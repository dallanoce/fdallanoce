#!/usr/bin/env python3
"""
PREZZI DEI CARBURANTI — dal Weekly Oil Bulletin della Commissione europea
al JSON letto dal sito

Lo lancia ogni settimana la GitHub Action .github/workflows/prezzi-carburanti.yml
e scrive assets/data/carburanti.json, che il tool "Costo del viaggio"
(assets/js/tools/viaggio.js) legge come prezzi di partenza.

DA DOVE ARRIVANO I PREZZI
Ogni settimana ogni Stato dell'UE comunica alla Commissione europea il prezzo
medio nazionale dei carburanti in vigore il lunedì; la Commissione li
pubblica il giovedì nel Weekly Oil Bulletin. Per l'Italia è il Ministero
(MIMIT) a mandarli. Questo script scarica il file "Prices with taxes, latest
prices" e ne legge la riga dell'Italia.

PERCHÉ NON IL CSV DEL MIMIT
La prima versione dello script leggeva il CSV giornaliero del MIMIT, con un
prezzo per ogni distributore. Il server del MIMIT però non risponde dai
runner di GitHub: la prima esecuzione è finita in timeout senza ricevere un
byte, e lo stesso succede da altri cloud. Il server della Commissione
risponde. Il prezzo cambia di poco: è una media nazionale invece di una
mediana, e c'è una settimana di ritardo invece di un giorno.

COSA C'È E COSA NO
Benzina (Euro-super 95), gasolio e GPL, tasse comprese. Il metano nel
bollettino non c'è: il sito usa il suo valore indicativo, come per
l'elettricità.

PERCHÉ IL BROWSER NON LO LEGGE DA SÉ
Il file è un .xlsx, e il server della Commissione non manda le intestazioni
CORS: un browser non avrebbe comunque il permesso di leggerlo da un altro
sito. Lo legge questo script una volta alla settimana, e il sito serve il
risultato come file statico. Nessun backend, nessuna chiave.

COSA FA
  1. scarica il file (o legge un file locale con --xlsx, per provarlo)
  2. apre il primo foglio: un .xlsx è uno zip di file XML, bastano zipfile
     e xml.etree della libreria standard
  3. trova le colonne per nome, non per posizione, e la riga "Italy"
  4. controlla che l'unità sia "1000 l" e divide per mille: €/L
  5. controlla che ogni prezzo sia plausibile e che la data non sia vecchia
  6. scrive il JSON solo se è cambiato, così il workflow non fa commit vuoti

Se qualcosa non torna lo script si ferma con un errore e non scrive niente:
il file vecchio resta dov'è e il sito continua a usarlo. Meglio un prezzo di
qualche settimana fa che uno sbagliato.

Solo libreria standard di Python: niente da installare.
"""

import argparse
import io
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path, PurePosixPath


# --------------------------------------------------------------------------
# PARAMETRI
# --------------------------------------------------------------------------

# "Prices with taxes, latest prices (xlsx)" sulla pagina
# https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en
#
# L'indirizzo resta lo stesso da una settimana all'altra: la Commissione
# sostituisce il file, non il link. Il nome del file che il server dichiara
# (oggi "…with Taxes - 2026-09-21.xlsx") non conta e può essere vecchio di
# mesi: la data vera dei prezzi sta dentro il foglio.
# Se un giorno smette di funzionare: aprire la pagina qui sopra, cercare
# "Prices with taxes latest prices (xlsx)", copiare il link di Download.
URL_XLSX = "https://energy.ec.europa.eu/document/download/264c2d0f-f161-4ea3-a777-78faae59bea0_en"

# La radice del repository: questo file sta in .github/scripts/
RADICE = Path(__file__).resolve().parents[2]
USCITA_PREDEFINITA = RADICE / "assets" / "data" / "carburanti.json"

# Il paese, scritto come nella colonna A del foglio
PAESE = "Italy"

# La chiave usata dal sito → un pezzo dell'intestazione della sua colonna.
# Le intestazioni sono in tre lingue ("GPL pour moteur LPG motor fuel"):
# basta un pezzo che compaia in una sola colonna. Attenzione al gasolio:
# "Gas oil de chauffage" è il gasolio da riscaldamento, un altro prodotto.
COLONNE = {
    "benzina": "euro-super 95",
    "gasolio": "gas oil automobile",
    "gpl": "lpg motor fuel",
}

UNITA_ATTESA = "1000 l"      # il foglio dà € per 1000 litri
PREZZO_MINIMO = 0.3          # € al litro: sotto è un refuso
PREZZO_MASSIMO = 4.0         # sopra pure

# Oltre questi giorni il file non si sta più aggiornando: forse la
# Commissione ha cambiato il link. Il bollettino salta qualche settimana
# a Natale, per questo il margine è di tre settimane e non di una.
GIORNI_MASSIMI = 21

TEMPO_MASSIMO_S = 60


# --------------------------------------------------------------------------
# LETTURA DEL FILE .xlsx
# --------------------------------------------------------------------------

# Gli spazi dei nomi XML del formato. ET li vuole scritti per intero
# dentro ogni nome: "{spazio}nome".
NS_FOGLIO = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def scarica(url):
    richiesta = urllib.request.Request(url, headers={
        # Uno User-Agent da browser generico: alcuni server rifiutano quello
        # predefinito di Python.
        "User-Agent": "Mozilla/5.0 (compatible; prezzi-carburanti/2.0)",
    })
    with urllib.request.urlopen(richiesta, timeout=TEMPO_MASSIMO_S) as risposta:
        return risposta.read()


def testo_di(elemento):
    """Tutto il testo dentro un elemento, anche se spezzato in più <t>."""
    return "".join(t.text or "" for t in elemento.iter(NS_FOGLIO + "t"))


def leggi_foglio(dati):
    """Il primo foglio del file come { numero di riga: { "B": valore } }.

    I valori sono stringhe così come stanno nel file: un numero resta
    "1969.6", e una data resta il suo numero di serie Excel. Ritorna anche
    se il file conta le date dal 1904 invece che dal 1900."""
    try:
        archivio = zipfile.ZipFile(io.BytesIO(dati))
    except zipfile.BadZipFile:
        raise ValueError("il file scaricato non è un .xlsx: forse il link è cambiato")

    # Quale file XML è il primo foglio: lo dice workbook.xml, con un
    # identificativo che il file delle relazioni traduce in un percorso.
    cartella = ET.fromstring(archivio.read("xl/workbook.xml"))
    proprieta = cartella.find(NS_FOGLIO + "workbookPr")
    dal_1904 = proprieta is not None and proprieta.get("date1904") in ("1", "true")

    primo = cartella.find(NS_FOGLIO + "sheets/" + NS_FOGLIO + "sheet")
    id_foglio = primo.get(NS_REL_ID + "id")
    relazioni = ET.fromstring(archivio.read("xl/_rels/workbook.xml.rels"))
    destinazione = next(r.get("Target") for r in relazioni.iter(NS_REL + "Relationship")
                        if r.get("Id") == id_foglio)
    # Il percorso è relativo a xl/, oppure assoluto dentro lo zip
    if destinazione.startswith("/"):
        percorso = destinazione.lstrip("/")
    else:
        percorso = str(PurePosixPath("xl") / destinazione)

    # Le stringhe non stanno nelle celle ma in una tabella comune, e la
    # cella ne contiene il numero d'ordine.
    stringhe = []
    if "xl/sharedStrings.xml" in archivio.namelist():
        tabella = ET.fromstring(archivio.read("xl/sharedStrings.xml"))
        stringhe = [testo_di(si) for si in tabella.iter(NS_FOGLIO + "si")]

    righe = {}
    foglio = ET.fromstring(archivio.read(percorso))
    for cella in foglio.iter(NS_FOGLIO + "c"):
        # "B17" → colonna "B", riga 17
        posto = re.fullmatch(r"([A-Z]+)(\d+)", cella.get("r", ""))
        if not posto:
            continue
        tipo = cella.get("t")
        valore = cella.find(NS_FOGLIO + "v")
        if tipo == "s" and valore is not None:
            testo = stringhe[int(valore.text)]
        elif tipo == "inlineStr":
            testo = testo_di(cella)
        elif valore is not None:
            testo = valore.text or ""
        else:
            continue
        righe.setdefault(int(posto.group(2)), {})[posto.group(1)] = testo

    return righe, dal_1904


def normalizza(testo):
    """Minuscolo e con gli spazi ridotti a uno: le intestazioni del foglio
    vanno a capo e hanno spazi doppi."""
    return " ".join(testo.lower().split())


def data_da_excel(valore, dal_1904):
    """Una data Excel è un numero di giorni da un'origine fissa. L'origine
    "30 dicembre 1899" e non 1° gennaio 1900 è voluta: compensa un vecchio
    errore di Excel, che conta un 29 febbraio 1900 mai esistito."""
    try:
        giorni = float(valore)
    except ValueError:
        raise ValueError("la data dei prezzi non è una data: " + valore[:40])
    origine = date(1904, 1, 1) if dal_1904 else date(1899, 12, 30)
    return origine + timedelta(days=int(giorni))


# --------------------------------------------------------------------------
# CALCOLO
# --------------------------------------------------------------------------

def prezzi_italia(dati, oggi):
    righe, dal_1904 = leggi_foglio(dati)

    # La riga delle intestazioni è quella che nomina la benzina.
    # Quella subito sotto ha le unità, e nella colonna A la data dei prezzi.
    riga_titoli = next((n for n in sorted(righe)
                        if any(COLONNE["benzina"] in normalizza(v) for v in righe[n].values())),
                       None)
    if riga_titoli is None:
        raise ValueError("intestazioni non trovate: il formato del file è cambiato?")
    titoli = righe[riga_titoli]
    unita = righe.get(riga_titoli + 1, {})

    colonne = {}
    for chiave, pezzo in COLONNE.items():
        trovate = [col for col, testo in titoli.items() if pezzo in normalizza(testo)]
        if len(trovate) != 1:
            raise ValueError(f"{chiave}: {len(trovate)} colonne con \"{pezzo}\" invece di una")
        colonna = trovate[0]
        if normalizza(unita.get(colonna, "")) != UNITA_ATTESA:
            raise ValueError(f"{chiave}: unità \"{unita.get(colonna)}\" invece di \"{UNITA_ATTESA}\"")
        colonne[chiave] = colonna

    aggiornato = data_da_excel(unita.get("A", ""), dal_1904)
    vecchiaia = (oggi - aggiornato).days
    if vecchiaia > GIORNI_MASSIMI:
        raise ValueError(f"prezzi del {aggiornato.isoformat()}, vecchi di {vecchiaia} giorni: "
                         "il file non si aggiorna più, forse il link è cambiato")
    if vecchiaia < 0:
        raise ValueError(f"prezzi datati nel futuro: {aggiornato.isoformat()}")

    riga_paese = next((celle for n, celle in sorted(righe.items())
                       if normalizza(celle.get("A", "")) == PAESE.lower()), None)
    if riga_paese is None:
        raise ValueError(f"riga \"{PAESE}\" non trovata")

    prezzi = {}
    for chiave, colonna in colonne.items():
        try:
            per_mille = float(riga_paese.get(colonna, ""))
        except ValueError:
            raise ValueError(f"{chiave}: nessun prezzo per {PAESE}")
        prezzo = round(per_mille / 1000, 3)
        if not PREZZO_MINIMO <= prezzo <= PREZZO_MASSIMO:
            raise ValueError(f"{chiave}: {prezzo} €/L non è plausibile")
        prezzi[chiave] = prezzo

    return {
        "aggiornato": aggiornato.isoformat(),
        "fonte": "Commissione europea, Weekly Oil Bulletin: media nazionale dell'Italia, tasse comprese",
        "prezzi": prezzi,
    }


# --------------------------------------------------------------------------
# SCRITTURA
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prezzi dei carburanti in Italia dal Weekly Oil Bulletin della Commissione europea.")
    parser.add_argument("--xlsx", help="legge questo file invece di scaricarlo (per provare lo script)")
    parser.add_argument("--out", default=str(USCITA_PREDEFINITA), help="dove scrivere il JSON")
    argomenti = parser.parse_args()

    try:
        dati = Path(argomenti.xlsx).read_bytes() if argomenti.xlsx else scarica(URL_XLSX)
        risultato = prezzi_italia(dati, date.today())
    except Exception as e:  # qualunque guasto: non si scrive niente
        print("ERRORE: " + str(e), file=sys.stderr)
        return 1

    testo = json.dumps(risultato, ensure_ascii=False, indent=2) + "\n"
    uscita = Path(argomenti.out)

    if uscita.exists() and uscita.read_text(encoding="utf-8") == testo:
        print("Nessuna variazione: " + str(uscita))
        return 0

    uscita.parent.mkdir(parents=True, exist_ok=True)
    uscita.write_text(testo, encoding="utf-8")
    print("Scritto " + str(uscita))
    print(testo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
