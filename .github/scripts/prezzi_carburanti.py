#!/usr/bin/env python3
"""
PREZZI DEI CARBURANTI — dal CSV del MIMIT al JSON letto dal sito

Lo lancia ogni settimana la GitHub Action .github/workflows/prezzi-carburanti.yml
e scrive assets/data/carburanti.json, che il tool "Costo del viaggio"
(assets/js/tools/viaggio.js) legge come prezzi di partenza.

PERCHÉ UNO SCRIPT E NON UNA CHIAMATA DAL BROWSER
Il MIMIT pubblica ogni giorno i prezzi di tutti i distributori italiani in un
CSV, ma il suo server non manda le intestazioni CORS: un browser non ha il
permesso di leggerlo da un altro sito. Lo legge quindi questo script, una
volta alla settimana, e il sito serve il risultato come file statico.
Il sito resta una cartella di file: nessun backend, nessuna funzione di
Cloudflare, nessuna chiave.

COSA FA
  1. scarica prezzo_alle_8.csv (o legge un file locale con --csv, per provarlo)
  2. tiene benzina, gasolio, GPL e metano, con il nome esatto: le varianti
     "speciali" (Blue Diesel, HiQ Perform+…) sono un altro prodotto
  3. benzina e gasolio solo in self-service, che è il prezzo che si paga di
     solito; GPL e metano in qualunque modalità, perché il self lì è raro
  4. scarta i prezzi comunicati più di 30 giorni prima dell'estrazione
     (impianti fermi) e quelli fuori da un intervallo plausibile (refusi)
  5. un prezzo per impianto, il più basso se ce n'è più d'uno
  6. per ogni carburante la MEDIANA nazionale: un distributore che sbaglia
     una virgola non la sposta, una media invece sì
  7. scrive il JSON solo se è cambiato, così il workflow non fa commit vuoti

Se un carburante resta con troppo pochi impianti lo script si ferma con un
errore e non scrive niente: il file vecchio resta dov'è e il sito continua a
usarlo. Meglio un prezzo di una settimana fa che uno sbagliato.

Solo libreria standard di Python: niente da installare.
"""

import argparse
import csv
import io
import json
import re
import statistics
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path


# --------------------------------------------------------------------------
# PARAMETRI
# --------------------------------------------------------------------------

URL_CSV = "https://www.mimit.gov.it/images/exportCSV/prezzo_alle_8.csv"

# La radice del repository: questo file sta in .github/scripts/
RADICE = Path(__file__).resolve().parents[2]
USCITA_PREDEFINITA = RADICE / "assets" / "data" / "carburanti.json"

# Il nome del carburante nel CSV (minuscolo) → la chiave usata dal sito
CARBURANTI = {
    "benzina": "benzina",
    "gasolio": "gasolio",
    "gpl": "gpl",
    "metano": "metano",
}

# Quelli che contano solo in self-service
SOLO_SELF = {"benzina", "gasolio"}

GIORNI_MASSIMI = 30          # un prezzo più vecchio di così è di un impianto fermo
PREZZO_MINIMO = 0.3          # € al litro o al kg: sotto è un refuso
PREZZO_MASSIMO = 4.0         # sopra pure
IMPIANTI_MINIMI = 100        # sotto, il file è incompleto: meglio non scrivere niente

TEMPO_MASSIMO_S = 120


# --------------------------------------------------------------------------
# LETTURA
# --------------------------------------------------------------------------

def scarica(url):
    richiesta = urllib.request.Request(url, headers={
        # Uno User-Agent da browser generico: alcuni server rifiutano quello
        # predefinito di Python.
        "User-Agent": "Mozilla/5.0 (compatible; prezzi-carburanti/1.0)",
    })
    with urllib.request.urlopen(richiesta, timeout=TEMPO_MASSIMO_S) as risposta:
        return risposta.read()


def decodifica(dati):
    # I campi che servono sono tutti ASCII: una lettera accentata mal
    # decodificata nel nome di un impianto non cambia niente.
    try:
        return dati.decode("utf-8")
    except UnicodeDecodeError:
        return dati.decode("latin-1")


def data_estrazione(prima_riga):
    """La prima riga del file è "Estrazione del 2026-11-17"."""
    trovata = re.search(r"(\d{4})-(\d{2})-(\d{2})", prima_riga)
    if not trovata:
        raise ValueError("prima riga inattesa: " + prima_riga.strip()[:80])
    return date(int(trovata.group(1)), int(trovata.group(2)), int(trovata.group(3)))


def separatore(intestazione):
    """Dal 10 febbraio 2026 i campi sono separati da "|"; prima da ";".
    Si riconoscono tutti e due, così lo script funziona anche su un file
    dell'archivio storico."""
    if "|" in intestazione:
        return "|"
    if ";" in intestazione:
        return ";"
    raise ValueError("separatore non riconosciuto: " + intestazione.strip()[:80])


def data_comunicazione(testo):
    """dtComu è "gg/mm/aaaa hh:mm:ss"."""
    try:
        return datetime.strptime(testo.strip()[:10], "%d/%m/%Y").date()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# CALCOLO
# --------------------------------------------------------------------------

def mediane(testo):
    righe = testo.splitlines()
    if len(righe) < 3:
        raise ValueError("file troppo corto")

    estrazione = data_estrazione(righe[0])
    sep = separatore(righe[1])
    lettore = csv.reader(io.StringIO("\n".join(righe[1:])), delimiter=sep)

    # Le colonne si cercano per nome, non per posizione
    intestazione = [nome.strip().lower() for nome in next(lettore)]
    try:
        i_id = intestazione.index("idimpianto")
        i_carb = intestazione.index("desccarburante")
        i_prezzo = intestazione.index("prezzo")
        i_self = intestazione.index("isself")
        i_data = intestazione.index("dtcomu")
    except ValueError as e:
        raise ValueError("colonna mancante nel CSV: " + str(e))

    # (impianto, carburante) → prezzo più basso trovato
    per_impianto = {}

    for campi in lettore:
        if len(campi) <= max(i_id, i_carb, i_prezzo, i_self, i_data):
            continue

        chiave = CARBURANTI.get(campi[i_carb].strip().lower())
        if not chiave:
            continue
        if chiave in SOLO_SELF and campi[i_self].strip() != "1":
            continue

        comunicato = data_comunicazione(campi[i_data])
        if comunicato is None or (estrazione - comunicato).days > GIORNI_MASSIMI:
            continue

        try:
            prezzo = float(campi[i_prezzo].strip().replace(",", "."))
        except ValueError:
            continue
        if not PREZZO_MINIMO <= prezzo <= PREZZO_MASSIMO:
            continue

        posto = (campi[i_id].strip(), chiave)
        if posto not in per_impianto or prezzo < per_impianto[posto]:
            per_impianto[posto] = prezzo

    prezzi = {}
    impianti = {}
    for chiave in CARBURANTI.values():
        valori = [p for (_, k), p in per_impianto.items() if k == chiave]
        if len(valori) < IMPIANTI_MINIMI:
            raise ValueError(f"{chiave}: solo {len(valori)} impianti validi, file incompleto?")
        prezzi[chiave] = round(statistics.median(valori), 3)
        impianti[chiave] = len(valori)

    return {
        "aggiornato": estrazione.isoformat(),
        "fonte": "MIMIT, prezzo_alle_8.csv: mediana nazionale, self-service per benzina e gasolio",
        "prezzi": prezzi,
        "impianti": impianti,
    }


# --------------------------------------------------------------------------
# SCRITTURA
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prezzi medi dei carburanti dal CSV del MIMIT.")
    parser.add_argument("--csv", help="legge questo file invece di scaricarlo (per provare lo script)")
    parser.add_argument("--out", default=str(USCITA_PREDEFINITA), help="dove scrivere il JSON")
    argomenti = parser.parse_args()

    try:
        dati = Path(argomenti.csv).read_bytes() if argomenti.csv else scarica(URL_CSV)
        risultato = mediane(decodifica(dati))
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
