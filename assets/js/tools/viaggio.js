/* ==========================================================================
   VIAGGIO — quanto costa andare da A a B, auto per auto

   Struttura del file:
     1. PARAMETRI   — servizi, carburanti, auto e consumi, isolati in cima
     2. CALCOLO     — funzioni pure: percorso e scelte dentro, costi fuori
     3. INTERFACCIA — mappa, ricerca degli indirizzi, tabelle, eventi

   DUE LINGUE, UN FILE SOLO
   Questo file serve sia /it/strumenti/viaggio/ sia /en/tools/trip-cost/.
   Le due pagine hanno gli stessi id proprio per questo: i consumi e le
   formule esistono una volta sola.

   L'UNICO TOOL CHE PARLA CON L'ESTERNO
   Gli altri tool del sito non mandano niente a nessuno. Questo non può:
   per sapere dove sta un indirizzo e quale strada si fa servono dei
   servizi esterni, tutti di OpenStreetMap e tutti senza chiave:

     - Nominatim       riceve il testo degli indirizzi e i punti cliccati
     - OSRM (FOSSGIS)  riceve le coordinate di partenza e arrivo
     - le tile         la cartina, un'immagine per ogni quadrato visibile

   La pagina lo dice prima che si scriva qualcosa, e l'indice dei tool lo
   segnala. I conti invece si fanno qui, come negli altri tool, e prezzi,
   scelte e indirizzi restano in localStorage.

   Tutti e due i servizi chiedono al massimo una richiesta al secondo e
   Nominatim vieta l'autocompletamento: per questo si cerca solo con Invio,
   con "Cerca" o con "Calcola", e le richieste allo stesso server passano
   in fila (vedi chiedi()).

   LA MAPPA LA DISEGNA LEAFLET
   È una libreria copiata in /assets/vendor/leaflet/ e caricata prima di
   questo file: da lei arriva la variabile globale L. Il resto del codice
   funziona anche se L non c'è — senza mappa si scrivono gli indirizzi.

   PERCHÉ IN MEMORIA CI SONO CHIAVI E NON NOMI
   Come in grigliata.js: localStorage è dello stesso dominio per tutti e
   due gli alberi, quindi si salvano le chiavi ("berlina", "gpl") e i nomi
   si traducono al momento di disegnarli.
   ========================================================================== */


/* --------------------------------------------------------------------------
   1. PARAMETRI
   -------------------------------------------------------------------------- */

/* Gli indirizzi dei servizi stanno qui e da nessun'altra parte: le
   condizioni d'uso del server dei percorsi chiedono espressamente di non
   cablarli nel codice, perché possono cambiare. */
const SERVIZI = {
  percorso:  "https://routing.openstreetmap.de/routed-car/route/v1/driving/",
  indirizzi: "https://nominatim.openstreetmap.org/",
  tile:      "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  crediti:   '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
};

/* Una richiesta al secondo per server, con un po' di margine. */
const ATTESA_MIN_MS = 1100;

/* Oltre questo tempo una richiesta si considera persa. */
const TEMPO_MASSIMO_MS = 15000;

/* Il file dei prezzi ufficiali, scritto ogni settimana da una GitHub
   Action (vedi .github/workflows/prezzi-carburanti.yml). Sta sullo stesso
   dominio del sito: leggerlo non manda niente a nessuno. */
const PREZZI_UFFICIALI_URL = "/assets/data/carburanti.json";

const CHIAVE_VIAGGIO = "viaggio";   /* dove si salva la configurazione */

/* Dove si apre la mappa la prima volta: l'Italia intera. */
const CENTRO_MAPPA = [42.4, 12.6];
const ZOOM_MAPPA = 5;

/* --- Carburanti ----------------------------------------------------------
   Quello che si compra, con l'unità in cui si paga. Il prezzo di riserva
   si usa solo se manca il file dei prezzi ufficiali: è un ordine di
   grandezza, e la pagina lo chiama "valore indicativo". L'elettricità non
   ha una fonte ufficiale settimanale, quindi usa sempre il suo. */
const CARBURANTI = {
  benzina:     { nome: { it: "Benzina",     en: "Petrol" },      unita: "L",   riserva: 1.70 },
  gasolio:     { nome: { it: "Gasolio",     en: "Diesel" },      unita: "L",   riserva: 1.65 },
  gpl:         { nome: { it: "GPL",         en: "LPG" },         unita: "L",   riserva: 0.72 },
  metano:      { nome: { it: "Metano",      en: "CNG" },         unita: "kg",  riserva: 1.45 },
  elettricita: { nome: { it: "Elettricità", en: "Electricity" }, unita: "kWh", riserva: 0.40 },
};

/* --- Auto ------------------------------------------------------------------
   L'ordine di questi oggetti è l'ordine delle righe e delle colonne. */
const TIPI = {
  utilitaria: { nome: { it: "Utilitaria", en: "City car" } },
  compatta:   { nome: { it: "Compatta",   en: "Compact" } },
  berlina:    { nome: { it: "Berlina",    en: "Saloon" } },
  suv:        { nome: { it: "SUV",        en: "SUV" } },
  monovolume: { nome: { it: "Monovolume", en: "MPV" } },
};

/* Ogni alimentazione dice quale carburante consuma. Il plug-in ne consuma
   due e ha un calcolo tutto suo: vedi fabbisogno(). */
const ALIMENTAZIONI = {
  benzina:   { nome: { it: "Benzina",   en: "Petrol" },   carburante: "benzina" },
  diesel:    { nome: { it: "Diesel",    en: "Diesel" },   carburante: "gasolio" },
  gpl:       { nome: { it: "GPL",       en: "LPG" },      carburante: "gpl" },
  metano:    { nome: { it: "Metano",    en: "CNG" },      carburante: "metano" },
  ibrido:    { nome: { it: "Ibrida",    en: "Hybrid" },   carburante: "benzina" },
  phev:      { nome: { it: "Plug-in",   en: "Plug-in" },  carburante: "benzina" },
  elettrico: { nome: { it: "Elettrica", en: "Electric" }, carburante: "elettricita" },
};

/* --- Strade ------------------------------------------------------------------
   Lo stesso viaggio costa diverso a seconda di dove passa: un'elettrica
   consuma poco in città e tanto in autostrada, un diesel il contrario. Per
   questo il percorso si divide in tre tipi di strada (più il traghetto, che
   non consuma niente). */
const REGIMI = {
  citta:      { it: "Città",       en: "Urban" },
  extra:      { it: "Extraurbano", en: "Out of town" },
  autostrada: { it: "Autostrada",  en: "Motorway" },
  traghetto:  { it: "Traghetto",   en: "Ferry" },
};

/* COME SI RICONOSCE UN'AUTOSTRADA
   Il server dei percorsi di FOSSGIS non marca i tratti come "autostrada" o
   "a pedaggio" (altri server OSRM lo fanno, e se ci sono quelle marcature
   vengono usate). Dice però due cose affidabili per ogni tratto: la sigla
   della strada ("A1", "A14;E45") e quanto tempo ci vuole a percorrerlo.

   Un tratto è autostrada se ha una sigla da autostrada o se lo si percorre
   in media almeno a SOGLIA_AUTOSTRADA_KMH: per i consumi conta la velocità,
   e una superstrada a 110 all'ora consuma come un'autostrada. Sotto
   SOGLIA_CITTA_KMH è città, in mezzo è extraurbano. */
const SOGLIA_CITTA_KMH = 50;
const SOGLIA_AUTOSTRADA_KMH = 90;

/* Le sigle da autostrada: A in Italia, Francia, Germania, Austria, Svizzera
   (anche scritte "A 7" o "A-7"), AP in Spagna, M nel Regno Unito. */
const SIGLA_AUTOSTRADA = /^(A|AP|M)[\s-]?(\d+)/;

/* IL PEDAGGIO È UNA STIMA
   Non sapendo quali tratti sono a pagamento, si contano a pedaggio i km su
   strade con una sigla da autostrada, tolte le principali autostrade
   italiane gratuite. La velocità qui non conta: una superstrada veloce non
   si paga. L'elenco non è completo e fuori dall'Italia le regole sono altre
   (vignette in Svizzera e Austria, niente pedaggio per le auto in Germania):
   la pagina lo dice, e la tariffa si può cambiare o togliere dal conto. */
const AUTOSTRADE_GRATUITE = [
  "A2",    /* del Mediterraneo, Salerno–Reggio Calabria */
  "A19",   /* Palermo–Catania */
  "A29",   /* Palermo–Mazara del Vallo */
  "A90",   /* Grande Raccordo Anulare di Roma */
  "A91",   /* Roma–Fiumicino */
];

/* --- Consumi ------------------------------------------------------------------
   Quanto consuma ogni tipo di auto ogni 100 km, su ogni tipo di strada:
   litri per benzina, diesel, GPL e ibride, kg per il metano, kWh per le
   elettriche.

   Sono consumi reali stimati, non quelli di omologazione, che su strada non
   si vedono mai. Per le elettriche sono kWh presi dalla presa, perdite di
   ricarica comprese: è quello che si paga.

   Il plug-in ha due serie: quanto consuma quando va a batteria e quanto
   quando la batteria è finita e va a benzina. Pesa più di un'ibrida
   normale, quindi consuma un po' di più in tutti e due i modi.

   La funzione strade() esiste solo per leggibilità: strade(città,
   extraurbano, autostrada). Le tabelle sono scritte per esteso di proposito, senza
   moltiplicatori nascosti: per correggere un numero si cambia quel numero. */
function strade(citta, extra, autostrada) {
  return { citta: citta, extra: extra, autostrada: autostrada };
}

const CONSUMI = {
  utilitaria: {
    benzina:       strade(6.8, 4.9, 6.3),
    diesel:        strade(5.3, 3.8, 5.0),
    gpl:           strade(8.6, 6.2, 8.0),
    metano:        strade(5.0, 3.6, 4.6),
    ibrido:        strade(4.0, 4.0, 5.6),
    phevElettrico: strade(14,  15,  20),
    phevBenzina:   strade(4.4, 4.4, 6.2),
    elettrico:     strade(13,  14,  19),
  },
  compatta: {
    benzina:       strade(7.8, 5.3, 6.8),
    diesel:        strade(6.0, 4.2, 5.4),
    gpl:           strade(9.8, 6.7, 8.6),
    metano:        strade(5.6, 3.9, 5.0),
    ibrido:        strade(4.4, 4.4, 6.0),
    phevElettrico: strade(16,  17,  22),
    phevBenzina:   strade(5.0, 5.0, 6.7),
    elettrico:     strade(15,  16,  21),
  },
  berlina: {
    benzina:       strade(9.0, 5.8, 7.3),
    diesel:        strade(6.8, 4.6, 5.8),
    gpl:           strade(11.3, 7.3, 9.2),
    metano:        strade(6.4, 4.3, 5.3),
    ibrido:        strade(4.9, 4.8, 6.5),
    phevElettrico: strade(18,  19,  24),
    phevBenzina:   strade(5.6, 5.5, 7.2),
    elettrico:     strade(16,  17,  22),
  },
  suv: {
    benzina:       strade(9.8, 6.5, 8.3),
    diesel:        strade(7.6, 5.2, 6.6),
    gpl:           strade(12.3, 8.2, 10.4),
    metano:        strade(7.0, 4.8, 6.0),
    ibrido:        strade(5.6, 5.4, 7.3),
    phevElettrico: strade(20,  21,  27),
    phevBenzina:   strade(6.4, 6.2, 8.1),
    elettrico:     strade(18,  19,  25),
  },
  monovolume: {
    benzina:       strade(9.5, 6.3, 8.0),
    diesel:        strade(7.4, 5.0, 6.4),
    gpl:           strade(12.0, 7.9, 10.0),
    metano:        strade(6.8, 4.6, 5.8),
    ibrido:        strade(5.5, 5.3, 7.1),
    phevElettrico: strade(20,  21,  27),
    phevBenzina:   strade(6.2, 6.0, 7.9),
    elettrico:     strade(19,  20,  26),
  },
};

/* --- Valori di partenza delle impostazioni ---------------------------------- */

/* Una tariffa media per un'auto di classe A sulle autostrade italiane.
   È una stima: le tratte a sistema chiuso si pagano a km, quelle a
   barriera a forfait, e ogni concessionaria ha la sua tariffa. */
const PEDAGGIO_PREDEFINITO = 0.08;

/* Quanti km fa un plug-in a batteria prima di passare alla benzina.
   Si parte con la batteria carica. */
const AUTONOMIA_PHEV_PREDEFINITA = 50;

/* "La tua auto" può essere di tutte le alimentazioni tranne il plug-in,
   che con un consumo solo non si descrive. */
const ALIMENTAZIONI_TUA_AUTO = ["benzina", "diesel", "gpl", "metano", "ibrido", "elettrico"];

/* Come si può scrivere il consumo della propria auto: per 100 km
   ("6 L/100 km") oppure al contrario ("16 km/L"), che è come lo dicono
   molti computer di bordo. */
const FORME_CONSUMO = ["per100", "km"];


/* --------------------------------------------------------------------------
   2. CALCOLO — nessun riferimento al DOM né alla rete fino alla sezione 3
   -------------------------------------------------------------------------- */

/* Accetta sia la virgola sia il punto come separatore decimale. */
function num(v) {
  const n = parseFloat(String(v).replace(",", "."));
  return isFinite(n) ? n : 0;
}

/* Le classi che il servizio dei percorsi attacca a un tratto ("motorway",
   "toll", "ferry"…). Stanno sugli incroci del tratto: si raccolgono tutte. */
function classiDelTratto(step) {
  const classi = new Set();
  (step.intersections || []).forEach(function (incrocio) {
    (incrocio.classes || []).forEach(function (nome) { classi.add(nome); });
  });
  return classi;
}

/* La sigla da autostrada di un tratto, ridotta all'essenziale ("A 7" →
   "A7", "A1-R5" → "A1"), oppure null. Una strada può avere più sigle
   separate da ";": "A14;E45" è la A14 e anche la europea E45. */
function siglaAutostrada(step) {
  const sigle = String(step.ref || "").split(";");
  for (const sigla of sigle) {
    const trovata = SIGLA_AUTOSTRADA.exec(sigla.trim());
    if (trovata) return trovata[1] + trovata[2];
  }
  return null;
}

/* Che tipo di strada è un tratto. */
function regimeDelTratto(step, classi, sigla) {
  if (step.mode === "ferry" || classi.has("ferry")) return "traghetto";

  const kmh = step.duration > 0 ? step.distance / step.duration * 3.6 : 0;
  if (classi.has("motorway") || sigla || kmh >= SOGLIA_AUTOSTRADA_KMH) return "autostrada";
  return kmh < SOGLIA_CITTA_KMH ? "citta" : "extra";
}

/* Se un tratto si paga. Una marcatura esplicita del server vince; senza,
   si stima dalla sigla (vedi AUTOSTRADE_GRATUITE). */
function aPedaggio(classi, sigla, regime) {
  if (regime === "traghetto") return false;
  if (classi.has("toll")) return true;
  return !!sigla && AUTOSTRADE_GRATUITE.indexOf(sigla) === -1;
}

/* La risposta del servizio dei percorsi ridotta a quello che serve:
   la linea da disegnare, la durata, e i tratti IN ORDINE di percorrenza —
   l'ordine conta per il plug-in, che va a batteria all'inizio.
   Restituisce null se un percorso non c'è. */
function leggiPercorso(risposta) {
  if (!risposta || risposta.code !== "Ok" || !risposta.routes || !risposta.routes.length) {
    return null;
  }
  const rotta = risposta.routes[0];
  const tratti = [];

  (rotta.legs || []).forEach(function (gamba) {
    (gamba.steps || []).forEach(function (step) {
      if (!(step.distance > 0)) return;   /* l'arrivo è un tratto lungo zero */
      const classi = classiDelTratto(step);
      const sigla = siglaAutostrada(step);
      const regime = regimeDelTratto(step, classi, sigla);
      tratti.push({
        km: step.distance / 1000,
        regime: regime,
        pedaggio: aPedaggio(classi, sigla, regime),
      });
    });
  });

  /* Il servizio scrive le coordinate come [lon, lat], Leaflet le vuole
     [lat, lon]. È l'errore classico di chi lavora con le mappe. */
  const linea = rotta.geometry && Array.isArray(rotta.geometry.coordinates)
    ? rotta.geometry.coordinates.map(function (p) { return [p[1], p[0]]; })
    : [];

  return { tratti: tratti, secondi: num(rotta.duration), linea: linea };
}

/* Quanti km per tipo di strada, quanti a pedaggio, quanti in tutto. */
function riepilogo(tratti) {
  const r = { citta: 0, extra: 0, autostrada: 0, traghetto: 0, pedaggio: 0, totale: 0 };
  tratti.forEach(function (t) {
    r[t.regime] += t.km;
    r.totale += t.km;
    if (t.pedaggio) r.pedaggio += t.km;
  });
  return r;
}

/* Quanto consuma una serie di consumi { citta, extra, autostrada } su
   questi tratti. Il traghetto porta l'auto e non la fa consumare. */
function consumoSuTratti(serie, tratti) {
  return tratti.reduce(function (somma, t) {
    if (t.regime === "traghetto") return somma;
    return somma + t.km * serie[t.regime] / 100;
  }, 0);
}

/* Quanto carburante serve a questa auto per questo viaggio, per
   carburante: { benzina: 31.2 } oppure, per un plug-in,
   { elettricita: 9, benzina: 12.4 }. */
function fabbisogno(tipo, alimentazione, tratti, autonomiaPhev) {
  const tabella = CONSUMI[tipo];

  if (alimentazione === "phev") {
    /* I tratti si percorrono in ordine: a batteria finché ce n'è, poi a
       benzina. Un tratto a metà fra le due cose si divide. */
    let batteria = Math.max(0, num(autonomiaPhev));
    let kWh = 0;
    let litri = 0;

    tratti.forEach(function (t) {
      if (t.regime === "traghetto") return;
      const aBatteria = Math.min(t.km, batteria);
      batteria -= aBatteria;
      kWh   += aBatteria * tabella.phevElettrico[t.regime] / 100;
      litri += (t.km - aBatteria) * tabella.phevBenzina[t.regime] / 100;
    });

    return { elettricita: kWh, benzina: litri };
  }

  const fuori = {};
  fuori[ALIMENTAZIONI[alimentazione].carburante] = consumoSuTratti(tabella[alimentazione], tratti);
  return fuori;
}

/* "La tua auto" se è attiva e ha un consumo sensato, altrimenti null.
   Il consumo è uno solo, uguale su tutte le strade, e si riporta sempre
   alla forma "per 100 km". */
function tuaAutoValida(tua) {
  if (!tua || !tua.attiva) return null;
  if (ALIMENTAZIONI_TUA_AUTO.indexOf(tua.alimentazione) === -1) return null;
  const valore = num(tua.consumo);
  if (!(valore > 0)) return null;

  const per100 = tua.forma === "km" ? 100 / valore : valore;
  return { alimentazione: tua.alimentazione, per100: per100 };
}

/* Il conto completo. Restituisce una matrice: una riga per tipo di auto
   scelto (più "la tua auto"), una colonna per alimentazione scelta.

   Una cella a cui manca un prezzo non viene indovinata: resta senza
   costo, con il motivo, e non entra nel confronto. Così il "più
   conveniente" non può essere tale solo perché un prezzo è vuoto. */
function calcola(percorso, scelte, prezzi) {
  const riep = riepilogo(percorso.tratti);
  const pedaggio = riep.pedaggio * Math.max(0, num(scelte.pedaggioTariffa));
  const aggiunta = scelte.pedaggioIncluso ? pedaggio : 0;
  const tua = tuaAutoValida(scelte.tuaAuto);
  const mancanti = new Set();

  /* Le colonne: le alimentazioni scelte, più quella della tua auto se non
     c'è già — altrimenti la sua cella non avrebbe dove stare. */
  const colonne = Object.keys(ALIMENTAZIONI).filter(function (a) {
    return scelte.alimentazioni.indexOf(a) !== -1 || (tua && tua.alimentazione === a);
  });

  function cella(fab) {
    let costo = 0;
    for (const carb of Object.keys(fab)) {
      if (!(fab[carb] > 0)) continue;   /* un plug-in tutto a batteria non compra benzina */
      const prezzo = num(prezzi[carb]);
      if (!(prezzo > 0)) {
        mancanti.add(carb);
        return {
          fabbisogno: fab, costo: null,
          errore: { it: "manca il prezzo: " + CARBURANTI[carb].nome.it,
                    en: "price missing: " + CARBURANTI[carb].nome.en },
        };
      }
      costo += fab[carb] * prezzo;
    }
    return { fabbisogno: fab, costo: costo + aggiunta, errore: null };
  }

  const righe = Object.keys(TIPI)
    .filter(function (t) { return scelte.tipi.indexOf(t) !== -1; })
    .map(function (tipo) {
      const celle = {};
      colonne.forEach(function (a) {
        /* Una colonna aggiunta solo per la tua auto resta vuota qui */
        celle[a] = scelte.alimentazioni.indexOf(a) !== -1
          ? cella(fabbisogno(tipo, a, percorso.tratti, scelte.autonomiaPhev))
          : null;
      });
      return { chiave: tipo, celle: celle };
    });

  if (tua) {
    const fab = {};
    fab[ALIMENTAZIONI[tua.alimentazione].carburante] =
      consumoSuTratti(strade(tua.per100, tua.per100, tua.per100), percorso.tratti);
    const celle = {};
    colonne.forEach(function (a) { celle[a] = null; });
    celle[tua.alimentazione] = cella(fab);
    righe.push({ chiave: "tua", celle: celle, tua: tua });
  }

  /* Il più conveniente di ogni riga e quello di tutta la tabella. A
     parità vince il primo, cioè quello più a sinistra e più in alto. */
  let migliore = null;
  righe.forEach(function (riga) {
    riga.minimo = null;
    riga.valide = 0;
    colonne.forEach(function (a) {
      const x = riga.celle[a];
      if (!x || x.costo == null) return;
      riga.valide += 1;
      if (riga.minimo === null || x.costo < riga.celle[riga.minimo].costo) riga.minimo = a;
      if (!migliore || x.costo < migliore.costo) {
        migliore = { riga: riga.chiave, colonna: a, costo: x.costo };
      }
    });
  });

  return {
    riepilogo: riep,
    pedaggio: pedaggio,
    pedaggioIncluso: !!scelte.pedaggioIncluso,
    colonne: colonne,
    righe: righe,
    migliore: migliore,
    prezziMancanti: Array.from(mancanti),
  };
}


/* --------------------------------------------------------------------------
   3. INTERFACCIA
   -------------------------------------------------------------------------- */

/* --- stato: lettura e scrittura ------------------------------------------ */

function statoIniziale() {
  return {
    prezzi: {},        /* solo i prezzi cambiati a mano: { gpl: 0.75 } */
    tipi: Object.keys(TIPI),
    alimentazioni: Object.keys(ALIMENTAZIONI),
    autonomiaPhev: AUTONOMIA_PHEV_PREDEFINITA,
    pedaggioTariffa: PEDAGGIO_PREDEFINITO,
    pedaggioIncluso: true,
    tuaAuto: { attiva: false, alimentazione: "benzina", consumo: 0, forma: "per100" },
    partenza: null,    /* { etichetta, lat, lon } */
    arrivo: null,
  };
}

/* Un numero letto da localStorage: null e stringhe vuote non sono zero. */
function numeroSalvato(v) {
  return v !== null && v !== "" && v !== undefined && isFinite(Number(v)) ? Number(v) : null;
}

function puntoSalvato(p) {
  if (!p || typeof p !== "object") return null;
  const lat = numeroSalvato(p.lat);
  const lon = numeroSalvato(p.lon);
  if (lat === null || lon === null || Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat: lat, lon: lon, etichetta: String(p.etichetta || "") };
}

/* Si rilegge quello che è stato salvato tenendo solo chiavi conosciute e
   valori del tipo giusto: un salvataggio vecchio o rovinato non deve
   rompere la pagina, al massimo si riparte dai valori di serie. */
function leggiViaggio() {
  const st = statoIniziale();

  let letto = null;
  try {
    letto = JSON.parse(localStorage.getItem(CHIAVE_VIAGGIO) || "null");
  } catch (e) {
    return st;   /* navigazione privata, storage negato o testo illeggibile */
  }
  if (!letto || typeof letto !== "object") return st;

  if (letto.prezzi && typeof letto.prezzi === "object") {
    Object.keys(CARBURANTI).forEach(function (k) {
      const v = numeroSalvato(letto.prezzi[k]);
      if (v !== null) st.prezzi[k] = v;
    });
  }

  const soloConosciute = function (elenco, dizionario) {
    return Array.isArray(elenco) ? elenco.filter(function (k) { return dizionario[k]; }) : [];
  };
  const tipi = soloConosciute(letto.tipi, TIPI);
  if (tipi.length) st.tipi = tipi;
  const alimentazioni = soloConosciute(letto.alimentazioni, ALIMENTAZIONI);
  if (alimentazioni.length) st.alimentazioni = alimentazioni;

  ["autonomiaPhev", "pedaggioTariffa"].forEach(function (k) {
    const v = numeroSalvato(letto[k]);
    if (v !== null && v >= 0) st[k] = v;
  });
  if (typeof letto.pedaggioIncluso === "boolean") st.pedaggioIncluso = letto.pedaggioIncluso;

  const tua = letto.tuaAuto;
  if (tua && typeof tua === "object") {
    st.tuaAuto.attiva = tua.attiva === true;
    if (ALIMENTAZIONI_TUA_AUTO.indexOf(tua.alimentazione) !== -1) st.tuaAuto.alimentazione = tua.alimentazione;
    if (FORME_CONSUMO.indexOf(tua.forma) !== -1) st.tuaAuto.forma = tua.forma;
    const consumo = numeroSalvato(tua.consumo);
    if (consumo !== null && consumo >= 0) st.tuaAuto.consumo = consumo;
  }

  st.partenza = puntoSalvato(letto.partenza);
  st.arrivo = puntoSalvato(letto.arrivo);
  return st;
}

/* Si salva con un ritardo: mentre si scrive un prezzo arrivano molti
   eventi, e non serve scrivere su disco a ogni tasto. */
let attesaSalvataggio = null;
function salvaViaggio() {
  clearTimeout(attesaSalvataggio);
  attesaSalvataggio = setTimeout(function () {
    try {
      localStorage.setItem(CHIAVE_VIAGGIO, JSON.stringify(stato));
    } catch (e) {
      /* spazio esaurito o storage negato: il tool funziona lo stesso */
    }
  }, 400);
}

let stato = leggiViaggio();

/* I prezzi ufficiali, quando arrivano: { aggiornato: "2026-11-16", prezzi: {…} } */
let ufficiali = null;

/* L'ultimo percorso calcolato. Sta solo in memoria: dipende dai server
   e dalle strade, e si ricalcola premendo Calcola. */
let percorso = null;

/* I risultati della ricerca di un indirizzo, per il menu "Intendevi…" */
const alternative = { partenza: [], arrivo: [] };

/* Dove va il prossimo clic sulla mappa */
let bersaglio = stato.partenza ? "arrivo" : "partenza";

/* Vero mentre si aspetta una risposta per Calcola */
let inCorso = false;


/* --- prezzi in uso --------------------------------------------------------- */

/* Il prezzo che vale, e da dove arriva: scritto a mano, ufficiale o di riserva. */
function prezzoDi(carb) {
  if (stato.prezzi[carb] !== undefined) return stato.prezzi[carb];
  if (ufficiali && ufficiali.prezzi && num(ufficiali.prezzi[carb]) > 0) return num(ufficiali.prezzi[carb]);
  return CARBURANTI[carb].riserva;
}

function origineDi(carb) {
  if (stato.prezzi[carb] !== undefined) return "mano";
  if (ufficiali && ufficiali.prezzi && num(ufficiali.prezzi[carb]) > 0) return "ufficiale";
  return "riserva";
}

function prezziCorrenti() {
  const fuori = {};
  Object.keys(CARBURANTI).forEach(function (k) { fuori[k] = prezzoDi(k); });
  return fuori;
}


/* --- formattazione ---------------------------------------------------------- */

/* Il locale segue la lingua della pagina: mai scritto a mano. */
const euro = new Intl.NumberFormat(I18N.locale, { style: "currency", currency: "EUR" });
const euroAlLitro = new Intl.NumberFormat(I18N.locale, {
  style: "currency", currency: "EUR", minimumFractionDigits: 3, maximumFractionDigits: 3,
});
const numero0 = new Intl.NumberFormat(I18N.locale, { maximumFractionDigits: 0 });
const numero1 = new Intl.NumberFormat(I18N.locale, { maximumFractionDigits: 1 });
const dataMedia = new Intl.DateTimeFormat(I18N.locale, { dateStyle: "medium" });

const $ = function (id) { return document.getElementById(id); };

/* Nomi di luoghi e testo dei servizi esterni finiscono in innerHTML:
   vanno neutralizzati. */
function esc(s) {
  return String(s).replace(/[&<>"]/g, function (ch) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch];
  });
}

/* Il valore memorizzato, scritto come lo scriverebbe chi legge la pagina. */
function perCampo(v) {
  const s = String(v);
  return I18N.lingua === "it" ? s.replace(".", ",") : s;
}

/* I km lunghi senza decimali, quelli corti con uno. */
function km(v) {
  return (v < 100 ? numero1 : numero0).format(v) + " km";
}

function durata(secondi) {
  const minuti = Math.round(secondi / 60);
  const ore = Math.floor(minuti / 60);
  return ore > 0 ? ore + " h " + String(minuti % 60).padStart(2, "0") + " min" : minuti + " min";
}

/* "31,2 L" oppure, per un plug-in, "9 kWh + 12,4 L". */
function testoQuantita(fab) {
  const parti = Object.keys(fab)
    .filter(function (k) { return fab[k] >= 0.05; })
    .map(function (k) { return numero1.format(fab[k]) + " " + CARBURANTI[k].unita; });
  return parti.length ? parti.join(" + ") : "0";
}

/* Il prezzo di quello che consuma un'alimentazione, per l'intestazione
   della colonna: "1,749 €/L", per il plug-in anche quello dei kWh. */
function prezzoColonna(alimentazione) {
  const carburanti = alimentazione === "phev"
    ? ["benzina", "elettricita"]
    : [ALIMENTAZIONI[alimentazione].carburante];
  return carburanti.map(function (k) {
    const p = prezzoDi(k);
    return (p > 0 ? euroAlLitro.format(p) : "—") + "/" + CARBURANTI[k].unita;
  }).join(" + ");
}

/* L'unità del consumo della tua auto: "L/100 km" oppure "km/L". */
function unitaConsumo(alimentazione, forma) {
  const u = CARBURANTI[ALIMENTAZIONI[alimentazione].carburante].unita;
  return forma === "km" ? "km/" + u : u + "/100 km";
}

/* Un indirizzo di Nominatim è lunghissimo ("Piazza del Duomo, Centro
   Storico, Municipio 1, Milano, Lombardia, 20122, Italia"): bastano i
   primi tre pezzi. */
function etichettaBreve(risultato) {
  return String(risultato.display_name || "").split(", ").slice(0, 3).join(", ");
}

/* Un punto cliccato, prima che arrivi il suo indirizzo. Le coordinate si
   scrivono sempre con il punto, anche in italiano: è la convenzione. */
function etichettaCoordinate(lat, lon) {
  return lat.toFixed(5) + ", " + lon.toFixed(5);
}

const NOMI_PUNTI = {
  partenza: { it: "Partenza", en: "Start" },
  arrivo:   { it: "Arrivo",   en: "Destination" },
};

const LETTERE_PUNTI = { partenza: "A", arrivo: "B" };


/* --- messaggi ---------------------------------------------------------------- */

/* La riga di stato sotto i campi del percorso. È role="status" nella
   pagina, quindi chi usa un lettore di schermo se la sente annunciare. */
function messaggioStato(testi, avviso) {
  const nodo = $("stato-percorso");
  nodo.textContent = testi ? I18N.t(testi) : "";
  nodo.classList.toggle("field__hint--warn", !!avviso);
}

let attesaMessaggio = null;
function messaggio(testi) {
  const nodo = $("toast");
  nodo.textContent = I18N.t(testi);
  nodo.classList.add("is-on");
  clearTimeout(attesaMessaggio);
  attesaMessaggio = setTimeout(function () { nodo.classList.remove("is-on"); }, 2500);
}

/* Un errore che sa già come spiegarsi a chi usa la pagina. */
function problema(testi) {
  const errore = new Error(testi.en);
  errore.testi = testi;
  return errore;
}


/* --- rete ------------------------------------------------------------------

   Tutte le richieste ai servizi esterni passano da chiedi(). Fa tre cose:

   1. mette in fila le richieste allo stesso server, almeno ATTESA_MIN_MS
      l'una dall'altra, come chiedono le condizioni d'uso;
   2. rinuncia dopo TEMPO_MASSIMO_MS, invece di lasciare la pagina appesa;
   3. trasforma ogni guasto in un problema() con un messaggio leggibile.

   La fila è una catena di promesse per server: ogni richiesta aspetta che
   la precedente sia partita da abbastanza tempo. */

const codaRichieste = {};

function pausa(ms) {
  return new Promise(function (fatto) { setTimeout(fatto, ms); });
}

function turno(host) {
  const precedente = codaRichieste[host] || Promise.resolve(0);
  const mio = precedente.then(function (partitaAlle) {
    const attesa = partitaAlle + ATTESA_MIN_MS - Date.now();
    return (attesa > 0 ? pausa(attesa) : Promise.resolve()).then(function () { return Date.now(); });
  });
  codaRichieste[host] = mio;
  return mio;
}

async function chiedi(url) {
  await turno(new URL(url).host);

  const interrompi = new AbortController();
  const timer = setTimeout(function () { interrompi.abort(); }, TEMPO_MASSIMO_MS);

  let risposta;
  try {
    risposta = await fetch(url, { signal: interrompi.signal, headers: { Accept: "application/json" } });
  } catch (e) {
    throw problema({
      it: "Non riesco a raggiungere il servizio: controlla la connessione e riprova.",
      en: "Cannot reach the service: check your connection and try again.",
    });
  } finally {
    clearTimeout(timer);
  }

  if (risposta.status === 429) {
    throw problema({
      it: "Il servizio ha ricevuto troppe richieste: riprova fra qualche secondo.",
      en: "The service has had too many requests: try again in a few seconds.",
    });
  }
  /* Il server dei percorsi risponde 400 anche quando non trova una strada:
     il corpo dice perché, e leggiPercorso() lo sa interpretare. */
  if (!risposta.ok && risposta.status !== 400) {
    throw problema({
      it: "Il servizio ha risposto con un errore (" + risposta.status + "): riprova più tardi.",
      en: "The service returned an error (" + risposta.status + "): try again later.",
    });
  }

  try {
    return await risposta.json();
  } catch (e) {
    throw problema({
      it: "Il servizio ha risposto qualcosa di illeggibile: riprova più tardi.",
      en: "The service sent back something unreadable: try again later.",
    });
  }
}

function urlRicerca(testo) {
  return SERVIZI.indirizzi + "search?format=jsonv2&limit=5" +
    "&accept-language=" + I18N.lingua + "&q=" + encodeURIComponent(testo);
}

function urlIndirizzo(lat, lon) {
  return SERVIZI.indirizzi + "reverse?format=jsonv2&zoom=16" +
    "&accept-language=" + I18N.lingua + "&lat=" + lat + "&lon=" + lon;
}

function urlPercorso(a, b) {
  const punto = function (p) { return p.lon.toFixed(6) + "," + p.lat.toFixed(6); };
  return SERVIZI.percorso + punto(a) + ";" + punto(b) +
    "?overview=full&geometries=geojson&steps=true";
}


/* --- mappa -------------------------------------------------------------------

   Se Leaflet non si è caricato la variabile L non esiste: la mappa resta
   un riquadro vuoto e il tool funziona scrivendo gli indirizzi. */

let mappa = null;
let stratoPercorso = null;
const segnaposto = { partenza: null, arrivo: null };

function preparaMappa() {
  if (typeof L === "undefined") {
    messaggioStato({
      it: "La mappa non si è caricata: scrivi gli indirizzi nei campi.",
      en: "The map did not load: type the addresses in the fields.",
    }, true);
    return;
  }

  /* La rotella del mouse non zooma finché la mappa non è stata cliccata:
     altrimenti chi scorre la pagina ci finisce dentro e resta incastrato. */
  mappa = L.map("mappa", { scrollWheelZoom: false }).setView(CENTRO_MAPPA, ZOOM_MAPPA);
  mappa.on("focus", function () { mappa.scrollWheelZoom.enable(); });
  mappa.on("blur", function () { mappa.scrollWheelZoom.disable(); });

  L.tileLayer(SERVIZI.tile, { maxZoom: 19, attribution: SERVIZI.crediti }).addTo(mappa);
  mappa.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');

  stratoPercorso = L.layerGroup().addTo(mappa);

  mappa.on("click", function (e) { puntoDallaMappa(bersaglio, e.latlng); });

  ["partenza", "arrivo"].forEach(disegnaSegnaposto);
  inquadraPunti();
}

/* Il segnaposto A o B. È un <div> con una lettera, disegnato da
   .mappa__pin: iconSize null perché Leaflet non gli scriva dimensioni in
   pixel, che sono già nel foglio di stile. */
function disegnaSegnaposto(quale) {
  if (!mappa) return;

  if (segnaposto[quale]) {
    mappa.removeLayer(segnaposto[quale]);
    segnaposto[quale] = null;
  }
  const p = stato[quale];
  if (!p) return;

  const marcatore = L.marker([p.lat, p.lon], {
    icon: L.divIcon({ className: "mappa__pin", html: LETTERE_PUNTI[quale], iconSize: null }),
    draggable: true,
    title: I18N.t(NOMI_PUNTI[quale]),
  });
  marcatore.on("dragend", function () { puntoDallaMappa(quale, marcatore.getLatLng()); });
  marcatore.addTo(mappa);
  segnaposto[quale] = marcatore;
}

/* Porta la mappa sui punti che ci sono. Si chiama dopo una ricerca, non
   dopo un clic: chi clicca sta già guardando dove vuole. */
function inquadraPunti() {
  if (!mappa) return;
  const punti = ["partenza", "arrivo"]
    .map(function (q) { return stato[q]; })
    .filter(Boolean)
    .map(function (p) { return [p.lat, p.lon]; });

  if (punti.length === 2) mappa.fitBounds(punti, { padding: [40, 40], maxZoom: 13 });
  else if (punti.length === 1) mappa.setView(punti[0], 11);
}

/* La linea del percorso: una larga color carta sotto, una color accento
   sopra, così si stacca dalla cartina in tutti e due i temi. */
function disegnaLinea() {
  if (!mappa) return;
  stratoPercorso.clearLayers();
  if (!percorso || !percorso.linea.length) return;

  L.polyline(percorso.linea, { className: "mappa__percorso-fondo", interactive: false }).addTo(stratoPercorso);
  const linea = L.polyline(percorso.linea, { className: "mappa__percorso", interactive: false }).addTo(stratoPercorso);
  mappa.fitBounds(linea.getBounds(), { padding: [32, 32] });
}


/* --- punti A e B ------------------------------------------------------------ */

/* Cambiare un punto rende vecchio il percorso: i costi sparirebbero
   comunque al prossimo Calcola, meglio non lasciarli lì a mentire. */
function invalidaPercorso() {
  const cera = !!percorso;
  percorso = null;
  if (stratoPercorso) stratoPercorso.clearLayers();
  $("risultati").hidden = true;
  if (cera) {
    messaggioStato({
      it: "I punti sono cambiati: premi Calcola per il nuovo percorso.",
      en: "The points have changed: press Calculate for the new route.",
    });
  }
}

function impostaPunto(quale, punto, opzioni) {
  stato[quale] = punto;
  $(quale + "-testo").value = punto ? punto.etichetta : "";
  disegnaSegnaposto(quale);
  invalidaPercorso();
  if (opzioni && opzioni.inquadra) inquadraPunti();
  salvaViaggio();
}

/* Il menu "Intendevi…" sotto un campo, quando la ricerca ha trovato più
   di un posto con quel nome. */
function mostraAlternative(quale) {
  const select = $(quale + "-alternative");
  const elenco = alternative[quale];

  if (elenco.length < 2) {
    select.hidden = true;
    select.innerHTML = "";
    return;
  }
  select.innerHTML =
    '<option value="" disabled selected>' + esc(I18N.t({ it: "Intendevi…", en: "Did you mean…" })) + "</option>" +
    elenco.map(function (p, i) {
      return '<option value="' + i + '">' + esc(p.etichetta) + "</option>";
    }).join("");
  select.hidden = false;
}

function nascondiAlternative(quale) {
  alternative[quale] = [];
  mostraAlternative(quale);
}

/* Cerca il testo scritto in un campo e mette il punto sul primo
   risultato; gli altri finiscono nel menu "Intendevi…". */
async function cercaIndirizzo(quale) {
  const testo = $(quale + "-testo").value.trim();
  if (!testo) {
    throw problema(quale === "partenza"
      ? { it: "Scrivi da dove parti, o clicca sulla mappa.", en: "Type where you start from, or click on the map." }
      : { it: "Scrivi dove arrivi, o clicca sulla mappa.",   en: "Type where you are going, or click on the map." });
  }

  const risultati = await chiedi(urlRicerca(testo));
  if (!Array.isArray(risultati) || !risultati.length) {
    throw problema({
      it: "Non trovo «" + testo + "»: prova ad aggiungere la città o la provincia.",
      en: "Cannot find “" + testo + "”: try adding the town or the region.",
    });
  }

  alternative[quale] = risultati.map(function (r) {
    return { lat: num(r.lat), lon: num(r.lon), etichetta: etichettaBreve(r) };
  });
  impostaPunto(quale, alternative[quale][0], { inquadra: true });
  mostraAlternative(quale);
}

/* Un punto scelto sulla mappa, con un clic o trascinando un segnaposto.
   Il campo mostra subito le coordinate; il nome della via arriva dopo,
   se il servizio risponde. Se nel frattempo il punto è cambiato di nuovo,
   la risposta vecchia si butta. */
async function puntoDallaMappa(quale, latlng) {
  const punto = {
    lat: Number(latlng.lat.toFixed(6)),
    lon: Number(latlng.lng.toFixed(6)),
    etichetta: etichettaCoordinate(latlng.lat, latlng.lng),
  };
  nascondiAlternative(quale);
  impostaPunto(quale, punto);

  /* Dopo la partenza il clic successivo va all'arrivo */
  if (quale === "partenza" && !stato.arrivo) scegliBersaglio("arrivo");

  try {
    const r = await chiedi(urlIndirizzo(punto.lat, punto.lon));
    if (stato[quale] !== punto || !r || !r.display_name) return;
    punto.etichetta = etichettaBreve(r);
    $(quale + "-testo").value = punto.etichetta;
    salvaViaggio();
  } catch (e) {
    /* resta il nome fatto di coordinate: va bene lo stesso */
  }
}

function scegliBersaglio(quale) {
  bersaglio = quale;
  $("clic-partenza").setAttribute("aria-pressed", String(quale === "partenza"));
  $("clic-arrivo").setAttribute("aria-pressed", String(quale === "arrivo"));
}

function inverti() {
  const a = stato.partenza;
  const b = stato.arrivo;
  const testoA = $("partenza-testo").value;
  const testoB = $("arrivo-testo").value;

  stato.partenza = b;
  stato.arrivo = a;
  $("partenza-testo").value = testoB;
  $("arrivo-testo").value = testoA;
  nascondiAlternative("partenza");
  nascondiAlternative("arrivo");
  disegnaSegnaposto("partenza");
  disegnaSegnaposto("arrivo");
  invalidaPercorso();
  salvaViaggio();
}


/* --- Calcola ------------------------------------------------------------------ */

async function calcolaPercorso() {
  if (inCorso) return;
  inCorso = true;
  $("calcola").disabled = true;

  try {
    /* Un campo scritto ma non ancora cercato si cerca adesso */
    for (const quale of ["partenza", "arrivo"]) {
      if (!stato[quale]) {
        messaggioStato({ it: "Cerco gli indirizzi…", en: "Looking up the addresses…" });
        await cercaIndirizzo(quale);
      }
    }

    messaggioStato({ it: "Calcolo il percorso…", en: "Working out the route…" });
    const nuovo = leggiPercorso(await chiedi(urlPercorso(stato.partenza, stato.arrivo)));
    if (!nuovo) {
      throw problema({
        it: "Non trovo una strada in auto fra questi due punti.",
        en: "There is no road route between these two points.",
      });
    }

    percorso = nuovo;
    disegnaLinea();
    aggiorna();
    messaggioStato(null);
    $("risultati").scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (e) {
    messaggioStato(e.testi || {
      it: "Qualcosa è andato storto: riprova.",
      en: "Something went wrong: try again.",
    }, true);
  } finally {
    inCorso = false;
    $("calcola").disabled = false;
  }
}


/* --- costruzione delle parti fisse --------------------------------------------

   Come in grigliata: le tabelle con dei campi si costruiscono una volta
   sola. Poi si aggiorna solo il testo intorno, perché ricostruire un campo
   mentre ci si scrive dentro gli farebbe perdere il cursore. */

function costruisciPrezzi() {
  document.querySelector("#tab-prezzi tbody").innerHTML = Object.keys(CARBURANTI).map(function (k) {
    const carb = CARBURANTI[k];
    return "<tr>" +
      '<td class="table__name">' + esc(I18N.t(carb.nome)) + "</td>" +
      '<td class="table__num">' +
        '<input class="field__input field__input--compact field__input--num field__input--sm"' +
        ' type="text" inputmode="decimal" data-campo="prezzo.' + k + '"' +
        ' value="' + perCampo(prezzoDi(k)) + '"' +
        ' aria-label="' + esc(I18N.t(carb.nome)) + '"></td>' +
      '<td><span class="table__unit">€/' + carb.unita + "</span></td>" +
      '<td><span class="table__unit" id="origine-' + k + '"></span></td>' +
      '<td class="table__num"><button class="btn btn--icon" type="button"' +
        ' data-azione="ripristina-prezzo" data-carburante="' + k + '" id="ripristina-' + k + '"' +
        ' title="' + esc(I18N.t({ it: "Torna al prezzo di riferimento", en: "Back to the reference price" })) + '">✕</button></td>' +
    "</tr>";
  }).join("");
}

/* Accanto a ogni prezzo, da dove arriva. Quando arrivano i prezzi
   ufficiali si aggiornano anche i campi che nessuno ha toccato — salvo
   quello in cui si sta scrivendo, per non spostare il cursore. */
function aggiornaPrezzi() {
  Object.keys(CARBURANTI).forEach(function (k) {
    const origine = origineDi(k);
    let testo;
    if (origine === "mano") {
      testo = I18N.t({ it: "scritto a mano", en: "set by hand" });
    } else if (origine === "ufficiale") {
      const data = /^\d{4}-\d{2}-\d{2}$/.test(ufficiali.aggiornato || "")
        ? " · " + dataMedia.format(new Date(ufficiali.aggiornato + "T12:00:00"))
        : "";
      testo = "MIMIT" + data;
    } else {
      testo = I18N.t({ it: "valore indicativo", en: "rough estimate" });
    }
    $("origine-" + k).textContent = testo;
    $("ripristina-" + k).hidden = origine !== "mano";

    const campo = document.querySelector('[data-campo="prezzo.' + k + '"]');
    if (campo && origine !== "mano" && document.activeElement !== campo) {
      campo.value = perCampo(prezzoDi(k));
    }
  });
}

/* Due gruppi di chip, uno per i tipi di auto e uno per le alimentazioni.
   Lo stato acceso/spento è aria-pressed, come nel resto del sito. */
function costruisciScelte() {
  const chip = function (gruppo, dizionario, scelti) {
    return Object.keys(dizionario).map(function (k) {
      return '<button class="chip" type="button" data-scelta="' + gruppo + '" data-chiave="' + k + '"' +
             ' aria-pressed="' + (scelti.indexOf(k) !== -1) + '">' +
             esc(I18N.t(dizionario[k].nome)) + "</button>";
    }).join("");
  };
  $("scelta-tipi").innerHTML = chip("tipi", TIPI, stato.tipi);
  $("scelta-alimentazioni").innerHTML = chip("alimentazioni", ALIMENTAZIONI, stato.alimentazioni);
}

/* Le unità tra cui scegliere dipendono dall'alimentazione: un'elettrica
   non si misura in litri. */
function costruisciFormeConsumo() {
  const t = stato.tuaAuto;
  $("tua-forma").innerHTML = FORME_CONSUMO.map(function (f) {
    return '<option value="' + f + '"' + (f === t.forma ? " selected" : "") + ">" +
           esc(unitaConsumo(t.alimentazione, f)) + "</option>";
  }).join("");
}

function costruisciTuaAuto() {
  const t = stato.tuaAuto;
  $("tua-attiva").checked = t.attiva;
  $("tua-alimentazione").innerHTML = ALIMENTAZIONI_TUA_AUTO.map(function (a) {
    return '<option value="' + a + '"' + (a === t.alimentazione ? " selected" : "") + ">" +
           esc(I18N.t(ALIMENTAZIONI[a].nome)) + "</option>";
  }).join("");
  $("tua-consumo").value = t.consumo > 0 ? perCampo(t.consumo) : "";
  costruisciFormeConsumo();
}

function riempiCampiFissi() {
  $("autonomia-phev").value = perCampo(stato.autonomiaPhev);
  $("pedaggio-tariffa").value = perCampo(stato.pedaggioTariffa);
  $("pedaggio-incluso").checked = stato.pedaggioIncluso;
  $("partenza-testo").value = stato.partenza ? stato.partenza.etichetta : "";
  $("arrivo-testo").value = stato.arrivo ? stato.arrivo.etichetta : "";
  costruisciTuaAuto();
  scegliBersaglio(bersaglio);
}


/* --- ricalcolo e rendering --------------------------------------------------- */

function scrivi(id, testo) {
  const nodo = $(id);
  if (nodo) nodo.textContent = testo;
}

function nomeRiga(chiave) {
  return chiave === "tua"
    ? I18N.t({ it: "La tua auto", en: "Your car" })
    : I18N.t(TIPI[chiave].nome);
}

function aggiorna() {
  aggiornaPrezzi();

  if (!percorso) {
    $("risultati").hidden = true;
    return;
  }

  const r = calcola(percorso, stato, prezziCorrenti());

  /* --- il sommario in cima --- */
  scrivi("tot-distanza", km(r.riepilogo.totale));
  scrivi("tot-durata", durata(percorso.secondi));
  if (r.migliore) {
    scrivi("tot-migliore", euro.format(r.migliore.costo));
    scrivi("nota-migliore", nomeRiga(r.migliore.riga) + " · " + I18N.t(ALIMENTAZIONI[r.migliore.colonna].nome));
  } else {
    scrivi("tot-migliore", "—");
    scrivi("nota-migliore", "");
  }
  scrivi("tot-pedaggio", euro.format(r.pedaggio));
  scrivi("nota-pedaggio",
    km(r.riepilogo.pedaggio) + " " + I18N.t({ it: "a pedaggio, stimati", en: "tolled, estimated" }) +
    (r.pedaggioIncluso ? "" : " · " + I18N.t({ it: "fuori dai costi", en: "not in the costs" })));

  disegnaRegimi(r.riepilogo);
  disegnaConfronto(r);
  disegnaAvvisi(r);

  $("risultati").hidden = false;
}

/* Quanti km su ogni tipo di strada. La barra è la quota del viaggio. */
function disegnaRegimi(riep) {
  const voci = ["citta", "extra", "autostrada", "traghetto"].filter(function (k) {
    return k !== "traghetto" || riep.traghetto > 0;
  });

  $("barre-regimi").innerHTML = voci.map(function (k) {
    const quota = riep.totale > 0 ? riep[k] / riep.totale * 100 : 0;
    return '<div class="bar">' +
             "<div>" + esc(I18N.t(REGIMI[k])) + "</div>" +
             '<div class="bar__track"><div class="bar__fill" data-quota="' + quota.toFixed(1) + '"></div></div>' +
             '<div class="bar__value">' + km(riep[k]) + " · " + numero0.format(quota) + "%</div>" +
           "</div>";
  }).join("");

  /* La lunghezza della barra entra nel CSS come proprietà --quota,
     non come attributo style scritto a mano. */
  $("barre-regimi").querySelectorAll(".bar__fill").forEach(function (n) {
    n.style.setProperty("--quota", n.dataset.quota + "%");
  });
}

/* La tabella del confronto. La prima intestazione ("Auto") è scritta
   nella pagina; le altre dipendono dalle alimentazioni scelte e si
   rifanno ogni volta. */
function disegnaConfronto(r) {
  const tabella = $("tab-confronto");
  const intestazione = tabella.querySelector("thead tr");
  intestazione.querySelectorAll("th[data-colonna]").forEach(function (n) { n.remove(); });
  intestazione.insertAdjacentHTML("beforeend", r.colonne.map(function (a) {
    return '<th class="table__num" data-colonna="' + a + '">' + esc(I18N.t(ALIMENTAZIONI[a].nome)) +
           '<br><span class="table__unit">' + esc(prezzoColonna(a)) + "</span></th>";
  }).join(""));

  const riga = function (voce, nome) {
    return "<tr>" +
      '<td class="table__name">' + nome + "</td>" +
      r.colonne.map(function (a) {
        const x = voce.celle[a];
        if (!x) return '<td class="table__num">—</td>';
        if (x.costo == null) {
          return '<td class="table__num" title="' + esc(I18N.t(x.errore)) + '">—</td>';
        }
        let classi = "table__num";
        if (voce.valide > 1 && voce.minimo === a) classi += " table__pick";
        if (r.migliore && r.migliore.riga === voce.chiave && r.migliore.colonna === a) classi += " table__best";
        return '<td class="' + classi + '">' + euro.format(x.costo) +
               '<br><span class="table__unit">' + esc(testoQuantita(x.fabbisogno)) + "</span></td>";
      }).join("") +
    "</tr>";
  };

  let corpi = "<tbody>" + r.righe
    .filter(function (v) { return v.chiave !== "tua"; })
    .map(function (v) { return riga(v, esc(nomeRiga(v.chiave))); })
    .join("") + "</tbody>";

  /* La tua auto sta in un gruppo a parte, con il consumo dichiarato
     scritto accanto: è un dato suo, non una stima del sito. */
  const tua = r.righe.find(function (v) { return v.chiave === "tua"; });
  if (tua) {
    const t = stato.tuaAuto;
    corpi += "<tbody>" +
      '<tr class="table__group"><th colspan="' + (r.colonne.length + 1) + '">' +
        esc(nomeRiga("tua")) + "</th></tr>" +
      riga(tua, esc(I18N.t(ALIMENTAZIONI[t.alimentazione].nome)) +
        '<br><span class="table__unit">' + esc(numero1.format(num(t.consumo)) + " " +
        unitaConsumo(t.alimentazione, t.forma)) + "</span>") +
      "</tbody>";
  }

  tabella.querySelectorAll("tbody").forEach(function (n) { n.remove(); });
  tabella.insertAdjacentHTML("beforeend", corpi);
}

/* Le cose da sapere per leggere bene la tabella: prezzi mancanti e
   traghetti. Se non c'è niente da dire, il riquadro sparisce. */
function disegnaAvvisi(r) {
  const voci = [];

  if (r.prezziMancanti.length) {
    voci.push(I18N.t({
      it: "Manca il prezzo di: " + r.prezziMancanti.map(function (k) { return CARBURANTI[k].nome.it; }).join(", ") +
          ". Le celle che lo usano restano vuote.",
      en: "Price missing for: " + r.prezziMancanti.map(function (k) { return CARBURANTI[k].nome.en; }).join(", ") +
          ". The cells that need it are left empty.",
    }));
  }
  if (r.riepilogo.traghetto > 0) {
    voci.push(I18N.t({
      it: "Il percorso comprende " + km(r.riepilogo.traghetto) + " di traghetto: non consumano carburante, " +
          "ma il biglietto non è compreso nei costi.",
      en: "The route includes " + km(r.riepilogo.traghetto) + " by ferry: they use no fuel, " +
          "but the ticket is not included in the costs.",
    }));
  }

  $("avvisi-viaggio").hidden = !voci.length;
  $("avvisi-viaggio").innerHTML = voci.map(function (v) { return "<p>" + esc(v) + "</p>"; }).join("");
}


/* --- eventi ------------------------------------------------------------------

   Un solo ascoltatore per tipo di evento sul documento, come in
   grigliata: le tabelle si ricostruiscono e non c'è niente da
   riagganciare ogni volta. */

document.addEventListener("input", function (e) {
  const bersaglioEvento = e.target;

  /* Scrivere in un campo indirizzo rende vecchio il punto che c'era:
     il nuovo testo va cercato. */
  if (bersaglioEvento.id === "partenza-testo" || bersaglioEvento.id === "arrivo-testo") {
    const quale = bersaglioEvento.id === "partenza-testo" ? "partenza" : "arrivo";
    if (stato[quale]) {
      stato[quale] = null;
      disegnaSegnaposto(quale);
      invalidaPercorso();
      salvaViaggio();
    }
    nascondiAlternative(quale);
    return;
  }

  const campo = bersaglioEvento.dataset ? bersaglioEvento.dataset.campo : null;
  if (!campo) return;

  const parti = campo.split(".");
  const valore = bersaglioEvento.type === "checkbox" ? bersaglioEvento.checked : bersaglioEvento.value;

  if (parti[0] === "prezzo") {
    stato.prezzi[parti[1]] = num(valore);

  } else if (parti[0] === "tuaAuto") {
    const nome = parti[1];
    stato.tuaAuto[nome] = nome === "consumo" ? num(valore) : valore;
    if (nome === "alimentazione") costruisciFormeConsumo();

  } else if (campo === "pedaggioIncluso") {
    stato.pedaggioIncluso = valore;

  } else if (campo === "autonomiaPhev" || campo === "pedaggioTariffa") {
    stato[campo] = Math.max(0, num(valore));
  }

  aggiorna();
  salvaViaggio();
});

/* Il menu "Intendevi…": sposta il punto su un altro risultato. */
document.addEventListener("change", function (e) {
  const id = e.target.id;
  if (id !== "partenza-alternative" && id !== "arrivo-alternative") return;

  const quale = id === "partenza-alternative" ? "partenza" : "arrivo";
  const scelta = alternative[quale][Number(e.target.value)];
  if (!scelta) return;
  impostaPunto(quale, scelta, { inquadra: true });
});

/* Invio in un campo indirizzo vale come "Cerca". */
document.addEventListener("keydown", function (e) {
  if (e.key !== "Enter") return;
  if (e.target.id !== "partenza-testo" && e.target.id !== "arrivo-testo") return;
  e.preventDefault();
  cercaDaCampo(e.target.id === "partenza-testo" ? "partenza" : "arrivo");
});

async function cercaDaCampo(quale) {
  messaggioStato({ it: "Cerco l'indirizzo…", en: "Looking up the address…" });
  try {
    await cercaIndirizzo(quale);
    messaggioStato(null);
    if (quale === "partenza" && !stato.arrivo) scegliBersaglio("arrivo");
  } catch (err) {
    messaggioStato(err.testi || { it: "Ricerca non riuscita.", en: "Search failed." }, true);
  }
}

document.addEventListener("click", function (e) {
  /* I chip delle scelte: almeno uno per gruppo deve restare acceso,
     altrimenti la tabella non avrebbe righe o colonne. */
  const chip = e.target.closest("[data-scelta]");
  if (chip) {
    const gruppo = chip.dataset.scelta;
    const chiave = chip.dataset.chiave;
    const scelti = stato[gruppo];
    const indice = scelti.indexOf(chiave);

    if (indice !== -1 && scelti.length === 1) {
      messaggio(gruppo === "tipi"
        ? { it: "Serve almeno un tipo di auto.", en: "At least one type of car is needed." }
        : { it: "Serve almeno un'alimentazione.", en: "At least one fuel type is needed." });
      return;
    }
    if (indice === -1) {
      /* Si rimette nell'ordine del dizionario, che è l'ordine della tabella */
      const ordine = Object.keys(gruppo === "tipi" ? TIPI : ALIMENTAZIONI);
      stato[gruppo] = ordine.filter(function (k) { return k === chiave || scelti.indexOf(k) !== -1; });
    } else {
      stato[gruppo] = scelti.filter(function (k) { return k !== chiave; });
    }
    chip.setAttribute("aria-pressed", String(indice === -1));
    aggiorna();
    salvaViaggio();
    return;
  }

  const bottone = e.target.closest("[data-azione]");
  if (!bottone) return;
  const azione = bottone.dataset.azione;

  if (azione === "cerca") {
    cercaDaCampo(bottone.dataset.punto);

  } else if (azione === "bersaglio") {
    scegliBersaglio(bottone.dataset.punto);

  } else if (azione === "inverti") {
    inverti();

  } else if (azione === "calcola") {
    calcolaPercorso();

  } else if (azione === "ripristina-prezzo") {
    const k = bottone.dataset.carburante;
    delete stato.prezzi[k];
    const campo = document.querySelector('[data-campo="prezzo.' + k + '"]');
    if (campo) campo.value = perCampo(prezzoDi(k));
    aggiorna();
    salvaViaggio();
  }
});


/* --- avvio ---------------------------------------------------------------------- */

costruisciPrezzi();
costruisciScelte();
riempiCampiFissi();
preparaMappa();
aggiorna();

/* I prezzi ufficiali arrivano dopo, senza fermare niente. Se il file non
   c'è — sito appena pubblicato, o aperto senza server — restano i valori
   indicativi e la pagina funziona lo stesso. */
fetch(PREZZI_UFFICIALI_URL, { cache: "no-cache" })
  .then(function (r) { return r.ok ? r.json() : null; })
  .then(function (dati) {
    if (!dati || !dati.prezzi) return;
    ufficiali = dati;
    aggiorna();
  })
  .catch(function () { /* nessun file: valori indicativi */ });
