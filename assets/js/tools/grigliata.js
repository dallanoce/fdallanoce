/* ==========================================================================
   GRIGLIATA — quanto comprare e quanto si spende

   Struttura del file:
     1. PARAMETRI   — catalogo, categorie e unità di misura, isolati in cima
     2. CALCOLO     — funzioni pure: stato dentro, risultato fuori
     3. INTERFACCIA — costruzione delle tabelle, rendering, eventi

   DUE LINGUE, UN FILE SOLO
   Questo file serve sia /it/strumenti/grigliata/ sia /en/tools/barbecue/.
   Le due pagine hanno gli stessi id nei campi proprio per questo: la
   formula esiste una volta sola.

   PERCHÉ IL CATALOGO HA UNA CHIAVE E NON SOLO UN NOME
   La configurazione si salva in localStorage, che è dello stesso dominio
   per tutti e due gli alberi: chi imposta la grigliata in italiano e poi
   passa all'inglese ritrova la sua lista. Se in memoria finissero i nomi
   già tradotti, dopo il cambio lingua si ritroverebbe una lista metà e
   metà. Quindi in memoria va la chiave ("salamelle") e il nome si traduce
   al momento di disegnarlo. Un prodotto aggiunto o rinominato dall'utente
   non ha chiave: quello che ha scritto vale così com'è, in tutte e due le
   lingue, perché è roba sua e non del sito.
   ========================================================================== */


/* --------------------------------------------------------------------------
   1. PARAMETRI
   -------------------------------------------------------------------------- */

const CHIAVE_ARCHIVIO = "grigliata";   /* dove si salva la configurazione */

/* Le categorie sono chiavi stabili con etichetta tradotta. L'ordine di
   questo oggetto è l'ordine in cui compaiono ovunque nella pagina. */
const CATEGORIE = {
  carne:       { it: "Carne",               en: "Meat" },
  verdure:     { it: "Verdure e contorni",  en: "Vegetables and sides" },
  pane:        { it: "Pane e formaggi",     en: "Bread and cheese" },
  alcolici:    { it: "Bevande alcoliche",   en: "Alcoholic drinks" },
  analcolici:  { it: "Bevande analcoliche", en: "Soft drinks" },
  consumabili: { it: "Consumabili",         en: "Consumables" },
};

/* Come cresce la quantità al crescere delle persone.
   "cibo", "alcol" e "soft" seguono le tre percentuali del gruppo: un
   bambino al 50% di cibo pesa mezza porzione. "testa" conta le teste e
   basta (la carbonella non sa chi è astemio). "fisso" non scala. */
const SCALE = {
  cibo:  { it: "Quota cibo",        en: "Food share" },
  alcol: { it: "Quota alcolici",    en: "Alcohol share" },
  soft:  { it: "Quota analcolici",  en: "Soft-drink share" },
  testa: { it: "Uguale per tutti",  en: "Same for everyone" },
  fisso: { it: "Quantità fissa",    en: "Fixed amount" },
};

/* In che cosa si misura un prodotto, e quindi come si compra. */
const UNITA = {
  pz: { it: "pezzi", en: "pieces" },
  kg: { it: "kg",    en: "kg" },
  L:  { it: "litri", en: "litres" },
};

/* La stessa unità in forma breve, per quando segue un numero. Il nome per
   esteso andrebbe accordato al singolare — "1 pezzi" non si può leggere — e
   l'abbreviazione toglie il problema invece di risolverlo con un plurale
   calcolato in due lingue. */
const UNITA_BREVI = {
  pz: { it: "pz", en: "pcs" },
  kg: { it: "kg", en: "kg" },
  L:  { it: "L",  en: "L" },
};

/* In che cosa è espresso il prezzo di listino. */
const PREZZO_UNITA = {
  kg:   { it: "€ al kg",         en: "€ per kg" },
  pz:   { it: "€ al pezzo",      en: "€ per piece" },
  L:    { it: "€ al litro",      en: "€ per litre" },
  conf: { it: "€ a confezione",  en: "€ per pack" },
};

/* Che cosa contiene una confezione, quando il prezzo è a confezione. */
const CONF_UNITA = {
  pz: { it: "pezzi", en: "pieces" },
  kg: { it: "kg",    en: "kg" },
  g:  { it: "g",     en: "g" },
  L:  { it: "litri", en: "litres" },
  ml: { it: "ml",    en: "ml" },
};

/* --- Gruppi di partenza ---------------------------------------------------
   Le tre percentuali sono il consumo del gruppo rispetto a un adulto. */

const NOMI_GRUPPI = {
  adulti:  { it: "Adulti",              en: "Adults" },
  bambini: { it: "Bambini",             en: "Children" },
  guida:   { it: "Astemi e chi guida",  en: "Non-drinkers and drivers" },
};

const GRUPPI_PREDEFINITI = [
  { chiave: "adulti",  n: 8, cibo: 100, alcol: 100, soft: 100 },
  { chiave: "bambini", n: 3, cibo: 50,  alcol: 0,   soft: 150 },
  { chiave: "guida",   n: 2, cibo: 100, alcol: 0,   soft: 150 },
];

/* --- Catalogo di partenza -------------------------------------------------

   Consumi e prezzi sono un punto di partenza plausibile per una grigliata
   in Italia, non un listino: si cambiano dalla pagina e restano salvati.

   Qualche nome resta in italiano anche in inglese perché è il nome della
   cosa (salamelle, pancetta, scamorza): tradurlo direbbe qualcosa di
   diverso da quello che si compra davvero. */

function prodotto(chiave, etichetta, categoria, campi) {
  return Object.assign({
    chiave: chiave,
    etichetta: etichetta,
    categoria: categoria,
    attivo: true,
    scala: "cibo",
    unita: "kg",
    pezzoQta: 0,      /* quanto pesa un pezzo, se il prodotto si conta a pezzi */
    pezzoUnita: "g",
    consumo: 0,       /* quanto ne consuma un adulto medio */
    prezzo: 0,
    prezzoUnita: "kg",
    confContenuto: 1, /* quanto c'è in una confezione, se il prezzo è a confezione */
    confUnita: "pz",
  }, campi);
}

const CATALOGO = [
  /* --- Carne: quantità legata alla quota "cibo" --- */
  prodotto("salamelle",  { it: "Salamelle",           en: "Salamelle" },            "carne", { unita: "pz", pezzoQta: 90,  consumo: 3,    prezzo: 8.50,  prezzoUnita: "kg" }),
  prodotto("salsicce",   { it: "Salsicce",            en: "Sausages" },             "carne", { unita: "pz", pezzoQta: 110, consumo: 1,    prezzo: 9.00,  prezzoUnita: "kg" }),
  prodotto("spiedini",   { it: "Spiedini",            en: "Skewers" },              "carne", { unita: "pz", pezzoQta: 90,  consumo: 2,    prezzo: 13.00, prezzoUnita: "kg" }),
  prodotto("braciole",   { it: "Braciole di maiale",  en: "Pork chops" },           "carne", { unita: "kg", consumo: 0.15, prezzo: 9.50 }),
  prodotto("costine",    { it: "Costine",             en: "Pork ribs" },            "carne", { unita: "kg", consumo: 0.20, prezzo: 8.00, attivo: false }),
  prodotto("pancetta",   { it: "Pancetta",            en: "Pancetta" },             "carne", { unita: "kg", consumo: 0.08, prezzo: 11.00 }),
  prodotto("pollo",      { it: "Pollo (ali e cosce)", en: "Chicken (wings, legs)" }, "carne", { unita: "kg", consumo: 0.15, prezzo: 6.50, attivo: false }),
  prodotto("hamburger",  { it: "Hamburger",           en: "Burgers" },              "carne", { unita: "pz", pezzoQta: 150, consumo: 1,    prezzo: 12.00, prezzoUnita: "kg", attivo: false }),
  prodotto("wurstel",    { it: "Würstel",             en: "Frankfurters" },         "carne", { unita: "pz", pezzoQta: 50,  consumo: 2,    prezzo: 7.00,  prezzoUnita: "kg", attivo: false }),

  /* --- Verdure e contorni --- */
  prodotto("zucchine",   { it: "Zucchine",  en: "Courgettes" }, "verdure", { unita: "kg", consumo: 0.10, prezzo: 2.50 }),
  prodotto("melanzane",  { it: "Melanzane", en: "Aubergines" }, "verdure", { unita: "kg", consumo: 0.10, prezzo: 2.80, attivo: false }),
  prodotto("peperoni",   { it: "Peperoni",  en: "Peppers" },    "verdure", { unita: "kg", consumo: 0.08, prezzo: 3.50, attivo: false }),
  prodotto("patate",     { it: "Patate",    en: "Potatoes" },   "verdure", { unita: "kg", consumo: 0.15, prezzo: 1.80 }),
  prodotto("insalata",   { it: "Insalata",  en: "Salad" },      "verdure", { unita: "kg", consumo: 0.08, prezzo: 3.00, attivo: false }),

  /* --- Pane e formaggi --- */
  prodotto("pane",       { it: "Pane",                 en: "Bread" },           "pane", { unita: "kg", consumo: 0.15, prezzo: 3.50 }),
  prodotto("scamorza",   { it: "Scamorza o provola",   en: "Scamorza cheese" }, "pane", { unita: "pz", pezzoQta: 250, consumo: 0.3, prezzo: 9.00, prezzoUnita: "kg" }),
  prodotto("formaggio",  { it: "Formaggio da griglia", en: "Grilling cheese" }, "pane", { unita: "kg", consumo: 0.06, prezzo: 12.00, attivo: false }),

  /* --- Alcolici: quantità legata alla quota "alcol" --- */
  prodotto("birra",      { it: "Birra (bottiglie da 66 cl)", en: "Beer (66 cl bottles)" }, "alcolici",
           { scala: "alcol", unita: "pz", pezzoQta: 660, pezzoUnita: "ml", consumo: 1.5,
             prezzo: 15.00, prezzoUnita: "conf", confContenuto: 12, confUnita: "pz" }),
  prodotto("vinoRosso",  { it: "Vino rosso",  en: "Red wine" },   "alcolici", { scala: "alcol", unita: "L", consumo: 0.30, prezzo: 5.00, prezzoUnita: "L" }),
  prodotto("vinoBianco", { it: "Vino bianco", en: "White wine" }, "alcolici", { scala: "alcol", unita: "L", consumo: 0.20, prezzo: 5.00, prezzoUnita: "L", attivo: false }),

  /* --- Analcolici: quantità legata alla quota "soft" --- */
  prodotto("acqua",      { it: "Acqua",                    en: "Water" },                        "analcolici", { scala: "soft", unita: "L",  consumo: 0.75, prezzo: 0.30, prezzoUnita: "L" }),
  prodotto("bibite",     { it: "Bibite (cola, aranciata)", en: "Soft drinks (cola, orange)" },   "analcolici", { scala: "soft", unita: "L",  consumo: 0.40, prezzo: 1.20, prezzoUnita: "L" }),
  prodotto("ghiaccio",   { it: "Ghiaccio",                 en: "Ice" },                          "analcolici", { scala: "soft", unita: "kg", consumo: 0.15, prezzo: 1.50, attivo: false }),

  /* --- Consumabili: si contano a testa o sono quantità fisse --- */
  prodotto("carbonella",   { it: "Carbonella",             en: "Charcoal" },          "consumabili", { scala: "testa", unita: "kg", consumo: 0.25, prezzo: 1.20 }),
  prodotto("accendifuoco", { it: "Accendifuoco",           en: "Firelighters" },      "consumabili", { scala: "fisso", unita: "pz", consumo: 1,    prezzo: 3.00, prezzoUnita: "pz" }),
  prodotto("condimenti",   { it: "Sale, olio, condimenti", en: "Salt, oil, dressing" }, "consumabili", { scala: "fisso", unita: "pz", consumo: 1,  prezzo: 8.00, prezzoUnita: "pz" }),
  prodotto("piatti",       { it: "Piatti",                 en: "Plates" },            "consumabili", { scala: "testa", unita: "pz", consumo: 2,    prezzo: 0.08, prezzoUnita: "pz" }),
  prodotto("bicchieri",    { it: "Bicchieri",              en: "Cups" },              "consumabili", { scala: "testa", unita: "pz", consumo: 3,    prezzo: 0.05, prezzoUnita: "pz" }),
  prodotto("posate",       { it: "Posate (set)",           en: "Cutlery (sets)" },    "consumabili", { scala: "testa", unita: "pz", consumo: 1.5,  prezzo: 0.10, prezzoUnita: "pz" }),
  prodotto("tovaglioli",   { it: "Tovaglioli",             en: "Napkins" },           "consumabili", { scala: "testa", unita: "pz", consumo: 4,    prezzo: 0.02, prezzoUnita: "pz" }),
  prodotto("alluminio",    { it: "Alluminio (rotolo)",     en: "Foil (roll)" },       "consumabili", { scala: "fisso", unita: "pz", consumo: 1,    prezzo: 2.50, prezzoUnita: "pz" }),
  prodotto("sacchi",       { it: "Sacchi per i rifiuti",   en: "Bin bags" },          "consumabili", { scala: "fisso", unita: "pz", consumo: 1,    prezzo: 2.00, prezzoUnita: "pz" }),
];

/* Le etichette del catalogo, raggiunte per chiave: servono a tradurre un
   prodotto letto da localStorage, dove c'è la chiave e non il nome. */
const NOMI_PRODOTTI = {};
CATALOGO.forEach(function (p) { NOMI_PRODOTTI[p.chiave] = p.etichetta; });

/* Il margine di sicurezza di partenza, in percentuale. */
const MARGINE_PREDEFINITO = 10;


/* --------------------------------------------------------------------------
   2. CALCOLO — nessun riferimento al DOM da qui alla fine della sezione
   -------------------------------------------------------------------------- */

/* Accetta sia la virgola sia il punto come separatore decimale: chi scrive
   "0,15" e chi scrive "0.15" intende la stessa cosa. */
function num(v) {
  const n = parseFloat(String(v).replace(",", "."));
  return isFinite(n) ? n : 0;
}

/* Quante persone e quante "quote" di consumo ci sono in tutto.
   Otto adulti e tre bambini al 50% fanno 11 persone ma 9,5 quote di cibo. */
function totali(st) {
  let persone = 0, cibo = 0, alcol = 0, soft = 0;

  st.gruppi.forEach(function (g) {
    const n = num(g.n);
    persone += n;
    cibo  += n * num(g.cibo)  / 100;
    alcol += n * num(g.alcol) / 100;
    soft  += n * num(g.soft)  / 100;
  });

  return { persone: persone, cibo: cibo, alcol: alcol, soft: soft };
}

/* Per quante quote va moltiplicato il consumo di questo prodotto. */
function bacino(prod, t) {
  switch (prod.scala) {
    case "cibo":  return t.cibo;
    case "alcol": return t.alcol;
    case "soft":  return t.soft;
    case "testa": return t.persone;
    default:      return 1;            /* fisso: la quantità non dipende da nessuno */
  }
}

/* Quanta parte del costo di questo prodotto tocca a questo gruppo (0..1).
   È la stessa proporzione del bacino, vista da un gruppo solo: chi non beve
   alcolici non compare nel bacino "alcol" e quindi non paga le birre. */
function quotaGruppo(prod, g, t) {
  const n = num(g.n);
  switch (prod.scala) {
    case "cibo":  return t.cibo  > 0 ? n * num(g.cibo)  / 100 / t.cibo  : 0;
    case "alcol": return t.alcol > 0 ? n * num(g.alcol) / 100 / t.alcol : 0;
    case "soft":  return t.soft  > 0 ? n * num(g.soft)  / 100 / t.soft  : 0;
    default:      return t.persone > 0 ? n / t.persone : 0;   /* testa e fisso: pro capite */
  }
}

/* Una quantità Q, espressa nell'unità del prodotto, tradotta nelle tre
   dimensioni possibili. Serve per incrociarla con il prezzo, che può essere
   espresso in un'altra: 39 salamelle da 90 g sono anche 3,51 kg. */
function quantita(prod, Q) {
  const fuori = { pz: null, kg: null, L: null };

  if (prod.unita === "pz") {
    fuori.pz = Q;
    const peso = num(prod.pezzoQta);
    if (peso > 0) {
      if (prod.pezzoUnita === "g") fuori.kg = Q * peso / 1000;
      else                         fuori.L  = Q * peso / 1000;
    }
  } else if (prod.unita === "kg") {
    fuori.kg = Q;
  } else {
    fuori.L = Q;
  }

  return fuori;
}

/* Il prezzo ridotto a "quanto costa una unità", con l'indicazione di quale
   unità sia. Un prezzo a confezione si divide per il contenuto.
   Restituisce null quando il dato non basta. */
function prezzoUnitario(prod) {
  const prezzo = num(prod.prezzo);

  if (prod.prezzoUnita === "conf") {
    const contenuto = num(prod.confContenuto);
    if (contenuto <= 0) return null;

    const perUnita = prezzo / contenuto;
    switch (prod.confUnita) {
      case "kg": return { dim: "kg", valore: perUnita };
      case "g":  return { dim: "kg", valore: perUnita * 1000 };
      case "L":  return { dim: "L",  valore: perUnita };
      case "ml": return { dim: "L",  valore: perUnita * 1000 };
      default:   return { dim: "pz", valore: perUnita };
    }
  }

  return { dim: prod.prezzoUnita, valore: prezzo };
}

/* Il conto completo: una riga per prodotto attivo, più i totali.
   Un prodotto a cui manca un dato non viene indovinato: finisce fra gli
   avvisi e resta fuori dal totale, così il totale non mente mai. */
function calcola(st) {
  const t = totali(st);
  const margine = 1 + num(st.margine) / 100;
  const righe = [];
  const avvisi = [];

  st.prodotti.forEach(function (prod) {
    if (!prod.attivo) return;

    const consumo = num(prod.consumo);
    const netta = consumo * bacino(prod, t);

    /* Il margine non si applica alle quantità fisse: un rotolo di
       alluminio resta un rotolo anche con il 20% di sicurezza. */
    const conMargine = prod.scala === "fisso" ? netta : netta * margine;

    /* I pezzi si comprano interi. Il toFixed evita che 3,0000000001
       diventi 4 per colpa della virgola mobile. */
    const acquisto = prod.unita === "pz"
      ? Math.ceil(Number(conMargine.toFixed(6)))
      : conMargine;

    const q = quantita(prod, acquisto);
    const unitario = prezzoUnitario(prod);

    let costo = null;
    let errore = null;

    if (consumo <= 0) {
      errore = { it: "manca il consumo a persona", en: "consumption per person is missing" };
    } else if (!unitario || !(unitario.valore > 0)) {
      errore = { it: "manca il prezzo", en: "price is missing" };
    } else if (q[unitario.dim] == null) {
      /* Prezzo al kg su un prodotto contato a pezzi: senza il peso del
         pezzo le due misure non si parlano. */
      errore = prod.unita === "pz"
        ? { it: "prezzo al " + unitario.dim + " ma il consumo è a pezzi: manca il peso del pezzo",
            en: "priced per " + unitario.dim + " but counted in pieces: the weight of one piece is missing" }
        : { it: "prezzo al " + unitario.dim + ": non è la misura di questo prodotto",
            en: "priced per " + unitario.dim + ": that is not how this product is measured" };
    } else {
      costo = q[unitario.dim] * unitario.valore;
    }

    if (errore) avvisi.push({ prodotto: prod, errore: errore });

    righe.push({
      prodotto: prod, netta: netta, conMargine: conMargine,
      acquisto: acquisto, q: q, unitario: unitario, costo: costo, errore: errore,
    });
  });

  const totale = righe.reduce(function (s, r) { return s + (r.costo || 0); }, 0);

  /* Totali per categoria, nell'ordine delle categorie */
  const perCategoria = new Map();
  righe.forEach(function (r) {
    const c = r.prodotto.categoria;
    if (!perCategoria.has(c)) perCategoria.set(c, { costo: 0, righe: [] });
    const voce = perCategoria.get(c);
    voce.costo += r.costo || 0;
    voce.righe.push(r);
  });

  /* Quanto costa il gruppo, sommando la sua fetta di ogni prodotto */
  const perGruppo = st.gruppi.map(function (g) {
    let costo = 0;
    righe.forEach(function (r) {
      if (r.costo != null) costo += r.costo * quotaGruppo(r.prodotto, g, t);
    });
    return { gruppo: g, costo: costo };
  });

  const kgCarne = righe
    .filter(function (r) { return r.prodotto.categoria === "carne"; })
    .reduce(function (s, r) { return s + (r.q.kg || 0); }, 0);

  const litri = righe
    .filter(function (r) { return r.prodotto.categoria === "alcolici" || r.prodotto.categoria === "analcolici"; })
    .reduce(function (s, r) { return s + (r.q.L || 0); }, 0);

  return {
    t: t, righe: righe, avvisi: avvisi, totale: totale,
    perCategoria: perCategoria, perGruppo: perGruppo,
    kgCarne: kgCarne, litri: litri,
    perPersona: t.persone > 0 ? totale / t.persone : 0,
  };
}


/* --------------------------------------------------------------------------
   3. INTERFACCIA
   -------------------------------------------------------------------------- */

/* --- stato: lettura e scrittura ------------------------------------------ */

/* Prototipi usati come rete di sicurezza quando si rilegge localStorage:
   i campi che mancano si prendono da qui, quelli sconosciuti si buttano. */
const PROTO_GRUPPO = { chiave: null, nome: null, n: 0, cibo: 100, alcol: 100, soft: 100 };
const PROTO_PRODOTTO = {
  chiave: null, nome: null, categoria: "consumabili", attivo: true,
  scala: "cibo", unita: "kg", pezzoQta: 0, pezzoUnita: "g", consumo: 0,
  prezzo: 0, prezzoUnita: "kg", confContenuto: 1, confUnita: "pz",
};

let contatore = 0;
function nuovoId(prefisso) { contatore += 1; return prefisso + contatore; }

function statoIniziale() {
  contatore = 0;
  return {
    evento: { nome: "", data: "" },
    margine: MARGINE_PREDEFINITO,
    gruppi: GRUPPI_PREDEFINITI.map(function (g) {
      return Object.assign({}, PROTO_GRUPPO, g, { id: nuovoId("g") });
    }),
    prodotti: CATALOGO.map(function (p) {
      /* In memoria va la chiave, non l'etichetta tradotta: vedi il commento
         in cima al file. */
      const copia = Object.assign({}, PROTO_PRODOTTO, p, { id: nuovoId("p") });
      delete copia.etichetta;
      return copia;
    }),
  };
}

/* Riporta il contatore oltre l'id più alto già in uso, altrimenti un
   prodotto aggiunto dopo un ricaricamento riceverebbe un id già preso. */
function allineaContatore(st) {
  let massimo = 0;
  st.gruppi.concat(st.prodotti).forEach(function (o) {
    const trovato = /(\d+)$/.exec(String(o.id || ""));
    if (trovato) massimo = Math.max(massimo, parseInt(trovato[1], 10));
  });
  contatore = Math.max(contatore, massimo);
}

function leggiStato() {
  const iniziale = statoIniziale();

  let salvato = null;
  try {
    salvato = localStorage.getItem(CHIAVE_ARCHIVIO);
  } catch (e) {
    return iniziale;   /* navigazione privata o storage negato: pazienza */
  }
  if (!salvato) return iniziale;

  try {
    const letto = JSON.parse(salvato);
    if (!letto || !Array.isArray(letto.gruppi) || !Array.isArray(letto.prodotti) || !letto.prodotti.length) {
      return iniziale;
    }

    const st = {
      evento: Object.assign({ nome: "", data: "" }, letto.evento || {}),
      margine: isFinite(Number(letto.margine)) ? Number(letto.margine) : MARGINE_PREDEFINITO,
      gruppi: letto.gruppi.map(function (g) {
        return Object.assign({}, PROTO_GRUPPO, g, { id: g.id || nuovoId("g") });
      }),
      prodotti: letto.prodotti.map(function (p) {
        return Object.assign({}, PROTO_PRODOTTO, p, { id: p.id || nuovoId("p") });
      }),
    };
    allineaContatore(st);
    return st;
  } catch (e) {
    return iniziale;   /* configurazione illeggibile: si riparte dai valori di serie */
  }
}

/* Si salva con un ritardo: mentre si trascina il cursore del margine
   arrivano decine di eventi, e non serve scrivere su disco a ogni pixel. */
let attesaSalvataggio = null;
function salva() {
  clearTimeout(attesaSalvataggio);
  attesaSalvataggio = setTimeout(function () {
    try {
      localStorage.setItem(CHIAVE_ARCHIVIO, JSON.stringify(stato));
    } catch (e) {
      /* spazio esaurito o storage negato: il tool funziona lo stesso */
    }
  }, 400);
}

let stato = leggiStato();

/* Le voci già prese al supermercato. Stanno solo in memoria di proposito:
   sono lo stato di una spesa in corso, non della configurazione. */
const spuntati = new Set();

let ultimoCalcolo = null;


/* --- nomi ed etichette ---------------------------------------------------- */

/* Il nome scritto dall'utente vince sempre; se non c'è, si traduce la
   chiave del catalogo. Un prodotto senza né l'uno né l'altra non esiste. */
function nomeProdotto(p) {
  if (p.nome) return p.nome;
  return NOMI_PRODOTTI[p.chiave] ? I18N.t(NOMI_PRODOTTI[p.chiave]) : "";
}

function nomeGruppo(g) {
  if (g.nome) return g.nome;
  return NOMI_GRUPPI[g.chiave] ? I18N.t(NOMI_GRUPPI[g.chiave]) : "";
}

/* Una categoria è una chiave conosciuta oppure, se l'ha scritta l'utente,
   il testo stesso. */
function nomeCategoria(chiave) {
  return CATEGORIE[chiave] ? I18N.t(CATEGORIE[chiave]) : chiave;
}

/* Il percorso inverso: dal testo digitato alla chiave, se corrisponde a una
   categoria del sito nella lingua corrente. */
function chiaveCategoria(testo) {
  const cercato = testo.trim().toLowerCase();
  const trovata = Object.keys(CATEGORIE).find(function (k) {
    return I18N.t(CATEGORIE[k]).toLowerCase() === cercato;
  });
  return trovata || testo.trim();
}

/* Le categorie di serie nel loro ordine, poi quelle aggiunte dall'utente. */
function categorieOrdinate() {
  const elenco = Object.keys(CATEGORIE);
  stato.prodotti.forEach(function (p) {
    if (elenco.indexOf(p.categoria) === -1) elenco.push(p.categoria);
  });
  return elenco;
}


/* --- formattazione -------------------------------------------------------- */

/* Il locale segue la lingua della pagina: mai scritto a mano.
   Lo stesso importo è "38,50 €" in italiano e "€38.50" in inglese. */
const euro = new Intl.NumberFormat(I18N.locale, { style: "currency", currency: "EUR" });
const euroPreciso = new Intl.NumberFormat(I18N.locale, {
  style: "currency", currency: "EUR", maximumFractionDigits: 3,
});
const numero = new Intl.NumberFormat(I18N.locale, { maximumFractionDigits: 2 });
const numeroCorto = new Intl.NumberFormat(I18N.locale, { maximumFractionDigits: 1 });

const $ = function (id) { return document.getElementById(id); };

/* I nomi li scrive l'utente e finiscono dentro innerHTML: vanno neutralizzati. */
function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
  });
}

/* Il valore memorizzato, scritto come lo scriverebbe chi legge la pagina. */
function perCampo(v) {
  const s = String(v);
  return I18N.lingua === "it" ? s.replace(".", ",") : s;
}

function qta(valore, unita) {
  return numero.format(valore) + " " + I18N.t(UNITA_BREVI[unita]);
}

/* "39 pezzi ≈ 3,51 kg": la seconda misura è quella che serve al banco. */
function qtaCompleta(riga) {
  let testo = qta(riga.acquisto, riga.prodotto.unita);
  if (riga.prodotto.unita === "pz") {
    if (riga.q.kg != null)     testo += " ≈ " + numero.format(riga.q.kg) + " kg";
    else if (riga.q.L != null) testo += " ≈ " + numero.format(riga.q.L) + " L";
  }
  return testo;
}

function dataLeggibile(iso) {
  const pezzi = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
  if (!pezzi) return "";
  const data = new Date(Number(pezzi[1]), Number(pezzi[2]) - 1, Number(pezzi[3]));
  return new Intl.DateTimeFormat(I18N.locale, { dateStyle: "long" }).format(data);
}

/* Un <select> a partire da un dizionario di etichette tradotte. */
function opzioni(dizionario, scelta) {
  return Object.keys(dizionario).map(function (k) {
    return '<option value="' + k + '"' + (k === scelta ? " selected" : "") + ">" +
           esc(I18N.t(dizionario[k])) + "</option>";
  }).join("");
}


/* --- costruzione delle tabelle -------------------------------------------

   Le tabelle si costruiscono una volta sola. Poi il ricalcolo tocca solo il
   testo delle celle di risultato e la visibilità delle righe: ricostruirle a
   ogni tasto premuto farebbe perdere il fuoco al campo che si sta usando. */

function costruisciGruppi() {
  document.querySelector("#tab-gruppi tbody").innerHTML = stato.gruppi.map(function (g) {
    return '' +
      "<tr>" +
        '<td><input class="field__input field__input--compact" type="text"' +
            ' data-campo="gruppi.' + g.id + '.nome" value="' + esc(nomeGruppo(g)) + '"></td>' +
        campoNumerico("gruppi." + g.id + ".n", g.n, 1) +
        campoNumerico("gruppi." + g.id + ".cibo", g.cibo, 10) +
        campoNumerico("gruppi." + g.id + ".alcol", g.alcol, 10) +
        campoNumerico("gruppi." + g.id + ".soft", g.soft, 10) +
        '<td class="table__num"><button class="btn btn--icon" type="button"' +
            ' data-azione="elimina-gruppo" data-id="' + g.id + '"' +
            ' title="' + esc(I18N.t({ it: "Elimina il gruppo", en: "Delete group" })) + '">✕</button></td>' +
      "</tr>";
  }).join("");

  document.querySelector("#tab-gruppi tfoot").innerHTML = '' +
    '<tr class="table__row--total">' +
      "<td>" + esc(I18N.t({ it: "Totale", en: "Total" })) + "</td>" +
      '<td class="table__num" id="gruppi-persone">—</td>' +
      '<td class="table__num" id="gruppi-cibo">—</td>' +
      '<td class="table__num" id="gruppi-alcol">—</td>' +
      '<td class="table__num" id="gruppi-soft">—</td>' +
      "<td></td>" +
    "</tr>";
}

function campoNumerico(campo, valore, passo) {
  return '<td class="table__num"><input class="field__input field__input--compact' +
         ' field__input--num field__input--xs" type="number" min="0" step="' + passo + '"' +
         ' data-campo="' + campo + '" value="' + valore + '"></td>';
}

/* Un <tbody> per categoria, con la sua riga di intestazione. Le categorie
   vuote non producono niente. */
function perCategoria(colonne, riga) {
  return categorieOrdinate().map(function (cat) {
    const prodotti = stato.prodotti.filter(function (p) { return p.categoria === cat; });
    if (!prodotti.length) return "";
    return '<tbody data-categoria="' + esc(cat) + '">' +
             '<tr class="table__group"><th colspan="' + colonne + '">' +
               esc(nomeCategoria(cat)) + "</th></tr>" +
             prodotti.map(riga).join("") +
           "</tbody>";
  }).join("");
}

function rigenera(idTabella, html) {
  const tabella = $(idTabella);
  tabella.querySelectorAll("tbody").forEach(function (c) { c.remove(); });
  tabella.insertAdjacentHTML("beforeend", html);
}

function costruisciCatalogo() {
  rigenera("tab-catalogo", perCategoria(5, function (p) {
    return '' +
      "<tr>" +
        '<td><input class="field__check" type="checkbox" data-campo="prodotti.' + p.id + '.attivo"' +
            (p.attivo ? " checked" : "") +
            ' aria-label="' + esc(I18N.t({ it: "Metti in lista", en: "Add to the list" })) + '"></td>' +
        '<td><input class="field__input field__input--compact" type="text"' +
            ' data-campo="prodotti.' + p.id + '.nome" value="' + esc(nomeProdotto(p)) + '"></td>' +
        '<td><select class="field__select field__select--compact" data-campo="prodotti.' + p.id + '.scala">' +
            opzioni(SCALE, p.scala) + "</select></td>" +
        '<td><select class="field__select field__select--compact" data-campo="prodotti.' + p.id + '.unita">' +
            opzioni(UNITA, p.unita) + "</select></td>" +
        '<td class="table__num"><button class="btn btn--icon" type="button"' +
            ' data-azione="elimina-prodotto" data-id="' + p.id + '"' +
            ' title="' + esc(I18N.t({ it: "Elimina dal catalogo", en: "Delete from the catalogue" })) + '">✕</button></td>' +
      "</tr>";
  }));
}

function costruisciConsumi() {
  rigenera("tab-consumi", perCategoria(5, function (p) {
    return '' +
      '<tr data-riga="' + p.id + '">' +
        '<td><span class="table__name" data-nome-di="' + p.id + '"></span><br>' +
            '<span class="table__unit" data-scala-di="' + p.id + '"></span></td>' +
        '<td class="table__num">' +
            '<input class="field__input field__input--compact field__input--num field__input--xs"' +
            ' type="text" inputmode="decimal" data-campo="prodotti.' + p.id + '.consumo"' +
            ' value="' + perCampo(p.consumo) + '"> ' +
            '<span class="table__unit" data-unita-di="' + p.id + '"></span></td>' +
        "<td>" +
          '<span data-pezzo="' + p.id + '">' +
            '<input class="field__input field__input--compact field__input--num field__input--xs"' +
            ' type="text" inputmode="decimal" data-campo="prodotti.' + p.id + '.pezzoQta"' +
            ' value="' + perCampo(p.pezzoQta) + '"> ' +
            '<select class="field__select field__select--compact field__select--auto"' +
            ' data-campo="prodotti.' + p.id + '.pezzoUnita">' +
              '<option value="g"' + (p.pezzoUnita === "g" ? " selected" : "") + ">g</option>" +
              '<option value="ml"' + (p.pezzoUnita === "ml" ? " selected" : "") + ">ml</option>" +
            "</select></span>" +
          '<span class="table__unit" data-senza-pezzo="' + p.id + '">—</span>' +
        "</td>" +
        '<td class="table__num" id="netta-' + p.id + '">—</td>' +
        '<td class="table__num" id="margine-' + p.id + '">—</td>' +
      "</tr>";
  }));
}

function costruisciPrezzi() {
  rigenera("tab-prezzi", perCategoria(6, function (p) {
    return '' +
      '<tr data-riga="' + p.id + '">' +
        '<td><span class="table__name" data-nome-di="' + p.id + '"></span></td>' +
        '<td class="table__num">' +
            '<input class="field__input field__input--compact field__input--num field__input--xs"' +
            ' type="text" inputmode="decimal" data-campo="prodotti.' + p.id + '.prezzo"' +
            ' value="' + perCampo(p.prezzo) + '"></td>' +
        '<td><select class="field__select field__select--compact" data-campo="prodotti.' + p.id + '.prezzoUnita">' +
            opzioni(PREZZO_UNITA, p.prezzoUnita) + "</select></td>" +
        "<td>" +
          '<span data-conf="' + p.id + '">' +
            '<input class="field__input field__input--compact field__input--num field__input--xs"' +
            ' type="text" inputmode="decimal" data-campo="prodotti.' + p.id + '.confContenuto"' +
            ' value="' + perCampo(p.confContenuto) + '"> ' +
            '<select class="field__select field__select--compact field__select--auto"' +
            ' data-campo="prodotti.' + p.id + '.confUnita">' + opzioni(CONF_UNITA, p.confUnita) + "</select>" +
          "</span>" +
          '<span class="table__unit" data-senza-conf="' + p.id + '">—</span>' +
        "</td>" +
        '<td class="table__num" id="unitario-' + p.id + '">—</td>' +
        '<td class="table__num" id="costo-' + p.id + '">—</td>' +
      "</tr>";
  }));
}

function costruisciTutto() {
  costruisciGruppi();
  costruisciCatalogo();
  costruisciConsumi();
  costruisciPrezzi();

  $("elenco-categorie").innerHTML = categorieOrdinate().map(function (c) {
    return '<option value="' + esc(nomeCategoria(c)) + '">';
  }).join("");

  aggiornaVisibilita();
}


/* --- visibilità e testi che dipendono dalle scelte ------------------------ */

function mostraTutti(selettore, visibile) {
  document.querySelectorAll(selettore).forEach(function (n) { n.hidden = !visibile; });
}

function scriviTutti(selettore, testo) {
  document.querySelectorAll(selettore).forEach(function (n) { n.textContent = testo; });
}

function aggiornaVisibilita() {
  const categorieAttive = {};

  stato.prodotti.forEach(function (p) {
    const aPezzi = p.unita === "pz";
    const aConfezione = p.prezzoUnita === "conf";

    mostraTutti('tr[data-riga="' + p.id + '"]', p.attivo);
    mostraTutti('[data-pezzo="' + p.id + '"]', aPezzi);
    mostraTutti('[data-senza-pezzo="' + p.id + '"]', !aPezzi);
    mostraTutti('[data-conf="' + p.id + '"]', aConfezione);
    mostraTutti('[data-senza-conf="' + p.id + '"]', !aConfezione);

    scriviTutti('[data-nome-di="' + p.id + '"]', nomeProdotto(p));
    scriviTutti('[data-scala-di="' + p.id + '"]', I18N.t(SCALE[p.scala]));
    scriviTutti('[data-unita-di="' + p.id + '"]', I18N.t(UNITA_BREVI[p.unita]));

    if (p.attivo) categorieAttive[p.categoria] = true;
  });

  /* Una categoria in cui non è rimasto niente di spuntato sparisce dalle
     tabelle dei consumi e dei prezzi: sarebbe una riga di sola intestazione. */
  ["tab-consumi", "tab-prezzi"].forEach(function (id) {
    $(id).querySelectorAll("tbody[data-categoria]").forEach(function (corpo) {
      corpo.hidden = !categorieAttive[corpo.dataset.categoria];
    });
  });
}


/* --- ricalcolo e rendering ------------------------------------------------ */

function scrivi(id, testo) {
  const nodo = $(id);
  if (nodo) nodo.textContent = testo;
}

function aggiorna() {
  const r = calcola(stato);
  const aTesta = function (v) { return r.t.persone > 0 ? euro.format(v / r.t.persone) : "—"; };

  /* --- il sommario in cima --- */
  scrivi("tot-spesa", euro.format(r.totale));
  scrivi("tot-persona", euro.format(r.perPersona));
  scrivi("tot-persone", numero.format(r.t.persone));
  scrivi("tot-carne", numeroCorto.format(r.kgCarne) + " kg");
  scrivi("tot-bevande", numeroCorto.format(r.litri) + " L");

  /* --- piede della tabella dei gruppi --- */
  const quote = I18N.t({ it: "quote", en: "shares" });
  scrivi("gruppi-persone", numero.format(r.t.persone));
  scrivi("gruppi-cibo", numero.format(r.t.cibo) + " " + quote);
  scrivi("gruppi-alcol", numero.format(r.t.alcol) + " " + quote);
  scrivi("gruppi-soft", numero.format(r.t.soft) + " " + quote);

  scrivi("margine-valore", "+" + numero.format(num(stato.margine)) + "%");

  /* --- celle calcolate nelle tabelle dei consumi e dei prezzi --- */
  stato.prodotti.forEach(function (p) {
    ["netta-", "margine-", "unitario-", "costo-"].forEach(function (pre) { scrivi(pre + p.id, "—"); });
  });

  r.righe.forEach(function (riga) {
    const p = riga.prodotto;
    scrivi("netta-" + p.id, qta(riga.netta, p.unita));
    scrivi("margine-" + p.id, qtaCompleta(riga));
    scrivi("unitario-" + p.id, riga.unitario && riga.unitario.valore > 0
      ? euroPreciso.format(riga.unitario.valore) + " / " + riga.unitario.dim
      : "—");
    scrivi("costo-" + p.id, riga.costo != null ? euro.format(riga.costo) : "—");
  });

  disegnaAvvisi(r);
  disegnaBarre(r);
  disegnaRisultati(r, aTesta);
  disegnaQuote(r);
  disegnaLista(r);

  $("intestazione-stampa").textContent = testoIntestazione(r);

  ultimoCalcolo = r;
  salva();
}

function disegnaAvvisi(r) {
  if (r.avvisi.length) {
    $("avvisi").innerHTML =
      '<div class="note"><p>' +
        esc(I18N.t({
          it: "Queste voci restano fuori dal totale perché manca un dato:",
          en: "These items are left out of the total because something is missing:",
        })) +
      "</p><ul>" +
      r.avvisi.map(function (a) {
        return "<li>" + esc(nomeProdotto(a.prodotto)) + " — " + esc(I18N.t(a.errore)) + "</li>";
      }).join("") +
      "</ul></div>";
    return;
  }

  $("avvisi").innerHTML = '<p class="muted">' + esc(r.righe.length
    ? I18N.t({
        it: "Tutte le voci in lista hanno consumo e prezzo: il totale è completo.",
        en: "Every item on the list has a consumption and a price: the total is complete.",
      })
    : I18N.t({
        it: "Non c'è niente in lista: scegli i prodotti qui sopra.",
        en: "Nothing on the list yet: pick some products above.",
      })) + "</p>";
}

function disegnaBarre(r) {
  const categorie = categorieOrdinate().filter(function (c) { return r.perCategoria.has(c); });
  const massimo = Math.max(1, ...categorie.map(function (c) { return r.perCategoria.get(c).costo; }));

  $("barre").innerHTML = categorie.map(function (c) {
    const voce = r.perCategoria.get(c);
    const percentuale = r.totale > 0 ? voce.costo / r.totale * 100 : 0;
    return '<div class="bar">' +
             "<div>" + esc(nomeCategoria(c)) + "</div>" +
             '<div class="bar__track"><div class="bar__fill"' +
               ' data-quota="' + (voce.costo / massimo * 100).toFixed(1) + '"></div></div>' +
             '<div class="bar__value">' + euro.format(voce.costo) + " · " +
               numeroCorto.format(percentuale) + "%</div>" +
           "</div>";
  }).join("");

  /* La lunghezza della barra è un numero calcolato: entra nel CSS come
     proprietà --quota, non come attributo style scritto a mano. */
  $("barre").querySelectorAll(".bar__fill").forEach(function (n) {
    n.style.setProperty("--quota", n.dataset.quota + "%");
  });
}

function disegnaRisultati(r, aTesta) {
  const categorie = categorieOrdinate().filter(function (c) { return r.perCategoria.has(c); });
  const daDefinire = I18N.t({ it: "da definire", en: "to be set" });

  const corpi = categorie.map(function (c) {
    const voce = r.perCategoria.get(c);
    return '<tbody>' +
      '<tr class="table__group"><th colspan="6">' + esc(nomeCategoria(c)) + "</th></tr>" +
      voce.righe.map(function (riga) {
        return "<tr" + (riga.errore ? ' class="table__row--warn"' : "") + ">" +
          '<td class="table__name">' + esc(nomeProdotto(riga.prodotto)) + "</td>" +
          '<td class="table__num">' + qta(riga.netta, riga.prodotto.unita) + "</td>" +
          '<td class="table__num">' + qta(riga.conMargine, riga.prodotto.unita) + "</td>" +
          '<td class="table__num">' + qtaCompleta(riga) + "</td>" +
          '<td class="table__num">' + (riga.costo != null ? euro.format(riga.costo) : esc(daDefinire)) + "</td>" +
          '<td class="table__num">' + (riga.costo != null ? aTesta(riga.costo) : "—") + "</td>" +
        "</tr>";
      }).join("") +
      '<tr class="table__row--sub">' +
        '<td colspan="4">' + esc(I18N.t({ it: "Subtotale", en: "Subtotal" })) + " " + esc(nomeCategoria(c)) + "</td>" +
        '<td class="table__num">' + euro.format(voce.costo) + "</td>" +
        '<td class="table__num">' + aTesta(voce.costo) + "</td>" +
      "</tr></tbody>";
  }).join("");

  const piede = '<tfoot><tr class="table__row--total">' +
      '<td colspan="4">' + esc(I18N.t({ it: "Totale", en: "Total" })) + "</td>" +
      '<td class="table__num">' + euro.format(r.totale) + "</td>" +
      '<td class="table__num">' + euro.format(r.perPersona) + "</td>" +
    "</tr></tfoot>";

  const tabella = $("tab-risultati");
  tabella.querySelectorAll("tbody, tfoot").forEach(function (n) { n.remove(); });
  tabella.insertAdjacentHTML("beforeend", corpi + piede);
}

function disegnaQuote(r) {
  const righe = r.perGruppo.map(function (voce) {
    const n = num(voce.gruppo.n);
    const quota = n > 0 ? voce.costo / n : 0;
    const differenza = quota - r.perPersona;
    const segno = differenza > 0.005 ? "+" : "";
    return "<tr>" +
      '<td class="table__name">' + esc(nomeGruppo(voce.gruppo)) + "</td>" +
      '<td class="table__num">' + numero.format(n) + "</td>" +
      '<td class="table__num">' + euro.format(voce.costo) + "</td>" +
      '<td class="table__num">' + (n > 0 ? euro.format(quota) : "—") + "</td>" +
      '<td class="table__num">' + (n > 0 ? segno + euro.format(differenza) : "—") + "</td>" +
    "</tr>";
  }).join("");

  document.querySelector("#tab-quote tbody").innerHTML = righe +
    '<tr class="table__row--total">' +
      "<td>" + esc(I18N.t({ it: "Totale", en: "Total" })) + "</td>" +
      '<td class="table__num">' + numero.format(r.t.persone) + "</td>" +
      '<td class="table__num">' + euro.format(r.totale) + "</td>" +
      '<td class="table__num">' + euro.format(r.perPersona) + "</td>" +
      "<td></td>" +
    "</tr>";
}

function disegnaLista(r) {
  const categorie = categorieOrdinate().filter(function (c) { return r.perCategoria.has(c); });

  if (!categorie.length) {
    $("lista").innerHTML = '<p class="muted">' +
      esc(I18N.t({ it: "Niente da comprare.", en: "Nothing to buy." })) + "</p>";
    return;
  }

  $("lista").innerHTML = categorie.map(function (c) {
    return '<p class="eyebrow eyebrow--tight">' + esc(nomeCategoria(c)) + "</p>" +
      '<div class="checklist">' + r.perCategoria.get(c).righe.map(function (riga) {
        const preso = spuntati.has(riga.prodotto.id);
        return '<label class="checklist__item' + (preso ? " is-done" : "") + '">' +
          '<input class="field__check" type="checkbox" data-spunta="' + riga.prodotto.id + '"' +
            (preso ? " checked" : "") + ">" +
          '<span class="checklist__text">' + esc(nomeProdotto(riga.prodotto)) +
            ' <span class="checklist__qty">— ' + qtaCompleta(riga) + "</span></span>" +
          '<span class="checklist__meta">' +
            (riga.costo != null ? euro.format(riga.costo) : "—") + "</span>" +
        "</label>";
      }).join("") + "</div>";
  }).join("");
}

/* La riga che compare solo sul foglio stampato: a schermo il sommario in
   cima dice già le stesse cose. */
function testoIntestazione(r) {
  const nome = stato.evento.nome || I18N.t({ it: "Grigliata", en: "Barbecue" });
  const data = dataLeggibile(stato.evento.data);
  return nome + (data ? " — " + data : "") + " · " +
    numero.format(r.t.persone) + " " + I18N.t({ it: "persone", en: "people" }) + " · " +
    euro.format(r.totale) + " (" + euro.format(r.perPersona) + " " +
    I18N.t({ it: "a testa", en: "each" }) + ")";
}


/* --- lista da copiare ----------------------------------------------------- */

function testoLista() {
  const r = ultimoCalcolo || calcola(stato);
  const righe = [testoIntestazione(r)];

  categorieOrdinate().filter(function (c) { return r.perCategoria.has(c); }).forEach(function (c) {
    righe.push("");
    righe.push(nomeCategoria(c).toUpperCase());
    r.perCategoria.get(c).righe.forEach(function (riga) {
      righe.push("- " + nomeProdotto(riga.prodotto) + ": " + qtaCompleta(riga) +
        (riga.costo != null ? "  (" + euro.format(riga.costo) + ")" : ""));
    });
  });

  return righe.join("\n");
}

let attesaMessaggio = null;
function messaggio(testo) {
  const nodo = $("toast");
  nodo.textContent = testo;
  nodo.classList.add("is-on");
  clearTimeout(attesaMessaggio);
  attesaMessaggio = setTimeout(function () { nodo.classList.remove("is-on"); }, 2500);
}

function copiaLista() {
  const testo = testoLista();
  const fatto = function () {
    messaggio(I18N.t({ it: "Lista copiata negli appunti.", en: "List copied to the clipboard." }));
  };
  const fallito = function () {
    messaggio(I18N.t({ it: "Copia non riuscita: seleziona il testo a mano.",
                       en: "Copy failed: select the text by hand." }));
  };

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(testo).then(fatto, fallito);
  } else {
    fallito();
  }
}


/* --- eventi ---------------------------------------------------------------

   Un solo ascoltatore sul documento invece di uno per campo: le righe si
   aggiungono e si tolgono di continuo, e così non c'è niente da riagganciare
   ogni volta che una tabella viene ricostruita. */

/* Campi che contengono testo o una scelta: tutti gli altri sono numeri. */
const CAMPI_TESTO = new Set(["nome", "categoria", "unita", "scala", "prezzoUnita", "pezzoUnita", "confUnita"]);

/* Campi che cambiano quali celle si vedono o come si chiamano le cose:
   dopo questi serve anche aggiornaVisibilita(). */
const CAMPI_STRUTTURA = new Set(["attivo", "nome", "unita", "scala", "prezzoUnita"]);

document.addEventListener("input", function (e) {
  const campo = e.target.dataset ? e.target.dataset.campo : null;
  if (!campo) return;

  const parti = campo.split(".");
  const valore = e.target.type === "checkbox" ? e.target.checked : e.target.value;

  if (parti[0] === "margine") {
    stato.margine = num(valore);

  } else if (parti[0] === "evento") {
    stato.evento[parti[1]] = valore;

  } else {
    const insieme = parti[0] === "gruppi" ? stato.gruppi : stato.prodotti;
    const oggetto = insieme.find(function (o) { return o.id === parti[1]; });
    if (!oggetto) return;

    const nome = parti[2];
    oggetto[nome] = (typeof valore === "boolean" || CAMPI_TESTO.has(nome)) ? valore : num(valore);

    /* Passando a "€ a confezione", il contenuto va espresso in una misura
       che il prodotto conosce: un pacco "da 12 kg" di birra contata a
       pezzi non vuol dire niente. */
    if (nome === "prezzoUnita" && valore === "conf") {
      const dimensione = { kg: "kg", g: "kg", L: "L", ml: "L", pz: "pz" }[oggetto.confUnita];
      if (quantita(oggetto, 1)[dimensione] == null) {
        oggetto.confUnita = oggetto.unita;
        const select = document.querySelector('select[data-campo="prodotti.' + oggetto.id + '.confUnita"]');
        if (select) select.value = oggetto.unita;
      }
    }

    if (CAMPI_STRUTTURA.has(nome)) aggiornaVisibilita();
  }

  aggiorna();
});

/* Spuntare una voce della lista della spesa non cambia nessun conto. */
document.addEventListener("change", function (e) {
  const id = e.target.dataset ? e.target.dataset.spunta : null;
  if (!id) return;

  if (e.target.checked) spuntati.add(id); else spuntati.delete(id);
  const voce = e.target.closest(".checklist__item");
  if (voce) voce.classList.toggle("is-done", e.target.checked);
});

document.addEventListener("click", function (e) {
  const bottone = e.target.closest("[data-azione]");
  if (!bottone) return;

  const azione = bottone.dataset.azione;

  if (azione === "aggiungi-gruppo") {
    stato.gruppi.push(Object.assign({}, PROTO_GRUPPO, {
      id: nuovoId("g"),
      nome: I18N.t({ it: "Nuovo gruppo", en: "New group" }),
      n: 1,
    }));
    costruisciGruppi();
    aggiorna();

  } else if (azione === "elimina-gruppo") {
    if (stato.gruppi.length <= 1) {
      messaggio(I18N.t({ it: "Serve almeno un gruppo.", en: "At least one group is needed." }));
      return;
    }
    stato.gruppi = stato.gruppi.filter(function (g) { return g.id !== bottone.dataset.id; });
    costruisciGruppi();
    aggiorna();

  } else if (azione === "aggiungi-prodotto") {
    aggiungiProdotto();

  } else if (azione === "elimina-prodotto") {
    const p = stato.prodotti.find(function (x) { return x.id === bottone.dataset.id; });
    if (!p) return;
    if (!confirm(I18N.t({
      it: 'Eliminare "' + nomeProdotto(p) + '" dal catalogo?',
      en: 'Delete "' + nomeProdotto(p) + '" from the catalogue?',
    }))) return;

    stato.prodotti = stato.prodotti.filter(function (x) { return x.id !== p.id; });
    spuntati.delete(p.id);
    costruisciTutto();
    aggiorna();

  } else if (azione === "copia") {
    copiaLista();

  } else if (azione === "stampa") {
    window.print();

  } else if (azione === "ripristina") {
    if (!confirm(I18N.t({
      it: "Rimettere tutto com'era all'inizio? La configurazione attuale va persa.",
      en: "Reset everything to the starting values? The current setup will be lost.",
    }))) return;

    try { localStorage.removeItem(CHIAVE_ARCHIVIO); } catch (err) { /* niente da rimuovere */ }
    stato = statoIniziale();
    spuntati.clear();
    riempiCampiFissi();
    costruisciTutto();
    aggiorna();
    messaggio(I18N.t({ it: "Valori di partenza ripristinati.", en: "Starting values restored." }));
  }
});

function aggiungiProdotto() {
  const nome = $("nuovo-nome").value.trim();
  if (!nome) {
    messaggio(I18N.t({ it: "Dai un nome al prodotto.", en: "Give the product a name." }));
    $("nuovo-nome").focus();
    return;
  }

  const unita = $("nuovo-unita").value;
  stato.prodotti.push(Object.assign({}, PROTO_PRODOTTO, {
    id: nuovoId("p"),
    nome: nome,
    categoria: chiaveCategoria($("nuovo-categoria").value) || "consumabili",
    scala: $("nuovo-scala").value,
    unita: unita,
    prezzoUnita: unita,   /* il prezzo parte nella stessa misura del consumo */
  }));

  $("nuovo-nome").value = "";
  costruisciTutto();
  aggiorna();
  messaggio(I18N.t({
    it: '"' + nome + '" aggiunto: ora imposta consumo e prezzo.',
    en: '"' + nome + '" added: now set its consumption and price.',
  }));
}


/* --- avvio ---------------------------------------------------------------- */

/* I campi che stanno nell'HTML e non vengono ricostruiti mai. */
function riempiCampiFissi() {
  $("evento-nome").value = stato.evento.nome || "";
  $("evento-data").value = stato.evento.data || "";
  $("margine").value = num(stato.margine);
}

$("nuovo-scala").innerHTML = opzioni(SCALE, "cibo");
$("nuovo-unita").innerHTML = opzioni(UNITA, "kg");

riempiCampiFissi();
costruisciTutto();
aggiorna();
