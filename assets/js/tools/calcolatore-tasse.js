/* ==========================================================================
   CALCOLATORE TASSE — regime forfettario

   Struttura del file, da riusare per i tool futuri:
     1. PARAMETRI  — tutti i numeri "di dominio", isolati in cima
     2. CALCOLO    — funzione pura: dati dentro, risultato fuori
     3. INTERFACCIA— lettura input, rendering, eventi

   Tenere i parametri separati dal calcolo significa che l'aggiornamento
   annuale delle aliquote è una modifica di due righe, non una caccia
   al numero sparso nel codice.

   DUE LINGUE, UN FILE SOLO
   Questo file serve sia /it/strumenti/calcolatore-tasse/ sia
   /en/tools/tax-calculator/. Le due pagine hanno gli stessi id nei campi
   proprio per questo: il calcolo esiste una volta sola, e un aggiornamento
   delle aliquote non può dimenticarsi di una delle due lingue.

   Etichette e messaggi che nascono qui — il testo di un <option>, un
   errore, la riga di stato — passano da I18N.t({ it, en }): dentro un
   <option> non si può mettere altro markup, quindi la traduzione non può
   stare nell'HTML della pagina come per il resto del testo.
   ========================================================================== */


/* --------------------------------------------------------------------------
   1. PARAMETRI

   ⚠️  DA VERIFICARE E AGGIORNARE OGNI ANNO.
   I valori qui sotto sono un punto di partenza plausibile, non una fonte
   ufficiale. Controllali sul sito dell'Agenzia delle Entrate e dell'INPS
   prima di pubblicare il tool.

   I numeri non cambiano con la lingua: si traduce solo l'etichetta.
   -------------------------------------------------------------------------- */

const COEFFICIENTI = [
  {
    id: "professioni",
    label: { it: "Professioni e servizi (78%)", en: "Professions and services (78%)" },
    valore: 0.78,
  },
  {
    id: "commercio",
    label: { it: "Commercio all'ingrosso (40%)", en: "Wholesale trade (40%)" },
    valore: 0.40,
  },
  {
    id: "alimentari",
    label: { it: "Commercio alimentari e bevande (40%)", en: "Food and drink retail (40%)" },
    valore: 0.40,
  },
  {
    id: "costruzioni",
    label: { it: "Costruzioni e immobiliare (86%)", en: "Construction and real estate (86%)" },
    valore: 0.86,
  },
  {
    id: "intermediari",
    label: { it: "Intermediari del commercio (62%)", en: "Trade intermediaries (62%)" },
    valore: 0.62,
  },
];

const ALIQUOTE = [
  {
    id: "ridotta",
    label: { it: "5% — primi 5 anni di attività", en: "5% — first 5 years of trading" },
    valore: 0.05,
  },
  {
    id: "ordinaria",
    label: { it: "15% — ordinaria", en: "15% — standard" },
    valore: 0.15,
  },
];

// "Gestione separata" e "artigiani e commercianti" sono nomi propri di
// gestioni INPS: in inglese si tengono, spiegandoli fra parentesi.
const CASSE = [
  {
    id: "separata",
    label: { it: "Gestione separata INPS", en: "INPS Gestione Separata (freelancers)" },
    aliquota: 0.2607,
    fisso: 0,
  },
  {
    id: "artigiani",
    label: { it: "Artigiani e commercianti", en: "Artisans and traders (INPS)" },
    aliquota: 0.24,
    fisso: 4600,
  },
  {
    id: "nessuna",
    label: { it: "Nessuna / cassa privata", en: "None / private fund" },
    aliquota: 0,
    fisso: 0,
  },
];


/* --------------------------------------------------------------------------
   2. CALCOLO — funzione pura, testabile, senza riferimenti al DOM
   -------------------------------------------------------------------------- */

function calcola({ ricavi, coefficiente, aliquota, cassa }) {
  // In forfettario il reddito non si calcola sui costi reali: si applica
  // un coefficiente forfettario ai ricavi.
  const redditoLordo = ricavi * coefficiente;

  // I contributi previdenziali si deducono prima di calcolare l'imposta.
  const contributi = redditoLordo * cassa.aliquota + cassa.fisso;

  const imponibileFiscale = Math.max(0, redditoLordo - contributi);
  const imposta = imponibileFiscale * aliquota;

  const netto = ricavi - contributi - imposta;
  const pressione = ricavi > 0 ? (contributi + imposta) / ricavi : 0;

  return { redditoLordo, contributi, imposta, netto, pressione };
}


/* --------------------------------------------------------------------------
   3. INTERFACCIA
   -------------------------------------------------------------------------- */

/* Il locale segue la lingua della pagina: non scriverlo mai a mano.
   Lo stesso importo diventa "38.391 €" in italiano e "€38,391" in inglese —
   due stringhe diverse, lo stesso numero. */
const euro = new Intl.NumberFormat(I18N.locale, {
  style: "currency", currency: "EUR", maximumFractionDigits: 0,
});

const percento = new Intl.NumberFormat(I18N.locale, {
  style: "percent", maximumFractionDigits: 1,
});

const $ = (id) => document.getElementById(id);

// Riempie una <select> a partire da un elenco di opzioni.
function popola(select, opzioni) {
  select.innerHTML = opzioni
    .map((o) => `<option value="${o.id}">${I18N.t(o.label)}</option>`)
    .join("");
}

function esegui() {
  const ricavi = Number($("ricavi").value);
  const stato = $("stato");
  const risultato = $("risultato");

  if (!Number.isFinite(ricavi) || ricavi <= 0) {
    risultato.hidden = true;
    stato.textContent = I18N.t({
      it: "Inserisci un importo di ricavi maggiore di zero.",
      en: "Enter a revenue figure greater than zero.",
    });
    return;
  }

  const esito = calcola({
    ricavi,
    coefficiente: COEFFICIENTI.find((c) => c.id === $("coefficiente").value).valore,
    aliquota:     ALIQUOTE.find((a) => a.id === $("aliquota").value).valore,
    cassa:        CASSE.find((c) => c.id === $("contributi").value),
  });

  $("out-reddito").textContent    = euro.format(esito.redditoLordo);
  $("out-contributi").textContent = euro.format(esito.contributi);
  $("out-imposta").textContent    = euro.format(esito.imposta);
  $("out-pressione").textContent  = percento.format(esito.pressione);
  $("out-netto").textContent      = euro.format(esito.netto);

  risultato.hidden = false;
  stato.textContent = I18N.t({
    it: "Stima aggiornata.",
    en: "Estimate updated.",
  });
}

// Avvio
popola($("coefficiente"), COEFFICIENTI);
popola($("aliquota"), ALIQUOTE);
popola($("contributi"), CASSE);

$("calcola").addEventListener("click", esegui);

// Ricalcola anche premendo Invio su un campo.
document.querySelectorAll(".field__input, .field__select").forEach((el) => {
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") esegui(); });
});
