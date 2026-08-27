/* ==========================================================================
   CALCOLATORE TASSE — regime forfettario

   Struttura del file, da riusare per i tool futuri:
     1. PARAMETRI  — tutti i numeri "di dominio", isolati in cima
     2. CALCOLO    — funzione pura: dati dentro, risultato fuori
     3. INTERFACCIA— lettura input, rendering, eventi

   Tenere i parametri separati dal calcolo significa che l'aggiornamento
   annuale delle aliquote è una modifica di due righe, non una caccia
   al numero sparso nel codice.
   ========================================================================== */


/* --------------------------------------------------------------------------
   1. PARAMETRI

   ⚠️  DA VERIFICARE E AGGIORNARE OGNI ANNO.
   I valori qui sotto sono un punto di partenza plausibile, non una fonte
   ufficiale. Controllali sul sito dell'Agenzia delle Entrate e dell'INPS
   prima di pubblicare il tool.
   -------------------------------------------------------------------------- */

const COEFFICIENTI = [
  { id: "professioni",  label: "Professioni e servizi (78%)",        valore: 0.78 },
  { id: "commercio",    label: "Commercio all'ingrosso (40%)",       valore: 0.40 },
  { id: "alimentari",   label: "Commercio alimentari e bevande (40%)", valore: 0.40 },
  { id: "costruzioni",  label: "Costruzioni e immobiliare (86%)",    valore: 0.86 },
  { id: "intermediari", label: "Intermediari del commercio (62%)",   valore: 0.62 },
];

const ALIQUOTE = [
  { id: "ridotta",   label: "5% — primi 5 anni di attività", valore: 0.05 },
  { id: "ordinaria", label: "15% — ordinaria",               valore: 0.15 },
];

const CASSE = [
  { id: "separata",   label: "Gestione separata INPS", aliquota: 0.2607, fisso: 0 },
  { id: "artigiani",  label: "Artigiani e commercianti", aliquota: 0.24, fisso: 4600 },
  { id: "nessuna",    label: "Nessuna / cassa privata",  aliquota: 0,    fisso: 0 },
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

const euro = new Intl.NumberFormat("it-IT", {
  style: "currency", currency: "EUR", maximumFractionDigits: 0,
});

const percento = new Intl.NumberFormat("it-IT", {
  style: "percent", maximumFractionDigits: 1,
});

const $ = (id) => document.getElementById(id);

// Riempie una <select> a partire da un elenco di opzioni.
function popola(select, opzioni) {
  select.innerHTML = opzioni
    .map((o) => `<option value="${o.id}">${o.label}</option>`)
    .join("");
}

function esegui() {
  const ricavi = Number($("ricavi").value);
  const stato = $("stato");
  const risultato = $("risultato");

  if (!Number.isFinite(ricavi) || ricavi <= 0) {
    risultato.hidden = true;
    stato.textContent = "Inserisci un importo di ricavi maggiore di zero.";
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
  stato.textContent = "Stima aggiornata.";
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
