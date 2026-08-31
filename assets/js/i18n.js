/* ==========================================================================
   I18N.JS — lingua della pagina e stringhe generate da JavaScript

   COME FUNZIONA
   Il sito ha due alberi paralleli, /it/ e /en/. Il testo di una pagina è
   scritto direttamente nella sua lingua: non c'è nessuno scambio di testo a
   runtime e quindi nessuno sfarfallio da prevenire. Una pagina senza
   JavaScript resta leggibile e corretta.

   Restano tre cose che l'HTML da solo non sa fare, ed è tutto quello che
   fa questo file:

     1. dire in che lingua siamo           → I18N.lingua
     2. tradurre il testo prodotto dal JS  → I18N.t({ it, en })
     3. trovare la pagina gemella          → I18N.alternativa()

   DUE SCELTE CHE EVITANO DUPLICAZIONE

   La lingua della pagina è <html lang>. È l'unica fonte di verità: la
   stessa che leggono browser, lettori di schermo e motori di ricerca. Non
   è ripetuta da nessun'altra parte, quindi non può andare fuori sincrono.

   La pagina gemella si legge da <link rel="alternate" hreflang="…">, che
   ogni pagina deve dichiarare comunque per i motori di ricerca. Il menu a
   tendina della testata legge quel tag: finché una pagina dichiara il suo
   gemello per la SEO, il selettore di lingua funziona da solo. Non esiste
   una tabella di percorsi da tenere aggiornata a mano, ed è questo che
   rende sostenibili gli indirizzi tradotti (/it/articoli/ ↔ /en/articles/).

   Una pagina che non dichiara un gemello — per esempio /stile/, che esiste
   solo in italiano — semplicemente non mostra il selettore.
   ========================================================================== */


/* --------------------------------------------------------------------------
   CONFIGURAZIONE

   Aggiungere una lingua significa: una voce qui, un albero di pagine, e le
   voci di menu in layout.js. Il costo è visibile di proposito.
   -------------------------------------------------------------------------- */

const LINGUE = {
  it: { nome: "Italiano", breve: "IT", locale: "it-IT", radice: "/it/" },

  // en-IE e non en-US: è inglese, ma di un paese dell'area euro, quindi
  // Intl formatta gli importi come "€38,391" invece di "€38,391.00" con
  // il simbolo fuori posto. La valuta del sito è l'euro.
  en: { nome: "English",  breve: "EN", locale: "en-IE", radice: "/en/" },
};

const LINGUA_PREDEFINITA = "en";   // l'inglese è la lingua servita per difetto
const CHIAVE_LINGUA = "lingua";    // dove il selettore salva la scelta


/* --------------------------------------------------------------------------
   LETTURA E SCRITTURA DELLA SCELTA

   localStorage può lanciare un'eccezione (navigazione privata, cookie di
   terze parti bloccati, spazio esaurito). Non è un errore che deve rompere
   la pagina: se non si può salvare, si tira avanti senza memoria.
   -------------------------------------------------------------------------- */

function leggiScelta() {
  try {
    const valore = localStorage.getItem(CHIAVE_LINGUA);
    return LINGUE[valore] ? valore : null;
  } catch (e) {
    return null;
  }
}

function scriviScelta(lingua) {
  try {
    localStorage.setItem(CHIAVE_LINGUA, lingua);
  } catch (e) {
    /* pazienza: la scelta vale solo per questa visita */
  }
}


/* --------------------------------------------------------------------------
   L'OGGETTO I18N — l'unica cosa che usano gli altri file
   -------------------------------------------------------------------------- */

const I18N = {

  /* La lingua di questa pagina, presa da <html lang>.
     Una lingua che non conosciamo ricade sulla predefinita, l'inglese. */
  get lingua() {
    const dichiarata = document.documentElement.lang;
    return LINGUE[dichiarata] ? dichiarata : LINGUA_PREDEFINITA;
  },

  /* Il locale da passare a Intl.NumberFormat e Intl.DateTimeFormat.
     Non scrivere mai "it-IT" a mano in un tool: usa questo. */
  get locale() {
    return LINGUE[this.lingua].locale;
  },

  /* Traduce una stringa generata dal JavaScript.
     Uso:  I18N.t({ it: "Calcola", en: "Calculate" })

     Serve per il testo che non può stare nell'HTML della pagina: il
     contenuto di un <option>, un messaggio di stato, un errore. */
  t(dizionario) {
    const testo = dizionario[this.lingua];
    return testo === undefined ? dizionario[LINGUA_PREDEFINITA] : testo;
  },

  /* La pagina gemella nell'altra lingua, letta dai tag <link rel="alternate">
     dichiarati nel <head>. Restituisce { lingua, href } oppure null.

     È una funzione e non una proprietà di proposito: viene chiamata quando
     il <body> è già stato letto, quindi non impone nessun vincolo su dove
     mettere questo script dentro il <head>. */
  alternativa() {
    const altra = document.querySelector(
      'link[rel="alternate"][hreflang]' +
      ':not([hreflang="' + this.lingua + '"])' +
      ':not([hreflang="x-default"])'
    );
    if (!altra) return null;

    const lingua = altra.hreflang;
    if (!LINGUE[lingua]) return null;

    return { lingua: lingua, href: altra.getAttribute("href") };
  },

  /* Registra la lingua scelta dall'utente, così la prossima visita a "/"
     ci arriva diretta. */
  salva(lingua) {
    if (LINGUE[lingua]) scriviScelta(lingua);
  },

  /* Quale lingua servire a chi arriva su "/" senza indicazioni.
     Ordine: scelta salvata → lingua del browser → inglese. */
  rilevata() {
    const salvata = leggiScelta();
    if (salvata) return salvata;

    // navigator.language è tipo "en-GB": ci interessano le prime due lettere.
    const browser = String(navigator.language || "").slice(0, 2).toLowerCase();
    return LINGUE[browser] ? browser : LINGUA_PREDEFINITA;
  },
};
