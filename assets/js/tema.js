/* ==========================================================================
   TEMA.JS — tema chiaro e tema scuro

   COME FUNZIONA
   Il sito è scuro di default. Chi preferisce il chiaro lo chiede con
   l'interruttore in alto a destra, e la scelta resta salvata per le visite
   successive.

   Il tema in uso è scritto in un solo posto: l'attributo data-tema sul tag
   <html>. Da lì lo legge il CSS (:root[data-tema="chiaro"] ridefinisce i
   colori) e da lì lo rilegge questo file. Stesso schema della lingua, che
   vive in <html lang> e non è ripetuta da nessun'altra parte.

   PERCHÉ QUESTO FILE VA NEL <head> SENZA "defer"
   L'attributo deve essere sul posto prima che il browser disegni la pagina.
   Con uno script differito, o messo in fondo al <body>, chi ha scelto il
   tema chiaro vedrebbe prima una pagina scura e poi il ribaltamento: il
   classico sfarfallio. Il file è minuscolo, il costo del blocco è nulla.

   PERCHÉ NON SEGUE LE IMPOSTAZIONI DEL SISTEMA
   Il sito parte scuro per tutti, anche per chi ha il computer in chiaro:
   è una decisione di progetto, non una dimenticanza. La @media
   (prefers-color-scheme) che c'era prima nel CSS è stata tolta apposta.
   Se un giorno si volesse tornare a seguire il sistema, il punto da
   cambiare è la funzione temaIniziale() qui sotto — non il foglio di
   stile, perché la scelta salvata deve comunque avere l'ultima parola.
   ========================================================================== */


/* --------------------------------------------------------------------------
   CONFIGURAZIONE
   -------------------------------------------------------------------------- */

const TEMI = ["scuro", "chiaro"];

const TEMA_PREDEFINITO = "scuro";   // il sito nasce scuro
const CHIAVE_TEMA = "tema";         // dove l'interruttore salva la scelta


/* --------------------------------------------------------------------------
   LETTURA E SCRITTURA DELLA SCELTA

   Come per la lingua: localStorage può lanciare un'eccezione (navigazione
   privata, cookie di terze parti bloccati, spazio esaurito) e non è un
   errore che deve rompere la pagina. Se non si può salvare, si tira avanti
   senza memoria.

   I nomi sono leggiTema/scriviTema e non leggiScelta/scriviScelta perché
   quelli esistono già in i18n.js: gli script del sito condividono un unico
   spazio globale, e due funzioni con lo stesso nome non danno errore — la
   seconda sostituisce la prima in silenzio.
   -------------------------------------------------------------------------- */

function leggiTema() {
  try {
    const valore = localStorage.getItem(CHIAVE_TEMA);
    return TEMI.indexOf(valore) !== -1 ? valore : null;
  } catch (e) {
    return null;
  }
}

function scriviTema(tema) {
  try {
    localStorage.setItem(CHIAVE_TEMA, tema);
  } catch (e) {
    /* pazienza: la scelta vale solo per questa visita */
  }
}

/* Il tema con cui aprire la pagina: quello scelto in passato, altrimenti
   il predefinito. */
function temaIniziale() {
  return leggiTema() || TEMA_PREDEFINITO;
}


/* --------------------------------------------------------------------------
   L'OGGETTO TEMA — l'unica cosa che usano gli altri file
   -------------------------------------------------------------------------- */

const TEMA = {

  /* Il tema di questa pagina, preso da <html data-tema>.
     Un valore che non conosciamo ricade sul predefinito. */
  get corrente() {
    const dichiarato = document.documentElement.dataset.tema;
    return TEMI.indexOf(dichiarato) !== -1 ? dichiarato : TEMA_PREDEFINITO;
  },

  /* Vero se siamo sullo scuro. Comodo per l'interruttore, che ragiona
     per acceso/spento e non per nome del tema. */
  get scuro() {
    return this.corrente === "scuro";
  },

  /* Mette il tema sulla pagina, senza ricordarlo.
     Serve all'avvio: applicare non è scegliere. */
  applica(tema) {
    if (TEMI.indexOf(tema) === -1) return;
    document.documentElement.dataset.tema = tema;
  },

  /* Sceglie un tema: lo applica e lo ricorda. Questo lo chiama
     l'interruttore, e nessun altro. */
  imposta(tema) {
    if (TEMI.indexOf(tema) === -1) return;
    this.applica(tema);
    scriviTema(tema);
  },

  /* Passa da un tema all'altro e restituisce quello nuovo. */
  alterna() {
    const nuovo = this.scuro ? "chiaro" : "scuro";
    this.imposta(nuovo);
    return nuovo;
  },
};


/* --------------------------------------------------------------------------
   AVVIO

   Succede subito, mentre il <head> viene letto e il <body> non esiste
   ancora: è tutto il senso di caricare questo file per primo.

   L'attributo viene scritto sempre, anche quando il tema è quello
   predefinito. Costa niente e tiene una regola sola: il tema in uso si
   legge da data-tema, punto — non "da data-tema, oppure è lo scuro se
   l'attributo non c'è".
   -------------------------------------------------------------------------- */

TEMA.applica(temaIniziale());
