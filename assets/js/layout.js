/* ==========================================================================
   LAYOUT.JS — testata e footer condivisi

   COME FUNZIONA
   Definisce due tag HTML personalizzati, <site-header> e <site-footer>.
   Ogni pagina li scrive una riga e basta: il contenuto viene generato qui.
   Modifichi la navigazione in questo file e cambia in tutto il sito.

   Il file va caricato nel <head> SENZA "defer": così i tag risultano già
   definiti quando il browser legge il <body> e la testata compare
   immediatamente, senza sfarfallio.

   Va caricato DOPO i18n.js e tema.js: dal primo usa I18N.lingua e
   I18N.alternativa() per costruire il menu nella lingua giusta, dal secondo
   TEMA.corrente per disegnare l'interruttore già nella posizione giusta.

   Non usa Shadow DOM di proposito, così il markup generato eredita
   normalmente le classi di style.css.
   ========================================================================== */


/* --------------------------------------------------------------------------
   CONFIGURAZIONE — l'unica parte da modificare

   Le voci di menu sono elencate una volta per lingua. Sono separate e non
   generate da una radice comune perché anche gli indirizzi sono tradotti:
   /it/articoli/ e /en/articles/ non si ricavano l'uno dall'altro.
   -------------------------------------------------------------------------- */

const SITE = {
  name: "Francesco Dallanoce",
  email: "dallanoce.fd@gmail.com",

  // "esatto" serve alla home: il suo indirizzo è il prefisso di tutti gli
  // altri, quindi senza questo flag risulterebbe attiva ovunque nel sito.
  nav: {
    it: [
      { href: "/it/",           label: "Home", esatto: true },
      { href: "/it/cv/",        label: "CV" },
      { href: "/it/articoli/",  label: "Articoli" },
      { href: "/it/strumenti/", label: "Tool" },
    ],
    en: [
      { href: "/en/",          label: "Home", esatto: true },
      { href: "/en/cv/",       label: "CV" },
      { href: "/en/articles/", label: "Articles" },
      { href: "/en/tools/",    label: "Tools" },
    ],
  },
};


/* --------------------------------------------------------------------------
   <site-header>
   -------------------------------------------------------------------------- */

class SiteHeader extends HTMLElement {
  connectedCallback() {
    const lingua = I18N.lingua;
    const path = location.pathname;
    const voci = SITE.nav[lingua] || SITE.nav[LINGUA_PREDEFINITA];

    const links = voci.map((item) => {
      // Una voce è attiva se siamo nella sua sezione o in una pagina figlia.
      // Quelle marcate "esatto" (la home) si accendono solo sul loro
      // indirizzo preciso: attive per prefisso lo sarebbero sempre, e due
      // voci con aria-current="page" insieme sono un errore, non un dettaglio.
      const active = item.esatto
        ? path === item.href
        : path === item.href || path.startsWith(item.href);
      return `<a class="nav__link${active ? " is-active" : ""}"
                 href="${item.href}"
                 ${active ? 'aria-current="page"' : ""}>${item.label}</a>`;
    }).join("");

    this.innerHTML = `
      <a class="skip" href="#contenuto">${I18N.t({
        it: "Vai al contenuto",
        en: "Skip to content",
      })}</a>
      <header class="masthead">
        <div class="wrap masthead__inner">
          <a class="masthead__name" href="${LINGUE[lingua].radice}">${SITE.name}</a>
          <div class="masthead__end">
            <nav class="nav" aria-label="${I18N.t({
              it: "Sezioni del sito",
              en: "Site sections",
            })}">${links}</nav>
            ${this.selettoreLingua(lingua)}
            ${this.interruttoreTema()}
          </div>
        </div>
      </header>`;

    this.attivaSelettore();
    this.attivaInterruttore();
  }

  /* Il menu a tendina delle lingue.

     È un <select> nativo e non un menu costruito a mano: si apre da
     tastiera, lo annunciano i lettori di schermo e si comporta come i
     controlli di sistema sul telefono, tutto senza una riga di codice.

     Compare solo se questa pagina dichiara una gemella: su una pagina che
     esiste in una lingua sola (per esempio /stile/) offrire il cambio
     lingua porterebbe a un 404. */
  selettoreLingua(lingua) {
    const altra = I18N.alternativa();
    if (!altra) return "";

    const opzioni = [lingua, altra.lingua].map((codice) => {
      const scelta = codice === lingua ? " selected" : "";
      return `<option value="${codice}"${scelta}>${LINGUE[codice].breve}</option>`;
    }).join("");

    return `
      <div class="langswitch">
        <select class="langswitch__select" id="selettore-lingua"
                aria-label="${I18N.t({ it: "Lingua", en: "Language" })}">
          ${opzioni}
        </select>
      </div>`;
  }

  /* L'interruttore chiaro/scuro.

     È un <button role="switch">: acceso o spento, un clic per cambiare. Il
     <select> della lingua accanto sembra il fratello ovvio, ma qui sarebbe
     stato peggio — aprire un menu per scegliere fra due cose che si vedono
     già disegnate sull'interruttore.

     A differenza del selettore di lingua, questo c'è su tutte le pagine:
     non dipende da niente che la pagina debba dichiarare.

     Le due icone sono scritte come codici (\u2600 sole, \u263E luna) invece
     che come caratteri: \uFE0E dopo il sole è il selettore di variante che
     impone il disegno testuale, e appiccicato al carattere vero sarebbe
     invisibile in questo file — cioè una riga che sembra sbagliata e non lo
     è. Sono caratteri e non immagini perché così prendono il colore dal
     testo e restano giusti in tutti e due i temi. */
  interruttoreTema() {
    const scuro = TEMA.scuro;

    return `
      <button class="temaswitch" type="button" role="switch"
              id="interruttore-tema"
              aria-checked="${scuro}"
              aria-label="${I18N.t({ it: "Tema scuro", en: "Dark theme" })}">
        <span class="temaswitch__icona" aria-hidden="true">\u2600\uFE0E</span>
        <span class="temaswitch__icona" aria-hidden="true">\u263E</span>
        <span class="temaswitch__pallino" aria-hidden="true"></span>
      </button>`;
  }

  /* Cambiare tema non ricarica niente: i colori sono variabili CSS, cambia
     l'attributo su <html> e la pagina si ridisegna da sola.

     aria-checked non è solo per i lettori di schermo: è anche quello che il
     CSS guarda per spostare il pallino. Un solo attributo da aggiornare,
     nessuna classe da tenere in sincrono con lo stato vero. */
  attivaInterruttore() {
    const bottone = this.querySelector("#interruttore-tema");
    if (!bottone) return;

    bottone.addEventListener("click", () => {
      TEMA.alterna();
      bottone.setAttribute("aria-checked", String(TEMA.scuro));
    });
  }

  /* Cambiare lingua ricarica la pagina gemella. Non è uno scambio di testo
     a runtime: le due lingue sono due documenti diversi, quindi si naviga.
     La scelta viene salvata così che il prossimo arrivo su "/" la rispetti. */
  attivaSelettore() {
    const select = this.querySelector("#selettore-lingua");
    if (!select) return;

    select.addEventListener("change", () => {
      const altra = I18N.alternativa();
      if (!altra || select.value === I18N.lingua) return;

      I18N.salva(altra.lingua);
      location.href = altra.href;
    });
  }
}


/* --------------------------------------------------------------------------
   <site-footer>
   -------------------------------------------------------------------------- */

class SiteFooter extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <footer class="footer">
        <div class="wrap footer__inner">
          <span>© ${new Date().getFullYear()} ${SITE.name}</span>
          <a href="mailto:${SITE.email}">${SITE.email}</a>
        </div>
      </footer>`;
  }
}


customElements.define("site-header", SiteHeader);
customElements.define("site-footer", SiteFooter);
