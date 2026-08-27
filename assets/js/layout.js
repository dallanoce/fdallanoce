/* ==========================================================================
   LAYOUT.JS — testata e footer condivisi

   COME FUNZIONA
   Definisce due tag HTML personalizzati, <site-header> e <site-footer>.
   Ogni pagina li scrive una riga e basta: il contenuto viene generato qui.
   Modifichi la navigazione in questo file e cambia in tutto il sito.

   Il file va caricato nel <head> SENZA "defer": così i tag risultano già
   definiti quando il browser legge il <body> e la testata compare
   immediatamente, senza sfarfallio.

   Va caricato DOPO i18n.js, da cui usa I18N.lingua e I18N.alternativa()
   per costruire il menu nella lingua giusta.

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
  name: "Nome Cognome",
  email: "ciao@esempio.it",

  nav: {
    it: [
      { href: "/it/cv/",        label: "CV" },
      { href: "/it/articoli/",  label: "Articoli" },
      { href: "/it/strumenti/", label: "Tool" },
    ],
    en: [
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
      const active = path === item.href || path.startsWith(item.href);
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
          </div>
        </div>
      </header>`;

    this.attivaSelettore();
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
