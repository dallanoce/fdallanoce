/* ==========================================================================
   LAYOUT.JS — testata e footer condivisi

   COME FUNZIONA
   Definisce due tag HTML personalizzati, <site-header> e <site-footer>.
   Ogni pagina li scrive una riga e basta: il contenuto viene generato qui.
   Modifichi la navigazione in questo file e cambia in tutto il sito.

   Il file va caricato nel <head> SENZA "defer": così i tag risultano già
   definiti quando il browser legge il <body> e la testata compare
   immediatamente, senza sfarfallio.

   Non usa Shadow DOM di proposito, così il markup generato eredita
   normalmente le classi di style.css.
   ========================================================================== */


/* --------------------------------------------------------------------------
   CONFIGURAZIONE — l'unica parte da modificare
   -------------------------------------------------------------------------- */

const SITE = {
  name: "Nome Cognome",
  email: "ciao@esempio.it",

  // Le voci di menu. Aggiungerne una qui la fa comparire ovunque.
  nav: [
    { href: "/cv/",       label: "CV" },
    { href: "/articoli/", label: "Articoli" },
    { href: "/tools/",    label: "Tool" },
  ],
};


/* --------------------------------------------------------------------------
   <site-header>
   -------------------------------------------------------------------------- */

class SiteHeader extends HTMLElement {
  connectedCallback() {
    const path = location.pathname;

    const links = SITE.nav.map((item) => {
      // Una voce è attiva se siamo nella sua sezione o in una pagina figlia.
      const active = path === item.href || path.startsWith(item.href);
      return `<a class="nav__link${active ? " is-active" : ""}"
                 href="${item.href}"
                 ${active ? 'aria-current="page"' : ""}>${item.label}</a>`;
    }).join("");

    this.innerHTML = `
      <a class="skip" href="#contenuto">Vai al contenuto</a>
      <header class="masthead">
        <div class="wrap masthead__inner">
          <a class="masthead__name" href="/">${SITE.name}</a>
          <nav class="nav" aria-label="Sezioni del sito">${links}</nav>
        </div>
      </header>`;
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
