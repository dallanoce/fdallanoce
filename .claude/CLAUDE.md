# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

The owner's personal website. A hand-written static site: plain HTML, CSS and
vanilla JavaScript, served exactly as it sits in the repo.

The site is **bilingual, Italian and English**, built as **two parallel trees**
under `/it/` and `/en/`. Italian is the source language. Code comments stay in
**Italian**, in the English pages too; this file and `README.md` are in English.

Four areas, each existing twice:

| Area | Italian | English |
| --- | --- | --- |
| Home | `/it/` | `/en/` |
| CV | `/it/cv/` | `/en/cv/` |
| Articles — links **out** to Medium, LinkedIn, dev.to | `/it/articoli/` | `/en/articles/` |
| Tools — self-contained browser utilities | `/it/strumenti/` | `/en/tools/` |

Plus three pages that live outside the trees: `/` (a language router),
`/404.html`, and `/stile/` (the style guide, Italian only, noindex).

## Hard constraints

These are deliberate decisions by the owner. Do not change them without asking first.

- **No build step.** No bundler, no transpiler, no `package.json`. What is in the
  repo is what ships.
- **No framework, no dependencies.** Vanilla JS only. Do not introduce React, Vue,
  Tailwind, a static site generator, or any npm package.
- **No platform lock-in.** The site must stay movable to any static host by copying
  files. Do not use host-specific features (Cloudflare Functions/Workers bindings,
  Netlify-specific redirect syntax, `_redirects`, etc.) without asking. This is why
  the language routing at `/` is client-side JavaScript and not a server rule.
- **Everything runs client-side.** Tools never call a backend and never transmit
  user input anywhere. This is stated to visitors on the tools index; keep it true.

The owner is an experienced developer but **not** a frontend developer. Favour
explicit, boring, well-commented code over clever or terse code. Explain frontend
reasoning when it is not obvious.

## Structure

```
.
├── index.html                     router: manda a /it/ o /en/, non ha contenuto
├── 404.html                       error page, deduce la lingua dall'URL sbagliato
├── _template.it.html              starting point for a new Italian page
├── _template.en.html              …and its English twin
├── it/                            albero italiano
│   ├── index.html
│   ├── cv/index.html
│   ├── articoli/index.html
│   └── strumenti/index.html
│       ├── calcolatore-tasse/     one tool = one folder
│       └── grigliata/
├── en/                            albero inglese, stessa forma
│   ├── index.html
│   ├── cv/index.html
│   ├── articles/index.html
│   └── tools/index.html
│       ├── tax-calculator/
│       └── barbecue/
├── stile/index.html               style guide, noindex, Italian only, not in the nav
└── assets/
    ├── css/style.css              the entire stylesheet
    └── js/
        ├── tema.js                light/dark theme; loaded first, before i18n.js
        ├── i18n.js                page language + JS strings; loaded before layout.js
        ├── layout.js              shared header, footer, language switcher, theme switch
        └── tools/                 per-tool logic, one file each, shared by both trees
```

`assets/` is **not** duplicated per language. One stylesheet, one copy of each
tool's logic.

## The four rules that keep the site consistent

Visual consistency across pages is an explicit goal, and it is enforced by
structure rather than by discipline. Preserve all four.

**1. Design tokens are the only source of raw values.**
Every colour, font size, spacing step and radius is a CSS custom property defined
in `:root` at the top of `style.css`. Pages must never contain a hex value, a
`px` value, or a `style=""` attribute. If a new value is needed, add a token —
do not inline it. **A new colour token must be defined twice**: in `:root` (dark,
the default) and in `:root[data-tema="chiaro"]` (light). Theme switching works
only because everything reads from tokens; one hardcoded colour is one element
that stays the wrong colour in one of the two themes.

**2. Header and footer are components.**
Every page writes `<site-header></site-header>` and `<site-footer></site-footer>`.
The markup comes from `layout.js`. Site name, email and the per-language nav items
live in the `SITE` object at the top of that file. Changing the nav means editing
one file.

**3. A page exists twice, and each copy declares the other.**
Every page has a twin in the other tree, and both name each other with
`<link rel="alternate" hreflang>`. That tag is the *only* place the pairing is
recorded: search engines need it anyway, and the language switcher reads it. Never
add a second mapping of paths somewhere else — one page, one declaration.

**4. New work starts from existing work.**
New page: copy `_template.it.html` **and** `_template.en.html`. Unsure which
classes exist: open `/stile/`, which renders every component next to the markup to
copy. When you add a new reusable component to `style.css`, **add it to `/stile/`
in the same change** — otherwise the catalogue rots and the owner starts inventing
one-off styles.

## Two languages

Italian is the source language: it is what `x-default` points at, what the router
falls back to, and what a visitor gets when nothing else is known.

**The shape of it.** A page's text is written directly in that page's language.
There is no runtime text swapping, no hidden twin elements, no flash of the wrong
language to prevent, and a page with JavaScript disabled still reads correctly in
one language. The cost is the obvious one: **a change to a page is a change to two
files**, and nothing in the repo enforces that you made both. That is the trade
that was chosen deliberately over keeping both languages inside one document.

**URLs are translated too.** `/it/articoli/` ↔ `/en/articles/`,
`/it/strumenti/` ↔ `/en/tools/`, `/it/strumenti/calcolatore-tasse/` ↔
`/en/tools/tax-calculator/`. `cv` happens to be the same word in both. Because the
pairing is declared per page in `hreflang`, translated slugs cost nothing extra to
maintain — there is no central routing table to keep in sync.

**Three places, three mechanisms.** This is the whole system:

| What | How |
| --- | --- |
| Text in a page | Write it in that page's language. Nothing else. |
| Which page is the twin | `<link rel="alternate" hreflang="…">` in `<head>` |
| Text generated by JavaScript | `I18N.t({ it: "…", en: "…" })` |

**What `i18n.js` actually does.** Three things, and nothing more:

- `I18N.lingua` — the page's language, read from `<html lang>`, the single source
  of truth. Never store the language a second time anywhere.
- `I18N.t({ it, en })` — for strings that cannot live in the HTML: the text of an
  `<option>`, a status line, an error message.
- `I18N.alternativa()` — the twin page, read from the `hreflang` tags. Returns
  `null` when the page has no twin.

It also exposes `I18N.locale` for `Intl` (`it-IT` / `en-IE`), `I18N.rilevata()`
for the router, and `I18N.salva()` to remember a choice.

**The language switcher** is a native `<select>` in the masthead, generated by
`layout.js`. Choosing a language saves the choice in `localStorage` and navigates
to the twin. It renders **only when the page declares a twin** — on `/stile/`,
which is Italian only, it is simply absent, because offering it there would lead
to a 404. If the switcher is missing from a new page, the cause is almost always
a missing or wrong `hreflang`.

**Arriving at `/`.** The root `index.html` has no content: it reads the saved
choice, then `navigator.language`, then falls back to Italian, and
`location.replace()`s into the right tree. `replace` and not `href` so the back
button leaves the site instead of bouncing back into the router. Its `<body>` is a
plain two-link language chooser, which is what a visitor without JavaScript gets.

**What is not translated.** Article titles — those are the real published titles,
one per article in every language; only the month next to them is a word of the
site and gets translated. Proper nouns, tags, and Italian tax terms with no
English equivalent (`regime forfettario`, `ATECO`), which are marked
`<span lang="it">` in the English pages. That is a bare `lang` **without** any
class: it tells a screen reader how to pronounce a foreign phrase, which is what
the attribute is actually for.

**Adding a third language** means: an entry in `LINGUE` in `i18n.js`, a nav list
in `SITE.nav` in `layout.js`, a third tree of pages, and a third `hreflang` on
every existing page. The last one is the expensive part, and it is a reason to
stop at two.

## Two themes

Dark is the default and light is the variant, and that asymmetry is written into
the CSS: `:root` holds the dark values, `:root[data-tema="chiaro"]` overrides the
colour tokens with the light ones. Nothing else changes between the two — the
type scale, the spacing and the radii are theme-independent and are declared once.

**The theme in use lives in one place**, the `data-tema` attribute on `<html>`,
exactly the way the language lives in `<html lang>`. The CSS reads it, `tema.js`
reads it back, and the switch's `aria-checked` is derived from it. The attribute
is always written, even for the default, so there is no second rule to remember.

**`prefers-color-scheme` is not consulted.** The site opens dark for everybody,
including a visitor whose machine is set to light. That is deliberate, not an
oversight: the old `@media (prefers-color-scheme: dark)` block was removed when
the switch was added. To follow the system preference again, the place to change
is `temaIniziale()` in `tema.js` — not the stylesheet, because the saved choice
has to keep the last word either way.

**`color-scheme` is set alongside each palette** so the browser draws its own
widgets — scrollbars, the open language dropdown, form controls — on the right
kind of background. Without it a Mac set to light would paint light scrollbars
onto the dark page.

**`tema.js` loads first, and never deferred.** It writes the attribute while the
`<head>` is being parsed, before the `<body>` exists. Deferred, or at the end of
the body, a visitor who chose light would see the page painted dark and then
flip. The script order on every page is `tema.js`, `i18n.js`, `layout.js`.

**The switch is rendered by `layout.js`**, at the right end of the masthead after
the language switcher. Unlike the language switcher it appears on every page:
it depends on nothing the page has to declare, so `/stile/` and `404.html` have
it too. It is a `<button role="switch">` and its state is `aria-checked`, which
is both what a screen reader announces and what the CSS reads to place the knob —
there is no `is-on` class to keep in sync with the truth.

**Changing theme does not navigate.** The colours are custom properties: the
attribute changes and the page repaints. Only the language switcher reloads.

## Recipes

### Add a page

1. `cp _template.it.html it/<sezione>/<nome>/index.html`
2. `cp _template.en.html en/<section>/<name>/index.html`
3. In **both**: update `<title>` and `<meta name="description">`, and fix the four
   language links — `canonical` points at the page itself, the two `alternate`
   tags point at the Italian and English versions, and `x-default` always points
   at the **Italian** one
4. Write the body using classes from `/stile/`
5. Link it from the relevant index page **in both trees** — the link text and the
   path in `.index__meta` are both language-specific

Always `folder/index.html`, never `folder.html` — this gives clean URLs (`/it/cv/`)
that stay stable if the site is ever migrated to a generator.

### Add a tool

1. Create the page in both trees, as above
2. Create **one** `assets/js/tools/<name>.js`, referenced at the bottom of both
   pages
3. Add a row to both tool indexes, and to the Tool section of both home pages if
   featured

**Both pages use the same element ids**, in Italian, and that is deliberate: one
file of logic serves both trees, so an update to a rate cannot reach one language
and miss the other. Ids are invisible to visitors; a duplicated calculation would
not be.

Labels, hints and buttons live in each page's HTML, in that page's language.
Anything the tool prints itself — `<option>` text, status messages, errors — goes
through `I18N.t({ it, en })`, because an `<option>` cannot contain markup. If the
tool formats numbers, pass `I18N.locale` to `Intl`; never write `"it-IT"` by hand.

Follow the structure of `calcolatore-tasse.js`, which is the reference model:

```
1. PARAMETERS  — all domain constants, isolated at the top
2. CALCULATION — a pure function: data in, result out, no DOM access
3. INTERFACE   — read inputs, render outputs, wire events
```

Keeping parameters separate from the calculation means an annual rate update is
a two-line edit rather than a hunt through the code.

### Add an article

One `<a class="index__row">` in `it/articoli/index.html` **and** one in
`en/articles/index.html`, grouped under the right year. `index__meta` holds the
destination **domain** (`medium.com`), which is what the reader wants to know
before clicking.

The article's title is the real published title: **identical in both files**, not
translated. Only the month in `index__note` differs.

## Design direction

Written down so future changes do not drift.

- **Concept.** The site is fundamentally an index: it points outward to articles
  and inward to tools. The design leans into that — it reads like a catalogue.
- **Signature element.** Every `.index__row` shows its destination in monospace
  on the right: an external domain for articles, an internal path for tools.
  This is information, not decoration; it distinguishes links that leave the
  site from links that stay. Keep it on every new index row.
- **Type.** Space Grotesk for everything, IBM Plex Mono for metadata, labels,
  paths and numbers. Space Grotesk is used for body copy too, which works here
  because body copy on this site is always short — the long-form writing lives
  on external platforms.
- **Colour.** Near-black paper with a green undertone, off-white ink, pine green
  accent — that is the default, dark. The light theme is the same palette turned
  over: cool off-white paper, near-black ink. Both are defined in `style.css`,
  dark in `:root` and light in `:root[data-tema="chiaro"]`.
- **Motion.** Minimal and deliberate: one micro-interaction, the accent bar that
  scales in on `.index__row` hover. Do not scatter more animation around.
  `prefers-reduced-motion` is respected globally.
- **Avoid.** Do not drift toward cream background + serif display + terracotta
  accent, or near-black + acid-green. Both are generic AI-portfolio defaults and
  were explicitly rejected.

## Gotchas

- **Every `.js` file shares one global scope.** They are plain `<script>` tags,
  not modules, so two files loaded on the same page cannot both declare a
  top-level `const` with the same name — that is a `SyntaxError` that kills the
  whole page, not a warning. This is why the code writes `I18N.t(...)` in full
  instead of aliasing it to a short `t` in each file. Watch for it whenever you
  add a tool: `tema.js`, `i18n.js`, `layout.js` and the tool's file are all
  loaded together. `LINGUE`, `LINGUA_PREDEFINITA`, `CHIAVE_LINGUA`, `I18N`,
  `SITE`, `TEMI`, `TEMA_PREDEFINITO`, `CHIAVE_TEMA` and `TEMA` are taken.
  **Function names collide too, and more quietly:** a second `function
  leggiScelta()` does not throw, it silently replaces the first. That is why
  `tema.js` calls its own pair `leggiTema` / `scriviTema`.
- **The three scripts load in order — `tema.js`, `i18n.js`, `layout.js` — and
  none may be deferred.** `layout.js` reads `I18N.lingua`, `I18N.alternativa()`
  and `TEMA.scuro` while building the header; `tema.js` has to write `data-tema`
  before the first paint. All three sit in `<head>` without `defer` so the custom
  elements are defined before `<body>` is parsed and neither the header nor the
  theme flashes. The files are tiny; the blocking cost is negligible.
- **`i18n.js` does not care where it sits relative to the `hreflang` tags,**
  because `I18N.alternativa()` is a function called after the body is parsed, not
  a value read at load time. Do not "optimise" it into a property.
- **`x-default` always points at the Italian page,** in the English files too.
  It tells search engines what to serve someone who matches no language, and the
  answer is always Italian. It is not a copy-paste mistake.
- **The `hreflang` and `canonical` URLs are root-relative** (`/en/cv/`), because
  the custom domain has not been chosen. The switcher works fine with these.
  Google, however, wants fully-qualified `hreflang` URLs — see the TODOs.
- **Changing language navigates.** The two languages are two documents, so the
  switcher does a real page load. There is no partial state to reason about.
- **The custom elements use light DOM, not Shadow DOM,** so the global stylesheet
  applies to the generated markup. Do not attach a shadow root.
- **All paths are absolute** (`/assets/...`, `/it/cv/`), so pages work at any
  folder depth. The trade-off: opening a file directly with `file://` does not
  work, and neither does the router at `/`. Local development needs
  `python3 -m http.server 8000`.
- **`python3 -m http.server` does not serve `404.html`.** It returns its own error
  page, so the 404 cannot be tested by browsing to a bad URL locally. Cloudflare
  Pages does serve it. To test it locally, open `/404.html` directly — note that
  it will pick its language from the browser, since the path has no `/it/` or
  `/en/` prefix to read.
- **The print palette needs both selectors.** `@media print` redefines the
  colour tokens (black on white, greys instead of green: the paper is white and
  browsers print text colour but not backgrounds). It is written
  `:root, :root[data-tema="chiaro"]` because a plain `:root` is 0,1,0 and would
  lose to the light theme's `:root[data-tema="chiaro"]` at 0,2,0 — someone
  printing from the light theme would get screen colours. The bare `:root` is
  still needed for a visitor with JavaScript off, who has no attribute at all.
- **A new colour token has to be added in three places:** `:root` (dark),
  `:root[data-tema="chiaro"]` (light) and the `@media print` block.
- **CSS specificity trap in `.prose`.** Vertical rhythm comes from
  `.prose > * + * { margin-top }` (specificity 0,1,0). Adding a type-based reset
  like `.prose p { margin: 0 }` (0,1,1) silently wins and collapses the spacing.
  This bug has already been fixed once. Reset with `.prose > * { margin: 0 }` and
  target exceptions with `.prose > * + h2`.
- **`.langswitch__select option` needs an explicit background.** The control is
  transparent so it sits on the masthead, but the open dropdown is drawn by the
  OS and does not inherit that — without the rule, dark mode gives light text on
  a light menu.
- **`Intl.NumberFormat("it-IT")` uses min-2 grouping.** `1442` renders without a
  thousands separator while `39000` renders as `39.000`. That is correct Italian
  convention, not a bug. The locale follows the site language, so the same
  figure reads `38.391 €` in Italian and `€38,391` in English — different
  strings, same number.
- **`grigliata.js` stores keys, never translated text.** `localStorage` is
  per-origin, so the same saved barbecue is read by both `/it/strumenti/grigliata/`
  and `/en/tools/barbecue/`. A built-in product is saved as `{chiave: "salamelle",
  nome: null}` and translated when drawn; a product the visitor typed has no key
  and keeps its literal name in both languages. Saving the displayed name instead
  would give anyone who switches language a half-translated shopping list. The
  same applies to category keys (`carne`, not `Carne`).
- **The tax constants in `calcolatore-tasse.js` are placeholders.** Rates and
  coefficients change yearly and have not been verified against Agenzia delle
  Entrate or INPS. Both pages carry a visible disclaimer. Never present the
  output as authoritative, and never remove the disclaimer.

## Deployment

Cloudflare Pages, connected to this repo. Build command: **empty**. Output
directory: the repository root. Every push to the default branch publishes.

The custom domain has not been chosen yet, so the site currently lives on a
`*.pages.dev` subdomain.

## Before finishing a change

- **The change was made in both trees**, and the two pages still say the same
  thing — this is the failure mode of this design, check it first
- Both pages were actually opened, and the switcher moves between them (not to
  the home page — that means a wrong `hreflang`)
- `canonical` points at the page itself; the two `alternate` tags point at each
  other; `x-default` points at the Italian page
- No `style=""` attributes and no raw hex or `px` values in any HTML file
- Every class used in HTML exists in `style.css`
- Internal links resolve (`folder/` must have an `index.html`)
- New page includes `style.css`, `tema.js`, `i18n.js`, `layout.js`,
  `<site-header>` and `<site-footer>`, with the three scripts in that order and
  none of them deferred
- New reusable component is documented in `/stile/`
- Checked at a narrow viewport, in **both themes**; keyboard focus is visible

## Open TODOs

- Replace the placeholder article rows — one per tree in `it/articoli/` and
  `en/articles/`, plus the featured one on both home pages. They still point at
  `medium.com/@utente` and `dev.to/utente`; the real Medium, LinkedIn and dev.to
  handles are not known yet
- Add `assets/cv.pdf` and re-enable the download link in both `cv/index.html`
  files (currently commented out). Decide whether there is one PDF or one per
  language
- Verify and update the tax constants
- Add a favicon
- Choose and attach the custom domain. When it exists, make the `hreflang` and
  `canonical` URLs absolute (`https://dominio/it/cv/`): Google ignores relative
  `hreflang`. Nothing else needs to change — that is why paths are absolute
