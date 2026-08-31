# Personal website

A static site with no build step: hand-written HTML, CSS and vanilla JavaScript,
served exactly as it sits in the repo. No dependencies, no framework, no lock-in
to any host.

The site is bilingual, Italian and English, built as two parallel trees under
`/it/` and `/en/`. Italian is the source language. Code comments are in Italian;
documentation is in English.

## Structure

```
.
├── index.html                     router: sends visitors to /it/ or /en/
├── 404.html                       error page
├── _template.it.html              copy this pair for every new page
├── _template.en.html
├── it/                            Italian tree
│   ├── index.html                 home
│   ├── cv/
│   ├── articoli/                  links out to published articles
│   └── strumenti/                 tool index
│       ├── calcolatore-tasse/     one tool = one folder
│       └── grigliata/
├── en/                            English tree, same shape
│   ├── index.html
│   ├── cv/
│   ├── articles/
│   └── tools/
│       ├── tax-calculator/
│       └── barbecue/
├── stile/                         style guide (noindex, Italian only, not in the nav)
└── assets/                        shared by both trees, never duplicated
    ├── css/style.css              the entire stylesheet
    └── js/
        ├── tema.js                light/dark theme, loaded first
        ├── i18n.js                page language and JS strings
        ├── layout.js              header, footer, language switcher, theme switch
        └── tools/                 per-tool logic
```

## Running it locally

Paths are absolute (`/assets/...`), so opening files with a double click will not
work. Serve the folder instead:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000`. It will redirect to `/it/` or `/en/` depending
on your browser's language.

Note that `http.server` does not serve `404.html` for unknown paths — it shows its
own error page. Cloudflare Pages does. Open `/404.html` directly to look at it.

## How the two languages work

Each page is written directly in its own language. There is no runtime text
swapping: `/it/cv/` contains Italian, `/en/cv/` contains English, and both work
with JavaScript disabled.

The URLs are translated too — `/it/articoli/` ↔ `/en/articles/`,
`/it/strumenti/` ↔ `/en/tools/`. Each page names its twin once, in its `<head>`:

```html
<link rel="canonical" href="/en/articles/">
<link rel="alternate" hreflang="it" href="/it/articoli/">
<link rel="alternate" hreflang="en" href="/en/articles/">
<link rel="alternate" hreflang="x-default" href="/it/articoli/">
```

That declaration is the only record of the pairing. Search engines need it, and
the language switcher in the header reads it to know where to go — so there is no
routing table to maintain, and no second place to update. A page that omits it
simply shows no switcher, which is what `/stile/` does.

**The cost of this design:** editing a page means editing two files. Nothing
enforces it. That is the thing to watch when changing content.

## Light and dark

The site is dark by default and has a switch at the right end of the header. The
theme in use is the `data-tema` attribute on `<html>`: `style.css` defines the
dark colours in `:root` and overrides them in `:root[data-tema="chiaro"]`, and
`tema.js` writes the attribute from the choice saved in `localStorage` — first
thing in the `<head>`, so a visitor who picked light never sees a dark flash.

The system preference is deliberately not consulted: everyone starts dark until
they say otherwise. Because every colour in the stylesheet is a token, adding a
component means picking existing tokens and getting both themes for free; writing
a literal colour anywhere means one element stuck in the wrong theme.

## Adding a page

1. `cp _template.it.html it/section/name/index.html`
2. `cp _template.en.html en/section/name/index.html`
3. In both: update the title, the description, and the four language links
4. Link it from the relevant index page in **both** trees

Always `folder/index.html`, never `folder.html`. This gives clean URLs (`/it/cv/`)
that stay stable if the site is ever migrated to a static site generator.

## Adding a tool

1. Create the page in both trees, as above
2. If it needs logic, create **one** `assets/js/tools/tool-name.js` and reference
   it at the bottom of both pages. Use `calcolatore-tasse.js` as the model:
   constants at the top, a pure calculation function, interface code last
3. Add a row to both tool indexes, and to the Tool section of both home pages if
   you want it featured

Both pages use the same element ids so a single file of logic drives them. Text
the tool prints itself — dropdown options, status messages — goes through
`I18N.t({ it, en })`; number formatting uses `I18N.locale`.

## Adding an article

One row in `it/articoli/index.html` and one in `en/articles/index.html`, under the
right year. The right-hand column holds the destination domain (`medium.com`,
`linkedin.com`) — that is what a reader wants to know before clicking.

The title is the real published title and is **identical in both files**. Only the
month is translated.

## Deploying to Cloudflare Pages

1. Push this folder to a GitHub repository.
2. In Cloudflare: **Workers & Pages → Create → Pages → Connect to Git**.
3. Build command: leave **empty**. Output directory: `/` (the repository root).
4. Every push publishes automatically.

The site goes live on `project-name.pages.dev`. When you have a custom domain,
attach it under **Custom domains** — no code changes needed, which is the reason
paths are absolute.

## Before going live

- [x] Replace `Nome Cognome` and `ciao@esempio.it` in `layout.js`
- [x] Update the name, titles and descriptions in all twelve pages
- [x] Fill the CV in both languages
- [ ] Replace the placeholder article rows in `it/articoli/` and `en/articles/`,
      and the one on both home pages, with real published pieces
- [ ] Add `assets/cv.pdf` and re-enable the download link in both `cv/index.html`
      files (currently commented out), or drop it
- [ ] Verify the rates and coefficients in `calcolatore-tasse.js`
- [ ] Add a favicon under `assets/`
- [ ] Once the domain exists, make the `hreflang` and `canonical` URLs absolute —
      Google ignores relative `hreflang`

## Conventions and gotchas

See [`.claude/CLAUDE.md`](.claude/CLAUDE.md). It documents the design direction,
the constraints behind these choices, and the traps worth knowing about before
changing anything — why `tema.js`, `i18n.js` and `layout.js` load in that order
and none may be deferred, why every JS file shares one global scope, the CSS
specificity pitfall in `.prose`, and what the tax constants actually are. Written
for Claude Code, useful to any human picking the repo back up after a few months.
