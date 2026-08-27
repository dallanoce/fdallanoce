# Personal website

A static site with no build step: hand-written HTML, CSS and vanilla JavaScript,
served exactly as it sits in the repo. No dependencies, no framework, no lock-in
to any host.

Site copy is in Italian. Documentation is in English.

## Structure

```
.
├── index.html                     home
├── 404.html                       error page
├── _template.html                 copy this for every new page
├── cv/                            CV
├── articoli/                      links out to published articles
├── tools/                         tool index
│   └── calcolatore-tasse/         one tool = one folder
├── stile/                         style guide (noindex, not in the nav)
└── assets/
    ├── css/style.css              the entire stylesheet
    └── js/
        ├── layout.js              shared header and footer
        └── tools/                 per-tool logic
```

## Running it locally

Paths are absolute (`/assets/...`), so opening files with a double click will not
work. Serve the folder instead:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## The three rules that keep the site consistent

**1. One stylesheet, tokens only.** Every colour, size and spacing step is a CSS
variable defined at the top of `style.css`. Pages contain classes, never raw
values. If you need a new colour, add a token rather than a hex in the page. Dark
mode is automatic and works only because everything reads from tokens.

**2. Header and footer are components.** Each page writes `<site-header>` and
`<site-footer>`; the markup comes from `layout.js`. Adding a nav item is a
one-line edit that applies everywhere. Your name and email live in the `SITE`
object at the top of that file.

**3. Always start from something that exists.** New page: copy `_template.html`.
Not sure which classes are available: open `/stile/`, which renders every
component next to the markup to copy.

## Adding a tool

1. `cp _template.html tools/tool-name/index.html`
2. If it needs logic, create `assets/js/tools/tool-name.js` and reference it at
   the bottom of the page. Use `calcolatore-tasse.js` as the model: constants at
   the top, a pure calculation function, interface code last.
3. Add a row to `tools/index.html`, and to the Tool section of `index.html` if
   you want it featured.

## Adding an article

One row in `articoli/index.html`, under the right year. The right-hand column
holds the destination domain (`medium.com`, `linkedin.com`) — that is what a
reader wants to know before clicking.

## Adding a page

Always `folder/index.html`, never `folder.html`. This gives clean URLs (`/cv/`)
that stay stable if the site is ever migrated to a static site generator.

## Deploying to Cloudflare Pages

1. Push this folder to a GitHub repository.
2. In Cloudflare: **Workers & Pages → Create → Pages → Connect to Git**.
3. Build command: leave **empty**. Output directory: `/` (the repository root).
4. Every push publishes automatically.

The site goes live on `project-name.pages.dev`. When you have a custom domain,
attach it under **Custom domains** — no code changes needed, which is the reason
paths are absolute.

## Before going live

- [ ] Replace `Nome Cognome` and `ciao@esempio.it` in `layout.js`
- [ ] Update the name, titles and descriptions in each `index.html`
- [ ] Add `assets/cv.pdf` and re-enable the download link in `cv/index.html`
      (currently commented out), or drop it
- [ ] Verify the rates and coefficients in `calcolatore-tasse.js`
- [ ] Add a favicon under `assets/`

## Conventions and gotchas

See [`CLAUDE.md`](CLAUDE.md). It documents the design direction, the constraints
behind these choices, and the traps worth knowing about before changing anything
— why `layout.js` must not be deferred, the CSS specificity pitfall in `.prose`,
and what the tax constants actually are. Written for Claude Code, useful to any
human picking the repo back up after a few months.
