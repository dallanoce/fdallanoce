# Porcini growth map — feasibility and architecture

**Status:** proposal, nothing decided, nothing built
**Date:** 2026-09-05
**Subject:** adding an interactive map of *Boletus edulis* fruiting probability over Italy
to this site, per `porcini_growth_app_summary.md`

This document exists so the decision is made once, on purpose, and does not have to be
re-derived later. It records what is feasible, what it costs, what it breaks, and what
is still open. It is not an implementation plan.

> Note: Cloudflare Pages serves the repository root, so this file is publicly reachable
> at `/docs/porcini-feasibility.md`, exactly as `README.md` already is. Nothing here is
> sensitive, but be aware of it before writing anything that shouldn't be.

---

## 1. Verdict

**Feasible, on free tiers, without giving up static hosting.**

Cloudflare Pages cannot compute anything on a schedule — it is a file server, and this
repo's build command is deliberately empty. It does not need to. The daily model run
happens elsewhere and its only output is **a handful of image files in a bucket**. The
site remains a pile of static files served as-is.

The obstacles are not technical. They are three of this site's own written constraints,
plus an honesty question about what the map claims to know. Those are in §2 and §9.

---

## 2. Decisions required before any code

These are the real gates. Each one is a deliberate constraint recorded in
`.claude/CLAUDE.md`, so each needs an explicit "yes, we are changing this".

### 2.1 "No framework, no dependencies" — a map library is unavoidable

A slippy map is pan, zoom, tile scheduling, projection maths and gesture handling. It is
not getting hand-written in vanilla JS, and attempting it would be a far worse outcome
than adding the dependency.

| Option | Size (gzipped) | Licence | Notes |
| --- | --- | --- | --- |
| **MapLibre GL JS** | ~200 KB | BSD-3-Clause | WebGL, smooth raster ramps, PMTiles-capable. Matches the target look. |
| **Leaflet** | ~42 KB | BSD-2-Clause | DOM/canvas, raster only, far smaller, coarser visual ceiling. |

Either can be **vendored** as a single file in `/assets/vendor/` and loaded with a plain
`<script>` tag. That preserves *no build step* completely. It does not preserve *no
dependencies*.

For scale: the entire site today is **1.7 MB**, of which the largest file is
`grigliata.js` at 50 KB. MapLibre alone is roughly half the weight of everything
currently in the repo. That is a change in the site's character, not just its bundle.

**Recommendation:** Leaflet if the map is a supporting element; MapLibre if the goal is
the full-bleed app in the reference screenshot. Do not add both.

### 2.2 "Everything runs client-side, tools never call a backend"

The tools index promises visitors:

> *i dati che inserisci restano sul tuo computer e non vengono inviati a nessun server*

Read it precisely — it is about **data the visitor enters**. Downloading a public
forecast raster does not break that sentence. Two features in the reference design do:

- **the geocoder search box** ("Cerca località o indirizzo…") sends the visitor's typed
  address to a third-party geocoding service;
- **a "locate me" button** sends their coordinates.

Choose one: drop both features, or move the privacy promise off the index page and down
into each tool where it is actually true. Leaving the blanket claim in place while
shipping a geocoder makes the site say something false.

### 2.3 "No platform lock-in"

Fully preservable, and §4 is designed around it. The compute is portable Python driven
by a cron; object storage is addressed by URL. Moving the whole thing off Cloudflare
means changing an endpoint and a hostname. **Do not** solve the daily run with
Cloudflare Workers Cron Triggers or Pages Functions — those are precisely the lock-in
the constraint exists to prevent (and see §5 for why they cannot run the code anyway).

### 2.4 Page layout — full-bleed app vs. the site's column

Every page here is a narrow catalogue column inside `.wrap.page`. The reference design
is an edge-to-edge map application with floating panels. These are different products.

Embedding the map inside the existing column is cheap and consistent. Going full-bleed
means a new layout mode, which under this repo's rules must also be added to `/stile/`
in the same change. Decide before writing CSS, not after.

---

## 3. The governing idea

**Precompute, never serve.** Nothing is calculated in response to a request. The
timeline slider in the reference design — 28 Aug through *Oggi* — is not a query. It is
**nine files sitting in a bucket**, and moving the slider changes a URL.

Consequence: there is no API to secure, no rate limit to enforce, no cold start, no
backend that can be down, and no per-visitor cost. A traffic spike costs bandwidth, and
on R2 bandwidth is free.

---

## 4. Architecture

```
┌─ TIER 1 — ONCE, by hand, never repeated ────────────────────────┐
│  CORINE forest mask (host species: beech, chestnut, oak, spruce)│
│  Copernicus DEM GLO-30  ──> slope, aspect, northness            │
│  SoilGrids  ──> texture, water retention                        │
│         └──>  S_static(x) : a single GeoTIFF, committed once    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─ TIER 2 — DAILY, ~06:00 UTC, GitHub Actions cron ───────────────┐
│  1. download yesterday's analysis + today's run (ICON-EU)       │
│  2. update rolling accumulators  P26, T20, thermal shock ──┐    │
│  3. FGI = S_static(x) × f(P26) × g(T20) × S_shock          │    │
│  4. render → Web-Mercator PNG, colour-ramped               │    │
│  5. upload to R2; expire files older than N days           │    │
│                          state persisted in R2 ────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─ TIER 3 — ON PAGE LOAD ─────────────────────────────────────────┐
│  Cloudflare Pages serves /it/strumenti/porcini/  (static)       │
│  map library + basemap + one PNG overlay fetched from R2        │
│  slider selects a different PNG. No API. No query. No server.   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Two optimisations that make it viable

**Bake the static terms once (Tier 1).** Forest presence, host species, slope, aspect
and soil texture do not change daily. Collapsing them into a single `S_static(x)` turns
the daily job into *weather × a lookup*. It also removes ~90% of the grid permanently:
Italian forest and wooded land is roughly a third of the country, so most cells are
structurally zero and compress to almost nothing.

**Keep rolling accumulators, don't re-read history (Tier 2).** The naive reading of the
spec is "every day, fetch 26 days of weather for every grid cell". That is hundreds of
thousands of API calls per run and it will not work on any free tier. Instead, persist
P26 and T20 as rasters in R2; each run adds one day and drops one. This is the single
change that takes the job from impossible to about ten minutes.

---

## 5. Where the daily job runs

| Option | Verdict |
| --- | --- |
| **GitHub Actions scheduled workflow** | **Recommended.** Free on public repos; ~450 of the 2,000 free private minutes/month. Runs the real Python geo stack. Portable: a `.yml` cron plus a script. |
| Cloudflare Workers Cron Triggers | **No.** Cannot run GRIB/raster Python, has CPU limits, and is the lock-in §2.3 forbids. |
| Cloudflare Pages Functions | **No.** Request-time only. Wrong shape entirely. |
| Small VPS with real cron (~€4/mo) | Escape hatch if data volume grows or Actions becomes limiting. |

Two Actions caveats worth knowing before choosing it, not after:

- cron is **best-effort** and can slip 10–30 minutes at busy times — irrelevant for a
  daily biological index;
- **scheduled workflows are auto-disabled after 60 days of repository inactivity.** On a
  personal site that ships in bursts, this is a genuine footgun. Mitigate with a
  monitoring ping, or accept that a quiet quarter silently stops the map.

The job writes to R2 with the `aws` CLI, since R2 is S3-compatible. That is the reason
this is not lock-in: the same script targets S3 or Backblaze by changing an endpoint.

---

## 6. The model, and the "stored model" question

This is where the design most invites over-building, so, explicitly:

**No model server. No inference endpoint. No GPU. No ML hosting bill.** Training and
inference are separated in time.

- **Training** is offline and rare — a laptop or a one-off Actions run, in a *separate*
  repo. Output: one model artifact.
- **Inference** is a `predict()` over a NumPy array inside the daily job that has already
  loaded the rasters. Nothing serves it; nothing waits on it.

### 6.1 Prefer a GLM over a forest — for storage reasons, not accuracy

| Artifact | Stored form | Robustness |
| --- | --- | --- |
| **Logistic regression / GLM** | ~20 float coefficients in a JSON file, committed to git | Survives library upgrades. Auditable. Reimplementable in JS if ever needed. |
| Random Forest / MaxEnt | multi-MB pickle in R2 or a GitHub Release | Breaks silently when the runner's `scikit-learn` updates. Pin the version exactly, or export to ONNX. |

The JSON-coefficients route also means Approach 1 → Approach 2 changes **nothing about
the architecture**. The heuristic curves are replaced by fitted coefficients in the same
step of the same job. That is the main reason to build it in these three tiers.

### 6.2 The sampling-bias problem with Approach 2

GBIF and iNaturalist record **where people walked and posted**, not where mushrooms grew.
Trained naively, the model will place its hotspots along marked trails, near roads, and
around cities — and it will look convincing while doing it. Any serious attempt at
Approach 2 has to confront this (bias-corrected background sampling, target-group
pseudo-absences, spatial block cross-validation) or it is a map of hiker density with a
mycological colour ramp.

---

## 7. Data sources

| Purpose | Source | Access | Notes |
| --- | --- | --- | --- |
| Daily weather + **soil** temp/moisture | **DWD ICON-EU** (~7 km) | `opendata.dwd.de`, GRIB2 | Free. Has `W_SO`/`T_SO` — the variables the spec actually needs. Attribution required. |
| Synoptic backup / forecast | ECMWF open data IFS (0.25°) | free, GRIB2 | CC-BY-4.0. Coarse (~28 km) but reliable. |
| Historical backfill + Approach 2 training | **ERA5-Land** (9 km, hourly) | Copernicus CDS, `cdsapi` | Account + API key; downloads are queued and slow. Do the backfill **once**, then keep the rolling archive. |
| Convenience / point-tool phase | Open-Meteo | REST | Free for non-commercial, ~10k calls/day. Fine for one location, hopeless for a national grid. |
| Forest mask + host species | CORINE Land Cover (100 m, classes 311/312/313); EU-Forest; INFC | download | Attribution required. |
| Terrain | Copernicus DEM GLO-30 | download | Slope, aspect, northness. |
| Soil texture | SoilGrids (ISRIC, 250 m) | download | Sand/clay/organic carbon → retention modifier. |
| Occurrences (Approach 2 only) | GBIF | `pygbif` | **Per-record licences vary** (CC0 / CC-BY / CC-BY-NC). Check before redistributing anything derived. |

Attribution obligations to budget for in the page footer: Copernicus (CORINE, DEM),
DWD, ECMWF, ISRIC, OpenStreetMap (ODbL), and GBIF if used.

---

## 8. Sizing and cost

Italy's bounding box is roughly 12° × 11.7°. Projected to Web Mercator at zoom 8:

- **~2,200 × 2,800 px**, about **450 m/px** at latitude 42 — already finer than 7 km
  weather data justifies; the visible detail comes from the forest mask;
- with ~90% transparent pixels and indexed colour, **300 KB – 1.5 MB per day**;
- nine days is under 10 MB total, and **the client fetches only the selected day** — the
  cost of one photograph.

**Start with one image overlay per day, not tiles.** Pre-project to Web Mercator and it
aligns with the basemap as a plain overlay: no tiling pipeline, no thousands of small
objects, no tile-cache invalidation.

If deep zoom is later wanted, the upgrade is **PMTiles** — one archive per day, read via
HTTP range requests, designed to work on static hosting with no server. Note that such
an archive will exceed **Cloudflare Pages' 25 MiB per-file limit**, which is the concrete
reason the rasters live in **R2** and never in this repository.

**Cost: zero, realistically.** R2's free tier is 10 GB storage, 1M class-A and 10M
class-B operations, and — the reason to prefer it over S3 — **no egress fees**. A viral
day costs operations, not bandwidth. GitHub Actions is free at this volume.

*(Provider limits change. Re-check Pages' file-size/file-count limits, R2's free tier and
Actions' minutes at build time rather than trusting these figures.)*

---

## 9. Honesty and legal

Two things that are not optional.

**The colour bar in the reference design reads 0.00–1.00 to two decimals.** The science
does not support that precision. `f(P26) × g(T20) × S_shock` is a plausibility heuristic
with hand-tuned curves; Approach 2 inherits the sampling bias in §6.2. Ship **three or
four coarse classes with honest labels** rather than a continuous index that implies
knowledge the model does not have. Say plainly on the page what it is: a habitat-and-
weather plausibility index, not a prediction, validated against nothing yet.

**A map telling people where to find porcini needs a real disclaimer.** Foraging in Italy
is regulated regionally — permits, daily quotas, closed periods — and *Boletus* has
genuinely dangerous look-alikes. One short, unmissable block, not a footnote. Consider
also that publishing precise fruiting locations has a conservation dimension; GBIF itself
obscures some coordinates for this reason.

---

## 10. Fitting this site's conventions

Assuming it gets built, the repo's own rules apply and are easy to violate here:

- **Two pages, two trees.** `/it/strumenti/porcini/` and `/en/tools/porcini/` — the slug
  may be identical in both languages, as `cv` already is. Confirm the English slug;
  "porcini" is borrowed in English, so it is defensible.
- **One JS file** at `/assets/js/tools/porcini.js`, shared by both trees, with **element
  ids in Italian** — same rule as `grigliata.js`, same reason.
- **Global scope is shared.** `tema.js`, `i18n.js`, `layout.js`, the vendored map
  library and the tool's own file all load into one scope. `LINGUE`,
  `LINGUA_PREDEFINITA`, `CHIAVE_LINGUA`, `I18N`, `SITE`, `TEMI`, `TEMA_PREDEFINITO`,
  `CHIAVE_TEMA`, `TEMA` are taken; MapLibre would additionally claim `maplibregl`.
  Duplicate top-level `const` is a `SyntaxError` that kills the page.
- **Colour ramp legend entries are design tokens**, and per `CLAUDE.md` a new colour
  token must be defined in **three** places: `:root`, `:root[data-tema="chiaro"]` and the
  `@media print` block.
- **Both themes.** The map's basemap and the overlay ramp must both work on dark and on
  light. A raster PNG baked for the dark theme will look wrong on the light one — decide
  whether to ship two ramps or one theme-neutral one.
- **Generated text goes through `I18N.t({ it, en })`**: legend labels, the date strip,
  status and error messages. Dates and numbers use `I18N.locale`, never a hardcoded
  `"it-IT"`.
- **New reusable components get added to `/stile/` in the same change.**
- Structure `porcini.js` like `grigliata.js`: parameters, then a pure calculation, then
  the interface.

---

## 11. Suggested phasing

1. **A point tool, not a map.** One location, forecast fetched from Open-Meteo in the
   browser, FGI computed in JS. **Zero infrastructure, zero new dependencies**, fits
   every existing constraint — and it validates the biology before a single byte of
   pipeline gets built. If the index turns out not to correspond to reality, this is
   where that is discovered, cheaply.
2. **The static-raster map.** Actions cron + R2 + a vendored map library, still the
   heuristic model. This is the reference screenshot.
3. **Approach 2.** Swap the hand-tuned curves for fitted JSON coefficients in the same
   job. The architecture does not change — which is the entire point of building it in
   three tiers.

Phase 1 is worth doing regardless of whether 2 and 3 ever happen.

---

## 12. Open questions

- Full-bleed map page, or map inside the existing `.wrap.page` column? (§2.4)
- Geocoder search box: drop it, or rewrite the privacy promise? (§2.2)
- MapLibre or Leaflet? (§2.1)
- How many days in the timeline, and is it history, forecast, or both?
- One colour ramp for both themes, or one baked per theme? (§10)
- English slug: `/en/tools/porcini/` or something else?
- What is the validation story — is there *any* ground truth available to check phase 1
  against before phase 2 is built?

---

## 13. Anti-scope — what this explicitly does not need

Recorded because each of these is a plausible-sounding wrong turn:

- a backend framework (FastAPI/Flask) — nothing is computed per request;
- PostgreSQL/PostGIS — there are no queries, only files;
- a model-serving endpoint — inference happens in the batch job;
- Cloudflare Workers or Pages Functions — see §2.3 and §5;
- npm, a bundler, or a build step — the map library is vendored as one file;
- user accounts or crowdsourced reports — phase 2 of the source spec proposes these; they
  would introduce a real backend, real personal data, and moderation, and they are a
  different project from this one.
