# Sito personale

Sito statico senza build step: HTML, CSS e JavaScript serviti così come sono.
Nessuna dipendenza, nessun framework, nessun vincolo di piattaforma.

## Struttura

```
.
├── index.html                     home
├── 404.html                       pagina di errore
├── _template.html                 da copiare per ogni pagina nuova
├── cv/                            CV
├── articoli/                      link agli articoli esterni
├── tools/                         indice dei tool
│   └── calcolatore-tasse/         un tool = una cartella
├── stile/                         guida di stile (non indicizzata)
└── assets/
    ├── css/style.css              tutto lo stile del sito
    └── js/
        ├── layout.js              testata e footer condivisi
        └── tools/                 logica dei singoli tool
```

## Le tre regole che tengono il sito omogeneo

**1. Un solo foglio di stile.** Tutti i colori, le dimensioni e le spaziature
sono variabili CSS definite in cima a `style.css`. Nelle pagine non compare mai
un valore grezzo: solo classi. Se serve un colore nuovo, si aggiunge un token,
non un hex nella pagina.

**2. Testata e footer sono componenti.** Ogni pagina scrive `<site-header>` e
`<site-footer>`; il contenuto arriva da `layout.js`. Aggiungere una voce di menu
è una riga in `layout.js` e cambia ovunque. Il nome del sito e l'email si
modificano nell'oggetto `SITE` in cima allo stesso file.

**3. Si parte sempre da un file esistente.** Pagina nuova: copia
`_template.html`. Non sai quali classi esistono: apri `/stile/`, che mostra ogni
componente con il markup da copiare.

## Aggiungere un tool

1. `cp _template.html tools/nome-tool/index.html`
2. Se serve logica, crea `assets/js/tools/nome-tool.js` e collegalo in fondo alla
   pagina. Guarda `calcolatore-tasse.js` come modello: parametri in cima,
   funzione di calcolo pura, interfaccia in fondo.
3. Aggiungi la riga in `tools/index.html` e, se lo vuoi in evidenza, in `index.html`.

## Aggiungere un articolo

Una riga in `articoli/index.html`. A destra va il dominio di destinazione
(`medium.com`, `linkedin.com`), che è quello che il lettore vuole sapere prima
di cliccare.

## Sviluppo in locale

I percorsi sono assoluti (`/assets/...`), quindi aprire i file con doppio clic
non funziona. Serve un server locale:

```sh
python3 -m http.server 8000
```

Poi apri `http://localhost:8000`.

## Deploy su Cloudflare Pages

1. Metti la cartella in un repo Git e caricalo su GitHub.
2. Su Cloudflare: **Workers & Pages → Create → Pages → Connect to Git**.
3. Build command: lascia **vuoto**. Output directory: `/` (la radice).
4. Ogni `git push` pubblica automaticamente.

Il sito è online su `nome-progetto.pages.dev`. Quando avrai il dominio, lo
aggiungi da **Custom domains** senza toccare il codice.

## Prima di andare online

- [ ] Sostituire `Nome Cognome` e `ciao@esempio.it` in `layout.js`
- [ ] Aggiornare nome, titoli e description in ogni `index.html`
- [ ] Mettere il PDF del CV in `assets/cv.pdf` o togliere il link
- [ ] Verificare aliquote e coefficienti in `calcolatore-tasse.js`
- [ ] Aggiungere una favicon in `assets/`
