/* ==========================================================================
   CORRELATION FIELD GUIDE — misurare la dipendenza, gestire la ridondanza

   Struttura del file:
     0. NUMERICA    — generatore, coefficienti, utilità: funzioni pure
     1. LABORATORIO 1 — lo stesso campione letto da cinque misure
     2. LABORATORIO 2 — che cosa fa la collinearità a un coefficiente
     3. LABORATORIO 3 — potatura di una matrice di correlazione
     4. AVVIO

   DUE LINGUE, UN FILE SOLO
   Questo file serve sia /it/strumenti/correlazione/ sia /en/tools/correlation/.
   Le due pagine hanno gli stessi id proprio per questo: la matematica esiste
   una volta sola. Tutto il testo che stampa il codice — etichette delle
   forme, note sotto i numeri, verdetti — passa da I18N.t({ it, en }), perché
   non può stare nell'HTML della pagina.

   TUTTO SI CALCOLA QUI
   Nessun dato precalcolato e nessuna chiamata di rete: i campioni li genera
   un generatore con seme, quindi ogni lettura è riproducibile e la pagina
   funziona anche senza connessione.

   NIENTE COLORI NEL CODICE
   Il JavaScript assegna solo classi (.pt, .ax, .gridline…) e passa i numeri
   calcolati alle proprietà --da, --largo, --tinta, --mescola. I colori
   stanno nei token della sezione 1 di style.css, ed è per questo che anche i
   grafici cambiano insieme al tema.
   ========================================================================== */


/* --------------------------------------------------------------------------
   0. NUMERICA
   -------------------------------------------------------------------------- */

/* Generatore pseudocasuale con seme: stesso seme, stesso campione. Serve
   perché un "ricampiona" deve dare un campione nuovo, ma un cambio di tema o
   un ridisegno no. */
function mulberry32(seme) {
  return function () {
    seme |= 0; seme = seme + 0x6D2B79F5 | 0;
    let t = Math.imul(seme ^ seme >>> 15, 1 | seme);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function creaRng(seme) {
  const u = mulberry32(seme);
  /* Box–Muller; u1 tenuto lontano da 0 perché il logaritmo resti finito */
  u.normale = function () {
    const u1 = Math.max(u(), 1e-12), u2 = u();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  return u;
}

const media = v => v.reduce((s, x) => s + x, 0) / v.length;

const devStd = v => {
  const m = media(v);
  return Math.sqrt(v.reduce((s, x) => s + (x - m) * (x - m), 0) / (v.length - 1));
};

function pearson(x, y) {
  const mx = media(x), my = media(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) {
    const a = x[i] - mx, b = y[i] - my;
    sxy += a * b; sxx += a * a; syy += b * b;
  }
  const d = Math.sqrt(sxx * syy);
  return d === 0 ? 0 : sxy / d;
}

/* Ranghi medi, che è il modo in cui Spearman vuole trattati i pari merito. */
function ranghi(v) {
  const idx = v.map((val, i) => [val, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(v.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const medio = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k][1]] = medio;
    i = j + 1;
  }
  return r;
}

const spearman = (x, y) => pearson(ranghi(x), ranghi(y));

function kendallTauB(x, y) {
  const n = x.length;
  let C = 0, D = 0, tx = 0, ty = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = Math.sign(x[i] - x[j]), b = Math.sign(y[i] - y[j]);
      if (a === 0 && b === 0) { tx++; ty++; }
      else if (a === 0) tx++;
      else if (b === 0) ty++;
      else if (a * b > 0) C++;
      else D++;
    }
  }
  const n0 = n * (n - 1) / 2;
  const den = Math.sqrt((n0 - tx) * (n0 - ty));
  return den === 0 ? 0 : (C - D) / den;
}

/* Matrice delle distanze doppiamente centrata: il cuore della covarianza di
   distanza di Székely–Rizzo. */
function distCentrata(v) {
  const n = v.length;
  const A = [];
  const mediaRiga = new Array(n).fill(0);
  let generale = 0;
  for (let i = 0; i < n; i++) {
    A.push(new Float64Array(n));
    for (let j = 0; j < n; j++) {
      const d = Math.abs(v[i] - v[j]);
      A[i][j] = d; mediaRiga[i] += d; generale += d;
    }
    mediaRiga[i] /= n;
  }
  generale /= n * n;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      A[i][j] = A[i][j] - mediaRiga[i] - mediaRiga[j] + generale;
  return A;
}

function correlazioneDistanza(x, y) {
  const n = x.length;
  const A = distCentrata(x), B = distCentrata(y);
  let dcov = 0, dvx = 0, dvy = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      dcov += A[i][j] * B[i][j];
      dvx += A[i][j] * A[i][j];
      dvy += B[i][j] * B[i][j];
    }
  }
  const den = Math.sqrt(Math.sqrt(dvx / (n * n)) * Math.sqrt(dvy / (n * n)));
  if (den === 0) return 0;
  return Math.sqrt(Math.max(0, dcov / (n * n))) / den;
}

/* Bin a frequenza uguale: i valori pari merito finiscono sempre nello stesso
   bin, che è quello che serve alla forma ordinale. */
function binQuantili(v, bin) {
  const s = [...v].sort((a, b) => a - b);
  const bordi = [];
  for (let k = 1; k < bin; k++) bordi.push(s[Math.floor(k * v.length / bin)]);
  return v.map(val => {
    let b = 0;
    while (b < bordi.length && val >= bordi[b]) b++;
    return b;
  });
}

function miNormalizzata(x, y, bin) {
  const n = x.length;
  const bx = binQuantili(x, bin), by = binQuantili(y, bin);
  const congiunta = new Map(), px = new Map(), py = new Map();
  for (let i = 0; i < n; i++) {
    const k = bx[i] + "|" + by[i];
    congiunta.set(k, (congiunta.get(k) || 0) + 1);
    px.set(bx[i], (px.get(bx[i]) || 0) + 1);
    py.set(by[i], (py.get(by[i]) || 0) + 1);
  }
  let I = 0;
  congiunta.forEach((c, k) => {
    const [a, b] = k.split("|").map(Number);
    const p = c / n, pa = px.get(a) / n, pb = py.get(b) / n;
    I += p * Math.log(p / (pa * pb));
  });
  let hx = 0, hy = 0;
  px.forEach(c => { const p = c / n; hx -= p * Math.log(p); });
  py.forEach(c => { const p = c / n; hy -= p * Math.log(p); });
  const d = Math.sqrt(hx * hy);
  return d === 0 ? 0 : Math.max(0, Math.min(1, I / d));
}

/* I numeri di questa pagina sono coefficienti, non importi: la formattazione
   segue comunque la lingua della pagina, così la virgola decimale è quella
   giusta. Il meno è il segno tipografico U+2212, non il trattino. */
function fmtNum(valore, decimali = 2) {
  const f = new Intl.NumberFormat(I18N.locale, {
    minimumFractionDigits: decimali,
    maximumFractionDigits: decimali,
  });
  const s = f.format(Math.abs(valore));
  return (valore < 0 && parseFloat(s.replace(",", ".")) !== 0 ? "−" : "") + s;
}

const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(nome, attributi) {
  const e = document.createElementNS(SVGNS, nome);
  for (const k in attributi) e.setAttribute(k, attributi[k]);
  return e;
}

/* Le tessere dei numeri sono .stat della sezione 10 con in più la barretta.
   Si costruiscono con il DOM e non con innerHTML perché la posizione della
   barretta è un numero calcolato, e un numero calcolato entra nel foglio di
   stile solo passando da una proprietà personalizzata. */
function tessera(chiave, valore, nota, barra) {
  const box = document.createElement("div");
  box.className = "stat";

  const k = document.createElement("span");
  k.className = "stat__key";
  k.textContent = chiave;
  box.appendChild(k);

  const v = document.createElement("span");
  v.className = "stat__value";
  v.textContent = valore;
  box.appendChild(v);

  if (barra) {
    const meter = document.createElement("span");
    meter.className = "stat__meter";
    const fill = document.createElement("span");
    fill.className = "stat__fill" + (barra.negativa ? " stat__fill--neg" : "");
    fill.style.setProperty("--da", barra.da + "%");
    fill.style.setProperty("--largo", barra.largo + "%");
    meter.appendChild(fill);
    box.appendChild(meter);
  }

  const n = document.createElement("span");
  n.className = "stat__note";
  n.textContent = nota;
  box.appendChild(n);

  return box;
}

function riempi(elemento, tessere) {
  elemento.textContent = "";
  tessere.forEach(t => elemento.appendChild(t));
}


/* --------------------------------------------------------------------------
   1. LABORATORIO 1 — lo stesso campione letto da cinque misure
   -------------------------------------------------------------------------- */

const N_CAMPIONE = 200;

/* Ogni forma è un generatore più la didascalia che spiega che cosa c'è da
   vedere. Le forme con due gruppi marcano i punti con g, che serve sia al
   colore sia al verdetto. */
const FORME = [
  {
    id: "lineare",
    etichetta: { it: "lineare", en: "linear" },
    didascalia: {
      it: "Una relazione a linea retta con rumore additivo. È l'unico caso in cui tutte e cinque le misure vanno più o meno d'accordo.",
      en: "A straight-line relation with additive noise. This is the one case where all five measures broadly agree.",
    },
    gen: (rng, nz) => Array.from({ length: N_CAMPIONE }, () => {
      const x = rng.normale();
      return { x, y: x + nz * 2.4 * rng.normale() };
    }),
  },
  {
    id: "monotona",
    etichetta: { it: "curva ma monotona", en: "curved but monotonic" },
    didascalia: {
      it: "Una curva esponenziale. Pearson la sottostima perché la relazione non è dritta; alle misure di rango non importa.",
      en: "An exponential curve. Pearson understates it because the relation is not straight; the rank measures do not care.",
    },
    gen: (rng, nz) => Array.from({ length: N_CAMPIONE }, () => {
      const x = rng();
      return { x, y: Math.exp(5.5 * x) + nz * 18 * rng.normale() };
    }),
  },
  {
    id: "quadratica",
    etichetta: { it: "a U", en: "U-shape" },
    didascalia: {
      it: "Una parabola perfetta. Pearson e tutte e due le misure di rango leggono quasi zero — la relazione non è monotona. La vedono solo la correlazione di distanza e l'informazione mutua.",
      en: "A perfect parabola. Pearson and both rank measures read near zero — the relation is not monotonic. Only distance correlation and MI see it.",
    },
    /* x percorsa in modo simmetrico invece che estratta a caso, così la
       parabola è esattamente bilanciata e r è zero per costruzione e non per
       fortuna */
    gen: (rng, nz) => Array.from({ length: N_CAMPIONE }, (_, i) => {
      const x = -1 + 2 * (i + 0.5) / N_CAMPIONE;
      return { x, y: x * x + nz * 0.32 * rng.normale() };
    }),
  },
  {
    id: "anello",
    etichetta: { it: "anello", en: "ring" },
    didascalia: {
      it: "Punti su una circonferenza: x e y sono del tutto incorrelate e del tutto dipendenti — fissa x e le y possibili sono due. Qui la correlazione di distanza supera di poco la sua soglia di indipendenza, mentre l'informazione mutua a bin supera la propria di parecchio. Nessuna delle due è sempre la più potente.",
      en: "Points on a circle: x and y are completely uncorrelated and completely dependent — fix x and there are only two possible y. Note that distance correlation only just clears its independence baseline here, while binned MI clears its own by a mile. Neither is uniformly the more powerful test.",
    },
    gen: (rng, nz) => Array.from({ length: N_CAMPIONE }, (_, i) => {
      const a = 2 * Math.PI * (i + 0.5) / N_CAMPIONE;
      const r = 1 + nz * 0.22 * rng.normale();
      return { x: Math.cos(a) * r, y: Math.sin(a) * r };
    }),
  },
  {
    id: "anomali",
    etichetta: { it: "cinque valori anomali", en: "five outliers" },
    didascalia: {
      it: "Una nuvola incorrelata più cinque punti a leva alta. Pearson è fabbricato quasi per intero da quelle cinque righe; le misure di rango si rifiutano di muoversi.",
      en: "An uncorrelated cloud plus five high-leverage points. Pearson is manufactured almost entirely by those five rows; the rank measures refuse to be moved.",
    },
    gen: (rng, nz) => {
      const p = [];
      for (let i = 0; i < N_CAMPIONE - 5; i++)
        p.push({ x: rng.normale() * 0.7, y: rng.normale() * 0.7 * (1 + nz), g: 0 });
      for (let i = 0; i < 5; i++) {
        const t = 4 + rng() * 1.3;
        p.push({ x: t, y: t + rng.normale() * 0.35, g: 1 });
      }
      return p;
    },
    legenda: [
      ["1", { it: "grosso del campione", en: "bulk of the sample" }],
      ["2", { it: "cinque punti a leva", en: "five leverage points" }],
    ],
  },
  {
    id: "simpson",
    etichetta: { it: "due gruppi", en: "two clusters" },
    didascalia: {
      it: "Paradosso di Simpson: dentro ogni gruppo la pendenza è negativa, quella complessiva è positiva. Il coefficiente descrive una popolazione che non esiste.",
      en: "Simpson's paradox: the slope inside each group is negative, the pooled slope is positive. The coefficient describes a population that does not exist.",
    },
    gen: (rng, nz) => {
      const p = [];
      for (let g = 0; g < 2; g++)
        for (let i = 0; i < N_CAMPIONE / 2; i++) {
          const t = rng() * 2;
          p.push({ x: g * 3 + t, y: g * 3.4 - 0.8 * t + nz * 0.9 * rng.normale(), g });
        }
      return p;
    },
    legenda: [
      ["1", { it: "gruppo A", en: "group A" }],
      ["3", { it: "gruppo B", en: "group B" }],
    ],
  },
  {
    id: "eteroschedastica",
    etichetta: { it: "varianza a ventaglio", en: "fanning variance" },
    didascalia: {
      it: "La relazione è lineare ma il rumore cresce con x. Il coefficiente va bene; ogni errore standard e ogni intervallo costruito su di lui no.",
      en: "The relation is linear but the noise grows with x. The coefficient is fine; every standard error and interval built on it is not.",
    },
    gen: (rng, nz) => Array.from({ length: N_CAMPIONE }, () => {
      const x = rng() * 3;
      return { x, y: 0.9 * x + nz * 1.5 * rng.normale() * x };
    }),
  },
  {
    id: "ordinale",
    etichetta: { it: "ordinale grossolana", en: "coarse ordinal" },
    didascalia: {
      it: "y è stata schiacciata su cinque livelli ordinati, quindi quasi tutte le coppie sono pari merito su y. Tau-b quelle coppie le toglie dal denominatore e Spearman no — una delle ragioni per cui i due coefficienti stanno su scale diverse e non vanno mai citati uno contro l'altro.",
      en: "y has been collapsed to five ordered levels, so most pairs are tied on y. Tau-b divides those tied pairs out of its denominator and Spearman does not — one of several reasons the two coefficients are on different scales and must never be quoted against each other.",
    },
    gen: (rng, nz) => Array.from({ length: N_CAMPIONE }, () => {
      const x = rng.normale();
      const grezzo = x + nz * 1.8 * rng.normale();
      return { x, y: Math.min(4, Math.max(0, Math.round(grezzo + 2))) };
    }),
  },
];

const STATO_SAND = { forma: "lineare", rumore: 0.30, vista: "valori", seme: 7 };

const elForme = document.getElementById("forme-chip");

FORME.forEach(f => {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "chip";
  b.textContent = I18N.t(f.etichetta);
  b.dataset.forma = f.id;
  b.setAttribute("aria-pressed", String(f.id === STATO_SAND.forma));
  b.addEventListener("click", () => {
    STATO_SAND.forma = f.id;
    sincronizzaChip();
    disegnaSandbox();
  });
  elForme.appendChild(b);
});

function sincronizzaChip() {
  [...elForme.children].forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.forma === STATO_SAND.forma)));
}

const elRumore = document.getElementById("rumore");
const elRumoreOut = document.getElementById("rumore-out");

elRumore.addEventListener("input", () => {
  STATO_SAND.rumore = elRumore.value / 100;
  elRumoreOut.textContent = fmtNum(STATO_SAND.rumore);
  disegnaSandbox();
});

const elVistaValori = document.getElementById("vista-valori");
const elVistaRanghi = document.getElementById("vista-ranghi");

function impostaVista(v) {
  STATO_SAND.vista = v;
  elVistaValori.setAttribute("aria-pressed", String(v === "valori"));
  elVistaRanghi.setAttribute("aria-pressed", String(v === "ranghi"));
  disegnaSandbox();
}

elVistaValori.addEventListener("click", () => impostaVista("valori"));
elVistaRanghi.addEventListener("click", () => impostaVista("ranghi"));

document.getElementById("ricampiona").addEventListener("click", () => {
  STATO_SAND.seme = (STATO_SAND.seme * 1664525 + 1013904223) >>> 0;
  disegnaSandbox();
});

function disegnaSandbox() {
  const forma = FORME.find(f => f.id === STATO_SAND.forma);
  const punti = forma.gen(creaRng(STATO_SAND.seme), STATO_SAND.rumore);
  const xs = punti.map(p => p.x), ys = punti.map(p => p.y);

  const r = pearson(xs, ys);
  const rho = spearman(xs, ys);
  const tau = kendallTauB(xs, ys);
  const dc = correlazioneDistanza(xs, ys);
  const mi = miNormalizzata(xs, ys, 7);

  /* --- grafico --- */
  const px = STATO_SAND.vista === "ranghi" ? ranghi(xs) : xs;
  const py = STATO_SAND.vista === "ranghi" ? ranghi(ys) : ys;

  const svg = document.getElementById("dispersione-punti");
  svg.textContent = "";

  const W = 420, H = 320, L = 34, R = 12, T = 12, B = 30;
  const xmin = Math.min(...px), xmax = Math.max(...px);
  const ymin = Math.min(...py), ymax = Math.max(...py);
  const padx = (xmax - xmin) * 0.06 || 1, pady = (ymax - ymin) * 0.06 || 1;
  const sx = v => L + (v - xmin + padx) / (xmax - xmin + 2 * padx) * (W - L - R);
  const sy = v => H - B - (v - ymin + pady) / (ymax - ymin + 2 * pady) * (H - T - B);

  for (let i = 0; i <= 4; i++) {
    const gy = T + i * (H - T - B) / 4;
    svg.appendChild(svgEl("line", { x1: L, y1: gy, x2: W - R, y2: gy, class: "gridline" }));
    const gx = L + i * (W - L - R) / 4;
    svg.appendChild(svgEl("line", { x1: gx, y1: T, x2: gx, y2: H - B, class: "gridline" }));
  }
  svg.appendChild(svgEl("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, class: "ax" }));
  svg.appendChild(svgEl("line", { x1: L, y1: T, x2: L, y2: H - B, class: "ax" }));

  punti.forEach((p, i) => {
    const cls = p.g === 1 ? (forma.id === "anomali" ? "pt pt--2" : "pt pt--3") : "pt";
    svg.appendChild(svgEl("circle", {
      cx: sx(px[i]).toFixed(1), cy: sy(py[i]).toFixed(1), r: 3.2, class: cls,
    }));
  });

  const etX = STATO_SAND.vista === "ranghi"
    ? I18N.t({ it: "rango di x  (1 … ", en: "rank of x  (1 … " }) + N_CAMPIONE + ")"
    : "x";
  const etY = STATO_SAND.vista === "ranghi"
    ? I18N.t({ it: "rango di y", en: "rank of y" })
    : "y";

  const tx = svgEl("text", { x: (L + W - R) / 2, y: H - 8, class: "axlab", "text-anchor": "middle" });
  tx.textContent = etX;
  svg.appendChild(tx);

  const ty = svgEl("text", {
    x: 11, y: (T + H - B) / 2, class: "axlab", "text-anchor": "middle",
    transform: "rotate(-90 11 " + ((T + H - B) / 2) + ")",
  });
  ty.textContent = etY;
  svg.appendChild(ty);

  /* --- legenda --- */
  const leg = document.getElementById("dispersione-legenda");
  leg.textContent = "";
  if (forma.legenda) {
    forma.legenda.forEach(([n, testo]) => {
      const s = document.createElement("span");
      const pallino = document.createElement("span");
      pallino.className = "swatch swatch--" + n;
      s.appendChild(pallino);
      s.appendChild(document.createTextNode(I18N.t(testo)));
      leg.appendChild(s);
    });
  }

  document.getElementById("dispersione-didascalia").textContent = I18N.t(forma.didascalia);

  /* --- numeri --- */
  const misure = [
    { k: "Pearson r", v: r, segno: true, nota: { it: "aderenza a una retta", en: "straight-line fit" } },
    { k: "Spearman ρ", v: rho, segno: true, nota: { it: "aderenza monotona", en: "monotonic fit" } },
    { k: "Kendall τb", v: tau, segno: true, nota: { it: "concordanza fra coppie", en: "pair concordance" } },
    { k: I18N.t({ it: "Corr. di distanza", en: "Distance corr" }), v: dc, segno: false,
      nota: { it: "soglia 0,17 · 0 solo se indipendenti", en: "baseline 0.17 · 0 only if independent" } },
    { k: I18N.t({ it: "Info mutua norm.", en: "Normalised MI" }), v: mi, segno: false,
      nota: { it: "soglia 0,07 · 7 bin", en: "baseline 0.07 · 7 bins" } },
  ];

  riempi(document.getElementById("letture"), misure.map(m => {
    const q = Math.min(1, Math.abs(m.v));
    /* Le misure con segno hanno lo zero al centro della barretta, quelle
       senza segno partono da sinistra: sono scale diverse e devono vedersi. */
    const barra = m.segno
      ? { da: m.v < 0 ? 50 - q * 50 : 50, largo: q * 50, negativa: m.v < 0 }
      : { da: 0, largo: q * 100, negativa: false };
    return tessera(m.k, fmtNum(m.v), I18N.t(m.nota), barra);
  }));

  document.getElementById("verdetto").innerHTML =
    verdetto(forma, punti, xs, ys, { r, rho, tau, dc, mi });
}

/* 95° percentile sotto vera indipendenza a n = 200, misurato per simulazione.
   Una lettura sotto queste soglie non è prova di niente. */
const SOGLIA_DCOR = 0.17, SOGLIA_MI = 0.07;

function verdetto(forma, punti, xs, ys, st) {
  const ar = Math.abs(st.r), arho = Math.abs(st.rho);

  /* Due forme portano l'etichetta di gruppo, quindi la diagnosi onesta si
     può calcolare davvero invece di indovinarla dai coefficienti. */
  if (forma.id === "simpson") {
    const g = k => {
      const s = punti.filter(p => p.g === k);
      return pearson(s.map(p => p.x), s.map(p => p.y));
    };
    return I18N.t({
      it: "<b>r complessivo = " + fmtNum(st.r) + ", ma dentro il gruppo A vale " + fmtNum(g(0)) +
          " e dentro il gruppo B " + fmtNum(g(1)) + ".</b> Ogni sottogruppo pende da una parte e il " +
          "coefficiente complessivo dall'altra. Descrive una popolazione che non esiste.",
      en: "<b>Pooled r = " + fmtNum(st.r) + ", but within group A it is " + fmtNum(g(0)) +
          " and within group B " + fmtNum(g(1)) + ".</b> Every subgroup slopes one way and the pooled " +
          "coefficient slopes the other. It describes a population that does not exist.",
    });
  }

  if (forma.id === "anomali") {
    const grosso = punti.filter(p => p.g === 0);
    const rb = pearson(grosso.map(p => p.x), grosso.map(p => p.y));
    return I18N.t({
      it: "<b>r = " + fmtNum(st.r) + " con i cinque punti a leva, " + fmtNum(rb) + " senza.</b> " +
          "Cinque righe su " + N_CAMPIONE + " sono tutta la relazione — ed è per questo che le misure " +
          "di rango, che un punto lontano non riesce a trascinare, stanno a " + fmtNum(st.rho) + ".",
      en: "<b>r = " + fmtNum(st.r) + " with the five leverage points, " + fmtNum(rb) + " without them.</b> " +
          "Five rows out of " + N_CAMPIONE + " are the entire relationship — which is why the rank measures, " +
          "which cannot be dragged by a distant point, sit at " + fmtNum(st.rho) + ".",
    });
  }

  if (forma.id === "eteroschedastica") {
    return I18N.t({
      it: "<b>La stima puntuale va bene; l'inferenza no.</b> Tutte e cinque le misure concordano sulla " +
          "forza, quindi r = " + fmtNum(st.r) + " riassume onestamente la pendenza — ma il rumore cresce " +
          "con x, e allora ogni errore standard, intervallo o p-value che assume varianza costante è " +
          "sbagliato. Serve un intervallo robusto all'eteroschedasticità, o bootstrap.",
      en: "<b>The point estimate is fine; the inference is not.</b> All five measures agree on the strength, " +
          "so r = " + fmtNum(st.r) + " is a fair summary of the slope — but the noise grows with x, so any " +
          "standard error, interval or p-value that assumes constant variance is wrong. Use a " +
          "heteroscedasticity-robust or bootstrap interval.",
    });
  }

  const quotaPari = 1 - new Set(ys).size / ys.length;

  if (ar < 0.18 && arho < 0.18 && (st.dc > SOGLIA_DCOR || st.mi > SOGLIA_MI)) {
    const viste = { it: [], en: [] };
    if (st.dc > SOGLIA_DCOR) { viste.it.push("La correlazione di distanza"); viste.en.push("Distance correlation"); }
    if (st.mi > SOGLIA_MI) {
      viste.it.push(viste.it.length ? "l'informazione mutua" : "L'informazione mutua");
      viste.en.push(viste.en.length ? "mutual information" : "Mutual information");
    }
    const doppio = viste.it.length > 1;
    return I18N.t({
      it: "<b>C'è una dipendenza che nessuna misura lineare o di rango riesce a vedere.</b> " +
          viste.it.join(" e ") + (doppio ? " superano entrambe la propria soglia" : " supera la sua soglia") +
          " di indipendenza mentre r, ρ e τ stanno a zero. La relazione c'è: solo, non è monotona.",
      en: "<b>A dependence is present that no linear or rank measure can see.</b> " +
          viste.en.join(" and ") + (doppio ? " both clear their" : " clears its") +
          " independence baseline while r, ρ and τ sit at nothing. The relation is real; it is just not monotonic.",
    });
  }

  if (ar > arho + 0.22) {
    return I18N.t({
      it: "<b>Pearson è molto più alto delle misure di rango.</b> Quel distacco è la firma di poche righe " +
          "estreme che fanno tutto il lavoro, invece di una relazione condivisa dal grosso dei dati. " +
          "Fanne un grafico prima di riportarlo.",
      en: "<b>Pearson is far above the rank measures.</b> That gap is the signature of a few extreme rows " +
          "doing the work, rather than a relationship the bulk of the data shares. Plot it before you report it.",
    });
  }

  if (arho > ar + 0.10) {
    return I18N.t({
      it: "<b>Monotona ma non lineare.</b> Le misure di rango trovano più struttura di Pearson, quindi la " +
          "retta è il riassunto sbagliato: trasforma la variabile, oppure riporta ρ.",
      en: "<b>Monotonic but not linear.</b> The rank measures find more structure than Pearson does, so a " +
          "straight line is the wrong summary — transform the variable, or report ρ instead.",
    });
  }

  if (quotaPari > 0.4 && arho - Math.abs(st.tau) > 0.08) {
    const distinti = new Set(ys).size;
    return I18N.t({
      it: "<b>y prende solo " + distinti + " valori distinti su " + N_CAMPIONE + " righe.</b> Con così tanti " +
          "pari merito il coefficiente difendibile è τb, che le coppie pari merito le esclude dal " +
          "denominatore, e la sua distanza da ρ qui è una differenza di scala, non un disaccordo.",
      en: "<b>y takes only " + distinti + " distinct values across " + N_CAMPIONE + " rows.</b> With ties this " +
          "heavy, τb — which excludes tied pairs from its denominator — is the defensible coefficient, and " +
          "its distance from ρ here is a scale difference, not a disagreement.",
    });
  }

  if (ar < 0.12 && st.dc <= SOGLIA_DCOR && st.mi <= SOGLIA_MI) {
    return I18N.t({
      it: "<b>Non c'è niente.</b> Tutte e cinque le misure stanno dentro la propria soglia di indipendenza, " +
          "che a questa numerosità è l'aspetto che ha l'indipendenza vera.",
      en: "<b>Nothing found.</b> All five measures sit inside their independence baselines, which at this " +
          "sample size is what genuine independence looks like.",
    });
  }

  return I18N.t({
    it: "<b>Le misure concordano.</b> Dipendenza lineare, monotona e generale puntano nella stessa " +
        "direzione e con forza simile, quindi qui Pearson è un riassunto adeguato.",
    en: "<b>The measures agree.</b> Linear, monotonic and general dependence all point the same way and to " +
        "a similar strength, so Pearson is an adequate summary here.",
  });
}


/* --------------------------------------------------------------------------
   2. LABORATORIO 2 — che cosa fa la collinearità a un coefficiente
   -------------------------------------------------------------------------- */

const STATO_COLL = { rho: 0.90, lam: 0, seme: 11 };
const N_SIM = 240, N_OSS = 60, N_TEST = 200;

function simula(rho, lamRelativo, seme) {
  const rng = creaRng(seme);
  const b1 = [], b2 = [], rmse = [];
  const lam = lamRelativo * N_OSS;

  for (let s = 0; s < N_SIM; s++) {
    let s11 = 0, s22 = 0, s12 = 0, s1y = 0, s2y = 0, m1 = 0, m2 = 0, my = 0;
    const X1 = new Float64Array(N_OSS), X2 = new Float64Array(N_OSS), Y = new Float64Array(N_OSS);

    for (let i = 0; i < N_OSS; i++) {
      const a = rng.normale();
      const x2 = rho * a + Math.sqrt(1 - rho * rho) * rng.normale();
      const y = a + x2 + rng.normale();
      X1[i] = a; X2[i] = x2; Y[i] = y;
      m1 += a; m2 += x2; my += y;
    }
    m1 /= N_OSS; m2 /= N_OSS; my /= N_OSS;

    for (let i = 0; i < N_OSS; i++) {
      const a = X1[i] - m1, b = X2[i] - m2, c = Y[i] - my;
      s11 += a * a; s22 += b * b; s12 += a * b; s1y += a * c; s2y += b * c;
    }

    const A = s11 + lam, D = s22 + lam;
    const det = A * D - s12 * s12;
    const e1 = det === 0 ? 0 : (D * s1y - s12 * s2y) / det;
    const e2 = det === 0 ? 0 : (A * s2y - s12 * s1y) / det;
    b1.push(e1); b2.push(e2);

    /* errore fuori campione per la stessa stima */
    let se = 0;
    const a0 = my - e1 * m1 - e2 * m2;
    for (let i = 0; i < N_TEST; i++) {
      const a = rng.normale();
      const x2 = rho * a + Math.sqrt(1 - rho * rho) * rng.normale();
      const y = a + x2 + rng.normale();
      const err = y - (a0 + e1 * a + e2 * x2);
      se += err * err;
    }
    rmse.push(Math.sqrt(se / N_TEST));
  }

  return { b1, b2, rmse };
}

function disegnaColl() {
  const { b1, b2, rmse } = simula(STATO_COLL.rho, STATO_COLL.lam, STATO_COLL.seme);
  const svg = document.getElementById("nuvola-punti");
  svg.textContent = "";

  const W = 420, H = 320, L = 40, R = 14, T = 14, B = 34;
  const lo = -1.6, hi = 3.6;
  const sx = v => L + (v - lo) / (hi - lo) * (W - L - R);
  const sy = v => H - B - (v - lo) / (hi - lo) * (H - T - B);

  for (let t = -1; t <= 3; t++) {
    svg.appendChild(svgEl("line", { x1: L, y1: sy(t), x2: W - R, y2: sy(t), class: "gridline" }));
    svg.appendChild(svgEl("line", { x1: sx(t), y1: T, x2: sx(t), y2: H - B, class: "gridline" }));
    const a = svgEl("text", { x: sx(t), y: H - B + 13, class: "axlab", "text-anchor": "middle" });
    a.textContent = fmtNum(t, 0);
    svg.appendChild(a);
    const c = svgEl("text", { x: L - 6, y: sy(t) + 3.5, class: "axlab", "text-anchor": "end" });
    c.textContent = fmtNum(t, 0);
    svg.appendChild(c);
  }
  svg.appendChild(svgEl("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, class: "ax" }));
  svg.appendChild(svgEl("line", { x1: L, y1: T, x2: L, y2: H - B, class: "ax" }));

  /* la direzione identificata: b1 + b2 = 2 */
  svg.appendChild(svgEl("line", {
    x1: sx(lo), y1: sy(2 - lo), x2: sx(hi), y2: sy(2 - hi), class: "markline",
  }));

  for (let i = 0; i < b1.length; i++) {
    const x = Math.max(lo, Math.min(hi, b1[i])), y = Math.max(lo, Math.min(hi, b2[i]));
    svg.appendChild(svgEl("circle", { cx: sx(x).toFixed(1), cy: sy(y).toFixed(1), r: 2.6, class: "pt" }));
  }
  svg.appendChild(svgEl("circle", { cx: sx(1), cy: sy(1), r: 5.5, class: "truept" }));

  const xl = svgEl("text", { x: (L + W - R) / 2, y: H - 6, class: "axlab", "text-anchor": "middle" });
  xl.textContent = I18N.t({ it: "coefficiente stimato su x1", en: "estimated coefficient on x1" });
  svg.appendChild(xl);

  const yl = svgEl("text", {
    x: 12, y: (T + H - B) / 2, class: "axlab", "text-anchor": "middle",
    transform: "rotate(-90 12 " + ((T + H - B) / 2) + ")",
  });
  yl.textContent = I18N.t({ it: "coefficiente stimato su x2", en: "estimated coefficient on x2" });
  svg.appendChild(yl);

  /* --- numeri --- */
  const somme = b1.map((v, i) => v + b2[i]);
  const vif = 1 / (1 - STATO_COLL.rho * STATO_COLL.rho);
  const kappa = Math.sqrt((1 + STATO_COLL.rho) / (1 - STATO_COLL.rho));
  const sd1 = devStd(b1), sdSomma = devStd(somme);
  const rmseMediano = [...rmse].sort((a, b) => a - b)[Math.floor(rmse.length / 2)];

  riempi(document.getElementById("coll-letture"), [
    tessera("VIF", fmtNum(vif, 1), "1 / (1 − r²)"),
    tessera(I18N.t({ it: "Indice di condizione", en: "Condition index" }), fmtNum(kappa, 1), "√(λmax / λmin)"),
    tessera(I18N.t({ it: "DS del coef. su x1", en: "SD of coef on x1" }), fmtNum(sd1),
      I18N.t({ it: "su 240 ristime", en: "across 240 refits" })),
    tessera(I18N.t({ it: "DS della somma", en: "SD of the sum" }), fmtNum(sdSomma), "coef(x1) + coef(x2)"),
    tessera(I18N.t({ it: "RMSE fuori campione", en: "Test RMSE" }), fmtNum(rmseMediano),
      I18N.t({ it: "mediano; il minimo è 1,00", en: "median; floor is 1.00" })),
  ]);

  const rapporto = sdSomma > 0 ? sd1 / sdSomma : 1;
  let messaggio;

  if (STATO_COLL.lam > 0) {
    messaggio = I18N.t({
      it: "<b>La ridge sta tenendo insieme la nuvola.</b> Le stime sono tirate verso lo zero — quindi " +
          "distorte — ma la loro dispersione è crollata, e l'errore di previsione non si è quasi mosso. " +
          "Quello scambio è tutto l'argomento a favore della penalizzazione.",
      en: "<b>Ridge is holding the cloud together.</b> The estimates are pulled toward zero — biased — but " +
          "their spread has collapsed, and the prediction error has barely moved. That trade is the entire " +
          "argument for the penalty.",
    });
  } else if (rapporto > 3) {
    messaggio = I18N.t({
      it: "<b>La ripartizione non è identificata; il totale sì.</b> Il singolo coefficiente è " +
          fmtNum(rapporto, 1) + " volte più variabile della somma dei due. Riporta la somma, costruisci una " +
          "feature combinata, oppure aggiungi una penalizzazione — ma non citare nessuno dei due da solo.",
      en: "<b>The split is unidentified; the total is not.</b> The individual coefficient is " +
          fmtNum(rapporto, 1) + "× more variable than the sum of the two. Report the sum, engineer a combined " +
          "feature, or add a penalty — but do not quote either coefficient on its own.",
    });
  } else if (STATO_COLL.rho > 0.5) {
    messaggio = I18N.t({
      it: "<b>Inflazione lieve.</b> La nuvola ha cominciato a stirarsi lungo la retta a somma costante, e gli " +
          "errori standard sono cresciuti di un fattore " + fmtNum(Math.sqrt(vif), 1) + ". Ancora usabile, " +
          "ma da segnalare nel resoconto.",
      en: "<b>Mild inflation.</b> The cloud has begun to stretch along the constant-sum line, and standard " +
          "errors are up by a factor of " + fmtNum(Math.sqrt(vif), 1) + ". Still usable, worth noting in the " +
          "write-up.",
    });
  } else {
    messaggio = I18N.t({
      it: "<b>Ben condizionato.</b> Le stime si sparpagliano in tondo attorno ai valori veri e ogni " +
          "coefficiente vuol dire quello che dice. L'errore fuori campione qui è il minimo teorico, e a ogni " +
          "posizione di questo cursore resta lì.",
      en: "<b>Well conditioned.</b> The estimates scatter roundly around the true values and each coefficient " +
          "means what it says. Test error here is the noise floor, and it stays there at every setting of this " +
          "slider.",
    });
  }

  document.getElementById("coll-verdetto").innerHTML = messaggio;
}

const elRho = document.getElementById("rho"), elRhoOut = document.getElementById("rho-out");
elRho.addEventListener("input", () => {
  STATO_COLL.rho = elRho.value / 100;
  elRhoOut.textContent = fmtNum(STATO_COLL.rho);
  disegnaColl();
});

const elLam = document.getElementById("lambda"), elLamOut = document.getElementById("lambda-out");
elLam.addEventListener("input", () => {
  STATO_COLL.lam = elLam.value / 100;
  elLamOut.textContent = fmtNum(STATO_COLL.lam);
  disegnaColl();
});

document.getElementById("risimula").addEventListener("click", () => {
  STATO_COLL.seme = (STATO_COLL.seme * 1664525 + 1013904223) >>> 0;
  disegnaColl();
});


/* --------------------------------------------------------------------------
   3. LABORATORIO 3 — potatura di una matrice di correlazione
   -------------------------------------------------------------------------- */

/* Pesi su tre fattori latenti (dimensione, posizione, età) seguiti da una
   dimensione unica per feature. Costruire la matrice così rende la struttura
   a blocchi vera e non decorativa, e permette a due coppie di essere
   duplicati esatti per costruzione.

   I nomi delle feature non si traducono: sono nomi di colonne, come li si
   troverebbe in un dataset. */
const FEATURE = [
  { nome: "area_sqm",       f: [.95, .10, .00],  u: 0, uw: .25 },
  { nome: "area_sqft",      f: [.95, .10, .00],  u: 0, uw: .25 },   /* identica: conversione di unità */
  { nome: "rooms",          f: [.88, .05, -.10], u: 1, uw: .28 },
  { nome: "bedrooms",       f: [.80, .02, -.05], u: 2, uw: .40 },
  { nome: "bathrooms",      f: [.72, .22, .10],  u: 3, uw: .62 },
  { nome: "garage_spaces",  f: [.55, .25, .12],  u: 4, uw: .78 },
  { nome: "year_built",     f: [.10, .15, .92],  u: 5, uw: .40 },
  { nome: "age_years",      f: [-.10, -.15, -.92], u: 5, uw: -.40 }, /* gemella negativa esatta */
  { nome: "dist_center_km", f: [-.05, -.90, .20], u: 6, uw: .35 },
  { nome: "zone_income",    f: [.20, .85, .05],  u: 7, uw: .38 },
  { nome: "zone_crime",     f: [-.05, -.74, -.10], u: 8, uw: .60 },
];

function costruisciMatrice() {
  const dim = 3 + 9;
  const vet = FEATURE.map(ft => {
    const v = new Array(dim).fill(0);
    for (let k = 0; k < 3; k++) v[k] = ft.f[k];
    v[3 + ft.u] = ft.uw;
    return v;
  });
  const prod = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const n = FEATURE.length;
  const M = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      M[i][j] = prod(vet[i], vet[j]) / Math.sqrt(prod(vet[i], vet[i]) * prod(vet[j], vet[j]));
  return M;
}

const MATRICE_R = costruisciMatrice();
const NOMI_FEATURE = FEATURE.map(f => f.nome);

function mediaAssolutaVerso(i, tenute) {
  let s = 0, c = 0;
  for (let j = 0; j < MATRICE_R.length; j++)
    if (j !== i && tenute[j]) { s += Math.abs(MATRICE_R[i][j]); c++; }
  return c ? s / c : 0;
}

/* caret::findCorrelation — della coppia peggiore si scarta quella che, in
   media, è più correlata con tutto il resto che è ancora in piedi. */
function pota(soglia) {
  const tenute = new Array(MATRICE_R.length).fill(true);
  for (;;) {
    let bi = -1, bj = -1, peggiore = 0;
    for (let i = 0; i < MATRICE_R.length; i++) {
      if (!tenute[i]) continue;
      for (let j = i + 1; j < MATRICE_R.length; j++) {
        if (!tenute[j]) continue;
        const a = Math.abs(MATRICE_R[i][j]);
        if (a > peggiore) { peggiore = a; bi = i; bj = j; }
      }
    }
    if (bi < 0 || peggiore < soglia) break;
    tenute[mediaAssolutaVerso(bi, tenute) >= mediaAssolutaVerso(bj, tenute) ? bi : bj] = false;
  }
  return tenute;
}

/* Componenti connesse della matrice sopra soglia: i gruppi di ridondanza. */
function componentiA(soglia) {
  const n = MATRICE_R.length, visti = new Array(n).fill(false), fuori = [];
  for (let i = 0; i < n; i++) {
    if (visti[i]) continue;
    const pila = [i], comp = [];
    visti[i] = true;
    while (pila.length) {
      const k = pila.pop();
      comp.push(k);
      for (let j = 0; j < n; j++)
        if (!visti[j] && Math.abs(MATRICE_R[k][j]) >= soglia) { visti[j] = true; pila.push(j); }
    }
    if (comp.length > 1) fuori.push(comp.sort((a, b) => a - b));
  }
  return fuori;
}

const elMatrice = document.getElementById("matrice");

function costruisciHeat() {
  const n = NOMI_FEATURE.length;
  elMatrice.style.gridTemplateColumns = "9.5rem repeat(" + n + ", 38px)";
  const frag = document.createDocumentFragment();

  const intestazione = document.createElement("div");
  intestazione.setAttribute("role", "row");
  intestazione.style.display = "contents";
  const angolo = document.createElement("div");
  angolo.setAttribute("role", "columnheader");
  intestazione.appendChild(angolo);

  NOMI_FEATURE.forEach((nome, j) => {
    const d = document.createElement("div");
    d.className = "heat__colhead";
    d.setAttribute("role", "columnheader");
    d.dataset.colonna = j;
    d.textContent = nome;
    intestazione.appendChild(d);
  });
  frag.appendChild(intestazione);

  NOMI_FEATURE.forEach((nome, i) => {
    const riga = document.createElement("div");
    riga.setAttribute("role", "row");
    riga.style.display = "contents";

    const testa = document.createElement("div");
    testa.className = "heat__rowhead";
    testa.setAttribute("role", "rowheader");
    testa.dataset.riga = i;
    testa.textContent = nome;
    riga.appendChild(testa);

    NOMI_FEATURE.forEach((_, j) => {
      const cella = document.createElement("div");
      cella.className = "heat__cell";
      cella.setAttribute("role", "cell");
      cella.dataset.i = i;
      cella.dataset.j = j;

      const v = MATRICE_R[i][j];
      const q = Math.min(1, Math.abs(v));
      /* La tinta è un token, la quantità di tinta un numero calcolato:
         tutti e due entrano nel CSS come proprietà, mai come colore scritto
         qui dentro. Fermata al 55% perché il numero stampato resti
         leggibile in tutti e due i temi. */
      cella.style.setProperty("--tinta", v >= 0 ? "var(--heat-pos)" : "var(--heat-neg)");
      cella.style.setProperty("--mescola", (q * 55).toFixed(1) + "%");

      /* Il numero si scrive per intero, zero iniziale compreso. La forma
         compatta ".92" funziona in inglese ma non in italiano, dove
         diventerebbe ",92": lo zero costa due pixel e li vale. */
      const segno = v < 0 ? "−" : "";
      cella.textContent = i === j ? "1"
        : q >= 0.995 ? segno + "1"
        : segno + fmtNum(q);

      const titolo = NOMI_FEATURE[i] + " × " + NOMI_FEATURE[j] + " = " + fmtNum(v, 3);
      cella.title = titolo;
      cella.setAttribute("aria-label", titolo);
      riga.appendChild(cella);
    });

    frag.appendChild(riga);
  });

  elMatrice.appendChild(frag);
}

const elSoglia = document.getElementById("soglia");
const elSogliaOut = document.getElementById("soglia-out");

function disegnaPotatura() {
  const soglia = elSoglia.value / 100;
  elSogliaOut.textContent = fmtNum(soglia);
  const tenute = pota(soglia);

  elMatrice.querySelectorAll(".heat__cell").forEach(c => {
    const i = +c.dataset.i, j = +c.dataset.j;
    c.classList.toggle("is-dropped", !tenute[i] || !tenute[j]);
  });
  elMatrice.querySelectorAll(".heat__rowhead")
    .forEach(h => h.classList.toggle("is-dropped", !tenute[+h.dataset.riga]));
  elMatrice.querySelectorAll(".heat__colhead")
    .forEach(h => h.classList.toggle("is-dropped", !tenute[+h.dataset.colonna]));

  let massimoRimasto = 0;
  for (let i = 0; i < MATRICE_R.length; i++)
    for (let j = i + 1; j < MATRICE_R.length; j++)
      if (tenute[i] && tenute[j]) massimoRimasto = Math.max(massimoRimasto, Math.abs(MATRICE_R[i][j]));

  const quante = tenute.filter(Boolean).length;
  const gruppi = componentiA(soglia);

  riempi(document.getElementById("pota-letture"), [
    tessera(I18N.t({ it: "Feature tenute", en: "Features kept" }),
      quante + " / " + NOMI_FEATURE.length,
      I18N.t({ it: "dopo la potatura greedy", en: "after greedy pruning" })),
    tessera(I18N.t({ it: "Scartate", en: "Dropped" }),
      String(NOMI_FEATURE.length - quante),
      I18N.t({ it: "ridondanti a questa soglia", en: "redundant at this threshold" })),
    tessera(I18N.t({ it: "Gruppi di ridondanza", en: "Redundancy groups" }),
      String(gruppi.length),
      I18N.t({ it: "connessi a |r| ≥ ", en: "connected at |r| ≥ " }) + fmtNum(soglia)),
    tessera(I18N.t({ it: "Massimo |r| rimasto", en: "Max |r| remaining" }),
      fmtNum(massimoRimasto),
      I18N.t({ it: "fra i sopravvissuti", en: "among the survivors" })),
  ]);

  disegnaGruppi(gruppi, tenute);
  disegnaLista(tenute);
}

function etichettaFeature(indice, tenuta) {
  const e = document.createElement("span");
  e.className = "tag " + (tenuta ? "tag--keep" : "tag--drop");
  e.textContent = NOMI_FEATURE[indice];
  return e;
}

function disegnaGruppi(gruppi, tenute) {
  const box = document.getElementById("gruppi");
  box.textContent = "";

  if (!gruppi.length) {
    const p = document.createElement("p");
    p.className = "figcap";
    p.textContent = I18N.t({
      it: "Nessuna coppia arriva a questa soglia: ogni feature sta per conto suo.",
      en: "No pair reaches this threshold — every feature stands alone.",
    });
    box.appendChild(p);
    return;
  }

  gruppi.forEach((comp, i) => {
    const riga = document.createElement("div");
    riga.className = "groups__row";

    const nome = document.createElement("span");
    nome.className = "groups__name";
    nome.textContent = I18N.t({ it: "gruppo ", en: "group " }) + (i + 1);
    riga.appendChild(nome);

    const lista = document.createElement("span");
    lista.className = "tags";
    comp.forEach(k => lista.appendChild(etichettaFeature(k, tenute[k])));
    riga.appendChild(lista);

    box.appendChild(riga);
  });
}

function disegnaLista(tenute) {
  const lista = document.getElementById("pota-lista");
  lista.textContent = "";
  NOMI_FEATURE.forEach((_, i) => lista.appendChild(etichettaFeature(i, tenute[i])));
}

elSoglia.addEventListener("input", disegnaPotatura);


/* --------------------------------------------------------------------------
   4. AVVIO
   -------------------------------------------------------------------------- */

elRumoreOut.textContent = fmtNum(STATO_SAND.rumore);
elRhoOut.textContent = fmtNum(STATO_COLL.rho);
elLamOut.textContent = fmtNum(STATO_COLL.lam);

costruisciHeat();
disegnaSandbox();
disegnaColl();
disegnaPotatura();
