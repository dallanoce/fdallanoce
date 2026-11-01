/* ==========================================================================
   INSIDE NANOGPT — il GPT di Karpathy, riga per riga

   Struttura del file:
     0. PARAMETRI      — la configurazione del modellino e dell'addestramento
     1. NUMERICA       — generatore con seme, tensori, funzioni elementari
     2. MODELLO        — pesi iniziali e passo in avanti (model.py)
     3. GRADIENTI      — il passo all'indietro, scritto a mano
     4. ADDESTRAMENTO  — get_batch, get_lr, AdamW, clip (train.py)
     5. GENERAZIONE    — generate() (model.py)
     6. INTERFACCIA    — un laboratorio per ogni sezione della pagina
     7. AVVIO

   I NOMI SONO QUELLI DI NANOGPT
   Le sezioni 0–5 usano gli stessi identificatori del codice Python, in
   inglese e con i trattini bassi: idx, tok_emb, pos_emb, c_attn, att, q, k,
   v, ln_1, lm_head, get_lr, warmup_iters, idx_cond… È un'eccezione voluta
   all'abitudine del sito di scrivere i nomi in italiano: la pagina serve a
   leggere model.py, e chi apre questo file deve poterci ritrovare le stesse
   parole. Il codice di contorno (sezione 6) torna all'italiano.

   Il riferimento è nanoGPT al commit 3adf61e (licenza MIT):
   https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291

   UN MODELLO VERO, MOLTO PICCOLO
   Qui gira un GPT completo — embedding, due Block con attenzione causale a
   due teste, MLP, LayerNorm, lm_head legata a wte — con dimensioni scelte
   perché ogni matrice si possa stampare con i suoi numeri. All'apertura i
   pesi sono quelli dell'inizializzazione di nanoGPT; il laboratorio di
   addestramento li allena nel browser, con un passo all'indietro scritto a
   mano (sezione 3) al posto dell'autograd di PyTorch.

   DUE LINGUE, UN FILE SOLO
   Questo file serve sia /it/strumenti/nanogpt/ sia /en/tools/nanogpt/. Le
   due pagine hanno gli stessi id; il testo che stampa il codice passa da
   I18N.t({ it, en }).

   NIENTE COLORI NEL CODICE
   Come in correlazione.js: il JavaScript assegna solo classi e passa i
   numeri calcolati alle proprietà --tinta, --mescola, --quota. I colori
   stanno nei token di style.css, e i grafici cambiano tema con la pagina.

   Le sezioni 0–5 non toccano la pagina: sono funzioni pure che si possono
   provare anche fuori dal browser.
   ========================================================================== */


/* --------------------------------------------------------------------------
   0. PARAMETRI
   -------------------------------------------------------------------------- */

/* Il GPTConfig del modellino della pagina. Stessi campi della dataclass di
   model.py; i valori di nanoGPT per Shakespeare sono fra parentesi.
   n_embd = 16 e n_head = 2 danno teste da hs = 8: q, k, v e att di una
   testa sono matrici 8 × 8, abbastanza piccole da stamparle intere. */
const config = {
  block_size: 8,      // (256) quanti caratteri di contesto vede il modello
  vocab_size: 65,     // (65)  i caratteri distinti di Tiny Shakespeare
  n_layer: 2,         // (6)   quanti Block in fila
  n_head: 2,          // (6)   teste di attenzione per Block
  n_embd: 16,         // (384) la larghezza C del flusso residuo
  dropout: 0.0,       // (0.2) spento: la pagina mostra il modello in valutazione
  bias: true,         // (False) i bias ci sono, come nel GPTConfig di default
};

/* Il vocabolario: i 65 caratteri distinti dell'intero Tiny Shakespeare, in
   ordine, esattamente come li produce data/shakespeare_char/prepare.py con
   sorted(list(set(data))). Scritto per intero e non ricavato dall'estratto
   che si scarica per l'addestramento, così gli id coincidono con quelli di
   nanoGPT anche per i caratteri che nell'estratto non compaiono. */
const chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".split("");

const stoi = {};
const itos = {};
chars.forEach((ch, i) => { stoi[ch] = i; itos[i] = ch; });

/* encoder: da stringa a lista di interi; decoder: il contrario.
   Un carattere fuori vocabolario in Python darebbe KeyError; qui diventa
   undefined, e sta all'interfaccia segnalarlo. */
function encode(s) { return [...s].map(c => stoi[c]); }
function decode(l) { return l.map(i => itos[i]).join(""); }

/* Gli iperparametri di train.py, ridotti in scala con il modello. Fra
   parentesi i valori di config/train_shakespeare_char.py. Il learning rate
   è più alto perché il modello è minuscolo, e i passi sono pochi perché
   l'addestramento deve finire mentre il lettore aspetta. */
const train_config = {
  batch_size: 16,              // (64)
  learning_rate: 1e-2,         // (1e-3)
  max_iters: 1500,             // (5000)
  lr_decay_iters: 1500,        // (5000) di solito uguale a max_iters
  min_lr: 1e-3,                // (1e-4) di solito learning_rate / 10
  warmup_iters: 100,           // (100)
  beta1: 0.9,                  // (0.9)
  beta2: 0.99,                 // (0.99)
  weight_decay: 1e-1,          // (1e-1)
  grad_clip: 1.0,              // (1.0)
  decay_lr: true,              // (True)
  eval_interval: 100,          // (250)
  eval_iters: 10,              // (200)
  gradient_accumulation_steps: 1,
};

/* Le configurazioni che la mappa sa confrontare con quella della pagina.
   shakespeare_char ha bias=False perché è il default di train.py; gpt2 è
   quella che impone from_pretrained('gpt2'). */
const PRESET = {
  pagina: config,
  shakespeare_char: { block_size: 256, vocab_size: 65, n_layer: 6, n_head: 6, n_embd: 384, dropout: 0.2, bias: false },
  gpt2: { block_size: 1024, vocab_size: 50257, n_layer: 12, n_head: 12, n_embd: 768, dropout: 0.0, bias: true },
};

/* Seme dei pesi iniziali: stesso seme, stesso modello a ogni visita. */
const SEME_PESI = 1337;


/* --------------------------------------------------------------------------
   1. NUMERICA
   -------------------------------------------------------------------------- */

/* Generatore pseudocasuale con seme, lo stesso di correlazione.js. Serve
   perché "step 0" deve essere lo stesso modello per tutti, e perché un
   addestramento rifatto con gli stessi parametri deve dare la stessa curva. */
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

/* Un tensore qui è un oggetto { shape, data }: la forma come in PyTorch e i
   numeri in fila, riga per riga, in un Float64Array. Le attivazioni sono
   sempre matrici: le B sequenze del batch stanno una sotto l'altra, quindi
   un tensore (B, T, C) di PyTorch qui è una matrice (B·T) × C. I parametri
   hanno anche grad, che il passo all'indietro riempie. */
function tensor(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return { shape: shape.slice(), data: new Float64Array(n) };
}

function parametro(shape) {
  const p = tensor(shape);
  p.grad = new Float64Array(p.data.length);
  return p;
}

/* Precisione doppia anche sulla funzione d'errore: Math non ha erf, e la
   formula approssimata più diffusa (Abramowitz–Stegun) sbaglia alla
   settima cifra, abbastanza da falsare il controllo dei gradienti.
   Serie di Taylor vicino allo zero, frazione continua di erfc più fuori,
   e ±1 oltre 6, dove erfc è già sotto la precisione di un double. */
function erf(x) {
  const a = Math.abs(x);
  let r;
  if (a < 2.5) {
    // 2/√π · Σ (−1)ⁿ x^(2n+1) / (n! (2n+1))
    const x2 = a * a;
    let termine = a, somma = a, n = 0;
    while (Math.abs(termine) > 1e-17 * somma) {
      n++;
      termine *= -x2 / n;
      somma += termine / (2 * n + 1);
    }
    r = somma * 2 / Math.sqrt(Math.PI);
  } else if (a < 6) {
    // erfc(a) = e^(−a²)/√π · 1/(a + ½/(a + 1/(a + 3⁄2/(a + …))))
    let f = a;
    for (let k = 60; k >= 1; k--) f = a + (k / 2) / f;
    r = 1 - Math.exp(-a * a) / (Math.sqrt(Math.PI) * f);
  } else {
    r = 1;
  }
  return x < 0 ? -r : r;
}

/* nn.GELU() di PyTorch, nella forma esatta (approximate='none'):
   GELU(x) = x · Φ(x) = ½ x (1 + erf(x / √2)) */
function gelu_scalare(x) {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/* La derivata: Φ(x) + x · φ(x), con φ la densità normale. */
function gelu_derivata(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2)) + x * Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/* La GELU approssimata con tanh del GPT-2 originale, solo per il grafico
   di confronto: nanoGPT usa quella esatta. */
function gelu_tanh(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

/* Softmax di un vettore, con il massimo sottratto prima dell'esponenziale:
   il risultato non cambia, ma e^1000 non diventa Infinity. Gli elementi a
   −Infinity escono a zero, che è quello che fa la maschera causale. */
function softmax(v) {
  let max = -Infinity;
  for (const z of v) if (z > max) max = z;
  const out = new Float64Array(v.length);
  let somma = 0;
  for (let i = 0; i < v.length; i++) {
    out[i] = v[i] === -Infinity ? 0 : Math.exp(v[i] - max);
    somma += out[i];
  }
  for (let i = 0; i < v.length; i++) out[i] /= somma;
  return out;
}


/* --------------------------------------------------------------------------
   2. MODELLO — model.py
   -------------------------------------------------------------------------- */

/* nn.Linear: y = x Wᵀ + b, con il peso salvato (out_features, in_features)
   come in PyTorch. È per questo che c_attn.weight ha forma (3C, C) anche se
   "trasforma C in 3C": la trasposta è dentro la formula. */
function linear(x, layer) {
  const [N, inF] = x.shape;
  const outF = layer.weight.shape[0];
  const W = layer.weight.data, b = layer.bias ? layer.bias.data : null;
  const y = tensor([N, outF]);
  for (let n = 0; n < N; n++) {
    const xo = n * inF;
    for (let o = 0; o < outF; o++) {
      let s = b ? b[o] : 0;
      const wo = o * inF;
      for (let i = 0; i < inF; i++) s += x.data[xo + i] * W[wo + i];
      y.data[n * outF + o] = s;
    }
  }
  return y;
}

/* LayerNorm di model.py, cioè F.layer_norm con eps = 1e-5:
   y = (x − μ) / √(σ² + ε) · weight + bias, riga per riga.
   La varianza è quella "di popolazione", divisa per C e non per C − 1. */
function layer_norm(x, ln) {
  const [N, C] = x.shape;
  const out = tensor([N, C]);
  const mean = new Float64Array(N), rstd = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    const o = n * C;
    let m = 0;
    for (let c = 0; c < C; c++) m += x.data[o + c];
    m /= C;
    let v = 0;
    for (let c = 0; c < C; c++) { const d = x.data[o + c] - m; v += d * d; }
    v /= C;
    const r = 1 / Math.sqrt(v + 1e-5);
    mean[n] = m; rstd[n] = r;
    for (let c = 0; c < C; c++) {
      out.data[o + c] = (x.data[o + c] - m) * r * ln.weight.data[c] + (ln.bias ? ln.bias.data[c] : 0);
    }
  }
  return { out, mean, rstd };
}

function somma(a, b) {
  const y = tensor(a.shape);
  for (let i = 0; i < y.data.length; i++) y.data[i] = a.data[i] + b.data[i];
  return y;
}

/* CausalSelfAttention.forward, ramo "manual implementation" (quello che
   PyTorch < 2.0 esegue davvero, e che flash attention calcola uguale).

   x arriva come (B·T) × C. Dove PyTorch fa view + transpose per ottenere
   (B, nh, T, hs), qui per ogni sequenza b e ogni testa h si ritagliano le
   colonne della testa: q[b][h] è la matrice T × hs di quella testa. */
function causal_self_attention(x, attn, B, T) {
  const C = config.n_embd;
  const n_head = config.n_head;
  const hs = C / n_head;

  // q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
  const qkv = linear(x, attn.c_attn);                     // (B·T) × 3C

  const q = [], k = [], v = [], att_raw = [], att = [], y_h = [];
  const y = tensor([B * T, C]);
  const scala = 1.0 / Math.sqrt(hs);                      // 1.0 / math.sqrt(k.size(-1))

  for (let b = 0; b < B; b++) {
    q.push([]); k.push([]); v.push([]); att_raw.push([]); att.push([]); y_h.push([]);
    for (let h = 0; h < n_head; h++) {
      // .view(B, T, n_head, hs).transpose(1, 2): le colonne della testa h
      const qh = tensor([T, hs]), kh = tensor([T, hs]), vh = tensor([T, hs]);
      for (let t = 0; t < T; t++) {
        const riga = (b * T + t) * 3 * C;
        for (let d = 0; d < hs; d++) {
          qh.data[t * hs + d] = qkv.data[riga + h * hs + d];
          kh.data[t * hs + d] = qkv.data[riga + C + h * hs + d];
          vh.data[t * hs + d] = qkv.data[riga + 2 * C + h * hs + d];
        }
      }

      // att = (q @ k.transpose(-2, -1)) — prima della scala, per la pagina
      const raw = tensor([T, T]);
      for (let i = 0; i < T; i++)
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let d = 0; d < hs; d++) s += qh.data[i * hs + d] * kh.data[j * hs + d];
          raw.data[i * T + j] = s;
        }

      // * scala, masked_fill(bias == 0, -inf), softmax(dim=-1)
      const a = tensor([T, T]);
      for (let i = 0; i < T; i++) {
        const riga = new Float64Array(T);
        for (let j = 0; j < T; j++) riga[j] = j <= i ? raw.data[i * T + j] * scala : -Infinity;
        a.data.set(softmax(riga), i * T);
      }

      // y = att @ v
      const yh = tensor([T, hs]);
      for (let i = 0; i < T; i++)
        for (let d = 0; d < hs; d++) {
          let s = 0;
          for (let j = 0; j <= i; j++) s += a.data[i * T + j] * vh.data[j * hs + d];
          yh.data[i * hs + d] = s;
        }

      // y.transpose(1, 2).contiguous().view(B, T, C): le teste fianco a fianco
      for (let t = 0; t < T; t++)
        for (let d = 0; d < hs; d++) y.data[(b * T + t) * C + h * hs + d] = yh.data[t * hs + d];

      q[b].push(qh); k[b].push(kh); v[b].push(vh);
      att_raw[b].push(raw); att[b].push(a); y_h[b].push(yh);
    }
  }

  // y = self.resid_dropout(self.c_proj(y)) — dropout spento
  const out = linear(y, attn.c_proj);
  return { qkv, q, k, v, att_raw, att, y_h, y, out, scala };
}

/* MLP.forward: c_fc allarga a 4C, GELU, c_proj torna a C.
   Φ(x) si tiene da parte: è metà della derivata, e ricalcolarla
   all'indietro vorrebbe dire rifare tutte le erf. */
function mlp_forward(x, mlp) {
  const h_fc = linear(x, mlp.c_fc);
  const h_gelu = tensor(h_fc.shape);
  const phi = new Float64Array(h_fc.data.length);
  for (let i = 0; i < h_fc.data.length; i++) {
    phi[i] = 0.5 * (1 + erf(h_fc.data[i] / Math.SQRT2));
    h_gelu.data[i] = h_fc.data[i] * phi[i];
  }
  const out = linear(h_gelu, mlp.c_proj);
  return { h_fc, h_gelu, phi, out };
}

/* F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=-1):
   la media, sulle posizioni con un target, di −log della probabilità che
   il modello dà al carattere giusto. */
function cross_entropy(logits, targets) {
  const [N, V] = logits.shape;
  const probs = tensor([N, V]);
  let totale = 0, contate = 0;
  for (let n = 0; n < N; n++) {
    const p = softmax(logits.data.subarray(n * V, (n + 1) * V));
    probs.data.set(p, n * V);
    if (targets[n] === -1) continue;
    totale += -Math.log(p[targets[n]]);
    contate++;
  }
  return { loss: totale / contate, probs, contate };
}

/* GPT.__init__ + _init_weights. Il modello è un albero di oggetti con le
   stesse chiavi dei moduli di PyTorch e nello stesso ordine, così
   named_parameters() restituisce gli stessi nomi di model.named_parameters().

   Inizializzazione come in nanoGPT: pesi di Linear ed Embedding da una
   normale con deviazione 0,02, bias a zero, LayerNorm a pesi 1 e bias 0, e
   i c_proj — gli strati che scrivono nel flusso residuo — con deviazione
   0,02 / √(2 · n_layer). lm_head.weight È transformer.wte.weight: lo stesso
   oggetto, non una copia (weight tying). */
function GPT(cfg, seme) {
  const rng = creaRng(seme);

  const casuale = (shape, std) => {
    const p = parametro(shape);
    for (let i = 0; i < p.data.length; i++) p.data[i] = rng.normale() * std;
    return p;
  };
  const costante = (shape, valore) => {
    const p = parametro(shape);
    p.data.fill(valore);
    return p;
  };

  const Linear = (in_f, out_f, bias, std = 0.02) => ({
    weight: casuale([out_f, in_f], std),
    bias: bias ? costante([out_f], 0) : null,
  });
  const Embedding = (num, dim) => ({ weight: casuale([num, dim], 0.02) });
  const LayerNorm = (ndim, bias) => ({
    weight: costante([ndim], 1),
    bias: bias ? costante([ndim], 0) : null,
  });

  const C = cfg.n_embd;
  const std_proj = 0.02 / Math.sqrt(2 * cfg.n_layer);

  const h = [];
  for (let l = 0; l < cfg.n_layer; l++) {
    h.push({
      ln_1: LayerNorm(C, cfg.bias),
      attn: {
        c_attn: Linear(C, 3 * C, cfg.bias),
        c_proj: Linear(C, C, cfg.bias, std_proj),
      },
      ln_2: LayerNorm(C, cfg.bias),
      mlp: {
        c_fc: Linear(C, 4 * C, cfg.bias),
        c_proj: Linear(4 * C, C, cfg.bias, std_proj),
      },
    });
  }

  const transformer = {
    wte: Embedding(cfg.vocab_size, C),
    wpe: Embedding(cfg.block_size, C),
    h,
    ln_f: LayerNorm(C, cfg.bias),
  };

  return {
    transformer,
    lm_head: { weight: transformer.wte.weight, bias: null },  // weight tying
  };
}

/* model.named_parameters(): i parametri con il loro nome completo. Un
   parametro condiviso compare una volta sola, con il primo nome che si
   incontra — come in PyTorch, dove lm_head.weight non viene ripetuto
   perché è lo stesso tensore di transformer.wte.weight. */
function named_parameters(model) {
  const lista = [];
  const visti = new Set();
  (function visita(prefisso, nodo) {
    for (const chiave of Object.keys(nodo)) {
      const valore = nodo[chiave];
      if (valore === null) continue;
      const nome = prefisso ? prefisso + "." + chiave : chiave;
      if (valore.data instanceof Float64Array) {
        if (!visti.has(valore)) { visti.add(valore); lista.push([nome, valore]); }
      } else if (Array.isArray(valore)) {
        valore.forEach((m, i) => visita(nome + "." + i, m));
      } else {
        visita(nome, valore);
      }
    }
  })("", model);
  return lista;
}

/* GPT.forward(idx, targets). idx è una lista di B sequenze di t interi.
   Restituisce logits e loss come nanoGPT, più cache: tutti i tensori
   intermedi con il loro nome, che il passo all'indietro e la pagina
   leggono senza ricalcolare niente. */
function forward(model, idx, targets) {
  const B = idx.length, t = idx[0].length;
  if (t > config.block_size) throw new Error("Cannot forward sequence of length " + t);
  const C = config.n_embd;
  const wte = model.transformer.wte.weight.data, wpe = model.transformer.wpe.weight.data;

  // pos = torch.arange(0, t)
  // tok_emb = self.transformer.wte(idx)   (b, t, n_embd)
  // pos_emb = self.transformer.wpe(pos)   (t, n_embd)
  // x = self.transformer.drop(tok_emb + pos_emb)
  const tok_emb = tensor([B * t, C]);
  const pos_emb = tensor([t, C]);
  for (let p = 0; p < t; p++)
    for (let c = 0; c < C; c++) pos_emb.data[p * C + c] = wpe[p * C + c];
  for (let b = 0; b < B; b++)
    for (let p = 0; p < t; p++)
      for (let c = 0; c < C; c++) tok_emb.data[(b * t + p) * C + c] = wte[idx[b][p] * C + c];

  const x0 = tensor([B * t, C]);
  for (let n = 0; n < B * t; n++)
    for (let c = 0; c < C; c++) x0.data[n * C + c] = tok_emb.data[n * C + c] + pos_emb.data[(n % t) * C + c];

  // for block in self.transformer.h: x = block(x)
  // dove Block.forward è  x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))
  let x = x0;
  const blocks = [];
  for (const block of model.transformer.h) {
    const ln_1 = layer_norm(x, block.ln_1);
    const attn = causal_self_attention(ln_1.out, block.attn, B, t);
    const x_mid = somma(x, attn.out);
    const ln_2 = layer_norm(x_mid, block.ln_2);
    const mlp = mlp_forward(ln_2.out, block.mlp);
    const x_out = somma(x_mid, mlp.out);
    blocks.push({ x_in: x, ln_1, attn, x_mid, ln_2, mlp, x_out });
    x = x_out;
  }

  // x = self.transformer.ln_f(x);  logits = self.lm_head(x)
  const ln_f = layer_norm(x, model.transformer.ln_f);
  const logits = linear(ln_f.out, model.lm_head);

  const cache = { idx, B, t, tok_emb, pos_emb, x0, blocks, x_final: x, ln_f, logits };
  if (!targets) return { logits, loss: null, cache };

  const ce = cross_entropy(logits, targets.flat());
  cache.targets = targets.flat();
  cache.probs = ce.probs;
  cache.contate = ce.contate;
  return { logits, loss: ce.loss, cache };
}


/* --------------------------------------------------------------------------
   3. GRADIENTI — loss.backward(), scritto a mano

   In nanoGPT questa sezione non esiste: la fa l'autograd di PyTorch. Qui
   ogni funzione della sezione 2 ha la sua gemella che percorre la strada
   al contrario: riceve il gradiente della loss rispetto alla propria
   uscita e restituisce quello rispetto al proprio ingresso, accumulando
   nel frattempo i gradienti dei pesi in .grad. Accumulando, e non
   sovrascrivendo: wte riceve gradiente due volte, come embedding e come
   lm_head, ed è così che la condivisione dei pesi si allena.
   -------------------------------------------------------------------------- */

function linear_backward(dy, x, layer) {
  const [N, inF] = x.shape;
  const outF = layer.weight.shape[0];
  const W = layer.weight.data, dW = layer.weight.grad;
  const db = layer.bias ? layer.bias.grad : null;
  const dx = tensor([N, inF]);
  for (let n = 0; n < N; n++) {
    const xo = n * inF;
    for (let o = 0; o < outF; o++) {
      const g = dy.data[n * outF + o];
      if (g === 0) continue;
      if (db) db[o] += g;
      const wo = o * inF;
      for (let i = 0; i < inF; i++) {
        dx.data[xo + i] += g * W[wo + i];
        dW[wo + i] += g * x.data[xo + i];
      }
    }
  }
  return dx;
}

/* La derivata della LayerNorm, nella forma compatta:
   dx = rstd · (dx̂ − media(dx̂) − x̂ · media(dx̂ · x̂)), con dx̂ = dy · weight */
function layer_norm_backward(dy, x, ln, cache) {
  const [N, C] = x.shape;
  const dx = tensor([N, C]);
  const w = ln.weight.data;
  for (let n = 0; n < N; n++) {
    const o = n * C, m = cache.mean[n], r = cache.rstd[n];
    let media_dxh = 0, media_dxh_xh = 0;
    for (let c = 0; c < C; c++) {
      const xh = (x.data[o + c] - m) * r;
      const g = dy.data[o + c];
      ln.weight.grad[c] += g * xh;
      if (ln.bias) ln.bias.grad[c] += g;
      const dxh = g * w[c];
      media_dxh += dxh;
      media_dxh_xh += dxh * xh;
    }
    media_dxh /= C; media_dxh_xh /= C;
    for (let c = 0; c < C; c++) {
      const xh = (x.data[o + c] - m) * r;
      dx.data[o + c] = r * (dy.data[o + c] * w[c] - media_dxh - xh * media_dxh_xh);
    }
  }
  return dx;
}

function causal_self_attention_backward(dout, x, attn, c, B, T) {
  const C = config.n_embd, n_head = config.n_head, hs = C / n_head;

  // c_proj
  const dy = linear_backward(dout, c.y, attn.c_proj);
  const dqkv = tensor([B * T, 3 * C]);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < n_head; h++) {
      const a = c.att[b][h].data, qh = c.q[b][h].data, kh = c.k[b][h].data, vh = c.v[b][h].data;

      // le colonne della testa h di dy, cioè dy_h (T × hs)
      const dyh = new Float64Array(T * hs);
      for (let t = 0; t < T; t++)
        for (let d = 0; d < hs; d++) dyh[t * hs + d] = dy.data[(b * T + t) * C + h * hs + d];

      // y = att @ v   →   datt = dy @ vᵀ,   dv = attᵀ @ dy
      const datt = new Float64Array(T * T);
      const dv = new Float64Array(T * hs);
      for (let i = 0; i < T; i++)
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let d = 0; d < hs; d++) {
            s += dyh[i * hs + d] * vh[j * hs + d];
            dv[j * hs + d] += a[i * T + j] * dyh[i * hs + d];
          }
          datt[i * T + j] = s;
        }

      // softmax riga per riga: dz_j = a_j (da_j − Σ_k a_k da_k).
      // Le posizioni mascherate hanno a = 0, quindi gradiente zero: la
      // maschera non lascia passare niente nemmeno all'indietro.
      // Poi la scala: d(raw) = d(scaled) · 1/√hs
      const draw = new Float64Array(T * T);
      for (let i = 0; i < T; i++) {
        let interno = 0;
        for (let j = 0; j <= i; j++) interno += a[i * T + j] * datt[i * T + j];
        for (let j = 0; j <= i; j++) draw[i * T + j] = a[i * T + j] * (datt[i * T + j] - interno) * c.scala;
      }

      // raw = q @ kᵀ   →   dq = draw @ k,   dk = drawᵀ @ q
      for (let i = 0; i < T; i++)
        for (let j = 0; j <= i; j++) {
          const g = draw[i * T + j];
          if (g === 0) continue;
          for (let d = 0; d < hs; d++) {
            dqkv.data[(b * T + i) * 3 * C + h * hs + d] += g * kh[j * hs + d];
            dqkv.data[(b * T + j) * 3 * C + C + h * hs + d] += g * qh[i * hs + d];
          }
        }
      for (let t = 0; t < T; t++)
        for (let d = 0; d < hs; d++) dqkv.data[(b * T + t) * 3 * C + 2 * C + h * hs + d] += dv[t * hs + d];
    }
  }

  // c_attn
  return linear_backward(dqkv, x, attn.c_attn);
}

function mlp_backward(dout, x, mlp, c) {
  const dh_gelu = linear_backward(dout, c.h_gelu, mlp.c_proj);
  const dh_fc = tensor(c.h_fc.shape);
  // GELU'(x) = Φ(x) + x · φ(x)
  for (let i = 0; i < dh_fc.data.length; i++) {
    const x_i = c.h_fc.data[i];
    dh_fc.data[i] = dh_gelu.data[i] * (c.phi[i] + x_i * Math.exp(-0.5 * x_i * x_i) / Math.sqrt(2 * Math.PI));
  }
  return linear_backward(dh_fc, x, mlp.c_fc);
}

/* loss.backward(). fattore scala la loss prima di derivare: è il
   loss / gradient_accumulation_steps di train.py. */
function backward(model, cache, fattore = 1) {
  const { B, t, blocks, ln_f, logits, probs, targets, contate } = cache;
  const C = config.n_embd, V = config.vocab_size, N = B * t;

  // cross_entropy: d logits = (softmax − one-hot) / numero di target
  const dlogits = tensor([N, V]);
  for (let n = 0; n < N; n++) {
    if (targets[n] === -1) continue;
    for (let v = 0; v < V; v++) dlogits.data[n * V + v] = probs.data[n * V + v] * fattore / contate;
    dlogits.data[n * V + targets[n]] -= fattore / contate;
  }

  // lm_head e ln_f
  const dln_f = linear_backward(dlogits, ln_f.out, model.lm_head);
  let dx = layer_norm_backward(dln_f, cache.x_final, model.transformer.ln_f, ln_f);

  // i Block al contrario. La somma residua copia il gradiente sui due rami:
  // è per questo che il segnale arriva intatto anche al primo Block.
  for (let l = blocks.length - 1; l >= 0; l--) {
    const blk = blocks[l], mod = model.transformer.h[l];
    const dln_2 = mlp_backward(dx, blk.ln_2.out, mod.mlp, blk.mlp);
    const dx_mid = somma(dx, layer_norm_backward(dln_2, blk.x_mid, mod.ln_2, blk.ln_2));
    const dln_1 = causal_self_attention_backward(dx_mid, blk.ln_1.out, mod.attn, blk.attn, B, t);
    dx = somma(dx_mid, layer_norm_backward(dln_1, blk.x_in, mod.ln_1, blk.ln_1));
  }

  // x = tok_emb + pos_emb: il gradiente torna alle righe di wte e wpe usate
  const dwte = model.transformer.wte.weight.grad, dwpe = model.transformer.wpe.weight.grad;
  for (let b = 0; b < B; b++)
    for (let p = 0; p < t; p++) {
      const n = b * t + p, tok = cache.idx[b][p];
      for (let c = 0; c < C; c++) {
        dwte[tok * C + c] += dx.data[n * C + c];
        dwpe[p * C + c] += dx.data[n * C + c];
      }
    }
}


/* --------------------------------------------------------------------------
   4. ADDESTRAMENTO — train.py
   -------------------------------------------------------------------------- */

/* data/shakespeare_char/prepare.py: il 90% del testo per l'addestramento,
   il resto per la validazione, tutti e due codificati in interi. */
function prepare(data) {
  const n = data.length;
  const train_data = data.slice(0, Math.floor(n * 0.9));
  const val_data = data.slice(Math.floor(n * 0.9));
  return {
    train: Uint8Array.from(encode(train_data)),
    val: Uint8Array.from(encode(val_data)),
  };
}

/* get_batch(split): batch_size finestre prese a caso; y è x spostata di
   un carattere. rng sostituisce torch.randint, con seme. */
function get_batch(dati, split, rng, batch_size, block_size) {
  const data = split === "train" ? dati.train : dati.val;
  const X = [], Y = [];
  for (let b = 0; b < batch_size; b++) {
    const i = Math.floor(rng() * (data.length - block_size));
    X.push(Array.from(data.subarray(i, i + block_size)));
    Y.push(Array.from(data.subarray(i + 1, i + 1 + block_size)));
  }
  return { X, Y };
}

/* get_lr(it): riscaldamento lineare, poi coseno fino a min_lr. Copiata riga
   per riga da train.py. */
function get_lr(it, tc = train_config) {
  // 1) linear warmup for warmup_iters steps
  if (it < tc.warmup_iters) return tc.learning_rate * (it + 1) / (tc.warmup_iters + 1);
  // 2) if it > lr_decay_iters, return min learning rate
  if (it > tc.lr_decay_iters) return tc.min_lr;
  // 3) in between, use cosine decay down to min learning rate
  const decay_ratio = (it - tc.warmup_iters) / (tc.lr_decay_iters - tc.warmup_iters);
  const coeff = 0.5 * (1.0 + Math.cos(Math.PI * decay_ratio));
  return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr);
}

/* GPT.configure_optimizers: i parametri a due o più dimensioni (le matrici
   dei Linear e gli embedding) hanno weight decay, gli altri (bias e
   LayerNorm) no. Poi AdamW. */
function configure_optimizers(model, weight_decay, learning_rate, betas) {
  const param_dict = named_parameters(model);
  const decay_params = param_dict.filter(([, p]) => p.shape.length >= 2);
  const nodecay_params = param_dict.filter(([, p]) => p.shape.length < 2);
  return {
    param_groups: [
      { params: decay_params, weight_decay },
      { params: nodecay_params, weight_decay: 0.0 },
    ],
    lr: learning_rate,
    betas,
    eps: 1e-8,
    step_count: 0,
    m: new Map(),
    v: new Map(),
  };
}

/* optimizer.step() di torch.optim.AdamW. Il weight decay è "disaccoppiato":
   rimpicciolisce il peso direttamente invece di passare dal gradiente. */
function optimizer_step(optimizer) {
  optimizer.step_count++;
  const [beta1, beta2] = optimizer.betas;
  const lr = optimizer.lr, eps = optimizer.eps, t = optimizer.step_count;
  const corr1 = 1 - Math.pow(beta1, t), corr2 = 1 - Math.pow(beta2, t);
  for (const group of optimizer.param_groups) {
    for (const [, p] of group.params) {
      if (!optimizer.m.has(p)) {
        optimizer.m.set(p, new Float64Array(p.data.length));
        optimizer.v.set(p, new Float64Array(p.data.length));
      }
      const m = optimizer.m.get(p), v = optimizer.v.get(p);
      const decadimento = 1 - lr * group.weight_decay;
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i];
        p.data[i] *= decadimento;
        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;
        p.data[i] -= lr * (m[i] / corr1) / (Math.sqrt(v[i] / corr2) + eps);
      }
    }
  }
}

/* optimizer.zero_grad(set_to_none=True) */
function zero_grad(model) {
  for (const [, p] of named_parameters(model)) p.grad.fill(0);
}

/* torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip): se la
   norma di tutti i gradienti messi insieme supera la soglia, si scalano
   tutti dello stesso fattore. Restituisce la norma prima del taglio. */
function clip_grad_norm_(model, max_norm) {
  let s = 0;
  const params = named_parameters(model);
  for (const [, p] of params) for (const g of p.grad) s += g * g;
  const total_norm = Math.sqrt(s);
  const clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1)
    for (const [, p] of params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= clip_coef;
  return total_norm;
}

/* estimate_loss(): la loss media su eval_iters batch di ciascuna parte. */
function estimate_loss(model, dati, rng, tc = train_config) {
  const out = {};
  for (const split of ["train", "val"]) {
    let s = 0;
    for (let k = 0; k < tc.eval_iters; k++) {
      const { X, Y } = get_batch(dati, split, rng, tc.batch_size, config.block_size);
      s += forward(model, X, Y).loss;
    }
    out[split] = s / tc.eval_iters;
  }
  return out;
}

/* Un'iterazione del ciclo di train.py, dal learning rate al zero_grad.
   stato tiene quello che in train.py sono variabili globali: iter_num,
   l'ottimizzatore, il generatore dei batch. */
function training_step(model, stato, dati, tc = train_config) {
  const lr = tc.decay_lr ? get_lr(stato.iter_num, tc) : tc.learning_rate;
  stato.optimizer.lr = lr;

  let lossf = 0;
  for (let micro_step = 0; micro_step < tc.gradient_accumulation_steps; micro_step++) {
    const { X, Y } = get_batch(dati, "train", stato.rng, tc.batch_size, config.block_size);
    const { loss, cache } = forward(model, X, Y);
    backward(model, cache, 1 / tc.gradient_accumulation_steps);
    lossf += loss / tc.gradient_accumulation_steps;
  }
  const norm = tc.grad_clip !== 0 ? clip_grad_norm_(model, tc.grad_clip) : null;
  optimizer_step(stato.optimizer);
  zero_grad(model);
  stato.iter_num++;
  return { loss: lossf, lr, norm };
}


/* --------------------------------------------------------------------------
   5. GENERAZIONE — GPT.generate()
   -------------------------------------------------------------------------- */

/* La distribuzione del prossimo carattere, cioè il corpo di un giro del
   ciclo di generate() fino a F.softmax compreso. Separata da generate()
   perché la pagina la disegna prima di estrarre. */
function next_token_probs(model, idx, temperature, top_k) {
  // if the sequence context is growing too long we must crop it at block_size
  const idx_cond = idx.length <= config.block_size ? idx : idx.slice(-config.block_size);
  // forward the model to get the logits for the index in the sequence
  const { logits: tutti } = forward(model, [idx_cond]);
  // pluck the logits at the final step and scale by desired temperature
  const V = config.vocab_size;
  const logits = Float64Array.from(tutti.data.subarray((idx_cond.length - 1) * V, idx_cond.length * V));
  for (let i = 0; i < V; i++) logits[i] /= temperature;
  // optionally crop the logits to only the top k options
  if (top_k !== null && top_k < V) {
    const v = Array.from(logits).sort((a, b) => b - a)[top_k - 1];
    for (let i = 0; i < V; i++) if (logits[i] < v) logits[i] = -Infinity;
  }
  // apply softmax to convert logits to (normalized) probabilities
  const probs = softmax(logits);
  return { idx_cond, logits, probs };
}

/* torch.multinomial(probs, num_samples=1) */
function multinomial(probs, rng) {
  const u = rng();
  let cumulata = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulata += probs[i];
    if (u < cumulata) return i;
  }
  return probs.length - 1;
}

function generate(model, idx, max_new_tokens, temperature = 1.0, top_k = null, rng = Math.random) {
  idx = idx.slice();
  for (let n = 0; n < max_new_tokens; n++) {
    const { probs } = next_token_probs(model, idx, temperature, top_k);
    const idx_next = multinomial(probs, rng);   // sample from the distribution
    idx.push(idx_next);                         // append sampled index to the running sequence
  }
  return idx;
}

/* GPT.get_num_params(): i parametri di una configurazione qualunque, senza
   costruire il modello, modulo per modulo. Con non_embedding si toglie wpe
   come fa nanoGPT; wte resta perché, condivisa con lm_head, è un peso vero
   dell'ultimo strato. */
function conta_parametri(cfg) {
  const C = cfg.n_embd, b = cfg.bias ? 1 : 0;
  const ln = C + b * C;
  const righe = [
    { nome: "transformer.wte", n: cfg.vocab_size * C },
    { nome: "transformer.wpe", n: cfg.block_size * C },
    { nome: "h[i].ln_1", n: ln, perBlock: true },
    { nome: "h[i].attn.c_attn", n: C * 3 * C + b * 3 * C, perBlock: true },
    { nome: "h[i].attn.c_proj", n: C * C + b * C, perBlock: true },
    { nome: "h[i].ln_2", n: ln, perBlock: true },
    { nome: "h[i].mlp.c_fc", n: C * 4 * C + b * 4 * C, perBlock: true },
    { nome: "h[i].mlp.c_proj", n: 4 * C * C + b * C, perBlock: true },
    { nome: "transformer.ln_f", n: ln },
    { nome: "lm_head", n: 0, legato: true },
  ];
  let totale = 0;
  for (const r of righe) totale += r.perBlock ? r.n * cfg.n_layer : r.n;
  return { righe, totale, non_embedding: totale - cfg.block_size * C };
}


/* --------------------------------------------------------------------------
   6. INTERFACCIA

   Da qui in poi il codice parla con la pagina, e i nomi tornano in
   italiano. Ogni laboratorio ha una funzione disegnaQualcosa() che legge
   lo STATO e la cache dell'ultimo forward(), e non calcola niente del
   modello per conto suo: quello che si vede è sempre quello che il modello
   ha calcolato davvero.
   -------------------------------------------------------------------------- */

const nodo = id => document.getElementById(id);

/* Il testo inserito con innerHTML passa da qui: i caratteri del prompt
   sono scritti da chi legge, e & è nel vocabolario di Shakespeare. */
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* Come in correlazione.js: decimali secondo la lingua della pagina e il
   meno tipografico U+2212 al posto del trattino. Un formattatore per ogni
   numero di decimali, creato una volta sola: crearne uno nuovo a ogni
   numero costa quanto disegnare la cella, e le celle sono migliaia. */
const FORMATTATORI = {};
function fmtNum(valore, decimali = 2) {
  if (!isFinite(valore)) return valore < 0 ? "−∞" : "—";
  const f = FORMATTATORI[decimali] || (FORMATTATORI[decimali] = new Intl.NumberFormat(I18N.locale, {
    minimumFractionDigits: decimali,
    maximumFractionDigits: decimali,
  }));
  const s = f.format(Math.abs(valore));
  return (valore < 0 && parseFloat(s.replace(",", ".")) !== 0 ? "−" : "") + s;
}

const fmtIntero = n => new Intl.NumberFormat(I18N.locale).format(n);

/* Lo spazio e l'a capo, che stampati sparirebbero. */
const mostra = ch => ch === " " ? "␣" : ch === "\n" ? "↵" : ch;
const eSpazio = ch => ch === " " || ch === "\n";
const nelVocabolario = ch => Object.prototype.hasOwnProperty.call(stoi, ch);

const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(nome, attributi, testo) {
  const e = document.createElementNS(SVGNS, nome);
  for (const k in attributi) e.setAttribute(k, attributi[k]);
  if (testo !== undefined) e.textContent = testo;
  return e;
}

/* Un testo in cui le parti fra backtick sono codice: finiscono in uno
   span .ident, che le tiene fuori dal maiuscolo di .stat__key. Così
   "`iter_num`" resta iter_num e "`μ`" non diventa una M. */
function scriviEtichetta(elemento, testo) {
  testo.split("`").forEach((pezzo, n) => {
    if (!pezzo) return;
    if (n % 2) {
      const s = document.createElement("span");
      s.className = "ident";
      s.textContent = pezzo;
      elemento.appendChild(s);
    } else {
      elemento.appendChild(document.createTextNode(pezzo));
    }
  });
}

/* Le tessere numeriche: .stat della sezione 10, come in correlazione.js. */
function tessera(chiave, valore, nota, hero) {
  const box = document.createElement("div");
  box.className = "stat" + (hero ? " stat--hero" : "");
  const k = document.createElement("span");
  k.className = "stat__key";
  scriviEtichetta(k, chiave);
  const v = document.createElement("span");
  v.className = "stat__value";
  v.textContent = valore;
  box.append(k, v);
  if (nota) {
    const n = document.createElement("span");
    n.className = "stat__note";
    n.textContent = nota;
    box.appendChild(n);
  }
  return box;
}

function riempi(elemento, figli) {
  elemento.textContent = "";
  figli.forEach(f => elemento.appendChild(f));
}

/* --- Gruppi di chip -------------------------------------------------------
   Un solo ascoltatore per gruppo, sul contenitore: i chip si possono
   ricreare (quelli delle posizioni cambiano con il prompt) senza perdere il
   collegamento. Si aggiornano al loro posto invece di essere ricostruiti,
   così il chip appena premuto non perde il focus della tastiera. */

function preparaChip(contenitore, alClic) {
  contenitore.addEventListener("click", e => {
    const b = e.target.closest("button.chip");
    if (b && contenitore.contains(b) && !b.disabled) alClic(b.dataset.valore);
  });
}

function aggiornaChip(contenitore, voci, attiva) {
  while (contenitore.children.length > voci.length) contenitore.lastChild.remove();
  voci.forEach((v, i) => {
    let b = contenitore.children[i];
    if (!b) {
      b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      contenitore.appendChild(b);
    }
    b.dataset.valore = String(v.valore);
    b.textContent = v.etichetta;
    b.setAttribute("aria-pressed", String(String(v.valore) === String(attiva)));
  });
}

/* I chip statici (scritti nell'HTML) hanno il valore in un data- tutto
   loro: data-preset, data-passo… Questa li accende e li spegne. */
function segnaChip(contenitore, attributo, attiva) {
  contenitore.querySelectorAll("button.chip").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset[attributo] === String(attiva))));
}


/* --- Matrici ---------------------------------------------------------------
   matrice(t, o) disegna un tensore { shape, data } come una .heat dentro
   una .tensor, con nome e forma sopra. Le opzioni:

     nome, forma       le due etichette sopra la matrice; testa: false le toglie
     dimensione        "compatta", "fine", "mini": celle piccole solo colore;
                       "larga": con il numero, per quattro decimali;
                       niente: con il numero. compatta: true vale "compatta"
     righe, colonne    etichette di riga e di colonna, o niente
     bande             per ogni colonna 1, 2 o 3: la fascia colorata sopra
     dims              i righelli: [righe, colonne], per esempio
                       ["T = 8", "C = 16"]; colonne può essere una lista,
                       un'etichetta per ogni gruppo fra due fessure
     fessure           le colonne prima delle quali la matrice si apre:
                       split, view e le teste affiancate
     valore(i, j)      chi legge il numero, se non è t.data[i·colonne + j]
     maschera(i, j)    vero per le celle a −∞; testoMaschera le scrive
     scelta {i, j}     la cella evidenziata; rigaScelta, colonnaScelta
                       mettono in grassetto le etichette
     evidenzia(i, j)   le celle con il filo sottile: una riga da seguire
     attenua(i, j)     vero per le celle che non c'entrano
     fantasma(i, j)    le copie aggiunte da un broadcast
     clic(i, j)        se c'è, le celle sono bottoni
     note              un testo per riga, in una colonna fuori matrice
     decimali, scala   scala è il valore che dà il colore pieno
     neutra            nessun colore: per gli indici, che non sono misure
     scorri            false: senza la .heat-scroll, quando la matrice sta
                       dentro un prodotto o un mazzo che scorre per conto suo

   Ogni elemento è messo nella griglia con riga e colonna esplicite: le
   righe delle celle sono alte quanto una cella, così in un prodotto la
   matrice di sinistra e il risultato restano allineati anche se solo una
   delle due ha le etichette.

   Il colore è una mescolanza fra --heat-mid e il token del segno, in una
   quantità proporzionale a |valore| / scala: tutto passa da proprietà
   personalizzate, nessun colore scritto qui. */

const CLASSE_DIMENSIONE = { compatta: " heat--compact", fine: " heat--fine", mini: " heat--mini", larga: " heat--wide" };

function matrice(t, o = {}) {
  const R = t.shape.length === 2 ? t.shape[0] : 1;
  const K = t.shape.length === 2 ? t.shape[1] : t.shape[0];
  const valore = o.valore || ((i, j) => t.data[i * K + j]);
  const maschera = o.maschera || (() => false);
  const decimali = o.decimali === undefined ? 2 : o.decimali;
  const dimensione = o.dimensione || (o.compatta ? "compatta" : "");
  const muta = dimensione === "compatta" || dimensione === "fine" || dimensione === "mini";
  const dims = o.dims || [];

  let scala = o.scala;
  if (!scala) {
    scala = 0;
    for (let i = 0; i < R; i++)
      for (let j = 0; j < K; j++) {
        const v = valore(i, j);
        if (!maschera(i, j) && isFinite(v)) scala = Math.max(scala, Math.abs(v));
      }
    scala = scala || 1;
  }

  const box = document.createElement("div");
  box.className = "tensor";
  if (o.chiave) box.dataset.diagramma = o.chiave;

  if (o.testa !== false) {
    const testa = document.createElement("div");
    testa.className = "tensor__head";
    const n = document.createElement("span");
    n.className = "tensor__name";
    n.textContent = o.nome || "";
    testa.appendChild(n);
    if (o.forma) {
      const f = document.createElement("span");
      f.className = "tensor__shape";
      f.textContent = o.forma;
      testa.appendChild(f);
    }
    box.appendChild(testa);
  }

  const griglia = document.createElement("div");
  griglia.className = "heat" + (CLASSE_DIMENSIONE[dimensione] || "");

  /* Le tracce della griglia. Colonne: etichette di riga, le celle con le
     fessure in mezzo, il righello delle righe, le note. Righe: etichette
     di colonna, bande, le celle, il righello delle colonne. */
  const fessure = new Set(o.fessure || []);
  const colonneG = [];
  const aggiungiColonna = misura => { colonneG.push(misura); return colonneG.length; };
  const colRighe = o.righe ? aggiungiColonna("auto") : 0;
  const colCella = [];
  for (let j = 0; j < K; j++) {
    if (j > 0 && fessure.has(j)) aggiungiColonna("var(--s-3)");
    colCella.push(aggiungiColonna("var(--cella-l, 38px)"));
  }
  const colRighello = dims[0] ? aggiungiColonna("auto") : 0;
  const colNote = o.note ? aggiungiColonna("auto") : 0;

  const righeG = [];
  const aggiungiRiga = misura => { righeG.push(misura); return righeG.length; };
  const rigaColonne = o.colonne ? aggiungiRiga("auto") : 0;
  const rigaBande = o.bande ? aggiungiRiga("auto") : 0;
  const rigaCella = [];
  for (let i = 0; i < R; i++) rigaCella.push(aggiungiRiga("var(--cella-a, 28px)"));
  const rigaRighello = dims[1] ? aggiungiRiga("auto") : 0;

  griglia.style.gridTemplateColumns = colonneG.join(" ");
  griglia.style.gridTemplateRows = righeG.join(" ");

  const metti = (el, riga, colonna) => {
    el.style.gridRow = String(riga);
    el.style.gridColumn = String(colonna);
    return el;
  };

  /* Una matrice muta è un'immagine per chi usa uno screen reader: 128
     celle senza numero lette una per una non aiutano nessuno. Una matrice
     con i numeri è una tabella, con righe vere. */
  if (muta) {
    griglia.setAttribute("role", "img");
    griglia.setAttribute("aria-label", (o.nome || "") + " " + (o.forma || ""));
  } else {
    griglia.setAttribute("role", "table");
    griglia.setAttribute("aria-label", o.nome || "");
  }

  if (o.colonne) {
    const riga = document.createElement("div");
    riga.style.display = "contents";
    if (!muta) riga.setAttribute("role", "row");
    o.colonne.forEach((c, j) => {
      const d = metti(document.createElement("div"), rigaColonne, colCella[j]);
      d.className = "heat__colhead" + (o.colonnaScelta === j ? " is-focus" : "");
      if (!muta) d.setAttribute("role", "columnheader");
      d.textContent = c;
      riga.appendChild(d);
    });
    griglia.appendChild(riga);
  }

  if (o.bande) {
    o.bande.forEach((b, j) => {
      const d = metti(document.createElement("div"), rigaBande, colCella[j]);
      d.className = "heat__band heat__band--" + b;
      d.setAttribute("aria-hidden", "true");
      griglia.appendChild(d);
    });
  }

  for (let i = 0; i < R; i++) {
    const riga = document.createElement("div");
    riga.style.display = "contents";
    if (!muta) riga.setAttribute("role", "row");

    if (o.righe) {
      const h = metti(document.createElement("div"), rigaCella[i], colRighe);
      h.className = "heat__rowhead" + (o.rigaScelta === i ? " is-focus" : "");
      if (!muta) h.setAttribute("role", "rowheader");
      h.textContent = o.righe[i];
      riga.appendChild(h);
    }

    for (let j = 0; j < K; j++) {
      const v = valore(i, j);
      const cella = metti(document.createElement(o.clic ? "button" : "div"), rigaCella[i], colCella[j]);
      cella.className = "heat__cell";
      if (o.clic) {
        cella.type = "button";
        cella.dataset.i = String(i);
        cella.dataset.j = String(j);
        cella.addEventListener("click", () => o.clic(i, j));
      } else if (!muta) {
        cella.setAttribute("role", "cell");
      }

      const nomeCella = (o.nome || "") + "[" + i + ", " + j + "]";
      if (maschera(i, j)) {
        const testo = o.testoMaschera === undefined ? "−∞" : o.testoMaschera;
        cella.classList.add("is-masked");
        if (!muta) cella.textContent = testo;
        cella.title = nomeCella + " = " + testo;
      } else {
        if (!o.neutra) {
          const q = Math.min(1, Math.abs(v) / scala);
          cella.style.setProperty("--tinta", v >= 0 ? "var(--heat-pos)" : "var(--heat-neg)");
          cella.style.setProperty("--mescola", (q * (muta ? 85 : 55)).toFixed(1) + "%");
        }
        if (!muta) cella.textContent = fmtNum(v, decimali);
        if (dimensione !== "mini") cella.title = nomeCella + " = " + fmtNum(v, Math.max(decimali, 4));
      }
      if (o.clic) cella.setAttribute("aria-label", cella.title);
      if (o.scelta && o.scelta.i === i && o.scelta.j === j) cella.classList.add("is-focus");
      if (o.evidenzia && o.evidenzia(i, j)) cella.classList.add("is-pick");
      if (o.attenua && o.attenua(i, j)) cella.classList.add("is-dim");
      if (o.fantasma && o.fantasma(i, j)) cella.classList.add("is-ghost");
      riga.appendChild(cella);
    }

    if (o.note) {
      const d = metti(document.createElement("div"), rigaCella[i], colNote);
      d.className = "heat__note";
      if (!muta) d.setAttribute("role", "cell");
      d.textContent = o.note[i];
      riga.appendChild(d);
    }
    griglia.appendChild(riga);
  }

  /* I righelli: uno accanto a tutte le righe, uno sotto le colonne o sotto
     ogni gruppo di colonne fra due fessure. */
  if (dims[0]) {
    const d = document.createElement("div");
    d.className = "heat__ruler heat__ruler--y";
    d.setAttribute("aria-hidden", "true");
    d.style.gridRow = rigaCella[0] + " / " + (rigaCella[R - 1] + 1);
    d.style.gridColumn = String(colRighello);
    d.textContent = dims[0];
    griglia.appendChild(d);
  }
  if (dims[1]) {
    const tagli = [0, ...[...fessure].filter(j => j > 0 && j < K).sort((a, b) => a - b), K];
    const gruppi = Array.isArray(dims[1]) ? dims[1] : [dims[1]];
    const tratti = Array.isArray(dims[1]) ? tagli.slice(0, -1).map((da, g) => [da, tagli[g + 1]]) : [[0, K]];
    tratti.forEach(([da, a], g) => {
      const d = document.createElement("div");
      d.className = "heat__ruler heat__ruler--x";
      d.setAttribute("aria-hidden", "true");
      d.style.gridRow = String(rigaRighello);
      d.style.gridColumn = colCella[da] + " / " + (colCella[a - 1] + 1);
      d.textContent = gruppi[g] || "";
      griglia.appendChild(d);
    });
  }

  if (o.scorri === false) {
    box.appendChild(griglia);
  } else {
    const scorri = document.createElement("div");
    scorri.className = "heat-scroll";
    scorri.appendChild(griglia);
    box.appendChild(scorri);
  }
  return box;
}

function operatore(simbolo) {
  const s = document.createElement("span");
  s.className = "tensors__op";
  s.setAttribute("aria-hidden", "true");
  s.textContent = simbolo;
  return s;
}

/* Un pezzo di un tensore, come nuova matrice: le righe da r0 a r1 e le
   colonne da c0 a c1 (escluse). Serve per q, k, v dentro qkv, per una
   riga di x, per le colonne di una testa. */
function ritaglia(t, r0, r1, c0, c1) {
  const K = t.shape[1];
  const out = tensor([r1 - r0, c1 - c0]);
  for (let i = r0; i < r1; i++)
    for (let j = c0; j < c1; j++) out.data[(i - r0) * (c1 - c0) + (j - c0)] = t.data[i * K + j];
  return out;
}

function trasposta(t) {
  const [R, K] = t.shape;
  const out = tensor([K, R]);
  for (let i = 0; i < R; i++)
    for (let j = 0; j < K; j++) out.data[j * R + i] = t.data[i * K + j];
  return out;
}

/* Un parametro 1D (un bias) come matrice di una riga. */
const comeRiga = p => ({ shape: [1, p.shape[0]], data: p.data });

/* Una matrice (R, K) fatta ripetendo un vettore: per riga (un bias, che
   vale per ogni posizione) o per colonna (una media, che vale per ogni
   canale). È quello che fa il broadcast, senza scriverlo in memoria. */
function ripetiRiga(vettore, R) {
  const K = vettore.length;
  const out = tensor([R, K]);
  for (let i = 0; i < R; i++) out.data.set(vettore, i * K);
  return out;
}
function ripetiColonna(vettore, K) {
  const R = vettore.length;
  const out = tensor([R, K]);
  for (let i = 0; i < R; i++) out.data.fill(vettore[i], i * K, (i + 1) * K);
  return out;
}


/* --- Disegni di operazioni -------------------------------------------------
   Le funzioni qui sotto mettono insieme le matrici nei quattro modi che
   servono a disegnare un modello: il prodotto riga per colonna, il mazzo
   delle teste, l'operazione con il suo codice, il flusso dall'alto in
   basso. Le classi sono nella sezione 12 di style.css. */

/* I quattro tipi di operazione, come li scrive la pagina. */
const TIPI_OP = {
  prodotto: { it: "prodotto · righe × colonne", en: "product · rows × columns" },
  forma: { it: "forma · nessun numero cambia", en: "shape · no number changes" },
  elemento: { it: "cella per cella", en: "cell by cell" },
  riga: { it: "lungo una dimensione", en: "along one dimension" },
};

/* L'operazione fra due tensori: il glifo, il codice che la esegue, il
   tipo, e se serve una nota. */
function op(tipo, glifo, codice, nota) {
  const d = document.createElement("div");
  d.className = "op op--" + tipo;
  const g = document.createElement("span");
  g.className = "op__glyph";
  g.setAttribute("aria-hidden", "true");
  g.textContent = glifo;
  const corpo = document.createElement("span");
  corpo.className = "op__body";
  const c = document.createElement("code");
  c.className = "op__code";
  c.textContent = codice;
  const k = document.createElement("span");
  k.className = "op__kind";
  k.textContent = I18N.t(TIPI_OP[tipo]);
  corpo.append(c, k);
  if (nota) {
    const n = document.createElement("span");
    n.className = "op__note";
    n.textContent = nota;
    corpo.appendChild(n);
  }
  d.append(g, corpo);
  return d;
}

/* Una tappa del flusso: una fila di tensori. E un passaggio: la freccia
   verso il basso con le operazioni che portano alla tappa dopo. */
function tappa(...figli) {
  const d = document.createElement("div");
  d.className = "flow__stage";
  figli.filter(Boolean).forEach(f => d.appendChild(f));
  return d;
}
function passaggio(...figli) {
  const d = document.createElement("div");
  d.className = "flow__op";
  figli.filter(Boolean).forEach(f => d.appendChild(f));
  return d;
}

/* Un'etichetta dentro un flusso, per dire "da qui in poi, una testa". */
function didascalia(testo) {
  const p = document.createElement("p");
  p.className = "lab__label";
  p.textContent = testo;
  return p;
}

/* Il prodotto riga per colonna: a @ b (+ bias) = out.
     a, b, out      tensori { shape, data }; bias un parametro 1D, o niente
     oa, ob, obias, oout   le opzioni di matrice() per ciascuno
     dimensione     una sola per tutte e quattro: le celle devono combaciare
     op             l'operazione, nell'angolo in alto a sinistra
     sel {i, j}     la cella scelta del risultato: accende la riga i di a,
                    la colonna j di b e del bias, e attenua il resto
     clic(i, j)     le celle del risultato sono bottoni
     chiave         il nome del disegno, per ritrovare il focus */
function prodotto(o) {
  const dimensione = o.dimensione || "";
  const sel = o.sel;
  const oppure = (f, g) => (i, j) => Boolean((f && f(i, j)) || (g && g(i, j)));
  const fuoriRiga = sel ? i => i !== sel.i : null;
  const fuoriColonna = sel ? (i, j) => j !== sel.j : null;
  /* La riga e la colonna che si moltiplicano hanno il filo: attenuare il
     resto non basta, quando i pesi sono quasi tutti vicini a zero e il loro
     colore è già quello dello sfondo. */
  const laRiga = sel ? i => i === sel.i : null;
  const laColonna = sel ? (i, j) => j === sel.j : null;
  const base = { dimensione, scorri: false };

  const g = document.createElement("div");
  g.className = "matmul" + (o.bias ? "" : " matmul--nobias");
  if (o.chiave) g.dataset.diagramma = o.chiave;

  const metti = (el, classe) => { el.classList.add(classe); g.appendChild(el); };

  if (o.op) {
    const d = document.createElement("div");
    d.appendChild(o.op);
    metti(d, "matmul__op");
  }
  const ob = o.ob || {}, oa = o.oa || {}, obias = o.obias || {}, oout = o.oout || {};
  metti(matrice(o.b, { ...ob, ...base,
    attenua: oppure(ob.attenua, fuoriColonna),
    evidenzia: oppure(ob.evidenzia, laColonna),
    colonnaScelta: sel ? sel.j : undefined }), "matmul__b");
  if (o.bias) {
    metti(matrice(comeRiga(o.bias), { ...obias, ...base, attenua: fuoriColonna, evidenzia: laColonna }), "matmul__bias");
  }
  metti(matrice(o.a, { ...oa, ...base,
    attenua: oppure(oa.attenua, fuoriRiga),
    evidenzia: oppure(oa.evidenzia, laRiga),
    rigaScelta: sel ? sel.i : undefined }), "matmul__a");
  metti(matrice(o.out, { ...oout, ...base, scelta: sel, clic: o.clic }), "matmul__out");

  const scorri = document.createElement("div");
  scorri.className = "heat-scroll";
  scorri.appendChild(g);
  return scorri;
}

/* Il mazzo delle teste: un tensore (B, nh, T, hs) con B = 1.
     carte          [{ t, nome }], una per testa, nell'ordine delle teste
     scelta         la testa davanti
     clic(h)        le carte dietro sono bottoni che portano h davanti
     matrice        le opzioni di matrice() per tutte le carte
     davanti        quelle in più per la carta davanti (clic sulle celle…)
     nome, forma    l'intestazione del mazzo intero
     dimensione     "mini" per le miniature: niente intestazioni */
function pila(carte, o = {}) {
  const n = carte.length;
  const davanti = o.scelta || 0;
  const mini = (o.matrice && o.matrice.dimensione) === "mini";

  const mazzo = document.createElement("div");
  mazzo.className = "deck" + (mini ? " deck--mini" : "");
  mazzo.style.setProperty("--n", String(n));

  /* La carta davanti è al livello 0, le altre dietro in ordine di testa.
     Nel documento vanno dalla più lontana alla più vicina: l'ultima
     disegnata copre le altre. */
  const livello = [];
  let prossimo = 1;
  for (let h = 0; h < n; h++) livello[h] = h === davanti ? 0 : prossimo++;
  const ordine = carte.map((_, h) => h).sort((a, b) => livello[b] - livello[a]);

  ordine.forEach(h => {
    const dietro = h !== davanti;
    const bottone = dietro && Boolean(o.clic);
    const carta = document.createElement(bottone ? "button" : "div");
    carta.className = "deck__card" + (dietro ? " is-back" : "");
    carta.style.setProperty("--livello", String(livello[h]));
    if (bottone) {
      carta.type = "button";
      carta.setAttribute("aria-label",
        I18N.t({ it: "Porta davanti la testa ", en: "Bring forward head " }) + h);
      carta.addEventListener("click", () => o.clic(h));
    }
    if (!mini && carte[h].nome) {
      const e = document.createElement("span");
      e.className = "deck__label";
      e.textContent = carte[h].nome;
      carta.appendChild(e);
    }
    const m = matrice(carte[h].t, { ...(o.matrice || {}), ...(dietro ? {} : o.davanti || {}), testa: false, scorri: false });
    if (dietro) m.setAttribute("aria-hidden", "true");
    carta.appendChild(m);
    mazzo.appendChild(carta);
  });

  if (mini) return mazzo;

  const box = document.createElement("div");
  box.className = "tensor";
  const testa = document.createElement("div");
  testa.className = "tensor__head";
  const nome = document.createElement("span");
  nome.className = "tensor__name";
  nome.textContent = o.nome || "";
  testa.appendChild(nome);
  if (o.forma) {
    const f = document.createElement("span");
    f.className = "tensor__shape";
    f.textContent = o.forma;
    testa.appendChild(f);
  }
  box.append(testa, mazzo);
  return box;
}

/* Le carte di un mazzo dalla lista per testa della cache: A.q[0], A.att[0]… */
const carte = (lista, nome) => lista.map((t, h) => ({ t, nome: nome + "[0, " + h + "]" }));

/* Una miniatura per lo schema del Block. */
const miniatura = (t, o = {}) => matrice(t, { ...o, dimensione: "mini", testa: false, scorri: false });

/* Ridisegnare dopo un clic su una cella cancella il bottone che aveva il
   focus: chi usa la tastiera si ritroverebbe in cima alla pagina. Questa
   ricorda quale cella di quale disegno era, ridisegna, e la ritrova. */
function conFocus(ridisegna) {
  const a = document.activeElement;
  const disegno = a && a.closest ? a.closest("[data-diagramma]") : null;
  const chiave = disegno ? disegno.dataset.diagramma : null;
  const i = a && a.dataset ? a.dataset.i : undefined;
  const j = a && a.dataset ? a.dataset.j : undefined;
  ridisegna();
  if (chiave && i !== undefined) {
    const cella = document.querySelector(
      '[data-diagramma="' + chiave + '"] [data-i="' + i + '"][data-j="' + j + '"]');
    if (cella) cella.focus({ preventScroll: true });
  }
}

/* Un prodotto scalare scritto termine per termine, per i .readout. */
function terminiProdotto(a, b, n, decimali = 2) {
  const t = [];
  for (let d = 0; d < n; d++) t.push(fmtNum(a(d), decimali) + "·" + fmtNum(b(d), decimali));
  return t.join(" + ");
}


/* --------------------------------------------------------------------------
   Lo stato della pagina
   -------------------------------------------------------------------------- */

const STATO = {
  model: GPT(config, SEME_PESI),
  prompt: "First Ci",
  pos: 7,                 // la posizione scelta in embedding, LayerNorm, MLP
  ln: "h.0.ln_1",
  strato: 0,              // Block e testa del laboratorio sull'attenzione
  testa: 0,
  passo: 1,
  cella: { i: 7, j: 2 },  // la cella cliccata nell'attenzione
  dropout_p: 0,
  mlp_strato: 0,
  mlp_passo: 1,
  schema_strato: 0,       // il Block disegnato nello schema
  preset: "pagina",
  uscita: null,           // l'ultimo forward(): logits, loss, cache
  /* La cella scelta in ciascun prodotto disegnato. Dove la riga è la
     posizione condivisa (pos) si tiene solo la colonna; logit è null
     finché nessuno clicca, e allora segue il carattere giusto. */
  scelte: {
    leggenda: { i: 0, j: 1 },
    leggenda_testa: 0,
    qkv: { i: 7, j: 20 },
    y: { i: 7, j: 0 },
    proj: { i: 7, j: 0 },
    fc: 0,
    mproj: 0,
    onehot: 0,
    logit: null,
  },
};

/* Lo stato dell'addestramento. Ha iter_num, optimizer e rng: gli stessi
   campi che training_step() si aspetta, come le globali di train.py. */
const ADD = {
  corsa: 0,               // cresce a ogni avvio e azzeramento: ferma le fette vecchie
  in_corso: false,
  iter_num: 0,
  optimizer: null,
  rng: null,
  rngEval: null,
  dati: null,
  storia: [],             // { iter, train, val } a ogni eval_interval
  ultimo: null,           // { loss, lr, norm } dell'ultimo passo
  ms_per_iter: null,
  campione: "",
  errore: false,
};

const GEN = {
  idx: encode(STATO.prompt),
  rng: creaRng(2024),
  nuovi: 0,               // quanti caratteri in fondo a idx sono stati generati
};

/* Il cuore della pagina: un forward() del prompt, con i bersagli spostati
   di un carattere e −1 sull'ultima posizione, che non ha un dopo. Tutti i
   laboratori leggono da qui. */
function ricalcola() {
  const idx = encode(STATO.prompt);
  const targets = idx.slice(1).concat([-1]);
  STATO.uscita = forward(STATO.model, [idx], [targets]);
  disegnaTutto();
}

function disegnaTutto() {
  disegnaStato();
  disegnaEmbedding();
  disegnaSchema();
  disegnaLN();
  disegnaAttenzione();
  disegnaTeste();
  disegnaMLP();
  disegnaTesta();
  disegnaLoss();
  disegnaGenerazione();
}

/* La posizione scelta è una sola per embedding, LayerNorm, MLP e lm_head:
   cambiarla da uno di loro ridisegna tutti e quattro. */
function cambiaPosizione(t) {
  STATO.pos = Number(t);
  STATO.scelte.logit = null;
  disegnaEmbedding();
  disegnaLN();
  disegnaMLP();
  disegnaTesta();
}

/* Le etichette delle posizioni: "0 F", "1 i"… */
function etichettePosizioni() {
  return [...STATO.prompt].map((ch, t) => t + " " + mostra(ch));
}

function vociPosizione() {
  return [...STATO.prompt].map((ch, t) => ({ valore: t, etichetta: t + " " + mostra(ch) }));
}


/* --------------------------------------------------------------------------
   Il prompt: cinque campi, un valore solo
   -------------------------------------------------------------------------- */

const CAMPI_PROMPT = ["prompt-emb", "prompt-schema", "prompt-ln", "prompt-attn", "prompt-mlp", "prompt-testa", "prompt-loss"];

CAMPI_PROMPT.forEach(id => {
  const campo = nodo(id);
  campo.addEventListener("input", () => {
    const pulito = [...campo.value].filter(nelVocabolario).join("").slice(0, config.block_size);
    if (!pulito.length) return;
    STATO.prompt = pulito;
    const T = pulito.length;
    STATO.pos = Math.min(STATO.pos, T - 1);
    STATO.cella = { i: Math.min(STATO.cella.i, T - 1), j: Math.min(STATO.cella.j, T - 1) };
    for (const k of ["qkv", "y", "proj"]) STATO.scelte[k].i = Math.min(STATO.scelte[k].i, T - 1);
    CAMPI_PROMPT.forEach(altro => { if (altro !== id) nodo(altro).value = pulito; });
    GEN.idx = encode(pulito);
    GEN.nuovi = 0;
    ricalcola();
  });
  /* Uscendo dal campo si vede che cosa è stato tenuto: i caratteri fuori
     vocabolario spariscono. */
  campo.addEventListener("change", () => { campo.value = STATO.prompt; });
});


/* --------------------------------------------------------------------------
   Come si leggono i disegni — tre esempi con numeri piccoli
   -------------------------------------------------------------------------- */

const ESEMPIO = {
  a: { shape: [2, 3], data: Float64Array.from([1, 2, 0, 0, 1, 3]) },
  b: { shape: [3, 2], data: Float64Array.from([2, 1, 0, 1, 1, 0]) },
  m: { shape: [2, 3], data: Float64Array.from([1, 2, 3, 4, 5, 6]) },
  sinistra: { shape: [2, 2], data: Float64Array.from([1, -1, 2, 0]) },
  destra: { shape: [2, 2], data: Float64Array.from([3, 1, -2, 4]) },
  teste: [
    { shape: [2, 3], data: Float64Array.from([1, 0, -1, 2, 1, 0]) },
    { shape: [2, 3], data: Float64Array.from([0, 3, 1, -1, 0, 2]) },
  ],
  bias: Float64Array.from([1, -2, 3]),
};

/* Il prodotto di due matrici piccole, solo per l'esempio: nel modello i
   prodotti li fa linear(), qui sopra. */
function prodottoEsempio(a, b) {
  const [R, K] = a.shape, M = b.shape[1];
  const out = tensor([R, M]);
  for (let i = 0; i < R; i++)
    for (let j = 0; j < M; j++)
      for (let k = 0; k < K; k++) out.data[i * M + j] += a.data[i * K + k] * b.data[k * M + j];
  return out;
}

function disegnaLeggenda() {
  const E = ESEMPIO;
  const sel = STATO.scelte.leggenda;
  const out = prodottoEsempio(E.a, E.b);
  const intero = { decimali: 0 };

  riempi(nodo("leggenda-prodotto"), [prodotto({
    chiave: "leggenda", sel,
    a: E.a, oa: { nome: "a", forma: "(n, k)", righe: ["0", "1"], ...intero, dims: [null, "k = 3"] },
    b: E.b, ob: { nome: "b", forma: "(k, m)", ...intero, dims: ["k = 3", null] },
    out, oout: { nome: "a @ b", forma: "(n, m)", ...intero, dims: ["n = 2", "m = 2"] },
    clic: (i, j) => conFocus(() => { STATO.scelte.leggenda = { i, j }; disegnaLeggenda(); }),
  })]);
  nodo("leggenda-lettura").innerHTML = "<b>(a @ b)[" + sel.i + ", " + sel.j + "]</b> = a[" + sel.i + "] · b[:, " + sel.j + "] = " +
    terminiProdotto(d => E.a.data[sel.i * 3 + d], d => E.b.data[d * 2 + sel.j], 3, 0) +
    " = <b>" + fmtNum(out.data[sel.i * 2 + sel.j], 0) + "</b>";

  const affiancate = { shape: [2, 4], data: Float64Array.from([1, -1, 3, 1, 2, 0, -2, 4]) };
  riempi(nodo("leggenda-forma"), [
    tappa(matrice(E.m, { nome: "m", forma: "(2, 3)", righe: ["0", "1"], ...intero, evidenzia: i => i === 0, dims: ["2", "3"] })),
    passaggio(op("forma", "ᵀ", "m.transpose(0, 1)", I18N.t({ it: "le righe diventano colonne", en: "rows become columns" }))),
    tappa(matrice(trasposta(E.m), { nome: "m.T", forma: "(3, 2)", colonne: ["0", "1"], ...intero,
      evidenzia: (i, j) => j === 0, dims: ["3", "2"] })),
    tappa(
      matrice(E.sinistra, { nome: "p", forma: "(2, 2)", ...intero, bande: [1, 1] }),
      matrice(E.destra, { nome: "q", forma: "(2, 2)", ...intero, bande: [2, 2] }),
    ),
    passaggio(op("forma", "‖", "torch.cat([p, q], dim=1)", I18N.t({ it: "fianco a fianco", en: "side by side" }))),
    tappa(matrice(affiancate, { nome: "pq", forma: "(2, 4)", ...intero, bande: [1, 1, 2, 2], dims: [null, "2 + 2 = 4"] })),
  ]);

  const piu = ripetiRiga(E.bias, 2);
  const somma = { shape: [2, 3], data: E.teste[0].data.map((v, n) => v + piu.data[n]) };
  riempi(nodo("leggenda-pila"), [
    tappa(pila(E.teste.map((t, h) => ({ t, nome: I18N.t({ it: "testa ", en: "head " }) + h })), {
      nome: "x", forma: "(nh, 2, 3)", scelta: STATO.scelte.leggenda_testa,
      clic: h => { STATO.scelte.leggenda_testa = h; disegnaLeggenda(); },
      matrice: intero,
    })),
    tappa(
      matrice(E.teste[0], { nome: "x[0]", forma: "(2, 3)", ...intero }),
      operatore("+"),
      matrice(piu, { nome: "b", forma: "(3,) → (2, 3)", ...intero, fantasma: i => i > 0 }),
      operatore("="),
      matrice(somma, { nome: "x[0] + b", forma: "(2, 3)", ...intero }),
    ),
  ]);
}


/* --------------------------------------------------------------------------
   La mappa e il pannello dei pesi
   -------------------------------------------------------------------------- */

function disegnaMappa() {
  const cfg = PRESET[STATO.preset];
  const T = cfg.block_size, C = cfg.n_embd;
  nodo("forma-idx").textContent = "(B, " + T + ")";
  nodo("forma-emb").textContent = "(B, " + T + ", " + C + ")";
  nodo("forma-block").textContent = "(B, " + T + ", " + C + ")";
  nodo("forma-attn").textContent = "att (B, " + cfg.n_head + ", " + T + ", " + T + ")";
  nodo("forma-mlp").textContent = "(B, " + T + ", " + 4 * C + ")";
  nodo("forma-logits").textContent = "(B, " + T + ", " + fmtIntero(cfg.vocab_size) + ")";
  nodo("forma-loss").textContent = "()";

  const conto = conta_parametri(cfg);
  const corpo = nodo("mappa-parametri");
  corpo.textContent = "";
  const riga = (celle, classe) => {
    const tr = document.createElement("tr");
    if (classe) tr.className = classe;
    celle.forEach((c, k) => {
      const td = document.createElement("td");
      if (k === 0) { td.className = "table__name"; td.innerHTML = '<span class="mono">' + esc(c) + "</span>"; }
      else { td.className = "table__num"; td.textContent = c; }
      tr.appendChild(td);
    });
    corpo.appendChild(tr);
  };
  conto.righe.forEach(r => {
    if (r.legato) {
      riga([r.nome, I18N.t({ it: "= wte", en: "= wte" }), "", "0"]);
    } else {
      riga([r.nome, fmtIntero(r.n), r.perBlock ? "× " + cfg.n_layer : "", fmtIntero(r.perBlock ? r.n * cfg.n_layer : r.n)]);
    }
  });
  riga([I18N.t({ it: "totale", en: "total" }), "", "", fmtIntero(conto.totale)], "table__row--total");
  riga(["get_num_params()", "", "− wpe", fmtIntero(conto.non_embedding)], "table__row--sub");

  /* dove stanno i parametri */
  const perBlock = n => conto.righe.find(r => r.nome === n).n * cfg.n_layer;
  const wte = conto.righe[0].n;
  const attn = perBlock("h[i].attn.c_attn") + perBlock("h[i].attn.c_proj");
  const mlp = perBlock("h[i].mlp.c_fc") + perBlock("h[i].mlp.c_proj");
  const pc = x => {
    const v = 100 * x / conto.totale;
    return fmtNum(v, v < 10 ? 1 : 0) + "%";
  };
  const milioni = fmtNum(conto.non_embedding / 1e6, 2) + "M";
  nodo("mappa-verdetto").innerHTML = I18N.t({
    it: "<b>" + (cfg === config ? fmtIntero(conto.non_embedding) : milioni) + " secondo get_num_params().</b> " +
        "wte ne pesa il " + pc(wte) + ", le MLP il " + pc(mlp) + ", l'attenzione il " + pc(attn) + ". " +
        (STATO.preset === "pagina"
          ? "In un modello così piccolo la tabella dei caratteri è una fetta grossa del totale."
          : "Le MLP hanno il doppio dei parametri dell'attenzione in ogni Block: 8·C² contro 4·C²."),
    en: "<b>" + (cfg === config ? fmtIntero(conto.non_embedding) : milioni) + " according to get_num_params().</b> " +
        "wte accounts for " + pc(wte) + ", the MLPs for " + pc(mlp) + ", attention for " + pc(attn) + ". " +
        (STATO.preset === "pagina"
          ? "In a model this small, the character table is a big slice of the total."
          : "In every Block the MLP has twice the parameters of attention: 8·C² against 4·C²."),
  });
}

nodo("config-chip").addEventListener("click", e => {
  const b = e.target.closest("button.chip");
  if (!b) return;
  STATO.preset = b.dataset.preset;
  segnaChip(nodo("config-chip"), "preset", STATO.preset);
  disegnaMappa();
});

function disegnaStato() {
  const ultimoEval = ADD.storia.length ? ADD.storia[ADD.storia.length - 1] : null;
  const passi = ADD.iter_num === 0
    ? I18N.t({ it: "passo 0", en: "step 0" })
    : fmtIntero(ADD.iter_num) + I18N.t({ it: " passi", en: " steps" });
  const nota = ADD.in_corso
    ? I18N.t({ it: "addestramento in corso", en: "training…" })
    : ADD.iter_num === 0
      ? I18N.t({ it: "inizializzazione di nanoGPT", en: "nanoGPT's initialisation" })
      : I18N.t({ it: "pesi addestrati qui", en: "trained right here" });

  riempi(nodo("stato-pesi"), [
    tessera(I18N.t({ it: "Pesi", en: "Weights" }), passi, nota, ADD.iter_num > 0),
    tessera("`val loss`", ultimoEval ? fmtNum(ultimoEval.val, 3) : "—",
      I18N.t({ it: "ln 65 = 4,174: tirare a indovinare", en: "ln 65 = 4.174: pure guessing" })),
    tessera(I18N.t({ it: "Parametri", en: "Parameters" }), fmtIntero(conta_parametri(config).totale),
      "get_num_params() = " + fmtIntero(conta_parametri(config).non_embedding)),
  ]);

  let testo;
  if (ADD.errore) {
    testo = I18N.t({
      it: "<b>Non riesco a scaricare il testo di addestramento.</b> La pagina va servita da un server web: aperta come file, il browser blocca la lettura.",
      en: "<b>Could not download the training text.</b> The page has to be served by a web server: opened as a file, the browser blocks the read.",
    });
  } else if (ADD.in_corso) {
    testo = I18N.t({
      it: "<b>Si sta allenando.</b> Ogni 100 passi tutti i laboratori della pagina si aggiornano con i pesi nuovi.",
      en: "<b>Training.</b> Every 100 steps each lab on this page redraws itself with the new weights.",
    });
  } else if (ADD.iter_num === 0) {
    testo = I18N.t({
      it: "<b>Il modello non sa ancora niente.</b> Ogni carattere ha circa la stessa probabilità, e l'attenzione si divide in parti uguali. Va bene così: è da qui che parte ogni GPT.",
      en: "<b>The model knows nothing yet.</b> Every character is about equally likely, and attention is split evenly. That is fine: this is where every GPT starts.",
    });
  } else {
    testo = I18N.t({
      it: "<b>I laboratori mostrano i pesi addestrati.</b> Torna al passo zero dal laboratorio di addestramento per confrontare.",
      en: "<b>The labs now show the trained weights.</b> Go back to step zero from the training lab to compare.",
    });
  }
  nodo("stato-verdetto").innerHTML = testo;
  etichettaBottoni();
}


/* --------------------------------------------------------------------------
   Parte prima — encode, get_batch, embedding
   -------------------------------------------------------------------------- */

function gettone(ch, id, classe) {
  const t = document.createElement("span");
  t.className = "token" + (classe ? " " + classe : "");
  const c = document.createElement("span");
  c.className = "token__ch" + (eSpazio(ch) ? " is-ws" : "");
  c.textContent = mostra(ch);
  const n = document.createElement("span");
  n.className = "token__id";
  n.textContent = id === undefined ? "?" : String(id);
  t.append(c, n);
  t.title = JSON.stringify(ch) + " → " + (id === undefined ? "KeyError" : id);
  return t;
}

function disegnaTesto() {
  const s = [...nodo("testo").value].slice(0, 80);
  const ids = s.map(ch => stoi[ch]);
  riempi(nodo("gettoni"), s.map((ch, k) =>
    gettone(ch, ids[k], nelVocabolario(ch) ? "" : "is-unknown")));

  const fuori = s.filter(ch => !nelVocabolario(ch));
  const noti = ids.filter(i => i !== undefined);
  nodo("decodifica").innerHTML =
    "<b>encode(s)</b> = [" + noti.join(", ") + "]<br>" +
    "<b>decode(encode(s))</b> = " + esc(JSON.stringify(decode(noti))) +
    (fuori.length
      ? "<br>" + I18N.t({
          it: "Fuori vocabolario, in Python KeyError: ",
          en: "Out of vocabulary, a KeyError in Python: ",
        }) + esc(fuori.map(c => JSON.stringify(c)).join(" "))
      : "");
}

nodo("testo").addEventListener("input", disegnaTesto);

/* L'inizio di Tiny Shakespeare, per il laboratorio di get_batch. */
const ESTRATTO =
  "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n" +
  "First Citizen:\nYou are all resolved rather to die than to famish?\n\nAll:\nResolved. resolved.\n\n" +
  "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n";

function disegnaBatch() {
  const data = encode(ESTRATTO);
  const i = Number(nodo("batch-i").value);
  nodo("batch-i-out").textContent = String(i);
  const block_size = config.block_size;
  const x = data.slice(i, i + block_size);
  const y = data.slice(i + 1, i + 1 + block_size);

  riempi(nodo("batch-x"), x.map(id => gettone(itos[id], id, "is-ctx")));
  riempi(nodo("batch-y"), y.map(id => gettone(itos[id], id, "is-target")));

  const corpo = nodo("batch-coppie");
  corpo.textContent = "";
  for (let t = 0; t < block_size; t++) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      '<td class="table__num">' + t + "</td>" +
      '<td><span class="mono">' + esc(x.slice(0, t + 1).map(id => mostra(itos[id])).join("")) + "</span></td>" +
      '<td><span class="mono table__pick">' + esc(mostra(itos[y[t]])) + "</span></td>";
    corpo.appendChild(tr);
  }
}

nodo("batch-i").max = String(ESTRATTO.length - config.block_size - 1);
nodo("batch-i").addEventListener("input", disegnaBatch);

function disegnaEmbedding() {
  const c = STATO.uscita.cache;
  const T = c.t, C = config.n_embd, V = config.vocab_size;
  const righe = etichettePosizioni();
  const p = STATO.pos;
  const idx = c.idx[0];
  const wte = STATO.model.transformer.wte.weight;
  const wpe = STATO.model.transformer.wpe.weight;
  const letta = new Set(idx);

  aggiornaChip(nodo("emb-pos"), vociPosizione(), p);

  /* Due letture di righe in parallelo, poi una somma cella per cella. */
  const colonna = valori => ({ shape: [valori.length, 1], data: Float64Array.from(valori) });
  const pos = Array.from({ length: T }, (_, t) => t);
  riempi(nodo("emb-tensori"), [
    tappa(
      matrice(colonna(idx), { nome: "idx", forma: "(B, T)", righe, neutra: true, decimali: 0,
        rigaScelta: p, scelta: { i: p, j: 0 }, dims: ["T = " + T, null] }),
      matrice(wte, { nome: "wte.weight", forma: "(V, C)", dimensione: "fine",
        attenua: i => !letta.has(i), evidenzia: i => i === idx[p], dims: ["V = " + V, "C = " + C] }),
      matrice(colonna(pos), { nome: "pos", forma: "(T,)", righe: pos.map(String), neutra: true, decimali: 0,
        rigaScelta: p, scelta: { i: p, j: 0 }, dims: ["T = " + T, null] }),
      matrice(wpe, { nome: "wpe.weight", forma: "(block_size, C)", dimensione: "compatta",
        righe: Array.from({ length: config.block_size }, (_, t) => String(t)),
        attenua: i => i >= T, evidenzia: i => i === p, dims: ["block_size = " + config.block_size, "C = " + C] }),
    ),
    passaggio(
      op("forma", "[i]", "tok_emb = self.transformer.wte(idx)",
        I18N.t({ it: "la riga idx[t] di wte, per ogni t", en: "row idx[t] of wte, for every t" })),
      op("forma", "[i]", "pos_emb = self.transformer.wpe(pos)",
        I18N.t({ it: "la riga t di wpe", en: "row t of wpe" })),
    ),
    tappa(
      matrice(c.tok_emb, { nome: "tok_emb", forma: "(B, T, C)", compatta: true, righe, evidenzia: i => i === p, rigaScelta: p, dims: ["T = " + T, "C = " + C] }),
      operatore("+"),
      matrice(c.pos_emb, { nome: "pos_emb", forma: "(T, C)", compatta: true, righe: pos.map(String), evidenzia: i => i === p, rigaScelta: p, dims: ["T = " + T, "C = " + C] }),
      operatore("="),
      matrice(c.x0, { nome: "x", forma: "(B, T, C)", compatta: true, righe, evidenzia: i => i === p, rigaScelta: p, dims: ["T = " + T, "C = " + C] }),
    ),
    passaggio(op("elemento", "+", "x = self.transformer.drop(tok_emb + pos_emb)",
      I18N.t({ it: "pos_emb (T, C) vale per ogni elemento del batch: il broadcast aggiunge B; drop ha p = 0",
               en: "pos_emb (T, C) applies to every batch element: broadcasting adds B; drop has p = 0" }))),
  ]);

  /* La riga scelta, con i numeri. Le tre righe condividono la scala, così
     i colori si confrontano. */
  const id = idx[p];
  const tok = ritaglia(c.tok_emb, p, p + 1, 0, C);
  const rigaPos = ritaglia(c.pos_emb, p, p + 1, 0, C);
  const x = ritaglia(c.x0, p, p + 1, 0, C);
  let scala = 0;
  for (const v of [...tok.data, ...rigaPos.data, ...x.data]) scala = Math.max(scala, Math.abs(v));
  riempi(nodo("emb-zoom"), [
    matrice(tok, { nome: "wte.weight[" + id + "]", forma: "(" + C + ",)  " + mostra(itos[id]), scala }),
    operatore("+"),
    matrice(rigaPos, { nome: "wpe.weight[" + p + "]", forma: "(" + C + ",)", scala }),
    operatore("="),
    matrice(x, { nome: "x[0, " + p + "]", forma: "(" + C + ",)", scala }),
  ]);

  disegnaUnoCaldo();
}

/* Il riquadro: la stessa lettura come prodotto per un one-hot. Sono 1.700
   celle in un riquadro quasi sempre chiuso: si disegna solo quando è aperto. */
function disegnaUnoCaldo() {
  if (!nodo("emb-onehot").closest("details").open) return;
  const c = STATO.uscita.cache;
  const T = c.t, C = config.n_embd, V = config.vocab_size;
  const p = STATO.pos;
  const idx = c.idx[0];
  const id = idx[p];
  const wte = STATO.model.transformer.wte.weight;
  const j = STATO.scelte.onehot;
  const unoCaldo = tensor([T, V]);
  idx.forEach((v, t) => { unoCaldo.data[t * V + v] = 1; });
  riempi(nodo("emb-onehot"), [prodotto({
    dimensione: "fine", chiave: "onehot", sel: { i: p, j },
    op: op("prodotto", "@", "one_hot(idx) @ wte.weight",
      I18N.t({ it: "= wte(idx), con 64 prodotti per zero in ogni cella", en: "= wte(idx), with 64 products by zero in each cell" })),
    a: unoCaldo, oa: { nome: "one_hot(idx)", forma: "(T, V)", neutra: false, scala: 1, dims: [null, "V = " + V] },
    b: wte, ob: { nome: "wte.weight", forma: "(V, C)", scelta: { i: id, j }, dims: ["V = " + V, null] },
    out: c.tok_emb, oout: { nome: "tok_emb", forma: "(T, C)", dims: ["T = " + T, "C = " + C] },
    clic: (i, jj) => conFocus(() => { STATO.scelte.onehot = jj; cambiaPosizione(i); }),
  })]);
  nodo("emb-onehot-lettura").innerHTML =
    "<b>tok_emb[" + p + ", " + j + "]</b> = Σ<sub>v</sub> one_hot[" + p + ", v] · wte[v, " + j + "] = 1 · wte[" + id + ", " + j +
    "] = <b>" + fmtNum(wte.data[id * C + j], 4) + "</b>" +
    I18N.t({
      it: " — l'unico 1 della riga " + p + " sta nella colonna " + id + " ('" + esc(mostra(itos[id])) + "'), e gli altri 64 termini valgono zero.",
      en: " — the only 1 in row " + p + " is in column " + id + " ('" + esc(mostra(itos[id])) + "'), and the other 64 terms are zero.",
    });
}

nodo("emb-onehot").closest("details").addEventListener("toggle", disegnaUnoCaldo);

preparaChip(nodo("emb-pos"), v => conFocus(() => cambiaPosizione(v)));


/* --------------------------------------------------------------------------
   Parte seconda — lo schema del Block
   -------------------------------------------------------------------------- */

/* Lo schema è scritto nella pagina, operazione per operazione; qui si
   riempiono solo le miniature: ogni [data-tensore] riceve il tensore che
   quell'operazione produce, per il Block scelto. Le forme si leggono dalle
   proporzioni: qkv è tre volte più largo di x, att è un triangolo per
   testa, lo strato nascosto dell'MLP è quattro volte più largo. */
function disegnaSchema() {
  const c = STATO.uscita.cache;
  const l = STATO.schema_strato;
  const b = c.blocks[l], A = b.attn;
  const T = c.t, C = config.n_embd;
  const futura = (i, j) => j > i;
  const scalate = A.att_raw[0].map(t => ({ shape: [T, T], data: t.data.map(z => z * A.scala) }));
  const mazzo = (lista, o = {}) => pila(lista.map(t => ({ t })), { matrice: { dimensione: "mini", ...o } });

  aggiornaChip(nodo("schema-strato"), [0, 1].map(v => ({ valore: v, etichetta: "h." + v })), l);

  const miniature = {
    x_in: () => miniatura(b.x_in),
    ln_1: () => miniatura(b.ln_1.out),
    qkv: () => miniatura(A.qkv),
    split: () => [0, 1, 2].map(n => miniatura(ritaglia(A.qkv, 0, T, n * C, (n + 1) * C))),
    teste: () => mazzo(A.q[0]),
    att_raw: () => mazzo(A.att_raw[0]),
    att_scala: () => mazzo(scalate),
    att_maschera: () => mazzo(scalate, { maschera: futura }),
    att: () => mazzo(A.att[0], { maschera: futura }),
    y_teste: () => mazzo(A.y_h[0]),
    y: () => miniatura(A.y),
    attn_out: () => miniatura(A.out),
    x_mid: () => miniatura(b.x_mid),
    ln_2: () => miniatura(b.ln_2.out),
    h_fc: () => miniatura(b.mlp.h_fc),
    h_gelu: () => miniatura(b.mlp.h_gelu),
    mlp_out: () => miniatura(b.mlp.out),
    x_out: () => miniatura(b.x_out),
  };

  document.querySelectorAll("#schema-block [data-tensore]").forEach(el => {
    const f = miniature[el.dataset.tensore];
    if (!f) return;
    const r = f();
    riempi(el, Array.isArray(r) ? r : [r]);
  });
}

preparaChip(nodo("schema-strato"), v => { STATO.schema_strato = Number(v); disegnaSchema(); });

/* Ogni operazione dello schema è un link alla sezione che la spiega. Prima
   che il browser ci arrivi, il laboratorio di destinazione viene messo sul
   Block e sul passo giusti. */
nodo("schema-block").addEventListener("click", e => {
  const a = e.target.closest("a");
  if (!a) return;
  const l = STATO.schema_strato;
  if (a.dataset.ln) {
    STATO.ln = "h." + l + "." + a.dataset.ln;
    disegnaLN();
  }
  if (a.dataset.passo) {
    STATO.strato = l;
    vaiAlPasso(Number(a.dataset.passo));
  }
  if (a.dataset.mlp) {
    STATO.mlp_strato = l;
    STATO.mlp_passo = Number(a.dataset.mlp);
    disegnaMLP();
  }
});


/* --------------------------------------------------------------------------
   Parte seconda — LayerNorm
   -------------------------------------------------------------------------- */

/* Quale LayerNorm: ingresso, uscita e parametri presi dalla cache. */
function sceltaLN() {
  const c = STATO.uscita.cache;
  if (STATO.ln === "ln_f") return { x: c.x_final, ln: c.ln_f, mod: STATO.model.transformer.ln_f };
  const l = Number(STATO.ln[2]);
  const qual = STATO.ln.slice(4);
  const b = c.blocks[l];
  return qual === "ln_1"
    ? { x: b.x_in, ln: b.ln_1, mod: STATO.model.transformer.h[l].ln_1 }
    : { x: b.x_mid, ln: b.ln_2, mod: STATO.model.transformer.h[l].ln_2 };
}

/* Barre verticali di un vettore, con lo zero a metà altezza. */
function barre(g, valori, x0, y0, larghezza, altezza, scala, classi) {
  const n = valori.length, passo = larghezza / n, mezzo = y0 + altezza / 2;
  g.appendChild(svgEl("line", { x1: x0, y1: mezzo, x2: x0 + larghezza, y2: mezzo, class: "ax" }));
  for (let k = 0; k < n; k++) {
    const v = valori[k];
    const h = Math.min(1, Math.abs(v) / scala) * (altezza / 2);
    g.appendChild(svgEl("rect", {
      x: (x0 + k * passo + passo * (classi.margine || 0.15)).toFixed(1),
      y: (v >= 0 ? mezzo - h : mezzo).toFixed(1),
      width: (passo * (1 - 2 * (classi.margine || 0.15))).toFixed(1),
      height: Math.max(0.5, h).toFixed(1),
      class: v >= 0 ? classi.pos : classi.neg,
    }));
  }
}

/* Il calcolo intero, su tutte le T righe: le statistiche di ogni riga,
   poi le due operazioni cella per cella, con quello che il broadcast
   ripete disegnato in trasparenza. La riga scelta ha il filo sottile. */
function disegnaFlussoLN(x, ln, mod, p) {
  const T = x.shape[0], C = config.n_embd;
  const righe = etichettePosizioni();
  const nome = STATO.ln;
  const mu = Float64Array.from(ln.mean);
  const sigma = Float64Array.from(ln.rstd, r => 1 / r);
  const xhat = tensor([T, C]);
  for (let i = 0; i < T; i++)
    for (let j = 0; j < C; j++) xhat.data[i * C + j] = (x.data[i * C + j] - mu[i]) * ln.rstd[i];

  const riga = i => i === p;
  const dimsTC = ["T = " + T, "C = " + C];
  const piena = (t, n, extra = {}) => matrice(t, { nome: n, forma: "(T, C)", compatta: true, righe,
    evidenzia: riga, rigaScelta: p, dims: dimsTC, ...extra });
  const colonna = (v, n) => matrice({ shape: [T, 1], data: v }, { nome: n, forma: "(T, 1)", dimensione: "larga",
    righe, decimali: 4, evidenzia: riga, rigaScelta: p, dims: ["T = " + T, "1"] });
  const perRiga = (v, n) => piena(ripetiColonna(v, C), n, { fantasma: (i, j) => j > 0, forma: "(T, 1) → (T, C)" });
  const perColonna = (v, n) => piena(ripetiRiga(v, T), n, { fantasma: i => i > 0, forma: "(C,) → (T, C)" });

  const fila = [
    tappa(piena(x, "x")),
    passaggio(op("riga", "μσ", "F.layer_norm(x, …, eps=1e-5)",
      I18N.t({ it: "per ogni riga, la media e la deviazione dei suoi C numeri", en: "for each row, the mean and deviation of its C numbers" }))),
    tappa(colonna(mu, "μ"), colonna(sigma, "√(σ² + ε)")),
    passaggio(op("elemento", "−÷", "x̂ = (x − μ) / √(σ² + ε)",
      I18N.t({ it: "μ e √(σ² + ε) valgono per tutta la riga: il broadcast li ripete su ogni colonna",
               en: "μ and √(σ² + ε) hold for the whole row: broadcasting repeats them on every column" }))),
    tappa(piena(x, "x"), operatore("−"), perRiga(mu, "μ"), operatore("÷"), perRiga(sigma, "√(σ² + ε)"),
      operatore("="), piena(xhat, "x̂")),
    passaggio(op("elemento", "⊙+", "x̂ * " + nome + ".weight + " + nome + ".bias",
      I18N.t({ it: "weight e bias valgono per ogni posizione: il broadcast li ripete su ogni riga",
               en: "weight and bias hold for every position: broadcasting repeats them on every row" }))),
    tappa(piena(xhat, "x̂"), operatore("⊙"), perColonna(mod.weight.data, nome + ".weight"),
      operatore("+"), perColonna(mod.bias ? mod.bias.data : new Float64Array(C), nome + ".bias"),
      operatore("="), piena(ln.out, nome + "(x)")),
  ];
  riempi(nodo("ln-diagramma"), fila);
}

function disegnaLN() {
  const C = config.n_embd;
  const scelte = ["h.0.ln_1", "h.0.ln_2", "h.1.ln_1", "h.1.ln_2", "ln_f"];
  aggiornaChip(nodo("ln-scelta"), scelte.map(s => ({ valore: s, etichetta: s })), STATO.ln);
  aggiornaChip(nodo("ln-pos"), vociPosizione(), STATO.pos);

  const { x, ln, mod } = sceltaLN();
  const p = STATO.pos;
  disegnaFlussoLN(x, ln, mod, p);
  const ingresso = Array.from(x.data.subarray(p * C, (p + 1) * C));
  const uscita = Array.from(ln.out.data.subarray(p * C, (p + 1) * C));
  const mu = ln.mean[p], rstd = ln.rstd[p];
  const varianza = 1 / (rstd * rstd) - 1e-5;
  const xhat = ingresso.map(v => (v - mu) * rstd);
  const varXhat = xhat.reduce((s, v) => s + v * v, 0) / C;

  const g = nodo("ln-grafico-g");
  g.textContent = "";
  const maxIn = Math.max(...ingresso.map(Math.abs)) || 1;
  const maxOut = Math.max(...uscita.map(Math.abs)) || 1;
  g.appendChild(svgEl("text", { x: 34, y: 14, class: "axlab" },
    I18N.t({ it: "ingresso x — scala ±", en: "input x — scale ±" }) + fmtNum(maxIn, 3)));
  barre(g, ingresso, 34, 22, 378, 112, maxIn, { pos: "vbar", neg: "vbar vbar--neg" });
  /* la media, che la LayerNorm toglie */
  const yMu = 22 + 56 - Math.max(-1, Math.min(1, mu / maxIn)) * 56;
  g.appendChild(svgEl("line", { x1: 34, y1: yMu.toFixed(1), x2: 412, y2: yMu.toFixed(1), class: "markline" }));
  g.appendChild(svgEl("text", { x: 2, y: (yMu + 3).toFixed(1), class: "axlab" }, "μ"));

  g.appendChild(svgEl("text", { x: 34, y: 166, class: "axlab" },
    I18N.t({ it: "uscita ln(x) — scala ±", en: "output ln(x) — scale ±" }) + fmtNum(maxOut, 2)));
  barre(g, uscita, 34, 174, 378, 112, maxOut, { pos: "vbar", neg: "vbar vbar--neg" });

  const peso = Array.from(mod.weight.data);
  const pesoMedio = peso.reduce((s, v) => s + v, 0) / C;
  const rapporto = 1e-5 / Math.max(varianza, 1e-30);

  riempi(nodo("ln-letture"), [
    tessera("`μ`", fmtNum(mu, 4), I18N.t({ it: "media dei 16 numeri", en: "mean of the 16 numbers" })),
    tessera("`σ²`", fmtNum(varianza, 5), I18N.t({ it: "varianza, divisa per C", en: "variance, divided by C" })),
    tessera("`ε / σ²`", fmtNum(100 * rapporto, 2) + "%", I18N.t({ it: "quanto pesa ε = 10⁻⁵", en: "how much ε = 10⁻⁵ weighs" })),
    tessera(I18N.t({ it: "Varianza di `x̂`", en: "Variance of `x̂`" }), fmtNum(varXhat, 3), I18N.t({ it: "prima di weight e bias", en: "before weight and bias" })),
    tessera(I18N.t({ it: "`weight` medio", en: "Mean `weight`" }), fmtNum(pesoMedio, 3), I18N.t({ it: "1 all'inizio", en: "1 at init" })),
  ]);

  nodo("ln-verdetto").innerHTML = rapporto > 0.005
    ? I18N.t({
        it: "<b>ε non è trascurabile.</b> I numeri in ingresso sono piccoli, σ² = " + fmtNum(varianza, 5) +
            ", e il 10⁻⁵ sotto la radice ne vale il " + fmtNum(100 * rapporto, 1) + "%: x̂ esce con varianza " +
            fmtNum(varXhat, 3) + " invece di 1. Succede al passo zero, dove gli embedding hanno deviazione 0,02.",
        en: "<b>ε is not negligible.</b> The incoming numbers are small, σ² = " + fmtNum(varianza, 5) +
            ", and the 10⁻⁵ under the root is " + fmtNum(100 * rapporto, 1) + "% of it: x̂ comes out with variance " +
            fmtNum(varXhat, 3) + " instead of 1. This happens at step zero, where embeddings have standard deviation 0.02.",
      })
    : I18N.t({
        it: "<b>Media zero, varianza uno, poi la scala imparata.</b> x̂ ha varianza " + fmtNum(varXhat, 3) +
            "; l'uscita no, perché weight e bias — " + fmtNum(pesoMedio, 2) + " di media, qui — la rimettono dove serve al " +
            "sotto-blocco che viene dopo.",
        en: "<b>Zero mean, unit variance, then the learned scale.</b> x̂ has variance " + fmtNum(varXhat, 3) +
            "; the output does not, because weight and bias — averaging " + fmtNum(pesoMedio, 2) + " here — put it back " +
            "where the next sub-block needs it.",
      });
}

preparaChip(nodo("ln-scelta"), v => { STATO.ln = v; disegnaLN(); });
preparaChip(nodo("ln-pos"), v => conFocus(() => cambiaPosizione(v)));


/* --------------------------------------------------------------------------
   Parte seconda — l'attenzione, passo per passo
   -------------------------------------------------------------------------- */

const N_PASSI = 10;

/* La maschera di dropout dell'anteprima del passo 7: fissa per Block e
   testa, così premere di nuovo p = 0,2 mostra la stessa. */
function mascheraDropout(T, p) {
  const rng = creaRng(500 + STATO.strato * 10 + STATO.testa);
  return Array.from({ length: T * T }, () => (rng() < p ? 0 : 1));
}

function disegnaAttenzione() {
  const c = STATO.uscita.cache;
  const T = c.t, C = config.n_embd, nh = config.n_head, hs = C / nh;
  const l = STATO.strato, h = STATO.testa, p = STATO.passo;
  const A = c.blocks[l].attn;
  const blocco = STATO.model.transformer.h[l].attn;
  const righe = etichettePosizioni();
  const sel = STATO.cella;
  const scelte = STATO.scelte;

  aggiornaChip(nodo("attn-strato"), [0, 1].map(v => ({ valore: v, etichetta: "h." + v })), l);
  aggiornaChip(nodo("attn-testa"), [0, 1].map(v => ({
    valore: v, etichetta: I18N.t({ it: "testa ", en: "head " }) + v })), h);
  segnaChip(nodo("attn-passi"), "passo", p);
  nodo("passo-prec").disabled = p === 1;
  nodo("passo-succ").disabled = p === N_PASSI;

  /* il testo del passo e la riga del listato */
  let righeCodice = [];
  document.querySelectorAll("#attn-passi-testo .step").forEach(s => {
    const attivo = Number(s.dataset.passo) === p;
    s.hidden = !attivo;
    if (attivo) {
      const [da, a] = s.dataset.righe.split("-").map(Number);
      for (let r = da; r <= (a || da); r++) righeCodice.push(r);
    }
  });
  document.querySelectorAll("#listato-attn .listing__line").forEach(r =>
    r.classList.toggle("is-current", righeCodice.includes(Number(r.dataset.riga))));

  segnaChip(nodo("dropout-chip"), "p", String(STATO.dropout_p));

  const q = A.q[0][h], k = A.k[0][h], v = A.v[0][h];
  const raw = A.att_raw[0][h], att = A.att[0][h], yh = A.y_h[0][h];
  const futura = (i, j) => j > i;
  const scala = A.scala;
  const scalata = { shape: [T, T], data: raw.data.map(z => z * scala) };
  const ridisegna = () => conFocus(disegnaAttenzione);
  const cliccaCella = (i, j) => { STATO.cella = { i, j }; ridisegna(); };
  const sceltaTesta = t => { STATO.testa = t; disegnaAttenzione(); };
  const nomeQ = "q[0, " + h + "]", nomeK = "k[0, " + h + "]", nomeV = "v[0, " + h + "]";
  const bandeTeste = Array.from({ length: C }, (_, j) => 1 + Math.floor(j / hs));
  const tagliTeste = Array.from({ length: nh - 1 }, (_, t) => (t + 1) * hs);
  const dimT = "T = " + T, dimC = "C = " + C, dimHs = "hs = " + hs;
  const testaDaVicino = didascalia(I18N.t({ it: "La testa " + h + ", da vicino", en: "Head " + h + ", up close" }));

  /* Un mazzo (B, nh, …) con la testa scelta davanti, che si può cambiare
     cliccando la carta dietro. */
  const mazzo = (lista, nome, forma, o = {}) => pila(carte(lista, nome), {
    nome, forma, scelta: h, clic: sceltaTesta,
    matrice: { dimensione: "compatta", righe, ...o },
  });
  /* att T × T con le etichette delle posizioni su righe e colonne */
  const quadrata = (t, nome, o = {}) => matrice(t, {
    nome, forma: "(T, T)", righe, colonne: righe, scelta: sel, clic: cliccaCella, ...o,
  });

  const fila = [];
  let dettaglio = "";

  if (p === 1) {
    const s = scelte.qkv;
    s.i = Math.min(s.i, T - 1);
    const x = c.blocks[l].ln_1.out;
    const bande = Array.from({ length: 3 * C }, (_, j) => 1 + Math.floor(j / C));
    fila.push(
      tappa(prodotto({
        dimensione: "compatta", chiave: "qkv", sel: s,
        op: op("prodotto", "@", "self.c_attn(x)", "x @ c_attn.weight.T + c_attn.bias"),
        a: x, oa: { nome: "ln_1(x)", forma: "(T, C)", righe, dims: [null, dimC] },
        b: trasposta(blocco.c_attn.weight), ob: { nome: "c_attn.weight.T", forma: "(C, 3C)", bande, dims: [dimC, null] },
        bias: blocco.c_attn.bias, obias: { nome: "+ c_attn.bias", forma: "(3C,)", bande },
        out: A.qkv, oout: { nome: "qkv", forma: "(T, 3C)", dims: [dimT, "3C = " + 3 * C] },
        clic: (i, j) => { scelte.qkv = { i, j }; ridisegna(); },
      })),
      passaggio(op("forma", "÷3", ".split(self.n_embd, dim=2)", "(B, T, 3C) → 3 × (B, T, C)")),
      tappa(...["q", "k", "v"].map((nome, n) => matrice(ritaglia(A.qkv, 0, T, n * C, (n + 1) * C), {
        nome, forma: "(B, T, C)", compatta: true, righe: n === 0 ? righe : null,
        bande: Array(C).fill(n + 1), dims: [n === 2 ? dimT : null, dimC],
        scelta: Math.floor(s.j / C) === n ? { i: s.i, j: s.j - n * C } : null,
      }))),
    );
    const parte = ["q", "k", "v"][Math.floor(s.j / C)];
    const W = blocco.c_attn.weight.data, b = blocco.c_attn.bias.data;
    dettaglio = "<b>qkv[" + s.i + ", " + s.j + "]</b> = ln_1(x)[" + s.i + "] · c_attn.weight[" + s.j + "] + c_attn.bias[" + s.j + "] = " +
      terminiProdotto(d => x.data[s.i * C + d], d => W[s.j * C + d], C) + " + " + fmtNum(b[s.j]) +
      " = <b>" + fmtNum(A.qkv.data[s.i * 3 * C + s.j], 4) + "</b>" +
      I18N.t({
        it: " — finisce in <b>" + parte + "</b>, colonna " + (s.j % C) + ", cioè nella testa " + Math.floor((s.j % C) / hs) + ".",
        en: " — it ends up in <b>" + parte + "</b>, column " + (s.j % C) + ", that is in head " + Math.floor((s.j % C) / hs) + ".",
      });
  }

  if (p === 2) {
    const qPiena = ritaglia(A.qkv, 0, T, 0, C);
    fila.push(
      tappa(matrice(qPiena, { nome: "q", forma: "(B, T, C)", compatta: true, righe, bande: bandeTeste, dims: [dimT, dimC] })),
      passaggio(op("forma", "⊞", "q.view(B, T, self.n_head, C // self.n_head)", "(B, T, C) → (B, T, nh, hs)")),
      tappa(matrice(qPiena, { nome: "q", forma: "(B, T, nh, hs)", compatta: true, righe, bande: bandeTeste,
        fessure: tagliTeste, dims: [dimT, Array(nh).fill(dimHs)] })),
      passaggio(op("forma", "ᵀ", ".transpose(1, 2)", "(B, T, nh, hs) → (B, nh, T, hs)")),
      tappa(
        mazzo(A.q[0], "q", "(B, nh, T, hs)", { dims: [dimT, dimHs] }),
        mazzo(A.k[0], "k", "(B, nh, T, hs)", { righe: null, dims: [dimT, dimHs] }),
        mazzo(A.v[0], "v", "(B, nh, T, hs)", { righe: null, dims: [dimT, dimHs] }),
      ),
    );
    dettaglio = I18N.t({
      it: "Da <b>(1, " + T + ", " + C + ")</b> a <b>(1, " + nh + ", " + T + ", " + hs + ")</b>. Le colonne 0–" + (hs - 1) +
          " di q sono la testa 0, le colonne " + hs + "–" + (C - 1) + " la testa 1: nessun numero è cambiato, solo il modo di leggerli. " +
          "Le teste diventano carte, una dietro l'altra: clicca quella dietro per portarla davanti.",
      en: "From <b>(1, " + T + ", " + C + ")</b> to <b>(1, " + nh + ", " + T + ", " + hs + ")</b>. Columns 0–" + (hs - 1) +
          " of q are head 0, columns " + hs + "–" + (C - 1) + " head 1: no number has changed, only the way they are read. " +
          "The heads become cards, one behind the other: click the one behind to bring it forward.",
    });
  }

  if (p === 3) {
    fila.push(
      tappa(
        mazzo(A.q[0], "q", "(B, nh, T, hs)"),
        operatore("@"),
        mazzo(A.k[0].map(trasposta), "k.T", "(B, nh, hs, T)", { righe: null }),
        operatore("="),
        mazzo(A.att_raw[0], "att", "(B, nh, T, T)"),
      ),
      testaDaVicino,
      tappa(
        matrice(k, { nome: nomeK, forma: "(T, hs)", righe, rigaScelta: sel.j, evidenzia: i => i === sel.j, dims: [dimT, dimHs] }),
        op("forma", "ᵀ", "k.transpose(-2, -1)",
          I18N.t({ it: "la riga j di k diventa la colonna j", en: "row j of k becomes column j" })),
        matrice(trasposta(k), { nome: nomeK + ".T", forma: "(hs, T)", colonne: righe, colonnaScelta: sel.j,
          evidenzia: (i, j) => j === sel.j, dims: [dimHs, dimT] }),
      ),
      tappa(prodotto({
        chiave: "att", sel,
        op: op("prodotto", "@", "q @ k.transpose(-2, -1)", "(T, hs) @ (hs, T) → (T, T)"),
        a: q, oa: { nome: nomeQ, forma: "(T, hs)", righe, dims: [null, dimHs] },
        b: trasposta(k), ob: { nome: nomeK + ".T", forma: "(hs, T)", colonne: righe, dims: [dimHs, null] },
        out: raw, oout: { nome: "att", forma: "(T, T)", dims: [dimT, dimT] },
        clic: cliccaCella,
      })),
    );
    dettaglio = "<b>att[" + sel.i + ", " + sel.j + "]</b> = q[" + sel.i + "] · k[" + sel.j + "] = " +
      terminiProdotto(d => q.data[sel.i * hs + d], d => k.data[sel.j * hs + d], hs) +
      " = <b>" + fmtNum(raw.data[sel.i * T + sel.j], 4) + "</b>" +
      (sel.j > sel.i
        ? I18N.t({ it: " — una posizione futura: la maschera del passo 5 la cancellerà.",
                   en: " — a future position: the mask in step 5 will erase it." })
        : "");
  }

  if (p === 4) {
    fila.push(
      tappa(quadrata(raw, "q @ k.T", { dims: [dimT, dimT] })),
      passaggio(op("elemento", "×", "* (1.0 / math.sqrt(k.size(-1)))", "× 1/√" + hs + " = × " + fmtNum(scala, 3))),
      tappa(quadrata(scalata, "att", { dims: [dimT, dimT] })),
    );
    dettaglio = "<b>att[" + sel.i + ", " + sel.j + "]</b> = " + fmtNum(raw.data[sel.i * T + sel.j], 4) +
      " × 1/√" + hs + " = " + fmtNum(raw.data[sel.i * T + sel.j], 4) + " × " + fmtNum(scala, 4) +
      " = <b>" + fmtNum(scalata.data[sel.i * T + sel.j], 4) + "</b>";
  }

  if (p === 5) {
    const tril = { shape: [T, T], data: Float64Array.from({ length: T * T }, (_, n) => (n % T <= Math.floor(n / T) ? 1 : 0)) };
    fila.push(
      tappa(
        quadrata(scalata, "att"),
        matrice(tril, { nome: "self.bias[:, :, :T, :T]", forma: "(1, 1, T, T)", righe, colonne: righe,
          decimali: 0, scala: 1, scelta: sel }),
      ),
      passaggio(op("elemento", "−∞", "att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))",
        I18N.t({ it: "dove bias vale 0, cioè sopra la diagonale", en: "where bias is 0, that is above the diagonal" }))),
      tappa(quadrata(scalata, "att", { maschera: futura, dims: [dimT, dimT] })),
    );
    dettaglio = sel.j > sel.i
      ? I18N.t({
          it: "<b>att[" + sel.i + ", " + sel.j + "] = −∞</b>: la posizione " + sel.j + " viene dopo la " + sel.i +
              ", e bias[" + sel.i + ", " + sel.j + "] = 0.",
          en: "<b>att[" + sel.i + ", " + sel.j + "] = −∞</b>: position " + sel.j + " comes after " + sel.i +
              ", and bias[" + sel.i + ", " + sel.j + "] = 0.",
        })
      : I18N.t({
          it: "<b>att[" + sel.i + ", " + sel.j + "] = " + fmtNum(scalata.data[sel.i * T + sel.j], 4) + "</b> resta: bias[" +
              sel.i + ", " + sel.j + "] = 1. Ogni riga " + "<i>i</i> tiene le colonne da 0 a <i>i</i>: la prima riga vede solo sé stessa.",
          en: "<b>att[" + sel.i + ", " + sel.j + "] = " + fmtNum(scalata.data[sel.i * T + sel.j], 4) + "</b> stays: bias[" +
              sel.i + ", " + sel.j + "] = 1. Row <i>i</i> keeps columns 0 to <i>i</i>: the first row sees only itself.",
        });
  }

  if (p === 6) {
    const espo = { shape: [T, T], data: scalata.data.map((z, n) => (n % T <= Math.floor(n / T) ? Math.exp(z) : 0)) };
    const somme = Array.from({ length: T }, (_, i) => {
      let s = 0;
      for (let j = 0; j < T; j++) s += espo.data[i * T + j];
      return s;
    });
    const sommeAtt = Array.from({ length: T }, (_, i) => {
      let s = 0;
      for (let j = 0; j < T; j++) s += att.data[i * T + j];
      return "Σ " + fmtNum(s);
    });
    fila.push(
      tappa(quadrata(scalata, "att", { maschera: futura, rigaScelta: sel.i, evidenzia: i => i === sel.i })),
      passaggio(op("elemento", "eˣ", "att.exp()",
        I18N.t({ it: "dentro F.softmax: e elevato a ogni cella, e^−∞ = 0", en: "inside F.softmax: e to the power of each cell, e^−∞ = 0" }))),
      tappa(quadrata(espo, "exp(att)", { maschera: futura, testoMaschera: "0", rigaScelta: sel.i,
        evidenzia: i => i === sel.i, note: somme.map(s => "Σ " + fmtNum(s)) })),
      passaggio(op("riga", "÷Σ", "F.softmax(att, dim=-1)",
        I18N.t({ it: "ogni riga divisa per la sua somma", en: "each row divided by its own sum" }))),
      tappa(quadrata(att, "att", { maschera: futura, testoMaschera: "0", note: sommeAtt, scala: 1 })),
    );
    if (sel.j > sel.i) {
      dettaglio = "<b>att[" + sel.i + ", " + sel.j + "] = e<sup>−∞</sup> / … = 0</b>";
    } else {
      const termini = [];
      for (let j = 0; j <= sel.i; j++) termini.push("e<sup>" + fmtNum(scalata.data[sel.i * T + j], 3) + "</sup>");
      const z = scalata.data[sel.i * T + sel.j];
      dettaglio = "<b>att[" + sel.i + ", " + sel.j + "]</b> = e<sup>" + fmtNum(z, 3) + "</sup> / (" + termini.join(" + ") +
        ") = " + fmtNum(Math.exp(z), 4) + " / " + fmtNum(somme[sel.i], 4) + " = <b>" + fmtNum(att.data[sel.i * T + sel.j], 4) + "</b>";
    }
  }

  if (p === 7) {
    const pd = STATO.dropout_p;
    const m = mascheraDropout(T, pd);
    const dopo = { shape: [T, T], data: att.data.map((a, n) => (pd ? a * m[n] / (1 - pd) : a)) };
    const somme = Array.from({ length: T }, (_, i) => {
      let s = 0;
      for (let j = 0; j < T; j++) s += dopo.data[i * T + j];
      return "Σ " + fmtNum(s);
    });
    const opzioni = { maschera: futura, testoMaschera: "0", scala: 1, clic: null, scelta: null };
    fila.push(tappa(
      quadrata(att, "att", opzioni),
      pd ? matrice({ shape: [T, T], data: Float64Array.from(m) }, {
        nome: "m", forma: "(T, T)", righe, colonne: righe, decimali: 0, scala: 1 }) : null,
    ));
    fila.push(passaggio(op("elemento", "p", "self.attn_dropout(att)",
      pd ? "att ⊙ m / (1 − p),  p = " + fmtNum(pd, 1)
         : I18N.t({ it: "p = 0: passa com'è", en: "p = 0: passes through unchanged" }))));
    fila.push(tappa(quadrata(dopo, "att", { ...opzioni, note: somme,
      attenua: (i, j) => pd > 0 && j <= i && m[i * T + j] === 0 })));
    dettaglio = pd === 0
      ? I18N.t({
          it: "<b>p = 0</b>: il passo restituisce att così com'è. È quello che succede sempre in valutazione, con model.eval().",
          en: "<b>p = 0</b>: the step returns att unchanged. That is what always happens in evaluation, with model.eval().",
        })
      : I18N.t({
          it: "<b>p = 0,2</b>: le celle dove m vale 0 sono state azzerate, le altre moltiplicate per 1/(1 − 0,2) = 1,25. " +
              "Le somme di riga non valgono più 1, ma la loro media sì: è un'anteprima, il modello di questa pagina non la usa.",
          en: "<b>p = 0.2</b>: the cells where m is 0 were zeroed, the others multiplied by 1/(1 − 0.2) = 1.25. " +
              "Row sums are no longer 1, but their average is: this is a preview, the model on this page does not use it.",
        });
  }

  if (p === 8) {
    const s = scelte.y;
    s.i = Math.min(s.i, T - 1);
    s.j = Math.min(s.j, hs - 1);
    fila.push(
      tappa(
        mazzo(A.att[0], "att", "(B, nh, T, T)", { maschera: futura }),
        operatore("@"),
        mazzo(A.v[0], "v", "(B, nh, T, hs)", { righe: null }),
        operatore("="),
        mazzo(A.y_h[0], "y", "(B, nh, T, hs)"),
      ),
      testaDaVicino,
      tappa(prodotto({
        chiave: "y", sel: s,
        op: op("prodotto", "@", "y = att @ v", "(T, T) @ (T, hs) → (T, hs)"),
        a: att, oa: { nome: "att", forma: "(T, T)", righe, maschera: futura, testoMaschera: "0", scala: 1, dims: [null, dimT] },
        b: v, ob: { nome: nomeV, forma: "(T, hs)", attenua: i => i > s.i, dims: [dimT, null] },
        out: yh, oout: { nome: "y[0, " + h + "]", forma: "(T, hs)", dims: [dimT, dimHs] },
        clic: (i, j) => { scelte.y = { i, j }; ridisegna(); },
      })),
    );
    const termini = [];
    for (let j = 0; j <= s.i; j++)
      termini.push(fmtNum(att.data[s.i * T + j], 3) + "·" + fmtNum(v.data[j * hs + s.j]));
    dettaglio = "<b>y[" + s.i + ", " + s.j + "]</b> = Σ<sub>j≤" + s.i + "</sub> att[" + s.i + ", j] · v[j, " + s.j + "] = " +
      termini.join(" + ") + " = <b>" + fmtNum(yh.data[s.i * hs + s.j], 4) + "</b>" +
      I18N.t({
        it: " — le righe di v dopo la " + s.i + " sono attenuate: la loro colonna in att è 0.",
        en: " — the rows of v after " + s.i + " are dimmed: their column in att is 0.",
      });
  }

  if (p === 9) {
    fila.push(
      tappa(mazzo(A.y_h[0], "y", "(B, nh, T, hs)", { dims: [dimT, dimHs] })),
      passaggio(op("forma", "ᵀ", "y.transpose(1, 2)", "(B, nh, T, hs) → (B, T, nh, hs)")),
      tappa(matrice(A.y, { nome: "y", forma: "(B, T, nh, hs)", compatta: true, righe, bande: bandeTeste,
        fessure: tagliTeste, dims: [dimT, Array(nh).fill(dimHs)] })),
      passaggio(op("forma", "‖", ".contiguous().view(B, T, C)", "(B, T, nh, hs) → (B, T, C)")),
      tappa(matrice(A.y, { nome: "y", forma: "(B, T, C)", compatta: true, righe, bande: bandeTeste,
        dims: [dimT, "C = nh · hs = " + C] })),
    );
    dettaglio = I18N.t({
      it: "<b>y.transpose(1, 2).contiguous().view(B, T, C)</b>: il contrario del passo 2. Le carte tornano fianco a fianco, " +
          "e le uscite delle due teste, " + hs + " numeri ciascuna, si incollano in un vettore da " + C + " per posizione.",
      en: "<b>y.transpose(1, 2).contiguous().view(B, T, C)</b>: the reverse of step 2. The cards go back side by side, " +
          "and the outputs of the two heads, " + hs + " numbers each, are glued into one " + C + "-number vector per position.",
    });
  }

  if (p === 10) {
    const s = scelte.proj;
    s.i = Math.min(s.i, T - 1);
    const xIn = c.blocks[l].x_in, xMid = c.blocks[l].x_mid;
    fila.push(
      tappa(prodotto({
        dimensione: "compatta", chiave: "proj", sel: s,
        op: op("prodotto", "@", "self.c_proj(y)", I18N.t({ it: "poi resid_dropout, con p = 0", en: "then resid_dropout, with p = 0" })),
        a: A.y, oa: { nome: "y", forma: "(T, C)", righe, dims: [null, dimC] },
        b: trasposta(blocco.c_proj.weight), ob: { nome: "c_proj.weight.T", forma: "(C, C)", dims: [dimC, null] },
        bias: blocco.c_proj.bias, obias: { nome: "+ c_proj.bias", forma: "(C,)" },
        out: A.out, oout: { nome: "attn(ln_1(x))", forma: "(T, C)", dims: [dimT, dimC] },
        clic: (i, j) => { scelte.proj = { i, j }; ridisegna(); },
      })),
      passaggio(op("elemento", "+", "x = x + self.attn(self.ln_1(x))",
        I18N.t({ it: "Block.forward, riga 104: la somma residua", en: "Block.forward, line 104: the residual sum" }))),
      tappa(
        matrice(xIn, { nome: "x", forma: "(T, C)", compatta: true, righe, scelta: s, dims: [null, dimC] }),
        operatore("+"),
        matrice(A.out, { nome: "attn(…)", forma: "(T, C)", compatta: true, scelta: s, dims: [null, dimC] }),
        operatore("="),
        matrice(xMid, { nome: "x", forma: "(T, C)", compatta: true, scelta: s, dims: [dimT, dimC] }),
      ),
    );
    const norma = (t, i) => {
      let q2 = 0;
      for (let j = 0; j < C; j++) q2 += t.data[i * C + j] ** 2;
      return Math.sqrt(q2);
    };
    let nx = 0, na = 0;
    for (let i = 0; i < T; i++) { nx += norma(xIn, i); na += norma(A.out, i); }
    const W = blocco.c_proj.weight.data, b = blocco.c_proj.bias.data;
    const n = s.i * C + s.j;
    dettaglio = "<b>attn[" + s.i + ", " + s.j + "]</b> = y[" + s.i + "] · c_proj.weight[" + s.j + "] + c_proj.bias[" + s.j + "] = " +
      terminiProdotto(d => A.y.data[s.i * C + d], d => W[s.j * C + d], C) + " + " + fmtNum(b[s.j]) +
      " = <b>" + fmtNum(A.out.data[n], 4) + "</b><br>" +
      "<b>x[" + s.i + ", " + s.j + "]</b> = " + fmtNum(xIn.data[n], 4) + " + " + fmtNum(A.out.data[n], 4) +
      " = <b>" + fmtNum(xMid.data[n], 4) + "</b>. " +
      I18N.t({
        it: "In media il ramo dell'attenzione aggiunge un vettore di norma " + fmtNum(na / T, 3) +
            " a un nastro di norma " + fmtNum(nx / T, 3) + ": una correzione, non una sostituzione.",
        en: "On average the attention branch adds a vector of norm " + fmtNum(na / T, 3) +
            " to a stream of norm " + fmtNum(nx / T, 3) + ": a correction, not a replacement.",
      });
  }

  riempi(nodo("attn-tensori"), fila);
  nodo("attn-dettaglio").innerHTML = dettaglio;
}

function vaiAlPasso(n) {
  STATO.passo = Math.max(1, Math.min(N_PASSI, n));
  disegnaAttenzione();
}

nodo("attn-passi").addEventListener("click", e => {
  const b = e.target.closest("button.chip");
  if (b) vaiAlPasso(Number(b.dataset.passo));
});
nodo("passo-prec").addEventListener("click", () => vaiAlPasso(STATO.passo - 1));
nodo("passo-succ").addEventListener("click", () => vaiAlPasso(STATO.passo + 1));
preparaChip(nodo("attn-strato"), v => { STATO.strato = Number(v); disegnaAttenzione(); });
preparaChip(nodo("attn-testa"), v => { STATO.testa = Number(v); disegnaAttenzione(); });
nodo("dropout-chip").addEventListener("click", e => {
  const b = e.target.closest("button.chip");
  if (!b) return;
  STATO.dropout_p = Number(b.dataset.p);
  disegnaAttenzione();
});


/* --- Perché √hs ------------------------------------------------------------ */

function disegnaScala() {
  const k = Number(nodo("hs").value);
  const hs = 2 ** k;
  nodo("hs-out").textContent = String(hs);
  const rng = creaRng(100 + k);

  const prodotto = () => {
    let s = 0;
    for (let d = 0; d < hs; d++) s += rng.normale() * rng.normale();
    return s;
  };
  let somma2 = 0;
  for (let n = 0; n < 2000; n++) { const z = prodotto(); somma2 += z * z; }
  const std = Math.sqrt(somma2 / 2000);

  const punteggi = Array.from({ length: 8 }, prodotto);
  const grezza = softmax(Float64Array.from(punteggi));
  const scalata = softmax(Float64Array.from(punteggi.map(z => z / Math.sqrt(hs))));

  const g = nodo("scala-grafico-g");
  g.textContent = "";
  const pannello = (dati, x0, titolo, classe) => {
    g.appendChild(svgEl("text", { x: x0, y: 14, class: "axlab" }, titolo));
    g.appendChild(svgEl("line", { x1: x0, y1: 200, x2: x0 + 190, y2: 200, class: "ax" }));
    for (let j = 0; j < 8; j++) {
      const h = dati[j] * 170;
      g.appendChild(svgEl("rect", {
        x: x0 + j * 24 + 3, y: (200 - h).toFixed(1), width: 18, height: Math.max(0.5, h).toFixed(1), class: classe,
      }));
    }
  };
  pannello(grezza, 8, I18N.t({ it: "softmax(q·k)", en: "softmax(q·k)" }), "vbar vbar--neg");
  pannello(scalata, 222, "softmax(q·k / √hs)", "vbar");

  riempi(nodo("scala-letture"), [
    tessera(I18N.t({ it: "Dev. std di `q·k`", en: "Std of `q·k`" }), fmtNum(std, 2), "√hs = " + fmtNum(Math.sqrt(hs), 2)),
    tessera(I18N.t({ it: "Dopo la scala", en: "After scaling" }), fmtNum(std / Math.sqrt(hs), 2), I18N.t({ it: "sempre circa 1", en: "always about 1" })),
    tessera(I18N.t({ it: "Peso max, grezzo", en: "Max weight, raw" }), fmtNum(Math.max(...grezza), 2), "softmax(q·k)"),
    tessera(I18N.t({ it: "Peso max, scalato", en: "Max weight, scaled" }), fmtNum(Math.max(...scalata), 2), "softmax(q·k/√hs)"),
  ]);

  nodo("scala-verdetto").innerHTML = Math.max(...grezza) > 0.9 && hs >= 16
    ? I18N.t({
        it: "<b>Senza la scala la softmax è satura.</b> Con hs = " + hs + " i punteggi hanno deviazione " + fmtNum(std, 1) +
            ", e una sola posizione si prende quasi tutto il peso: le altre ricevono un gradiente vicino a zero e smettono di imparare.",
        en: "<b>Without the scale the softmax saturates.</b> With hs = " + hs + " the scores have standard deviation " + fmtNum(std, 1) +
            ", and a single position takes almost all the weight: the others get a near-zero gradient and stop learning.",
      })
    : I18N.t({
        it: "<b>Con teste strette la differenza è piccola.</b> Alza hs: GPT-2 usa 64, e lì la softmax senza scala diventa un one-hot.",
        en: "<b>With narrow heads the difference is small.</b> Raise hs: GPT-2 uses 64, and there the unscaled softmax becomes a one-hot.",
      });
}

nodo("hs").addEventListener("input", disegnaScala);


/* --- Tutte le teste -------------------------------------------------------- */

function disegnaTeste() {
  const c = STATO.uscita.cache;
  const righe = etichettePosizioni();
  const fila = [];
  c.blocks.forEach((b, l) => b.attn.att[0].forEach((att, h) => {
    fila.push(matrice(att, {
      nome: "h." + l + " · " + I18N.t({ it: "testa ", en: "head " }) + h, forma: "att (T, T)",
      righe, colonne: righe, maschera: (i, j) => j > i, testoMaschera: "", scala: 1,
    }));
  }));
  riempi(nodo("teste-tutte"), fila);
}


/* --------------------------------------------------------------------------
   Parte seconda — MLP e GELU
   -------------------------------------------------------------------------- */

function disegnaMLP() {
  const c = STATO.uscita.cache;
  const T = c.t, C = config.n_embd, C4 = 4 * config.n_embd;
  const l = STATO.mlp_strato, passo = STATO.mlp_passo;
  aggiornaChip(nodo("mlp-strato"), [0, 1].map(v => ({ valore: v, etichetta: "h." + v })), l);
  aggiornaChip(nodo("mlp-pos"), vociPosizione(), STATO.pos);
  segnaChip(nodo("mlp-passi"), "passo", passo);

  /* il testo del passo e la riga del listato di MLP */
  let righeCodice = [];
  document.querySelectorAll("#mlp-passi-testo .step").forEach(s => {
    const attivo = Number(s.dataset.passo) === passo;
    s.hidden = !attivo;
    if (attivo) righeCodice = s.dataset.righe.split(",").map(Number);
  });
  document.querySelectorAll("#listato-mlp .listing__line").forEach(r =>
    r.classList.toggle("is-current", righeCodice.includes(Number(r.dataset.riga))));

  const b = c.blocks[l];
  const m = b.mlp;
  const mod = STATO.model.transformer.h[l].mlp;
  const p = STATO.pos;
  const righe = etichettePosizioni();
  const riga = i => i === p;
  const dimT = "T = " + T, dimC = "C = " + C, dim4C = "4C = " + C4;
  /* Un clic su una cella sceglie anche la posizione: il grafico qui sotto
     mostra la riga su cui si è cliccato. */
  const scegli = (chiave, i, j) => conFocus(() => { STATO.scelte[chiave] = j; cambiaPosizione(i); });
  const jFc = Math.min(STATO.scelte.fc, C4 - 1), jProj = Math.min(STATO.scelte.mproj, C - 1);
  const fila = [];
  let dettaglio = "";

  if (passo === 1) {
    const x = b.ln_2.out, W = mod.c_fc.weight.data, bias = mod.c_fc.bias.data;
    fila.push(tappa(prodotto({
      dimensione: "fine", chiave: "fc", sel: { i: p, j: jFc },
      op: op("prodotto", "@", "x = self.c_fc(x)", "x @ c_fc.weight.T + c_fc.bias"),
      a: x, oa: { nome: "ln_2(x)", forma: "(T, C)", dims: [null, dimC] },
      b: trasposta(mod.c_fc.weight), ob: { nome: "c_fc.weight.T", forma: "(C, 4C)", dims: [dimC, null] },
      bias: mod.c_fc.bias, obias: { nome: "+ c_fc.bias", forma: "(4C,)" },
      out: m.h_fc, oout: { nome: "c_fc(x)", forma: "(T, 4C)", dims: [dimT, dim4C] },
      clic: (i, j) => scegli("fc", i, j),
    })));
    dettaglio = "<b>c_fc(x)[" + p + ", " + jFc + "]</b> = ln_2(x)[" + p + "] · c_fc.weight[" + jFc + "] + c_fc.bias[" + jFc + "] = " +
      terminiProdotto(d => x.data[p * C + d], d => W[jFc * C + d], C) + " + " + fmtNum(bias[jFc]) +
      " = <b>" + fmtNum(m.h_fc.data[p * C4 + jFc], 4) + "</b>";
  }

  if (passo === 2) {
    const n = p * C4 + jFc;
    const opzioni = { forma: "(T, 4C)", compatta: true, righe, evidenzia: riga, rigaScelta: p,
      scelta: { i: p, j: jFc }, clic: (i, j) => scegli("fc", i, j), dims: [dimT, dim4C] };
    fila.push(
      tappa(matrice(m.h_fc, { nome: "c_fc(x)", chiave: "gelu-prima", ...opzioni })),
      passaggio(op("elemento", "Φ", "x = self.gelu(x)", "GELU(x) = x · Φ(x)")),
      tappa(matrice(m.h_gelu, { nome: "gelu(x)", chiave: "gelu-dopo", ...opzioni })),
    );
    dettaglio = "<b>gelu(x)[" + p + ", " + jFc + "]</b> = x · Φ(x) = " + fmtNum(m.h_fc.data[n], 4) + " · " +
      fmtNum(m.phi[n], 4) + " = <b>" + fmtNum(m.h_gelu.data[n], 4) + "</b>" +
      I18N.t({
        it: " — Φ(x) è la parte di x che passa: quasi tutta se x è grande, quasi niente se è molto negativo.",
        en: " — Φ(x) is the share of x that gets through: almost all of it when x is large, almost none when it is very negative.",
      });
  }

  if (passo === 3) {
    const x = m.h_gelu, W = mod.c_proj.weight.data, bias = mod.c_proj.bias.data;
    fila.push(tappa(prodotto({
      dimensione: "fine", chiave: "mproj", sel: { i: p, j: jProj },
      op: op("prodotto", "@", "x = self.c_proj(x)", "x @ c_proj.weight.T + c_proj.bias"),
      a: x, oa: { nome: "gelu(x)", forma: "(T, 4C)", dims: [null, dim4C] },
      b: trasposta(mod.c_proj.weight), ob: { nome: "c_proj.weight.T", forma: "(4C, C)", dims: [dim4C, null] },
      bias: mod.c_proj.bias, obias: { nome: "+ c_proj.bias", forma: "(C,)" },
      out: m.out, oout: { nome: "c_proj(x)", forma: "(T, C)", dims: [dimT, dimC] },
      clic: (i, j) => scegli("mproj", i, j),
    })));
    dettaglio = "<b>c_proj(x)[" + p + ", " + jProj + "]</b> = gelu(x)[" + p + "] · c_proj.weight[" + jProj + "] + c_proj.bias[" + jProj + "] = Σ<sub>64</sub> … = " +
      terminiProdotto(d => x.data[p * C4 + d], d => W[jProj * C4 + d], 4) + " + … + " + fmtNum(bias[jProj]) +
      " = <b>" + fmtNum(m.out.data[p * C + jProj], 4) + "</b>" +
      I18N.t({ it: " (i primi 4 termini su 64)", en: " (the first 4 of 64 terms)" });
  }

  if (passo === 4) {
    const n = p * C + jProj;
    const opzioni = { forma: "(T, C)", compatta: true, scelta: { i: p, j: jProj }, evidenzia: riga };
    fila.push(
      tappa(matrice(m.out, { nome: "c_proj(x)", righe, ...opzioni, dims: [dimT, dimC] })),
      passaggio(op("elemento", "p", "x = self.dropout(x)", I18N.t({ it: "p = 0: passa com'è", en: "p = 0: passes through unchanged" }))),
      passaggio(op("elemento", "+", "x = x + self.mlp(self.ln_2(x))",
        I18N.t({ it: "Block.forward, riga 105: la seconda somma residua", en: "Block.forward, line 105: the second residual sum" }))),
      tappa(
        matrice(b.x_mid, { nome: "x", righe, ...opzioni, dims: [null, dimC] }),
        operatore("+"),
        matrice(m.out, { nome: "mlp(…)", ...opzioni, dims: [null, dimC] }),
        operatore("="),
        matrice(b.x_out, { nome: "x", ...opzioni, dims: [dimT, dimC] }),
      ),
    );
    dettaglio = "<b>x[" + p + ", " + jProj + "]</b> = " + fmtNum(b.x_mid.data[n], 4) + " + " + fmtNum(m.out.data[n], 4) +
      " = <b>" + fmtNum(b.x_out.data[n], 4) + "</b>" +
      I18N.t({
        it: l === config.n_layer - 1
          ? ". È l'uscita dell'ultimo Block: da qui si va a ln_f e lm_head."
          : ". È l'ingresso del Block h." + (l + 1) + ".",
        en: l === config.n_layer - 1
          ? ". This is the output of the last Block: from here on, ln_f and lm_head."
          : ". This is the input of Block h." + (l + 1) + ".",
      });
  }

  riempi(nodo("mlp-tensori"), fila);
  nodo("mlp-dettaglio").innerHTML = dettaglio;

  /* Il grafico: i 4C numeri nascosti della posizione scelta. */
  const prima = Array.from(m.h_fc.data.subarray(p * C4, (p + 1) * C4));
  const dopo = Array.from(m.h_gelu.data.subarray(p * C4, (p + 1) * C4));
  const scala = Math.max(...prima.map(Math.abs)) || 1;

  const g = nodo("mlp-grafico-g");
  g.textContent = "";
  g.appendChild(svgEl("text", { x: 8, y: 14, class: "axlab" },
    "c_fc(ln_2(x))[" + p + "] → gelu — " + I18N.t({ it: "scala ±", en: "scale ±" }) + fmtNum(scala, 3)));
  barre(g, prima, 8, 24, 624, 188, scala, { pos: "vbar vbar--3", neg: "vbar vbar--3", margine: 0.08 });
  barre(g, dopo, 8, 24, 624, 188, scala, { pos: "vbar", neg: "vbar vbar--neg", margine: 0.3 });

  const negativi = prima.filter(z => z < 0).length;
  const spenti = dopo.filter(z => Math.abs(z) < 0.05 * scala).length;
  riempi(nodo("mlp-letture"), [
    tessera(I18N.t({ it: "Negativi prima", en: "Negative before" }), negativi + " / " + C4,
      I18N.t({ it: "la GELU li schiaccia", en: "GELU squashes them" })),
    tessera(I18N.t({ it: "Quasi spenti dopo", en: "Nearly off after" }), spenti + " / " + C4,
      I18N.t({ it: "|gelu| < 5% della scala", en: "|gelu| < 5% of scale" })),
    tessera(I18N.t({ it: "Minimo dopo", en: "Minimum after" }), fmtNum(Math.min(...dopo), 3),
      I18N.t({ it: "la GELU non scende sotto −0,17", en: "GELU never goes below −0.17" })),
  ]);
}

preparaChip(nodo("mlp-strato"), v => { STATO.mlp_strato = Number(v); disegnaMLP(); });
preparaChip(nodo("mlp-pos"), v => conFocus(() => cambiaPosizione(v)));
nodo("mlp-passi").addEventListener("click", e => {
  const b = e.target.closest("button.chip");
  if (!b) return;
  STATO.mlp_passo = Number(b.dataset.passo);
  disegnaMLP();
});

function disegnaGelu() {
  const x = Number(nodo("gelu-x").value) / 100;
  nodo("gelu-x-out").textContent = fmtNum(x);

  const g = nodo("gelu-grafico-g");
  g.textContent = "";
  const W = 420, H = 260, L = 30, R = 10, T = 10, B = 26;
  const sx = v => L + (v + 4) / 8 * (W - L - R);
  const sy = v => H - B - (v + 1) / 5 * (H - T - B);

  for (let t = -4; t <= 4; t += 2) {
    g.appendChild(svgEl("line", { x1: sx(t), y1: T, x2: sx(t), y2: H - B, class: "gridline" }));
    g.appendChild(svgEl("text", { x: sx(t), y: H - 8, class: "axlab", "text-anchor": "middle" }, fmtNum(t, 0)));
  }
  for (let t = -1; t <= 4; t++) {
    g.appendChild(svgEl("line", { x1: L, y1: sy(t), x2: W - R, y2: sy(t), class: t === 0 ? "ax" : "gridline" }));
    g.appendChild(svgEl("text", { x: L - 6, y: sy(t) + 3.5, class: "axlab", "text-anchor": "end" }, fmtNum(t, 0)));
  }

  const curva = (f, classe) => {
    let d = "";
    for (let n = 0; n <= 160; n++) {
      const v = -4 + n / 20;
      d += (n ? "L" : "M") + sx(v).toFixed(1) + " " + sy(f(v)).toFixed(1);
    }
    g.appendChild(svgEl("path", { d, class: classe }));
  };
  curva(v => Math.max(0, v), "curve curve--2");
  curva(gelu_scalare, "curve");
  curva(gelu_tanh, "curve curve--3");

  g.appendChild(svgEl("line", { x1: sx(x), y1: T, x2: sx(x), y2: H - B, class: "cursor" }));
  g.appendChild(svgEl("circle", { cx: sx(x), cy: sy(gelu_scalare(x)), r: 4, class: "dot" }));

  riempi(nodo("gelu-letture"), [
    tessera("`GELU(x)`", fmtNum(gelu_scalare(x), 4), "x · Φ(x)", true),
    tessera("`Φ(x)`", fmtNum(0.5 * (1 + erf(x / Math.SQRT2)), 4), I18N.t({ it: "quanta parte di x passa", en: "how much of x gets through" })),
    tessera("`ReLU(x)`", fmtNum(Math.max(0, x), 4), "max(0, x)"),
    tessera("`|GELU − tanh|`", fmtNum(Math.abs(gelu_scalare(x) - gelu_tanh(x)) * 1e4, 2) + "·10⁻⁴",
      I18N.t({ it: "esatta contro approssimata", en: "exact against approximate" })),
  ]);
}

nodo("gelu-x").addEventListener("input", disegnaGelu);


/* --------------------------------------------------------------------------
   Parte terza — logits e loss
   -------------------------------------------------------------------------- */

/* ln_f e lm_head: il vettore di ogni posizione contro la tabella wte,
   trasposta. La cella scelta, finché nessuno clicca, è quella del
   carattere giusto: il suo logit è un prodotto scalare con l'embedding
   di quel carattere. */
function disegnaTesta() {
  const c = STATO.uscita.cache;
  const T = c.t, C = config.n_embd, V = config.vocab_size;
  const righe = etichettePosizioni();
  const p = STATO.pos;
  const wte = STATO.model.transformer.wte.weight;
  aggiornaChip(nodo("testa-pos"), vociPosizione(), p);

  const riga = c.logits.data.subarray(p * V, (p + 1) * V);
  let migliore = 0;
  for (let j = 1; j < V; j++) if (riga[j] > riga[migliore]) migliore = j;
  const target = c.targets[p];
  const j = STATO.scelte.logit !== null ? STATO.scelte.logit : target !== -1 ? target : migliore;

  riempi(nodo("testa-tensori"), [
    tappa(matrice(c.x_final, { nome: "x", forma: "(B, T, C)", compatta: true, righe,
      evidenzia: i => i === p, rigaScelta: p, dims: ["T = " + T, "C = " + C] })),
    passaggio(op("riga", "μσ", "x = self.transformer.ln_f(x)",
      I18N.t({ it: "l'ultima LayerNorm, fuori dai Block", en: "the last LayerNorm, outside the Blocks" }))),
    tappa(prodotto({
      dimensione: "fine", chiave: "logit", sel: { i: p, j },
      op: op("prodotto", "@", "logits = self.lm_head(x)",
        I18N.t({ it: "lm_head.weight è wte.weight: nessun bias", en: "lm_head.weight is wte.weight: no bias" })),
      a: c.ln_f.out, oa: { nome: "ln_f(x)", forma: "(T, C)", dims: [null, "C = " + C] },
      b: trasposta(wte), ob: { nome: "wte.weight.T", forma: "(C, V)", dims: ["C = " + C, null] },
      out: c.logits, oout: { nome: "logits", forma: "(T, V)", dims: ["T = " + T, "V = " + V] },
      clic: (i, jj) => conFocus(() => { cambiaPosizione(i); STATO.scelte.logit = jj; disegnaTesta(); }),
    })),
  ]);

  nodo("testa-dettaglio").innerHTML =
    "<b>logits[" + p + ", " + j + "]</b> = ln_f(x)[" + p + "] · wte.weight[" + j + "] = " +
    terminiProdotto(d => c.ln_f.out.data[p * C + d], d => wte.data[j * C + d], C) +
    " = <b>" + fmtNum(c.logits.data[p * V + j], 4) + "</b>" +
    I18N.t({
      it: " — la colonna " + j + " è l'embedding di '" + esc(mostra(itos[j])) + "'" +
          (j === target ? ", il carattere giusto dopo la posizione " + p : j === migliore ? ", il preferito del modello" : "") +
          ": più il vettore gli somiglia, più quel carattere è probabile.",
      en: " — column " + j + " is the embedding of '" + esc(mostra(itos[j])) + "'" +
          (j === target ? ", the right character after position " + p : j === migliore ? ", the model's favourite" : "") +
          ": the more the vector resembles it, the more likely that character is.",
    });
}

preparaChip(nodo("testa-pos"), v => conFocus(() => cambiaPosizione(v)));

function disegnaLoss() {
  const c = STATO.uscita.cache;
  const T = c.t, V = config.vocab_size;
  const righe = etichettePosizioni();
  const targets = c.targets;

  /* Il calcolo di F.cross_entropy, a pezzi: softmax di ogni riga, la
     probabilità del carattere giusto, −log, la media. La riga con
     target −1 non entra. */
  const giusto = (i, j) => targets[i] === j;
  const ignorata = i => targets[i] === -1;
  const pT = tensor([T, 1]), nll = tensor([T, 1]);
  let sommaNll = 0, quante = 0;
  for (let i = 0; i < T; i++) {
    if (ignorata(i)) continue;
    pT.data[i] = c.probs.data[i * V + targets[i]];
    nll.data[i] = -Math.log(pT.data[i]);
    sommaNll += nll.data[i];
    quante++;
  }
  const colonna = (t, nome, extra = {}) => matrice(t, { nome, forma: "(T, 1)", dimensione: "larga", righe, decimali: 3,
    maschera: i => ignorata(i), testoMaschera: "—", ...extra });
  const dimTV = ["T = " + T, "V = " + V];
  riempi(nodo("loss-logits"), [
    tappa(matrice(c.logits, { nome: "logits", forma: "(B, T, V)", compatta: true, righe,
      evidenzia: giusto, attenua: i => ignorata(i), dims: dimTV })),
    passaggio(op("riga", "÷Σ", "F.cross_entropy(logits.view(-1, logits.size(-1)), …)",
      I18N.t({ it: "dentro: la softmax di ogni riga, eˣ diviso la somma", en: "inside: the softmax of each row, eˣ over the sum" }))),
    tappa(matrice(c.probs, { nome: "probs", forma: "(B·T, V)", compatta: true, righe,
      evidenzia: giusto, attenua: i => ignorata(i), dims: dimTV })),
    passaggio(op("forma", "[t]", "probs[n, targets[n]]",
      I18N.t({ it: "di ogni riga, solo la probabilità del carattere giusto; targets = −1 è saltato",
               en: "from each row, only the probability of the right character; targets = −1 is skipped" }))),
    tappa(
      colonna(pT, "p(target)", { dims: ["T = " + T, null] }),
      op("elemento", "−log", "-log(p)"),
      colonna(nll, "−log p", { dims: ["T = " + T, null] }),
      op("riga", "x̄", ".mean()", I18N.t({ it: "su " + quante + " righe", en: "over " + quante + " rows" })),
      matrice({ shape: [1, 1], data: Float64Array.from([quante ? sommaNll / quante : NaN]) },
        { nome: "loss", forma: "()", dimensione: "larga", decimali: 3 }),
    ),
  ]);

  const corpo = nodo("loss-tabella");
  corpo.textContent = "";
  let somma = 0, contate = 0;
  for (let i = 0; i < T; i++) {
    const p = c.probs.data.subarray(i * V, (i + 1) * V);
    const ordine = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 3);
    const tre = ordine.map(j => '<span class="mono">' + esc(mostra(itos[j])) + "</span> " + fmtNum(p[j])).join(" · ");
    const t = targets[i];
    let pt = "—", nl = I18N.t({ it: "ignorato", en: "ignored" });
    if (t !== -1) {
      pt = fmtNum(p[t], 3);
      nl = fmtNum(-Math.log(p[t]), 3);
      somma += -Math.log(p[t]);
      contate++;
    }
    const tr = document.createElement("tr");
    tr.innerHTML =
      '<td class="table__num">' + i + "</td>" +
      '<td><span class="mono">' + esc(mostra(STATO.prompt[i])) + "</span></td>" +
      "<td>" + (t === -1 ? '<span class="mono">−1</span>' : '<span class="mono table__pick">' + esc(mostra(itos[t])) + "</span>") + "</td>" +
      '<td class="table__num">' + pt + "</td>" +
      '<td class="table__num">' + nl + "</td>" +
      "<td>" + tre + "</td>";
    corpo.appendChild(tr);
  }

  const loss = contate ? somma / contate : NaN;
  const ultima = c.probs.data.subarray((T - 1) * V, T * V);
  let migliore = 0;
  for (let j = 1; j < V; j++) if (ultima[j] > ultima[migliore]) migliore = j;

  riempi(nodo("loss-letture"), [
    tessera("`loss`", contate ? fmtNum(loss, 3) : "—",
      I18N.t({ it: "media su ", en: "mean over " }) + contate + I18N.t({ it: " posizioni", en: " positions" }), true),
    tessera("`ln 65`", fmtNum(Math.log(V), 3), I18N.t({ it: "tirare a indovinare", en: "pure guessing" })),
    tessera(I18N.t({ it: "Perplessità", en: "Perplexity" }), contate ? fmtNum(Math.exp(loss), 1) : "—",
      I18N.t({ it: "e^loss: fra quanti caratteri è indeciso", en: "e^loss: how many characters it hesitates between" })),
    tessera(I18N.t({ it: "Dopo il prompt", en: "After the prompt" }), "'" + mostra(itos[migliore]) + "'",
      "p = " + fmtNum(ultima[migliore], 3)),
  ]);

  nodo("loss-verdetto").innerHTML = !contate
    ? I18N.t({ it: "<b>Serve un prompt di almeno due caratteri</b> perché ci sia un bersaglio.",
               en: "<b>The prompt needs at least two characters</b> for there to be a target." })
    : loss > 3.9
      ? I18N.t({
          it: "<b>Loss " + fmtNum(loss, 2) + ", praticamente ln 65.</b> Il modello dà a ogni carattere circa 1/65 di probabilità: " +
              "è la firma di un modello appena inizializzato. Addestralo e torna qui.",
          en: "<b>Loss " + fmtNum(loss, 2) + ", practically ln 65.</b> The model gives each character about a 1/65 chance: " +
              "the signature of a freshly initialised model. Train it and come back.",
        })
      : I18N.t({
          it: "<b>Loss " + fmtNum(loss, 2) + ".</b> In media il modello è indeciso fra " + fmtNum(Math.exp(loss), 1) +
              " caratteri invece che fra 65. Le posizioni con −log p alto sono quelle che non poteva sapere: l'inizio di una parola nuova.",
          en: "<b>Loss " + fmtNum(loss, 2) + ".</b> On average the model hesitates between " + fmtNum(Math.exp(loss), 1) +
              " characters instead of 65. The positions with a high −log p are the ones it could not know: the start of a new word.",
        });
}


/* --------------------------------------------------------------------------
   Parte quarta — get_lr, gruppi, addestramento
   -------------------------------------------------------------------------- */

function leggiLR() {
  const tc = {
    learning_rate: Number(nodo("lr-max").value) / 1000,
    warmup_iters: Number(nodo("lr-warmup").value),
    lr_decay_iters: Number(nodo("lr-decay").value),
    min_lr: Number(nodo("lr-min").value) / 10000,
  };
  /* con warmup_iters = lr_decay_iters train.py dividerebbe per zero */
  if (tc.lr_decay_iters <= tc.warmup_iters) tc.lr_decay_iters = tc.warmup_iters + 1;
  return tc;
}

function disegnaLR() {
  const tc = leggiLR();
  nodo("lr-max-out").textContent = fmtNum(tc.learning_rate, 3);
  nodo("lr-warmup-out").textContent = String(tc.warmup_iters);
  nodo("lr-decay-out").textContent = String(Number(nodo("lr-decay").value));
  nodo("lr-min-out").textContent = fmtNum(tc.min_lr, 4);

  const g = nodo("lr-grafico-g");
  g.textContent = "";
  const W = 420, H = 240, L = 44, R = 18, T = 10, B = 26, MAX = 3000;
  const top = Math.max(tc.learning_rate, tc.min_lr) * 1.1;
  const sx = v => L + v / MAX * (W - L - R);
  const sy = v => H - B - v / top * (H - T - B);

  for (let t = 0; t <= MAX; t += 1000) {
    g.appendChild(svgEl("line", { x1: sx(t), y1: T, x2: sx(t), y2: H - B, class: "gridline" }));
    g.appendChild(svgEl("text", { x: sx(t), y: H - 8, class: "axlab", "text-anchor": "middle" }, fmtIntero(t)));
  }
  for (let n = 0; n <= 4; n++) {
    const v = top * n / 4;
    g.appendChild(svgEl("line", { x1: L, y1: sy(v), x2: W - R, y2: sy(v), class: n ? "gridline" : "ax" }));
    g.appendChild(svgEl("text", { x: L - 6, y: sy(v) + 3.5, class: "axlab", "text-anchor": "end" }, fmtNum(v, 3)));
  }
  g.appendChild(svgEl("line", { x1: sx(tc.lr_decay_iters), y1: T, x2: sx(tc.lr_decay_iters), y2: H - B, class: "markline" }));

  let d = "";
  for (let it = 0; it <= MAX; it += 5) d += (it ? "L" : "M") + sx(it).toFixed(1) + " " + sy(get_lr(it, tc)).toFixed(1);
  g.appendChild(svgEl("path", { d, class: "curve" }));

  const meta = Math.round((tc.warmup_iters + tc.lr_decay_iters) / 2);
  riempi(nodo("lr-letture"), [
    tessera("`get_lr(0)`", fmtNum(get_lr(0, tc), 5), "learning_rate / (warmup_iters + 1)"),
    tessera("`get_lr(" + tc.warmup_iters + ")`", fmtNum(get_lr(tc.warmup_iters, tc), 5),
      I18N.t({ it: "fine del riscaldamento", en: "end of warmup" })),
    tessera("`get_lr(" + meta + ")`", fmtNum(get_lr(meta, tc), 5),
      I18N.t({ it: "metà discesa: la media dei due", en: "halfway down: the average of the two" })),
    tessera("`get_lr(" + MAX + ")`", fmtNum(get_lr(MAX, tc), 5), "min_lr"),
  ]);
}

["lr-max", "lr-warmup", "lr-decay", "lr-min"].forEach(id => nodo(id).addEventListener("input", disegnaLR));

function disegnaGruppi() {
  const corpo = nodo("gruppi-tabella");
  corpo.textContent = "";
  const optimizer = configure_optimizers(STATO.model, train_config.weight_decay, train_config.learning_rate,
    [train_config.beta1, train_config.beta2]);
  optimizer.param_groups.forEach((gruppo, k) => {
    const totale = gruppo.params.reduce((s, [, p]) => s + p.data.length, 0);
    const testa = document.createElement("tr");
    testa.className = "table__group";
    testa.innerHTML = '<th colspan="4" scope="colgroup">' + (k === 0
      ? '<span class="ident">decay_params — dim() ≥ 2</span>'
      : '<span class="ident">nodecay_params — dim() &lt; 2</span>') +
      " · " + gruppo.params.length + I18N.t({ it: " tensori, ", en: " tensors, " }) + fmtIntero(totale) +
      I18N.t({ it: " numeri", en: " numbers" }) + "</th>";
    corpo.appendChild(testa);
    gruppo.params.forEach(([nome, p]) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        '<td class="table__name"><span class="mono">' + esc(nome) + "</span></td>" +
        '<td><span class="mono">(' + p.shape.join(", ") + (p.shape.length === 1 ? "," : "") + ")</span></td>" +
        '<td class="table__num">' + fmtIntero(p.data.length) + "</td>" +
        '<td class="table__num">' + fmtNum(gruppo.weight_decay, 1) + "</td>";
      corpo.appendChild(tr);
    });
  });
}


/* --- Il laboratorio di addestramento -------------------------------------- */

function etichettaBottoni() {
  const finito = ADD.iter_num >= train_config.max_iters;
  const testo = ADD.in_corso
    ? I18N.t({ it: "Pausa", en: "Pause" })
    : finito
      ? I18N.t({ it: "Riaddestra da zero", en: "Retrain from zero" })
      : ADD.iter_num > 0
        ? I18N.t({ it: "Riprendi", en: "Resume" })
        : I18N.t({ it: "Addestra", en: "Train" });
  nodo("addestra").textContent = testo;
  nodo("addestra-rapido").textContent = testo;
  nodo("passi-chip").querySelectorAll("button.chip").forEach(b => { b.disabled = ADD.in_corso || ADD.iter_num > 0; });
}

/* Ogni eval_interval passi: estimate_loss() e un campione di testo, come
   il blocco "evaluate the loss" di train.py. Poi tutti i laboratori si
   ridisegnano con i pesi di adesso. */
function valuta() {
  const l = estimate_loss(STATO.model, ADD.dati, ADD.rngEval, train_config);
  ADD.storia.push({ iter: ADD.iter_num, train: l.train, val: l.val });
  const idx = generate(STATO.model, encode("\n"), 200, 1.0, null, creaRng(1000 + ADD.iter_num));
  ADD.campione = decode(idx.slice(1));
  ricalcola();
}

async function avviaOFerma() {
  if (ADD.in_corso) {           // Pausa
    ADD.in_corso = false;
    ADD.corsa++;
    ricalcola();
    disegnaAddestramento();
    return;
  }
  if (ADD.iter_num >= train_config.max_iters) azzera();

  if (!ADD.dati) {
    try {
      const risposta = await fetch("/assets/data/shakespeare.txt");
      if (!risposta.ok) throw new Error(String(risposta.status));
      ADD.dati = prepare(await risposta.text());
      ADD.errore = false;
    } catch (e) {
      ADD.errore = true;
      disegnaStato();
      disegnaAddestramento();
      return;
    }
  }
  if (!ADD.optimizer) {
    ADD.optimizer = configure_optimizers(STATO.model, train_config.weight_decay, train_config.learning_rate,
      [train_config.beta1, train_config.beta2]);
    ADD.rng = creaRng(42);
    ADD.rngEval = creaRng(7);
  }
  ADD.in_corso = true;
  const corsa = ++ADD.corsa;
  disegnaStato();
  disegnaAddestramento();
  setTimeout(() => fetta(corsa), 0);
}

/* Il ciclo gira a fette di una quarantina di millisecondi, poi restituisce
   il controllo al browser: la pagina resta scorrevole e i bottoni
   rispondono anche mentre il modello si allena. */
function fetta(corsa) {
  if (corsa !== ADD.corsa || !ADD.in_corso) return;
  const max = train_config.max_iters;
  const t0 = performance.now();
  let passi = 0, tempo = 0;
  while (performance.now() - t0 < 40 && ADD.iter_num < max) {
    if (ADD.iter_num % train_config.eval_interval === 0) valuta();
    const s = performance.now();
    ADD.ultimo = training_step(STATO.model, ADD, ADD.dati, train_config);
    tempo += performance.now() - s;
    passi++;
  }
  if (passi) {
    const m = tempo / passi;
    ADD.ms_per_iter = ADD.ms_per_iter === null ? m : 0.8 * ADD.ms_per_iter + 0.2 * m;
  }
  if (ADD.iter_num >= max) {
    valuta();
    ADD.in_corso = false;
    disegnaStato();
    disegnaAddestramento();
    return;
  }
  disegnaAddestramento();
  disegnaStato();
  setTimeout(() => fetta(corsa), 0);
}

function azzera() {
  ADD.in_corso = false;
  ADD.corsa++;
  ADD.iter_num = 0;
  ADD.optimizer = null;
  ADD.storia = [];
  ADD.ultimo = null;
  ADD.ms_per_iter = null;
  ADD.campione = "";
  STATO.model = GPT(config, SEME_PESI);
  GEN.idx = encode(STATO.prompt);
  GEN.nuovi = 0;
  ricalcola();
  disegnaAddestramento();
}

function disegnaAddestramento() {
  const max = train_config.max_iters;
  const g = nodo("curva-loss-g");
  g.textContent = "";
  const W = 420, H = 280, L = 34, R = 18, T = 10, B = 26, TOP = 4.6;
  const sx = v => L + v / max * (W - L - R);
  const sy = v => H - B - Math.min(v, TOP) / TOP * (H - T - B);

  for (let n = 0; n <= 4; n++) {
    const it = Math.round(max * n / 4);
    g.appendChild(svgEl("line", { x1: sx(it), y1: T, x2: sx(it), y2: H - B, class: "gridline" }));
    g.appendChild(svgEl("text", { x: sx(it), y: H - 8, class: "axlab", "text-anchor": "middle" }, fmtIntero(it)));
  }
  for (let v = 0; v <= 4; v++) {
    g.appendChild(svgEl("line", { x1: L, y1: sy(v), x2: W - R, y2: sy(v), class: v ? "gridline" : "ax" }));
    g.appendChild(svgEl("text", { x: L - 6, y: sy(v) + 3.5, class: "axlab", "text-anchor": "end" }, fmtNum(v, 0)));
  }
  g.appendChild(svgEl("line", { x1: L, y1: sy(Math.log(65)), x2: W - R, y2: sy(Math.log(65)), class: "markline" }));

  ["train", "val"].forEach((parte, k) => {
    if (!ADD.storia.length) return;
    let d = "";
    ADD.storia.forEach((s, n) => { d += (n ? "L" : "M") + sx(s.iter).toFixed(1) + " " + sy(s[parte]).toFixed(1); });
    g.appendChild(svgEl("path", { d, class: k ? "curve curve--2" : "curve" }));
  });

  const ultimoEval = ADD.storia.length ? ADD.storia[ADD.storia.length - 1] : null;
  const restanti = ADD.ms_per_iter ? (max - ADD.iter_num) * ADD.ms_per_iter / 1000 : null;
  riempi(nodo("addestra-letture"), [
    tessera("`iter_num`", fmtIntero(ADD.iter_num) + " / " + fmtIntero(max),
      restanti !== null && ADD.in_corso ? I18N.t({ it: "ancora ~", en: "about " }) + fmtNum(restanti, 0) + " s" + I18N.t({ it: "", en: " left" }) : ""),
    tessera("`lr`", ADD.ultimo ? fmtNum(ADD.ultimo.lr, 5) : "—", "get_lr(iter_num)"),
    tessera("`loss`", ADD.ultimo ? fmtNum(ADD.ultimo.loss, 3) : "—", I18N.t({ it: "l'ultimo batch", en: "the last batch" })),
    tessera("`val loss`", ultimoEval ? fmtNum(ultimoEval.val, 3) : "—", "estimate_loss()", ultimoEval && ultimoEval.iter > 0),
    tessera("`‖g‖`", ADD.ultimo ? fmtNum(ADD.ultimo.norm, 3) : "—",
      I18N.t({ it: "tagliata a 1 se più grande", en: "clipped to 1 if larger" })),
    tessera(I18N.t({ it: "Ms per passo", en: "Ms per step" }), ADD.ms_per_iter ? fmtNum(ADD.ms_per_iter, 1) : "—",
      I18N.t({ it: "forward + backward + AdamW", en: "forward + backward + AdamW" })),
  ]);

  let testo;
  if (ADD.errore) {
    testo = I18N.t({
      it: "<b>Non riesco a scaricare il testo.</b> La pagina va aperta da un server web, non come file.",
      en: "<b>Could not download the text.</b> The page has to be opened from a web server, not as a file.",
    });
  } else if (!ADD.storia.length) {
    testo = I18N.t({
      it: "<b>Pronto.</b> Il primo punto della curva sarà intorno a 4,17: è la loss di un modello che dà 1/65 a ogni carattere.",
      en: "<b>Ready.</b> The first point on the curve will be around 4.17: the loss of a model that gives each character 1/65.",
    });
  } else if (ADD.in_corso) {
    testo = I18N.t({
      it: "<b>In corso.</b> La curva scende in fretta all'inizio — il modello impara quali caratteri sono frequenti — e poi " +
          "rallenta, quando deve imparare quali seguono quali.",
      en: "<b>Running.</b> The curve drops fast at first — the model learns which characters are common — and then slows " +
          "down, when it has to learn which ones follow which.",
    });
  } else if (ADD.iter_num >= max) {
    const gap = ultimoEval.val - ultimoEval.train;
    testo = I18N.t({
      it: "<b>Fatto: val loss " + fmtNum(ultimoEval.val, 3) + ".</b> " +
          (Math.abs(gap) < 0.1
            ? "Train e val sono vicine: con 7.760 parametri il modello è troppo piccolo per imparare a memoria. "
            : "La distanza fra train e val è di " + fmtNum(gap, 2) + ". ") +
          "Ora risali la pagina: l'attenzione, la loss e la generazione mostrano i pesi imparati.",
      en: "<b>Done: val loss " + fmtNum(ultimoEval.val, 3) + ".</b> " +
          (Math.abs(gap) < 0.1
            ? "Train and val are close: with 7,760 parameters the model is too small to memorise. "
            : "The gap between train and val is " + fmtNum(gap, 2) + ". ") +
          "Now scroll back up: attention, the loss and generation all show the learned weights.",
    });
  } else {
    testo = I18N.t({
      it: "<b>In pausa al passo " + fmtIntero(ADD.iter_num) + ".</b> I laboratori della pagina mostrano i pesi di adesso.",
      en: "<b>Paused at step " + fmtIntero(ADD.iter_num) + ".</b> The labs on the page show the current weights.",
    });
  }
  nodo("addestra-verdetto").innerHTML = testo;
  nodo("addestra-campione").textContent = ADD.campione || "—";
  etichettaBottoni();
}

nodo("addestra").addEventListener("click", avviaOFerma);
nodo("addestra-rapido").addEventListener("click", avviaOFerma);
nodo("azzera").addEventListener("click", azzera);
nodo("passi-chip").addEventListener("click", e => {
  const b = e.target.closest("button.chip");
  if (!b || b.disabled) return;
  train_config.max_iters = Number(b.dataset.iters);
  train_config.lr_decay_iters = train_config.max_iters;
  segnaChip(nodo("passi-chip"), "iters", train_config.max_iters);
  disegnaAddestramento();
});


/* --------------------------------------------------------------------------
   Parte quinta — generazione
   -------------------------------------------------------------------------- */

function leggiManopole() {
  const temperature = Number(nodo("temperatura").value) / 100;
  const k = Number(nodo("topk").value);
  return { temperature, top_k: k >= config.vocab_size ? null : k };
}

function disegnaGenerazione() {
  const { temperature, top_k } = leggiManopole();
  nodo("temperatura-out").textContent = fmtNum(temperature);
  nodo("topk-out").textContent = top_k === null ? "None" : String(top_k);

  const idx = GEN.idx;
  const { probs } = next_token_probs(STATO.model, idx, temperature, top_k);

  /* gli ultimi 40 caratteri: in verde la finestra idx_cond, tratteggiati
     quelli che il modello non vede più, in ambra l'ultimo generato */
  const da = Math.max(0, idx.length - 40);
  const gettoni = [];
  for (let n = da; n < idx.length; n++) {
    const dentro = n >= idx.length - config.block_size;
    const nuovo = GEN.nuovi > 0 && n === idx.length - 1;
    gettoni.push(gettone(itos[idx[n]], idx[n], nuovo ? "is-target" : dentro ? "is-ctx" : "is-out"));
  }
  riempi(nodo("gen-idx"), gettoni);

  const ordine = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 15);
  const contenitore = nodo("gen-barre");
  contenitore.textContent = "";
  ordine.forEach((j, n) => {
    const r = document.createElement("div");
    r.className = "bar" + (n === 0 ? " bar--pick" : "");
    const e = document.createElement("span");
    e.className = "bar__label";
    e.textContent = mostra(itos[j]);
    const traccia = document.createElement("div");
    traccia.className = "bar__track";
    const riempimento = document.createElement("div");
    riempimento.className = "bar__fill";
    /* scalata sul favorito: al passo zero le probabilità sono tutte
       attorno a 0,015 e su una scala da 0 a 1 non si vedrebbe niente */
    riempimento.style.setProperty("--quota", (100 * probs[j] / probs[ordine[0]]).toFixed(1) + "%");
    traccia.appendChild(riempimento);
    const v = document.createElement("div");
    v.className = "bar__value";
    v.textContent = fmtNum(probs[j], 3);
    r.append(e, traccia, v);
    contenitore.appendChild(r);
  });

  let entropia = 0, ammessi = 0;
  for (const p of probs) if (p > 0) { entropia -= p * Math.log2(p); ammessi++; }
  riempi(nodo("gen-letture"), [
    tessera(I18N.t({ it: "Entropia", en: "Entropy" }), fmtNum(entropia, 2) + " bit",
      I18N.t({ it: "log₂ 65 = 6,02: tutto uguale", en: "log₂ 65 = 6.02: all equal" })),
    tessera(I18N.t({ it: "Candidati", en: "Candidates" }), ammessi + " / " + config.vocab_size,
      I18N.t({ it: "con probabilità > 0", en: "with probability > 0" })),
    tessera(I18N.t({ it: "Favorito", en: "Favourite" }), "'" + mostra(itos[ordine[0]]) + "' " + fmtNum(probs[ordine[0]], 2),
      I18N.t({ it: "la barra ambra", en: "the amber bar" })),
    tessera("`len(idx)`", String(idx.length), "idx_cond = idx[:, -" + config.block_size + ":]"),
  ]);

  let testo;
  const taglio = top_k === null ? "" : I18N.t({
    it: " E con top_k = " + top_k + " restano in gioco solo i " + top_k + " candidati migliori: gli altri sono a −∞.",
    en: " And with top_k = " + top_k + " only the best " + top_k + " candidates stay in play: the rest are at −∞.",
  });
  if (ADD.iter_num === 0) {
    testo = I18N.t({
      it: "<b>Modello al passo zero.</b> La distribuzione è quasi piatta, e nessuna manopola la può raddrizzare: " +
          "temperature e top_k cambiano come si estrae, non quello che il modello sa.",
      en: "<b>Model at step zero.</b> The distribution is nearly flat, and no knob can fix it: " +
          "temperature and top_k change how you sample, not what the model knows.",
    });
  } else if (temperature < 0.85) {
    testo = I18N.t({
      it: "<b>Temperatura bassa.</b> La distribuzione si stringe sul favorito: il testo diventa ripetitivo, " +
          "perché il modello sceglie sempre la strada più battuta.",
      en: "<b>Low temperature.</b> The distribution narrows onto the favourite: the text turns repetitive, " +
          "because the model always takes the most beaten path.",
    });
  } else if (temperature > 1.15) {
    testo = I18N.t({
      it: "<b>Temperatura alta.</b> Le differenze fra i logits si appiattiscono e i caratteri improbabili " +
          "cominciano a uscire: il testo si sfalda.",
      en: "<b>High temperature.</b> The differences between logits flatten and unlikely characters start " +
          "coming out: the text falls apart.",
    });
  } else {
    testo = I18N.t({
      it: "<b>La distribuzione del modello, quasi intatta.</b> Con temperature vicina a 1 si estrae da " +
          "quello che il modello ha imparato, così com'è.",
      en: "<b>The model's own distribution, nearly untouched.</b> With temperature close to 1 you sample " +
          "from what the model learned, as it is.",
    });
  }
  nodo("gen-verdetto").innerHTML = testo + (ADD.iter_num === 0 ? "" : taglio);
  nodo("gen-testo").textContent = decode(idx);
}

/* Un giro del ciclo di generate(): la distribuzione, l'estrazione, la
   coda. Il +100 chiama generate() vera e propria. */
nodo("genera-1").addEventListener("click", () => {
  const { temperature, top_k } = leggiManopole();
  const { probs } = next_token_probs(STATO.model, GEN.idx, temperature, top_k);
  GEN.idx.push(multinomial(probs, GEN.rng));
  GEN.nuovi++;
  disegnaGenerazione();
});
nodo("genera-100").addEventListener("click", () => {
  const { temperature, top_k } = leggiManopole();
  GEN.idx = generate(STATO.model, GEN.idx, 100, temperature, top_k, GEN.rng);
  GEN.nuovi += 100;
  disegnaGenerazione();
});
nodo("genera-azzera").addEventListener("click", () => {
  GEN.idx = encode(STATO.prompt);
  GEN.nuovi = 0;
  disegnaGenerazione();
});
["temperatura", "topk"].forEach(id => nodo(id).addEventListener("input", disegnaGenerazione));


/* --------------------------------------------------------------------------
   7. AVVIO
   -------------------------------------------------------------------------- */

riempi(nodo("vocabolario"), chars.map((ch, i) => gettone(ch, i)));
disegnaTesto();
disegnaBatch();
disegnaMappa();
disegnaLeggenda();
disegnaScala();
disegnaGelu();
disegnaLR();
disegnaGruppi();
ricalcola();
disegnaAddestramento();
