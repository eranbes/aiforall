// 06-attention.js — level 4: where context enters the number.

import { Plot, color, div, seq } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, checkbox, select, stats, legend, fmt } from '../ui.js';
import { Mat, matmul, matmulT, softmaxRows, causalMask, dot, rope } from '../engine/tensor.js';

/* ------------------------------------------------------------------
   A hand-built toy. The embeddings below are WRITTEN BY HAND so the
   attention pattern is legible; a trained model discovers patterns like
   these, it is not given them. The mechanism is exactly the real one.
------------------------------------------------------------------ */

const DIMS = ['animate', 'thing', 'place', 'action', 'pronoun', 'warm', 'the', 'past'];

const TOKENS = [
  ['the',     [0, 0, 0, 0, 0, 0, 1, 0]],
  ['cat',     [1, 1, 0, 0, 0, 0.3, 0, 0]],
  ['sat',     [0, 0, 0, 1, 0, 0, 0, 1]],
  ['on',      [0, 0, 0.6, 0, 0, 0, 0, 0]],
  ['the',     [0, 0, 0, 0, 0, 0, 1, 0]],
  ['mat',     [0, 1, 1, 0, 0, 0, 0, 0]],
  ['because', [0, 0, 0, 0, 0, 0, 0, 0]],
  ['it',      [0, 0, 0, 0, 1, 0, 0, 0]],
  ['was',     [0, 0, 0, 1, 0, 0, 0, 1]],
  ['warm',    [0, 0, 0, 0, 0, 1, 0, 0]],
];

const N = TOKENS.length;
const D = DIMS.length;
const E = Mat.from(TOKENS.map((t) => t[1]));
const WORDS = TOKENS.map((t) => t[0]);

/** W_q: a pronoun asks "who is animate / a thing?"; everything else asks weakly for itself. */
const WQ = (() => {
  const m = Mat.zeros(D, D);
  for (let i = 0; i < D; i++) m.set(i, i, 0.2);           // weak self-similarity
  m.set(4, 0, 4.0);                                       // pronoun -> animate
  m.set(4, 1, 3.5);                                       // pronoun -> thing
  m.set(5, 0, 1.5);                                       // warm    -> animate
  return m;
})();
/** W_k: keys are (almost) the embedding itself. */
const WK = (() => {
  const m = Mat.zeros(D, D);
  for (let i = 0; i < D; i++) m.set(i, i, 1);
  return m;
})();
const WV = WK;

const Q = matmul(E, WQ);
const K = matmul(E, WK);
const V = matmul(E, WV);

/** Positional head: one rotary pair per token, rotated by its position. */
function ropeQK(base) {
  const q = [], k = [];
  for (let i = 0; i < N; i++) {
    q.push(rope(Float32Array.from([1, 0]), i, base));
    k.push(rope(Float32Array.from([1, 0]), i, base));
  }
  return { q, k };
}

export default {
  id: '06-attention',
  num: '06',
  group: 'inside the model',
  title: 'Attention: where context enters',
  blurb: 'An n-gram sees two tokens back. Attention lets every token read every earlier token, and decide for itself which ones matter. It is four operations, and you can watch all four.',

  render(root) {
    const state = {
      head: 'content',
      scale: true,
      mask: true,
      soft: true,
      query: 7,            // "it"
      base: 10,            // rotary base for the positional head
    };

    /* ---------- compute the four stages ---------- */
    const compute = () => {
      const S = Mat.zeros(N, N);
      if (state.head === 'content') {
        const raw = matmulT(Q, K);
        S.d.set(raw.d);
      } else {
        const { q, k } = ropeQK(state.base);
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) S.set(i, j, dot(q[i], k[j]));
      }
      const scaled = S.clone();
      if (state.scale) for (let i = 0; i < scaled.d.length; i++) scaled.d[i] /= Math.sqrt(D);

      const masked = scaled.clone();
      if (state.mask) causalMask(masked);

      const P = masked.clone();
      if (state.soft) softmaxRows(P);

      const out = matmul(P, V);
      return { S, scaled, masked, P, out };
    };

    let data = compute();

    /* ---------- the score matrix ---------- */
    const heat = new Plot({
      width: 430, height: 390, pad: { l: 74, r: 14, t: 66, b: 16 },
      xlim: [0, N], ylim: [0, N],
    });

    heat.onDraw((p) => {
      const M = state.soft ? data.P : data.masked;
      const m = M.toArray().map((r) => r.map((v) => (v < -1e8 ? NaN : v)));
      const finite = m.flat().filter((v) => !Number.isNaN(v));
      const max = Math.max(1e-9, ...finite.map(Math.abs));

      const cw = p.iw / N, ch = p.ih / N;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const v = m[i][j];
          const x = p.pad.l + j * cw, y = p.pad.t + i * ch;
          p.ctx.fillStyle = Number.isNaN(v)
            ? color('surface')
            : (state.soft ? seq(v / max) : div(v / max));
          p.ctx.fillRect(x, y, cw - 1, ch - 1);
          if (Number.isNaN(v)) {
            p.ctx.strokeStyle = color('border');
            p.ctx.lineWidth = 1;
            p.ctx.beginPath();
            p.ctx.moveTo(x + 3, y + 3); p.ctx.lineTo(x + cw - 4, y + ch - 4);
            p.ctx.stroke();
          }
        }
      }
      // highlight the selected query row
      p.ctx.strokeStyle = color('accent');
      p.ctx.lineWidth = 2;
      p.ctx.strokeRect(p.pad.l - 1, p.pad.t + state.query * ch - 1, p.iw + 2, ch);

      p.ctx.font = '10px ui-monospace, Menlo, monospace';
      p.ctx.fillStyle = color('muted');
      p.ctx.textAlign = 'right'; p.ctx.textBaseline = 'middle';
      for (let i = 0; i < N; i++) {
        p.ctx.fillStyle = i === state.query ? color('accent') : color('muted');
        p.ctx.fillText(WORDS[i], p.pad.l - 6, p.pad.t + i * ch + ch / 2);
      }
      p.ctx.save();
      p.ctx.textAlign = 'left'; p.ctx.textBaseline = 'middle';
      for (let j = 0; j < N; j++) {
        p.ctx.save();
        p.ctx.translate(p.pad.l + j * cw + cw / 2, p.pad.t - 6);
        p.ctx.rotate(-Math.PI / 4);
        p.ctx.fillStyle = color('muted');
        p.ctx.fillText(WORDS[j], 0, 0);
        p.ctx.restore();
      }
      p.ctx.restore();
      p.text(p.pad.l - 68, p.pad.t - 30, 'query ↓', { color: 'muted', size: 10, px: true, align: 'left' });
      p.text(p.pad.l, p.pad.t - 46, 'key →', { color: 'muted', size: 10, px: true, align: 'left' });
    });

    heat.interact({
      down: (_, ev) => {
        const { x, y } = heat.atPx(ev);
        const i = Math.floor((y - heat.pad.t) / (heat.ih / N));
        if (i >= 0 && i < N && x > heat.pad.l) { state.query = i; refresh(); }
      },
      hover: (_, ev) => { heat.canvas.style.cursor = ev ? 'pointer' : 'default'; },
    });

    /* ---------- the selected row, as weights and as an output ---------- */
    const rowPlot = new Plot({
      width: 430, height: 200, pad: { l: 44, r: 12, t: 26, b: 48 },
      xlim: [0, N], ylim: [0, 1], ylabel: 'weight',
    });
    rowPlot.onDraw((p) => {
      const w = Array.from(data.P.row(state.query)).map((v) => (v < -1e8 ? 0 : v));
      p.ylim = [Math.min(0, ...w) * 1.1, Math.max(0.1, ...w) * 1.25];
      p.axes({ xticks: 0, yticks: 4, zero: true });
      p.bars(w, { color: w.map((v, j) => (j > state.query && state.mask ? 'border' : (j === state.query ? 'c4' : 'c3'))), labels: WORDS });
      p.text(p.pad.l, 12, `row "${WORDS[state.query]}" — ${state.soft ? 'after softmax, sums to 1' : 'raw scores'}`,
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    const outPlot = new Plot({
      width: 430, height: 190, pad: { l: 74, r: 12, t: 24, b: 40 },
      xlim: [0, D], ylim: [0, 1],
    });
    outPlot.onDraw((p) => {
      const o = Array.from(data.out.row(state.query));
      const inp = Array.from(E.row(state.query));
      p.ylim = [Math.min(0, ...o, ...inp) - 0.1, Math.max(0.6, ...o, ...inp) * 1.2];
      p.axes({ xticks: 0, yticks: 4, zero: true });
      p.bars(inp, { color: 'border', width: 0.76, labels: DIMS });
      p.bars(o, { color: 'c1', width: 0.42 });
      p.text(p.pad.l, 12, 'grey = the token\'s own vector · blue = what attention returns',
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    const read = stats([
      { k: 'query token', v: '—' }, { k: 'attends most to', v: '—' },
      { k: 'that weight', v: '—' }, { k: 'visible keys', v: '—' },
    ]);

    const refresh = () => {
      data = compute();
      const w = Array.from(data.P.row(state.query)).map((v) => (v < -1e8 ? 0 : v));
      let bi = 0;
      for (let j = 0; j < N; j++) if (w[j] > w[bi]) bi = j;
      read.set('query token', `"${WORDS[state.query]}" @ ${state.query}`);
      read.set('attends most to', `"${WORDS[bi]}" @ ${bi}`);
      read.set('that weight', state.soft ? fmt.pct(w[bi], 1) : fmt.n(w[bi], 2));
      read.set('visible keys', state.mask ? `${state.query + 1} of ${N}` : `${N} of ${N}`);
      heat.render(); rowPlot.render(); outPlot.render(); scalePlot.render(); ropePlot.render();
    };

    root.append(card('The whole mechanism, on one sentence',
      'Click any row of the matrix to choose the query token. Row <i>i</i>, column <i>j</i> is "how much does token <i>i</i> want to read token <i>j</i>".',
      row(
        el('div', {}, heat.canvas),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          select({
            label: 'head', value: 'content',
            options: [{ v: 'content', t: 'content head — "what is like me?"' }, { v: 'position', t: 'positional head — "what is near me?"' }],
            onChange: (v) => { state.head = v; refresh(); },
          }),
          checkbox({ label: 'scale by 1/√d', checked: true, onChange: (v) => { state.scale = v; refresh(); } }),
          checkbox({ label: 'causal mask (hide the future)', checked: true, onChange: (v) => { state.mask = v; refresh(); } }),
          checkbox({ label: 'softmax the rows', checked: true, onChange: (v) => { state.soft = v; refresh(); } }),
          read,
          note(`The embeddings here are <b>hand-written</b>, one column per named feature, so the pattern is
                readable. A trained model finds patterns like these itself — the mechanism below is
                exactly the real one.`)))));

    root.append(card('One row, in full',
      'Attention output for the selected token is a weighted average of all visible value vectors. Nothing more exotic than that.',
      row(
        el('div', {}, rowPlot.canvas),
        el('div', {}, outPlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 260px' } },
          el('div', { class: 'eq', html: 'scores = <span class="hl">Q·Kᵀ</span> / √d' }),
          el('div', { class: 'eq', html: 'weights = softmax( mask(scores) )' }),
          el('div', { class: 'eq', html: 'out = <span class="hl">weights · V</span>' }),
          prose(`
            <p>Select <b>"it"</b> and look at the output: it has acquired the <code class="inline">animate</code>
            and <code class="inline">thing</code> features it did not have on its own. The pronoun has
            <i>read</i> its antecedent.</p>
            <p>That is the entire point. The vector at position 7 stops meaning "it" and starts meaning
            "it, the cat". Context has entered the number.</p>
          `)))));

    /* ---------- why 1/sqrt(d) ---------- */
    const scalePlot = new Plot({
      width: 430, height: 240, pad: { l: 50, r: 14, t: 16, b: 42 },
      xlim: [1, 256], ylim: [0, 1], xlabel: 'head dimension d', ylabel: 'largest softmax weight',
    });
    scalePlot.onDraw((p) => {
      // random q,k of dimension d have dot products with std ~ sqrt(d).
      // Without the 1/sqrt(d) scale the softmax saturates as d grows.
      const curve = (scaled) => {
        const pts = [];
        for (let d = 2; d <= 256; d += 2) {
          // expected max score among 10 keys ~ sqrt(d) * E[max of 10 normals] ≈ sqrt(d) * 1.54
          const spread = Math.sqrt(d) * 1.54 * (scaled ? 1 / Math.sqrt(d) : 1);
          // softmax of {spread, 0, 0, ... } over 10 keys
          const e = Math.exp(spread);
          pts.push([d, e / (e + 9)]);
        }
        return pts;
      };
      p.axes({ xticks: 5, yticks: 5, zero: false });
      p.line(curve(false), { color: 'c2', width: 2.5 });
      p.line(curve(true), { color: 'c3', width: 2.5 });
      p.text(200, 0.97, 'unscaled', { color: 'c2', size: 11 });
      p.text(200, 0.3, 'scaled by 1/√d', { color: 'c3', size: 11 });
    });

    root.append(card('Why divide by √d',
      'Not a fudge factor. Dot products of d-dimensional vectors grow like √d, so without the divide, wider heads drive softmax into a one-hot spike — and a spike has almost no gradient to learn from.',
      row(
        el('div', {}, scalePlot.canvas),
        grow(prose(`
          <p>Each score is a sum of <i>d</i> products. Sum <i>d</i> independent terms and the spread of the
          total grows like <b>√d</b>. At d = 256 the raw scores are ~16× wider than at d = 1.</p>
          <p>Feed wide scores to softmax and the largest wins outright: weights go to 1 and 0. The layer
          stops averaging and starts selecting a single key, and — worse during training — the gradient
          through a saturated softmax is almost zero.</p>
          <p>Dividing by √d holds the spread constant as heads get wider. Toggle <b>scale by 1/√d</b> above
          and watch the matrix contrast jump.</p>`),
          note(`In TensorFold's reference implementation the scale is a <b>parameter</b>, not computed
                inside attention: the caller passes it. <code class="inline">forward.zig:209</code> passes
                <code class="inline">1.0/16.0</code> — which is 1/√256, the head dimension of Qwen3.5-2B.`)))));

    /* ---------- RoPE ---------- */
    const ropePlot = new Plot({
      width: 420, height: 280, pad: { l: 40, r: 14, t: 14, b: 34 },
      xlim: [-1.3, 1.3], ylim: [-1.3, 1.3],
    });
    const posI = slider({ label: 'position of the query', min: 0, max: 12, step: 1, value: 5, fmt: String, onInput: () => ropePlot.render() });
    const posJ = slider({ label: 'position of the key', min: 0, max: 12, step: 1, value: 2, fmt: String, onInput: () => ropePlot.render() });
    const ropeRead = stats([{ k: 'relative distance', v: '—' }, { k: 'q · k', v: '—' }]);

    ropePlot.onDraw((p) => {
      const i = posI.get(), j = posJ.get();
      const base = state.base;
      const qv = rope(Float32Array.from([1, 0]), i, base);
      const kv = rope(Float32Array.from([1, 0]), j, base);
      p.axes({ xticks: 4, yticks: 4, zero: true, grid: true });
      p.circle(0, 0, p.X(1) - p.X(0), { color: 'border', fill: false, width: 1 });
      p.arrow(0, 0, qv[0], qv[1], { color: 'c1', width: 3 });
      p.arrow(0, 0, kv[0], kv[1], { color: 'c5', width: 3 });
      p.text(qv[0] * 1.15, qv[1] * 1.15, `q @ ${i}`, { color: 'c1', size: 11 });
      p.text(kv[0] * 1.15, kv[1] * 1.15, `k @ ${j}`, { color: 'c5', size: 11 });
      const d = dot(qv, kv);
      ropeRead.set('relative distance', String(i - j));
      ropeRead.set('q · k', fmt.n(d, 4));
    });

    root.append(card('RoPE: position as a rotation',
      'Attention as written has no idea what order the tokens are in — it is a weighted average, and averages do not care about order. Rotary embeddings fix that by rotating each vector by an angle proportional to its position.',
      row(
        el('div', {}, ropePlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          posI, posJ, ropeRead,
          prose(`
            <p>Rotate the query by <i>i</i>θ and the key by <i>j</i>θ. Their dot product depends only on
            the angle <b>between</b> them, which is <b>(i − j)θ</b> — the <i>relative</i> distance.</p>
            <p>Move both sliders together, keeping the gap fixed: <code class="inline">q · k</code> does not
            change. That is the property the whole scheme exists for. A model trained on short sequences
            has at least a chance on longer ones, because nothing depends on absolute position.</p>`),
          note(`Switch the head above to <b>positional</b> to see this drive real attention: with no content
                information at all, tokens attend to their neighbours purely through rotation.`)))));

    /* ---------- what attention must remember ---------- */
    const bytesOf = (n) => {
      if (n < 1024) return `${fmt.n(n, 0)} B`;
      if (n < 1024 ** 2) return `${fmt.n(n / 1024, 1)} KiB`;
      if (n < 1024 ** 3) return `${fmt.n(n / 1024 ** 2, 1)} MiB`;
      return `${fmt.n(n / 1024 ** 3, 2)} GiB`;
    };
    const kv = { context: 32768, layers: 6, kvHeads: 2, headDim: 256, dtype: 2 };
    const kvRead = stats([
      { k: 'bytes / token', v: '—' }, { k: 'context', v: '—' }, { k: 'KV payload', v: '—' },
    ]);
    const kvTable = el('table', {});
    const renderKV = () => {
      const perToken = kv.layers * 2 * kv.kvHeads * kv.headDim * kv.dtype;
      kvRead.set('bytes / token', `${fmt.n(perToken, 0)} B`);
      kvRead.set('context', fmt.n(kv.context, 0));
      kvRead.set('KV payload', bytesOf(perToken * kv.context));
      kvTable.replaceChildren(
        el('thead', {}, el('tr', {}, el('th', { text: 'tokens' }), el('th', { text: 'KV payload' }))),
        el('tbody', {}, [4096, 32768, 131072, 1000000].map((n) => el('tr', {},
          el('td', { text: fmt.n(n, 0) }),
          el('td', { text: bytesOf(perToken * n) })))));
    };
    renderKV();

    root.append(card('What attention has to remember',
      'Every key and value you just watched being computed must be <b>kept</b> — the next token attends to them too. Recomputing them for the whole history every step would make generation quadratic, so they are cached.',
      row(
        grow(
          prose(`<p>That is the <b>KV cache</b>, and it is the single largest runtime cost of a long
            conversation. Each attention layer stores a key and a value per token, per KV head:</p>`),
          el('div', { class: 'eq', html: 'bytes/token = attention layers × 2 × kv heads × head_dim × dtype<br><b>6 × 2 × 2 × 256 × 2 = 12,288 bytes/token</b>' }),
          el('div', { class: 'sub', style: { marginTop: '10px' }, text: 'Qwen3.5-2B · verified against state.zig:204-208 and its cache-size test' }),
          kvTable,
          note(`<b>The factor of 2 is K and V.</b> Queries are not cached — a query belongs to the token being
                generated right now and is thrown away. Keys and values belong to the past, which is why GQA
                (two KV heads for eight query heads) cuts this by four.`)),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          slider({ label: 'context length', min: 1024, max: 1000000, step: 1024, value: kv.context, fmt: (v) => fmt.n(v, 0), onInput: (v) => { kv.context = v; renderKV(); } }),
          slider({ label: 'attention layers', min: 1, max: 24, step: 1, value: kv.layers, fmt: String, onInput: (v) => { kv.layers = v; renderKV(); } }),
          slider({ label: 'KV heads', min: 1, max: 16, step: 1, value: kv.kvHeads, fmt: String, onInput: (v) => { kv.kvHeads = v; renderKV(); } }),
          slider({ label: 'head dimension', min: 32, max: 512, step: 32, value: kv.headDim, fmt: String, onInput: (v) => { kv.headDim = v; renderKV(); } }),
          slider({ label: 'bytes per number', min: 1, max: 4, step: 1, value: kv.dtype, fmt: (v) => `${v} B`, onInput: (v) => { kv.dtype = v; renderKV(); } }),
          kvRead,
          note(`<b>Only 6 of this model's 24 layers pay it.</b> The other 18 (lesson 17) keep a fixed-size
                state instead, which is why this cache is smaller than the layer count suggests.`)))));

    /* ---------- in the engine ---------- */
    root.append(card('In the real engine', null, row(
      grow(
        el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/core/attention_ref.zig:5-29 — the ground truth every kernel is checked against</span>\n` +
          `var m: f64 = -std.math.inf(f64);\n` +
          `for (keys) |j| {\n` +
          `    var s: f64 = 0;\n` +
          `    for (0..d) |e| s += q[e] * k[j * d + e];   <span class="cm">// dot product</span>\n` +
          `    m = @max(m, s * scale);                    <span class="cm">// running max</span>\n` +
          `}\n` +
          `var l: f64 = 0;\n` +
          `for (keys) |j| {\n` +
          `    ...\n` +
          `    const p = @exp(s * scale - m);             <span class="cm">// shifted exp</span>\n` +
          `    l += p;\n` +
          `    for (0..d) |e| out[e] += p * v[j * d + e]; <span class="cm">// weighted sum of V</span>\n` +
          `}\n` +
          `for (out, tol) |*o, *t| o.* /= l;              <span class="cm">// normalise</span>` }),
        prose(`
          <p>Twenty lines, in f64, deliberately slow. Every Metal and CUDA attention kernel in the repo is
          validated against this function — the same arithmetic you just watched, reordered until it fits
          in registers.</p>`)),
      el('div', { class: 'col', style: { flex: '1 1 300px' } },
        note(`<b>There is no mask in the reference.</b> Causality is expressed as
              <code class="inline">keys</code>, the list of visible positions — the mask is "which keys you
              hand in". A mask of −∞ is an implementation convenience, not the concept.`),
        note(`<b>Flash attention is this loop, fused.</b> The reference makes two passes: one to find the
              max, one to accumulate. The kernel cannot afford to keep the scores around, so it keeps a
              running max and a running sum and <i>rescales</i> what it already accumulated whenever a
              bigger score shows up (<code class="inline">kernels/metal/ops/sdpa.metal:10-19</code>).
              Same answer, one pass, nothing but registers.`),
        note(`<b>GQA: 8 query heads, 2 key/value heads.</b> Qwen3.5-2B gives four query heads to each KV
              head (<code class="inline">kv = head / gqa</code>). The KV cache shrinks fourfold, and four
              queries share one fetch of K and V — which matters because decoding is bound by memory
              bandwidth, not arithmetic.`),
        note(`<b>RoPE is fused into the cache write.</b> The engine never stores an unrotated key: rotation
              happens as K is written into the KV cache
              (<code class="inline">layout.metal:47-73</code>), with θ = 1e7 inline and only 64 of the 256
              head dimensions rotating at all.`)))));

    refresh();
  },
};
