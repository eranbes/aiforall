// 04-logits.js — level 6: one hidden vector times one output matrix.

import { Plot, div } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, select, legend, fmt, clamp } from '../ui.js';
import { dot, cosine } from '../engine/tensor.js';

const QUERY = { x: 1.35, y: 0.78 };
const TOY_TOKENS = ['▁the', '▁cat', '▁runs', '▁blue', '▁model', '▁reads', '▁is', '▁a', '▁vector', '▁matrix', '▁and', '▁to'];
// A deliberately small, fixed matrix: enough structure to make the geometry visible, not a trained model.
const W = [
  [0.82, 0.28, 0.08, 0.22, -0.10, 0.14],
  [-0.16, 0.94, 0.20, 0.08, -0.15, 0.25],
  [-0.22, 0.76, 0.58, -0.08, 0.12, 0.10],
  [0.14, -0.12, 0.06, 0.92, 0.28, -0.20],
  [-0.08, 0.10, -0.16, 0.14, 0.98, 0.18],
  [0.05, 0.24, 0.68, -0.14, 0.22, 0.74],
  [0.62, -0.10, 0.18, -0.04, 0.38, -0.12],
  [0.54, 0.22, -0.18, 0.10, -0.10, 0.06],
  [-0.10, 0.05, 0.42, 0.18, 0.68, 0.54],
  [0.12, -0.14, 0.36, 0.54, 0.74, -0.08],
  [0.30, 0.26, 0.10, 0.06, 0.08, 0.34],
  [-0.24, 0.18, -0.12, 0.16, 0.06, 0.40],
];
const H0 = [0.82, 0.48, 0.34, 0.22, 0.44, 0.30];
const EMBED_TOKENS = ['▁cat', '▁dog', '▁kitten', '▁puppy', '▁run', '▁walk', '▁jump', '▁blue', '▁red', '▁green', '▁ocean', '▁river', '▁model', '▁code', '▁matrix', '▁learn'];
const EMBED_POINTS = [
  [1.20, 1.50], [1.40, 1.20], [0.88, 1.62], [1.18, 1.10],
  [-0.62, 0.92], [-0.30, 0.56], [-0.82, 1.28],
  [-1.22, -0.18], [-0.92, -0.42], [-1.42, -0.08],
  [-1.78, -1.04], [-1.40, -1.28],
  [0.20, -1.50], [0.62, -1.32], [0.10, -1.00], [0.70, -0.80],
].map(([x, y]) => ({ x, y }));

const signed = (v, d = 2) => (v >= 0 ? '+' : '') + fmt.n(v, d);
const comma = (n) => Math.round(n).toLocaleString('en-US');

export default {
  id: '04-logits',
  num: '04',
  group: 'inside the model',
  title: 'Where do the logits come from?',
  blurb: 'At the very end, a hidden vector meets every row of one enormous matrix. Make the dot product concrete, then watch that one operation produce a score for every possible next token.',

  render(root) {
    /* ---------- dot product as geometry ---------- */
    const q = { ...QUERY };
    const v = { x: 0.42, y: 1.18 };
    const dotStats = stats([
      { k: 'q · v', v: '—' },
      { k: 'cosine similarity', v: '—' },
      { k: 'projection of v onto q', v: '—' },
    ]);
    const dotPlot = new Plot({
      width: 500, height: 330, xlim: [-2.1, 2.1], ylim: [-2.1, 2.1], equal: true,
      pad: { l: 42, r: 16, t: 16, b: 34 }, xlabel: 'dimension 1', ylabel: 'dimension 2',
    });
    let grabVector = false;
    const setVector = (x, y) => { v.x = clamp(x, -1.9, 1.9); v.y = clamp(y, -1.9, 1.9); dotPlot.render(); };
    dotPlot.onDraw((p) => {
      const qn = Math.hypot(q.x, q.y) ** 2;
      const d = dot([q.x, q.y], [v.x, v.y]);
      const scale = d / qn;
      const proj = { x: scale * q.x, y: scale * q.y };
      p.axes({ xticks: 5, yticks: 5, grid: true });
      p.arrow(0, 0, q.x, q.y, { color: 'c2', width: 3 });
      p.text(q.x * 1.05, q.y * 1.05, 'q (fixed)', { color: 'c2', align: 'left' });
      p.arrow(0, 0, v.x, v.y, { color: 'accent', width: 3 });
      p.text(v.x * 1.03, v.y * 1.03, 'v (drag)', { color: 'accent', align: 'left' });
      p.line([[v.x, v.y], [proj.x, proj.y]], { color: 'muted', width: 1.5, dash: [4, 4] });
      p.arrow(0, 0, proj.x, proj.y, { color: 'c4', width: 2, dash: [3, 3] });
      p.circle(v.x, v.y, 7, { color: 'accent', fill: true });
      p.text(proj.x, proj.y, '  projection', { color: 'c4', align: 'left' });
      dotStats.set('q · v', signed(d));
      dotStats.set('cosine similarity', signed(cosine([q.x, q.y], [v.x, v.y])));
      dotStats.set('projection of v onto q', `(${fmt.n(proj.x)}, ${fmt.n(proj.y)})`);
    });
    dotPlot.interact({
      down: (pt) => {
        grabVector = Math.hypot(pt.x - v.x, pt.y - v.y) < 0.38;
        if (grabVector) setVector(pt.x, pt.y);
      },
      move: (pt) => { if (grabVector) setVector(pt.x, pt.y); },
      up: () => { grabVector = false; },
    });
    const dotButtons = btns(
      button('aligned', () => setVector(q.x * 0.9, q.y * 0.9), { ghost: true }),
      button('perpendicular', () => setVector(-q.y * 0.9, q.x * 0.9), { ghost: true }),
      button('opposed', () => setVector(-q.x * 0.9, -q.y * 0.9), { ghost: true }),
    );
    root.append(card('A dot product is a similarity score',
      'The dot product multiplies matching coordinates and adds them: <b>q · v = q₁v₁ + q₂v₂</b>. Drag the tip of <span style="color:var(--accent)">v</span>. The dashed segment is its projection onto <span style="color:var(--c2)">q</span>.',
      row(grow(dotPlot.canvas, dotButtons), el('div', { class: 'col', style: { flex: '0 1 260px' } },
        dotStats,
        note('<b>Read the signs.</b> Aligned vectors have a large positive dot product. Perpendicular vectors contribute zero because their projection is zero. Opposed vectors have a negative score. Cosine divides away the lengths, leaving only the angle.')))));

    /* ---------- a small output projection ---------- */
    const h = H0.slice();
    const logits = () => W.map((r) => dot(h, r));
    const hiddenPlot = new Plot({ width: 370, height: 245, xlim: [0, 6], ylim: [-1.35, 1.35], pad: { l: 38, r: 12, t: 14, b: 34 }, ylabel: 'hᵢ' });
    const matrixPlot = new Plot({ width: 470, height: 300, xlim: [0, 6], ylim: [-0.5, TOY_TOKENS.length - 0.5], pad: { l: 72, r: 12, t: 30, b: 18 }, plain: true });
    const logitsPlot = new Plot({ width: 470, height: 300, xlim: [0, TOY_TOKENS.length], ylim: [-1, 2], pad: { l: 42, r: 12, t: 14, b: 48 }, ylabel: 'logit' });
    const logitStats = stats([{ k: 'largest logit', v: '—' }, { k: 'winner', v: '—' }, { k: 'one row dot product', v: '—' }]);
    const hControls = h.map((_, i) => slider({
      label: `h<sub>${i + 1}</sub>`, min: -1.2, max: 1.2, step: 0.01, value: h[i], fmt: (x) => fmt.n(x),
      onInput: (x) => { h[i] = x; renderProjection(); renderLookup?.(); }, width: '100%',
    }));
    const renderProjection = () => {
      const ls = logits();
      const winner = ls.reduce((best, x, i) => x > ls[best] ? i : best, 0);
      hiddenPlot.render(); matrixPlot.render(); logitsPlot.ylim = [Math.min(-1, Math.min(...ls) - 0.25), Math.max(1, Math.max(...ls) + 0.25)]; logitsPlot.render();
      logitStats.set('largest logit', fmt.n(ls[winner]));
      logitStats.set('winner', TOY_TOKENS[winner], 'accent');
      logitStats.set('one row dot product', `${TOY_TOKENS[winner]} · h = ${fmt.n(ls[winner])}`);
    };
    hiddenPlot.onDraw((p) => { p.axes({ xticks: 6, yticks: 5, xfmt: (x) => `h${x + 1}` }); p.bars(h, { color: 'c3', baseline: 0, width: 0.66, fmtv: (x) => fmt.n(x) }); });
    matrixPlot.onDraw((p) => {
      p.matrix(W, { x0: 0, x1: 6, y0: -0.5, y1: TOY_TOKENS.length - 0.5, cmap: div, labels: { rows: TOY_TOKENS, cols: ['1', '2', '3', '4', '5', '6'] }, text: true, fmtv: (x) => fmt.n(x, 1), max: 1 });
      p.text(3, TOY_TOKENS.length - 0.05, 'W  (one row per token)', { color: 'muted', size: 12, baseline: 'bottom' });
    });
    logitsPlot.onDraw((p) => {
      const ls = logits();
      const winner = ls.reduce((best, x, i) => x > ls[best] ? i : best, 0);
      p.axes({ xticks: 6, yticks: 5, xfmt: () => '', yfmt: (x) => fmt.n(x, 1) });
      p.bars(ls, { color: 'c5', highlight: winner, labels: TOY_TOKENS, baseline: 0, width: 0.72, fmtv: (x) => fmt.n(x) });
    });
    root.append(card('Logits = hidden state · every output row',
      'This is a complete toy output layer. The six sliders are a hidden state <b>h</b>; each row of <b>W</b> is one token vector. The model computes <b>logits[i] = h · W[i]</b>, then a later softmax can turn those scores into probabilities.',
      el('div', { class: 'sub', style: { marginBottom: '8px' }, text: 'change h — the highlighted bar is the token with the highest dot product' }),
      row(grow(...hControls), el('div', { class: 'col', style: { flex: '0 1 390px' } }, hiddenPlot.canvas, logitStats)),
      row(grow(matrixPlot.canvas), grow(logitsPlot.canvas)),
      legend([{ c: 'c3', label: 'hidden coordinates' }, { c: 'c5', label: 'logit for each token' }, { c: 'accent', label: 'winner' }])));
    renderProjection();

    /* ---------- tied input/output matrix ---------- */
    const selected = { i: 4 };
    const pick = select({
      label: 'choose a token row', options: TOY_TOKENS.map((t, i) => ({ v: String(i), t })), value: String(selected.i),
      onChange: (value) => { selected.i = Number(value); renderLookup(); },
    });
    const lookup = el('div', { class: 'code', style: { minHeight: '76px' } });
    const lookupStats = stats([{ k: 'W[token] · h', v: '—' }, { k: 'row length', v: '6 numbers' }]);
    const renderLookup = () => {
      const i = selected.i;
      const rowValues = W[i].map((x, j) => `w${j + 1}=${signed(x)}`).join('   ');
      lookup.textContent = `input lookup:  embedding[${i}] = W[${i}]  =  [ ${rowValues} ]\noutput score:  logit[${i}] = h · W[${i}]  =  ${fmt.n(dot(h, W[i]))}`;
      lookupStats.set('W[token] · h', fmt.n(dot(h, W[i])));
    };
    root.append(card('The embedding table is the same matrix, read the other way',
      'For an input token, the model <b>looks up one row</b>: <code>embedding[id] = W[id]</code>. At the output, it reads that row as a candidate and takes a dot product with h. Tying these weights means the geometry used to read a token is also the geometry used to score it.',
      row(grow(pick, lookup), el('div', { class: 'col', style: { flex: '0 1 250px' } }, lookupStats, note('The row is not a mysterious class label. It is a vector. Similar hidden states score nearby rows similarly.'))),
      note('<b>One matrix, two directions.</b> Input: integer → one row. Output: one hidden vector → all rows. The expensive part is the second direction because every vocabulary row participates.')));
    renderLookup();

    /* ---------- scale ---------- */
    const hidden = 2048;
    const vocab = 248320;
    const intermediate = 6144;
    const layers = 24;
    const outputParams = hidden * vocab;
    const outputBytes4 = outputParams / 2;
    const oneMlp = 3 * hidden * intermediate;
    const allMlp = layers * oneMlp;
    const scaleStats = stats([
      { k: 'output parameters', v: `${comma(outputParams)}  (${fmt.n(outputParams / 1e6, 1)}M)` },
      { k: 'MACs for one token', v: comma(outputParams) },
      { k: '4-bit weight bytes', v: `${comma(outputBytes4)}  (${fmt.n(outputBytes4 / (1024 * 1024), 1)} MiB)` },
    ]);
    const scalePlot = new Plot({ width: 510, height: 240, xlim: [0, 3], ylim: [0, allMlp * 1.08], pad: { l: 54, r: 14, t: 16, b: 54 }, ylabel: 'parameters', plain: true });
    scalePlot.onDraw((p) => {
      p.axes({ xticks: 3, yticks: 4, xfmt: () => '', yfmt: (x) => `${fmt.n(x / 1e6, 0)}M` });
      p.bars([oneMlp, outputParams, allMlp], { color: ['c3', 'accent', 'c4'], labels: ['one MLP block', 'final projection', '24 MLP blocks'], baseline: 0, width: 0.64, fmtv: (x) => `${fmt.n(x / 1e6, 1)}M` });
    });
    root.append(card('The last dot product is not small',
      'TensorFold’s Qwen3.5-2B geometry is pinned in <code>notes/map-forward.md</code> (source anchor <code>zig/src/families/qwen3_5/config.zig:7-21</code>): <b>hidden = 2048</b>, <b>vocab = 248320</b>, <b>intermediate = 6144</b>, and <b>layers = 24</b>. The final <code>lm_head</code> projection is explicitly the last operation in <code>forward.zig:996-999</code>.',
      row(grow(prose(`<p><b>One token:</b> 2048 × 248,320 = <b>${comma(outputParams)}</b> weights. One multiply-accumulate per weight means ${comma(outputParams)} MACs to score the vocabulary.</p><p>At 4 bits per weight, reading those weights alone is ${comma(outputBytes4)} bytes (${fmt.n(outputBytes4 / (1024 * 1024), 1)} MiB), before scales and other overhead. The map records 64-value affine quantisation; 4 bits is the payload arithmetic shown here.</p>`), scaleStats), scalePlot.canvas),
      note('<b>Scale comparison.</b> The third bar is 24 copies of the three-matrix SwiGLU MLP shape (3 × 2048 × 6144); it is not the whole model. The output projection is one matrix, yet it is larger than one complete MLP block.')));

    /* ---------- nearest neighbours ---------- */
    const neighbourState = { i: 0 };
    const neighbourPlot = new Plot({ width: 550, height: 360, xlim: [-2.25, 2.05], ylim: [-1.85, 1.95], equal: true, pad: { l: 42, r: 16, t: 16, b: 34 }, xlabel: 'embedding dimension 1', ylabel: 'embedding dimension 2' });
    const rankList = el('div', { style: { font: '12px ui-monospace, Menlo, monospace', lineHeight: '1.8', minWidth: '210px' } });
    const renderNeighbours = () => {
      const source = EMBED_POINTS[neighbourState.i];
      const ranked = EMBED_POINTS.map((point, i) => ({ i, sim: cosine([source.x, source.y], [point.x, point.y]) })).sort((a, b) => b.sim - a.sim);
      rankList.replaceChildren(el('div', { class: 'sub', text: `nearest to ${EMBED_TOKENS[neighbourState.i]}` }), ...ranked.map((r, rank) => el('div', {},
        el('span', { style: { color: rank === 0 ? 'var(--accent)' : 'var(--muted)', display: 'inline-block', width: '26px' }, text: String(rank + 1).padStart(2, '0') }),
        el('span', { style: { display: 'inline-block', width: '92px', color: r.i === neighbourState.i ? 'var(--accent)' : 'var(--fg)', fontWeight: r.i === neighbourState.i ? '600' : '' }, text: EMBED_TOKENS[r.i] }),
        el('span', { style: { color: 'var(--muted)' }, text: fmt.n(r.sim, 3) }))));
      neighbourPlot.render();
    };
    neighbourPlot.onDraw((p) => {
      p.axes({ xticks: 5, yticks: 5 });
      const source = EMBED_POINTS[neighbourState.i];
      p.scatter(EMBED_POINTS.map((point, i) => ({ ...point, color: i === neighbourState.i ? 'accent' : 'c3', r: i === neighbourState.i ? 7 : 5 })), { stroke: 'bg', width: 1 });
      EMBED_POINTS.forEach((point, i) => p.text(point.x, point.y + 0.12, EMBED_TOKENS[i], { color: i === neighbourState.i ? 'accent' : 'fg', size: 10, weight: i === neighbourState.i ? '600' : '' }));
      p.circle(source.x, source.y, 0.22, { color: 'accent', fill: false, width: 1.5 });
    });
    neighbourPlot.interact({ down: (pt) => {
      let best = 0, bd = Infinity;
      EMBED_POINTS.forEach((point, i) => { const d = (point.x - pt.x) ** 2 + (point.y - pt.y) ** 2; if (d < bd) { bd = d; best = i; } });
      if (bd < 0.25) { neighbourState.i = best; renderNeighbours(); }
    } });
    root.append(card('Nearest neighbours are geometry',
      'Click a point. The list ranks every token by cosine similarity to it. These 16 two-dimensional vectors are <b>illustrative and hand-made, not learned</b>: animals, actions, colours, water, and model terms form loose clusters so the geometry is legible.',
      row(grow(neighbourPlot.canvas), el('div', { class: 'col', style: { flex: '0 1 260px', maxHeight: '330px', overflowY: 'auto' } }, rankList)),
      legend([{ c: 'c3', label: 'other token vectors' }, { c: 'accent', label: 'selected token' }]),
      note('<b>In a real vocabulary, this space has 2048 dimensions.</b> The same cosine idea still applies, but human 2D intuition fails: the whole vocabulary lives in a high-dimensional geometry we can only inspect through projections and measurements.')));
    renderNeighbours();
  },
};
