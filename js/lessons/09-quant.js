// 09-quant.js — level 1: the metal, where weights become packed codes.
import { Plot, color } from '../plot.js';
import { card, prose, note, row, grow, select, slider, textbox, stats, legend, fmt, el } from '../ui.js';
import { mulberry32, cosine, softmax, argmax } from '../engine/tensor.js';

const rng = mulberry32(0x09c0ffee);
const normal = () => { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const WEIGHTS = Array.from({ length: 8 }, () => 0.78 * normal() + 0.15 * Math.sin(rng() * 8));
const R = 8, C = 16;
const ORIGINAL = Array.from({ length: R }, () => Array.from({ length: C }, () => 0.7 * normal()));
const INPUT = Array.from({ length: C }, (_, i) => 0.35 + Math.sin(i * 0.8));
const bitsOf = [2, 3, 4, 6, 8];

/**
 * Unpack `count` unsigned codes of `bits` width from little-endian packed 32-bit words.
 * Code i sits at bit offset (i % perWord) * bits — low bits hold the low-index code.
 * This is the MLX affine layout: 32/bits codes per word, reconstructed as scale·q + bias.
 */
function unpack(words, bits, count) {
  const perWord = 32 / bits;
  const mask = (1 << bits) - 1;
  const out = [];
  for (let i = 0; i < count; i++) {
    const w = words[Math.floor(i / perWord)] >>> 0;
    out.push((w >>> ((i % perWord) * bits)) & mask);
  }
  return out;
}

function quantise(values, bits, group) {
  const levels = (1 << bits) - 1, q = [], rec = [], params = [];
  for (let start = 0; start < values.length; start += group) {
    const end = Math.min(values.length, start + group), chunk = values.slice(start, end);
    const lo = Math.min(...chunk), hi = Math.max(...chunk), scale = (hi - lo) / levels || 1;
    params.push({ scale, bias: lo });
    for (let i = start; i < end; i++) { const code = Math.max(0, Math.min(levels, Math.round((values[i] - lo) / scale))); q[i] = code; rec[i] = scale * code + lo; }
  }
  return { q, rec, params };
}
function matrixQuant(bits, group) { return ORIGINAL.map((row) => quantise(row, bits, group).rec); }
function metrics(A, B) {
  const a = A.flat(), b = B.flat(), err = a.map((v, i) => v - b[i]);
  const rms = Math.sqrt(err.reduce((s, x) => s + x * x, 0) / err.length);
  const max = Math.max(...err.map(Math.abs));
  const outA = ORIGINAL.map((r) => r.reduce((s, x, i) => s + x * INPUT[i], 0));
  const outB = B.map((r) => r.reduce((s, x, i) => s + x * INPUT[i], 0));
  const changed = outA.reduce((n, _, i) => n + (argmax(outA) !== argmax(outB) ? 1 : 0), 0) / outA.length;
  const pa = softmax(outA), pb = softmax(outB);
  const kl = pa.reduce((s, p, i) => s + p * Math.log(p / Math.max(pb[i], 1e-12)), 0);
  return { rms, max, cos: cosine(a, b), outA, outB, changed, kl };
}
function table(values, q, rec) {
  const rows = values.map((v, i) => `<tr><td>${i}</td><td>${fmt.n(v, 3)}</td><td>${q[i]}</td><td>${fmt.n(rec[i], 3)}</td><td>${fmt.sign(rec[i] - v, 3)}</td></tr>`).join('');
  return `<table><thead><tr><th>i</th><th>original</th><th>code</th><th>reconstructed</th><th>error</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function barsPlot(a, b, title) {
  const lim = Math.max(1, ...a.map(Math.abs), ...b.map(Math.abs)) * 1.15;
  const p = new Plot({ width: 520, height: 230, xlim: [-0.5, a.length - 0.5], ylim: [-lim, lim], plain: true });
  p.onDraw((g) => { g.axes({ xticks: a.length, yticks: 5, grid: true, zero: true }); const w = 0.34; a.forEach((v, i) => { g.rect(i - w, 0, i - 0.03, v, { color: 'c1', fill: true }); g.rect(i + 0.03, 0, i + w, b[i], { color: 'c3', fill: true }); }); g.text(4, lim * .91, title, { color: 'muted', size: 12, align: 'center' }); });
  return p;
}
function curvePlot(points, xlabel, xlim = [2, 8]) {
  const p = new Plot({ width: 520, height: 230, xlim, ylim: [0, Math.max(0.01, ...points.map((x) => x[1])) * 1.15], xlabel, ylabel: 'error', plain: true });
  p.onDraw((g) => { g.axes({ xticks: 7, yticks: 4 }); g.line(points, { color: 'accent', width: 2 }); g.scatter(points, { color: 'c2', r: 4 }); });
  return p;
}

export default {
  id: '09-quant', num: '09', group: 'the metal', title: 'Quantisation: weights as codes',
  blurb: 'A weight need not be a floating-point number. Pack an unsigned integer code, then restore it with one scale and bias per group — and measure exactly what that shortcut costs.',
  render(root) {
    let bits = 4, group = 8;
    const qCard = card('Write a dequantiser', 'Affine quantisation is deliberately simple: choose a group range, round to an unsigned code, then w = scale · q + bias.');
    const controls = row(select({ label: 'bits', options: bitsOf.map((v) => ({ v, t: `${v}-bit` })), value: bits, onChange: (v) => { bits = +v; update(); updateMetrics(); updateCurves(); } }), slider({ label: 'group size', min: 2, max: 8, step: 1, value: group, fmt: (v) => `${v}`, onInput: (v) => { group = +v; update(); updateMetrics(); updateCurves(); } }));
    const readout = el('div'); const plot = new Plot({ width: 520, height: 230, xlim: [-0.5, 7.5], ylim: [-2.5, 2.5], plain: true });
    qCard.append(controls, readout, plot.canvas, legend([{ c: 'c1', label: 'original' }, { c: 'c3', label: 'reconstructed grid' }]));
    function update() { const z = quantise(WEIGHTS, bits, group); readout.innerHTML = `<p>codes are unsigned 0…${(1 << bits) - 1}; ${z.params.length} group${z.params.length > 1 ? 's' : ''}: ${z.params.map((x) => `scale ${fmt.n(x.scale, 3)}, bias ${fmt.n(x.bias, 3)}`).join(' · ')}</p>${table(WEIGHTS, z.q, z.rec)}`; plot.onDraw((g) => { g.axes({ xticks: 8, yticks: 5, zero: true }); WEIGHTS.forEach((v, i) => { g.rect(i - .35, 0, i - .04, v, { color: 'c1', fill: true }); g.rect(i + .04, 0, i + .35, z.rec[i], { color: 'c3', fill: true }); }); }); }
    update();

    const format = card('The exact MLX affine format', 'TensorFold follows MLX’s packed layout: three tensors per quantised weight — packed words, scales, biases. For Qwen3.5-2B, groups contain 64 values; K = scales.shape[-1] × group and words are K·bits/32 wide.');
    const test = [0x76543210];
    const expected = [-1, 1, 3, 5, 7, 9, 11, 13], got = unpack(test, 4, 8).map((q) => 2 * q - 1);
    format.append(prose('<p>Codes are little-endian: the low code occupies the low bits. The repository test vector is one 32-bit word, <code>0x76543210</code>, scale 2, bias −1, group 8.</p>'), el('div', { class: got.every((v, i) => v === expected[i]) ? 'note' : 'note err', html: `<strong>${got.every((v, i) => v === expected[i]) ? 'PASS' : 'FAIL'}</strong> unpack → [${got.join(', ')}]` }));
    const pack = card('See the nibbles', 'Edit the word; each hexadecimal digit is one 4-bit unsigned code.');
    const wordInput = textbox({ label: '32-bit word (hex)', value: '76543210', rows: 1, onInput: updateWord }); const wordView = el('div'); pack.append(wordInput, wordView);
    function updateWord() { const w = Number.parseInt(wordInput.querySelector('input,textarea')?.value || '76543210', 16) >>> 0; const qs = unpack([w], 4, 8); wordView.innerHTML = `<p class="code">${qs.map((q, i) => `nibble ${i}: 0x${q.toString(16)} → ${fmt.n(2 * q - 1, 2)}`).join('<br>')}</p>`; } updateWord();

    const cost = card('Measure the cost on an output', 'The same approximation can look tiny in weight space and still move a decision. These are fixed seeded weights and input, so reloads are comparable.');
    const metricNode = stats([{ k: 'RMS error', v: '' }, { k: 'max error', v: '' }, { k: 'cosine', v: '' }, { k: 'argmax changes', v: '' }, { k: 'KL (nats)', v: '' }]); const outputPlot = barsPlot([], [], ''); cost.append(metricNode, outputPlot.canvas, legend([{ c: 'c1', label: 'original logits' }, { c: 'c3', label: 'quantised logits' }]));
    function updateMetrics() { const B = matrixQuant(bits, group), m = metrics(ORIGINAL, B); metricNode.set('RMS error', fmt.n(m.rms, 4)); metricNode.set('max error', fmt.n(m.max, 4)); metricNode.set('cosine', fmt.n(m.cos, 5)); metricNode.set('argmax changes', m.changed ? 'yes' : 'no'); metricNode.set('KL (nats)', fmt.n(m.kl, 5)); outputPlot.xlim = [-.5, m.outA.length - .5]; outputPlot.ylim = [-Math.max(1, ...m.outA.map(Math.abs), ...m.outB.map(Math.abs)) * 1.15, Math.max(1, ...m.outA.map(Math.abs), ...m.outB.map(Math.abs)) * 1.15]; outputPlot.onDraw((g) => { g.axes({ xticks: m.outA.length, yticks: 5, zero: true }); m.outA.forEach((v, i) => { g.rect(i - .35, 0, i - .04, v, { color: 'c1', fill: true }); g.rect(i + .04, 0, i + .35, m.outB[i], { color: 'c3', fill: true }); }); }); }
    updateMetrics();

    const curves = card('Bits, groups, and metadata', 'Bits-per-weight includes the two 16-bit table entries per group: bits + 32/group. Smaller groups buy local scales and biases, but spend more metadata.');
    const curveNode = el('div'); curves.append(curveNode); function updateCurves() { curveNode.innerHTML = ''; const byBits = bitsOf.map((b) => { const m = metrics(ORIGINAL, matrixQuant(b, group)); return [b, m.rms]; }); curveNode.append(curvePlot(byBits, 'bits', [2, 8]).canvas); const byGroup = [2, 4, 8, 16, 32].map((g) => { const m = metrics(ORIGINAL, matrixQuant(bits, g)); return [bits + 32 / g, m.rms]; }).sort((a, b) => a[0] - b[0]); curveNode.append(curvePlot(byGroup, 'bits/weight', [bits + 1, bits + 16]).canvas); curveNode.insertAdjacentHTML('beforeend', `<p>At ${bits}-bit/group ${group}: <strong>${fmt.n(bits + 32 / group, 2)} bits/weight</strong> (payload ${bits} + scale/bias overhead ${fmt.n(32 / group, 2)}).</p>`); }
    updateCurves();

    const kernel = card('The kernel never materialises W', 'Metal decodes nibbles in registers and folds each group’s scale/bias into the dot product.');
    const x = [0.4, -1.2, 0.7, 2], q = [1, 3, 0, 2], s = 1.7, b = -0.25; const lhs = x.reduce((t, v, i) => t + v * (s * q[i] + b), 0), rhs = s * x.reduce((t, v, i) => t + v * q[i], 0) + b * x.reduce((t, v) => t + v, 0); kernel.append(prose(`<p>With x = [${x.join(', ')}], q = [${q.join(', ')}], s = ${s}, b = ${b}:<br>Σx(sq+b) = <strong>${fmt.n(lhs, 6)}</strong>; s·Σxq + b·Σx = <strong>${fmt.n(rhs, 6)}</strong>.</p><p><code>simd_qmm.metal</code> and <code>tile_qmv.metal</code> use this identity. The latter also records a <code>0x6400</code> nibble→half trick: decode and accumulate without writing a full matrix.</p>`));
    const clean = card('Distribution-level honesty', '“The error is small” is not a quality metric until it is measured where the model makes decisions. KL here is in nats, with the full toy vocabulary; TensorFold’s CUDA check also records top-1 agreement and NLL in f64.'); clean.append(note('The KL value above compares softmax(original logits) with softmax(quantised logits). It exposes probability mass moved by quantisation even when cosine similarity looks excellent.'));
    root.append(qCard, format, pack, cost, curves, kernel, clean);
    return () => { plot.destroy(); outputPlot.destroy(); };
  },
};
