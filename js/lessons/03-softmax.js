// 03-softmax.js — level 7: logits become probabilities.

import { Plot } from '../plot.js';
import { card, el, prose, note, row, grow, slider, select, stats, legend, fmt, clamp } from '../ui.js';
import { softmax } from '../engine/tensor.js';

const TOKENS = ['the', 'cat', 'sat', 'on', 'a', 'mat', '.', '<eos>'];
const BAR_COLORS = TOKENS.map((_, i) => `c${(i % 6) + 1}`);
const labels = (xs = TOKENS) => xs.map((t) => t === '<eos>' ? 'eos' : t);
const arrText = (xs, digits = 6) => `[${xs.map((x) => Number.isFinite(x) ? Number(x.toFixed(digits)) : String(x)).join(', ')}]`;

export default {
  id: '03-softmax',
  num: '03',
  group: 'the surface',
  title: 'Logits become probabilities',
  blurb: 'The model hands the sampler a score for every token. Softmax turns those scores into positive numbers that sum to one — while preserving exactly the comparisons that matter.',

  render(root) {
    const state = {
      logits: [2.5, 1.8, 1.2, 0.7, 0.1, -0.7, -1.4, -2.1],
      pairA: 0,
      pairB: 1,
      shift: 0,
      temperature: 1,
      stabilityMagnitude: 800,
      ceTarget: 0,
      ceLogits: [2.2, 1.3, 0.6, 0.2, -0.3, -0.8, -1.5, -2.2],
    };

    const logitsPlot = new Plot({
      width: 510, height: 300, pad: { l: 48, r: 12, t: 22, b: 58 },
      xlim: [0, TOKENS.length], ylim: [-5.5, 5.5], ylabel: 'logit z',
    });
    const probabilityPlot = new Plot({
      width: 510, height: 300, pad: { l: 48, r: 12, t: 22, b: 58 },
      xlim: [0, TOKENS.length], ylim: [0, 1], ylabel: 'probability p',
    });

    logitsPlot.onDraw((p) => {
      p.axes({ xticks: 4, yticks: 5, zero: true });
      p.bars(state.logits, { color: BAR_COLORS, labels: labels(), width: 0.72, fmtv: (v) => fmt.n(v, 1), baseline: -5 });
      p.text(p.pad.l, 10, 'drag a bar vertically', { color: 'muted', size: 10, px: true, align: 'left' });
    });
    probabilityPlot.onDraw((p) => {
      const probs = softmax(state.logits, 1);
      p.ylim = [0, Math.max(0.2, Math.max(...probs) * 1.2)];
      p.axes({ xticks: 4, yticks: 4, zero: false });
      p.bars(Array.from(probs), { color: BAR_COLORS, labels: labels(), width: 0.72, fmtv: (v) => fmt.pct(v, 0), baseline: 0 });
      p.text(p.pad.l, 10, `Σ p = ${fmt.n(Array.from(probs).reduce((a, b) => a + b, 0), 3)}`, { color: 'muted', size: 10, px: true, align: 'left' });
    });

    let dragging = -1;
    const setDraggedLogit = (point) => {
      if (dragging < 0) return;
      state.logits[dragging] = clamp(point.y, -5, 5);
      refresh();
    };
    logitsPlot.interact({
      down: (point) => {
        dragging = clamp(Math.floor(point.x), 0, TOKENS.length - 1);
        setDraggedLogit(point);
      },
      move: setDraggedLogit,
      up: () => { dragging = -1; },
      hover: (point) => { logitsPlot.canvas.style.cursor = point ? 'ns-resize' : 'default'; },
    });

    const primaryStats = stats([
      { k: 'sum of logits', v: '—' },
      { k: 'sum of p', v: '—' },
      { k: 'largest p', v: '—' },
      { k: 'rank preserved?', v: 'yes' },
    ]);

    const refresh = () => {
      const probs = Array.from(softmax(state.logits, 1));
      const sum = state.logits.reduce((a, b) => a + b, 0);
      const winner = probs.reduce((best, v, i) => v > probs[best] ? i : best, 0);
      primaryStats.set('sum of logits', fmt.n(sum, 2));
      primaryStats.set('sum of p', fmt.n(probs.reduce((a, b) => a + b, 0), 6));
      primaryStats.set('largest p', `${labels()[winner]}  ${fmt.pct(probs[winner], 1)}`);
      primaryStats.set('rank preserved?', 'yes', 'ok');
      logitsPlot.render();
      probabilityPlot.render();
      shiftPlot.render();
      shiftProbPlot.render();
      temperaturePlot.render();
      cePlot.render();
      updateRatio();
      updateShift();
      updateStability();
      updateTemperature();
      updateCrossEntropy();
    };

    root.append(card('Scores in, probabilities out',
      'These eight bars are logits: unconstrained scores, not percentages. Drag any bar. The second plot is the same vector after <b>softmax</b>; its bars stay positive and their heights add to one.',
      row(
        grow(logitsPlot.canvas, probabilityPlot.canvas),
        el('div', { class: 'col', style: { flex: '0 1 245px' } }, primaryStats,
          note('<b>Monotonic.</b> Raise one score and its probability rises; the ordering never flips. Softmax does not decide which token is best — it puts calibrated weight behind the ranking.'))),
      legend(TOKENS.map((t, i) => ({ c: `c${(i % 6) + 1}`, label: `${t} = z${i}` })))));

    const rawPlot = new Plot({
      width: 430, height: 260, pad: { l: 46, r: 12, t: 18, b: 54 },
      xlim: [0, TOKENS.length], ylim: [-1, 1], ylabel: 'raw / Σ raw',
    });
    const clampPlot = new Plot({
      width: 430, height: 260, pad: { l: 46, r: 12, t: 18, b: 54 },
      xlim: [0, TOKENS.length], ylim: [0, 1], ylabel: 'clamp(z) / Σ clamp(z)',
    });
    rawPlot.onDraw((p) => {
      const sum = state.logits.reduce((a, b) => a + b, 0);
      const vals = state.logits.map((z) => sum === 0 ? NaN : z / sum);
      const finite = vals.filter(Number.isFinite);
      const extent = finite.length ? Math.max(0.25, ...finite.map(Math.abs)) * 1.2 : 1;
      p.ylim = [-extent, extent];
      p.axes({ xticks: 4, yticks: 4, zero: true });
      p.bars(vals, { color: vals.map((v) => v < 0 ? 'err' : 'c2'), labels: labels(), width: 0.72, fmtv: (v) => Number.isFinite(v) ? fmt.n(v, 2) : 'NaN', baseline: 0 });
    });
    clampPlot.onDraw((p) => {
      const positive = state.logits.map((z) => Math.max(0, z));
      const sum = positive.reduce((a, b) => a + b, 0);
      const vals = positive.map((z) => sum ? z / sum : 0);
      p.axes({ xticks: 4, yticks: 4, zero: false });
      p.bars(vals, { color: 'c5', labels: labels(), width: 0.72, fmtv: (v) => fmt.pct(v, 0), baseline: 0 });
    });

    const ratioStats = stats([
      { k: 'pᵢ / pⱼ', v: '—' },
      { k: 'exp(zᵢ − zⱼ)', v: '—' },
      { k: 'gap zᵢ − zⱼ', v: '—' },
      { k: 'difference', v: '—' },
    ]);
    const pairA = select({
      label: 'token i', options: TOKENS.map((t, i) => ({ v: String(i), t: `${t}  (z${i})` })), value: '0',
      onChange: (v) => { state.pairA = Number(v); refresh(); },
    });
    const pairB = select({
      label: 'token j', options: TOKENS.map((t, i) => ({ v: String(i), t: `${t}  (z${i})` })), value: '1',
      onChange: (v) => { state.pairB = Number(v); refresh(); },
    });
    const updateRatio = () => {
      const probs = Array.from(softmax(state.logits, 1));
      const i = state.pairA, j = state.pairB;
      const gap = state.logits[i] - state.logits[j];
      const fromP = probs[j] === 0 ? Infinity : probs[i] / probs[j];
      const fromGap = Math.exp(gap);
      ratioStats.set('pᵢ / pⱼ', fmt.exp(fromP, 4));
      ratioStats.set('exp(zᵢ − zⱼ)', fmt.exp(fromGap, 4));
      ratioStats.set('gap zᵢ − zⱼ', fmt.n(gap, 2));
      ratioStats.set('difference', fmt.exp(fromP - fromGap, 3), Math.abs(fromP - fromGap) < 1e-5 ? 'ok' : 'warn');
    };

    root.append(card('Why exponentiate?',
      'Two tempting shortcuts fail. Dividing raw logits by their sum lets negative “probabilities” through; clamping negatives to zero throws away order and makes flat dead zones. Exponentiation is the smallest useful repair: it stays positive and turns equal score gaps into equal probability ratios.',
      row(
        el('div', {}, rawPlot.canvas, el('div', { class: 'sub', style: { maxWidth: '430px' }, text: 'normalise z directly — negative bars are not probabilities' })),
        el('div', {}, clampPlot.canvas, el('div', { class: 'sub', style: { maxWidth: '430px' }, text: 'clamp(z, 0, ∞) — zeroed scores cannot recover smoothly' })),
        el('div', { class: 'col', style: { flex: '1 1 245px' } }, pairA, pairB, ratioStats,
          note('<b>The ratio test.</b> Softmax cancels its shared denominator: pᵢ / pⱼ = exp(zᵢ) / exp(zⱼ) = exp(zᵢ − zⱼ). Drag the first chart and the two readouts track each other.')))));

    const shiftPlot = new Plot({
      width: 510, height: 280, pad: { l: 48, r: 12, t: 20, b: 58 },
      xlim: [0, TOKENS.length], ylim: [-6, 10], ylabel: 'logit',
    });
    const shiftProbPlot = new Plot({
      width: 510, height: 260, pad: { l: 48, r: 12, t: 20, b: 58 },
      xlim: [0, TOKENS.length], ylim: [0, 1], ylabel: 'probability',
    });
    const shiftStats = stats([
      { k: 'constant C', v: '0.00' },
      { k: 'max |Δp|', v: '—' },
      { k: 'sum after shift', v: '—' },
    ]);
    const shiftSlider = slider({
      label: 'add the same C to every logit', min: -8, max: 8, step: 0.01, value: 0,
      fmt: (v) => fmt.sign(v, 2), onInput: (v) => { state.shift = v; refresh(); },
    });
    shiftPlot.onDraw((p) => {
      const shifted = state.logits.map((z) => z + state.shift);
      const lo = Math.min(-5.5, ...state.logits, ...shifted) - 0.3;
      const hi = Math.max(5.5, ...state.logits, ...shifted) + 0.3;
      p.ylim = [lo, hi];
      p.axes({ xticks: 4, yticks: 5, zero: true });
      p.bars(state.logits, { color: 'border', labels: labels(), width: 0.72, baseline: 0 });
      p.bars(shifted, { color: 'accent', labels: null, width: 0.42, baseline: 0 });
      p.text(p.pad.l, 10, 'grey = z · coloured = z + C', { color: 'muted', size: 10, px: true, align: 'left' });
    });
    shiftProbPlot.onDraw((p) => {
      const original = Array.from(softmax(state.logits, 1));
      const shifted = Array.from(softmax(state.logits.map((z) => z + state.shift), 1));
      p.axes({ xticks: 4, yticks: 4, zero: false });
      p.bars(original, { color: 'border', labels: labels(), width: 0.72, baseline: 0 });
      p.bars(shifted, { color: 'accent', labels: null, width: 0.42, baseline: 0 });
      p.text(p.pad.l, 10, 'grey = p(z) · coloured = p(z + C)', { color: 'muted', size: 10, px: true, align: 'left' });
    });
    const updateShift = () => {
      const shifted = state.logits.map((z) => z + state.shift);
      const p = Array.from(softmax(state.logits, 1));
      const q = Array.from(softmax(shifted, 1));
      const maxDiff = Math.max(...p.map((v, i) => Math.abs(v - q[i])));
      shiftStats.set('constant C', fmt.sign(state.shift, 2));
      shiftStats.set('max |Δp|', maxDiff === 0 ? '0 (same)' : fmt.exp(maxDiff, 2), maxDiff < 1e-6 ? 'ok' : 'warn');
      shiftStats.set('sum after shift', fmt.n(shifted.reduce((a, b) => a + b, 0), 2));
    };

    root.append(card('Shift every score: nothing changes',
      'Softmax only sees differences. This control adds a constant C to <i>all</i> logits; the coloured bars slide, but the probability vector is invariant. That is not a curiosity: it is the licence to subtract a large number before exponentiating.',
      row(grow(shiftPlot.canvas, shiftProbPlot.canvas), el('div', { class: 'col', style: { flex: '0 1 280px' } }, shiftSlider, shiftStats,
        note('z and z + C have the same pairwise gaps. Their exponentials are all multiplied by the same factor, which the final division cancels.')))));

    const stabilityReadout = el('pre', { class: 'code', text: '' });
    const stabilitySlider = slider({
      label: 'largest logit M', min: 0, max: 1000, step: 1, value: 800, fmt: (v) => String(Math.round(v)),
      onInput: (v) => { state.stabilityMagnitude = v; updateStability(); },
    });
    const updateStability = () => {
      const m = state.stabilityMagnitude;
      const z = [m, m - 1, m - 2];
      const rawExp = z.map((v) => Math.exp(v));
      const rawSum = rawExp.reduce((a, b) => a + b, 0);
      const naive = rawExp.map((v) => v / rawSum);
      const max = Math.max(...z);
      const stableExp = z.map((v) => Math.exp(v - max));
      const stableSum = stableExp.reduce((a, b) => a + b, 0);
      const stable = stableExp.map((v) => v / stableSum);
      stabilityReadout.textContent = [
        `z                         = ${arrText(z)}`,
        `Math.exp(z)              = ${arrText(rawExp)}`,
        `sum(Math.exp(z))         = ${String(rawSum)}`,
        `naive p = exp(z) / sum   = ${arrText(naive)}`,
        '',
        `z - max(z)               = ${arrText(z.map((v) => v - max))}`,
        `Math.exp(z - max(z))     = ${arrText(stableExp)}`,
        `stable sum               = ${String(stableSum)}`,
        `stable p                 = ${arrText(stable)}`,
      ].join('\n');
    };

    root.append(card('Overflow is a real bug, not a footnote',
      'JavaScript can represent a large logit, but not its exponential. At the default M = 800, every raw exponential is <b>Infinity</b>; Infinity divided by Infinity is <b>NaN</b>. Subtracting the maximum first produces the same mathematics with small, finite numbers.',
      row(grow(stabilityReadout), el('div', { class: 'col', style: { flex: '0 1 300px' } }, stabilitySlider,
        note('<b>Look at the actual values.</b> Move M below the overflow threshold and back. The stable column remains finite because its largest exponent is always exp(0) = 1.')))));

    const temperaturePlot = new Plot({
      width: 510, height: 280, pad: { l: 48, r: 12, t: 20, b: 58 },
      xlim: [0, TOKENS.length], ylim: [0, 1], ylabel: 'probability',
    });
    const temperatureStats = stats([
      { k: 'temperature T', v: '1.00' },
      { k: 'top probability', v: '—' },
      { k: 'p spread (max−min)', v: '—' },
    ]);
    const temperatureSlider = slider({
      label: 'temperature <span class="badge">lesson 01</span>', min: 0.1, max: 3, step: 0.01, value: 1,
      fmt: (v) => fmt.n(v, 2), onInput: (v) => { state.temperature = v; refresh(); },
    });
    temperaturePlot.onDraw((p) => {
      const probs = Array.from(softmax(state.logits, state.temperature));
      p.ylim = [0, Math.max(0.2, Math.max(...probs) * 1.2)];
      p.axes({ xticks: 4, yticks: 4, zero: false });
      p.bars(probs, { color: 'c3', labels: labels(), width: 0.72, fmtv: (v) => fmt.pct(v, 0), baseline: 0 });
      p.text(p.pad.l, 10, `softmax(z / ${fmt.n(state.temperature, 2)})`, { color: 'muted', size: 10, px: true, align: 'left' });
    });
    const updateTemperature = () => {
      const probs = Array.from(softmax(state.logits, state.temperature));
      temperatureStats.set('temperature T', fmt.n(state.temperature, 2));
      temperatureStats.set('top probability', fmt.pct(Math.max(...probs), 1));
      temperatureStats.set('p spread (max−min)', fmt.n(Math.max(...probs) - Math.min(...probs), 3));
    };

    root.append(card('Temperature is the same operation',
      'The sampler knob from lesson 01 divides logits by T <i>before</i> softmax. Nothing else changes: lower T magnifies the gaps and sharpens the distribution; higher T shrinks them and flattens it. The ranking stays the same.',
      row(grow(temperaturePlot.canvas), el('div', { class: 'col', style: { flex: '0 1 280px' } }, temperatureSlider, temperatureStats,
        el('div', { class: 'eq', html: 'p<sub>i</sub>(T) = exp(<span class="hl">z<sub>i</sub> / T</span>) / Σ<sub>j</sub> exp(<span class="hl">z<sub>j</sub> / T</span>)' }),
        note('<b>T → 0</b> approaches argmax. <b>T = 1</b> leaves the learned distribution alone. <b>T &gt; 1</b> gives lower-ranked tokens more room.')))));

    const cePlot = new Plot({
      width: 510, height: 280, pad: { l: 48, r: 12, t: 20, b: 58 },
      xlim: [0, TOKENS.length], ylim: [0, 1], ylabel: 'probability',
    });
    const ceStats = stats([
      { k: 'correct token', v: '—' },
      { k: 'p(correct)', v: '—' },
      { k: 'loss', v: '—' },
    ]);
    const targetSelect = select({
      label: 'which token was correct?', options: TOKENS.map((t, i) => ({ v: String(i), t: `${t}  (token ${i})` })), value: '0',
      onChange: (v) => { state.ceTarget = Number(v); confidenceSlider.set(state.ceLogits[state.ceTarget]); refresh(); },
    });
    const confidenceSlider = slider({
      label: 'correct token\'s logit', min: -100, max: 4, step: 0.1, value: state.ceLogits[0],
      fmt: (v) => fmt.n(v, 1), onInput: (v) => { state.ceLogits[state.ceTarget] = v; refresh(); },
    });
    cePlot.onDraw((p) => {
      const probs = Array.from(softmax(state.ceLogits, 1));
      p.ylim = [0, Math.max(0.2, Math.max(...probs) * 1.2)];
      p.axes({ xticks: 4, yticks: 4, zero: false });
      p.bars(probs, { color: 'c4', labels: labels(), width: 0.72, highlight: state.ceTarget, fmtv: (v) => fmt.pct(v, 0), baseline: 0 });
      p.text(p.pad.l, 10, `highlight = ${labels()[state.ceTarget]}`, { color: 'accent', size: 10, px: true, align: 'left' });
    });
    const updateCrossEntropy = () => {
      const probs = Array.from(softmax(state.ceLogits, 1));
      const pCorrect = probs[state.ceTarget];
      const loss = -Math.log(pCorrect);
      ceStats.set('correct token', labels()[state.ceTarget], 'accent');
      ceStats.set('p(correct)', Number.isFinite(pCorrect) ? fmt.exp(pCorrect, 3) : String(pCorrect));
      ceStats.set('loss', Number.isFinite(loss) ? fmt.n(loss, 3) : String(loss), loss > 8 ? 'warn' : 'fg');
    };

    root.append(card('Cross-entropy: softmax’s partner',
      'Once softmax assigns probabilities, training asks one focused question: how much mass did you put on the answer that was actually correct? The loss is <b>−ln p[correct]</b>. Pick a target, then drag its logit down until its probability approaches zero.',
      row(grow(cePlot.canvas), el('div', { class: 'col', style: { flex: '0 1 300px' } }, targetSelect, confidenceSlider, ceStats,
        el('div', { class: 'eq', html: 'loss = −ln <span class="hl">p[correct]</span>' }),
        note('<b>Only one number matters.</b> Two distributions with the same p(correct) have the same loss, regardless of how the remaining probability is split. As p → 0, −ln p → ∞. Lesson 11 will use this feedback signal.')))));

    root.append(card('The stability trick in TensorFold', null,
      row(grow(
        prose('<p>The verified serving map identifies the host sampler’s shifted softmax in <code class="inline">zig/src/core/lanes/sampling.zig:135–178</code>. Its exponentiation step is:</p>'),
        el('pre', { class: 'code', text: '// sampling.zig:674\nfor (probs, scaled) |*p, x| p.* = exp(x - scaled[0]);' }),
        prose('<p><code class="inline">scaled[0]</code> is the leading scaled score in that sampler path. Subtracting it is exactly the invariance you just tested — and it keeps the largest exponent at one.</p>')),
        note('<b>Same answer, safer arithmetic.</b> The shift does not change any probability ratio; it only chooses a numerically convenient origin for the scores.'))));

    refresh();
    return () => {};
  },
};
