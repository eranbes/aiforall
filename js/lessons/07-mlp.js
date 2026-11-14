// 07-mlp.js — level 3: the per-token computation inside a transformer block.

import { Plot, color, div } from '../plot.js';
import { card, el, prose, note, row, grow, slider, select, stats, legend, fmt } from '../ui.js';
import { Mat, matmul, act, mulberry32 } from '../engine/tensor.js';


const vec = (a, d = 2) => `[${a.map((x) => fmt.n(x, d)).join(', ')}]`;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const ACTIVATIONS = {
  relu: { fn: act.relu, formula: 'max(0, z)' },
  silu: { fn: act.silu, formula: 'z · σ(z) = z / (1 + e<sup>−z</sup>)' },
  gelu: { fn: act.gelu, formula: '½z(1 + tanh(0.7979(z + 0.0447z³)))' },
  tanh: { fn: act.tanh, formula: 'tanh(z)' },
  sigmoid: { fn: act.sigmoid, formula: 'σ(z) = 1 / (1 + e<sup>−z</sup>)' },
};

export default {
  id: '07-mlp',
  num: '07',
  group: 'inside the model',
  title: 'The token-wise MLP',
  blurb: 'Attention moves information between tokens. The MLP then transforms each token on its own — with most of the block’s learned weights living in these projections.',

  render(root) {
    /* ---------- one neuron ---------- */
    const nstate = { w: 1.15, b: -0.25, x: 1.3, fn: 'silu' };
    const nplot = new Plot({ width: 430, height: 260, xlim: [-4, 4], ylim: [-3, 4.5], pad: { l: 44, r: 14, t: 14, b: 36 }, xlabel: 'pre-activation z', ylabel: 'activation(z)' });
    const nformula = el('div', { class: 'eq' });
    const nstats = stats([{ k: 'z = w·x + b', v: '—' }, { k: 'output', v: '—' }]);
    const nfn = select({
      label: 'activation', options: Object.keys(ACTIVATIONS), value: nstate.fn,
      onChange: (v) => { nstate.fn = v; updateNeuron(); },
    });
    const nw = slider({ label: 'weight w', min: -2, max: 2, step: 0.01, value: nstate.w, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { nstate.w = v; updateNeuron(); } });
    const nb = slider({ label: 'bias b', min: -2, max: 2, step: 0.01, value: nstate.b, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { nstate.b = v; updateNeuron(); } });
    const nx = slider({ label: 'input x', min: -2, max: 2, step: 0.01, value: nstate.x, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { nstate.x = v; updateNeuron(); } });
    function updateNeuron() {
      const spec = ACTIVATIONS[nstate.fn];
      const z = nstate.w * nstate.x + nstate.b;
      const y = spec.fn(z);
      nformula.innerHTML = `<div><b>f(z)</b> = ${spec.formula}</div><b>y = ${nstate.fn}(w·x + b)</b> = ${nstate.fn}(${fmt.sign(nstate.w, 2)} · ${fmt.sign(nstate.x, 2)} ${fmt.sign(nstate.b, 2)}) = <b>${fmt.n(y, 4)}</b>`;
      nstats.set('z = w·x + b', fmt.n(z, 4));
      nstats.set('output', fmt.n(y, 4), 'accent');
      nplot.render();
    }
    nplot.onDraw((p) => {
      const spec = ACTIVATIONS[nstate.fn];
      p.axes({ xticks: 5, yticks: 5, zero: true });
      p.fn(spec.fn, { color: 'accent', width: 2.5, samples: 220 });
      const z = nstate.w * nstate.x + nstate.b;
      p.line([[z, p.ylim[0]], [z, spec.fn(z)]], { color: 'c5', dash: [4, 4], width: 1.2 });
      p.scatter([[z, spec.fn(z)]], { color: 'c5', stroke: 'bg', r: 5 });
      p.text(z, spec.fn(z), `  ${fmt.n(spec.fn(z), 2)}`, { color: 'c5', align: z > 2.5 ? 'right' : 'left', size: 11 });
    });
    updateNeuron();
    root.append(card('One neuron, no mystery', 'A neuron is an affine function followed by a chosen nonlinearity. Drag <b>w</b>, <b>b</b>, or <b>x</b>; switch the function and the curve, marked point, and substituted formula all change.',
      row(grow(nplot.canvas, nformula), el('div', { class: 'col', style: { flex: '0 1 280px' } }, nfn, nw, nb, nx, nstats))));

    /* ---------- why a nonlinearity ---------- */
    const W1 = Mat.from([[1.2, -0.7], [0.4, 1.1]]);
    const W2 = Mat.from([[0.6, 0.8], [-1.0, 0.5]]);
    const x = Mat.from([[1.1, -0.8]]);
    const collapsed = matmul(W1, W2);
    const linearTwo = matmul(matmul(x, W1), W2).row(0);
    const linearOne = matmul(x, collapsed).row(0);
    const hiddenNonlinear = matmul(x, W1).row(0).map(act.relu);
    const nonlinearOut = matmul(Mat.from([Array.from(hiddenNonlinear)]), W2).row(0);
    const linearProof = el('div', { class: 'eq', html: `x·W₁·W₂ = ${vec(linearTwo)}<br>x·(W₁W₂) = ${vec(linearOne)}<br><b>same: ${Math.max(...linearOne.map((v, i) => Math.abs(v - linearTwo[i]))) < 1e-6 ? 'yes' : 'no'}</b>` });
    const nonlinearProof = el('div', { class: 'eq', html: `x·W₁ = ${vec(matmul(x, W1).row(0))}<br>ReLU(x·W₁) = ${vec(hiddenNonlinear)}<br>ReLU(x·W₁)·W₂ = <b>${vec(nonlinearOut)}</b>` });
    const xor = new Plot({ width: 430, height: 270, xlim: [-2, 2], ylim: [-2, 2], pad: { l: 38, r: 14, t: 14, b: 32 }, xlabel: 'x₁', ylabel: 'x₂', equal: true });
    xor.onDraw((p) => {
      p.field((a, b) => a * b < 0 ? color('c3') : color('c1'), { res: 6 });
      p.axes({ xticks: 5, yticks: 5, zero: true, grid: false, frame: true });
      const pts = [{ x: -1, y: -1, label: '0', c: 'c1' }, { x: -1, y: 1, label: '1', c: 'c3' }, { x: 1, y: -1, label: '1', c: 'c3' }, { x: 1, y: 1, label: '0', c: 'c1' }];
      p.scatter(pts.map((q) => ({ x: q.x, y: q.y, color: q.c, r: 7 })), { stroke: 'bg', width: 2 });
      for (const q of pts) p.text(q.x, q.y, q.label, { color: 'bg', size: 10, weight: 'bold' });
    });
    root.append(card('Why bend the space?', 'Two linear layers are still one linear layer: matrix multiplication collapses them. Insert ReLU (or SiLU, GELU, …), and the composition can make a boundary a single matrix cannot.',
      row(grow(
        el('div', { class: 'sub', text: 'linear: W₁ then W₂' }), linearProof,
        el('div', { class: 'sub', text: 'XOR: alternating labels need a bend' }), xor.canvas,
        legend([{ c: 'c1', label: 'class 0' }, { c: 'c3', label: 'class 1' }])),
        el('div', { class: 'col', style: { flex: '0 1 340px' } },
          note('<b>Collapse:</b> x·W₁·W₂ and x·(W₁W₂) print the same two numbers because associativity is exact here.'),
          el('div', { class: 'sub', text: 'with ReLU between them' }), nonlinearProof,
          prose('<p>The XOR panel is the same idea geometrically: one straight line cannot put the two diagonal 1s on one side and the two diagonal 0s on the other. A hidden nonlinearity lets later layers compose regions instead of only tilting one plane.</p>')))));

    /* ---------- SwiGLU ---------- */
    const G = [[1.1, -0.8, 0.2, 1.0], [-0.6, 0.9, 1.2, -0.5]];
    const U = [[0.7, 0.4, -1.0, 0.8], [1.0, -0.7, 0.3, 0.5]];
    const D = [[0.8, -0.4], [-0.6, 0.9], [0.7, 0.3], [-0.5, 0.8]];
    const mstate = { x1: 1.0, x2: -0.7 };
    const mplots = [];
    const heatLabels = ['ch 0', 'ch 1', 'ch 2', 'ch 3'];
    const stage = (title) => {
      const plot = new Plot({ width: 220, height: 125, xlim: [0, 4], ylim: [0, 1], pad: { l: 26, r: 8, t: 22, b: 22 }, plain: true });
      plot._stageTitle = title;
      plot.onDraw((p) => { p.axes({ xticks: 0, yticks: 0, grid: false, zero: false, frame: true }); });
      mplots.push(plot);
      return plot;
    };
    const gplot = stage('gate = x·Wg');
    const uplot = stage('up = x·Wu');
    const splot = stage('SiLU(gate)');
    const hplot = stage('h = SiLU(g) ⊙ up');
    const bars = new Plot({ width: 450, height: 190, xlim: [0, 4], ylim: [-2.5, 2.5], pad: { l: 38, r: 12, t: 20, b: 34 }, xlabel: 'intermediate channel', ylabel: 'value' });
    const mstats = stats([{ k: 'down(h)', v: '—' }, { k: 'gate suppression', v: '—' }]);
    const mx1 = slider({ label: 'token x₀', min: -2, max: 2, step: 0.01, value: mstate.x1, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { mstate.x1 = v; updateMLP(); } });
    const mx2 = slider({ label: 'token x₁', min: -2, max: 2, step: 0.01, value: mstate.x2, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { mstate.x2 = v; updateMLP(); } });
    function mlpValues() {
      const xv = [mstate.x1, mstate.x2];
      const gm = matmul(Mat.from([xv]), Mat.from(G)).row(0);
      const um = matmul(Mat.from([xv]), Mat.from(U)).row(0);
      const sm = gm.map(act.silu);
      const hm = sm.map((v, i) => v * um[i]);
      const dm = matmul(Mat.from([hm]), Mat.from(D)).row(0);
      return { gm: Array.from(gm), um: Array.from(um), sm: Array.from(sm), hm: Array.from(hm), dm: Array.from(dm) };
    }
    function updateMLP() {
      const v = mlpValues();
      const vals = [v.gm, v.um, v.sm, v.hm];
      const lim = Math.max(1, ...vals.flat().map((a) => Math.abs(a))) * 1.12;
      for (const [i, p] of mplots.entries()) {
        p.ylim = [-lim, lim];
        p.onDraw((q) => { q.axes({ xticks: 0, yticks: 0, grid: false, zero: true, frame: true }); q.matrix([vals[i]], { x0: 0, x1: 4, y0: 0, y1: 1, cmap: div, labels: { cols: heatLabels }, text: true, fmtv: (z) => fmt.n(z, 2), max: lim }); q.text(2, lim * 0.9, p._stageTitle, { color: 'fg', size: 11 }); });
      }
      bars.ylim = [-lim, lim];
      bars._v = v;
      bars.render();
      mstats.set('down(h)', vec(v.dm));
      const suppressed = v.gm.reduce((n, z, i) => n + (Math.abs(z) > 0.25 && Math.abs(v.sm[i]) < Math.abs(z) * 0.6 ? 1 : 0), 0);
      mstats.set('gate suppression', `${suppressed} / 4 channels`);
    }
    bars.onDraw((p) => {
      p.axes({ xticks: 0, yticks: 5, zero: true, frame: true });
      const v = bars._v || { hm: [0, 0, 0, 0], dm: [0, 0] };
      p.bars(v.hm, { color: ['c1', 'c2', 'c3', 'c4'], labels: heatLabels, fmtv: (z) => fmt.n(z, 2) });
      p.text(2, p.ylim[1] * 0.88, 'elementwise product h', { color: 'fg', size: 11 });
      p.text(3.25, p.ylim[0] * 0.88, `down → ${vec(v.dm)}`, { color: 'accent', size: 11, align: 'center' });
    });
    updateMLP();
    root.append(card('A gated MLP: three matrices', 'Modern transformer MLPs use two up-projections and one down-projection. The gate is not a second token mixer: it is a channel-wise control signal.',
      row(grow(
        row(gplot.canvas, uplot.canvas), row(splot.canvas, hplot.canvas), bars.canvas),
        el('div', { class: 'col', style: { flex: '0 1 280px' } }, mx1, mx2, mstats,
          note('<b>SwiGLU:</b> <code>down( SiLU(gate(x)) ⊙ up(x) )</code>. The ⊙ is elementwise: a small SiLU(gate) value suppresses that channel before the expensive down projection.'),
          prose('<p>The Qwen3.5 reference fuses this elementwise step in <code>zig/kernels/metal/qwen3_5/mlp_act.metal:12-18</code>; the host launches gate and up projections, then down at <code>zig/src/families/qwen3_5/forward.zig:94-101</code>.</p>')))));

    /* ---------- mixture of experts ---------- */
    const E = 8;
    const rng = mulberry32(707);
    const logits = Array.from({ length: E }, () => (rng() * 2 - 1) * 2);
    const bias = [0.12, -0.08, 0.04, 0.16, -0.11, 0.07, -0.04, 0.1];
    const costs = [0.8, 1.0, 1.2, 0.9, 1.4, 1.1, 0.7, 1.3];
    const rstate = { k: 2, mode: 'sigmoid' };
    const rsliders = logits.map((v, i) => slider({ label: `expert ${i} logit`, min: -4, max: 4, step: 0.01, value: v, fmt: (z) => fmt.sign(z, 2), onInput: (z) => { logits[i] = z; updateRouter(); } }));
    const rk = slider({ label: 'top-k experts', min: 1, max: E, step: 1, value: rstate.k, fmt: (z) => String(z), onInput: (z) => { rstate.k = z; updateRouter(); } });
    const rmode = select({ label: 'router weighting', options: [{ v: 'sigmoid', t: 'sigmoid + bias rank (moe_ref)' }, { v: 'softmax', t: 'softmax rank + weights (route_ref)' }], value: rstate.mode, onChange: (v) => { rstate.mode = v; updateRouter(); } });
    const rplot = new Plot({ width: 450, height: 245, xlim: [0, E], ylim: [0, 1], pad: { l: 42, r: 12, t: 18, b: 38 }, xlabel: 'expert', ylabel: 'combination weight' });
    const rtokens = el('div', { class: 'row', style: { flexWrap: 'wrap', gap: '6px' } });
    const rstats = stats([{ k: 'selected', v: '—' }, { k: 'active expert params (toy)', v: '—' }, { k: 'total expert params (toy)', v: `${fmt.n(costs.reduce((a, b) => a + b, 0), 1)}B` }]);
    function route() {
      const order = Array.from({ length: E }, (_, i) => i).sort((a, b) => (rstate.mode === 'sigmoid' ? (logits[b] + bias[b]) - (logits[a] + bias[a]) : logits[b] - logits[a]) || a - b);
      const picks = order.slice(0, rstate.k);
      const weights = Array(E).fill(0);
      if (rstate.mode === 'sigmoid') {
        const raw = picks.map((i) => act.sigmoid(logits[i]));
        const total = raw.reduce((a, b) => a + b, 0) || 1;
        picks.forEach((i, j) => { weights[i] = raw[j] / total; });
      } else {
        const raw = picks.map((i) => Math.exp(logits[i] - logits[picks[0]]));
        const total = raw.reduce((a, b) => a + b, 0) || 1;
        picks.forEach((i, j) => { weights[i] = raw[j] / total; });
      }
      return { picks, weights };
    }
    function updateRouter() {
      const r = route();
      const selectedCost = r.picks.reduce((s, i) => s + costs[i], 0);
      rplot._weights = r.weights;
      rplot._picks = r.picks;
      rplot.render();
      rstats.set('selected', r.picks.map((i) => `E${i}`).join(', '), 'accent');
      rstats.set('active expert params (toy)', `${fmt.n(selectedCost, 1)}B`, 'accent');
      rtokens.replaceChildren(...Array.from({ length: E }, (_, i) => el('span', { class: 'badge', text: `E${i} ${r.weights[i] ? fmt.pct(r.weights[i], 0) : 'off'}`, style: { borderColor: r.weights[i] ? 'var(--accent)' : 'var(--border)', color: r.weights[i] ? 'var(--accent)' : 'var(--muted)' } })));
    }
    rplot.onDraw((p) => {
      p.axes({ xticks: 0, yticks: 5, zero: false, frame: true, yfmt: (v) => fmt.pct(v, 0) });
      const weights = rplot._weights || Array(E).fill(0);
      const picks = rplot._picks || [];
      p.bars(weights, { color: weights.map((w, i) => picks.includes(i) ? 'accent' : 'border'), labels: Array.from({ length: E }, (_, i) => `E${i}`), fmtv: (v) => v ? fmt.pct(v, 0) : '' });
    });
    updateRouter();
    root.append(card('Mixture of experts: route, then skip', 'A sparse MLP keeps many experts in the checkpoint but runs only a few for each token. Change logits and <b>top-k</b>; the highlighted bars and active-parameter count move.',
      row(grow(rplot.canvas, rtokens, legend([{ c: 'accent', label: 'selected expert' }, { c: 'border', label: 'unpicked' }])),
        el('div', { class: 'col', style: { flex: '0 1 300px' } }, rmode, rk, ...rsliders, rstats,
          note('<b>Two real routing references differ:</b> <code>zig/src/core/moe_ref.zig:5-31</code> ranks by logit+bias and weights winners by σ(logit); <code>zig/src/core/route_ref.zig:19-32</code> uses softmax weights. In Nemotron, <code>zig/kernels/metal/nemotron/expert_up.metal:66-84</code> returns early for unpicked experts. A label like <b>30B-A3B</b> means roughly 30B parameters are stored, while routing activates about 3B for each token — capacity without paying all 30B every time.')))));

    /* ---------- where the parameters are ---------- */
    const hidden = 2048, intermediate = 6144;
    const q = 8, kv = 2, hd = 256;
    const attentionParams = hidden * (q * hd) + hidden * (kv * hd) + hidden * (kv * hd) + (q * hd) * hidden;
    const mlpParams = 3 * hidden * intermediate;
    const pplot = new Plot({ width: 450, height: 250, xlim: [0, 2], ylim: [0, mlpParams / 1e6 * 1.15], pad: { l: 48, r: 12, t: 18, b: 38 }, xlabel: 'projection family', ylabel: 'millions of weights' });
    pplot.onDraw((p) => {
      p.axes({ xticks: 0, yticks: 5, zero: true, frame: true, yfmt: (v) => fmt.n(v, 0) });
      p.bars([attentionParams / 1e6, mlpParams / 1e6], { color: ['c4', 'accent'], labels: ['attention', 'SwiGLU MLP'], fmtv: (v) => fmt.n(v, 2) });
    });
    const pstats = stats([{ k: 'attention projections', v: `${fmt.n(attentionParams / 1e6, 2)}M` }, { k: 'three-matrix MLP', v: `${fmt.n(mlpParams / 1e6, 2)}M`, }, { k: 'MLP / attention', v: `${fmt.n(mlpParams / attentionParams, 2)}×`, color: 'accent' }]);
    root.append(card('Where the parameters are', 'For the real Qwen3.5-2B geometry, the MLP is not a side dish. It is several times larger than the four attention projections in a full-attention layer.',
      row(grow(pplot.canvas), el('div', { class: 'col', style: { flex: '0 1 330px' } }, pstats,
        prose(`<p><b>Mapped dimensions:</b> hidden = ${hidden}, intermediate = ${intermediate}, query heads = ${q}, KV heads = ${kv}, head dim = ${hd} (all pinned in <code>zig/src/families/qwen3_5/config.zig:7-21</code>).</p><p>Attention count here is Q + K + V + O, excluding biases: ${hidden}×${q * hd} + 2·(${hidden}×${kv * hd}) + ${q * hd}×${hidden}. SwiGLU is exactly 3×${hidden}×${intermediate}; the map calls this about 37.7M parameters per layer at <code>notes/map-forward.md §6</code>.</p>`),
        note('<b>Shape, not magic:</b> attention decides which token information to read; the much wider MLP decides how to rewrite this token after that read.')))));
  },
};
