// 05-blocks.js — level 5: the residual stream, layer loop, and cache.

import { Plot, div } from '../plot.js';
import { card, el, prose, note, code, row, grow, slider, checkbox, stats, legend, btns, button, fmt } from '../ui.js';
import { mulberry32, norm } from '../engine/tensor.js';

const DIM = 16;
const BLOCKS = 8;
const LAYERS = 24;
const QWEN_ATTENTION = [3, 7, 11, 15, 19, 23];

export default {
  id: '05-blocks',
  num: '05',
  group: 'inside the model',
  title: 'The hidden state is a stream',
  blurb: 'A token’s hidden state is not replaced 24 times. It travels through a stack of blocks, and each block adds a delta to the running vector.',

  render(root) {
    /* Synthetic, deterministic vectors: these show the shape of the computation, not real weights. */
    const rng = mulberry32(0x05b10c);
    const embedding = Array.from({ length: DIM }, () => (rng() * 2 - 1) * 0.62);
    const deltas = Array.from({ length: BLOCKS }, () =>
      Array.from({ length: DIM }, () => (rng() * 2 - 1) * 0.22));
    const states = [embedding];
    for (const delta of deltas) states.push(states.at(-1).map((x, j) => x + delta[j]));

    const streamState = { layer: 0 };
    const layerRead = stats([
      { k: 'stream row', v: 'embedding' },
      { k: '‖x‖₂', v: '—' },
      { k: 'delta norm', v: '—' },
    ]);

    const heat = new Plot({
      width: 620, height: 232, pad: { l: 62, r: 14, t: 20, b: 28 },
      xlim: [0, DIM], ylim: [0, BLOCKS + 1], plain: false,
    });
    heat.onDraw((p) => {
      p.axes({ xticks: 4, yticks: 0, grid: false, zero: false, frame: true, xfmt: (v) => `d${v}` });
      p.matrix(states, {
        x0: 0, x1: DIM, y0: 0, y1: BLOCKS + 1, cmap: div, max: 0.95, gap: 2,
        labels: { rows: ['x₀', ...Array.from({ length: BLOCKS }, (_, i) => `x${i + 1}`)] },
      });
      const row = BLOCKS - streamState.layer;
      p.rect(0, row, DIM, row + 1, { color: 'accent', width: 2, radius: 3 });
      p.text(p.pad.l, 7, 'each row is the same token after another block',
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    const current = new Plot({
      width: 620, height: 190, pad: { l: 42, r: 14, t: 22, b: 34 },
      xlim: [0, DIM], ylim: [-1, 1], ylabel: 'value',
    });
    current.onDraw((p) => {
      const v = states[streamState.layer];
      const hi = Math.max(0.5, ...v.map(Math.abs)) * 1.18;
      p.ylim = [-hi, hi];
      p.axes({ xticks: 4, yticks: 4, xfmt: (x) => `d${x}` });
      p.bars(v, { color: v.map((x) => x >= 0 ? 'c3' : 'c5'), width: 0.72, baseline: 0 });
      p.text(p.pad.l, 8, `x${streamState.layer}  =  ${streamState.layer ? `x${streamState.layer - 1} + Δ${streamState.layer}` : 'embedding'}`,
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    const lane = el('div', {
      style: {
        display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '5px',
        padding: '10px 8px', border: '1px solid var(--border)', borderRadius: '9px',
        background: 'var(--surface-2)', font: '12px ui-monospace, Menlo, monospace',
      },
    });
    const renderLayer = () => {
      const k = streamState.layer;
      lane.replaceChildren(
        el('span', { style: { color: 'var(--muted)', padding: '5px 2px' }, text: 'embedding' }),
        el('span', { style: { color: 'var(--muted)' }, text: '→' }),
        ...deltas.map((delta, i) => el('div', {
          onClick: () => { streamState.layer = i + 1; layerSlider.set(i + 1, true); renderLayer(); },
          title: `click to show layer ${i + 1}`,
          style: {
            cursor: 'pointer', padding: '5px 7px', borderRadius: '6px',
            border: `1px solid var(${i + 1 === k ? '--accent' : '--border'})`,
            color: `var(${i + 1 === k ? '--accent' : '--fg'})`,
            background: i + 1 === k ? 'var(--surface)' : 'transparent',
          },
          text: `block ${i + 1}`,
        }, el('span', { style: { color: 'var(--muted)' }, text: ' + Δ' }))),
      );
      layerRead.set('stream row', `x${k}`);
      layerRead.set('‖x‖₂', fmt.n(norm(states[k]), 3));
      layerRead.set('delta norm', k ? fmt.n(norm(deltas[k - 1]), 3) : '—');
      heat.render();
      current.render();
    };

    const layerSlider = slider({
      label: 'step through blocks', min: 0, max: BLOCKS, step: 1, value: 0,
      fmt: (v) => v === 0 ? 'embedding' : `after block ${v}`,
      onInput: (v) => { streamState.layer = v; renderLayer(); },
    });

    root.append(card('The residual stream',
      'Follow one token (one row vector) through a small stack. The colours are <b>synthetic, deterministic deltas</b> — illustrative of the shape of the computation, not real TensorFold weights.',
      row(
        grow(heat.canvas, current.canvas),
        el('div', { class: 'col', style: { flex: '0 1 310px' } },
          lane, layerSlider, layerRead,
          btns(
            button('previous', () => { layerSlider.set(Math.max(0, streamState.layer - 1), true); }, { ghost: true }),
            button('next', () => { layerSlider.set(Math.min(BLOCKS, streamState.layer + 1), true); }),
          ),
          note('<b>The invariant:</b> a block does not replace <code class="inline">x</code>. It writes a delta, and the stream continues as <code class="inline">x + Δ</code>. That lets information from the embedding survive all the way to the logits.'),
        ))));
    renderLayer();

    /* A deterministic numerical comparison of residual and replacement updates. */
    const normRng = mulberry32(0x24cafe);
    const gains = Array.from({ length: LAYERS }, () => 0.018 + normRng() * 0.018);
    const forcing = Array.from({ length: LAYERS }, () =>
      Array.from({ length: DIM }, () => (normRng() * 2 - 1) * 0.003));
    const seedVector = Array.from({ length: DIM }, () => (normRng() * 2 - 1) * 0.5);
    const curve = (residual) => {
      let x = seedVector.slice();
      const out = [[0, norm(x)]];
      for (let i = 0; i < LAYERS; i++) {
        const fx = x.map((v, j) => gains[i] * v + forcing[i][j]);
        x = residual ? x.map((v, j) => v + fx[j]) : fx;
        out.push([i + 1, norm(x)]);
      }
      return out;
    };
    const residualCurve = curve(true);
    const replaceCurve = curve(false);
    const residualState = { on: true };
    const normPlot = new Plot({
      width: 620, height: 280, pad: { l: 52, r: 14, t: 18, b: 42 },
      xlim: [0, LAYERS], ylim: [0, 1], xlabel: 'layer', ylabel: '‖x‖₂',
    });
    normPlot.onDraw((p) => {
      p.ylim = [0, Math.max(...residualCurve.map((q) => q[1])) * 1.12];
      p.axes({ xticks: 6, yticks: 5, zero: false, frame: true });
      p.line(replaceCurve, { color: residualState.on ? 'muted' : 'accent', width: residualState.on ? 1.5 : 3 });
      p.line(residualCurve, { color: residualState.on ? 'accent' : 'muted', width: residualState.on ? 3 : 1.5 });
      p.text(p.pad.l + 4, p.pad.t + 4, residualState.on ? 'active: x ← x + f(x)' : 'active: x ← f(x)',
        { color: residualState.on ? 'accent' : 'muted', size: 10, px: true, align: 'left' });
    });
    const normRead = stats([
      { k: 'active update', v: 'x ← x + f(x)' },
      { k: 'norm at layer 24', v: '—' },
      { k: 'same f, same seed', v: 'yes' },
    ]);
    const residualToggle = checkbox({
      label: 'keep the residual: x ← x + f(x)', checked: true,
      onChange: (v) => {
        residualState.on = v;
        normRead.set('active update', v ? 'x ← x + f(x)' : 'x ← f(x)');
        normRead.set('norm at layer 24', fmt.n((v ? residualCurve : replaceCurve).at(-1)[1], 4));
        normPlot.render();
      },
    });
    normRead.set('norm at layer 24', fmt.n(residualCurve.at(-1)[1], 4));
    root.append(card('Why residuals exist',
      'Here the same tiny deterministic function <code class="inline">f</code> is applied for 24 layers. With a residual, each layer makes a small update; without one, the next layer receives only the small output of <code class="inline">f</code>.',
      row(
        grow(normPlot.canvas),
        el('div', { class: 'col', style: { flex: '0 1 290px' } }, residualToggle, normRead,
          legend([{ c: 'accent', label: 'selected update' }, { c: 'muted', label: 'other update' }]),
          note('This is a numerical demonstration, not a trained model: the seed, starting vector, gains and offsets are fixed, so reloads reproduce the same two curves.')))));

    root.append(card('The layer loop in the engine', null,
      row(
        grow(prose(`
          <p>TensorFold’s forward map describes the whole 24-block pass in about twenty lines: normalize the stream, run either a delta mixer or an attention block, normalize again, run the gated MLP, and write the next delta. After the loop comes the final norm and the projection to logits.</p>
          <p>The surprising part is what is <i>not</i> in the loop: there is no residual <code class="inline">+</code> operator there. <code class="inline">s.h</code> carries the stream and <code class="inline">s.r</code> carries the last sublayer’s delta. The next norm kernel fuses <code class="inline">h += r</code> with re-normalisation — an engineering choice that avoids a separate pass over memory.</p>
        `)),
        el('div', { class: 'col', style: { flex: '1 1 430px' } },
          code(`for (m.weights.blocks, 0..) |block, i| {\n    enc.norm(s.h, if (i == 0) null else s.r, block.input_norm, s.x, total);\n    switch (block.mixer) {\n        .delta => |d| delta(enc, s, d, i, segments, record, total),\n        .attention => |a| attention(enc, s, a, i, segments, total),\n    }\n    enc.norm(s.h, s.r, block.post_norm, s.x, total);\n    enc.projection(block.gate, s.x, 0, s.gate, 0, total);\n    enc.projection(block.up, s.x, 0, s.up, 0, total);\n    enc.pipe("mlp_act");\n    ...\n    enc.projection(block.down, s.act, 0, s.r, 0, total);\n}`),
          el('div', { class: 'sub', style: { marginTop: '8px' }, text: 'map excerpt · forward.zig:87–106' })))));

    const kinds = Array.from({ length: LAYERS }, (_, i) => QWEN_ATTENTION.includes(i) ? 'attention' : 'gated delta-net');
    const kindPlot = new Plot({
      width: 760, height: 220, pad: { l: 42, r: 14, t: 20, b: 52 },
      xlim: [0, LAYERS], ylim: [0, 1.18], xlabel: 'layer index',
    });
    kindPlot.onDraw((p) => {
      p.axes({ xticks: 0, yticks: 0, grid: false, zero: false, frame: true });
      p.bars(Array.from({ length: LAYERS }, () => 1), {
        color: kinds.map((k) => k === 'attention' ? 'c5' : 'c3'), width: 0.82,
        labels: Array.from({ length: LAYERS }, (_, i) => String(i)),
      });
      p.text(p.pad.l, 8, 'the six attention blocks are every fourth layer (3, 7, 11, 15, 19, 23)',
        { color: 'muted', size: 10, px: true, align: 'left' });
    });
    root.append(card('A “transformer” is a family, not a fixed recipe',
      'Qwen3.5-2B has 24 layers, but only 6 are attention: <b>every 4th layer is full attention; the other 18 are gated-delta-net linear attention</b>. The dispatch in the loop is therefore a real choice, not a diagram shortcut.',
      kindPlot.canvas,
      legend([{ c: 'c5', label: 'full attention (6)' }, { c: 'c3', label: 'gated-delta-net linear mixer (18)' }]),
      note('This lesson treats each mixer as a box that reads the stream and writes a delta. What happens inside the attention boxes — and what they must remember between tokens — is the next lesson.')));

    return () => {
      heat.destroy(); current.destroy(); normPlot.destroy(); kindPlot.destroy();
    };
  },
};
