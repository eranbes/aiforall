// 17-deltanet.js — the 18 layers of Qwen3.5-2B that are not attention.

import { Plot, color, div, seq } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, fmt } from '../ui.js';
import { mulberry32, cosine } from '../engine/tensor.js';

const DK = 8, DV = 8;          // toy widths; the real ones are 128 × 128
const PAIRS = [
  { name: 'alpha', c: 'c1' },
  { name: 'beta', c: 'c2' },
  { name: 'gamma', c: 'c3' },
  { name: 'delta', c: 'c4' },
  { name: 'epsilon', c: 'c5' },
  { name: 'zeta', c: 'c6' },
];

/* Deterministic unit keys and distinct value patterns. */
const rng = mulberry32(17);
const unit = (n) => {
  const v = Array.from({ length: n }, () => rng() * 2 - 1);
  const m = Math.hypot(...v);
  return v.map((x) => x / m);
};
for (const p of PAIRS) { p.k = unit(DK); p.v = unit(DV).map((x) => x * 1.2); }

export default {
  id: '17-deltanet',
  num: '17',
  group: 'one real model',
  title: 'The other eighteen layers',
  blurb: 'Only 6 of Qwen3.5-2B\'s 24 layers are attention. The other 18 are gated delta-net: a fixed-size associative memory that is updated by gradient descent as the tokens arrive. No introduction to transformers will have mentioned it, and it is most of the model.',

  render(root) {
    const S = {
      state: Array.from({ length: DV }, () => new Float64Array(DK)),
      written: [],
      g: 0.98,
      beta: 1.0,
      query: 0,
      ctx: 8000,
    };

    const reset = () => {
      S.state = Array.from({ length: DV }, () => new Float64Array(DK));
      S.written = [];
      refresh();
    };

    /** Exactly the kernel's inner loop, in JS. */
    const write = (p) => {
      // 1. decay: state *= g
      for (const rowV of S.state) for (let i = 0; i < DK; i++) rowV[i] *= S.g;
      // 2. what does the memory currently return for this key?
      const recalled = S.state.map((rowV) => {
        let s = 0;
        for (let i = 0; i < DK; i++) s += rowV[i] * p.k[i];
        return s;
      });
      // 3. the error, scaled by the write strength
      const delta = recalled.map((r, dv) => (p.v[dv] - r) * S.beta);
      // 4. write the correction along the key direction
      for (let dv = 0; dv < DV; dv++) for (let i = 0; i < DK; i++) S.state[dv][i] += p.k[i] * delta[dv];
      S.written.push(p.name);
      refresh();
    };

    const readWith = (k) => S.state.map((rowV) => {
      let s = 0;
      for (let i = 0; i < DK; i++) s += rowV[i] * k[i];
      return s;
    });

    /* ---------- the state matrix ---------- */
    const statePlot = new Plot({
      width: 300, height: 290, pad: { l: 34, r: 10, t: 26, b: 10 },
      xlim: [0, DK], ylim: [0, DV],
    });
    statePlot.onDraw((p) => {
      const m = S.state.map((r) => Array.from(r));
      const max = Math.max(0.25, ...m.flat().map(Math.abs));
      p.matrix(m, { cmap: div, max, gap: 2 });
      p.text(p.pad.l, 12, `state: ${DV} × ${DK} numbers, and that is all there is`,
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    /* ---------- recall ---------- */
    const recallPlot = new Plot({
      width: 350, height: 290, pad: { l: 44, r: 12, t: 26, b: 46 },
      xlim: [0, DV], ylim: [-1.6, 1.6], ylabel: 'value',
    });
    recallPlot.onDraw((p) => {
      const q = PAIRS[S.query];
      const out = readWith(q.k);
      p.ylim = [Math.min(-0.6, ...out, ...q.v) * 1.2, Math.max(0.6, ...out, ...q.v) * 1.2];
      p.axes({ xticks: 0, yticks: 5, zero: true });
      p.bars(q.v, { color: 'border', width: 0.76, labels: q.v.map((_, i) => 'd' + i) });
      p.bars(out, { color: q.c, width: 0.42 });
      const sim = cosine(Float64Array.from(out), Float64Array.from(q.v));
      p.text(p.pad.l, 12, `query "${q.name}"  ·  cosine ${fmt.n(sim, 3)}`,
        { color: S.written.includes(q.name) ? 'ok' : 'muted', size: 10, px: true, align: 'left' });
    });

    const read = stats([
      { k: 'pairs written', v: '0' }, { k: 'state size', v: `${DV * DK} numbers` },
      { k: 'recall of query', v: '—' }, { k: 'mean recall', v: '—' },
    ]);

    const writtenBox = el('div', { class: 'tokens' });

    const refresh = () => {
      let tot = 0, n = 0;
      for (const p of PAIRS) {
        if (!S.written.includes(p.name)) continue;
        tot += cosine(Float64Array.from(readWith(p.k)), Float64Array.from(p.v));
        n++;
      }
      const q = PAIRS[S.query];
      const sim = cosine(Float64Array.from(readWith(q.k)), Float64Array.from(q.v));
      read.set('pairs written', String(S.written.length));
      read.set('recall of query', fmt.n(sim, 3), sim > 0.9 ? 'ok' : sim > 0.6 ? 'warn' : 'err');
      read.set('mean recall', n ? fmt.n(tot / n, 3) : '—', n && tot / n > 0.9 ? 'ok' : n ? 'warn' : null);
      writtenBox.replaceChildren(...(S.written.length
        ? S.written.map((w, i) => el('span', { class: 'tok', style: { opacity: String(Math.pow(S.g, (S.written.length - 1 - i) * 1)) } }, w))
        : [el('span', { style: { fontSize: '12px', color: 'var(--muted)' }, text: 'memory is empty' })]));
      statePlot.render(); recallPlot.render(); costPlot.render();
    };

    root.append(card('An associative memory you can write to',
      'This is the whole mechanism. Write a key→value pair into the state; read it back with a key. The state never grows — writing more just overwrites the same numbers.',
      row(
        el('div', {}, statePlot.canvas),
        el('div', {}, recallPlot.canvas,
          el('div', { style: { height: '6px' } }),
          legend([{ c: 'border', label: 'the value that was written' }, { c: 'c1', label: 'what comes back out' }])),
        el('div', { class: 'col', style: { flex: '1 1 270px' } },
          read,
          el('div', { class: 'sub', style: { margin: '4px 0 2px' }, text: 'written so far (faded = decayed)' }),
          writtenBox,
          el('div', { class: 'sub', style: { margin: '6px 0 2px' }, text: 'write a pair' }),
          btns(...PAIRS.map((p) => button(p.name, () => write(p), { ghost: true }))),
          el('div', { class: 'sub', style: { margin: '6px 0 2px' }, text: 'read with' }),
          btns(...PAIRS.map((p, i) => button(p.name, () => { S.query = i; refresh(); }, { ghost: S.query !== i }))),
          slider({ label: 'g — decay (forget gate)', min: 0.6, max: 1, step: 0.005, value: 0.98, fmt: (v) => v.toFixed(3), onInput: (v) => { S.g = v; refresh(); } }),
          slider({ label: 'β — write strength', min: 0.05, max: 1, step: 0.05, value: 1, fmt: (v) => v.toFixed(2), onInput: (v) => { S.beta = v; refresh(); } }),
          btns(button('reset memory', reset, { ghost: true }))))));

    root.append(card('What you just saw, and its price',
      null,
      row(
        grow(prose(`
          <p><b>Write all six pairs, then read each one back.</b> The first ones come back degraded: the keys are
          not orthogonal, so every write smears a little onto its neighbours. A fixed number of parameters can
          only hold so many associations, and the failure mode is interference, not an error.</p>
          <p>That is the fundamental difference from attention. Attention keeps <i>every</i> token and can recall
          any of them exactly — at the cost of storing them all. This keeps a fixed-size summary and recalls
          approximately — at no growing cost at all.</p>
          <p><b>Drop g to 0.9</b> and write six pairs. The early ones are gone: the decay gate is a deliberate
          forgetting rate, and it is learned per head, per layer.</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b>β is a learning rate.</b> Set it to 0.2 and write the same pair five times — recall climbs
                toward the target instead of landing on it. You have seen this curve before, in lesson 11.`),
          note(`<b>Why the keys are normalised.</b> The kernel L2-normalises q and k per head before the update
                (<code class="inline">gdn_pre.metal:49-59</code>), which keeps the write magnitude bounded no
                matter how large the projections get.`)))));

    /* ---------- it is gradient descent ---------- */
    root.append(card('This layer is doing gradient descent at inference time',
      'Not an analogy. The update rule is one step of least-squares on the memory\'s own prediction error.',
      row(
        grow(
          el('div', { class: 'eq', html: 'loss = ½ ‖ <span class="hl">S·k</span> − <span class="n">v</span> ‖²' }),
          el('div', { class: 'eq', html: '∂loss/∂S = (S·k − v) ⊗ k' }),
          el('div', { class: 'eq', html: 'S ← S − <span class="n">β</span> · (S·k − v) ⊗ k&nbsp;&nbsp;<span style="color:var(--muted)">— which is exactly the kernel below</span>' }),
          el('pre', { class: 'code', html:
            `<span class="cm">// zig/kernels/metal/qwen3_5/gdn_chain.metal:53-65</span>\n` +
            `float kv_mem = 0.0f;\n` +
            `for (int i = 0; i &lt; n_per_t; ++i) {\n` +
            `    state[i] = state[i] * <b>gc</b>;             <span class="cm">// decay: forget a little</span>\n` +
            `    kv_mem += state[i] * kc[i];           <span class="cm">// what the memory predicts for k</span>\n` +
            `}\n` +
            `kv_mem = simd_sum(kv_mem);\n` +
            `auto <b>delta = (vc - kv_mem) * bc</b>;          <span class="cm">// the prediction error × β</span>\n` +
            `float out = 0.0f;\n` +
            `for (int i = 0; i &lt; n_per_t; ++i) {\n` +
            `    <b>state[i] = state[i] + kc[i] * delta</b>;  <span class="cm">// the gradient step</span>\n` +
            `    out += state[i] * qc[i];              <span class="cm">// read with the query</span>\n` +
            `}\n` +
            `out = simd_sum(out);` })),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Lesson 11 said training is the only backwards part of the story.</b> That was true of the
                weights. This layer runs the same algorithm <i>forwards</i>, during inference, on a state that
                is thrown away when the conversation ends. The weights that decide <i>how</i> to learn were
                themselves learned by gradient descent.`),
          note(`<b>The classical name is the delta rule</b> — Widrow–Hoff, 1960. Modern linear-attention work
                rediscovered it as "fast weights": a network whose activations <i>are</i> another network's
                weights.`),
          note(`<b>Why a subtraction and not just a write.</b> Writing <code class="inline">S += k ⊗ v</code>
                (plain linear attention) accumulates without bound and never corrects itself. Subtracting what
                the memory already returns means a repeated key updates rather than piles up.`)))));

    /* ---------- the gates ---------- */
    root.append(card('Where g and β come from',
      'Both gates are computed per head, per token, by the kernel that runs just before the chain.',
      row(
        grow(el('pre', { class: 'code', html:
          `<span class="cm">// zig/kernels/metal/qwen3_5/gdn_pre.metal:40-67</span>\n` +
          `for (int tap = 0; tap &lt; TAPS; tap++) {        <span class="cm">// TAPS = 4: a short causal</span>\n` +
          `    const int row = windows[w * TAPS + tap];  <span class="cm">// convolution over 4 positions</span>\n` +
          `    acc += float(CW[c * TAPS + tap]) * xv;\n` +
          `}\n` +
          `const float conv = float(bfloat(acc));\n` +
          `const float sig = 1.0f / (1.0f + exp(-conv));\n` +
          `vals[j] = conv * sig;                         <span class="cm">// SiLU</span>\n` +
          `...\n` +
          `const float sp = max(s, 0) + log(1 + exp(-abs(s)));   <span class="cm">// softplus</span>\n` +
          `<b>G[...]    = exp(-exp(ALOG[hv]) * sp);</b>                 <span class="cm">// decay in (0, 1)</span>\n` +
          `<b>BETA[...] = 1 / (1 + exp(-Bin[...]));</b>                 <span class="cm">// sigmoid, in (0, 1)</span>` })),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          prose(`
            <p>Three things happen before the recurrence:</p>
            <ul>
              <li>A <b>4-tap causal convolution</b> over the last four positions of all 6,144 channels — a cheap
              local mixer, so a token can see its immediate neighbours without attention.</li>
              <li><b>SiLU</b>, then L2 normalisation of the query and key per head.</li>
              <li>The two gates: <b>g</b> through <code class="inline">exp(−exp(a)·softplus(…))</code>, which is
              guaranteed to land in (0,1); <b>β</b> through a plain sigmoid.</li>
            </ul>`),
          note(`<b>That convolution explains two config constants.</b>
                <code class="inline">conv_taps = 4</code> and <code class="inline">conv_dim = 6144</code> — and
                6144 is exactly q (16×128) + k (16×128) + v (16×128). The kernel computes
                <code class="inline">C = 2*NK*DK + NV*DV</code>.`),
          note(`<b>It also means there is a second kind of state.</b> Besides the memory matrix, each layer must
                keep the last 3 positions of those 6,144 channels so the convolution can continue — the kernel
                writes them to <code class="inline">CO</code> at the end.`)))));

    /* ---------- the cost argument ---------- */
    const costPlot = new Plot({
      width: 520, height: 300, pad: { l: 62, r: 16, t: 18, b: 44 },
      xlim: [0, 32], ylim: [0, 1], xlabel: 'context length (thousands of tokens)', ylabel: 'state per stream (MiB)',
    });
    costPlot.onDraw((p) => {
      // verified from the kernels and config:
      //   delta-net: 16 heads × 128 × 128 fp32 = 1 MiB per layer, × 18 layers — constant
      //   attention: 6 layers × 2 (K,V) × 2 kv_heads × 256 dims × 2 bytes = 12,288 B per token
      const gdn = 18;                       // MiB, constant
      const perTok = 12288;
      const attn = [], total = [], flat = [];
      for (let k = 0; k <= 32; k += 0.5) {
        const a = (k * 1000 * perTok) / 2 ** 20;
        attn.push([k, a]); flat.push([k, gdn]); total.push([k, a + gdn]);
      }
      p.ylim = [0, Math.max(40, attn[attn.length - 1][1] * 1.15)];
      p.axes({ xticks: 5, yticks: 5, zero: false });
      p.area(flat, { color: 'c4', alpha: 0.18 });
      p.line(flat, { color: 'c4', width: 2.5 });
      p.line(attn, { color: 'c1', width: 2.5 });
      p.line(total, { color: 'muted', width: 1.5, dash: [4, 4] });
      const cross = (gdn * 2 ** 20) / perTok / 1000;
      p.line([[cross, 0], [cross, p.ylim[1]]], { color: 'c5', width: 1.5, dash: [3, 3] });
      p.scatter([[cross, gdn]], { color: 'c5', r: 5, stroke: 'bg' });
      p.text(cross + 0.4, gdn, ` crossover ≈ ${fmt.n(cross * 1000, 0)} tokens`, { color: 'c5', size: 10, align: 'left' });
      p.text(p.pad.l + 6, p.pad.t + 10, '18 delta-net layers: 18 MiB, forever', { color: 'c4', size: 11, px: true, align: 'left' });
      p.text(p.pad.l + 6, p.pad.t + 26, '6 attention layers: 12 KiB per token', { color: 'c1', size: 11, px: true, align: 'left' });
    });

    root.append(card('Why a model would do this at all',
      'The state is 16 heads × 128 × 128 floats per layer — one mebibyte, no matter how long the conversation is.',
      row(
        el('div', {}, costPlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          prose(`
            <p>Past about <b>1,500 tokens</b>, the six attention layers cost more memory than all eighteen
            delta-net layers combined. At 32,000 tokens attention needs 375 MiB and the delta-net layers still
            need 18.</p>
            <p><b>Below that crossover the recurrence is the more expensive mixer</b>, and it is worth being
            precise about why. Every token, each delta-net layer must read and write its entire state:
            18 layers × 16 heads × 128 × 128 × 4 B = <b>18,874,368 bytes per token</b>, whether the conversation
            is two tokens long or two million. Attention touches 12,288 bytes per token of context — less, until
            context reaches 1,536 tokens. In MAC terms a delta-net layer equals an attention layer at just
            <b>256 keys</b>.</p>
            <p>So this is not "recurrence is cheaper". It is a <i>flat</i> cost versus a <i>growing</i> one. The
            architecture is a bet that conversations are long — and on short ones it loses.</p>
            <p>That is the trade the hybrid is making — and why "the transformer" is no longer one architecture.
            Qwen3.5-2B is three quarters recurrent network by layer count.</p>`),
          note(`<b>So why keep any attention?</b> Because exact recall still matters: "what was the variable
                called on line 40?" is a lookup, not a summary. Six layers of exact attention, interleaved every
                fourth layer, appear to be enough to carry it.`),
          note(`<b>Why you have not heard of this.</b> It postdates every popular explainer. The mechanism is
                not in "Attention Is All You Need", and most courses stop there — which is exactly why this
                course reads the engine instead of the canon.`)))));

    refresh();
  },
};
