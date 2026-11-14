// 16-qwen-anatomy.js — what a real model implementation actually contains.

import { Plot, color, mix } from '../plot.js';
import { card, el, prose, note, row, grow, btns, button, stats, legend, fmt } from '../ui.js';

/* Measured from the repo, not estimated:
   find zig/src/families/qwen3_5 -name '*.zig' | xargs wc -l   →  28,711 lines, 179 files
   cat zig/kernels/metal/qwen3_5/*                             →   1,601 lines, 18 files   */
const CATEGORIES = [
  { k: 'CUDA backend', lines: 7404, c: 'c1', what: 'The same model again for NVIDIA: its own engine, cache, prefill, paging, graph capture, penalties, rope tables, warmup.' },
  { k: 'HIP backend', lines: 5801, c: 'c2', what: 'And again for AMD. Separate engine, separate state, separate drafter, separate cost model.' },
  { k: 'tests', lines: 5168, c: 'c3', what: 'Contracts: synthetic safetensors, mock drivers, GPU regressions, byte-equal disk round-trips, paged-vs-contiguous equality.' },
  { k: 'drafting / MTP', lines: 3402, c: 'c4', what: 'Speculative decoding (lesson 14): DFlash2 drafters, MTP heads, tree walkers, batched drafting, acceptance priors.' },
  { k: 'weights / formats', lines: 2662, c: 'c5', what: 'Reading the checkpoint: MLX affine, EXL3, ModelOpt NVFP4/FP8, plain BF16, shard tables, HF name fallbacks.' },
  { k: 'core — the Metal path', lines: 2538, c: 'c6', what: 'The model itself on Apple silicon: config, forward, state, weights, draw, prefill, snapshot, backend.' },
  { k: 'Living Weights', lines: 1736, c: 'muted', what: 'Runtime training (--slide): a trainer, gradients, pockets beside each MLP, disk shards.' },
];

const METAL_PATH = [
  ['config.zig', 584, 'Pinned geometry and format admission. Refuses any checkpoint whose numbers differ.'],
  ['backend.zig', 256, 'Qwen recurrence, windows and keyed draws behind the lane engine\'s backend interface.'],
  ['forward.zig', 237, 'Row projections with segment-local recurrence and causal attention. The whole 24-layer pass.'],
  ['state.zig', 213, 'Per-stream committed recurrence and KV state, plus pooled window snapshots.'],
  ['draw.zig', 160, 'The host draw over a row of bf16 logits — greedy and keyed sampling (lesson 01).'],
  ['weights.zig', 101, 'Views into one checkpoint buffer; a tied head aliases the packed embedding.'],
  ['kernels.zig', 81, 'Compiles and indexes every Metal pipeline; looks projections up by shape.'],
  ['snapshot.zig', 71, 'A committed prompt state in host memory, independent of page placement (lesson 13).'],
  ['model.zig', 46, 'A native Qwen3.5-2B: validated config, checkpoint views, Metal pipelines.'],
  ['prefill.zig', 42, 'Prompt reuse and capture hooks around the unchanged forward.'],
  ['qwen3_5.zig', 8, 'The entry point. Eight lines.'],
];

const KERNELS = [
  ['qmm_248320_2048', 137, 'out', 'The output projection: hidden 2048 → one logit per vocabulary entry (248,320).'],
  ['qmm_6144_2048', 137, 'mlp', 'MLP gate and up: 2048 → intermediate 6144.'],
  ['qmm_2048_6144', 137, 'mlp', 'MLP down: 6144 → 2048.'],
  ['qmm_2048_2048', 137, 'attn', 'Square projections — query (8 heads × 256) and output.'],
  ['qmm_512_2048', 137, 'attn', 'Key and value: 2 KV heads × 256 = 512. GQA, visible in a filename.'],
  ['qmm_4096_2048', 137, 'gdn', 'A wide projection on the delta-net path.'],
  ['qmm_16_2048', 137, 'gdn', 'Sixteen outputs — one per linear head. Per-head scalars, not a vector.'],
  ['attn_partial', 93, 'attn', 'Flash-style partial attention: block-amortised online softmax, one rescale per 4 keys.'],
  ['attn_merge', 33, 'attn', 'Merges the partial results into the final attention output.'],
  ['layout', 86, 'attn', 'Embedding, head norm, queries, keys, attention gate — and RoPE fused into the KV-cache write.'],
  ['paged', 112, 'attn', 'The KV-page twins of the keys and partial-attention kernels.'],
  ['gdn_pre', 79, 'gdn', 'Gated delta-net, stage 1 of 3.'],
  ['gdn_chain', 82, 'gdn', 'The recurrent chain — the part that replaces attention in 18 of 24 layers.'],
  ['gdn_post', 38, 'gdn', 'Gated delta-net, stage 3 of 3.'],
  ['norm', 42, 'norm', 'RMSNorm with the previous sublayer\'s residual folded in (lesson 05).'],
  ['norm_nores', 39, 'norm', 'The same kernel for layer 0, which has no residual to add.'],
  ['mlp_act', 19, 'mlp', 'The elementwise SwiGLU step: act(gate) ⊙ up.'],
];

const KIND_COLOR = { out: 'c5', mlp: 'c3', attn: 'c1', gdn: 'c4', norm: 'c6' };

export default {
  id: '16-qwen-anatomy',
  num: '16',
  group: 'one real model',
  title: 'Anatomy of one model',
  blurb: 'Qwen3.5-2B as TensorFold actually implements it: 179 files and 28,711 lines of Zig, plus 1,601 lines of Metal. The mathematics you learned is in there. So is everything else, and the ratio is the lesson.',

  render(root) {
    /* ---------- where the lines are ---------- */
    const totalLines = CATEGORIES.reduce((a, c) => a + c.lines, 0);
    let picked = 5;   // the Metal path, by default

    const breakdown = new Plot({
      width: 660, height: 150, pad: { l: 12, r: 12, t: 26, b: 40 }, xlim: [0, 1], ylim: [0, 1],
    });
    breakdown.onDraw((p) => {
      let x = p.pad.l;
      const y = p.pad.t + 6, h = 52;
      CATEGORIES.forEach((c, i) => {
        const w = (c.lines / totalLines) * p.iw;
        p.ctx.fillStyle = color(c.c);
        p.ctx.globalAlpha = i === picked ? 1 : 0.55;
        p.ctx.fillRect(x, y, Math.max(1, w - 2), h);
        p.ctx.globalAlpha = 1;
        if (w > 54) {
          p.ctx.fillStyle = color('bg');
          p.ctx.font = 'bold 10px ui-monospace, Menlo, monospace';
          p.ctx.textAlign = 'center'; p.ctx.textBaseline = 'middle';
          p.ctx.fillText(`${(c.lines / totalLines * 100).toFixed(0)}%`, x + w / 2, y + h / 2);
        }
        if (i === picked) {
          p.ctx.strokeStyle = color('fg');
          p.ctx.lineWidth = 2;
          p.ctx.strokeRect(x - 1, y - 2, w, h + 4);
        }
        x += w;
      });
      p.text(p.pad.l, p.pad.t - 10, `${totalLines.toLocaleString()} lines of Zig for one model family — click a band`,
        { color: 'muted', size: 11, px: true, align: 'left' });
    });
    breakdown.interact({
      down: (_, ev) => {
        const px = breakdown.atPx(ev).x - breakdown.pad.l;
        let acc = 0;
        for (let i = 0; i < CATEGORIES.length; i++) {
          acc += (CATEGORIES[i].lines / totalLines) * breakdown.iw;
          if (px <= acc) { picked = i; detail(); breakdown.render(); return; }
        }
      },
      hover: (_, ev) => { breakdown.canvas.style.cursor = ev ? 'pointer' : 'default'; },
    });

    const detailBox = el('div', { class: 'note' });
    const detail = () => {
      const c = CATEGORIES[picked];
      detailBox.innerHTML = `<b style="color:var(--${c.c})">${c.k}</b> — ${c.lines.toLocaleString()} lines,
        ${(c.lines / totalLines * 100).toFixed(1)}% of the family.<br>${c.what}`;
    };
    detail();

    root.append(card('Where the 28,711 lines actually go',
      'Measured with <code class="inline">wc -l</code> over <code class="inline">zig/src/families/qwen3_5/</code>, not estimated.',
      breakdown.canvas,
      el('div', { style: { height: '8px' } }),
      legend(CATEGORIES.map((c) => ({ c: c.c, label: c.k }))),
      el('div', { style: { height: '10px' } }),
      detailBox,
      el('div', { style: { height: '12px' } }),
      row(
        grow(prose(`
          <p>The arithmetic this course spent eleven lessons on — attention, the MLP, the norms, the sampler —
          lives in the band marked <b>core</b>: 2,538 lines, <b>8.8%</b> of the family.</p>
          <p>Nearly half the family (46%) is the same model implemented twice more, for CUDA and for HIP. Another
          18% is tests. 12% is speculative decoding. 9% is reading five different checkpoint formats.</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b>This is the honest shape of ML engineering.</b> The model is a small, elegant core surrounded
                by an enormous amount of work that exists because it must run on three vendors' hardware, read
                eight weight formats, survive concurrency, and be provably correct.`),
          note(`<b>And this is one family of thirty.</b> <code class="inline">zig/src/families/</code> holds
                336,067 lines across 1,849 files. Each new architecture repeats most of this structure.`)))));

    /* ---------- the Metal path ---------- */
    const metalTotal = METAL_PATH.reduce((a, f) => a + f[1], 0);
    root.append(card('What a Mac actually runs',
      `Strip away the other vendors, the formats, the drafters and the tests, and the model has these eleven files at its spine — <b>${metalTotal.toLocaleString()} lines of Zig</b> plus <b>1,601 lines of Metal</b>. (About 22 files in total once helpers are counted; the core band above reads 2,538 lines because it also includes CPU reference kernels and admission checks that no decode step touches.)`,
      row(
        grow(el('table', {},
          el('thead', {}, el('tr', {},
            el('th', { text: 'file' }), el('th', { text: 'lines' }), el('th', { text: 'what it is for' }))),
          el('tbody', {}, ...METAL_PATH.map(([f, n, w]) => el('tr', {},
            el('td', { text: f }),
            el('td', { text: String(n) }),
            el('td', { style: { textAlign: 'left', color: 'var(--fg)' }, text: w })))))),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>config.zig is the largest file.</b> Not the forward pass — the admission check. It pins the
                geometry at compile time and refuses any checkpoint whose numbers differ, because every kernel
                is specialised to those exact widths.`),
          note(`<b>forward.zig is 237 lines.</b> The entire 24-layer model, including the hybrid dispatch between
                attention and delta-net layers. Lesson 05 quoted the loop; this is the file it lives in.`),
          note(`<b>qwen3_5.zig is 8 lines.</b> The public entry point of a 2-billion-parameter language model.`)))));

    /* ---------- the kernels ---------- */
    const kernelBox = el('div', { class: 'col', style: { gap: '3px' } });
    let kfilter = 'all';
    const renderKernels = () => {
      kernelBox.replaceChildren(...KERNELS.filter((k) => kfilter === 'all' || k[2] === kfilter).map(([name, n, kind, what]) =>
        el('div', {
          style: {
            display: 'flex', gap: '10px', alignItems: 'baseline', padding: '6px 10px',
            borderRadius: '7px', background: 'var(--surface-2)',
            borderLeft: `3px solid var(--${KIND_COLOR[kind]})`,
          },
        },
          el('span', { style: { font: '11.5px ui-monospace,Menlo,monospace', color: `var(--${KIND_COLOR[kind]})`, minWidth: '160px' }, text: name }),
          el('span', { style: { font: '10px ui-monospace,Menlo,monospace', color: 'var(--muted)', minWidth: '44px' }, text: n + ' ln' }),
          el('span', { style: { fontSize: '12.5px' }, text: what }))));
    };
    renderKernels();

    root.append(card('You can read the architecture off the filenames',
      'Eighteen Metal files. Seven of them are the same quantised matmul compiled for one specific matrix shape — and the shapes <i>are</i> the model.',
      row(
        grow(
          btns(...[['all', 'all 18'], ['attn', 'attention'], ['gdn', 'delta-net'], ['mlp', 'MLP'], ['norm', 'norms'], ['out', 'output']]
            .map(([k, t]) => button(t, () => { kfilter = k; renderKernels(); }, { ghost: kfilter !== k }))),
          el('div', { style: { height: '10px' } }),
          kernelBox),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          el('pre', { class: 'code', html:
            `<span class="cm">// zig/src/families/qwen3_5/kernels.zig:27-36</span>\n` +
            `pub fn projection(self: *const Kernels, n: usize, k: usize) ?Projection {\n` +
            `    inline for (sources.all, 0..) |s, i| {\n` +
            `        if (s.columns &gt; 0) {\n` +
            `            const nk = comptime <b>dimensions(s.key)</b>;\n` +
            `            if (<b>n == nk[0] and k == nk[1]</b>) return .{ ... };\n` +
            `        }\n` +
            `    }\n` +
            `    return null;\n` +
            `}\n\n` +
            `<span class="cm">// ...and dimensions() parses them out of the kernel's NAME, at compile time</span>\n` +
            `fn dimensions(comptime key: []const u8) [2]usize {\n` +
            `    const tail = key[4..];\n` +
            `    const at = std.mem.indexOfScalar(u8, tail, '_').?;\n` +
            `    return .{ parseInt(tail[0..at]), parseInt(tail[at + 1 ..]) };\n` +
            `}` }),
          note(`<b>A matmul is dispatched by its shape.</b> There is no general matrix-multiply routine taking
                dimensions at runtime. The engine asks "do I have a kernel for 248320 × 2048?" and gets one
                compiled for exactly that.`),
          note(`<b>Why that is worth the rigidity.</b> Loop bounds, tile sizes and register budgets become
                compile-time constants, so nothing is computed from variables in the inner loop. It is also why
                <code class="inline">config.zig</code> must refuse a checkpoint of different widths — there
                would be no kernel for it.`)))));

    /* ---------- the decode step ---------- */
    const seq = new Plot({ width: 660, height: 210, pad: { l: 0, r: 0, t: 0, b: 0 }, plain: true });
    const STEPS = [
      ['qwen35_embed', 'out', 'look up the token\'s row'],
      ['norm / norm_nores', 'norm', 'RMSNorm, residual folded in'],
      ['qmm_2048_2048 ×2', 'attn', 'queries, output proj'],
      ['qmm_512_2048 ×2', 'attn', 'keys, values (GQA)'],
      ['layout', 'attn', 'RoPE + write to KV cache'],
      ['attn_partial', 'attn', 'online-softmax attention'],
      ['attn_merge', 'attn', 'combine partials'],
      ['gdn_pre / chain / post', 'gdn', 'the other 18 layers'],
      ['norm', 'norm', 'pre-MLP norm'],
      ['qmm_6144_2048 ×2', 'mlp', 'gate + up'],
      ['mlp_act', 'mlp', 'SwiGLU'],
      ['qmm_2048_6144', 'mlp', 'down'],
      ['qwen35_head_norm', 'norm', 'final norm'],
      ['qmm_248320_2048', 'out', 'logits'],
      ['draw.zig', 'out', 'sample one token'],
    ];
    seq.onDraw((p) => {
      const c = p.ctx;
      const cols = 5, cw = p.w / cols, rh = 62;
      STEPS.forEach(([name, kind, what], i) => {
        const x = (i % cols) * cw + 7, y = Math.floor(i / cols) * rh + 12;
        c.fillStyle = color('surface');
        c.strokeStyle = color(KIND_COLOR[kind]);
        c.lineWidth = 1.4;
        c.beginPath(); c.roundRect(x, y, cw - 14, rh - 14, 8); c.fill(); c.stroke();
        c.fillStyle = color(KIND_COLOR[kind]);
        c.font = 'bold 9.5px ui-monospace, Menlo, monospace';
        c.textAlign = 'left'; c.textBaseline = 'top';
        c.fillText(name.length > 20 ? name.slice(0, 19) + '…' : name, x + 8, y + 8);
        c.fillStyle = color('muted');
        c.font = '9.5px -apple-system, system-ui, sans-serif';
        wrap(c, what, x + 8, y + 23, cw - 28, 11);
        if (i % cols !== cols - 1 && i < STEPS.length - 1) {
          c.strokeStyle = color('border');
          c.beginPath(); c.moveTo(x + cw - 13, y + (rh - 14) / 2); c.lineTo(x + cw - 6, y + (rh - 14) / 2); c.stroke();
        }
      });
    });

    root.append(card('One token, as a sequence of kernel launches',
      'This is what "running a language model" is, concretely: a fixed order of GPU programs, dispatched 24 times, then once more for the logits.',
      seq.canvas,
      el('div', { style: { height: '6px' } }),
      legend([
        { c: 'c1', label: 'attention layers (6 of 24)' },
        { c: 'c4', label: 'delta-net layers (18 of 24)' },
        { c: 'c3', label: 'MLP (every layer)' },
        { c: 'c6', label: 'norms' },
        { c: 'c5', label: 'embedding + output' },
      ]),
      el('div', { style: { height: '10px' } }),
      note(`Each box is a real kernel in <code class="inline">zig/kernels/metal/qwen3_5/</code>. The next lesson
            opens the <b>delta-net</b> boxes — the three kernels that do the work in 18 of the 24 layers, and that
            no introduction to transformers will have told you about.`)));
  },
};

function wrap(c, text, x, y, maxW, lh) {
  const words = String(text).split(' ');
  let line = '', yy = y;
  for (const w of words) {
    const t = line ? line + ' ' + w : w;
    if (c.measureText(t).width > maxW && line) { c.fillText(line, x, yy); line = w; yy += lh; }
    else line = t;
  }
  if (line) c.fillText(line, x, yy);
}
