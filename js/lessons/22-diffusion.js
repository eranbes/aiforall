// 22-diffusion.js — the other kind of generation: refine the whole output, not append to it.

import { Plot, color, seq } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, loop, fmt } from '../ui.js';
import { mulberry32 } from '../engine/tensor.js';
import { NGram, tokenize, detokenize } from '../engine/ngram.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';

const N = 32;                       // the toy "latent" grid
const cell = (i, j) => {
  const y = (i - (N - 1) / 2) / (N / 2);
  const x = (j - (N - 1) / 2) / (N / 2);
  return { x, y, r: Math.hypot(x, y) };
};

/** What the prompt asks for: a ring with a dot. */
const TARGET = (() => {
  const t = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const { r } = cell(i, j);
      const ring = Math.exp(-((r - 0.62) ** 2) / 0.012);
      const dot = Math.exp(-(r ** 2) / 0.02);
      t[i * N + j] = Math.min(1, ring * 0.95 + dot);
    }
  }
  return t;
})();

/** What "an image, any image" looks like — the unconditional branch. */
const GENERIC = (() => {
  const t = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const { r } = cell(i, j);
      t[i * N + j] = 0.45 * Math.exp(-(r ** 2) / 0.6);
    }
  }
  return t;
})();

export default {
  id: '22-diffusion',
  num: '22',
  group: 'the real thing',
  title: 'The other kind of generation',
  blurb: 'Image models in this same repo share the weights-on-disk, the quantised matmul and the Metal kernels — and share almost nothing else. There is no token, no vocabulary, no growing context. The whole output exists from step one and is refined in place.',

  render(root) {
    const stops = [];

    /* ================= 1. append vs refine ================= */
    const D = {
      x: null, t: 0, steps: 24, guidance: 1, seed: 7,
      running: false, acc: 0, done: false,
    };
    const reset = () => {
      const rng = mulberry32(D.seed);
      D.x = Float64Array.from({ length: N * N }, () => (rng() * 2 - 1) * 0.8);
      D.t = 0; D.done = false;
      refresh();
    };

    /**
     * One Euler step of a flow-matching sampler.
     * The model's job is to predict a VELOCITY: which way this latent must move,
     * right now, to look more like what the prompt asked for. Here we cheat and
     * compute it from a known target; a trained model has learned it from data.
     */
    const stepOnce = () => {
      if (D.done) return;
      const dt = 1 / D.steps;
      const inv = 1 / Math.max(1e-3, 1 - D.t);
      for (let k = 0; k < D.x.length; k++) {
        const vCond = (TARGET[k] - D.x[k]) * inv;     // "make it the prompt"
        const vUncond = (GENERIC[k] - D.x[k]) * inv;  // "make it an image"
        const v = vUncond + D.guidance * (vCond - vUncond);   // classifier-free guidance
        D.x[k] += v * dt;
      }
      D.t += dt;
      if (D.t >= 1 - 1e-9) { D.t = 1; D.done = true; }
      refresh();
    };

    const imgPlot = new Plot({
      width: 300, height: 300, pad: { l: 0, r: 0, t: 0, b: 0 }, xlim: [0, N], ylim: [0, N], plain: true,
    });
    imgPlot.onDraw((p) => {
      if (!D.x) return;
      const m = [];
      for (let i = 0; i < N; i++) {
        const r = [];
        for (let j = 0; j < N; j++) r.push(Math.max(0, Math.min(1, D.x[i * N + j])));
        m.push(r);
      }
      p.matrix(m, { cmap: (v) => seq(v), gap: 0, max: 1 });
    });

    /* the autoregressive side, for contrast */
    const lm = new NGram(3).train(DEFAULT_CORPUS);
    const A = { ids: [], running: false, acc: 0 };
    const resetA = () => {
      A.ids = tokenize('The engine reads the').map((t) => lm.index.get(t)).filter((i) => i !== undefined);
      textBox.textContent = detokenize(A.ids.map((i) => lm.vocab[i]));
    };
    const textBox = el('div', {
      style: {
        font: '13px/1.7 ui-monospace, Menlo, monospace', whiteSpace: 'pre-wrap',
        background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '9px',
        padding: '12px 14px', height: '300px', width: '330px', maxWidth: '100%', overflowY: 'auto',
      },
    });
    const stepA = () => {
      const { logits } = lm.logits(lm.context(A.ids));
      let best = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      A.ids.push(best);
      if (A.ids.length > 90) A.ids = A.ids.slice(-70);
      textBox.textContent = detokenize(A.ids.map((i) => lm.vocab[i]));
      textBox.scrollTop = textBox.scrollHeight;
    };

    const dStats = stats([
      { k: 'step', v: '—' }, { k: 'time t', v: '—' },
      { k: 'pixels changed', v: '—' }, { k: 'output size', v: `${N}×${N} fixed` },
    ]);

    const refresh = () => {
      dStats.set('step', `${Math.round(D.t * D.steps)} / ${D.steps}`);
      dStats.set('time t', fmt.n(D.t, 3));
      dStats.set('pixels changed', D.done ? '0 — finished' : `all ${N * N}, every step`);
      imgPlot.render();
    };

    const runBtn = button('▶ run both', () => {
      D.running = !D.running; A.running = D.running;
      runBtn.textContent = D.running ? '⏸ pause' : '▶ run both';
    });
    stops.push(loop((dt) => {
      if (!D.running) return;
      D.acc += dt;
      if (D.acc > 90) { D.acc = 0; stepOnce(); stepA(); }
    }, textBox));

    root.append(card('Append, or refine',
      'Left: lesson 01\'s loop — one token is chosen and <b>appended</b>, and the text gets longer. Right: a diffusion sampler — every one of the 1,024 values is <b>rewritten</b> every step, and the output was always this size.',
      row(
        el('div', { class: 'col' },
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'autoregressive: the output grows' }),
          textBox),
        el('div', { class: 'col' },
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'diffusion: the output is refined in place' }),
          imgPlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 260px' } },
          dStats,
          slider({ label: 'steps', min: 2, max: 60, step: 1, value: 24, fmt: String, onInput: (v) => { D.steps = v; reset(); } }),
          slider({ label: 'seed', min: 1, max: 40, step: 1, value: 7, fmt: String, onInput: (v) => { D.seed = v; reset(); } }),
          btns(
            runBtn,
            button('one step ▸', () => { stepOnce(); stepA(); }, { ghost: true }),
            button('reset', () => { reset(); resetA(); }, { ghost: true })),
          note(`<b>Nothing here is a token.</b> The grid starts as seeded noise and is pushed toward something
                the prompt would accept. There is no vocabulary to pick from, no position to append at, and
                nothing to cache between steps — every step reads the whole canvas and rewrites it.`)))));

    /* ================= 2. the velocity ================= */
    root.append(card('What the model predicts is a direction',
      'MOSS-Image is trained by <b>flow matching</b>: given a noisy latent and a timestep, predict the velocity that moves it toward a real image. Sampling is then numerical integration — Euler steps along that field.',
      row(
        grow(
          el('div', { class: 'eq', html: 'v = model( <span class="hl">latents</span>, <span class="n">timestep</span>, prompt )' }),
          el('div', { class: 'eq', html: 'latents ← latents + v · Δt' }),
          el('pre', { class: 'code', html:
            `<span class="cm">// zig/src/families/moss_image/cuda_model.zig:217-218</span>\n` +
            `<span class="cm">/// The velocity of \`latents\` (image rows, 128) at \`timestep\` under prompt \`p\`,</span>\n` +
            `<span class="cm">/// skipping block \`skip\` if given.</span>\n` +
            `pub fn forward(t: *Model, p: *const Prompt, latents: []const f32,\n` +
            `               timestep: f32, skip: ?usize, velocity: []f32) !Timing {` }),
          prose(`
            <p>Read the signature and the whole architecture falls out: latents in, a timestep, a prompt — and
            <b>velocity</b> out. No logits, no vocabulary, no position. The same 1,024 numbers go in and come
            back nudged.</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b>"Sampler" means something else here.</b> In lesson 01 it drew one token from a probability
                distribution. In an image pipeline it is an <i>ODE solver</i> — Euler, DPM++, Heun — integrating
                this velocity field. Same word, unrelated job.`),
          note(`<b>"Steps" means something else too.</b> Not output length: the number of integration steps.
                Drag <b>steps</b> above from 60 down to 3 and watch the result degrade without getting shorter.
                More steps buy accuracy, not more picture.`),
          note(`<b>The timestep is an input, not a counter.</b> It is embedded into a vector (a sinusoid, like
                RoPE's) and used to modulate every block — one modulation shared by all 42 of them, rebuilt
                each step. The network must behave differently at t = 0.1 than at t = 0.9.`)))));

    /* ================= 3. guidance ================= */
    const gPlot = new Plot({
      width: 460, height: 250, pad: { l: 50, r: 16, t: 20, b: 40 },
      xlim: [0, 8], ylim: [0, 1], xlabel: 'guidance scale', ylabel: 'error against the prompt',
    });
    gPlot.onDraw((p) => {
      // run the toy sampler to completion for each guidance value and measure the result
      const pts = [], over = [];
      for (let g = 0; g <= 8; g += 0.25) {
        const rng = mulberry32(D.seed);
        let x = Float64Array.from({ length: N * N }, () => (rng() * 2 - 1) * 0.8);
        let t = 0;
        const dt = 1 / D.steps;
        for (let s = 0; s < D.steps; s++) {
          const inv = 1 / Math.max(1e-3, 1 - t);
          for (let k = 0; k < x.length; k++) {
            const vc = (TARGET[k] - x[k]) * inv, vu = (GENERIC[k] - x[k]) * inv;
            x[k] += (vu + g * (vc - vu)) * dt;
          }
          t += dt;
        }
        let err = 0, clipped = 0;
        for (let k = 0; k < x.length; k++) {
          err += (x[k] - TARGET[k]) ** 2;
          if (x[k] > 1.02 || x[k] < -0.02) clipped++;
        }
        pts.push([g, Math.sqrt(err / x.length)]);
        over.push([g, clipped / x.length]);
      }
      p.ylim = [0, Math.max(0.2, ...pts.map((q) => q[1])) * 1.15];
      p.axes({ xticks: 5, yticks: 5, zero: false });
      p.line(pts, { color: 'c2', width: 2.5 });
      p.line(over, { color: 'c5', width: 2, dash: [4, 3] });
      p.line([[D.guidance, 0], [D.guidance, p.ylim[1]]], { color: 'accent', width: 1.5, dash: [3, 3] });
      p.line([[3, 0], [3, p.ylim[1]]], { color: 'c3', width: 1.5 });
      p.text(3, p.ylim[1], ' the card\'s default: 3 ', { color: 'c3', size: 10, align: 'left', baseline: 'top' });
      p.text(p.pad.l + 6, p.pad.t + 8, 'red: distance from what the prompt asked for', { color: 'c2', size: 10, px: true, align: 'left' });
      p.text(p.pad.l + 6, p.pad.t + 22, 'amber: fraction of values pushed out of range', { color: 'c5', size: 10, px: true, align: 'left' });
    });

    root.append(card('Guidance: two passes, subtracted',
      'The pipeline runs the model <b>twice per step</b> — once with your prompt, once with an empty one — and pushes away from the empty answer. That is why image models feel twice as slow as their parameter count suggests.',
      row(
        el('div', {}, gPlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          el('div', { class: 'eq', html: 'v = v<sub>empty</sub> + <span class="n">g</span> · ( v<sub>prompt</sub> − v<sub>empty</sub> )' }),
          slider({
            label: 'guidance scale <span class="badge">--guidance</span>', min: 0, max: 8, step: 0.25, value: 1,
            fmt: (v) => v.toFixed(2), onInput: (v) => { D.guidance = v; reset(); gPlot.render(); },
          }),
          prose(`
            <p>At <b>g = 1</b> the empty branch cancels exactly and you get the prompt's own answer. At
            <b>g = 0</b> you get a generic blob — the prompt is ignored. Above 1 you are <i>extrapolating</i>
            away from the generic answer, which sharpens the subject and, pushed far enough, overshoots the
            valid range entirely.</p>
            <p>That overshoot is the amber curve, and it is the toy version of a very familiar artifact:
            over-guided images with blown-out colours and crunchy edges.</p>`),
          note(`<b>MOSS-Image's card default is guidance 3</b>, with the negative branch <b>skipping block 22</b>
                — a deliberate asymmetry between the two passes, documented in
                <code class="inline">docs/recipes/moss-image.md</code> and implemented as the
                <code class="inline">skip: ?usize</code> argument in the signature above.`)))));

    /* ================= 4. what transfers ================= */
    root.append(card('What this course still buys you here', null, row(
      grow(el('table', {},
        el('thead', {}, el('tr', {}, el('th', { text: 'from the course' }), el('th', { text: 'applies?' }), el('th', { text: 'why' }))),
        el('tbody', {}, ...[
          ['safetensors, mmap, memory floor (10)', 'yes', 'the same loader reads a diffusers checkpoint folder'],
          ['quantised / bf16 matmul kernels (08, 09)', 'yes', 'the denoiser is a transformer; same GEMMs'],
          ['attention, RoPE (06, 21)', 'yes', '42 joint blocks, full attention over image + prompt rows'],
          ['tokenizer, embeddings (02, 04)', 'yes', 'the prompt encoder is a text model'],
          ['the decode loop (01)', 'no', 'nothing is appended'],
          ['KV cache (05, 13)', 'no', 'no sequence grows, so nothing to reuse'],
          ['sampler: temperature, top-k, top-p (01)', 'no', '"sampler" here is an ODE solver'],
          ['speculative decoding (14)', 'no', 'no next token to guess'],
          ['context window, compaction (21)', 'no', 'the output is a fixed grid'],
        ].map(([a, b, c]) => el('tr', {},
          el('td', { style: { textAlign: 'left' }, text: a }),
          el('td', { style: { color: b === 'yes' ? 'var(--ok)' : 'var(--err)' }, text: b }),
          el('td', { style: { textAlign: 'left' }, text: c })))))),
      el('div', { class: 'col', style: { flex: '1 1 300px' } },
        note(`<b>The prompt encoder is an old friend.</b> MOSS-Image's text encoder is <i>the checkpoint's
              Qwen3.5 text model</i> — the same family lesson 16 took apart file by file. An image pipeline is
              a language model, a denoiser and a decoder wired together.`),
        note(`<b>The real geometry</b>, from the recipe: a 5B joint transformer, <b>42 blocks</b>, hidden
              <b>3,072</b>, 24 heads of 128, MLP 8,192, denoising <b>128-channel latents</b> by flow matching.
              Qwen-Image-2.1 next door is 7B, 32 blocks, hidden 4,096, 64-channel latents.`),
        note(`<b>And the sharpest line in the repo:</b> <i>"It is a denoiser, not a token family. It does not
              register with the lane engine, the server or <code class="inline">families.detect</code>."</i>
              Everything from lessons 12–15 — lanes, caches, drafts, SSE — simply does not apply to it.`)))));

    /* ================= 5. honesty ================= */
    root.append(card('Why there is no picture from the real model on this page', null, row(
      grow(prose(`
        <p>Both image families in this repo were built when we ran <code class="inline">zig build</code> —
        <code class="inline">tf-moss-image</code>, <code class="inline">tf-moss-image-generate</code>,
        <code class="inline">tf-qwen-image</code> are sitting in <code class="inline">zig-out/bin</code>. Neither
        can produce an image <i>on this machine</i>, for two separate and honest reasons:</p>
        <ul>
          <li><b>MOSS-Image refuses the GPU.</b> Both its model and its decoder check for tensor units and
          abort: <code class="inline">"has no tensor units; this family runs on the M5's alone, the only chip
          it was checked on"</code> (<code class="inline">decoder.zig:116-118</code>,
          <code class="inline">encoder.zig:55-56</code>). This is an M1 Max.</li>
          <li><b>Qwen-Image is incomplete on purpose.</b> The recipe says the prompt encoder and image decoder
          <i>"are not ported"</i> — only the denoising transformer runs here. There is nothing to turn latents
          into pixels.</li>
        </ul>
        <p>So the panels above are a toy I wrote, running the same arithmetic at 32×32 with a velocity field I
        computed rather than learned. The mechanism is honest; the model is not real.</p>`)),
      el('div', { class: 'col', style: { flex: '1 1 300px' } },
        note(`<b>This is the course's rule working as intended.</b> Lesson 18 could check its claims against a
              running engine, so it did. This one cannot, so it says so rather than showing you a picture and
              implying the repo made it.`),
        note(`<b>What it would take.</b> An M5 Mac (or an NVIDIA GPU for the CUDA path), the
              <code class="inline">OpenMOSS-Team/MOSS-Image-Base</code> checkpoint, and a CASE file generated by
              <code class="inline">tools/zig/moss_image_case.py</code> through the publisher's own torch
              pipeline — the tool compares every step against that reference before running the schedule.`)))));

    reset(); resetA();
    return () => stops.forEach((s) => s());
  },
};
