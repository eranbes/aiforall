// 11-training.js — the exception: the one part of the story that really runs backwards.

import { Plot, color, mix } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, loop, fmt } from '../ui.js';
import { mulberry32 } from '../engine/tensor.js';
import { CharLM } from '../engine/trainer.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';

/* ---------- card 2: a loss surface you can see ---------- */
const PTS = [[1, 1], [2, 2], [3, 2.4], [4, 3.3]];
const lossAt = (w, b) => {
  let s = 0;
  for (const [x, y] of PTS) { const e = w * x + b - y; s += e * e; }
  return s / PTS.length;
};
const gradAt = (w, b) => {
  let gw = 0, gb = 0;
  for (const [x, y] of PTS) { const e = w * x + b - y; gw += 2 * e * x; gb += 2 * e; }
  return [gw / PTS.length, gb / PTS.length];
};

export default {
  id: '11-training',
  num: '11',
  group: 'backwards for real',
  title: 'Where the numbers came from',
  blurb: 'Every lesson so far assumed the weights already existed. They were found by running the whole computation in reverse: measure the error, ask each weight how it contributed, nudge it. That is the only part of this course that is genuinely backwards.',

  render(root) {
    const stops = [];

    /* ================= gradient descent ================= */
    const gd = {
      w: -0.4, b: 2.2, lr: 0.05, path: [], running: false, acc: 0,
    };
    gd.path.push([gd.w, gd.b]);

    const surf = new Plot({
      width: 440, height: 340, pad: { l: 46, r: 14, t: 14, b: 38 },
      xlim: [-1.5, 2.2], ylim: [-1.5, 3], xlabel: 'w', ylabel: 'b',
    });

    surf.onDraw((p) => {
      const lo = 0, hi = 14;
      p.field((w, b) => {
        const t = Math.min(1, Math.max(0, (lossAt(w, b) - lo) / (hi - lo)));
        // banded colouring makes the bowl's contours visible
        const band = Math.floor(Math.sqrt(t) * 9) / 9;
        return mix(color('surface-2'), color('c2'), band * 0.85);
      }, { res: 3 });
      p.axes({ xticks: 5, yticks: 5, zero: true, grid: false, frame: true });

      if (gd.path.length > 1) p.line(gd.path, { color: 'accent', width: 2 });
      p.scatter(gd.path.slice(0, -1), { color: 'accent', r: 2.5, alpha: 0.7 });

      const [gw, gb] = gradAt(gd.w, gd.b);
      const n = Math.hypot(gw, gb) || 1;
      p.arrow(gd.w, gd.b, gd.w - (gw / n) * 0.45, gd.b - (gb / n) * 0.45, { color: 'c3', width: 2.5 });
      p.scatter([[gd.w, gd.b]], { color: 'c5', r: 6, stroke: 'bg' });
      p.text(p.pad.l + 6, p.pad.t + 10, 'green arrow = −gradient (downhill)', { color: 'muted', size: 10, px: true, align: 'left' });
    });

    surf.interact({
      down: (pt) => { gd.w = pt.x; gd.b = pt.y; gd.path = [[gd.w, gd.b]]; refreshGD(); },
    });

    const fit = new Plot({
      width: 330, height: 240, pad: { l: 40, r: 14, t: 14, b: 34 },
      xlim: [0, 5], ylim: [0, 4], xlabel: 'x', ylabel: 'y',
    });
    fit.onDraw((p) => {
      p.axes({ xticks: 5, yticks: 4, zero: false });
      p.fn((x) => gd.w * x + gd.b, { color: 'accent', width: 2.5 });
      for (const [x, y] of PTS) {
        const pred = gd.w * x + gd.b;
        p.line([[x, y], [x, pred]], { color: 'c2', width: 1.5, dash: [3, 3] });
      }
      p.scatter(PTS, { color: 'c5', r: 5, stroke: 'bg' });
    });

    const gdStats = stats([{ k: 'loss', v: '—' }, { k: 'w', v: '—' }, { k: 'b', v: '—' }, { k: 'steps', v: '0' }]);

    const refreshGD = () => {
      gdStats.set('loss', fmt.n(lossAt(gd.w, gd.b), 4));
      gdStats.set('w', fmt.sign(gd.w, 3));
      gdStats.set('b', fmt.sign(gd.b, 3));
      gdStats.set('steps', String(gd.path.length - 1));
      surf.render(); fit.render();
    };

    const gdStep = () => {
      const [gw, gb] = gradAt(gd.w, gd.b);
      gd.w -= gd.lr * gw;
      gd.b -= gd.lr * gb;
      if (!isFinite(gd.w) || Math.abs(gd.w) > 1e6) { gd.running = false; }
      gd.path.push([gd.w, gd.b]);
      if (gd.path.length > 400) gd.path = gd.path.slice(-300);
      refreshGD();
    };

    const lrS = slider({
      label: 'learning rate', min: 0.005, max: 0.35, step: 0.005, value: 0.05,
      fmt: (v) => v.toFixed(3), onInput: (v) => { gd.lr = v; },
    });
    const runGD = button('▶ descend', () => {
      gd.running = !gd.running;
      runGD.textContent = gd.running ? '⏸ pause' : '▶ descend';
    });
    stops.push(loop((dt) => {
      if (!gd.running) return;
      gd.acc += dt;
      if (gd.acc > 60) { gd.acc = 0; gdStep(); }
    }, surf.canvas));

    root.append(card('Gradient descent, on a surface you can see',
      'Fitting a line to four points has two parameters, so its loss is a landscape you can draw. <b>Click anywhere</b> to drop the parameters there and press descend.',
      row(
        el('div', {}, surf.canvas),
        el('div', { class: 'col', style: { flex: '0 1 330px' } },
          fit.canvas,
          gdStats,
          lrS,
          btns(runGD, button('step', gdStep, { ghost: true }),
            button('reset', () => { gd.w = -0.4; gd.b = 2.2; gd.path = [[gd.w, gd.b]]; refreshGD(); }, { ghost: true })))),
      el('div', { style: { height: '12px' } }),
      row(
        grow(prose(`
          <p>The gradient is the direction of steepest <i>increase</i>, so every step moves against it.
          Nothing searches, nothing guesses: the slope is computed exactly, and the parameters slide down it.</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b>Push the learning rate past ~0.25.</b> The steps overshoot the valley, land further up the
                opposite wall, and the loss explodes to <code class="inline">Infinity</code> within a few
                iterations. That is a diverged training run, and it is this undramatic.`),
          note(`<b>Too small is its own failure.</b> At 0.005 the path crawls; with a billion parameters and
                a corpus of trillions of tokens, that is the difference between weeks and never.`)))));

    /* ================= the chain rule ================= */
    const cr = { x: 1.3, w: 0.9, b: -0.4, y: 1.0, phase: 0 };
    const crPlot = new Plot({ width: 660, height: 260, pad: { l: 0, r: 0, t: 0, b: 0 }, plain: true });

    crPlot.onDraw((p) => {
      const z = cr.w * cr.x + cr.b;
      const a = Math.tanh(z);
      const L = (a - cr.y) * (a - cr.y);
      const dL_da = 2 * (a - cr.y);
      const da_dz = 1 - a * a;
      const dL_dz = dL_da * da_dz;
      const dL_dw = dL_dz * cr.x;
      const dL_db = dL_dz;

      const c = p.ctx;
      const nodes = [
        { x: 60, y: 60, t: 'w', v: cr.w, g: dL_dw, leaf: true },
        { x: 60, y: 130, t: 'x', v: cr.x, g: null, leaf: true },
        { x: 60, y: 200, t: 'b', v: cr.b, g: dL_db, leaf: true },
        { x: 230, y: 110, t: 'z = w·x + b', v: z, g: dL_dz },
        { x: 400, y: 110, t: 'a = tanh(z)', v: a, g: dL_da },
        { x: 560, y: 110, t: 'L = (a − y)²', v: L, g: 1 },
      ];
      const edges = [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]];

      for (const [i, j] of edges) {
        const A = nodes[i], B = nodes[j];
        const fwd = cr.phase === 0;
        c.strokeStyle = color(fwd ? 'c1' : 'c6');
        c.lineWidth = 2;
        c.beginPath();
        c.moveTo(A.x + 54, A.y);
        c.lineTo(B.x - 54, B.y);
        c.stroke();
        const mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2 - 10;
        c.fillStyle = color(fwd ? 'c1' : 'c6');
        c.font = '10px ui-monospace, Menlo, monospace';
        c.textAlign = 'center';
        if (!fwd) {
          const localD = j === 3 ? (i === 0 ? cr.x : i === 2 ? 1 : cr.w)
            : j === 4 ? da_dz : dL_da;
          c.fillText('×' + fmt.n(localD, 2), mx, my);
        }
      }

      for (const n of nodes) {
        const isLeaf = n.leaf;
        c.fillStyle = color('surface');
        c.strokeStyle = color(cr.phase === 1 && n.g !== null ? 'c6' : 'border');
        c.lineWidth = cr.phase === 1 && n.g !== null ? 2 : 1;
        c.beginPath();
        c.roundRect(n.x - 54, n.y - 26, 108, 52, 9);
        c.fill(); c.stroke();
        c.fillStyle = color('fg');
        c.font = '11px ui-monospace, Menlo, monospace';
        c.textAlign = 'center'; c.textBaseline = 'middle';
        c.fillText(n.t, n.x, n.y - 11);
        c.fillStyle = color('c1');
        c.font = 'bold 13px ui-monospace, Menlo, monospace';
        c.fillText(fmt.n(n.v, 3), n.x, n.y + 6);
        if (cr.phase === 1 && n.g !== null) {
          c.fillStyle = color('c6');
          c.font = '10px ui-monospace, Menlo, monospace';
          c.fillText('∂L/∂ = ' + fmt.sign(n.g, 3), n.x, n.y + 20);
        }
        if (isLeaf) {
          c.fillStyle = color('muted');
          c.font = '9px ui-monospace, Menlo, monospace';
          c.fillText(n.g === null ? 'input' : 'parameter', n.x, n.y - 34);
        }
      }
      c.fillStyle = color('muted');
      c.font = '10px ui-monospace, Menlo, monospace';
      c.textAlign = 'left';
      c.fillText(cr.phase === 0
        ? 'forward: every node computes its value from the left'
        : 'backward: every node receives ∂L/∂itself, multiplies by the local derivative, passes it left',
        14, 240);
    });

    const crStats = stats([{ k: '∂L/∂w', v: '—' }, { k: '∂L/∂b', v: '—' }, { k: 'loss', v: '—' }]);
    const refreshCR = () => {
      const z = cr.w * cr.x + cr.b, a = Math.tanh(z);
      const dL_dz = 2 * (a - cr.y) * (1 - a * a);
      crStats.set('∂L/∂w', fmt.sign(dL_dz * cr.x, 4));
      crStats.set('∂L/∂b', fmt.sign(dL_dz, 4));
      crStats.set('loss', fmt.n((a - cr.y) ** 2, 4));
      crPlot.render();
    };

    root.append(card('The chain rule, drawn',
      'One neuron, one target. Press <b>backward</b> and watch the gradient arrive at the loss as 1, then travel left, multiplying by one local derivative per edge.',
      crPlot.canvas,
      el('div', { style: { height: '10px' } }),
      row(
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          btns(
            button('forward', () => { cr.phase = 0; refreshCR(); }),
            button('backward ←', () => { cr.phase = 1; refreshCR(); }, { ghost: true })),
          slider({ label: 'w', min: -2, max: 2, step: 0.05, value: 0.9, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { cr.w = v; refreshCR(); } }),
          slider({ label: 'b', min: -2, max: 2, step: 0.05, value: -0.4, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { cr.b = v; refreshCR(); } }),
          slider({ label: 'target y', min: -1, max: 1, step: 0.05, value: 1, fmt: (v) => fmt.sign(v, 2), onInput: (v) => { cr.y = v; refreshCR(); } }),
          crStats),
        grow(
          el('div', { class: 'eq', html: '∂L/∂w = <span class="hl">∂L/∂a</span> · <span class="n">∂a/∂z</span> · <span class="hl">∂z/∂w</span>' }),
          prose(`
            <p>That is the entire algorithm. Backpropagation is not a separate idea from the chain rule —
            it is the chain rule applied to a graph, computing each local derivative once and reusing it
            for everything downstream.</p>
            <p>Each node needs only two things: its own local derivative, and the gradient handed to it from
            the right. It never needs to know what the rest of the network looks like. That locality is why
            the same code trains a two-node graph and a 400-billion-parameter one.</p>
            <p>Notice <code class="inline">∂a/∂z = 1 − a²</code>. Push <b>w</b> to either extreme: tanh
            saturates, that derivative collapses toward zero, and <i>every</i> gradient to its left is
            multiplied by almost nothing. That is the vanishing gradient, visible in one number.</p>`)))));

    /* ================= gradient check ================= */
    const lm = new CharLM({ text: DEFAULT_CORPUS, ctx: 3, emb: 12, hid: 64, seed: 3 });
    const checkRng = mulberry32(11);
    const checkBatch = lm.batch(1, checkRng);
    const checkBox = el('div', { class: 'col' });

    const runCheck = () => {
      const [ids, target] = checkBatch[0];
      const rows = [];
      for (const [j, k] of [[0, 1], [7, 5], [31, 12], [63, 20]]) {
        const an = lm.gradW2(ids, target, j, k);
        const nu = lm.numericGradW2(ids, target, j, k);
        const rel = Math.abs(an - nu) / Math.max(1e-9, Math.abs(an) + Math.abs(nu));
        rows.push({ j, k, an, nu, rel });
      }
      const worst = Math.max(...rows.map((r) => r.rel));
      checkBox.replaceChildren(
        el('div', { class: 'eq', style: { whiteSpace: 'pre' }, html:
          ['  W2[j][k]        analytic        numeric       rel.err',
            ...rows.map((r) => `  [${String(r.j).padStart(2)}][${String(r.k).padStart(2)}]   ${fmt.sign(r.an, 8).padStart(13)}  ${fmt.sign(r.nu, 8).padStart(13)}   ${r.rel.toExponential(1).padStart(8)}`),
          ].join('\n') }),
        el('div', { style: { height: '8px' } }),
        el('span', { class: worst < 1e-5 ? 'badge ok' : 'badge err' },
          worst < 1e-5 ? `✓ hand-written gradients agree with finite differences (worst ${worst.toExponential(1)})`
            : `✗ mismatch: ${worst.toExponential(1)}`));
    };

    root.append(card('Proving the gradients are right',
      'A hand-written gradient is an assertion. Here is the test that settles it: nudge one weight by ±ε, re-run the model, and compare the measured change in loss against what the chain rule predicted.',
      row(
        grow(checkBox,
          el('div', { style: { height: '10px' } }),
          btns(button('re-run the check', runCheck, { ghost: true }))),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          el('div', { class: 'eq', html: '∂L/∂θ ≈ ( L(θ+ε) − L(θ−ε) ) / 2ε' }),
          note(`<b>This is how you debug a model.</b> If the analytic gradient and the numeric estimate
                disagree, training will quietly fail to converge and nothing will tell you why. The check
                costs two forward passes per weight — unusable for training, perfect for verification.`),
          note(`<b>Why not just use finite differences everywhere?</b> Each weight would need its own pair
                of forward passes. Backpropagation computes <i>every</i> gradient in one backward pass that
                costs about as much as one forward pass. That ratio is the reason deep learning is possible
                at all.`)))));

    /* ================= train it ================= */
    const tr = { running: false, lr: 0.5, batch: 32, losses: [], rng: mulberry32(5), acc: 0 };
    const lossPlot = new Plot({
      width: 440, height: 240, pad: { l: 52, r: 14, t: 14, b: 38 },
      xlim: [0, 200], ylim: [0, 4], xlabel: 'training step', ylabel: 'loss (nats / char)',
    });
    lossPlot.onDraw((p) => {
      p.xlim = [0, Math.max(200, tr.losses.length)];
      const hi = Math.max(1, ...tr.losses.slice(0, 5), 1.5);
      p.ylim = [0, hi * 1.1];
      p.axes({ xticks: 5, yticks: 5, zero: false });
      const uniform = Math.log(lm.V);
      p.line([[0, uniform], [p.xlim[1], uniform]], { color: 'muted', width: 1.5, dash: [5, 4] });
      p.text(p.xlim[1] * 0.98, uniform, `random guessing = ln ${lm.V} = ${fmt.n(uniform, 2)}`,
        { color: 'muted', size: 10, align: 'right', baseline: 'bottom' });
      p.line(tr.losses.map((l, i) => [i, l]), { color: 'accent', width: 2 });
    });

    const sampleBox = el('div', {
      style: {
        font: '13px/1.65 ui-monospace, Menlo, monospace', whiteSpace: 'pre-wrap',
        background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '9px',
        padding: '12px 14px', minHeight: '120px',
      },
    });
    const trStats = stats([
      { k: 'steps', v: '0' }, { k: 'loss', v: '—' },
      { k: 'perplexity', v: '—' }, { k: 'parameters', v: '—' },
    ]);

    const showSample = () => {
      const rng = mulberry32(7 + lm.steps);
      sampleBox.textContent = lm.generate('The ', 220, { temperature: 0.7, rng });
    };

    const refreshTrain = () => {
      const l = tr.losses.length ? tr.losses[tr.losses.length - 1] : null;
      trStats.set('steps', String(lm.steps));
      trStats.set('loss', l == null ? '—' : fmt.n(l, 3));
      trStats.set('perplexity', l == null ? '—' : fmt.n(Math.exp(l), 1));
      trStats.set('parameters', String(lm.E.length + lm.W1.length + lm.b1.length + lm.W2.length + lm.b2.length));
      lossPlot.render();
    };

    const trainSteps = (n) => {
      for (let i = 0; i < n; i++) tr.losses.push(lm.step(lm.batch(tr.batch, tr.rng), tr.lr));
      refreshTrain();
    };

    const runTrain = button('▶ train', () => {
      tr.running = !tr.running;
      runTrain.textContent = tr.running ? '⏸ pause' : '▶ train';
    });
    stops.push(loop(() => {
      if (!tr.running) return;
      trainSteps(4);
      tr.acc++;
      if (tr.acc % 12 === 0) showSample();
    }, lossPlot.canvas));

    root.append(card('Now train one, here, from random noise',
      'A character-level neural language model: embeddings, one hidden layer, a softmax over characters. The gradients are the hand-written ones you just verified. It starts knowing nothing.',
      row(
        el('div', {}, lossPlot.canvas,
          el('div', { style: { height: '8px' } }),
          legend([{ c: 'accent', label: 'training loss' }, { c: 'muted', label: 'random guessing' }])),
        el('div', { class: 'col', style: { flex: '1 1 330px' } },
          trStats,
          slider({ label: 'learning rate', min: 0.05, max: 2, step: 0.05, value: 0.5, fmt: (v) => v.toFixed(2), onInput: (v) => { tr.lr = v; } }),
          slider({ label: 'batch size', min: 4, max: 128, step: 4, value: 32, fmt: String, onInput: (v) => { tr.batch = v; } }),
          btns(runTrain,
            button('+100 steps', () => { trainSteps(100); showSample(); }, { ghost: true }),
            button('sample', showSample, { ghost: true })))),
      el('div', { style: { height: '12px' } }),
      el('div', { class: 'sub', style: { margin: '0 0 6px' }, text: 'sampled from the model as it stands right now' }),
      sampleBox,
      el('div', { style: { height: '12px' } }),
      row(
        grow(prose(`
          <p>Watch the loss fall below the dashed line. That line is <code class="inline">ln V</code> — the
          loss of a model that knows nothing and spreads its probability evenly over every character.
          Beating it is the first evidence that anything was learned at all.</p>
          <p>The samples go through phases: noise, then letter frequencies, then plausible letter pairs,
          then fragments of real words, then short real words. Nobody programmed those stages. They are the
          order in which it is cheapest to reduce the loss.</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Loss is in nats per character.</b> Exponentiate it and you get perplexity: the effective
                number of characters the model is choosing between. It starts at <i>V</i> and drops.`),
          note(`<b>This is the same loop as GPT-scale training.</b> Sample a batch, forward, measure loss,
                backward, nudge every weight, repeat. What changes at scale is the size of the model, the
                size of the corpus, and the number of machines — not the algorithm on this page.`)))));

    /* ================= closing ================= */
    root.append(card('And that is the whole stack', null, row(
      grow(prose(`
        <p>You have now built every level of the descent:</p>
        <ul>
          <li>bytes in a file → dequantised weights → matrix multiplies →</li>
          <li>blocks that add to a residual stream → attention that moves information between tokens →</li>
          <li>a final projection → logits → softmax → a sampler → a token → text.</li>
        </ul>
        <p>And now the last loose end is tied: the weights were found by gradient descent, which is this
        same computation run backwards, a very large number of times.</p>
        <p><a href="#/01-sampling">Go back to lesson 01</a>. Nothing on that page should be mysterious any
        more — the logits it samples from are the output of everything you have since taken apart.</p>`)),
      el('div', { class: 'col', style: { flex: '1 1 320px' } },
        note(`<b>TensorFold does not train.</b> It is an inference engine: it reads finished weights and
              runs them forward as fast as the hardware allows. Training and serving are different
              problems, solved by different software — which is why this lesson is the only one whose
              anchor is not a kernel.`),
        note(`<b>Except, a little.</b> The repo ships an experimental mode called Living Weights
              (<code class="inline">--slide</code>) that teaches a running model new facts by writing into
              its own checkpoint files, and its README is candid about the limits: related facts disturb
              each other, recall drops after a few dozen facts, and learned facts cost decode speed.`)))));

    refreshGD(); refreshCR(); runCheck(); refreshTrain(); showSample();
    return () => stops.forEach((s) => s());
  },
};
