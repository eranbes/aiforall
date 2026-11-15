// 01-sampling.js — levels 11, 10 and 8: the decode loop and the sampler.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, checkbox, stats, legend, loop, textbox, fmt, jumpToCard } from '../ui.js';
import { NGram, tokenize, detokenize } from '../engine/ngram.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';
import { sample, entropy, perplexity, keyedUniform } from '../engine/sample.js';
import { softmax } from '../engine/tensor.js';

const SHOW = 16;              // how many tokens of the vocabulary to draw
const DRAW = 8;               // how many candidates to show in the Gumbel race
const label = (t) => (t === '\n' ? '⏎' : t);
const SEED_TEXT = 'The engine reads the';

export default {
  id: '01-sampling',
  num: '01',
  group: 'the surface',
  title: 'How a token is chosen',
  blurb: 'A reply is not written, it is drawn — one token at a time, from a probability distribution. Here is the exact procedure, in the exact order a production engine applies it.',

  render(root) {
    const stops = [];

    const state = {
      model: new NGram(3).train(DEFAULT_CORPUS),
      ids: [],
      params: { temperature: 0.9, topK: 0, topP: 1, minP: 0 },
      greedy: false,
      seed: 7,
      running: false,
      acc: 0,
    };

    /** How many of state.ids came from the prompt — i.e. what prefill processed. */
    let promptLen = 0;
    const seedTokens = (text) => {
      state.ids = tokenize(text).map((t) => state.model.index.get(t)).filter((i) => i !== undefined);
      if (!state.ids.length) state.ids = [0];
      promptLen = state.ids.length;
    };
    seedTokens(SEED_TEXT);

    /** Everything the page shows is derived from this one call. */
    const current = () => {
      const ctx = state.model.context(state.ids);
      const { logits, used } = state.model.logits(ctx);
      const res = sample(logits, {
        ...state.params,
        temperature: state.greedy ? 0 : state.params.temperature,
        seed: state.seed,
        position: state.ids.length,
      });
      return { logits, used, res };
    };

    /* ---------------- generated text ---------------- */
    const outEl = el('div', {
      style: {
        font: '14px/1.7 ui-monospace, Menlo, monospace', whiteSpace: 'pre-wrap',
        background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '9px',
        padding: '12px 14px', minHeight: '116px', maxHeight: '240px', overflowY: 'auto',
      },
    });

    const renderText = () => {
      const toks = state.ids.map((i) => state.model.vocab[i]);
      // three regions: the prompt (one prefill pass), earlier replies, the newest token
      const promptText = detokenize(toks.slice(0, promptLen));
      const whole = detokenize(toks);
      const head = detokenize(toks.slice(0, -1));
      const generated = whole.slice(promptText.length);
      const newest = whole.slice(head.length);
      const older = generated.slice(0, generated.length - newest.length);
      outEl.replaceChildren(
        el('span', {
          style: {
            color: 'var(--c5)', fontWeight: '600',
            background: 'color-mix(in srgb, var(--c5) 16%, transparent)',
            borderRadius: '4px',
          },
          text: promptText,
        }),
        el('span', { style: { color: 'var(--muted)' }, text: older }),
        el('span', { style: { color: 'var(--accent)', fontWeight: '600' }, text: newest }),
        el('span', { style: { color: 'var(--accent)' }, text: '▌' }));
      outEl.scrollTop = outEl.scrollHeight;
    };

    /* ---------------- the distribution ---------------- */
    const dist = new Plot({
      width: 660, height: 318, pad: { l: 46, r: 12, t: 22, b: 66 },
      xlim: [0, SHOW], ylim: [0, 1], ylabel: 'probability',
    });

    let view = null;

    dist.onDraw((p) => {
      const { logits, res } = current();
      const raw = softmax(logits, 1);
      const order = res.order.slice(0, SHOW);
      const tempered = softmax(logits, state.greedy ? 1 : state.params.temperature);

      const rawV = order.map((i) => raw[i]);
      const tmpV = order.map((i) => (res.greedy ? (i === res.chosen ? 1 : 0) : tempered[i]));
      p.ylim = [0, Math.max(0.08, Math.max(...rawV, ...tmpV) * 1.18)];

      const cols = order.map((i) => {
        if (res.greedy) return i === res.chosen ? 'c3' : 'muted';
        if (!res.steps.afterK.has(i)) return 'c2';
        if (!res.steps.afterP.has(i)) return 'c5';
        if (!res.steps.afterMinP.has(i)) return 'c6';
        return 'c3';
      });

      p.axes({ xticks: 0, yticks: 4, zero: false });
      p.bars(rawV, { color: 'border', width: 0.78 });
      p.bars(tmpV, { color: cols, width: 0.46, labels: order.map((i) => label(state.model.vocab[i])) });

      const slot = p.iw / SHOW;
      const ci = order.indexOf(res.chosen);
      if (ci >= 0) {
        const x = p.pad.l + ci * slot + slot / 2;
        const y = p.Y(Math.max(rawV[ci], tmpV[ci]));
        p.text(x, y - 16, '▼', { color: 'accent', size: 13, px: true });
        p.text(x, y - 28, label(state.model.vocab[res.chosen]), { color: 'accent', size: 11, px: true, weight: 'bold' });
      }
      p.text(p.pad.l, 12, 'wide grey = before temperature · coloured = what survives your filters', { color: 'muted', size: 10, px: true, align: 'left' });

      view = { order, slot };
    });

    dist.interact({
      down: (_, ev) => {
        if (!view) return;
        const id = view.order[Math.floor((dist.atPx(ev).x - dist.pad.l) / view.slot)];
        if (id !== undefined) step(id);
      },
      hover: (_, ev) => { dist.canvas.style.cursor = ev && dist.atPx(ev).x > dist.pad.l ? 'pointer' : 'default'; },
    });

    /* ---------------- the Gumbel race ---------------- */
    const race = new Plot({
      width: 420, height: 268, pad: { l: 46, r: 14, t: 18, b: 54 },
      xlim: [0, DRAW], ylim: [-4, 4], ylabel: 'score',
    });

    race.onDraw((p) => {
      const { res } = current();
      if (res.greedy) {
        p.axes({ xticks: 0, yticks: 4 });
        p.text(p.w / 2, p.h / 2, 'greedy: no noise, no race', { color: 'muted', size: 12, px: true });
        return;
      }
      const n = Math.min(DRAW, res.keep);
      const scaled = [], scores = [];
      for (let j = 0; j < n; j++) {
        const id = res.order[j];
        scaled.push(res.scaled[j]);
        scores.push(res.scaled[j] + res.gumbel.get(id));
      }
      const lo = Math.min(...scaled, ...scores), hi = Math.max(...scaled, ...scores);
      const padY = Math.max(0.5, (hi - lo) * 0.18);
      p.ylim = [lo - padY, hi + padY];
      p.axes({ xticks: 0, yticks: 5, zero: false });

      const slot = p.iw / DRAW;
      const winner = res.order.indexOf(res.chosen);
      for (let j = 0; j < n; j++) {
        const x = (j + 0.5) * (DRAW / DRAW) * (p.iw / DRAW);
        const wx = p.pad.l + x;
        const won = j === winner;
        // noise segment: from the scaled logit to the noisy score
        p.ctx.save();
        p.ctx.strokeStyle = color(won ? 'accent' : 'muted');
        p.ctx.lineWidth = won ? 3 : 1.5;
        p.ctx.globalAlpha = won ? 1 : 0.6;
        p.ctx.beginPath();
        p.ctx.moveTo(wx, p.Y(scaled[j]));
        p.ctx.lineTo(wx, p.Y(scores[j]));
        p.ctx.stroke();
        p.ctx.restore();
        p.circle(wx, p.Y(scaled[j]), won ? 4 : 3, { color: 'muted', px: true });
        p.circle(wx, p.Y(scores[j]), won ? 6 : 4, { color: won ? 'accent' : 'c4', px: true });
        p.text(wx, p.pad.t + p.ih + 8, label(state.model.vocab[res.order[j]]).slice(0, 7),
          { color: won ? 'accent' : 'muted', size: 10, px: true, baseline: 'top' });
      }
      p.text(p.pad.l, 8, 'grey = logit/T · coloured = + Gumbel noise · highest wins',
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    /* ---------------- readouts ---------------- */
    const read = stats([
      { k: 'entropy', v: '—' }, { k: 'perplexity', v: '—' },
      { k: 'tokens kept', v: '—' }, { k: 'p(chosen)', v: '—' },
      { k: 'position', v: '—' }, { k: 'context used', v: '—' },
    ]);

    const refresh = () => {
      const { used, res } = current();
      read.set('entropy', fmt.n(entropy(res.probs), 2) + ' bits');
      read.set('perplexity', fmt.n(perplexity(res.probs), 1));
      read.set('tokens kept', `${res.kept.length} / ${state.model.vocabSize}`);
      read.set('p(chosen)', fmt.pct(res.probs[res.chosen] || 0, 1));
      read.set('position', String(state.ids.length));
      read.set('context used', used === 0 ? 'none' : `${used} token${used > 1 ? 's' : ''}`);
      dist.render();
      race.render();
      tempPlot.render();
      renderText();
    };

    /** One iteration of the decode loop. */
    function step(forcedId) {
      const { res } = current();
      state.ids.push(forcedId != null ? forcedId : res.chosen);
      if (state.ids.length > 220) state.ids = state.ids.slice(-180);
      refresh();
    }

    /* ---------------- controls ---------------- */
    const tempS = slider({
      label: 'temperature <span class="badge">--temperature</span>', min: 0, max: 2, step: 0.01, value: 0.9,
      fmt: (v) => (v === 0 ? '0 (greedy)' : v.toFixed(2)),
      onInput: (v) => { state.params.temperature = v; refresh(); },
    });
    const topKS = slider({
      label: 'top-k <span class="badge">--top-k</span>', min: 0, max: 40, step: 1, value: 0,
      fmt: (v) => (v === 0 ? 'off' : String(v)),
      onInput: (v) => { state.params.topK = v; refresh(); },
    });
    const topPS = slider({
      label: 'top-p <span class="badge">--top-p</span>', min: 0.05, max: 1, step: 0.01, value: 1,
      fmt: (v) => (v >= 1 ? 'off (1.00)' : v.toFixed(2)),
      onInput: (v) => { state.params.topP = v; refresh(); },
    });
    const minPS = slider({
      label: 'min-p <span class="badge">--min-p</span>', min: 0, max: 0.5, step: 0.005, value: 0,
      fmt: (v) => (v === 0 ? 'off' : v.toFixed(3)),
      onInput: (v) => { state.params.minP = v; refresh(); },
    });
    const greedyC = checkbox({
      label: 'greedy (argmax — deterministic, no noise)',
      onChange: (v) => { state.greedy = v; refresh(); },
    });
    const seedS = slider({
      label: 'seed <span class="badge">--prompt-seed</span>', min: 1, max: 99, step: 1, value: 7, fmt: (v) => String(v),
      onInput: (v) => { state.seed = v; refresh(); },
    });

    const runBtn = button('▶ run', () => {
      state.running = !state.running;
      runBtn.textContent = state.running ? '⏸ pause' : '▶ run';
    });

    stops.push(loop((dt) => {
      if (!state.running) return;
      state.acc += dt;
      if (state.acc > 110) { state.acc = 0; step(); }
    }, outEl));

    /* ---------------- what to actually do ---------------- */
    const guide = card('Do this, in order',
      'Seven cards follow. Here is what each one is for and what to press. Nothing below needs reading first.',
      el('div', { class: 'col', style: { gap: '8px' } },
        ...[
          ['Prefill, then the decode loop', 'Press <b>step ▸</b> five times.', 'The amber text was already there — that is the prompt, prefilled in one pass. Each press adds one blue word and costs one more pass.'],
          ['Before the loop: your prompt', 'Just look at the two rows of boxes.', 'Your prompt costs one pass; each reply word costs a whole pass of its own.'],
          ['The sampler', 'Drag <b>temperature</b> to 0, press step a few times. Then to 1.8, step again.', 'At 0 it repeats itself; at 1.8 it rambles. The coloured bars are the words it was choosing between.'],
          ['The draw is not a die roll', 'Press <b>step</b> and watch the tall blue line.', 'The winner is not always the most likely word — noise is added to each candidate first.'],
          ['What temperature actually does', 'Read the three short paragraphs.', 'Now the maths behind what you just felt. Skip if you are satisfied.'],
          ['In the real engine', 'Skim it.', 'The same knobs on a production server, with the source they come from.'],
          ['The model behind the numbers', 'Edit the text box, press <b>retrain</b>.', 'The predictions come from that text and nothing else. Change it and the model changes.'],
        ].map(([where, action, why], i) => el('div', {
          class: 'jump',
          role: 'button',
          tabindex: '0',
          title: `go to “${where}”`,
          onclick: () => jumpToCard(where),
          onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); jumpToCard(where); } },
        },
          el('span', { class: 'n', text: String(i + 1) }),
          el('div', {},
            el('div', { style: { fontSize: '13.5px' }, html: `<b>${where}</b> — ${action}` }),
            el('div', { style: { fontSize: '12.5px', color: 'var(--muted)', marginTop: '1px' }, text: why })),
          el('span', { class: 'go', text: '↓' })))));
    guide.classList.add('hero');
    root.append(guide);

    root.append(card('Prefill, then the decode loop',
      'The <b style="color:var(--c5)">amber text</b> is your prompt. It was processed in <b>one pass</b> before anything was generated — that is <b>prefill</b>, and it already happened. Press <b>step ▸</b> to run the <b>decode</b> loop: each press costs one more full pass and produces exactly one word.',
      row(
        grow(outEl,
          el('div', { style: { height: '8px' } }),
          legend([
            { c: 'c5', label: 'prompt — prefilled in 1 pass' },
            { c: 'accent', label: 'generated — 1 pass per word' },
          ]),
          el('div', { style: { height: '10px' } }),
          btns(
            button('step ▸', () => step()),
            runBtn,
            button('reset', () => { seedTokens(SEED_TEXT); refresh(); }, { ghost: true }))),
        el('div', { class: 'col', style: { flex: '0 1 260px' } },
          note(`The loop is <b>logits → sampler → token → append → repeat</b>. The model is asked again
                from scratch for every token; the only thing that changed is that the context got one
                token longer.`),
          read))));

    /* ---------------- before the loop ---------------- */
    const phase = new Plot({
      width: 660, height: 172, pad: { l: 0, r: 0, t: 0, b: 0 }, plain: true,
    });
    phase.onDraw((p) => {
      const c = p.ctx;
      const promptN = 8, replyN = 6;
      const boxW = 26, boxH = 26, gap = 5;
      const y0 = 32, y1 = 116;

      c.font = '10px ui-monospace, Menlo, monospace';
      c.textAlign = 'left'; c.textBaseline = 'middle';

      // prefill: every prompt token in ONE pass
      c.fillStyle = color('muted');
      c.fillText('prefill — your prompt, all at once, one pass', 8, 14);
      for (let i = 0; i < promptN; i++) {
        const x = 8 + i * (boxW + gap);
        c.fillStyle = color('c5');
        c.globalAlpha = 0.85;
        c.beginPath(); c.roundRect(x, y0, boxW, boxH, 5); c.fill();
        c.globalAlpha = 1;
      }
      const px = 8 + promptN * (boxW + gap);
      c.strokeStyle = color('c5');
      c.lineWidth = 1.5;
      c.setLineDash([3, 3]);
      c.beginPath();
      c.moveTo(8, y0 + boxH + 8); c.lineTo(px - gap, y0 + boxH + 8);
      c.stroke();
      c.setLineDash([]);
      c.fillStyle = color('c5');
      c.fillText('1 pass over the weights', 8, y0 + boxH + 20);

      // decode: one pass per token
      c.fillStyle = color('muted');
      c.fillText('decode — the reply, one token per pass', 8, y1 - 12);
      for (let i = 0; i < replyN; i++) {
        const x = 8 + i * (boxW + gap);
        c.fillStyle = color('accent');
        c.globalAlpha = 0.9;
        c.beginPath(); c.roundRect(x, y1, boxW, boxH, 5); c.fill();
        c.globalAlpha = 1;
        c.strokeStyle = color('accent');
        c.lineWidth = 1.2;
        c.setLineDash([3, 3]);
        c.beginPath(); c.moveTo(x, y1 + boxH + 6); c.lineTo(x + boxW, y1 + boxH + 6); c.stroke();
        c.setLineDash([]);
      }
      c.fillStyle = color('accent');
      c.fillText(`${replyN} passes over the weights`, 8 + replyN * (boxW + gap) + 6, y1 + boxH / 2);
    });

    root.append(card('Before the loop: your prompt',
      'The loop above starts at token one of the <i>reply</i>. Something happens first, and it has completely different economics.',
      row(
        grow(phase.canvas),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          prose(`
            <p><b>Prefill</b> runs your whole prompt through the model in a <i>single</i> pass. A thousand
            prompt tokens cost roughly one pass, because they can all be processed side by side — the model
            already knows every one of them.</p>
            <p><b>Decode</b> cannot do that. Token 2 of the reply depends on token 1, which did not exist yet.
            So the reply costs <b>one full pass over every weight, per token</b>.</p>`),
          note(`<b>This is why the first word is slow and the rest are steady</b>, why prompt tokens are priced
                lower than reply tokens by most APIs, and why a cache that skips prefill (lesson 13) is worth so
                much. Lesson 08 measures the two costs; lesson 15 draws the timeline; lesson 18 times it on a
                real engine.`),
          note(`<b>The toy above has no visible prefill</b> — a trigram just looks two words up in a table. The
                phases only separate once the model is big enough for a pass over the weights to cost something.`)))));

    root.append(card('The sampler',
      'Grey bars are the model\'s opinion. Coloured bars are what survives your filters, renormalised. <b>Click any bar</b> to override the sampler and force that token.',
      row(
        grow(dist.canvas,
          el('div', { style: { height: '8px' } }),
          legend([
            { c: 'c3', label: 'survives — can be drawn' },
            { c: 'c2', label: 'cut by top-k' },
            { c: 'c5', label: 'cut by top-p' },
            { c: 'c6', label: 'cut by min-p' },
            { c: 'border', label: 'probability before temperature' },
          ])),
        el('div', { class: 'col', style: { flex: '0 1 290px' } },
          tempS, topKS, topPS, minPS, greedyC, seedS,
          note(`The seed holds the noise still. Turn a knob with the seed fixed and any token that
                changes, changed <b>because of the filter</b> — not because of luck.`)))));

    /* ---------------- the draw itself ---------------- */
    root.append(card('The draw is not a die roll',
      'Every surviving candidate gets its own noise, added to its scaled logit. The highest total wins. This is Gumbel-max sampling, and it is exactly equivalent to drawing from the distribution.',
      row(
        el('div', {}, race.canvas),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          el('div', { class: 'eq', html: 'chosen = argmax<sub>j</sub> ( <span class="hl">z<sub>j</sub> / T</span> + <span class="n">g<sub>j</sub></span> )' }),
          el('div', { class: 'eq', html: 'g<sub>j</sub> = −ln( −ln <span class="n">u<sub>j</sub></span> )' }),
          el('div', { class: 'eq', html: 'u<sub>j</sub> = uniform( <span class="n">seed</span>, <span class="n">position</span>, <span class="n">token id</span> )' }),
          prose(`
            <p>The noise is not drawn from a stream. It is <b>computed</b> from three integers by hashing
            them — so the same (seed, position, token) always yields the same noise, no matter what else
            the machine is doing.</p>
            <p>That single design choice buys three things that matter enormously in a real server:</p>
            <ul>
              <li><b>Batching is safe.</b> Eight requests sharing a GPU cannot perturb each other's randomness,
              because there is no shared stream to perturb.</li>
              <li><b>The GPU can sample in parallel.</b> A thousand threads each score their own candidate
              and the winner falls out of a max-reduction — no sequential CDF walk.</li>
              <li><b>Speculative decoding becomes checkable.</b> A draft model proposes tokens; the big model
              verifies them. Acceptance is an exact integer comparison, because "what the big model would
              have chosen" is a reproducible function, not a dice throw.</li>
            </ul>
          `)))));

    /* ---------------- temperature geometry ---------------- */
    const tempPlot = new Plot({
      width: 420, height: 250, pad: { l: 46, r: 14, t: 14, b: 38 },
      xlim: [1, 12], ylim: [0, 1], xlabel: 'token rank', ylabel: 'probability',
    });
    const TEMPS = [0.3, 0.7, 1.0, 1.6];
    tempPlot.onDraw((p) => {
      const { logits, res } = current();
      const order = res.order.slice(0, 12);
      const series = TEMPS.map((T) => {
        const s = softmax(logits, T);
        return order.map((id, k) => [k + 1, s[id]]);
      });
      p.ylim = [0, Math.max(0.1, Math.max(...series.flat().map((d) => d[1])) * 1.1)];
      p.axes({ xticks: 6, yticks: 4, zero: false });
      series.forEach((pts, i) => {
        const on = Math.abs(TEMPS[i] - state.params.temperature) < 0.16;
        p.line(pts, { color: `c${i + 1}`, width: on ? 3 : 1.5, alpha: on ? 1 : 0.45 });
        p.scatter(pts, { color: `c${i + 1}`, r: on ? 3.5 : 2, alpha: on ? 1 : 0.45 });
      });
      const cur = softmax(logits, Math.max(0.01, state.params.temperature));
      p.line(order.map((id, k) => [k + 1, cur[id]]), { color: 'accent', width: 2, dash: [5, 4] });
    });

    root.append(card('What temperature actually does',
      'Temperature divides the logits before the softmax. It adds no randomness — the randomness was always there. It reshapes how steep the ranking is.',
      row(
        el('div', {}, tempPlot.canvas,
          el('div', { style: { height: '8px' } }),
          legend(TEMPS.map((t, i) => ({ c: `c${i + 1}`, label: `T = ${t.toFixed(1)}` })).concat([{ c: 'accent', label: 'your setting' }]))),
        grow(
          el('div', { class: 'eq', html: 'p<sub>i</sub> = exp(<span class="hl">z<sub>i</sub></span> / <span class="n">T</span>) / Σ<sub>j</sub> exp(<span class="hl">z<sub>j</sub></span> / <span class="n">T</span>)' }),
          prose(`
            <p><b>T → 0</b> drives the biggest logit's share to 1: the distribution collapses onto the argmax.
            That is why <code class="inline">temperature 0</code> and "greedy" are the same request.</p>
            <p><b>T = 1</b> is the distribution the model actually learned. Nothing is being corrected.</p>
            <p><b>T &gt; 1</b> flattens the ranking, lending mass to tokens the model rated poorly.</p>
            <p>Note what temperature does <i>not</i> do: it never changes the <b>order</b> of the tokens.
            Rank 1 stays rank 1 at any temperature — which is exactly why top-k can be applied before it.</p>
          `)))));

    /* ---------------- grounding ---------------- */
    root.append(card('In the real engine',
      'These are not teaching knobs. They are the serve flags of the engine in <code class="inline">TensorFold/</code>, and the order below is its source, not a simplification.',
      row(
        grow(
          el('pre', { class: 'code', html:
            `<span class="cm">// zig/src/core/lanes/sampling.zig:135-178  —  chooseOrdered()</span>\n` +
            `if (s.temperature &lt;= 0) return order[0].id;  <span class="cm">// greedy</span>\n` +
            `const k = @max(1, @min(... s.top_k ..., width)); <span class="cm">// top-k: prefix cut</span>\n` +
            `for (scaled, order[0..k]) |*x, o| x.* = o.value / t; <span class="cm">// temperature</span>\n` +
            `    ... keep = @min(keep, below + 1);            <span class="cm">// top-p</span>\n` +
            `    const floor = scaled[0] + s.minLog();        <span class="cm">// min-p (log space)</span>\n` +
            `const gumbel = -log(-log(uniform(s.seed, position, order[j].id)));\n` +
            `const score = scaled[j] + gumbel;               <span class="cm">// the draw</span>` }),
          prose(`
            <p>Three implementations of that rule live in the repo and must agree: a 23-line sorting
            oracle (<code class="inline">sampling_reference.zig</code>), this canonical host path, and a
            Metal kernel that runs the race across 1024 threads
            (<code class="inline">sample.metal</code>).</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>top-k before temperature.</b> Surprising, and sound: temperature is monotonic, so it
                never reorders candidates — a prefix cut is the same prefix either way. <b>top-p is not
                order-free</b>: it is computed on the temperature-scaled probabilities, so swapping those
                two would select a different nucleus.`),
          note(`<b>top-p includes the crossing token.</b> The engine keeps <code class="inline">below + 1</code>
                candidates: the token that pushes the cumulative mass past <i>p</i> is kept, not dropped.`),
          note(`<b>min-p is additive.</b> Rather than compare probabilities, the engine adds
                <code class="inline">ln(min_p)</code> to the top scaled logit and keeps everything above
                that floor. Same set, no division, no renormalisation.`),
          note(`<b>The only thing that changes upstream.</b> Replacing this n-gram with a 30B transformer
                changes nothing on this page. It changes <i>where the logits come from</i> — which is the
                entire rest of the course.`)))));

    /* ---------------- the model ---------------- */
    const ta = textbox({
      label: 'training text — edit it and the model is rebuilt from scratch',
      rows: 7, value: DEFAULT_CORPUS,
    });
    // The visible effect of retraining is ~3000px up the page, so the button
    // looked dead. Report what the new model is, right here, and offer a jump.
    const trained = el('div', { class: 'note', style: { display: 'none' } });
    const seeAbove = button('see it above ↑', () => jumpToCard('Prefill, then the decode loop'), { ghost: true });
    seeAbove.style.display = 'none';

    const retrain = () => {
      const text = ta.get() || DEFAULT_CORPUS;
      const model = new NGram(3).train(text);
      state.model = model;
      seedTokens(tokenize(text).slice(0, 4).join(' '));
      refresh();
      const words = tokenize(text).length;
      trained.innerHTML = `<b>Rebuilt from this text.</b> ${words.toLocaleString()} tokens,
        <b>${model.vocab.length.toLocaleString()}</b> distinct — so every prediction on this page now
        comes from those ${model.vocab.length.toLocaleString()} words and nothing else. The output and the
        bars at the top of the page have been reset and now use this model.`;
      trained.style.display = '';
      seeAbove.style.display = '';
    };

    root.append(card('The model behind the numbers',
      'A trigram model: it counts which token followed which pair of tokens, and normalises. Primitive — but the probabilities are real, and they came from this text, which you can read in full.',
      row(
        grow(ta,
          el('div', { style: { height: '10px' } }),
          btns(
            button('retrain', retrain),
            seeAbove,
            button('restore default', () => { ta.set(DEFAULT_CORPUS); retrain(); }, { ghost: true })),
          el('div', { style: { height: '10px' } }),
          trained),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Where it fails.</b> It sees two tokens of context. Ask it for a long sentence and it
                loses the thread immediately — it has no idea what it said five tokens ago.`),
          note(`<b>And that is the motivation for everything below.</b> Attention exists to make the
                distribution depend on <i>all</i> the earlier tokens, not the last two. Same loop,
                same sampler — a far better guess.`)))));

    refresh();
    return () => stops.forEach((s) => s());
  },
};
