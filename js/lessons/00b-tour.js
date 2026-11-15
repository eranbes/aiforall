// 00b-tour.js — the entire story, five ideas, one page.
// If someone only ever reads one lesson, it should be this one.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, fmt, presets } from '../ui.js';
import { NGram, tokenize, detokenize } from '../engine/ngram.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';
import { sample } from '../engine/sample.js';
import { softmax, mulberry32 } from '../engine/tensor.js';

const label = (t) => (t === '\n' ? '⏎' : t);

/** A step marker so the page reads as five numbered beats, not five cards. */
const beat = (n, claim) => el('div', {
  style: { display: 'flex', gap: '12px', alignItems: 'baseline', marginBottom: '10px' },
},
  el('span', {
    style: {
      font: '12px ui-monospace,Menlo,monospace', fontWeight: '700',
      color: '#06121c', background: 'var(--accent)', borderRadius: '50%',
      width: '22px', height: '22px', display: 'inline-flex',
      alignItems: 'center', justifyContent: 'center', flex: '0 0 22px',
    },
  }, String(n)),
  el('div', { style: { fontSize: '15px', fontWeight: '600' }, html: claim }));

export default {
  id: '00b-tour',
  num: '10m',
  group: 'orientation',
  title: 'The whole thing in ten minutes',
  blurb: 'Five ideas. Each one has a button. Press every button and you will understand, in outline, how every language model works — including the ones you pay for.',

  render(root) {
    const model = new NGram(3).train(DEFAULT_CORPUS);

    /* ---------------- 1. a reply is drawn, one word at a time ---------------- */
    const S = { ids: tokenize('The engine reads the').map((t) => model.index.get(t)).filter((i) => i !== undefined) };
    const out = el('div', {
      style: {
        font: '14px/1.8 ui-monospace,Menlo,monospace', whiteSpace: 'pre-wrap',
        background: 'var(--surface-2)', border: '1px solid var(--border)',
        borderRadius: '9px', padding: '12px 14px', minHeight: '64px',
      },
    });
    const draw = () => {
      out.textContent = detokenize(S.ids.map((i) => model.vocab[i])) + ' ▌';
    };
    const stepOne = () => {
      const { logits } = model.logits(model.context(S.ids));
      const res = sample(logits, { temperature: 0.9, seed: 7, position: S.ids.length });
      S.ids.push(res.chosen);
      draw();
      barPlot.render();
    };
    draw();

    root.append(card(null, null,
      beat(1, 'A reply is not written. It is <b>drawn, one word at a time.</b>'),
      out,
      el('div', { style: { height: '10px' } }),
      btns(
        button('add a word ▸', stepOne),
        button('reset', () => {
          S.ids = tokenize('The engine reads the').map((t) => model.index.get(t)).filter((i) => i !== undefined);
          draw(); barPlot.render();
        }, { ghost: true })),
      el('div', { style: { height: '10px' } }),
      prose(`<p>Press it a few times. There is no plan and no sentence waiting to come out — the model
        picks <i>one</i> word, adds it to the end, and then starts over with a slightly longer text.
        Everything else in AI is about making that single choice well.</p>`)));

    /* ---------------- 2. the choice is made from a list of numbers ---------------- */
    const barPlot = new Plot({
      width: 660, height: 230, pad: { l: 44, r: 12, t: 16, b: 52 },
      xlim: [0, 10], ylim: [0, 1], ylabel: 'chance',
    });
    barPlot.onDraw((p) => {
      const { logits } = model.logits(model.context(S.ids));
      const probs = softmax(logits, 0.9);
      const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 10);
      const vals = order.map((i) => probs[i]);
      p.ylim = [0, Math.max(0.1, ...vals) * 1.2];
      p.axes({ xticks: 0, yticks: 4, zero: false });
      p.bars(vals, {
        color: order.map((_, i) => (i === 0 ? 'c3' : 'c1')),
        labels: order.map((i) => label(model.vocab[i])),
        fmtv: (v) => (v > 0.04 ? fmt.pct(v, 0) : ''),
      });
    });

    root.append(card(null, null,
      beat(2, 'The word is picked from a <b>list of chances</b> — one for every word it knows.'),
      barPlot.canvas,
      el('div', { style: { height: '8px' } }),
      prose(`<p>This is the <i>only</i> thing a language model produces: a score for every possible next
        word. Press <b>add a word</b> above again and watch the whole list change. The green bar is the
        favourite — but the model does not always take it, which is why the same question can get
        different answers.</p>`)));

    /* ---------------- 3. words are not words ---------------- */
    const chips = el('div', { class: 'tokens', style: { lineHeight: '2.2' } });
    const renderChips = (text) => {
      const pieces = text.match(/\w+|[^\s\w]|\s+/g) || [];
      chips.replaceChildren(...pieces.map((t, i) => el('span', {
        class: 'tok',
        style: { background: `color-mix(in srgb, var(--c${(i % 6) + 1}) 20%, var(--surface-2))` },
      }, t.replace(/ /g, '␣'))));
    };
    renderChips('The engine reads the weights.');

    root.append(card(null, null,
      beat(3, 'It does not see letters or words. It sees <b>numbered pieces</b>.'),
      chips,
      el('div', { style: { height: '10px' } }),
      presets([
        { label: 'a sentence', value: 'The engine reads the weights.' },
        { label: 'a long word', value: 'unbelievably complicated' },
        { label: 'emoji', value: 'hello 🌍 ok' },
      ].map((p) => ({ ...p, onPick: (v) => renderChips(v) })), 'The engine reads the weights.'),
      el('div', { style: { height: '10px' } }),
      prose(`<p>Each coloured chip is one piece, and each piece is really just a number — an entry in a
        list of about 250,000. "The" might be 1,109. A long or unusual word gets chopped into several
        pieces. This is why models miscount letters, and why text in some languages costs more.</p>`)));

    /* ---------------- 4. the numbers come from multiplication ---------------- */
    const mat = new Plot({
      width: 660, height: 220, pad: { l: 70, r: 14, t: 26, b: 20 }, xlim: [0, 12], ylim: [0, 6],
    });
    const mState = { h: [0.8, -0.4, 0.6, 0.2, -0.7, 0.5] };
    const rng0 = mulberry32(5);
    const W = Array.from({ length: 6 }, () => Array.from({ length: 12 }, () => rng0() * 2 - 1));
    mat.onDraw((p) => {
      p.matrix(W, { cmap: (v) => (v >= 0 ? color('c1') : color('c2')), gap: 2, max: 1 });
      p.ctx.save();
      p.ctx.font = '10px ui-monospace,Menlo,monospace';
      p.ctx.fillStyle = color('muted');
      p.ctx.textAlign = 'right'; p.ctx.textBaseline = 'middle';
      const ch = p.ih / 6;
      mState.h.forEach((v, i) => p.ctx.fillText(fmt.sign(v, 2), p.pad.l - 8, p.pad.t + i * ch + ch / 2));
      p.ctx.textAlign = 'left';
      p.ctx.fillText('your text, as 6 numbers →', p.pad.l - 60, p.pad.t - 12);
      p.ctx.restore();
    });

    root.append(card(null, null,
      beat(4, 'Those chances come from <b>multiplying numbers by a huge table</b>. That table is the model.'),
      mat.canvas,
      el('div', { style: { height: '10px' } }),
      prose(`<p>Your text becomes a list of numbers (left). They get multiplied against a table of learned
        numbers (the grid) and added up, and out comes a score for every possible next word. That is
        genuinely all a "neural network" is doing here — multiply, add, repeat.</p>
        <p>This toy grid is 6 × 12 = 72 numbers. A small real model has <b>2 billion</b>, and it does this
        multiplication about 24 times for every single word it produces.</p>`)));

    /* ---------------- 5. the table was learned ---------------- */
    const PTS = [[1, 1], [2, 2], [3, 2.4], [4, 3.3]];
    const F = { w: -0.3, b: 2.4, hist: [] };
    const loss = () => PTS.reduce((s, [x, y]) => s + (F.w * x + F.b - y) ** 2, 0) / PTS.length;
    const learn = () => {
      let gw = 0, gb = 0;
      for (const [x, y] of PTS) { const e = F.w * x + F.b - y; gw += 2 * e * x; gb += 2 * e; }
      F.w -= 0.06 * (gw / PTS.length);
      F.b -= 0.06 * (gb / PTS.length);
      F.hist.push(loss());
      if (F.hist.length > 120) F.hist.shift();
      fitPlot.render(); lossPlot.render();
      fitRead.set('error', fmt.n(loss(), 4), loss() < 0.1 ? 'ok' : null);
      fitRead.set('nudges', String(F.hist.length));
    };
    const fitRead = stats([{ k: 'error', v: '—' }, { k: 'nudges', v: '0' }]);

    const fitPlot = new Plot({
      width: 330, height: 220, pad: { l: 40, r: 14, t: 14, b: 32 },
      xlim: [0, 5], ylim: [0, 4], xlabel: 'x', ylabel: 'y',
    });
    fitPlot.onDraw((p) => {
      p.axes({ xticks: 5, yticks: 4, zero: false });
      p.fn((x) => F.w * x + F.b, { color: 'accent', width: 2.5 });
      for (const [x, y] of PTS) p.line([[x, y], [x, F.w * x + F.b]], { color: 'c2', width: 1.5, dash: [3, 3] });
      p.scatter(PTS, { color: 'c5', r: 5, stroke: 'bg' });
    });
    const lossPlot = new Plot({
      width: 330, height: 220, pad: { l: 46, r: 14, t: 14, b: 32 },
      xlim: [0, 120], ylim: [0, 1], xlabel: 'nudges', ylabel: 'how wrong',
    });
    lossPlot.onDraw((p) => {
      p.ylim = [0, Math.max(0.5, F.hist[0] || 1) * 1.1];
      p.axes({ xticks: 4, yticks: 4, zero: false });
      if (F.hist.length > 1) p.line(F.hist.map((v, i) => [i, v]), { color: 'c3', width: 2.5 });
    });

    root.append(card(null, null,
      beat(5, 'Nobody wrote those numbers. They were <b>nudged into place</b>, billions of times.'),
      row(
        el('div', {}, fitPlot.canvas),
        el('div', {}, lossPlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 220px' } },
          fitRead,
          btns(
            button('nudge ×20', () => { for (let i = 0; i < 20; i++) learn(); }),
            button('reset', () => { F.w = -0.3; F.b = 2.4; F.hist = []; fitPlot.render(); lossPlot.render(); fitRead.set('error', '—'); fitRead.set('nudges', '0'); }, { ghost: true })))),
      el('div', { style: { height: '10px' } }),
      prose(`<p>Here the "model" has just two numbers and the job is to fit a line to four dots. Press
        <b>nudge</b>: measure how wrong it is, change each number a little in the direction that makes it
        less wrong, repeat. The green curve is the wrongness going down.</p>
        <p>Training a real model is this exact loop — on billions of numbers instead of two, with "how
        wrong" meaning "how surprised it was by the next word in a real document". That is the whole of
        machine learning, and the rest is engineering.</p>`)));

    /* ---------------- what now ---------------- */
    const next = card('That was the whole story',
      'Everything else is detail underneath those five ideas. You do not owe this course 23 lessons.',
      row(
        grow(prose(`
          <ul>
            <li>Curious <b>how the choice is made</b> — temperature, why the same prompt varies?
            → lesson <b>01</b>.</li>
            <li>Curious about <b>those numbered pieces</b>? → lesson <b>02</b>.</li>
            <li>Curious <b>what the big table does</b>, and how it handles context? → lessons <b>04–06</b>.</li>
            <li>Curious <b>why it is slow or expensive</b>? → lesson <b>08</b>.</li>
            <li>Curious <b>how the nudging really works</b>? → lesson <b>11</b>.</li>
          </ul>
          <p>Follow only the ones you actually want. Each is about twenty minutes and stands on its own.</p>`)),
        el('div', { class: 'col', style: { flex: '0 1 280px' } },
          btns(button('go to lesson 01 →', () => { location.hash = '#/01-sampling'; })),
          note(`<b>If you stop here, you still got the real thing.</b> Those five ideas are not a
                simplification that gets replaced later — they are what the remaining lessons are made of.`))));
    next.classList.add('hero');
    root.append(next);
  },
};
