// 21-context.js — the edges of the context window: stretching it, and surviving overflow.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, checkbox, fmt } from '../ui.js';

/* Verified constants for Qwen3.5-2B (zig/src/families/qwen3_5/config.zig:7-21):
   head_dim = 256, rotary_dim = 64, theta = 1e7.
   So 32 rotating pairs, each with its own frequency.                           */
const ROTARY_DIM = 64;
const PAIRS = ROTARY_DIM / 2;
const THETA = 1e7;

/** Wavelength of pair i, in tokens: the distance over which it completes one full turn. */
const wavelength = (i, base = THETA) => 2 * Math.PI * Math.pow(base, (2 * i) / ROTARY_DIM);

export default {
  id: '21-context',
  num: '21',
  group: 'the real thing',
  title: 'The edges of the window',
  blurb: 'A model trained at 8k can answer at 128k, and a conversation that no longer fits does not have to fail. Both tricks happen outside the weights — one rescales the angles positions are encoded with, the other throws away part of your conversation on purpose.',

  render(root) {
    const S = { window: 8192, factor: 4, highlight: -1 };

    /* ---------------- 1. what RoPE actually encodes ---------------- */
    const waveStats = stats([
      { k: 'fastest pair', v: '—' }, { k: 'slowest pair', v: '—' },
      { k: 'resolved in window', v: '—' }, { k: 'rotary dims', v: `${ROTARY_DIM} of 256` },
    ]);

    const wavePlot = new Plot({
      width: 660, height: 320, pad: { l: 66, r: 16, t: 24, b: 44 },
      xlim: [0, PAIRS - 1], ylim: [0, 1], xlabel: 'rotary dimension pair', ylabel: 'wavelength (tokens)',
    });

    wavePlot.onDraw((p) => {
      const pts = [];
      for (let i = 0; i < PAIRS; i++) pts.push([i, Math.log10(wavelength(i))]);
      p.ylim = [0, Math.max(...pts.map((q) => q[1])) * 1.08];
      p.axes({ xticks: 8, yticks: 6, zero: false, yfmt: (v) => (v === 0 ? '1' : `1e${Math.round(v)}`) });

      const line = Math.log10(S.window);
      // dims whose wavelength exceeds the window never complete a turn inside it
      let split = PAIRS;
      for (let i = 0; i < PAIRS; i++) if (wavelength(i) > S.window) { split = i; break; }

      p.ctx.save();
      p.ctx.fillStyle = color('c3');
      p.ctx.globalAlpha = 0.1;
      p.ctx.fillRect(p.pad.l, p.pad.t, (split / (PAIRS - 1)) * p.iw, p.ih);
      p.ctx.fillStyle = color('c2');
      p.ctx.fillRect(p.pad.l + (split / (PAIRS - 1)) * p.iw, p.pad.t, p.iw - (split / (PAIRS - 1)) * p.iw, p.ih);
      p.ctx.restore();

      p.line([[0, line], [PAIRS - 1, line]], { color: 'c5', width: 2, dash: [5, 4] });
      p.text(PAIRS - 1, line, `context window ${S.window.toLocaleString()} `, { color: 'c5', size: 10, align: 'right', baseline: 'bottom' });
      p.line(pts, { color: 'accent', width: 2.5 });
      p.scatter(pts, { color: 'accent', r: 2.5 });

      p.text(p.pad.l + 8, p.pad.t + 14,
        `${split} pairs turn at least once inside the window`, { color: 'c3', size: 11, px: true, align: 'left' });
      p.text(p.pad.l + p.iw - 8, p.pad.t + 14,
        `${PAIRS - split} never complete a turn`, { color: 'c2', size: 11, px: true, align: 'right' });

      waveStats.set('fastest pair', fmt.n(wavelength(0), 1) + ' tokens');
      waveStats.set('slowest pair', wavelength(PAIRS - 1).toExponential(1) + ' tokens');
      waveStats.set('resolved in window', `${split} of ${PAIRS}`);
    });

    root.append(card('Position is encoded at many different speeds',
      'Lesson 06 rotated one pair of numbers by an angle proportional to position. A real model rotates <b>32 pairs at 32 different rates</b> — fast ones resolve neighbours, slow ones encode "far away". These are Qwen3.5-2B\'s real constants: 64 rotary dims of 256, θ = 1e7.',
      row(
        grow(wavePlot.canvas,
          el('div', { style: { height: '8px' } }),
          legend([
            { c: 'c3', label: 'completes at least one rotation inside the window' },
            { c: 'c2', label: 'never completes one — encodes "far", not "how far"' },
            { c: 'c5', label: 'the context window' },
          ])),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          waveStats,
          slider({
            label: 'context window', min: 1024, max: 262144, step: 1024, value: 8192,
            fmt: (v) => (v >= 1024 ? `${Math.round(v / 1024)}k` : String(v)),
            onInput: (v) => { S.window = v; wavePlot.render(); extendPlot.render(); bandPlot.render(); },
          }),
          note(`<b>The fastest pair turns every ~6 tokens.</b> The slowest has a wavelength of about
                <b>50 million</b> — inside any real window it barely moves, so it encodes a coarse sense of
                position rather than a precise one.`),
          note(`<b>Only 64 of 256 dimensions rotate at all</b> in this model. The other 192 carry content with
                no positional signal whatsoever — verified in
                <code class="inline">config.zig:7-21</code> and the fused RoPE write in
                <code class="inline">layout.metal:47-73</code>.`)))));

    /* ---------------- 2. stretching it ---------------- */
    /**
     * YaRN's band boundaries, straight from `zig/src/core/rope_yarn.zig:1-26`:
     *   dimOf(rot, b, max_pos) = 64 · ln(max_pos / (rot·2π)) / (2·ln b)
     * with beta_fast = 32 and beta_slow = 1. Below `low` a pair is left alone,
     * above `high` it is divided by the factor, between them it is blended.
     */
    const dimOf = (rot, b, maxPos) => (ROTARY_DIM * Math.log(maxPos / (rot * 2 * Math.PI))) / (2 * Math.log(b));
    const yarnBands = (trained, base = THETA) => {
      const low = Math.max(Math.floor(dimOf(32, base, trained)), 0);
      let high = Math.min(Math.ceil(dimOf(1, base, trained)), ROTARY_DIM - 1);
      if (low === high) high += 0.001;
      return { low, high };
    };
    /** The blend weight: 1 = untouched, 0 = fully divided by the factor. */
    const efOf = (k, low, high) => 1 - Math.min(1, Math.max(0, (k - low) / (high - low)));

    const bandStats = stats([
      { k: 'untouched', v: '—' }, { k: 'blended', v: '—' },
      { k: 'divided by factor', v: '—' }, { k: 'attention factor', v: '—' },
    ]);

    const bandPlot = new Plot({
      width: 660, height: 86, pad: { l: 0, r: 0, t: 22, b: 18 }, plain: true,
    });
    bandPlot.onDraw((p) => {
      const { low, high } = yarnBands(S.window);
      const c = p.ctx;
      const w = p.w / PAIRS;
      let a = 0, b = 0, d = 0;
      for (let k = 0; k < PAIRS; k++) {
        const ef = efOf(k, low, high);
        const kind = ef === 1 ? 'c3' : ef === 0 ? 'c1' : 'c5';
        if (ef === 1) a++; else if (ef === 0) d++; else b++;
        c.fillStyle = color(kind);
        c.globalAlpha = S.factor > 1 ? 1 : 0.25;
        c.fillRect(k * w + 1, 26, w - 2, 34);
        c.globalAlpha = 1;
        if (k % 4 === 0) {
          c.fillStyle = color('muted');
          c.font = '9px ui-monospace, Menlo, monospace';
          c.textAlign = 'center'; c.textBaseline = 'top';
          c.fillText(String(k), k * w + w / 2, 64);
        }
      }
      c.fillStyle = color('muted');
      c.font = '10px ui-monospace, Menlo, monospace';
      c.textAlign = 'left'; c.textBaseline = 'top';
      c.fillText(S.factor > 1
        ? `each of the 32 pairs, at ${S.factor.toFixed(1)}× — low = ${low}, high = ${Math.floor(high)}`
        : 'at 1× nothing is rescaled — raise the factor',
        2, 6);
      bandStats.set('untouched', `${a} pairs`);
      bandStats.set('blended', `${b} pairs`);
      bandStats.set('divided by factor', `${d} pairs`);
      bandStats.set('attention factor', fmt.n(0.1 * Math.log(Math.max(1, S.factor)) + 1, 4));
    });

    const extendPlot = new Plot({
      width: 660, height: 300, pad: { l: 54, r: 16, t: 24, b: 44 },
      xlim: [0, 1], ylim: [-1.3, 1.3], xlabel: 'position', ylabel: 'the pair\'s rotation',
    });

    extendPlot.onDraw((p) => {
      const trained = S.window;
      const want = trained * S.factor;
      p.xlim = [0, want];
      p.ylim = [-1.35, 1.35];
      p.axes({ xticks: 5, yticks: 4, zero: true, xfmt: (v) => (v >= 1000 ? `${Math.round(v / 1024)}k` : String(Math.round(v))) });

      // a mid-frequency pair, shown three ways
      const i = 10;
      const w = wavelength(i);
      const sample = (f) => {
        const pts = [];
        for (let x = 0; x <= want; x += Math.max(1, want / 400)) pts.push([x, Math.cos((2 * Math.PI * x) / (w * f))]);
        return pts;
      };
      // trained region shading
      p.ctx.save();
      p.ctx.fillStyle = color('c3');
      p.ctx.globalAlpha = 0.08;
      p.ctx.fillRect(p.pad.l, p.pad.t, (Math.min(trained, want) / want) * p.iw, p.ih);
      p.ctx.restore();
      p.line([[trained, -1.35], [trained, 1.35]], { color: 'c3', width: 1.5, dash: [4, 4] });
      p.text(trained, 1.28, ' trained window', { color: 'c3', size: 10, align: 'left' });

      p.line(sample(1), { color: 'c2', width: 2 });          // extrapolate: same rate, unseen angles
      p.line(sample(S.factor), { color: 'accent', width: 2.5 }); // interpolate: slowed to fit

      p.text(p.pad.l + 8, p.pad.t + 14, 'red: unchanged rate — positions past the window produce angles never trained on',
        { color: 'c2', size: 10, px: true, align: 'left' });
      p.text(p.pad.l + 8, p.pad.t + 28, `blue: rate divided by ${S.factor.toFixed(1)} — the whole range squeezed into trained angles`,
        { color: 'accent', size: 10, px: true, align: 'left' });
    });

    root.append(card('Stretching the window without retraining',
      'If the model never saw position 40,000, its angles there are new. Two bad options and one compromise: leave the rate alone and extrapolate into unseen angles, or slow every rate down and lose the ability to tell neighbours apart.',
      row(
        grow(extendPlot.canvas,
          el('div', { style: { height: '10px' } }),
          el('div', { class: 'sub', style: { margin: '0 0 2px' }, text: 'what YaRN actually does to each of the 32 pairs' }),
          bandPlot.canvas,
          legend([
            { c: 'c3', label: 'left alone (ef = 1)' },
            { c: 'c5', label: 'blended' },
            { c: 'c1', label: 'divided by the factor (ef = 0)' },
          ])),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          slider({
            label: 'extension factor <span class="badge">--rope-yarn</span>', min: 1, max: 16, step: 0.5, value: 4,
            fmt: (v) => `${v.toFixed(1)}× → ${Math.round(S.window * v / 1024)}k`,
            onInput: (v) => { S.factor = v; extendPlot.render(); bandPlot.render(); },
          }),
          bandStats,
          prose(`
            <p><b>Pure extrapolation</b> (red) keeps resolution but asks the model to interpret angles it has
            never been trained on. <b>Pure interpolation</b> (blue) divides every frequency by the factor —
            safe, but neighbouring tokens become that much closer in angle, blurring the fast dimensions that
            distinguish "the previous word" from "two words ago".</p>
            <p><b>YaRN does neither uniformly.</b> It inverts the wavelength formula to find two boundaries —
            the pair whose wavelength is <i>trained/32</i>, and the pair whose wavelength is <i>trained</i> —
            then leaves everything below the first alone, divides everything above the second, and ramps
            linearly between. Naive position interpolation divides all 32 pairs; at the repo's own
            Qwen3.6 settings YaRN divides <b>10</b>, blends <b>7</b>, and leaves <b>15</b> untouched.</p>`),
          note(`<b>The attention factor is logarithmic.</b> YaRN also scales attention by
                <code class="inline">0.1·ln(factor) + 1</code>, applied to cos and sin inside the rotary kernel
                — so ×4 buys only +13.9% and ×16 only +27.7%. It is exactly 1 at factor 1, which is why one
                kernel serves both the plain and extended paths.`),
          note(`<b>It is a serve-time flag, not a retrain.</b> <code class="inline">--rope-yarn F</code> applies
                a factor over the checkpoint's trained window; builds without support refuse the flag rather
                than ignore it. The frequency table is bit-pinned to transformers — the repo asserts it against
                torch's own values.`)))));

    /* ---------------- 3. the hard limit ---------------- */
    const L = { prompt: 2012, reply: 8000, window: 8192 };
    const limitBox = el('pre', { class: 'code' });
    const refreshLimit = () => {
      const room = L.window - L.prompt;
      const ok = L.prompt < L.window && L.reply <= room;
      limitBox.innerHTML = ok
        ? `<span style="color:var(--ok)">200 OK</span> — prompt ${L.prompt} + reply ${L.reply} ≤ ${L.window}`
        : L.prompt >= L.window
          ? `<span style="color:var(--err)">400 context_length_exceeded</span>\n\n` +
            `"This server's maximum context length is ${L.window} tokens, but the\n` +
            ` rendered prompt has ${L.prompt} tokens and leaves no room for a reply,\n` +
            ` which exceeds the context window. Compact or shorten the conversation."`
          : `<span style="color:var(--err)">400 context_length_exceeded</span>\n\n` +
            `"This server's maximum context length is ${L.window} tokens, but the\n` +
            ` rendered prompt has ${L.prompt} tokens and requests ${L.reply} reply\n` +
            ` tokens, which exceeds the context window. Reduce the prompt to at most\n` +
            ` ${Math.max(0, L.window - L.reply)} prompt tokens or request at most ${room} reply tokens,\n` +
            ` including chat template and thinking tokens."`;
      limitStats.set('room for a reply', `${Math.max(0, room)} tokens`);
      limitStats.set('verdict', ok ? 'accepted' : 'refused', ok ? 'ok' : 'err');
    };
    const limitStats = stats([{ k: 'room for a reply', v: '—' }, { k: 'verdict', v: '—' }]);

    root.append(card('The window is a hard refusal, not a truncation',
      'The engine will not quietly drop your tokens. It computes <code class="inline">room = window − prompt</code> and refuses before a byte of reply is generated. These are the real messages from the running server.',
      row(
        grow(limitBox),
        el('div', { class: 'col', style: { flex: '0 1 320px' } },
          limitStats,
          slider({ label: 'rendered prompt tokens', min: 100, max: 12000, step: 100, value: 2012, fmt: (v) => v.toLocaleString(), onInput: (v) => { L.prompt = v; refreshLimit(); } }),
          slider({ label: 'max_tokens requested', min: 100, max: 9000, step: 100, value: 8000, fmt: (v) => v.toLocaleString(), onInput: (v) => { L.reply = v; refreshLimit(); } }),
          note(`<b>"Rendered" prompt.</b> The count includes the chat template, the tool schemas (lesson 20) and
                any thinking tokens — not just your text. People are routinely surprised by a few hundred tokens
                they never wrote.`),
          note(`<b>An advertised window is not a promise.</b> This model reports
                <code class="inline">context_length: 8192</code> and
                <code class="inline">capabilities.context_fitted: false</code> — the second says the server has
                not guaranteed that every workload at that length fits in memory.`)))));

    /* ---------------- 4. compaction ---------------- */
    root.append(card('Or: throw part of the conversation away, on purpose',
      'With <code class="inline">--compact-at</code>, a conversation that would overflow is <b>compacted</b> instead of refused. The engine keeps the system prompt and the most recent turns word for word, and asks the model to rewrite everything older as a structured note.',
      row(
        grow(
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'observed on a server started with --context 2048 --compact-at auto --compact-keep 400' }),
          el('pre', { class: 'code', html:
            `<span class="cm">// 22 messages sent; the reply\'s "tensorfold" object came back:</span>\n` +
            `"context_window": 2048,\n` +
            `"context_used":   1741,\n` +
            `"compaction": {\n` +
            `    "messages_compacted": <b>20</b>,\n` +
            `    "kept_messages":      <b>1</b>,\n` +
            `    "note_tokens":        3828,\n` +
            `    "note":               "# Goal\\nProvide a comprehensive overview…"\n` +
            `}` }),
          el('div', { style: { height: '10px' } }),
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'and the note the model wrote, verbatim (first lines)' }),
          el('pre', { class: 'code', html:
            `<b># Goal</b>\n` +
            `Provide a comprehensive overview of packing in Lisbon, emphasizing\n` +
            `safety, location, and the unique nature of each sector.\n\n` +
            `<b># Constraints and preferences</b>\n` +
            `*   <b>Tone:</b> Informative, encouraging, and safety-conscious.\n` +
            `*   <b>Formatting:</b> Use Markdown sections: Goal, Constraints and\n` +
            `    preferences, Progress, Decisions, Next steps, Critical context.\n` +
            `*   <b>Critical context:</b> Keep exact names, paths, numbers, and\n` +
            `    error text.\n\n` +
            `<b># Progress</b> …` })),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b>The summariser is the model you are serving.</b> Nothing separate runs. The note above was
                written by the same 2B checkpoint, mid-request, before it answered — which is why 20 messages
                collapsed to one kept message and the reply still landed.`),
          note(`<b>It is a summary, so it loses things.</b> The headings exist to protect what matters most —
                goal, constraints, decisions, next steps, exact names and numbers — but a detail nobody wrote
                down under one of those headings is simply gone. Compaction trades <i>fidelity</i> for
                <i>continuing at all</i>.`),
          note(`<b>Later compactions update the note</b> rather than starting over, and a client that sends the
                compacted conversation back keeps it. The same request produces the same compaction, so the
                behaviour stays reproducible.`),
          note(`<b>Off by default.</b> Without the flag every reply is unchanged and an overflowing conversation
                gets the 400 above. That is the right default: silently discarding a user's context is a
                decision an application should make, not a server.`)))));

    refreshLimit();
  },
};
