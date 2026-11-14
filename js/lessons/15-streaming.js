// 15-streaming.js — the last mile: turning committed token ids into bytes on a socket.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, textbox, legend, loop, fmt, checkbox } from '../ui.js';
import { BPE, display, fromByteString } from '../engine/bpe.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';

const charCount = (s) => [...s].length;
const afterChars = (s, n) => [...s].slice(n).join('');

/**
 * A faithful port of TensorFold's incremental detokeniser
 * (zig/src/server/reply_text.zig:452-485).
 *
 * The committed prefix advances ONLY when re-decoding the window both produced more
 * CHARACTERS and did not end in U+FFFD. Otherwise the bytes are held back until the
 * continuation token arrives.
 */
class Incremental {
  constructor(decode) {
    this.decode = decode;
    this.tokens = [];
    this.text = '';
    this.prefix = 0;
    this.read = 0;
  }
  extend(more) {
    this.tokens.push(...more);
    const before = this.decode(this.tokens.slice(this.prefix, this.read));
    const after = this.decode(this.tokens.slice(this.prefix));
    const nb = charCount(before);
    let emitted = '';
    if (charCount(after) > nb && !after.endsWith('\ufffd')) {
      emitted = afterChars(after, nb);
      this.text += emitted;
      this.prefix = this.read;
      this.read = this.tokens.length;
    }
    return emitted;
  }
  /** A character is still split across tokens, so streaming waits for the rest. */
  pending() { return this.tokens.length > 0 && this.read < this.tokens.length; }
}

export default {
  id: '15-streaming',
  num: '15',
  group: 'shipping it',
  title: 'Streaming it back',
  blurb: 'The model produced an integer. Turning integers into bytes on a socket, one at a time, without printing garbage, is harder than it looks — and it is where several visible oddities of chat UIs come from.',

  render(root) {
    const stops = [];

    /* a byte-level tokenizer with few merges, so multi-byte characters split across tokens */
    const bpe = new BPE();
    bpe.train(DEFAULT_CORPUS, 120);

    const state = {
      text: 'ok 🌍 日本語 done',
      tokens: [],
      i: 0,
      naive: '',
      inc: null,
      frames: [],
      running: false,
      acc: 0,
      held: 0,
    };

    const reset = () => {
      state.tokens = bpe.tokenize(state.text);
      state.i = 0;
      state.naive = '';
      state.inc = new Incremental((toks) => fromByteString(toks.join('')));
      state.frames = [];
      state.held = 0;
      render();
    };

    const stepToken = () => {
      if (state.i >= state.tokens.length) { state.running = false; runBtn.textContent = '▶ stream'; return; }
      const t = state.tokens[state.i++];
      // the naive way: decode this one token on its own
      state.naive += fromByteString(t);
      // the correct way: re-decode the window and diff
      const emitted = state.inc.extend([t]);
      if (emitted) state.frames.push({ i: state.i, text: emitted });
      else state.held++;
      render();
    };

    /* ---------- panes ---------- */
    const tokStrip = el('div', { class: 'tokens' });
    const naivePane = el('div', { class: 'eq', style: { minHeight: '46px', whiteSpace: 'pre-wrap' } });
    const incPane = el('div', { class: 'eq', style: { minHeight: '46px', whiteSpace: 'pre-wrap' } });
    const framePane = el('div', {
      style: {
        font: '11.5px/1.6 ui-monospace, Menlo, monospace', whiteSpace: 'pre-wrap',
        background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '9px',
        padding: '10px 12px', maxHeight: '190px', overflowY: 'auto',
      },
    });
    const read = stats([
      { k: 'tokens sent', v: '0' }, { k: 'frames emitted', v: '0' },
      { k: 'bytes held back', v: '0' }, { k: 'naive output', v: '—' },
    ]);

    const render = () => {
      tokStrip.replaceChildren(...state.tokens.map((t, i) => el('span', {
        class: 'tok',
        style: {
          opacity: i < state.i ? '1' : '0.35',
          borderColor: i === state.i - 1 ? 'var(--accent)' : 'var(--border)',
          background: t.length === 1 ? 'color-mix(in srgb, var(--c2) 18%, var(--surface-2))' : 'var(--surface-2)',
        },
        title: t.length === 1 ? `raw byte 0x${t.charCodeAt(0).toString(16).padStart(2, '0')}` : 'merged token',
      }, display(t))));

      const broken = (state.naive.match(/\ufffd/g) || []).length;
      naivePane.innerHTML = `<span style="color:var(--err)">${escapeHtml(state.naive) || '&nbsp;'}</span>`;
      incPane.innerHTML = `<span style="color:var(--ok)">${escapeHtml(state.inc ? state.inc.text : '')}</span>` +
        (state.inc && state.inc.pending()
          ? `<span style="color:var(--warn)"> ▌held: ${state.inc.tokens.length - state.inc.read} token(s)</span>`
          : '');
      framePane.textContent = state.frames.length
        ? state.frames.map((f) => `data: {"delta":${JSON.stringify(f.text)}}`).join('\n')
          + (state.i >= state.tokens.length ? '\ndata: [DONE]' : '')
        : '(nothing sent yet)';
      framePane.scrollTop = framePane.scrollHeight;

      read.set('tokens sent', `${state.i} / ${state.tokens.length}`);
      read.set('frames emitted', String(state.frames.length));
      read.set('bytes held back', String(state.inc ? state.inc.tokens.length - state.inc.read : 0));
      read.set('naive output', broken ? `${broken} × U+FFFD` : 'clean', broken ? 'err' : 'ok');
    };

    const runBtn = button('▶ stream', () => {
      state.running = !state.running;
      runBtn.textContent = state.running ? '⏸ pause' : '▶ stream';
    });
    stops.push(loop((dt) => {
      if (!state.running) return;
      state.acc += dt;
      if (state.acc > 220) { state.acc = 0; stepToken(); }
    }, tokStrip));

    root.append(card('One token at a time, without printing garbage',
      'Press <b>next token</b>. The red pane decodes each token on its own; the green pane uses the engine\'s algorithm. Red chips are single raw bytes — a character that needs several of them.',
      row(
        grow(
          textbox({
            label: 'text the model is "generating"', rows: 1, value: state.text,
            onInput: (v) => { state.text = v; reset(); },
          }),
          el('div', { style: { height: '10px' } }),
          tokStrip,
          el('div', { style: { height: '14px' } }),
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'naive: decode(token) per token' }),
          naivePane,
          el('div', { style: { height: '10px' } }),
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'incremental: decode the window, emit the difference' }),
          incPane,
          el('div', { style: { height: '12px' } }),
          btns(
            button('next token ▸', stepToken),
            runBtn,
            button('reset', reset, { ghost: true }),
            button('emoji', () => { state.text = 'ok 🌍 日本語 done'; reset(); }, { ghost: true }),
            button('plain ascii', () => { state.text = 'the engine reads the weights'; reset(); }, { ghost: true }))),
        el('div', { class: 'col', style: { flex: '0 1 320px' } },
          read,
          el('div', { class: 'sub', style: { margin: '6px 0 4px' }, text: 'what actually goes on the socket' }),
          framePane,
          note(`Watch the <b>held</b> counter. When a character is only half-arrived, the correct answer is to
                send <i>nothing</i> — not a replacement character, not a partial byte. The frame is emitted one
                token later, which is why fast streams sometimes arrive in small bursts.`)))));

    /* ---------- the algorithm ---------- */
    root.append(card('The rule is two conditions',
      null,
      row(
        grow(el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/server/reply_text.zig:452-485</span>\n` +
          `pub fn extend(inc: *Incremental, a: Allocator, ctx: anytype, decode: anytype, more: []const u32) ![]const u8 {\n` +
          `    try inc.tokens.appendSlice(a, more);\n` +
          `    const before = try decode(ctx, a, inc.tokens.items[inc.prefix..inc.read]);\n` +
          `    const after  = try decode(ctx, a, inc.tokens.items[inc.prefix..]);\n` +
          `    const nb = charCount(before);\n` +
          `    if (<b>charCount(after) &gt; nb</b> and <b>!std.mem.endsWith(u8, after, "\\u{fffd}")</b>) {\n` +
          `        try inc.text.appendSlice(a, afterChars(after, nb));\n` +
          `        inc.prefix = inc.read;\n` +
          `        inc.read = inc.tokens.items.len;\n` +
          `    }\n` +
          `    return inc.text.items;\n` +
          `}\n\n` +
          `<span class="cm">/// A character is still split across tokens, so streaming waits for the rest.</span>\n` +
          `pub fn pending(inc: *const Incremental) bool {\n` +
          `    return inc.tokens.items.len &gt; 0 and inc.read &lt; inc.tokens.items.len;\n` +
          `}` })),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Why count characters, not bytes?</b> A decoder can emit a U+FFFD replacement (3 bytes) for an
                incomplete sequence. Byte growth would look like progress; character growth plus the trailing-FFFD
                check is what actually distinguishes "new text" from "a broken guess at new text".`),
          note(`<b>Why decode a window and diff it,</b> rather than decode each token? Because the decoder chain
                is context-dependent: pieces get joined, spaces get stripped, byte-level pieces combine. The only
                correct incremental text is <code class="inline">decode(window) − decode(window minus the new
                tokens)</code>.`),
          note(`<b>The code above is running on this page</b> — the green pane is a direct port. The red pane is
                the obvious implementation, and it is wrong for any non-ASCII text.`)))));

    /* ---------- who stops the stream ---------- */
    root.append(card('Who decides the reply is over',
      'Three different stop conditions, owned by two different components, for one good reason.',
      row(
        grow(el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/server/chat_generation.zig:91-103 — commit()</span>\n` +
          `for (chunk) |t| {\n` +
          `    if (g.reason != null) break;\n` +
          `    try g.collected.append(g.a, t);\n` +
          `    landed += 1;\n` +
          `    if (g.eos(t)) {\n` +
          `        g.reason = "stop";\n` +
          `    } else if (try g.stopHit()) {\n` +
          `        g.reason = "stop";\n` +
          `        g.srv.engine.cancel(g.id); <span class="cm">// the engine ends EOS and length itself;</span>\n` +
          `                                   <span class="cm">// a stop string is the server's</span>\n` +
          `    } else if (g.collected.items.len &gt;= g.max_tokens) g.reason = "length";\n` +
          `}` })),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          el('div', { class: 'eq', html: 'engine owns: <span class="hl">EOS</span> · <span class="hl">length</span>' }),
          el('div', { class: 'eq', html: 'server owns: <span class="hl">stop strings</span>' }),
          prose(`
            <p>The split is forced by information: a stop string like <code class="inline">"\\nUser:"</code> exists
            in <i>text</i>, and only the server has detokenised text. The engine sees integers, so it can only
            recognise the end-of-sequence id and count tokens.</p>
            <p>That is also why a stop string must explicitly <code class="inline">cancel</code> the engine: the
            engine was perfectly happy to keep going.</p>`),
          note(`<b>It only decodes the tail.</b> <code class="inline">stopHit</code> re-decodes the last few
                tokens each round rather than the whole reply — otherwise stop-string checking would be O(n²)
                over the length of the reply.`)))));

    /* ---------- the wire ---------- */
    root.append(card('The wire format is 35 lines',
      'Everything hard happened upstream. "Streaming an LLM" is text framing.',
      row(
        grow(el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/server/sse.zig — the entire protocol</span>\n` +
          `pub fn open(conn: *Conn) error{Closed}!void {\n` +
          `    conn.startResponse(200, null) catch return error.Closed;\n` +
          `    conn.addHeader("Content-Type", "text/event-stream") catch return error.Closed;\n` +
          `    conn.addHeader("Cache-Control", "no-cache") catch return error.Closed;\n` +
          `    conn.addHeader("Connection", "close") catch return error.Closed;\n` +
          `    return conn.finish("");\n` +
          `}\n\n` +
          `<span class="cm">/// \`\`data: {json}\`\` and a blank line, sent at once.</span>\n` +
          `pub fn data(conn: *Conn, _: Allocator, payload: json.Value) error{Closed}!void { ... }\n\n` +
          `<span class="cm">/// An SSE comment line, which clients skip: it keeps a stream alive.</span>\n` +
          `pub fn comment(conn: *Conn) error{Closed}!void {\n` +
          `    return conn.writeAll(": keepalive\\n\\n");\n` +
          `}\n\n` +
          `pub fn done(conn: *Conn) error{Closed}!void {\n` +
          `    return conn.writeAll("data: [DONE]\\n\\n");\n` +
          `}` })),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Each frame is one <code class="inline">writeAll</code></b>, so a frame can never be interleaved
                with another writer's bytes. The JSON is built in a per-frame arena that is destroyed immediately,
                so a 10,000-token stream does not grow the request's memory.`),
          note(`<b><code class="inline">: keepalive</code> exists because of prefill.</b> A long prompt can take
                seconds before the first token appears, and a client that times out a silent socket would
                disconnect mid-think. The comment frame is bytes that mean nothing and prove the server is alive.`),
          note(`<b><code class="inline">data: [DONE]</code> is not JSON.</b> It is a literal sentinel every
                OpenAI-compatible client special-cases — an accident of the original API that is now load-bearing
                everywhere.`)))));

    /* ---------- the timeline ---------- */
    const tl = { prompt: 2000, prefillRate: 3000, decodeRate: 40, tokens: 120 };
    const tlPlot = new Plot({
      width: 660, height: 190, pad: { l: 12, r: 12, t: 30, b: 42 },
      xlim: [0, 4], ylim: [0, 1],
    });
    tlPlot.onDraw((p) => {
      const prefill = tl.prompt / tl.prefillRate;            // seconds
      const decode = tl.tokens / tl.decodeRate;
      const total = prefill + decode;
      p.xlim = [0, Math.max(1, total * 1.08)];
      const y = 0.52;
      const barH = 34;
      const yPx = p.Y(y) - barH / 2;

      p.ctx.fillStyle = color('c5');
      p.ctx.fillRect(p.X(0), yPx, p.X(prefill) - p.X(0), barH);
      p.ctx.fillStyle = color('c1');
      p.ctx.fillRect(p.X(prefill), yPx, p.X(total) - p.X(prefill), barH);

      // token ticks
      p.ctx.strokeStyle = color('bg');
      p.ctx.lineWidth = 1;
      const step = decode / tl.tokens;
      for (let i = 0; i < tl.tokens; i++) {
        const x = p.X(prefill + i * step);
        if (i % Math.max(1, Math.round(tl.tokens / 60)) !== 0) continue;
        p.ctx.beginPath(); p.ctx.moveTo(x, yPx); p.ctx.lineTo(x, yPx + barH); p.ctx.stroke();
      }

      p.text(p.X(prefill) / 2 + p.X(0) / 2, yPx - 9, 'prefill (prompt)', { color: 'c5', size: 11, px: true });
      p.text((p.X(prefill) + p.X(total)) / 2, yPx - 9, 'decode (one token at a time)', { color: 'c1', size: 11, px: true });

      p.ctx.strokeStyle = color('accent');
      p.ctx.setLineDash([4, 3]);
      p.ctx.beginPath(); p.ctx.moveTo(p.X(prefill), yPx - 20); p.ctx.lineTo(p.X(prefill), yPx + barH + 14); p.ctx.stroke();
      p.ctx.setLineDash([]);
      p.text(p.X(prefill) + 6, yPx + barH + 20, `TTFT ${fmt.n(prefill, 2)} s`, { color: 'accent', size: 11, px: true, align: 'left' });
      p.text(p.X(total), yPx + barH + 20, `total ${fmt.n(total, 2)} s`, { color: 'muted', size: 11, px: true, align: 'right' });
      p.axes({ xticks: 6, yticks: 0, grid: false, zero: false, xfmt: (v) => `${v.toFixed(1)}s` });
    });

    root.append(card('Why the first token is slow and the rest are even',
      'A reply has two phases with completely different cost models. Users feel them separately, which is why both numbers are reported.',
      tlPlot.canvas,
      el('div', { style: { height: '10px' } }),
      row(
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          slider({ label: 'prompt length (tokens)', min: 100, max: 32000, step: 100, value: 2000, fmt: (v) => v.toLocaleString(), onInput: (v) => { tl.prompt = v; tlPlot.render(); } }),
          slider({ label: 'prefill speed (tokens/s)', min: 200, max: 20000, step: 100, value: 3000, fmt: (v) => v.toLocaleString(), onInput: (v) => { tl.prefillRate = v; tlPlot.render(); } }),
          slider({ label: 'decode speed (tokens/s)', min: 5, max: 200, step: 1, value: 40, fmt: String, onInput: (v) => { tl.decodeRate = v; tlPlot.render(); } }),
          slider({ label: 'reply length (tokens)', min: 10, max: 1000, step: 10, value: 120, fmt: String, onInput: (v) => { tl.tokens = v; tlPlot.render(); } })),
        grow(prose(`
          <p><b>Prefill</b> processes the whole prompt at once — matrix × matrix, arithmetic-bound, hundreds to
          thousands of tokens per second. It happens before you see anything, so it <i>is</i> time-to-first-token.</p>
          <p><b>Decode</b> produces one token per pass over the weights — matrix × vector, memory-bound, tens of
          tokens per second. Lesson 08 computed this ceiling; lesson 14 beat it by guessing.</p>
          <p>Push the prompt to 32,000 tokens and watch TTFT dominate the entire reply. That is the number a
          prompt cache attacks: if the prefix was computed before, prefill is skipped entirely.</p>`),
          note(`<b>Reported rates exclude the first token.</b> <code class="inline">decode_tokens =
                collected.len - 1</code>, because including the prefill-delayed first token in a tokens/second
                figure would make long prompts look like slow decoding.`)))));

    reset();
    return () => stops.forEach((s) => s());
  },
};

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}
