// 18-live.js — the whole course, checked against a real engine on this machine.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, textbox, fmt, presets } from '../ui.js';
import { probe, capabilities, sampleOnce, measureDistribution, stream, HOSTED } from '../engine/live.js';
import { sample, entropy, perplexity } from '../engine/sample.js';
import { NGram, tokenize } from '../engine/ngram.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';

const SHOW = 14;
const label = (t) => (t === '' ? '∅' : t.replace(/ /g, '␣').replace(/\n/g, '⏎'));

export default {
  id: '18-live',
  num: '18',
  group: 'the real thing',
  title: 'Against a real engine',
  blurb: 'Everything so far ran on toys small enough to read. Here the lab talks to an actual TensorFold server on this machine — a 2B model on Metal, 4-bit weights — and checks the story the course has been telling.',

  render(root) {
    const S = {
      model: null, ctx: null, connected: false, caps: {}, err: null,
      prompt: 'The capital of France is',
      counts: null, n: 0, ms: 0, busy: false,
      temperature: 1.2, topP: 1, topK: 0, draws: 60,
      ngram: new NGram(3).train(DEFAULT_CORPUS),
    };

    /* ---------------- connection ---------------- */
    const statusLine = el('div', { class: 'note' });
    const capLine = el('div', { class: 'tokens' });
    const connBtn = button('connect', () => connect());

    const renderStatus = () => {
      if (S.connected) {
        statusLine.className = 'note';
        statusLine.innerHTML = `<b style="color:var(--ok)">connected</b> — serving
          <code class="inline">${S.model}</code>${S.ctx ? `, context ${S.ctx.toLocaleString()} tokens` : ''}.
          Every number below came out of that process, on this machine, just now.`;
        capLine.replaceChildren(...Object.entries(S.caps).map(([k, v]) => el('span', {
          class: 'tok',
          style: { borderColor: v ? 'var(--ok)' : 'var(--border)', opacity: v ? '1' : '0.5' },
        }, `${k}: ${v}`)));
      } else if (HOSTED) {
        // Static host: there is no proxy and never will be. Say so plainly
        // rather than naming a port the reader never started.
        statusLine.className = 'note';
        statusLine.innerHTML = `<b>You are reading the published copy, so there is no engine behind this page.</b>
          <br><span style="color:var(--muted)">A static host serves files; it cannot run a 2-billion-parameter
          model for every visitor. <a class="xref" href="#/01-sampling">Lessons 01</a>–<a class="xref"
          href="#/17-deltanet">17</a> are complete here — they compute everything in this tab.
          This lesson and the four after it <i>measure</i> a real server, so they need the lab on your own
          machine. The commands below are the whole recipe.</span>`;
        capLine.replaceChildren();
      } else {
        statusLine.className = 'note err';
        statusLine.innerHTML = `<b>no engine on 127.0.0.1:8080.</b>
          ${S.err ? `<br><span style="color:var(--muted)">${S.err}</span>` : ''}
          <br>Start one and press connect.`;
        capLine.replaceChildren();
      }
    };

    const connect = async () => {
      connBtn.disabled = true; connBtn.textContent = 'connecting…';
      const r = await probe();
      S.connected = r.ok;
      S.err = r.ok ? null : r.error;
      if (r.ok && r.models.length) {
        S.model = r.models[0].id;
        S.ctx = r.models[0].context;
        try { S.caps = await capabilities(); } catch { S.caps = {}; }
      }
      connBtn.disabled = false;
      connBtn.textContent = S.connected ? 'reconnect' : 'connect';
      renderStatus();
    };

    // The recipe sits below the two columns, at full card width: these are long
    // shell lines and a narrow column wraps them mid-URL.
    const recipe = el('pre', { class: 'code', html:
      `<span class="cm"># 1. the lab you are reading, served locally</span>\n` +
      `git clone https://github.com/eranbes/aiforall &amp;&amp; cd aiforall\n` +
      `./serve.py 8777\n\n` +
      `<span class="cm"># 2. the engine: Zig pinned by .zig-version, plus Xcode's Metal compiler</span>\n` +
      `git clone https://github.com/ashhart/TensorFold\n` +
      `xcodebuild -downloadComponent MetalToolchain\n` +
      `cd TensorFold &amp;&amp; zig build native -Doptimize=ReleaseFast\n\n` +
      `<span class="cm"># 3. weights, then serve them on :8080</span>\n` +
      `./zig-out/native/bin/tensorfold-native pull mlx-community/Qwen3.5-2B-MLX-4bit\n` +
      `./zig-out/native/bin/tensorfold-native serve mlx-community/Qwen3.5-2B-MLX-4bit \\\n` +
      `    --name local-model --context 8192 --no-thinking\n\n` +
      `<span class="cm"># then open http://127.0.0.1:8777/#/18-live and press connect</span>` });

    root.append(card('Is anything listening?',
      'The lab is a static page; the engine is a separate native process built from the Zig source this course has been reading.',
      row(
        grow(statusLine,
          el('div', { style: { height: '10px' } }),
          capLine,
          el('div', { style: { height: '10px' } }),
          btns(connBtn)),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>What the checkpoint reports.</b> <code class="inline">tensorfold-native info</code> on these
                weights prints <i>“format: 4-bit, group 64”</i> and <i>“memory floor: 3.63 GiB (weights plus a
                2.00 GiB runtime margin)”</i> — lesson 09's quantisation format and lesson 10's memory-floor
                formula, printed by the engine itself.`),
          note(`<b>Why a proxy.</b> A page on :8777 may not call :8080 unless the engine allows the origin.
                Rather than require a particular launch flag, <code class="inline">serve.py</code> forwards
                <code class="inline">/engine/*</code> from the same origin. The engine is never modified.`))),
      el('div', { style: { height: '12px' } }),
      recipe));

    /* ---------------- measure the distribution ---------------- */
    const dist = new Plot({
      width: 660, height: 300, pad: { l: 52, r: 12, t: 24, b: 62 },
      xlim: [0, SHOW], ylim: [0, 1], ylabel: 'measured frequency',
    });

    dist.onDraw((p) => {
      p.axes({ xticks: 0, yticks: 4, zero: false });
      if (!S.counts || !S.counts.size) {
        p.text(p.w / 2, p.h / 2, S.busy ? `drawing ${S.n}/${S.draws} samples…` : 'press measure',
          { color: 'muted', size: 12, px: true });
        return;
      }
      const sorted = [...S.counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, SHOW);
      const freq = sorted.map(([, c]) => c / S.n);
      p.ylim = [0, Math.max(0.1, ...freq) * 1.2];
      p.bars(freq, {
        color: sorted.map((_, i) => (i === 0 ? 'c3' : 'c1')),
        labels: sorted.map(([t]) => label(t).slice(0, 10)),
        width: 0.66,
        fmtv: (v) => (v >= 0.05 ? fmt.pct(v, 0) : ''),
      });
      p.text(p.pad.l, 12, `${S.n} independent draws · ${S.counts.size} distinct tokens · ${fmt.n(S.ms / S.n, 0)} ms each`,
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    const read = stats([
      { k: 'measured entropy', v: '—' }, { k: 'measured perplexity', v: '—' },
      { k: 'top token share', v: '—' }, { k: 'distinct tokens', v: '—' },
      { k: 'trigram entropy', v: '—' }, { k: 'requests / s', v: '—' },
    ]);

    const trigramBaseline = () => {
      const ids = tokenize(S.prompt).map((t) => S.ngram.index.get(t)).filter((i) => i !== undefined);
      const { logits } = S.ngram.logits(S.ngram.context(ids.length ? ids : [0]));
      const res = sample(logits, { temperature: 1, seed: 1, position: 0 });
      read.set('trigram entropy', fmt.n(entropy(res.probs), 2) + ' bits');
    };

    const measure = async () => {
      if (!S.connected || S.busy) return;
      S.busy = true; S.counts = null; S.n = 0;
      dist.render();
      try {
        const r = await measureDistribution(S.model, S.prompt, {
          n: S.draws, temperature: S.temperature,
          onProgress: (done) => { S.n = done; dist.render(); },
        });
        S.counts = r.counts; S.n = r.n; S.ms = r.ms;
        const freq = [...r.counts.values()].map((c) => c / r.n);
        read.set('measured entropy', fmt.n(entropy(freq), 2) + ' bits');
        read.set('measured perplexity', fmt.n(perplexity(freq), 1));
        read.set('top token share', fmt.pct(Math.max(...freq), 0));
        read.set('distinct tokens', String(r.counts.size));
        read.set('requests / s', fmt.n(r.n / (r.ms / 1000), 1));
      } catch (e) {
        S.err = String(e.message || e);
        renderStatus();
      }
      S.busy = false;
      trigramBaseline();
      dist.render();
    };

    let promptPresets = null;
    const promptBox = textbox({
      label: 'prompt', rows: 2, value: S.prompt,
      onInput: (v) => { S.prompt = v; if (promptPresets) promptPresets.select(null); },
    });

    root.append(card('The distribution, recovered by sampling',
      'This backend reports <code class="inline">logprobs: false</code> — it will not hand over the probability vector. So we do what the sampler itself does: <b>draw from it, many times, and count</b>.',
      row(
        grow(dist.canvas,
          el('div', { style: { height: '10px' } }),
          promptBox,
          el('div', { style: { height: '8px' } }),
          btns(button('measure ▸', measure)),
          el('div', { style: { height: '8px' } }),
          (promptPresets = presets([
            { label: '"2 + 2 ="', value: '2 + 2 =' },
            { label: 'open-ended', value: 'Write one word:' },
          ].map((p) => ({ ...p, onPick: (v) => { promptBox.set(v); S.prompt = v; promptPresets.select(v); measure(); } })), S.prompt))),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          read,
          slider({ label: 'temperature <span class="badge">request field</span>', min: 0, max: 2, step: 0.05, value: 1.2, fmt: (v) => (v === 0 ? '0 (greedy)' : v.toFixed(2)), onInput: (v) => { S.temperature = v; } }),
          slider({ label: 'draws', min: 10, max: 200, step: 10, value: 60, fmt: String, onInput: (v) => { S.draws = v; } }),
          note(`<b>Set temperature to 0 and measure again.</b> Every draw returns the same token and the
                measured entropy collapses to 0 bits — greedy decoding, observed rather than asserted.`),
          note(`<b>Why this is fast.</b> Sixty requests with the same prompt hit the prompt cache (lesson 13),
                so only the first pays prefill. Watch the milliseconds-per-request figure.`)))));

    /* ---------------- determinism ---------------- */
    const detBox = el('div', { class: 'col', style: { gap: '6px' } });
    const detBadge = el('span', { class: 'badge' });

    const runDeterminism = async () => {
      if (!S.connected) return;
      detBox.replaceChildren(el('div', { class: 'sub', text: 'asking…' }));
      const prompt = 'Write one word:';
      const rows = [];
      try {
        const a = await sampleOnce(S.model, prompt, { temperature: 1.5, seed: 1234 });
        const b = await sampleOnce(S.model, prompt, { temperature: 1.5, seed: 1234 });
        rows.push(['seed 1234', a.token], ['seed 1234 again', b.token]);
        for (const s of [1, 2, 3, 4]) {
          const r = await sampleOnce(S.model, prompt, { temperature: 1.5, seed: s });
          rows.push([`seed ${s}`, r.token]);
        }
        const same = a.token === b.token;
        detBadge.className = same ? 'badge ok' : 'badge err';
        detBadge.textContent = same
          ? '✓ same seed → same token, at temperature 1.5'
          : '✗ same seed gave different tokens';
      } catch (e) {
        detBadge.className = 'badge err';
        detBadge.textContent = String(e.message || e);
      }
      detBox.replaceChildren(...rows.map(([k, v], i) => el('div', {
        style: {
          display: 'flex', gap: '12px', alignItems: 'baseline', fontSize: '13px',
          padding: '5px 10px', borderRadius: '7px',
          background: i < 2 ? 'color-mix(in srgb, var(--ok) 12%, var(--surface-2))' : 'var(--surface-2)',
        },
      },
        el('span', { style: { font: '11px ui-monospace,Menlo,monospace', color: 'var(--muted)', minWidth: '120px' }, text: k }),
        el('span', { style: { font: '13px ui-monospace,Menlo,monospace', color: 'var(--accent)' }, text: label(v) }))));
    };

    root.append(card('The keyed draw, proven on a real server',
      'Lesson 01 claimed the engine\'s randomness is a <b>pure function</b> of (seed, position, token id) rather than a stream. That is a testable claim, and this is the test.',
      row(
        grow(btns(button('run the test ▸', runDeterminism)),
          el('div', { style: { height: '10px' } }),
          detBox,
          el('div', { style: { height: '10px' } }),
          detBadge),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          prose(`
            <p>Two requests with the same <code class="inline">seed</code> at temperature 1.5 return the
            <i>identical</i> token — not because the model is confident, but because the noise was computed
            from the seed rather than drawn from a stream.</p>
            <p>Change the seed and the token changes. That is exactly the property that makes speculative
            decoding checkable (lesson 14): "what the model would have produced" is a value you can recompute,
            so a draft can be accepted by integer equality.</p>`),
          note(`<b>This is also why batching is safe</b> (lesson 12). Your request's randomness cannot be
                perturbed by whoever else happens to share your round, because there is no shared stream to
                perturb.`)))));

    /* ---------------- the whole loop ---------------- */
    const out = el('div', {
      style: {
        font: '13px/1.7 ui-monospace, Menlo, monospace', whiteSpace: 'pre-wrap',
        background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '9px',
        padding: '12px 14px', minHeight: '120px', maxHeight: '260px', overflowY: 'auto',
      },
    });
    const timing = stats([
      { k: 'time to first token', v: '—' }, { k: 'tokens', v: '0' },
      { k: 'decode rate', v: '—' }, { k: 'total', v: '—' },
    ]);

    let abort = null;
    const runStream = async () => {
      if (!S.connected) return;
      abort?.abort();
      abort = new AbortController();
      out.textContent = '';
      let n = 0; const t0 = performance.now(); let tFirst = null;
      try {
        await stream(S.model, S.prompt, {
          temperature: Math.min(1, S.temperature), maxTokens: 160, signal: abort.signal,
          onToken: (piece, whole) => {
            if (tFirst === null) {
              tFirst = performance.now();
              timing.set('time to first token', fmt.n((tFirst - t0) / 1000, 2) + ' s');
            }
            n++;
            out.textContent = whole;
            out.scrollTop = out.scrollHeight;
            const secs = (performance.now() - tFirst) / 1000;
            timing.set('tokens', String(n));
            if (secs > 0.2) timing.set('decode rate', fmt.n(n / secs, 1) + ' tok/s');
          },
        });
        timing.set('total', fmt.n((performance.now() - t0) / 1000, 2) + ' s');
      } catch (e) {
        if (e.name !== 'AbortError') out.textContent += `\n\n[stream failed: ${e.message}]`;
      }
    };

    root.append(card('And the whole loop, end to end',
      'Streamed over SSE — the exact frames lesson 15 took apart, parsed here by twenty lines of client code.',
      row(
        grow(out,
          el('div', { style: { height: '10px' } }),
          btns(button('generate ▸', runStream), button('stop', () => abort?.abort(), { ghost: true }))),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          timing,
          note(`<b>Compare the two numbers.</b> Time-to-first-token is prefill over your whole prompt; the
                decode rate is one pass over the weights per token. Lesson 08 predicted the second from memory
                bandwidth alone, and lesson 15 drew exactly this timeline.`),
          note(`<b>Measured on an M1 Max: ~70 tokens/s.</b> The ceiling from lesson 08's arithmetic is
                1.75 GB of weights plus 18.9 MB of delta-net state per token, against ~400 GB/s of published
                bandwidth — about <b>226 tok/s</b>. Reaching 31% of that is unremarkable and honest: nobody
                hits peak bandwidth, the KV cache is read too, and the recurrent state (lesson 17) is read and
                written every token. The <i>shape</i> of the prediction holds; the constant does not.`)))));

    root.append(card('What just happened, in the terms of this course', null, row(
      grow(prose(`
        <ol>
          <li>Your prompt was <b>tokenized</b> by a byte-level BPE over a 248,320-entry vocabulary (lesson 02).</li>
          <li>The ids indexed an <b>embedding table</b> and became vectors (lesson 04).</li>
          <li>They passed through <b>24 blocks</b> — 6 attention, 18 gated delta-net (lessons 05, 06, 17).</li>
          <li>Every matmul ran as a <b>shape-specialised Metal kernel</b> over <b>4-bit group-64 weights</b>
          unpacked in registers (lessons 08, 09, 16).</li>
          <li>A 2048-vector met a 2048 × 248,320 matrix to make <b>logits</b> (lesson 04).</li>
          <li><b>softmax</b> turned them into the distribution you measured above (lesson 03).</li>
          <li>The <b>sampler</b>, with its keyed draw, picked one (lessons 01, 14).</li>
          <li>The id was <b>incrementally detokenised</b> and framed as SSE (lesson 15).</li>
          <li>All of it read from a <b>memory-mapped safetensors file</b> (lesson 10), whose numbers came from
          <b>gradient descent</b> (lesson 11).</li>
        </ol>`)),
      el('div', { class: 'col', style: { flex: '1 1 300px' } },
        note(`<b>Nothing in that list is a metaphor.</b> Every step has a file and a line number in
              <code class="inline">notes/</code>, and a page in this course where you built a working version
              of it yourself.`),
        note(`<b>The gap that remains</b> is scale and training data, not mechanism. The trigram from lesson 01
              and the 2B model above differ in how good the guess is — not in what the guess <i>is</i>.`)))));

    renderStatus();
    trigramBaseline();
    connect();
  },
};
