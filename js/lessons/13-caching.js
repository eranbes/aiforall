// 13-caching.js — why the second message of a conversation is so much faster.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, checkbox, fmt } from '../ui.js';

/* A conversation, in tokens. These counts are illustrative; the arithmetic is not. */
const START = [
  { who: 'system', label: 'system prompt + tool schemas', tokens: 1200, fixed: true },
  { who: 'user', label: 'user: "summarise this file"', tokens: 3400 },
  { who: 'assistant', label: 'assistant reply', tokens: 260 },
];

export default {
  id: '13-caching',
  num: '13',
  group: 'shipping it',
  title: 'Caches: not recomputing the past',
  blurb: 'Every turn of a conversation resends the whole conversation. Without a cache the engine would re-read 30,000 tokens to answer "and shorter please". The fix is simple, and the way it breaks is the single most common performance surprise in LLM applications.',

  render(root) {
    const S = {
      turns: START.map((t) => ({ ...t })),
      cached: 0,            // tokens of prefix already in the cache
      prefillRate: 3000,    // tokens/s
      decodeRate: 40,       // tokens/s
      editedTurn: -1,       // which turn the user "edited" (invalidates from there on)
      cacheOn: true,
      budgetGiB: 16,
      bytesPerToken: 12288, // lesson 05's Qwen3.5-2B figure
    };

    const total = () => S.turns.reduce((a, t) => a + t.tokens, 0);
    /** Tokens reusable from cache: the prefix up to the first edited turn. */
    const reusable = () => {
      if (!S.cacheOn) return 0;
      let acc = 0;
      for (let i = 0; i < S.turns.length; i++) {
        if (i === S.editedTurn) break;
        acc += S.turns[i].tokens;
      }
      return Math.min(acc, S.cached);
    };

    /* ---------- the prompt bar ---------- */
    const bar = new Plot({
      width: 660, height: 150, pad: { l: 12, r: 12, t: 34, b: 46 }, xlim: [0, 1], ylim: [0, 1],
    });
    bar.onDraw((p) => {
      const n = total();
      const hit = reusable();
      const x0 = p.pad.l, w = p.iw;
      const yTop = p.pad.t + 12, h = 42;

      // cached region
      const cw = (hit / n) * w;
      p.ctx.fillStyle = color('c3');
      p.ctx.globalAlpha = 0.5;
      p.ctx.fillRect(x0, yTop, cw, h);
      p.ctx.globalAlpha = 1;
      p.ctx.fillStyle = color('c5');
      p.ctx.fillRect(x0 + cw, yTop, w - cw, h);

      // turn boundaries
      let acc = 0;
      p.ctx.strokeStyle = color('bg');
      p.ctx.lineWidth = 1.5;
      for (const t of S.turns) {
        acc += t.tokens;
        const x = x0 + (acc / n) * w;
        p.ctx.beginPath(); p.ctx.moveTo(x, yTop); p.ctx.lineTo(x, yTop + h); p.ctx.stroke();
      }
      // edited marker
      if (S.editedTurn >= 0 && S.editedTurn < S.turns.length) {
        let pre = 0;
        for (let i = 0; i < S.editedTurn; i++) pre += S.turns[i].tokens;
        const x = x0 + (pre / n) * w;
        p.ctx.strokeStyle = color('c2');
        p.ctx.lineWidth = 2.5;
        p.ctx.beginPath(); p.ctx.moveTo(x, yTop - 10); p.ctx.lineTo(x, yTop + h + 10); p.ctx.stroke();
        p.text(x, yTop - 16, 'edited here ↓', { color: 'c2', size: 10, px: true });
      }

      p.text(x0, p.pad.t - 8, `prompt: ${n.toLocaleString()} tokens`, { color: 'muted', size: 11, px: true, align: 'left' });
      p.text(x0 + cw / 2, yTop + h / 2, hit ? `reused ${hit.toLocaleString()}` : '', { color: 'bg', size: 11, px: true, weight: 'bold' });
      p.text(x0 + cw + (w - cw) / 2, yTop + h / 2, `recompute ${(n - hit).toLocaleString()}`, { color: 'bg', size: 11, px: true, weight: 'bold' });

      const prefill = (n - hit) / S.prefillRate;
      p.text(x0, yTop + h + 24, `prefill ${fmt.n(prefill, 2)} s`, { color: 'accent', size: 12, px: true, align: 'left', weight: 'bold' });
      const cold = n / S.prefillRate;
      p.text(x0 + w, yTop + h + 24,
        hit ? `cold would be ${fmt.n(cold, 2)} s — saved ${fmt.n(cold - prefill, 2)} s` : 'no cache hit',
        { color: 'muted', size: 11, px: true, align: 'right' });
    });

    const read = stats([
      { k: 'prompt tokens', v: '—' }, { k: 'reused', v: '—' },
      { k: 'time to first token', v: '—' }, { k: 'vs cold start', v: '—' },
      { k: 'kv cache size', v: '—' },
    ]);

    const turnsBox = el('div', { class: 'col', style: { gap: '4px' } });

    const refresh = () => {
      const n = total(), hit = reusable();
      const ttft = (n - hit) / S.prefillRate;
      const cold = n / S.prefillRate;
      read.set('prompt tokens', n.toLocaleString());
      read.set('reused', `${hit.toLocaleString()} (${fmt.pct(n ? hit / n : 0, 0)})`);
      read.set('time to first token', fmt.n(ttft, 2) + ' s');
      read.set('vs cold start', hit ? `${fmt.n(cold / Math.max(0.001, ttft), 1)}× faster` : '1.0×',
        hit ? 'ok' : 'warn');
      read.set('kv cache size', fmt.n(n * S.bytesPerToken / 2 ** 20, 1) + ' MiB');

      turnsBox.replaceChildren(...S.turns.map((t, i) => {
        const isEdited = i === S.editedTurn;
        return el('div', {
          style: {
            display: 'flex', gap: '10px', alignItems: 'center', fontSize: '12.5px',
            padding: '6px 10px', borderRadius: '8px',
            border: `1px solid ${isEdited ? 'var(--c2)' : 'var(--border)'}`,
            background: 'var(--surface-2)',
          },
        },
          el('span', { class: 'badge', text: t.who }),
          el('span', { style: { flex: '1' }, text: t.label }),
          el('span', { style: { font: '11px ui-monospace,Menlo,monospace', color: 'var(--muted)' }, text: `${t.tokens.toLocaleString()} tok` }),
          el('button', {
            class: 'ghost',
            style: { padding: '3px 8px', fontSize: '11px' },
            onClick: () => { S.editedTurn = isEdited ? -1 : i; refresh(); },
          }, isEdited ? 'un-edit' : 'edit this'));
      }));
      bar.render();
      growth.render();
    };

    const addTurn = () => {
      const i = S.turns.length;
      S.cached = total();                       // the previous prompt is now cached
      S.turns.push({ who: 'user', label: `user: follow-up #${Math.ceil(i / 2)}`, tokens: 120 });
      S.turns.push({ who: 'assistant', label: 'assistant reply', tokens: 300 });
      refresh();
    };

    root.append(card('A conversation, turn by turn',
      'Press <b>send next turn</b>. Each turn resends everything before it — but only the green part has to be computed again.',
      row(
        grow(bar.canvas,
          el('div', { style: { height: '6px' } }),
          legend([
            { c: 'c3', label: 'reused from cache — free' },
            { c: 'c5', label: 'must be prefilled now' },
            { c: 'c2', label: 'edit point: everything after it is invalid' },
          ]),
          el('div', { style: { height: '14px' } }),
          turnsBox,
          el('div', { style: { height: '12px' } }),
          btns(
            button('send next turn ▸', addTurn),
            button('reset', () => { S.turns = START.map((t) => ({ ...t })); S.cached = 0; S.editedTurn = -1; refresh(); }, { ghost: true }))),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          read,
          checkbox({ label: 'prompt cache enabled', checked: true, onChange: (v) => { S.cacheOn = v; refresh(); } }),
          slider({ label: 'prefill speed (tokens/s)', min: 300, max: 12000, step: 100, value: 3000, fmt: (v) => v.toLocaleString(), onInput: (v) => { S.prefillRate = v; refresh(); } }),
          note(`<b>Press "edit this" on the system prompt.</b> Every token after the edit point becomes
                invalid, the reuse collapses to zero, and the turn costs a full cold prefill — no matter how
                small the edit was.`)))));

    /* ---------- why the prefix rule is absolute ---------- */
    root.append(card('Why a cache can only ever be a prefix',
      null,
      row(
        grow(prose(`
          <p>The cached thing is the <b>KV cache</b> from lesson 05: for every token, the keys and values each
          attention layer computed for it. Reusing them is only valid if those tensors would come out identical
          this time.</p>
          <p>Attention at position <i>i</i> reads every position <i>&lt; i</i>. So token 5,000's keys and values
          depend on tokens 0…4,999 — every one of them. Change token 3 and tokens 4 onward are all different
          numbers, even though the text after them is byte-identical.</p>
          <p>That is why the match is a <b>prefix</b> match and nothing cleverer is possible. A cache cannot skip
          a changed paragraph in the middle and resume: there is nothing valid to resume from.</p>`),
          el('div', { class: 'eq', html: 'reuse = length of the longest common <span class="hl">prefix</span> of (old prompt, new prompt)' })),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b>The practical rule.</b> Put everything stable at the <i>front</i> — system prompt, tool
                schemas, few-shot examples, the long document — and everything variable at the <i>back</i>.
                Reverse that and you pay a full prefill on every request.`),
          note(`<b>The classic own-goal.</b> Injecting the current timestamp, a request id, or a shuffled tool
                list into the system prompt changes token ~20 of every request. The cache hit rate goes to zero
                and nobody can see why the app is slow.`),
          note(`<b>Why replies are cacheable too.</b> The assistant's own tokens went through the same forward
                pass, so their K/V are already in the cache — a conversation that only ever appends is a pure
                prefix extension, which is the best case this design has.`)))));

    /* ---------- cost of keeping it ---------- */
    const growth = new Plot({
      width: 460, height: 260, pad: { l: 62, r: 16, t: 16, b: 40 },
      xlim: [0, 128], ylim: [0, 1], xlabel: 'context length (thousands of tokens)', ylabel: 'KV cache (GiB)',
    });
    growth.onDraw((p) => {
      const pts = [];
      for (let k = 0; k <= 128; k += 2) pts.push([k, (k * 1000 * S.bytesPerToken) / 2 ** 30]);
      p.ylim = [0, Math.max(0.5, pts[pts.length - 1][1]) * 1.1];
      p.axes({ xticks: 5, yticks: 5, zero: false });
      p.area(pts, { color: 'accent', alpha: 0.12 });
      p.line(pts, { color: 'accent', width: 2 });
      p.line([[0, S.budgetGiB], [128, S.budgetGiB]], { color: 'c2', width: 1.5, dash: [5, 4] });
      p.text(126, S.budgetGiB, `--prompt-cache-gib ${S.budgetGiB} `, { color: 'c2', size: 10, align: 'right', baseline: 'bottom' });
      const fits = (S.budgetGiB * 2 ** 30) / S.bytesPerToken / 1000;
      p.scatter([[Math.min(128, fits), S.budgetGiB]], { color: 'c2', r: 5, stroke: 'bg' });
      p.text(p.pad.l + 6, p.pad.t + 12,
        `budget holds ~${fmt.n(fits, 0)}k tokens of prefix`, { color: 'muted', size: 11, px: true, align: 'left' });
    });

    root.append(card('Keeping it costs memory, so it is a budget',
      'A cache you cannot afford to keep is not a cache. At Qwen3.5-2B\'s 12,288 bytes per token, prefixes get expensive fast.',
      row(
        el('div', {}, growth.canvas,
          el('div', { style: { height: '8px' } }),
          row(
            slider({ label: 'cache budget (GiB)', min: 1, max: 64, step: 1, value: 16, fmt: (v) => v + ' GiB', onInput: (v) => { S.budgetGiB = v; refresh(); }, width: '200px' }),
            slider({ label: 'bytes per token', min: 2048, max: 262144, step: 2048, value: 12288, fmt: (v) => (v / 1024).toFixed(0) + ' KiB', onInput: (v) => { S.bytesPerToken = v; refresh(); }, width: '200px' }))),
        grow(prose(`
          <p>Lesson 05's formula, applied: this model's six attention layers cost 12 KiB per token of context. A
          128k-token prefix is 1.5 GiB — for <i>one</i> conversation.</p>
          <p>So the cache is a fixed budget with an eviction policy, and prefixes compete for it. Serve many users
          with long shared prompts and the budget is the thing that decides whether they hit or miss.</p>
          <p>Drag <b>bytes per token</b> up to a model with 40 attention layers and no GQA. The same budget now
          holds a fraction of the context — which is exactly why GQA (lesson 06) matters commercially, not just
          mathematically.</p>`),
          note(`<b>Memory is shared with the streams.</b> Every concurrent lane from lesson 12 needs its own KV
                cache, from the same pool. Prompt cache, batch size and context length are three ways of spending
                one budget, and the engine makes you choose.`)))));

    /* ---------- in the engine ---------- */
    root.append(card('In the real engine',
      'TensorFold exposes four distinct layers of "do not recompute this", documented in its README and serve flags.',
      row(
        grow(el('pre', { class: 'code', html:
          `  --prompt-cache-gib GIB     <span class="cm">// in-memory retained prefixes; by default half the</span>\n` +
          `                             <span class="cm">// room left after streams, 16 GiB floor on Metal</span>\n` +
          `  --learn                    <span class="cm">// keep SHARED prefixes on disk, so a new conversation</span>\n` +
          `                             <span class="cm">// resumes a system prompt after a restart</span>\n` +
          `  --learn-gib 32             <span class="cm">// disk budget; least recently used go first</span>\n` +
          `  --learn-conversations      <span class="cm">// keep every conversation's state on disk too</span>\n` +
          `  --recall                   <span class="cm">// draft from a record of token ids seen before,</span>\n` +
          `                             <span class="cm">// so repeated text is verified in a few rounds</span>` }),
          prose(`
            <p>The four layers answer four different questions: <b>prompt cache</b> — is this prefix still in
            memory? <b>learn</b> — was it computed before the server restarted? <b>learn-conversations</b> — was
            <i>this</i> conversation evicted? <b>recall</b> — have I seen this <i>text</i> before, even in a
            different conversation?</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Recall is a cache that feeds the drafter, not the KV cache.</b> It records token ids the
                engine has read and written, then proposes them as drafts — so repeated text is <i>verified</i>
                in a few rounds (lesson 14) rather than re-generated token by token. Replies are unchanged,
                because acceptance is still exact equality.`),
          note(`<b>Disk caches must be invalidated by computation, not by text.</b> An upgrade that changes the
                kernels changes the numbers, so a stored prefix is only reusable if it would compute the same
                bits — the README is explicit that <code class="inline">--learn</code> resumes only after an
                upgrade "that computes the same bits".`),
          note(`<b>Not every model supports every layer.</b> The README lists which families accept
                <code class="inline">--prompt-cache-gib</code> and which refuse <code class="inline">--learn</code>
                at startup. Caching is not a generic service bolted on top; it is per-architecture state
                management.`)))));

    refresh();
  },
};
