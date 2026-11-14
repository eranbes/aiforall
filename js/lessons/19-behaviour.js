// 19-behaviour.js — the three stages between your API call and the sampler
// that explain most "why is the model doing that?" bugs.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, textbox, checkbox, fmt } from '../ui.js';
import { probe, tokenize, pieces } from '../engine/live.js';
import { sample, applyPenalties } from '../engine/sample.js';
import { NGram, tokenize as words } from '../engine/ngram.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';

const SPECIAL = /^<\|.*\|>$|^<\/?think>$/;
const show = (s) => (s === '\n' ? '⏎' : s === '\n\n' ? '⏎⏎' : s.replace(/ /g, '␣'));

export default {
  id: '19-behaviour',
  num: '19',
  group: 'the real thing',
  title: 'Why your app misbehaves',
  blurb: 'Three stages sit between your API call and the sampler you built: a template that turns messages into one string, penalties that edit the logits before anything else, and a grammar that can forbid tokens outright. Each one quietly changes output, and none of them is the model.',

  render(root) {
    /* ================= 1. the chat template ================= */
    const T = { connected: false, model: null, tokens: [], pieces: [], busy: false, err: null };

    const msgBox = textbox({
      label: 'messages (JSON)', rows: 5,
      value: JSON.stringify([
        { role: 'system', content: 'Be terse.' },
        { role: 'user', content: 'hi' },
      ], null, 2),
    });

    const chips = el('div', { class: 'tokens', style: { lineHeight: '2.2' } });
    const tokStats = stats([
      { k: 'tokens', v: '—' }, { k: 'specials', v: '—' },
      { k: 'your text', v: '—' }, { k: 'template overhead', v: '—' },
    ]);
    const tokErr = el('div', { class: 'sub' });

    const render = async () => {
      if (!T.connected || T.busy) return;
      let messages;
      try { messages = JSON.parse(msgBox.get()); } catch (e) { tokErr.textContent = 'invalid JSON: ' + e.message; return; }
      T.busy = true; tokErr.textContent = 'rendering…';
      try {
        const r = await tokenize(T.model, messages);
        T.tokens = r.tokens;
        T.pieces = await pieces(T.model, r.tokens);
        const specials = T.pieces.filter((p) => SPECIAL.test(p)).length;
        const yourChars = messages.map((m) => String(m.content || '')).join('').length;
        tokStats.set('tokens', String(r.count));
        tokStats.set('specials', String(specials));
        tokStats.set('your text', yourChars + ' chars');
        tokStats.set('template overhead', fmt.pct(specials / Math.max(1, r.count), 0));
        chips.replaceChildren(...T.pieces.map((p, i) => {
          const sp = SPECIAL.test(p);
          return el('span', {
            class: 'tok',
            title: `id ${T.tokens[i]}`,
            style: {
              background: sp ? 'color-mix(in srgb, var(--c4) 22%, var(--surface-2))' : 'var(--surface-2)',
              borderColor: sp ? 'var(--c4)' : 'var(--border)',
              fontWeight: sp ? '600' : '400',
            },
          }, show(p));
        }));
        tokErr.textContent = '';
      } catch (e) {
        tokErr.textContent = String(e.message || e);
      }
      T.busy = false;
    };

    const connect = async () => {
      const r = await probe();
      T.connected = r.ok;
      if (r.ok && r.models.length) T.model = r.models[0].id;
      tokErr.textContent = r.ok ? '' : 'no engine on :8080 — this card needs the live server from lesson 18.';
      if (r.ok) render();
    };

    root.append(card('Your messages are not what the model sees',
      'The engine renders them through the checkpoint\'s own Jinja template into <b>one flat string</b>, then tokenizes that. These are the real ids from the running server — purple chips are control tokens you never wrote.',
      row(
        grow(msgBox,
          el('div', { style: { height: '8px' } }),
          btns(
            button('render ▸', render),
            button('drop the system message', () => {
              msgBox.set(JSON.stringify([{ role: 'user', content: 'hi' }], null, 2)); render();
            }, { ghost: true }),
            button('add an assistant turn', () => {
              msgBox.set(JSON.stringify([
                { role: 'system', content: 'Be terse.' },
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'Hello.' },
                { role: 'user', content: 'again' },
              ], null, 2)); render();
            }, { ghost: true })),
          el('div', { style: { height: '10px' } }),
          chips,
          tokErr),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          tokStats,
          note(`<b>Look at the tail.</b> The template opens a <code class="inline">&lt;think&gt;</code> block,
                and because this server was started with <code class="inline">--no-thinking</code> the engine
                appends <code class="inline">&lt;/think&gt;</code> to close it. "Thinking off" is a
                <b>prompt suffix</b>, not a model switch — <code class="inline">prompt.zig:36-78</code> detects
                a rendered prompt ending in a bare <code class="inline">&lt;think&gt;</code> and closes it.`),
          note(`<b>Why this is the top source of bugs.</b> Two token ids of difference at the front of the
                prompt — a role renamed, a stray newline, a timestamp injected — invalidates the entire prefix
                cache (lesson 13) and can change the reply. You did not change your text; you changed the
                string.`)))));

    /* ================= 2. penalties ================= */
    const model = new NGram(3).train(DEFAULT_CORPUS);
    const seedIds = words('The engine reads the').map((t) => model.index.get(t)).filter((i) => i !== undefined);
    const P = {
      reply: [],                 // tokens the model has already emitted
      repetition: 1, presence: 0, frequency: 0,
      temperature: 0,            // greedy by default: penalties still bite
    };

    const base = () => model.logits(model.context(seedIds.concat(P.reply))).logits;

    const penStats = stats([
      { k: 'reply length', v: '0' }, { k: 'distinct tokens', v: '0' },
      { k: 'greedy pick', v: '—' }, { k: 'changed by penalties', v: '—' },
    ]);

    const penPlot = new Plot({
      width: 660, height: 300, pad: { l: 48, r: 12, t: 24, b: 64 },
      xlim: [0, 12], ylim: [0, 1], ylabel: 'logit',
    });

    penPlot.onDraw((p) => {
      const raw = base();
      const after = Float64Array.from(raw);
      applyPenalties(after, {
        history: seedIds.concat(P.reply), reply: P.reply,
        repetition: P.repetition, presence: P.presence, frequency: P.frequency,
      });
      const order = Array.from(raw.keys()).sort((a, b) => raw[b] - raw[a]).slice(0, 12);
      const rawV = order.map((i) => raw[i]);
      const aftV = order.map((i) => after[i]);
      const lo = Math.min(...rawV, ...aftV), hi = Math.max(...rawV, ...aftV);
      p.ylim = [lo - 0.6, hi + 0.6];
      p.axes({ xticks: 0, yticks: 5, zero: true });
      p.bars(rawV, { color: 'border', width: 0.74, labels: order.map((i) => show(model.vocab[i])), baseline: lo - 0.6 });
      p.bars(aftV, {
        color: order.map((i) => (P.reply.includes(i) ? 'c2' : 'c3')),
        width: 0.4, baseline: lo - 0.6,
      });
      // which token greedy picks, before and after
      const gi = order[0];
      let bi = order[0];
      for (const i of order) if (after[i] > after[bi]) bi = i;
      p.text(p.pad.l, 12,
        `greedy before: "${show(model.vocab[gi])}"   →   after penalties: "${show(model.vocab[bi])}"`,
        { color: gi === bi ? 'muted' : 'c5', size: 11, px: true, align: 'left' });
      penStats.set('greedy pick', show(model.vocab[bi]), gi === bi ? null : 'warn');
      penStats.set('changed by penalties', gi === bi ? 'no' : 'yes', gi === bi ? null : 'warn');
    });

    const replyBox = el('div', { class: 'tokens' });
    const refreshPen = () => {
      replyBox.replaceChildren(...(P.reply.length
        ? P.reply.map((i) => el('span', { class: 'tok', style: { borderColor: 'var(--c2)' } }, show(model.vocab[i])))
        : [el('span', { style: { fontSize: '12px', color: 'var(--muted)' }, text: 'nothing emitted yet' })]));
      penStats.set('reply length', String(P.reply.length));
      penStats.set('distinct tokens', String(new Set(P.reply).size));
      penPlot.render();
    };

    const emitGreedy = () => {
      const raw = base();
      const after = Float64Array.from(raw);
      applyPenalties(after, {
        history: seedIds.concat(P.reply), reply: P.reply,
        repetition: P.repetition, presence: P.presence, frequency: P.frequency,
      });
      const res = sample(after, { temperature: P.temperature, seed: 5, position: P.reply.length });
      P.reply.push(res.chosen);
      refreshPen();
    };

    root.append(card('Penalties run before everything else',
      'Not after the sampler, not inside it — on the <b>raw logits</b>, before temperature and before any cut. Grey bars are the model\'s opinion; coloured bars are what the sampler will actually see. Red means the token is already in the reply.',
      row(
        grow(penPlot.canvas,
          el('div', { style: { height: '8px' } }),
          legend([
            { c: 'border', label: 'raw logit' },
            { c: 'c3', label: 'after penalties — untouched' },
            { c: 'c2', label: 'after penalties — already emitted' },
          ]),
          el('div', { style: { height: '12px' } }),
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'the reply so far (what presence/frequency count)' }),
          replyBox,
          el('div', { style: { height: '10px' } }),
          btns(
            button('emit one token ▸', emitGreedy),
            button('emit 10', () => { for (let i = 0; i < 10; i++) emitGreedy(); }, { ghost: true }),
            button('clear reply', () => { P.reply = []; refreshPen(); }, { ghost: true }))),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          penStats,
          slider({ label: 'repetition <span class="badge">HF</span>', min: 0.5, max: 2, step: 0.05, value: 1, fmt: (v) => (v === 1 ? 'off (1.00)' : v.toFixed(2)), onInput: (v) => { P.repetition = v; penPlot.render(); } }),
          slider({ label: 'presence <span class="badge">OpenAI</span>', min: 0, max: 3, step: 0.05, value: 0, fmt: (v) => (v === 0 ? 'off' : v.toFixed(2)), onInput: (v) => { P.presence = v; penPlot.render(); } }),
          slider({ label: 'frequency <span class="badge">OpenAI</span>', min: 0, max: 2, step: 0.05, value: 0, fmt: (v) => (v === 0 ? 'off' : v.toFixed(2)), onInput: (v) => { P.frequency = v; penPlot.render(); } }),
          checkbox({ label: 'sample instead of greedy', onChange: (v) => { P.temperature = v ? 0.9 : 0; } }),
          note(`<b>Press "emit 10" with everything off</b> — the trigram loops. Now raise <b>frequency</b> to
                0.4 and clear the reply. The loop breaks, because each repeat costs the token
                <code class="inline">presence + frequency × count</code> more every time.`),
          note(`<b>They apply at temperature 0.</b> The engine's own comment says so: <i>"a reply's token loses
                presence + frequency * its count, <b>greedy too</b>"</i>. "Deterministic" does not mean
                "unmodified".`)))));

    root.append(card('Two penalty families that are not the same operation', null, row(
      grow(el('pre', { class: 'code', html:
        `<span class="cm">// zig/src/hip_serve/draw.zig:190-216 — process()</span>\n` +
        `if (r.repetition != 1) {\n` +
        `    for (r.history) |id| counts[id] = 1;          <span class="cm">// prompt + reply, counted ONCE</span>\n` +
        `    for (values, counts) |*v, seen| if (seen != 0) {\n` +
        `        v.* = if (v.* &lt; 0) v.* * r.repetition   <span class="cm">// negative: pushed down</span>\n` +
        `              else v.* / r.repetition;            <span class="cm">// positive: pulled toward 0</span>\n` +
        `    };\n` +
        `}\n` +
        `if (r.penalties.active()) {\n` +
        `    for (r.reply) |id| counts[id] += 1;           <span class="cm">// REPLY only, with multiplicity</span>\n` +
        `    for (values, counts) |*v, count| if (count != 0) {\n` +
        `        v.* -= r.penalties.of(count);             <span class="cm">// presence + frequency*count</span>\n` +
        `    };\n` +
        `}` })),
      el('div', { class: 'col', style: { flex: '1 1 320px' } },
        note(`<b>repetition is multiplicative and sign-dependent.</b> Dividing a positive logit by 1.2 moves it
              toward zero; multiplying a negative one by 1.2 moves it further away. The same knob therefore
              behaves differently depending on whether the model liked the token at all.`),
        note(`<b>It also counts your prompt.</b> <code class="inline">history</code> is prompt + reply, so a
              high repetition penalty discourages the model from quoting the document you just gave it —
              a common and baffling failure in summarisation.`),
        note(`<b>presence and frequency count only the reply</b>, with multiplicity. presence is a one-off
              charge for having appeared at all; frequency scales with how often. They are additive, so their
              effect does not depend on the logit's sign.`)))));

    /* ================= 3. constrained decoding ================= */
    const VOCAB = ['{', '}', '"', 'name', 'age', ':', ',', '42', '"Ada"', ' ', 'the', 'hello'];
    const G = { emitted: [], on: true };

    /**
     * A tiny JSON-object grammar as an explicit state machine: which vocabulary
     * entries may legally come next? Real engines compile a full grammar to a
     * mask per position; this is the same idea at nine states.
     */
    const legal = (seq) => {
      let st = 'start';
      // a state names what the grammar expects NEXT; consuming that token moves on
      for (const t of seq) {
        switch (st) {
          case 'start': st = 'brace'; break;                        // consumed '{'
          case 'brace': st = t === '}' ? 'done' : 'keyword'; break; // '"' opens a key, '}' ends an empty object
          case 'keyword': st = 'keyclose'; break;                   // consumed name | age
          case 'keyclose': st = 'colon'; break;                     // consumed the key's closing '"'
          case 'colon': st = 'value'; break;                        // consumed ':'
          case 'value': st = 'after'; break;                        // consumed 42 | "Ada"
          case 'after': st = t === ',' ? 'comma' : 'done'; break;   // ',' continues, '}' ends
          case 'comma': st = 'keyword'; break;                      // consumed '"'
          default: break;
        }
      }
      switch (st) {
        case 'start': return new Set(['{']);
        case 'brace': return new Set(['"', '}']);
        case 'keyword': return new Set(['name', 'age']);
        case 'keyclose': return new Set(['"']);
        case 'colon': return new Set([':']);
        case 'value': return new Set(['42', '"Ada"']);
        case 'after': return new Set([',', '}']);
        case 'comma': return new Set(['"']);
        default: return new Set();                                  // complete
      }
    };

    const gPlot = new Plot({
      width: 660, height: 250, pad: { l: 48, r: 12, t: 24, b: 56 },
      xlim: [0, VOCAB.length], ylim: [0, 1], ylabel: 'logit',
    });
    const gSeq = el('div', { class: 'tokens' });
    const gStats = stats([{ k: 'legal next', v: '—' }, { k: 'masked out', v: '—' }, { k: 'string so far', v: '—' }]);

    // a deterministic "model opinion" that likes prose, not JSON — so the mask has work to do
    const OPINION = [0.2, -0.4, 0.1, 0.3, 0.0, -0.2, -0.1, 0.4, 0.2, 1.1, 1.6, 1.3];

    gPlot.onDraw((p) => {
      const ok = legal(G.emitted);
      const vals = VOCAB.map((t, i) => (G.on && !ok.has(t) ? -8 : OPINION[i]));
      p.ylim = [-9, 2.4];
      p.axes({ xticks: 0, yticks: 5, zero: true });
      p.bars(OPINION, { color: 'border', width: 0.74, labels: VOCAB.map(show), baseline: -9 });
      p.bars(vals, {
        color: VOCAB.map((t) => (G.on && !ok.has(t) ? 'c2' : 'c3')),
        width: 0.42, baseline: -9,
      });
      p.text(p.pad.l, 12,
        G.on ? 'grammar on: illegal tokens forced to −∞ before the sampler sees them' : 'grammar off: the model may write anything',
        { color: G.on ? 'c3' : 'muted', size: 11, px: true, align: 'left' });
      gStats.set('legal next', [...ok].map(show).join(' ') || '— (complete)');
      gStats.set('masked out', `${VOCAB.length - ok.size} of ${VOCAB.length}`);
      gStats.set('string so far', G.emitted.join('') || '∅');
    });

    const stepGrammar = () => {
      const ok = legal(G.emitted);
      const vals = VOCAB.map((t, i) => (G.on && !ok.has(t) ? -1e9 : OPINION[i]));
      const res = sample(Float64Array.from(vals), { temperature: 0.7, seed: 3, position: G.emitted.length });
      if (G.on && !ok.size) return;
      G.emitted.push(VOCAB[res.chosen]);
      gSeq.replaceChildren(...G.emitted.map((t) => el('span', { class: 'tok', style: { borderColor: 'var(--c3)' } }, show(t))));
      gPlot.render();
    };

    root.append(card('A grammar does not persuade the model — it deletes options',
      'Constrained decoding ("JSON mode", required tool calls) masks every token the grammar forbids to −∞ <b>before</b> the sampler runs. The model\'s preference is irrelevant for illegal tokens: they cannot be drawn at all.',
      row(
        grow(gPlot.canvas,
          el('div', { style: { height: '8px' } }),
          legend([{ c: 'c3', label: 'legal — can be drawn' }, { c: 'c2', label: 'forced to −∞' }, { c: 'border', label: 'what the model wanted' }]),
          el('div', { style: { height: '10px' } }),
          gSeq,
          el('div', { style: { height: '10px' } }),
          btns(
            button('next token ▸', stepGrammar),
            button('emit 8', () => { for (let i = 0; i < 8; i++) stepGrammar(); }, { ghost: true }),
            button('reset', () => { G.emitted = []; gSeq.replaceChildren(); gPlot.render(); }, { ghost: true }),
            checkbox({ label: 'grammar on', checked: true, onChange: (v) => { G.on = v; gPlot.render(); } }))),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          gStats,
          note(`<b>Turn the grammar off and press "emit 8".</b> This toy model prefers prose — it writes
                <code class="inline">the hello the…</code>. Turn it on and the same preferences produce valid
                JSON, because everything else was removed.`),
          note(`<b>What it costs.</b> Someone must compute the legal set at every step, from the grammar and
                the text so far. TensorFold delegates that to <code class="inline">libtfgrammar.dylib</code>
                (xgrammar) — the server we built <i>warned at startup</i> that it is missing, and answers
                HTTP 400 for structured output without it.`),
          note(`<b>And what it does not fix.</b> A grammar guarantees the output <i>parses</i>. It says nothing
                about whether the values are true. Valid JSON full of wrong numbers is the most common way this
                is misunderstood.`)))));

    refreshPen();
    connect();
  },
};
