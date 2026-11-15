// 02-tokens.js — level 9: text is not characters, it is tokens.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, textbox, legend, fmt, presets } from '../ui.js';
import { BPE, display, pretokenize, toByteString, fromByteString } from '../engine/bpe.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';

const TRAIN_TEXT = DEFAULT_CORPUS + '\n' + DEFAULT_CORPUS.toLowerCase();
const MAX_MERGES = 500;
const SAMPLE = 'The engine reads the weights and the attention layer scores every key.';
// Text the tokenizer has never seen: nothing in it comes from the training corpus.
const HELDOUT = 'Margaret sharpened her pencil, glanced at the rain, and rewrote the fourth paragraph entirely.';

export default {
  id: '02-tokens',
  num: '02',
  group: 'the surface',
  title: 'Text is tokens',
  blurb: 'The model never sees your text. It sees a list of integers. Which integers, and why those, is decided by an algorithm you can watch learn — in about forty lines.',

  render(root) {
    /* ---------- train once, keep every intermediate state ---------- */
    const full = new BPE();
    const history = [];        // {rank, pair, count, token, vocabSize}
    full.train(TRAIN_TEXT, MAX_MERGES, (s) => history.push(s));

    const allMerges = full.merges.slice();
    const atMerges = (n) => {
      const t = new BPE();
      t.merges = allMerges.slice(0, n);
      t.ranks = new Map(t.merges.map(([a, b], i) => [a + '\u0000' + b, i]));
      t.rebuildVocab();
      return t;
    };

    // compression curves: characters per token as the vocabulary grows.
    // Two texts — one the tokenizer was trained on, one it has never seen.
    const curveFor = (text) => {
      const out = [];
      for (let n = 0; n <= allMerges.length; n += 10) {
        const k = atMerges(n).tokenize(text).length;
        out.push([256 + n, text.length / k]);
      }
      return out;
    };
    const curve = curveFor(SAMPLE);
    const curveHeld = curveFor(HELDOUT);

    const state = { n: Math.min(220, allMerges.length), tok: null, text: SAMPLE };
    state.tok = atMerges(state.n);

    /* ---------- live tokenization view ---------- */
    const chips = el('div', { class: 'tokens', style: { lineHeight: '2.1' } });
    const tokStats = stats([
      { k: 'characters', v: '—' }, { k: 'tokens', v: '—' },
      { k: 'chars / token', v: '—' }, { k: 'vocabulary', v: '—' },
      { k: 'bytes', v: '—' },
    ]);

    const renderTokens = () => {
      const toks = state.tok.tokenize(state.text);
      const ids = state.tok.encode(state.text);
      chips.replaceChildren(...toks.map((t, i) => {
        const isByte = t.length === 1;
        return el('span', {
          class: 'tok',
          title: `id ${ids[i]}  ·  ${t.length} byte${t.length > 1 ? 's' : ''}${isByte ? '  ·  raw byte, never merged' : ''}`,
          style: {
            background: `color-mix(in srgb, var(--c${(i % 6) + 1}) 20%, var(--surface-2))`,
            borderColor: isByte ? 'var(--c2)' : 'var(--border)',
          },
        }, display(t));
      }));
      const bytes = toByteString(state.text).length;
      tokStats.set('characters', String(state.text.length));
      tokStats.set('tokens', String(toks.length));
      tokStats.set('chars / token', fmt.n(toks.length ? state.text.length / toks.length : 0, 2));
      tokStats.set('vocabulary', String(state.tok.vocabSize));
      tokStats.set('bytes', String(bytes));
      // round-trip proof
      const back = state.tok.decode(ids);
      rt.textContent = back === state.text ? '✓ decode(encode(text)) === text' : '✗ round-trip failed';
      rt.className = back === state.text ? 'badge ok' : 'badge err';
      mergePlot.render();
      renderMergeList();
    };

    const rt = el('span', { class: 'badge' });

    // Declared before the preset row so onInput can clear the highlight: once
    // you type, no preset is selected any more and saying otherwise would lie.
    let presetRow = null;
    const input = textbox({
      label: 'your text — anything, any language, any emoji',
      rows: 3, value: SAMPLE,
      onInput: (v) => { state.text = v; renderTokens(); if (presetRow) presetRow.select(null); },
    });

    const vocabS = slider({
      label: 'merges learned', min: 0, max: allMerges.length, step: 1, value: state.n,
      fmt: (v) => `${v} → vocab ${256 + v}`,
      onInput: (v) => { state.n = v; state.tok = atMerges(v); renderTokens(); },
    });

    root.append(card('Tokenize anything',
      'Each chip is one token — one integer the model will look up. Red-edged chips are single raw bytes that were never merged into anything bigger. Hover a chip for its id.',
      row(
        grow(input,
          el('div', { style: { height: '12px' } }),
          chips,
          el('div', { style: { height: '12px' } }),
          (presetRow = presets([
            { label: 'a sentence', value: SAMPLE },
            { label: 'emoji + 日本語', value: 'héllo 🌍 日本語 — ok?' },
            { label: 'numbers', value: '3.14159 and 2026 and 1000000' },
            { label: 'nonsense', value: 'zzqqxj Xyzzy-- \t\u0000\u00ff' },
          ].map((p) => ({ ...p, onPick: (v) => { input.set(v); state.text = v; renderTokens(); presetRow.select(v); } })), SAMPLE))),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          vocabS, tokStats, rt,
          note(`Nothing is ever "out of vocabulary". Worst case a character falls back to its raw
                bytes — which is why the vocabulary <b>starts</b> with all 256 of them.`)))));

    /* ---------- watch it learn ---------- */
    const mergeList = el('div', {
      style: {
        font: '12px ui-monospace, Menlo, monospace', maxHeight: '240px', overflowY: 'auto',
        border: '1px solid var(--border)', borderRadius: '9px', padding: '8px 10px', background: 'var(--surface-2)',
      },
    });

    const renderMergeList = () => {
      const shown = history.slice(Math.max(0, state.n - 14), state.n).reverse();
      mergeList.replaceChildren(...(shown.length ? shown : [{ rank: -1 }]).map((m) => m.rank < 0
        ? el('div', { style: { color: 'var(--muted)' }, text: 'no merges yet — every token is a single byte' })
        : el('div', { style: { display: 'flex', gap: '8px', padding: '2px 0' } },
          el('span', { style: { color: 'var(--muted)', minWidth: '34px' }, text: '#' + m.rank }),
          el('span', { style: { color: 'var(--c2)' }, text: display(m.pair[0]) }),
          el('span', { style: { color: 'var(--muted)' }, text: '+' }),
          el('span', { style: { color: 'var(--c5)' }, text: display(m.pair[1]) }),
          el('span', { style: { color: 'var(--muted)' }, text: '→' }),
          el('span', { style: { color: 'var(--c3)', fontWeight: '600' }, text: display(m.token) }),
          el('span', { style: { color: 'var(--muted)', marginLeft: 'auto' }, text: `seen ${m.count}×` }))));
    };

    const mergePlot = new Plot({
      width: 430, height: 250, pad: { l: 48, r: 14, t: 14, b: 40 },
      xlim: [256, 256 + allMerges.length], ylim: [0, 6],
      xlabel: 'vocabulary size', ylabel: 'characters per token',
    });
    mergePlot.onDraw((p) => {
      p.ylim = [0, Math.max(2, Math.max(...curve.map((c) => c[1])) * 1.15)];
      p.axes({ xticks: 5, yticks: 4, zero: false });
      p.area(curve, { color: 'accent', alpha: 0.1 });
      p.line(curve, { color: 'accent', width: 2 });
      p.line(curveHeld, { color: 'c4', width: 2 });

      const v = 256 + state.n;
      p.line([[v, 0], [v, p.ylim[1]]], { color: 'c5', width: 1.5, dash: [4, 4] });
      for (const [text, col] of [[SAMPLE, 'accent'], [HELDOUT, 'c4']]) {
        const ratio = text.length / state.tok.tokenize(text).length;
        p.scatter([[v, ratio]], { color: col, r: 5, stroke: 'bg' });
        p.text(v, ratio, `  ${fmt.n(ratio, 2)}`, { color: col, align: 'left', size: 11 });
      }
    });

    root.append(card('Watch it learn',
      'Training is one rule repeated: find the most frequent adjacent pair, fuse it into a new symbol, write the rule down. Drag <b>merges learned</b> above and watch both panels move.',
      row(
        grow(el('div', { class: 'sub', style: { margin: '0 0 6px' }, text: 'the most recent merges, newest first' }), mergeList),
        el('div', {}, mergePlot.canvas,
          el('div', { style: { height: '8px' } }),
          legend([
            { c: 'accent', label: 'text it was trained on' },
            { c: 'c4', label: 'text it has never seen' },
          ]),
          el('div', { class: 'sub', style: { marginTop: '8px', maxWidth: '420px' },
            html: 'The gap between the two lines is the tokenizer <b>overfitting its corpus</b>. Real vocabularies are trained on trillions of tokens, which is why the gap closes — and why a domain they never saw (rare languages, DNA, base64) still tokenizes badly.' }))))); 

    root.append(card('The tradeoff you are sliding along', null, row(
      grow(prose(`
        <p><b>One token per character</b> (vocab ≈ 256): tiny embedding table, but every sentence is
        hundreds of tokens. Attention cost grows with the square of that number, so you pay dearly.</p>
        <p><b>One token per word</b> (vocab ≈ 500k): short sequences, but every typo, name, number and
        foreign word is unrepresentable — and the output layer becomes enormous.</p>
        <p><b>BPE</b> sits between: frequent strings become single tokens, rare ones decompose. You chose
        the position on that curve when you chose the vocabulary size.</p>
        <p>Watch the curve flatten. Past a point, more vocabulary buys almost no compression — but every
        extra token costs another row in the output matrix, forever.</p>
      `)),
      el('div', { class: 'col', style: { flex: '1 1 320px' } },
        note(`<b>Why "␣the" is one token.</b> The space is glued to the word, so " the" and "the"
              are different tokens. That is why a stray leading space can change a model's output.`),
        note(`<b>Why numbers behave oddly.</b> "1000000" may split into chunks like "100" "000" "0"
              depending on what the training text contained. Arithmetic is being done on a representation
              that does not know about place value.`),
        note(`<b>Why the same prompt costs different amounts in different languages.</b> Vocabularies
              are trained mostly on English. Try the 日本語 button and watch chars-per-token collapse.`)))));

    /* ---------- real engine ---------- */
    root.append(card('In the real engine', null, row(
      grow(prose(`
        <p>Qwen3.5-2B, the small checkpoint TensorFold serves on Metal, pins its geometry at compile time:</p>`),
        el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/families/qwen3_5/config.zig:7-21</span>\n` +
          `pub const hidden = 2048;\n` +
          `pub const vocab = <b>248320</b>;\n` +
          `pub const layers = 24;\n` +
          `pub const query_heads = 8;\n` +
          `pub const kv_heads = 2;\n` +
          `pub const head_dim = 256;` }),
        prose(`
        <p>248,320 tokens. Your tokenizer above has a few hundred. Everything else is identical in kind:
        a byte-level merge table, applied greedily, producing integers.</p>`)),
      el('div', { class: 'col', style: { flex: '1 1 320px' } },
        note(`<b>The number that will matter later.</b> The final layer must turn a 2048-wide vector into
              one score per token: a 2048 × 248,320 matrix. That is <b>508 million</b> parameters —
              a quarter of this 2B model — spent purely on saying "which token next".`),
        note(`<b>Decoding is harder than encoding.</b> A streaming server cannot just decode each token
              to text: one character can span several tokens, and a token can end mid-UTF-8-sequence.
              Emit too eagerly and you print <code class="inline">\\ufffd</code>. That is why engines buffer
              partial bytes until they form a valid character.`)))));

    renderTokens();
  },
};
