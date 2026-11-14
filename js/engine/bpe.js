// bpe.js — byte-level byte-pair encoding, the algorithm behind nearly every
// production tokenizer (GPT-2/3/4, Llama, Qwen, Mistral).
//
// Two ideas, and that is the whole thing:
//   1. Work on BYTES, not characters. Then no input is ever out of vocabulary:
//      any text in any language, any emoji, any corrupt byte, is representable.
//   2. Repeatedly merge the most frequent adjacent pair into a new symbol.
//      Frequent sequences ("ing", " the") become single tokens; rare ones stay in pieces.

const PRETOK = /'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu;

const enc = new TextEncoder();
const dec = new TextDecoder('utf-8', { fatal: false });

/** UTF-8 bytes -> a latin1 string, one char per byte. Keeps everything a string. */
export function toByteString(s) {
  const b = enc.encode(s);
  let out = '';
  for (let i = 0; i < b.length; i++) out += String.fromCharCode(b[i]);
  return out;
}
/** latin1 byte-string -> real text. */
export function fromByteString(s) {
  const b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i) & 0xff;
  return dec.decode(b);
}

/** Human-readable form of a token: spaces shown as ␣, newlines as ⏎. */
export function display(tokenByteString) {
  const t = fromByteString(tokenByteString);
  return t.replace(/ /g, '␣').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
}

export function pretokenize(text) {
  return text.match(PRETOK) || [];
}

export class BPE {
  constructor() {
    this.merges = [];          // [[a, b], ...] in learned order
    this.ranks = new Map();    // "a\u0000b" -> rank
    this.vocab = [];           // token byte-strings
    this.vocabIndex = new Map();
  }

  /** Vocabulary = 256 raw bytes + one entry per merge. */
  rebuildVocab() {
    this.vocab = [];
    for (let i = 0; i < 256; i++) this.vocab.push(String.fromCharCode(i));
    for (const [a, b] of this.merges) this.vocab.push(a + b);
    this.vocabIndex = new Map(this.vocab.map((t, i) => [t, i]));
  }

  /**
   * Learn `numMerges` merges from `text`.
   * @param {(step:{rank:number, pair:[string,string], count:number, token:string, vocabSize:number})=>void} [onStep]
   */
  train(text, numMerges, onStep) {
    // words: each pre-token as an array of single-byte symbols, with a count
    const freq = new Map();
    for (const w of pretokenize(text)) {
      const b = toByteString(w);
      freq.set(b, (freq.get(b) || 0) + 1);
    }
    const words = [...freq.entries()].map(([w, c]) => ({ syms: [...w], count: c }));

    this.merges = [];
    this.ranks = new Map();

    for (let step = 0; step < numMerges; step++) {
      // count every adjacent pair
      const pairs = new Map();
      for (const { syms, count } of words) {
        for (let i = 0; i < syms.length - 1; i++) {
          const key = syms[i] + '\u0000' + syms[i + 1];
          pairs.set(key, (pairs.get(key) || 0) + count);
        }
      }
      if (!pairs.size) break;

      let bestKey = null, bestCount = 0;
      for (const [k, v] of pairs) if (v > bestCount) { bestCount = v; bestKey = k; }
      if (bestCount < 2) break;                       // nothing worth merging

      const [a, b] = bestKey.split('\u0000');
      const merged = a + b;

      for (const w of words) {
        const s = w.syms;
        if (s.length < 2) continue;
        const out = [];
        for (let i = 0; i < s.length; i++) {
          if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { out.push(merged); i++; }
          else out.push(s[i]);
        }
        w.syms = out;
      }

      this.ranks.set(bestKey, this.merges.length);
      this.merges.push([a, b]);
      onStep && onStep({
        rank: this.merges.length - 1, pair: [a, b], count: bestCount,
        token: merged, vocabSize: 256 + this.merges.length,
      });
    }
    this.rebuildVocab();
    return this;
  }

  /** Apply the learned merges to one pre-token, lowest rank first (greedy, deterministic). */
  _mergeWord(byteStr) {
    let syms = [...byteStr];
    if (syms.length < 2) return syms;
    for (;;) {
      let bestRank = Infinity, bestI = -1;
      for (let i = 0; i < syms.length - 1; i++) {
        const r = this.ranks.get(syms[i] + '\u0000' + syms[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
      }
      if (bestI < 0) break;
      syms = syms.slice(0, bestI).concat(syms[bestI] + syms[bestI + 1], syms.slice(bestI + 2));
    }
    return syms;
  }

  /** Text -> token byte-strings. */
  tokenize(text) {
    const out = [];
    for (const w of pretokenize(text)) out.push(...this._mergeWord(toByteString(w)));
    return out;
  }

  /** Text -> token ids. */
  encode(text) {
    return this.tokenize(text).map((t) => {
      const id = this.vocabIndex.get(t);
      return id === undefined ? -1 : id;
    });
  }

  decode(ids) {
    return fromByteString(ids.map((i) => this.vocab[i] ?? '').join(''));
  }

  get vocabSize() { return this.vocab.length; }
}

/** Characters per token: the compression ratio that decides how much your prompt costs. */
export function compression(text, tokens) {
  return tokens.length ? text.length / tokens.length : 0;
}
