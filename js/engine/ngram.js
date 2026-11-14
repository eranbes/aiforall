// ngram.js — an interpolated n-gram language model.
//
// This is a real language model: it produces a genuine probability distribution over a
// vocabulary, conditioned on context. It is also the weakest one worth building, which is
// the point — the sampler on top of it is EXACTLY the sampler a 30B transformer uses.
// Swap the thing that produces logits; everything downstream is unchanged.

const TOKEN_RE = /[A-Za-z']+|[0-9]+|\n|[^\sA-Za-z0-9']/g;

export function tokenize(text) {
  return text.match(TOKEN_RE) || [];
}

/** Join tokens back into text with English-ish spacing rules. */
export function detokenize(toks) {
  let out = '';
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (t === '\n') { out += '\n'; continue; }
    const noSpace = i === 0 || out.endsWith('\n') || /^[.,;:!?)'\]]$/.test(t) || /[([]$/.test(out.trimEnd().slice(-1));
    out += (noSpace ? '' : ' ') + t;
  }
  return out;
}

export class NGram {
  /** @param {number} order highest context length (3 = trigram) */
  constructor(order = 3) {
    this.order = order;
    this.vocab = [];
    this.index = new Map();      // token string -> id
    this.tables = [];            // tables[k] : Map(contextKey -> Map(id -> count)) for context length k
    this.unigram = null;         // Float32Array of counts
    this.total = 0;
  }

  id(tok) {
    let i = this.index.get(tok);
    if (i === undefined) {
      i = this.vocab.length;
      this.vocab.push(tok);
      this.index.set(tok, i);
    }
    return i;
  }

  train(text) {
    const toks = tokenize(text);
    const ids = toks.map((t) => this.id(t));
    this.tables = Array.from({ length: this.order }, () => new Map());
    const counts = new Float32Array(this.vocab.length);
    for (const i of ids) counts[i]++;
    this.unigram = counts;
    this.total = ids.length;

    for (let k = 1; k <= this.order - 1; k++) {
      const table = this.tables[k];
      for (let i = k; i < ids.length; i++) {
        const key = ids.slice(i - k, i).join(',');
        let m = table.get(key);
        if (!m) { m = new Map(); table.set(key, m); }
        m.set(ids[i], (m.get(ids[i]) || 0) + 1);
      }
    }
    this.ids = ids;
    return this;
  }

  get vocabSize() { return this.vocab.length; }

  /**
   * Interpolated distribution over the whole vocabulary given a context of token ids.
   * Longer contexts get more weight when they have been seen at all.
   * @returns {{probs: Float32Array, used: number}} used = longest context length that matched
   */
  distribution(ctx) {
    const V = this.vocab.length;
    const probs = new Float32Array(V);
    const WEIGHTS = [0.03, 0.17, 0.8];   // unigram, bigram-context, trigram-context
    let used = 0;
    let acc = 0;

    // order 0: unigram
    for (let i = 0; i < V; i++) probs[i] = WEIGHTS[0] * (this.unigram[i] / this.total);
    acc += WEIGHTS[0];

    for (let k = 1; k <= this.order - 1; k++) {
      if (ctx.length < k) break;
      const key = ctx.slice(ctx.length - k).join(',');
      const m = this.tables[k].get(key);
      if (!m) continue;
      let tot = 0;
      for (const v of m.values()) tot += v;
      const w = WEIGHTS[Math.min(k, WEIGHTS.length - 1)];
      for (const [id, cnt] of m) probs[id] += w * (cnt / tot);
      acc += w;
      used = k;
    }
    for (let i = 0; i < V; i++) probs[i] /= acc;
    return { probs, used };
  }

  /**
   * Logits, so the sampler sees the same kind of input a transformer produces.
   * A transformer's logits are unnormalised scores; log p is exactly that, up to a constant.
   */
  logits(ctx) {
    const { probs, used } = this.distribution(ctx);
    const out = new Float32Array(probs.length);
    for (let i = 0; i < probs.length; i++) out[i] = Math.log(Math.max(1e-12, probs[i]));
    return { logits: out, used };
  }

  /** Context ids for the tail of a token-id array. */
  context(ids) { return ids.slice(Math.max(0, ids.length - (this.order - 1))); }
}
