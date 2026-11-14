// trainer.js — a small neural language model with hand-written gradients.
//
// Architecture (the smallest thing that is honestly a neural LM):
//   ids[ctx]  --embedding lookup-->  x  (ctx*emb)
//   h      = tanh(x @ W1 + b1)       (hid)
//   logits = h @ W2 + b2             (vocab)
//   loss   = -log softmax(logits)[target]
//
// Every gradient below is written out by hand. Nothing is hidden in a framework,
// so you can check each line against the chain rule — and the lesson does exactly
// that, numerically, on screen.

import { mulberry32 } from './tensor.js';

/**
 * Softmax in f64. The shared tensor.js softmax returns Float32Array, whose ~1e-7
 * relative error is fine for display but fatal for finite-difference gradient
 * checking: dividing that noise by 2*eps swamps the gradient being measured.
 */
function softmax64(z, T = 1) {
  const t = Math.max(1e-6, T);
  const out = new Float64Array(z.length);
  let m = -Infinity;
  for (let i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
  let s = 0;
  for (let i = 0; i < z.length; i++) { out[i] = Math.exp((z[i] - m) / t); s += out[i]; }
  for (let i = 0; i < z.length; i++) out[i] /= s;
  return out;
}

export class CharLM {
  constructor({ text, ctx = 3, emb = 12, hid = 64, seed = 1 }) {
    const chars = [...new Set(text)].sort();
    this.vocab = chars;
    this.index = new Map(chars.map((c, i) => [c, i]));
    this.V = chars.length;
    this.ctx = ctx;
    this.emb = emb;
    this.hid = hid;
    this.rng = mulberry32(seed);

    this.data = [...text].map((c) => this.index.get(c));

    const n = (a, b, s) => Float64Array.from({ length: a * b }, () => (this.rng() * 2 - 1) * s);
    this.E = n(this.V, emb, 0.8);                    // embedding table
    this.W1 = n(ctx * emb, hid, Math.sqrt(1 / (ctx * emb)));
    this.b1 = new Float64Array(hid);
    this.W2 = n(hid, this.V, Math.sqrt(1 / hid));
    this.b2 = new Float64Array(this.V);
    this.steps = 0;
  }

  /** Forward pass for one example. Returns every intermediate, for backward and for display. */
  forward(ids) {
    const { ctx, emb, hid, V } = this;
    const x = new Float64Array(ctx * emb);
    for (let t = 0; t < ctx; t++) {
      const id = ids[t];
      for (let e = 0; e < emb; e++) x[t * emb + e] = this.E[id * emb + e];
    }
    const z = new Float64Array(hid);
    for (let j = 0; j < hid; j++) {
      let s = this.b1[j];
      for (let i = 0; i < x.length; i++) s += x[i] * this.W1[i * hid + j];
      z[j] = s;
    }
    const h = new Float64Array(hid);
    for (let j = 0; j < hid; j++) h[j] = Math.tanh(z[j]);

    const logits = new Float64Array(V);
    for (let k = 0; k < V; k++) {
      let s = this.b2[k];
      for (let j = 0; j < hid; j++) s += h[j] * this.W2[j * V + k];
      logits[k] = s;
    }
    return { x, z, h, logits };
  }

  loss(ids, target) {
    const { logits } = this.forward(ids);
    const p = softmax64(logits);
    return -Math.log(Math.max(1e-12, p[target]));
  }

  /**
   * One SGD step over a batch. Returns the mean loss BEFORE the update.
   * Gradients accumulate across the batch, then are applied once.
   */
  step(batch, lr) {
    const { ctx, emb, hid, V } = this;
    const gE = new Float64Array(this.E.length);
    const gW1 = new Float64Array(this.W1.length);
    const gb1 = new Float64Array(hid);
    const gW2 = new Float64Array(this.W2.length);
    const gb2 = new Float64Array(V);
    let total = 0;

    for (const [ids, target] of batch) {
      const { x, h, logits } = this.forward(ids);
      const p = softmax64(logits);
      total += -Math.log(Math.max(1e-12, p[target]));

      // dL/dlogits = p - onehot(target)   <- the whole reason softmax+cross-entropy are paired
      const dlogits = new Float64Array(V);
      for (let k = 0; k < V; k++) dlogits[k] = p[k];
      dlogits[target] -= 1;

      // logits = h @ W2 + b2
      const dh = new Float64Array(hid);
      for (let j = 0; j < hid; j++) {
        const hj = h[j];
        let acc = 0;
        for (let k = 0; k < V; k++) {
          gW2[j * V + k] += hj * dlogits[k];
          acc += this.W2[j * V + k] * dlogits[k];
        }
        dh[j] = acc;
      }
      for (let k = 0; k < V; k++) gb2[k] += dlogits[k];

      // h = tanh(z)  =>  dz = dh * (1 - h^2)
      const dz = new Float64Array(hid);
      for (let j = 0; j < hid; j++) dz[j] = dh[j] * (1 - h[j] * h[j]);
      for (let j = 0; j < hid; j++) gb1[j] += dz[j];

      // z = x @ W1 + b1
      const dx = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) {
        const xi = x[i];
        let acc = 0;
        for (let j = 0; j < hid; j++) {
          gW1[i * hid + j] += xi * dz[j];
          acc += this.W1[i * hid + j] * dz[j];
        }
        dx[i] = acc;
      }

      // x was a lookup: route each slice of dx back to the row it came from
      for (let t = 0; t < ctx; t++) {
        const id = ids[t];
        for (let e = 0; e < emb; e++) gE[id * emb + e] += dx[t * emb + e];
      }
    }

    const s = lr / batch.length;
    for (let i = 0; i < this.E.length; i++) this.E[i] -= s * gE[i];
    for (let i = 0; i < this.W1.length; i++) this.W1[i] -= s * gW1[i];
    for (let i = 0; i < this.b1.length; i++) this.b1[i] -= s * gb1[i];
    for (let i = 0; i < this.W2.length; i++) this.W2[i] -= s * gW2[i];
    for (let i = 0; i < this.b2.length; i++) this.b2[i] -= s * gb2[i];
    this.steps++;
    return total / batch.length;
  }

  /** Analytic gradient of a single example w.r.t. one W2 entry, for the gradient check. */
  gradW2(ids, target, j, k) {
    const { h, logits } = this.forward(ids);
    const p = softmax64(logits);
    const dlogit = p[k] - (k === target ? 1 : 0);
    return h[j] * dlogit;
  }

  /** The same gradient, estimated by nudging the weight and re-running the model. */
  numericGradW2(ids, target, j, k, eps = 1e-5) {
    const idx = j * this.V + k;
    const original = this.W2[idx];
    this.W2[idx] = original + eps;
    const a = this.loss(ids, target);
    this.W2[idx] = original - eps;
    const b = this.loss(ids, target);
    this.W2[idx] = original;
    return (a - b) / (2 * eps);
  }

  /** A deterministic training batch drawn from the corpus. */
  batch(size, rng) {
    const out = [];
    for (let n = 0; n < size; n++) {
      const i = this.ctx + Math.floor(rng() * (this.data.length - this.ctx - 1));
      out.push([this.data.slice(i - this.ctx, i), this.data[i]]);
    }
    return out;
  }

  /** Mean loss over a fixed slice — the honest measure of progress. */
  evaluate(count = 200) {
    const rng = mulberry32(99);
    let total = 0;
    const b = this.batch(count, rng);
    for (const [ids, target] of b) total += this.loss(ids, target);
    return total / b.length;
  }

  /** Generate text with the lesson-01 sampler semantics. */
  generate(prompt, length, { temperature = 0.8, rng = Math.random } = {}) {
    let ids = [...prompt].map((c) => this.index.get(c)).filter((i) => i !== undefined);
    while (ids.length < this.ctx) ids.unshift(0);
    let out = '';
    for (let n = 0; n < length; n++) {
      const { logits } = this.forward(ids.slice(-this.ctx));
      const p = softmax64(logits, temperature);
      let r = rng(), pick = p.length - 1;
      for (let k = 0; k < p.length; k++) { r -= p[k]; if (r <= 0) { pick = k; break; } }
      out += this.vocab[pick];
      ids.push(pick);
    }
    return out;
  }
}
