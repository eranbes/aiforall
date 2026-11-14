// sample.js — the sampler, in the order a production engine applies it.
//
// This mirrors TensorFold's canonical host sampler, `chooseOrdered`
// (zig/src/core/lanes/sampling.zig:135-178):
//
//   greedy? → top-k → temperature → top-p → min-p → keyed Gumbel argmax
//
// Two things in that line surprise most people:
//
//   1. top-k is applied BEFORE temperature. It can be: temperature is monotonic,
//      so it never reorders the candidates, so a prefix cut is the same prefix either way.
//      top-p is NOT order-free — it is computed on the temperature-scaled probabilities.
//
//   2. The draw is not "roll a die against a cumulative distribution". It adds Gumbel
//      noise to each scaled logit and takes the argmax. Mathematically identical, but it
//      makes the choice a PURE FUNCTION of (seed, position, token id) — no RNG stream to
//      keep in sync across batches, GPUs, restarts or speculative branches.

const M64 = (1n << 64n) - 1n;

function splitmix64(x) {
  x = (x + 0x9e3779b97f4a7c15n) & M64;
  let z = x;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M64;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M64;
  return (z ^ (z >> 31n)) & M64;
}

/**
 * The keyed uniform: same construction as the engine's `uniform(seed, position, id)`
 * (splitmix64 over the three keys). Not bit-compatible with it — same property.
 * @returns {number} in (0, 1)
 */
export function keyedUniform(seed, position, id) {
  let h = splitmix64(BigInt(seed >>> 0));
  h = splitmix64(h ^ BigInt(position >>> 0));
  h = splitmix64(h ^ BigInt(id >>> 0));
  return (Number(h >> 11n) + 0.5) / 9007199254740992;
}

/**
 * Penalties — the stage that runs BEFORE everything else in this file.
 *
 * A faithful port of `zig/src/hip_serve/draw.zig:190-216` (`process`) and the
 * formula at `zig/src/core/lanes/sampling.zig:27-38`. Two families coexist and
 * they are NOT the same operation:
 *
 *   repetition (HuggingFace) — MULTIPLICATIVE and sign-dependent, over
 *     prompt + reply, each token counted ONCE. A negative logit is multiplied
 *     (pushed further down); a positive one is divided (pulled toward zero).
 *   presence + frequency (OpenAI) — ADDITIVE, over the REPLY ONLY, counted
 *     with multiplicity: a token held `count` times loses
 *     `presence + frequency * count`.
 *
 * Both run on the raw logits, before temperature and before any cut — so they
 * change greedy output too. The engine's own doc comment says "greedy too".
 *
 * @param {Float64Array|number[]} logits  modified in place
 * @param {{history?:number[], reply?:number[], repetition?:number,
 *          presence?:number, frequency?:number}} opts
 */
export function applyPenalties(logits, { history = [], reply = [], repetition = 1, presence = 0, frequency = 0 } = {}) {
  if (!(repetition > 0) || !isFinite(repetition)) throw new Error('BadRepetitionPenalty');

  if (repetition !== 1) {
    const seen = new Set(history);            // counted once, prompt AND reply
    for (const id of seen) {
      if (id >= logits.length) continue;
      logits[id] = logits[id] < 0 ? logits[id] * repetition : logits[id] / repetition;
    }
  }

  if (presence !== 0 || frequency !== 0) {
    const counts = new Map();                 // reply only, with multiplicity
    for (const id of reply) counts.set(id, (counts.get(id) || 0) + 1);
    for (const [id, count] of counts) {
      if (id >= logits.length) continue;
      logits[id] -= presence + frequency * count;
    }
  }
  return logits;
}

/**
 * @param {ArrayLike<number>} logits raw model outputs
 * @param {{temperature?:number, topK?:number, topP?:number, minP?:number,
 *          seed?:number, position?:number}} opts
 * @returns {{probs:Float32Array, kept:number[], order:number[], chosen:number,
 *            gumbel:Map<number,number>, steps:{afterK:Set<number>,afterP:Set<number>,afterMinP:Set<number>},
 *            greedy:boolean}}
 */
export function sample(logits, { temperature = 1, topK = 0, topP = 1, minP = 0, seed = 1, position = 0 } = {}) {
  const width = logits.length;
  const probs = new Float32Array(width);

  // candidates sorted by value desc, ties to the lower id — the engine's `Candidate` order
  const order = Array.from({ length: width }, (_, i) => i)
    .sort((a, b) => (logits[b] - logits[a]) || (a - b));

  // 1. greedy: temperature <= 0 returns the first maximum.
  if (temperature <= 0) {
    probs[order[0]] = 1;
    return {
      probs, kept: [order[0]], order, chosen: order[0], greedy: true,
      gumbel: new Map(),
      steps: { afterK: new Set(order), afterP: new Set([order[0]]), afterMinP: new Set([order[0]]) },
    };
  }

  // 2. top-k: a prefix cut of the sorted order.
  const k = Math.max(1, Math.min(topK > 0 ? topK : width, width));
  const afterK = new Set(order.slice(0, k));

  // 3. temperature: divide the surviving logits.
  const t = Math.max(temperature, 1e-6);
  const scaled = new Float64Array(k);
  for (let j = 0; j < k; j++) scaled[j] = logits[order[j]] / t;

  // 4. top-p: softmax (shifted by the largest), keep `below + 1` — the token that
  //    *crosses* the threshold is included, not excluded.
  let keep = k;
  const p = new Float64Array(k);
  let total = 0;
  for (let j = 0; j < k; j++) { p[j] = Math.exp(scaled[j] - scaled[0]); total += p[j]; }
  for (let j = 0; j < k; j++) p[j] /= total;
  if (topP > 0 && topP < 1) {
    let cum = 0, below = 0;
    for (let j = 0; j < k; j++) { cum += p[j]; if (cum < topP) below++; }
    keep = Math.min(keep, below + 1);
  }
  const afterP = new Set(order.slice(0, keep));

  // 5. min-p: an ADDITIVE floor in log space — scaled[0] + ln(min_p) — never a
  //    probability comparison. Same result, no division.
  if (minP > 0) {
    const floor = scaled[0] + Math.log(minP);
    let n = 0;
    for (let j = 0; j < keep; j++) if (scaled[j] >= floor) n++;
    keep = Math.max(1, n);
  }
  const kept = order.slice(0, keep);
  const afterMinP = new Set(kept);

  // the effective distribution you are actually drawing from
  let mass = 0;
  for (let j = 0; j < keep; j++) mass += p[j];
  for (let j = 0; j < keep; j++) probs[order[j]] = p[j] / mass;

  // 6. the draw: argmax over scaled logit + Gumbel noise, keyed per candidate.
  const gumbel = new Map();
  let best = 0, bestScore = -Infinity;
  for (let j = 0; j < keep; j++) {
    const u = keyedUniform(seed, position, order[j]);
    const g = -Math.log(-Math.log(u));
    gumbel.set(order[j], g);
    const score = scaled[j] + g;
    if (j === 0 || score > bestScore) { best = j; bestScore = score; }
  }

  return { probs, kept, order, chosen: order[best], gumbel, scaled, keep, steps: { afterK, afterP, afterMinP }, greedy: false };
}

/** Shannon entropy in bits: how undecided the model is at this step. */
export function entropy(p) {
  let h = 0;
  for (const v of p) if (v > 0) h -= v * Math.log2(v);
  return h;
}

/** Perplexity = 2^entropy: the "effective number of choices". */
export const perplexity = (p) => Math.pow(2, entropy(p));

/** Negative log-likelihood of the true token — the training loss, per token. */
export const nll = (p, target) => -Math.log(Math.max(1e-12, p[target]));
