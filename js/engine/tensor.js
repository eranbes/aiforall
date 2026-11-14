// tensor.js — flat Float32Array matrices + the ops an inference engine actually needs.
// Mirrors, in readable JS, what TensorFold's kernels do on Metal/CUDA.

export class Mat {
  /** @param {number} rows @param {number} cols @param {Float32Array} [data] */
  constructor(rows, cols, data) {
    this.r = rows; this.c = cols;
    this.d = data || new Float32Array(rows * cols);
  }
  static zeros(r, c) { return new Mat(r, c); }
  static from(arr2d) {
    const r = arr2d.length, c = arr2d[0].length;
    const m = new Mat(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m.d[i * c + j] = arr2d[i][j];
    return m;
  }
  static randn(r, c, scale = 1, rng = Math.random) {
    const m = new Mat(r, c);
    for (let i = 0; i < m.d.length; i++) m.d[i] = gauss(rng) * scale;
    return m;
  }
  get(i, j) { return this.d[i * this.c + j]; }
  set(i, j, v) { this.d[i * this.c + j] = v; }
  row(i) { return this.d.subarray(i * this.c, (i + 1) * this.c); }
  clone() { return new Mat(this.r, this.c, this.d.slice()); }
  toArray() {
    const out = [];
    for (let i = 0; i < this.r; i++) out.push(Array.from(this.row(i)));
    return out;
  }
  map(f) {
    const m = new Mat(this.r, this.c);
    for (let i = 0; i < this.d.length; i++) m.d[i] = f(this.d[i], i);
    return m;
  }
  fill(v) { this.d.fill(v); return this; }
}

function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Deterministic RNG so every lesson reproduces across reloads. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** C = A @ B  — the operation that is ~95% of all LLM FLOPs. */
export function matmul(A, B, out) {
  if (A.c !== B.r) throw new Error(`matmul shape: ${A.r}x${A.c} @ ${B.r}x${B.c}`);
  const C = out || new Mat(A.r, B.c);
  const { r: n, c: k } = A, m = B.c;
  C.d.fill(0);
  for (let i = 0; i < n; i++) {
    const aoff = i * k, coff = i * m;
    for (let p = 0; p < k; p++) {
      const a = A.d[aoff + p];
      if (a === 0) continue;
      const boff = p * m;
      for (let j = 0; j < m; j++) C.d[coff + j] += a * B.d[boff + j];
    }
  }
  return C;
}

/** A @ B^T — how attention scores are formed (Q @ K^T). */
export function matmulT(A, B, out) {
  if (A.c !== B.c) throw new Error(`matmulT shape: ${A.r}x${A.c} @ (${B.r}x${B.c})^T`);
  const C = out || new Mat(A.r, B.r);
  for (let i = 0; i < A.r; i++) {
    for (let j = 0; j < B.r; j++) {
      let s = 0;
      const ao = i * A.c, bo = j * B.c;
      for (let p = 0; p < A.c; p++) s += A.d[ao + p] * B.d[bo + p];
      C.d[i * C.c + j] = s;
    }
  }
  return C;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
export function norm(a) { return Math.sqrt(dot(a, a)); }
export function cosine(a, b) {
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot(a, b) / d;
}

export function addRow(A, bias) {
  for (let i = 0; i < A.r; i++) for (let j = 0; j < A.c; j++) A.d[i * A.c + j] += bias[j];
  return A;
}
export function scale(A, s) {
  for (let i = 0; i < A.d.length; i++) A.d[i] *= s;
  return A;
}
export function addInto(A, B) {
  for (let i = 0; i < A.d.length; i++) A.d[i] += B.d[i];
  return A;
}

/** Numerically stable softmax over a plain array; returns a new Float32Array. */
export function softmax(logits, temperature = 1) {
  const n = logits.length;
  const out = new Float32Array(n);
  let max = -Infinity;
  const t = Math.max(1e-6, temperature);
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp((logits[i] - max) / t);
    out[i] = e; sum += e;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/** Row-wise softmax in place (attention weights). */
export function softmaxRows(A) {
  for (let i = 0; i < A.r; i++) {
    const row = A.row(i);
    let max = -Infinity;
    for (let j = 0; j < A.c; j++) if (row[j] > max) max = row[j];
    let s = 0;
    for (let j = 0; j < A.c; j++) { row[j] = Math.exp(row[j] - max); s += row[j]; }
    for (let j = 0; j < A.c; j++) row[j] /= s;
  }
  return A;
}

/** Causal mask: position i may not attend to j > i. */
export function causalMask(A, neg = -1e9) {
  for (let i = 0; i < A.r; i++) for (let j = i + 1; j < A.c; j++) A.d[i * A.c + j] = neg;
  return A;
}

/** RMSNorm — what modern transformers (Qwen, Llama, Nemotron) use instead of LayerNorm. */
export function rmsnorm(x, weight = null, eps = 1e-6) {
  const out = new Float32Array(x.length);
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * (weight ? weight[i] : 1);
  return out;
}

export function layernorm(x, eps = 1e-5) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  v /= n;
  const inv = 1 / Math.sqrt(v + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv;
  return out;
}

/** Rotary position embedding, applied in place to one head's vector. */
export function rope(vec, pos, base = 10000) {
  const d = vec.length;
  for (let i = 0; i < d; i += 2) {
    const theta = pos / Math.pow(base, i / d);
    const c = Math.cos(theta), s = Math.sin(theta);
    const a = vec[i], b = vec[i + 1];
    vec[i] = a * c - b * s;
    vec[i + 1] = a * s + b * c;
  }
  return vec;
}

/* ---- activations ---- */
export const act = {
  relu: (x) => (x > 0 ? x : 0),
  drelu: (x) => (x > 0 ? 1 : 0),
  tanh: Math.tanh,
  dtanh: (x) => 1 - Math.tanh(x) ** 2,
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  dsigmoid: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
  silu: (x) => x / (1 + Math.exp(-x)),
  dsilu: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 + x * (1 - s)); },
  gelu: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
  dgelu: (x) => {
    const k = 0.7978845608, c = 0.044715;
    const u = k * (x + c * x * x * x), t = Math.tanh(u);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * c * x * x);
  },
  step: (x) => (x > 0 ? 1 : 0),
  identity: (x) => x,
};

export const argmax = (a) => {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > bv) { bv = a[i]; bi = i; }
  return bi;
};

export const topIndices = (a, k) =>
  Array.from(a.keys ? a.keys() : a.map((_, i) => i))
    .sort((i, j) => a[j] - a[i])
    .slice(0, k);
