# Forward-pass code map — token ids → logits

Paths are relative to the TensorFold repo root (`TensorFold/`).
Every anchor below was read; line numbers are correct as of this read.

---

## 0. Concrete config — Qwen3.5-2B (the Metal reference target)

- **Anchor**: `zig/src/families/qwen3_5/config.zig:7-21` (constants) and `zig/src/families/qwen3_5/config.zig:95-104` (where they are checked against `config.json`)
- **What it does**: The Metal 2B path pins the geometry at compile time: every kernel is specialised for these exact widths (the `.metal` files literally hard-code `2048`, `6144`, `248320`, `256`). `parse` then reads `config.json`'s `text_config` and *refuses* any checkpoint whose numbers differ, so there is no runtime shape plumbing at all.
- **Excerpt**:

```zig
pub const hidden = 2048;
pub const vocab = 248320;
pub const layers = 24;
pub const intermediate = 6144;
pub const linear_heads = 16;
pub const linear_dim = 128;
pub const conv_dim = 6144;
pub const conv_taps = 4;
pub const query_heads = 8;
pub const kv_heads = 2;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const group = 64;
pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000000;
```

And the admission check (`config.zig:95-104`):

```zig
    const dims = .{
        .{ "hidden_size", hidden },              .{ "vocab_size", vocab },                  .{ "num_hidden_layers", layers },
        .{ "intermediate_size", intermediate },  .{ "linear_num_key_heads", linear_heads }, .{ "linear_num_value_heads", linear_heads },
        .{ "linear_key_head_dim", linear_dim },  .{ "linear_value_head_dim", linear_dim },  .{ "linear_conv_kernel_dim", conv_taps },
        .{ "num_attention_heads", query_heads }, .{ "num_key_value_heads", kv_heads },      .{ "head_dim", head_dim },
    };
    inline for (dims) |d| if (try integer(text, d[0]) != d[1]) return error.UnsupportedQwenGeometry;
    if (try number(try field(text, "rms_norm_eps")) != 1e-6) return error.UnsupportedQwenConfig;
```

**The numbers, in one table** (all from `zig/src/families/qwen3_5/config.zig`, cross-checked against the real
`config.json` fixture at `config.zig:126-212`):

| Quantity | Value | Source |
|---|---|---|
| `n_layers` | 24 | `config.zig:9` (`num_hidden_layers`) |
| `n_heads` (query) | 8 | `config.zig:15` (`num_attention_heads`) |
| `n_kv_heads` | 2 | `config.zig:16` (`num_key_value_heads`) |
| GQA group | 4 (= 8 / 2) | `attn_partial.metal:23` (`constexpr int G = 4`) |
| `head_dim` | 256 | `config.zig:17` (`head_dim`) |
| rotary dim | 64 (= `partial_rotary_factor` 0.25 × 256) | `config.zig:18`, checked `config.zig:114` |
| hidden size | 2048 | `config.zig:7` (`hidden_size`) |
| MLP intermediate | 6144 | `config.zig:10` (`intermediate_size`) |
| vocab size | 248320 | `config.zig:8` (`vocab_size`) |
| RMSNorm eps | 1e-6 | `config.zig:20` (`rms_norm_eps`) |
| RoPE theta | 1e7 | `config.zig:21` (`rope_theta`) |
| quant group | 64, 4-bit affine | `config.zig:19`, enforced `config.zig:75` |
| attention scale | 1/16 (= 1/√256) | `forward.zig:209` |
| context | ≤ 262144 | `config.zig:120-121` (`max_position_embeddings`) |
| layer schedule | every 4th layer is full attention; the other 3 are gated-delta-net linear attention | `config.zig:37-39`, `qwen35_layers.zig:11-33` |

**Gotcha**: only **6 of 24** layers hold a KV cache. `linear(i) == i % 4 != 3`, so layers 3, 7, 11, 15, 19, 23 are
real attention; the rest are linear/recurrent mixers with a fixed-size state instead of a growing cache. Teaching a
"standard transformer" from this model needs that caveat up front.

MoE numbers for the two MoE families (for §7):

| Model | experts | top-k | expert width | router | source |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 256 | 8 | 512 (`moe_intermediate_size`) | sigmoid + bias | `zig/src/families/qwen3_5/cuda_geometry.zig:348-352`, fixture `cuda_geometry.zig:417-419` |
| Nemotron-H A3B | 128 | 6 | 1856, shared 3712, `routed_scaling_factor` 2.5 | sigmoid + bias, normalised | `zig/src/families/nemotron/fixtures/config.json:87-101` |

---

## 1. Reference attention (naive scaled dot-product)

- **Anchor**: `zig/src/core/attention_ref.zig:5-29`
- **What it does**: One query head's attention in `f64`, the ground truth every backend's attention kernel is
  checked against. Pass 1 finds the row max of `scale · q·k_j`; pass 2 re-computes the scores, exponentiates
  against that max, and accumulates both `Σ p_j v_j` and a parallel error budget `tol`. The final divide by `l`
  normalises. Causality is *not* a mask here — the caller passes `keys`, the list of visible positions, so the
  mask is expressed as "which keys exist".
- **Excerpt**:

```zig
/// softmax(scale q . k_j) v over `keys`; tol 2^-7 of sum p_j |v_j| / sum p_j (bf16 weights) + a bf16 half-ulp.
pub fn head(q: []const f64, k: []const f64, v: []const f64, keys: []const usize, d: usize, scale: f64, out: []f64, tol: []f64) void {
    var m: f64 = -std.math.inf(f64);
    for (keys) |j| {
        var s: f64 = 0;
        for (0..d) |e| s += q[e] * k[j * d + e];
        m = @max(m, s * scale);
    }
    var l: f64 = 0;
    @memset(out, 0);
    @memset(tol, 0);
    for (keys) |j| {
        var s: f64 = 0;
        for (0..d) |e| s += q[e] * k[j * d + e];
        const p = @exp(s * scale - m);
        l += p;
        for (0..d) |e| {
            out[e] += p * v[j * d + e];
            tol[e] += p * @abs(v[j * d + e]);
        }
    }
    for (out, tol) |*o, *t| {
        o.* /= l;
        t.* = t.* / l * std.math.pow(f64, 2, -7) + @abs(o.*) * std.math.pow(f64, 2, -8) + 1e-6;
    }
}
```

- **Teaching value**: Attention *is* these 20 lines. A dot product per key, subtract the max, exponentiate, weighted-average the values. Everything in §2 is the same arithmetic reordered so it fits in registers.
- **Gotchas**: (a) the scale is a *parameter*, not computed — the caller supplies `1/√d` (see `forward.zig:209` passing `1.0/16.0`); (b) there is no explicit `-inf` mask: masking is "which `keys` you hand in"; (c) `tol` makes this a *contract*, not just an implementation — kernels must land inside the budget, and the budget is explicitly sized for bf16 V weights.
- **Self-check**: the file's own test (`attention_ref.zig:31-38`) states the contract crisply: one key returns that key's value verbatim; equal scores give the plain average.

### 1b. A second, differently-shaped reference: multi-head with explicit `1/√d_h`

- **Anchor**: `zig/src/core/joint_schema/reference_attn.zig:8-32`
- **What it does**: Full multi-head attention over an explicit head loop, including the Q/K/V projections sliced
  out of one fused `in_weight` and the `1/√dh` scale computed in place. Unmasked (it is a non-causal encoder head),
  so it is the cleanest place to see *where* the scale comes from and how `dh = W / heads` splits a flat vector into heads.
- **Excerpt**:

```zig
/// MultiheadAttention, no mask; bf16 mode rounds Q, K, V and the context once, scores, softmax and P·V stay exact.
pub fn mha(c: Ctx, m: tensors.Attention, q: []const f64, nq: usize, kv: []const f64, nk: usize, heads: usize) ![]f64 {
    const W = m.in_weight.dim(1);
    const dh = W / heads;
    const qp = try ref.linearRows(c, q, nq, m.in_weight, m.in_bias, 0, W);
    const kp = try ref.linearRows(c, kv, nk, m.in_weight, m.in_bias, W, W);
    const vp = try ref.linearRows(c, kv, nk, m.in_weight, m.in_bias, 2 * W, W);
    const ctx = try c.a.alloc(f64, nq * W);
    const s = try c.a.alloc(f64, nk);
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(dh)));
    for (0..nq) |i| for (0..heads) |hh| {
        const qi = qp[i * W + hh * dh ..][0..dh];
        for (s, 0..) |*x, j| {
            var d: f64 = 0;
            for (qi, kp[j * W + hh * dh ..][0..dh]) |a, b| d += a * b;
            x.* = d * scale;
        }
        ref.softmax(s);
        const o = ctx[i * W + hh * dh ..][0..dh];
        @memset(o, 0);
        for (s, 0..) |p, j| for (o, vp[j * W + hh * dh ..][0..dh]) |*x, v| {
            x.* += p * v;
        };
        for (o) |*x| x.* = c.r(x.*);
    };
    return ref.linear(c, ctx, nq, m.out);
}
```

- **Teaching value**: "Multi-head" is just *the same single-head routine run on disjoint slices of one flat vector* — `[i*W + hh*dh ..][0..dh]` is the whole idea.
- The textbook softmax it calls is `zig/src/core/joint_schema/reference.zig:96-105` (`softmax`): max, subtract, `exp`, sum, divide.

---

## 2. Production attention (Metal kernels)

Two production kernels are worth mapping — one general and self-documenting, one Qwen3.5-specific and fused.

### 2a. The clearest online-softmax statement — `ops/sdpa.metal`

- **Anchor**: `zig/kernels/metal/ops/sdpa.metal:10-19` (the rescaling step) and `:23-47` (the streaming loop)
- **What it does**: `tf_softmax_step` is *the* flash-attention rescale, isolated into a 10-line function: on each
  new score, lift the running max to `peak`, multiply the accumulated sum **and** the accumulated output vector by
  `exp(old_max - peak)`, then add the new `exp(score - peak)`-weighted value. `tf_sdpa_one_pass` streams the whole
  KV cache through it with simdgroup `s` taking keys `s, s+32, s+64, …` and lane `l` owning dims `4l..4l+3`.
- **Excerpt** (the online-softmax rescaling, verbatim):

```cpp
template <int PER>
inline void tf_softmax_step(float score, thread float& top, thread float& mass, thread float* acc,
                            const device bfloat* v) {
  const float peak = max(top, score);
  const float rescale = fast::exp(top - peak);
  const float weight = fast::exp(score - peak);
  top = peak;
  mass = mass * rescale + weight;
  for (int j = 0; j < PER; ++j) acc[j] = acc[j] * rescale + weight * v[j];
}
```

And the loop body, with the causal mask and the GQA head mapping (`sdpa.metal:27-46`):

```cpp
  const int head = int(tg.x), row = int(tg.y);
  const int rows = int(grid_tg.y);
  const int out_at = head * rows + row;
  const int q_at = QT ? int(grid_tg.x) * row + head : out_at;
  const int kv = head / gqa;
  float qv[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < 4; ++j) qv[j] = static_cast<float>(scale) * q[q_at * 128 + 4 * lane + j];
  float top = TF_FLOOR, mass = 0.0f;
  const device bfloat* kp = k + kv * k_head + long(sg) * k_seq + 4 * lane;
  const device bfloat* vp = v + kv * v_head + long(sg) * v_seq + 4 * lane;
  for (int i = int(sg); i < n; i += 32) {
    if (!CAUSAL || i <= n - rows + row) {
      float kr[4];
      for (int j = 0; j < 4; ++j) kr[j] = kp[j];
      float score = 0.0f;
      for (int j = 0; j < 4; ++j) score += qv[j] * kr[j];
      tf_softmax_step<4>(simd_sum(score), top, mass, acc, vp);
    }
    kp += 32 * int(k_seq);
    vp += 32 * int(v_seq);
  }
```

- **How it differs from §1**: (i) **one pass, not two** — no pre-scan for the max, so K is read once instead of
  twice; (ii) the dot product is a **cross-lane reduction** (`simd_sum(score)`), each of 32 lanes holding 4 of the
  128 dims; (iii) the scale is folded into `qv` at load time, so no per-score multiply; (iv) the normalising divide
  is deferred to the very end; (v) the causal mask is a **branch that skips work**, never a `-inf` added to a score.
- **Teaching value**: You never need the whole score row in memory. Carrying `(running max, running sum, running output)` and rescaling on the fly gives bit-comparable results in O(1) extra space — that single insight *is* flash attention.
- **Gotchas**: `TF_FLOOR = -0x1.fffffep+127f` (`sdpa.metal:7`) not `-INFINITY`, because `exp(-inf - -inf)` is NaN; `fast::exp` is the hardware approximation; the file's header comment claims bit-identity with MLX 0.32.3's `sdpa_vector`, so the rounding order is a *contract*, not an accident. The two-pass variants (`tf_sdpa_blocks` at `:71`, merged by `tf_sdpa_merge` at `:168`) split the KV range across threadgroups and then merge partials with the same `exp(m_i - m)` trick — that's how long contexts stay parallel.

### 2b. The Qwen3.5 production path — `qwen3_5/attn_partial.metal` + `attn_merge.metal`

- **Anchor**: `zig/kernels/metal/qwen3_5/attn_partial.metal:21-32` (the tiling constants and head mapping) and `:44-71` (the streaming body)
- **What it does**: Splits the KV range into 128-key chunks (`CK`), one threadgroup per chunk, and inside a chunk
  splits again 4 ways (`SPLIT`) with 4 keys per lane-step (`BLK`). Each simdgroup keeps its own `(m, l, o)` triple,
  scores 4 keys at a time, takes the block max `bm` *first*, then performs **one** rescale for all 4 keys instead of
  4 separate rescales. The partials go to `PM`/`PL`/`PO` and `attn_merge` folds them.
- **Excerpt** (`attn_partial.metal:44-71`):

```cpp
    float m = -INFINITY, l = 0.0f;
    for (int base = k0 + s; base < k1; base += SPLIT * BLK) {
      float sc[BLK];
      int rows[BLK];
      float bm = -INFINITY;
      for (int j = 0; j < BLK; j++) {
        const int pos = base + j * SPLIT;
        rows[j] = pos < k1 ? (pos < P ? pos : pos) : -1;
        float d = 0.0f;
        if (rows[j] >= 0) {
          const device bfloat* kr = K + (size_t(h) * CAP + rows[j]) * D + int(lane) * DPL;
          for (int i = 0; i < DPL; i++) d = fma(q[i], float(kr[i]), d);
        }
        sc[j] = simd_sum(d) * scale[0];
        if (rows[j] >= 0) bm = metal::max(bm, sc[j]);
      }
      const float mn = metal::max(m, bm);
      const float a = metal::exp(m - mn);
      l *= a;
      for (int i = 0; i < DPL; i++) o[i] *= a;
      for (int j = 0; j < BLK; j++) {
        if (rows[j] < 0) continue;
        const float b = metal::exp(sc[j] - mn);
        l += b;
        const device bfloat* vr = V + (size_t(h) * CAP + rows[j]) * D + int(lane) * DPL;
        for (int i = 0; i < DPL; i++) o[i] = fma(b, float(vr[i]), o[i]);
      }
      m = mn;
    }
```

The cross-chunk merge is the same algebra again (`attn_merge.metal:22-33`):

```cpp
  float mx_ = -INFINITY;
  for (int c = 0; c < NCH; c++) mx_ = metal::max(mx_, PM[base + c]);
  float lsum = 0.0f, acc[DPL];
  for (int i = 0; i < DPL; i++) acc[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const float e = PL[base + c] > 0.0f ? metal::exp(PM[base + c] - mx_) : 0.0f;
    lsum = fma(PL[base + c], e, lsum);
    for (int i = 0; i < DPL; i++) acc[i] = fma(PO[size_t(base + c) * D + int(lane) * DPL + i], e, acc[i]);
  }
  for (int i = 0; i < DPL; i++) OUT[((w * 8) + qh) * D + int(lane) * DPL + i] = bfloat(acc[i] / lsum);
```

- **How it differs from §2a**: block-amortised rescaling (one `exp(m - mn)` per 4 keys, not per key), `fma`
  everywhere, `D = 256` with `DPL = 8` dims per lane, and a **three-level** softmax merge (within a simdgroup →
  across the 4 `SPLIT` simdgroups at `attn_partial.metal:78-93` → across the `NCH` chunk threadgroups in
  `attn_merge`). Every level is the same `exp(m_i - m)` reweighting.
- **Gotchas**: the causal mask here is *purely a loop bound* — `const int last = P + w; const int k1 = min(k0 + CK, last + 1);` (`attn_partial.metal:36-38`), where `P` is the committed cache length and `w` the row's offset inside the current chunk of new tokens. No comparison per key at all; row `w` simply never reads key `> P + w`. Also note `rows[j] = pos < k1 ? (pos < P ? pos : pos) : -1` — both ternary branches are identical, a leftover from a paged variant; it reduces to `pos < k1 ? pos : -1`. And `attn_merge` divides by `lsum` without guarding against zero (safe only because row `w` always sees at least its own key).

---

## 3. GQA / MQA — sharing K/V heads across Q heads

- **Anchor**: `zig/kernels/metal/ops/sdpa.metal:31` (the one-line mapping) and `zig/kernels/metal/qwen3_5/attn_partial.metal:21-32` (the inverse mapping)
- **What it does**: Two directions of the same arithmetic. In `ops/sdpa.metal` the kernel is launched per **query**
  head and derives its KV head by integer division: `kv = head / gqa`. In Qwen3.5's `attn_partial` the kernel is
  launched per **KV** head (so K and V are read once for the whole group) and derives the query head by
  multiplication: `qh = h * G + g`, where `g = sgi / SPLIT` selects which of the `G = 4` queries in the group this
  simdgroup serves.
- **Excerpt** — the query-major direction (`ops/sdpa.metal:27-33`):

```cpp
  const int head = int(tg.x), row = int(tg.y);
  const int rows = int(grid_tg.y);
  const int out_at = head * rows + row;
  const int q_at = QT ? int(grid_tg.x) * row + head : out_at;
  const int kv = head / gqa;
  float qv[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int j = 0; j < 4; ++j) qv[j] = static_cast<float>(scale) * q[q_at * 128 + 4 * lane + j];
```

And the KV-major direction (`attn_partial.metal:21-34`):

```cpp
constexpr int D = 256;
constexpr int G = 4;
constexpr int CK = 128;
constexpr int SPLIT = 4;
constexpr int BLK = 4;

  const uint lane = thread_index_in_simdgroup;
  const uint sgi = simdgroup_index_in_threadgroup;
  const int g = int(sgi) / SPLIT, s = int(sgi) % SPLIT;
  const int c = int(threadgroup_position_in_grid.y);
  const int P = dims[0], W = dims[1], CAP = dims[2], NCH = dims[3], MAXD = dims[4];
  const int h = int(threadgroup_position_in_grid.z) / W;
  const int w = int(threadgroup_position_in_grid.z) % W;
  constexpr int DPL = D / 32;
  const int qh = h * G + g;
```

- **Teaching value**: GQA is *one integer division*. `kv = q_head / (n_heads / n_kv_heads)`. Nothing else in the
  attention math changes — the cache just gets `n_heads / n_kv_heads` times smaller. For Qwen3.5-2B that is 8 query
  heads over 2 KV heads, a 4× cache saving. MQA is the same formula at `gqa = n_heads` (one shared KV head).
- **Gotchas**: the launch shape tells you which direction a kernel chose — `forward.zig:214` dispatches
  `.{ 512, nch, c.kv_heads * seg.rows }`, i.e. the grid's z axis is **KV heads × rows**, not query heads. That's
  deliberate: a group's 4 queries share one `K`/`V` fetch, so GQA converts an arithmetic saving into a
  **bandwidth** saving. Note also the K/V pointer in `attn_partial.metal:54`: `K + (size_t(h) * CAP + rows[j]) * D`
  indexes by `h` (the KV head), never by `qh`.

---

## 4. RoPE — rotary position embeddings

### 4a. The rotation, in production (fused, table-free)

- **Anchor**: `zig/kernels/metal/qwen3_5/layout.metal:47-73`
- **What it does**: `qwen35_rotate` computes one output dimension of one rotated head. It recomputes
  `cos`/`sin` from scratch per element (no precomputed table), applies the **rotate-half** convention (dim `i`
  pairs with dim `i + 32`), and leaves every dim `≥ 64` untouched — that's `partial_rotary_factor = 0.25` over a
  256-wide head. It is then called straight from the Q-writer and the K-writer, so RoPE is *fused into the KV-cache
  write*: no separate rope pass, no intermediate buffer.
- **Excerpt**:

```cpp
inline bfloat qwen35_rotate(const device bfloat* x, uint d, uint position) {
  if (d >= 64) return x[d];
  const uint i = d % 32;
  const float angle = float(position) * metal::pow(10000000.0f, -float(i) / 32.0f);
  const float cs = metal::cos(angle), sn = metal::sin(angle);
  const float a = float(x[i]), b = float(x[i + 32]);
  return bfloat(d < 32 ? a * cs - b * sn : b * cs + a * sn);
}

kernel void qwen35_queries(const device bfloat* x [[buffer(0)]],
                           device bfloat* out [[buffer(1)]],
                           constant uint4& dims [[buffer(2)]],
                           uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z;
  const uint m = dims.z + r;
  out[(size_t(h) * dims.y + m) * 256 + d] = qwen35_rotate(x + (size_t(m) * 8 + h) * 256, d, dims.x + r);
}

kernel void qwen35_keys(const device bfloat* k [[buffer(0)]],
                        const device bfloat* v [[buffer(1)]],
                        device bfloat* keys [[buffer(2)]],
                        device bfloat* values [[buffer(3)]],
                        constant uint4& dims [[buffer(4)]],
                        uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z;
  const uint m = dims.z + r;
  const size_t dst = (size_t(h) * dims.y + dims.x + r) * 256 + d;
  keys[dst] = qwen35_rotate(k + (size_t(m) * 2 + h) * 256, d, dims.x + r);
  values[dst] = v[(size_t(m) * 2 + h) * 256 + d];
}
```

- **Teaching value**: Position is injected by **rotating pairs of dimensions by an angle proportional to the
  position** — a 2-D rotation matrix per pair, with a per-pair frequency. The dot product `q·k` then depends only
  on the *difference* of positions, which is why it generalises.
- **Gotchas**: (a) `theta = 1e7`, not the classic 1e4 — visible inline as the literal `10000000.0f`;
  (b) only the first 64 of 256 dims rotate, the rest pass through verbatim (`if (d >= 64) return x[d]`);
  (c) rotate-half pairs `(i, i+32)`, **not** adjacent `(2i, 2i+1)` — this matches HF `transformers`, not the
  original RoPE paper, and the two are not interchangeable; (d) `V` is written *unrotated* in the same kernel —
  RoPE touches Q and K only; (e) the exponent is `-i/32` over 32 pairs, which is the `2i/dim` form with `dim = 64`.

### 4b. The reference rotation (with QK-norm fused in)

- **Anchor**: `zig/src/core/qk_ref.zig:6-24`
- **What it does**: The `f64` reference for the whole Q/K input path: RMS-norm the head by `w`, then rotate-half
  RoPE its first `rot` dims. `null` weights mean "skip the norm"; `null` cos/sin mean "skip the rope". This is the
  clearest statement of *where QK-norm sits relative to RoPE* (norm first, then rotate).
- **Excerpt**:

```zig
/// One head: RMS-normed by w (none: as is), then rotate-half RoPE on its first `rot` dims (no cos/sin: none).
pub fn head(x: []const f64, w: ?[]const f64, eps: f64, cos: ?[]const f64, sin: ?[]const f64, rot: usize, y: []f64) void {
    const n = x.len;
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const inv = if (w != null) 1 / @sqrt(ss / @as(f64, @floatFromInt(n)) + eps) else 1;
    var z: [512]f64 = undefined;
    for (x, 0..) |v, i| z[i] = v * inv * (if (w) |ww| ww[i] else 1);
    const half = rot / 2;
    for (y, 0..) |*o, i| {
        if (cos == null or i >= rot) {
            o.* = z[i];
            continue;
        }
        const p = if (i < half) i + half else i - half;
        const rotated = if (i < half) -z[p] else z[p];
        o.* = z[i] * cos.?[i] + rotated * sin.?[i];
    }
}
```

- Its test (`qk_ref.zig:109-116`) is the one-line statement of the convention: *"rotate-half pairs (d, d + rot/2) and leave the rest"*.

### 4c. The frequency table — `1/θ^(2i/dim)`

- **Anchor**: `zig/src/core/rope_freq.zig:5-12`
- **What it does**: Produces `transformers`' default inverse frequencies, with the rounding steps in exactly
  torch's order: an **fp32** exponent, `pow` in f64 rounded **once** to fp32, then an fp32 reciprocal.
- **Excerpt**:

```zig
/// out[i] = 1 / theta^(2i / dim), i < out.len: an fp32 exponent, theta^e rounded once to fp32, an fp32 reciprocal.
pub fn inverseFrequencies(theta: f64, dim: usize, out: []f32) void {
    for (out, 0..) |*f, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
        const p: f32 = @floatCast(std.math.pow(f64, theta, e));
        f.* = 1.0 / p;
    }
}
```

- **Gotchas**: a glorious callout lives in this file's last test (`rope_freq.zig:95-108`), *"planted: the fp32-pow
  form misses transformers' words for theta 1e7 over 64 (15 of 32, up to 5 ulp)"* — doing the `pow` in fp32
  instead of f64-then-round gives different bits for **15 of 32** frequencies. The test for theta 1e7 / 64 dims
  (`rope_freq.zig:27-40`) pins the exact fp32 words Qwen3.5 must produce.

### 4d. YaRN context extension (`--rope-yarn`)

- **Anchor**: `zig/src/core/rope_yarn.zig:5-26`
- **What it does**: `transformers`' static YaRN for a 64-dim rotary width. It computes the two cutoff dimensions
  (`low` from β=32, `high` from β=1), then per frequency blends the **extrapolated** `1/pos_freq` against the
  **interpolated** `1/(factor · pos_freq)` by a ramp `ef` clamped to `[0,1]`. It also returns the attention
  temperature `scale = 0.1·ln(factor) + 1`, which the kernels fold into cos/sin.
- **Excerpt**:

```zig
/// _compute_yarn_parameters (beta 32, 1; truncated ends) in fp32; flashnext keeps its copy, which its receipts hash.
pub fn yarn(base: f64, factor: f64, trained: usize) struct { freq: [32]f32, scale: f32 } {
    const dim: f64 = 64;
    const tp: f64 = @floatFromInt(trained);
    const dimOf = struct {
        fn f(rot: f64, b: f64, max_pos: f64) f64 {
            return 64 * @log(max_pos / (rot * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.f;
    const low = @max(@floor(dimOf(32, base, tp)), 0);
    var high = @min(@ceil(dimOf(1, base, tp)), dim - 1);
    if (low == high) high += 0.001;
    var out: [32]f32 = undefined;
    for (0..32) |k| {
        const kf: f64 = @floatFromInt(k);
        const pos_freq: f32 = @floatCast(std.math.pow(f64, base, @as(f64, @as(f32, @floatCast(2 * kf / dim)))));
        const extra: f32 = 1.0 / pos_freq;
        const inter: f32 = 1.0 / (@as(f32, @floatCast(factor)) * pos_freq);
        const ef: f32 = 1 - @as(f32, @floatCast(std.math.clamp((kf - low) / (high - low), 0, 1)));
        out[k] = inter * (1 - ef) + extra * ef;
    }
    return .{ .freq = out, .scale = @floatCast(0.1 * @log(factor) + 1) };
}
```

- **Teaching value**: YaRN is a *per-frequency* decision. High-frequency dims (which already see many full cycles
  inside the trained window) are left alone; low-frequency dims get their wavelength stretched by `factor`; the
  middle band is linearly ramped between the two.
- **Where the flag lives**: `zig/src/server/cli.zig:112` declares `--rope-yarn` (CUDA builds only), `cli.zig:236`
  documents it (`rope_yarn: ?f64 = null, // YaRN with this factor over the checkpoint's trained window
  (Qwen3.5/3.6 on CUDA)`), and `zig/src/server/cli_gpu_flags.zig:19` parses it. `cli_test.zig:360-364` asserts
  that Metal/HIP/XPU families **refuse** the flag. Real invocation, from `docs/recipes/qwen3.6-moe.md:204`:
  `tensorfold serve nvidia/Qwen3.6-35B-A3B-NVFP4 --drafter … --rope-yarn 4 --context 1048576`.
- **Gotchas**: it is *static* YaRN — the frequencies are computed once at load, not per token, so YaRN changes
  **every** position's embedding and therefore changes replies even at short lengths
  (`docs/recipes/qwen3.6-moe.md:207`). The `if (low == high) high += 0.001` line exists purely to avoid a 0/0 in
  the ramp. The test at `rope_yarn.zig:30-45` pins all 32 fp32 words against transformers 5.19.0 for
  theta 1e7 / factor 4 / 262144 trained.

---

## 5. Normalisation — RMSNorm (and how LayerNorm differs)

### 5a. RMSNorm, fused with the residual add

- **Anchor**: `zig/kernels/metal/qwen3_5/norm.metal:17-41`
- **What it does**: One threadgroup per row, 16 elements per thread (`E = 16`, `TPG = 2048/16 = 128` threads).
  In a single pass it (1) adds the residual `R` into `H` and writes the **updated residual stream** back to `HO`,
  (2) accumulates `Σ x²` on the post-add value, (3) reduces across simdgroups via threadgroup memory, and (4)
  writes the normalised, weighted row to `XO`. So the residual add and the norm are **one kernel, one read of H**.
- **Excerpt**:

```cpp
constexpr int K = 2048;

  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  const int base = int(m) * K + int(t) * E;
  float hv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    bfloat h = H[base + i];
    h = bfloat(float(h) + float(R[base + i]));
    HO[base + i] = h;
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  for (int i = 0; i < E; i++) XO[base + i] = bfloat(float(Wt[int(t) * E + i]) * (hv[i] * inv));
```

- **Where epsilon is set**: `zig/src/families/qwen3_5/config.zig:20` (`pub const eps: f32 = 1e-6;`), validated
  against `config.json`'s `rms_norm_eps` at `config.zig:104`, and bound into buffer 3 at
  `zig/src/families/qwen3_5/forward.zig:43` (`self.e.setValue(c.eps, 3);`).
- **Teaching value**: RMSNorm has **no mean subtraction and no bias** — it is purely `x · rsqrt(mean(x²) + ε) · w`.
  That's one reduction instead of two, which is the whole reason it displaced LayerNorm in LLMs.
- **Gotchas**: (a) the residual add is fused in, and the *pre-norm* value is what carries forward (`HO`), the
  *post-norm* value (`XO`) only feeds the next projection — classic pre-norm residual topology; (b) the add is
  rounded to bf16 **before** squaring (`h = bfloat(float(h) + float(R[...]))`), so the norm is computed on the
  bf16-rounded stream, which a reimplementation must match; (c) a `norm_nores.metal` sibling exists for layer 0,
  which has no residual yet — see `forward.zig:37-53`, where `norm()` picks between `"norm"` and `"norm_nores"`
  purely on whether a residual buffer was passed; (d) there is a **second** RMSNorm in the attention block,
  `qwen35_head_norm` (`layout.metal:20-45`), which normalises each 256-wide *head* independently with
  `eps = 1e-6f` hard-coded in the kernel — that's QK-norm, and it divides by `256.0f` (head_dim), not by hidden.

### 5b. The host-side RMSNorm (rounding-exact mirror)

- **Anchor**: `zig/src/families/qwen3_5/cpu_kernels.zig:186-191`
- **What it does**: The same row norm on CPU cores, written to reproduce the Metal kernel's rounding points and
  summation order exactly (the `sumSquares` helper at `cpu_kernels.zig:165-179` replicates "16 inputs per thread in
  an fma chain, 32 threads per simdgroup, groups of 512").
- **Excerpt**:

```zig
/// qwen35_norm's row: out = bf16(w * (x * rsqrt(mean(x²) + eps))); the row a multiple of 512 wide.
pub fn rmsnorm(x: []const u16, w: []const u16, eps: f32, out: []u16) void {
    std.debug.assert(x.len % 512 == 0 and w.len == x.len and out.len == x.len);
    const inv = rsqrt(sumSquares(x) / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, w) |*o, h, ww| o.* = bf16of(f32of(ww) * (f32of(h) * inv));
}
```

- **Gotcha**: `rsqrt` here (`cpu_kernels.zig:182-184`) is deliberately `1/√x` **through f64, rounded once** —
  `metal::rsqrt`'s portable form. The test at `cpu_kernels.zig:193-...` proves plain fp32 `1.0/@sqrt(y)` differs.

### 5c. LayerNorm, for contrast

- **Anchor**: `zig/src/core/joint_schema/reference.zig:68-82`
- **What it does**: The real affine LayerNorm the codebase keeps for encoder/vision towers: subtract the mean,
  divide by the **biased** standard deviation, scale by `weight`, **add `bias`**, `eps = 1e-5`.
- **Excerpt**:

```zig
/// Affine LayerNorm (eps 1e-5) over each row, biased variance; one rounding per output.
pub fn layerNorm(c: Ctx, x: []const f64, n: usize, norm: tensors.Norm) ![]f64 {
    const d = norm.weight.dim(0);
    const y = try c.a.alloc(f64, n * d);
    for (0..n) |row| {
        const v = x[row * d ..][0..d];
        var mean: f64 = 0;
        for (v) |e| mean += e;
        mean /= @floatFromInt(d);
        var vari: f64 = 0;
        for (v) |e| vari += (e - mean) * (e - mean);
        const inv = 1 / @sqrt(vari / @as(f64, @floatFromInt(d)) + 1e-5);
        for (y[row * d ..][0..d], v, 0..) |*o, e, i| o.* = c.r((e - mean) * inv * at(norm.weight, i) + at(norm.bias, i));
    }
    return y;
}
```

- **Teaching value**: side by side, the diff is exactly three things: LayerNorm centres (`e - mean`), uses variance
  not raw mean-square, and has a bias term. RMSNorm drops all three. Default eps also differs: 1e-5 vs 1e-6 — and
  that split is codified at `zig/src/core/contract/rows.zig:68`:
  `.eps = if (op == .layer_norm) 1e-5 else 1e-6`.
- Production LayerNorm kernels do exist, for vision towers only — e.g. `zig/kernels/metal/core/qwen_vision.metal:36-40`
  (`qwv_layer_norm`, "LayerNorm of the fp32 residual stream with weight and bias").

---

## 6. MLP / feed-forward — SwiGLU

- **Anchor**: `zig/kernels/metal/qwen3_5/mlp_act.metal:12-18` (the activation) and `zig/src/families/qwen3_5/forward.zig:94-101` (the three projections)
- **What it does**: The classic gated MLP. Two independent projections `gate = x·W_g` and `up = x·W_u`, both
  `2048 → 6144`; the activation kernel computes `silu(gate) * up` elementwise; then one `down` projection
  `6144 → 2048` returns to the residual stream. SiLU is inlined as `gf / (1 + exp(-gf))` — i.e. `x · σ(x)` —
  so gate+activation+multiply is a single fused elementwise kernel over 6144 columns.
- **Excerpt** (`mlp_act.metal`):

```cpp
[[kernel]] void custom_kernel_qwen35_mlp_act_bfloat16_t_bfloat16_t_bfloat16_t(
  const device bfloat16_t* GATE [[buffer(0)]],
  const device bfloat16_t* UP [[buffer(1)]],
  device bfloat16_t* HOUT [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {
constexpr int N = 6144;

  const uint i = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (i >= uint(N)) return;
  const float gf = float(GATE[m * N + i]);
  HOUT[m * N + i] = bfloat(gf / (1.0f + metal::exp(-gf)) * float(UP[m * N + i]));
```

and the host side (`forward.zig:94-101`):

```zig
        enc.projection(block.gate, s.x, 0, s.gate, 0, total);
        enc.projection(block.up, s.x, 0, s.up, 0, total);
        enc.pipe("mlp_act");
        e.setBuffer(s.gate, 0, 0);
        e.setBuffer(s.up, 0, 1);
        e.setBuffer(s.act, 0, 2);
        enc.run(.{ c.intermediate, total, 1 }, .{ 256, 1, 1 });
        enc.projection(block.down, s.act, 0, s.r, 0, total);
```

- **Teaching value**: "SwiGLU" names a shape, not a mystery: **two** up-projections, one gated through SiLU by the
  other, then one down-projection. That's why a gated MLP has 3 weight matrices (3 × 2048 × 6144 ≈ 37.7M params per
  layer) where a plain MLP has 2.
- **Gotchas**: (a) the width ratio is 3× hidden (6144/2048), not the 4× of the original transformer — gating costs
  a third matrix, so the width shrinks to compensate; (b) `exp` not `fast::exp`, so SiLU is the precise form;
  (c) nothing is clamped — bf16 `gate` values are small enough in practice that `exp(-gf)` cannot overflow; (d) for
  contrast, Nemotron's expert MLP uses **ReLU²** (`relu2`), not SwiGLU: `const float h = max(float(bfloat(acc[j])), 0.0f); ACT[...] = bfloat(h * h);`
  at `zig/kernels/metal/nemotron/expert_up.metal:78-81`, and GELU/GEGLU variants exist for Gemma
  (`tf_geglu_rows`, see `zig/src/core/contract/gemma_cuda_records.zig:10`).

---

## 7. MoE routing — top-k experts per token

### 7a. The routing reference

- **Anchor**: `zig/src/core/moe_ref.zig:5-31`
- **What it does**: The `f64` reference every backend's MoE glue is checked against. Experts are **ranked** by
  `sigmoid-free logit + bias` but **weighted** by `σ(logit)` of the winner — the bias steers load balancing without
  distorting the weights. Ties go to the lower id. `normalize` divides by the sum of chosen weights; `scale` is the
  config's `routed_scaling_factor`; `shared` appends a virtual expert `experts` with weight 1.
- **Excerpt**:

```zig
/// Sigmoid routing, as cuda/moe_glue.zig's Route.
pub const Route = struct { experts: usize, topk: usize, shared: bool, normalize: bool = false, scale: f64 = 1 };

/// A row's picks by the fp32 logit + bias (the kernels' ranking exactly; ties: lower id), weights in float64.
pub fn topk(logits: []const f32, bias: ?[]const f32, r: Route, pick: []u32, wts: []f64) void {
    var taken: [4096]bool = @splat(false);
    var sum: f64 = 0;
    for (0..r.topk) |k| {
        var best: ?usize = null;
        var bv: f32 = -std.math.inf(f32);
        for (logits[0..r.experts], 0..) |l, e| {
            if (taken[e]) continue;
            const v = l + (if (bias) |b| b[e] else 0); // one fp32 rounding, as the kernel's __fadd_rn
            if (best == null or v > bv) {
                best = e;
                bv = v;
            }
        }
        taken[best.?] = true;
        pick[k] = @intCast(best.?);
        wts[k] = 1 / (1 + @exp(-@as(f64, logits[best.?])));
        sum += wts[k];
    }
    for (wts[0..r.topk]) |*w| w.* = (if (r.normalize) w.* / sum else w.*) * r.scale;
    if (r.shared) {
        pick[r.topk] = @intCast(r.experts);
        wts[r.topk] = 1;
    }
}
```

A second, **softmax**-style reference (for models that route by softmax rather than sigmoid) is
`zig/src/core/route_ref.zig:19-32`, where weights are `e^(l_k − l_0) / Σ`:

```zig
/// The `ids.len` largest logits (the lower id among equal ones), largest first, and weights e^(l_k - l_0) / their sum.
pub fn topk(logits: []const f64, ids: []usize, weights: []f64) void {
    for (ids, 0..) |*id, k| {
        var best: ?usize = null;
        for (logits, 0..) |v, i| {
            if (std.mem.indexOfScalar(usize, ids[0..k], i) != null) continue;
            if (best == null or v > logits[best.?]) best = i;
        }
        id.* = best.?;
    }
    var total: f64 = 0;
    for (ids) |id| total += @exp(logits[id] - logits[ids[0]]);
    for (ids, weights) |id, *w| w.* = @exp(logits[id] - logits[ids[0]]) / total;
}
```

### 7b. The router logits kernel

- **Anchor**: `zig/kernels/metal/core/moe_route.metal:11-57`
- **What it does**: `tf_route_logits` computes `[rows, E] = x·Wᵀ` for up to 16 rows, with a threadgroup covering
  4 experts × 4 rows and 8 simdgroups producing 8 partials that are summed through threadgroup memory. The router
  weight arrives as `uint4` pairs holding **two bf16 values per 32-bit word**, unpacked by shifting
  (`as_type<float>(words[j] << 16)` for the low half, `& 0xffff0000u` for the high).
- **Excerpt**:

```cpp
  for (int ii = 0; ii < PERL; ii++) {
    float inter[4][4];
    for (int h = 0; h < 2; h++) {
      const uint4 v = w[ii * 2 + h];
      const uint words[4] = {v.x, v.y, v.z, v.w};
      for (int j = 0; j < 4; j++) {
        const int e = h * 8 + j * 2;
        inter[e / 4][e % 4] = as_type<float>(words[j] << 16);
        inter[(e + 1) / 4][(e + 1) % 4] = as_type<float>(words[j] & 0xffff0000u);
      }
    }
    const int k = 32 * (PERL * int(lane) + ii) + 4 * int(s);
    for (int c = 0; c < 4; c++) {
      const device bfloat* x = X + size_t(min(r0 + c, rows - 1)) * TF_K + k;
      for (int tm = 0; tm < 4; tm++) {
        const float xv = float(x[tm]);
        for (int tn = 0; tn < 4; tn++) acc[c][tn] += xv * inter[tm][tn];
      }
    }
  }
```

- **Gotcha**: bf16 → fp32 needs **no instruction** — just place the 16 bits in the high half of a 32-bit word. That
  trick appears all over this codebase.

### 7c. The top-k selection kernel

- **Anchor**: `zig/kernels/metal/core/moe_route.metal:73-119`
- **What it does**: One simdgroup per row. Each lane holds `PER = ceil(E/32)` experts; the top-k loop finds each
  lane's best, reduces across the 32 lanes with a `simd_shuffle_xor` butterfly (ties resolved by lower id at every
  step), records the winner, then **sets that expert's score to `-INFINITY`** so the next iteration can't pick it.
  Weights are the winners' sigmoids, normalised and multiplied by `a.scale`.
- **Excerpt**:

```cpp
    float w[TF_TOPK];
    for (int k = 0; k < TF_TOPK; k++) {
      float best = -INFINITY, bsc = 0.0f;
      int bid = TF_E;
      for (int j = 0; j < PER; j++) {
        const int id = j * 32 + int(lane);
        if (id < TF_E && (c[j] > best || (c[j] == best && id < bid))) { best = c[j]; bid = id; bsc = sc[j]; }
      }
      for (int off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_xor(best, off);
        const int oi = simd_shuffle_xor(bid, off);
        const float os = simd_shuffle_xor(bsc, off);
        if (ob > best || (ob == best && oi < bid)) { best = ob; bid = oi; bsc = os; }
      }
      w[k] = bsc;
      if (int(lane) == bid % 32) c[bid / 32] = -INFINITY;
      if (lane == 0) { picks[r * TF_TOPK + k] = bid; PICK[r * TF_TOPK + k] = bid; }
    }
    if (lane == 0) {
      float total = w[0];
      for (int k = 1; k < TF_TOPK; k++) total = total + w[k];
      for (int k = 0; k < TF_TOPK; k++) WTS[r * TF_TOPK + k] = (w[k] / total) * a.scale;
    }
```

The scoring that feeds it (`moe_route.metal:88-95`) shows the rank/weight split cleanly:

```cpp
    float c[PER], sc[PER];
    for (int j = 0; j < PER; j++) {
      const int id = j * 32 + int(lane);
      if (id < TF_E) {
        sc[j] = tf_sigmoid_precise(LOGITS[r * TF_E + id]);
        c[j] = sc[j] + BIAS[id];
      } else {
        sc[j] = 0.0f; c[j] = -INFINITY;
      }
    }
```

- **Gotchas**: (a) `tf_sigmoid_precise` (`moe_route.metal:59-63`) evaluates `exp(|x|)` and flips, to stay accurate
  for large negative logits; (b) the second half of this kernel (`:121-164`) does something beyond routing — it
  builds the **pick lists** (`MINE` / `THEIRS` / `LIDS` / `LMEM`) partitioning experts between this Mac and a peer
  via `simd_prefix_exclusive_sum`, because MoE weights can be split across machines; (c) here `c[j] = sc[j] + BIAS[id]`
  ranks by *sigmoid + bias* whereas `moe_ref.topk` ranks by *logit + bias* — the two are only order-equivalent
  because sigmoid is monotone, but the fp32 rounding differs, which is why each backend has its own reference.

### 7d. Nemotron's single-kernel router (a smaller, self-contained read)

- **Anchor**: `zig/kernels/metal/nemotron/route.metal:17-48`
- **What it does**: The entire router for 128 experts / top-6 in 30 lines: sigmoid every logit, add the bias for
  ranking, then 6 rounds of `simd_max` + `simd_min`-on-ties, writing ids to `IDX` and normalised scaled weights to `WT`.
- **Excerpt**:

```cpp
  const uint lane = thread_index_in_simdgroup;
  const uint r = threadgroup_position_in_grid.x;
  float sel[NE / 32], prob[NE / 32];
  for (int j = 0; j < NE / 32; j++) {
    const int e = int(lane) + 32 * j;
    const float g = float(G[int(r) * NE + e]);
    prob[j] = 1.0f / (1.0f + metal::exp(-g));
    sel[j] = prob[j] + bias[e];
  }
  float total = 0.0f;
  float picked[K];
  for (int k = 0; k < K; k++) {
    float best = -INFINITY;
    int best_e = 1 << 20;
    for (int j = 0; j < NE / 32; j++) {
      const int e = int(lane) + 32 * j;
      if (sel[j] > best) { best = sel[j]; best_e = e; }
    }
    const float top = simd_max(best);
    const int winner = simd_min(best == top ? best_e : (1 << 20));
    float p = 0.0f;
    for (int j = 0; j < NE / 32; j++) {
      if (int(lane) + 32 * j == winner) { p = prob[j]; sel[j] = -INFINITY; }
    }
    p = simd_sum(p);
    picked[k] = p;
    total += p;
    if (lane == 0) IDX[int(r) * K + k] = uint(winner);
  }
  if (lane == 0) {
    const float denominator = total + 1e-20f;
    for (int k = 0; k < K; k++) WT[int(r) * K + k] = picked[k] / denominator * scaling[0];
  }
```

- **Gotchas**: `simd_min(best == top ? best_e : 1<<20)` is a tie-break idiom worth a callout — broadcast the max,
  then take the smallest id that achieved it. `+ 1e-20f` guards an all-zero denominator. The template instantiation
  at `route.metal:52` hard-codes `<128, 6>`: experts and top-k are **compile-time** constants, one specialisation
  per model.

### 7e. "Only a few experts run per token"

- **Anchor**: `zig/kernels/metal/nemotron/expert_up.metal:66-84`
- **What it does**: This is the payoff of routing. Grid z indexes **unique picked experts** `u` (and returns
  immediately if `u >= UCOUNT[0]`, so unpicked experts cost nothing). `START[u]`/`COUNT[u]` give that expert's slice
  of the `MEMBERS` list — the token slots routed to it — so an expert's 4-bit weights are loaded **once** and
  reused across all its tokens. `p / TOPK` recovers which token row a slot belongs to.
- **Excerpt**:

```cpp
  const uint lane = thread_index_in_simdgroup;
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  const int first = START[u], last = START[u] + COUNT[u];

  #pragma clang loop unroll(disable)
  for (int m = first; m < last; m++) {
    const int p = MEMBERS[m];
    float acc[RPS];
    tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                          X + size_t(p / TOPK) * K, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) {
        const float h = metal::max(float(bfloat(acc[j])), 0.0f);
        ACT[size_t(p) * N + row0 + j] = bfloat(h * h);
      }
  }
```

- **Teaching value**: An MoE layer holds 128 experts' weights but each token touches 6. The saving is realised by a
  *gather-by-expert* data layout: group the token slots by expert first, then loop experts and stream their tokens.
  The weights are the expensive thing to move, so the loop nests weights outermost.
- **Gotcha**: the activation is ReLU² (`h*h` after `max(·,0)`), and `#pragma clang loop unroll(disable)` is there
  on purpose — unrolling would blow the register budget that keeps `acc[RPS]` resident.

---

## 8. KV cache — layout, growth, memory formula

- **Anchor**: `zig/src/families/qwen3_5/state.zig:14-30` (layout), `:94-99` (allocation), `:204-208` (the formula)
- **What it does**: Each stream owns a `Cache` whose `blocks[c.layers]` is a tagged union: `.delta` (recurrence
  state + conv window) for the 18 linear layers, `.attention` (two buffers, keys and values) for the 6 attention
  layers. A KV buffer is `capacity × kv_heads × head_dim × 2` bytes — i.e. indexed `[kv_head][position][head_dim]`,
  **head-major**, so one head's whole history is contiguous (exactly what `attn_partial.metal:54`'s
  `(h * CAP + rows[j]) * D` expects). Growth is one `len += rows` per committed step; the memory was preallocated at
  `capacity`, so a token never allocates.
- **Excerpt** — the formula:

```zig
/// Bytes a stream's Cache takes at `capacity` rows: its logits, delta layers' states, attention layers' KV.
pub fn cacheBytes(capacity: usize) u64 {
    var bytes: u64 = c.vocab * 2;
    for (0..c.layers) |i| bytes += if (c.linear(i)) delta_bytes + conv_bytes else 2 * capacity * c.kv_heads * c.head_dim * 2;
    return bytes;
}
```

the allocation it describes (`state.zig:94-99`):

```zig
            } else {
                const bytes = capacity * c.kv_heads * c.head_dim * 2;
                b.* = .{ .attention = .{ .keys = try out.memory.alloc(bytes), .values = try out.memory.alloc(bytes) } };
            }
```

and the paged layout, which states the per-token cost directly (`state.zig:13-30`):

```zig
/// Layers with a KV cache (every fourth): a page holds 256 positions of each.
pub const attention_layers = c.layers / 4;
comptime {
    var n = 0;
    for (0..c.layers) |i| n += @intFromBool(!c.linear(i));
    if (n != attention_layers) @compileError("Qwen3.5's attention layers are every fourth");
}

/// Attention layer `a`'s keys in a page (its values the part after): [layer][keys, values][kv head][256][head_dim].
pub fn pagePart(a: usize, values: bool) usize {
    return (a * 2 + @intFromBool(values)) * c.kv_heads * kvp.page_tokens * c.head_dim * 2;
}

/// A paged stream: its fixed part the recurrences, conv windows and logits, its keys and values pages of the pool.
pub fn pagedLayout() kv.Layout {
    const fixed = c.vocab * 2 + (c.layers - attention_layers) * (delta_bytes + conv_bytes);
    return .{ .fixed_bytes = fixed, .token_bytes = attention_layers * 2 * c.kv_heads * c.head_dim * 2, .page_tokens = kvp.page_tokens, .snap_bytes = 0 };
}
```

- **The formula, instantiated**: `bytes_per_token = attention_layers × 2 × kv_heads × head_dim × dtype_bytes`
  = `6 × 2 × 2 × 256 × 2` = **12,288 bytes/token (12 KiB)**. The file's own test asserts exactly this
  (`state.zig:210-213`): *"a stream's cache bytes: about 19.5 MB of states and logits, and 12 KiB of keys and
  values a row"*, with `cacheBytes(1) - cacheBytes(0) == 12_288`. At the 262,144-token max context that is
  ~3.0 GiB of KV per stream.
- **Teaching value**: the generic textbook formula is `layers × 2 × kv_heads × head_dim × context × dtype`; this
  file is that formula as executable code, with `layers` replaced by *layers that actually cache*. The `× 2` is K
  and V; `kv_heads` (not `n_heads`) is where GQA pays off.
- **Gotchas**: (a) the fixed overhead is large and *context-independent* — `cacheBytes(0)` is
  `18_874_368 + 663_552 + 496_640` bytes ≈ 19.5 MB, of which 18.9 MB is the 18 delta layers' recurrence states
  (`delta_bytes = 16 heads × 128 × 128 × 4` = 1,048,576 **each**, in fp32) and 496,640 is the per-stream logits
  buffer `vocab × 2`; (b) there is an optional **paged** mode (`TF_QWEN35_METAL_KV_PAGES=1`) where all attention
  layers point into one shared arena and a per-stream page table maps positions, 256 positions per page — see
  `state.zig:70` and the `paging()` helper at `forward.zig:230-234`; (c) `commit` (`state.zig:107-121`) is what
  advances `len`, and it also snapshots recurrence state so a rejected speculative branch can be rewound
  (`keep`, `state.zig:123-140`).

---

## 9. The layer loop and residual connections

- **Anchor**: `zig/src/families/qwen3_5/forward.zig:87-106`
- **What it does**: The whole transformer, in 20 lines. Embed (`forward.zig:81-86`), then for each of the 24
  blocks: input-norm (fused residual add), the mixer (delta or attention — both write their output into `s.r`,
  the residual buffer), post-norm (fused residual add again), the SwiGLU MLP, and `down` writing back into `s.r`.
  After the loop, a final norm and the `lm_head` projection to `vocab` logits. `head == .last` is the decode fast
  path: it offsets into the last row only, so one row of logits is produced instead of `total`.
- **Excerpt**:

```zig
    for (m.weights.blocks, 0..) |block, i| {
        enc.norm(s.h, if (i == 0) null else s.r, block.input_norm, s.x, total);
        switch (block.mixer) {
            .delta => |d| delta(enc, s, d, i, segments, record, total),
            .attention => |a| attention(enc, s, a, i, segments, total),
        }
        enc.norm(s.h, s.r, block.post_norm, s.x, total);
        enc.projection(block.gate, s.x, 0, s.gate, 0, total);
        enc.projection(block.up, s.x, 0, s.up, 0, total);
        enc.pipe("mlp_act");
        e.setBuffer(s.gate, 0, 0);
        e.setBuffer(s.up, 0, 1);
        e.setBuffer(s.act, 0, 2);
        enc.run(.{ c.intermediate, total, 1 }, .{ 256, 1, 1 });
        enc.projection(block.down, s.act, 0, s.r, 0, total);
    }
    enc.norm(s.h, s.r, m.weights.norm, s.x, total);
    if (head == .all) enc.projection(m.weights.head(), s.x, 0, s.logits, 0, total) else {
        enc.projection(m.weights.head(), s.x, (total - 1) * c.hidden * 2, s.logits, 0, 1);
    }
```

The `norm` helper that hides the residual add (`forward.zig:37-54`):

```zig
    fn norm(self: Encoder, h: Buffer, residual: ?Buffer, weight: wts.Tensor, x: Buffer, rows: usize) void {
        if (residual) |r| {
            self.pipe("norm");
            self.e.setBuffer(h, 0, 0);
            self.e.setBuffer(r, 0, 1);
            self.tensor(weight, 2);
            self.e.setValue(c.eps, 3);
            self.e.setBuffer(h, 0, 4);
            self.e.setBuffer(x, 0, 5);
        } else {
            self.pipe("norm_nores");
            self.e.setBuffer(h, 0, 0);
            self.tensor(weight, 1);
            self.e.setValue(c.eps, 2);
            self.e.setBuffer(x, 0, 3);
        }
        self.run(.{ 128, rows, 1 }, .{ 128, 1, 1 });
    }
```

- **Teaching value**: There is **no `+` operator for the residual anywhere in the loop.** Two buffers carry the
  whole topology: `s.h` is the residual stream, `s.r` is "what the last sublayer produced". The residual add
  happens inside the *next* norm kernel. Every sublayer reads `s.x` (normalised) and writes `s.r` (a delta); the
  norm then does `h += r` and re-normalises. Pre-norm residual architecture, zero extra memory traffic.
- **Gotchas**: (a) `if (i == 0) null else s.r` — layer 0 has nothing to add yet, so it dispatches a *different
  kernel* (`norm_nores`) rather than adding a zeroed buffer; (b) the sublayer dispatch is a `switch` on a tagged
  union, because 18 of 24 layers aren't attention at all; (c) the whole 24-layer forward is **one command buffer**
  on a `.serial` encoder (`forward.zig:79-80`, committed at `:107`), so the GPU is handed the entire model in one
  submission; (d) for the attention block, the full sub-sequence is readable at `forward.zig:168-228`:
  Q/K/V projections → per-head QK-norm → RoPE+cache-write → `attn_partial` → `attn_merge` → a **sigmoid output
  gate** (`qwen35_attention_gate`, `layout.metal:76-86`, reading the second half of the 4096-wide `qraw` as the
  gate — this is Qwen3.5's `attn_output_gate: true`) → the `out` projection; (e) `forward.zig:209` is where the
  attention scale enters, as the literal `@as(f32, 1.0 / 16.0)` = `1/√256`.

---

## Cross-reference: reference ↔ production

| Concept | Reference (readable) | Production (fast) | Key difference |
|---|---|---|---|
| Attention | `core/attention_ref.zig:5-29` (f64, 2-pass) | `kernels/metal/ops/sdpa.metal:10-19`, `qwen3_5/attn_partial.metal:44-71` | 2-pass max→softmax becomes 1-pass online rescaling; `simd_sum` dot products; 3-level partial merge |
| Multi-head | `core/joint_schema/reference_attn.zig:8-32` | GQA head mapping `ops/sdpa.metal:31` | explicit head loop becomes grid dims; `heads` becomes `head / gqa` |
| QK-norm + RoPE | `core/qk_ref.zig:6-24` (f64, cos/sin supplied) | `qwen3_5/layout.metal:20-45`, `:47-54` | cos/sin recomputed per element in-kernel; fused into the KV write |
| RoPE frequencies | `core/rope_freq.zig:5-12` (torch-exact) | inlined constant `pow(10000000.0f, -i/32)` at `layout.metal:50` | table lookup replaced by recomputation — cheaper than a load |
| YaRN | `core/rope_yarn.zig:5-26` | CUDA only; `families/qwen3_5/cuda_rope_table.zig` | computed once at load, baked into the cos/sin table |
| RMSNorm | `families/qwen3_5/cpu_kernels.zig:186-191` | `qwen3_5/norm.metal:17-41` | residual add fused in; simdgroup tree reduction |
| LayerNorm | `core/joint_schema/reference.zig:68-82` | `kernels/metal/core/qwen_vision.metal:36-40` | vision towers only; LLM blocks are RMSNorm |
| MoE routing | `core/moe_ref.zig:8-31` (sigmoid+bias), `core/route_ref.zig:19-32` (softmax) | `kernels/metal/core/moe_route.metal:73-119`, `nemotron/route.metal:17-48` | linear scan becomes `simd_shuffle_xor` butterfly; selection fused with cross-machine pick lists |
| Expert MLP | — | `nemotron/expert_up.metal:66-84` | gather-by-expert so weights load once per expert, not per token |

## Suggested reading order for a lesson

1. `core/attention_ref.zig` — 20 lines, f64, no GPU. Attention, complete.
2. `core/joint_schema/reference_attn.zig:8-32` — where `1/√d_h` and the head split come from.
3. `kernels/metal/ops/sdpa.metal:10-19` — the single function that turns (1) into flash attention.
4. `kernels/metal/ops/sdpa.metal:31` + `qwen3_5/attn_partial.metal:32` — GQA is one division (or one multiply).
5. `core/qk_ref.zig:6-24`, then `qwen3_5/layout.metal:47-54` — RoPE, reference then fused.
6. `qwen3_5/norm.metal:17-41` vs `core/joint_schema/reference.zig:68-82` — RMSNorm vs LayerNorm, side by side.
7. `qwen3_5/mlp_act.metal:12-18` + `forward.zig:94-101` — SwiGLU in two files.
8. `core/moe_ref.zig:8-31`, then `nemotron/route.metal:17-48`, then `nemotron/expert_up.metal:66-84` — route, select, run only the chosen.
9. `families/qwen3_5/state.zig:204-208` — the memory formula, as code with a test that pins 12 KiB/token.
10. `families/qwen3_5/forward.zig:87-106` — the whole model, with residuals hiding inside the norms.
