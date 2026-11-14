# Context-edges code map — extending the window, and surviving overflow

Paths are relative to the TensorFold repo root (`TensorFold/`).
Every anchor below was read; line numbers are correct as of this read.

Companion to `map-forward.md` (which covers RoPE *inside* the forward pass: fused into the KV-cache write,
`theta = 1e7`, rotate-half, 64 of 256 dims rotating) and `map-serving.md` (the refusal arithmetic).
This map covers only the two **edges**: what happens when you ask for positions the model never saw, and
what happens when the conversation no longer fits.

---

## 0. The numbers, in one table

| Quantity | Value | Defined in |
|---|---|---|
| RoPE base (`theta`) for Qwen3.5/3.6 | 1e7 | `zig/src/families/qwen3_5/cuda_rope.zig:7` |
| rotary dims / frequency pairs | 64 dims = 32 pairs | `zig/src/core/rope_yarn.zig:6` (`const dim: f64 = 64`), loop `:17` |
| YaRN `beta_fast`, `beta_slow` | 32, 1 | `rope_yarn.zig:13-14` (`dimOf(32, …)`, `dimOf(1, …)`); enforced `cuda_rope.zig:63-65` |
| YaRN attention factor (mscale) | `0.1·ln(factor) + 1` | `rope_yarn.zig:25` |
| …at factor 4 | 1.138629436111989 | asserted `rope_yarn.zig:37` |
| Qwen3.6 trained window | 262,144 | `cuda_rope.zig:106-108` (fixture), test `:125` |
| Qwen3.6 YaRN×4 window | 1,048,576 | `cuda_rope.zig:106`, test `:124` |
| rope table on device | `[33]f32` = scale, then 32 freqs | `cuda_rope.zig:12` |
| `--compact-at auto` reserve | `max(15% of window, 16384)` | `zig/src/server/compact.zig:47-52` |
| …when 16384 > window | 15% of window | `compact.zig:58-63` |
| note budget | `min(8192, 80% of reserve)` | `compact.zig:54-56` |
| `--compact-keep` default | `min(20000, window/4)` | `compact.zig:65-68` |
| note generation cap | 1024 tokens per chunk | `compact.zig:261` |
| note prose cap | "Under 800 words" | `compact.zig:20` |
| stored note file size cap | 1 MiB read limit | `zig/src/server/compact_mem.zig:62` |
| Nemotron Metal fallback window | 262,144 (`1 << 18`) | `zig/src/families/nemotron/limits.zig:4` |
| …advertised `context_length` | 262,193 | `limits.zig:6-12`, asserted `zig/src/server/status_routes.zig:354` |

**Live engine, observed** (`curl -s http://127.0.0.1:8080/v1/models`, this session):
`"context_length": 8192`, `"max_model_len": 8192`, `"capabilities": {… "context_fitted": false …}`,
`"id": "local-model"`. So the running instance is an 8k window, *not* memory-fitted. I could not observe
whether it was launched with `--compact-at`; `/v1/models` does not report serve flags.

---

## 1. Per-dimension RoPE frequencies, concretely

- **Anchor**: `zig/src/core/rope_freq.zig:1-11`
- **What it does**: The one place in the repo that computes transformers' default inverse frequencies.
  `out[i] = 1 / theta^(2i/dim)`, with the rounding order pinned: the exponent `2i/dim` is rounded to **fp32
  first**, `pow` is evaluated in **f64** and rounded **once** to fp32, then the reciprocal is taken in fp32.
  Every family that ropes (Gemma 4's CUDA path, LFM3, the XPU shim, the GLM vision tower) imports this rather
  than rolling its own.
- **Excerpt**:

```zig
//! transformers' default RoPE inverse frequencies in fp32, 1 / theta^(f32(2i) / dim), each step rounded as torch's.
const std = @import("std");

/// out[i] = 1 / theta^(2i / dim), i < out.len: an fp32 exponent, theta^e rounded once to fp32, an fp32 reciprocal.
pub fn inverseFrequencies(theta: f64, dim: usize, out: []f32) void {
    for (out, 0..) |*f, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
        const p: f32 = @floatCast(std.math.pow(f64, theta, e));
        f.* = 1.0 / p;
    }
}
```

- **The wavelength view** (derived from this formula, not quoted): pair `k` turns at angle
  `pos · theta^(-2k/dim)`, so its **wavelength is `λ_k = 2π · theta^(k/32)` tokens**. For Qwen3.5/3.6
  (`theta = 1e7`, 32 pairs): `λ_0 = 2π ≈ 6.3` tokens, `λ_16 ≈ 2π·10^3.5 ≈ 19,870`, `λ_31 ≈ 2π·10^6.78 ≈ 3.8e7`
  tokens — the slowest pair has not completed one revolution in any window anyone serves. That spread is the
  whole reason context extension is possible *and* the reason it is lossy.
- **Teaching value**: RoPE is a bank of 32 clocks, each ticking `10^(7/32) ≈ 1.66×` slower than the last. Position
  is encoded by *which combination of clock hands* you see. Fast clocks resolve "the previous few tokens"; slow
  clocks resolve "roughly where in the document".
- **Gotchas**: (a) the function takes `out.len`, not `dim`, as the pair count — `dim` is only the exponent
  denominator, which is how partial rotary (`rotary_dim = 64` inside `head_dim = 256`) and Gemma's proportional
  form both reuse it; (b) `proportional` (`rope_freq.zig:14-19`) fills the first `rotated` pairs and **zeroes the
  rest** — a zero inverse frequency is an identity rotation, which is how "only some dims rotate" is expressed as
  data rather than as a branch; (c) nothing here depends on the window length. The frequencies are a property of
  the *model*, not of the request.

---

## 2. The planted test: why `pow` in fp32 is the wrong answer

- **Anchor**: `zig/src/core/rope_freq.zig:77-91`
- **What it does**: A deliberately *negative* test. It computes the same table the obvious way — a single
  `std.math.pow(f32, …)` — and asserts that this disagrees with torch's bits for **exactly 15 of the 32
  frequencies**, by **up to 5 ulp**. It is not checking that the fast form is good enough; it is nailing down
  how wrong it is, so that anyone who "simplifies" `inverseFrequencies` into one fp32 `pow` fails loudly instead
  of silently shipping a model whose positions drift from the reference implementation.
- **Excerpt**:

```zig
test "planted: the fp32-pow form misses transformers' words for theta 1e7 over 64 (15 of 32, up to 5 ulp)" {
    var want: [32]f32 = undefined;
    inverseFrequencies(1e7, 64, &want);
    var differ: usize = 0;
    var most: u32 = 0;
    for (want, 0..) |w, i| {
        const f32pow: f32 = 1.0 / std.math.pow(f32, 1e7, @as(f32, @floatFromInt(2 * i)) / 64.0);
        const a: u32 = @bitCast(w);
        const b: u32 = @bitCast(f32pow);
        if (a != b) differ += 1;
        most = @max(most, if (a > b) a - b else b - a);
    }
    try std.testing.expectEqual(@as(usize, 15), differ);
    try std.testing.expectEqual(@as(u32, 5), most);
}
```

- **What it guards**: the *positive* tests next to it (`rope_freq.zig:25-35`, `:37-46`, `:48-57`, `:59-80`) hold
  hard-coded `u32` bit patterns captured from "transformers 5.22.1's default rope init, torch 2.15.0 CPU"
  (comment at `:24`). Those prove equality. The planted test proves that equality is *fragile* — that there is a
  natural refactor one keystroke away which breaks it. Together they say: this table is a bit-exact contract with
  PyTorch, and the contract lives in the rounding order, not in the mathematics.
- **Teaching value**: "The math is right" is not the same as "the bits are right". A 5-ulp error in an inverse
  frequency is a phase error that grows linearly with position — invisible at token 10, measurable at token
  100,000. This is why a reimplementation that passes a tolerance test can still fail a long-context test.
- **Gotchas**: (a) `most = 5` is the maximum *integer distance between bit patterns*, which for same-exponent
  floats equals ulps — a clever, allocation-free way to express "5 ulp"; (b) the Gemma proportional test at
  `rope_freq.zig:59-80` carries an even sharper comment: torch's **AVX2 and AVX-512** `pow` give word 29 one ulp
  up (`0x3e561912`) while the scalar path gives `0x3e561911`, and the repo asserts the **scalar** value — so
  "torch's bits" is itself not one answer, and the file picks a side on purpose; (c) the Metal mirror of this
  computation is a one-line kernel using `metal::precise::pow` (`zig/kernels/metal/gemma4/glue.metal:421-425`),
  i.e. the GPU is required to match MLX's `pow`, not torch's.

---

## 3. YaRN frequencies: which clocks get slowed and which do not

- **Anchor**: `zig/src/core/rope_yarn.zig:1-26`
- **What it does**: transformers' `_compute_yarn_parameters`, truncated-ends variant, for exactly 64 rotary dims.
  `dimOf(rot, b, max_pos)` inverts the wavelength formula: it returns the (fractional) pair index whose
  wavelength is `max_pos / rot`. With `beta_fast = 32` and `beta_slow = 1` that gives `low` = the pair whose
  wavelength is `trained/32`, and `high` = the pair whose wavelength is `trained`. For each pair it builds two
  candidates — `extra` (the untouched frequency) and `inter` (the frequency divided by `factor`) — and blends
  them with a linear ramp `ef` that is 1 below `low` and 0 above `high`. Finally it returns the **attention
  factor** `0.1·ln(factor) + 1`.
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

- **The three bands, for Qwen3.6 (`theta = 1e7`, `trained = 262144`, `factor = 4`)**. Evaluating `dimOf` gives
  `low = floor(14.24) = 14` and `high = ceil(21.12) = 22` (**derived** from the formula above — the repo does
  not print these two numbers). The split is **independently confirmed bit-for-bit** by comparing the two
  verbatim torch tables that *are* in the repo, `rope_freq.zig:26-33` (plain) against `rope_yarn.zig:29-35`
  (YaRN×4):

| Pairs | Count | `ef` | What happens | Bit evidence |
|---|---|---|---|---|
| `k = 0 … 14` | **15** | 1 | **untouched** — pure extrapolation, identical to plain RoPE | words 0–14 are byte-identical in both tables (e.g. `0x3f800000`, `0x3a6301e2`) |
| `k = 15 … 21` | **7** | (0,1) | **blended** ramp between the two forms | e.g. `k=15` plain `0x3a092e02` vs YaRN `0x39f8a364`, which is neither the plain word nor plain/4 |
| `k = 22 … 31` | **10** | 0 | **divided by exactly `factor`** — pure interpolation | `k=22` plain `0x37812dab` vs YaRN `0x36812dab`: same mantissa, exponent down by 2 = ÷4 |

- **Teaching value**: This *is* YaRN's defining trick, and it falls straight out of §1's wavelength picture.
  The fast clocks already distinguish neighbouring tokens perfectly well, and slowing them down would destroy
  local resolution — so leave them alone. Only the clocks whose wavelength is comparable to the trained window
  (between `trained/32` and `trained`) are stretched, because those are the ones that would otherwise wrap into
  angles the model never saw. Naive "linear position interpolation" divides *all 32*; YaRN divides *10 of 32*
  and blends 7. That is the whole difference.
- **Gotchas**: (a) `pos_freq` here is computed with the **same double-rounding dance** as §1
  (`@as(f64, @as(f32, @floatCast(2*kf/dim)))`) — the YaRN path is bit-pinned to transformers too, asserted at
  `rope_yarn.zig:27-38` against "transformers 5.19.0's `yarn` rope init under torch 2.14.1"; (b) `if (low ==
  high) high += 0.001` is a divide-by-zero guard, not a modelling choice — without it a degenerate band makes
  `(kf-low)/(high-low)` a NaN that poisons every frequency; (c) the function **hard-codes 64 rotary dims** and
  returns `[32]f32`, so it is Qwen-geometry-specific, not a generic utility; (d) the scale is `0.1·ln(factor)+1`
  — a *log* in the factor, so ×4 buys only +13.9% attention temperature and ×16 only +27.7%; (e) FlashNext keeps
  a byte-identical copy at `zig/src/families/flashnext/rotary_config.zig:4-25` because its startup receipts hash
  it (noted in `docs/recipes/adding-a-zig-family.md:50`).

---

## 4. The attention factor is applied to cos and sin, not to the scores

- **Anchor**: `zig/kernels/cuda/common/mla.cuh:317-331`
- **What it does**: The rotary kernel itself. One block per (row, head), one thread per pair; thread `i` reads
  GPT-J-paired elements `2i, 2i+1`, computes the angle as `inv_freq[i] · (pos[0] + r)` in fp32, and then
  **multiplies both `cos` and `sin` by `scale`** before the rotation. That `scale` is YaRN's attention factor.
  Folding the mscale into the rotation means the *keys and queries* come out pre-scaled, so the softmax
  temperature change costs zero extra instructions in attention.
- **Excerpt**:

```cpp
__device__ __forceinline__ void rope(const __nv_bfloat16* x, int64_t xs, int64_t xh, __nv_bfloat16* out, int64_t os,
                                     int64_t oh, const int* pos, const float* inv_freq, float scale) {
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int i = threadIdx.x;  // pair i at angle fp32 inv_freq[i] (pos + r); explicit rn ops, no fma contraction
  const __nv_bfloat16* src = x + r * xs + h * xh + 2 * i;
  const float x0 = bf(src[0]), x1 = bf(src[1]);
  float s, c;
  sincosf(__fmul_rn(inv_freq[i], (float)((int64_t)pos[0] + r)), &s, &c);
  s = __fmul_rn(s, scale);  // YaRN's attention factor on cos and sin (exact at 1)
  c = __fmul_rn(c, scale);
  __nv_bfloat16* dst = out + r * os + h * oh + 2 * i;
  dst[0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
  dst[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
}
```

- **Teaching value**: Extending context changes **two** things, not one: the *angles* (frequencies) and the
  *temperature* (attention factor). Longer contexts mean more keys competing in one softmax, so the logits are
  scaled up slightly to keep the distribution from flattening. And the cheapest place to put a scalar that
  multiplies every query and key is inside the rotation you were already doing.
- **Gotchas**: (a) the comment "exact at 1" is load-bearing — plain RoPE passes `scale = 1.0`, and `x·1.0f` is
  exactly `x` in IEEE, so **the same kernel serves both paths with no branch and no bit change**; (b) note
  `rope` is a *rotate-pairs* (GPT-J, `2i`/`2i+1`) kernel here, whereas `map-forward.md` documents the Qwen3.5
  Metal path as rotate-**half** — the two conventions coexist in the repo and are not interchangeable;
  (c) every multiply is an explicit `__fmul_rn`/`__fsub_rn` with the comment "no fma contraction": the compiler
  is forbidden from fusing, because fusing would change the bits; (d) `pos[0]` is read from device memory and
  the row offset `r` added — so one launch ropes a whole chunk at consecutive positions.

The Qwen3.5/3.6 MTP head takes the same table through a templated variant
(`zig/kernels/cuda/qwen36_mtp.cu:128-132`): `Y == nullptr` selects plain `theta` rope, otherwise `Y[0]` is the
attention factor and `Y[1..33]` the frequencies, "as the target's rope".

---

## 5. `--rope-yarn FACTOR`: resolution, the window it buys, and every refusal

- **Anchor**: `zig/src/families/qwen3_5/cuda_rope.zig:22-39`
- **What it does**: The single decision point for "what rope does this checkpoint get, and how large a window
  may `--context` ask for". Two sources may name YaRN: the checkpoint's own `rope_parameters`/`rope_scaling`
  block, or the `--rope-yarn` flag. If the config has a block, the flag must agree **exactly** or
  `RopeYarnConflict`; the block's `original_max_position_embeddings × factor` must equal
  `max_position_embeddings` or `UnsupportedRopeScaling`. If only the flag is given, the trained window *is*
  `max_position_embeddings` and the ceiling becomes `max × factor`. `--rope-yarn 1` is explicitly a no-op that
  resolves to plain rope.
- **Excerpt**:

```zig
/// The tower's rope and ceiling: its YaRN block (rope_parameters/rope_scaling) or `flag`; both must name one factor.
pub fn resolve(text: std.json.ObjectMap, flag: ?f64) Error!Resolved {
    const max = try positive(text, "max_position_embeddings");
    const rope = text.get("rope_parameters") orelse return error.BadQwenConfig;
    if (rope != .object) return error.BadQwenConfig;
    if (flag) |f| if (!std.math.isFinite(f) or f < 1) return error.RopeYarnFactor;
    if (try scaling(text, rope.object)) |y| {
        const factor = try number(y, "factor");
        const trained = try positive(y, "original_max_position_embeddings");
        if (!(factor > 1) or @as(f64, @floatFromInt(trained)) * factor != @as(f64, @floatFromInt(max))) return error.UnsupportedRopeScaling;
        if (flag) |f| if (f != factor) return error.RopeYarnConflict;
        return .{ .rope = .{ .yarn = yarn(factor, trained) }, .ceiling = max };
    }
    const f = flag orelse return .{ .rope = .plain, .ceiling = max };
    if (f == 1) return .{ .rope = .plain, .ceiling = max };
    const ceiling = @as(f64, @floatFromInt(max)) * f;
    if (ceiling != @floor(ceiling) or ceiling > std.math.maxInt(u32)) return error.RopeYarnFactor;
    return .{ .rope = .{ .yarn = yarn(f, max) }, .ceiling = @intFromFloat(ceiling) };
}
```

- **Where a factor is validated or refused** (all verified):

| Check | Outcome | Anchor |
|---|---|---|
| not a number (`--rope-yarn four`) | `error.Usage` at CLI parse | `zig/src/server/cli_gpu_flags.zig:19`, test `zig/src/server/cli_test.zig:363` |
| flag listed only on CUDA builds | hidden/refused elsewhere | `zig/src/server/cli.zig:112` (`.native = cuda_build`) |
| non-finite, or `< 1` | `RopeYarnFactor` | `cuda_rope.zig:26` |
| `max × factor` not an integer, or `> u32` max | `RopeYarnFactor` | `cuda_rope.zig:37` |
| flag disagrees with the config's `factor` | `RopeYarnConflict` | `cuda_rope.zig:31` |
| tested bad values `0.5, 0, -4, inf, nan, 1.0000001` | all `RopeYarnFactor` | `cuda_rope.zig:132` |
| `rope_type` is `dynamic`, `longrope`, or any non-`yarn` | `UnsupportedRopeScaling` | `cuda_rope.zig:61`, cases `:146-148` |
| block sets `mscale`, `mscale_all_dim` or `attention_factor` | `UnsupportedRopeScaling` | `cuda_rope.zig:62` |
| `beta_fast ≠ 32` or `beta_slow ≠ 1` | `UnsupportedRopeScaling` | `cuda_rope.zig:63-65` |
| `truncate: false` | `UnsupportedRopeScaling` | `cuda_rope.zig:66` |
| family is not `qwen3_5` / `qwen3_5_moe` on CUDA | refused before any family branch | `zig/src/native/cuda_refusal.zig:24-25`, test `:113-118` |
| backend is Metal | "the Metal engine has no YaRN rope" | `zig/src/native/metal.zig:132-134` |
| backend is HIP | "the native HIP engine has no YaRN rope" | `zig/src/native/hip_families.zig:40` |
| backend is XPU | "the native Intel GPU engine has no YaRN rope" | `zig/src/native/xpu.zig:189-192` |
| `--slide` plus YaRN | `QwenYarnNoLearner` — "the learner's attention ropes without YaRN" | `zig/src/families/qwen3_5/cuda_native_explain.zig:13`, test `cuda_learn_shape.zig:82-84` |
| a YaRN config reaching a path that cannot honour it | `InvalidRotary` / `UnsupportedQwenConfig` rather than silently read as plain rope | `zig/src/families/qwen3_5/config.zig:536-538`, `cuda_draft.zig:581-584` |

- **Teaching value**: "Extend the context" is a *configuration* decision made once at load, not a per-request
  knob. The engine would rather refuse than quietly serve a rope it cannot reproduce exactly — including
  refusing a *parameter it does not implement* (`mscale`, a non-default `beta_fast`) rather than ignoring it.
- **Gotchas**: (a) the ceiling is a hard `u32` product, so `--rope-yarn 4` on a 262,144 checkpoint yields
  exactly 1,048,576 and nothing else; (b) both routes (flag on the stock checkpoint, or a staged `config.json`
  carrying the YaRN block) produce the **same table and the same ceiling** — asserted in
  `cuda_rope.zig:111-130`; (c) `cuda_rope.zig:62` refuses an explicit `attention_factor` — you cannot override
  the mscale, you get `0.1·ln(factor)+1` or nothing; (d) a legacy `rope_scaling` block with `type: "default"`
  is *also* refused (`cuda_rope.zig:170-172`), because only a scaled rope has that block at all.

### 5b. Uploading the table invalidates every captured CUDA graph

- **Anchor**: `zig/src/families/qwen3_5/cuda_rope_table.zig:7-23`
- **What it does**: Copies the 33 floats to the device, records the pointer on the engine, prints the startup
  line, and then **destroys every captured CUDA graph** — because a graph bakes in the table's address.
- **Excerpt**:

```zig
/// The engine ropes with `rope` from its next forward: the table uploaded, graphs captured with the old one dropped.
pub fn set(e: *Engine, rope: Rope) !void {
    free(e);
    e.g.rope = rope;
    switch (rope) {
        .plain => {},
        .yarn => |y| {
            e.rope_buf = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(&y.table));
            e.rope_table = e.rope_buf.?.ptr;
            std.debug.print("info: YaRN rope, factor {d} over the trained {d} positions: windows up to {d}\n", .{ y.factor, y.trained, e.g.context });
        },
    }
    // a captured graph keeps the table address it was captured with: each shape captures again on its next run
    var it = e.graphs.valueIterator();
    while (it.next()) |g| g.deinit();
    e.graphs.clearRetainingCapacity();
}
```

- **Teaching value**: Context extension is 132 bytes of data (`33 × f32`) plus one scalar multiply. Everything
  else about the model is unchanged — same weights, same kernels, same geometry. The cost is not in the engine;
  it is in the model's ability to use angles it was never trained on.
- **Gotchas**: `rope_table == 0` **is** the plain-rope signal (`cuda_engine.zig:96`, set to 0 at
  `:219`), and the table is uploaded "before any weight or forward" (`cuda_engine.zig:224`). The table is also
  hashed into the engine's on-disk identity (`cuda_disk.zig:156-159` hashes the literal table bytes, "the actual
  kernel inputs, independent of a finite probe") — so a YaRN engine never reuses a plain engine's cached
  artifacts.

---

## 6. Where the window is enforced: `--context N` and the refusal

### 6a. Choosing the window at startup

- **Anchor**: `zig/src/native/cuda_window.zig:40-47`
- **What it does**: Four lines that define `--context`. No flag → the family default, clamped to the model's own
  window. `--context 0` → the model's whole window (and an error if the checkpoint names none).
  `--context N` → `N`, but **never past the model's native window**, which is itself the YaRN ceiling from §5
  when YaRN applies (`cuda_native.zig:71-73`, `cuda.zig:184-188`).
- **Excerpt**:

```zig
/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
pub fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}
```

- **Gotchas**: `contextWindow` is "prompt **plus reply**" tokens, not prompt tokens. The default is
  `@min(default, native)`, so on a 262k checkpoint you get the family default (16,384 in the test at
  `cuda_window.zig:50`), **not** 262k — a long window is opt-in because it costs memory per stream
  (`docs/recipes/qwen3.6-moe.md:211`: "A 1M window takes 22.2 GiB a stream, so a GB10 admits two").

### 6b. The per-request refusal

- **Anchor**: `zig/src/server/chat.zig:170-177`
- **What it does**: After rendering the prompt (template, tools and thinking tokens included), compute
  `room = window - n`. If `room < 1` the prompt leaves no space for even one reply token → refuse. If the client
  asked for `max_tokens` and it exceeds `room`, refuse and **say both repairs**: the largest prompt that would
  fit, and the largest reply that would fit. Otherwise silently clamp `max_tokens` to `room`.
- **Excerpt**:

```zig
    const window: i64 = srv.info.context_window;
    const n: i64 = @intCast(rendered.ids.len);
    if (window > 0) {
        const room = window - n;
        if (room < 1) return cx.fail(.context_length, "{s} {d} tokens{s}, but the rendered prompt has {d} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.", .{ errors.context_limit, window, if (srv.info.capability(.context_fitted)) ", the most this server's memory budget fits" else "", n });
        if (input.max_tokens != null and limit > room) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens and requests {d} reply tokens, which exceeds the context window. Reduce the prompt to at most {d} prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.", .{ errors.context_limit, window, n, limit, @max(0, window - limit), room });
        limit = if (input.fill_room) room else @min(limit, room);
    }
```

- **Teaching value**: The window is a budget over **prompt + reply**, and the prompt that counts is the
  *rendered* one — chat template, tool schemas and thinking scaffolding included. A learner who counts their own
  message characters will always under-estimate.
- **Gotchas**: (a) `errors.context_limit = "This server's maximum context length is"` with the comment
  "OpenAI's wording, which clients match to compact" (`zig/src/server/errors.zig:20`) — the error *string* is a
  compatibility surface; (b) the HTTP shape is `invalid_request_error` with `code:
  "context_length_exceeded"` and a `param` key (`zig/src/server/openai.zig:97-100`); (c) when `window == 0` the
  whole block is skipped and the **engine** refuses later (Nemotron Metal: `limits.zig:14-16`,
  `error.PromptTooLong`); (d) a streamed request that overflows still opens the stream and reports the error
  after the role chunk (`openai.zig:512`, `516`, test `stream_preflight_test.zig:280-282`).

### 6c. What the server advertises — and why that is not a promise

- **Anchor**: `zig/src/server/status_routes.zig:104-122`
- **What it does**: `/v1/models` emits `context_length` and `max_model_len` as **equal** integers, taken from
  `reported_context_window orelse context_window`, and omits both when that is zero.
- **Excerpt**:

```zig
    const window = info.reported_context_window orelse info.context_window;
    if (window > 0) {
        try m.put(a, "context_length", try json.intValue(a, window));
        try m.put(a, "max_model_len", try json.intValue(a, window));
    }
```

- **The caveat, stated by the repo** — `README.md:277-281`:
  > These are the loaded engine's effective prompt-plus-reply token limit, including any startup memory fitting;
  > they are not the default reply limit. An unspecified engine window omits both fields. Clients should
  > still handle per-request memory refusals: **an advertised context is not a guarantee that every workload fits.**
- **Gotchas**: (a) `reported_context_window` is documented as a "**Discovery-only** limit for an engine that
  keeps its existing prefill admission and error policy" (`zig/src/core/engine_api.zig:176-177`) — i.e. the
  advertised number and the enforced number are *different fields by design*; (b) the concrete divergence:
  Nemotron Metal with no `--context` allocates `262144 + 64 + 1 = 262209` positions and advertises
  `262209 − 16 = 262193` (`zig/src/families/nemotron/limits.zig:2-12`, asserted
  `status_routes.zig:350-356`) while `info.context_window` stays **0** — so the client is told 262,193 and
  §6b's check never runs; (c) the `context_fitted` capability flag says "the context was limited to memory"
  (`docs/api.md:152`), and when it is true §6b's refusal appends ", the most this server's memory budget fits";
  the live engine reports `context_fitted: false` at an 8,192-token window.

---

## 7. Compaction: the thresholds

- **Anchor**: `zig/src/server/compact.zig:47-76`
- **What it does**: All the arithmetic of "when do we compact, and how much room does the summary get".
  `--compact-at auto` reserves `max(15% of window, 16384)` for the reply; `--compact-at F` reserves whatever is
  left above `floor(F × window)`. `reserveInWindow` is the fix for the pathological case where the 16,384 floor
  exceeds the entire window — then the reserve falls back to 15%. The note generation itself gets
  `min(8192, 80% of that reserve)` tokens of budget, and the verbatim tail defaults to
  `min(20000, window/4)`.
- **Excerpt**:

```zig
pub fn reserveTokens(window: i64, at: CompactAt) i64 {
    return switch (at) {
        .auto => @max(@divTrunc(window * 15, 100), 16384),
        .fraction => |f| @max(1, window - threshold(window, f)),
    };
}

pub fn noteBudgetTokens(window: i64, at: CompactAt) i64 {
    return @min(8192, @max(1, @divTrunc(reserveInWindow(window, at) * 4, 5)));
}

/// Auto's 16384 floor applies only while the window holds it; a smaller window reserves fifteen percent.
fn reserveInWindow(window: i64, at: CompactAt) i64 {
    const reserve = reserveTokens(window, at);
    if (at != .auto or reserve < window) return reserve;
    return @max(@as(i64, 1), @divTrunc(window * 15, 100));
}

pub fn keepTokens(window: u32, override: ?u32) usize {
    if (override) |k| return k;
    return @min(20000, @as(usize, window) / 4);
}
```

- **Worked numbers, asserted in-repo** (`zig/src/server/compact_test.zig:247-256`):
  `reserveTokens(1000, .auto) = 16384`; `reserveTokens(200000, .auto) = 30000`;
  `noteBudgetTokens(200000, .auto) = 8192`; `reserveTokens(2000, {fraction 0.5}) = 1000`;
  `noteBudgetTokens(2000, {fraction 0.5}) = 800`; `keepTokens(100000, null) = 20000`;
  `keepTokens(400, null) = 100`; `keepTokens(400, 7) = 7`.
- **Validation**: `--compact-at` must be `auto` or a fraction in `(0, 1]`, else
  `"argument --compact-at: expected auto or a fraction in (0, 1]: '1.5'"`; `--compact-keep` must be an integer
  `≥ 0` (`zig/src/server/cli.zig:474-486`). Each later occurrence replaces **both** fields, so `--compact-at
  auto --compact-at 0.5` ends at the fraction — a fixed bug, `CHANGELOG.md:111`, tested
  `compact_at_cli_test.zig:11-22`.
- **Gotchas**: (a) `reserveTokens(1000, .auto)` returns `16384`, *larger than the window* — the fallback lives
  in `reserveInWindow`, not in `reserveTokens`, so the public function really can return nonsense and callers
  must use the right one; this was a real bug (`CHANGELOG.md:1653`); (b) `--compact-keep 0` is legal and means
  "keep no tail by token budget" — `keptStart` still falls back to the **first** cuttable message
  (`compact.zig:226`), so at least one turn survives; (c) the trigger is `prompt + wanted_reply`, where
  `wanted_reply` is the request's `max_tokens` or the server default (`compact.zig:78-88`).

---

## 8. Compaction: what is kept verbatim, and the tool-call invariant

- **Anchor**: `zig/src/server/compact.zig:208-226` (`keptStart`), with `prefixLen`/`isLead`/`isCut` at `:378-392`
- **What it does**: Decides the boundary `tail`. The **leading** `system`/`developer` messages are the prefix
  and are always kept (`prefixLen`). Then only `user` and `assistant` messages are candidate **cut points**
  (`isCut`) — `tool` messages are *not*. Walking turn-group boundaries backwards, it accumulates token counts
  for each group `[start_g, start_{g+1})` and returns the first boundary at which the accumulated tail reaches
  `keep`.
- **Excerpt**:

```zig
pub fn keptStart(srv: *Server, a: Allocator, msgs: []const Value, keep: usize, marker: ?usize) chat.Failure!usize {
    const prefix = prefixLen(msgs);
    var starts: std.ArrayList(usize) = .empty;
    for (msgs, 0..) |m, i| {
        if (i < prefix or (marker != null and i == marker.?)) continue;
        if (isCut(m)) try starts.append(a, i);
    }
    if (starts.items.len == 0) return prefix;
    var acc: usize = 0;
    var g = starts.items.len;
    while (g > 0) {
        g -= 1;
        const from = starts.items[g];
        const to = if (g + 1 < starts.items.len) starts.items[g + 1] else msgs.len;
        acc += try countRange(srv, a, msgs, from, to, marker);
        if (acc >= keep) return from;
    }
    return starts.items[0];
}
```

- **Teaching value**: "Never separate a tool call from its result" is not a special case or a post-hoc check —
  it is a consequence of the cut-point *alphabet*. Because `tool` is excluded from `isCut`, a boundary can never
  land between an assistant's `tool_calls` and the `tool` messages answering it. Choosing what may be a boundary
  is more robust than checking boundaries afterwards.
- **Gotchas**: (a) the invariant is tested directly —
  `compact_test.zig:279-311`, "the kept span never starts between a tool call and its results", with
  `[user, assistant+tool_calls, tool, tool, user]` and `keep = 8` returning index **1** (the assistant), never 2
  or 3; (b) `prefixLen` only counts system/developer messages at the **head**; a later `system` message is
  rewritten to `srv.late_system` by `messages.normalize` (`zig/src/server/messages.zig:26-27`, `:162`) and is
  therefore *not* protected; (c) the count is a per-message tokenization of `"role: content" + " call NAME ARGS"`
  (`compact.zig:368-374`, `:341-353`), i.e. an approximation of the rendered prompt, not the prompt itself;
  (d) a **split turn**: if `tail` lands on an `assistant` message, `splitIndex` (`:405-416`) finds that turn's
  `user` message and the cut-off front half of the *current* turn is summarized separately under `split_task`
  (`:23-26`) and inserted as "Earlier in this turn:" before the kept reply (`:296`).

---

## 9. Compaction: who writes the memory note

- **Anchor**: `zig/src/server/compact.zig:251-286` (`oneNote`), with the task prompt at `:15-21`
- **What it does**: **The served model writes its own note.** `oneNote` builds a three-message conversation —
  a `system` message holding `note_task`, an optional `user` message `"Previous note:\n…"`, and a `user` message
  `"Messages to add:\n<transcript>"` — then runs it through the ordinary `chat.prepare`/`chat.generate` path as
  an `own = true` request with `enable_thinking: false`, `temperature: 0`, `max_tokens: 1024`. If even that
  prompt will not fit, it degrades: retry at `max_tokens = 1`, then size the reply to the room actually left,
  then give up and return the previous note unchanged.
- **Excerpt** (the task prompt, `compact.zig:15-21`):

```zig
const note_task =
    \\Rewrite the memory note for this conversation so it keeps everything still needed.
    \\Markdown sections: Goal, Constraints and preferences, Progress, Decisions, Next steps, Critical context.
    \\Progress lists done, in progress and blocked. Decisions keep every earlier one and its reason.
    \\Next steps stay in order. Critical context keeps exact names, paths, numbers and error text.
    \\Under 800 words.
;
```

And the generation itself (`compact.zig:252-262`, `:282-286`):

```zig
    const a = cx.a;
    var list: std.ArrayList(Value) = .empty;
    try list.append(a, try msg(a, "system", task));
    if (prev.len > 0) try list.append(a, try msg(a, "user", try std.fmt.allocPrint(a, "Previous note:\n{s}", .{prev})));
    const script = try transcript(a, msgs);
    try list.append(a, try msg(a, "user", try std.fmt.allocPrint(a, "Messages to add:\n{s}", .{script})));
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "temperature", .{ .float = 0 });
    var prepared = chat.prepare(srv, cx, .{ .own = true, .messages = .{ .array = list.items }, .fields = .{ .object = fields }, .max_tokens = 1024, .temperature = 0 }, gone) catch |e| blk: {
    …
    prepared.request.drafts = true; // the note stays a drafted generation when replies pass --no-drafts
    prepared.drafts = true;
    const reply = try chat.generate(srv, cx, prepared, null, gone);
    return .{ .text = reply.content, .tokens = reply.completion_tokens };
```

- **The two mechanical sections** the server appends itself, *not* the model: `"## Files read and changed\n"`
  (`compact.zig:28`) and `"## Kept reply context\n"` (`:29`), assembled by `finishNote` (`:321-339`). File paths
  are harvested from `tool_calls` arguments under the keys `path`, `file_path`, `filename`, `file`, `paths`
  (`compact.zig:448-460`) — so "the files the tool calls read or changed" is a **mechanical extraction**, not
  something the model is trusted to remember.
- **Teaching value**: Compaction is inference, not string processing. Reducing a conversation is itself a
  generation — which means it consumes tokens, takes time, and a compacted request runs **extra forward passes
  before its reply** (`docs/api.md:872`). The reply you get was preceded by one or more invisible ones.
- **Gotchas**: (a) `own = true` marks it "the server's own request (a compaction summary …): it never takes
  `--instruct-sampling`'s set" (`zig/src/server/chat.zig:30`) — operator sampling defaults must not leak into
  the summary; (b) long middles are chunked: `update` (`:228-248`) packs as many messages as fit in
  `chunkRoom = window − noteBudget` (`:90-94`) and calls `oneNote` repeatedly, **each call seeing the previous
  note**, so a very long history is folded incrementally; (c) drafting is force-enabled for the note even under
  `--no-drafts`, because the note is not the client's reply; (d) `withoutExtra` (`:341-354`) strips the two
  mechanical sections from the previous note *before* feeding it back, so the model never re-summarizes its own
  appendices — and it handles both `\n\n` and `\r\n\r\n` (tested `:587-592`).

---

## 10. Compaction: updating the previous note instead of starting over

- **Anchor**: `zig/src/server/compact.zig:136-152` (the head of `rebuild`)
- **What it does**: Before summarizing anything, `rebuild` looks for a note it already has, from two places:
  the stored `--compact-memory` slot (if its hash still matches) or a **marker message** in the conversation the
  client sent back. The found note becomes `prev` and is handed to `oneNote` as "Previous note:". `skip` then
  excludes the messages that note already covers, so they are never re-summarized. The marker is the literal
  string `"Memory of the earlier conversation:"` (`compact.zig:13`).
- **Excerpt**:

```zig
fn rebuild(srv: *Server, cx: *Cx, input: chat.Input, gone: anytype, at: CompactAt, msgs: []Value) chat.Failure!Ready {
    const a = cx.a;
    const marker = try findMarker(a, msgs);
    const slot = stored.loadMem(srv, a, msgs);
    const hash_ok = if (slot) |m| stored.hashes(a, msgs, m) else false;
    var prev: []const u8 = "";
    if (hash_ok) {
        prev = slot.?.note;
    } else if (marker) |i| {
        prev = try markerBody(a, msgs[i]) orelse "";
    }
    const skip: usize = if (hash_ok) slot.?.covered else if (marker) |i| i + 1 else 0;
    const prefix = prefixLen(msgs);
    var tail = try keptStart(srv, a, msgs, keepTokens(srv.info.context_window, srv.config.compact_keep), marker);
    if (tail < prefix) tail = prefix;
    const split_at = splitIndex(msgs, prefix, tail, marker);
    const middle_to = split_at orelse tail;
```

- **The retry ladder** (`compact.zig:158-205`): after building the note, `assemble` reconstructs the message
  list — leading prefix, then the note as one `srv.late_system` message beginning with the marker line, then
  optionally the split summary, then the verbatim tail — and tries `chat.prepare`. If it *still* does not fit:
  first drop the split summary, then advance `tail` to the next cut point, summarize the newly-dropped span into
  the note, and try again. The loop is bounded by `guard = msgs.len + 2`. If even that fails, the original
  request's own `context_length` refusal is what the client sees (test
  `compact_test.zig:330-344`, "a request that cannot fit after compaction is still context_length_exceeded").
- **The stored-memory format**: `--compact-memory DIR` writes `<key>.md` (the note) and `<key>.json`
  (`{"covered":N,"hash":"<sha256 hex>"}`), where `key` is the first 16 hex chars of SHA-256 over the **first
  system message and first user message** only (`compact_mem.zig:11-20`), and `hash` is SHA-256 over the
  compact JSON of the first `covered` messages (`:72-80`). A later request reuses the note only if that prefix
  hashes identically (`:22-26`). Files are written through a private atomic replace, directory mode `0o700`
  (`compact_mem.zig:45-49`, `:65-70`).
- **Teaching value**: The note is **state carried in the conversation itself**, not server-side session state.
  A client that echoes the compacted messages back keeps its memory with no cooperation from the server; the
  `--compact-memory` directory is only an optimization that avoids regenerating a note the client discarded.
- **Gotchas**: (a) `findMarker` returns the **last** marker found (`:394-400`), and the marker message is then
  excluded everywhere by the `marker` parameter threaded through `collect`, `countRange`, `keptStart`,
  `countBetween` — one `?usize` carries "pretend this message does not exist"; (b) the marker is matched by
  **prefix of the content**, and content assembled from `text` parts counts too (`markerBody` at `:402-414`,
  test `:559-566`); a stray `\r` after the marker line is stripped (test `:568-574`); (c) the note is injected
  with role `srv.late_system`, which is `"system"` or `"user"` depending on whether the chat template keeps a
  later system message in place (`zig/src/server/server.zig:559-562`) — so on some templates the memory note
  arrives as a *user* message; (d) if everything is dropped, `assemble` appends one empty `user` message
  (`:302`) rather than sending an empty array.

---

## 11. Determinism of compaction

- **Anchor**: `zig/src/server/compact.zig:1` (the module's own claim) with `:258-261` (what enforces it)
- **What it does**: The file's first line states the property, and the mechanism is three fields on the note
  request: `enable_thinking: false`, `temperature: 0` in the template fields, and `temperature = 0` on the
  request itself, with `max_tokens = 1024`.
- **Excerpt**:

```zig
//! Note and reply are ordinary greedy or keyed generations, so the same request compacts and answers the same way.
```

```zig
    const fields = try json.newObject(a);
    try fields.put(a, "enable_thinking", .{ .bool = false });
    try fields.put(a, "temperature", .{ .float = 0 });
    var prepared = chat.prepare(srv, cx, .{ .own = true, .messages = .{ .array = list.items }, .fields = .{ .object = fields }, .max_tokens = 1024, .temperature = 0 }, gone) catch |e| blk: {
```

- **Why that is sufficient**: `docs/request-seeds.md:6` — "**Greedy requests do not draw a seed.**" A
  temperature-0 generation has no entropy source, so the note is a pure function of (template, weights,
  transcript, previous note). The surrounding selection logic is already deterministic: token counts, the cut
  point, the ramp of retries. Hence "the same request gives the same compaction and the same reply"
  (`README.md:263`).
- **Direct test**: `compact_test.zig:313-328`, "the compacted prompt fits and the same request compacts the
  same way" — a 1,200-pad conversation on a 2,000-token window is sent twice and the two `note` strings are
  compared with `expectEqualStrings`; the same test also asserts `first.prompt_tokens + 32 <= 2000`.
- **Teaching value**: Determinism is not a feature bolted on; it is the *absence* of a seed. Everything that
  could have introduced variation (sampling, thinking, operator sampling defaults via `--instruct-sampling`) is
  explicitly switched off for this one internal generation.
- **Gotchas**: (a) "or **keyed**" in the module doc covers the `--prompt-seed` mode, where a sampled draw is
  keyed from prompt IDs plus `TENSORFOLD_SEED_SALT` (`docs/request-seeds.md:13-14`) — determinism survives that
  too, by a different route; (b) determinism is per-server: it does **not** promise the same note across
  different `--context` values, since `keepTokens` and `chunkRoom` both depend on the window; (c) the note is a
  *drafted* generation (`compact.zig:282-283`), and the repo's claim elsewhere is that drafted replies equal
  their undrafted ones bit for bit (`docs/recipes/qwen3.6-moe.md:216`), which is what makes forcing drafts on
  safe here.

---

## 12. What the client sees

- **Anchor**: `zig/src/server/compact.zig:122-134` (`stamp`)
- **What it does**: Writes the context accounting into the reply's `runtime` object — which is serialized under
  the top-level key `tensorfold`. `context_window` and `context_used` go on **every** reply once compaction is
  enabled; the nested `compaction` object appears **only** when a compaction actually ran.
- **Excerpt**:

```zig
pub fn stamp(cx: *Cx, reply: *chat.Reply, s: Stamp) chat.Failure!void {
    if (reply.runtime != .object) return;
    const o = reply.runtime.object;
    try o.put(cx.a, "context_window", try json.intValue(cx.a, s.window));
    try o.put(cx.a, "context_used", try json.intValue(cx.a, s.used));
    if (!s.compacted) return;
    const c = try json.newObject(cx.a);
    try c.put(cx.a, "messages_compacted", try json.intValue(cx.a, s.messages_compacted));
    try c.put(cx.a, "kept_messages", try json.intValue(cx.a, s.kept_messages));
    try c.put(cx.a, "note_tokens", try json.intValue(cx.a, s.note_tokens));
    try c.put(cx.a, "note", .{ .string = s.note });
    try o.put(cx.a, "compaction", .{ .object = c });
}
```

- **Confirmed field names** (code, docs and tests agree):

| Field | Path in the reply | Meaning | Anchor |
|---|---|---|---|
| `context_window` | `tensorfold.context_window` | `srv.info.context_window` | `compact.zig:125` |
| `context_used` | `tensorfold.context_used` | `prepared.prompt_len`, the rendered prompt actually run | `compact.zig:126`, set at `:107`, `:113`, `:116` |
| `messages_compacted` | `tensorfold.compaction.messages_compacted` | messages between prefix and tail that became the note | `compact.zig:129`, counted `:181` |
| `kept_messages` | `tensorfold.compaction.kept_messages` | messages from tail to the end, kept verbatim | `compact.zig:130`, counted `:182` |
| `note_tokens` | `tensorfold.compaction.note_tokens` | completion tokens spent generating the note(s) | `compact.zig:131` |
| `note` | `tensorfold.compaction.note` | the note text itself | `compact.zig:132` |

  `docs/api.md:877-882` states the same list. The two-level nesting is exercised in
  `compact_test.zig:496-497` and `:503-504` (`p.get("tensorfold")` → `t.get("compaction")`).
- **Teaching value**: The server tells you it threw something away, and gives you the replacement text verbatim.
  Compaction is observable, not silent — a client can diff `note` across turns to watch the conversation's
  memory evolve.
- **Gotchas**: (a) with compaction **off** the `runtime` object is untouched — no `context_window`, no
  `context_used` (test `compact_test.zig:261-277`: `plain.runtime.get("context_window") == null`). These fields
  are a compaction-mode feature, not universal telemetry; (b) `context_used` is the prompt after compaction, so
  on a compacted turn it is *smaller* than the conversation the client sent; (c) on a streamed reply the stamp
  is written onto the **final** chunk (`zig/src/server/openai.zig:579`); (d) compaction applies to chat
  completions with a `messages` array and to the Responses API; "a completions request with a plain string
  prompt is never compacted" (`docs/api.md:883-884`), which matches the guard `input.messages != .array` at
  `compact.zig:110`.

---

## 13. The honest limits

Four places in the repo state a *measured* or *explicitly disclaimed* limit rather than a hope.

### 13a. The startup line that hands quality to the operator

- **Anchor**: `zig/src/native/cuda_window.zig:34-38`
- **Excerpt**:

```zig
/// The startup line of a window past the checkpoint's trained one, which only the family's switch serves.
pub fn pastTrained(a: Allocator, window: i64, trained: i64, switch_name: []const u8) ![]const u8 {
    const words = "{s}: a {d}-token window, past the checkpoint's trained {d} tokens; past them the quality is the operator's to check";
    return std.fmt.allocPrint(a, words, .{ switch_name, window, trained });
}
```

  Asserted verbatim at `cuda_window.zig:60-69`. The engine will open the window and will **not** claim the
  output is good.

### 13b. The cost of static YaRN on short prompts

- **Anchor**: `docs/recipes/qwen3.8-flash-next.md:131-143`
- **Quoted**: "`--context` past the trained 262,144 tokens, up to 1,048,576, rotates with static YaRN x4
  (transformers' `_compute_yarn_parameters`: factor 4 over the trained window, beta 32 and 1, cos and sin scaled
  by 0.1 ln 4 + 1) in every rotary kernel … **for every prompt**; at 262,144 or less the plain rotary runs as
  before. **Qwen recommends YaRN only where long prompts need it, since static scaling can cost short texts.**"
  Same point for the MoE family, `docs/recipes/qwen3.6-moe.md:207-208`: "YaRN turns every position, so replies
  differ from plain rope at any length; without the flag or the block, every byte is as before."
- **This is the core trade**: static YaRN is not conditional on prompt length. Turning it on changes the answer
  to a 10-token question.

### 13c. Measured long-context behaviour

- **Anchor**: `docs/recipes/qwen3.8-flash-next.md:139-146` and `docs/recipes/qwen3.6-moe.md:214-220`
- Flash Next on an M5 Ultra, 1,048,576- and 262,144-token sessions: needles **found** at 138,830, 553,139 and
  1,032,827 prompt tokens, at 34 s / 169 s / 417 s to first token; one stream decodes 137–216 tok/s at the long
  window against 142–214 tok/s at 262,144.
- Qwen3.6-35B-A3B on one GB10 with `--rope-yarn 4 --context 1048576`: a passphrase 5% into a 300,000-token
  prompt returns after 173 s of prefill (1,740 tok/s); a 1,048,512-token prompt prefills in 1,546 s (680 tok/s),
  returns the passphrase, decodes 64 tokens at 31 tok/s to the window's last position, and "one token more is
  refused with the server's context-window error."
- So the measured cost of 4× context is **not** accuracy collapse on a needle test; it is ~26 minutes of prefill
  and an order-of-magnitude drop in prefill throughput. Note what is *not* measured, stated plainly
  (`qwen3.6-moe.md:220`): "The MTP drafter's cost over a continuation of many thousands of tokens is not
  measured."

### 13d. A numerical check that the rope is still right past the trained window

- **Anchor**: `zig/src/families/qwen3_5/cuda_mtp_yarn_check.zig:1-31` and `:50-72`
- **What it does**: Uploads the factor-4/262,144 table, absorbs three rows at position **300,000**, and checks
  each KV head's ring key against a float64 recomputation of the YaRN rotation, inside an explicitly derived
  error bound. It prints "PASS mtp YaRN: ring keys at 300,000 to 300,002 against float64 YaRN rope (worst … of
  the bound)".
- **Excerpt**:

```zig
    const start: u64 = 300000;
    const rows = 3;
    try d.absorb(ctx, tap, start, rows, null, ids[1 .. 1 + rows]);
…
            for (0..64) |dim| {
                const a = n[dim % 32];
                const b = n[dim % 32 + 32];
                const angle: f32 = @as(f32, @floatFromInt(pos)) * t[1 + dim % 32];
                const cs = @cos(@as(f64, angle)) * t[0];
                const sn = @sin(@as(f64, angle)) * t[0];
                const want = if (dim < 32) a * cs - b * sn else b * cs + a * sn;
```

- **Teaching value — and the honest boundary**: this test proves the **arithmetic** is correct at position
  300,000. It proves nothing about whether the *model* is correct there. That gap is exactly what §13a hands to
  the operator and what §13b warns costs short prompts. A learner should leave with: the engine can guarantee
  bit-level fidelity to a reference formula at any position; it cannot guarantee the trained weights interpret
  those angles usefully.
- **Note the rotate-half structure visible in the check**: `n[dim % 32]` and `n[dim % 32 + 32]` — the first 32
  of the 64 rotary dims pair with the second 32, matching `map-forward.md`'s rotate-half finding, and one
  frequency `t[1 + dim % 32]` serves two dims.

---

## 14. Suggested lesson ordering

Twelve steps, from "what a frequency is" to "what the server throws away". Each names the **one** interactive
control or visual that makes the idea land.

| # | Anchor | The idea | The one control / visual |
|---|---|---|---|
| 1 | `zig/src/core/rope_freq.zig:1-11` | RoPE is 32 clocks, each ~1.66× slower than the last | **Log-scale wavelength ruler**: 32 dots at `λ_k = 2π·θ^(k/32)`, with a draggable `theta` slider (1e4 → 1e7). The dots fan out; a vertical line marks "the trained window". Learner sees which clocks are inside the window and which have never completed a turn. |
| 2 | `zig/src/core/rope_freq.zig:77-91` | bit-exactness is the contract; the obvious refactor breaks it | **32-cell bit-diff strip**: one cell per frequency, green = identical to torch's word, red = differs, with the ulp distance on hover. A toggle flips between `inverseFrequencies` and the one-line fp32 `pow`; 15 cells turn red and "max 5 ulp" appears. |
| 3 | `zig/src/core/rope_yarn.zig:1-26` | YaRN slows only the slow clocks | **`factor` slider (1 → 16) over the §1 ruler**, with the three bands shaded. Dots 0–14 stay pinned, 22–31 slide left by exactly `factor`, 15–21 slide partially. The `low`/`high` markers move as the learner changes `trained`. |
| 4 | `zig/kernels/cuda/common/mla.cuh:317-331` | extension also changes attention temperature, and it is folded into cos/sin | **A single read-out of `0.1·ln(factor)+1`** beside the factor slider from step 3, next to the literal line `s = __fmul_rn(s, scale)` highlighted — plus a "factor = 1" button that shows the value snap to exactly 1.0 and the kernel become a no-op. |
| 5 | `zig/src/families/qwen3_5/cuda_rope.zig:22-39` | the window is a product, decided once at load | **Two-input form: `max_position_embeddings` × `--rope-yarn`** → ceiling, with live refusal messages. Entering `0.5`, `nan`, or a factor that disagrees with the config's block shows the exact error name (`RopeYarnFactor`, `RopeYarnConflict`). |
| 6 | `zig/src/families/qwen3_5/cuda_rope_table.zig:7-23` | context extension is 132 bytes of data | **A hex dump of the `[33]f32` table**, cell 0 tinted differently (the attention factor) — and a counter "bytes changed in the model: 0". |
| 7 | `zig/src/native/cuda_window.zig:40-47` | `--context` is prompt **plus** reply, and clamped | **Three-slider budget bar** (`requested`, `native`, `default`) that renders the chosen window and flips to the error name when `requested > native`. |
| 8 | `zig/src/server/chat.zig:170-177` | the refusal is arithmetic over the *rendered* prompt | **A stacked bar** `[template | tools | messages | thinking | reply]` against the window, with a `max_tokens` slider. Crossing the line prints the real error string, including both suggested repairs. |
| 9 | `zig/src/server/status_routes.zig:104-122` + `families/nemotron/limits.zig:1-16` | advertised ≠ enforced | **Side-by-side cards**: "what `/v1/models` says" (262,193) vs "what the engine allocates" (262,209) vs "what `context_window` is" (0). One button, "send a 262,200-token prompt", shows which of the two checks actually fires. |
| 10 | `zig/src/server/compact.zig:47-76` | the trigger is a reserve, and the reserve has a pathological case | **Window-size slider (512 → 262,144) with an `auto` / `fraction` toggle**, plotting reserve, note budget and keep side by side. Dragging below ~109,227 shows the 16,384 floor take over; dragging to 1,000 shows the 15% fallback rescue it. |
| 11 | `zig/src/server/compact.zig:208-226` | the tool-call invariant is a consequence of the cut alphabet | **A clickable message strip** (`system, user, assistant+call, tool, tool, user, …`) where legal cut points glow. Dragging the `--compact-keep` slider moves the boundary and it visibly *skips over* the `tool` messages. |
| 12 | `zig/src/server/compact.zig:122-134` + `:251-286` | the model summarizes itself, deterministically, and says so | **A live reply inspector** showing the `tensorfold` block, with a "send the same request again" button that diffs the two `note` strings and reports "identical" — the determinism claim, demonstrated rather than asserted. |

**Closing frame for the lesson, in one sentence each:**
- *Why a 32k-trained model can answer at 128k*: the 15 fastest of its 32 position clocks are left exactly as
  trained, and only the 10 slowest are slowed by the extension factor, with 7 blended between — so local
  structure is untouched and only the "roughly where am I" signal is reinterpreted
  (`rope_yarn.zig:13-23`, bands cross-checked against `rope_freq.zig:26-33` vs `rope_yarn.zig:29-35`).
- *What it costs*: every prompt rotates differently, so short answers change too
  (`docs/recipes/qwen3.8-flash-next.md:134-135`); prefill throughput falls (1,740 → 680 tok/s between 300k and
  1M, `docs/recipes/qwen3.6-moe.md:216-219`); and the quality past the trained window is, in the engine's own
  startup words, "the operator's to check" (`cuda_window.zig:36`).
- *What the server throws away when a conversation no longer fits*: the messages between the leading
  system/developer prefix and the last `--compact-keep` tokens of `user`/`assistant` turns — never splitting an
  assistant's tool call from its `tool` results — replaced by one message beginning "Memory of the earlier
  conversation:" that the model itself wrote greedily under a six-section task prompt, plus a mechanically
  extracted list of the file paths its tool calls named (`compact.zig:208-226`, `:251-286`, `:288-302`,
  `:448-460`).

---

## 15. Explicitly not verified

- Whether the **live** engine on `127.0.0.1:8080` was started with `--compact-at`, `--compact-keep` or
  `--compact-memory`: `/v1/models` does not expose serve flags, and I did not send a generation request. The
  compaction behaviour above is read from source and from the in-repo host tests, not observed live.
- `low = 14` / `high = 22` for `(1e7, 262144, 4)` are **my evaluation** of `rope_yarn.zig:13-14`; the repo does
  not print them. The resulting three-band split *is* verified, by bit-comparing the two torch tables the repo
  does contain (`rope_freq.zig:26-33` against `rope_yarn.zig:29-35`): pairs 0–14 byte-identical, pairs 22–31
  exactly one quarter (same mantissa, exponent −2), pairs 15–21 neither.
- No measured accuracy-vs-length curve (e.g. perplexity or RULER scores by length) exists in the repo that I
  found; the long-context evidence is needle-retrieval plus timings, not a degradation curve.
