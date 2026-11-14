# Qwen3.5 family anatomy — `families/qwen3_5` end to end

Paths are relative to the TensorFold repo root (`TensorFold/`).
Every anchor below was read; line numbers are correct as of this read.

**Read `notes/map-forward.md` first.** It already covers, and this map does *not* re-derive:
the pinned constants (`config.zig:7-21`), the admission check (`config.zig:95-104` — see the
correction below), the layer loop (`forward.zig:87-106`), residual-add fusion into the norm kernel,
RoPE fused into the KV write (`layout.metal:47-73`), the attention kernels (`attn_partial.metal`,
`attn_merge.metal`), the SwiGLU MLP (`mlp_act.metal:12-18`), and the KV formula (`state.zig:204-208`).
This map is the rest of the family: **every file, the delta-net layers, the full state model, the
exact kernel dispatch order, drafting, and the tests.**

**Two corrections to `map-forward.md`** (both re-verified by grep at read time):
- the admission check is `config.zig:97-104` (`const dims = .{` is line 97, `rms_norm_eps` is 104), not `:95-104`;
- the `paging()` helper is `forward.zig:233-237`, not `:230-234`. `forward.zig:209` (`1.0/16.0`),
  `config.zig:7-21`, `config.zig:37-39` and `state.zig:204-208` are all exactly right.

---

## 0. The first surprise: this is three engines in one directory

`zig/src/families/qwen3_5/` holds **180 files** (177 `.zig`, 2 JSON fixtures, 1 Metal-side
`kernels.zig` lives elsewhere). Only **~22** of them are the Metal Qwen3.5-2B model that the course
targets. The rest are the CUDA engine for the *same family tree* at other sizes (Qwen3.8-27B dense,
Qwen3.6-35B-A3B MoE) and a HIP backend.

| Group | Files | What it is |
|---|---|---|
| Metal 2B (the lesson path) | 22 | `qwen3_5.zig` + config/weights/model/kernels/forward/state/backend/prefill/snapshot/qualify/draw/tests |
| Shared host loader `weights/` | 11 | safetensors index, HF/ROCm tensor names, host-side projection decoding |
| CUDA engine | 112 (`cuda.zig` + 111 `cuda_*.zig`) | 2B/27B/35B on NVIDIA: engine, caches, pages, MTP, DFlash2 drafter, EXL3/ModelOpt readers, trainer |
| HIP backend `backend/hip/` | 33 | the same model on AMD: `forward/`(6) `model/`(3) `engine/`(16) `draft/`(8) |
| Fixtures | 2 | `fixtures/lighton-ocr-{0.8b,4b}.json` real `config.json` copies (104 / 112 lines) |

- **Teaching value**: "one model implementation" is not one file set — it is *one architecture
  described once (`config.zig`) and then re-implemented per backend*. The Metal path is the small,
  readable one; the CUDA path is 10× the code for the same mathematics plus serving machinery.
- **Gotcha**: `config.zig` is shared by **all** backends (Metal, CUDA host checks, HIP), which is why
  a 584-line "config" file sits next to a 237-line "whole forward pass".

---

## 1. File inventory

### 1a. Metal Qwen3.5-2B — the lesson path

| File | Lines | Purpose (first line of its own doc comment) |
|---|---:|---|
| `qwen3_5.zig` | 8 | Family entry: "Native Metal Qwen3.5-2B, with its tied affine head and hybrid recurrent/attention state." Re-exports config/weights/kernels/Model/state/forward/backend. |
| `config.zig` | 584 | "Qwen3.5-2B MLX affine admission: pinned geometry and formats, tied or untied head." Constants + `parse` + fixtures + refusal tests. |
| `model.zig` | 46 | "A native Qwen3.5-2B model: validated config, shared affine checkpoint views and Metal pipelines." |
| `weights.zig` | 101 | "Views into one checkpoint buffer: a tied head aliases the packed embedding; an untied one is its own projection." |
| `kernels.zig` | 81 | "Qwen pipelines use the existing row decoder's fixed arithmetic, including on GPUs without tensor units." Compiles + indexes all pipelines. |
| `forward.zig` | 237 | "Shared row projections with segment-local recurrence and causal attention, encoded through the native Metal queue." |
| `state.zig` | 213 | "Per-stream committed recurrence and KV state, with a shared pool of window snapshots for partial acceptance." |
| `backend.zig` | 256 | "Qwen recurrence, windows and keyed draws behind the existing lane engine's backend interface." |
| `prefill.zig` | 42 | "Prompt reuse and capture hooks around the unchanged Qwen forward." |
| `snapshot.zig` | 71 | "A committed prompt state in host memory, independent of stream capacity and attention page placement." |
| `qualify.zig` | 65 | "Admit wider native lanes only after real-weight forward rows and committed caches match one-row runs on this GPU." |
| `draw.zig` | 160 | "The host draw over a row of bf16 logits, shared by the Metal and CUDA backends: greedy, keyed sampling or a mask." |
| `draw_topk.zig` | 135 | "A finite BF16 row's bounded top-k, ordered by value then id, with scratch owned by the draw." (`capacity = 128`) |
| `cpu_kernels.zig` | 339 | "The Qwen3.5-2B affine projection and RMS norm on the host's cores: the Metal kernels' rounding points and orders." |
| `paged_test.zig` | 154 | GPU test: paged keys/attention vs the contiguous kernels, bit for bit, through shuffled pages. |
| `draft_prior.zig` | 4 | The DFlash2 drafter's acceptance-by-depth prior (CUDA/HIP only). |
| `loader.zig` | 22 | Host-loader module root (`weights/`), and the test aggregator for it. |
| `hip.zig` | 71 | Host loader bindings for the native HIP engine. |
| `cuda.zig` | 93 | CUDA family entry: "the engine and its native family entry; Metal's 2B is qwen3_5.zig." |
| `config_checks.zig` | 230 | "What a Qwen3.5 config must declare for the HIP engine: activation, gate, state, layers, rope, end ids, sizes." |
| `slicing_decl_test.zig` | 105 | "Qwen3.5's TP cuts as mesh declarations: packed segments, kv heads kept by their query ranks." |
| `pocket_shards_test.zig` | 259 | Living-Weights pocket in a Flash-Next-shaped folder (shared learning test). |

Out-of-directory companions of the Metal path:

| File | Lines | Purpose |
|---|---:|---|
| `zig/kernels/metal/qwen3_5/kernels.zig` | 19 | The generated-kernel table: key, Metal function name, `@embedFile` source, and projection launch geometry. |
| `zig/kernels/metal/qwen3_5/*.metal` | see §7 | 17 kernels in 11 files (7 `qmm_*`, `norm`, `norm_nores`, `gdn_pre/chain/post`, `mlp_act`, `attn_partial`, `attn_merge`, `layout`, `paged`). |
| `zig/kernels/metal/sources.zig:13-18` | — | Registers `qwen35`, `qwen35_layout`, and `qwen35_paged = layout.metal ++ core/kv_pages.h ++ paged.metal`. |
| `zig/src/native/qwen35.zig` | — | The served host: load, stream admission, `TF_QWEN35_METAL_KV_PAGES` page planning, prompt-cache budget. |
| `zig/tests/qwen35_{forward,exact,check,fixtures}.zig` | 35/126/28/118 | See §10. |

### 1b. Shared host loader — `weights/`

| File | Lines | Purpose |
|---|---:|---|
| `weights/table.zig` | 200 | "One tensor table over a checkpoint's safetensors files, with the Python loader's `_Shards` key stripping and widths." |
| `weights/checkpoint.zig` | 328 | "A Qwen3.5 / Qwen3.6 checkpoint on the host: the Python ROCm loader's tensor names, fallbacks and conversions." |
| `weights/host.zig` | 90 | "What the loader hands the GPU, still on the host: the Python TextModel's layers with the bytes its kernels read." |
| `weights/config.zig` | 119 | "The HIP loader's configuration: the family's geometry (`../config.zig`) with the checkpoint's quantization table." |
| `weights/projection.zig` | 71 | "The Python loader's `_packed`, `_float`, `_conv` and `_halves`, and torch's `1 + w` norms: read and checked." |
| `weights/shard.zig` | 96 | "One safetensors file mapped read-only; tensors above core's max rank and unnamed dtypes stay out of its index." |
| `weights/slicing.zig` | 124 | (no doc comment) TP slicing declarations over `config.zig` + `host.zig`. |
| `weights/checkpoint_test.zig` | 464 | Host tests of the index on small synthetic safetensors, against the Python loader's rules. |
| `weights/checkpoint_hf_test.zig` | 307 | Host tests of an HF BF16 checkpoint: names, dtypes, the conv layout, norms and files. |
| `weights/real_test.zig` | 91 | "`TF_QWEN_DIR` names a model directory; every tensor the loader reads resolves." |
| `weights/shard_integrity_test.zig` | 86 | A released layer keeps a heap copy's bytes; a malformed header refuses. |

### 1c. CUDA engine (112 files) — grouped

Engine core: `cuda_engine.zig` 592 ("qlinear projections, qwen35.cu glue, one captured forward over every
stream's window"), `cuda_native.zig` 600 (lane backend), `cuda_forward.zig` 197, `cuda_forward_extra.zig` 14,
`cuda_round.zig` 101, `cuda_round_timing.zig` 44, `cuda_layers.zig` 82, `cuda_scratch.zig` 77,
`cuda_shapes.zig` 122, `cuda_signatures.zig` 95, `cuda_geometry.zig` 546 (reads 2B/27B widths from
`config.json`), `cuda_attention.zig` 29, `cuda_recurrent.zig` 93, `cuda_rope.zig` 173,
`cuda_rope_table.zig` 29, `cuda_prompt.zig` 182, `cuda_prompt_int8.zig` 98, `cuda_prompt_tail.zig` 58,
`cuda_prefill.zig` 186, `cuda_projection.zig` 95, `cuda_projection_many.zig` 140, `cuda_readout.zig` 90,
`cuda_joint_readout.zig` 87, `cuda_penalties.zig` 169, `cuda_routed.zig` 67, `cuda_native_moe.zig` 43,
`cuda_mt2.zig` 50, `cuda_engine_tail.zig` 18, `cuda_completion.zig` 42, `cuda_decide.zig` 148,
`cuda_native_explain.zig` 26, `cuda_vision.zig` 16, `cuda_vision_prompt.zig` 139, `cuda_host_embed.zig` 71,
`cuda_stage.zig` 115, `cuda_stage_forward.zig` 139, `cuda_slide.zig` 130, `cuda_wide.zig` 45.

State / memory: `cuda_cache.zig` 418 ("recurrences, conv tails, keys and values, the drafter's tap ring,
the last window's log"), `cuda_pages.zig` 279, `cuda_prefix.zig` 195, `cuda_reuse.zig` 586,
`cuda_disk.zig` 166, `cuda_reserve.zig` 76, `cuda_warm.zig` 127, `cuda_cost_adapter.zig` 152,
`cuda_cost_identity.zig` 70, `cuda_window_costs.zig` 61, `cuda_startup_profile.zig` 54.

Weights / quant readers: `cuda_weights.zig` 533, `cuda_modelopt.zig` 504 (NVFP4/FP8/bf16),
`cuda_modelopt_experts.zig` 91, `cuda_exl3_weights.zig` 348, `cuda_exl3_admit.zig` 35,
`cuda_expert_pack.zig` 52, `cuda_bf16_bytes.zig` 132, `cuda_bf16_rows.zig` 2.

Drafting (MTP + DFlash2 + DSpine): `cuda_mtp.zig` 564, `cuda_mtp_batch.zig` 258, `cuda_mtp_bf16.zig` 222,
`cuda_mtp_weights.zig` 99, `cuda_mtp_absorb.zig` 54, `cuda_mtp_budget.zig` 41, `cuda_draft.zig` 593,
`cuda_draft_weights.zig` 277, `cuda_draft_frame.zig` 217, `cuda_draft_config.zig` 184,
`cuda_draft_memory.zig` 86, `cuda_draft_precision.zig` 32, `cuda_draft_tree.zig` 20,
`cuda_drafter_open.zig` 92, `cuda_dspine_frame.zig` 138, `cuda_dspine_weights.zig` 113,
`cuda_dspine_config.zig` 40.

Training (Living Weights): `cuda_train.zig` 566, `cuda_train_ops.zig` 420, `cuda_train_load.zig` 147,
`cuda_train_more.zig` 190, `cuda_train_init.zig` 27, `cuda_learn_hooks.zig` 80, `cuda_learn_shape.zig` 88,
`cuda_pocket.zig` 211.

Tests (28 of the 112): `cuda_admit_test` 114, `cuda_bf16_admission_test` 51, `cuda_disk_test` 282,
`cuda_draft_config_test` 112, `cuda_draft_copy_test` 25, `cuda_dspine_config_test` 79,
`cuda_engine_tail_test` 98, `cuda_exl3_admission_test` 55, `cuda_exl3_loader_test` 84,
`cuda_modelopt_moe_test` 129, `cuda_mtp_area_test` 69, `cuda_mtp_batch_check` 155, `cuda_mtp_check` 137,
`cuda_mtp_yarn_check` 76, `cuda_native_grants_test` 26, `cuda_pages_init_test` 164,
`cuda_pages_lifetime_test` 85, `cuda_pend_test` 138, `cuda_pocket_test` 173, `cuda_prefill_test` 316,
`cuda_prefix_fault_test` 183, `cuda_quant_admission_test` 49, `cuda_read_cost_test` 50,
`cuda_request_starts_test` 69, `cuda_reserve_test` 38, `cuda_reuse_pages_test` 122,
`cuda_startup_profile_test` 59, `cuda_train_init_test` 209, `cuda_warm_test` 111.

### 1d. HIP backend — `backend/hip/` (33 files)

`forward/`: `state.zig` 249 (page table + per-linear-layer conv window and state), `window.zig` 195
(a lane round's forward), `forward.zig` 204 (prompt span in steps of SPAN rows), `pages.zig` 131
(reference-counted KV pages), `plan.zig` 164 (device plan; launches depend only on `Shape`),
`round_shape.zig` 37.
`model/`: `bridge.zig` 88 (the forward's view of an uploaded model), `view.zig` 152 ("projections, fp32
norms, conv taps, DeltaNet tables, MLPs and the MTP head"), `weights.zig` 260 (tensors as device buffers).
`engine/`: `engine.zig` 554, `native.zig` 494, `hip_lanes.zig` 523, `hip_prompt_test.zig` 395,
`mtp.zig` 340, `vision.zig` 275, `round.zig` 254, `hip_costs_test.zig` 172, `worker.zig` 146,
`hip_costs.zig` 126, `joint_readout.zig` 84, `hip_prefix.zig` 63, `scratch.zig` 48, `prefix.zig` 41,
`memory.zig` 36 ("bytes of a stream's state and of the prompt cache"), `round_graphs.zig` 12.
`draft/`: `draft.zig` 446, `weights.zig` 240, `frame.zig` 188, `config.zig` 159, `context.zig` 150,
`select.zig` 142, `config_test.zig` 126, `weights_test.zig` 119.

---

## 2. Entry points — one architecture, three engines

- **Anchor**: `zig/src/families/qwen3_5/qwen3_5.zig:1-8`, `cuda.zig:1-4`, `hip.zig:1-4`, and the
  registration at `zig/src/tensorfold.zig:65`
- **What it does**: `qwen3_5.zig` is the Metal 2B surface and nothing else — eight `pub const` lines.
  `cuda.zig` and `hip.zig` are sibling surfaces over the same `config.zig`. `tensorfold.zig:65`
  exposes only the Metal one as `tf.qwen35`, which is why every Metal-side tool writes `tf.qwen35.forward`.
- **Excerpt** (`qwen3_5.zig`, the whole file):

```zig
//! Native Metal Qwen3.5-2B, with its tied affine head and hybrid recurrent/attention state.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const Model = @import("model.zig").Model;
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const backend = @import("backend.zig");
```

- **Teaching value**: a "model" in this engine is five nouns — *config, weights, kernels, state,
  forward* — plus one adapter (*backend*) that hands it to the server. Every family in the repo has
  the same five.
- **Gotchas**: `cuda.zig:1` says it outright ("Metal's 2B is qwen3_5.zig"); the two engines share only
  `config.zig` and `draw.zig`. `model.zig:33` loads the checkpoint with `addFileSelected(..., "", "language_model.")`
  — the vision tower's tensors are *filtered out at map time*, not ignored later.

---

## 3. Checkpoint admission — a whitelist, not a parser

- **Anchor**: `zig/src/families/qwen3_5/config.zig:73-85` (quantization) and `:105-123` (schedule, rope, context)
- **What it does**: `parse` refuses anything that is not exactly this checkpoint. Beyond the dims
  table that `map-forward` covers: the quantization object must be *4 bits, group 64, mode affine* **and
  contain no other keys at all**; the per-layer schedule read from `layer_types` must agree with
  `linear(i) == i % 4 != 3` for all 24 layers; rope must be `default` (no YaRN) with `theta = 1e7`,
  `partial_rotary_factor = 0.25`, `mrope_interleaved = true` and `mrope_section = [11, 11, 10]`;
  `max_position_embeddings` must be ≤ 262144.
- **Excerpt** (`config.zig:73-85`):

```zig
fn quantization(v: std.json.Value) !void {
    const o = try object(v);
    if (try integer(o, "bits") != 4 or try integer(o, "group_size") != group) return error.UnsupportedQwenQuantization;
    if (o.get("mode")) |mode| {
        if (mode != .string or !std.mem.eql(u8, mode.string, "affine")) return error.UnsupportedQwenQuantization;
    }
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size") and !std.mem.eql(u8, k, "mode"))
            return error.UnsupportedQwenQuantization;
    }
}
```

and the schedule / rope / context gate (`config.zig:105-122`):

```zig
    var full: [layers]bool = undefined;
    try layer_pattern.read(&full, text);
    for (full, 0..) |f, i| if (f == linear(i)) return error.UnsupportedQwenGeometry; // the 2B's schedule only
    const rope = try object(try field(text, "rope_parameters"));
    try string(rope, "rope_type", "default");
    checks.defaultRope(text, rope) catch return error.UnsupportedQwenRotary;
    try boolean(rope, "mrope_interleaved", true);
    if (try number(try field(rope, "rope_theta")) != theta or
        try number(try field(rope, "partial_rotary_factor")) != 0.25) return error.UnsupportedQwenRotary;
    const sections = try field(rope, "mrope_section");
    if (sections != .array or sections.array.items.len != 3) return error.UnsupportedQwenRotary;
    for (sections.array.items, [_]i64{ 11, 11, 10 }) |s, expected| {
        if (s != .integer or s.integer != expected) return error.UnsupportedQwenRotary;
    }
    try quantization(root.get("quantization") orelse root.get("quantization_config") orelse return error.UnsupportedQwenQuantization);
    if (root.get("quantization_config")) |q| try quantization(q);
    const context = try integer(text, "max_position_embeddings");
    if (context > 262144) return error.UnsupportedQwenConfig;
```

- **Teaching value**: when every kernel hard-codes 2048/6144/248320/256, the config parser's job is
  not to *configure* anything — it is to **prove the checkpoint matches the compiled-in model** and
  fail loudly otherwise. One `?` of flexibility would mean a silently wrong answer.
- **Gotchas**: (a) an *unknown extra key* inside `quantization` is a refusal — that is how a future
  `mxfp4` export gets rejected instead of mis-read; (b) both `quantization` and `quantization_config`
  are checked when both exist, because MLX writes one and HF writes the other;
  (c) the refusal table in the file's own test (`config.zig:220-228`) enumerates the 9 mutations and
  their errors, including `"mode":"mxfp4"` → `UnsupportedQwenQuantization`, `mrope_section [16,8,8]` →
  `UnsupportedQwenRotary`, `max_position_embeddings 524288` → `UnsupportedQwenConfig`,
  `full_attention_interval 2` → `LayerTypesContradictFullAttentionInterval`, `rope_type yarn` →
  `UnsupportedQwenConfig`; (d) the fixture at `config.zig:126-212` is a real `config.json`, so the
  test proves the *actual* published checkpoint is admitted.

---

## 4. Weight loading for this family

- **Anchor**: `zig/src/families/qwen3_5/weights.zig:54-101` (the whole loader), `:33-52` (the two readers)
- **What it does**: `load` walks the 24 layers once and builds a `[24]Layer` where each layer's
  `mixer` is a tagged union, `delta` or `attention`, chosen by `c.linear(i)`. Every projection goes
  through `mlx_affine.read(ckpt, name, .{ .n, .k, .bits = 4, .group = 64 })` which yields the triple
  (`weight` codes, `scales`, `biases`); every non-projection tensor goes through `expect`, which
  compares **dtype and full shape** and refuses with `error.UnexpectedQwenTensor`. Nothing is copied:
  a `Tensor` is a `(buffer, offset)` view into the mapped checkpoint.
- **Excerpt** — the delta-net layer's tensor names and shapes (`weights.zig:77-97`):

```zig
        if (c.linear(i)) {
            block.mixer = .{ .delta = .{
                .qkv = try projection(ckpt, p, "linear_attn.in_proj_qkv", c.conv_dim, c.hidden),
                .z = try projection(ckpt, p, "linear_attn.in_proj_z", c.hidden, c.hidden),
                .a = try projection(ckpt, p, "linear_attn.in_proj_a", c.linear_heads, c.hidden),
                .b = try projection(ckpt, p, "linear_attn.in_proj_b", c.linear_heads, c.hidden),
                .out = try projection(ckpt, p, "linear_attn.out_proj", c.hidden, c.hidden),
                .conv = try tensor(ckpt, p, "linear_attn.conv1d.weight", .bf16, &.{ c.conv_dim, c.conv_taps, 1 }),
                .norm = try tensor(ckpt, p, "linear_attn.norm.weight", .bf16, &.{c.linear_dim}),
                .a_log = try tensor(ckpt, p, "linear_attn.A_log", .f32, &.{c.linear_heads}),
                .dt_bias = try tensor(ckpt, p, "linear_attn.dt_bias", .bf16, &.{c.linear_heads}),
            } };
        } else {
            block.mixer = .{ .attention = .{
                .q = try projection(ckpt, p, "self_attn.q_proj", 2 * c.hidden, c.hidden),
                .k = try projection(ckpt, p, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden),
                .v = try projection(ckpt, p, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden),
                .out = try projection(ckpt, p, "self_attn.o_proj", c.hidden, c.hidden),
                .q_norm = try tensor(ckpt, p, "self_attn.q_norm.weight", .bf16, &.{c.head_dim}),
                .k_norm = try tensor(ckpt, p, "self_attn.k_norm.weight", .bf16, &.{c.head_dim}),
            } };
        }
```

**The complete tensor name list this engine expects** (prefix `language_model.model.layers.{i}.`):

| Tensor | Shape | dtype | Layers |
|---|---|---|---|
| `input_layernorm.weight` | 2048 | bf16 | all 24 |
| `post_attention_layernorm.weight` | 2048 | bf16 | all 24 |
| `mlp.gate_proj` / `mlp.up_proj` | 6144 × 2048 | affine-4 g64 | all 24 |
| `mlp.down_proj` | 2048 × 6144 | affine-4 g64 | all 24 |
| `linear_attn.in_proj_qkv` | **6144** × 2048 | affine-4 g64 | 18 delta |
| `linear_attn.in_proj_z` | 2048 × 2048 | affine-4 g64 | 18 delta |
| `linear_attn.in_proj_a` / `in_proj_b` | **16** × 2048 | affine-4 g64 | 18 delta |
| `linear_attn.out_proj` | 2048 × 2048 | affine-4 g64 | 18 delta |
| `linear_attn.conv1d.weight` | 6144 × 4 × 1 | bf16 | 18 delta |
| `linear_attn.norm.weight` | **128** (per head dim) | bf16 | 18 delta |
| `linear_attn.A_log` | 16 | **f32** | 18 delta |
| `linear_attn.dt_bias` | 16 | bf16 | 18 delta |
| `self_attn.q_proj` | **4096** × 2048 | affine-4 g64 | 6 attention |
| `self_attn.k_proj` / `v_proj` | 512 × 2048 | affine-4 g64 | 6 attention |
| `self_attn.o_proj` | 2048 × 2048 | affine-4 g64 | 6 attention |
| `self_attn.q_norm.weight` / `k_norm.weight` | 256 | bf16 | 6 attention |

Model-level: `language_model.model.embed_tokens` (248320 × 2048, affine-4), `language_model.model.norm.weight`
(2048, bf16), and optionally `language_model.lm_head` (248320 × 2048, affine-4).

- **Where a mismatch is refused**: three places, all before any GPU work.
  1. `weights.zig:35-38` — wrong dtype/rank/shape → `error.UnexpectedQwenTensor` (logged with the name).
  2. `weights.zig:55-59` — head tiedness disagreement → `error.QwenHeadTiednessMismatch`:
     ```zig
         const ships_head = ckpt.has("language_model.lm_head.weight");
         if (ships_head != !tied) {
             std.log.err("config.json says tie_word_embeddings={} but the checkpoint {s} a language_model.lm_head projection", .{ tied, if (ships_head) "ships" else "lacks" });
             return error.QwenHeadTiednessMismatch;
         }
     ```
  3. a missing tensor → `ckpt.get` fails inside `expect`/`mlx_affine.read`.
- **Teaching value**: the three surprising numbers in that table *are* the architecture.
  `q_proj` is **4096 = 2 × 2048** wide because Qwen3.5 gates attention: half is Q, half is the gate
  (§7, `qwen35_attention_gate`). `in_proj_qkv` is **6144 = 2·(16·128) + 16·128** — Q, K and V for the
  recurrent mixer in one matrix. `in_proj_a`/`in_proj_b` are **16 outputs** — one scalar per head, the
  decay and the write strength.
- **Gotchas**: (a) `A_log` is the only **f32** tensor in the model; everything else is bf16 or 4-bit
  codes; (b) `linear_attn.norm.weight` is 128 long, not 2048 — it is a *per-head-dim* norm shared by
  all 16 heads; (c) `head()` (`weights.zig:28-30`) returns `separate_head orelse embedding`, so the
  tied case reuses the **same buffer and offset** — `tf-qwen35-check` asserts exactly that
  (`zig/tests/qwen35_check.zig:12-22`); (d) `conv1d.weight` is stored `[6144][4][1]`, i.e. channel-major
  with the 4 taps contiguous, which is why `gdn_pre` indexes `CW[c * TAPS + tap]`.

---

## 5. The gated delta-net layers (18 of 24)

### 5.0 What a gated delta-net layer computes

Per layer, per head `h` (16 heads), the layer carries a **matrix** state `S_h ∈ R^{128×128}` (fp32)
instead of a growing list of keys and values. For each token it computes, from the normalised residual
`x`:

| Quantity | How | Width |
|---|---|---|
| `qkv` | one affine-4 projection, then a 4-tap causal depthwise conv + SiLU | 6144 = 2048 (q) + 2048 (k) + 2048 (v) |
| `q`, `k` | L2/RMS-normalised per head, then scaled (`1/dk` for q, `1/√dk` for k) | 16 × 128 each |
| `v` | the conv+SiLU output, unnormalised | 16 × 128 |
| `g` (decay) | `exp(-exp(A_log_h) · softplus(a_h + dt_bias_h))` ∈ (0, 1] | 1 per head |
| `β` (write strength) | `sigmoid(b_h)` ∈ (0, 1) | 1 per head |
| `z` (output gate) | a separate 2048-wide projection of `x` | 2048 |

Then the **delta rule**, in this exact order (`gdn_chain.metal:53-65`):

$$S \leftarrow g\,S,\qquad \delta = \beta\,(v - S k),\qquad S \leftarrow S + \delta\,k^{\!\top},\qquad y = S q$$

and the output is `y`, RMS-normalised per head and gated: `out = silu(z) * norm(y)` — then one
2048×2048 `out_proj` back into the residual stream.

Read it as *associative memory with a write-correction*: `S k` is "what the memory currently returns
for this key", `v - S k` is the error, `β` is how much of the error to write, `g` forgets a little of
everything first. Attention keeps every (k, v) pair and re-reads them all; this keeps one fixed-size
summary and edits it.

### 5.1 `gdn_pre` — conv, SiLU, head norms, gate and β in one kernel

- **Anchor**: `zig/kernels/metal/qwen3_5/gdn_pre.metal:37-48` (conv + SiLU), `:49-59` (q/k norm),
  `:60-69` (v, decay, β), `:71-77` (next conv window)
- **What it does**: one threadgroup of 32 lanes per (head-slot, row), where `head` runs
  `0..47` = 16 q heads, 16 k heads, 16 v heads (`forward.zig:146` launches grid `{32, 3*16, rows}`).
  Each lane owns `PER = 128/32 = 4` channels. For each channel it reads **4 taps** — taps that fall
  before the chunk come from the cached conv state `CS`, later taps from this pass's `QKV` rows —
  multiplies by `CW`, rounds to bf16, applies SiLU, and rounds again. q and k slots then do a
  simd-wide sum of squares for the per-head RMS norm; v slots additionally compute the layer's two
  scalars per head.
- **Excerpt** (`gdn_pre.metal:37-48`, the conv and SiLU):

```cpp
  for (int j = 0; j < PER; j++) {
    const int c = c0 + int(lane) * PER + j;
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int row = windows[w * TAPS + tap];
      const float xv = row < TAPS - 1 ? float(CS[row * C + c]) : float(QKV[(row - (TAPS - 1)) * C + c]);
      acc += float(CW[c * TAPS + tap]) * xv;
    }
    const float conv = float(bfloat(acc));
    const float sig = float(bfloat(1.0f / (1.0f + metal::exp(-conv))));
    vals[j] = float(bfloat(conv * sig));
  }
```

and the gate/β (`gdn_pre.metal:60-69`):

```cpp
  } else {
    const int hv = int(head) - 2 * NK;
    for (int j = 0; j < PER; j++) Vout[(w * NV + hv) * DV + lane * PER + j] = bfloat(vals[j]);
    if (lane == 0) {
      const float s = float(bfloat(float(Ain[w * NV + hv]) + float(DT[hv])));
      const float sp = float(bfloat(metal::max(s, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(s)))));
      G[w * NV + hv] = metal::exp(-metal::exp(float(ALOG[hv])) * sp);
      BETA[w * NV + hv] = bfloat(1.0f / (1.0f + metal::exp(-float(Bin[w * NV + hv]))));
    }
  }
```

- **Teaching value**: the "convolution" in a linear-attention layer is tiny and local — **4 taps,
  depthwise, causal**. It exists so each token's q/k/v can see the 3 tokens before it *without*
  touching the recurrent state. That is why the state has two parts: a matrix and a 3-row window.
- **Gotchas**: (a) `softplus` is written in the numerically stable form
  `max(s,0) + log1p(exp(-|s|))`, not `log(1+exp(s))`; (b) the q scale is `1/dk = 1/128` while the k
  scale is `1/√dk` — **asymmetric**, and both are rounded to bf16 *before* use
  (`gdn_pre.metal:54`), so the kernel reproduces a specific reference's rounding;
  (c) the norm's epsilon is folded as `ss/DK + (1e-6/DK)`, not `+1e-6`; (d) `G` is written as **f32**
  while `BETA` is bf16 — decay multiplies the whole state every step, so its precision matters most;
  (e) lines 71-77 write the *next* conv window (`CO`) in the same kernel: there is no separate
  "commit conv state" kernel on Metal; (f) the `windows` table is built on the host, one `[4]i32` per
  row, at `forward.zig:68-74`.

### 5.2 `gdn_chain` — the recurrence itself

- **Anchor**: `zig/kernels/metal/qwen3_5/gdn_chain.metal:43-75` (the token loop), `:53-75` excerpted
- **What it does**: launched as grid `{32, 128, 16}` with threadgroup `{32, 4, 1}`
  (`forward.zig:155`) — one simdgroup per (output dim `dv`, head), 32 lanes each holding
  `n_per_t = 4` of the 128 key dims. **The grid has no row axis**: each simdgroup walks the window's
  `W` rows sequentially inside the kernel, keeping its 4 state floats in registers, and software-pipelines
  the next row's q/k/v/g/β loads one iteration ahead.
- **Excerpt** (`gdn_chain.metal:53-75`):

```cpp
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] * gc;
            kv_mem += state[i] * kc[i];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (vc - kv_mem) * bc;
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] + kc[i] * delta;
            out += state[i] * qc[i];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[(node * Hv + hv_idx) * Dv + dv_idx] = static_cast<InT>(out);
          }
          for (int i = 0; i < n_per_t; ++i) { kc[i] = kn[i]; qc[i] = qn[i]; }
          vc = vn; gc = gn; bc = bn;
          if (dims[1] || node + 1 == W) {
            const int saved = dims[1] ? node : 0;
            auto o_state = state_out + ((saved * Hv + hv_idx) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
          }
```

- **Teaching value**: this is the whole difference between attention and a linear mixer, in 13 lines.
  The state is **a matrix in registers**; one token costs a fixed amount of arithmetic no matter how
  long the context is; and the loop is **serial in time** — you cannot parallelise over tokens, only
  over state elements.
- **Cost, O(1) vs O(n)** (arithmetic from the verified constants, not a measurement):

  | | 18 delta layers | 6 attention layers |
  |---|---|---|
  | per-token state touched | `18 × 16 × 128 × 128 × 4 B` = **18,874,368 B**, constant | `12,288 B × context`, grows |
  | per-token MACs / layer | ≈ 4 × (16·128·128) ≈ **1.05 M**, constant | `8 heads × 256 dims × n × 2` = `4096·n` |
  | parallelism | `128 × 16 × 32` = 65,536 threads, independent of rows | grows with keys (`nch` chunks) |
  | serial dimension | **tokens** (the `node` loop) | none (all keys independent) |

  Equal-cost crossovers: the recurrent layers read/write as many bytes per token as the attention
  layers at **1,536 tokens** of context (`18,874,368 / 12,288`); in MAC terms a delta layer equals an
  attention layer at **256 keys** (`1.05 M / 4096`). Past those points attention keeps getting more
  expensive per token and the delta-net does not. Below them, the recurrence is the *more* expensive
  mixer — a 1-MiB state read per layer is not free.
- **Gotchas**: (a) `dims[1]` is `record`: when the caller is verifying a drafted window, **every**
  row's state is written out (`saved = node`) so a partial acceptance can roll back; otherwise only
  the last row's is kept (`saved = 0`) — that one flag is the entire speculative-decoding story for
  recurrent layers; (b) `hk_idx = hv_idx / (Hv / Hk)` is a GQA-style mapping that is the *identity*
  here (16/16), but the same generated kernel shape serves the 27B where it is not
  (`cuda_recurrent.zig:10-13` requires `v_heads == 48 and k_heads == 16`);
  (c) the two `simd_sum`s are inside the token loop — 2 cross-lane reductions per token per
  (head, dv), i.e. 2 × 16 × 128 per layer per token; (d) the state is **fp32** while q/k/v/β are bf16
  (`docs/recipes/adding-a-zig-family.md:255` states the rule: "keep recurrent state fp32");
  (e) the file is machine-generated — `// Generated by tools/zig/gen_qwen35_kernels.py; do not edit.`
  — so the Zig-side launch and the Metal signature cannot drift.

### 5.3 `gdn_post` — the gated output norm

- **Anchor**: `zig/kernels/metal/qwen3_5/gdn_post.metal:22-36`
- **What it does**: one simdgroup per (head, row). RMS-normalises the head's 128 outputs with the
  shared 128-wide weight, multiplies by `silu(z)` taken from this head's slice of the 2048-wide `z`
  projection, and writes the 2048-wide mixer output.
- **Excerpt** (`gdn_post.metal:22-36`):

```cpp
  constexpr int PER = DV / 32;
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = float(bfloat(float(NW[d]) * (yv[j] * inv)));
    const float zf = float(Z[m * ZS + ZO + hv * DV + d]);
    OUT[m * NV * DV + hv * DV + d] = bfloat(zf / (1.0f + metal::exp(-zf)) * x);
  }
```

- **Teaching value**: "gated" in *gated* delta-net is this one line — the mixer's output is
  multiplied elementwise by `silu` of a second projection of the same input. The same trick appears
  in the SwiGLU MLP and in attention's output gate; it is the family's signature.
- **Gotchas**: this is the **only** place the `z` projection is consumed, and `z` is a full
  2048×2048 matmul — the gate costs as much as `out_proj`. The norm is per head (128 wide,
  `eps = 1e-6` passed in), not per token.

### 5.4 Reference implementations

- **Anchor**: `zig/src/core/gdn_prep_ref.zig:5-33` (prep, fp64 with error bounds) and
  `zig/src/core/contract/delta_slots_ref.zig:96-152` (the recurrence, fp64 with a running error budget)
- **What it does**: `gdn_prep_ref` is the ground truth for `gdn_pre`: `conv` (with an fp32 error bound
  `1.1 · taps · 2^-24 · Σ|x·w|`, "silu' is at most 1.1"), `norm` (the asymmetric q/k scaling), and
  `gate` (the exact `exp(-exp(a_log)·softplus(a+dt_bias))`). `delta_slots_ref` is the fp64 recurrence,
  computing not just the output but a **per-element error allowance** that any backend's kernel must
  land inside.
- **Excerpt** (`gdn_prep_ref.zig:28-33`) — the gate, in one readable function:

```zig
/// g = exp(-exp(a_log) * softplus(a + dt_bias)).
pub fn gate(a: f64, dt_bias: f64, a_log: f64) f64 {
    const x = a + dt_bias;
    const sp = if (x > 20) x else std.math.log1p(@exp(x));
    return @exp(-@exp(a_log) * sp);
}
```

and the recurrence with its error budget (`delta_slots_ref.zig:110-127`):

```zig
            for (0..128) |k| {
                const si = (span.state_slot * 48 + head) * 128 * 128 + dim * 128 + k;
                const initial: f64 = if (parent < 0) f.state[si] else states[@as(usize, @intCast(parent)) * 128 + k];
                const error_before = if (parent < 0) 0 else errors[@as(usize, @intCast(parent)) * 128 + k];
                const state = initial * f.decay[at];
                const err = error_before * @abs(f.decay[at]) + u * @abs(state);
                const key = v(f.key[(row * 16 + kh) * 128 + k]);
                states[r * 128 + k] = state;
                errors[r * 128 + k] = err;
                dot += state * key;
                dot_mag += @abs(state * key);
                dot_err += err * @abs(key);
            }
            dot_err += tree * dot_mag;
            const incoming = v(f.value[at * 128 + dim]);
            const mix = v(f.mixing[at]);
            const correction = (incoming - dot) * mix;
            const correction_err = (dot_err + u * (@abs(incoming) + @abs(dot))) * @abs(mix) + u * @abs(correction);
```

- **Teaching value**: the reference and the kernel are the *same five operations*; only the data
  layout and the precision differ. `decay`/`mixing` in the reference are `g`/`β` in the kernel —
  reading them side by side is the fastest way to understand either.
- **Gotchas**: (a) `delta_slots_ref` is parameterised but its defaults are the **27B's** geometry
  (`nk = 16, nv = 48, dk = dv = 128, taps = 4`, `delta_slots_ref.zig:6-16`) and it is wired into the
  CUDA/HIP kernel-contract gates — **there is no fp64 reference in-repo that is specialised to the 2B
  Metal `gdn_chain`**; the 2B's guarantee is the *self-consistency* check in §9 instead;
  (b) the reference tracks a `parent` per row (a draft **tree**), which the Metal path does not support
  (`state.zig:124` refuses any non-linear path with `error.UnsupportedQwenTree`);
  (c) `gdn_prep_ref`'s error term exists because an fp32 conv over 4 taps can lose bits — the contract
  is "inside this budget", never "bit-equal to fp64".

---

## 6. State management — what a session actually owns

### 6.1 The per-stream `Cache`

- **Anchor**: `zig/src/families/qwen3_5/state.zig:8-11` (the sizes), `:64-75` (the layout),
  `:82-102` (allocation), `:108-136` (commit and keep)
- **What it does**: a `Cache` is one stream's committed state: a logits row, and for each of the 24
  layers either a `Delta{recurrence, conv}` or an `Attention{keys, values}`. All buffers are
  `shared | untracked` Metal buffers (`state.zig:7`), i.e. CPU-visible unified memory — which is why
  `commit`/`keep`/`snapshot` are plain `@memcpy`s on the host with no blit encoder.
- **Excerpt** (`state.zig:8-11` and `:64-75`):

```zig
pub const delta_bytes = c.linear_heads * c.linear_dim * c.linear_dim * 4;
pub const conv_bytes = (c.conv_taps - 1) * c.conv_dim * 2;
pub const window_rows = 16;
pub const batch_rows = 32;
…
pub const Delta = struct { recurrence: mtl.Buffer, conv: mtl.Buffer };
pub const Attention = struct { keys: mtl.Buffer, values: mtl.Buffer };
pub const Mixer = union(enum) { delta: Delta, attention: Attention };

pub const Cache = struct {
    memory: Storage,
    capacity: usize,
    len: usize = 0,
    paged: ?Paged = null, // its keys and values in the pool's pages (TF_QWEN35_METAL_KV_PAGES=1), not buffers of its own
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    blocks: [c.layers]Mixer,
    logits: mtl.Buffer,
};
```

**The complete per-session memory model** (all arithmetic from these constants; `state.zig:210-213`
asserts the first two rows):

| Part | Formula | Bytes |
|---|---|---|
| recurrent state, per delta layer | `16 × 128 × 128 × 4` | 1,048,576 |
| conv window, per delta layer | `(4 − 1) × 6144 × 2` | 36,864 |
| × 18 delta layers | | 19,537,920 |
| logits row | `248320 × 2` | 496,640 |
| **fixed part** `cacheBytes(0)` | | **20,034,560** (≈ 19.1 MiB) |
| KV per token | `6 layers × 2 × 2 kv_heads × 256 × 2` | **12,288** |
| e.g. 70,000-token stream | `20,034,560 + 70,000 × 12,288` | ≈ 880 MB |

So the recurrent state equals the KV cache at **1,590 tokens** (`19,537,920 / 12,288`). Under ~1.6k
tokens of context, the 18 "cheap" layers dominate a session's memory; above it the 6 attention layers do.

- **Teaching value**: a hybrid model has **two kinds of state with opposite scaling** — a large fixed
  cost that does not grow, and a small per-token cost that does. Every capacity decision in the
  engine follows from that one table.
- **Gotchas**: (a) `capacity` is fixed at stream admission (`backend.zig:109`:
  `min(options.capacity, ids.len + max_new + window_rows)`), and KV is indexed `[kv_head][position][dim]`
  with stride `capacity`, so a cache cannot grow; (b) in paged mode **all** attention layers of a
  stream point at the *same* arena buffer and are disambiguated by `pagePart(a, values)`
  (`state.zig:22-24`) plus the stream's page table; (c) the delta buffers are explicitly zeroed at
  init (`state.zig:90-91`) — a recurrent model has no "empty cache" sentinel, zero *is* the empty state.

### 6.2 `Scratch` — the shared, and much larger, allocation

- **Anchor**: `state.zig:139-196`, especially `:187-193`
- **What it does**: one `Scratch` per backend (not per stream) holds every intermediate buffer for a
  whole forward: `h`, `x`, `r` (residual stream), `qkv`, `z`, `a`, `b`, `q`, `k`, `v`, `g`, `beta`,
  `y`, `mix`, `qraw`, `knorm`, `queries`, `gate`, `up`, `act`, the attention partials `pm`/`pl`/`po`,
  the logits block, and — per delta layer — a **snapshot** pair for rollback.
- **Excerpt** (`state.zig:187-194`):

```zig
        const chunks = (capacity + 127) / 128;
        out.pm = try mem.alloc(rows * c.query_heads * chunks * 4);
        out.pl = try mem.alloc(rows * c.query_heads * chunks * 4);
        out.po = try mem.alloc(rows * c.query_heads * chunks * c.head_dim * 4);
        out.logits = try mem.alloc(batch_rows * c.vocab * 2);
        for (&out.snapshots, 0..) |*s, i| if (c.linear(i)) {
            s.* = .{ .recurrence = try mem.alloc(batch_rows * delta_bytes), .conv = try mem.alloc(rows * conv_bytes) };
        };
```

With the served settings (`backend.zig:37` → `rows = max(chunk = 128, batch_rows = 32) = 128`), at
`capacity = 70,000` (`chunks = 547`), the two dominant terms are, by arithmetic:

| Buffer | Formula | Bytes |
|---|---|---|
| delta snapshots | `18 × (32 × 1,048,576 + 128 × 36,864)` | 688,914,432 |
| `po` (attention partials) | `128 × 8 × 547 × 256 × 4` | 573,571,072 |
| `logits` | `32 × 248320 × 2` | 15,892,480 |

- **Teaching value**: the *engine's* scratch, not the sessions' caches, is usually the biggest single
  allocation — and `po` grows with the **context length** while the per-stream KV grows with the
  tokens actually used.
- **Gotchas**: (a) `snapshots[i].recurrence` is sized `batch_rows × delta_bytes` = 32 MiB **per delta
  layer** because `record` mode writes every row's state; (b) `po` is fp32 (`×4`) — the attention
  partials are kept at higher precision than anything else in the pass; (c) `Scratch` is created
  before `qualify.check` runs, and `qualify` makes a *second*, 1-row `Scratch` (`qualify.zig:31`), so
  loading briefly holds two.

### 6.3 Commit, partial keep, and snapshots

- **Anchor**: `state.zig:108-136` (commit/keep), `snapshot.zig:38-71` (host image)
- **What it does**: `commit` copies each delta layer's chosen snapshot row into the committed state
  and advances `len`; `keep(path)` is the speculative-decoding commit — it rewinds to
  `last.start + path.len`, copies the state from the snapshot row the accepted prefix ended on, and
  copies that row's logits. Attention layers need nothing in either case: their keys were written
  in place at the right positions, and `len` alone decides what is visible.
- **Excerpt** (`state.zig:108-119`):

```zig
    pub fn commit(self: *Cache, scratch: *const Scratch, base: usize, rows: usize, record: bool) void {
        for (&self.blocks, scratch.snapshots) |*b, snapshot| {
            if (b.* == .delta) {
                const d = b.delta;
                const saved = if (record) base + rows - 1 else 0;
                @memcpy(d.recurrence.contents()[0..delta_bytes], (snapshot.?.recurrence.contents() + saved * delta_bytes)[0..delta_bytes]);
                @memcpy(d.conv.contents()[0..conv_bytes], (snapshot.?.conv.contents() + (base + rows - 1) * conv_bytes)[0..conv_bytes]);
            }
        }
        if (record) self.last = .{ .start = self.len, .base = base, .rows = rows } else self.last = null;
        self.len += rows;
    }
```

- **Teaching value**: **rejecting a drafted token is cheap for attention and expensive for a
  recurrent layer.** Attention just forgets; the recurrence must be restored from a saved copy — which
  is exactly why `record` mode stores 32 rows of 1-MiB state per delta layer.
- **Gotchas**: (a) `keep` requires the path to be the identity prefix `0,1,2,…`
  (`state.zig:124`) — chains only, no trees, on Metal; (b the conv snapshot is indexed by
  `base + rows - 1` even when `record` is false, because the conv window always advances;
  (c) `snapshot.zig:46-70` serialises a cache into host bytes in a *layout-independent* order so a
  contiguous cache can be restored into a paged one and vice versa, asserting
  `off == snap.bytes.len` at the end.

### 6.4 Session admission and the paged option

- **Anchor**: `zig/src/native/qwen35.zig:45-70` (`pagesOn`, `planPages`) and `:98-112` (admission)
- **What it does**: unpaged, the server asks `metal_admit.streams(..., q.state.cacheBytes(window), ...)`
  — how many whole caches fit under the load ceiling. With `TF_QWEN35_METAL_KV_PAGES=1`, it instead
  plans a shared page pool from `state.pagedLayout()` (fixed part = logits + 18 delta states;
  `token_bytes` = 12,288; `page_tokens` = 256) and fails with a sentence naming the shortfall.
- **Excerpt** (`native/qwen35.zig:51-66`):

```zig
fn planPages(a: std.mem.Allocator, metal: *q.backend.Metal, device: mtl.Device, o: api.Open, window: u64) !?[]const u8 {
    const kv = q.state.kv;
    const l = q.state.pagedLayout();
    const limit = (cache_fit.ceiling(cache_fit.ram() orelse 0, device.maxWorkingSet(), null) catch unreachable).bytes;
    const used = @max(cache_fit.footprint() orelse 0, device.allocated());
    const room = limit -| used;
    const lanes_n: u32 = @max(o.lanes, 2);
    const p = kv.plan(l, .{ .room = room, .engine = l.fixed_bytes, .lanes = lanes_n, .fixed = o.lanes_fixed, .window = window, .engine_paged = true }) catch |e| return switch (e) {
        error.NoStream => try std.fmt.allocPrint(a, "TF_QWEN35_METAL_KV_PAGES: one stream's {d}-token window of KV pages ({d:.2} GiB) does not fit the {d:.2} GiB left under the load limit; lower --context", .{ window, gib(l.pages(window) * l.pageBytes()), gib(room) }),
        error.TooMany => try std.fmt.allocPrint(a, "TF_QWEN35_METAL_KV_PAGES: --parallel {d} needs a fixed part and a page for each stream beside the first, more than the {d:.2} GiB left under the load limit", .{ lanes_n, gib(room) }),
    };
    try metal.attachPages(l, p.lanes, p.pages, window);
```

- **Teaching value**: "how many users fit" is one division — ceiling minus weights, divided by
  `cacheBytes(context)`. The paged variant replaces *per-stream reservation* with *per-stream fixed part
  plus shared pages*, which is only worth it because the fixed part (19.1 MiB) is small next to a long
  context's KV.
- **Gotchas**: (a) paging is **opt-in via environment variable** and gets its own two compiled kernels
  (§7); (b) with paging on, `--parallel` sets the stream count outright, bypassing the byte-based
  admission (`native/qwen35.zig:100`); (c) prompt-cache budgeting uses `pagedLayout().fixed_bytes` in
  paged mode and full `cacheBytes(window)` otherwise (`native/qwen35.zig:126-130`).

---

## 7. The kernel dispatch list

`kernels.zig:11` counts them: `total = sources.all.len (15) + layout_names.len (5) + paged_names.len (2)`
= **22 pipelines**, compiled from source at load (`kernels.zig:58-80`), each from its own
`mtl.Library` except the 5 layout kernels and the 2 paged kernels, which share one library each.

### 7.1 Per forward pass, in encode order

One serial compute encoder, one command buffer per `forward.run` (`forward.zig:77-78`, `:107-109`).

| # | Kernel | Grid / group | Anchor | What it does |
|---|---|---|---|---|
| 1 | `qwen35_embed` | `{1024, rows}` / 256 | `forward.zig:80-86`, `layout.metal:5-18` | unpacks two 4-bit codes per byte from the packed embedding into bf16 |
| | **× 24 layers:** | | `forward.zig:87-102` | |
| 2 | `norm` (or `norm_nores` at layer 0) | `{128, rows}` / 128 | `forward.zig:88`, `norm.metal:27-40` | `h += r`, then RMSNorm → `x` |
| 3 | **delta layers (18)** | | `forward.zig:123-166` | |
| 3a | `qmm_6144_2048` | per `projection()` | `forward.zig:125` | `in_proj_qkv` |
| 3b | `qmm_2048_2048` | | `:126` | `in_proj_z` |
| 3c | `qmm_16_2048` ×2 | | `:127-128` | `in_proj_a`, `in_proj_b` |
| 3d | `gdn_pre` | `{32, 48, seg.rows}` / 32 | `:133-146` | conv+SiLU, q/k norms, v, `g`, `β`, next conv window |
| 3e | `gdn_chain` | `{32, 128, 16}` / `{32,4,1}` | `:147-155` | the recurrence over the segment's rows |
| 3f | `gdn_post` | `{32, 16, rows}` / 32 | `:158-164` | per-head norm × `silu(z)` |
| 3g | `qmm_2048_2048` | | `:165` | `out_proj` → `r` |
| 4 | **attention layers (6)** | | `forward.zig:168-231` | |
| 4a | `qmm_4096_2048` | | `:170` | `q_proj` (Q **and** gate) |
| 4b | `qmm_512_2048` ×2 | | `:171-172` | `k_proj`, `v_proj` |
| 4c | `qwen35_head_norm` ×2 | `{64·rows·heads}` / 64 | `:173-184`, `layout.metal:20-45` | per-head RMSNorm of Q then K |
| 4d | `qwen35_queries` | `{256, 8, rows}` / 256 | `:188-192`, `layout.metal:56-63` | RoPE into the query scratch |
| 4e | `qwen35_keys` *or* `qwen35_keys_paged` | `{256, 2, rows}` / 256 | `:195-202`, `layout.metal:65-76`, `paged.metal:3-16` | RoPE + write K/V into the cache |
| 4f | `attn_partial` *or* `qwen35_attn_partial_paged` | `{512, nch, 2·rows}` / 512 | `:205-215` | flash-attention partials per 128-key chunk |
| 4g | `attn_merge` | `{32, 8, rows}` / 32 | `:216-222` | merges the `nch` partials |
| 4h | `qwen35_attention_gate` | `{2048, rows}` / 256 | `:225-229`, `layout.metal:78-86` | `out = y · sigmoid(gate half of q_proj)` |
| 4i | `qmm_2048_2048` | | `:230` | `o_proj` → `r` |
| 5 | `norm` | `{128, rows}` / 128 | `:93` | post-attention norm (residual fused) |
| 6 | `qmm_6144_2048` ×2 | | `:94-95` | `gate_proj`, `up_proj` |
| 7 | `mlp_act` | `{6144, rows}` / 256 | `:96-100` | `silu(gate) · up` |
| 8 | `qmm_2048_6144` | | `:101` | `down_proj` → `r` |
| 9 | `norm` | | `:103` | final norm |
| 10 | `qmm_248320_2048` | | `:104-106` | `lm_head`; `.last` offsets to the final row only |

**Per decode step (one row, one stream)**, this is: 1 embed + 24×2 norms + 18×(4 projections + 3 GDN
kernels) + 6×(3 projections + 2 head_norms + 4 attention kernels + gate + o_proj) + 24×(2+1+1 MLP) +
1 final norm + 1 head = **263 dispatches** in a single command buffer. Counted from the table:
`1 + 48 + 18×7 + 6×11 + 24×4 + 1 + 1 = 339`… the exact figure depends on how `projection()` calls are
counted, so treat the table (not a total) as the contract.

- **Teaching value**: a decode step is *hundreds* of small kernel launches, all in one command buffer,
  with one `wait()` at the end (`forward.zig:109`). The per-layer shape of the pass — norm, mixer,
  norm, MLP — is visible as a repeating block in the encode order.
- **Gotchas**: (a) `projection()` (`forward.zig:25-36`) looks the pipeline up **by (outputs, inputs)**
  — `kernels.zig:30-35` matches `qmm_N_K` by parsing the key name, so an unknown shape returns `null`
  and the `.?` panics; the seven compiled shapes are exactly the seven the 2B needs; (b) the
  per-segment loops mean a shared round over 2 streams encodes 3d/3e and 4d-4g **twice**, once per
  cache, while all projections and `gdn_post` run once over the concatenated rows — segment-local
  recurrence, shared projections, as the file's doc line says; (c) `qwen35_head_norm` uses
  `metal::precise::rsqrt` (`layout.metal:39`) where `norm.metal:39` uses plain `metal::rsqrt`;
  (d) the paged twins differ *only* in the K/V address computation (`paged.metal:1`: "the contiguous
  kernels with only the K/V address changed") and `paged.metal:36` static-asserts
  `TF_PAGE_TOKENS % CK == 0` so a 128-key chunk never straddles a page.

### 7.2 The projection kernels

- **Anchor**: `zig/kernels/metal/qwen3_5/kernels.zig:3-11` (the seven shapes), `forward.zig:25-36` (launch)
- **What it does**: one specialised kernel per matrix shape, each with `columns = 32` (16 for
  `qmm_16_2048`) and `threads = 256`; the launch is
  `{threads × ceil(outputs/columns), ceil(rows/8), 1}`. Inside, the kernel hard-codes `K`, `N`, the
  number of accumulation chains `S = 16`, simdgroups `SGS = 8`, tiles `NT = 4` and the group size
  `GS = 64` (`qmm_2048_2048.metal:30-36`).
- **Excerpt** (`forward.zig:25-36`):

```zig
    fn projection(self: Encoder, linear: wts.Linear, x: Buffer, offset: usize, y: Buffer, y_offset: usize, rows: usize) void {
        const p = self.m.kernels.projection(linear.outputs, linear.inputs).?;
        self.e.setPipeline(p.pipeline);
        self.e.setBuffer(x, offset, 0);
        self.e.setValue([2]i32{ @intCast(rows), @intCast(linear.inputs) }, 1);
        self.tensor(linear.weight, 2);
        self.tensor(linear.scales, 3);
        self.tensor(linear.biases, 4);
        self.e.setValue([1]f32{1}, 5);
        self.e.setBuffer(y, y_offset, 6);
        self.run(.{ p.threads * ((linear.outputs + p.columns - 1) / p.columns), (rows + 7) / 8, 1 }, .{ p.threads, 1, 1 });
    }
```

- **Teaching value**: "the matmul" in a 4-bit engine is not one kernel — it is one kernel **per
  shape**, generated, so the loop bounds and the accumulation order are compile-time constants.
- **Gotchas**: `cpu_kernels.zig` is the host twin of exactly this kernel — it reproduces the Metal
  chain order (`reduceChains`, `cpu_kernels.zig:89-97`) and has a portability digest
  (`cpu_kernels.zig:277`: `portable_digest: u64 = 0xae93e2f573b522de`) asserting byte-identical output
  at 1-16 threads on x86_64 and arm64.

---

## 8. MTP / drafting — the Metal 2B has **no** draft head

- **Anchor**: `zig/src/families/qwen3_5/backend.zig:193-195`
- **What it does**: the lane engine's `draft` vtable entry refuses any nonzero depth outright.
- **Excerpt**:

```zig
    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.QwenCheckpointHasNoDraftHead;
    }
```

- **Why, and what fills the gap**: `docs/recipes/qwen3.5-2b-native.md` states it plainly — "Although
  the config describes an MTP layer, this checkpoint has no MTP weights; context-copy proposals use
  the shared lane engine with drafts enabled." The config fixture indeed carries
  `"mtp_num_hidden_layers": 1` (`config.zig:175`) which this engine ignores. Speculation still happens,
  but the proposals come from the shared lane engine (prompt/context copying), and they are *verified*
  through the same `forward.run(..., record = true, .all)` path (`backend.zig:156-186`) with
  `window_rows = 16` rows per stream and `batch_rows = 32` rows per shared forward.
- **Where drafting *is* implemented for this family**: the CUDA and HIP backends.
  `cuda_mtp.zig` (564 lines) runs "the Qwen3.6 MTP layer as the drafter on CUDA: chains or trees from a
  kept row's final normed state and next token", with weights from a side file
  (`cuda_mtp_weights.zig:1`: "mtp-4bit.safetensors beside the checkpoint"); `cuda_draft.zig` (593) and
  `backend/hip/draft/draft.zig` (446) implement the DFlash2 external drafter;
  `draft_prior.zig:4` holds the shared acceptance prior
  `{0.75, 0.65, 0.58, 0.52, 0.47, 0.43, 0.4, 0.37, 0.35, 0.33, 0.31, 0.29, 0.28, 0.27, 0.26}` for depths 1-15.
- **Teaching value**: speculative decoding needs **a cheap proposer and a cheap verifier**. This
  checkpoint has no proposer, so the engine keeps only the verifier — and the verifier is just the
  normal forward with `record = true` and 16 rows instead of 1.
- **Gotchas**: `draft_prior.zig` sitting in the directory makes it look like the Metal path drafts;
  it does not — the file's own doc comment says "one table for the CUDA and HIP engines".

---

## 9. Profiling and load-time qualification

There is **no `TF_QWEN35_*` profiling switch for the Metal 2B** (the only `TF_QWEN35_*` variable is
`TF_QWEN35_METAL_KV_PAGES`, read at `native/qwen35.zig:47`). The analogue of the 27B's
`TF_Q27_PROFILE` lives on the CUDA side as a family of switches in the test driver
`zig/tests/qwen35_cuda_main.zig:17-19, 100-109, 117, 147, 217-242`:
`TF_Q35_STATE_STEPS`, `TF_Q35_PREFILL`, `TF_Q35_BESIDE`, `TF_Q35_DELTA`, `TF_Q35_RESUME`,
`TF_Q35_RESUME_TREE`, `TF_Q35_SHIFT`, `TF_Q35_WINDOW` — notably its `state` mode reports "raw GDN/conv
and active KV bytes" (`qwen35_cuda_main.zig:17`), which is the measured version of §6.1's table.
Startup phases are timed by `cuda_startup_profile.zig` (54 lines, "disjoint awake-clock startup
phases"); CUDA window prices are cached by `cuda_window_costs.zig`.

What the Metal path does at load instead is a **numerical qualification**:

- **Anchor**: `zig/src/families/qwen3_5/qualify.zig:30-65`, invoked at `backend.zig:39-46`
- **What it does**: on the real weights, it runs 4 caches of slightly different capacities through
  1-row and 3-row prompts, then one 16+16-row shared forward with `record = true`, and compares
  **every row's logits** against a 1-row run, then **every committed byte** (logits, recurrence, conv,
  and the live prefix of each KV head) via `equal`. It then exercises `keep` with paths of length 3
  and 1 and re-compares against caches built one row at a time.
- **Excerpt** (`backend.zig:39-46`) — the consequence of failure:

```zig
        @import("qualify.zig").check(model, &out.scratch) catch |err| switch (err) {
            error.QwenCacheLengthMismatch, error.QwenRecurrenceMismatch, error.QwenAttentionCacheMismatch, error.QwenWindowLogitsMismatch, error.QwenCommittedLogitsMismatch => {
                std.log.warn("Qwen wider lanes failed the local exactness check ({s}); keeping one-row lane rounds", .{@errorName(err)});
                return out;
            },
            else => return err,
        };
        out.width = st.window_rows;
```

- **Teaching value**: the engine's correctness claim is **self-consistency, not cross-backend
  equality** — "a 16-row window must produce the same bytes as 16 one-row steps". That is the only
  property a user can actually feel (drafted = plain, concurrent = solo), and it is checked on the
  user's own GPU at load.
- **Gotchas**: (a) a failed check **degrades** the engine to one row per round instead of refusing to
  serve; a GPU *execution* failure still aborts; (b) the token ids in `qualify.zig:40` are a fixed
  public set; (c) `qualify` allocates a second `Scratch` with capacity 64 — the check deliberately
  uses tiny caches so it is fast.

---

## 10. The tests as contracts

| Test / tool | Where | Invariant it pins |
|---|---|---|
| `cacheBytes` test | `state.zig:210-213` | `cacheBytes(0) == 18_874_368 + 663_552 + 496_640` and **12,288 bytes per token** |
| `attention_layers` comptime check | `state.zig:15-19` | exactly `layers/4` layers are non-linear, else `@compileError` |
| config refusal table | `config.zig:220-232` | 9 named mutations of the real `config.json` each produce their specific error |
| config fixture parse | `config.zig:126-212` + its tests | the published checkpoint's config is admitted |
| `paged_test.zig:39-154` | family dir (GPU) | paged `qwen35_keys`/`attn_partial_paged` produce the **same bytes** as the contiguous kernels through shuffled pages, at `(len, rows)` = (700,9), (255,2), (0,16), (1024,1) — i.e. across page and 128-key chunk boundaries |
| `draw.zig:22-72` | family dir (host) | bounded top-k draws equal the full-vocabulary fp64 reference draw, incl. masks, `±0`, ties, keyed positions |
| `draw.zig:140-158` | family dir | every nonfinite bf16 encoding is refused (`NonfiniteQwenLogits`), empty mask → `NoAllowedQwenToken` |
| `draw.zig:104-115` | family dir | the served top-k path makes **no heap allocation** (`FailingAllocator`, `alloc_index == 0`) |
| `cpu_kernels.zig:280-339` | family dir (host) | the host projection is byte-identical at 1-16 threads and at every column split, on x86_64 and arm64 (digest `0xae93e2f573b522de`) |
| `tf-qwen35-exact` | `zig/tests/qwen35_exact.zig:104-126` | for prefixes {0,127,128,129,255,256} × windows {1,2,7,8,15,16,17,32}: every window row's logits equal a 1-row run; caches equal byte for byte; a partial `keep` equals a cache built only from the accepted rows; and the *next* step after a keep still matches (48 cases + a 4-stream shared-round case) |
| `tf-qwen35-exact` `shared` | `qwen35_exact.zig:71-103` | two streams of **unequal** prefixes (127 and 256) in one shared forward each equal their solo runs, then survive different partial keeps |
| `tf-qwen35-check` | `zig/tests/qwen35_check.zig:12-22` | the tied head is literally the embedding buffer and offset (or the untied head is its own), and all 22 pipelines compile |
| `tf-qwen35-forward` | `zig/tests/qwen35_forward.zig:26-34` | teacher-forced bf16 logits for a token fixture, 32 rows at a time — the input to the external fidelity comparison (`tools/zig/qwen35_fidelity.py`) |
| `qwen35_fixtures.zig:8-118` | `zig/tests` | captured per-operation fixtures must match exactly, reading weights through the real checkpoint views |
| `slicing_decl_test.zig` (105) | family dir | the TP cuts keep packed segments whole and kv heads with their query ranks |
| `weights/checkpoint_test.zig` (464), `checkpoint_hf_test.zig` (307) | family dir | host index, tensor names, dtypes, the conv layout and norm conventions match the Python loader's rules |
| `weights/real_test.zig` (91) | family dir | with `TF_QWEN_DIR` set, every tensor the loader asks for resolves in a real checkpoint |
| CUDA/HIP GDN contracts | `core/contract/delta_slots_records.zig:6-12`, `hip_records_gdn.zig`, `docs/kernel-contract.md:303-305, 346-349` | each `tf_q35_gdn_slots_*` replay is bit-equal with and without write-back; HIP's fused prompt prework is bit-equal to the 4-kernel chain it replaces |

Build wiring worth knowing: `zig/build/core_checks.zig:86-90` runs `paged_test.zig`;
`zig/build/cpu_contract.zig:7` runs `cpu_kernels.zig`; `zig/build/mesh.zig:12` runs
`slicing_decl_test.zig`; `zig/build/metal_targets.zig:314-315` registers `tf-qwen35-forward` and
`tf-qwen35-exact`; `zig/build/cuda_family_host.zig:7-8` runs the device-free
"Metal 2B format scope" config tests; `zig/src/living_shards_test.zig:11` pulls in
`pocket_shards_test.zig`.

---

## 11. Suggested lesson ordering

Each step names the anchor to show and the one control that makes it land.

1. **"Here is a whole model."** `qwen3_5.zig:1-8` (8 lines), then the §1a table.
   *Control*: a file-tree view where clicking a file reveals its doc comment and line count — the
   learner sees 22 files become 5 nouns.
2. **"The config is a bouncer, not a knob."** `config.zig:7-21` (from map-forward) → `config.zig:73-85`
   and `:105-122`.
   *Control*: an editable `config.json` text box; every edit prints the exact Zig error
   (`UnsupportedQwenQuantization`, `UnsupportedQwenRotary`, …) using the refusal table at `:220-228`.
3. **"Weights are names and shapes."** `weights.zig:77-97` + the tensor table in §4.
   *Control*: a toggle "layer i = 3?" that flips the mixer union and swaps the tensor list,
   highlighting `q_proj = 4096` vs `in_proj_qkv = 6144`.
4. **"18 of 24 layers are not attention."** `config.zig:37-39` with `state.zig:15-19`.
   *Control*: a 24-cell layer strip coloured by mixer, with the running memory total beside it.
5. **The delta rule, on paper.** §5.0's four formulas, then `gdn_chain.metal:53-65`.
   *Control*: a 4×4 state matrix the learner can step token by token, with sliders for `g` (decay) and
   `β` (write strength); set `β = 1` to see the memory overwrite the key exactly, `g < 1` to watch it
   fade.
6. **The conv and the two scalars.** `gdn_pre.metal:37-48` then `:60-69`, against
   `gdn_prep_ref.zig:28-33`.
   *Control*: a 4-tap window animation over a token row, and a `a_log`/`dt_bias` slider pair showing
   `g` move through (0,1].
7. **The gate.** `gdn_post.metal:22-36`.
   *Control*: a single `silu(z)` curve with the head's output vector scaled live by it.
8. **Why it is cheaper — and when it is not.** §5.2's cost table.
   *Control*: a context-length slider from 0 to 70,000 plotting two lines, `12,288 × n` and the flat
   `18,874,368`, crossing at 1,536.
9. **Two kinds of state.** `state.zig:8-11`, `:64-75`, then the §6.1 table.
   *Control*: the same slider driving a stacked bar — fixed recurrent block vs growing KV block — with
   the 1,590-token crossover marked.
10. **Committing and rejecting.** `state.zig:108-119` then `:121-136`.
    *Control*: a 16-row drafted window where the learner picks an acceptance length; attention rows
    simply disappear, while the delta layers visibly copy one snapshot row back.
11. **The dispatch list.** §7.1's table beside `forward.zig:87-102`.
    *Control*: a scrubbable timeline of the encode order for one decode step, filtered by
    "delta layer / attention layer / MLP", showing the same 4-step block repeating 24 times.
12. **One shape, one kernel.** `forward.zig:25-36` + `kernels.zig:3-11`.
    *Control*: enter any `(outputs, inputs)` pair; the lookup either names the `qmm_N_K` pipeline or
    shows the `null` that would panic.
13. **No draft head.** `backend.zig:193-195`, and the recipe's one sentence.
    *Control*: a request with `"draft": 3` that returns `QwenCheckpointHasNoDraftHead`, next to the
    CUDA path's `cuda_mtp.zig` line count for contrast.
14. **How we know it is right.** `qualify.zig:30-65` and `backend.zig:39-46`, then
    `qwen35_exact.zig:104-126`.
    *Control*: a 6×8 grid of (prefix, window) cases that fills in green as the exactness sweep runs,
    with one cell deliberately flipped to show the "keeping one-row lane rounds" warning path.
