# Map: the bottom of the stack — "it is just numbers in a file"

Repo root: `TensorFold/`. All anchors are paths relative to that root and were read at the line numbers given.

Reading order for a lesson author: safetensors header → mmap + name lookup → MLX 4-bit affine layout → the two matmul kernels → bf16/fp8/fp4 dtype zoo → KL/top-1 quality check → memory-bound decode.

---

## 1. safetensors: the 8-byte header length

- **Anchor**: `zig/src/core/safetensors.zig:8-11`
- **What it does**: The entire on-disk contract in three lines. A safetensors file is: 8 bytes little-endian `u64` header length, then exactly that many bytes of JSON, then the raw tensor data region. `headerSize` is the only place that validates the prefix: the file must be at least 8 bytes, the header must fit in the file after the prefix, and it is capped at `max_header_bytes = 100_000_000` (line 6) so a corrupt length cannot drive an allocation.
- **Excerpt**:
```zig
pub fn headerSize(header_len: u64, file_size: u64) error{BadSafetensors}!usize {
    if (file_size < 8 or header_len > max_header_bytes or header_len > file_size - 8) return error.BadSafetensors;
    return std.math.cast(usize, header_len) orelse error.BadSafetensors;
}
```
- **Teaching value**: The whole format is "a length, a JSON table of contents, and a blob". There is no magic number and no per-tensor framing — the JSON *is* the index.
- **Gotchas**: `file_size - 8` is computed only after the `file_size < 8` guard, so the subtraction cannot wrap. The cap is checked **before** any allocation; the test suite asserts a failing allocator is never even reached (`zig/safetensors_header_host.zig:88-100`).

## 2. safetensors: dtypes and the data region offsets

- **Anchor**: `zig/src/core/safetensors.zig:13-58`
- **What it does**: `DType` is the closed set of element types the engine admits, with `size()` giving bytes per element and `parse()` mapping the JSON spelling (`"BF16"`, `"F8_E4M3"`, …) to the tag. `Entry` is one tensor's header record: dtype, rank (`max_rank = 5`), shape, and the half-open byte range `[begin, end)` **relative to the start of the data region**, i.e. relative to `8 + header_len`.
- **Excerpt**:
```zig
pub const DType = enum {
    bool,
    u8,
    i8,
    u16,
    i16,
    f16,
    bf16,
    u32,
    i32,
    f32,
    u64,
    i64,
    f64,
    f8_e4m3, // FP8 checkpoints: codes as stored, the scales come as their own tensors
    f8_e5m2,
    f8_e8m0,

    pub fn size(self: DType) usize {
        return switch (self) {
            .bool, .u8, .i8, .f8_e4m3, .f8_e5m2, .f8_e8m0 => 1,
            .u16, .i16, .f16, .bf16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
        };
    }
```
- **Teaching value**: A "tensor" on disk is four facts: a dtype, a shape, and two byte offsets. Everything else (names, layers, attention) is convention layered on top of that.
- **Gotchas**: `rank` is capped at 5 — enough for vision patch convolutions such as `BF16 [1024, 3, 2, 14, 14]` (`docs/quantization.md`, "Native safetensors metadata"). FP8 dtypes have no arithmetic here: they are *codes*, and their scales arrive as separate tensors.

## 3. safetensors: computing a tensor's byte range (the parser)

- **Anchor**: `zig/src/core/safetensors.zig:110-135`
- **What it does**: For each JSON entry it multiplies the shape into an expected byte count `n = dtype.size() * Π shape`, reads `data_offsets: [begin, end]`, and refuses unless `end - begin == n` **and** `end <= data_len`. `data_len` is the caller's measured data-region size, so a header can never point outside the file. After parsing, all ranges are sorted by `begin` and checked for overlap.
- **Excerpt**:
```zig
        var e: Entry = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .begin = 0, .end = 0 };
        var n: usize = dtype.size();
        for (shape, 0..) |d, i| {
            e.shape[i] = uint(d) orelse return error.BadSafetensors;
            n = std.math.mul(usize, n, e.shape[i]) catch return error.BadSafetensors;
        }
        const offs = list(o.get("data_offsets")) orelse return error.BadSafetensors;
        if (offs.len != 2) return error.BadSafetensors;
        e.begin = uint(offs[0]) orelse return error.BadSafetensors;
        e.end = uint(offs[1]) orelse return error.BadSafetensors;
        if (e.end < e.begin or e.end - e.begin != n or e.end > data_len) return error.BadSafetensors;
        try out.put(arena, kv.key_ptr.*, e);
    }
    const ranges = try arena.dupe(Entry, out.values());
    defer arena.free(ranges);
    std.mem.sort(Entry, ranges, {}, struct {
        fn less(_: void, a: Entry, b: Entry) bool {
            return a.begin < b.begin;
        }
    }.less);
    var end: usize = 0;
    for (ranges) |e| {
        if (e.begin == e.end) continue;
        if (e.begin < end) return error.OverlappingTensors;
        end = e.end;
    }
```
- **Teaching value**: The byte range is *redundant* with the shape and dtype — and the loader uses that redundancy as a checksum. `shape × dtype == end - begin` is the one invariant that makes "just numbers in a file" safe to trust.
- **Gotchas**: `__metadata__` is skipped (line 93). Zero-length tensors are exempted from the overlap check (`if (e.begin == e.end) continue`). Integers past `i64` arrive as JSON `number_string` and are parsed via the `uint` helper, so `18446744073709551616` is rejected as a shape rather than silently truncated. `parseHeaderPrefixes` lets a text-only loader index just `language_model.*` and ignore a vision tower's layouts.

## 4. Checkpoint loading: memory mapping (and the header-only alternative)

- **Anchor**: `zig/src/core/safetensors.zig:226-239` (mmap) and `:189-201` (header only)
- **What it does**: `File.openPrefixes` maps the whole shard read-only with `populate = false` (lazy faulting), reads the 8-byte length straight out of `map.memory[0..8]`, and parses the header out of `map.memory[8..][0..header_len]`. `data` is stored as `8 + header_len` — the offset every `Entry.begin` is relative to. `Metadata.read` is the no-mapping path: two positional reads (8 bytes, then the header) and nothing of the tensor data is ever touched; it re-checks the file length after reading so a file growing underneath is refused.
- **Excerpt**:
```zig
    pub fn openPrefixes(gpa: std.mem.Allocator, io: Io, path: []const u8, prefixes: []const []const u8) !File {
        var file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len = std.math.cast(usize, try file.length(io)) orelse return error.BadSafetensors;
        if (len < 8) return error.BadSafetensors;
        var map = try Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
        errdefer map.destroy(io);
        const header_len = try headerSize(std.mem.readInt(u64, map.memory[0..8], .little), len);
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const names = try parseHeaderPrefixes(arena.allocator(), map.memory[8..][0..header_len], len - 8 - header_len, prefixes);
        const own = try arena.allocator().dupeSentinel(u8, path, 0);
        return .{ .path = own, .file = file, .map = map, .data = 8 + header_len, .names = names, .arena = arena };
    }
```
- **Teaching value**: Loading a 30 GB model costs almost nothing up front: you mmap and parse a few hundred KB of JSON. The weights only enter RAM as pages when a kernel (or the uploader) touches them.
- **Gotchas**: `map.section == null` means the mapping fell back to a heap copy — then `MADV_DONTNEED` would *zero* the only copy of the bytes, which is why `releaseExcept` guards on it (`:265-282`). `release`/`adviseRandom` are Linux-only no-ops elsewhere. On unified memory (Apple) the page-cache copy competes with the GPU's, hence the deliberate `drop`/`releaseExcept` dance after upload.

## 5. Checkpoint loading: a tensor name → its bytes

- **Anchor**: `zig/src/core/safetensors.zig:288-290` and `zig/src/core/checkpoint.zig:88-98,117-131`
- **What it does**: `File.get` is the whole address computation: look the name up in the header map, then slice the mapping from `data + begin` to `data + end`. `Checkpoint` is the multi-shard namespace: `openModelPrefixes` maps every shard named by `model.safetensors.index.json`'s `weight_map` (else every `model*.safetensors`, sorted), and `get` searches the shards in order, records the name in `used`, and errors by name when it is missing.
- **Excerpt** (`safetensors.zig:288-290`, then `checkpoint.zig:117-131`):
```zig
    pub fn get(self: *const File, name: []const u8) ?Tensor {
        const e = self.names.get(name) orelse return null;
        return .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .bytes = self.map.memory[self.data + e.begin .. self.data + e.end] };
    }
```
```zig
    pub fn get(self: *Checkpoint, short: []const u8) !Tensor {
        var buf: [320]u8 = undefined;
        const name = try self.scoped(&buf, short);
        for (self.files.items) |*f| if (f.get(name)) |t| {
            const key = f.names.getKey(name).?;
            try self.used.put(self.gpa, key, {});
            var out = t;
            if (self.copies) |c| if (c.get(key)) |bytes| {
                out.bytes = bytes;
            };
            return out;
        };
        std.log.err("checkpoint has no tensor {s}", .{name});
        return error.MissingTensor;
    }
```
- **Teaching value**: `"model.layers.3.mlp.down_proj.weight"` → a hash lookup → a byte slice of an mmap. That is the entire path from a model-architecture name to numbers.
- **Gotchas**: `unused()` (`checkpoint.zig:190-200`) makes the loader refuse a checkpoint with tensors nobody read — a leftover name means the family misunderstood the file. `scope` prefixes every name (`"language_model."` for a model with a vision tower beside it). `copies` lets a layer-at-a-time loader substitute converted bytes for the mapped ones without changing call sites.

## 6. Checkpoint loading: tensor names → modules (prefix canonicalisation)

- **Anchor**: `zig/src/core/quant/keys.zig:8-16`
- **What it does**: Checkpoints spell the same module many ways (`model.layers.0.mlp.down_proj.weight`, `language_model.model.layers.0.mlp.down_proj`, `text_model.layers.0.mlp.down_proj`). `canonical` strips one trailing `.weight` and then each known prefix at most once, in a fixed order, returning a slice of the input. This is how a per-module quantisation override in `config.json` finds its tensor whatever the spelling.
- **Excerpt**:
```zig
/// The prefixes a module path may carry, each stripped at most once and in this order.
const prefixes = [_][]const u8{ "model.language_model.", "language_model.", "text_model.", "model." };

/// The module `key` names, without a trailing `.weight` or the known prefixes; a slice of `key`.
pub fn canonical(key: []const u8) []const u8 {
    var k = key;
    if (std.mem.endsWith(u8, k, ".weight")) k = k[0 .. k.len - ".weight".len];
    for (prefixes) |prefix| {
        if (std.mem.startsWith(u8, k, prefix)) k = k[prefix.len..];
    }
    return k;
}
```
- **Teaching value**: Nothing in the file format says what a tensor *is*. The mapping from name to role is a convention the loader has to re-derive, and different exporters disagree.
- **Gotchas**: Order matters and prefixes compose: `model.language_model.model.layers.0...` reduces because `model.language_model.` is stripped first and then `model.` — the test at `:33-45` pins exactly that. A prefix only counts at the start and the suffix only at the end: `layers.0.weight_proj` survives intact. NB: the file named in the brief, `zig/src/core/checkpoint_names.zig`, is **not** this — it is portable *shard filename* validation (`plainName`, `shardName`, Windows reserved names) plus the `weight_map` schema check.

## 7. The memory floor reported by `tensorfold info`

- **Anchor**: `zig/src/cli/info.zig:66-77`, with the bytes counted at `zig/src/core/checkpoint_weights.zig:13-27` and the margin at `zig/src/platform/memory_policy.zig:2`
- **What it does**: `info` prints `weights: X GiB` and then `memory floor: X + 2 GiB`. The weight total comes from the headers alone — one shard's `Metadata` held at a time, summing `entry.end - entry.begin` over every tensor, skipping `vision_model.*` / `vision_projector.*` / `mlp1.*` when the text side is being priced. The runtime margin is a single constant, `2 << 30` (2 GiB). (On Metal + Flash Next it prints a richer startup/cold-pack budget instead.)
- **Excerpt** (`info.zig:66-77`):
```zig
    try out.print("weights: {d:.2} GiB ({d} bytes)\n", .{ @as(f64, @floatFromInt(weights)) / GiB, weights });
    if (flash_metal) {
        const memory = @import("flashnext_memory");
        const inventory = try memory.weights_plan.read(core.Checkpoint, core.quant.mlx, a, io, dir);
        const plan = try memory.budget(weights, memory.trained, inventory, 0);
        try out.print("startup budget: {d:.2} GiB (conservative raw/packed text weights, both prompt planes/selectors, KV/state, keepalive and {d} GiB runtime reserve; trained window)\n", .{ @as(f64, @floatFromInt(plan.steady)) / GiB, runtime_margin >> 30 });
        try out.print("cold pack budget: {d:.2} GiB (source pages, one output phase and conversion scratch; skipped with a validated pack; vision and requested storage are additional at load)\n", .{@as(f64, @floatFromInt(plan.building)) / GiB});
    } else try out.print("memory floor: {d:.2} GiB (weights plus a {d:.2} GiB runtime margin)\n", .{
        @as(f64, @floatFromInt(weights + runtime_margin)) / GiB,
        @as(f64, @floatFromInt(runtime_margin)) / GiB,
    });
    return 0;
}
```
- **Excerpt** (`checkpoint_weights.zig:13-27`):
```zig
pub fn count(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision: bool) !u64 {
    try @import("save_guard").check(io, dir);
    const paths = try ck.shardFiles(a, io, dir);
    defer ck.freeShardFiles(a, paths);
    var total: u64 = 0;
    for (paths) |path| {
        var header = try st.Metadata.read(a, io, path);
        defer header.deinit();
        for (header.names.keys(), header.names.values()) |name, entry| {
            if (!vision and visionTensor(name)) continue;
            total = try std.math.add(u64, total, entry.end - entry.begin);
        }
    }
    return total;
}
```
- **Teaching value**: "How much RAM do I need?" is answered by *adding up byte ranges in JSON headers* — no model is loaded, no GPU is touched. Memory floor = sum of tensor bytes + a fixed 2 GiB.
- **Gotchas**: A repeated name counts each time (the doc comment says so) — duplicated tensors cost real bytes. The 2 GiB margin is shared by the Metal residency bound and the preflight check (`zig/src/metal/residency.zig:24`), so `info`'s number and the server's admission number are the same policy. The `info` test asserts `memory floor: 2.00 GiB` for an 8 KiB fixture.

---

## 8. MLX 4-bit affine: the format declaration (group size and bits)

- **Anchor**: `zig/src/core/quant/mlx.zig:18-21` and `:48-60`, admitted widths at `zig/src/core/quant/mlx_geometry.zig:29-34`
- **What it does**: A width is a `(bits, group)` pair read from `config.json`'s `quantization` object (`bits`, `group_size`, optional `mode: "affine"`). Admitted bit widths are **2, 3, 4, 5, 6, 8** (not 7); admitted group sizes are **32, 64, 128**. `Config` holds a global width plus per-module overrides keyed both by the config's own spelling and by `keys.canonical`.
- **Excerpt**:
```zig
pub const bit_widths = [_]u8{ 2, 3, 4, 5, 6, 8 };

/// One affine width: groups of `group` weights share a scale and a bias, `bits` per weight.
pub const Width = struct { bits: u8, group: u16 };
```
```zig
fn affineWidth(o: std.json.ObjectMap) ?Width {
    const bits = intOrNull(o.get("bits")) orelse return null;
    const group = intOrNull(o.get("group_size")) orelse return null;
    if (o.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return null;
    if (std.mem.indexOfScalar(i64, &.{ 2, 3, 4, 5, 6, 8 }, bits) == null) return null;
    if (std.mem.indexOfScalar(i64, &.{ 32, 64, 128 }, group) == null) return null;
    return .{ .bits = @intCast(bits), .group = @intCast(group) };
}
```
- **Teaching value**: Quantisation is a *per-group* affine map. The only free parameters are how many bits a code has and how many weights share one scale/bias pair.
- **Gotchas**: 7-bit is deliberately excluded from device formats even though the scalar oracle decodes it (`mlx_geometry.scalarShape` allows `bits 2..8`, `shape` narrows to the list). A checkpoint may be *mixed width*: two spellings of one module that declare different widths is a hard refusal (`mlx.zig:42-47`). Real shipped widths: `-oQ4`/`-oQ6`/`-oQ8` Nemotron packs, `Qwen3.8-Flash-Next-MLX-4bit-MTP` at 4-bit g32, `oQ4e-HIP` at 8-bit g64 (`docs/recipes/qwen3.8-flash-next.md:505-507`).

## 9. MLX 4-bit affine: the three tensors and their shapes

- **Anchor**: `zig/src/core/quant/mlx.zig:81-82,97-110`, geometry checked at `zig/src/core/quant/mlx_geometry.zig:37-56`
- **What it does**: One quantised projection is **three** safetensors tensors sharing a module name: `{module}.weight` (packed `i32`/`u32` words, shape `[N, K*bits/32]`), `{module}.scales` and `{module}.biases` (shape `[N, K/group]`, dtype f32/bf16/f16). `read` fetches all three by name, re-labels the words as `.i32`, and runs the geometry check. An expert stack leads with `E` (rank 3).
- **Excerpt** (`mlx.zig:81-82,97-110`):
```zig
/// An affine projection as stored: i32 words [N, K * bits / 32], tables [N, K / group]; a stack leads with E + 1.
pub const Host = struct { words: Tensor, scales: Tensor, biases: Tensor, bits: u8, group: u16 };
```
```zig
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    var buf: [256]u8 = undefined;
    var other: [256]u8 = undefined;
    var words = try t.get(join(&buf, &.{ key, ".weight" }));
    const w = try t.width(key);
    if ((words.dtype != .u32 and words.dtype != .i32) or words.rank != 2) return refuse(key, "weight is not packed int32 words");
    words.dtype = .i32;
    const scale = try t.get(join(&buf, &.{ key, ".scales" }));
    const bias = try t.get(join(&other, &.{ key, ".biases" }));
    _ = geometry.shape(words, scale, bias, w.bits, w.group, false) catch
        return refuse(key, "packed words and group tables have incompatible or unrepresentable geometry");
    const pair = try tables(a, scale, bias);
    return .{ .words = words, .scales = pair[0], .biases = pair[1], .bits = w.bits, .group = w.group };
}
```
- **Excerpt** (`mlx_geometry.zig:46-56`, the dimension algebra):
```zig
    if (!std.mem.eql(usize, scales.shape[0..rank], biases.shape[0..rank]) or
        !std.mem.eql(usize, words.shape[0 .. lead + 1], scales.shape[0 .. lead + 1]))
        return error.UnexpectedTensor;
    const k = try product(scales.shape[lead + 1], group);
    const packed_bits = try product(k, bits);
    if (packed_bits % 32 != 0 or words.shape[lead + 1] != packed_bits / 32) return error.UnexpectedTensor;
    const n = std.math.cast(u32, words.shape[lead]) orelse return error.UnexpectedTensor;
    const columns = std.math.cast(u32, k) orelse return error.UnexpectedTensor;
    const rows = try product(if (stacked) words.shape[0] else 1, n);
    return .{ .n = n, .k = columns, .elements = try product(rows, k) };
}
```
- **Teaching value**: `K` is never stored. It is *derived*: `K = scales.shape[-1] * group`, and the words tensor must then be exactly `K*bits/32` wide. Shape consistency across three tensors is the only thing proving the file is a valid quantised matrix.
- **Gotchas**: `matches()` detects the format by the mere existence of `{key}.scales` beside the weight (`mlx.zig:84-87`) — presence of a sibling tensor, not a flag. Scales/biases are kept at their stored dtype when it is f32/bf16/f16 and only widened to f32 otherwise (`tables`, `:91-94`), so the kernels read bf16 tables directly. `k*bits % 32 != 0` is refused: K must pack into whole 32-bit words.

## 10. MLX 4-bit affine: the exact dequantisation formula and bit packing

- **Anchor**: `zig/src/core/quant/mlx.zig:217-237` (fp64 oracle) and `:240-272` (fp32, torch-order)
- **What it does**: These are the two reference dequantisers. Codes are packed **little-endian, low code first**, `bits` at a time, into a row of 32-bit words; a code may straddle a word boundary (3, 5, 6, 7 bits), so both read the current word plus the next as a 64-bit pair and shift. The value is `w = scale * code + bias`, where the group index is `g = r * groups + (c / group)` — **scales and biases are per row and per group**, not shared across rows.
- **Excerpt** (`mlx.zig:217-237`, the fp64 oracle):
```zig
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const dims = try geometry.scalarShape(h.words, h.scales, h.biases, h.bits, h.group, h.words.rank == 3);
    if (!convert.tableDtype(h.scales.dtype) or !convert.tableDtype(h.biases.dtype)) return error.UnexpectedTensor;
    const rank = h.words.rank;
    var n_rows: usize = 1;
    for (h.words.shape[0 .. rank - 1]) |d| n_rows *= d;
    const groups = h.scales.shape[h.scales.rank - 1];
    const k = groups * h.group;
    const words_row = h.words.shape[rank - 1];
    if (out.len != dims.elements) return error.UnexpectedTensor;
    const mask: u64 = (@as(u64, 1) << @intCast(h.bits)) - 1;
    for (0..n_rows) |r| for (0..k) |c| {
        const bit = c * h.bits;
        const at = r * words_row + (bit >> 5);
        const lo: u64 = std.mem.readInt(u32, h.words.bytes[4 * at ..][0..4], .little);
        const hi: u64 = if ((bit & 31) + h.bits > 32) std.mem.readInt(u32, h.words.bytes[4 * (at + 1) ..][0..4], .little) else 0;
        const code: f64 = @floatFromInt(((lo | hi << 32) >> @intCast(bit & 31)) & mask);
        const g = r * groups + c / h.group;
        out[r * k + c] = code * table64(h.scales.dtype, h.scales.bytes, g) + table64(h.biases.dtype, h.biases.bytes, g);
    };
}
```
- **Excerpt** (`mlx.zig:254-271`, the fp32 path that reproduces torch bit for bit):
```zig
    for (0..n_rows) |r| for (0..k) |c| {
        // a code may run into the next word (3, 5, 6 and 7 bits): read the pair
        const bit = c * bits;
        const at = r * row_words + (bit >> 5);
        const lo: u64 = std.mem.readInt(u32, words.bytes[4 * at ..][0..4], .little);
        const hi: u64 = if ((bit & 31) + bits > 32)
            std.mem.readInt(u32, words.bytes[4 * (at + 1) ..][0..4], .little)
        else
            0;
        const code: f32 = @floatFromInt(((lo | hi << 32) >> @intCast(bit & 31)) & mask);
        const g = r * scale.shape[1] + c / group;
        const s = convert.load(scale.dtype, scale.bytes, g);
        const b = convert.load(bias.dtype, bias.bytes, g);
        // torch multiplies, then adds: two roundings
        const prod = code * s;
        out[r * k + c] = prod + b;
    };
    return out;
}
```
- **Teaching value**: Dequantisation is one multiply and one add per weight: **`w = scale * q + bias`** with `q` an unsigned `bits`-wide integer. There is no zero-point subtraction and no symmetric/asymmetric branch: the "zero point" is folded into `bias` (which is therefore typically negative).
- **Gotchas**:
  - **Codes are unsigned** (`0..2^bits-1`), never sign-extended. The affine map does all the signing.
  - **Low code first**: the test at `mlx.zig:279-285` feeds one word `0x76543210` with `scale = 2.0` (bf16 `0x4000`), `bias = -1.0` (bf16 `0xBF80`), `bits = 4`, `group = 8` and expects `{-1, 1, 3, 5, 7, 9, 11, 13}` — i.e. code 0 is the **low** nibble of the word, and `w_i = 2*i - 1`.
  - **8 nibbles per 32-bit word** at 4 bits; code `c` of a row lives at bit `c*4`, so word `c>>3`, shift `(c*4) & 31`.
  - The fp32 path keeps `(code*s) + b` as two separate roundings on purpose — fusing it into an FMA would diverge from torch/MLX.
  - `reference` folds a stack's leading dimensions (`E`) into `n_rows`, so an expert stack decodes with the same loop.

### Exact recipe for a JavaScript dequantiser (4-bit affine, from the above alone)

Given, for a module, `words` of shape `[N, K*bits/32]` (read as **unsigned** 32-bit little-endian), `scales` and `biases` of shape `[N, K/group]` (dtype f32, bf16 or f16), and `(bits, group)` from `config.json`:

```
groups   = scales.shape[1]
K        = groups * group
rowWords = words.shape[1]            // must equal K*bits/32
mask     = (1 << bits) - 1

for r in 0..N-1:
  for c in 0..K-1:
    bit  = c * bits
    at   = r * rowWords + (bit >>> 5)
    lo   = words[at] >>> 0                 // unsigned 32-bit, little-endian
    hi   = ((bit & 31) + bits > 32) ? (words[at + 1] >>> 0) : 0
    code = Number(((BigInt(lo) | (BigInt(hi) << 32n)) >> BigInt(bit & 31)) & BigInt(mask))
    g    = r * groups + ((c / group) | 0)
    W[r][c] = code * scale(g) + bias(g)
```
For 4 bits the BigInt is unnecessary (`(bit & 31) + 4 <= 32` always): `code = (words[at] >>> (bit & 31)) & 0xF`, and `at = r*rowWords + (c >>> 3)`.
bf16 table entries widen exactly: `f32bits = u16 << 16` (see §16/§17).
Self-check against the repo's own test vector: one word `0x76543210`, scale 2.0, bias −1.0, group 8 → `[-1, 1, 3, 5, 7, 9, 11, 13]`.

## 11. Other quantisation formats TensorFold reads

| Format | Codes | Scale granularity | Formula | Anchor |
| --- | --- | --- | --- | --- |
| MLX affine | unsigned `bits` ∈ {2,3,4,5,6,8}, packed in i32 words | one scale **and** bias per `group` ∈ {32,64,128} per row | `scale*q + bias` | `zig/src/core/quant/mlx.zig:217-237` |
| Block FP8 | e4m3 bytes | one `scale_inv` per **128×128 block** | `e4m3(code) * scale_inv` | `zig/src/core/quant/fp8.zig:13-19,56-68` |
| NVFP4 (ModelOpt) | e2m1 nibbles, 2 per byte | e4m3 scale per **16 inputs**, plus one f32 global per matrix | `e2m1(q) * e4m3(s) * global` | `zig/src/core/quant/nvfp4.zig:9-15,26-42` |
| csf0_v1 NVFP4 | e2m1 nibbles | e4m3 per 32 inputs, stored as a per-row base byte + 4-bit per-group offset | `e2m1(q) * e4m3(base+offset) * weight_scale_2` | `docs/quantization.md`, "csf0_v1" |
| EXL3 / EAQ | tail-biting trellis, int16 tiles `[K/16, N/16, words]` + `suh` fp16 `[K]`, `svh` fp16 `[N]` | per-matrix Hadamard scales | codebook lookup (`inst3` / `mcg` / `mul1`) | `zig/src/core/exl3_layout.zig:27-30`, `zig/src/core/exl3_format.zig:1` |
| oQ4/oQ6/oQ8 | **not a separate format** — these are oMLX "oQ" *quantisation recipes* producing MLX affine files | as MLX affine | `scale*q + bias` | `docs/recipes/nemotron-3.5.md:26-35` |

- **Excerpt** (`fp8.zig:13-19` — e4m3 decoded from first principles):
```zig
pub fn e4m3(b: u8) f64 {
    if (b & 0x7f == 0x7f) return std.math.nan(f64);
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xF;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8.0 * std.math.pow(f64, 2, -6) else (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}
```
- **Excerpt** (`nvfp4.zig:26-42` — three multiplies, nibble unpacking):
```zig
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale.rank != rank or h.weight.dtype != .u8 or h.scale.dtype != .f8_e4m3) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1] * 2;
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale.shape[d]) return error.UnexpectedTensor;
    if (h.scale.shape[rank - 2] != n or h.scale.shape[rank - 1] * group != k or out.len != elements(h)) return error.UnexpectedTensor;
    const stack = elements(h) / (n * k);
    if (h.global.len != stack) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const row = e * n + r;
        const byte = h.weight.bytes[row * (k / 2) + c / 2];
        const nib: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
        out[row * k + c] = e2m1(nib) * fp8.e4m3(h.scale.bytes[row * (k / group) + c / group]) * h.global[e];
    };
}
```
- **Teaching value**: Every format is the same idea at different granularities — *small integers plus a scale*. What differs is how many weights share a scale (16 / 32 / 64 / 128 / a 128×128 block) and whether the scale itself is quantised (NVFP4's scales are themselves FP8).
- **Gotchas**: e2m1 magnitudes are the fixed table `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (`nvfp4.zig:13`) — a lookup, not an exponent/mantissa computation. e4m3 `0x7F`/`0xFF` are NaN, which is why csf0_v1 refuses a row whose `base + offset` exceeds `0x7E`. NVFP4 scales are *themselves* FP8, so dequantisation is a product of **three** numbers. EXL3 is not a scale-and-code format at all: it is a trellis whose 65,536-state codebook is hashed and verified at load (`zig/src/core/eaq_container.zig:251-262`).

## 12. The format interface: one `conforms` list, nothing above names a format

- **Anchor**: `zig/src/core/quant/quant.zig:37-61`
- **What it does**: `read_order` is the detection order (`dense`, then `mlx`); the first module that `matches` a tensor reads it. `conforms` is a compile-time contract listing everything a format module must provide — host read, TP slicing, device upload, the kernel view, and an **fp64 `reference`**.
- **Excerpt**:
```zig
/// The order a tensor is offered to the formats: the first that recognizes it reads it.
const read_order = [_]Format{ .dense, .mlx };

/// What a format module provides, checked at compile time.
pub fn conforms(comptime M: type) void {
    const required = .{
        "id", "decoder", // which format, and the device decoder its kernels use
        "Config", "detect", // the checkpoint's own declaration of the format (null: not this format)
        "Host", "matches", "read", "rows", "bytes", "halves", // a tensor's host side: recognized, read and split
        "sliceRows", "sliceCols", "sliceStack", // tensor parallelism: output rows, input columns, experts
        "Device", "upload", // load: onto the device, repacked there when the decoder reads a layout of its own
        "Matrix", "View", "view", // what the kernels read of an uploaded tensor
        "elements", "reference", // fp64 dequantization of every weight, for the tests
    };
    inline for (required) |name| {
        if (!@hasDecl(M, name)) @compileError("a format module needs `" ++ name ++ "`");
    }
}
```
- **Teaching value**: Every weight format must be able to state, in fp64, what each of its weights *is*. That fp64 oracle is what makes "did quantisation break the model?" an answerable question.
- **Gotchas**: `upload` for MLX is a straight copy — "this format keeps its layout" (`mlx.zig:177-181`), so what the GPU holds is byte-identical to what is on disk. The `bf16` format is opt-in and deliberately absent from `read_order` so an unquantized checkpoint never shadows a quantised one.

---

## 13. Quantisation error: KL divergence and top-1 agreement

- **Anchor**: `zig/src/cli/cuda_kl.zig:88-111` (loop), `:7-37` (file format and log-softmax)
- **What it does**: `tensorfold kl MODEL --tokens IDS --dump FILE` writes one precision's logits rows to disk (a `u32` row count, a `u32` vocab, then per row the fed token as `u32` plus the whole bf16 logit row). `--against FILE` then runs the other precision teacher-forced on the same tokens and reports mean/max `KL(reference ‖ this run)` in nats, top-1 agreement, the first differing row, and the mean NLL of the fed tokens under each.
- **Excerpt**:
```zig
    for (0..rows) |i| {
        const at = header_bytes + i * rec;
        const t = std.mem.readInt(u32, ref[at..][0..4], .little);
        const want = ref[at + 4 ..][0 .. v * 2];
        const row = try download(gpa, e, if (i == 0) e.b.p_logits else e.b.logits, v * 2);
        defer gpa.free(row);
        logSoftmax(want, lp);
        logSoftmax(row, lq);
        var kl: f64 = 0;
        for (lp, lq) |p, q| kl += @exp(p) * (p - q);
        kl_sum += kl;
        if (kl > kl_max) {
            kl_max = kl;
            kl_at = i;
        }
        const same = argmax(row, v) == argmax(want, v);
        top1 += @intFromBool(same);
        if (!same and first_diff == null) first_diff = i;
        nll[0] -= lp[t];
        nll[1] -= lq[t];
        if (i + 1 < rows) _ = try e.step(t, null);
    }
```
- **Teaching value**: Quality loss from quantisation is measured as a *distribution* distance (`KL = Σ p·(log p − log q)`, exact over the full vocabulary) plus the blunter "does it still pick the same token?" — not as weight error.
- **Gotchas**: `logSoftmax` is done in f64 from bf16 logits with the max subtracted (`:27-37`), so the comparison is not itself a precision artifact. **Published numbers to cite:**
  - Flash Next, full 248,320-entry vocabulary, 31,728 scored tokens on WikiText-2 and HumanEval: KL(bf16 ‖ quant) 0.039 for oQ4e vs 0.059 for the 5-bit-mixer variant — ~1.5× (`docs/recipes/qwen3.8-flash-next.md:506-516`). MMLU-Pro 55.9% for uniform 4-bit vs 69.1% for oQ4e.
  - Spark X2.6, 8-bit MLX affine vs the bf16 release: mean KL 0.0021 (`docs/recipes/spark-x2.6-4b-metal.md:76-78`).
  - Nemotron Super VL, NVFP4 `checkpoint` vs `full`: mean KL 0.0077 nats/token, same top token 756/768 = 98.4% (`docs/quantization.md`, NVFP4 section).
  - The bf16 engine itself vs HF fp32: mean KL 0.00076, top-1 99.6% — HF's *own* bf16 is 0.00048 from the same fp32 run, top-1 99.3% (`docs/recipes/spark-x2.6-4b-metal.md:51-55`). Useful callout: bf16 is itself a lossy quantisation.
  - There is also a weight-level comparator, `lora_quant.agreement` (`zig/src/core/lora_quant.zig:64`), reporting the share of identical scales, biases, codes and whole groups; `tf-merge-check` prints it (`zig/src/core/lora_gate.zig:113-115`).

---

## 14. The quantised matmul kernel: `simd_qmm.metal`

- **Anchor**: `zig/kernels/metal/core/simd_qmm.metal:46-68` (decomposition), `:117-145` (dequant inside the MMA)
- **What it does**: `tf_sq_tiles<S, NT, RT, H>` computes `OUT[R, N] = X[R, K] · W[N, K]^T` for 4-bit MLX affine `W`. A threadgroup owns an `8*RT × 8*NT` output tile (`rb = tg.y * 8*RT`, `nb = tg.x * 8*NT`); within a simdgroup, lane `l` maps to `fm = ((l/4) & 4) + ((l/2) % 4)` (the **weight-row / output-column** index inside an 8×8 tile) and `fn = ((l/4) & 2)*2 + (l%2)*2` (the **row-pair** index, so each lane holds two X rows). K is cut into 64-input groups `g`; the `S` simdgroups of the threadgroup take chunks `c, c+SGS, …` and a fixed binary tree sums their partials through `threadgroup float* red`.
- **Excerpt** (`:46-68`, who computes what):
```cpp
// R rows in 8-row tiles, NT 8-column tiles; K chunks c, c + SGS, ... summed by a fixed tree; loops unrolled (+15-20%).
template <int S, int NT, int RT, bool H>
inline void tf_sq_tiles(const device uint4* X, const device uint* W, const device bfloat* SC, const device bfloat* BI,
                        constant TfSqArgs& a, device bfloat* OUT, const device uint* FL, uint sgi, uint sgs, uint lane,
                        uint3 tg, threadgroup float* red) {
  const int K = a.K, N = a.N, R = a.R, G = K / 64, SGS = int(sgs);
  const float one = a.one;
  const int sg = int(sgi);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int nb = int(tg.x) * (8 * NT);
  const int rb = int(tg.y) * (8 * RT);
  const device uint2* W2 = (const device uint2*)W;
  int wrow[NT];
  #pragma unroll
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  #pragma unroll
  for (int rt = 0; rt < RT; rt++) {
    xr0[rt] = min(rb + 8 * rt + fn, R - 1);
    xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1);
  }
```
- **Excerpt** (`:117-145`, dequantisation *inside* the simdgroup MMA, nothing materialised):
```cpp
      #pragma unroll
      for (int s = 0; s < 8; s++) {
        const float ps = tf_sq_pre(s);
        const uint mask = 0xFu << (4 * s);
        simdgroup_matrix<float, 8, 8> bm[RT];
        #pragma unroll
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = tf_sq_bf8(xa[rt], s) * ps;
          bm[rt].thread_elements()[1] = tf_sq_bf8(xb[rt], s) * ps;
        }
        #pragma unroll
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = float(wv[t].x & mask);
          am.thread_elements()[1] = float(wv[t].y & mask);
          #pragma unroll
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      #pragma unroll
      for (int t = 0; t < NT; t++) {
        const float sc = fast ? float(SC[size_t(wrow[t]) * G + g]) * 4096.0f : float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);
        #pragma unroll
        for (int rt = 0; rt < RT; rt++) {
          acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
          acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
        }
      }
```
- **Teaching value**: The weights are **never dequantised into memory**. The affine map is distributed over the dot product:
  `Σ_c x_c·(s·q_c + b) = s·(Σ_c x_c·q_c) + b·(Σ_c x_c)`.
  So the kernel accumulates two things per group — the integer-code dot product `P` and the plain input sum `xs0/xs1` (computed by `tf_sq_sum8` plus three `simd_shuffle_xor` reductions at `:85-90`) — and folds in `scale` and `bias` **once per 64-input group** instead of once per weight. That is the trick that makes a quantised matmul cheaper than a dense one.
- **Gotchas**:
  - `tf_sq_pre(s)` returns `2^-4s` as a raw exponent construction (`:21-22`): it pre-scales the *input* so the weight nibble can be masked **in place** (`wv[t].x & mask`, still sitting at bit position `4s`) and converted to float with no shift. One mask instead of a shift-and-mask per code.
  - `a.one` is a run-time `1.0f` multiplied into every `fma` so the compiler **cannot reorder or fold** the sums — bit-exact reproducibility is enforced against reassociation.
  - The `fast` path (`:99-115`) uses `half` operands: a nibble becomes `as_type<half>(0x3400 | q) - 0.25h`, i.e. `q·2^-12` exactly in half; the group scale is then pre-multiplied by `4096.0f` to compensate. It is only taken when a prepass kernel, `tf_sq_fits` (`:25-44`), has proved every input in that 16-row × 64-input tile is zero or a normal half (`|v| ∈ [2^-14, 65504]` and `float(half(f)) == f`), recorded as a flag word in `FL`.
  - `min(…, N-1)` / `min(…, R-1)` clamps instead of branching: out-of-range lanes compute a duplicate row and are discarded at the store (`if (row < R && n < N)`).
  - Variants in the same file: `tf_sq_mma` (plain), `tf_sq_mmh` (half fast path, takes `FL`), `tf_sq_mmp` (P adjacent K chunks per simdgroup, wider tiles) — `:171-300`.
  - Width-general (2/3/5/6/8-bit) twin: `zig/kernels/metal/core/simd_qmm_bits.metal:28-54` — `tf_sqw_load` loads `2*BITS` bytes per lane and `tf_sqw_code` does the general shift/straddle, so only 4-bit (and 2-bit, via `tf_sqw_pre2`) gets the mask-in-place shortcut.

## 15. The decode-shaped quantised matvec: `tile_qmv.metal`

- **Anchor**: `zig/kernels/metal/core/tile_qmv.metal:11-18` (code unpack), `:181-207` (per-lane decomposition), `:209-220` (the kernel)
- **What it does**: The decode twin of the tiles: no simdgroup matrices, just fma chains in the same order. One `tg.y` handles one `X` row; `TQ_SG = 2` simdgroups per threadgroup, each owning `NC = 4` output columns. Inside a simdgroup, **lane `l` handles K-group `g0 + l`** — 32 groups (2,048 inputs) per round — and stages its `NC` code-dot partials plus the input sum into `threadgroup float* part`. Then **lane `c` (c < NC) chains column `c`'s 32 groups in order**, folding `scale` and `bias` per group.
- **Excerpt** (`:11-18`, the branch-free nibble decode):
```cpp
// A 4-bit word's eight codes in k order as exact integers: codes i and i + 4 as half 1024 + code, minus 1024.
inline void tf_tq4_codes(uint word, thread float* cf) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const half2 h = as_type<half2>(((word >> (4 * i)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
    cf[i] = float(h.x); cf[i + 4] = float(h.y);
  }
}
```
- **Excerpt** (`:182-207`, the two-phase lane decomposition):
```cpp
template <int BITS, int NC>
inline float tf_tq_columns(const device bfloat* x, float one, const device uint8_t* w, const device bfloat* sc,
                           const device bfloat* bi, int K, uint lane, threadgroup float* part) {
  const int G = K / 64;
  const size_t stride = size_t(K) * BITS / 8;
  threadgroup float* sums = part + NC * 32;
  const int c = min(int(lane), NC - 1);
  float acc = 0.0f;
  for (int g0 = 0; g0 < G; g0 += 32) {
    const int g = g0 + int(lane);
    if (g < G) {
      float p[NC];
      sums[lane] = tf_tq_group<BITS, NC>(x, w, stride, g, p, one);
      for (int j = 0; j < NC; ++j) part[j * 32 + lane] = p[j];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const device bfloat* s = sc + c * G + g0;
    const device bfloat* b = bi + c * G + g0;
    const threadgroup float* pc = part + c * 32;
    const int n = min(32, G - g0);
    #pragma unroll(8)
    for (int j = 0; j < n; ++j) acc = fma(float(b[j]), sums[j], fma(float(s[j]), pc[j], acc));
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  return acc;
}
```
- **Teaching value**: Same algebra as the tiles (`acc += s·Σxq + b·Σx`), but the parallel decomposition inverts: at R=1 there is no output tile to fill, so the **K axis** is what gets spread over the 32 lanes, and a barrier-mediated transpose in threadgroup memory hands each column's 32 partials to one lane for the ordered fold.
- **Gotchas**: `tf_tq4_codes` decodes **two codes at once** with no shifts past the first: OR-ing `0x6400` onto a nibble builds the half `1024 + q` directly (half's exponent field makes `1024+q` exactly representable for `q ≤ 15`), and subtracting `1024.0h` leaves the integer. Codes `i` and `i+4` come from the two 16-bit halves of the same `uint`. For 6/8-bit the general path reads a 64-bit pair and shifts (`tf_tq_group`, `:57-69`). `tf_tile_qmv_few` (`:222-240`) takes `TQ_RT = 4` rows over one code load — the same "load weights once, reuse across rows" lever as the bf16 kernel. `tf_tile_qmv_glu` (`:258-...`) fuses gate and up projections plus SwiGLU into one launch so the activation never round-trips to memory. The ordered fold is deliberate: it reproduces the MMA path's summation order so one-row and windowed results agree bit for bit.

## 16. The dense/unquantised matvec: `bf16_matvec.metal`

- **Anchor**: `zig/kernels/metal/core/bf16_matvec.metal:7-31`
- **What it does**: `Y[r, j] = bf16(Σ_i X[r, i] · W[j, i])`, with `W` stored bf16 `[n, k]`, accumulating in fp32. **One simdgroup computes one output element `j`** (per row `r`); its 32 lanes each take 8 contiguous inputs (`i = 8*lane`, striding by 256 = 32 lanes × 8), load them as two `bfloat4` → `float4`, and `dot` them. `simd_sum` reduces the 32 lane accumulators, lane 0 stores. Instantiated for ROWS = 1, 2, 4, 8, 16, 32 (`:33-46`).
- **Excerpt**:
```cpp
// Y[r, j] = bf16(sum_i X[r, i] * W[j, i]) over bf16 W [n, k] in fp32 (torch's bf16 linear): a simdgroup a row j.
template <uint ROWS>
inline void tf_bf16_matvec_t(const device bfloat* X, const device bfloat* W, constant tf_bf16_mv_args& a,
    device bfloat* Y, uint3 tg, uint sg, uint lane) {
  constexpr uint SG = 8;
  const uint j = tg.x * SG + sg;
  if (j >= a.n) return;
  const device bfloat* w = W + (size_t)j * a.k;
  float acc[ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  for (uint i = 8 * lane; i < a.k; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat* x = X + (size_t)r * a.k + i;
      acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));
    }
  }
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    if (lane == 0 && r < a.rows) Y[(size_t)r * a.ldy + j] = bfloat(s);
  }
}
```
- **Teaching value**: The cleanest reference matmul in the repo and the right place to start a lesson: a thread owns 8 numbers, a simdgroup owns one output, the whole kernel is one `dot` in a loop. The quantised kernels are this plus on-the-fly decoding.
- **Gotchas**: The **weight row `w` is loaded once and reused across all ROWS** — that is the entire reason multi-row (batched) decode is nearly free: the `r` loop adds FLOPs but no weight bytes. `SG = 8` simdgroups per threadgroup, so `tg.x` strides 8 output elements. Output is rounded back to bf16 but sums stay fp32 — exactly torch's bf16 `linear`. `k` must be a multiple of 8 for the `bfloat4` pair loads; the stride of 256 assumes a 32-wide simdgroup.

## 17. dtype zoo: bf16 is a truncated fp32

- **Anchor**: `zig/src/core/bf16_round.zig:17-32`, `zig/src/core/quant/convert.zig:12-20`, GPU-side `zig/kernels/metal/core/simd_qmm.metal:8-12`
- **What it does**: `value(h) = @bitCast(@as(u32, h) << 16)` — a bf16 is literally the **top 16 bits of an fp32**: same sign, same 8-bit exponent, mantissa truncated from 23 to 7 bits. `bits(v)` is the other direction with round-to-nearest-even implemented as an integer add (`u + 0x7fff + ((u >> 16) & 1)`), plus a NaN branch so a NaN stays a NaN. `convert.load` is the host reader for the three float widths.
- **Excerpt** (`bf16_round.zig:17-32`):
```zig
/// fp32 to bf16 bits, ties to even, as __float2bfloat16_rn rounds (NaN stays NaN).
pub fn bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// The fp32 value of bf16 bits.
pub fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// fp32 rounded to bf16 and back: a kernel's `__bfloat162float(__float2bfloat16_rn(v))`.
pub fn round32(v: f32) f32 {
    return value(bits(v));
}
```
- **Excerpt** (`convert.zig:12-20` — the three float widths side by side):
```zig
/// One element as fp32 (bf16 and fp16 widen exactly).
pub fn load(dtype: DType, bytes: []const u8, i: usize) f32 {
    return switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)))),
        else => unreachable,
    };
}
```
- **Excerpt** (`simd_qmm.metal:8-12` — the same shift on the GPU, unpacking 8 bf16 from a `uint4`):
```cpp
// The bf16 at index e (0..7) of 8 packed bf16, as fp32.
inline float tf_sq_bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
```
- **Teaching value**: bf16 → fp32 is a **left shift by 16 and nothing else**. No exponent re-biasing, no table, no special cases: bf16 has fp32's full dynamic range and a quarter of its precision. This is the single most convincing demonstration that floats are just bit patterns.
- **Gotchas**:
  - The integer rounding trick is exercised against three historical spellings over every zero, subnormal, infinity, NaN and tie plus millions of random words (`bf16_round.zig:100-130`); the old "plain `+`" version overflowed negative NaNs, and `bits` of fp32's most negative value is `-Inf` (`0xff80`).
  - The GPU version exploits the identity in reverse: an fp32 whose low 16 bits are zero *is* a bf16, so masking with `0xFFFF0000` extracts the odd element with **no shift at all**; the even element is `w << 16`. Two bf16 per `uint`, eight per `uint4`.
  - `bf16_round.bf16(f64)` (`:5-15`) is a separate **single-rounding** f64→bf16 for references — rounding twice (f64→f32→bf16) is not the same number.
  - fp16 is *not* a truncation: it needs a real float cast (different exponent width), hence `@floatCast`.
  - fp8/fp4 conversions: `fp8.e4m3` / `fp8.toE4m3` (`zig/src/core/quant/fp8.zig:13-30`), `nvfp4.e2m1` (`:12-15`), and an independent set in `zig/src/families/drama3/metal/quant_reference.zig:5-47` — including a `roundBf16` that walks neighbouring representable values instead of using the integer trick, deliberately so the production trick is checked against logic it does not share.

---

## 18. Decode is memory-bound: 1 token = read every weight once

- **Anchor**: `docs/recipes/spark-x2.6-4b-metal.md:62-64,73-78`, `docs/optimizing.md:102-106`, `docs/WINDOWS.md:117`, `docs/recipes/qwen3.8-flash-next.md:498-501`, `zig/tests/flashnext_widths_rows.zig:322`
- **What it does**: These are the repo's own statements of the arithmetic. A single-stream decode step reads the entire weight set once; throughput is `bandwidth / weight_bytes`, and extra concurrent streams are nearly free because the weights are read **once per round for every row**.
- **Excerpt** (`docs/recipes/spark-x2.6-4b-metal.md:62-64`, verbatim):
```
- Decode: 106 tokens/s with one stream (9.5 ms a step: 7.7 GiB of weights a step, the memory bandwidth), 179-190 in
  all with two, 296-326 with four (12.8 ms a round: the weights read once for every row).
- Cold prompts: 2,350 tokens 9,438 tokens/s, 9,354 tokens 10,940, 37,174 tokens 9,058, 148,588 tokens 5,002 (29.7 s).
```
- **Excerpt** (`docs/recipes/spark-x2.6-4b-metal.md:73-78`, verbatim):
```
- **Streams (`--parallel`).** One stream decodes at 106 tokens/s, bound by reading the 7.7 GiB of weights each step.
  The weights are read once a round for every stream's row: two streams give 179-190 tokens/s in all, four 296-326
  (12.8 ms a round against 9.6), each reply equal to its run alone; the first token moves from 0.05-0.07 s to
  0.13-0.17 s at four. The row kernel holds up to 32 rows a launch.
- **Weights.** The release's bf16 is exact. The CUDA page's MLX affine conversions also load: the 8-bit one decodes at
  166 tokens/s (4.1 GiB a step) for a mean KL of 0.0021 against the release and greedy replies that part within a
  hundred tokens on two of four chats; 4-bit was not measured. Neither is needed on a Mac.
```
- **Excerpt** (`docs/optimizing.md:102-106`, verbatim):
```
## A null on a bandwidth-bound box

A GB10 is bandwidth-bound for most decode work. A compute-side change (fewer instructions, better tiles, more
overlap) can measure as a null there and still win on a discrete card. A null on a GB10 is not a verdict for that kind
of change: open the PR with the GB10 numbers and ask for a run on a discrete card.
```
- **Excerpt** (`docs/recipes/qwen3.8-flash-next.md:498-501`, verbatim):
```
The native HIP build serves Flash Next on gfx1151 (AMD Ryzen AI MAX+ 395, Radeon 8060S, 128 GB unified memory) from MLX
affine checkpoints whose fused stacks each have one bit width and group size. Decode on this iGPU is bandwidth-bound and
the dense mixers (attention, Gated DeltaNet projections, hyper-connection mixers, n-gram key/value projections) dominate
each decoded token's reads, so the checkpoint choice sets both speed and quality. Three checkpoints load:
```
- **The arithmetic, with the repo's own numbers**:
  - **Check the bf16 claim**: 7.7 GiB / 9.5 ms ≈ 810 GiB/s ≈ the box's memory bandwidth. 1 / 9.5 ms = 105 tok/s; measured 106.
  - **Check the quantised claim**: the same model at 8-bit reads 4.1 GiB/step and decodes at 166 tok/s. Bytes ratio 7.7/4.1 = 1.88×; speed ratio 166/106 = 1.57× — the ratio tracks *bytes*, not FLOPs (the arithmetic is unchanged, and some of it even increases).
  - **Check batching**: 4 streams take 12.8 ms vs 9.6 ms for 1 — 4× the arithmetic for 1.33× the time, because the weight traffic is unchanged.
  - **Windows page**: "Qwen3.8-27B (16 GiB): 40 tokens/s plain, **at the card's memory bandwidth**" (`docs/WINDOWS.md:117`) → 16 GiB × 40/s = 640 GiB/s.
  - **Roofline constant in code**: `const peak = 800.0; // GB/s: the M3 Ultra's memory bandwidth, the roofline a weight-bound launch is held to` (`zig/tests/flashnext_widths_rows.zig:322`); the benchmark prints each kernel's achieved GB/s as a percentage of it (`:357-362`).
  - **Nominal bandwidth table** for the chips served: `zig/src/cluster/node.zig:96-100` (`M3 Ultra 819`, `M2 Ultra 800`, `M1 Ultra 800`, `M4 Max 546`, `M3 Max 400` GB/s), with the doc comment "the engine measures it" when unknown.
- **Teaching value**: Generating one token requires touching every weight exactly once, so a decode step is a *streaming read* of the model file, and tokens/second ≈ bandwidth ÷ model size. This is why quantisation makes models faster (fewer bytes), why batching is nearly free (same bytes, more arithmetic), and why a faster GPU core can change nothing.
- **Gotchas**:
  - Prompt/prefill is the **opposite** regime: many rows share one weight read, so prefill is compute-bound. `docs/optimization-ledger.md:336-338` records an FP4 kernel win on prompt16 and a measured null on FP8 block experts "as decode there is bandwidth-bound, so FP8-only families are `failed`".
  - `docs/optimizing.md:12-14`: "For a bandwidth-bound kernel [the ceiling on its gain] is its time minus the time its bytes take at the device floor."
  - On an MoE model only the *routed* experts are read, so "every weight" means all dense layers plus the selected experts — `docs/design/fluid-subgraph-residency.md:11`: "Let B be cold expert bytes read per generated token and S be measured sustained SSD bytes per second. With B greater than zero, storage alone bounds single-stream throughput by S / B tokens per second."
  - Speculative decoding breaks the "1 token = 1 weight pass" rule in the useful direction: one pass verifies several drafted tokens (MTP acceptance 3.05 tokens/round, `docs/recipes/qwen3.8-flash-next.md:506-507`).

---

## 19. Real quantities to quote in the course

| Quantity | Value | Where it is defined |
| --- | --- | --- |
| safetensors header prefix | 8 bytes, little-endian u64 | `zig/src/core/safetensors.zig:8-11` |
| Max header size | 100,000,000 bytes | `zig/src/core/safetensors.zig:6` |
| Max tensor rank | 5 | `zig/src/core/safetensors.zig:47` |
| MLX affine bit widths | 2, 3, 4, 5, 6, 8 (never 7) | `zig/src/core/quant/mlx.zig:18`; `mlx_geometry.zig:31` |
| MLX affine group sizes | 32, 64, 128 | `zig/src/core/quant/mlx_geometry.zig:32` |
| 4-bit codes per 32-bit word | 8 (low code first) | `zig/src/core/quant/mlx.zig:81`, test at `:279-285` |
| FP8 block scale granularity | 128 × 128 | `zig/src/core/quant/fp8.zig:10` |
| NVFP4 scale granularity | 16 inputs | `zig/src/core/quant/nvfp4.zig:9` |
| csf0_v1 NVFP4 granularity | 32 inputs (row base + 4-bit offset) | `docs/quantization.md`, csf0_v1 section |
| e2m1 magnitudes | {0, 0.5, 1, 1.5, 2, 3, 4, 6} | `zig/src/core/quant/nvfp4.zig:13` |
| e4m3 max finite | 448 (`0x7E`); `0x7F`/`0xFF` are NaN | `zig/src/core/quant/fp8.zig:13-19` |
| Runtime memory margin | 2 GiB (`2 << 30`) | `zig/src/platform/memory_policy.zig:2` |
| Flash Next vocabulary | 248,320 | `docs/recipes/qwen3.8-flash-next.md:520`; the `248320 × 2560` head shape at `zig/tests/flashnext_widths_rows.zig:360` |
| Flash Next hidden size | 2,560 | decode shapes at `zig/tests/flashnext_widths_rows.zig:357-360` |
| Flash Next head_dim | 256 (attention); 128 (linear k/v, indexer) | `zig/src/families/qwen4/cuda_attention.zig:14-16`, `native_admission.zig:9-10,78-81` |
| Flash Next conv kernel width | 4 | `zig/src/families/qwen4/native_admission.zig:11` |
| Q/K/V projection shapes | `q: [2*heads*head_dim, hidden]`, `k/v: [kv_heads*head_dim, hidden]`, `o: [hidden, heads*head_dim]` | `zig/src/families/qwen4/model.zig:109-112`, `weights.zig:54-57` |
| Draft vocabulary (Flash Next) | 79,591 sorted ids, padded to a multiple of 64 | `docs/recipes/qwen3.8-flash-next.md:548` |
| EXL3 trellis tile | int16 `[K/16, N/16, words]`, `suh` fp16 `[K]`, `svh` fp16 `[N]` | `zig/src/core/exl3_layout.zig:27-30` |
| EXL3 codebook states | 65,536, sha256-verified at load | `zig/src/core/eaq_container.zig:251-262` |
| Decode weight traffic (Spark X2.6, bf16) | 7.7 GiB/step → 106 tok/s | `docs/recipes/spark-x2.6-4b-metal.md:62-64` |
| Decode weight traffic (same, 8-bit MLX) | 4.1 GiB/step → 166 tok/s | `docs/recipes/spark-x2.6-4b-metal.md:76-78` |
| Roofline constant in tests | 800 GB/s (M3 Ultra) | `zig/tests/flashnext_widths_rows.zig:322` |

## 20. Reference vs optimised, paired

| Concept | Reference / naive | Optimised / kernel | How they differ |
| --- | --- | --- | --- |
| MLX affine dequant | `mlx.reference` — fp64, scalar, one weight at a time (`mlx.zig:217-237`) | `simd_qmm.metal:117-145` | The reference materialises `w`; the kernel never does — it distributes `s` and `b` over the dot product, folding them once per 64-input group |
| MLX affine dequant, torch-exact | `mlx.dequant` — fp32, `(q*s)+b` as two separate roundings (`mlx.zig:240-272`) | — | Exists purely to reproduce torch/MLX bit for bit; the fp64 `reference` is the *quality* oracle, this one is the *bit-compatibility* oracle |
| Quantised matmul, prompt | `tile_qmm.metal` — 32×32 staged tiles, codes decoded once per chunk (`:1-63`) | `simd_qmm.metal` — 8-row simdgroup-matrix tiles; `simd_qmm_bits.metal` for 2–8 bit | `simd_qmm` is row-exact at small R; `tile_qmm` amortises code decode over 32 rows and uses a 256-entry constant table (`tf_tq_pairs`) for nibble→bf16 pairs |
| Quantised matmul, decode | `tile_qmv.metal:182-207` — scalar fma chains, "the tiles' twin" | `simd_qmm.metal` fast path with the `tf_sq_fits` prepass | The twin keeps register pressure low at R=1–4 and reproduces the MMA summation order exactly, so one-row and tiled results agree bit for bit |
| Dense matmul | `bf16_matvec.metal:7-31` | — | This *is* the clean reference: 8 inputs a lane, one output a simdgroup, fp32 sums |
| FP4/FP8 projection | `drama3/metal/quant_reference.zig:49-57` — fp64 scalar `fp4(q)*fp8(s)*bf16(x)` | the family's Metal kernels | The reference is "never used for inference" by design — an independent oracle for the operator gate |
| bf16 rounding | `quant_reference.roundBf16` — walks neighbouring representables (`:37-47`) | `bf16_round.bits` — integer add trick (`:17-22`) | Deliberately different implementations so one checks the other |
| FP8 / NVFP4 host side | `fp8.reference`, `nvfp4.reference` (fp64) | each backend's own kernel layouts | `quant.zig:16-17` notes fp8/nvfp4 are "host only (no Format: their kernels' layouts are each backend's)" |

- **Excerpt** (`zig/src/families/drama3/metal/quant_reference.zig:49-57` — the clearest "it is just numbers" statement in the repo: one output of one layer of a real LLM, from raw bytes, in nine lines):
```zig
pub fn projection(codes: []const u8, scales: []const u8, global: f32, n: usize, k: usize, x: []align(1) const u16) f64 {
    var sum: f64 = 0;
    for (0..k) |i| {
        const byte = codes[n * (k / 2) + i / 2];
        const code = if (i % 2 == 0) byte % 16 else byte / 16;
        sum += fp4(code) * fp8(scales[n * (k / 16) + i / 16]) * fromBf16(x[i]);
    }
    return sum * global;
}
```
- **Teaching value**: This is the landing point of the whole course. Unpack a nibble, look up its magnitude, multiply by two scales and an input, sum. Everything above it in the stack is bookkeeping.
