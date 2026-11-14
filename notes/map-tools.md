# TensorFold tool-calling code map (schemas in, `tool_calls` out)

Repo root: `TensorFold/`. All paths relative to that root.
Every anchor below was read at the line numbers given. Reference/backend-agnostic files preferred; the shared
markup parser (`tool_parse.zig`) is the canonical implementation, with the family-specific streamers beside it.

**The pipeline in one line:**
`openai.zig:plan` (`tools` + `tool_choice` → `tool_specs.active`) → `prompt.zig:renderRequest` → `hf_text.zig:renderFn`
(the **checkpoint's own Jinja template** renders the schemas) → `server.zig:checkFeatures` (optional grammar, or HTTP 400)
→ decode → `tool_stream.Streamer.feed` per round (SSE argument deltas) **and** `tool_parse.parse` at the end
(whole-reply `tool_calls`) → `openai_calls.attach` (ids, `finish_reason`, `kept_tool_calls`).

**The one-sentence answer to "what validates anything":** *nothing on the request side beyond "is it a list of objects
with a name", nothing on the response side beyond "did the markup parse and is the name one the request offered".*
Argument **schemas are only enforced when a grammar is compiled** (`tool_choice: "required"`, a named function, a
`"strict": true` tool, or `TENSORFOLD_TOOL_GRAMMAR=1`) — and that path needs `libtfgrammar`, else HTTP 400.

Live server observed at `http://127.0.0.1:8080/v1/models` during this mapping: `"id": "local-model"`,
`"capabilities": {"structures": true, "logprobs": false, ...}`, `"context_length": 8192`. So forced tool calls are
enforceable on this box (`structures: true`), but the `logprobs.tool_calls` groups cannot be demonstrated here
(`logprobs: false`).

---

## 1. The request's `tools` and `tool_choice` become one list of specs

- **Anchor**: `zig/src/server/tool_specs.zig:72-97`
- **What it does**: `active` is the *only* validation the request's tool list gets: it must be a JSON array, each
  element an object, each element must yield a non-empty function name (`toolName` reads `function.name`, falling
  back to a bare `name`). `tool_choice: "none"` returns an empty list (the template never sees the tools at all);
  a named function filters the list down to that one tool and 400s on an unknown name; `"required"` with no tools
  is a 400. The returned slice is what the template, the parser and the grammar all later use.
- **Excerpt**:
```zig
pub fn active(cx: *Cx, tools: ?Value, choice: ?Value) errors.Refused![]Value {
    var specs: []Value = &.{};
    if (tools) |t| if (t != .null) {
        if (t != .array) return cx.other("tools must be a list");
        for (t.array, 0..) |tool, i| {
            if (tool != .object) return cx.fail(.other, "tools[{d}] must be an object", .{i});
            if ((try toolName(cx.a, tool)).len == 0) return cx.fail(.other, "tools[{d}] must include a function name", .{i});
        }
        specs = t.array;
    };
    if (choiceDisables(choice)) return &.{};
    if (specs.len == 0) {
        if (try choiceRequiresCall(cx.a, choice)) return cx.other("tool_choice requires a tool call, but the request offers no tools");
        return &.{};
    }
    const c = choice orelse return specs;
    if (!try isFunctionChoice(cx.a, c)) return specs;
    const function = c.get("function") orelse return cx.other("tool_choice function must include a function object");
    if (function != .object) return cx.other("tool_choice function must include a function object");
    const requested = try nameText(cx.a, function.get("name"));
    if (requested.len == 0) return cx.other("tool_choice function must include a name");
    var named: std.ArrayList(Value) = .empty;
    for (specs) |tool| if (std.mem.eql(u8, try toolName(cx.a, tool), requested)) try named.append(cx.a, tool);
    if (named.items.len == 0) return cx.fail(.other, "tool_choice requested unknown tool '{s}'", .{requested});
    return named.items;
}
```
- **Teaching value**: `tool_choice` is implemented as a **filter over the tool list before the prompt is built**,
  not as a decoding mode. `"none"` is "don't tell the model about the tools"; a named function is "offer exactly one".
- **Gotchas**: the **JSON Schema in `parameters` is never inspected here** — not for validity, not for size, not for
  depth. A tool with `parameters: "banana"` passes this gate. There is **no cap on the number of tools** anywhere in
  `zig/src` (searched). `tool_choice` accepts `{"type": ...}` *or* `{"mode": ...}` (`tool_specs.zig:41-62`), and all
  comparisons are case-insensitive and whitespace-trimmed.

Where it is called, and the two flags it sets for later stages:

- **Anchor**: `zig/src/server/openai.zig:139-140, 159-160, 170-174`
```zig
        input.messages = try messages.normalizeWith(cx, body.get("messages"), "system", srv.needs_user_after_tool, srv.media());
        input.tools = if (srv.text.native_deepseek) try @import("deepseek_request.zig").active(cx, body.get("tools"), body.get("tool_choice")) else try tool_specs.active(cx, body.get("tools"), body.get("tool_choice"));
```
```zig
    if (input.tools.len > 0 and try tool_specs.choiceRequiresCall(cx.a, body.get("tool_choice"))) try sampling.put(cx.a, "tool_call_required", .{ .bool = true });
    if (body.field("parallel_tool_calls")) |p| if (p == .bool and !p.bool) try sampling.put(cx.a, "tool_call_single", .{ .bool = true });
```
```zig
    var policy: tool_stream.Policy = .{};
    if (body.field("parallel_tool_calls")) |p| {
        if (p != .bool) return cx.refuse("parallel_tool_calls must be a boolean");
        policy.single = !p.bool;
    }
```
`tool_call_required` / `tool_call_single` are carried as **named keys in the opaque sampling object**, so later layers
can ask "did the client name this?" (same trick as every other sampling knob, see `map-serving.md` §3).

---

## 2. Rendering schemas into the prompt is the **template's** job, not the server's

- **Anchor**: `zig/src/server/prompt.zig:36-50`
- **What it does**: `renderRequest` normalises the messages, rewrites historical `tool_calls[].function.arguments`
  into objects (§3), and hands the tool list to the text model as **one render option**: `options.tools`. The server
  does not format schemas, pick a wire form, or add instructions; it binds a variable. `tool_choice` and
  `response_format` are *also* passed through as template options, for templates that read them.
- **Excerpt**:
```zig
pub fn renderRequest(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool, require_off: bool) errors.Refused![]const u32 {
    const messages_in = if (srv.text.vtable.format == .kimi) messages else try messages_mod.normalizeWith(cx, messages, srv.late_system, srv.needs_user_after_tool, srv.media());
    const normalized = if (srv.text.vtable.format == .kimi or srv.text.native_deepseek) messages_in else try messages_mod.toolArguments(cx, messages_in);
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else cx.named_effort, // off: only an effort the request named
        .clear_thinking = cx.clear_thinking,
        .tool_choice = cx.template_fields.get("tool_choice"),
        .response_format = cx.template_fields.get("response_format"),
        .images = srv.info.vision != null,
    };
```
- **Teaching value**: "how does the model know about my tools?" has a boring answer: the checkpoint ships a Jinja
  template, the server sets `tools` to the request's array, and the template decides everything the model reads.
- **Gotchas**: a checkpoint can ship a **separate tool template**; the render picks it *only when tools are present*:
  `hf_text.zig:423` — `const compiled = if (options.tools != null and t.tool_compiled != null) &t.tool_compiled.? else ...`
  (that is HuggingFace's `chat_template.tool_use` variant, see the fixture at `hf_text.zig:500-510`).
  Two template families are **not** Jinja and build the tool block in Zig instead: DeepSeek-V4.x
  (`hf_text.zig:393-399` → `zig/src/families/deepseek_v41/serve/template.zig:239-243`) and Kimi
  (`zig/src/server/kimi_prompt.zig:77-83`).

The binding itself:

- **Anchor**: `zig/src/server/hf_text.zig:421-447`
```zig
    fn renderFn(ctx: *anyopaque, a: Allocator, messages: Value, options: model_text.RenderOptions, problem: *[]const u8) model_text.Error!model_text.Rendered {
        const t = self(ctx);
        const compiled = if (options.tools != null and t.tool_compiled != null) &t.tool_compiled.? else if (t.compiled) |*c| c else {
            if (t.native_deepseek) return t.renderDeepseek(a, messages, options, problem);
            problem.* = "this checkpoint has no chat template: send a completion prompt instead";
            return error.Template;
        };
        var globals: std.json.ObjectMap = .empty;
        if (t.specials == .object) {
            var it = t.specials.object.iterator();
            while (it.next()) |e| try globals.put(a, e.key_ptr.*, e.value_ptr.*);
        }
        try globals.put(a, "thinking_mode", .{ .string = if (options.enable_thinking) "thinking" else "chat" });
        var context: std.json.ObjectMap = .empty;
        try context.put(a, "enable_thinking", .{ .bool = options.enable_thinking });
        if (options.reasoning_effort) |e| try context.put(a, "reasoning_effort", .{ .string = e });
        if (t.clear_thinking) |clear| try context.put(a, "clear_thinking", .{ .bool = options.clear_thinking orelse clear });
        var diag: template.Diag = .{};
        const tools: std.json.Value = if (options.tools) |tl| try toStd(a, tl) else .null;
        const out = template.renderProvenanceGlobals(a, compiled, try toStd(a, messages), tools, .{ .object = context }, .{ .object = globals }, options.add_generation_prompt, &diag) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            problem.* = diag.msg;
            return error.Template;
        };
        return .{ .text = out.s, .spans = out.spans };
    }
```

---

## 3. What the model literally sees: the raw JSON schema, plus a format lecture

- **Anchor**: `zig/src/server/fixtures/thinking/qwen38.jinja:57-70` (a real Qwen-family template shipped as a test fixture)
- **What it does**: when `tools` is a non-empty iterable the template opens an **extra system turn**, writes
  `# Tools`, dumps **each tool object verbatim as JSON** (`tool | tojson` — the whole OpenAI envelope, `type`,
  `function`, `description`, `parameters` schema and all, one per line between `<tools>` and `</tools>`), and then
  spells out the exact call markup the model must emit, with a reminder block. The user's own system prompt is
  appended *after* the tool block.
- **Excerpt** (Jinja, not Zig — this is the template file):
```jinja
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
```
- **Teaching value**: the model sees **your JSON Schema as text**, inside a system message, next to prose telling it
  which tags to type. Tool calling is prompt engineering that the checkpoint author shipped — the "protocol" is a
  string literal in a template.
- **Gotchas**: the schema is **not** compacted, deduplicated or trimmed — `tojson` writes it as sent, so a fat
  `description` or `$defs` block costs real prompt tokens on **every turn**. Each family writes a different form:
  GLM-5.3 lists `<tools>` with per-tool JSON then calls with `<arg_key>/<arg_value>`
  (`zig/src/server/fixtures/thinking/glm53.jinja:32-45, 162-163`); Kimi's server-side builder writes
  `"# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n"` + sorted compact JSON
  (`kimi_prompt.zig:77-83`); DeepSeek writes a `## Tools` block describing a `<｜DSML｜ calls>` envelope
  (`zig/src/families/deepseek_v41/serve/template.zig:239-243`).

Round-tripping *previous* calls back into the prompt (so a multi-turn agent loop renders):

- **Anchor**: `zig/src/server/messages.zig:204-243`
- **What it does**: `toolArguments` turns each historical `tool_calls[].function.arguments` **string** into a real
  JSON object before the template sees it, because templates do `arguments.items()`. Whitespace-only arguments
  become `{}`; anything that does not parse as an object is wrapped as `{"_invalid_arguments": <original>}` so the
  render still succeeds instead of raising.
- **Excerpt** (lines 219-234):
```zig
            const args = function.get("arguments").?;
            if (args == .object) continue;
            var parsed: ?Value = args;
            if (args == .string and std.mem.trim(u8, args.string, " \t\r\n").len == 0) {
                parsed = .{ .object = try json.newObject(cx.a) }; // a call with no arguments, as clients spell it
            } else if (args == .string) {
                parsed = switch (try json.parseText(cx.a, args.string)) {
                    .ok => |v| v,
                    .err => null,
                };
            }
            if (parsed == null or parsed.? != .object) {
                const wrapped = try json.newObject(cx.a);
                try wrapped.put(cx.a, "_invalid_arguments", args);
                parsed = .{ .object = wrapped };
            }
```
- **Teaching value**: OpenAI's wire format sends arguments as a *string*; templates want a *dict*. Somebody has to
  parse that, and a malformed string from the client is **not** an error — it is rendered under a marker key.
- **Gotchas**: `_invalid_arguments` is a real key the model will see in the prompt. Kimi and native DeepSeek skip
  this normalisation entirely (`prompt.zig:38`).

---

## 4. The markup the parser accepts (all families, one scanner)

- **Anchor**: `zig/src/server/tool_parse.zig:55-107`
- **What it does**: `envelopes` finds every call block in the finished text, in four passes, then sorts and drops
  nested ones. The accepted openers are: `<tool_call>…</tool_call>` (any case), namespaced
  `<NAME:tool_call>…</NAME:tool_call>` (any case), Gemma's `<|tool_call>…<tool_call|>`, and DeepSeek's DSML
  `<｜DSML｜…>` blocks (whose spellings live in `tool_dsml.zig:34`). A DSML block that opens but never closes is
  kept with `complete = false`.
- **Excerpt** (lines 55-81, the first two of the four passes):
```zig
pub fn envelopes(a: Allocator, text: []const u8) Allocator.Error![]Envelope {
    var found: std.ArrayList(Envelope) = .empty;
    var pos: usize = 0;
    var dsml = tool_dsml.findOpen(text, 0); // the DSML opener at or after `pos`, searched again only once `pos` passes it
    while (findCI(text, "<tool_call>", pos)) |s| {
        if (dsml != null and dsml.?.at < pos) dsml = tool_dsml.findOpen(text, pos);
        if (dsml) |outer| if (outer.at < s) {
            pos = tool_dsml.blockEnd(text, outer) orelse break;
            continue;
        };
        const e = findCI(text, "</tool_call>", s + 11) orelse break;
        try found.append(a, .{ .start = s, .end = e + 12, .payload = strip(text[s + 11 .. e]) });
        pos = e + 12;
    }
    pos = 0;
    var closers: tool_closers.Closers = .{ .text = text };
    while (std.mem.indexOfScalarPos(u8, text, pos, '<')) |s| {
        pos = s + 1;
        var j = s + 1;
        if (j >= text.len or !(std.ascii.isAlphabetic(text[j]) or text[j] == '_')) continue;
        j += 1;
        while (j < text.len and tool_closers.nameByte(text[j])) j += 1;
        if (!std.ascii.startsWithIgnoreCase(text[j..], ":tool_call>")) continue;
        const e = (try closers.next(a, text[s + 1 .. j], j + 11)) orelse continue;
        const end = e + j - s + 12; // ``</NAME:tool_call>``
        try found.append(a, .{ .start = s, .end = end, .payload = strip(text[j + 11 .. e]) });
        pos = end;
    }
```
- **Teaching value**: the "protocol" is recovered by **string search over the decoded text**, after the fact — not
  by a token-level state machine. Everything outside an envelope is prose.
- **Gotchas**: a call written *inside* an earlier envelope's arguments is part of that envelope, never its own call
  (the DSML skip-ahead above, and the dedupe at `:103-105`:
  `for (found.items) |env| if (kept.items.len == 0 or env.start >= kept.items[kept.items.len - 1].end) try kept.append(a, env);`).
  Openers are matched **case-insensitively**, so `<TOOL_CALL>` works (see `tool_single_stream_test.zig:201-210`).

What one envelope's payload may contain:

- **Anchor**: `zig/src/server/tool_parse.zig:293-339`
- **What it does**: `payload` is the per-block reader. It accepts, in order: Gemma's `call:NAME{…}` /`:NAME{…}`
  (`gemmaCall`, `tool_parse.zig:502`); a JSON object (name from `name`/`tool`/`function`/`call`, arguments from
  `arguments`/`args`/`parameters`, **or every other key collected as the arguments**); a JSON array (first item that
  reads as a call); Qwen's XML `<function=NAME><parameter=K>V</parameter></function>` (`functionBlock`,
  `tool_parse.zig:357`); GLM's `NAME<arg_key>K</arg_key><arg_value>V</arg_value>…` (`glmCall`, `tool_parse.zig:453`).
- **Excerpt** (lines 308-333, the JSON-object branch):
```zig
        if (v == .object) {
            const function = v.get("function");
            const holder = if (function != null and function.? == .object) function.? else v;
            const name_keys: []const []const u8 = if (holder.object == v.object) &.{ "name", "tool", "function", "call" } else &.{ "name", "tool", "function" };
            const name = if (holder.firstTruthy(name_keys)) |n| strip(try tool_specs.pyStr(a, n)) else "";
            var arguments: ?Value = null;
            for ([_][]const u8{ "arguments", "args", "parameters" }) |k| if (holder.has(k)) {
                arguments = holder.get(k).?;
                break;
            };
            const args = arguments orelse blk: {
                const loose = try json.newObject(a);
                for (holder.object.keys(), holder.object.values()) |k, item| {
                    const skip = for ([_][]const u8{ "name", "tool", "function", "call", "type" }) |x| {
                        if (std.mem.eql(u8, x, k)) break true;
                    } else false;
                    if (!skip) try loose.put(a, k, item);
                }
                break :blk Value{ .object = loose };
            };
            if (name.len == 0) return error.Invalid;
            const object = try jsonObject(a, args);
            if (complete and !tool_params.finite(object)) return error.Invalid;
            return .{ .name = name, .arguments = object };
        }
```
- **Teaching value**: the parser is **deliberately forgiving about shape** and **strict about one thing only**: there
  must be a name. Everything a model commonly gets wrong (`args` instead of `arguments`, flat keys instead of a
  nested object, a list wrapper, a ```json fence) is absorbed.
- **Gotchas**: the only hard rejections here are "no name" and (on a complete block) a `NaN`/`Infinity` anywhere in
  the arguments (`tool_params.finite`, `tool_params.zig:81-92`). **No schema is consulted to accept or reject.**
  A reply that is *only* bare JSON naming an offered tool is also a call, with no envelope at all
  (`bareCalls`, `tool_parse.zig:250-277`), and that path strips a ```json fence first (`stripFence`, `:279`).

---

## 5. `parse`: text → the `tool_calls` array, with rejects kept as content

- **Anchor**: `zig/src/server/tool_parse.zig:109-182`
- **What it does**: the whole-reply parser. It builds a lowercase set of **offered** names, tries the bare-JSON path,
  then walks the envelopes: prose between envelopes accumulates into `residue`; a block that parses *and* names an
  offered tool becomes an OpenAI call; a block that fails to parse (`malformed`) or names a tool the request did not
  offer (`unoffered`) is **copied back into the residue verbatim** and counted. The result carries `content` (the
  stripped residue), `calls` (null when none), both reject counts and up to eight unoffered names.
- **Excerpt** (lines 127-170, the decision):
```zig
    for (envs) |env| {
        try residue.appendSlice(a, text[cursor..env.start]);
        cursor = env.end;
        if (!env.complete) {
            try residue.appendSlice(a, text[env.start..env.end]);
            continue;
        }
        if (max_calls) |m| if (calls.items.len >= m) continue;
        const parsed: ?[]Call = blk: {
            if (tool_dsml.whole(env.payload)) |b| {
                break :blk tool_dsml.calls(a, b.inner, b.spelling) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Invalid => null,
                };
            }
            const one = payload(a, env.payload, &schemas, max_calls != null) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Invalid => break :blk null,
            };
            if (one) |c| {
                const list = try a.alloc(Call, 1);
                list[0] = c;
                break :blk list;
            }
            break :blk null;
        };
        const reject: enum { none, unoffered, malformed } = blk: {
            const list = parsed orelse break :blk .malformed;
            if (list.len == 0) break :blk .malformed;
            var unoffered = false;
            for (list) |c| if ((try callName(a, c, &known)) == null) {
                unoffered = true;
                try keepName(a, &kept_names, strip(c.name));
            };
            break :blk if (unoffered) .unoffered else .none;
        };
        if (reject != .none) {
            try residue.appendSlice(a, text[env.start..env.end]);
            rejected += 1;
            if (reject == .unoffered) kept_unoffered += 1 else kept_malformed += 1;
            continue;
        }
```
- **Teaching value**: **a bad tool call is never an error and never silently dropped — it is returned as text.**
  The client sees the exact markup the model wrote, which is what lets an agent loop say "that tool does not exist"
  and lets the model self-correct.
- **Gotchas**: `parse` returns early with `calls = null` when `tools.len == 0` (`:110`) — **a server with no `tools`
  in the request never parses call markup at all**, so `<tool_call>` in a toolless reply is just text. Name matching
  is **case-insensitive** and the *offered* spelling is returned, not the model's (`callName`, `:233-236`). An
  unoffered name is judged per-envelope: one bad call in a DSML block rejects the whole block.
  This is exactly what `zig build test-tool-parse-host` covers.

Reject bookkeeping and its wire shape:

- **Anchor**: `zig/src/server/tool_parse.zig:185-192`
```zig
fn keepName(a: Allocator, names: *std.ArrayList([]const u8), raw: []const u8) Allocator.Error!void {
    var end = @min(raw.len, 64);
    while (end > 0 and end < raw.len and (raw[end] & 0xc0) == 0x80) end -= 1;
    const name = raw[0..end];
    if (name.len == 0 or names.items.len >= 8) return;
    for (names.items) |kept| if (std.mem.eql(u8, kept, name)) return;
    try names.append(a, name);
}
```
Concrete limits: **8 names, 64 UTF-8 bytes each, deduplicated, truncated on a codepoint boundary**.

---

## 6. `attach`: the reply's `tool_calls`, ids, `finish_reason` and `kept_tool_calls`

- **Anchor**: `zig/src/server/openai_calls.zig:15-45`
- **What it does**: the single place a finished chat reply gains calls. Native-format families (Kimi, DeepSeek) hand
  over their own `native_calls`; everyone else goes through `tool_parse.parse` with the policy's call limit. Rejected
  envelopes are logged, counted into `/metrics`, and reported in the response's `tensorfold.kept_tool_calls` block.
  A reply cut by `max_tokens` mid-call is recovered through `tool_stream.cutCalls` so the whole reply matches what
  the stream already sent. On success `content` becomes the filtered prose and `finish_reason` becomes `"tool_calls"`.
- **Excerpt** (lines 26-45):
```zig
    const parsed = try tool_parse.parse(r.a, reply.content, tools, r.plan.policy.maxCalls());
    if (parsed.rejected_envelopes > 0) {
        log.line("warning: kept {d} malformed or unoffered tool envelopes as content", .{parsed.rejected_envelopes});
        r.srv.metrics.keptToolCalls(r.srv.io, parsed.kept_unoffered, parsed.kept_malformed);
        if (reply.runtime == .object) try reply.runtime.object.put(r.a, "kept_tool_calls", try keptBlock(r.a, parsed));
    }
    // max_tokens cut a call the stream had begun: the whole reply returns it cut too, as the stream did (#1238)
    const cut = if (!r.plan.policy.single and std.mem.eql(u8, reply.finish_reason, "length")) try tool_stream.cutCalls(r.a, reply.content, tools) else null;
    if (cut != null) reply.content = try @import("reply_text.zig").cutProse(r.a, reply.content);
    const found = cut orelse parsed.calls;
    const calls = (if (r.turn != null) cortheon.clientCalls(found) else found) orelse {
        if (r.plan.policy.single or parsed.calls != null) reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
        return null;
    };
    if (cut != null) return try numbered(r, calls);
    reply.content = try r.plan.policy.parsedContent(r.a, parsed.content);
    reply.finish_reason = "tool_calls";
    if (r.plan.policy.single) reply.tool_calls_streamed = false;
    return try numbered(r, if (r.plan.policy.single and calls.len > 1) calls[0..1] else calls);
}
```
- **Teaching value**: `finish_reason: "tool_calls"` is **assigned by the server after parsing**, not emitted by the
  model. A reply with prose *and* a call keeps both: content and `tool_calls` in the same message.
- **Gotchas**: a `length`-cut call still comes back as a call, with `finish_reason` left at `"length"` — clients that
  only run calls on `"tool_calls"` will miss it. Call ids are **not random per reply**: `numbered`
  (`openai_calls.zig:48-51`) replaces each id with an HMAC-keyed id derived from the request digest + choice + index
  (`ids.Calls`, documented at `docs/api.md:590-606`), so a replayed request to the same process gets identical ids;
  `TENSORFOLD_TOOL_ID_KEY_FILE` makes them stable across restarts. The default inside `tool_parse.openaiCall` is
  `call_` + 24 hex (`tool_parse.zig:239-248`).

The counters and the public block:

- **Anchor**: `zig/src/server/metrics.zig:143-144` (and the same pair again at `:206-207` for the second renderer)
```zig
        try gauge(w, "kept_tool_calls_unoffered_total", "counter", "Tool-call envelopes kept as text because they named a tool the request did not offer.", snap.kept_unoffered);
        try gauge(w, "kept_tool_calls_malformed_total", "counter", "Tool-call envelopes kept as text because their markup did not parse.", snap.kept_malformed);
```
The block itself is built at `zig/src/server/openai_calls.zig:64-75` (`unoffered`, `malformed`, `unoffered_names`).

`zig build test-tool-visibility-http-host` ("Rejected tool envelopes survive actual chat JSON and SSE replies",
`zig/build/server.zig:112-114`) is the end-to-end guarantee: `late_system_retained_test.zig:235-311` asserts that a
kept unoffered call and a kept malformed call appear as `tensorfold.kept_tool_calls`
`{unoffered, malformed, unoffered_names}` **identically on the whole JSON reply and on the stream's final chunk**,
that a clean reply has **no** such field, that the name list caps at 8 × 64 bytes, and that both `/metrics` counters
move; `late_system_test.zig:546-552` asserts the unoffered call's markup itself survives in `content` both whole and
streamed.

---

## 7. Streaming: arguments as deltas, assembled by a text state machine

- **Anchor**: `zig/src/server/tool_stream.zig:250-272` (`.function` state) and `:273-301` (`.value` state)
- **What it does**: `Streamer.feed` is fed the **whole answer so far** on every round and emits only the new deltas.
  In `.function` it waits for `<parameter=KEY>`, looks that key's schema up, and emits the JSON prologue
  (`{"key":` or `{"key":"`); in `.value` it streams the parameter's bytes as escaped JSON-string content as they
  arrive, holding back a possible partial `</parameter>` and the framing newline. A *typed* parameter (schema takes
  a non-string) is buffered instead and emitted once, decoded, at the closing tag.
- **Excerpt** (lines 250-272):
```zig
                .function => {
                    if (opening(rest, "<parameter=")) |m| {
                        const key = reply_text.pyStrip(m.name);
                        s.value_schema = try tool_params.schemaOf(&s.schemas, s.current, key, s.a);
                        s.typed = try typedSchema(s.a, s.value_schema);
                        s.buffer.clearRetainingCapacity();
                        const quoted = try json.quote(s.a, key, .{ .ascii = false });
                        try out.append(s.a, try s.args(try std.mem.concat(s.a, u8, &.{ if (s.args_open) "," else "{", quoted, if (s.typed) ":" else ":\"" })));
                        s.args_open = true;
                        s.pos += m.len;
                        s.state = .value;
                        s.lead = !m.newline;
                        s.held = "";
                        continue;
                    }
                    if (closing(rest, "</function>")) |n| {
                        try out.append(s.a, try s.args(if (s.args_open) "}" else "{}"));
                        s.pos += n;
                        s.state = .closing;
                        continue;
                    }
                    return;
                },
```
- **Teaching value**: OpenAI's `tool_calls[].function.arguments` deltas are **synthesised**: the model is writing XML
  (or GLM tags, or DSML), and the server translates that, token by token, into fragments of a JSON string that
  concatenate to valid JSON. The arguments JSON the client reassembles never existed in the model's output.
- **Gotchas**: a stream **cannot be rewound**, so the streamer has to commit to "this is a call" before it knows the
  call is valid. Its escape hatches are `giveUp` (`:319-326`) → `.buffered`, which waits for the closer and then
  parses the whole envelope with `tool_parse.parse` and emits finished calls (`.buffered` state, `:237-249`), and
  `.abort` (when nothing was streamed yet, e.g. the reply starts with `{`, `[` or a fence — `:129-137`), after which
  only the final whole-reply parse produces calls. An **unoffered name seen while streaming forces the buffered
  path** (`:199-206`), which is how a rejected envelope can still end up as plain content. The incremental
  `.value` path also decodes typed values through the same `tool_params.decode` the whole-reply parser uses
  (`:295-298`), so streamed and whole arguments agree byte for byte.

The fallback that emits a call that was *not* followed byte-by-byte:

- **Anchor**: `zig/src/server/tool_streaming.zig:9-37`
```zig
pub fn deltas(a: Allocator, calls: []const Value) Allocator.Error![]Value {
    var out: std.ArrayList(Value) = .empty;
    for (calls, 0..) |call, index| {
        const function = call.get("function") orelse continue;
        if (function != .object) continue;
        const head_fn = try json.newObject(a);
        try head_fn.put(a, "name", .{ .string = if (function.get("name")) |n| (if (n.truthy()) try tool_specs.pyStr(a, n) else "") else "" });
        try head_fn.put(a, "arguments", .{ .string = "" });
        const head = try json.newObject(a);
        try head.put(a, "index", try json.intValue(a, index));
        const id = call.get("id");
        try head.put(a, "id", .{ .string = if (id != null and id.?.truthy()) try tool_specs.pyStr(a, id.?) else try std.fmt.allocPrint(a, "call_{d}", .{index}) });
        const kind = call.get("type");
        try head.put(a, "type", .{ .string = if (kind != null and kind.?.truthy()) try tool_specs.pyStr(a, kind.?) else "function" });
        try head.put(a, "function", .{ .object = head_fn });
        try out.append(a, try wrapCalls(a, .{ .object = head }));
        const args = function.get("arguments");
        const text = if (args != null and args.?.truthy()) try tool_specs.pyStr(a, args.?) else "";
        if (text.len > 0) {
            const arg_fn = try json.newObject(a);
            try arg_fn.put(a, "arguments", .{ .string = text });
            const more = try json.newObject(a);
            try more.put(a, "index", try json.intValue(a, index));
            try more.put(a, "function", .{ .object = arg_fn });
            try out.append(a, try wrapCalls(a, .{ .object = more }));
        }
    }
    return out.items;
}
```
Two frames per call: a head (`index`, `id`, `type`, `name`, empty `arguments`) then **the whole arguments string in
one delta**. So "streamed arguments" is per-family: Qwen XML / GLM / DSML stream incrementally, a JSON-envelope call
arrives whole at its closer.

Where the deltas join the SSE stream (per decode round, from the answer only — never the reasoning):

- **Anchor**: `zig/src/server/chat_generation.zig:155-159`
```zig
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
```
The streamer only exists for streamed, non-Kimi requests with tools: `chat.zig:276` —
`if (sink != null and input.tools.len > 0 and srv.text.vtable.format != .kimi) gen.calls = tool_stream.Streamer.init(a, input.tools)`.
A call the reply's end cut off is finished with `finishCut` (`chat_generation.zig:298-303`,
`tool_stream.zig:360-374`). Ids on streamed heads are rewritten to the same keyed ids the whole reply uses, at
`openai.zig:454` → `openai_calls.streamed` (`openai_calls.zig:54-62`).

---

## 8. Forced calls: `tool_choice` → a compiled grammar, or HTTP 400

- **Anchor**: `zig/src/server/server.zig:345-357`
- **What it does**: `checkFeatures` is the gate. `tool_call_required` (set in §1 from `"required"` or a named
  function) **requires** the backend's `structures` capability; without it the request is refused by name before any
  token. With it, a structural-tag grammar for the template's own call markup is compiled and attached to the
  request. Separately, an `auto` request still gets a grammar when any offered tool is `"strict": true` (or
  `TENSORFOLD_TOOL_GRAMMAR=1`) **and** a grammar compiler is already available — otherwise it silently stays free.
- **Excerpt** (lines 345-357):
```zig
    pub fn checkFeatures(srv: *Server, cx: *Cx, f: Value, has_tools: bool, thinking: bool, prompt_ids: []const u32, tools: []const Value, request: *api.Request) errors.Refused!void {
        _ = prompt_ids;
        errdefer srv.release(false, request.*); // a refused request frees what it compiled here
        const required = if (f.get("tool_call_required")) |r| r.truthy() else false;
        const one_call = if (f.get("tool_call_single")) |x| x.truthy() else false;
        const all = has_tools and !required and (tool_grammar.autoAllEnv() catch return cx.fail(.server, "invalid TF_DSV41_TOOL_GRAMMAR: expected 0 or 1", .{}));
        if (has_tools and required) {
            if (!srv.info.capability(.structures)) return cx.fail(.request, "backend '{s}' cannot enforce tool_choice \"required\" or a named function: it has no token grammar support; send \"auto\" or \"none\"", .{srv.backendName()});
            request.structure = try tool_grammar.structure(srv, cx, tools, !one_call, thinking);
        } else if (has_tools and srv.info.capability(.structures) and tool_grammar.autoWanted(tools, all) and srv.grammarCompiler() != null) {
            // auto with a strict tool (any, TENSORFOLD_TOOL_GRAMMAR=1): text or calls; no grammar library, no grammar
            request.structure = try tool_grammar.optionalStructure(srv, cx, tools, all, !one_call, thinking);
        }
```
- **Teaching value**: "required tool call" is not a sampler flag — it is a **grammar over the template's own call
  markup**, compiled per request, that makes any other continuation unsamplable.
- **Gotchas**: there are **two different refusals**. (a) Backend without the capability →
  `backend '<name>' cannot enforce tool_choice "required" or a named function: it has no token grammar support; send "auto" or "none"`.
  (b) Capability present but the grammar library missing → the message from `compiledStructure` below. The asymmetry
  is deliberate: a missing library is fixable by installing a file; a missing backend hook is not
  (`docs/api.md:666-669`). Note the `auto` + strict path requires `srv.grammarCompiler() != null` **before** it
  attaches anything, so a strict tool without the library degrades to unconstrained instead of 400. A
  `response_format`/`guided_*` grammar sent *with* tools wins and frees the tool grammar (`server.zig:358-363`).

The exact 400 when `libtfgrammar` is absent:

- **Anchor**: `zig/src/server/server.zig:374-379`, with the reason built at `zig/src/server/tool_grammar.zig:44-53`
```zig
    pub fn compiledStructure(srv: *Server, cx: *Cx, kind: grammar.Kind, text: []const u8, field: []const u8, thinking: bool) errors.Refused!api.Structure {
        const c = srv.grammarCompiler() orelse return cx.fail(.request, "{s}: {s}", .{ if (kind == .structural_tag) field else "structured output", srv.grammar_problem.? });
        var err: [512]u8 = undefined;
        const compiled = c.compile(srv.io, @fromBackingInt(@intCast(@backingInt(kind))), text, &err) catch
            return cx.fail(.request, "{s}: the grammar cannot be enforced: {s}", .{ field, std.mem.sliceTo(&err, 0) });
        return .{ .kind = kind, .field = field, .text = text, .after = if (thinking) srv.text.tokenId(srv.markers.close) else null, .compiler = c, .compiled = compiled };
    }
```
```zig
pub fn structure(srv: *Server, cx: *Cx, tools: []const Value, parallel: bool, thinking: bool) errors.Refused!api.Structure {
    const rendered = try probed(srv, cx);
    const s = style(rendered) orelse .json; // unrecognized markup uses the canonical call form the parser accepts
    const text = if (lfm2.isStyle(rendered, probe, key)) try lfm2.build(cx.a, tools, parallel) else try build(cx.a, s, tools, parallel);
    const reason = try std.fmt.allocPrint(cx.a, "tool_choice \"required\" or a named function on backend '{s}'", .{srv.backendName()});
    var st = try srv.compiledStructure(cx, .structural_tag, text, reason, thinking);
    // The structure names the request's field; the sentence above words the 400.
    st.field = "tool_choice";
    return st;
}
```
For `kind == .structural_tag` the `field` argument *is* that reason sentence, so the body reads
`tool_choice "required" or a named function on backend '<backend>': <problem>`. The library problem string is
composed at `zig/src/platform/optional_libraries.zig:23-24, 42` as
`names[0] ++ " is required for " ++ feature ++ "; " ++ install`, i.e. on macOS:
`libtfgrammar.dylib is required for structured output and required tool calls (HTTP 400 when missing); restore the matching release archive's lib/ directory or run zig build grammar -Dxgrammar=DIR; prepare DIR with git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar`
(`.so` on Linux, `tfgrammar.dll` on Windows; `clone` is `zig/src/platform/grammar_library.zig:3`). Other
`grammar_problem` values from `server.zig:403-420` include `"the checkpoint has no tokenizer.json"`,
`"config.json is not JSON"` and `"structured output needs the checkpoint's config.json vocab_size"`.
A grammar xgrammar refuses to compile gives `<reason>: the grammar cannot be enforced: <xgrammar's text>`.

How the grammar learns the family's markup (no hardcoded family list):

- **Anchor**: `zig/src/server/tool_grammar.zig:68-76` and `:173-189`
```zig
fn probed(srv: *Server, cx: *Cx) errors.Refused![]const u8 {
    const probe_turn = (try json.parse(cx.a, "[{\"role\": \"user\", \"content\": \"x\"}, {\"role\": \"assistant\", \"content\": \"\", \"tool_calls\": [{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"" ++ probe ++ "\", \"arguments\": {\"" ++ key ++ "\": \"v\"}}}]}]")).ok;
    var scratch: Cx = .{ .a = cx.a };
    return if (prompt.renderIds(srv, &scratch, probe_turn, &.{}, false, null, false)) |ids| try srv.text.decode(cx.a, ids) else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Refused => "",
    };
}
```
```zig
fn call(a: Allocator, s: Style, name: []const u8, schema: Value) Allocator.Error!Value {
    const ws = try node(a, "regex", &.{.{ "pattern", .{ .string = blanks } }});
    const elements = switch (s) {
        .deepseek => try list(a, &.{
            try constant(a, try std.fmt.allocPrint(a, "<｜DSML｜invoke name=\"{s}\">", .{name})),
            ws,
            try arguments(a, schema, "deepseek_v4_1_xml"),
            ws,
            try constant(a, "</｜DSML｜invoke>"),
        }),
        .xml => try list(a, &.{
            try constant(a, "<tool_call>"),                                        ws,
            try constant(a, try std.fmt.allocPrint(a, "<function={s}>", .{name})), try arguments(a, schema, "qwen_xml"),
            try constant(a, "</function>"),                                        ws,
            try constant(a, "</tool_call>"),
        }),
        .gemma => try list(a, &.{ try constant(a, try std.fmt.allocPrint(a, "<|tool_call>call:{s}", .{name})), try arguments(a, schema, "json"), try constant(a, "<tool_call|>") }),
```
The server **asks the template to render a fake call** to `tfprobe_fn` with `{tfprobe_k: "v"}`, matches the result
against six known shapes (`style`, `tool_grammar.zig:28-39`: `xml`, `json`, `gemma`, `glm`, `minimax`, `deepseek`),
and builds the grammar in that shape. An unrecognised template falls back to `.json`, i.e. the canonical
`<tool_call>{"name":…,"arguments":{…}}</tool_call>` that the parser always accepts. With thinking on, the constraint
starts after the think-close token (`compiledStructure`'s `.after`, `server.zig:379`).

---

## 9. Where the schema *is* enforced — and where it is only a type hint

- **Anchor**: `zig/src/server/tool_grammar.zig:84-97, 142-148, 226-232`
- **What it does**: `anyCall` decides, per tool, whether the grammar carries the tool's real parameter schema or just
  "any JSON object": the schema is used when the request is forced (`build(..., all = true)` from `structure`) or the
  tool is `"strict": true`; otherwise `anyObject()` with `additionalProperties: true`. `autoWanted` is the switch
  that gives an `auto` request a grammar at all.
- **Excerpt** (lines 84-96 and 142-146):
```zig
pub fn autoWanted(tools: []const Value, all: bool) bool {
    if (tools.len == 0) return false;
    if (all) return true;
    for (tools) |t| if (strict(t)) return true;
    return false;
}

/// OpenAI's function.strict (or a bare tool's strict) is true.
fn strict(tool: Value) bool {
    const function = if (tool.get("function")) |f| (if (f == .object) f else tool) else tool;
    const s = function.get("strict") orelse return false;
    return s == .bool and s.bool;
}
```
```zig
fn anyCall(a: Allocator, s: Style, tools: []const Value, all: bool) Allocator.Error!Value {
    const calls = try a.alloc(Value, tools.len);
    for (tools, calls) |t, *c| c.* = try call(a, s, try tool_specs.toolName(a, t), if (all or strict(t)) try parameters(a, t) else try anyObject(a));
    if (calls.len == 1) return calls[0];
    return node(a, "or", &.{.{ "elements", .{ .array = calls } }});
}
```
- **Teaching value**: there is exactly one mechanism that makes arguments match a schema — **constrained decoding**.
  Everything else is hope plus a forgiving parser.
- **Gotchas**: `"strict": true` on a tool changes behaviour **even under `tool_choice: "auto"`** (CHANGELOG #1688):
  `optional` (`tool_grammar.zig:122-140`) builds `triggered_tags` whose trigger is the call opener, so free prose is
  still allowed but a started call must validate, and `stop_after_first` is the inverse of `parallel`. Under
  `required`, **every** offered tool's schema is enforced (`all = true` in `build`, `:100`), whether or not it is
  strict. `parameters` falls back to `input_schema`, then to an empty `{"type":"object","properties":{}}`
  (`:216-224`). LFM2's Pythonic markup has no trigger form, so `optionalStructure` returns null for it (`:58`).

The non-grammar path, where the schema only *types* a value:

- **Anchor**: `zig/src/server/tool_params.zig:121-139` with `zig/src/server/tool_kinds.zig:224-235`
```zig
pub fn decode(a: Allocator, value: []const u8, schema: Schema, python: bool) Allocator.Error!Value {
    const kinds = (try tool_kinds.schemaKinds(a, schema.value, schema.root)) orelse return .{ .string = value };
    if (!tool_kinds.takesJson(kinds)) return .{ .string = value };
    const closed: ?[]const u8 = if (kinds.contains(.array) or kinds.contains(.object)) try closedJson(a, value) else null;
    for ([_]?[]const u8{ value, closed }) |candidate| {
        const text = candidate orelse continue;
        const parsed = switch (try json.parseText(a, text)) {
            .ok => |v| v,
            .err => continue,
        };
        if (!finite(parsed)) continue;
        return if (tool_kinds.accepts(kinds, parsed)) parsed else .{ .string = value };
    }
    if (!python) return .{ .string = value };
    const parsed = (try pythonLiteral(a, value)) orelse return .{ .string = value };
    if (!finite(parsed)) return .{ .string = value };
    return if (tool_kinds.accepts(kinds, parsed)) parsed else .{ .string = value };
}
```
- **Teaching value**: for XML/GLM markup the only thing the schema does is answer "should `3` be the number 3 or the
  string `"3"`?". A value the schema would reject is **kept as the text the model wrote** — never an error.
- **Gotchas**: `decode(..., python = true)` (the Qwen XML path, `tool_parse.zig:382`) also accepts **Python
  literals** — `True`, `None`, `'single quotes'`, tuples, dicts — via `pythonLiteral` (`tool_params.zig:141-157`,
  recursion capped at depth 200, `:166`). `tool_kinds.accepts` (`:231-235`) returns false for any string, which is
  why a quoted value stays the model's text. Schema reference resolution is memoised and capped at
  `max_ref_depth = 32` (`tool_kinds.zig:32, 108-127`); a `$ref` cycle resolves to "no kinds", i.e. "constrains
  nothing". A parameter no tool declares gets `emptySchema` and stays a string (`tool_params.zig:62-72`).

---

## 10. Limits, and the one-call policy

- **Anchor**: `zig/src/server/tool_stream.zig:510-546`
- **What it does**: `Policy` is the whole of `parallel_tool_calls: false`. `maxCalls()` returns `1`, which `parse`
  applies as "stop accepting once one call is kept"; `text`/`content`/`parsedContent` run a `TextFilter`
  (`:437-507`) that hides call markup from streamed prose while **keeping whole envelopes the parser rejected**
  visible.
- **Excerpt** (lines 509-525):
```zig
/// ``parallel_tool_calls: false`` keeps one call and hides call markup from streamed prose.
pub const Policy = struct {
    single: bool = false,
    filter: TextFilter = .{},

    pub fn maxCalls(p: *const Policy) ?usize {
        return if (p.single) 1 else null;
    }

    /// Filters a str delta's text (dict deltas lose tool_calls and filter content), or passes it through.
    pub fn text(p: *Policy, a: Allocator, s: []const u8) Allocator.Error![]const u8 {
        return if (p.single) p.filter.feed(a, s) else s;
    }

    pub fn flush(p: *Policy) []const u8 {
        return if (p.single) p.filter.finish() else "";
    }
```
- **Teaching value**: "one tool call" is enforced **twice** — by the grammar when the request is forced
  (`!one_call` → no `star` node, `tool_grammar.zig:108, 117`) and by the parser/policy when it is not. The model may
  still write five calls; four are dropped from the response and **still counted in usage**
  (`docs/api.md:584-586`).
- **Gotchas** — the concrete numbers, each with the file that defines them:
  - **1** accepted call under `parallel_tool_calls: false` (`tool_stream.zig:514-516`); a *rejected* envelope does
    **not** consume that budget (`tool_parse_visibility_test.zig:24-35`).
  - **8** kept unoffered names, **64** bytes each (`tool_parse.zig:185-192`).
  - **0–8** blank characters between grammar elements: `blanks = "[ \\t\\r\\n]{0,8}"` (`tool_grammar.zig:22`).
  - **32** blank characters max between JSON tokens inside constrained arguments: `value_blanks = 32`
    (`tool_grammar.zig:25`, emitted as `max_whitespace_cnt` at `:230-232`).
  - **32** `$ref` depth (`tool_kinds.zig:32`); **200** Python-literal nesting depth (`tool_params.zig:166`).
  - **256** characters of unrecognised text after `<tool_call>` before the streamer gives up and buffers the
    envelope instead of streaming it (`tool_stream.zig:229-235`).
  - **512** bytes of xgrammar error text in a compile refusal (`server.zig:376`).
  - **16,384** grammar parser states, a fixed work limit, and the last **8** grammars kept, for `grammar_budget`
    (`docs/api.md:680-690`; refused at `server.zig:385-394`).
  - **No limit** on tool count, schema size, argument size, nesting of arguments, or call count without
    `parallel_tool_calls: false` (searched `zig/src`; the only ceiling is the context window and `max_tokens`).

---

## 11. The honest failure modes

- **Anchor**: `zig/src/server/tool_parse_visibility_test.zig:1-35` (the executable specification)
- **What it does**: pins the three "model got it wrong" cases against the real parser. A call to an undeclared tool
  (XML *or* JSON form) and an unparseable payload both come back with `calls == null`,
  `rejected_envelopes == 1`, and `content` **byte-identical to the markup the model wrote** — with and without the
  single-call limit. A rejected envelope ahead of two good calls does not consume the one allowed call.
- **Excerpt** (line 6, then lines 10-22; the three omitted lines 7-9 are the usual
  `std.heap.ArenaAllocator.init` boilerplate):
```zig
test "tool visibility: a first unoffered call remains content in single-call mode" {
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"execute\"}}]")).ok.array;
    const forms = [_][]const u8{
        "<tool_call><function=tools.session_move><parameter=folder>project</parameter></function></tool_call>",
        "<tool_call>{\"name\":\"tools.session_move\",\"arguments\":{}}</tool_call>",
        "<tool_call>not a tool payload</tool_call>",
    };
    for (forms) |text| for ([_]?usize{ null, 1 }) |limit| {
        const result = try parse(a, text, tools, limit);
        try std.testing.expect(result.calls == null);
        try std.testing.expectEqual(@as(usize, 1), result.rejected_envelopes);
        try std.testing.expectEqualStrings(text, result.content);
    };
}
```
- **Teaching value**: the failure contract is "**the caller always gets the truth**": either a call whose *name* was
  checked against the offered tools, or the model's raw text in `content` plus a count in
  `tensorfold.kept_tool_calls`. There is no third outcome (no 500, no silent drop).
- **Gotchas**, the part that surprises app developers:
  - **An invented tool name** → not an error; `content` keeps the markup, `kept_tool_calls.unoffered` increments,
    the name appears in `unoffered_names`, `finish_reason` stays `"stop"`.
  - **Arguments that do not match the schema** → **never validated** on the non-grammar path. Missing required
    properties, wrong types, extra properties: all pass straight through into `function.arguments` as compact JSON
    (`tool_parse.zig:239-248`). Only `NaN`/`Infinity` and "no name" are rejected (`tool_parse.zig:329-336`).
    Validating arguments is the **caller's** job unless the request compiled a grammar (§9).
  - **Bare JSON that names no offered tool** is treated as a *structured answer* and stays content
    (`bareCalls`, `tool_parse.zig:271-274`).
  - **An envelope the reply never closed** stays text (`envelopes` keeps `complete = false` blocks;
    `reply_text.zig:238-241`), unless the model's end token landed inside an otherwise whole call — then the server
    appends the missing closers and re-parses **strictly**, keeping the call only if it then parses
    (`closeCall`/`argumentsClose`, `tool_parse.zig:195-231`).
  - **GLM repair**: an `<arg_key>` the model forgot is put back before the word ahead of a lone `</arg_key>`
    (`repairGlmKeys`, `tool_parse.zig:425-451`) — the parser actively *fixes* known model mistakes.
  - **Tools disable live logprobs**: `chat_generation.zig:163-166` returns no live logprob rows when
    `tools.len > 0`; the per-call `logprobs.tool_calls` groups are rebuilt afterwards from the markup ranges
    (`logprob_parts.zig:197-198`, documented at `docs/api.md:404-408`) and cover the **markup the model wrote**, not
    the arguments JSON the reply returns.

---

## 12. The two build steps that are the ground truth

- **Anchor**: `zig/build/server.zig:110-114`
- **What it does**: names the two host-only test binaries that cover this map. `test-tool-parse-host` runs the
  parser/visibility/limits tests with no device; `test-tool-visibility-http-host` runs the filtered tests named
  `"tool visibility HTTP:"`, which drive real chat JSON and SSE exchanges through the server.
- **Excerpt**:
```zig
    b.step("test-tool-parse-host", "Tool parser visibility, schemas and single-call limits without devices").dependOn(&run.step);
    const http = b.addRunArtifact(b.addTest(.{ .root_module = server, .filters = &.{"tool visibility HTTP:"} }));
    http.has_side_effects = true;
    b.step("test-tool-visibility-http-host", "Rejected tool envelopes survive actual chat JSON and SSE replies").dependOn(&http.step);
```
- **Teaching value**: the behaviours a lesson teaches are pinned by name in the build graph — a learner can run the
  exact step that guarantees the claim.
- **Gotchas**: the HTTP step's tests live in **two** files, not one: `late_system_retained_test.zig:235-311`
  (`kept_tool_calls` fields, name caps, `/metrics`) and `late_system_test.zig:546-552` (unoffered markup retained,
  whole and streamed). `tool_parse_visibility_test.zig` is reached through `tool_parse_test.zig:598-600`.
  Forced calls and the grammar gate have their own suite, `tool_required_test.zig` (notably `:98-118` "unsupported
  required tools name the actual backend before any grammar or admission", `:240-265` "required without tools,
  unknown names and competing grammars are refused", `:423-...` "auto tools: a strict tool gets the optional-call
  grammar"); the single-call stream contract is `tool_single_stream_test.zig`.

---

## Suggested lesson ordering

Teach it **outside-in**: what you send, what the model reads, what it types, what comes back, and only then what can
force it.

| # | Anchor | The idea | The one interactive control / visual |
|---|---|---|---|
| 1 | `tool_specs.zig:72-97` | `tool_choice` filters the tool list *before* the prompt exists | A three-way toggle `none / auto / {function: "get_weather"}` over a two-tool request; the panel shows the resulting **spec list** (empty, both, one) and the 400 text when you pick an unknown name. |
| 2 | `fixtures/thinking/qwen38.jinja:57-70` + `hf_text.zig:439-440` | The model sees your JSON Schema as **text in a system turn** | A split pane: left, the editable `tools` JSON; right, the rendered prompt with the `<tools>` block highlighted and a live **prompt-token counter** — add a long `description` and watch the count climb on every turn. |
| 3 | `messages.zig:204-243` | The prompt round-trips *previous* calls; bad arguments become `_invalid_arguments` | A "history" editor where a past call's `arguments` can be set to `"{bad"`; the rendered prompt shows the `_invalid_arguments` key appearing. |
| 4 | `tool_parse.zig:293-339` | One payload reader, many spellings | A **markup sandbox**: paste what a model wrote (XML, JSON, `args`, flat keys, a fenced JSON block, GLM tags, Gemma's `call:`) and see the extracted `{name, arguments}` or the reason it is not a call. Preloaded buttons per family. |
| 5 | `tool_parse.zig:109-182` + `tool_parse_visibility_test.zig:6-22` | Rejected calls become **content**, counted, never errors | Two checkboxes on the sandbox: *name is offered* and *markup parses*. The output panel flips between `tool_calls: [...]` and `content: "<tool_call>…"` + `kept_tool_calls {unoffered, malformed}`. |
| 6 | `openai_calls.zig:15-45` | `finish_reason: "tool_calls"` is the **server's** verdict | A reply-shape diff: toggle "offered the tool" and watch `finish_reason` move between `stop` and `tool_calls`, and `content` lose/keep the markup. |
| 7 | `tool_stream.zig:250-301` + `tool_streaming.zig:9-37` | Argument deltas are **synthesised** from XML/GLM/DSML | A **token-by-token scrubber**: drag through a Qwen XML call and watch two columns — raw model text on the left, SSE `arguments` fragments on the right — with the reassembled JSON building underneath. A second tab shows a JSON-envelope call arriving in one delta. |
| 8 | `tool_stream.zig:319-326, 237-249` | A stream cannot rewind, so it buffers when unsure | On the same scrubber, a "make the name unoffered" switch at the call's opener: the delta column stops and the envelope reappears as content at the closer. |
| 9 | `server.zig:345-357, 374-379` | Forced calls = a compiled grammar; no library = HTTP 400 | A capability panel fed by the live `/v1/models` JSON (`structures`, `logprobs`), with a `tool_choice: required` button that renders either the grammar or the exact refusal string, backend name interpolated. |
| 10 | `tool_grammar.zig:68-76, 173-189` | The grammar is derived by **probing the template** | A "probe" visual: render the fake `tfprobe_fn` call for each template, show the detected `Style`, then line the structural tag's `const_string` elements up against that markup. |
| 11 | `tool_grammar.zig:142-148` + `tool_params.zig:121-139` | Schemas bind **only** under a grammar; otherwise they just type values | One slider: *unconstrained → strict tool → required*. Under each setting, feed the same wrong-typed arguments and show what the client receives (string passthrough vs. a schema-shaped object vs. impossible-to-emit). |
| 12 | `tool_stream.zig:510-516` + `tool_parse.zig:185-192` | The limits that actually exist | A counter board for one reply: calls accepted (1 vs ∞), names kept (8 × 64 B), blanks allowed (0–8 / 32) — each number linking to its defining line. |
