# TensorFold lanes / concurrency / reuse code map

Repo root: `TensorFold/`. All paths below are relative to that root.
Every anchor was read at the line numbers given (pinned with numbered grep output at read time).
Companion maps: `notes/map-serving.md` (HTTP → decode loop), `notes/map-forward.md` (one forward pass),
`notes/map-weights.md` (checkpoint → matmul).

**The pipeline in one line:**
TCP accept gate (`http_admission.zig`) → per-request reservation (`request_gate.zig` + `request_admission.zig`)
→ `engine.submit` → **one host queue** (`lane_host.zig:45` `queued`) → admission pick
(`lane_host_admit.zig:11`) + KV byte ledger (`kv_ledger.zig:82`) → **admitted list** (`lane_host.zig:46`)
→ prompt pass (resumed from the prompt cache at `Plan.from`) → the stream joins `lanes/engine.zig:44` `live`
→ one engine thread calls `Engine.step` (`lanes/engine.zig:118`) forever: either **one shared round for many
streams** (`roundStreams`, `:258`) or **one single-reply round** (`familyRound`, `:243`).

Reuse has four independent layers, each with its own store:
1. **KV pages / slots** — preallocated at load (`kv_pool.zig:125`), leased per request.
2. **Prompt cache (RAM)** — token-exact prefix states (`prompt_cache.zig`), the reason turn 2 is fast.
3. **Learned states (disk)** — `--learn`, keyed by a *computation identity* (`prompt_imprint.zig`, `metal_identity.zig:9`).
4. **Recall (disk + RAM record of token ids)** — `--recall`, used only for *drafting* (`lanes/recall.zig`).

---

## 1. Connection admission (`--max-connections`)

- **Anchor**: `zig/src/server/http_admission.zig:19-27`
- **What it does**: The accept loop holds a single atomic counter of open sockets. `acquire` (`:8`) CAS-increments it
  below `limit` and otherwise writes a canned `503 ... Retry-After: 1` (`:6`) and closes. From `pressure(limit)` —
  three quarters full — one remote address may hold only `perPeer(limit)` slots, so a single client cannot occupy
  the whole server while others are queueing. The default limit is `128` (`:5`), so pressure starts at 96 and a
  single address is then capped at 32.
- **Excerpt**:
```zig
/// The open-connection count from which one address's share applies (96 of 128).
pub fn pressure(limit: u32) u32 {
    return limit - limit / 4;
}

/// One address's default share under pressure, so a single host cannot fill the server.
pub fn perPeer(limit: u32) u32 {
    return @max(1, limit / 4);
}
```
- **Teaching value**: a "concurrency limit" is three separate gates at three layers: sockets, in-flight requests, and
  GPU lanes. This one is the cheapest and the dumbest — it never looks at the model.
- **Gotchas**: this gate is *not* `--parallel`. A server with `--parallel 1` still accepts 128 connections; 127 of
  them will sit in the engine queue. Loopback and `--trusted-proxy` addresses are exempt from the per-address share
  (`:33-34`), which is what makes a reverse proxy usable. The refusal is written by the accept thread
  (`server.zig:449-453` `admitConnection`) before any HTTP parsing happens.

---

## 2. Per-request reservation (`--max-inflight`, `--max-inflight-per-key`, rate)

- **Anchor**: `zig/src/server/request_admission.zig:11-21`
- **What it does**: `Limits` is the whole opt-in admission policy. All four fields default to 0, and `enabled()`
  returning false means *no* request is counted, reserved or refused — admission is off unless a flag names a
  number. When enabled, `Controller.acquireAt` (`:100`) checks the global in-flight cap first, then finds the
  caller's principal slot (API-key digest, or `null` for a keyless server) among `principal_capacity = 128` slots
  (`:5`), then the per-key cap, then a token-bucket rate in integer nanosecond credits.
- **Excerpt**:
```zig
pub const Limits = struct {
    max_inflight: u32 = 0,
    max_inflight_per_key: u32 = 0,
    requests_per_second: u32 = 0,
    request_burst: u32 = 0,

    /// All zero, the default: no request is counted, reserved or refused, and /metrics gains nothing.
    pub fn enabled(l: Limits) bool {
        return l.max_inflight > 0 or l.max_inflight_per_key > 0 or l.requests_per_second > 0;
    }
};
```
- **Teaching value**: admission control is a *lease held for the whole reply*, not a check at the door — the counter
  only drops when the stream ends, so it bounds concurrency rather than arrival rate.
- **Gotchas**: this is a **429 refusal, not a queue** — `Reason` (`:23`) is one of `global_limit`, `key_limit`,
  `principal_capacity`, `rate_limit`, each with a wire code and message (`code`/`message` at the end of the file).
  The engine queue is where requests *wait*; `--max-inflight` is where they get *rejected*. `max_limit = 128` (`:4`)
  caps the flag itself. A rate-limited caller gets a computed `retry_after_s` from its missing credit.

---

## 3. Where a request is admitted on the HTTP side

- **Anchor**: `zig/src/server/request_gate.zig:58-79`
- **What it does**: `enter` is the one gate every engine-using POST passes before its handler runs. In order: bump
  `srv.generating`, enter the load ledger, restart the idle window, check the drain flag, take the admission lease
  (if admission is enabled at all), then check model-kind and wake a parked engine. The `Held` value returned is
  released in exactly reverse order by `Held.release` (`:28`) when the reply ends.
- **Excerpt**:
```zig
pub fn enter(srv: *Server, a: Allocator, key_digest: ?[32]u8, group: RouteGroup, kind: RouteKind) Entered {
    // counted before the drain flag is read: a stop that sees no request in progress sees every later one refused
    _ = srv.generating.fetchAdd(1, .seq_cst);
    srv.load.enter(group);
    if (srv.idle_manager) |m| m.begin(); // engine work starts: the idle window restarts at this route's end
    var e: Entered = .{ .held = .{ .srv = srv, .group = group }, .refusal = null };
    if (@import("builtin").is_test) if (@import("routes.zig").counted_hook) |h| h();
    if (srv.draining.load(.seq_cst)) {
        e.refusal = .draining;
        return e;
    }
    // --max-inflight and the rest, when set: a reservation for the whole reply, taken before the body is read
    if (srv.config.admission.enabled()) switch (srv.admission.acquire(srv.io, srv.config.admission, key_digest)) {
        .accepted => |lease| e.held.lease = lease,
        .rejected => |refusal| {
            e.refusal = .{ .overloaded = refusal };
            return e;
        },
    };
    e.refusal = modelKind(srv, kind) orelse wake(srv, a);
    return e;
}
```
- **Teaching value**: the counter is incremented *before* the drain flag is read. That ordering is the whole
  correctness argument for graceful shutdown: a stop that observes zero in-flight requests knows every later
  arrival will be refused.
- **Gotchas**: the reservation is taken **before the request body is read**, so an overload refusal costs no
  parsing and leaves the body unread (`Refusal.overloaded` doc at `:42-43`). `routes.zig:118-160` calls
  `request_gate.enter` and `defer entered.held.release()` for every text route — see `notes/map-serving.md` §1.

---

## 4. What `/health`'s `live.connections` and `live.waiting` actually count

- **Anchor**: `zig/src/server/live.zig:16-23`
- **What it does**: `live.snapshot` is the exact JSON that `/health` embeds (`status_routes.zig:59`). `connections`
  is **not** the socket count: it is `running + waiting`, i.e. requests the *engine* holds. `running` and `waiting`
  come straight from the two lane-host lists: `running = h.admitted.items.len`, `waiting = h.queued.items.len`
  (`lane_host_status.zig:25-26`). The same numbers render the terminal status line (`live.zig:27`).
- **Excerpt**:
```zig
/// /health's ``live``: open requests, how many wait, decode and prefill tokens a second.
pub fn snapshot(a: Allocator, status: api.Status) Allocator.Error!json.Value {
    const o = try json.newObject(a);
    try o.put(a, "connections", try json.intValue(a, status.running + status.waiting));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = round1(status.decode_tokens_per_second) });
    try o.put(a, "prefill_tokens_per_second", .{ .float = round1(status.prefill_tokens_per_second) });
    return .{ .object = o };
}
```
- **Teaching value**: a dashboard number is only as good as its definition. Here "connections" means
  *requests inside the engine*, and `waiting > 0` is the single observable that says "you are lane-bound, not
  compute-bound".
- **Gotchas**: `running` counts admitted jobs including those still *prefilling*, not just decoding. The decode
  rate is a 2-second window (`lane_host_status.zig:27`: tokens / 2.0) and the prefill rate is zeroed once stale
  (`:28`). An idle HTTP keep-alive socket shows up in neither number.

---

## 5. `--parallel N|auto` → lanes

- **Anchor**: `zig/src/server/cli.zig:512-523`
- **What it does**: `--parallel auto` (the default) is literally the number **8**. A number is clamped to
  `1..4096`. `parallelFixed` records *whether the user named a number*, which becomes `Options.lanes_fixed`
  (`core/engine_api.zig:293-295`): with `auto`, an engine whose memory fits fewer streams silently serves fewer;
  with an explicit `N` it **refuses to load** instead (`kv_pool.zig:132` → `error.TooMany`).
- **Excerpt**:
```zig
/// ``--parallel``: "auto" is up to 8 requests at once; a number caps it.
pub fn parallel(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " ");
    if (std.ascii.eqlIgnoreCase(t, "auto")) return 8;
    const n = std.fmt.parseInt(i64, t, 10) catch return null;
    return @intCast(@max(1, @min(n, 4096)));
}

/// ``--parallel`` named a number (an engine that fits fewer refuses), not "auto" (it serves what fits).
pub fn parallelFixed(text: []const u8) bool {
    return !std.ascii.eqlIgnoreCase(std.mem.trim(u8, text, " "), "auto") and parallel(text) != null;
}
```
- **Teaching value**: "auto" is not adaptive magic — it is a constant (8) plus a *permission to degrade*. The
  interesting bit is `lanes_fixed`: the same number means "at most" or "exactly" depending on whether you typed it.
- **Gotchas**: `--parallel` also switches *drivers*, not just a count (§13). `cli.zig:533-534` feeds
  `.lanes = parallel(args.parallel) orelse 8` and `.lanes_fixed = parallelFixed(...)` into the engine options.
  The lane count is further clipped at runtime by `memory_lanes` (`lane_host_admit.zig:16`).

---

## 6. The lane table: one queue, one admitted list, one live list

- **Anchor**: `zig/src/core/lane_host.zig:43-48` and `zig/src/core/lanes/engine.zig:43-47`
- **What it does**: There are exactly three containers, all plain `ArrayList`s, and they are the whole scheduler
  state. `queued` (`:45`) holds submitted-but-not-started jobs (foreground ahead of background, arrival order
  inside each); `admitted` (`:46`) holds jobs that hold a lane; `filling` (`:47`) holds admitted jobs whose prompt
  is still going in a chunk at a time. Inside the lane core, `live` holds the `*Stream`s in admission order — that
  list *is* the set of lanes.
- **Excerpt** (`lane_host.zig:43-48`):
```zig
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    filling: std.ArrayList(*Job) = .empty, // admitted jobs whose prompt is still going in, a chunk a round, oldest first
    cancels: std.ArrayList(Id) = .empty,
```
  and (`lanes/engine.zig:43-47`):
```zig
    rule: depth.Rule,
    live: std.ArrayList(*Stream) = .empty, // admitted streams in admission order (Python `_live`)
    drafted: u64 = 0,
    accepted: u64 = 0,
    shared_rounds: u64 = 0,
```
- **Teaching value**: a production continuous-batching scheduler is three arrays and a mutex. No priority heaps, no
  work-stealing: the ordering policy lives in `submitFn`'s insert position and in `takeTurns`' sort.
- **Gotchas**: `activeCount` (`lanes/engine.zig:93-96`) counts `live` entries that are not finished — finished
  streams linger in `live` until `sweep`. `submitFn` (`lane_host.zig:202`) inserts a foreground request *ahead of
  every background one* by walking back from the tail, which is the only priority rule in the system.

---

## 7. Admission *into a lane*: `pick`, and what makes a request wait

- **Anchor**: `zig/src/core/lane_host_admit.zig:11-21`
- **What it does**: `pick` returns the index of the queued job to start, or null to hold. Order of refusals:
  OS memory pressure holds everything; a stale device (mid-weight-update) starts nothing; the lane count is
  `@min(info_.lanes, memory_lanes orelse info_.lanes)` and `admitted.len` must be below it. Then requests waiting
  for a *prefix another prompt is currently filling* are passed over, so two requests sharing a system prompt do
  not both prefill it. `choose` (`:35`) takes the queue head, but if the head does not `fits` (KV room) it lets
  *later, smaller* requests through while an older stream is still running — bounded anti-starvation, not pure FIFO.
- **Excerpt**:
```zig
/// The queued job to admit (its index), or null to hold; the caller holds the lock.
pub fn pick(h: *LaneHost) ?usize {
    if (pressure.holds(h)) return null; // under memory pressure nothing starts or is read ahead
    if (!h.stale) fetch.queue(h); // disk states read ahead for queued requests, a lane free or not
    // a stale device takes no reply: `learn.refuse` ends the queued ones instead
    if (h.stale or h.queued.items.len == 0 or h.admitted.items.len >= @min(h.info_.lanes, h.memory_lanes orelse h.info_.lanes)) return null;
    prefix.release(h); // requests whose awaited cut is in (or will not come) go on first
    while (true) { // a request that waits for a cut another prompt is filling stays queued, passed over
        const at = choose(h) orelse return null;
        if (!prefix.waits(h, h.queued.items[at])) return at;
    }
}
```
- **Teaching value**: "is there a free lane?" is only the first of five questions. Memory pressure, device staleness,
  KV bytes, and *another request's in-flight prefix* all gate admission before the lane count matters.
- **Gotchas**: the head-of-line rule in `choose:44-50` is subtle: once the head has waited, only requests admitted
  *before it started waiting* may be overtaken — after that nothing jumps it. Disk-state prefetch for *queued*
  requests starts here (`fetch.queue`, §25), so a second-turn request can have its state read from disk while
  other streams still decode. `admit.batches`/`admit.many` (`:61`, `:71`) admit several at once when the backend can
  prefill up to `lanes.Engine.max_together = 16` prompts in one pass (`lanes/open.zig:41`).

---

## 8. Taking the lane: the KV byte promise

- **Anchor**: `zig/src/core/lane_host.zig:353-380`
- **What it does**: `takeJob` is the single transition from `queued` to `admitted`, under the host lock. It asks
  `admit.pick`, then prices the job against the KV ledger: a job that can never fit the *whole* budget fails
  immediately with `ledger.too_big` (rather than waiting forever), a job that does not fit *right now* stays queued
  and nothing is taken, and otherwise the job gets a monotonically increasing `seq` and moves to `admitted`.
- **Excerpt**:
```zig
    pub fn takeJob(h: *LaneHost) Taken {
        h.lock();
        const at = admit.pick(h) orelse { // no lane free, or none may go: a shared cache pool's head waits for room
            h.unlock();
            return .none;
        };
        if (ledger.never(h, h.queued.items[at])) { // past the whole budget alone: no release will ever make it fit
            const job = h.queued.orderedRemove(at);
            h.unlock();
            h.finish(job, .failed, ledger.too_big);
            return .failed;
        }
        if (!ledger.promise(h, h.queued.items[at])) { // bytes before any stream: the job stays queued, nothing taken
            h.unlock();
            return .none;
        }
        const job = h.queued.orderedRemove(at);
        job.noteAdmission();
        job.seq = h.admissions;
        h.admissions += 1;
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return .failed;
        };
        h.unlock();
        return .{ .job = job };
    }
```
- **Teaching value**: admission is *reservation*, not optimism. The request's prompt + reply + one verify window of
  KV is promised before a single token runs, so the engine never discovers mid-decode that it has no room.
- **Gotchas**: `job.seq` is what `choose` later compares against `waits_since` for anti-starvation. `noteAdmission`
  records the admitted timestamp that becomes `Stats.queued_seconds` in the reply (`lane_host.zig:317`) — the
  observable "how long did I wait for a lane". `admitOne` (`:339`) loops over this: the run loop calls
  `while (h.admitOne()) {}`, so a burst drains in one pass.

---

## 9. The engine thread: one loop, rounds between admissions

- **Anchor**: `zig/src/core/lane_host.zig:507-528`
- **What it does**: One thread owns the lane core. Each iteration: take cancels, step memory pressure, answer
  status/score/learn questions, **admit everything admissible**, run prompt-fill chunks, sweep finished streams,
  deliver tokens, then — if any stream is active — run exactly one `core.step()`. If nothing is active it waits on
  a condition variable with a 100 ms timeout (`:548`).
- **Excerpt**:
```zig
    fn run(h: *LaneHost) void {
        defer scoring.close(h);
        if (!@import("lane_thread.zig").start(h)) return;
        // a family planner: prompt pieces and decode rows in one pass; else whole or `prefill_step` chunked prompts
        if (h.rounds) |rd| return rounds.run(h, rd);
        while (true) {
            var activity = @import("lane_activity.zig").Activity.enter(h.activity);
            defer activity.end();
            ends.takeCancels(h);
            pressure.step(h);
            rows_host.answer(h);
            scoring.answer(h);
            learn.refuse(h);
            while (h.admitOne()) {}
            fill.run(h);
            h.core.sweep() catch |err| {
                ends.failRound(h, err);
                continue;
            };
            ends.deliverPending(h);
            status.noteLive(h);
```
- **Teaching value**: new requests join **between rounds**, never inside one. That one sentence explains both the
  latency floor (you wait at most one round to join) and why a round's shape can be planned statically.
- **Gotchas**: the admit loop runs *before* the round, so a request that arrives while the GPU is busy is admitted
  at the top of the next iteration. `ends.deliverPending` runs twice per iteration (before and after the step) so
  tokens reach HTTP threads promptly. `h.share.debt_ns` makes the round *timed* only while prompt work owes decode
  time (`--decode-share`).

---

## 10. The shared round: `step` → `takeTurns` → `roundStreams`

- **Anchor**: `zig/src/core/lanes/engine.zig:118-148`
- **What it does**: `step` collects the unfinished, unpaused streams from `live`. If more than one is live **and**
  `cfg.family_streams` is on, it lands any step-ahead token (a shared round is synchronous), compacts out streams
  that just finished, marks every stream `shared_round`, picks the turn-takers with `takeTurns`, bumps
  `shared_rounds`, and calls `roundStreams`. Otherwise each live stream runs its own `familyRound` (or
  `pipelinedRound`). `e.alone` — exactly one live stream — is what unlocks the lone-stream fast paths (wider
  copies, per-stream measured prices).
- **Excerpt**:
```zig
    pub fn step(e: *Engine) !void {
        _ = e.arena.reset(.retain_capacity);
        const a = e.arena.allocator();
        try trail.event(e, &.{ f("ev", str("step")), f("index", int(e.steps)) });
        e.steps += 1;
        var live: std.ArrayList(*Stream) = .empty;
        for (e.live.items) |s| {
            if (!s.finished and !s.paused) try live.append(a, s);
        }
        if (live.items.len > 1 and e.cfg.family_streams) {
            // a shared round is synchronous: a stream that ran a step ahead lands its queued token first
            for (live.items) |s| {
                if (s.inflight != null) try e.landInflight(s);
            }
            var kept: usize = 0;
            for (live.items) |s| {
                if (s.finished) continue;
                live.items[kept] = s;
                kept += 1;
            }
            live.shrinkRetainingCapacity(kept);
        }
        e.alone = live.items.len == 1;
        if (!e.alone) for (live.items) |s| {
            s.shared_round = true;
        };
        if (live.items.len > 1 and e.cfg.family_streams) {
            const chosen = try e.takeTurns(live.items);
            e.shared_rounds += 1;
            for (chosen) |s| s.served = e.shared_rounds;
            try e.roundStreams(chosen);
        } else {
```
- **Teaching value**: the batch is re-formed from scratch **every round**. There is no persistent "batch object" —
  which is exactly why a finishing lane and an arriving request cost nothing structurally.
- **Gotchas**: a lane that finishes mid-round is dropped from `live` by the `kept` compaction *before* the round is
  planned, and permanently removed by `sweep()` at the end of `step` (`:157`). `e.alone` flips the engine between
  two genuinely different cost models — a two-request server is not "the one-request server twice".

---

## 11. `takeTurns`: a round is *not* "one token per lane"

- **Anchor**: `zig/src/core/lanes/engine.zig:163-188`
- **What it does**: The round's participants are chosen least-recently-served first (`s.served` is the
  `shared_rounds` counter at its last turn), then clipped by two budgets: at most `cfg.batch_streams` streams, and
  at most `cfg.batch_rows` **rows** in total. Each stream's `need` is its minimum rows — `1` for a plain stream, or
  `@min(base_width, batch_rows)` bounded by its forced tokens for a drafting stream. The first chosen stream is
  never cut (`chosen.items.len > 0 and ...`), so a round always has at least one lane. The chosen set is re-sorted
  into stream order and traced as a `turns` event.
- **Excerpt**:
```zig
    fn takeTurns(e: *Engine, live: []*Stream) ![]*Stream {
        const a = e.arena.allocator();
        const order = try a.alloc(usize, live.len);
        for (order, 0..) |*o, i| o.* = i;
        const Ctx = struct {
            live: []*Stream,
            fn served(c: @This(), i: usize) i64 {
                return if (c.live[i].served) |v| @intCast(v) else -1;
            }
            fn less(c: @This(), x: usize, y: usize) bool {
                const sx = c.served(x);
                const sy = c.served(y);
                return sx < sy or (sx == sy and x < y);
            }
        };
        std.mem.sort(usize, order, Ctx{ .live = live }, Ctx.less);
        var chosen: std.ArrayList(usize) = .empty;
        var rows: u64 = 0;
        for (order) |i| {
            const s = live[i];
            const need: u64 = if (s.drafts) @min(@min(e.cfg.base_width, e.cfg.batch_rows), 1 + s.force.items.len) else 1;
            if (chosen.items.len == e.cfg.batch_streams) break;
            if (chosen.items.len > 0 and rows + need > e.cfg.batch_rows) continue;
            try chosen.append(a, i);
            rows += need;
        }
```
- **Teaching value**: the scheduling unit is a **GPU row**, not a request. With speculative drafting one lane can
  consume 16 rows, so "128 concurrent streams" and "16 rows per forward" are compatible only because lanes take
  turns across rounds.
- **Gotchas**: a lane can be *live but not served* in a given round — it keeps its KV and its place, it just gets
  no token that round. That is the latency cost of throughput, and it is explicitly round-robin fair
  (least-recently-served), not fair-share by tokens. Constants: `batch_rows` defaults to **32** and
  `batch_streams` to **32** when streams are on (`lanes/config.zig:127`, `:170`); the Qwen3.8-27B Metal runner
  raises streams to **128** (`families/qwen27/attention_plan.zig:3`, re-exported at `native/qwen27_lanes.zig:28`),
  and the Flash Next shared forward caps a round at **16 rows** (`families/flashnext/batch_meta.zig:3`).

---

## 12. The shared forward itself: one `verify` for the whole round

- **Anchor**: `zig/src/core/lanes/engine.zig:258-274`
- **What it does**: `roundStreams` plans one `Plan` per stream (forced tokens, a copied continuation, or held
  drafts), allocates draft widths across all of them *jointly* (`win.allocate`), builds one `be.Window` per stream,
  and issues **one** `backend.verify(windows, out)` for the whole round. Only afterwards is each stream concluded,
  committed and re-drafted separately. `total` accumulates the round's rows for cost accounting.
- **Excerpt**:
```zig
    fn roundStreams(e: *Engine, entries: []*Stream) !void {
        e.clock.start();
        try e.late.flush(e, Engine.askDrafts);
        const a = e.arena.allocator();
        const plans = try a.alloc(Plan, entries.len);
        for (entries, plans) |s, *p| p.* = try win.plan(e, s, null);
        try win.allocate(e, plans);
        try @import("grammar_parallel.zig").constrain(e, plans);
        const windows = try a.alloc(be.Window, plans.len);
        const out = try a.alloc(be.Verified, plans.len);
        var total: u64 = 0;
        for (plans, windows, out) |p, *w, *o| {
            w.* = try win.build(e, p, false);
            o.* = .{ .sampled = try a.alloc(u32, w.rows()), .drafts = try a.alloc(u32, w.rows() - 1), .rows = if (p.stream.logprobs != null) try a.alloc(LogRow, w.rows()) else &.{} };
            total += w.rows();
        }
        try e.backend.verify(windows, out);
        const outcomes = try a.alloc(Outcome, plans.len);
```
- **Teaching value**: batching happens at exactly one call — `verify` over a list of windows. Everything around it
  (planning, sampling, commit, drafting) stays per-stream, which is why per-request sampling settings and grammars
  survive batching.
- **Gotchas**: `sampled` is `w.rows()` long but `drafts` is `w.rows() - 1` — a window of N rows verifies N-1 drafts
  and samples one bonus token. Draft budgets are decided jointly *after* verify
  (`shared_drafts.budgets` / `rule.budgets`, `:285-290`), and with `cfg.draft_streams` every stream's draft head
  runs in one batch (`:292`). Grammar masks for all streams are built in parallel
  (`grammar_parallel.constrain`), not per stream.

---

## 13. Batched vs single-reply drivers, and `FZ_BATCH_MTP_SERIAL`

- **Anchor**: `zig/src/families/flashnext/batch_meta.zig:61-71`
- **What it does**: For Flash Next, shared sessions are opened **only** when `--parallel` named a number above 1
  (`fixed`) and the server is neither a two-Mac pair nor learning. `auto` — the default — keeps the single-reply
  host. `refusal` returns the exact sentence the loader prints when an explicit `--parallel N > 1` cannot be
  honoured. The generic switch is `lanes/config.zig:126`: `family_streams` is on only when the family's exact
  window is ≥ 2 rows (or plain sharing is allowed) *and* the backend reports exact streams and hidden rows;
  `batch_streams` is `max_streams` or 32 when streams are on, and **1** when they are not (`:170`).
- **Excerpt**:
```zig
/// An explicit --parallel above one shares sessions; auto (the default, as on CUDA), one, learning and pairs do not.
pub fn shared(fixed: bool, streams: u32, paired: bool, learning: bool) bool {
    return fixed and streams > 1 and !paired and !learning;
}

/// Why an explicit --parallel above one cannot open; null when it can (auto keeps the single-reply host).
pub fn refusal(fixed: bool, streams: u32, paired: bool, learning: bool) ?[]const u8 {
    if (!fixed or streams <= 1) return null;
    if (paired) return "Flash Next shared sessions are single-Mac only; use --parallel 1 with --speed-up";
    if (learning) return "Flash Next shared sessions spill evicted states (--spill-gib) but do not write states behind (--learn, --learn-conversations) yet; use --parallel 1 with those";
    return null;
}
```
- **Teaching value**: "batched" is a *load-time* decision, not a per-request one. The same binary has two decode
  drivers, and which one you get depends on a CLI string, the family, and whether another feature (learning,
  two-Mac link) claims the slot.
- **Gotchas**: `FZ_BATCH_MTP_SERIAL=1` does **not** turn off shared rounds — it only makes each stream's MTP drafts
  use their own command buffers instead of one shared buffer (`families/flashnext/backend.zig:50`, read from the
  environment at `:79`). Sibling knobs read in the same line: `FZ_BATCH_MTP_CHAIN`, `FZ_VERIFY_OVERLAP=0`,
  `FZ_SETTLE_SYNC`. `lanes/config.zig:170` is the one place `batch_streams` becomes 1 — that is the real
  "single-reply" switch inside the core. The lone-stream driver is additionally selected per job at runtime by
  `lone.fits` (`lane_host.zig:403-406`, `startsAlone` at `:395`).

---

## 14. The prompt cache: token-exact prefix matching

- **Anchor**: `zig/src/core/prompt_cache_index.zig:7-17`
- **What it does**: `find` is the entire matching rule: walk a nonowning index sorted longest-position-first and
  return the **first** kept entry whose tokens are an exact prefix of this prompt, whose canonical decode spans
  agree through its position, whose images match, and which leaves at least one row to run (`e.at < prompt.len`).
  Matching granularity is **token-exact over the whole prefix** — not block-hashed — and the cheap last-token
  check (`prompt[n - 1] != e.tokens[n - 1]`) rejects most candidates before the full compare.
- **Excerpt**:
```zig
/// The longest kept token prefix with matching canonical spans through its position, leaving at least one row.
pub fn find(s: *pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?*pc.Entry {
    std.debug.assert(s.by_position.items.len == s.entries.items.len); // add and Store.remove keep them equal
    for (s.by_position.items[below(s, prompt.len)..]) |e| { // longest first: the first match is the scan's winner
        if (e.expired or e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
        const n = e.tokens.len;
        if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
        if (!modes.equal(e.decode_spans, spans, e.at) or !@import("cache_images.zig").matches(s, e)) continue;
        return e;
    }
    return null;
}
```
- **Teaching value**: prefix caching is a *linear scan over a few dozen entries* sorted by length, with a binary
  search (`below`, `:34`) to skip states longer than the prompt. No radix tree is needed when the store holds tens
  of states, not millions.
- **Gotchas**: a state stores `tokens` = position **plus `lookahead` more** (`prompt_cache.zig:85`), so a match
  validates more tokens than it resumes — that is how a family whose rows depend on the next token stays exact.
  `usable` (`prompt_cache.zig:243-245`) rejects positions that are not chunk starts for "planned" families, so for
  those the granularity is effectively the chunk grid, not any token. `modes.equal` on `decode_spans` means two
  prompts with identical tokens but different *row arithmetic* (e.g. int8 vs bf16 prompt precision) never share a
  state. Paged families use a radix variant instead (`core/kv_prompts.zig`, `families/qwen3_5/cuda_prefix.zig:99`).

---

## 15. Turning a partial hit into "only run the new tokens"

- **Anchor**: `zig/src/core/prompt_cache.zig:263-281`
- **What it does**: `beginUser` is the one call a backend makes at the start of a prompt pass. It looks up the best
  entry, asks the family to `restore` it into the live state, and returns a `Plan{ from, marks }`. `from` is the
  position the prompt pass starts at — *the GPU runs only `prompt[from..]`*. `marks` are the positions where this
  pass should stop and keep new states. If a restore fails, the entry is dropped and the **next shorter** RAM state
  is tried before falling back to a pass from 0.
- **Excerpt**:
```zig
    pub fn beginUser(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, user_end: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        var l = try s.lookupUser(a, prompt, @intCast(prompt.len), history_len, rewind_len, user_end, shared, starts, spans);
        // the first lookup's marks, kept for the pass from the start if no shorter kept state resumes
        var fresh_marks: ?[]const u32 = null;
        while (l.entry) |e| {
            if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| {
                const at = e.at; // read before the reserve may evict the entry
                s.resumedWithReason(e, prompt, null);
                if (fresh_marks) |m| if (m.ptr != l.marks.ptr) a.free(m);
                // Restored: this state may go too; a peer restores before it drops.
                s.reserve(prompt, @intCast(prompt.len), null, l.marks, spans, @import("prompt_cache_reserve.zig").endpointFor(s, prompt, at, history_len, starts, spans));
                return .{ .from = at, .marks = l.marks };
            } else |err| {
                // a refused state is dropped, then the next shorter RAM state is tried before the pass from the start
                if (fresh_marks == null) fresh_marks = l.marks else a.free(l.marks);
                s.resumedWithReason(e, prompt, err);
                l = try s.lookupUserAt(a, prompt, @intCast(prompt.len), history_len, rewind_len, user_end, shared, starts, spans, false);
            }
        }
```
- **Teaching value**: the hit is turned into work avoidance by a single integer. `Plan.from` is the whole payoff of
  prefix caching: prefill cost becomes proportional to `prompt.len - from`.
- **Gotchas**: the backend consumes `from` through `Stream.reuse` (`lanes/stream.zig:84-87`): `saved` is the
  family's copy and `at` must equal `reuse.at`, or the backend sets `s.reuse_failed` and starts at 0
  (e.g. `families/flint/backend.zig:213-216`, `families/gemma4/cuda_backend.zig:326-332`,
  `families/glm/cuda_lanes.zig:382-389` which also records *why*). `s.cached` then reports how many prompt tokens
  were skipped, and that is what the prefill rate and the reply's `prefilled` event report
  (`lane_host.zig:447-469`). A server-side caller is `native/flashnext_host.zig:372-373`, which passes
  `history_len`, `rewind_len`, `user_end` and `shared_prefixes` computed by `server/prompt.zig`
  (see `notes/map-serving.md` §5b).

---

## 16. Where new states are kept — why turn 2 is fast

- **Anchor**: `zig/src/core/prompt_cache.zig:412-423`
- **What it does**: `marksRewind` decides the positions a pass stops at to keep states. The first and most
  important mark is **the end of the conversation history** (`history_len`) — the prompt minus the new user turn.
  Keeping *there* is what makes the next turn resume: the next prompt is `history + new reply + new question`, so
  the kept state is an exact prefix of it. Additional marks: the active rewind position, shared system-block cuts,
  a "stable prefix" fork point, a repeat-detected mark, and (for short prompts) the chunk end.
- **Excerpt**:
```zig
    fn marksRewind(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, rewind: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, previous: []const u32) ![]const u32 {
        const short = !s.rules.planned and prompt.len < s.rules.min_prompt; // the history's mark only, where the pass cuts a chunk anyway
        if (short and s.rules.chunk == 0) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        // a warm pass (its history the whole prompt) keeps where a state can be keyed, its lookahead before the end
        const warm_pass = s.rules.warm != .off and history_len == prompt.len and s.rules.lookahead > 0;
        const repeat_at = s.repeat_log.mark(s.rules.repeat_min, prompt, from, s.rules.lookahead, s.rules.planned);
        const history: u32 = if (warm_pass) @intCast(prompt.len -| s.rules.lookahead) else if (history_len == 0) repeat_at else history_len;
        try want.append(a, if (short) cuts.chunkEnd(s.rules.chunk, from, history, prompt.len, starts, spans) else history);
        if (rewind > 0 and s.active_rewind_at == rewind) try want.append(a, rewind);
        // the prompt this one parts from: the one it resumed, or (fork rule) on a miss the kept one sharing the most
        const kept = if (previous.len > 0 or !s.rules.fork) previous else forks.longestShared(Entry, s.entries.items, prompt);
        const part: u32 = if (kept.len == 0 or short) 0 else @intCast(std.mem.indexOfDiff(u32, kept, prompt) orelse @min(kept.len, prompt.len));
```
- **Teaching value**: a prefix cache only pays off if someone *chose where to cut*. The cut that matters for chat is
  "end of history", and it is chosen by the server telling the cache where the history ends — not by the cache
  guessing.
- **Gotchas**: `Store.keep` (`:444`) is also called **after the reply** with the reply's tokens included
  (`native/flashnext_host.zig:302` keeps at each planned `at`), so the state the second turn resumes already
  contains the first answer — the second turn prefills only the new user message. A short prompt
  (`< rules.min_prompt`) keeps *only* the history mark. `chunkEnd` (`prompt_cuts.zig:14`) snaps a mark back to a
  whole-chunk boundary so the resume point is always a legal pass start. A "repeat log" lets a prompt with no
  declared history earn a mark after being seen `repeat_min` times (`prompt_cache_repeat.zig`).

---

## 17. Prompt-cache eviction order

- **Anchor**: `zig/src/core/prompt_cache_policy.zig:10-32`
- **What it does**: `victimIn` is the eviction policy, in one scan with one prefix compare per entry. It builds
  three candidates: `best` = the least recently used **redundant** state (superseded by a longer state on the same
  token path, or extended by a live continuation), `rung` = a superseded ladder checkpoint (sacrificed only after
  all other redundant copies), and `oldest` = the plain LRU fallback. Never evicted: entries with `borrowers > 0`
  (an unfinished prompt pass holds them), entries protected by the active reservation guard, `shared` cuts,
  `branched` states, and the active rewind state (which sorts last).
- **Excerpt**:
```zig
pub fn victimIn(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32, own_only: bool) ?usize {
    var best: ?usize = null;
    var rung: ?usize = null; // a superseded ladder checkpoint: after every other redundant copy, before any newest state
    var oldest: ?usize = null;
    var oldest_rewind = false;
    for (s.entries.items, 0..) |e, i| { // one prefix compare an entry: the superseded scan and the oldest scan share it
        if (e.borrowers > 0 or e == skip) continue;
        if (s.reservation_guard) |g| if (g.protects(s, e)) continue;
        if (own_only and !own(e, skip, prompt)) continue;
        const on = e.tokens.len <= prompt.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
        const rewind = !e.expired and e.at == s.active_rewind_at and on; // the active rewind state goes last
        if (e.borrowers == 0 and e != skip and (oldest == null or (if (rewind != oldest_rewind) oldest_rewind else e.used < s.entries.items[oldest.?].used))) {
            oldest = i;
            oldest_rewind = rewind;
        }
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or rewind) continue;
        const now = (born > e.born or (born == e.born and !e.rewind)) and prompt.len > e.tokens.len and on;
        if (!now and !extended(s, e)) continue;
        const pick = if (e.intermediate) &rung else &best;
        if (pick.* == null or e.used < s.entries.items[pick.*.?].used) pick.* = i;
    }
    return best orelse rung orelse oldest;
}
```
- **Teaching value**: "LRU" is the *last* resort. A good prefix cache first throws away states that are provably
  redundant — a shorter state on the same conversation path whose longer successor still lives.
- **Gotchas**: `born` (`prompt_cache.zig:90`) is the prompt length of the request that created the state, and it is
  how "superseded" is decided — a newer, longer request's state supersedes an older shorter one on the same path.
  `shared = true` (`:94`) marks system-prompt cuts other conversations resume, so they survive their own
  conversation's progress. Eviction is driven by `giveBack` (`prompt_cache.zig:218-230`), which loops
  `remove(victimBut(...))` until the caller's `left.fits()` is satisfied and returns false if it can never fit.
  `used` (`:91`) is a store-internal logical clock, not wall time; wall times in `seen` are explicitly
  "never policy" (`:92`).

---

## 18. KV capacity planning at load (`--prompt-cache-gib` vs streams)

- **Anchor**: `zig/src/core/kv_pool.zig:125-135`
- **What it does**: `plan` splits the device's post-weights room, in a fixed priority order: the engine's own
  sequence first, then `lanes - 1` extra streams' fixed parts plus a page each, then the prompt cache, then
  everything left becomes pool pages. `fit` is how many extra streams actually fit; if that is fewer than asked
  **and** `--parallel` named a number (`b.fixed`), it returns `error.TooMany` instead of serving fewer. The cache
  is then split into snapshot slots and copied-KV bytes by `split` (`:118`, at most `most_snaps = 64` slots, `:74`).
- **Excerpt**:
```zig
pub fn plan(l: Layout, b: Budget) error{ NoStream, TooMany }!Plan {
    const own = ownPages(l, b);
    if (b.engine + own > b.room or b.lanes == 0) return error.NoStream;
    var left = b.room - b.engine - own;
    const extra: u64 = b.lanes - 1;
    const each = l.fixed_bytes + (1 + l.runPages()) * l.pageBytes();
    const fit = @min(extra, left / each);
    if (fit < extra and b.fixed) return error.TooMany;
    left -= fit * l.fixed_bytes;
    const cache = @min(b.cache, left -| held(l, b, fit, windows(l, b, left) - 1));
    const taken = @min(b.cache, left -| held(l, b, fit, 0)) - cache; // what the windows past the engine's own took
```
- **Teaching value**: lanes and prompt cache are the **same memory**. Every extra concurrent request is paid for in
  cache bytes, and `--prompt-cache-gib` is the knob that reverses the trade.
- **Gotchas**: `Layout.need` (`:53`) is the per-request reservation formula — `pages(prompt + max_tokens + window) +
  runPages()`, i.e. a request reserves room for *its whole possible reply plus one verify window*, which is why a
  huge `max_tokens` reduces achievable concurrency. `holdingGib` (`:77`) rounds a slot count up to whole GiB, and
  `holding` (`:155`) answers "what `--prompt-cache-gib` keeps all N slots". A cache the user *asked* for
  (`Budget.asked`, `:103`) is reserved before every stream window except the engine's own; otherwise the windows
  take theirs first (`taken`). `--prompt-cache-gib` defaults to null, meaning "what 70% of RAM leaves past the
  loaded server" (`server/cli.zig:197`); `0` means no cache at all.

---

## 19. Per-request KV promise and release (the slot's lifetime)

- **Anchor**: `zig/src/core/kv_ledger.zig:82-97`
- **What it does**: The ledger is the byte budget admitted jobs promise from. `promise` prices a job — a job that
  can sit on one of the **standing streams** paid for at load pays only what it adds past that stream — fails with
  `error.LaneMemory` if it does not hold, and records the lease. `finish` returns the bytes and the standing stream
  and empties the lease, so an empty lease returns nothing twice. `fits` (`:51`) additionally counts the prompt
  store's *reclaimable* pages as room, because eviction can free them.
- **Excerpt**:
```zig
    pub fn promise(l: *Ledger, lease: *Lease, bytes: u64) error{LaneMemory}!void {
        std.debug.assert(lease.bytes == 0 and !lease.bound);
        const p = l.price(bytes);
        if (!l.holds(p)) return error.LaneMemory;
        lease.bound = l.bound < l.standing.streams;
        l.bound += @intFromBool(lease.bound);
        l.promised += p;
        lease.bytes = p;
    }

    /// `lease`'s bytes and standing stream go back and it is empty again; an empty lease returns nothing.
    pub fn finish(l: *Ledger, lease: *Lease) void {
        l.promised -= lease.bytes;
        l.bound -= @intFromBool(lease.bound);
        lease.* = .{};
    }
```
- **Teaching value**: capacity is preallocated (pages exist from load), but *ownership* is leased per request. The
  first N requests are nearly free because their streams were already paid for at load.
- **Gotchas**: `error.LaneMemory` from a prompt pass is **not** a failure — `failedPass` (`lane_host.zig:421-428`)
  requeues the job (`ends.requeue`). `finish` is called from `LaneHost.finish` via `ledger.release`
  (`lane_host.zig:308`), and `h.memory_lanes = null` there forces the next admission to re-ask the backend how many
  lanes memory allows. The page pool itself (`kv_pool.zig:237` `Pool.init`) is allocated once with a free stack and
  a per-page holder count; a request's pages are a `Lease` of page ids in token order (`:186`), and
  `available() = nfree - promised` (`:263-265`) is what admission tests. For a paged family the whole ledger *is*
  the pool (`Ledger.ofPages`, `:27-30`), so "bytes" and "pages" never disagree.

---

## 20. Slot reset between requests, and where `cache_len` advances

- **Anchor**: `zig/src/core/lanes/open.zig:84-103`
- **What it does**: `openedCommitted` is the reset point. After the prompt pass, the stream's context is truncated
  to `prompt_len`, its logprob rows cleared, `pending` cleared, and `cache_len` set to `prompt_len` — i.e. the
  lane's notion of "how much KV is mine" is re-established from the prompt, whether the pass ran from 0 or resumed
  from a cached prefix. A cancelled stream releases its backend state here (`e.backend.release(s)`) and never gets
  a first token.
- **Excerpt**:
```zig
fn openedCommitted(e: *Engine, s: *Stream, reach: Reach, hook: ?be.Committed) !void {
    if (s.promptOnly()) {
        if (s.isCancelled()) {
            e.backend.release(s);
            return error.Cancelled;
        }
        s.cache_len = s.prompt_len;
        s.pending = null;
        return;
    }
    if (e.backend.first_before_draft and e.cfg.family_mtp and s.drafts) return @import("open_first.zig").opened(e, s, reach, hook);
    if (s.isCancelled()) { // cancelled in its last chunk: no first token
        e.backend.release(s);
        return error.Cancelled;
    }
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.rows.clearRetainingCapacity();
    s.pending = null;
    s.cache_len = s.prompt_len;
    const position: u64 = s.prompt_len;
```
- **Teaching value**: a "slot" is not cleared memory — it is three counters reset to the prompt length. Reuse is
  safe because nothing past `cache_len` is ever read.
- **Gotchas**: during decode `cache_len` advances by the number of **accepted** rows, not by 1 — see the second
  excerpt below (`lanes/engine.zig:419-430`), which is the speculative-decoding bookkeeping
  `notes/map-serving.md` points at. The pipelined path advances it by exactly one before queueing a draw
  (`engine.zig:569-571`). At the end the stream is appended to `live` (`open.zig:145`) and finally handed back by
  `handover`/`sweep` (`lanes/engine.zig:111-115`); its KV lease goes back through `ledger.release`.
- **Second excerpt (the decode-time advance,** `lanes/engine.zig:419-430`**)**:
```zig
        var committed_rows: []LogRow = &.{};
        if (rows_lp.len > 0) {
            committed_rows = try a.alloc(LogRow, committed.items.len);
            for (committed_rows, committed.items, path[0..committed.items.len]) |*r, t, row| r.* = rows_lp[row].forToken(t);
        }
        const landed = try s.commit(e.gpa, committed.items, committed_rows);
        const got = committed.items[0..landed];
        if (s.finished and got.len < path.len) path = path[0 .. got.len + 1];
        const keep = path.len;
        s.cache_len += keep;
        s.pending = committed.items[keep - 1];
        const chain = accept.isChain(parents);
```

---

## 21. Learned prefixes on disk (`--learn`): the index

- **Anchor**: `zig/src/core/prompt_imprint.zig:26-42`
- **What it does**: `Imprint.open` is the on-disk learned-state store. States live under
  `<root>/<identity-as-16-hex-digits>/`, with an append-only `index` file of records (`Meta`: key, position `at`,
  the tokens read, the chunk starts below it, bytes) and one payload file per state written by the family. Opening
  cleans up `.part` files, loads the index keeping any valid prefix across a torn tail, removes payloads the index
  does not name, stamps "last opened", and totals what *other* identities hold so the shared cap can be enforced.
- **Excerpt**:
```zig
    /// The states learned under `identity` in `root` (made when missing); other identities past `cap` go, oldest first.
    pub fn open(gpa: Allocator, root: []const u8, identity: u64, cap: u64) !Imprint {
        var m: Imprint = paths: {
            const r = try gpa.dupe(u8, root);
            errdefer gpa.free(r);
            const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ root, identity }, 0);
            errdefer gpa.free(dir);
            try makePath(gpa, dir);
            break :paths .{ .gpa = gpa, .root = r, .dir = dir, .cap = cap };
        };
        errdefer m.deinit();
        @import("lanes").learned_dirs.cleanupParts(root) catch return error.ImprintRead;
        try m.load();
        try @import("lanes").learned_dirs.removeUnindexed(gpa, m.dir, m.metas.items);
        m.stamp();
        m.others = @import("lanes").learned_dirs.totalOthers(root, m.dir) catch return error.ImprintRead;
        return m;
    }
```
- **Teaching value**: persisting a GPU state needs *three* keys, not one: which prompt (token hash), which position,
  and which **computation** produced it (the directory name).
- **Gotchas**: the state key is `keyOf` = `Wyhash(0x1e47, tokens)` over the state's tokens (`:57`) — a pure token
  hash, with the index record carrying the tokens so a collision or corruption is caught (`add` rejects
  `key != keyOf(tokens)`, `:152`). Matching is `best` (`:103`): longest learned state that is a token prefix,
  leaves a row, and (for planned families) agrees on the chunk starts below it. Eviction is LRU by `victim`
  (`:94`), and `fits` (`:88`) first sweeps *other identities* oldest-opened-first before refusing. An index record
  whose `version` is newer than `VERSION = 2` makes the whole store refuse rather than guess (`readIndex`,
  `:252-271`). `--learn-gib` defaults to **32** GiB across every model and build
  (`core/engine_api.zig:315`, `server/cli.zig:206`), with `--learn-min-free-gib` 4 and `--learn-daily-gib` 32
  (`cli.zig:205`, `:208`). `--learn-conversations` writes *every* kept state behind, not just shared cuts
  (`engine_api.zig:316-317`), and plugs in a different store (`Store.disk`, `prompt_cache.zig:135`) instead of
  `Store.imprint`.

---

## 22. Validating learned state after an upgrade: the identity hash

- **Anchor**: `zig/src/native/metal_identity.zig:9-25`
- **What it does**: This is the answer to "what happens after an upgrade that changes the computation". The
  directory name is a Wyhash over: the running **executable's own file identity**, every checkpoint shard's stamp,
  the MTP head's safetensors header, the prompt chunk size, whether a head exists, the **kernel source hash**, a
  **numeric probe** (a fixed 4,143-token prompt actually run through the model), the precision mode, the OS build,
  the GPU name, and a family/backend profile. Anything that changes the arithmetic changes the hash, so old states
  simply live in a different directory — never read, and reclaimed later as "other identities".
- **Excerpt**:
```zig
/// What learned and disk states read under: checkpoint, shards, prefill step, head, kernels, a probe, OS build, chip.
pub fn learned(a: Allocator, io: std.Io, dir: []const u8, b: *nemotron.backend.Metal) !struct { id: u64, rate: f64 } {
    var h = std.hash.Wyhash.init(0x6e65);
    try @import("sys").file_identity.current(&h, a, io);
    try shardStamps(&h, a, io, dir);
    if (b.head != null) h.update(try safetensorsHeader(a, dir, "mtp-4bit.safetensors"));
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const probe = try nemotron.learned_prompts.probe(b);
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    for ([_]u64{ b.o.chunk, @intFromBool(b.head != null), nemotron.kernels.sourceHash(), probe }) |x| h.update(std.mem.asBytes(&x));
    const precision = @import("tensorfold").frags.precision;
    precision.hashMode(&h, b.m.device.tensorUnits() and try precision.requested());
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(b.m.device.name()));
    @import("disk_identity.zig").profile(&h, .{ .family = "nemotron_h", .backend = "metal" });
    return .{ .id = withPolicy(h.final(), b), .rate = 4143.0 / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}
```
- **Teaching value**: cross-version cache safety is solved by *keying on the computation*, not by versioning the
  format. You cannot accidentally read a state produced by different kernels, because you never look in its folder.
- **Gotchas**: the probe is a real forward pass at startup, which also measures the machine's prefill rate
  (`rate = 4143 / seconds`) — and it is why `--learn` needs `--context 4096` or more (`native/metal.zig:246-248`
  prints exactly that refusal). A shard rewritten **in place with identical bytes** still changes the identity,
  because the stamp includes mtime (the test at the bottom of this file asserts it). `withPolicy` (`:29-32`)
  extends the hash with the resolved K-slice split, so a policy change also invalidates. Sibling derivations:
  `families/flashnext/flashnext_disk.zig:138-142` (engine identity hash, learned source hash, three device modes,
  probe) and `native/cuda_disk.zig:262`. A weight update at runtime calls `Store.clear`
  (`prompt_cache.zig:357-365`), which expires every RAM state and detaches `imprint`/`disk` entirely:
  "old arithmetic".

---

## 23. Recall (`--recall`): the record

- **Anchor**: `zig/src/core/lanes/recall_record.zig:98-123`
- **What it does**: Recall's store is a flat, append-only log of token ids in 1 Mi-token chunks (`chunk_tokens`,
  `:10`) with an 8-token key index. `push` appends one id; a full chunk allocates a new one; `mark`
  (`maxInt(u32)`) ends a segment so no key spans it and no copy crosses it. Every position whose preceding 8 ids
  form a complete key is linked into a bucket by `back` distance, newest-first — a singly linked suffix index built
  in O(1) per token with no rebuild on growth (`heads` is a fixed power-of-two table chosen at `init`).
- **Excerpt**:
```zig
    /// One id at the end; a full newest chunk takes a new one first.
    pub fn push(r: *Record, t: u32) !void {
        const p = r.end;
        if (@as(usize, p) == @as(usize, r.base()) + r.chunks.items.len * chunk_tokens) {
            const ids = try r.gpa.alloc(u32, chunk_tokens);
            errdefer r.gpa.free(ids);
            const back = try r.gpa.alloc(u32, chunk_tokens);
            errdefer r.gpa.free(back);
            try r.chunks.append(r.gpa, .{ .ids = ids, .back = back });
        }
        const c = r.chunks.items[r.chunks.items.len - 1];
        const at = p % chunk_tokens;
        c.ids[at] = t;
        c.back[at] = 0;
        r.end = p + 1;
        if (t == mark) {
            r.run = 0;
            return;
        }
        std.mem.copyForwards(u32, r.last[0 .. key_len - 1], r.last[1..]);
        r.last[key_len - 1] = t;
        r.run += 1;
        if (r.run < key_len) return;
        const b = r.bucket(&r.last);
        if (r.heads[b] != 0) c.back[at] = p - (r.heads[b] - 1);
        r.heads[b] = p + 1;
    }
```
- **Teaching value**: "remember what the server has seen" can be a chunked token log plus a hash-chain index —
  8 bytes per token (ids + back links), no model state at all.
- **Gotchas**: Recall stores **token ids only**, never KV — it can only make *drafts*, never skip prefill. Lookup
  cost is bounded by `max_steps = 64` candidates (`:12`) and a match is extended back at most `max_match = 64`
  (`:14`). `ramBytes` (`:70-72`) is `chunks * chunk_tokens * 8 + heads * 4`, i.e. about twice the disk size. The
  store is keyed by **tokenizer identity**, not by model weights: `recall_store.zig:156-186` refuses a directory
  written under a different tokenizer digest/vocab with a named error, and the default directory is
  `$HOME/.cache/tensorfold/recall/<tokenizer identity>` (`:551-555`). Each chunk file carries a 32-byte header
  with magic, version, vocab size and the identity's first 16 bytes (`:417-424`); any id ≥ vocab stops the load
  and is never written (`:357`). `--recall-gib` defaults to 1/16 GiB (`server/cli.zig:211`), and
  `cap_chunks = max(2, cap_bytes / chunk_bytes)` (`recall_store.zig:100`).

---

## 24. Recall drafting: priced before the record is even read

- **Anchor**: `zig/src/core/lanes/recall.zig:112-136`
- **What it does**: `Reader.propose` is the drafting path. Before touching the record it asks whether a copy can
  *pay*: `Rates.pays` (`:61`) compares the expected tokens from a band's learned landing rate against `x.head` —
  the stream's own recent tokens-per-round. If it cannot pay, the lookup happens only as an occasional probe. On a
  hit it needs `min_match` (default 16, `bands[0]`) matched ids and at least 2 copied ids. The answer is memoised
  by `(context tail, width)` so the same ask inside one round is free.
- **Excerpt**:
```zig
    pub fn propose(x: *Reader, ctx: []const u32, most: usize, round: u64) []const u32 {
        if (ctx.len == 0 or most < 2 or round < x.quiet_until) return &.{};
        const key = memoKey(ctx, most);
        if (x.memo) |m| if (m == key) return x.out.items;
        x.out.clearRetainingCapacity();
        x.out.ensureTotalCapacity(x.gpa, most) catch return &.{};
        x.memo = key;
        x.shared.lock();
        defer x.shared.unlock();
        const rates = &x.shared.rates;
        var probing = false; // the price comes first: the record is walked only when a copy can pay, or for a probe
        if (!rates.pays(most, x.head)) {
            probing = rates.probe();
            if (!probing) return &.{};
        }
        x.lookups += 1;
        const m = x.shared.record.longest(ctx);
        if (m.len < x.params.min_match) return &.{};
        const got = x.shared.record.copy(m.at, x.out.allocatedSlice()[0..most]);
        if (got.len < 2) return &.{};
        const b = Rates.band(m.len);
        if (!probing and expected(rates.rate(b), got.len) <= x.head) {
            probing = rates.probe();
            if (!probing) return &.{};
        }
```
- **Teaching value**: a speculative source must *earn its place*. Recall is a bandit: three match-length bands
  (16/32/64, `:25`) each with a learned landing rate, exponentially-backed-off probes when a band loses, and the
  stream's own draft performance as the bar to beat.
- **Gotchas**: priors are deliberately asymmetric — `landed = {0.75, 2, 4.5}` vs `missed = {0.5, 0.5, 0.5}`
  (`:31-32`) encode 0.6/0.8/0.9 per-id success, "worth half a miss each". A copy that lands nothing silences the
  stream for `quiet` (4) rounds (`observe`, `:148-158`); `noteHead` (`:162-165`) is the EMA of the stream's own
  rounds and is what Recall must beat (`lanes/engine.zig:397`, `:537` call it). The engine only consults Recall
  when the stream's own proposer gave nothing (`lanes/windows.zig:205-220` `copyProposal`), and a *priced*
  proposer suppresses Recall entirely. Grafts use a different entry point, `candidates` (`:168-193`), merged behind
  the stream's own suffix matches (`lanes/fill.zig:100-108`: own context wins ties). Wiring:
  `core/lane_host_recall.zig:24-30` creates a per-job `Reader` only if `job.request.drafts`; `:38-41`/`:45-50`
  write the prompt and the reply back into the record unless the job is a background, untaught or prompt-only
  pass. Per-reply telemetry is `{"recall": {...}}` (`recall.zig:196-199`).

---

## 25. Reading disk states ahead for *queued* requests

- **Anchor**: `zig/src/core/lane_host_fetch.zig:13-30`
- **What it does**: Called from `admit.pick` on **every** scheduler iteration, whether a lane is free or not. For
  each queued, foreground, token-keyed request it chooses the best disk state and starts an asynchronous read, so
  the state is already in memory when the request is finally admitted. A request whose chosen state is currently
  being *written* waits briefly rather than racing the writer. When the read finishes, `wake` (`:72-77`) signals
  the engine thread — which may be idle precisely because every queued request was waiting for a read.
- **Excerpt**:
```zig
/// Reads no queued request names go; then reads start for queued requests a disk state serves (the lock held).
pub fn queue(h: *LaneHost) void {
    const store = h.cache orelse return;
    const r = store.disk orelse return;
    if (r.fetch) |f| f.sweep(h, named);
    if (h.admitted.items.len == 0) return; // alone: its state is read where it is admitted, as before
    var fetcher: ?*prefetch.Prefetch = null;
    for (h.queued.items) |job| {
        if (job.fetch.looked or job.request.background or !keyed(job.request)) continue; // warm passes read where they run
        const f = fetcher orelse prefetch.ensure(r, h.io, .{ .ctx = h, .call = wake }) orelse return;
        fetcher = f;
        if (!f.on()) return; // TENSORFOLD_RESTORE_PREFETCH=0: nothing is looked up or read here
        write.reap(r);
        const q = job.request;
```
- **Teaching value**: queueing is an opportunity, not just a delay — the time a request spends waiting for a lane is
  spent loading its cached state from SSD.
- **Gotchas**: this only runs when **something else is already admitted** (`admitted.len == 0` returns early): a
  lone request reads its state inline where it is admitted, as before. `held` (`:45-55`) is what makes
  `admit.pick` pass over a request whose read is still in flight, and `admit.holds` (`lane_host_admit.zig:29-32`)
  lets the engine sleep when *every* queued request is waiting on a read or a prefix. `keyed` (`:68-70`) excludes
  prompt-only, logprob, untaught and image requests. `TENSORFOLD_RESTORE_PREFETCH=0` disables it.

---

## Walkthroughs the lesson needs

### A. Eight requests arrive at once (`--parallel auto`, no `--max-inflight`)

1. Eight sockets are accepted: `http_admission.acquire` (`http_admission.zig:8`) admits all eight — the default
   limit is 128 and per-address pressure starts at 96 (`:5`, `:20`).
2. Eight HTTP threads each call `request_gate.enter` (`request_gate.zig:58`). Admission is **off by default**
   (`request_admission.zig:18`), so no 429s; each takes the in-flight count and the idle-window hold.
3. Each thread templates and tokenises, then calls `engine.submit` → `submitFn` (`lane_host.zig:202`), which inserts
   the job into `queued` in arrival order (foreground ahead of background) and signals the engine thread.
4. The engine thread's loop runs `while (h.admitOne()) {}` (`lane_host.zig:517`). Each `takeJob` (`:353`) asks
   `admit.pick` (`lane_host_admit.zig:11`): lanes are `min(info_.lanes = 8, memory_lanes)`, so up to 8 are
   admitted — each after `ledger.promise` reserves `pages(prompt + max_tokens + window)` worth of bytes
   (`kv_pool.zig:53`, `kv_ledger.zig:82`). A ninth request stays in `queued` and shows up as `/health` →
   `live.waiting = 1` (`lane_host_status.zig:26`, `live.zig:19-20`).
5. Prompts go in: either batched (up to `max_together = 16` in one pass, `lanes/open.zig:41`, via
   `admit.batches`/`admit.many`) or a chunk at a time through `filling`. Each pass starts at `Plan.from` from the
   prompt cache (`prompt_cache.zig:263`), so a shared system prompt is prefilled once — and a request waiting for a
   prefix another pass is filling is deliberately passed over (`lane_host_admit.zig:18-20`).
6. Each opened stream appends itself to `live` (`lanes/open.zig:145`, list at `lanes/engine.zig:44`) and takes part
   **from the next round**.
7. Decode: every loop iteration calls `core.step()` (`lanes/engine.zig:118`). With 8 live streams and
   `family_streams` on, `takeTurns` (`:163`) picks the least-recently-served streams that fit `batch_streams`
   (32 by default) and `batch_rows` (32 by default) — so with speculative drafting not all 8 necessarily get a
   token in the same round. `roundStreams` (`:258`) issues **one** `backend.verify` for the whole set.
8. As each lane finishes, `deliver`/`finish` (`lane_host.zig:298`, `:306`) release its ledger lease and reset
   `memory_lanes`, and the next `admitOne` pulls the next queued request in — between rounds, never inside one.
   The reply's `Stats.queued_seconds` (`lane_host.zig:317`) reports how long it waited for its lane.

### B. The second turn of a conversation

1. The server re-renders the whole conversation and computes `history_len` (the prompt minus the new user turn),
   plus `rewind_len`/`user_end` (`server/prompt.zig`, see `notes/map-serving.md` §5b).
2. The backend calls `beginUser` (`prompt_cache.zig:263`), which calls `find` (`prompt_cache_index.zig:7`):
   longest-first scan for a kept state whose tokens are an exact prefix and whose decode spans match.
3. The state it finds is the one kept at **turn 1's reply end** — `Store.keep` was called after the reply with the
   reply's tokens included (`prompt_cache.zig:444`, e.g. `native/flashnext_host.zig:302`) — and/or at the history
   mark chosen by `marksRewind` (`:412`).
4. `restore` makes the live GPU state that state's; `Plan.from = e.at` is returned, the backend sets `s.cached`
   from `Stream.reuse` (`lanes/stream.zig:84`), and the prompt pass runs **only `prompt[from..]`** — the new user
   message. `s.cache_len` is then set to `prompt_len` (`lanes/open.zig:90`/`:102`) and decode advances it by
   accepted rows (`lanes/engine.zig:428`).
5. If the state was evicted from RAM but `--learn`/`--learn-conversations` wrote it, it comes back from disk:
   either prefetched while the request was queued (`lane_host_fetch.zig:13`) or found by `Imprint.best`
   (`prompt_imprint.zig:103`) under the identity directory for *this exact build* (`metal_identity.zig:9`).
   After an upgrade that changes the arithmetic the directory name differs, so nothing stale is ever read.
6. Independently, `--recall` lets the second turn *draft* from token ids seen in the first
   (`lanes/recall.zig:112`), which speeds up decode even when the prefix cache missed.

---

## Suggested lesson ordering

1. **"Where does a request wait?"** — `http_admission.zig:19-27` → `request_gate.zig:58-79` →
   `lane_host.zig:43-48`. *Visual*: three nested boxes (sockets / in-flight / lanes) with live counters; sliders for
   `--max-connections` and `--max-inflight` showing which box refuses and which box queues.
2. **`/health` as ground truth** — `live.zig:16-23` + `lane_host_status.zig:25-26`. *Interactive*: a "fire N
   requests" button with `--parallel` fixed at 2, watching `connections` rise while `waiting` absorbs the excess.
3. **The scheduler loop** — `lane_host.zig:507-528`. *Visual*: a round timeline where new arrivals snap to the next
   round boundary. One control: arrival time, to show the "join between rounds" latency floor.
4. **Choosing who runs** — `lanes/engine.zig:118-148` then `:163-188`. *Interactive*: sliders for `batch_rows`
   (default 32) and drafts-per-lane; lanes go grey in rounds where they did not fit. This is the
   throughput-vs-latency lesson.
5. **One forward, many windows** — `lanes/engine.zig:258-274`. *Visual*: the row-concatenation diagram (each lane's
   window stacked into one `verify`), linking back to `notes/map-forward.md`.
6. **Admission is reservation** — `lane_host.zig:353-380` + `kv_ledger.zig:82-97` + `kv_pool.zig:125-135`.
   *Interactive*: a memory bar with three sliders — `--parallel`, `max_tokens`, `--prompt-cache-gib` — showing
   lanes and cache slots trading against each other, and `--parallel N` turning red (`error.TooMany`) where `auto`
   would silently serve fewer.
7. **Prefix cache: the hit** — `prompt_cache_index.zig:7-17` then `prompt_cache.zig:263-281`. *Interactive*: edit a
   conversation's text and watch `Plan.from` collapse to the first changed token — the most convincing single demo
   of why turn 2 is fast.
8. **Prefix cache: the keep** — `prompt_cache.zig:412-423`. *Visual*: the prompt as a bar with marks at history
   end / rewind / shared cut; toggle "keep at history end" off and watch turn 2 become a full prefill.
9. **Prefix cache: the eviction** — `prompt_cache_policy.zig:10-32`. *Interactive*: a fixed-slot cache plus a
   branching conversation tree; highlight which state is "redundant" vs "oldest", and which are pinned
   (`shared`, `branched`, `borrowers > 0`).
10. **Disk, keyed by the computation** — `prompt_imprint.zig:26-42` then `metal_identity.zig:9-25`. *Interactive*:
    a "pretend you upgraded" switch that changes the kernel hash and shows the directory name — and therefore every
    learned state — change.
11. **Recall: drafting from history** — `lanes/recall_record.zig:98-123` then `lanes/recall.zig:112-136`.
    *Visual*: the token log with its 8-token key chain; a band-rate panel showing `pays` / probe backoff live.
12. **Closing the loop** — `lanes/open.zig:84-103` + `lanes/engine.zig:419-430`. *Visual*: one lane's `cache_len`
    advancing by the number of *accepted* draft rows, tying this map back to speculative decoding in
    `notes/map-serving.md`.
