// 12-batching.js — what happens when more than one person is talking to the model.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, loop, checkbox, fmt } from '../ui.js';
import { mulberry32 } from '../engine/tensor.js';

/* Cost model: a round reads every weight once (fixed), then pays a little per ROW.
   Lesson 08 measured why the fixed part dominates.                                */
const roundMs = (rows, base, per) => base + rows * per;

export default {
  id: '12-batching',
  num: '12',
  group: 'shipping it',
  title: 'Batching: eight people, one GPU',
  blurb: 'A decode round reads all the weights whether it serves one request or sixteen. That fact decides how a serving engine is built — and the real scheduler is not the obvious one.',

  render(root) {
    const stops = [];

    const S = {
      lanes: 6,            // --parallel
      batchRows: 32,       // cfg.batch_rows
      batchStreams: 32,    // cfg.batch_streams
      drafting: false,     // do streams propose drafts (lesson 14)?
      baseWidth: 8,        // rows a drafting stream asks for
      accept: 0.6,
      arrival: 2.5,
      replyLen: 60,
      base: 24, per: 1.2,
      speed: 6,
      // runtime
      t: 0, acc: 0, running: false, nextId: 1, rng: mulberry32(4),
      live: [], queue: [], done: [], rejected: 0,
      rounds: 0, tokens: 0, nextArrival: 0.3,
      strip: [],           // per round: array of {id, chosen, rows}
    };

    const reset = () => {
      Object.assign(S, {
        t: 0, acc: 0, nextId: 1, rng: mulberry32(4), live: [], queue: [], done: [],
        rejected: 0, rounds: 0, tokens: 0, nextArrival: 0.3, strip: [],
      });
      refresh();
    };

    /**
     * takeTurns — zig/src/core/lanes/engine.zig:163-188.
     * Least-recently-served first, then clipped by TWO budgets: at most
     * batch_streams streams, and at most batch_rows ROWS in total. A stream
     * that does not fit the row budget is simply skipped this round.
     */
    const takeTurns = () => {
      const order = [...S.live].sort((a, b) => a.served - b.served || a.id - b.id);
      const chosen = [];
      let rows = 0;
      for (const s of order) {
        const need = S.drafting ? Math.min(S.baseWidth, S.batchRows) : 1;
        if (chosen.length === S.batchStreams) break;
        if (chosen.length > 0 && rows + need > S.batchRows) continue;
        chosen.push({ s, need });
        rows += need;
      }
      return { chosen, rows };
    };

    const stepRound = () => {
      // arrivals (requests join BETWEEN rounds, never inside one)
      const { chosen, rows } = takeTurns();
      const dur = roundMs(Math.max(1, rows), S.base, S.per) / 1000;

      S.nextArrival -= dur;
      while (S.nextArrival <= 0) {
        if (S.live.length + S.queue.length >= S.lanes + 12) S.rejected++;
        else S.queue.push({ id: S.nextId++, arrivedAt: S.t, tokens: S.replyLen });
        S.nextArrival += -Math.log(1 - S.rng()) / S.arrival;
      }
      while (S.live.length < S.lanes && S.queue.length) {
        const r = S.queue.shift();
        S.live.push({ ...r, left: r.tokens, startedAt: S.t, served: S.rounds, got: 0 });
      }

      // the chosen streams advance; the skipped ones get nothing this round
      const frame = S.live.map((s) => ({ id: s.id, chosen: false, rows: 0 }));
      for (const { s, need } of chosen) {
        let emitted = 1;
        if (S.drafting) {
          emitted = 1;                                   // the bonus token is always committed
          for (let i = 1; i < need; i++) { if (S.rng() < S.accept) emitted++; else break; }
        }
        s.left -= emitted;
        s.got += emitted;
        s.served = S.rounds;
        S.tokens += emitted;
        const f = frame.find((x) => x.id === s.id);
        if (f) { f.chosen = true; f.rows = need; }
      }
      S.strip.push(frame);
      if (S.strip.length > 60) S.strip.shift();

      for (const s of S.live.filter((x) => x.left <= 0)) {
        S.done.push({ id: s.id, wait: s.startedAt - s.arrivedAt, total: S.t + dur - s.arrivedAt, tokens: s.tokens });
      }
      S.live = S.live.filter((s) => s.left > 0);

      S.rounds++;
      S.t += dur;
      refresh();
    };

    /* ---------- the round strip ---------- */
    const strip = new Plot({
      width: 660, height: 210, pad: { l: 56, r: 12, t: 26, b: 28 }, xlim: [0, 1], ylim: [0, 1],
    });
    strip.onDraw((p) => {
      const c = p.ctx;
      const ids = [...new Set(S.strip.flat().map((f) => f.id))].sort((a, b) => a - b).slice(-10);
      const rowH = Math.min(18, p.ih / Math.max(1, ids.length));
      const cw = p.iw / 60;
      ids.forEach((id, r) => {
        const y = p.pad.t + r * rowH;
        c.fillStyle = color('muted');
        c.font = '10px ui-monospace, Menlo, monospace';
        c.textAlign = 'right'; c.textBaseline = 'middle';
        c.fillText('req #' + id, p.pad.l - 6, y + rowH / 2);
        S.strip.forEach((frame, i) => {
          const f = frame.find((x) => x.id === id);
          if (!f) return;
          c.fillStyle = f.chosen ? color('c3') : color('c2');
          c.globalAlpha = f.chosen ? 0.9 : 0.35;
          c.fillRect(p.pad.l + i * cw, y + 1, Math.max(1, cw - 1), rowH - 3);
          c.globalAlpha = 1;
        });
      });
      c.fillStyle = color('muted');
      c.textAlign = 'left';
      c.fillText(`last ${S.strip.length} rounds · green = took a turn · red = skipped`, p.pad.l, p.pad.t - 12);
    });

    const read = stats([
      { k: 'throughput', v: '—' }, { k: 'rows per round', v: '—' },
      { k: 'skipped turns', v: '—' }, { k: 'round time', v: '—' },
      { k: 'completed', v: '0' }, { k: 'refused (429)', v: '0' },
    ]);

    const refresh = () => {
      const { rows } = takeTurns();
      const dur = roundMs(Math.max(1, rows), S.base, S.per);
      const frames = S.strip.flat();
      const skipped = frames.length ? frames.filter((f) => !f.chosen).length / frames.length : 0;
      read.set('throughput', S.t > 0 ? fmt.n(S.tokens / S.t, 1) + ' tok/s' : '—');
      read.set('rows per round', `${rows} / ${S.batchRows}`);
      read.set('skipped turns', fmt.pct(skipped, 0), skipped > 0.2 ? 'warn' : null);
      read.set('round time', fmt.n(dur, 1) + ' ms');
      read.set('completed', String(S.done.length));
      read.set('refused (429)', String(S.rejected), S.rejected ? 'err' : null);
      strip.render();
      curve.render();
    };

    const runBtn = button('▶ run', () => {
      S.running = !S.running;
      runBtn.textContent = S.running ? '⏸ pause' : '▶ run';
    });
    stops.push(loop((dt) => {
      if (!S.running) return;
      S.acc += dt * S.speed;
      const { rows } = takeTurns();
      const dur = roundMs(Math.max(1, rows), S.base, S.per);
      while (S.acc >= dur) { S.acc -= dur; stepRound(); }
    }, strip.canvas));

    root.append(card('A round is not "one token per lane"',
      'The obvious design — every active request advances one token per round — is not what the engine does. Turns are handed out <b>least-recently-served first</b> and clipped by a budget of GPU <b>rows</b>, so a lane can be skipped entirely.',
      row(
        grow(strip.canvas,
          el('div', { style: { height: '8px' } }),
          legend([{ c: 'c3', label: 'took a turn this round' }, { c: 'c2', label: 'skipped — someone else used the rows' }]),
          el('div', { style: { height: '12px' } }),
          btns(runBtn, button('one round ▸', stepRound, { ghost: true }), button('reset', reset, { ghost: true }))),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          read,
          slider({ label: 'lanes <span class="badge">--parallel</span>', min: 1, max: 16, step: 1, value: 6, fmt: String, onInput: (v) => { S.lanes = v; refresh(); } }),
          slider({ label: 'row budget <span class="badge">batch_rows</span>', min: 4, max: 64, step: 4, value: 32, fmt: String, onInput: (v) => { S.batchRows = v; refresh(); } }),
          checkbox({ label: 'streams propose drafts', onChange: (v) => { S.drafting = v; refresh(); } }),
          el('div', { class: 'sub', style: { margin: '-4px 0 2px' },
            html: 'A <b>draft</b> is a cheap guess at the next several tokens, which the big model verifies in one pass. Lesson 14 builds one; all you need here is that a drafting stream asks for several rows instead of one.' }),
          slider({ label: 'rows a drafting stream asks for', min: 2, max: 24, step: 1, value: 8, fmt: String, onInput: (v) => { S.baseWidth = v; refresh(); } }),
          slider({ label: 'arrival rate (req/s)', min: 0.2, max: 12, step: 0.2, value: 2.5, fmt: (v) => v.toFixed(1), onInput: (v) => { S.arrival = v; } }),
          note(`<b>Turn drafting on.</b> Each drafting stream now asks for 8 rows instead of 1, the 32-row budget
                is exhausted after four of them, and the rest are skipped — visibly, in the strip. Speculation
                buys one stream speed by spending a shared budget.`)))));

    root.append(card('The scheduler, as written',
      null,
      row(
        grow(el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/core/lanes/engine.zig:163-188 — takeTurns()</span>\n` +
          `<span class="cm">// participants are least-recently-served first (s.served = the round of its last turn)</span>\n` +
          `const need: u64 = if (s.drafts)\n` +
          `    @min(@min(e.cfg.base_width, e.cfg.batch_rows), 1 + s.force.items.len)\n` +
          `else 1;\n` +
          `if (chosen.items.len == <b>e.cfg.batch_streams</b>) break;\n` +
          `if (chosen.items.len &gt; 0 and rows + need &gt; <b>e.cfg.batch_rows</b>) continue;` }),
          prose(`
            <p>Two budgets, not one. <code class="inline">batch_rows</code> and
            <code class="inline">batch_streams</code> both default to <b>32</b> when streams are on
            (<code class="inline">lanes/config.zig:127</code>, <code class="inline">:170</code>). The scheduling
            unit is a <b>GPU row</b>, not a request: a plain stream needs 1, a drafting stream needs up to
            <code class="inline">base_width</code>.</p>
            <p>The <code class="inline">continue</code> rather than <code class="inline">break</code> matters —
            a stream that does not fit is skipped, but a later, smaller one may still fit in the remaining rows.</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Requests join between rounds, never inside one.</b> The engine thread admits everything
                admissible, then runs exactly one <code class="inline">core.step()</code>
                (<code class="inline">lane_host.zig:507-528</code>). That single fact sets the latency floor:
                you wait at most one round to join, and a round's shape can be planned statically.</p>`),
          note(`<b>Fairness is by turn, not by tokens.</b> Least-recently-served means everyone gets a turn
                eventually, but a drafting stream that commits 6 tokens in its turn gets 6× the output of a plain
                stream that commits 1.`),
          note(`<b>Admission is five questions, not one.</b> <code class="inline">lane_host_admit.zig:11-21</code>:
                memory pressure holds everything; a stale device starts nothing; the lane count must have room;
                KV bytes must fit; and a request waiting on a prefix <i>another prompt is currently filling</i>
                is passed over — so two requests sharing a system prompt do not both prefill it.`)))));

    /* ---------- three gates ---------- */
    root.append(card('Three independent limits, often confused for one',
      null,
      row(
        grow(el('table', {},
          el('thead', {}, el('tr', {},
            el('th', { text: 'limit' }), el('th', { text: 'counts' }), el('th', { text: 'when exceeded' }))),
          el('tbody', {},
            el('tr', {},
              el('td', { text: '--max-connections' }),
              el('td', { style: { textAlign: 'left' }, text: 'open sockets (default 128)' }),
              el('td', { style: { textAlign: 'left' }, text: 'connection refused; one address\u2019s share capped from 96' })),
            el('tr', {},
              el('td', { text: '--max-inflight' }),
              el('td', { style: { textAlign: 'left' }, text: 'requests in progress (default off)' }),
              el('td', { style: { textAlign: 'left' }, text: 'HTTP 429 — never a queue' })),
            el('tr', {},
              el('td', { text: '--parallel' }),
              el('td', { style: { textAlign: 'left' }, text: 'lanes, i.e. concurrent streams' }),
              el('td', { style: { textAlign: 'left' }, text: 'the request waits — this is the only place requests queue' }))))),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b><code class="inline">--parallel auto</code> is the literal constant 8</b>
                (<code class="inline">server/cli.zig:513-515</code>). The real content of the flag is whether the
                count is <i>fixed</i>: <code class="inline">auto</code> permits silent degradation if memory is
                short, while an explicit <code class="inline">N</code> makes the load <b>refuse</b> with
                <code class="inline">error.TooMany</code> rather than quietly give you fewer.`),
          note(`<b><code class="inline">/health</code>'s <code class="inline">live.connections</code> is not
                sockets.</b> It is <code class="inline">running + waiting</code> — admitted plus queued requests
                (<code class="inline">live.zig:19-20</code>). Eight connections with four waiting means four are
                actually decoding.`),
          note(`<b>Queued requests are not idle.</b> Their saved state is read from disk <i>while they wait</i>
                (<code class="inline">lane_host_fetch.zig:13-30</code>), and the engine sleeps when every queued
                request is blocked on a read.`)))));

    /* ---------- the tradeoff curve ---------- */
    const curve = new Plot({
      width: 460, height: 280, pad: { l: 54, r: 20, t: 18, b: 40 },
      xlim: [1, 32], ylim: [0, 1], xlabel: 'rows in one round',
    });
    curve.onDraw((p) => {
      const agg = [], one = [];
      for (let b = 1; b <= 32; b++) {
        const d = roundMs(b, S.base, S.per) / 1000;
        agg.push([b, b / d]);
        one.push([b, 1 / d]);
      }
      const aggMax = Math.max(...agg.map((x) => x[1]));
      p.ylim = [0, aggMax * 1.12];
      p.axes({ xticks: 6, yticks: 5, zero: false });
      p.line(agg, { color: 'c3', width: 2.5 });
      const scale = aggMax / Math.max(...one.map((x) => x[1]));
      p.line(one.map(([b, v]) => [b, v * scale]), { color: 'c1', width: 2.5 });
      p.line([[S.batchRows, 0], [S.batchRows, p.ylim[1]]], { color: 'c5', width: 1.5, dash: [4, 4] });
      p.text(p.pad.l + 4, p.pad.t + 6, 'total tokens/s (everyone)', { color: 'c3', size: 11, px: true, align: 'left' });
      p.text(p.pad.l + 4, p.pad.t + 22, 'tokens/s for one stream', { color: 'c1', size: 11, px: true, align: 'left' });
      const d1 = roundMs(1, S.base, S.per) / 1000, dn = roundMs(S.batchRows, S.base, S.per) / 1000;
      p.text(p.pad.l + 4, p.pad.t + 42,
        `at ${S.batchRows} rows: ${fmt.n(S.batchRows / dn, 0)} tok/s total vs ${fmt.n(1 / d1, 1)} alone — ${fmt.n((S.batchRows / dn) / (1 / d1), 1)}× the work`,
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    root.append(card('Why the row budget exists at all',
      'Both curves come from two numbers: a fixed cost per round and a marginal cost per row.',
      row(
        el('div', {}, curve.canvas,
          el('div', { style: { height: '8px' } }),
          row(
            slider({ label: 'fixed cost per round (ms)', min: 4, max: 60, step: 1, value: 24, fmt: (v) => v + ' ms', onInput: (v) => { S.base = v; refresh(); }, width: '200px' }),
            slider({ label: 'marginal cost per row (ms)', min: 0.1, max: 8, step: 0.1, value: 1.2, fmt: (v) => v.toFixed(1) + ' ms', onInput: (v) => { S.per = v; refresh(); }, width: '200px' }))),
        grow(prose(`
          <p>When <b>fixed ≫ marginal</b> — the memory-bound regime of lesson 08 — adding rows is nearly free and
          total throughput climbs almost linearly. That is why batching exists.</p>
          <p>But the round gets longer with every row, and <i>every</i> stream in the round waits for the slowest
          part of it. Past some width you are buying throughput with everyone's latency, which is precisely why
          there is a cap called <code class="inline">batch_rows</code> rather than "as many as show up".</p>
          <p>Drag <b>marginal cost per row</b> up — a long context, where attention over the KV cache dominates —
          and the green curve flattens while the blue one dives.</p>`),
          note(`<b>This is why the same engine quotes wildly different tokens/second.</b> The question is always
                <i>whose</i> tokens: one stream's, or the sum over everyone sharing the round.`)))));

    reset();
    return () => stops.forEach((s) => s());
  },
};
