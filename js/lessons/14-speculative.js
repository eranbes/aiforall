// 14-speculative.js — getting more than one token per pass over the weights.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, loop, fmt } from '../ui.js';
import { NGram, tokenize, detokenize } from '../engine/ngram.js';
import { DEFAULT_CORPUS } from '../engine/corpus.js';
import { sample } from '../engine/sample.js';

const label = (t) => (t === '\n' ? '⏎' : t);

export default {
  id: '14-speculative',
  num: '14',
  group: 'shipping it',
  title: 'Speculative decoding',
  blurb: 'Decoding one token reads every weight in the model. Verifying eight tokens reads them exactly once too. So guess eight, check them all in one pass, and keep the ones that were right — with provably identical output.',

  render(root) {
    const stops = [];

    /* ---------- two real models: a strong target, a weak drafter ---------- */
    const target = new NGram(3).train(DEFAULT_CORPUS);   // sees 2 tokens of context
    const drafter = new NGram(2).train(DEFAULT_CORPUS);  // sees 1 token — cheaper, worse
    // the drafter must speak the target's vocabulary for ids to be comparable
    drafter.vocab = target.vocab;
    drafter.index = target.index;

    const SEED = 7;
    const state = {
      ids: tokenize('The engine reads the').map((t) => target.index.get(t)).filter((i) => i !== undefined),
      depth: 4,
      temp: 0.5,          // the target's sampling temperature
      gamma: 0.15,        // drafter cost as a fraction of one target pass
      rounds: [],         // history of {draft, picks, accepted}
      drafted: 0, accepted: 0, roundsRun: 0, tokensOut: 0,
      running: false, acc: 0,
    };

    /** Sampling must be the SAME function the target would use alone: seed + absolute position. */
    const targetPick = (ids, position) => {
      const { logits } = target.logits(target.context(ids));
      return sample(logits, { temperature: state.temp, seed: SEED, position }).chosen;
    };
    /** The drafter proposes its single best guess — greedy, as a real draft head does. */
    const drafterPick = (ids) => {
      const { logits } = drafter.logits(drafter.context(ids));
      return sample(logits, { temperature: 0 }).chosen;
    };

    /**
     * One speculative round, exactly as the engine does it:
     *   1. the drafter proposes `depth` tokens, cheaply, one at a time
     *   2. the target scores every proposed position IN ONE PASS
     *   3. walk the chain: keep a draft iff it equals the target's own pick there
     *   4. commit the TARGET's tokens, plus one bonus token that speculation gets free
     */
    const roundOnce = () => {
      const base = state.ids.slice();
      const pos0 = base.length;

      const draft = [];
      let ctx = base.slice();
      for (let i = 0; i < state.depth; i++) {
        const t = drafterPick(ctx);
        draft.push(t);
        ctx = ctx.concat(t);
      }

      // the target's pick at every row, computed as if the drafts were real context
      const picks = [];
      for (let i = 0; i <= state.depth; i++) {
        picks.push(targetPick(base.concat(draft.slice(0, i)), pos0 + i));
      }

      // acceptPath: first row whose drafted token differs from the target's pick ends it
      let kept = 0;
      while (kept < state.depth && draft[kept] === picks[kept]) kept++;

      const committed = picks.slice(0, kept + 1);   // target's tokens + the free bonus
      state.ids = base.concat(committed);
      state.rounds.unshift({ draft, picks, kept, committed });
      if (state.rounds.length > 7) state.rounds.pop();
      state.drafted += state.depth;
      state.accepted += kept;
      state.roundsRun++;
      state.tokensOut += committed.length;
      if (state.ids.length > 200) state.ids = state.ids.slice(-150);
      refresh();
    };

    /* ---------- round visual ---------- */
    const roundsBox = el('div', { class: 'col', style: { gap: '10px' } });

    const chip = (text, kind) => el('span', {
      class: 'tok',
      style: {
        background: kind === 'ok' ? 'color-mix(in srgb, var(--ok) 22%, var(--surface-2))'
          : kind === 'bad' ? 'color-mix(in srgb, var(--err) 22%, var(--surface-2))'
            : kind === 'bonus' ? 'color-mix(in srgb, var(--accent) 22%, var(--surface-2))'
              : 'var(--surface-2)',
        borderColor: kind === 'ok' ? 'var(--ok)' : kind === 'bad' ? 'var(--err)' : kind === 'bonus' ? 'var(--accent)' : 'var(--border)',
        opacity: kind === 'dead' ? '0.42' : '1',
      },
    }, text);

    const renderRounds = () => {
      roundsBox.replaceChildren(...state.rounds.map((r, idx) => {
        const draftRow = el('div', { class: 'tokens' },
          el('span', { style: { color: 'var(--muted)', minWidth: '96px', fontSize: '11px' }, text: 'drafter says' }),
          ...r.draft.map((t, i) => chip(label(target.vocab[t]), i < r.kept ? 'ok' : i === r.kept ? 'bad' : 'dead')));
        const targetRow = el('div', { class: 'tokens' },
          el('span', { style: { color: 'var(--muted)', minWidth: '96px', fontSize: '11px' }, text: 'target picks' }),
          ...r.picks.map((t, i) => chip(label(target.vocab[t]), i < r.kept ? 'ok' : i === r.kept ? 'bonus' : 'dead')));
        return el('div', {
          style: {
            borderLeft: `2px solid ${idx === 0 ? 'var(--accent)' : 'var(--border)'}`,
            paddingLeft: '12px', opacity: idx === 0 ? '1' : String(Math.max(0.35, 1 - idx * 0.14)),
          },
        },
          el('div', { style: { fontSize: '11px', color: 'var(--muted)', marginBottom: '4px' },
            html: `round −${idx} · drafted ${r.draft.length} · <b style="color:var(--ok)">accepted ${r.kept}</b> · committed ${r.committed.length} token${r.committed.length > 1 ? 's' : ''}` }),
          draftRow, targetRow);
      }));
    };

    const outEl = el('div', {
      style: {
        font: '13px/1.7 ui-monospace, Menlo, monospace', whiteSpace: 'pre-wrap',
        background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '9px',
        padding: '11px 13px', minHeight: '76px', maxHeight: '150px', overflowY: 'auto',
      },
    });

    const read = stats([
      { k: 'acceptance rate', v: '—' }, { k: 'tokens / round', v: '—' },
      { k: 'target passes', v: '0' }, { k: 'tokens emitted', v: '0' },
      { k: 'speedup', v: '—' },
    ]);

    const refresh = () => {
      const alpha = state.drafted ? state.accepted / state.drafted : 0;
      const perRound = state.roundsRun ? state.tokensOut / state.roundsRun : 0;
      const cost = 1 + state.depth * state.gamma;      // one target pass + depth drafter passes
      read.set('acceptance rate', state.drafted ? fmt.pct(alpha, 1) : '—');
      read.set('tokens / round', state.roundsRun ? fmt.n(perRound, 2) : '—');
      read.set('target passes', String(state.roundsRun));
      read.set('tokens emitted', String(state.tokensOut));
      read.set('speedup', state.roundsRun ? fmt.n(perRound / cost, 2) + '×' : '—',
        perRound / cost > 1 ? 'ok' : 'warn');
      outEl.textContent = detokenize(state.ids.map((i) => target.vocab[i]));
      outEl.scrollTop = outEl.scrollHeight;
      renderRounds();
      econ.render();
    };

    const runBtn = button('▶ run', () => {
      state.running = !state.running;
      runBtn.textContent = state.running ? '⏸ pause' : '▶ run';
    });
    stops.push(loop((dt) => {
      if (!state.running) return;
      state.acc += dt;
      if (state.acc > 420) { state.acc = 0; roundOnce(); }
    }, outEl));

    root.append(card('Why guessing is worth it',
      null,
      row(
        grow(prose(`
          <p>Lesson 08 measured it: generating one token is <b>memory-bound</b>. The GPU reads every weight in the
          model, multiplies each one exactly once, and sits idle waiting for the next cache line. The arithmetic
          units are mostly unemployed.</p>
          <p>So running the model on <i>eight</i> token positions at once costs almost the same wall-clock time as
          running it on one. The weights get read once either way; only the (cheap, parallel) arithmetic grows.</p>
          <p>That asymmetry is the whole opportunity. If something cheap can <i>guess</i> the next eight tokens,
          the big model can check all eight guesses in a single pass — and every guess that was right is a token
          you got for free.</p>`)),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          note(`<b>The catch, and why this is not an approximation.</b> A guess is kept only if it is
                <i>literally the token the big model would have produced anyway</i>. Not "close enough", not
                "probably fine" — the same <code class="inline">u32</code>. The output is bit-identical to
                decoding one token at a time.`),
          note(`<b>What makes that checkable</b> is lesson 01's design: the target's draw is a pure function of
                (seed, position, token id). "What the big model would have chosen" is therefore a value you can
                compute and compare, not an event you had to witness.`)))));

    root.append(card('A real speculative decoder, running here',
      'The <b>target</b> is the trigram model from lesson 01. The <b>drafter</b> is a bigram — same vocabulary, one token less context, much cheaper and often wrong. Everything below is actually computed.',
      row(
        grow(outEl,
          el('div', { style: { height: '10px' } }),
          btns(
            button('one round ▸', roundOnce),
            runBtn,
            button('reset', () => {
              state.ids = tokenize('The engine reads the').map((t) => target.index.get(t)).filter((i) => i !== undefined);
              state.rounds = []; state.drafted = 0; state.accepted = 0; state.roundsRun = 0; state.tokensOut = 0;
              refresh();
            }, { ghost: true })),
          el('div', { style: { height: '14px' } }),
          roundsBox),
        el('div', { class: 'col', style: { flex: '0 1 300px' } },
          read,
          slider({
            label: 'draft depth <span class="badge">k</span>', min: 1, max: 10, step: 1, value: 4,
            fmt: String, onInput: (v) => { state.depth = v; refresh(); },
          }),
          slider({
            label: 'drafter cost (× a target pass)', min: 0.02, max: 0.6, step: 0.01, value: 0.15,
            fmt: (v) => v.toFixed(2), onInput: (v) => { state.gamma = v; refresh(); },
          }),
          slider({
            label: 'target temperature', min: 0, max: 1.4, step: 0.05, value: 0.5,
            fmt: (v) => (v === 0 ? '0 (greedy)' : v.toFixed(2)),
            onInput: (v) => { state.temp = v; refresh(); },
          }),
          legend([
            { c: 'ok', label: 'guessed right — free token' },
            { c: 'err', label: 'first wrong guess — round ends' },
            { c: 'accent', label: 'bonus token, always free' },
          ]),
          note(`Every round commits the <b>target's</b> tokens, never the drafter's — even for an accepted row.
                That is why a lucky-but-wrong draft cannot leak into the output.`),
          note(`<b>Temperature governs acceptance.</b> Measured on this page over 30 rounds: at
                <b>T = 0</b> acceptance is ~49% and the speedup ~1.85×; at <b>T = 1.2</b> acceptance falls to
                ~18% and the speedup to ~1.08×. A confident target is predictable, and only a predictable
                target can be guessed. This is why speculation pays most on code and structured output, and
                least on deliberately creative sampling.`)))));

    /* ---------- the acceptance rule ---------- */
    root.append(card('The acceptance rule is one comparison',
      null,
      row(
        grow(el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/core/lanes/accept.zig:39-62</span>\n` +
          `pub fn acceptPath(gpa: Allocator, tokens: []const u32, parents: []const i32, picks: []const u32) ![]u32 {\n` +
          `    ...\n` +
          `    while (true) {\n` +
          `        const at = path.items[path.items.len - 1];\n` +
          `        const want = <b>picks[at]</b>;              <span class="cm">// what the TARGET sampled here</span>\n` +
          `        var next: ?u32 = null;\n` +
          `        for (parents, 0..) |q, row| {\n` +
          `            if (q == @as(i32, @intCast(at)) and <b>tokens[row] == want</b>) {\n` +
          `                next = @intCast(row);  <span class="cm">// the draft matched: extend the path</span>\n` +
          `                break;\n` +
          `            }\n` +
          `        }\n` +
          `        try path.append(gpa, next orelse break);\n` +
          `    }\n` +
          `}\n\n` +
          `<span class="cm">/// The target's pick at an accepted row's parent:</span>\n` +
          `<span class="cm">/// the reply's token, never the draft's proposal.</span>\n` +
          `pub fn targetPick(picks: []const u32, parents: []const i32, row: u32) u32 {\n` +
          `    return picks[@intCast(parents[@intCast(row)])];\n` +
          `}` })),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b><code class="inline">tokens[row] == want</code> is the entire test.</b> No rejection sampling,
                no probability correction, no tolerance. Other systems need a probabilistic acceptance rule
                because their target's draw is a random event; here it is a function.`),
          note(`<b>Trees, not just chains.</b> <code class="inline">parents</code> is per-row, so a drafter may
                propose a <i>tree</i> of alternatives and the walk picks the branch that matched. One target pass
                can then verify several competing continuations at once.`),
          note(`<b>The tree is validated, not trusted.</b> <code class="inline">depths()</code> returns
                <code class="inline">error.ParentAfterChild</code> rather than looping, and
                <code class="inline">sanitizeTree</code> truncates to a node budget and drops orphaned subtrees.`),
          note(`<b>Rejection is a signal, not just a loss.</b> The accepted/drafted counters feed
                <code class="inline">observeDepth</code> — an exponential moving average of per-depth acceptance
                that decides how deep to draft next round. The engine learns how much to guess.`)))));

    /* ---------- the economics ---------- */
    const econ = new Plot({
      width: 460, height: 290, pad: { l: 50, r: 16, t: 16, b: 42 },
      xlim: [1, 12], ylim: [0, 4], xlabel: 'draft depth k', ylabel: 'speedup vs. one token per pass',
    });
    const ALPHAS = [0.5, 0.7, 0.85, 0.95];
    econ.onDraw((p) => {
      // expected committed tokens for a chain draft of depth k at per-token acceptance a:
      //   E = sum_{i=0..k} a^i   = (1 - a^(k+1)) / (1 - a)
      // cost of the round, in target passes: 1 + k*gamma
      const curve = (a) => {
        const pts = [];
        for (let k = 1; k <= 12; k++) {
          const E = (1 - Math.pow(a, k + 1)) / (1 - a);
          pts.push([k, E / (1 + k * state.gamma)]);
        }
        return pts;
      };
      const all = ALPHAS.map(curve);
      p.ylim = [0, Math.max(2, ...all.flat().map((d) => d[1])) * 1.12];
      p.axes({ xticks: 6, yticks: 5, zero: false });
      p.line([[1, 1], [12, 1]], { color: 'muted', width: 1.5, dash: [5, 4] });
      p.text(11.8, 1, 'no speculation', { color: 'muted', size: 10, align: 'right', baseline: 'bottom' });
      all.forEach((pts, i) => {
        p.line(pts, { color: `c${i + 1}`, width: 2 });
        // mark each curve's best depth: speculation has an optimum, it is not "more is better"
        let best = 0;
        pts.forEach((q, j) => { if (q[1] > pts[best][1]) best = j; });
        p.scatter([pts[best]], { color: `c${i + 1}`, r: 5, stroke: 'bg' });
      });
      const meas = state.drafted ? state.accepted / state.drafted : null;
      if (meas != null && meas > 0 && meas < 1) {
        p.line(curve(meas), { color: 'accent', width: 2.5, dash: [4, 3] });
      }
      p.scatter([[state.depth, 0]], { color: 'c5', r: 4 });
    });

    root.append(card('More guessing is not better',
      'Each extra draft token costs drafter time whether or not it survives, and its chance of surviving decays geometrically. There is an optimum, and the engine chases it.',
      row(
        el('div', {}, econ.canvas,
          el('div', { style: { height: '8px' } }),
          legend(ALPHAS.map((a, i) => ({ c: `c${i + 1}`, label: `acceptance ${fmt.pct(a, 0)}` }))
            .concat([{ c: 'accent', label: 'your measured rate' }, { c: 'c5', label: 'your depth' }]))),
        grow(
          el('div', { class: 'eq', html: 'E[tokens] = Σ<sub>i=0..k</sub> α<sup>i</sup> = (1 − α<sup>k+1</sup>) / (1 − α)' }),
          el('div', { class: 'eq', html: 'speedup = E[tokens] / (1 + k·γ)' }),
          prose(`
            <p>α is the chance a single drafted token survives; γ is the drafter's cost as a fraction of one
            target pass. The dots mark each curve's best depth.</p>
            <p>At <b>α = 0.5</b> the fifth draft token survives 3% of the time — you are paying the drafter for
            near-certain waste. At <b>α = 0.95</b> deep drafting pays, which is why engines that draft aggressively
            invest in very good drafters (a trained MTP head, not a smaller model).</p>
            <p>Raise the drafter-cost slider above and watch every curve collapse toward 1. A drafter that costs
            40% of the target buys nothing at any depth unless it is almost always right.</p>`),
          note(`<b>Why the bonus token matters.</b> Even when the very first guess is wrong, the round still
                commits one token — the target's own pick at row 0. Speculation can waste drafter time; it can
                never make the target slower per round.`)))));

    /* ---------- in the engine ---------- */
    root.append(card('In the real engine', null, row(
      grow(
        el('pre', { class: 'code', html:
          `<span class="cm">// zig/src/core/lanes/engine.zig:327-352 — conclude()</span>\n` +
          `path = try accept.acceptPath(a, tokens, parents, sampled);\n` +
          `for (path[1..]) |r| try committed.append(a, accept.targetPick(sampled, parents, r));\n` +
          `try committed.append(a, sampled[path[path.len - 1]]);   <span class="cm">// the free bonus token</span>\n` +
          `if (s.proposer) |pr| pr.round(@intCast(rows), @intCast(path.len - 1));\n` +
          `if (rows > 1) {\n` +
          `    const kept = path.len - 1;\n` +
          `    e.drafted += rows - 1;\n` +
          `    e.accepted += kept;      <span class="cm">// becomes acceptance_rate in the response</span>\n` +
          `}` }),
        prose(`
          <p>A round of <code class="inline">rows</code> verified positions commits <code class="inline">path.len</code>
          tokens: always at least one, at most <code class="inline">rows</code>. Row 0 is excluded from both
          counters because it is the already-known pending token.</p>`)),
      el('div', { class: 'col', style: { flex: '1 1 300px' } },
        note(`<b>Where the drafts come from.</b> <code class="inline">drafter.zig:9-72</code> defines the
              interface: <code class="inline">taps</code> (which target layers it reads),
              <code class="inline">absorb</code>, <code class="inline">hold</code>,
              <code class="inline">held</code>. A drafter may be a separate small model
              (<code class="inline">--drafter</code>) or an MTP head bolted onto the target that reuses its
              hidden states.`),
        note(`<b>It is observable over HTTP.</b> Replies carry a <code class="inline">speculative</code> block
              with the acceptance rate, and <code class="inline">token_sha256</code> exists so a drafted reply
              and a <code class="inline">"draft": false</code> reply can be proven to have produced the identical
              token sequence.`),
        note(`<b>Rounds leak into the API.</b> The server's mailbox records chunk boundaries
              (<code class="inline">m.chunks</code>) because a round can deliver several tokens at once — which
              is why streamed output sometimes arrives in visible bursts rather than evenly.`)))));

    refresh();
    return () => stops.forEach((s) => s());
  },
};
