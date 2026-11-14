// 08-kernels.js — level 2: the same multiply, mapped onto a GPU.

import { Plot, color, div } from '../plot.js';
import { card, el, prose, note, eq, code, btns, button, row, grow, slider, select, stats, legend, loop, fmt } from '../ui.js';
import { Mat, matmul, mulberry32 } from '../engine/tensor.js';

const R = 6, K = 4, N = 5;
const THREAD_COLORS = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'];

function makeMatrices() {
  const rng = mulberry32(0x08c0ffee);
  const ints = (rows, cols) => Mat.from(Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => Math.floor(rng() * 5) - 2)));
  return { A: ints(R, K), B: ints(K, N) };
}

function matrixValues(m) { return m.toArray(); }
function dotAt(A, B, i, j, upto) {
  let s = 0;
  for (let k = 0; k < upto; k++) s += A.get(i, k) * B.get(k, j);
  return s;
}
function operationState(step) {
  const total = R * N * K;
  const done = Math.min(Math.floor(step / K), R * N);
  const cell = Math.min(done, R * N - 1);
  return { total, done, i: Math.floor(cell / N), j: cell % N, upto: step === total ? K : step % K };
}
function fmtCell(v) { return Number.isInteger(v) ? String(v) : v.toFixed(2); }

function matrixPlot(width, height, xlim, ylim) {
  return new Plot({ width, height, xlim, ylim, pad: { l: 38, r: 10, t: 30, b: 30 } });
}

function drawHighlight(p, rows, cols, i, j, kind) {
  if (kind === 'row') {
    const rowY = rows - i - 1;
    p.rect(0, rowY, cols, rowY + 1, { color: 'accent', width: 2, fill: false, radius: 1 });
  }
  if (kind === 'col') {
    p.rect(j, 0, j + 1, rows, { color: 'accent', width: 2, fill: false, radius: 1 });
  }
}

function shapePlot(title, rowsX, colsX, rowsW, colsW, rowsY, colsY) {
  const p = new Plot({ width: 420, height: 230, xlim: [0, 12], ylim: [0, 6], pad: { l: 8, r: 8, t: 10, b: 10 }, plain: true });
  p.onDraw((q) => {
    const boxes = [
      { x: 0.5, y: 3.2, w: 2.6, h: 1.6, label: `${rowsX} × ${colsX}`, sub: 'activations X', c: 'c3' },
      { x: 4.6, y: 3.2, w: 2.6, h: 1.6, label: `${rowsW} × ${colsW}`, sub: 'weights W', c: 'c4' },
      { x: 8.8, y: 3.2, w: 2.6, h: 1.6, label: `${rowsY} × ${colsY}`, sub: 'output Y', c: 'c1' },
    ];
    for (const b of boxes) {
      q.rect(b.x, b.y, b.x + b.w, b.y + b.h, { color: b.c, fill: true, alpha: 0.2, radius: 4 });
      q.rect(b.x, b.y, b.x + b.w, b.y + b.h, { color: b.c, fill: false, radius: 4 });
      q.text(b.x + b.w / 2, b.y + 1.03, b.label, { color: b.c, size: 15, weight: 'bold' });
      q.text(b.x + b.w / 2, b.y + 0.35, b.sub, { color: 'muted', size: 10 });
    }
    q.arrow(3.25, 4, 4.45, 4, { color: 'muted', width: 1.5 });
    q.arrow(7.35, 4, 8.65, 4, { color: 'muted', width: 1.5 });
    q.text(6, 2.4, title, { color: 'fg', size: 12, weight: 'bold' });
    q.text(6, 1.35, `${rowsX} × ${colsX}  @  ${rowsW} × ${colsW}  →  ${rowsY} × ${colsY}`, { color: 'muted', size: 11 });
  });
  return p;
}

export default {
  id: '08-kernels',
  num: '08',
  group: 'the metal',
  title: 'The kernel: where the multiply runs',
  blurb: 'A matrix multiply is simple enough to write in three loops. The hard part is making the same weights travel through memory only once. Watch the loops become threads, tiles, and a bandwidth ceiling.',

  render(root) {
    const { A, B } = makeMatrices();
    const C = matmul(A, B);
    const state = { step: 0, playing: false };
    let animationStop = null;

    const aPlot = matrixPlot(220, 250, [0, K], [0, R]);
    const bPlot = matrixPlot(220, 250, [0, N], [0, K]);
    const cPlot = matrixPlot(220, 250, [0, N], [0, R]);
    const readout = stats([
      { k: 'output cell', v: 'C[0,0]' },
      { k: 'terms included', v: '0 / 4' },
      { k: 'running sum', v: '0' },
      { k: 'final answer', v: '—' },
    ]);
    const equation = el('div', { class: 'eq' });
    const stepLabel = el('div', { class: 'note' });

    function renderMatmul() {
      const s = operationState(state.step);
      const shown = Array.from({ length: R }, (_, i) => Array.from({ length: N }, (_, j) => {
        const index = i * N + j;
        if (index < s.done) return C.get(i, j);
        if (index === s.done) return dotAt(A, B, i, j, s.upto);
        return 0;
      }));
      const av = matrixValues(A), bv = matrixValues(B);
      aPlot.onDraw((p) => {
        p.matrix(av, { x0: 0, x1: K, y0: 0, y1: R, cmap: div, text: true, labels: { rows: ['r0','r1','r2','r3','r4','r5'], cols: ['k0','k1','k2','k3'] }, max: 2 });
        drawHighlight(p, R, K, s.i, s.j, 'row');
      });
      bPlot.onDraw((p) => {
        p.matrix(bv, { x0: 0, x1: N, y0: 0, y1: K, cmap: div, text: true, labels: { rows: ['k0','k1','k2','k3'], cols: ['n0','n1','n2','n3','n4'] }, max: 2 });
        drawHighlight(p, K, N, s.i, s.j, 'col');
      });
      cPlot.onDraw((p) => {
        p.matrix(shown, { x0: 0, x1: N, y0: 0, y1: R, cmap: div, text: true, labels: { rows: ['r0','r1','r2','r3','r4','r5'], cols: ['n0','n1','n2','n3','n4'] }, max: 16 });
        p.rect(s.j, R - s.i - 1, s.j + 1, R - s.i, { color: 'accent', width: 2.5, fill: false, radius: 1 });
      });
      const terms = Array.from({ length: s.upto }, (_, k) => `${fmtCell(A.get(s.i, k))}·${fmtCell(B.get(k, s.j))}`).join(' + ');
      const sum = dotAt(A, B, s.i, s.j, s.upto);
      equation.innerHTML = s.upto === 0
        ? `C[${s.i},${s.j}] = <span class="hl">0</span>  &nbsp; waiting for the first multiply`
        : `C[${s.i},${s.j}] = ${terms} = <span class="hl">${fmtCell(sum)}</span>`;
      stepLabel.innerHTML = `<b>Step ${state.step} / ${s.total}</b> · row <b>${s.i}</b> of A meets column <b>${s.j}</b> of B · add term ${Math.min(s.upto + 1, K)} of ${K}`;
      readout.set('output cell', `C[${s.i},${s.j}]`);
      readout.set('terms included', `${s.upto} / ${K}`);
      readout.set('running sum', fmtCell(sum));
      readout.set('final answer', state.step >= s.total ? fmtCell(C.get(s.i, s.j)) : '—');
      aPlot.render(); bPlot.render(); cPlot.render();
    }

    const back = button('← previous', () => { state.step = Math.max(0, state.step - 1); renderMatmul(); }, { ghost: true });
    const next = button('next multiply →', () => { state.step = Math.min(R * N * K, state.step + 1); renderMatmul(); });
    const play = button('▶ play', () => {
      if (state.playing) { state.playing = false; if (animationStop) animationStop(); animationStop = null; play.textContent = '▶ play'; return; }
      state.playing = true; play.textContent = 'Ⅱ pause';
      animationStop = loop((dt) => {
        if (dt < 90) return;
        state.step++;
        if (state.step >= R * N * K) { state.step = R * N * K; state.playing = false; if (animationStop) animationStop(); animationStop = null; play.textContent = '▶ play'; }
        renderMatmul();
      }, play);
    });
    const reset = button('reset', () => { state.step = 0; renderMatmul(); }, { ghost: true });

    root.append(card('One multiply, one cell at a time',
      'A is 6 × 4, B is 4 × 5, so C is 6 × 5. Step through the exact scalar operations: the highlighted row and column meet, and one product joins the running sum.',
      row(
        el('div', { class: 'col', style: { flex: '1 1 220px' } }, el('div', { class: 'note', text: 'A · active row' }), aPlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 220px' } }, el('div', { class: 'note', text: 'B · active column' }), bPlot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 220px' } }, el('div', { class: 'note', text: 'C · accumulated output' }), cPlot.canvas),
        grow(stepLabel, equation, btns(back, play, next, reset), readout)
      ),
      note('The CPU-shaped algorithm is three loops: choose an output cell, walk its shared K axis, and accumulate. A GPU kernel does not change that answer; it changes who owns each cell and where the data waits between products.'),
    ));

    const threadPlot = matrixPlot(470, 320, [0, N], [0, R]);
    const owners = Array.from({ length: R }, () => Array(N).fill(0));
    for (let lane = 0; lane < 32; lane++) {
      const qid = Math.floor(lane / 4);
      const fm = (qid & 4) + (Math.floor(lane / 2) % 4);
      const fn = (qid & 2) * 2 + (lane % 2) * 2;
      for (const rr of [fn, fn + 1]) if (rr < R && fm < N) owners[rr][fm] = qid;
    }
    threadPlot.onDraw((p) => {
      p.matrix(owners, { x0: 0, x1: N, y0: 0, y1: R, max: 7, gap: 2,
        cmap: (v) => color(THREAD_COLORS[Math.min(THREAD_COLORS.length - 1, Math.max(0, Math.round(v * 7 / 7))) ]),
        text: true, fmtv: (v) => `q${v}`, labels: { rows: ['r0','r1','r2','r3','r4','r5'], cols: ['n0','n1','n2','n3','n4'] } });
      p.text(0, -0.35, 'each colour = a lane quartet (lane / 4)', { color: 'muted', size: 10, px: false, align: 'left' });
    });

    root.append(card('Threads do not multiply the whole matrix',
      'The output grid is the contract. The mapping is the implementation: lanes are assigned pieces of that grid, while a threadgroup handles a tile so loaded values can be reused.',
      row(
        el('div', { class: 'col', style: { flex: '1 1 460px' } }, threadPlot.canvas, legend(THREAD_COLORS.map((c, i) => ({ c, label: `q${i} = lanes ${i * 4}–${i * 4 + 3}` })))),
        grow(
          prose(`<p>For the prompt-shaped kernel, TensorFold's mapped <code class="inline">simd_qmm</code> decomposition gives a threadgroup an <b>8·RT × 8·NT</b> output tile. Inside the simdgroup, lane <i>l</i> maps to a weight row (the output column) and a row pair (two output rows). The colours above show that ownership for our cropped 6 × 5 toy tile.</p>
          <p>That is tiling: load a block of X and a block of W into fast threadgroup/register storage, then reuse each loaded value across the products it participates in. A naive cell-at-a-time implementation would ask memory for the same values again and again.</p>`),
          note(`<b>Map, verbatim:</b> “<code>tf_sq_tiles&lt;S, NT, RT, H&gt;</code> computes <code>OUT[R, N] = X[R, K] · W[N, K]^T</code> … lane <code>l</code> maps to the weight-row / output-column index inside an 8×8 tile.”<br><span class="muted">Source recorded in <code>notes/map-weights.md</code>, §14, anchor <code>zig/kernels/metal/core/simd_qmm.metal:46-68</code>.</span>`),
          note(`<b>Decode reverses the decomposition:</b> the mapped <code>tile_qmv</code> kernel has no simdgroup matrices. “Lane <code>l</code> handles K-group <code>g0 + l</code>” (32 groups per round), writes each lane's partials to threadgroup memory, then a lane folds one output column's groups in order.<br><span class="muted">Source recorded in <code>notes/map-weights.md</code>, §15, anchor <code>zig/kernels/metal/core/tile_qmv.metal:11-18,181-207</code>.</span>`),
        )
      )
    ));

    const promptShape = shapePlot('prefill: many rows share W', 8, K, K, N, 8, N);
    const decodeShape = shapePlot('decode: one row streams through W', 1, K, K, N, 1, N);
    root.append(card('Prompt and decode are different traffic patterns',
      'The algebra is still X · Wᵀ. The shape changes what the memory system can reuse: a prompt brings many token rows at once; decode brings one new row.',
      row(
        grow(promptShape.canvas, prose(`<p><b>Prompt / prefill:</b> matrix × matrix. The same weight block contributes to many token rows, so arithmetic accumulates around each load.</p>`)),
        grow(decodeShape.canvas, prose(`<p><b>Decode:</b> matrix × vector (one row). Each weight participates in one dot product for this token, so the step is dominated by streaming the weight bytes.</p>`)),
      ),
      note(`<b>TensorFold's two mapped kernels:</b> the map's reference table names <code>simd_qmm</code> as “quantised matmul, prompt” and <code>tile_qmv.metal</code> as “quantised matmul, decode”. Same mathematics; deliberately different parallel decomposition.`),
    ));

    const calc = { params: 4, bits: 4, bw: 400, peak: 10.4 };
    const intensityRead = stats([
      { k: 'FLOPs / step', v: '—' }, { k: 'weight bytes / step', v: '—' },
      { k: 'arithmetic intensity', v: '—' }, { k: 'machine roof', v: '—' },
      { k: 'predicted ceiling', v: '—' },
    ]);
    const calcText = el('div', { class: 'note' });
    const scenario = el('div', { class: 'prose' });
    const paramsSlider = slider({ label: 'parameters (billions)', min: 1, max: 70, step: 0.5, value: calc.params, fmt: (v) => `${v.toFixed(1)}B`, onInput: (v) => { calc.params = v; refreshCalc(); } });
    const bitsSelect = select({ label: 'bits per weight', options: [{ v: 2, t: '2-bit' }, { v: 4, t: '4-bit' }, { v: 8, t: '8-bit' }, { v: 16, t: '16-bit (bf16)' }], value: 4, onChange: (v) => { calc.bits = Number(v); refreshCalc(); } });
    const bwSlider = slider({ label: 'memory bandwidth (GB/s)', min: 100, max: 800, step: 10, value: calc.bw, fmt: (v) => `${v.toFixed(0)} GB/s`, onInput: (v) => { calc.bw = v; refreshCalc(); } });
    const peakSlider = slider({ label: 'compute peak (TFLOP/s)', min: 1, max: 30, step: 0.1, value: calc.peak, fmt: (v) => `${v.toFixed(1)} TF`, onInput: (v) => { calc.peak = v; refreshCalc(); } });

    function refreshCalc() {
      const params = calc.params * 1e9;
      const flops = 2 * params;
      const bytes = params * calc.bits / 8;
      const intensity = flops / bytes;
      const roof = calc.peak * 1000 / calc.bw;
      const tok = calc.bw * 1e9 / bytes;
      intensityRead.set('FLOPs / step', `${fmt.n(flops / 1e12, 2)} TF`);
      intensityRead.set('weight bytes / step', `${fmt.n(bytes / 1e9, 2)} GB`);
      intensityRead.set('arithmetic intensity', `${fmt.n(intensity, 2)} FLOP/byte`);
      intensityRead.set('machine roof', `${fmt.n(roof, 1)} FLOP/byte`);
      intensityRead.set('predicted ceiling', `${fmt.n(tok, 1)} tokens/s`);
      const bound = intensity < roof ? 'memory-bound' : 'compute-bound';
      calcText.innerHTML = `<b>${bound}</b>: ${fmt.n(intensity, 2)} FLOP/byte available versus a ${fmt.n(roof, 1)} FLOP/byte machine roof. Ideal streaming time is <code class="inline">bytes ÷ bandwidth</code>, so this predicts ${fmt.n(tok, 1)} tokens/s.<br><span class="muted">Idealized lower bound: weight bytes only; scales, biases, activations, and launch overhead are omitted.</span>`;
      scenario.innerHTML = `<p><b>Illustrative ceilings</b> at ${calc.bw.toFixed(0)} GB/s, using the selected ${calc.bits}-bit weight size (not benchmarks):</p><table><tbody>${[2,4,8,14,27].map((b) => `<tr><td>${b}B parameters</td><td>${fmt.n(calc.bw * 8 / (b * calc.bits), 1)} tokens/s</td><td>${fmt.n(b * calc.bits / 8, 2)} GB read</td></tr>`).join('')}</tbody></table>`;
      refreshBatch();
    }

    root.append(card('Arithmetic intensity: count the bytes',
      'For one decode step, FLOPs = 2 × weights. Bytes read = weights × bytes per weight. The ratio tells you whether faster arithmetic can matter at all.',
      row(
        grow(paramsSlider, bitsSelect, bwSlider, peakSlider, intensityRead),
        grow(calcText, scenario, prose(`<p>At the default 400 GB/s, the bandwidth is the important number. That is the Apple-published memory-bandwidth figure for the M1 Max class of chip, not a measurement from this page. The compute slider is an explicit “what if” roofline, so you can see the comparison rather than take “GPU” as a magic speed label.</p>`)),
      ),
      eq('decode time ≈ <span class="hl">weight bytes</span> ÷ bandwidth  &nbsp; · &nbsp; tokens/s ≈ bandwidth ÷ weight bytes'),
      note(`<b>Repository evidence (quoted by the map):</b><br><code>“Decode: 106 tokens/s with one stream (9.5 ms a step: 7.7 GiB of weights a step, the memory bandwidth)”</code> — <code>docs/recipes/spark-x2.6-4b-metal.md:62-64</code>.<br><code>“A GB10 is bandwidth-bound for most decode work.”</code> — <code>docs/optimizing.md:102-106</code>.<br><code>“Qwen3.8-27B (16 GiB): 40 tokens/s plain, at the card's memory bandwidth”</code> — <code>docs/WINDOWS.md:117</code>.<br>Those are the same equation as the calculator: bytes per step, divided by sustained bandwidth. They are measurements of particular systems; this page's table is only a ceiling.`),
    ));

    const batch = { n: 1 };
    const batchRead = stats([
      { k: 'batch arithmetic intensity', v: '—' }, { k: 'weight read', v: '—' },
      { k: 'round time', v: '—' }, { k: 'aggregate ceiling', v: '—' }, { k: 'per-request wait', v: '—' },
    ]);
    const batchSlider = slider({ label: 'concurrent rows', min: 1, max: 32, step: 1, value: 1, fmt: (v) => String(v), onInput: (v) => { batch.n = v; refreshBatch(); } });
    const batchPlot = new Plot({ width: 500, height: 270, xlim: [1, 32], ylim: [0, 1], pad: { l: 48, r: 18, t: 25, b: 40 }, xlabel: 'rows in one round', ylabel: 'relative value' });
    batchPlot.onDraw((p) => {
      const params = calc.params * 1e9, bytes = params * calc.bits / 8;
      const bwTime = bytes / (calc.bw * 1e9);
      const roof = calc.peak * 1e12;
      const xs = Array.from({ length: 32 }, (_, i) => i + 1);
      const times = xs.map((b) => Math.max(bwTime, (2 * params * b) / roof));
      const through = xs.map((b, i) => [b, (b / times[i]) / (32 / Math.max(...xs.map((x, j) => x / times[j])))]);
      const lat = xs.map((b, i) => [b, times[i] / Math.max(...times)]);
      p.axes({ xticks: 7, yticks: 4, zero: true, grid: true });
      p.line(through, { color: 'c3', width: 2.5 });
      p.line(lat, { color: 'c5', width: 2.5, dash: [5, 4] });
      const bi = batch.n - 1;
      p.circle(batch.n, through[bi][1], 4, { color: 'c3' });
      p.circle(batch.n, lat[bi][1], 4, { color: 'c5' });
      p.text(28, through[27][1] + 0.08, 'aggregate tokens/s', { color: 'c3', size: 10, align: 'right' });
      p.text(28, lat[27][1] - 0.08, 'per-request wait', { color: 'c5', size: 10, align: 'right' });
    });
    function refreshBatch() {
      const params = calc.params * 1e9, bytes = params * calc.bits / 8;
      const intensity = (2 * params * batch.n) / bytes;
      const t = Math.max(bytes / (calc.bw * 1e9), (2 * params * batch.n) / (calc.peak * 1e12));
      batchRead.set('batch arithmetic intensity', `${fmt.n(intensity, 2)} FLOP/byte`);
      batchRead.set('weight read', `${fmt.n(bytes / 1e9, 2)} GB once`);
      batchRead.set('round time', `${fmt.n(t * 1000, 2)} ms`);
      batchRead.set('aggregate ceiling', `${fmt.n(batch.n / t, 1)} tokens/s`);
      batchRead.set('per-request wait', `${fmt.n(t * 1000, 2)} ms (shared round)`);
      batchPlot.render();
    }

    root.append(card('Batching: more arithmetic, the same weight read',
      'Eight concurrent requests do eight dot products, but the threadgroup can reuse the loaded weight tile across rows. The arithmetic intensity rises with batch size; the memory traffic does not.',
      row(
        el('div', { class: 'col', style: { flex: '1 1 500px' } }, batchPlot.canvas, legend([{ c: 'c3', label: 'aggregate throughput' }, { c: 'c5', label: 'per-request wait' }])),
        grow(batchSlider, batchRead, prose(`<p>Move the slider. In the memory-bound region, aggregate throughput rises almost linearly while each request still waits for the shared round. Once arithmetic crosses the machine roof, throughput flattens and latency grows faster.</p><p>This is why batching improves hardware utilisation but is not free for interactive latency.</p>`)),
      ),
      note(`<b>Map, verbatim:</b> “The weight row <code>w</code> is loaded once and reused across all <code>ROWS</code> — that is the entire reason multi-row (batched) decode is nearly free: the <code>r</code> loop adds FLOPs but no weight bytes.” The mapped dense reference is <code>zig/kernels/metal/core/bf16_matvec.metal:7-31</code>.`),
    ));

    root.append(card('The kernel is a data-movement decision', null,
      row(
        grow(code(`for (output row r, output col n) {
  acc = 0;
  for (k = 0; k < K; ++k)
    acc += X[r,k] * W[n,k];
  Y[r,n] = acc;
}`), prose(`<p>The answer is just a dot product. Performance comes from deciding which loop runs across lanes, which values stay in registers, and how many rows share one weight load.</p>`)),
        grow(note(`<b>Prompt:</b> tile the output and reuse X/W blocks.<br><b>Decode:</b> stream the row through the same weights, then reuse them across concurrent rows when available.<br><b>Measure:</b> count bytes first. If the roofline says memory-bound, a cleverer multiply cannot outrun the read.`), eq('same math · different shape · different bottleneck')),
      )
    ));

    refreshCalc();
    renderMatmul();
    return () => { if (animationStop) animationStop(); aPlot.destroy(); bPlot.destroy(); cPlot.destroy(); threadPlot.destroy(); promptShape.destroy(); decodeShape.destroy(); batchPlot.destroy(); };
  },
};
