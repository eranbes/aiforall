// 10-file.js — level 0: a model is a header and a byte range.

import { card, el, prose, note, btns, button, row, grow, slider, select, stats, textbox, fmt } from '../ui.js';

const DTYPE_BYTES = { F32: 4, BF16: 2, F16: 2, U8: 1 };
const REGION_COLORS = { prefix: 'c2', header: 'c3', data: 'c1' };
const enc = new TextEncoder();
const dec = new TextDecoder();

function product(shape) {
  return shape.reduce((n, d) => n * d, 1);
}

function makeFile(specs) {
  const header = {};
  const chunks = [];
  let cursor = 0;
  for (let s = 0; s < specs.length; s++) {
    const spec = specs[s];
    const n = product(spec.shape) * DTYPE_BYTES[spec.dtype];
    header[spec.name] = {
      dtype: spec.dtype,
      shape: [...spec.shape],
      data_offsets: [cursor, cursor + n],
    };
    const bytes = new Uint8Array(n);
    for (let i = 0; i < n; i++) bytes[i] = (17 * (s + 1) + i) & 255;
    chunks.push(bytes);
    cursor += n;
  }
  const json = enc.encode(JSON.stringify(header));
  const file = new Uint8Array(8 + json.length + cursor);
  new DataView(file.buffer).setBigUint64(0, BigInt(json.length), true);
  file.set(json, 8);
  let at = 8 + json.length;
  for (const chunk of chunks) { file.set(chunk, at); at += chunk.length; }
  return file;
}

// This is deliberately the same small sequence a loader needs: length, JSON, slice.
function parseFile(file) {
  if (file.length < 8) throw new Error('file is shorter than its length prefix');
  const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
  const headerLength = Number(view.getBigUint64(0, true));
  if (headerLength > 100_000_000 || headerLength > file.length - 8) throw new Error('bad header length');
  const header = JSON.parse(dec.decode(file.subarray(8, 8 + headerLength)));
  const entries = Object.entries(header).map(([name, e]) => ({
    name, dtype: e.dtype, shape: [...e.shape], begin: e.data_offsets[0], end: e.data_offsets[1],
  }));
  return { headerLength, header, entries, dataStart: 8 + headerLength, dataLen: file.length - 8 - headerLength };
}

function expectedBytes(entry) {
  return product(entry.shape) * (DTYPE_BYTES[entry.dtype] || 0);
}

function validate(parsed, corruption) {
  const entries = parsed.entries.map((e) => ({ ...e, shape: [...e.shape] }));
  let dataLen = parsed.dataLen;
  if (corruption === 'shape' && entries[0]) entries[0].shape[0] += 1;
  if (corruption === 'overlap' && entries[1] && entries[0]) {
    const n = expectedBytes(entries[1]);
    entries[1].begin = entries[0].begin;
    entries[1].end = entries[0].begin + n;
  }
  if (corruption === 'truncated') dataLen = Math.max(0, dataLen - 1);
  if (corruption === 'rank' && entries[0]) entries[0].shape = [1, 1, 1, 1, 1, 1];

  const size = entries.every((e) => expectedBytes(e) === e.end - e.begin);
  const bounds = entries.every((e) => e.end >= e.begin && e.end <= dataLen);
  const rank = entries.every((e) => e.shape.length <= 5);
  const sorted = entries.every((e, i) => i === 0 || entries[i - 1].begin <= e.begin);
  let previousEnd = 0;
  let overlap = false;
  for (const e of [...entries].sort((a, b) => a.begin - b.begin)) {
    if (e.begin !== e.end && e.begin < previousEnd) overlap = true;
    previousEnd = Math.max(previousEnd, e.end);
  }
  return [
    ['shape × dtype = end − begin', size, 'the redundant byte-count checksum'],
    ['end ≤ data_len', bounds, `data region is ${dataLen} bytes`],
    ['rank ≤ 5', rank, 'the loader admits at most five dimensions'],
    ['sorted / no overlap', sorted && !overlap, 'half-open ranges must not collide'],
  ];
}

function f32Bits(value) {
  const b = new ArrayBuffer(4);
  const d = new DataView(b);
  d.setFloat32(0, Math.fround(value), false);
  return d.getUint32(0, false) >>> 0;
}

function bitsF32(bits) {
  const b = new ArrayBuffer(4);
  const d = new DataView(b);
  d.setUint32(0, bits >>> 0, false);
  return d.getFloat32(0, false);
}

function f16Bits(value) {
  const bits = f32Bits(value);
  const sign = (bits >>> 16) & 0x8000;
  const abs = bits & 0x7fffffff;
  if (abs >= 0x7f800000) return sign | (abs > 0x7f800000 ? 0x7e00 : 0x7c00);
  if (abs > 0x477fefff) return sign | 0x7c00;
  if (abs < 0x33000000) return sign;
  let exponent = ((abs >>> 23) & 255) - 127 + 15;
  let fraction = abs & 0x7fffff;
  if (exponent <= 0) {
    fraction |= 0x800000;
    const shift = 14 - exponent;
    return sign | ((fraction + (1 << (shift - 1))) >>> shift);
  }
  fraction = (fraction + 0x1000) >>> 13;
  if (fraction === 0x400) { exponent++; fraction = 0; }
  return sign | (exponent << 10) | fraction;
}

function f16Value(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >>> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export default {
  id: '10-file',
  num: '10',
  group: 'the metal',
  title: 'The model is a file',
  blurb: 'At the bottom there is no hidden object: an eight-byte length, JSON, and raw tensor bytes. Build one, break its contract, and inspect the bits yourself.',

  render(root) {
    const state = {
      specs: [
        { name: 'embedding.weight', dtype: 'F32', shape: [2, 4] },
        { name: 'lm_head.weight', dtype: 'BF16', shape: [3, 2] },
      ],
      corruption: 'none',
      parsed: null,
      file: null,
    };

    const headerPre = el('pre', { class: 'code' });
    const parserCode = el('pre', { class: 'code', text:
      `const header_len = view.getBigUint64(0, true);
const header = JSON.parse(new TextDecoder().decode(bytes.slice(8, 8 + Number(header_len))));
const data = bytes.slice(8 + Number(header_len));
const bytesFor = e => e.shape.reduce((n, d) => n * d, 1) * dtypeSize[e.dtype];
// e.data_offsets is relative to data, never to the file start.` });
    const offsets = el('div', { class: 'code', style: { whiteSpace: 'pre-wrap' } });
    const dump = el('div', { style: { fontFamily: 'ui-monospace, Menlo, monospace', fontSize: '0.78rem', lineHeight: '1.8', overflowWrap: 'anywhere' } });
    const regionRead = el('div', { class: 'note', text: 'Click a coloured byte to identify its region.' });
    const validator = el('div', { class: 'stats' });
    const rebuild = () => {
      state.file = makeFile(state.specs);
      state.parsed = parseFile(state.file);
      headerPre.textContent = JSON.stringify(state.parsed.header, null, 2);
      offsets.textContent = state.parsed.entries.map((e) => `${e.name}: [${e.begin}, ${e.end})  ${e.end - e.begin} bytes`).join('\n');
      dump.replaceChildren();
      for (let i = 0; i < state.file.length; i++) {
        const region = i < 8 ? 'prefix' : i < state.parsed.dataStart ? 'header' : 'data';
        const byte = state.file[i].toString(16).padStart(2, '0');
        dump.append(el('button', {
          class: 'ghost', text: byte, title: `${region}: byte ${i}`,
          style: { color: `var(--${REGION_COLORS[region]})`, padding: '0 0.25rem', minWidth: '2.3rem', fontFamily: 'inherit' },
          onClick: () => {
            regionRead.innerHTML = region === 'prefix'
              ? '<b>8-byte prefix.</b> Little-endian u64 header length; it says how many JSON bytes follow.'
              : region === 'header'
                ? '<b>JSON region.</b> Names, dtypes, shapes, and offsets are the table of contents.'
                : '<b>data region.</b> The tensor payload. Its offsets start at zero <i>inside this region</i>, not at file byte zero.';
          },
        }));
        if ((i + 1) % 16 === 0) dump.append(el('br'));
      }
      renderValidation();
    };

    const tensorCtl = slider({
      label: 'embedding rows', min: 1, max: 8, step: 1, value: 2, fmt: String,
      onInput: (v) => { state.specs[0].shape[0] = v; rebuild(); },
    });
    const addTensor = button('add a tensor', () => {
      if (!state.specs.some((s) => s.name === 'scratch')) state.specs.push({ name: 'scratch', dtype: 'F16', shape: [2, 2] });
      rebuild();
    }, { ghost: true });
    const removeTensor = button('remove it', () => {
      state.specs = state.specs.filter((s) => s.name !== 'scratch');
      rebuild();
    }, { ghost: true });
    const corruptionCtl = select({
      label: 'corrupt the parsed header/data', value: 'none',
      options: [
        { v: 'none', t: 'none — valid file' },
        { v: 'shape', t: 'wrong shape' },
        { v: 'overlap', t: 'overlapping ranges' },
        { v: 'truncated', t: 'truncated data' },
        { v: 'rank', t: 'rank > 5' },
      ],
      onChange: (v) => { state.corruption = v; renderValidation(); },
    });
    const renderValidation = () => {
      validator.replaceChildren();
      for (const [label, pass, detail] of validate(state.parsed, state.corruption)) {
        const value = el('div', { class: 'v', text: `${pass ? 'PASS' : 'FAIL'} · ${detail}`, style: { color: `var(--${pass ? 'ok' : 'err'})` } });
        validator.append(el('div', { class: 'stat' }, el('div', { class: 'k', text: label }), value));
      }
    };

    root.append(card('A real safetensors file, made here',
      'The format from <code class="inline">zig/src/core/safetensors.zig:8-11</code> is exactly: little-endian u64 header length → that many JSON bytes → data. The parser above is not a sketch: it reads the bytes we just built back into entries.',
      row(grow(
        prose('<p>Change a shape or add a tensor. The JSON gets a new length, and every later range moves. The ranges are <code class="inline">[begin, end)</code> relative to the first data byte, which is file offset <code class="inline">8 + header_len</code>.</p>'),
        row(tensorCtl, btns(addTensor, removeTensor))),
      grow(
        el('div', { class: 'lbl', text: 'generated header (pretty-printed)' }), headerPre,
        el('div', { class: 'lbl', text: 'parsed ranges' }), offsets,
      ),
      parserCode,
      note('The visible parser uses the same two slices as the builder: first JSON, then data. A tensor lookup only needs its name and these two offsets.'),
    )));

    root.append(card('See every byte',
      'The three regions are colour-coded. Click any byte to make the boundary concrete; the eight prefix bytes are part of the file but not part of any tensor range.',
      el('div', { class: 'legend' },
        el('span', {}, el('i', { style: { background: 'var(--c2)' } }), 'length prefix'),
        el('span', {}, el('i', { style: { background: 'var(--c3)' } }), 'JSON header'),
        el('span', {}, el('i', { style: { background: 'var(--c1)' } }), 'tensor data')),
      dump, regionRead,
      note('The real loader caps the header at 100,000,000 bytes and checks the file is at least 8 bytes before subtracting 8. That order prevents a corrupt length from wrapping or causing an oversized allocation.'),
    ));

    root.append(card('The header is a contract, not a suggestion',
      'TensorFold checks the redundant facts in <code class="inline">zig/src/core/safetensors.zig:110-135</code>. Corrupt one fact and the named invariant catches it before a kernel reads arbitrary bytes.',
      corruptionCtl, validator,
      note('Wrong shape breaks the byte-count checksum. Overlap keeps each individual size correct but makes two tensors claim the same bytes. Truncation leaves the header unchanged while the data region becomes too short. Rank is capped at 5. These checks are why a plain JSON table is safe to trust.'),
    ));

    root.append(card('mmap: opening 70 GB without copying it',
      'TensorFold’s <code class="inline">File.openPrefixes</code> (<code class="inline">safetensors.zig:226-239</code>) maps read-only with <code class="inline">populate = false</code>. <code class="inline">File.get</code> (<code class="inline">:288-290</code>) then returns <code class="inline">map.memory[data + begin .. data + end]</code>.',
      row(
        el('div', { style: { flex: '1 1 300px', minWidth: '260px' } },
          el('div', { style: { display: 'flex', height: '3rem', alignItems: 'stretch', margin: '1rem 0' } },
            el('div', { style: { flex: '0 0 16%', background: 'var(--c2)', color: 'var(--surface)', padding: '0.5rem', fontSize: '0.75rem' }, text: 'u64' }),
            el('div', { style: { flex: '0 0 34%', background: 'var(--c3)', color: 'var(--surface)', padding: '0.5rem', fontSize: '0.75rem' }, text: 'JSON index' }),
            el('div', { style: { flex: '1', background: 'var(--c1)', color: 'var(--surface)', padding: '0.5rem', fontSize: '0.75rem' }, text: '70 GB of pages' })),
          stats([{ k: 'up front', v: 'map + parse JSON' }, { k: 'on first touch', v: 'page fault' }, { k: 'cache', v: 'OS page cache' }]),
        ),
        grow(prose('<p>Mapping creates virtual addresses, not a second 70 GB copy. The OS brings pages into RAM when a matmul first touches them; untouched pages stay out. The page cache is the model cache. A name lookup plus two offsets becomes a byte slice.</p><p>TensorFold’s map notes that MLX <code class="inline">upload</code> is a straight byte-for-byte copy (<code class="inline">mlx.zig:177-181</code>): after upload, GPU memory holds exactly the bytes represented by the file, just in another memory domain.</p>')),
      ),
    ));

    const floatInput = textbox({ label: 'fp32 value (try 70000)', value: '70000', rows: 1 });
    const bitStrip = el('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '0.15rem', margin: '0.5rem 0' } });
    const floatRead = stats([
      { k: 'Math.fround(value)', v: '—' }, { k: 'DataView bits', v: '—' },
      { k: 'bf16 after dropping 16 bits', v: '—' }, { k: 'fp16', v: '—' },
    ]);
    let currentBits = f32Bits(70000);
    const renderFloat = () => {
      const value = bitsF32(currentBits);
      bitStrip.replaceChildren();
      for (let i = 0; i < 32; i++) {
        const bit = (currentBits >>> (31 - i)) & 1;
        const colour = i === 0 ? 'c2' : i < 9 ? 'c3' : 'c1';
        bitStrip.append(el('button', {
          class: 'ghost', text: String(bit), title: `toggle bit ${31 - i}`,
          style: { color: `var(--${colour})`, borderColor: `var(--${colour})`, background: bit ? 'var(--surface-2)' : 'var(--surface)' },
          onClick: () => { currentBits = (currentBits ^ (2 ** (31 - i))) >>> 0; renderFloat(); },
        }));
        if (i === 0 || i === 8) bitStrip.append(el('span', { text: ' ' }));
      }
      const f32 = Math.fround(value);
      const bfBits = (currentBits >>> 16) & 0xffff;
      const bfValue = bitsF32((bfBits << 16) >>> 0);
      const f16 = f16Value(f16Bits(value));
      floatRead.set('Math.fround(value)', fmt.exp(f32, 6));
      floatRead.set('DataView bits', `0x${currentBits.toString(16).padStart(8, '0')}  ·  ${currentBits.toString(2).padStart(32, '0')}`);
      floatRead.set('bf16 after dropping 16 bits', `${fmt.exp(bfValue, 6)}  (0x${bfBits.toString(16).padStart(4, '0')})`);
      floatRead.set('fp16', `${fmt.exp(f16, 6)}  (different exponent width)`);
    };
    floatInput.querySelector('input')?.addEventListener('input', () => {
      const n = Number(floatInput.get());
      if (Number.isFinite(n) || n === Infinity) { currentBits = f32Bits(n); renderFloat(); }
    });
    renderFloat();

    root.append(card('Floating point is bits with a decoder',
      'Toggle any bit. Sign is the first bit, exponent is the next 8, and mantissa is the last 23. The displayed value is decoded with a <code class="inline">DataView</code>; the starting word is produced with <code class="inline">Math.fround</code>.',
      floatInput, bitStrip,
      el('div', { class: 'lbl', text: 'sign  |  exponent (8 bits)  |  mantissa (23 bits)' }),
      floatRead,
      row(
        grow(prose('<p><b>bf16 is fp32 with the bottom 16 bits removed.</b> TensorFold’s <code class="inline">bf16_round.zig:17-32</code> verifies the conversion as a 16-bit shift: same sign, same 8-bit exponent, only 7 mantissa bits remain. That is why the bf16 value above keeps fp32’s range while losing precision.</p>')),
        note('<b>Try 70000.</b> bf16 survives with a nearby finite value because its exponent is as wide as fp32. fp16 has only a 5-bit exponent, so it overflows to ∞ above 65504. The safetensors dtype set also admits <code class="inline">f8_e4m3</code>, <code class="inline">f8_e5m2</code>, and <code class="inline">f8_e8m0</code> (<code class="inline">safetensors.zig:13-58</code>).'),
      ),
    ));

    const floorSlider = slider({ label: 'model weights (GiB)', min: 0.25, max: 70, step: 0.25, value: 7, fmt: (v) => `${v.toFixed(2)} GiB`, onInput: updateFloor });
    const floorRead = stats([{ k: 'header byte ranges sum to', v: '—' }, { k: 'runtime margin', v: '2.00 GiB' }, { k: 'memory floor', v: '—' }]);
    function updateFloor() {
      const gib = floorSlider.get();
      floorRead.set('header byte ranges sum to', `${gib.toFixed(2)} GiB`);
      floorRead.set('memory floor', `${(gib + 2).toFixed(2)} GiB`, 'accent');
    }
    updateFloor();

    root.append(card('The memory floor is addition',
      'The <code class="inline">tensorfold info</code> floor comes from <code class="inline">info.zig:66-77</code> and <code class="inline">checkpoint_weights.zig:13-27</code>: sum every <code class="inline">end - begin</code> in the headers, then add the fixed <code class="inline">2 << 30</code> runtime margin from <code class="inline">platform/memory_policy.zig:2</code>. No weights are loaded to answer this question.',
      floorSlider, floorRead,
      prose('<p>Examples: a 1 GiB header total needs 3 GiB; 7 GiB needs 9 GiB; 70 GiB needs 72 GiB. The slider is doing the same operation as the CLI: the file’s JSON already says how many bytes every tensor occupies.</p>'),
    ));

    root.append(card('End of the descent', null,
      prose('<p><b>bytes → dequantised weights → matmuls → blocks → logits → softmax → sampler → token → text.</b> You have now reached the bytes at the bottom: a JSON index and ranges into a blob.</p><p>One thing is still unexplained: <i>where did these numbers come from?</i> That is training.</p>'),
      btns(button('back to the beginning: sampling', () => { window.location.hash = '#/01-sampling'; }), button('continue to training', () => { window.location.hash = '#/11-training'; }, { ghost: true })),
    ));

    rebuild();
    return () => {};
  },
};
