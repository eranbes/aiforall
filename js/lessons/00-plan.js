// 00-plan.js — the method: descend the stack instead of climbing it.

import { Plot, color, mix } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, checkbox, loop, stats } from '../ui.js';

/** The course, top of the stack to bottom. `ready` flips on as lessons land. */
export const STACK = [
  { n: 11, id: '01-sampling', label: 'A reply comes back', q: 'Text appears in your terminal. Where from?', anchor: 'HTTP route → SSE stream', ready: true },
  { n: 10, id: '01-sampling', label: 'One token at a time', q: 'The reply is not written. It is drawn, repeatedly.', anchor: 'decode loop', ready: true },
  { n: 9, id: '02-tokens', label: 'The pieces are tokens', q: 'Not words, not letters. Integers from a merge table.', anchor: 'byte-level BPE', ready: true },
  { n: 8, id: '01-sampling', label: 'A token is picked', q: 'Picked how? temperature, top-k, top-p, min-p.', anchor: 'sampler', ready: true },
  { n: 7, id: '03-softmax', label: 'From a probability vector', q: '248,320 numbers that sum to one. Where from?', anchor: 'softmax', ready: true },
  { n: 6, id: '04-logits', label: 'From logits', q: 'One vector times one big matrix. That is all.', anchor: '2048 × 248320 projection', ready: true },
  { n: 5, id: '05-blocks', label: 'From a stack of blocks', q: 'Each block rewrites the vector. 24 of them, here.', anchor: 'layer loop + residuals', ready: true },
  { n: 4, id: '06-attention', label: 'A block looks around', q: 'Attention: where context enters the number.', anchor: 'core/attention_ref.zig', ready: true },
  { n: 3, id: '07-mlp', label: 'A block thinks', q: 'The MLP: gate, up, activate, down.', anchor: 'feed-forward + MoE routing', ready: true },
  { n: 2, id: '08-kernels', label: 'All of it is matmul', q: 'Which thread multiplies which numbers?', anchor: 'metal/core/simd_qmm.metal', ready: true },
  { n: 1, id: '09-quant', label: 'On 4-bit weights', q: 'Eight weights per 32-bit word, unpacked mid-kernel.', anchor: 'quantised matmul', ready: true },
  { n: 0, id: '10-file', label: 'Which are numbers in a file', q: 'A JSON header, then raw bytes. Nothing more.', anchor: 'safetensors header', ready: true },
];

const CHAIN = [
  { a: 'Claude wrote me a function.', q: 'but what is a "reply"?' },
  { a: 'A sequence of tokens, emitted one by one.', q: 'but how is each token chosen?' },
  { a: 'Drawn at random from a probability distribution.', q: 'but where do the probabilities come from?' },
  { a: 'softmax over a vector of ~150,000 logits.', q: 'but where do the logits come from?' },
  { a: 'The last hidden state multiplied by the output matrix.', q: 'but where does the hidden state come from?' },
  { a: 'Dozens of transformer blocks, each adding to it.', q: 'but what does a block compute?' },
  { a: 'Attention over previous tokens, then an MLP.', q: 'but what is attention, mechanically?' },
  { a: 'Three matrix multiplies, a mask, a softmax, one more multiply.', q: 'but what is a matrix multiply, on this hardware?' },
  { a: 'Thousands of GPU threads each summing a few hundred products.', q: 'but what are they multiplying?' },
  { a: '4-bit integers and a scale, unpacked from a memory-mapped file.', q: 'and where did those come from?' },
  { a: 'Gradient descent. Which is the only part of the story that is backwards for real.', q: null },
];

export default {
  id: '00-plan',
  num: '00',
  group: 'orientation',
  title: 'Preface: why we go backwards',
  blurb: 'An interactive course on how language models actually work, taken apart from the top down.',

  render(root) {
    const stops = [];

    /* ---------------- start here ---------------- */
    const startCard = card('Start: the 10-minute tour',
      'This page (00) is a <b>preface</b> — it explains how the course is built, not how a model works. Start with the tour instead: five ideas, five buttons, and you will have the whole picture in outline.',
      row(
        grow(el('div', { class: 'col', style: { gap: '10px' } },
          ...[
            ['1', 'Take the <b>10-minute tour</b> first.', 'Five ideas, each with one button. At the end you will understand, in outline, how every language model works.'],
            ['2', 'Then follow only what you are curious about.', 'The tour ends with five links. Pick one. There is no obligation to read 23 lessons.'],
            ['3', 'Ignore the rest of the sidebar until it is useful.', 'Lessons 12–22 are reference depth. They are there for when a question sends you looking.'],
          ].map(([n, what, why]) => el('div', {
            style: {
              display: 'flex', gap: '12px', alignItems: 'baseline',
              padding: '10px 12px', borderRadius: '9px',
              background: 'var(--surface-2)', border: '1px solid var(--border)',
            },
          },
            el('span', {
              style: {
                font: '13px ui-monospace,Menlo,monospace', fontWeight: '700', color: 'var(--accent)',
                minWidth: '18px',
              },
            }, n),
            el('div', {},
              el('div', { style: { fontSize: '14px' }, html: what }),
              el('div', { style: { fontSize: '12.5px', color: 'var(--muted)', marginTop: '2px' }, text: why })))),
          el('div', { style: { height: '4px' } }),
          btns(
            button('take the 10-minute tour →', () => { location.hash = '#/00b-tour'; }),
            button('I have an engine running — show me the real thing', () => { location.hash = '#/18-live'; }, { ghost: true })))),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Do not read the sidebar as a syllabus.</b> There are 23 lessons because the subject is
                large, not because you must finish them. Lessons <b>01–11</b> are the course; everything after
                is optional depth you can take when a question sends you there.`),
          note(`<b>Nothing here is a video or a wall of text.</b> Every page has controls. If you are reading
                for more than a minute without dragging something, you are using it wrong.`))));
    startCard.classList.add('hero');
    root.append(startCard);

    root.append(card('Why backwards', null, row(
      grow(prose(`
        <p>Built bottom-up, every concept arrives as an <i>unmotivated definition</i>. You learn what a
        sigmoid is six weeks before anything needs one. Nothing you learn is load-bearing yet, so none of it sticks.</p>
        <p>Built top-down, every concept arrives as the <i>answer to a question you already have</i>.
        You watch a model emit a token, ask how it picked that one, and the sampler is the answer. The
        sampler reads a probability vector, so you ask where that comes from, and softmax is the answer.
        Each layer is motivated by the one above it.</p>
        <p>The risk of top-down is hand-waving: "a transformer predicts the next token" explains nothing.
        We avoid it with a rule.</p>
      `)),
      el('div', { class: 'col', style: { flex: '1 1 300px' } },
        note(`<b>The rule.</b> Every answer must bottom out in code you can open, or a number you can
              change and watch move. No metaphors that cannot be cashed out.`),
        note(`<b>The anchor.</b> Every claim is checked against <code class="inline">TensorFold</code>,
              a real LLM inference engine in Zig (980k lines, Metal + CUDA kernels), cloned next to this
              page. When a lesson says "the engine does X", there is a file and a line number.`),
        note(`<b>The one exception.</b> Training really does run backwards — the chain rule flows from
              loss to weights. We reach it last, when "backwards" has become literal.`)))));

    /* ---------------- the descent diagram ---------------- */
    const W = 680, H = 520;
    const plot = new Plot({ width: W, height: H, pad: { l: 0, r: 0, t: 0, b: 0 }, plain: true });

    let hover = -1;
    let reverse = false;   // false = our study order (top→bottom)
    let t = 0;             // animation phase

    const ROW_H = 40, GAP = 6, X0 = 150, X1 = W - 14, TOP = 26;

    const rungY = (i) => TOP + i * (ROW_H + GAP);

    plot.onDraw((p) => {
      const c = p.ctx;
      const n = STACK.length;

      c.font = '10px ui-monospace, Menlo, monospace';
      c.fillStyle = color('muted');
      c.textAlign = 'left'; c.textBaseline = 'middle';
      c.fillText(reverse ? 'HOW THE MACHINE RUNS  ↑' : 'HOW WE WILL STUDY IT  ↓', 14, 12);
      c.textAlign = 'right';
      c.fillText(reverse ? 'bottom → top' : 'top → bottom', X1, 12);

      for (let i = 0; i < n; i++) {
        const s = STACK[i];
        const y = rungY(i);
        const k = i / (n - 1);
        const hue = mix(color('c1'), color('c4'), k);
        const on = hover === i;

        // spine
        c.strokeStyle = color('border');
        c.lineWidth = 2;
        if (i < n - 1) {
          c.beginPath();
          c.moveTo(X0 - 26, y + ROW_H);
          c.lineTo(X0 - 26, y + ROW_H + GAP);
          c.stroke();
        }

        // rung
        c.fillStyle = on ? color('surface-2') : color('surface');
        c.strokeStyle = on ? hue : color('border');
        c.lineWidth = on ? 2 : 1;
        c.beginPath();
        c.roundRect(X0, y, X1 - X0, ROW_H, 9);
        c.fill(); c.stroke();

        // depth bar
        c.fillStyle = hue;
        c.globalAlpha = on ? 1 : 0.75;
        c.beginPath();
        c.roundRect(X0, y, 4, ROW_H, 2);
        c.fill();
        c.globalAlpha = 1;

        // level marker
        c.fillStyle = on ? hue : color('muted');
        c.font = `${on ? 'bold ' : ''}11px ui-monospace, Menlo, monospace`;
        c.textAlign = 'right'; c.textBaseline = 'middle';
        c.fillText(`level ${s.n}`, X0 - 40, y + ROW_H / 2);

        c.beginPath();
        c.arc(X0 - 26, y + ROW_H / 2, on ? 6 : 4, 0, 7);
        c.fillStyle = on ? hue : color('border');
        c.fill();

        // label + question
        c.textAlign = 'left';
        c.fillStyle = color('fg');
        c.font = '13px -apple-system, system-ui, sans-serif';
        c.fillText(s.label, X0 + 16, y + 14);
        c.fillStyle = color('muted');
        c.font = '11px -apple-system, system-ui, sans-serif';
        c.fillText(s.q, X0 + 16, y + 29);

        // anchor tag
        c.textAlign = 'right';
        c.font = '10px ui-monospace, Menlo, monospace';
        c.fillStyle = on ? color('fg') : color('muted');
        const tag = s.anchor.length > 34 ? '…' + s.anchor.slice(-33) : s.anchor;
        c.fillText(tag, X1 - 12, y + ROW_H / 2);
      }

      // travelling token
      const pos = (t % 1) * (STACK.length - 1);
      const idx = reverse ? STACK.length - 1 - pos : pos;
      const y = rungY(idx) + ROW_H / 2;
      c.beginPath();
      c.arc(X0 - 26, y, 7, 0, 7);
      c.fillStyle = color('accent');
      c.fill();
      c.strokeStyle = color('bg');
      c.lineWidth = 2;
      c.stroke();
    });

    stops.push(loop((dt) => { t += dt / 9000; plot.render(); }, plot.canvas));

    plot.interact({
      hover: (_, ev) => {
        const i = ev ? Math.floor((plot.atPx(ev).y - TOP) / (ROW_H + GAP)) : -1;
        const valid = ev && i >= 0 && i < STACK.length && plot.atPx(ev).x > 150;
        const next = valid ? i : -1;
        if (next !== hover) { hover = next; plot.canvas.style.cursor = next >= 0 ? 'pointer' : 'default'; plot.render(); }
      },
      down: (_, ev) => {
        const i = Math.floor((plot.atPx(ev).y - TOP) / (ROW_H + GAP));
        const s = STACK[i];
        if (s && s.ready) location.hash = '#/' + s.id;
      },
    });

    root.append(card('The descent', 'Hover a rung. The dot is one token making the trip the machine makes; flip the switch to see which way we will walk it.',
      row(
        el('div', {}, plot.canvas),
        el('div', { class: 'col', style: { flex: '1 1 240px' } },
          checkbox({
            label: 'show the machine\'s direction (forwards)',
            onChange: (v) => { reverse = v; plot.render(); },
          }),
          prose(`
            <p>The machine runs <b>upwards</b>: bytes → weights → matmuls → blocks → logits → a token → text.</p>
            <p>We will walk <b>downwards</b>, because that is the order in which the questions occur to you.</p>
            <p>Both directions pass through the same eleven places. By the end you will have built every one
            of them, in the browser, from scratch — and read the production version in Zig.</p>
          `),
          note('Every rung is a built lesson — click one to jump straight there. Levels 11 down to 0 are lessons 01 to 10.')))));

    /* ---------------- the question chain ---------------- */
    let depth = 0;
    const chainBox = el('div', { class: 'col' });
    const counter = stats([{ k: 'depth', v: '0 / 10' }]);

    const drawChain = () => {
      chainBox.replaceChildren();
      for (let i = 0; i <= depth; i++) {
        const step = CHAIN[i];
        const isLast = i === depth;
        chainBox.append(el('div', {
          style: {
            display: 'flex', gap: '10px', alignItems: 'baseline',
            padding: '7px 0 7px 0',
            borderLeft: `2px solid ${i === 0 ? 'var(--border)' : 'var(--c1)'}`,
            paddingLeft: '14px',
            marginLeft: `${i * 10}px`,
            opacity: isLast ? '1' : '0.62',
          },
        },
          el('span', { class: 'badge', text: String(i) }),
          el('span', { style: { fontSize: '14px' } }, step.a)));
      }
      counter.set('depth', `${depth} / ${CHAIN.length - 1}`);
      deeper.disabled = depth >= CHAIN.length - 1;
      deeper.textContent = depth >= CHAIN.length - 1 ? 'bottom reached — nothing below but arithmetic' : CHAIN[depth].q;
      back.disabled = depth === 0;
    };

    const deeper = button('', () => { if (depth < CHAIN.length - 1) { depth++; drawChain(); } });
    const back = button('↑ back up', () => { if (depth > 0) { depth--; drawChain(); } }, { ghost: true });
    drawChain();

    root.append(card('The question chain',
      'Press the question. Each answer is one level down; the course is this chain, slowed to one lesson per step.',
      chainBox,
      el('div', { style: { height: '10px' } }),
      btns(deeper, back),
      el('div', { style: { height: '8px' } }),
      counter));

    /* ---------------- after the descent ---------------- */
    const TRACKS = [
      {
        name: 'Track B — how a real engine ships it',
        why: 'The descent explains what the engine computes. It does not explain how a server turns that into a product that survives eight users at once.',
        items: [
          { id: '12-batching', t: 'Batching and lanes', q: 'Eight requests arrive at once. What happens?', ready: true },
          { id: '13-caching', t: 'Caches and prefixes', q: 'Why is the second message so much faster?', ready: true },
          { id: '14-speculative', t: 'Speculative decoding', q: 'More than one token per pass over the weights.', ready: true },
          { id: '15-streaming', t: 'Streaming it back', q: 'Integers to bytes on a socket, without garbage.', ready: true },
        ],
      },
      {
        name: 'Track C — one real model, read end to end',
        why: 'Concepts are portable; implementations are not. These read a single architecture exactly as it is written, in Zig and Metal.',
        items: [
          { id: '16-qwen-anatomy', t: 'Anatomy of a model', q: 'What is in an implementation, file by file?', ready: true },
          { id: '17-deltanet', t: 'The other 18 layers', q: 'Most of this model is not attention. So what is it?', ready: true },
        ],
      },
      {
        name: 'Track D — against the real thing',
        why: 'Stop reading the engine and run it. Then the three stages between your API call and the sampler, which cause most real bugs.',
        items: [
          { id: '18-live', t: 'Against a real engine', q: 'Same sampler, real model. Does the story hold?', ready: true },
          { id: '19-behaviour', t: 'Why your app misbehaves', q: 'Templates, penalties, grammars — none of them is the model.', ready: true },
          { id: '20-tools', t: 'Tool calling is a convention', q: 'A paragraph of prose and a regular expression.', ready: true },
          { id: '21-context', t: 'The edges of the window', q: 'Stretching it, and surviving overflow.', ready: true },
          { id: '22-diffusion', t: 'The other kind of generation', q: 'Same kernels, no tokens. Why images differ.', ready: true },
        ],
      },
    ];

    root.append(card('After the bottom: three more tracks',
      'The descent ends at bytes in a file. Three things remain that the descent deliberately skipped.',
      row(...TRACKS.map((tr) => el('div', { class: 'col', style: { flex: '1 1 340px' } },
        el('div', { style: { fontSize: '13px', fontWeight: '600' }, text: tr.name }),
        el('div', { class: 'sub', style: { margin: '2px 0 8px' }, text: tr.why }),
        ...tr.items.map((it) => el(it.ready ? 'a' : 'div', {
          href: it.ready ? '#/' + it.id : null,
          style: {
            display: 'block', textDecoration: 'none', color: 'inherit',
            border: '1px solid var(--border)', borderRadius: '9px',
            padding: '8px 11px', marginBottom: '6px',
            opacity: it.ready ? '1' : '0.5',
            borderLeft: `3px solid ${it.ready ? 'var(--accent)' : 'var(--border)'}`,
          },
        },
          el('div', { style: { fontSize: '13px' } }, it.t,
            it.ready ? null : el('span', { class: 'badge', style: { marginLeft: '8px' }, text: 'building' })),
          el('div', { style: { fontSize: '11.5px', color: 'var(--muted)' }, text: it.q })))))),
      note(`<b>Coverage, honestly.</b> Even with both tracks, this course cites about <b>1.6%</b> of
            TensorFold's source files. One backend of six, one quantisation format of eight, two model
            families of thirty. The spine has to stay unbroken to be teachable; the rest of the engine is
            overwhelmingly about going fast on specific hardware and staying correct under concurrency —
            not about different mathematics.`)));

    /* ---------------- what you will build ---------------- */
    root.append(card('What you will have built by the end', null, row(
      grow(prose(`
        <ul>
          <li>A sampler with every knob a real server exposes, drawing from live distributions.</li>
          <li>A byte-pair tokenizer you train yourself, on text you paste in.</li>
          <li>Attention, written out as the four operations it is, with every intermediate matrix on screen.</li>
          <li>A 4-bit quantiser matching the format on disk, with the error it introduces measured, not asserted.</li>
          <li>A transformer that trains in your browser and generates text — badly at first, then less badly.</li>
          <li>Backpropagation, drawn as a graph, checked against numerical gradients.</li>
        </ul>
      `)),
      el('div', { class: 'col', style: { flex: '1 1 280px' } },
        note(`<b>No install.</b> Everything on these pages is plain JavaScript running in this tab.
              Nothing is pre-baked; every number you see was computed here, just now.`),
        note(`<b>The reference engine.</b> <code class="inline">TensorFold/</code> sits beside this page:
              a Zig engine serving real checkpoints on Metal and CUDA. We read its source as the ground
              truth for every claim, and its kernels as the answer to "but how does it go fast?".`)))));

    return () => stops.forEach((s) => s());
  },
};
