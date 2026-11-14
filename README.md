# aiforall

An interactive course on how large language models actually work — built **backwards**.

Most material starts at a neuron and hopes you reach a language model in six weeks. This starts
at a language model answering a request and takes it apart, one layer at a time, until nothing is
left but numbers in a file.

Everything runs in the browser. No install, no build step, no dependencies, no pre-baked data:
every number on every page is computed in your tab, just now.

## Start here

**→ [The whole thing in ten minutes](#/00b-tour)** — five ideas, five buttons. If you read nothing
else, read that. It is not a simplification that gets replaced later; it is what the rest is made of.

Then follow only what you are curious about. Lessons **01–11** are the course proper; 12–22 are
reference depth for when a question sends you looking.

## Run it locally

```sh
git clone https://github.com/eranbes/aiforall
cd aiforall
./serve.py            # or: python3 -m http.server 8777
open http://127.0.0.1:8777/
```

`serve.py` is `http.server` plus `Cache-Control: no-store` (so edited ES modules actually reload)
and a proxy at `/engine/*` to a local inference server on port 8080. Any static server works for
lessons 00–17; there is nothing to compile.

**Lessons 18–22 talk to a real engine** running on your own machine and degrade gracefully when
there is none — they say "no engine on 127.0.0.1:8080" rather than pretending. To light them up,
build TensorFold and serve a checkpoint (see Track D below).

## The descent

| Level | Lesson | The question it answers |
| --- | --- | --- |
| 11 | `01-sampling` | Text appears in your terminal. Where from? |
| 10 | `01-sampling` | The reply is not written — it is drawn, repeatedly. |
| 9 | `02-tokens` | The pieces are not words or letters. What are they? |
| 8 | `01-sampling` | A token is picked — how? |
| 7 | `03-softmax` | From a probability vector. Where does *that* come from? |
| 6 | `04-logits` | One vector times one very large matrix. |
| 5 | `05-blocks` | A stack of blocks, each one adding to a running vector. |
| 4 | `06-attention` | Where context enters the number. |
| 3 | `07-mlp` | The other half of the block, and most of the parameters. |
| 2 | `08-kernels` | All of it is matmul. Which thread multiplies which numbers? |
| 1 | `09-quant` | On 4-bit weights, unpacked inside the kernel. |
| 0 | `10-file` | A JSON header, then raw bytes. Nothing more. |
| — | `11-training` | The one part that really is backwards: gradients. |

Lesson `00-plan` explains the method and is the place to start.

## Track B — how a real engine ships it

The descent explains *what* the engine computes. It does not explain how a server turns that into a
product. These lessons cover the operationally important part, and the measurements on each page are
taken live in your browser.

| Lesson | The question it answers |
| --- | --- |
| `12-batching` | Eight requests arrive at once. What actually happens? |
| `13-caching` | Why is the second message of a conversation so much faster? |
| `14-speculative` | How do you get more than one token per pass over the weights? |
| `15-streaming` | How do integers become bytes on a socket without printing garbage? |

## Track C — one real model, read end to end

Concepts are portable; implementations are not. These read a single architecture —
`zig/src/families/qwen3_5/` — as it is actually written.

| Lesson | The question it answers |
| --- | --- |
| `16-qwen-anatomy` | What is in a model implementation, file by file? |
| `17-deltanet` | 18 of its 24 layers are not attention. What are they? |

## Track D — against the real thing

The last lesson stops reading the engine and runs it: the lab drives **your** sampler with logits
pulled from a TensorFold server on this machine.

| Lesson | The question it answers |
| --- | --- |
| `18-live` | Same sampler, real 2B model. Does the story hold? |
| `19-behaviour` | Why does my app misbehave when the model is fine? |
| `20-tools` | What is "function calling", mechanically? |
| `21-context` | Why does 128k work on a 32k model, and what happens when it overflows? |
| `22-diffusion` | Same repo, same kernels, no tokens — how do image models differ? |

Building the engine needs two exact things:

```sh
# 1. Zig pinned to the version in TensorFold/.zig-version (a @compileError enforces it)
curl -fsSLO https://ziglang.org/download/0.17.0/zig-aarch64-macos-0.17.0.tar.xz
tar -xJf zig-aarch64-macos-0.17.0.tar.xz
export PATH="$PWD/zig-aarch64-macos-0.17.0:$PATH"

# 2. Xcode's Metal compiler (839 MB; `xcrun -f metal` resolving is NOT enough)
xcodebuild -downloadComponent MetalToolchain

cd TensorFold && zig build -Doptimize=ReleaseFast
./zig-out/bin/tensorfold pull mlx-community/Qwen3.5-2B-MLX-4bit
./zig-out/bin/tensorfold serve mlx-community/Qwen3.5-2B-MLX-4bit --name local-model --context 8192
```

`serve.py` forwards `/engine/*` to `127.0.0.1:8080` so the lab works against an engine started exactly
as TensorFold's own docs describe. (The engine also has `--allowed-origin`, but depending on it would
make the lesson work only for servers launched a particular way.) The engine is never modified.

### Coverage, honestly

The course cites 100 of TensorFold's 6,105 source files (1.6%) and 24k of its 1.09M lines (2.2%).
It covers one backend of six, one quantisation format of eight, and two model families of thirty.
That is deliberate: the spine from text to bytes has to stay unbroken to be teachable. The rest of
the engine is overwhelmingly about going fast on specific hardware, staying correct under
concurrency, and supporting many model shapes — not about different mathematics.

## The rule

Every answer must bottom out in **code you can open** or **a number you can change and watch move**.
No metaphors that cannot be cashed out.

To keep that honest, the course is anchored to a real production inference engine rather than to
toy code: [TensorFold](https://github.com/ashhart/TensorFold), an LLM inference engine written in
Zig with Metal and CUDA kernels (~980k lines). It is cloned into `TensorFold/` and read as the
ground truth for every claim about how engines really behave.

Claims are traceable. Before a lesson says "the engine does X", the file and line number are
verified and recorded in `notes/`:

| Map | Covers |
| --- | --- |
| `notes/map-serving.md` | HTTP routes, SSE, chat templating, tokenizer, sampler, decode loop, speculative decoding |
| `notes/map-forward.md` | Attention (reference + Metal), GQA, RoPE, norms, MLP, MoE routing, KV cache, layer loop |
| `notes/map-weights.md` | safetensors layout, checkpoint loading, MLX 4-bit affine, FP8/NVFP4, matmul kernels, dtypes |
| `notes/map-lanes.md` | Admission, lanes and shared rounds, the prompt cache, learned disk state, recall, KV leasing |
| `notes/map-qwen35.md` | One family end to end: file inventory, gated delta-net, state sizes, kernel dispatch |

That is 5,422 lines of verified anchors behind 18 lessons.

Some findings that shaped the lessons — several of which contradicted the obvious guess:

- The sampler's noise is a **pure function** `splitmix64(seed, position, token_id)`, not an RNG
  stream — which is why a GPU can sample 1024 candidates in parallel and why speculative decoding
  can verify a draft with an exact integer comparison.
- **top-k is applied before temperature** (sound, because temperature never reorders candidates);
  top-p is applied after, and is *not* order-free.
- A shared round is **not one token per lane**. Turns go to the least-recently-served streams,
  clipped by a budget of 32 GPU *rows* — so a drafting stream can consume the budget and others are
  skipped entirely that round.
- Qwen3.5-2B has 24 layers but **only 6 are attention** — the other 18 are gated delta-net, a
  fixed-size associative memory updated by the delta rule. Below ~1,536 tokens of context that
  recurrence is the *more* expensive mixer, not the cheaper one.
- Residual addition **never appears as a `+`** in the layer loop: it is fused into the next
  normalisation kernel, and layer 0 dispatches a different kernel rather than adding zeros.
- Matmul kernels are **dispatched by matrix shape**, parsed at compile time out of the kernel's own
  filename (`qmm_248320_2048`), which is why a checkpoint of different widths is refused outright.
- Learned disk state is keyed by the **computation**, not the text: the directory name hashes the
  binary's identity, shard mtimes, kernel source and a real 4,143-token probe forward pass.

## Layout

```
index.html          shell: sidebar, lesson frame, routing target
styles.css          one stylesheet, dark + light via CSS custom properties
js/app.js           router, nav, theme
js/ui.js            DOM + control helpers (slider, card, stats, loop, …)
js/plot.js          canvas plotting: lines, bars, heatmaps, fields, interaction
js/engine/          the maths, written to be read
  tensor.js         matrices, matmul, softmax, norms, RoPE, activations
  sample.js         the sampler, in the engine's exact order
  ngram.js          a real (tiny) language model, so early lessons have real logits
  bpe.js            byte-level byte-pair encoding
  corpus.js         default training text
js/lessons/         one file per lesson; index.js is the registry
notes/              verified code maps of TensorFold
TensorFold/         the reference engine (git clone)
```

## Writing a lesson

```js
export default {
  id: '06-attention',      // must equal the filename stem
  num: '06',
  group: 'inside the model',
  title: 'Where context enters',
  blurb: 'One or two sentences: the question this answers.',
  render(root) {
    root.append(card('…', '…', /* nodes */));
    return () => {};       // cleanup, required if you call loop()
  },
};
```

Register it in `js/lessons/index.js`. Colours must come from the theme (`var(--accent)` in CSS,
`'accent'` in canvas calls) so both themes work. Seed every random number with `mulberry32` so a
reload reproduces the page.
