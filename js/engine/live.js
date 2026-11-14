// live.js — talk to a real TensorFold server from the lab.
//
// The engine sends no CORS headers, so every call goes through serve.py's
// same-origin proxy at engine/… (see serve.py). If no engine is running the
// proxy answers 503 with a JSON error rather than hanging.
//
// The path is RELATIVE on purpose: the site also publishes to a subpath
// (e.g. /aiforall/ on GitHub Pages), where an absolute '/engine' would escape
// the site root. There is no proxy on a static host, so these lessons simply
// report "no engine" — which is what they are written to do.

const BASE = new URL('engine/v1', document.baseURI).pathname;

/**
 * True when this page is served from somewhere that cannot have an engine
 * behind it — GitHub Pages and friends are static file hosts, so there is no
 * proxy and nothing is listening. Lessons use this to explain the situation
 * instead of blaming a port the reader never started.
 */
export const HOSTED = !['127.0.0.1', 'localhost', '[::1]', ''].includes(location.hostname);

async function call(path, body) {
  const res = await fetch(BASE + path, body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {});
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { throw new Error(`non-JSON reply (${res.status}): ${text.slice(0, 160)}`); }
  if (json.error) throw new Error(json.error.message || JSON.stringify(json.error));
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 160)}`);
  return json;
}

/** The served model names, or throw if nothing is listening. */
export async function models() {
  const j = await call('/models');
  return (j.data || []).map((m) => ({
    id: m.id,
    context: m.context_length ?? m.max_model_len ?? null,
    reasoning: m.reasoning || null,
  }));
}


/** Capability flags the engine advertises for a model (logprobs, penalties, …). */
export async function capabilities() {
  const j = await call('/models');
  const m = (j.data || [])[0] || {};
  return m.capabilities || {};
}

/**
 * The engine's own renderer: messages → the exact token ids the model receives.
 * This is the chat template applied for real, not a reconstruction of it.
 */
export async function tokenize(model, messages, tools) {
  const body = { model, messages };
  if (tools && tools.length) body.tools = tools;   // schemas are rendered into the prompt
  const j = await call('/tokenize', body);
  return { tokens: j.tokens || [], count: j.count ?? (j.tokens || []).length, max: j.max_model_len };
}

/** ids → text. A list gives the joined string; a single id gives that piece. */
export async function detokenize(model, tokens) {
  const j = await call('/detokenize', { model, tokens });
  return j.prompt ?? j.text ?? '';
}

/** Each id's own piece, so specials like <|im_start|> become visible. */
export async function pieces(model, tokens, concurrency = 8) {
  const out = new Array(tokens.length);
  for (let i = 0; i < tokens.length; i += concurrency) {
    const slice = tokens.slice(i, i + concurrency);
    const got = await Promise.all(slice.map((t) => detokenize(model, [t]).catch(() => '\uFFFD')));
    got.forEach((g, k) => { out[i + k] = g; });
  }
  return out;
}

/**
 * One token, sampled by the engine itself.
 * `seed` is optional; with it, the draw is reproducible (lesson 01's keyed noise).
 */
export async function sampleOnce(model, prompt, { temperature = 1, seed } = {}) {
  const body = {
    model, messages: [{ role: 'user', content: prompt }],
    max_tokens: 1, temperature,
  };
  if (seed !== undefined) body.seed = seed;
  const j = await call('/chat/completions', body);
  const c = j.choices && j.choices[0];
  return { token: (c && c.message && c.message.content) || '', usage: j.usage || {} };
}

/**
 * Recover the model's next-token distribution by SAMPLING it, when the backend
 * will not report logprobs. `n` draws, a few in flight at a time.
 * @returns {Promise<{counts: Map<string, number>, n: number, ms: number}>}
 */
export async function measureDistribution(model, prompt, { n = 60, temperature = 1, concurrency = 8, onProgress } = {}) {
  const counts = new Map();
  const t0 = performance.now();
  let done = 0;
  for (let i = 0; i < n; i += concurrency) {
    const batch = Math.min(concurrency, n - i);
    const got = await Promise.all(
      Array.from({ length: batch }, () => sampleOnce(model, prompt, { temperature }).catch(() => null)),
    );
    for (const g of got) {
      if (!g) continue;
      counts.set(g.token, (counts.get(g.token) || 0) + 1);
    }
    done += batch;
    onProgress && onProgress(done, n);
  }
  return { counts, n: done, ms: performance.now() - t0 };
}

/**
 * The model's actual distribution over the next token, from logprobs.
 * Only works where the backend advertises `capabilities.logprobs`.
 *
 * @returns {{token: string, candidates: {token: string, logprob: number, p: number}[], usage: object}}
 */
export async function nextToken(model, prompt, { topK = 20, temperature = 1, raw = false } = {}) {
  const body = raw
    ? { model, prompt, max_tokens: 1, temperature, logprobs: Math.min(20, topK) }
    : {
      model, max_tokens: 1, temperature,
      messages: [{ role: 'user', content: prompt }],
      logprobs: true, top_logprobs: Math.min(20, topK),
    };
  const j = await call(raw ? '/completions' : '/chat/completions', body);
  const choice = j.choices && j.choices[0];
  if (!choice) throw new Error('no choices in reply');

  let entry;
  if (raw) {
    const lp = choice.logprobs || {};
    const top = (lp.top_logprobs && lp.top_logprobs[0]) || {};
    entry = {
      token: (lp.tokens && lp.tokens[0]) || choice.text || '',
      top_logprobs: Object.entries(top).map(([token, logprob]) => ({ token, logprob })),
    };
  } else {
    const content = (choice.logprobs && (choice.logprobs.content || choice.logprobs.reasoning_content)) || [];
    entry = content[0];
    if (!entry) throw new Error('reply carried no logprobs — does this engine advertise capabilities.logprobs?');
  }

  const candidates = (entry.top_logprobs || [])
    .map((t) => ({ token: t.token, logprob: t.logprob, p: Math.exp(t.logprob) }))
    .sort((a, b) => b.logprob - a.logprob);

  return { token: entry.token, candidates, usage: j.usage || {}, finish: choice.finish_reason };
}

/** A full reply, streamed token by token through the callback. */
export async function stream(model, prompt, { onToken, temperature = 0.7, maxTokens = 128, signal } = {}) {
  const res = await fetch(BASE + '/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model, messages: [{ role: 'user', content: prompt }],
      max_tokens: maxTokens, temperature, stream: true,
    }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`stream failed: HTTP ${res.status}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '', out = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    // SSE frames are separated by a blank line (lesson 15)
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, i).trim();
      buf = buf.slice(i + 2);
      if (!frame.startsWith('data:')) continue;        // ": keepalive" comments land here
      const payload = frame.slice(5).trim();
      if (payload === '[DONE]') return out;            // the literal sentinel, not JSON
      try {
        const j = JSON.parse(payload);
        const piece = j.choices?.[0]?.delta?.content || '';
        if (piece) { out += piece; onToken && onToken(piece, out); }
      } catch { /* a partial frame: wait for the rest */ }
    }
  }
  return out;
}

/** Is an engine reachable at all? Never throws. */
export async function probe() {
  try {
    const list = await models();
    return { ok: true, models: list };
  } catch (e) {
    return { ok: false, error: String(e.message || e) };
  }
}
