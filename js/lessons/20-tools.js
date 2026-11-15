// 20-tools.js — "function calling" is a paragraph of prose and a parser.

import { Plot, color } from '../plot.js';
import { card, el, prose, note, btns, button, row, grow, slider, stats, legend, textbox, checkbox, fmt, presets } from '../ui.js';
import { probe, tokenize } from '../engine/live.js';

/* The markup this checkpoint's own chat template instructs the model to emit.
   Verbatim from chat_template.jinja in the downloaded snapshot. */
const EXAMPLE = `<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>`;

const TOOL = (name = 'get_weather') => ({
  type: 'function',
  function: {
    name,
    description: 'Get current weather for a city',
    parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
  },
});

/**
 * The parser. This is the entire "function calling" feature on the server side:
 * find the markup, pull out the name and parameters, hand back JSON.
 */
export function parseToolCalls(text) {
  const calls = [];
  const callRe = /<tool_call>\s*<function=([^>]+)>([\s\S]*?)<\/function>\s*<\/tool_call>/g;
  let m;
  while ((m = callRe.exec(text)) !== null) {
    const name = m[1].trim();
    const args = {};
    const paramRe = /<parameter=([^>]+)>\n?([\s\S]*?)\n?<\/parameter>/g;
    let p;
    while ((p = paramRe.exec(m[2])) !== null) args[p[1].trim()] = p[2];
    calls.push({ type: 'function', function: { name, arguments: JSON.stringify(args) } });
  }
  const content = text.replace(callRe, '').trim();
  return { calls, content };
}

export default {
  id: '20-tools',
  num: '20',
  group: 'the real thing',
  title: 'Tool calling is a convention',
  blurb: 'There is no "function calling" inside the model. There is a paragraph of instructions pushed into your prompt, a format the model is asked to imitate, and a regular expression on the way out. Every surprising failure follows from that.',

  render(root) {
    const S = { connected: false, model: null, counts: {}, busy: false };

    /* ---------------- 1. what a tool costs ---------------- */
    const costPlot = new Plot({
      width: 460, height: 240, pad: { l: 56, r: 14, t: 20, b: 40 },
      xlim: [0, 6], ylim: [0, 1], xlabel: 'tools declared', ylabel: 'prompt tokens',
    });
    const costStats = stats([
      { k: 'no tools', v: '—' }, { k: 'one tool', v: '—' },
      { k: 'cost per tool', v: '—' }, { k: 'of an 8k window', v: '—' },
    ]);

    const measureCost = async () => {
      if (!S.connected || S.busy) return;
      S.busy = true;
      const msg = [{ role: 'user', content: 'What is the weather in Paris?' }];
      const out = {};
      for (const n of [0, 1, 2, 3, 4, 5]) {
        const tools = Array.from({ length: n }, (_, i) => TOOL(`tool_${i}`));
        const body = n === 0 ? { messages: msg } : { messages: msg, tools };
        try {
          const r = await tokenize(S.model, body.messages, body.tools);
          out[n] = r.count;
        } catch { out[n] = null; }
      }
      S.counts = out;
      const zero = out[0], one = out[1];
      if (zero && one) {
        costStats.set('no tools', `${zero} tokens`);
        costStats.set('one tool', `${one} tokens`);
        costStats.set('cost per tool', `~${one - zero} tokens`);
        costStats.set('of an 8k window', fmt.pct((one - zero) / 8192, 1));
      }
      S.busy = false;
      costPlot.render();
    };

    costPlot.onDraw((p) => {
      const ks = Object.keys(S.counts).map(Number).filter((k) => S.counts[k]);
      if (!ks.length) {
        p.axes({ xticks: 5, yticks: 4, zero: false });
        p.text(p.w / 2, p.h / 2, S.busy ? 'measuring…' : 'press measure', { color: 'muted', size: 12, px: true });
        return;
      }
      const pts = ks.map((k) => [k, S.counts[k]]);
      p.ylim = [0, Math.max(...pts.map((q) => q[1])) * 1.15];
      p.axes({ xticks: 5, yticks: 5, zero: false });
      p.area(pts, { color: 'c5', alpha: 0.14 });
      p.line(pts, { color: 'c5', width: 2.5 });
      p.scatter(pts, { color: 'c5', r: 4, stroke: 'bg' });
      p.text(p.pad.l + 6, p.pad.t + 8, 'every request pays this, every turn',
        { color: 'muted', size: 10, px: true, align: 'left' });
    });

    root.append(card('A tool is a paragraph in your prompt',
      'Declaring a tool does not configure the model. It injects a system message containing your JSON schema verbatim, plus a page of instructions telling the model what to imitate. Measured against the running engine.',
      row(
        el('div', {}, costPlot.canvas,
          el('div', { style: { height: '8px' } }),
          btns(button('measure ▸', measureCost))),
        el('div', { class: 'col', style: { flex: '1 1 320px' } },
          costStats,
          note(`<b>Measured here: 19 tokens becomes 274</b> when a single two-field tool is declared. That is
                about <b>255 tokens of overhead per tool</b>, resent on every turn of the conversation, forever.
                Declare twelve tools and a third of a 8k window is gone before the user types anything.`),
          note(`<b>The schema goes in raw.</b> The template does
                <code class="inline">{{ tool | tojson }}</code> — your JSON Schema, serialised, inside a
                <code class="inline">&lt;tools&gt;</code> block in a system message. Long descriptions and deep
                nesting are paid for in tokens at full price.`)))));

    /* ---------------- 2. what the model is told ---------------- */
    root.append(card('What the model is actually told to do',
      'Verbatim from this checkpoint\'s <code class="inline">chat_template.jinja</code>. Not an API contract — an instruction, in English, that the model may or may not follow.',
      row(
        grow(el('pre', { class: 'code', html:
          `<span class="cm"># Tools</span>\n\n` +
          `You have access to the following functions:\n\n` +
          `&lt;tools&gt;\n` +
          `{"type":"function","function":{"name":"get_weather", ... }}\n` +
          `&lt;/tools&gt;\n\n` +
          `<b>If you choose to call a function ONLY reply in the following\n` +
          `format with NO suffix:</b>\n\n` +
          `&lt;tool_call&gt;\n` +
          `&lt;function=example_function_name&gt;\n` +
          `&lt;parameter=example_parameter_1&gt;\n` +
          `value_1\n` +
          `&lt;/parameter&gt;\n` +
          `&lt;/function&gt;\n` +
          `&lt;/tool_call&gt;\n\n` +
          `&lt;IMPORTANT&gt;\n` +
          `Reminder:\n` +
          `- Function calls MUST follow the specified format ...\n` +
          `- Required parameters MUST be specified\n` +
          `- You may provide optional reasoning ... BEFORE the function call,\n` +
          `  but NOT after` })),
        el('div', { class: 'col', style: { flex: '1 1 300px' } },
          note(`<b>Read the capitalised words.</b> "ONLY", "MUST", "NO suffix", "IMPORTANT". This is a prompt
                engineer begging a text predictor to stay in format. Nothing enforces it — unless a grammar is
                attached (lesson 19), in which case the illegal tokens are removed outright.`),
          note(`<b>It is XML, not JSON.</b> The model writes
                <code class="inline">&lt;parameter=city&gt;Paris&lt;/parameter&gt;</code>. The tidy
                <code class="inline">{"city":"Paris"}</code> your API client receives was <i>constructed by the
                server</i> from that markup. TensorFold recognises <b>four envelope families</b> and several
                payload forms — Qwen's XML, plain JSON, GLM's <code class="inline">&lt;arg_key&gt;</code>
                pairs, Gemma's <code class="inline">call:NAME{…}</code> — because every checkpoint was trained
                to write something different. The uniform API hides all of it.`),
          note(`<b>Which is why a model can "forget" tools.</b> It is reading an instruction several thousand
                tokens up, competing with everything else in the prompt.`)))));

    /* ---------------- 3. the parser, and breaking it ---------------- */
    let casePresets = null;
    const markup = textbox({
      label: 'what the model emitted (edit it — break it on purpose)',
      rows: 9, value: `I'll check that for you.\n\n${EXAMPLE}`,
      onInput: () => { reparse(); if (casePresets) casePresets.select(null); },
    });
    const parsedBox = el('pre', { class: 'code' });
    const verdict = el('span', { class: 'badge' });
    const schemaCheck = el('div', { class: 'col', style: { gap: '4px' } });

    const reparse = () => {
      const { calls, content } = parseToolCalls(markup.get());
      parsedBox.innerHTML = JSON.stringify({
        content: content || null,
        tool_calls: calls.length ? calls : null,
        finish_reason: calls.length ? 'tool_calls' : 'stop',
      }, null, 2).replace(/</g, '&lt;');

      verdict.className = calls.length ? 'badge ok' : 'badge err';
      verdict.textContent = calls.length
        ? `✓ ${calls.length} call${calls.length > 1 ? 's' : ''} parsed`
        : '✗ no call found — this is delivered to your app as ordinary text';

      // the checks YOUR code has to do, because the markup carries no guarantees
      const declared = { get_weather: ['city'] };
      const rows = [];
      for (const c of calls) {
        const name = c.function.name;
        const args = JSON.parse(c.function.arguments);
        const known = Object.prototype.hasOwnProperty.call(declared, name);
        rows.push([`tool "${name}" exists`, known]);
        if (known) {
          for (const req of declared[name]) {
            rows.push([`required "${req}" present`, Object.prototype.hasOwnProperty.call(args, req)]);
          }
          for (const k of Object.keys(args)) {
            if (!declared[name].includes(k)) rows.push([`unexpected argument "${k}"`, false]);
          }
        }
      }
      schemaCheck.replaceChildren(...(rows.length ? rows : [['nothing to validate', null]]).map(([t, ok]) => el('div', {
        style: { display: 'flex', gap: '8px', alignItems: 'baseline', fontSize: '12.5px' },
      },
        el('span', { class: ok === null ? 'badge' : ok ? 'badge ok' : 'badge err', text: ok === null ? '–' : ok ? '✓' : '✗' }),
        el('span', { text: t }))));
    };

    // The first value is also the textbox's initial content, so the matching
    // preset starts highlighted rather than the row looking unselected.
    casePresets = presets([
      { label: 'valid call', value: `I'll check that for you.\n\n${EXAMPLE}` },
      { label: 'typo in the tag', value: EXAMPLE.replace('<tool_call>', '<toolcall>') },
      { label: 'missing required arg', value: EXAMPLE.replace(/<parameter=city>[\s\S]*?<\/parameter>\n/, '') },
      { label: 'invented tool', value: EXAMPLE.replace('get_weather', 'send_email') },
      { label: 'prose after the call', value: `${EXAMPLE}\n\nLet me know if you need more!` },
    ].map((p) => ({ ...p, onPick: (v) => { markup.set(v); casePresets.select(v); reparse(); } })),
    `I'll check that for you.\n\n${EXAMPLE}`);

    root.append(card('The parser is the entire feature',
      'This is a working implementation of the server side, running here. Edit the left pane — misspell <code class="inline">&lt;tool_call&gt;</code>, drop a <code class="inline">&lt;/parameter&gt;</code>, invent a tool, add prose after the call — and watch what your application receives.',
      row(
        grow(markup,
          el('div', { style: { height: '8px' } }),
          btns(button('parse ▸', reparse)),
          el('div', { style: { height: '8px' } }),
          casePresets),
        el('div', { class: 'col', style: { flex: '1 1 330px' } },
          el('div', { class: 'sub', style: { margin: '0 0 4px' }, text: 'what your HTTP client receives' }),
          parsedBox,
          verdict,
          el('div', { class: 'sub', style: { margin: '8px 0 4px' }, text: 'the engine checks the name; everything below this line is your job' }),
          schemaCheck))));

    root.append(card('The failure modes, and who owns them', null, row(
      grow(prose(`
        <ul>
          <li><b>A typo in the tag</b> → no call is found, and the markup is handed to your app as ordinary
          message content. Your code sees a chat reply full of angle brackets.</li>
          <li><b>An invented tool name</b> → the parser above accepts it, but the <i>real engine does not</i>.
          TensorFold builds a set of the names your request offered and classifies an envelope naming anything
          else as <code class="inline">.unoffered</code>: the markup is copied back into
          <code class="inline">content</code> verbatim and counted in
          <code class="inline">tensorfold.kept_tool_calls</code>
          (<code class="inline">tool_parse.zig:153-168</code>). Names are checked. Keep reading.</li>
          <li><b>A missing required parameter</b> → also parses perfectly. JSON Schema was <i>advice in the
          prompt</i>, not a validator on the output.</li>
          <li><b>Prose after the call</b> → the template explicitly forbids it ("but NOT after"), because trailing
          text makes the boundary ambiguous. The model obeys that most of the time.</li>
          <li><b>Arguments are a string</b>, not an object: <code class="inline">"{\\"city\\":\\"Paris\\"}"</code>.
          It can be malformed JSON, and then <code class="inline">JSON.parse</code> throws inside your handler.</li>
        </ul>`)),
      el('div', { class: 'col', style: { flex: '1 1 300px' } },
        note(`<b>Names are validated; argument values are not.</b> That is the real split, and it is easy to
              get backwards. The engine verifies the tool was offered — but
              <code class="inline">tool_params.decode</code> only <i>types</i> the argument values, keeping
              anything your schema would reject as the model's own string. A required field you declared can be
              missing, and an extra field you never declared can be present, and both reach your handler.`),
        note(`<b>The one real enforcement</b> is constrained decoding (lesson 19): with a grammar attached, the
              illegal tokens cannot be sampled at all. TensorFold routes that through
              <code class="inline">libtfgrammar.dylib</code> and answers HTTP 400 when it is missing — which is
              exactly what the server we built does, because we did not build the grammar library.`),
        note(`<b>Observed on the live engine:</b> a request with one tool returned
              <code class="inline">finish_reason: "tool_calls"</code>,
              <code class="inline">content: null</code>, and one call with
              <code class="inline">arguments: "{\\"city\\":\\"Paris\\"}"</code> — the markup above, parsed.`)))));

    const connect = async () => {
      const r = await probe();
      S.connected = r.ok;
      if (r.ok && r.models.length) S.model = r.models[0].id;
      if (r.ok) measureCost();
    };

    reparse();
    connect();
  },
};
