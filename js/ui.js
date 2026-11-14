// ui.js — tiny DOM + control helpers. No framework, no build step.

export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2).toLowerCase(), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat(9)) {
    if (k == null || k === false) continue;
    n.append(k.nodeType ? k : document.createTextNode(String(k)));
  }
  return n;
}

export const row = (...kids) => el('div', { class: 'row' }, kids);
export const col = (...kids) => el('div', { class: 'col' }, kids);
export const grow = (...kids) => el('div', { class: 'col grow' }, kids);
export const prose = (html) => el('div', { class: 'prose', html });
export const note = (html) => el('div', { class: 'note', html });
export const eq = (html) => el('div', { class: 'eq', html });
export const code = (text) => el('pre', { class: 'code', html: text });

export function card(title, sub, ...kids) {
  const c = el('div', { class: 'card' });
  if (title) c.append(el('h3', { text: title }));
  if (sub) c.append(el('p', { class: 'sub', html: sub }));
  c.append(...kids.flat(9).filter(Boolean));
  return c;
}

/** Range slider with live numeric readout. Returns node augmented with get/set. */
export function slider({ label, min = 0, max = 1, step = 0.01, value = 0, fmt = (v) => v.toFixed(2), onInput, width = '100%' }) {
  const out = el('b', { text: fmt(value) });
  const input = el('input', { type: 'range', min, max, step, value });
  const node = el('div', { class: 'ctl', style: { width } },
    el('div', { class: 'lbl' }, el('span', { html: label }), out),
    input);
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    out.textContent = fmt(v);
    onInput && onInput(v);
  });
  node.get = () => parseFloat(input.value);
  node.set = (v, fire = false) => {
    input.value = v;
    out.textContent = fmt(parseFloat(input.value));
    if (fire && onInput) onInput(parseFloat(input.value));
  };
  return node;
}

export function button(label, onClick, { ghost = false } = {}) {
  return el('button', { class: ghost ? 'ghost' : '', onClick }, label);
}
export const btns = (...kids) => el('div', { class: 'btns' }, kids.flat(9).filter(Boolean));

export function select({ label, options, value, onChange }) {
  const s = el('select');
  for (const o of options) {
    const { v, t } = typeof o === 'string' ? { v: o, t: o } : o;
    s.append(el('option', { value: v, selected: v === value }, t));
  }
  s.addEventListener('change', () => onChange && onChange(s.value));
  const node = el('div', { class: 'ctl' }, label ? el('div', { class: 'lbl' }, el('span', { html: label })) : null, s);
  node.get = () => s.value;
  node.set = (v) => { s.value = v; };
  return node;
}

export function checkbox({ label, checked = false, onChange }) {
  const i = el('input', { type: 'checkbox', checked });
  i.addEventListener('change', () => onChange && onChange(i.checked));
  const node = el('label', { class: 'check' }, i, el('span', { html: label }));
  node.get = () => i.checked;
  node.set = (v) => { i.checked = v; };
  return node;
}

export function textbox({ label, value = '', rows = 3, onInput, placeholder = '' }) {
  const t = rows > 1
    ? el('textarea', { rows, placeholder }, value)
    : el('input', { type: 'text', value, placeholder });
  t.addEventListener('input', () => onInput && onInput(t.value));
  const node = el('div', { class: 'ctl' }, label ? el('div', { class: 'lbl' }, el('span', { html: label })) : null, t);
  node.get = () => t.value;
  node.set = (v) => { t.value = v; };
  return node;
}

/** Stat readouts. `defs` = [{k, v}]. Returns node with .set(key, value). */
export function stats(defs) {
  const map = new Map();
  const node = el('div', { class: 'stats' });
  for (const d of defs) {
    const v = el('div', { class: 'v', text: d.v ?? '—' });
    map.set(d.k, v);
    node.append(el('div', { class: 'stat' }, el('div', { class: 'k', text: d.k }), v));
  }
  node.set = (k, val, color) => {
    const n = map.get(k);
    if (!n) return;
    n.textContent = val;
    if (color) n.style.color = `var(--${color})`;
  };
  return node;
}

export const legend = (items) =>
  el('div', { class: 'legend' }, items.map(({ c, label }) =>
    el('span', {}, el('i', { style: { background: `var(--${c})` } }), label)));

/** requestAnimationFrame loop; returns stop(). Auto-stops when node detached. */
export function loop(fn, node) {
  let on = true, last = performance.now();
  const tick = (t) => {
    if (!on) return;
    if (node && !node.isConnected) { on = false; return; }
    const dt = Math.min(100, t - last); last = t;
    fn(dt);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  return () => { on = false; };
}

export const fmt = {
  n: (x, d = 2) => (Object.is(x, -0) ? 0 : x).toFixed(d),
  sign: (x, d = 2) => (x >= 0 ? '+' : '') + x.toFixed(d),
  pct: (x, d = 1) => (100 * x).toFixed(d) + '%',
  exp: (x, d = 2) => Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(d) : x.toFixed(d),
};

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
