// app.js — router, nav, theme. No framework.

import { LESSONS } from './lessons/index.js';
import { refreshTheme } from './plot.js';
import { el } from './ui.js';

const navEl = document.getElementById('nav');
const rootEl = document.getElementById('lesson-root');
const titleEl = document.getElementById('lesson-title');
const blurbEl = document.getElementById('lesson-blurb');
const crumbEl = document.getElementById('lesson-crumb');
const pagerEl = document.getElementById('pager');

/* ---------- theme ---------- */
const savedTheme = localStorage.getItem('aiforall.theme');
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
document.getElementById('theme').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('aiforall.theme', next);
  refreshTheme();
});

/* ---------- nav ---------- */
let lastGroup = null;
for (const l of LESSONS) {
  if (l.group !== lastGroup) {
    navEl.append(el('div', { class: 'group', text: l.group }));
    lastGroup = l.group;
  }
  navEl.append(el('a', { href: '#/' + l.id, 'data-id': l.id },
    el('span', { class: 'num', text: l.num }),
    el('span', { text: l.title })));
}

/* ---------- routing ---------- */
let cleanup = null;

function show(id) {
  const i = Math.max(0, LESSONS.findIndex((l) => l.id === id));
  const lesson = LESSONS[i];

  if (cleanup) { try { cleanup(); } catch { /* lesson teardown is best-effort */ } cleanup = null; }
  rootEl.replaceChildren();

  crumbEl.textContent = `${lesson.group} · step ${lesson.num}`;
  titleEl.textContent = lesson.title;
  blurbEl.innerHTML = lesson.blurb || '';
  document.title = `${lesson.num} ${lesson.title} — aiforall`;

  for (const a of navEl.querySelectorAll('a')) a.classList.toggle('active', a.dataset.id === lesson.id);

  try {
    cleanup = lesson.render(rootEl) || null;
  } catch (err) {
    rootEl.append(el('div', { class: 'card' },
      el('h3', { text: 'This lesson failed to render' }),
      el('pre', { class: 'code', text: String(err && err.stack || err) })));
    console.error(err);
  }

  // prev / next
  pagerEl.replaceChildren();
  const prev = LESSONS[i - 1], next = LESSONS[i + 1];
  pagerEl.append(prev
    ? el('a', { href: '#/' + prev.id, class: 'prev' }, el('span', { class: 'dir', text: '← previous' }), prev.title)
    : el('span'));
  pagerEl.append(next
    ? el('a', { href: '#/' + next.id, class: 'next' }, el('span', { class: 'dir', text: 'next →' }), next.title)
    : el('span'));

  window.scrollTo(0, 0);
}

function route() {
  const id = location.hash.replace(/^#\/?/, '') || LESSONS[0].id;
  show(id);
}

window.addEventListener('hashchange', route);
route();
