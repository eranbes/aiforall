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

/* ---------- sidebar collapse ---------- */
const setNav = (collapsed) => {
  document.body.classList.toggle('nav-collapsed', collapsed);
  localStorage.setItem('aiforall.nav', collapsed ? 'collapsed' : 'open');
};
// No stored preference: start collapsed on narrow screens, where the full list
// would push the lesson itself below the fold.
const storedNav = localStorage.getItem('aiforall.nav');
setNav(storedNav ? storedNav === 'collapsed' : window.innerWidth <= 860);
document.getElementById('navclose').addEventListener('click', () => setNav(true));
document.getElementById('navopen').addEventListener('click', () => setNav(false));
// Esc toggles it too, so the keyboard works without hunting for the button.
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') setNav(!document.body.classList.contains('nav-collapsed'));
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

/* ---------- cross-references ---------- */
// Prose numbers its siblings in several shapes: "lesson 13", "lessons 12–22",
// "lessons 05, 06, 17" — and the number is often wrapped for emphasis, as in
// `lesson <b>01</b>`. So matching has to run over the flattened rendered text
// and then wrap a Range, not a single text node.
const BY_NUM = new Map(LESSONS.map((l) => [String(l.num).padStart(2, '0'), l]));
const MENTION = /\blessons?\s+\d{2}(?:\s*(?:[–—-]|,\s*(?:and\s+)?)\s*\d{2})*/gi;
const SKIP = new Set(['A', 'CODE', 'PRE', 'SCRIPT', 'STYLE', 'TEXTAREA', 'BUTTON', 'CANVAS']);

/** Flatten scope to text, keeping a character → (text node, offset) map. */
function flatten(scope) {
  let text = '';
  const map = []; // one entry per character
  const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    let skip = false;
    for (let p = n.parentElement; p && p !== scope; p = p.parentElement) {
      if (SKIP.has(p.tagName)) { skip = true; break; }
    }
    for (let i = 0; i < n.nodeValue.length; i++) map.push({ node: n, offset: i, skip });
    text += n.nodeValue;
  }
  return { text, map };
}

function linkLessonMentions(scope, selfId) {
  const { text, map } = flatten(scope);
  const hits = [];
  MENTION.lastIndex = 0;
  for (let m; (m = MENTION.exec(text));) {
    const lo = m.index, hi = lo + m[0].length;
    if (map.slice(lo, hi).some((c) => c.skip)) continue; // code, buttons, links
    // One link per number mentioned; a lone "lesson 13" takes the word too.
    const nums = [...m[0].matchAll(/\d{2}/g)];
    let first = true;
    for (const num of nums) {
      const lesson = BY_NUM.get(num[0]);
      if (!lesson || lesson.id === selfId) { first = false; continue; } // never self-link
      hits.push({
        lo: first ? lo : lo + num.index,
        hi: lo + num.index + num[0].length,
        lesson,
      });
      first = false;
    }
  }

  // Right to left: wrapping mutates the tree, so earlier offsets stay valid.
  for (let i = hits.length - 1; i >= 0; i--) {
    const { lo, hi, lesson } = hits[i];
    const start = map[lo], end = map[hi - 1];
    const range = document.createRange();
    range.setStart(start.node, start.offset);
    range.setEnd(end.node, end.offset + 1);
    const a = el('a', { class: 'xref', href: '#/' + lesson.id, title: lesson.title });
    // extractContents copes with a range that only partly covers <b>01</b>.
    a.append(range.extractContents());
    range.insertNode(a);
  }
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

  // Prose says "lesson 13" in 48 places; make every one of them clickable.
  linkLessonMentions(rootEl, lesson.id);

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
