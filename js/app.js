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
// Prose numbers its siblings in three shapes: "lesson 13", "lessons 12–22" and
// "lessons 05, 06, 17". Rather than hand-write anchors in ~50 places, rewrite
// the text nodes after each render. Each number becomes its own link.
const BY_NUM = new Map(LESSONS.map((l) => [String(l.num).padStart(2, '0'), l]));
// the whole phrase, including any trailing range/list of further numbers
const MENTION = /\blessons?\s+\d{2}(?:\s*(?:[–—-]|,\s*(?:and\s+)?)\s*\d{2})*/gi;
const NUM = /\d{2}/g;
const SKIP = new Set(['A', 'CODE', 'PRE', 'SCRIPT', 'STYLE', 'TEXTAREA', 'BUTTON', 'CANVAS']);

/** Split one matched phrase into links for each number, plain text between. */
function renderMention(phrase, selfId) {
  const parts = [];
  let cut = 0;
  let linked = false;
  NUM.lastIndex = 0;
  for (let m; (m = NUM.exec(phrase));) {
    const lesson = BY_NUM.get(m[0]);
    if (!lesson || lesson.id === selfId) continue; // never link a page to itself
    // A lone "lesson 13" reads better as one link, word included.
    const start = linked ? m.index : 0;
    if (start > cut) parts.push(phrase.slice(cut, start));
    parts.push(el('a', { class: 'xref', href: '#/' + lesson.id, title: lesson.title },
      phrase.slice(start, m.index + m[0].length)));
    cut = m.index + m[0].length;
    linked = true;
  }
  if (!linked) return null;
  if (cut < phrase.length) parts.push(phrase.slice(cut));
  return parts;
}

function linkLessonMentions(scope, selfId) {
  const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      MENTION.lastIndex = 0;
      if (!MENTION.test(n.nodeValue)) return NodeFilter.FILTER_REJECT;
      for (let p = n.parentElement; p && p !== scope; p = p.parentElement) {
        if (SKIP.has(p.tagName)) return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const targets = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) targets.push(n);

  for (const node of targets) {
    const frag = document.createDocumentFragment();
    let last = 0;
    MENTION.lastIndex = 0;
    for (let m; (m = MENTION.exec(node.nodeValue));) {
      const parts = renderMention(m[0], selfId);
      if (!parts) continue;
      if (m.index > last) frag.append(node.nodeValue.slice(last, m.index));
      frag.append(...parts);
      last = m.index + m[0].length;
    }
    if (!last) continue;
    frag.append(node.nodeValue.slice(last));
    node.replaceWith(frag);
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
