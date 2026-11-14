// plot.js — minimal immediate-mode 2D plotting on canvas, theme-aware.

const live = new Set();
let themeTick = 0;
const cache = new Map();

export function color(name) {
  const key = themeTick + ':' + name;
  let c = cache.get(key);
  if (!c) {
    c = getComputedStyle(document.body).getPropertyValue('--' + name).trim() || name;
    cache.set(key, c);
  }
  return c;
}

/** Call when the theme flips: invalidates color cache and redraws every live plot. */
export function refreshTheme() {
  themeTick++;
  cache.clear();
  for (const p of live) p.render();
}

export function mix(a, b, t) {
  const pa = hex(a), pb = hex(b);
  return `rgb(${Math.round(pa[0] + (pb[0] - pa[0]) * t)},${Math.round(pa[1] + (pb[1] - pa[1]) * t)},${Math.round(pa[2] + (pb[2] - pa[2]) * t)})`;
}
function hex(h) {
  h = h.trim();
  if (h.startsWith('rgb')) return h.match(/\d+/g).map(Number);
  if (h.length === 4) h = '#' + [...h.slice(1)].map((c) => c + c).join('');
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export class Plot {
  constructor(opts = {}) {
    const {
      width = 420, height = 300, xlim = [-1, 1], ylim = [-1, 1],
      pad = {}, equal = false, xlabel = '', ylabel = '', plain = false,
    } = opts;
    this.w = width; this.h = height;
    this.xlim = [...xlim]; this.ylim = [...ylim];
    this.pad = { l: 40, r: 12, t: 12, b: 28, ...pad };
    this.equal = equal; this.xlabel = xlabel; this.ylabel = ylabel;
    this.canvas = document.createElement('canvas');
    if (plain) this.canvas.className = 'plain';
    this.canvas.style.width = width + 'px';
    this.canvas.style.height = height + 'px';
    this.canvas.style.maxWidth = '100%';
    this.ctx = this.canvas.getContext('2d');
    this._scale();
    this._draw = null;
    live.add(this);
  }
  _scale() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = this.w * dpr;
    this.canvas.height = this.h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  /** Register the draw callback; called on render() and on theme change. */
  onDraw(fn) { this._draw = fn; this.render(); return this; }
  render() {
    if (!this.canvas.isConnected && this._mounted) { live.delete(this); return; }
    if (this.canvas.isConnected) this._mounted = true;
    if (this._draw) { this.clear(); this._draw(this); }
  }
  destroy() { live.delete(this); }

  get iw() { return this.w - this.pad.l - this.pad.r; }
  get ih() { return this.h - this.pad.t - this.pad.b; }

  X(x) {
    const [a, b] = this.xlim;
    return this.pad.l + ((x - a) / (b - a)) * this.iw;
  }
  Y(y) {
    const [a, b] = this.ylim;
    return this.pad.t + this.ih - ((y - a) / (b - a)) * this.ih;
  }
  invX(px) {
    const [a, b] = this.xlim;
    return a + ((px - this.pad.l) / this.iw) * (b - a);
  }
  invY(py) {
    const [a, b] = this.ylim;
    return a + ((this.pad.t + this.ih - py) / this.ih) * (b - a);
  }
  /** Mouse/touch event -> world coords. */
  at(ev) {
    const r = this.canvas.getBoundingClientRect();
    const t = ev.touches ? ev.touches[0] : ev;
    const sx = this.w / r.width, sy = this.h / r.height;
    return { x: this.invX((t.clientX - r.left) * sx), y: this.invY((t.clientY - r.top) * sy) };
  }

  /** Event -> canvas pixel coords (for px-space layouts like diagrams). */
  atPx(ev) {
    const r = this.canvas.getBoundingClientRect();
    const t = ev.touches ? ev.touches[0] : ev;
    return { x: (t.clientX - r.left) * (this.w / r.width), y: (t.clientY - r.top) * (this.h / r.height) };
  }

  clear() {
    this.ctx.clearRect(0, 0, this.w, this.h);
    return this;
  }

  axes({ xticks = 5, yticks = 5, grid = true, zero = true, xfmt = (v) => trimNum(v), yfmt = (v) => trimNum(v), frame = false } = {}) {
    const c = this.ctx;
    c.save();
    c.font = '10px ui-monospace, Menlo, monospace';
    c.fillStyle = color('muted');
    c.strokeStyle = color('grid');
    c.lineWidth = 1;
    const xs = ticks(this.xlim, xticks), ys = ticks(this.ylim, yticks);
    if (grid) {
      c.beginPath();
      for (const x of xs) { const px = Math.round(this.X(x)) + 0.5; c.moveTo(px, this.pad.t); c.lineTo(px, this.pad.t + this.ih); }
      for (const y of ys) { const py = Math.round(this.Y(y)) + 0.5; c.moveTo(this.pad.l, py); c.lineTo(this.pad.l + this.iw, py); }
      c.stroke();
    }
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (const x of xs) c.fillText(xfmt(x), this.X(x), this.pad.t + this.ih + 6);
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (const y of ys) c.fillText(yfmt(y), this.pad.l - 6, this.Y(y));
    if (zero) {
      c.strokeStyle = color('border');
      c.lineWidth = 1.4;
      c.beginPath();
      if (this.ylim[0] < 0 && this.ylim[1] > 0) { const py = Math.round(this.Y(0)) + 0.5; c.moveTo(this.pad.l, py); c.lineTo(this.pad.l + this.iw, py); }
      if (this.xlim[0] < 0 && this.xlim[1] > 0) { const px = Math.round(this.X(0)) + 0.5; c.moveTo(px, this.pad.t); c.lineTo(px, this.pad.t + this.ih); }
      c.stroke();
    }
    if (frame) {
      c.strokeStyle = color('border');
      c.lineWidth = 1;
      c.strokeRect(this.pad.l + 0.5, this.pad.t + 0.5, this.iw - 1, this.ih - 1);
    }
    if (this.xlabel) { c.textAlign = 'center'; c.textBaseline = 'bottom'; c.fillStyle = color('muted'); c.fillText(this.xlabel, this.pad.l + this.iw / 2, this.h - 1); }
    if (this.ylabel) {
      c.save(); c.translate(10, this.pad.t + this.ih / 2); c.rotate(-Math.PI / 2);
      c.textAlign = 'center'; c.textBaseline = 'top'; c.fillText(this.ylabel, 0, 0); c.restore();
    }
    c.restore();
    return this;
  }

  clip(fn) {
    const c = this.ctx;
    c.save();
    c.beginPath();
    c.rect(this.pad.l, this.pad.t, this.iw, this.ih);
    c.clip();
    fn(this);
    c.restore();
    return this;
  }

  line(pts, { color: col = 'accent', width = 2, dash = null, alpha = 1 } = {}) {
    if (pts.length < 2) return this;
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.strokeStyle = resolve(col);
    c.lineWidth = width;
    c.lineJoin = 'round'; c.lineCap = 'round';
    if (dash) c.setLineDash(dash);
    c.beginPath();
    let pen = false;
    for (const p of pts) {
      const [x, y] = Array.isArray(p) ? p : [p.x, p.y];
      if (!isFinite(x) || !isFinite(y)) { pen = false; continue; }
      const px = this.X(x), py = this.Y(y);
      if (!pen) { c.moveTo(px, py); pen = true; } else c.lineTo(px, py);
    }
    c.stroke();
    c.restore();
    return this;
  }

  /** Plot y=f(x) sampled across the x-range. */
  fn(f, opts = {}) {
    const n = opts.samples || 240;
    const [a, b] = this.xlim;
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const x = a + ((b - a) * i) / n;
      pts.push([x, f(x)]);
    }
    return this.clip(() => this.line(pts, opts));
  }

  area(pts, { color: col = 'accent', alpha = 0.14, base = null } = {}) {
    if (pts.length < 2) return this;
    const c = this.ctx;
    const y0 = base == null ? this.ylim[0] : base;
    c.save();
    c.globalAlpha = alpha;
    c.fillStyle = resolve(col);
    c.beginPath();
    c.moveTo(this.X(pts[0][0]), this.Y(y0));
    for (const [x, y] of pts) c.lineTo(this.X(x), this.Y(y));
    c.lineTo(this.X(pts[pts.length - 1][0]), this.Y(y0));
    c.closePath();
    c.fill();
    c.restore();
    return this;
  }

  scatter(pts, { color: col = 'accent', r = 4, stroke = null, alpha = 1, width = 1.5 } = {}) {
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    for (const p of pts) {
      const x = p.x ?? p[0], y = p.y ?? p[1];
      const fill = p.color ? resolve(p.color) : resolve(col);
      c.beginPath();
      c.arc(this.X(x), this.Y(y), p.r ?? r, 0, 7);
      c.fillStyle = fill;
      c.fill();
      if (stroke) { c.strokeStyle = resolve(stroke); c.lineWidth = width; c.stroke(); }
    }
    c.restore();
    return this;
  }

  /** Scalar field via offscreen pixel blit. f(x,y) -> css color string. */
  field(f, { res = 3 } = {}) {
    const c = this.ctx;
    const nx = Math.ceil(this.iw / res), ny = Math.ceil(this.ih / res);
    const off = document.createElement('canvas');
    off.width = nx; off.height = ny;
    const octx = off.getContext('2d');
    const img = octx.createImageData(nx, ny);
    const [x0, x1] = this.xlim, [y0, y1] = this.ylim;
    for (let j = 0; j < ny; j++) {
      const y = y1 - ((j + 0.5) / ny) * (y1 - y0);
      for (let i = 0; i < nx; i++) {
        const x = x0 + ((i + 0.5) / nx) * (x1 - x0);
        const [r, g, b, a = 255] = toRGBA(f(x, y));
        const k = (j * nx + i) * 4;
        img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = a;
      }
    }
    octx.putImageData(img, 0, 0);
    c.save();
    c.imageSmoothingEnabled = true;
    c.drawImage(off, this.pad.l, this.pad.t, this.iw, this.ih);
    c.restore();
    return this;
  }

  /** Matrix heatmap in world coords: rows x cols grid over [x0,x1]x[y0,y1]. */
  matrix(m, { x0 = this.xlim[0], x1 = this.xlim[1], y0 = this.ylim[0], y1 = this.ylim[1], cmap = seq, labels = null, gap = 1, text = false, fmtv = (v) => v.toFixed(2), max = null } = {}) {
    const rows = m.length, cols = m[0].length;
    const c = this.ctx;
    const hi = max ?? Math.max(1e-9, ...m.flat().map(Math.abs));
    const cw = (this.X(x1) - this.X(x0)) / cols;
    const ch = (this.Y(y0) - this.Y(y1)) / rows;
    c.save();
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = m[i][j];
        c.fillStyle = cmap(v / hi);
        c.fillRect(this.X(x0) + j * cw, this.Y(y1) + i * ch, cw - gap, ch - gap);
        if (text && cw > 26 && ch > 14) {
          c.fillStyle = Math.abs(v / hi) > 0.55 ? color('bg') : color('fg');
          c.font = '10px ui-monospace, Menlo, monospace';
          c.textAlign = 'center'; c.textBaseline = 'middle';
          c.fillText(fmtv(v), this.X(x0) + j * cw + cw / 2 - gap / 2, this.Y(y1) + i * ch + ch / 2 - gap / 2);
        }
      }
    }
    c.restore();
    if (labels) {
      c.save();
      c.font = '10px ui-monospace, Menlo, monospace';
      c.fillStyle = color('muted');
      if (labels.cols) {
        c.textAlign = 'center'; c.textBaseline = 'bottom';
        labels.cols.forEach((t, j) => c.fillText(t, this.X(x0) + j * cw + cw / 2, this.Y(y1) - 3));
      }
      if (labels.rows) {
        c.textAlign = 'right'; c.textBaseline = 'middle';
        labels.rows.forEach((t, i) => c.fillText(t, this.X(x0) - 4, this.Y(y1) + i * ch + ch / 2));
      }
      c.restore();
    }
    return this;
  }

  bars(vals, { color: col = 'accent', labels = null, width = 0.72, highlight = -1, fmtv = null, baseline = 0 } = {}) {
    const c = this.ctx;
    c.save();
    const n = vals.length;
    const slot = this.iw / n;
    for (let i = 0; i < n; i++) {
      const v = vals[i];
      const x = this.pad.l + i * slot + (slot * (1 - width)) / 2;
      const w = slot * width;
      const yTop = this.Y(Math.max(v, baseline)), yBot = this.Y(Math.min(v, baseline));
      c.fillStyle = i === highlight ? resolve('accent') : (Array.isArray(col) ? resolve(col[i]) : resolve(col));
      c.globalAlpha = highlight >= 0 && i !== highlight ? 0.45 : 1;
      c.fillRect(x, yTop, w, Math.max(1, yBot - yTop));
      if (labels && slot > 7) {
        c.globalAlpha = 1;
        c.fillStyle = color('muted');
        c.font = '10px ui-monospace, Menlo, monospace';
        const txt = String(labels[i]);
        if (slot < 34) {
          // crowded: lay the labels on a diagonal so they stay readable
          c.save();
          c.translate(x + w / 2, this.Y(baseline) + 5);
          c.rotate(-Math.PI / 4);
          c.textAlign = 'right'; c.textBaseline = 'middle';
          c.fillText(txt.length > 12 ? txt.slice(0, 11) + '…' : txt, 0, 0);
          c.restore();
        } else {
          c.textAlign = 'center'; c.textBaseline = 'top';
          c.fillText(txt, x + w / 2, this.Y(baseline) + 4);
        }
      }
      if (fmtv && slot > 24) {
        c.globalAlpha = 1;
        c.fillStyle = color('fg');
        c.font = '10px ui-monospace, Menlo, monospace';
        c.textAlign = 'center'; c.textBaseline = 'bottom';
        c.fillText(fmtv(v), x + w / 2, yTop - 2);
      }
    }
    c.restore();
    return this;
  }

  arrow(x0, y0, x1, y1, { color: col = 'accent', width = 2, head = 7, dash = null } = {}) {
    const c = this.ctx;
    const ax = this.X(x0), ay = this.Y(y0), bx = this.X(x1), by = this.Y(y1);
    const a = Math.atan2(by - ay, bx - ax);
    const len = Math.hypot(bx - ax, by - ay);
    if (len < 0.5) return this;
    const h = Math.min(head, len * 0.5);
    c.save();
    c.strokeStyle = resolve(col); c.fillStyle = resolve(col); c.lineWidth = width; c.lineCap = 'round';
    if (dash) c.setLineDash(dash);
    c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx - Math.cos(a) * h * 0.8, by - Math.sin(a) * h * 0.8); c.stroke();
    c.setLineDash([]);
    c.beginPath();
    c.moveTo(bx, by);
    c.lineTo(bx - Math.cos(a - 0.4) * h, by - Math.sin(a - 0.4) * h);
    c.lineTo(bx - Math.cos(a + 0.4) * h, by - Math.sin(a + 0.4) * h);
    c.closePath(); c.fill();
    c.restore();
    return this;
  }

  text(x, y, str, { color: col = 'fg', size = 11, align = 'center', baseline = 'middle', px = false, mono = true, weight = '' } = {}) {
    const c = this.ctx;
    c.save();
    c.fillStyle = resolve(col);
    c.font = `${weight} ${size}px ${mono ? 'ui-monospace, Menlo, monospace' : '-apple-system, sans-serif'}`.trim();
    c.textAlign = align; c.textBaseline = baseline;
    c.fillText(str, px ? x : this.X(x), px ? y : this.Y(y));
    c.restore();
    return this;
  }

  circle(x, y, r, { color: col = 'accent', fill = true, width = 2, alpha = 1, px = false }) {
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.beginPath();
    c.arc(px ? x : this.X(x), px ? y : this.Y(y), r, 0, 7);
    if (fill) { c.fillStyle = resolve(col); c.fill(); }
    else { c.strokeStyle = resolve(col); c.lineWidth = width; c.stroke(); }
    c.restore();
    return this;
  }

  rect(x0, y0, x1, y1, { color: col = 'accent', fill = false, width = 1.5, alpha = 1, dash = null, radius = 6, px = false }) {
    const c = this.ctx;
    const ax = px ? x0 : this.X(x0), ay = px ? y0 : this.Y(y1);
    const bw = (px ? x1 - x0 : this.X(x1) - this.X(x0)), bh = (px ? y1 - y0 : this.Y(y0) - this.Y(y1));
    c.save();
    c.globalAlpha = alpha;
    if (dash) c.setLineDash(dash);
    c.beginPath();
    c.roundRect(ax, ay, bw, bh, radius);
    if (fill) { c.fillStyle = resolve(col); c.fill(); }
    else { c.strokeStyle = resolve(col); c.lineWidth = width; c.stroke(); }
    c.restore();
    return this;
  }

  /** Drag/click interaction in world coords. */
  interact({ down, move, up, hover, wheel }) {
    const cv = this.canvas;
    let dragging = false;
    const get = (e) => this.at(e);
    const onDown = (e) => {
      e.preventDefault();
      dragging = true;
      down && down(get(e), e);
    };
    const onMove = (e) => {
      const p = get(e);
      if (dragging) { e.preventDefault(); move && move(p, e); }
      else hover && hover(p, e);
    };
    const onUp = (e) => {
      if (!dragging) return;
      dragging = false;
      up && up(get(e), e);
    };
    cv.addEventListener('pointerdown', onDown);
    cv.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    cv.addEventListener('pointerleave', () => hover && hover(null));
    if (wheel) cv.addEventListener('wheel', (e) => { e.preventDefault(); wheel(e.deltaY, get(e)); }, { passive: false });
    cv.style.cursor = 'crosshair';
    return this;
  }
}

function resolve(c) {
  if (!c) return color('fg');
  if (c.startsWith('#') || c.startsWith('rgb') || c.startsWith('hsl')) return c;
  return color(c);
}
function toRGBA(c) {
  if (Array.isArray(c)) return c;
  const s = resolve(c);
  if (s.startsWith('rgba')) { const m = s.match(/[\d.]+/g).map(Number); return [m[0], m[1], m[2], m[3] * 255]; }
  return [...hex(s), 255];
}

function trimNum(v) {
  const r = Math.round(v * 1000) / 1000;
  return Math.abs(r) < 1e-9 ? '0' : String(r);
}
function ticks([a, b], n) {
  const span = b - a;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

/* colormaps: t in [-1,1] (div) or [0,1] (seq) */
export const div = (t) => {
  t = Math.max(-1, Math.min(1, t));
  const base = color('surface-2');
  return t >= 0 ? mix(base, color('c1'), t) : mix(base, color('c2'), -t);
};
export const seq = (t) => mix(color('surface-2'), color('accent'), Math.max(0, Math.min(1, t)));
export const heat = (t) => {
  t = Math.max(0, Math.min(1, t));
  return t < 0.5 ? mix(color('surface-2'), color('c5'), t * 2) : mix(color('c5'), color('c2'), (t - 0.5) * 2);
};
