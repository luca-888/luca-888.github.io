// KV Cache 压缩视频的公共图元：配色、缓动、文字与形状，沿用 Fused Linear Cross Entropy 视频（videos/fused-linear-cross-entropy/lib.js）。
const K = { paper: '#f1ede4', paper2: '#e6e0d3', ink: '#141414', ink2: '#3a3a3a', gray: '#8a8474', lg: '#b9b2a2',
  red: '#e5322d', blue: '#2350d8', yel: '#f4c21b', yelD: '#9a7500', purple: '#7c3ff5', teal: '#0fa391' };
const ZH = '"PingFang SC", "Helvetica Neue", sans-serif', HN = '"Helvetica Neue", "PingFang SC", sans-serif';
const font = (w, s, f = ZH) => `${w} ${s}px ${f}`;
let X, T = 0;
const clamp = (x) => Math.min(1, Math.max(0, x));
const clampR = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (x) => 1 - Math.pow(1 - clamp(x), 3);
const io = (x) => { x = clamp(x); return x < .5 ? 4 * x ** 3 : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const P = (t0, d = .7) => ease((T - t0) / d);                    // 缓出
const S = (t0, d = 1) => io((T - t0) / d);                       // 缓入缓出
const mix = (a, b, p) => a + (b - a) * p;
const R = (x, y, w, h, c) => { if (w > 0 && h > 0) { X.fillStyle = c; X.fillRect(x, y, w, h); } };
function TX(s, x, y, f, c, a = 'left') { X.font = f; X.fillStyle = c; X.textAlign = a; X.textBaseline = 'alphabetic'; const available = a === 'center' ? Math.min(x, 1920 - x) * 2 - 32 : a === 'right' ? x - 72 : 1824 - x; X.fillText(s, x, y, Math.max(40, available)); }
function A(a, fn) { if (a <= .003) return; X.save(); X.globalAlpha *= Math.min(1, a); fn(); X.restore(); }
function IN(t0, fn, d = .7, dy = 20) { const p = P(t0, d); if (p <= 0) return; X.save(); X.globalAlpha *= p; X.translate(0, (1 - p) * dy); fn(); X.restore(); }
function POP(t0, x, y, w, h, fn, d = .5) { const p = P(t0, d); if (p <= 0) return; X.save(); X.globalAlpha *= Math.min(1, p * 1.5); X.translate(x + w / 2, y + h / 2); X.scale(.86 + .14 * p, .86 + .14 * p); X.translate(-x - w / 2, -y - h / 2); fn(); X.restore(); }
const win = (a, b, f = .3) => Math.min(P(a, f), 1 - P(b, f));   // a 时淡入、b 时淡出
function RICH(s, x, y, f, base, align = 'left') {
  const ACC = { b: K.blue, r: K.red, y: K.yelD, p: K.purple, t: K.teal, g: K.gray };
  const parts = []; const re = /\{(\w):([^}]*)\}/g; let last = 0, m;
  while ((m = re.exec(s))) { if (m.index > last) parts.push([s.slice(last, m.index), null]); parts.push([m[2], m[1]]); last = re.lastIndex; }
  if (last < s.length) parts.push([s.slice(last), null]);
  X.font = f; const w = parts.reduce((a, [t]) => a + X.measureText(t).width, 0); let cx = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  X.textAlign = 'left'; X.textBaseline = 'alphabetic';
  X.save(); const limit = align === 'center' ? Math.min(x, 1920 - x) * 2 - 32 : align === 'right' ? x - 72 : 1824 - x; if (w > limit) { X.translate(cx, 0); X.scale(limit / w, 1); cx = 0; }
  for (const [t, k] of parts) { X.fillStyle = k ? ACC[k] : base; X.fillText(t, cx, y); cx += X.measureText(t).width; } X.restore();
}
const PAT = {};
function hatch(col, bg) { const k = col + bg; if (PAT[k]) return PAT[k]; const g = 14, c = document.createElement('canvas'); c.width = c.height = g; const q = c.getContext('2d');
  q.fillStyle = bg; q.fillRect(0, 0, g, g); q.strokeStyle = col; q.lineWidth = 4; q.beginPath(); for (let o = -g; o <= g * 2; o += g) { q.moveTo(o, g); q.lineTo(o + g, 0); } q.stroke(); return (PAT[k] = X.createPattern(c, 'repeat')); }
const HR = (x, y, w, h, col = K.lg, bg = K.paper2) => { if (w > 0 && h > 0) { X.fillStyle = hatch(col, bg); X.fillRect(x, y, w, h); } };
const box = (x, y, w, h, lw = 4, col = K.ink) => { X.strokeStyle = col; X.lineWidth = lw; X.strokeRect(x, y, w, h); };
function line(x1, y1, x2, y2, col = K.ink, lw = 3, dash) { X.strokeStyle = col; X.lineWidth = lw; X.setLineDash(dash || []); X.beginPath(); X.moveTo(x1, y1); X.lineTo(x2, y2); X.stroke(); X.setLineDash([]); }
function poly(pts, col = K.ink, lw = 4) { X.strokeStyle = col; X.lineWidth = lw; X.lineJoin = 'round'; X.beginPath(); pts.forEach(([x, y], i) => (i ? X.lineTo(x, y) : X.moveTo(x, y))); X.stroke(); }
function arrow(x1, y1, x2, y2, col = K.ink, lw = 5) { const a = Math.atan2(y2 - y1, x2 - x1), hl = lw * 3.4, hw = lw * 2.2; if (Math.hypot(x2 - x1, y2 - y1) < hl) return;
  line(x1, y1, x2 - Math.cos(a) * hl * .9, y2 - Math.sin(a) * hl * .9, col, lw);
  X.fillStyle = col; X.beginPath(); X.moveTo(x2, y2); X.lineTo(x2 - Math.cos(a) * hl + Math.sin(a) * hw, y2 - Math.sin(a) * hl - Math.cos(a) * hw); X.lineTo(x2 - Math.cos(a) * hl - Math.sin(a) * hw, y2 - Math.sin(a) * hl + Math.cos(a) * hw); X.fill(); }
// 画到一半的箭头：p 从 0 到 1
const arrowP = (x1, y1, x2, y2, p, col, lw) => { if (p > .02) arrow(x1, y1, mix(x1, x2, p), mix(y1, y2, p), col, lw); };
function cell(x, y, w, h, col, label, size = 30, tc = '#fff', fam = ZH) { R(x, y, w, h, col); if (label != null) TX(label, x + w / 2, y + h / 2 + size * .36, font(700, size, fam), tc, 'center'); }
function span(x1, x2, y, col = K.ink, lw = 4) { line(x1, y, x2, y, col, lw); line(x1, y - 10, x1, y + 10, col, lw); line(x2, y - 10, x2, y + 10, col, lw); }
const dot = (x, y, r, c) => { X.fillStyle = c; X.beginPath(); X.arc(x, y, r, 0, 7); X.fill(); };
// 节拍：第 k 幕的每句配音一个节拍，pre[i] 为第 i 句之前留给画面的空档。返回每句的起止（幕内时间）；at(B, i, f) 为第 i 句进行到 f 处的时刻
function beatTimes(k, pre = []) { let t = 0; return (window.BEATS[k] || []).map((b, i) => { t += pre[i] ?? .4; const s = t; t += b.dur; return { s, e: t, d: b.dur }; }); }
const at = (B, i, f = 0) => B[i].s + B[i].d * f;
window.BEAT_AT = {};
// 每幕一个结论标题（左上），底部一行字幕由 film.html 画
const head = (t0, s) => IN(t0, () => RICH(s, 96, 150, font(700, 60), K.ink), .5, 0);
const note = (t0, s, y = 952) => IN(t0, () => RICH(s, 96, y, font(600, 34), K.ink2), .5, 0);
// 显存大小的统一写法
const GB = (b) => (b / 2 ** 30).toFixed(b >= 10 * 2 ** 30 ? 1 : 2) + ' GB', MB = (b) => Math.round(b / 2 ** 20) + ' MB';
// 带标签的矩阵块：实色或斜纹，标签放在块内（太窄时放在外侧）
function mat(x, y, w, h, col, label, opt = {}) {
  if (opt.hatch) HR(x, y, w, h, col, opt.bg || K.paper2); else R(x, y, w, h, col);
  if (label) { const f = font(700, opt.size || 30); X.font = f; const tw = X.measureText(label).width;
    if (tw + 24 < w) TX(label, x + w / 2, y + h / 2 + (opt.size || 30) * .36, f, opt.tc || '#fff', 'center');
    else TX(label, x + w + 14, y + h / 2 + (opt.size || 30) * .36, f, opt.oc || col); }
}
