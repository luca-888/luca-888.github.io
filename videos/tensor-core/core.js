// Tensor Core 影片共用：配色、时间工具、绘图图元与播放器（图元沿用 ZeRO 与 FSDP 影片）。
// 数字的出处见 docs/tensor-core-notes.md。
const W = 1920, H = 1080;
const KC = { paper: '#f1ede4', paper2: '#e6e0d3', ink: '#141414', ink2: '#3a3a3a', gray: '#8a8474', lg: '#cfc8b8',
  red: '#e5322d', blue: '#2350d8', yel: '#f4c21b', yelD: '#9a7500', purple: '#7c3ff5', teal: '#0fa391', tealD: '#0b7d70' };
// 语义色：矩阵 A blue，矩阵 B yellow，累加结果 C / D purple，有用的计算 teal，指令开销与瓶颈 red
const COL = { a: KC.blue, b: KC.yel, c: KC.purple, math: KC.teal, ovh: KC.red };
const MONO = '"SF Mono",Menlo,monospace';
const ZH = '"PingFang SC","Helvetica Neue",sans-serif', HN = '"Helvetica Neue","PingFang SC",sans-serif';

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eout = (x) => 1 - Math.pow(1 - clamp(x), 3);
const p = (t, a, b) => ease((t - a) / (b - a));
const po = (t, a, b) => eout((t - a) / (b - a));
const lin = (t, a, b) => clamp((t - a) / (b - a));
const win = (t, a, b, f = 0.4) => p(t, a, a + f) * (1 - p(t, b - f, b));
const mix = (a, b, k) => a + (b - a) * k;

const cv = document.getElementById('cv'), g = cv.getContext('2d');
function rr(x, y, w, h, r, fill, alpha = 1) {
  if (alpha <= 0.003 || w <= 0 || h <= 0) return;
  g.globalAlpha = alpha; g.fillStyle = fill; g.beginPath(); g.roundRect(x, y, w, h, r); g.fill(); g.globalAlpha = 1;
}
function box(x, y, w, h, col, lw = 4, a = 1, dash) {
  if (a <= 0.003) return;
  g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.strokeRect(x, y, w, h); g.setLineDash([]); g.globalAlpha = 1;
}
function txt(s, x, y, { size = 40, color = KC.ink, align = 'left', weight = 400, font = ZH, alpha = 1, base = 'middle' } = {}) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = color; g.textAlign = align; g.textBaseline = base;
  g.font = `${weight} ${size}px ${font}`; g.fillText(s, x, y); g.globalAlpha = 1;
}
function line(x0, y0, x1, y1, color, w = 2, alpha = 1, dash) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash || []);
  g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
}
// {b:蓝} {p:紫} {y:黄} {t:青} {r:红} 标出强调色
function rich(s, x, y, size, base = KC.ink, weight = 700, alpha = 1, align = 'left') {
  if (alpha <= 0.003) return;
  const ACC = { b: KC.blue, r: KC.red, y: KC.yelD, p: KC.purple, t: KC.tealD };
  const parts = []; const re = /\{(\w):([^}]*)\}/g; let last = 0, m;
  while ((m = re.exec(s))) { if (m.index > last) parts.push([s.slice(last, m.index), null]); parts.push([m[2], m[1]]); last = re.lastIndex; }
  if (last < s.length) parts.push([s.slice(last), null]);
  g.font = `${weight} ${size}px ${ZH}`; g.globalAlpha = alpha; g.textBaseline = 'alphabetic'; g.textAlign = 'left';
  const total = parts.reduce((a, [t]) => a + g.measureText(t).width, 0);
  let cx = align === 'center' ? x - total / 2 : x;
  for (const [t, k] of parts) { g.fillStyle = k ? ACC[k] : base; g.fillText(t, cx, y); cx += g.measureText(t).width; }
  g.globalAlpha = 1;
}
let HATCH;
function hatch() {
  if (HATCH) return HATCH;
  const c = document.createElement('canvas'); c.width = c.height = 14; const q = c.getContext('2d');
  q.strokeStyle = KC.red; q.lineWidth = 3;
  q.beginPath(); for (let o = -14; o <= 28; o += 14) { q.moveTo(o, 14); q.lineTo(o + 14, 0); } q.stroke();
  return (HATCH = g.createPattern(c, 'repeat'));
}
function hatchRect(x, y, w, h, a = 1) {
  if (a <= 0.003 || w <= 0 || h <= 0) return;
  g.globalAlpha = a; g.fillStyle = hatch(); g.fillRect(x, y, w, h); g.globalAlpha = 1;
}
function arrow(x0, y0, x1, y1, k, col, w = 5, a = 1, head = 18) {
  if (k <= 0 || a <= 0.003) return;
  const x = mix(x0, x1, k), y = mix(y0, y1, k), ang = Math.atan2(y1 - y0, x1 - x0);
  const bx = x - Math.cos(ang) * head, by = y - Math.sin(ang) * head;
  line(x0, y0, bx, by, col, w, a);
  g.globalAlpha = a; g.fillStyle = col; g.beginPath(); g.moveTo(x, y);
  g.lineTo(bx + Math.sin(ang) * head * 0.6, by - Math.cos(ang) * head * 0.6);
  g.lineTo(bx - Math.sin(ang) * head * 0.6, by + Math.cos(ang) * head * 0.6); g.fill(); g.globalAlpha = 1;
}
// 带白字的实心块
function block(x, y, w, h, fill, label, a = 1, { size = 26, color = '#fff', font = ZH, sub } = {}) {
  rr(x, y, w, h, 0, fill, a);
  if (label === undefined) return;
  const cy = y + h / 2 - (sub ? size * 0.45 : 0);
  txt(label, x + w / 2, cy, { size, weight: 700, color, align: 'center', font, alpha: a });
  if (sub) txt(sub, x + w / 2, cy + size * 1.05, { size: size * 0.78, weight: 500, color, align: 'center', font, alpha: a });
}
function badge(x, y, n, a = 1) {
  if (a <= 0.003) return;
  g.globalAlpha = a; g.fillStyle = KC.ink; g.beginPath(); g.arc(x, y, 22, 0, Math.PI * 2); g.fill(); g.globalAlpha = 1;
  txt(String(n), x, y + 1, { size: 26, weight: 700, color: '#fff', align: 'center', font: HN, alpha: a });
}

// 音效事件：[时刻, 种类]，由 render.mjs 合成
const EVENTS = [];
const ev = (at, kind) => EVENTS.push([at, kind]);

// 播放器：预览时带进度条；被 puppeteer 打开时（navigator.webdriver）只暴露 renderAt
function start(draw, dur) {
  const RENDER = navigator.webdriver || new URLSearchParams(location.search).has('render');
  window.DURATION = dur; window.EVENTS = EVENTS; window.READY = document.fonts.ready;
  window.renderAt = (t) => { g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; draw(t); };
  const bar = document.getElementById('bar');
  if (RENDER) { bar.style.display = 'none'; return; }
  const seek = bar.querySelector('input'), btn = bar.querySelector('button'), tl = bar.querySelector('span');
  seek.max = dur; seek.step = 0.01;
  let now = Number(new URLSearchParams(location.search).get('t')) || 0, playing = new URLSearchParams(location.search).has('play'), last = performance.now();
  const show = () => { window.renderAt(now); seek.value = now; tl.textContent = `${now.toFixed(2)} / ${dur}s`; };
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? '暂停' : '播放'; if (now >= dur) now = 0; last = performance.now(); };
  seek.oninput = () => { now = Number(seek.value); show(); };
  document.addEventListener('keydown', (e) => { if (e.code === 'Space') { e.preventDefault(); btn.click(); } });
  (function loop(ts) {
    if (playing) { now += (ts - last) / 1000; if (now >= dur) { now = dur; playing = false; btn.textContent = '播放'; } }
    last = ts; show(); requestAnimationFrame(loop);
  })(performance.now());
}

function poly(pts, fill, a = 1) {
  if (a <= 0.003) return;
  g.globalAlpha = a; g.fillStyle = fill; g.beginPath(); g.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts.slice(1)) g.lineTo(x, y); g.closePath(); g.fill(); g.globalAlpha = 1;
}
// 矩阵格子：rows × cols，每格 s 像素；fill(i, j) 返回颜色（null 不画）
function grid(x, y, rows, cols, s, fill, { a = 1, gap = 2, stroke } = {}) {
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const c = fill(i, j); if (c) rr(x + j * s + gap / 2, y + i * s + gap / 2, s - gap, s - gap, 0, c, a);
  }
  if (stroke) box(x, y, cols * s, rows * s, stroke, 3, a);
}
// 矩阵乘的立方体：m 行（向下）、n 列（向右）、k 深度（向左上）。
// 正面是输出 C（m × n），顶面是 B（k × n），左面是 A（m × k）。
function cube(x, y, m, n, k, s, { a = 1, d = 0.5, cols = [COL.a, COL.b, COL.c], lines = true } = {}) {
  const dx = -s * d * 0.45, dy = -s * d * 0.3;         // 每一层深度的偏移
  const P = (i, j, kk) => [x + j * s + kk * dx, y + i * s + kk * dy];
  const top = [P(0, 0, 0), P(0, n, 0), P(0, n, k), P(0, 0, k)];
  const left = [P(0, 0, 0), P(0, 0, k), P(m, 0, k), P(m, 0, 0)];
  const front = [P(0, 0, 0), P(0, n, 0), P(m, n, 0), P(m, 0, 0)];
  poly(top, cols[1], a); poly(left, cols[0], a); poly(front, cols[2], a);
  if (!lines) return;
  const L = (p0, p1) => line(p0[0], p0[1], p1[0], p1[1], KC.paper, 1.5, a * 0.75);
  for (let j = 1; j < n; j++) { L(P(0, j, 0), P(m, j, 0)); L(P(0, j, 0), P(0, j, k)); }
  for (let i = 1; i < m; i++) { L(P(i, 0, 0), P(i, n, 0)); L(P(i, 0, 0), P(i, 0, k)); }
  for (let kk = 1; kk < k; kk++) { L(P(0, 0, kk), P(0, n, kk)); L(P(0, 0, kk), P(m, 0, kk)); }
}
// 横向条形：value 按 scale 像素/单位
function hbar(x, y, value, scale, h, col, a = 1) { rr(x, y, Math.max(2, value * scale), h, 0, col, a); return value * scale; }
