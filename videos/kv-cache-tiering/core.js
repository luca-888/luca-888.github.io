// KV cache 分层存储影片共用：配色、时间工具、画布图元与播放器。沿用系列的纸色版式（左上角色块标签 + 粗体标题、实心色块）；改自 linear-attention/core.js。
const W = 1920, H = 1080;
const C = { paper: '#f1ede4', paper2: '#e6e0d3', ink: '#141414', ink2: '#3a3a3a', gray: '#8a8474', lg: '#cfc8b8',
  red: '#e5322d', blue: '#2350d8', yel: '#f4c21b', yelD: '#9a7500', purple: '#7c3ff5', purpleL: '#b79af9', teal: '#0fa391', tealD: '#0b7d70', amber: '#e08a1e', green: '#4c9a2a' };
const ZH = '"PingFang SC","Helvetica Neue",sans-serif', MONO = '"SF Mono",Menlo,monospace';

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eout = (x) => 1 - Math.pow(1 - clamp(x), 3);
const p = (t, a, b) => ease((t - a) / (b - a));
const po = (t, a, b) => eout((t - a) / (b - a));
const lin = (t, a, b) => clamp((t - a) / (b - a));
const mix = (a, b, k) => a + (b - a) * k;
const mixc = (c0, c1, t) => '#' + [0, 1, 2].map((i) => Math.round(parseInt(c0.substr(1 + i * 2, 2), 16) * (1 - t) + parseInt(c1.substr(1 + i * 2, 2), 16) * t).toString(16).padStart(2, '0')).join('');

const cv = document.getElementById('cv'), g = cv.getContext('2d');
function rr(x, y, w, h, fill, alpha = 1) {
  if (alpha <= 0.003 || w <= 0 || h <= 0) return;
  g.globalAlpha = alpha; g.fillStyle = fill; g.fillRect(x, y, w, h); g.globalAlpha = 1;
}
function sr(x, y, w, h, col, lw = 3, alpha = 1, dash) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash || []); g.strokeRect(x, y, w, h); g.setLineDash([]); g.globalAlpha = 1;
}
function txt(s, x, y, { size = 30, color = C.ink, align = 'left', weight = 600, font = ZH, alpha = 1 } = {}) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = color; g.textAlign = align; g.textBaseline = 'alphabetic';
  g.font = `${weight} ${size}px ${font}`; g.fillText(s, x, y); g.globalAlpha = 1;
}
const tw = (s, size, weight = 600, font = ZH) => { g.font = `${weight} ${size}px ${font}`; return g.measureText(s).width; };
// 带强调色的一行字：{t:…} teal、{r:…} red、{b:…} blue、{p:…} purple、{y:…} 深黄、{g:…} gray、{a:…} amber
function rich(s, x, y, size, weight = 700, alpha = 1, base = C.ink) {
  const ACC = { b: C.blue, r: C.red, y: C.yelD, p: C.purple, t: C.tealD, g: C.gray, a: C.amber };
  const re = /\{(\w):([^}]*)\}/g; let last = 0, m; const parts = [];
  while ((m = re.exec(s))) { if (m.index > last) parts.push([s.slice(last, m.index), base]); parts.push([m[2], ACC[m[1]]]); last = re.lastIndex; }
  if (last < s.length) parts.push([s.slice(last), base]);
  for (const [t, col] of parts) { txt(t, x, y, { size, weight, color: col, alpha }); x += tw(t, size, weight); }
}
function line(x0, y0, x1, y1, color, w = 3, alpha = 1, dash) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash || []);
  g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
}
function arrow(x0, y0, x1, y1, col = C.ink, alpha = 1, lw = 4) {
  if (alpha <= 0.003) return;
  const a = Math.atan2(y1 - y0, x1 - x0);
  line(x0, y0, x1 - 12 * Math.cos(a), y1 - 12 * Math.sin(a), col, lw, alpha);
  g.globalAlpha = alpha; g.fillStyle = col; g.beginPath(); g.moveTo(x1, y1);
  g.lineTo(x1 - 20 * Math.cos(a - 0.45), y1 - 20 * Math.sin(a - 0.45)); g.lineTo(x1 - 20 * Math.cos(a + 0.45), y1 - 20 * Math.sin(a + 0.45)); g.fill(); g.globalAlpha = 1;
}
// 指向画面上某一处的标注：一个小圆点、一条细线、线尾的一行字
function callout(px, py, tx, ty, s, { color = C.ink, size = 26, align = 'left', alpha = 1 } = {}) {
  line(px, py, tx, ty, color, 2, alpha);
  g.globalAlpha = alpha; g.fillStyle = color; g.beginPath(); g.arc(px, py, 5, 0, 7); g.fill(); g.globalAlpha = 1;
  txt(s, tx + (align === 'left' ? 10 : -10), ty + size * 0.35, { size, color, align, alpha, weight: 700 });
}

// ---------- 对着同一条基线的条 ----------
// parts：[数量, 颜色]，依次从左排；full 为整条对应的数量。
const MX = 1440, MW = 384;
function meter(y, label, parts, value, { note = '', alpha = 1, x = MX, w = MW, full = 1, color = C.ink } = {}) {
  txt(label, x, y, { size: 26, weight: 700, alpha }); txt(value, x + w, y, { size: 28, weight: 700, align: 'right', alpha, color });
  rr(x, y + 14, w, 32, C.paper2, alpha);
  let at = x;
  for (const [n, col] of parts) { if (n <= 0) continue; const bw = Math.max(3, w * n / full); rr(at, y + 14, bw, 32, col, alpha); at += bw; }
  if (note) txt(note, x, y + 14 + 32 + 28, { size: 21, weight: 500, color: C.gray, alpha });
}
// 代码面板：lines 为字符串数组，{t:…} 等强调色可用
function code(x, y, w, lines, { size = 22, alpha = 1, title = '' } = {}) {
  const lh = size * 1.45, h = lines.length * lh + 28 + (title ? 34 : 0);
  rr(x, y, w, h, '#fbf9f4', alpha); sr(x, y, w, h, C.lg, 2, alpha);
  if (title) txt(title, x + 18, y + 30, { size: 19, weight: 500, color: C.gray, font: MONO, alpha });
  lines.forEach((s, i) => richMono(s, x + 18, y + (title ? 34 : 0) + 14 + lh * (i + 0.8), size, alpha));
  return h;
}
function richMono(s, x, y, size, alpha) {
  const ACC = { b: C.blue, r: C.red, y: C.yelD, p: C.purple, t: C.tealD, g: C.gray, a: C.amber };
  const re = /\{(\w):([^}]*)\}/g; let last = 0, m; const parts = [];
  while ((m = re.exec(s))) { if (m.index > last) parts.push([s.slice(last, m.index), C.ink2]); parts.push([m[2], ACC[m[1]]]); last = re.lastIndex; }
  if (last < s.length) parts.push([s.slice(last), C.ink2]);
  for (const [t, col] of parts) { txt(t, x, y, { size, weight: 500, color: col, font: MONO, alpha }); x += tw(t, size, 500, MONO); }
}

// ---------- 图片 ----------
const IMG = {}; const imgLoads = [];
function loadImg(name, src) { const im = new Image(); IMG[name] = im; imgLoads.push(new Promise((r) => { im.onload = r; im.onerror = r; })); im.src = src; }

// ---------- 播放器：预览时带进度条；被 puppeteer 打开时只暴露 renderAt ----------
function start(draw, dur) {
  const RENDER = navigator.webdriver || new URLSearchParams(location.search).has('render');
  window.DUR = dur; window.READY = Promise.all([document.fonts.ready, ...imgLoads]);
  window.renderAt = (t) => { g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; draw(t); };
  const bar = document.getElementById('bar');
  if (RENDER) { bar.style.display = 'none'; cv.style.cssText = 'width:1920px;height:1080px;max-height:none'; return; }
  const seek = bar.querySelector('input'), btn = bar.querySelector('button'), tl = bar.querySelector('span');
  seek.max = dur; seek.step = 0.01;
  let now = Number(new URLSearchParams(location.search).get('t')) || 0, playing = false, last = 0;
  const show = () => { window.renderAt(now); seek.value = now; tl.textContent = `${now.toFixed(2)} / ${dur.toFixed(1)}s`; };
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? '暂停' : '播放'; if (now >= dur) now = 0; last = performance.now(); };
  seek.oninput = () => { now = Number(seek.value); show(); };
  document.addEventListener('keydown', (e) => { if (e.code === 'Space') { e.preventDefault(); btn.click(); } });
  window.READY.then(() => (function loop(ts) {
    if (playing) { now += (ts - last) / 1000; last = ts; if (now >= dur) { now = dur; playing = false; btn.textContent = '播放'; } }
    show(); requestAnimationFrame(loop);
  })(performance.now()));
}
