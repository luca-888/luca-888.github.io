// 稀疏 Attention 影片共用：配色、时间工具、画布图元、token 页与播放器。沿用系列的纸色版式（左上角色块标签 + 粗体标题、实心色块）。
const W = 1920, H = 1080;
const C = { paper: '#f1ede4', paper2: '#e6e0d3', ink: '#141414', ink2: '#3a3a3a', gray: '#8a8474', lg: '#cfc8b8',
  red: '#e5322d', blue: '#2350d8', yel: '#f4c21b', yelD: '#9a7500', purple: '#7c3ff5', purpleL: '#b79af9', teal: '#0fa391', tealD: '#0b7d70' };
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
// 带强调色的一行字：{t:…} teal、{r:…} red、{b:…} blue、{p:…} purple、{y:…} 深黄、{g:…} gray
function rich(s, x, y, size, weight = 700, alpha = 1, base = C.ink) {
  const ACC = { b: C.blue, r: C.red, y: C.yelD, p: C.purple, t: C.tealD, g: C.gray };
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

// ---------- token 页 ----------
// 贯穿全片的图形：上下文画成一页格子，一格是一个 token 的 KV entry，按阅读顺序从左上排到右下，最新的 token 在右下角。
// 128K 上下文 = 512 列 × 256 行；格子的个数都按真实个数画。colorOf(i) 返回第 i 格的颜色，返回空则不画。
const N = 131072, COLS = 512, ROWS = 256;
const PAGE = { x: 96, y: 296, pitch: 2.5 };          // 各幕里这一页的位置：1280 × 640
const PW = COLS * PAGE.pitch, PH = ROWS * PAGE.pitch, PR = PAGE.x + PW, PB = PAGE.y + PH;
const pageCache = {};
function pageImg(key, cols, rows, colorOf, S = 5) {
  if (pageCache[key]) return pageCache[key];
  const c = document.createElement('canvas'); c.width = cols * S; c.height = rows * S;
  const q = c.getContext('2d'), groups = {};
  for (let i = 0; i < cols * rows; i++) { const col = colorOf(i); if (col) (groups[col] ||= []).push(i); }
  for (const col in groups) { q.fillStyle = col; for (const i of groups[col]) q.fillRect((i % cols) * S, Math.floor(i / cols) * S, S - 1, S - 1); }
  return (pageCache[key] = c);
}
// w、h 缺省为 cols × pitch、rows × pitch；r0、r1 只画其中的若干行（可为小数），用于逐行扫过
function page(key, colorOf, { x = PAGE.x, y = PAGE.y, cols = COLS, rows = ROWS, pitch = PAGE.pitch, alpha = 1, S = 5, w = cols * pitch, h = rows * pitch, r0 = 0, r1 = rows } = {}) {
  if (alpha <= 0.003 || r1 - r0 <= 0.001) return;
  g.globalAlpha = alpha; g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  g.drawImage(pageImg(key, cols, rows, colorOf, S), 0, r0 * S, cols * S, (r1 - r0) * S, x, y + h * r0 / rows, w, h * (r1 - r0) / rows); g.globalAlpha = 1;
}
// 按内容选出的 n 个位置（示意）：一部分成小段聚在一起，一部分零散；个数是准确的
function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function pick(n, total, seed) {
  const r = rng(seed), set = new Set();
  while (set.size < n * 0.6) { const at = Math.floor(r() * total), len = 4 + Math.floor(r() * 36); for (let j = 0; j < len && set.size < n * 0.6; j++) if (at + j < total) set.add(at + j); }
  while (set.size < n) set.add(Math.floor(r() * total));
  return set;
}

// ---------- 读取与缓存的两条 ----------
// 两条对着同一条基线：整条 = 128K 个 entry。parts：[个数, 颜色, 条高占比（默认 1）]，依次从左排。
const MX = 1440, MW = 384;
function meter(y, label, parts, value, { note = '', alpha = 1, x = MX, w = MW, full = N } = {}) {
  txt(label, x, y, { size: 28, weight: 700, alpha }); txt(value, x + w, y, { size: 30, weight: 700, align: 'right', alpha });
  rr(x, y + 16, w, 36, C.paper2, alpha);
  let at = x, down = 0;
  for (const [n, col, hk = 1] of parts) {
    if (n <= 0) continue;
    if (hk < 1) { rr(x, y + 16 + 36 + 4 + down, Math.max(3, w * n / full), 36 * hk, col, alpha); down += 36 * hk + 4; continue; }
    const bw = Math.max(3, w * n / full); rr(at, y + 16, bw, 36, col, alpha); at += bw;
  }
  if (note) txt(note, x, y + 16 + 36 + down + 32, { size: 22, weight: 500, color: C.gray, alpha });
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
