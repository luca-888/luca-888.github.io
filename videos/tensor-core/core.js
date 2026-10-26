// Tensor Core 影片共用：配色、时间工具、画布图元、乘加方块堆、芯片图、字幕与播放器。沿用系列的纸色版式（左上角色块标签 + 粗体标题、实心色块）。
const W = 1920, H = 1080;
const C = { paper: '#f1ede4', paper2: '#e6e0d3', ink: '#141414', ink2: '#3a3a3a', gray: '#8a8474', lg: '#cfc8b8',
  red: '#e5322d', blue: '#2350d8', yel: '#f4c21b', yelD: '#9a7500', purple: '#7c3ff5', teal: '#0fa391', tealD: '#0b7d70', tealL: '#3fc0ae' };
const ZH = '"PingFang SC","Helvetica Neue",sans-serif', MONO = '"SF Mono",Menlo,monospace';

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eout = (x) => 1 - Math.pow(1 - clamp(x), 3);
const p = (t, a, b) => ease((t - a) / (b - a));       // 缓入缓出
const po = (t, a, b) => eout((t - a) / (b - a));      // 缓出
const lin = (t, a, b) => clamp((t - a) / (b - a));    // 匀速：用于按真实速率推进的量
const mix = (a, b, k) => a + (b - a) * k;

const cv = document.getElementById('cv'), g = cv.getContext('2d');
function rr(x, y, w, h, fill, alpha = 1) {
  if (alpha <= 0.003 || w <= 0 || h <= 0) return;
  g.globalAlpha = alpha; g.fillStyle = fill; g.fillRect(x, y, w, h); g.globalAlpha = 1;
}
function sr(x, y, w, h, col, lw = 3, alpha = 1) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = lw; g.strokeRect(x, y, w, h); g.globalAlpha = 1;
}
function txt(s, x, y, { size = 30, color = C.ink, align = 'left', weight = 600, font = ZH, alpha = 1 } = {}) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = color; g.textAlign = align; g.textBaseline = 'alphabetic';
  g.font = `${weight} ${size}px ${font}`; g.fillText(s, x, y); g.globalAlpha = 1;
}
const tw = (s, size, weight = 600, font = ZH) => { g.font = `${weight} ${size}px ${font}`; return g.measureText(s).width; };
// 带强调色的一行字：{t:…} teal、{r:…} red、{b:…} blue、{p:…} purple、{y:…} 深黄
function rich(s, x, y, size, weight = 700, alpha = 1) {
  const ACC = { b: C.blue, r: C.red, y: C.yelD, p: C.purple, t: C.tealD };
  const re = /\{(\w):([^}]*)\}/g; let last = 0, m; const parts = [];
  while ((m = re.exec(s))) { if (m.index > last) parts.push([s.slice(last, m.index), C.ink]); parts.push([m[2], ACC[m[1]]]); last = re.lastIndex; }
  if (last < s.length) parts.push([s.slice(last), C.ink]);
  for (const [t, col] of parts) { txt(t, x, y, { size, weight, color: col, alpha }); x += tw(t, size, weight); }
}
function line(x0, y0, x1, y1, color, w = 3, alpha = 1, dash) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash || []);
  g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
}
function arrow(x0, y0, x1, y1, k, col = C.ink, alpha = 1, lw = 4) {
  if (k <= 0 || alpha <= 0.003) return;
  const a = Math.atan2(y1 - y0, x1 - x0), x = mix(x0, x1, k), y = mix(y0, y1, k);
  line(x0, y0, x - 12 * Math.cos(a), y - 12 * Math.sin(a), col, lw, alpha);
  g.globalAlpha = alpha; g.fillStyle = col; g.beginPath(); g.moveTo(x, y);
  g.lineTo(x - 20 * Math.cos(a - 0.45), y - 20 * Math.sin(a - 0.45)); g.lineTo(x - 20 * Math.cos(a + 0.45), y - 20 * Math.sin(a + 0.45)); g.fill(); g.globalAlpha = 1;
}
function poly(pts, fill) { g.fillStyle = fill; g.beginPath(); pts.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))); g.closePath(); g.fill(); }

// ---------- 乘加方块 ----------
// 一个小方块是一次乘加。等轴测的方块堆：nx × ny × nz，(ox, oy) 是底面最靠前的角，s 是方块边长。
// count 个方块按“自下而上、由后向前”的顺序出现；不足一层的部分逐个画，整层的部分画成带网格的整块。
const SETS = {
  teal: { top: C.tealL, left: C.teal, right: C.tealD }, gray: { top: C.lg, left: C.gray, right: '#6f6a5c' },
  ghost: { top: '#ebe6da', left: C.paper2, right: '#ddd6c6' },
};
function stack(ox, oy, nx, ny, nz, s, { count = nx * ny * nz, set = SETS.teal, alpha = 1, grid = true } = {}) {
  if (alpha <= 0.003 || count <= 0) return;
  const P = (i, j, k) => [ox + (i - j) * s * 0.866, oy - (nx + ny - i - j) * s * 0.5 - k * s];
  const per = nx * ny, full = Math.min(nz, Math.floor(count / per)), rest = count >= per * nz ? 0 : count - full * per;
  const gw = s >= 30 ? 1.5 : 1, gcol = C.paper;
  g.globalAlpha = alpha;
  const seg = (a, b) => { g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); };
  if (full > 0) {
    poly([P(0, 0, full), P(nx, 0, full), P(nx, ny, full), P(0, ny, full)], set.top);
    poly([P(0, ny, 0), P(nx, ny, 0), P(nx, ny, full), P(0, ny, full)], set.left);
    poly([P(nx, 0, 0), P(nx, ny, 0), P(nx, ny, full), P(nx, 0, full)], set.right);
    if (grid && s >= 5) {
      g.strokeStyle = gcol; g.lineWidth = gw;
      for (let i = 0; i <= nx; i++) { seg(P(i, 0, full), P(i, ny, full)); seg(P(i, ny, 0), P(i, ny, full)); }
      for (let j = 0; j <= ny; j++) { seg(P(0, j, full), P(nx, j, full)); seg(P(nx, j, 0), P(nx, j, full)); }
      for (let k = 0; k <= full; k++) { seg(P(0, ny, k), P(nx, ny, k)); seg(P(nx, 0, k), P(nx, ny, k)); }
    }
  }
  for (let n = 0; n < rest; n++) {                 // 最上面没铺满的一层
    const i = n % nx, j = Math.floor(n / nx), k = full;
    poly([P(i, j, k + 1), P(i + 1, j, k + 1), P(i + 1, j + 1, k + 1), P(i, j + 1, k + 1)], set.top);
    poly([P(i, j + 1, k), P(i + 1, j + 1, k), P(i + 1, j + 1, k + 1), P(i, j + 1, k + 1)], set.left);
    poly([P(i + 1, j, k), P(i + 1, j + 1, k), P(i + 1, j + 1, k + 1), P(i + 1, j, k + 1)], set.right);
    if (grid && s >= 5) {
      g.strokeStyle = gcol; g.lineWidth = gw;
      seg(P(i, j, k + 1), P(i + 1, j, k + 1)); seg(P(i, j, k + 1), P(i, j + 1, k + 1)); seg(P(i + 1, j, k + 1), P(i + 1, j + 1, k + 1)); seg(P(i, j + 1, k + 1), P(i + 1, j + 1, k + 1));
      seg(P(i, j + 1, k), P(i + 1, j + 1, k)); seg(P(i, j + 1, k), P(i, j + 1, k + 1)); seg(P(i + 1, j + 1, k), P(i + 1, j + 1, k + 1)); seg(P(i + 1, j, k), P(i + 1, j + 1, k)); seg(P(i + 1, j, k), P(i + 1, j, k + 1));
    }
  }
  g.globalAlpha = 1;
}
// 单个方块，(x, y) 是它底面最靠前的角
const cube = (x, y, s, set = SETS.teal, alpha = 1) => stack(x, y, 1, 1, 1, s, { set, alpha });
// 一组小格：n 行 m 列，hi(r, q) 返回该格的颜色（不返回则用 fill）
function cells(x, y, n, m, c, fill, alpha = 1, hi) {
  for (let r = 0; r < n; r++) for (let q = 0; q < m; q++) rr(x + q * c, y + r * c, c - 6, c - 6, (hi && hi(r, q)) || fill, alpha);
}

// ---------- 芯片 ----------
// 自绘的 Blackwell 芯片，结构照 NVIDIA 官方渲染图（NVIDIA newsroom 的 Blackwell 渲染图）：
// 金色加强环 → 基板（走线与两侧的小元件）→ interposer → 上下各 4 组 HBM → 中间两个并排的 die。
// die 内按 8 个 GPC × 20 格画出 160 格，对应 Blackwell 最多 160 个 SM。build 0→1 逐层出现，lit 是已点亮的格数，hot 高亮其中一格。
const CHIP_HOT = 2 * 40 + 20 + 2 * 5 + 3;   // 高亮格的序号（die 2、上排右侧 GPC、第 3 行第 4 列）
const mixc = (c0, c1, t) => '#' + [0, 1, 2].map((i) => Math.round(parseInt(c0.substr(1 + i * 2, 2), 16) * (1 - t) + parseInt(c1.substr(1 + i * 2, 2), 16) * t).toString(16).padStart(2, '0')).join('');
function chip(x, y, w, { build = 1, lit = 160, hot = 0, alpha = 1, detail = true, off = null, paint = null } = {}) {
  if (alpha <= 0.003) return;
  const u = w / 100, R = (a, b, c, d, f, al = 1) => rr(x + a * u, y + b * u, c * u, d * u, f, al * alpha);
  const GOLD = '#c8a544', GOLD2 = '#e3c56a', SUB = '#17171a', TR = '#2e2e33';
  const tint = (t) => (t < 0.5 ? mixc('#c2a85a', '#58b3a2', t * 2) : mixc('#58b3a2', '#9483d6', t * 2 - 1));   // 金 → 青 → 紫，对应官方图 die 上的虹彩
  const rnd = (i) => { const v = Math.sin(i * 127.1 + 311.7) * 43758.5453; return v - Math.floor(v); };
  const L = (a, b) => po(build, a, b);            // 各层出现的进度
  const ring = L(0, 0.2), sub = L(0.12, 0.38), ip = L(0.32, 0.52), hbm = L(0.45, 0.7), die = L(0.62, 0.9);
  R(1.2, 1.2, 100, 92, 'rgba(20,20,20,0.14)', ring); R(0, 0, 100, 92, GOLD, ring); R(0, 0, 100, 1.1, GOLD2, ring); R(0, 0, 1.1, 92, GOLD2, ring);
  R(5, 5, 90, 82, SUB, sub);
  if (detail && sub > 0) {
    for (let i = 0; i < 46; i++) { const yy = 9 + i * 1.62, d = (i % 5) * 0.9; R(7 + d, yy, 11.2 - d, 0.28, TR, sub); R(81.8, yy, 11.2 - d, 0.28, TR, sub); }
    for (let i = 0; i < 26; i++) for (let j = 0; j < 4; j++) { R(21 + i * 2.25, 6.2 + j * 0.9, 0.5, 0.4, TR, sub); R(21 + i * 2.25, 82.6 + j * 0.9, 0.5, 0.4, TR, sub); }
    for (let i = 0; i < 22; i++) {
      const yy = 12 + i * 3.1 + rnd(i) * 1.2;
      if (rnd(i + 40) > 0.3) R(8.6 + rnd(i + 9) * 1.2, yy, 1.5, 1.0, GOLD, sub);
      if (rnd(i + 80) > 0.3) R(89.6 + rnd(i + 5) * 1.2, yy, 1.5, 1.0, GOLD, sub);
      if (rnd(i + 17) > 0.6) { R(11.4, yy + 0.2, 2.4, 0.7, GOLD2, sub); R(86, yy + 0.2, 2.4, 0.7, GOLD2, sub); }
    }
  }
  R(19, 9.5, 62, 73, '#202024', ip);
  [11.5, 67.5].forEach((yy, a) => [21.5, 51].forEach((x0, b) => { for (let k = 0; k < 2; k++) {
    const al = po(hbm, (a * 4 + b * 2 + k) / 12, (a * 4 + b * 2 + k) / 12 + 0.4), xx = x0 + k * 13.9;
    R(xx, yy, 13.6, 13, GOLD, al); R(xx, yy, 13.6, 0.7, GOLD2, al); R(xx + 0.9, yy + 1.6, 11.8, 10.4, '#d4b352', al);
  } }));
  const DX = 20.5, DY = 26.5, DW = 59, DH = 39;
  R(DX, DY, DW, DH, '#35342e', die); R(DX, DY, DW, 0.5, '#8d8566', die);
  let n = 0;
  for (let d = 0; d < 2; d++) for (let gr = 0; gr < 2; gr++) for (let gc = 0; gc < 2; gc++) {
    const gx = DX + 1.4 + d * 29.4 + gc * 13.6, gy = DY + 1.8 + gr * 19.6;
    R(gx - 0.35, gy - 0.35, 13.3, 16.5, '#26251f', die);
    for (let r = 0; r < 4; r++) for (let q = 0; q < 5; q++, n++) {
      const cx = gx + q * 2.52, cy = gy + r * 3.96, t = (cx - DX) / DW * 0.85 + (cy - DY) / DH * 0.15;
      const dead = off && off.has(n), on = dead ? 0 : die * clamp(lit - n), isHot = n === CHIP_HOT ? hot : 0;
      R(cx, cy, 2.2, 3.6, '#4a483f', die * (1 - clamp(lit - n)));
      R(cx, cy, 2.2, 3.6, tint(t), on);
      if (detail) { R(cx + 0.25, cy + 0.3, 1.7, 1.25, 'rgba(255,255,255,0.22)', on); R(cx + 0.25, cy + 2.2, 0.75, 1.1, 'rgba(0,0,0,0.25)', on); R(cx + 1.2, cy + 2.2, 0.75, 1.1, 'rgba(0,0,0,0.25)', on); }
      if (dead) R(cx, cy, 2.2, 3.6, '#2a2924', die);
      if (paint) { const pc = paint(n); if (pc) R(cx, cy, 2.2, 3.6, pc[0], die * pc[1]); }
      if (isHot > 0) { R(cx, cy, 2.2, 3.6, C.teal, isHot); sr(x + (cx - 0.3) * u, y + (cy - 0.3) * u, 2.8 * u, 4.2 * u, '#fff', 0.3 * u, isHot * alpha); }
    }
  }
  R(DX + 1, DY + 18.3, DW - 2, 2.4, '#1c1b17', die);
  if (detail) for (let i = 0; i < 44; i++) R(DX + 1.6 + i * 1.3, DY + 18.8, 0.8, 1.4, i % 7 === 3 ? GOLD2 : '#8d8566', die);
  R(DX + DW / 2 - 0.45, DY, 0.9, DH, '#14130f', die);
}
// 第 n 格在画面上的位置（左上角与宽高），顺序与 chip() 的编号一致
function chipCell(x, y, w, n) {
  const u = w / 100, d = Math.floor(n / 80), i = n % 20;
  const gpc = Math.floor(n / 20) % 4, grr = Math.floor(gpc / 2), gcc = gpc % 2, r = Math.floor(i / 5), q = i % 5;
  const gx = 20.5 + 1.4 + d * 29.4 + gcc * 13.6, gy = 26.5 + 1.8 + grr * 19.6;
  return [x + (gx + q * 2.52) * u, y + (gy + r * 3.96) * u, 2.2 * u, 3.6 * u];
}
// B200 启用 148 个 SM：每个 GPC 关掉的格数取 SemiAnalysis 的估计（20,20,20,18 / 18,18,18,16），关的位置为示意
const B200_OFF = new Set([[3, 2], [4, 2], [5, 2], [6, 2], [7, 4]].flatMap(([gpc, k]) => Array.from({ length: k }, (_, i) => gpc * 20 + 19 - i * 3)));
const B200_ON = Array.from({ length: 160 }, (_, n) => n).filter((n) => !B200_OFF.has(n));
// 高亮格在画面上的中心
const chipHotXY = (x, y, w) => { const u = w / 100; return [x + (20.5 + 1.4 + 29.4 + 13.6 + 3 * 2.52 + 1.1) * u, y + (26.5 + 1.8 + 2 * 3.96 + 1.8) * u]; };

// ---------- 字幕 ----------
// 配音逐句烧录在画面底部：一句按标点切成短句，按读音长度分配这句的发声时长。
// 有录音（vo.js，由 vo.py 生成）用实测时长；没有录音时按估计：汉字 0.21 s，英文词或数字 0.32 s，句中标点 0.15 s（语速 1.15 下的经验值）。
const SPOKEN = (x) => (x.match(/[一-鿿]/g) || []).length * 0.21 + (x.match(/[A-Za-z0-9][A-Za-z0-9._\-]*/g) || []).length * 0.32 + (x.match(/[，；：、]/g) || []).length * 0.15;
function cues(text, t0, dur) {
  const parts = text.split(/(?<=[，。？！；：])/).map((x) => x.trim()).filter(Boolean);
  const all = parts.reduce((a, x) => a + SPOKEN(x), 0);
  let t = t0; return parts.map((x) => { const d = dur * SPOKEN(x) / all, c = [t, t + d, x.replace(/[，。；：]$/, '')]; t += d; return c; });
}
function subtitle(beats, voice, u) {
  for (let j = 0; j < beats.length; j++) {
    const [t0, , text] = beats[j]; if (!text) continue;
    const d = voice?.[j]?.text === text ? voice[j].dur : SPOKEN(text);
    const next = j + 1 < beats.length ? beats[j + 1][0] : Infinity;
    for (const [a, b, s] of cues(text, t0, d)) {
      if (u < a || u >= Math.min(next, b + (s === cues(text, t0, d).at(-1)[2] ? 0.4 : 0))) continue;
      const size = 40, w = tw(s, size, 600) + 48;
      rr(W / 2 - w / 2, 1004, w, 60, 'rgba(20,20,20,0.78)');
      txt(s, W / 2, 1047, { size, weight: 600, color: '#fff', align: 'center' });
      return;
    }
  }
}

// ---------- 等轴测长方体 ----------
// 与 stack() 同一套投影；(ox, oy) 是 i = j = k = 0 的后角在地面上的位置，s 是一个单位的边长。
// 画 [i0, i0 + di] × [j0, j0 + dj] × [k0, k0 + dk]，只画看得见的三个面（顶、j 端、i 端）。
const Q = (ox, oy, s, i, j, k) => [ox + (i - j) * s * 0.866, oy + (i + j) * s * 0.5 - k * s];
function box(ox, oy, s, i0, j0, k0, di, dj, dk, set = SETS.teal, alpha = 1, edge = null) {
  if (alpha <= 0.003) return;
  const i1 = i0 + di, j1 = j0 + dj, k1 = k0 + dk, P = (i, j, k) => Q(ox, oy, s, i, j, k);
  g.globalAlpha = alpha;
  poly([P(i0, j0, k1), P(i1, j0, k1), P(i1, j1, k1), P(i0, j1, k1)], set.top);
  poly([P(i0, j1, k0), P(i1, j1, k0), P(i1, j1, k1), P(i0, j1, k1)], set.left);
  poly([P(i1, j0, k0), P(i1, j1, k0), P(i1, j1, k1), P(i1, j0, k1)], set.right);
  if (edge) {
    g.strokeStyle = edge; g.lineWidth = 2; g.beginPath();
    [[P(i0, j0, k1), P(i1, j0, k1), P(i1, j1, k1), P(i0, j1, k1), P(i0, j0, k1)], [P(i0, j1, k1), P(i0, j1, k0), P(i1, j1, k0), P(i1, j0, k0), P(i1, j0, k1)], [P(i1, j1, k0), P(i1, j1, k1)]]
      .forEach((ps) => ps.forEach((q, n) => (n ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))));
    g.stroke();
  }
  g.globalAlpha = 1;
}

// ---------- 占位 ----------
// 需要实测或实机材料、草稿期尚未获取的位置：虚线框 + “占位”标签，写明这里将放什么、需要什么。
function placeholder(x, y, w, h, title, note, alpha = 1) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = 'rgba(255,255,255,0.55)'; g.fillRect(x, y, w, h);
  g.strokeStyle = C.red; g.lineWidth = 2.5; g.setLineDash([12, 8]); g.strokeRect(x, y, w, h); g.setLineDash([]); g.globalAlpha = 1;
  rr(x, y, 92, 38, C.red, alpha); txt('占位', x + 46, y + 28, { size: 24, weight: 700, color: '#fff', align: 'center', alpha });
  txt(title, x + 112, y + 29, { size: 26, weight: 700, color: C.ink, alpha });
  (Array.isArray(note) ? note : [note]).forEach((s, i) => txt(s, x + 20, y + 72 + i * 32, { size: 22, weight: 500, color: C.ink2, alpha }));
}

// ---------- 播放器：预览时带进度条；被 puppeteer 打开时只暴露 renderAt ----------
function start(draw, dur) {
  const RENDER = navigator.webdriver || new URLSearchParams(location.search).has('render');
  window.DUR = dur; window.READY = document.fonts.ready;
  window.renderAt = (t) => { g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; draw(t); };
  const bar = document.getElementById('bar');
  if (RENDER) { bar.style.display = 'none'; return; }
  const seek = bar.querySelector('input'), btn = bar.querySelector('button'), tl = bar.querySelector('span');
  seek.max = dur; seek.step = 0.01;
  let now = Number(new URLSearchParams(location.search).get('t')) || 0, playing = false, last = 0;
  const show = () => { window.renderAt(now); seek.value = now; tl.textContent = `${now.toFixed(2)} / ${dur.toFixed(1)}s`; };
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? '暂停' : '播放'; if (now >= dur) now = 0; last = performance.now(); };
  seek.oninput = () => { now = Number(seek.value); show(); };
  document.addEventListener('keydown', (e) => { if (e.code === 'Space') { e.preventDefault(); btn.click(); } });
  (function loop(ts) {
    if (playing) { now += (ts - last) / 1000; last = ts; if (now >= dur) { now = dur; playing = false; btn.textContent = '播放'; } }
    show(); requestAnimationFrame(loop);
  })(performance.now());
}
