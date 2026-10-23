// 矩阵乘法阶梯影片共用：配色、时间工具、画布图元、矩阵网格、地址条、代码框、终端、阶梯图、时间线、字幕与播放器。
// 版式沿用系列（纸色底、左上角色块标签 + 粗体标题、实心色块）；网格的画法与配色沿用 inference-study 的 1.3 讲解页：从显存读取 = 橙，片上数据 = 绿，选中的线程 = 蓝，同一 warp = 浅蓝。
const W = 1920, H = 1080;
const C = { paper: '#f1ede4', paper2: '#e6e0d3', ink: '#141414', ink2: '#3a3a3a', gray: '#8a8474', lg: '#cfc8b8',
  red: '#e5322d', blue: '#2350d8', yel: '#f4c21b', yelD: '#9a7500', purple: '#7c3ff5', teal: '#0fa391', tealD: '#0b7d70', tealL: '#3fc0ae',
  hot: '#c4561d', hotL: '#eeb48f', use: '#2f8a57', useL: '#a9d6b9', thr: '#2f6fdc', warp: '#a9c2ee', cell: '#e2ddd0' };
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
// 带强调色的一行字：{t:…} teal、{r:…} red、{b:…} blue、{p:…} purple、{y:…} 深黄、{o:…} 显存橙、{g:…} 片上绿
function rich(s, x, y, size, weight = 700, alpha = 1) {
  const ACC = { b: C.blue, r: C.red, y: C.yelD, p: C.purple, t: C.tealD, o: C.hot, g: C.use };
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
  // 渲染时画布按 1920 × 1080 原尺寸铺满视口（预览页的进度条留出的 44 px 会让截图出现黑边）
  if (RENDER) { bar.style.display = 'none'; cv.style.cssText = 'display:block;width:1920px;height:1080px;max-height:none'; return; }
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

// ---------- 本片的图元 ----------
// 方格网格：n × n，格宽 c（含间隙 gap），fn(r, q) 返回该格颜色，不返回则为底色
function grid(x, y, n, c, fn, alpha = 1, gap = 2, m = n) {
  for (let r = 0; r < n; r++) for (let q = 0; q < m; q++) rr(x + q * c, y + r * c, c - gap, c - gap, (fn && fn(r, q)) || C.cell, alpha);
}
// 一个实心节点：色块 + 居中白字（可带一行小字）
function node(x, y, w, h, s, col, alpha = 1, sub = '', size = 30) {
  rr(x, y, w, h, col, alpha);
  txt(s, x + w / 2, y + h / 2 + (sub ? -2 : size * 0.36), { size, weight: 800, color: '#fff', align: 'center', alpha });
  if (sub) txt(sub, x + w / 2, y + h / 2 + 30, { size: 20, weight: 600, color: 'rgba(255,255,255,0.85)', align: 'center', alpha });
}
// 小标签：色块底 + 白字
function pill(s, x, y, col, alpha = 1, size = 24) { const w = tw(s, size, 700) + 24; rr(x, y - size - 6, w, size + 16, col, alpha); txt(s, x + 12, y + 2, { size, weight: 700, color: '#fff', alpha }); return w; }
// 代码框：lines 的每项是字符串或 [[片段, 颜色], …]；hl(i) 返回该行底色
function code(x, y, lines, { size = 21, lh = 36, w = 760, hl = null, alpha = 1, title = '' } = {}) {
  if (alpha <= 0.003) return;
  const top = title ? 40 : 0;
  rr(x, y, w, lines.length * lh + 24 + top, '#e7e1d4', alpha);
  if (title) txt(title, x + 18, y + 30, { size: 20, weight: 700, color: C.gray, alpha, font: MONO });
  lines.forEach((ln, i) => {
    const yy = y + top + 12 + i * lh, bg = hl && hl(i);
    if (bg) rr(x, yy, w, lh, bg, alpha);
    let xx = x + 18;
    for (const [s, col] of (typeof ln === 'string' ? [[ln, C.ink2]] : ln)) { txt(s, xx, yy + lh * 0.7, { size, weight: 500, color: col || C.ink2, font: MONO, alpha }); xx += tw(s, size, 500, MONO); }
  });
}
// 终端：深色底，等宽字原样输出；hl(i) 为真时该行高亮
function term(x, y, w, lines, { size = 19, lh = 32, hl = null, alpha = 1 } = {}) {
  if (alpha <= 0.003) return;
  rr(x, y, w, lines.length * lh + 28, '#1d1c19', alpha);
  lines.forEach((s, i) => {
    const yy = y + 14 + i * lh, on = hl && hl(i);
    if (on) rr(x + 6, yy, w - 12, lh, 'rgba(244,194,27,0.22)', alpha * on);
    txt(s, x + 18, yy + lh * 0.72, { size, weight: 500, color: s.startsWith('$') ? '#9fd3b4' : on ? '#fff' : '#cfc8b8', font: MONO, alpha });
  });
}
// 时间线的一条：segs = [[起, 止, 颜色, 文字]]，单位为 0..T；reveal 为 0..1 的揭示进度
function lane(x, y, w, h, T, segs, reveal = 1, alpha = 1, name = '', nameCol = C.ink, ns = 20) {
  if (name) txt(name, x - 12, y + h * 0.68, { size: ns, weight: 700, color: nameCol, align: 'right', alpha });
  rr(x, y, w, h, C.paper2, alpha);
  const cut = T * reveal;
  for (const [a, b, col, s] of segs) {
    if (a >= cut) continue;
    const bb = Math.min(b, cut), x0 = x + a / T * w, x1 = x + bb / T * w;
    rr(x0 + 1, y, x1 - x0 - 2, h, col, alpha);
    if (s && bb === b && x1 - x0 > tw(s, 20, 700) + 8) txt(s, (x0 + x1) / 2, y + h * 0.66, { size: 20, weight: 700, color: '#fff', align: 'center', alpha });
  }
}

// ---------- 阶梯 ----------
// 主线上的 15 版，8192 阶实测 TFLOPS；f = FP32 线（蓝），t = Tensor Core 线（teal）。
// cut：这一版去除的瓶颈（结尾标注），cat：瓶颈所在位置（结尾的色带）
const STEPS = [
  ['naive', 5.41, 'f', '起点', -1], ['smem16', 8.22, 'f', '显存读取', 0], ['smem32', 9.22, 'f', '复用次数', 0],
  ['reg8x8', 24.72, 'f', '共享内存带宽', 1], ['vec', 35.28, 'f', '访存指令', 1], ['warptile', 37.00, 'f', 'bank 冲突', 1], ['dbuf', 44.30, 'f', '显存延迟', 0],
  ['wmma', 93.29, 't', 'FP32 峰值', 2], ['mma', 269.41, 't', 'bank 冲突', 1], ['mma_async', 308.77, 't', '搬运延迟', 0],
  ['wgmma', 484.21, 't', '寄存器中转', 1], ['wgmma_ws', 479.74, 't', '同步等待', 2], ['wgmma_l2', 740.05, 't', 'L2 容量', 3],
  ['wgmma_pipe', 740.51, 't', '无变化', 4], ['wgmma_persist', 657.00, 't', '变慢 11%', 4],
  ['wgmma_cluster', 757.26, 't', 'L2 读出量', 3], ['wgmma_tstore', 777.63, 't', '写回', 3],
];
const CUB = { f: 51.74, t: 781.86 };
const CATS = [['显存', C.hot], ['片上存储', C.use], ['计算单元与同步', C.purple], ['L2 与写回', C.blue], ['未去除瓶颈', C.gray]];
const pctOf = (i) => STEPS[i][1] / CUB[STEPS[i][2]] * 100;
const fmtP = (x) => (x >= 99 && x < 100 ? x.toFixed(1) : Math.round(x)) + '%';
const fmtT = (v) => (v >= 100 ? (Math.abs(v - 777.63) < 0.3 ? '777.6' : String(Math.round(v))) : v.toFixed(1));
// ladder：upto 为已点亮的版数（小数部分是最后一级的生长进度）；ghost 为未点亮各级轮廓的不透明度；
// cub = [FP32 基线, BF16 基线] 的不透明度；names 显示版本名；ann(i) 返回第 i 级瓶颈标注的不透明度；band 为色带不透明度
function ladder(x, y, w, h, { upto = 1, ghost = 0.5, cub = [0, 0], names = false, values = 'last', ann = null, band = 0, alpha = 1, axis = true, flash = -1, size = 1, cubLabel = true, pct = false, ring = -1, skip = [] } = {}) {
  if (alpha <= 0.003) return;
  const Y = (v) => y + h - h * Math.log10(Math.max(1, v)) / 3, vis = STEPS.map((_, i) => i).filter((i) => !skip.includes(i)), n = vis.length, sw = w / n, gap = Math.max(3, sw * 0.12);
  if (axis) for (const [v, s] of [[1, '1'], [10, '10'], [100, '100'], [1000, '1000']]) {
    line(x, Y(v), x + w, Y(v), C.lg, 1.5, alpha);
    txt(s, x - 12, Y(v) + 8 * size, { size: 20 * size, weight: 600, color: C.gray, align: 'right', alpha, font: MONO });
  }
  const full = Math.floor(upto), frac = upto - full;
  STEPS.forEach(([name, v, line_, cut, cat], i) => {
    if (skip.includes(i)) return;
    const bx = x + vis.indexOf(i) * sw + gap, bw = sw - 2 * gap, col = line_ === 'f' ? C.blue : C.teal;
    let a = 0, vv = v;
    if (i < full) a = 1; else if (i === full && frac > 0) { a = 1; const pv = i ? STEPS[i - 1][1] : 1; vv = pv * Math.pow(v / pv, ease(frac)); }
    if (a === 0) {
      const fa = flash >= 0 ? clamp(1 - Math.abs(flash - i) / 1.2) : 0;
      if (ghost > 0) { sr(bx, Y(v), bw, y + h - Y(v), C.lg, 2, alpha * ghost); rr(bx, Y(v), bw, y + h - Y(v), col, alpha * fa * 0.45); }
    } else rr(bx, Y(vv), bw, y + h - Y(vv), col, alpha);
    const isLast = i === (frac > 0 ? full : full - 1);
    if (a && (values === 'all' || (values === 'last' && isLast) || (values === 'ends' && (i === 0 || i === n - 1)))) { txt(fmtT(vv), bx + bw / 2, Y(vv) - (pct ? 34 : 10) * size, { size: 22 * size, weight: 800, color: line_ === 'f' ? C.blue : C.tealD, align: 'center', alpha, font: MONO });
      if (pct) txt(fmtP(pctOf(i)), bx + bw / 2, Y(vv) - 10 * size, { size: 18 * size, weight: 600, color: C.gray, align: 'center', alpha, font: MONO }); }
    if (names) {
      g.save(); g.translate(bx + bw / 2 + 6, y + h + 14); g.rotate(-Math.PI / 4);
      txt(name, 0, 0, { size: 19 * size, weight: 600, color: a ? C.ink2 : C.gray, align: 'right', alpha, font: MONO }); g.restore();
    }
    if (ann && cat >= 0) {
      const aa = ann(i); if (aa > 0) {
        const lines = cut.length > 4 && !cut.includes(' ') ? [cut.slice(0, cut.length - 2), cut.slice(-2)] : [cut];
        lines.forEach((s, j) => txt(s, bx + bw / 2, Y(v) - 46 - (lines.length - 1 - j) * 24, { size: 19, weight: 700, color: CATS[cat][1], align: 'center', alpha: alpha * aa }));
      }
    }
    if (band > 0 && cat >= 0) rr(bx - gap + 1, y + h + 4, sw - 2, 10, CATS[cat][1], alpha * band);
  });
  const cl = (k, i0, i1, v, s) => {
    if (cub[k] <= 0) return; const x0 = x + i0 * sw, x1 = x + i1 * sw; line(x0, Y(v), x1, Y(v), C.ink, 2.5, alpha * cub[k], [10, 7]);
    if (cubLabel) txt(s, k ? x1 - 4 : x0 + 4, Y(v) - 12, { size: 20 * size, weight: 700, color: C.ink, align: k ? 'right' : 'left', alpha: alpha * cub[k] });
  };
  cl(0, 0, 7, CUB.f, 'cuBLAS FP32 51.7'); cl(1, 7, n, CUB.t, 'cuBLAS BF16 781.9');
  const o_hl = ring; if (o_hl >= 0) { const bx = x + o_hl * sw; sr(bx + 2, Y(STEPS[o_hl][1]) - 6, sw - 4, y + h - Y(STEPS[o_hl][1]) + 12, C.red, 3, alpha); }
  return Y;
}
// 各幕右下角的小阶梯：第 k 级在 t0 起生长 0.9 s
function mini(u, k, t0, { x = 1310, y = 660, w = 500, h = 260, alpha = 1, cub = [0, 0], from = k } = {}) {
  if (alpha <= 0.003) return;
  txt('8192 阶 · TFLOPS', x - 40, y - 26, { size: 20, weight: 700, color: C.gray, alpha });
  let up = from; for (let j = from; j <= k; j++) { const s = t0 + (j - from) * 1.0; if (u >= s) up = j + Math.min(1, (u - s) / 0.9) * 0.999; }
  if (u < t0) up = from;
  ladder(x, y, w, h, { upto: up, ghost: 0.45, cub, alpha, size: 0.9 });
}
