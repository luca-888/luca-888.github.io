// MoE Kernel 影片共用：示意路由数据、时间工具与播放器。
// 数字取 Qwen3-30B-A3B 的 config：128 experts、top-8、hidden 2048、moe_intermediate_size 768、norm_topk_prob。
// 路由分数为示意数据（固定种子）。
const W = 1920, H = 1080;
let DUR = 45;
const E = 128, K = 8, NT = 16;

// 各段起止（秒）：开幕、稠密 FFN、切成 expert、router、加权求和、一批 token、提问
const B = { open: [0, 10], dense: [10, 16.3], split: [16.3, 20.8], router: [20.8, 28], combine: [28, 35], batch: [35, 42.2], ask: [42.2, 45] };

function mulberry32(s) { return () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let x = Math.imul(s ^ (s >>> 15), 1 | s); x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x; return ((x ^ (x >>> 14)) >>> 0) / 4294967296; }; }
const rnd = mulberry32(11);
const gauss = () => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
const pop = Array.from({ length: E }, () => gauss() * 0.35);
const route = Array.from({ length: NT }, () => {
  const logit = pop.map((v) => v + gauss());
  const mx = Math.max(...logit), ex = logit.map((l) => Math.exp(l - mx)), s = ex.reduce((a, b) => a + b);
  const prob = ex.map((v) => v / s);
  const top = [...prob.keys()].sort((a, b) => prob[b] - prob[a]).slice(0, K);
  const ts = top.reduce((a, e) => a + prob[e], 0);
  return { prob, top, w: top.map((e) => prob[e] / ts) };
});
const counts = new Array(E).fill(0);
route.forEach((r) => r.top.forEach((e) => counts[e]++));
const ZERO = counts.filter((c) => c === 0).length, MAXC = Math.max(...counts);
const HIST = Array.from({ length: MAXC + 1 }, (_, c) => counts.filter((v) => v === c).length);   // 分到 c 个 token 的 expert 数
const T0 = route[0];
const PMAX = Math.max(...T0.prob);
const IS_TOP = new Array(E).fill(false); T0.top.forEach((e) => (IS_TOP[e] = true));
const TOP_ASC = [...T0.top].sort((a, b) => a - b);
const WOF = (e) => T0.w[T0.top.indexOf(e)];
const DEMO_E = [...T0.top].sort((a, b) => Math.abs(a - E / 2) - Math.abs(b - E / 2))[0];

// 一批 token 的到达时刻与截至 t 每个 expert 已分到的 token 数
const BT = (i) => 36.6 + i * 0.22;
function liveCounts(t, lag = 0.3) {
  const c = new Array(E).fill(0);
  for (let i = 0; i < NT; i++) if (t >= BT(i) + lag) route[i].top.forEach((e) => c[e]++);
  return c;
}

// 开幕：6 个 token 依次分派
const OT = 6, OFIRE = (i) => 1.3 + i * 0.5;
function openCounts(t, lag = 0.4) {
  const c = new Array(E).fill(0);
  for (let i = 0; i < OT; i++) if (t >= OFIRE(i) + lag) route[i].top.forEach((e) => c[e]++);
  return c;
}

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = (x) => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eout = (x) => 1 - Math.pow(1 - clamp(x), 3);
const p = (t, a, b) => ease((t - a) / (b - a));
const po = (t, a, b) => eout((t - a) / (b - a));
const lin = (t, a, b) => clamp((t - a) / (b - a));
const win = (t, a, b, f = 0.4) => p(t, a, a + f) * (1 - p(t, b - f, b));
const mix = (a, b, k) => a + (b - a) * k;
const inB = (t, k, f = 0.45) => win(t, B[k][0], B[k][1], f);

const cv = document.getElementById('cv'), g = cv.getContext('2d');
function rr(x, y, w, h, r, fill, alpha = 1) {
  if (alpha <= 0.003 || w <= 0 || h <= 0) return;
  g.globalAlpha = alpha; g.fillStyle = fill; g.beginPath(); g.roundRect(x, y, w, h, r); g.fill(); g.globalAlpha = 1;
}
function txt(s, x, y, { size = 40, color = '#000', align = 'left', weight = 400, font, alpha = 1, spacing = 0, base = 'middle' } = {}) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = color; g.textAlign = align; g.textBaseline = base;
  g.font = `${weight} ${size}px ${font}`; g.letterSpacing = `${spacing}px`;
  g.fillText(s, x, y); g.letterSpacing = '0px'; g.globalAlpha = 1;
}
function line(x0, y0, x1, y1, color, w = 2, alpha = 1, dash) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash || []);
  g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
}
// 三次贝塞尔（水平出、水平进），k 为画出的比例
function curve(x0, y0, x1, y1, k, color, w, alpha, dash) {
  if (alpha <= 0.003 || k <= 0) return;
  g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath();
  const cx = (x0 + x1) / 2, n = 40;
  for (let i = 0; i <= n * k; i++) {
    const u = i / n, a = (1 - u) ** 3, b = 3 * (1 - u) ** 2 * u, c = 3 * (1 - u) * u * u, d = u ** 3;
    const x = a * x0 + b * cx + c * cx + d * x1, y = a * y0 + b * y0 + c * y1 + d * y1;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
}

// 播放器：预览时带进度条；被 puppeteer 打开时（navigator.webdriver）只暴露 renderAt
function start(draw, dur = DUR) {
  DUR = dur;
  const RENDER = navigator.webdriver || new URLSearchParams(location.search).has('render');
  window.DUR = DUR; window.READY = document.fonts.ready;
  window.renderAt = (t) => { g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; draw(t); };
  const bar = document.getElementById('bar');
  if (RENDER) { bar.style.display = 'none'; return; }
  const seek = bar.querySelector('input'), btn = bar.querySelector('button'), tl = bar.querySelector('span');
  seek.max = DUR; seek.step = 0.01;
  let now = Number(new URLSearchParams(location.search).get('t')) || 0, playing = false, last = 0;
  const show = () => { window.renderAt(now); seek.value = now; tl.textContent = `${now.toFixed(2)} / ${DUR}s`; };
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? '暂停' : '播放'; if (now >= DUR) now = 0; last = performance.now(); };
  seek.oninput = () => { now = Number(seek.value); show(); };
  document.addEventListener('keydown', (e) => { if (e.code === 'Space') { e.preventDefault(); btn.click(); } });
  (function loop(ts) {
    if (playing) { now += (ts - last) / 1000; last = ts; if (now >= DUR) { now = DUR; playing = false; btn.textContent = '播放'; } }
    show(); requestAnimationFrame(loop);
  })(performance.now());
}
