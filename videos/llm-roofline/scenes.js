// LLM Roofline 影片的各幕（顺序与 id 见 script.js）。每幕一个 draw(u)，u 是幕内时间（秒）；节拍起点取自 script.js，音效事件登记进 EV。
// 贯穿全片的两张图：一步的时间轴（读取 = 字节 ÷ 带宽，计算 = 运算 ÷ 算力，按真实比例画）与 roofline（对数坐标，同一个场景是图上的一个点）。
// 颜色：读取与权重 blue，计算 teal，KV cache purple，activation 深黄，CUDA graph amber，实测与下限之差 red。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const I = Object.fromEntries(SCRIPT.map((s, i) => [s.id, i]));   // 幕的 id → 序号
const EV = SCRIPT.map(() => []);                                 // 每幕的音效：[幕内时间, 种类]
const ev = (k, u, kind) => EV[k].push([u, kind]);
const fade = (u, a, d = 0.45) => po(u, a, a + d);
const FILM = 'LLM 推理的量级估算';

// ---------- 本片的数（推导见 docs/llm-roofline-notes.md） ----------
// Qwen3-8B：36 层；每层 7 个矩阵；lm_head 每步全读，embedding 每步只取一行。H100 SXM：989 TFLOPS（BF16 dense），3.35 TB/s。
const BW = 3.35, PK = 989, RIDGE = PK / BW;                      // TB/s、TFLOPS、FLOP/byte
const P_LAYER = 192.94e6, P_HEAD = 622.3e6, P_READ = 36 * P_LAYER + P_HEAD;   // 每步参与计算的参数
const GB_READ = 2 * P_READ / 1e9;                                 // 15.1 GB
const T_MEM = GB_READ / BW;                                       // 4.52 ms（GB ÷ TB/s = ms）
const T_COMP = 2 * P_READ / 1e9 / PK;                             // 0.0153 ms（GFLOP ÷ TFLOPS = ms）
const KV_TOK = 36 * 2 * 8 * 128 * 2;                              // 147456 byte = 144 KiB
const kvGB = (b, ctx = 2048) => b * ctx * KV_TOK / 1e9;
const M = { itl1: 6.84, itl8: 7.03, tps1: 145, tps8: 1118 };      // 实测（vLLM bench，H100）
const LEDGER = { total: 79.18, used: 16.17, act: 1.68, graph: 0.45, kv: 54.99, budget: 72.84 };   // 实测启动日志，GiB

// ---------- 页眉与文字 ----------
function header(k, u) {
  const s = SCRIPT[k], a = fade(u, 0, 0.5);
  txt(FILM, 96, 76, { size: 28, weight: 700, color: C.ink2 });
  if (s.src) txt(s.src, 1824, 76, { size: 22, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 226 + (1 - a) * 16, 52, 700, a);
}
const note = (s, x, y, o = {}) => txt(s, x, y, { size: 22, weight: 500, color: C.gray, ...o });
const lab = (s, x, y, o = {}) => txt(s, x, y, { size: 26, weight: 700, ...o });
function rtxt(s, x, y, ang, o = {}, oy = 0) {
  if ((o.alpha ?? 1) <= 0.003) return;
  g.save(); g.translate(x, y); g.rotate(ang); txt(s, 0, oy, o); g.restore();
}
// 沿屋顶斜线的箭头：从强度 v0 到 v1，off 为向右下方偏离斜线的像素
function sarrow(v0, v1, off, col, alpha = 1, k = 1, c = RC) {
  if (alpha <= 0.003 || k <= 0) return [0, 0];
  const x0 = rx(v0, c), y0 = ry(attain(v0), c), x1 = rx(v1, c), y1 = ry(attain(v1), c), L = Math.hypot(x1 - x0, y1 - y0), nx = -(y1 - y0) / L, ny = (x1 - x0) / L;
  const ax = x0 + nx * off, ay = y0 + ny * off;
  arrow(ax, ay, ax + (x1 - x0) * k, ay + (y1 - y0) * k, col, alpha, 4);
  return [ax + (x1 - x0) / 2, ay + (y1 - y0) / 2];
}
function badge(n, x, y, col, alpha = 1) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = col; g.beginPath(); g.arc(x, y, 17, 0, 7); g.fill(); g.globalAlpha = 1;
  txt(n, x, y + 8, { size: 22, weight: 800, color: '#fff', align: 'center', alpha });
}
function dot(x, y, r, col, alpha = 1) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = col; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
  g.strokeStyle = '#fff'; g.lineWidth = 3; g.stroke(); g.globalAlpha = 1;
}
function diamond(x, y, r, col, alpha = 1) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.fillStyle = col; g.beginPath(); g.moveTo(x, y - r * 1.3); g.lineTo(x + r * 1.3, y); g.lineTo(x, y + r * 1.3); g.lineTo(x - r * 1.3, y); g.closePath(); g.fill();
  g.strokeStyle = '#fff'; g.lineWidth = 3; g.stroke(); g.globalAlpha = 1;
}
// 弧形箭头：从 (x0, y0) 到 (x1, y1)，bend 为向左侧（行进方向）鼓出的像素
function arc(x0, y0, x1, y1, bend, col, alpha = 1, k = 1) {
  if (alpha <= 0.003 || k <= 0) return;
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2, L = Math.hypot(x1 - x0, y1 - y0), nx = -(y1 - y0) / L, ny = (x1 - x0) / L;
  const cx = mx + nx * bend, cy = my + ny * bend, n = 30, pts = [];
  for (let i = 0; i <= n * k; i++) { const t = i / n; pts.push([(1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1, (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * cy + t * t * y1]); }
  g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = 4; g.beginPath(); pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke(); g.globalAlpha = 1;
  if (pts.length > 2) { const [ax, ay] = pts[pts.length - 2], [bx, by] = pts[pts.length - 1]; arrow(ax, ay, bx + (bx - ax) * 0.01, by + (by - ay) * 0.01, col, alpha, 4); }
}
// 占位框：这里将放实测，写明需要什么硬件
function placeholder(x, y, w, h, lines, alpha = 1) {
  if (alpha <= 0.003) return;
  rr(x, y, w, h, '#fbf9f4', alpha); sr(x, y, w, h, C.gray, 2.5, alpha, [10, 8]);
  txt('占位 · 待实测', x + 20, y + 38, { size: 22, weight: 700, color: C.red, alpha });
  lines.forEach((s, i) => txt(s, x + 20, y + 74 + i * 32, { size: 21, weight: 500, color: C.ink2, alpha }));
}

// ---------- 对话窗口：按实测速度逐 token 流出（文本为示意，速度为实测） ----------
const PROMPT = '解释一下 Transformer 是怎样生成文本的。';
const ANSWER = 'Transformer 是一种以 attention 为核心的神经网络。它先把输入的每个 token 映射成一个向量，再让这些向量依次经过几十层计算：attention 让每个位置读取其他位置的信息，前馈网络再对每个位置单独变换。生成文本时，模型根据已有的内容计算下一个 token 的概率，选出一个，追加到序列末尾，然后用更新后的序列预测下一个，如此循环，直到输出结束符。因为每一步只产出一个 token，回答的速度取决于完成一步需要多久。今天的大语言模型几乎都采用这种结构，区别主要在层数、宽度、attention 的形式与训练数据。';
const TOKS = (() => {   // 中文按 1–2 个字一个 token，英文单词一个 token（接近 Qwen 分词的平均长度）
  const r = (() => { let s = 7; return () => ((s = (s * 16807) % 2147483647) / 2147483647); })(), out = []; let i = 0;
  while (i < ANSWER.length) {
    const m = /^[A-Za-z0-9]+ ?/.exec(ANSWER.slice(i));
    const n = m ? m[0].length : /[，。：、]/.test(ANSWER[i + 1] || '') ? 2 : r() < 0.4 ? 2 : 1;
    out.push(ANSWER.slice(i, i + n)); i += n;
  }
  return out;
})();
const CW = {};
const cw = (c, size) => (CW[size + c] ??= tw(c, size, 500));
function wrap(s, maxW, size) {
  const lines = []; let cur = '', w = 0;
  for (const c of s.match(/[A-Za-z0-9.\-]+|[\s\S]/g) || []) {
    const d = c.length > 1 ? tw(c, size, 500) : cw(c, size);
    if (w + d > maxW && cur && !/^[，。：、；）]$/.test(c)) { lines.push(cur); cur = ''; w = 0; if (c === ' ') continue; }
    cur += c; w += d;
  }
  if (cur) lines.push(cur);
  return lines;
}
// n：已生成的 token 数（可为小数）；offset：从第几个 token 开始（小窗口错开内容）
function chat(x, y, w, h, n, { size = 26, alpha = 1, offset = 0, title = 'Qwen3-8B · H100', rate = '' } = {}) {
  if (alpha <= 0.003) return;
  rr(x, y, w, h, '#fbf9f4', alpha); sr(x, y, w, h, C.lg, 2, alpha);
  const pad = size * 0.7, bar = size * 1.6;
  rr(x, y, w, bar, C.paper2, alpha);
  txt(title, x + pad, y + bar * 0.68, { size: size * 0.72, weight: 600, color: C.ink2, alpha });
  if (rate) txt(rate, x + w - pad, y + bar * 0.68, { size: size * 0.72, weight: 700, color: C.blue, alpha, align: 'right' });
  let s = ''; const k = Math.floor(n);
  for (let i = 0; i < k; i++) s += TOKS[(offset + i) % TOKS.length];
  const lh = size * 1.5, top = y + bar + pad;
  const q = size > 20 ? wrap(PROMPT, w - 2 * pad, size) : [];
  const lines = wrap(s, w - 2 * pad, size), room = Math.floor((h - bar - 2 * pad) / lh) - q.length - (q.length ? 0.4 : 0);
  q.forEach((l, i) => txt(l, x + pad, top + lh * (i + 0.75), { size, weight: 600, color: C.gray, alpha }));
  const shown = lines.slice(Math.max(0, lines.length - Math.floor(room))), y0 = top + lh * (q.length + (q.length ? 0.4 : 0));
  g.save(); g.beginPath(); g.rect(x, y + bar, w, h - bar); g.clip();
  shown.forEach((l, i) => txt(l, x + pad, y0 + lh * (i + 0.75), { size, weight: 500, color: C.ink, alpha }));
  if (k > 0 && n % 1 < 0.5) { const last = shown[shown.length - 1] || ''; rr(x + pad + tw(last, size, 500) + 3, y0 + lh * (shown.length - 1) + lh * 0.2, size * 0.5, size * 0.9, C.blue, alpha * 0.8); }
  g.restore();
}

// ---------- H100 示意：中间是计算单元，两侧各 3 个 HBM ----------
function h100(cx, cy, s, alpha = 1, flow = -1) {
  if (alpha <= 0.003) return;
  const W2 = 300 * s, H2 = 190 * s;
  rr(cx - W2, cy - H2, 2 * W2, 2 * H2, C.paper2, alpha); sr(cx - W2, cy - H2, 2 * W2, 2 * H2, C.ink2, 2, alpha);
  const dw = 150 * s, dh = 150 * s;
  rr(cx - dw, cy - dh, 2 * dw, 2 * dh, '#2f3b3a', alpha);
  for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) rr(cx - dw + 8 * s + i * 24.3 * s, cy - dh + 8 * s + j * 24.3 * s, 19 * s, 19 * s, C.teal, alpha * 0.85);
  const hw = 62 * s, hh = 92 * s, gap = 12 * s;
  for (const side of [-1, 1]) for (let j = 0; j < 3; j++) {
    const hx = side < 0 ? cx - W2 + 18 * s : cx + W2 - 18 * s - hw, hy = cy - (1.5 * hh + gap) + j * (hh + gap);
    rr(hx, hy, hw, hh, C.blue, alpha); for (let q = 1; q < 6; q++) line(hx + 6 * s, hy + q * hh / 6, hx + hw - 6 * s, hy + q * hh / 6, '#ffffff', 1.5, alpha * 0.35);
    if (flow >= 0) for (let q = 0; q < 4; q++) {   // 数据从 HBM 流向计算单元
      const f = (flow * 1.2 + q / 4 + j * 0.13) % 1, x0 = side < 0 ? hx + hw : hx, x1 = side < 0 ? cx - dw : cx + dw;
      rr(mix(x0, x1, f) - 5 * s, hy + hh / 2 - 5 * s + (q - 1.5) * 9 * s, 10 * s, 10 * s, C.blue, alpha * Math.sin(Math.PI * f));
    }
  }
}

// ---------- 一步的时间轴 ----------
// rows：[{ label, segs: [[毫秒, 颜色], …], val, a, k }]，k 为长出的进度；ticks：[{ ms, label, col, a, dash, row }]，row 为标签所在的行（默认在顶部）
const ROWH = 78;
function timeline(x, y, w, maxMs, rows, { alpha = 1, ticks = [], step = 1, unit = 'ms', zoom = null } = {}) {
  if (alpha <= 0.003) return;
  const px = (ms) => x + w * ms / maxMs, yAxis = y + rows.length * ROWH + 4;
  rows.forEach((r, i) => {
    const a = alpha * (r.a ?? 1), yy = y + i * ROWH; if (a <= 0.003) return;
    txt(r.label, x, yy + 26, { size: 24, weight: 700, color: r.lcol || C.ink, alpha: a });
    rr(x, yy + 38, w, 28, C.paper2, a * 0.6);
    let at = 0; const k = r.k ?? 1;
    for (const [ms, col, name] of r.segs) {
      const d = ms * k, bw = w * d / maxMs; rr(px(at), yy + 38, Math.max(d > 0 ? 2.5 : 0, bw), 28, col, a);
      if (name && bw > tw(name, 19, 700) + 16) txt(name, px(at) + 8, yy + 59, { size: 19, weight: 700, color: '#fff', alpha: a });
      at += d;
    }
    if (r.val) txt(r.val, Math.max(px(at) + 14, x + tw(r.label, 24, 700) + 24), yy + 62, { size: 24, weight: 700, color: r.vcol || C.ink, alpha: a * clamp(k * 3 - 2) });
  });
  line(x, yAxis, x + w, yAxis, C.ink, 2, alpha);
  for (let ms = 0; ms <= maxMs + 1e-6; ms += step) {
    line(px(ms), yAxis, px(ms), yAxis + 8, C.ink, 2, alpha);
    txt(ms + (ms === 0 ? '' : ''), px(ms), yAxis + 34, { size: 20, weight: 500, color: C.gray, align: 'center', alpha });
  }
  txt(unit, x + w + 14, yAxis + 34, { size: 20, weight: 500, color: C.gray, alpha });
  for (const t of ticks) {
    const a = alpha * (t.a ?? 1); if (a <= 0.003) continue;
    const tx = px(t.ms), y0 = rows.length ? y - 10 : y - 44;
    line(tx, y0, tx, yAxis, t.col || C.ink, 3, a, t.dash);
    txt(t.label, tx + (t.align === 'right' ? -10 : 10), y0 + 4, { size: 23, weight: 700, color: t.col || C.ink, alpha: a, align: t.align || 'left' });
  }
  if (zoom) {   // 放大框：把 [0, zoom.ms] 放大到框内，用来看清很短的条
    const a = alpha * (zoom.a ?? 1), { bx, by, bw, bh, ms, row } = zoom; if (a <= 0.003) return;
    const yy = y + row * ROWH + 38;
    sr(x - 4, yy - 6, Math.max(10, w * ms / maxMs) + 8, 40, C.ink, 2, a);
    line(x + Math.max(10, w * ms / maxMs) + 4, yy + 14, bx, by + bh / 2, C.ink, 2, a);
    rr(bx, by, bw, bh, '#fbf9f4', a); sr(bx, by, bw, bh, C.ink, 2, a);
    const segs = rows[row].segs; let at = 0;
    for (const [d, col] of segs) { rr(bx + 12 + (bw - 24) * at / ms, by + 12, (bw - 24) * d / ms, bh - 24, col, a); at += d; }
    txt(zoom.label, bx + bw + 14, by + bh / 2 + 8, { size: 20, weight: 600, color: C.gray, alpha: a });
  }
}

// ---------- roofline ----------
// 对数坐标：横轴 arithmetic intensity 0.5–8192 FLOP/byte，纵轴 1–5000 TFLOPS
const RC = { x: 190, y: 300, w: 760, h: 600 };
const LX0 = Math.log10(0.5), LX1 = Math.log10(8192), LY0 = 0, LY1 = Math.log10(5000);
const rx = (v, c = RC) => c.x + c.w * (Math.log10(v) - LX0) / (LX1 - LX0);
const ry = (v, c = RC) => c.y + c.h - c.h * (Math.log10(v) - LY0) / (LY1 - LY0);
const attain = (v, bw = BW, pk = PK) => Math.min(pk, bw * v);
function axes(c = RC, alpha = 1) {
  if (alpha <= 0.003) return;
  for (const v of [1, 10, 100, 1000]) {
    line(rx(v, c), c.y, rx(v, c), c.y + c.h, C.lg, 1.5, alpha * 0.7);
    txt(String(v), rx(v, c), c.y + c.h + 32, { size: 21, weight: 500, color: C.gray, align: 'center', alpha });
    line(c.x, ry(v, c), c.x + c.w, ry(v, c), C.lg, 1.5, alpha * 0.7);
    txt(String(v), c.x - 12, ry(v, c) + 7, { size: 21, weight: 500, color: C.gray, align: 'right', alpha });
  }
  line(c.x, c.y + c.h, c.x + c.w, c.y + c.h, C.ink, 2.5, alpha); line(c.x, c.y, c.x, c.y + c.h, C.ink, 2.5, alpha);
  txt('arithmetic intensity（FLOP / byte）', c.x + c.w / 2, c.y + c.h + 70, { size: 23, weight: 600, color: C.ink2, align: 'center', alpha });
  rtxt('可达到的算力（TFLOPS）', c.x - 66, c.y + c.h / 2, -Math.PI / 2, { size: 23, weight: 600, color: C.ink2, align: 'center', alpha });
}
// 一条屋顶：pa 为斜线画出的进度，pb 为平顶画出的进度
function roofLine({ bw = BW, pk = PK, col = C.ink, lw = 6, dash, pa = 1, pb = 1, alpha = 1, slope = '', flat = '', ridge = '' }, c = RC) {
  if (alpha <= 0.003) return;
  const rI = pk / bw, x0 = rx(0.5, c), y0 = ry(attain(0.5, bw, pk), c), xr = rx(rI, c), yr = ry(pk, c), x1 = c.x + c.w;
  const yClip = Math.max(c.y + c.h, y0);
  g.save(); g.beginPath(); g.rect(c.x, c.y - 40, c.w, c.h + 40); g.clip();
  if (pa > 0) line(x0, y0, mix(x0, xr, pa), mix(y0, yr, pa), col, lw, alpha, dash);
  if (pb > 0) line(xr, yr, mix(xr, x1, pb), yr, col, lw, alpha, dash);
  g.restore();
  const ang = Math.atan2(yr - y0, xr - x0);
  if (slope) rtxt(slope, mix(x0, xr, 0.62), mix(y0, yr, 0.62), ang, { size: 22, weight: 700, color: col, alpha: alpha * clamp(pa * 2 - 1), align: 'center' }, -14);
  if (flat) txt(flat, x1 - 4, yr - 16, { size: 22, weight: 700, color: col, align: 'right', alpha: alpha * clamp(pb * 2 - 1) });
  if (ridge) { const a = alpha * clamp(pb * 3); line(xr, yr, xr, c.y + c.h, col, 2, a * 0.8, [6, 6]); txt(ridge, xr + 8, c.y + c.h - 14, { size: 22, weight: 700, color: col, alpha: a }); }
  void yClip;
}
// 图上的点：v 为强度，perf 默认取屋顶上的值
function rpt(v, { col = C.blue, shape = 'dot', alpha = 1, label = '', dx = 24, dy = 44, align = 'left', perf, size = 23, r = 13, lcol } = {}, c = RC) {
  if (alpha <= 0.003) return [0, 0];
  const x = rx(v, c), y = ry(perf ?? attain(v), c);
  (shape === 'dia' ? diamond : dot)(x, y, r, col, alpha);
  if (label) label.split('\n').forEach((l, i) => txt(l, x + dx, y + dy + i * (size + 8), { size, weight: 700, color: lcol || col, align, alpha }));
  return [x, y];
}

// ============ 各幕 ============
function sOpen(u) {
  const k = I.open, shrink = p(u, bt(k, 2), bt(k, 2) + 1.0);
  // 标题：封面上的大字，最后缩成左上角的片名
  txt(FILM, 96, mix(250, 76, shrink), { size: mix(84, 28, shrink), weight: 700, color: mixc(C.ink, C.ink2, shrink) });
  txt('Roofline：生成一个 token 的最短用时', 96, 322, { size: 40, weight: 600, color: C.gray, alpha: 1 - shrink });
  const a0 = fade(u, 0.3), t0 = bt(k, 0);
  chat(96, mix(400, 300, shrink), 860, mix(420, 500, shrink), Math.max(0, (u - t0) * M.tps1), { alpha: a0, rate: '145 tok/s（实测）' });
  h100(1420, mix(600, 560, shrink), 1.05, fade(u, 0.8), u);
  txt('H100 SXM 80GB（示意）', 1420, mix(830, 790, shrink), { size: 22, weight: 600, color: C.gray, align: 'center', alpha: fade(u, 0.8) });
  // 时间轴：第一个刻度落在实测值
  const ta = fade(u, bt(k, 1));
  timeline(96, 930, 1640, 8, [], { alpha: ta, ticks: [{ ms: M.itl1, label: '每个 token 6.84 ms（实测）', a: fade(u, bt(k, 1) + 0.6), align: 'right' }] });
  if (u >= bt(k, 1) + 0.6) txt('?', 96 + 1640 * M.itl1 / 8 + 20, 924, { size: 40, weight: 800, color: C.red, alpha: fade(u, bt(k, 1) + 1.4) * (1 - shrink) });
}

(() => { const k = I.open; ev(k, bt(k, 1) + 0.6, 'land'); ev(k, bt(k, 2), 'whoosh'); })();

function sWork(u) {
  const k = I.work; header(k, u);
  // 左：36 层；一个指针从第 1 层走到第 36 层
  const SX = 96, SY = 300, SW = 250, LH = 15.4;
  const descA = lin(u, bt(k, 0) + 0.3, bt(k, 0) + 3.8), descC = lin(u, bt(k, 2) + 0.2, bt(k, 2) + 3.6);
  const cur = u < bt(k, 1) ? descA * 35 : u < bt(k, 2) ? 0 : descC * 35;
  for (let i = 0; i < 36; i++) {
    const done = u < bt(k, 1) ? i <= cur : u < bt(k, 2) ? i === 0 : i <= cur;
    rr(SX, SY + i * LH, SW, LH - 4, done ? C.blue : C.paper2, fade(u, 0.1 + i * 0.01));
  }
  lab('36 层', SX, SY + 36 * LH + 34, { alpha: fade(u, 0.3) });
  const ptr = SY + Math.round(cur) * LH + (LH - 4) / 2;
  arrow(SX + SW + 50, ptr, SX + SW + 8, ptr, C.ink, fade(u, bt(k, 0)));
  // 中：放大第 1 层，7 个矩阵按真实面积（每格的高 = 输入维度，宽 = 输出维度）
  const za = fade(u, bt(k, 1), 0.6), S = 0.0334, ZX = 470;
  line(SX + SW, SY, ZX - 20, 300, C.ink, 1.5, za * 0.5); line(SX + SW, SY + LH - 4, ZX - 20, 740, C.ink, 1.5, za * 0.5);
  const mats = [['q', 4096, 4096, 0], ['k', 4096, 1024, 0], ['v', 4096, 1024, 0], ['o', 4096, 4096, 0], ['gate', 4096, 12288, 1], ['up', 4096, 12288, 1], ['down', 12288, 4096, 1]];
  let xr = [ZX, ZX]; const RY = [318, 560];
  mats.forEach(([name, din, dout, row], i) => {
    const w = (row ? Math.max(din, dout) : dout) * S, h = (row ? Math.min(din, dout) : din) * S, x = xr[row], y = RY[row];
    xr[row] += w + 22;
    const t0 = bt(k, 1) + 0.5 + i * 0.75, sw = lin(u, t0, t0 + 0.7);
    rr(x, y, w, h, C.paper2, za); rr(x, y, w * (u >= bt(k, 2) ? 1 : sw), h, C.blue, za * 0.85);
    if (sw > 0 && sw < 1 && u < bt(k, 2)) rr(x + w * sw - 3, y - 8, 6, h + 16, C.teal, za);
    sr(x, y, w, h, C.ink, 2, za);
    txt(name, x + 6, y + h + 30, { size: 22, weight: 700, color: C.ink2, alpha: za });
    txt(`${din} × ${dout}`, x + w - 4, y + h + 30, { size: 18, weight: 500, color: C.gray, alpha: za * (w > 300 ? 1 : 0), align: 'right' });
  });
  note('attention', ZX, 300, { alpha: za }); note('FFN', ZX, 542, { alpha: za });
  // 计数器：第 1 层之后快进其余 35 层与 lm_head
  const perLayer = 2 * P_LAYER / 1e9, sweep1 = clamp((u - bt(k, 1) - 0.5) / (7 * 0.75));
  const layers = u < bt(k, 1) ? 0 : u < bt(k, 2) ? sweep1 : 1 + descC * 35, head = lin(u, bt(k, 2) + 3.6, bt(k, 2) + 3.9);
  const gb = Math.min(GB_READ, layers * perLayer + head * 2 * P_HEAD / 1e9), ca = fade(u, bt(k, 1) + 0.3);
  const CY = 800;
  txt('从显存读出', ZX, CY, { size: 26, weight: 600, color: C.ink2, alpha: ca });
  txt(`${gb.toFixed(gb < 1 ? 2 : 1)} GB`, ZX, CY + 66, { size: 60, weight: 800, color: C.blue, alpha: ca, font: ZH });
  txt('乘加运算', ZX + 420, CY, { size: 26, weight: 600, color: C.ink2, alpha: ca });
  txt(`${gb.toFixed(gb < 1 ? 2 : 1)} GFLOP`, ZX + 420, CY + 66, { size: 60, weight: 800, color: C.tealD, alpha: ca });
  const ra = fade(u, bt(k, 2) + 4.1);
  txt('运算 ÷ 字节 = 1', 1824, CY + 66, { size: 44, weight: 800, align: 'right', alpha: ra });
  note('每个权重：读 2 字节（BF16），做 1 次乘加 = 2 次运算', ZX, CY + 116, { alpha: ca });
  note('embedding 表每步只取一行，不计入', 1824, CY + 116, { alpha: ra, align: 'right' });
}
// 音效登记（不依赖帧）
(() => { const k = I.work; for (let i = 0; i < 36; i += 3) ev(k, bt(k, 0) + 0.3 + i / 35 * 3.5, 'tick'); ev(k, bt(k, 1), 'whoosh'); ev(k, bt(k, 2) + 4.1, 'land'); })();

function sSpeeds(u) {
  const k = I.speeds; header(k, u);
  h100(430, 470, 0.86, fade(u, 0.2), u);
  const a1 = fade(u, bt(k, 0) + 0.6);
  txt('Tensor Core：989 TFLOPS（BF16）', 430, 680, { size: 26, weight: 700, color: C.tealD, align: 'center', alpha: a1 });
  txt('HBM3：3.35 TB/s', 430, 718, { size: 26, weight: 700, color: C.blue, align: 'center', alpha: a1 });
  const fa = fade(u, bt(k, 1) - 2.5);
  rich('读取：{b:15.1 GB} ÷ {b:3.35 TB/s} = {b:4.52 ms}', 900, 380, 36, 700, fa);
  rich('计算：{t:15.1 GFLOP} ÷ {t:989 TFLOPS} = {t:0.015 ms}', 900, 450, 36, 700, fade(u, bt(k, 1) - 1.6));
  note('两者之比约 300 : 1', 900, 500, { alpha: fade(u, bt(k, 1) + 2) });
  const g1 = p(u, bt(k, 1), bt(k, 1) + 1.8), z = fade(u, bt(k, 1) + 2);
  timeline(96, 790, 1640, 8, [
    { label: '读取', segs: [[T_MEM, C.blue]], k: g1 },
    { label: '计算', segs: [[T_COMP, C.teal]], val: '0.015 ms', k: g1, vcol: C.tealD },
  ], { alpha: fade(u, bt(k, 1) - 0.4), ticks: [
    { ms: T_MEM, label: '下限 4.52 ms', col: C.blue, a: fade(u, bt(k, 2)), dash: [8, 6] },
    { ms: M.itl1, label: '实测 6.84 ms', a: fade(u, bt(k, 3)) },
  ], zoom: { a: z, ms: 0.05, row: 1, bx: 300, by: 860, bw: 420, bh: 48, label: '放大：0 – 0.05 ms' } });
  // 实测与下限之差
  const ra = fade(u, bt(k, 3) + 0.6), x0 = 96 + 1640 * T_MEM / 8, x1 = 96 + 1640 * M.itl1 / 8;
  rr(x0, 790 + 38, x1 - x0, 28, C.red, ra * 0.85);
  txt('下限 ÷ 实测 = 66%', x1 + 14, 790 + 62, { size: 24, weight: 700, color: C.red, alpha: ra });
}
(() => { const k = I.speeds; ev(k, bt(k, 1), 'pop'); ev(k, bt(k, 2), 'land'); ev(k, bt(k, 3), 'land'); })();

function sRoofline(u) {
  const k = I.roofline; header(k, u);
  const FX = 1060;
  rich('arithmetic intensity = 运算 ÷ 字节', FX, 340, 34, 700, fade(u, bt(k, 0)));
  rich('decode：{t:15.1 GFLOP} ÷ {b:15.1 GB} = {r:1}', FX, 404, 30, 600, fade(u, bt(k, 0) + 1.2));
  rich('H100：{t:989 TFLOPS} ÷ {b:3.35 TB/s} = {t:295}', FX, 460, 30, 600, fade(u, bt(k, 1)));
  note('每读取 1 字节做 295 次运算，计算单元才不闲置', FX, 500, { alpha: fade(u, bt(k, 1) + 1) });
  const ca = fade(u, bt(k, 2) - 0.6);
  axes(RC, ca);
  roofLine({ col: C.ink, pa: p(u, bt(k, 2), bt(k, 2) + 1.6), pb: p(u, bt(k, 2) + 1.8, bt(k, 2) + 3.2), alpha: ca, slope: '带宽 3.35 TB/s', flat: '算力 989 TFLOPS', ridge: '拐点 295' });
  const ma = fade(u, bt(k, 2) + 3.4);
  txt('受带宽限制', rx(4), ry(600), { size: 24, weight: 700, color: C.blue, alpha: ma, align: 'center' });
  txt('受算力限制', rx(1900), ry(350), { size: 24, weight: 700, color: C.tealD, alpha: ma, align: 'center' });
  rpt(1, { col: C.blue, alpha: fade(u, bt(k, 3), 0.35), label: 'decode，batch 1\n3.35 TFLOPS，峰值的 0.3%' });
  // 2009 年论文原图
  const fa = fade(u, bt(k, 2) + 0.4), FW = 280, FH = FW * 1560 / 1200, fx = 1060, fy = 560;
  if (IMG.fig && IMG.fig.complete && IMG.fig.naturalWidth) { g.globalAlpha = fa; g.drawImage(IMG.fig, fx, fy, FW, FH); g.globalAlpha = 1; }
  sr(fx, fy, FW, FH, C.lg, 2, fa);
  note('Williams 等，2009 年论文 Fig. 1（原图）', fx + FW + 28, fy + FH - 40, { alpha: fa });
  note('原文称 operational intensity', fx + FW + 28, fy + FH - 8, { alpha: fa });
}
loadImg('fig', '../../src/assets/llm-roofline/williams2009-fig1a.png');
(() => { const k = I.roofline; ev(k, bt(k, 1), 'pop'); ev(k, bt(k, 2), 'whoosh'); ev(k, bt(k, 2) + 1.8, 'whoosh'); ev(k, bt(k, 3), 'land'); })();

// 右侧面板的位置（第 5–10 幕，左边是 roofline）
const PX = 1060, PW = 764;
function sBatch(u) {
  const k = I.batch; header(k, u);
  axes(); roofLine({ slope: '带宽 3.35 TB/s', flat: '算力 989 TFLOPS', ridge: '拐点 295' });
  const mv = p(u, bt(k, 2), bt(k, 2) + 2), v = Math.pow(8, mv);
  if (mv > 0) rpt(1, { col: C.blue, alpha: 0.3 });
  rpt(v, { col: C.blue, label: mv > 0.98 ? 'batch 8' : 'decode，batch 1' });
  const da = fade(u, bt(k, 2) + 2.3);
  sarrow(10, RIDGE, 70, C.gray, da, p(u, bt(k, 2) + 2.3, bt(k, 2) + 3.6));
  txt('batch ≈ 295 时到达拐点', rx(RIDGE) + 16, ry(PK) + 120, { size: 22, weight: 700, color: C.gray, alpha: fade(u, bt(k, 2) + 3.4) });
  // 右上：8 个对话窗口
  const CWd = 180, CHt = 112;
  for (let i = 0; i < 8; i++) {
    const a = i === 0 ? fade(u, 0) : fade(u, bt(k, 0) + 1.2 + i * 0.25), x = PX + (i % 4) * (CWd + 14), y = 300 + Math.floor(i / 4) * (CHt + 14);
    const start = i === 0 ? -20 : bt(k, 0) + 1.2 + i * 0.25;
    chat(x, y, CWd, CHt, Math.max(0, (u - start) * M.tps1), { size: 13, alpha: a, offset: 23 * i, title: `请求 ${i + 1}` });
  }
  const sa = fade(u, bt(k, 1));
  const tk = M.itl1 + (M.itl8 - M.itl1) * p(u, bt(k, 1) + 1.0, bt(k, 1) + 2.4);
  timeline(PX, 600, PW - 60, 8, [
    { label: '读取（权重）', segs: [[T_MEM, C.blue]], val: '4.52 ms', vcol: C.blue },
    { label: '计算 × 8', segs: [[T_COMP * 8, C.teal]], val: '0.12 ms', vcol: C.tealD, k: p(u, bt(k, 1), bt(k, 1) + 1) },
  ], { alpha: sa, step: 2, ticks: [{ ms: tk, label: `实测 ${tk.toFixed(2)} ms`, a: sa, align: 'right' }] });
  rich(`实测吞吐：{g:145} → {t:1118 tok/s}`, PX, 860, 30, 700, fade(u, bt(k, 1) + 2.4));
  note('上下文不超过 384 token，KV cache 的读取可忽略', PX, 900, { alpha: fade(u, bt(k, 1) + 2.4) });
}
(() => { const k = I.batch; for (let i = 1; i < 8; i++) ev(k, bt(k, 0) + 1.2 + i * 0.25, 'pop'); ev(k, bt(k, 1) + 1.0, 'click'); ev(k, bt(k, 2), 'whoosh'); })();

function sPrefill(u) {
  const k = I.prefill; header(k, u);
  axes(); roofLine({ slope: '带宽 3.35 TB/s', flat: '算力 989 TFLOPS', ridge: '拐点 295' });
  rpt(1, { col: C.blue, label: fade(u, bt(k, 2)) > 0 ? 'decode：受带宽限制' : 'decode' });
  rpt(1880, { col: C.teal, shape: 'dia', alpha: fade(u, bt(k, 2), 0.35), label: 'prefill，2048 token\n受算力限制', dx: 0, dy: 56, align: 'center', lcol: C.tealD });
  // 右上：2048 个 token 一次送入
  const pa = fade(u, bt(k, 0)), cells = Math.floor(2048 * lin(u, bt(k, 0), bt(k, 0) + 2.5));
  lab('prompt：2048 token，一次送入', PX, 320, { alpha: pa });
  for (let i = 0; i < cells; i++) rr(PX + (i % 128) * 5.6, 340 + Math.floor(i / 128) * 5.6, 4.4, 4.4, C.teal, pa);
  const sa = fade(u, bt(k, 1) - 0.3), g1 = p(u, bt(k, 1), bt(k, 1) + 1.6);
  timeline(PX, 480, PW - 60, 32, [
    { label: '读取（权重）', segs: [[T_MEM, C.blue]], val: '4.52 ms', vcol: C.blue },
    { label: '计算', segs: [[28.8, C.teal]], val: '28.8 ms', vcol: C.tealD, k: g1 },
  ], { alpha: sa, step: 8 });
  note('估算值', PX + PW - 60, 480 - 14, { alpha: sa, align: 'right' });
  placeholder(PX, 730, PW, 168, ['prefill 用时随 prompt 长度的实测曲线（64 → 8192 token，batch 1）', '需要 1 张 H100（vLLM，Qwen3-8B BF16）', '估算的拐点约在 300 token：此前约 4.5 ms 不变，此后线性增长'], fade(u, bt(k, 2) + 1));
}
(() => { const k = I.prefill; ev(k, bt(k, 0), 'whoosh'); ev(k, bt(k, 1), 'pop'); ev(k, bt(k, 2), 'land'); })();

function sKv(u) {
  const k = I.kv; header(k, u);
  axes(); roofLine({ slope: '带宽 3.35 TB/s', flat: '算力 989 TFLOPS', ridge: '拐点 295' });
  const gb = p(u, bt(k, 1), bt(k, 1) + 4), B = Math.pow(64, gb);
  rpt(B, { col: C.blue, label: `矩阵乘法，batch ${Math.round(B)}`, ...(B > 30 ? { dx: -24, dy: -22, align: 'right' } : {}) });
  rpt(4, { col: C.purple, alpha: fade(u, bt(k, 1), 0.35), label: 'attention ≈ 4\n与 batch 无关' });
  // 右上：每个请求自己的 KV cache
  const ka = fade(u, bt(k, 0)), grow = p(u, bt(k, 0) + 0.3, bt(k, 0) + 4);
  lab('每个请求独有的 KV cache', PX, 320, { alpha: ka, color: C.purple });
  for (let i = 0; i < 6; i++) {
    const y = 340 + i * 36, a = ka * fade(u, bt(k, 0) + i * 0.15);
    txt(`请求 ${i + 1}`, PX, y + 24, { size: 20, weight: 600, color: C.ink2, alpha: a });
    rr(PX + 90, y + 6, 560, 22, C.paper2, a * 0.6); rr(PX + 90, y + 6, 560 * grow, 22, C.purple, a);
  }
  txt(`上下文 ${Math.round(2048 * grow)} token`, PX + 650, 340 + 6 * 36 + 20, { size: 20, weight: 600, color: C.purple, alpha: ka, align: 'right' });
  note(`… 共 ${Math.round(B)} 个`, PX + 90, 340 + 6 * 36 + 20, { alpha: ka * fade(u, bt(k, 1)) });
  rich('每 token {p:144 KiB} = 36 层 × 2 × 8 个 KV head × 128 × 2 字节', PX, 625, 24, 600, fade(u, bt(k, 0) + 2));
  const ta = fade(u, bt(k, 2) - 0.4), kvms = kvGB(B) / BW;
  timeline(PX, 680, PW - 60, 12, [
    { label: '读取', segs: [[T_MEM, C.blue, '权重'], [kvms * p(u, bt(k, 2), bt(k, 2) + 1.4), C.purple, 'KV']], val: `${(T_MEM + kvms).toFixed(1)} ms`, vcol: C.ink },
    { label: '计算', segs: [[T_COMP * B, C.teal]], val: `${(T_COMP * B).toFixed(2)} ms`, vcol: C.tealD },
  ], { alpha: ta, step: 2 });
  rich(`64 × 2048 token：KV {p:${kvGB(64).toFixed(1)} GB} > 权重 {b:15.1 GB}`, PX, 930, 26, 700, fade(u, bt(k, 2) + 1.6));
}
(() => { const k = I.kv; for (let i = 0; i < 6; i++) ev(k, bt(k, 0) + i * 0.15, 'pop'); ev(k, bt(k, 1), 'land'); ev(k, bt(k, 2), 'pop'); })();

function sLedger(u) {
  const k = I.ledger; header(k, u);
  const BX = 96, BY = 310, BWd = 1728, BH = 120, s = BWd / LEDGER.total, L = LEDGER;
  const segs = [
    [L.used, C.blue, '权重与非 torch', 0, bt(k, 0)], [L.act, C.yel, 'activation 峰值', 1, bt(k, 1)], [L.graph, C.amber, 'CUDA graph', 2, bt(k, 1) + 1.2],
    [L.kv, C.purple, 'KV cache', 3, bt(k, 2)],
  ];
  rr(BX, BY, BWd, BH, C.paper2, fade(u, 0)); sr(BX, BY, BWd, BH, C.ink, 2, fade(u, 0));
  txt('H100 显存 79.18 GiB', BX, BY - 14, { size: 22, weight: 600, color: C.gray, alpha: fade(u, 0) });
  let x = BX;
  segs.forEach(([gib, col, name, i, t0]) => {
    const k1 = p(u, t0, t0 + 1.0), w = gib * s;
    rr(x, BY, w * k1, BH, col, 1);
    const a = fade(u, t0 + 0.6);
    if (i === 3) txt(`${name} ${gib.toFixed(2)} GiB`, x + 16, BY + 34, { size: 26, weight: 700, color: '#fff', alpha: a });
    else if (w > 200) { txt(name, x + 16, BY + 50, { size: 26, weight: 700, color: '#fff', alpha: a }); txt(`${gib.toFixed(2)} GiB`, x + 16, BY + 92, { size: 30, weight: 800, color: '#fff', alpha: a }); }
    else { callout(x + w / 2, BY + BH, x + w / 2 + 40 + i * 0, BY + BH + 46 + (i - 1) * 40, `${name} ${gib.toFixed(2)} GiB`, { size: 22, alpha: a, color: i === 1 ? C.yelD : C.amber }); }
    x += w;
  });
  const ra = fade(u, bt(k, 2) + 1);
  g.save(); g.beginPath(); g.rect(x, BY, BX + BWd - x, BH); g.clip();
  for (let i = 0; i < 14; i++) line(x + i * 24, BY, x + i * 24 - 30, BY + BH, C.gray, 2, ra * 0.5);
  g.restore();
  txt('预留 8%', x + (BX + BWd - x) / 2, BY + BH + 34, { size: 22, weight: 700, color: C.gray, align: 'center', alpha: ra });
  // 195 个请求：KV 段排成 195 格
  const kx = BX + (L.used + L.act + L.graph) * s, kw = L.kv * s, ga = fade(u, bt(k, 3) + 1.2);
  for (let i = 0; i < 195; i++) {
    const c = i % 65, r = Math.floor(i / 65), cwid = (kw - 8) / 65, ch = 22, a = ga * fade(u, bt(k, 3) + 1.2 + i * 0.008, 0.2);
    rr(kx + 4 + c * cwid + 1.5, BY + 48 + r * (ch + 2), cwid - 3, ch, C.purpleL, a);
  }
  // 日志原文（vLLM 0.30 启动日志，节选）
  const LG = [
    [`{g:INFO [model_runner.py:428]} Model loading took {b:15.27 GiB} memory and 12.389277 seconds`, bt(k, 0) + 0.3, C.blue],
    [`{g:INFO [gpu_worker.py:888]} … Actual usage is {b:16.17 GiB} for consumed memory (weights + non-torch),`, bt(k, 0) + 0.6, C.blue],
    [`    {y:1.68 GiB} for peak activation, and {a:0.45 GiB} for CUDAGraph memory.`, bt(k, 1) + 0.3, C.yel],
    [`{g:INFO [gpu_worker.py:640]} Available KV cache memory: {p:54.99 GiB}`, bt(k, 2) + 0.3, C.purple],
    [`{g:INFO [kv_cache_utils.py:2395]} GPU KV cache size: {p:400,448 tokens}, Maximum concurrency for 2,048 tokens per request: {p:195.53x}`, bt(k, 3) + 0.3, C.purple],
  ];
  const LY = 610, la = fade(u, bt(k, 0) + 0.2);
  rr(BX, LY, BWd, 40 + LG.length * 38, '#fbf9f4', la); sr(BX, LY, BWd, 40 + LG.length * 38, C.lg, 2, la);
  txt('vllm serve Qwen/Qwen3-8B · 启动日志（实测，节选）', BX + 18, LY - 14, { size: 20, weight: 600, color: C.gray, alpha: la, font: ZH });
  LG.forEach(([s0, t0, col], i) => {
    const a = fade(u, t0, 0.3), y = LY + 22 + i * 38;
    rr(BX + 6, y + 2, 6, 30, col, a);
    richMono(s0, BX + 24, y + 24, 19.5, a * 0.35 + a * 0.65);
  });
  // 换算
  const ca = fade(u, bt(k, 3));
  rich('{p:54.99 GiB} ÷ {p:144 KiB} = {p:400,448 token}', BX, 896, 34, 700, ca);
  rich('÷ 每个请求 2048 token = 最多 {p:195 个请求}同时运行', BX + 760, 896, 34, 700, fade(u, bt(k, 3) + 1.2));
  note('195 格：每格是一个 2048 token 请求的 KV cache', BX, 944, { alpha: fade(u, bt(k, 3) + 1.2) });
}
(() => { const k = I.ledger; ev(k, bt(k, 0), 'pop'); ev(k, bt(k, 1), 'pop'); ev(k, bt(k, 1) + 1.2, 'pop'); ev(k, bt(k, 2), 'land'); ev(k, bt(k, 3) + 1.2, 'whoosh'); })();

function sBound(u) {
  const k = I.bound; header(k, u);
  axes(); roofLine({ slope: '带宽 3.35 TB/s', flat: '算力 989 TFLOPS', ridge: '拐点 295' });
  rpt(195, { col: C.blue, label: '矩阵乘法，batch 195', dx: -24, dy: -22, align: 'right', alpha: fade(u, bt(k, 1)) });
  rpt(4, { col: C.purple, label: 'attention ≈ 4', alpha: fade(u, bt(k, 1)) });
  rich('一步 ≥ max（{b:字节 ÷ 带宽}，{t:运算 ÷ 算力}）', PX, 340, 34, 800, fade(u, bt(k, 0)));
  const ia = fade(u, bt(k, 1) - 0.3);
  rich('字节 = 权重 {b:15.1 GB} + KV {p:58.9 GB} = 74.0 GB', PX, 406, 26, 600, ia);
  rich('运算 = 2 × 7.57 B × 195 = {t:2.95 TFLOP}', PX, 448, 26, 600, ia);
  const g1 = p(u, bt(k, 1), bt(k, 1) + 1.8);
  timeline(PX, 500, PW - 60, 25, [
    { label: '读取', segs: [[T_MEM, C.blue, '权重'], [kvGB(195) / BW, C.purple, 'KV cache']], val: '22.1 ms', k: g1 },
    { label: '计算', segs: [[2.98, C.teal]], val: '3.0 ms', vcol: C.tealD, k: g1 },
  ], { alpha: fade(u, bt(k, 1) - 0.3), step: 5 });
  const ra = fade(u, bt(k, 2));
  rich('吞吐上限 ≈ {t:8800 tok/s}，batch 1 下限的 40 倍', PX, 750, 26, 700, ra);
  rich('每个 token 的下限：{b:4.52} → {r:22.1 ms}', PX, 792, 26, 700, fade(u, bt(k, 2) + 1));
  placeholder(PX, 818, PW, 122, ['decode 每步用时随 batch 的实测点（上下文 2048，batch 1 → 256）', '需要 1 张 H100；实测点将叠在这条估算上'], fade(u, bt(k, 2) + 1.8));
}
(() => { const k = I.bound; ev(k, bt(k, 0), 'pop'); ev(k, bt(k, 1), 'pop'); ev(k, bt(k, 2), 'land'); })();

function sLevers(u) {
  const k = I.levers; header(k, u);
  axes();
  // B200：平顶与斜线一起抬高，拐点几乎不动
  const ba = fade(u, bt(k, 3));
  roofLine({ bw: 8, pk: 2250, col: C.gray, lw: 4, dash: [4, 6], alpha: ba, pa: p(u, bt(k, 3), bt(k, 3) + 1.2), pb: p(u, bt(k, 3) + 1.2, bt(k, 3) + 2.2), flat: 'B200' });
  roofLine({ flat: 'H100' });
  const rgA = fade(u, bt(k, 3) + 2.2);
  line(rx(281), ry(2250), rx(281), RC.y + RC.h, C.gray, 2, rgA, [6, 6]); line(rx(RIDGE), ry(PK), rx(RIDGE), RC.y + RC.h, C.ink, 2, rgA, [6, 6]);
  txt('拐点 281 / 295', rx(300) + 12, RC.y + RC.h - 14, { size: 22, weight: 700, color: C.ink, alpha: rgA });
  rpt(1, { col: C.blue, label: 'decode', dx: -24, dy: -22, align: 'right' });
  // ① FP8：每个权重 1 字节，强度 × 2（斜线上方）
  const a1 = fade(u, bt(k, 0) + 0.3), k1 = p(u, bt(k, 0) + 0.3, bt(k, 0) + 1.3);
  const m1 = sarrow(1, 2, -34, C.tealD, a1, k1); badge('1', m1[0] - 26, m1[1] - 22, C.tealD, a1);
  // ③ speculative decoding：一次验证 4 个 token，强度 × 4（斜线下方）
  const a3 = fade(u, bt(k, 2)), k3 = p(u, bt(k, 2), bt(k, 2) + 1.2);
  const m3 = sarrow(1, 4, 40, C.blue, a3, k3); badge('3', m3[0] + 24, m3[1] + 24, C.blue, a3);
  // ② MLA：attention 的强度 ≈ 4 → ≈ 240
  const a2 = fade(u, bt(k, 1)), k2 = p(u, bt(k, 1) + 0.4, bt(k, 1) + 2);
  rpt(4, { col: C.purple, alpha: a2 * (1 - 0.6 * k2), r: 11 });
  const m2 = sarrow(4, 240, 44, C.purple, a2, k2); badge('2', m2[0] + 24, m2[1] + 24, C.purple, a2);
  rpt(240, { col: C.purple, alpha: fade(u, bt(k, 1) + 2), label: 'attention（MLA）≈ 240', dx: -24, dy: -26, align: 'right' });
  // 右侧：三个手段的说明与 H100 / B200 对照
  const item = (n, col, head, sub, y, a) => { badge(n, PX + 17, y - 8, col, a); txt(head, PX + 50, y, { size: 26, weight: 700, color: col, alpha: a }); if (sub) note(sub, PX + 50, y + 36, { alpha: a }); };
  item('1', C.tealD, 'FP8 权重：每个权重 1 字节', '强度 × 2；FP8 的平顶升至 1979 TFLOPS', 330, a1);
  item('2', C.purple, 'MLA：所有 query head 共用一份 latent', '', 430, a2);
  const kvRow = (name, kb, y) => { txt(name, PX + 50, y + 26, { size: 21, weight: 600, color: C.ink2, alpha: a2 }); rr(PX + 290, y + 6, 360 * kb / 327, 26, C.purple, a2); txt(`${kb} KB`, PX + 300 + 360 * kb / 327, y + 28, { size: 21, weight: 700, color: C.purple, alpha: a2 }); };
  kvRow('Qwen2.5-72B（GQA）', 327, 450); kvRow('DeepSeek-V3（MLA）', 70, 490);
  note('每 token 的 KV cache（BF16）', PX + 50, 550, { alpha: a2 });
  item('3', C.blue, 'speculative decoding：读取一次，验证多个 token', '图中按一次验证 4 个 token 示意', 620, a3);
  const ta = fade(u, bt(k, 3) + 0.6);
  const tbl = [['', 'H100', 'B200'], ['算力（BF16）', '989 TFLOPS', '2250 TFLOPS'], ['带宽', '3.35 TB/s', '8 TB/s'], ['拐点', '295', '281']];
  tbl.forEach((r, i) => r.forEach((c, j) => txt(c, PX + 50 + [0, 230, 470][j], 735 + i * 46, { size: 23, weight: i === 0 || j === 0 || i === 3 ? 700 : 600, color: j === 2 ? C.gray : C.ink, alpha: ta })));
  line(PX + 50, 748, PX + 700, 748, C.lg, 2, ta);
}
(() => { const k = I.levers; ev(k, bt(k, 0) + 0.3, 'whoosh'); ev(k, bt(k, 1) + 0.4, 'whoosh'); ev(k, bt(k, 2), 'whoosh'); ev(k, bt(k, 3), 'pop'); })();

function sEnd(u) {
  const k = I.end; header(k, u);
  chat(96, 300, 860, 500, 300 + u * M.tps1, { rate: '145 tok/s（实测）' });
  const sa = fade(u, bt(k, 0) + 0.5);
  const steps = [['1', '字节数：每一步从显存读取的量', C.blue], ['2', '运算次数：每一步的乘加次数', C.tealD], ['3', '各除以带宽与算力，取较大的一个', C.ink]];
  steps.forEach(([n, s, col], i) => { const a = fade(u, bt(k, 1) + i * 0.5); rr(1060, 330 + i * 90, 52, 52, col, a); txt(n, 1086, 368 + i * 90, { size: 30, weight: 800, color: '#fff', align: 'center', alpha: a }); txt(s, 1132, 368 + i * 90, { size: 30, weight: 700, color: C.ink, alpha: a }); });
  timeline(96, 900, 1640, 8, [], { ticks: [
    { ms: T_MEM, label: '下限 = 15.1 GB ÷ 3.35 TB/s = 4.52 ms', col: C.blue, a: sa, dash: [8, 6], align: 'right' },
    { ms: M.itl1, label: '实测 6.84 ms' },
  ] });
  const x0 = 96 + 1640 * T_MEM / 8, x1 = 96 + 1640 * M.itl1 / 8;
  rr(x0, 900 - 2, x1 - x0, 8, C.red, sa * 0.85);
}
(() => { const k = I.end; ev(k, bt(k, 0) + 0.5, 'land'); for (let i = 0; i < 3; i++) ev(k, bt(k, 1) + i * 0.5, 'pop'); })();

// ---------- 字幕 ----------
// 配音稿逐句显示在画面底部：按标点切成不超过 26 字的短句。
// 时间来源：vo.js（vo.py 合成后生成，含每句实测时长）存在且文字一致时，按实测时长在这一句内按字数分配；否则按字数估算，不越过下一个节拍。
function cues(text, lim = 26) {
  const out = [];
  for (const p0 of (text.match(/[^，。：；？！]+[，。：；？！]?/g) || []).map((x) => x.trim()).filter(Boolean)) {
    const t = p0.replace(/[，。；]$/, '');
    if (out.length && out[out.length - 1].length + t.length + 1 <= lim && !/[？！]$/.test(out[out.length - 1])) out[out.length - 1] += (out[out.length - 1].endsWith('：') ? '' : '，') + t;
    else out.push(t);
  }
  return out;
}
const est = (t) => (t.match(/[一-鿿]/g) || []).length / 4.6 + (t.match(/[A-Za-z0-9][A-Za-z0-9.,%\-]*/g) || []).length * 0.35;
const VOB = typeof VO === 'undefined' ? null : VO.beats;
const SUBS = SCRIPT.map((sc, k) => sc.beats.flatMap(([t0, , text], j) => {
  if (!text) return [];
  const cs = cues(text), end = j + 1 < sc.beats.length ? sc.beats[j + 1][0] - 0.1 : sc.dur - 0.1, total = cs.reduce((a, c) => a + est(c), 0);
  const v = VOB?.[k]?.[j], span = v && v.text === text ? v.dur : Math.min(total, end - t0);
  let t = t0;
  return cs.map((c) => { const d = span * est(c) / total; const r = [t, t + d, c]; t += d; return r; });
}));
function subtitle(k, u) {
  const c = SUBS[k].find(([a, b]) => u >= a && u < b); if (!c) return;
  const w = tw(c[2], 34, 600);
  rr(960 - w / 2 - 20, 1006, w + 40, 54, C.paper, 0.92);
  txt(c[2], 960, 1045, { size: 34, weight: 600, align: 'center', color: C.ink });
}

const SCENES = { open: sOpen, work: sWork, speeds: sSpeeds, roofline: sRoofline, batch: sBatch, prefill: sPrefill, kv: sKv, ledger: sLedger, bound: sBound, levers: sLevers, end: sEnd };
function draw(t) {
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  rr(0, 0, W, H, C.paper);
  SCENES[SCRIPT[k].id](t - AT[k]);
  subtitle(k, t - AT[k]);
}
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
