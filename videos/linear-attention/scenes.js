// 线性 Attention 影片的各幕（顺序与 id 见 script.js）。每幕一个 draw(u)，u 是幕内时间（秒）；节拍起点取自 script.js，音效事件登记进 EV。
// 贯穿全片的图形：一个 head 的 state，128 × 128 的热图（core.js 的 heat）；开场与结尾是它与 full attention 一层 KV 的真实面积对比。
// 颜色与文章的自绘图一致：teal 线性 attention 层与它的 state，purple full attention 层与它的 KV；
// state 热图正值 teal、负值 amber；第 6 幕 amber / blue / green 为旧 value、新 value 与无关的另一条关联。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const I = Object.fromEntries(SCRIPT.map((s, i) => [s.id, i]));   // 幕的 id → 序号
const EV = SCRIPT.map(() => []);                      // 每幕的音效：[幕内时间, 种类]
const ev = (k, u, kind) => EV[k].push([u, kind]);
const evRun = (k, a, b, step, kind) => { for (let u = a; u < b - 1e-6; u += step) ev(k, u, kind); };
const fade = (u, a, d = 0.45) => po(u, a, a + d);
const KiB = 1024, MiB = KiB * KiB, GiB = MiB * KiB;
const GHOST = '#e4ded0';

// 页眉：左上角是片名，下面是这一幕的标签与标题，右上角是出处
function header(k, u) {
  const s = SCRIPT[k], a = fade(u, 0, 0.5);
  txt('线性 Attention', 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 76, { size: 24, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 226 + (1 - a) * 16, 56, 700, a);
}
const note = (s, x, y, o = {}) => txt(s, x, y, { size: 22, weight: 500, color: C.gray, ...o });
const lab = (s, x, y, o = {}) => txt(s, x, y, { size: 26, weight: 700, ...o });

// ---------- 公式：斜体衬线，parts 为字符串或 [字符串, 颜色]；返回各段的起点与终点 ----------
const MF = '"STIX Two Text","STIXGeneral","Times New Roman",serif';
// 简单的上下标：x_t、k_i^T、S_{t−1}；返回 [[文字, 字号比例, 纵向偏移比例], …]
function mparse(s) {
  const out = []; let i = 0, buf = '';
  const flush = () => { if (buf) out.push([buf, 1, 0]); buf = ''; };
  while (i < s.length) {
    const c = s[i];
    if ((c === '_' || c === '^') && i + 1 < s.length) {
      flush(); let grp;
      if (s[i + 1] === '{') { const j = s.indexOf('}', i + 2); grp = s.slice(i + 2, j); i = j + 1; } else { grp = s[i + 1]; i += 2; }
      out.push([grp, 0.62, c === '_' ? 0.24 : -0.42]);
    } else { buf += c; i++; }
  }
  flush(); return out;
}
const mw = (s, size) => { let w = 0; for (const [t, k] of mparse(s)) { g.font = `italic 400 ${size * k}px ${MF}`; w += g.measureText(t).width + (k < 1 ? 1 : 0); } return w; };
function mrow(parts, x, y, size = 44, alpha = 1, base = C.ink) {
  const pos = [];
  for (const pt of parts) {
    const [s, col] = typeof pt === 'string' ? [pt, base] : pt, x0 = x;
    for (const [t, k, dy] of mparse(s)) {
      g.font = `italic 400 ${size * k}px ${MF}`;
      if (alpha > 0.003) { g.globalAlpha = alpha; g.fillStyle = col; g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillText(t, x, y + dy * size); g.globalAlpha = 1; }
      x += g.measureText(t).width + (k < 1 ? 1 : 0);
    }
    pos.push([x0, x]);
  }
  return pos;
}
const mtxt = (s, x, y, size = 30, o = {}) => mrow([[s, o.color || C.ink]], o.align === 'center' ? x - mw(s, size) / 2 : x, y, size, o.alpha ?? 1);
function ellipse(cx, cy, rx, ry, col, alpha = 1, lw = 3) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = lw; g.beginPath(); g.ellipse(cx, cy, rx, ry, 0, 0, 7); g.stroke(); g.globalAlpha = 1;
}
// 公式下方的花括号与说明
function brace(x0, x1, y, s, col, alpha = 1, align = 'center') {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath();
  const m = (x0 + x1) / 2; g.moveTo(x0, y); g.quadraticCurveTo(x0, y + 12, x0 + 12, y + 12); g.lineTo(m - 10, y + 12); g.quadraticCurveTo(m, y + 12, m, y + 22);
  g.quadraticCurveTo(m, y + 12, m + 10, y + 12); g.lineTo(x1 - 12, y + 12); g.quadraticCurveTo(x1, y + 12, x1, y); g.stroke(); g.globalAlpha = 1;
  txt(s, align === 'center' ? m : align === 'right' ? x1 : x0, y + 52, { size: 24, weight: 700, color: col, align, alpha });
}

// ---------- 示例数据（随机向量，seed 固定） ----------
// 画面上的示例向量取平滑的随机向量，外积呈清楚的格纹；第 4 幕的模拟用独立高斯的随机单位向量
function sunit(seed, w = 9) { const x = unit(seed), y = new Float32Array(D); let s = 0; for (let i = 0; i < D; i++) { let a = 0; for (let j = -w; j <= w; j++) a += x[(i + j + D) % D]; y[i] = a; s += a * a; } s = Math.sqrt(s); for (let i = 0; i < D; i++) y[i] /= s; return y; }
const V = (i) => sunit(1000 + i), K = (i) => sunit(2000 + i);
const outer = (v, k, w = 1) => addOuter(new Float32Array(D * D), v, k, w);
const SUM4 = (() => { const S = new Float32Array(D * D); for (let i = 0; i < 12; i++) addOuter(S, V(i), K(i)); return S; })();
const OUT_SCALE = 0.03;
const S_NEXT = (() => { const S = new Float32Array(D * D); for (let i = 0; i < 16; i++) addOuter(S, V(i), K(i)); return S; })();   // 第 12 幕：越过 draft 之后的 state（示意）
                               // 单个外积的满色值：单位向量的分量约为 ±0.09，乘积约 ±0.008
const S_SCALE = 0.11;

// 一个 head 的 state 方块，标出 d_v × d_k
function stateBox(key, M, x, y, s, { alpha = 1, scale = S_SCALE, dims = false, label = '' } = {}) {
  heat(key, M, x, y, s, s, { scale, alpha });
  if (dims) { note('d_k = 128', x + s / 2, y - 12, { align: 'center', alpha }); note('128', x - 12, y + s / 2 + 8, { align: 'right', alpha }); }
  if (label) mtxt(label, x + s / 2, y + s + 42, 34, { align: 'center', alpha, color: C.tealD });
}

// ================= 1 开场 =================
// 一句普通的话逐个 token 读进来：每读一个，中间的 state 加上一个外积，图样变了，大小不变；底下 full attention 的 KV 每个 token 多一格。
const SENT = ['小明', '把', '钥匙', '放在', '厨房', '的', '抽屉', '里', '，', '然后', '出门', '上班', '了', '。'];
const NT = SENT.length, tokT = (i) => 0.9 + i * 0.5;
const ACC = []; { const S = new Float32Array(D * D); for (let i = 0; i < NT; i++) { addOuter(S, V(100 + i), K(100 + i)); ACC.push(new Float32Array(S)); } }
const SQ = { x: 740, y: 420, s: 380 };
// 读到第几个 token（可为小数：小数部分是 state 换图的过渡）
const readN = (u) => { let n = 0; while (n < NT && u >= tokT(n)) n++; return n; };
function sentence(u, alpha = 1, y = 340, all = false) {
  let x = 96; const n = all ? NT : readN(u);
  SENT.forEach((t, i) => {
    const w = tw(t, 36, 600), cur = !all && i === n - 1;
    txt(t, x, y, { size: 36, weight: 600, color: i < n ? C.ink : C.lg, alpha });
    if (cur) rr(x, y + 12, w, 5, C.teal, alpha);
    x += w + 14;
  });
}
function stateAt(u, alpha = 1, all = false) {
  const n = all ? NT : readN(u);
  if (n === 0) { rr(SQ.x, SQ.y, SQ.s, SQ.s, C.paper, alpha); sr(SQ.x, SQ.y, SQ.s, SQ.s, C.ink, 2, alpha); }
  else {
    const k = all ? 1 : clamp((u - tokT(n - 1)) / 0.25);
    if (n > 1) heat('acc' + (n - 2), ACC[n - 2], SQ.x, SQ.y, SQ.s, SQ.s, { scale: S_SCALE, alpha: alpha * (1 - k), frame: false });
    heat('acc' + (n - 1), ACC[n - 1], SQ.x, SQ.y, SQ.s, SQ.s, { scale: S_SCALE, alpha: alpha * (n > 1 ? k : k) });
  }
  lab('state', SQ.x, SQ.y - 20, { color: C.tealD, alpha });
  note('128 × 128，尺寸不随 token 数变化', SQ.x, SQ.y + SQ.s + 34, { alpha });
}
// full attention 的 KV：每个 token 一格
function kvRow(u, alpha = 1, all = false) {
  const n = all ? NT : readN(u), y = 900, cw = 56;
  lab('full attention 的 KV', 96, y - 20, { color: C.purple, alpha });
  for (let i = 0; i < n; i++) rr(96 + i * (cw + 6), y, cw, 40, C.purple, alpha * (all ? 1 : clamp((u - tokT(i)) / 0.2)));
  txt(`${n} 份`, 96 + n * (cw + 6) + 12, y + 31, { size: 26, weight: 700, color: C.purple, alpha });
}
function bigTitle(out, alpha = 1) {
  const size = mix(110, 28, out), w1 = tw('线性', 110, 800) + 30, w2 = tw('线性 ', 28, 700);
  txt('线性', 96, mix(190, 76, out), { size, weight: mix(800, 700, out), color: out > 0.5 ? C.ink2 : C.ink, alpha });
  txt('Attention', mix(96 + w1, 96 + w2, out), mix(190, 76, out), { size, weight: mix(800, 700, out), color: mixc(C.teal, C.ink2, out), alpha });
}
const S1 = { out: [SCRIPT[I.open].dur - 1.4, SCRIPT[I.open].dur - 0.2] };
ev(I.open, 0.2, 'land'); for (let i = 0; i < NT; i++) ev(I.open, tokT(i), 'tick'); ev(I.open, S1.out[0], 'whoosh');
function s1(u) {
  const out = p(u, ...S1.out), keep = 1 - out;
  bigTitle(out, fade(u, 0.1, 0.6));
  sentence(u, keep); stateAt(u, keep); kvRow(u, keep);
}

// ================= 两种读法 =================
// 同一个 q 读开场那句话的 14 个 token。key 含公共分量（两两点积约 0.5，示意），full attention 用 softmax，线性 attention 直接用点积。
const RT = 6;                                          // 目标：“抽屉”
const RK = (() => { const c = unit(9000); return SENT.map((_, i) => { const r = unit(9100 + i), k = new Float32Array(D); let n = 0; for (let j = 0; j < D; j++) { k[j] = c[j] + r[j]; n += k[j] * k[j]; } n = Math.sqrt(n); return k.map((x) => x / n); }); })();
const RQ = (() => { const c = unit(9000), r = unit(9100 + RT), q = new Float32Array(D); let n = 0; for (let j = 0; j < D; j++) { q[j] = c[j] + 1.15 * r[j]; n += q[j] * q[j]; } n = Math.sqrt(n); return q.map((x) => x / n); })();
const RV = SENT.map((_, i) => sunit(9300 + i));
const RDOT = RK.map((k) => k.reduce((a, x, j) => a + x * RQ[j], 0));
const RSM = (() => { const e = RDOT.map((d) => Math.exp(9 * d)), s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); })();
const readOut = (w) => { const o = new Float32Array(D); w.forEach((wi, i) => { for (let j = 0; j < D; j++) o[j] += wi * RV[i][j]; }); return o; };
const RO_FULL = readOut(RSM), RO_LIN = readOut(RDOT);
const RCOS = [cosine(RO_FULL, RV[RT]), cosine(RO_LIN, RV[RT])];
const S_R = { full: bt(I.read, 1), lin: bt(I.read, 2) };
ev(I.read, 0.5, 'whoosh'); ev(I.read, 1.6, 'pop'); evRun(I.read, S_R.full + 0.3, S_R.full + 1.7, 0.1, 'tick'); ev(I.read, S_R.full + 3.0, 'land'); evRun(I.read, S_R.lin + 0.3, S_R.lin + 1.7, 0.1, 'tick'); ev(I.read, S_R.lin + 3.0, 'land');
function sRead(u) {
  header(I.read, u);
  const X0 = 330, STEP = 80, BW = 48, a0 = fade(u, 0.4);
  txt('当前 token 的 query q：设它与“抽屉”的 key 最匹配', 96, 316, { size: 26, weight: 700, color: C.ink2, alpha: a0 });
  rr(96 + tw('当前 token 的 query q：设它与“抽屉”的 key 最匹配', 26, 700) + 16, 296, 22, 22, C.yel, fade(u, 1.6));
  note('分词为 Qwen3.5 tokenizer 的实际结果', 1824, 316, { align: 'right', alpha: a0 });
  const rows = [
    { y: 560, w: RSM, st: S_R.full, name: 'full attention', sub: 'softmax 权重：和为 1', col: C.purple, h: 170, cos: RCOS[0], ro: RO_FULL, fmt: (x) => x.toFixed(2) },
    { y: 880, w: RDOT, st: S_R.lin, name: '线性 attention', sub: `点积权重：和为 ${RDOT.reduce((a, b) => a + b, 0).toFixed(1)}`, col: C.teal, h: 170, cos: RCOS[1], ro: RO_LIN, fmt: (x) => x.toFixed(2) },
  ];
  rows.forEach((r, n) => {
    const a = fade(u, r.st);
    lab(r.name, 96, r.y - 120, { color: n ? C.tealD : C.purple, alpha: a });
    note(r.sub, 96, r.y - 84, { alpha: a });
    line(X0 - 20, r.y, X0 + 13 * STEP + BW + 20, r.y, C.ink, 2, a);
    SENT.forEach((t, i) => {
      const grow = p(u, r.st + 0.3 + i * 0.1, r.st + 0.8 + i * 0.1), hh = r.h * r.w[i] * grow, x = X0 + i * STEP;
      rr(x, r.y - hh, BW, hh, i === RT ? r.col : mixc(r.col, C.paper, 0.55), a);
      txt(t, x + BW / 2, r.y + 32, { size: 22, weight: i === RT ? 800 : 500, align: 'center', color: i === RT ? C.ink : C.gray, alpha: a });
      if (i === RT || n === 1 && i === 0) txt(r.fmt(r.w[i]), x + BW / 2, r.y - hh - 10, { size: 20, weight: 700, align: 'center', alpha: a * grow });
    });
    // 读出：与 v_抽屉 对比
    const ra = fade(u, r.st + 3.0), RX = 1500, RW = 324;
    note('v（抽屉）', RX, r.y - 140, { alpha: ra, size: 20 }); vec('rv' + RT, RV[RT], RX, r.y - 128, RW, 22, { scale: 0.18, alpha: ra });
    note('读出 o', RX, r.y - 80, { alpha: ra, size: 20 }); vec('ro' + n, r.ro.map((x) => x / Math.max(1e-6, Math.hypot(...r.ro))), RX, r.y - 68, RW, 22, { scale: 0.18, alpha: ra });
    txt(`相似度 ${r.cos.toFixed(2)}`, RX, r.y - 6, { size: 28, weight: 800, color: r.cos < 0.6 ? C.red : C.ink, alpha: ra });
  });
}


// ================= 2 去掉 softmax =================
// 两列对照：左 full attention，右线性 attention，逐行对齐（权重 / 输出 / 能不能把 q 提出来 / 要存什么）。
// 右列的公式按项移动（同一项在前后两式里同色、同 id），看得出每一步挪了什么。q 深黄、k 蓝、v 绿、S teal。
const QC = C.yelD, KC = C.blue, VC = C.green;
// 公式变形：A、B 为 [[id, 文字, 颜色], …]；同 id 的项从 A 的位置移到 B 的位置，文字不同则原地交叉淡化；只在一边出现的项淡入淡出
function layout(F, x, size) { const pos = {}; for (const [id, t] of F) { const w = mw(t, size); pos[id] = [x, w]; x += w; } return pos; }
function morph(A, B, k, x, y, size, alpha = 1) {
  const pa = layout(A, x, size), pb = layout(B, x, size), e = ease(k), ia = Object.fromEntries(A.map((f) => [f[0], f])), ib = Object.fromEntries(B.map((f) => [f[0], f]));
  for (const [id, t, col] of A) {
    if (!ib[id]) { mrow([[t, col]], pa[id][0], y, size, alpha * (1 - e)); continue; }
    const xx = mix(pa[id][0], pb[id][0], e), lift = Math.sin(Math.PI * e) * (ib[id][1] === t && Math.abs(pb[id][0] - pa[id][0]) > 40 ? -26 : 0);
    if (ib[id][1] === t) mrow([[t, col]], xx, y + lift, size, alpha);
    else { mrow([[t, col]], xx, y, size, alpha * (1 - e)); mrow([[ib[id][1], ib[id][2]]], xx, y, size, alpha * e); }
  }
  for (const [id, t, col] of B) if (!ia[id]) mrow([[t, col]], pb[id][0], y, size, alpha * e);
  return pb;
}
const MF0 = [['o', 'o_t', C.ink], ['eq', ' = ', C.ink], ['sum', 'Σ_i ', C.ink], ['lp', '(', C.ink], ['q', 'q_t^T', QC], ['k', 'k_i', KC], ['rp', ')', C.ink], ['v', ' v_i', VC]];
const MF1 = [['o', 'o_t', C.ink], ['eq', ' = ', C.ink], ['sum', 'Σ_i ', C.ink], ['v', 'v_i ', VC], ['lp', '(', C.ink], ['k', 'k_i^T', KC], ['q', 'q_t', QC], ['rp', ')', C.ink]];
const MF2 = [['o', 'o_t', C.ink], ['eq', ' = ', C.ink], ['lp2', '(', C.tealD], ['sum', 'Σ_i ', C.tealD], ['v', 'v_i ', VC], ['k', 'k_i^T', KC], ['rp2', ')', C.tealD], ['q', ' q_t', QC]];
const MF3 = [['o', 'o_t', C.ink], ['eq', ' = ', C.ink], ['S', 'S_t', C.tealD], ['q', ' q_t', QC]];
const S2 = { f: 0.5, m: bt(I.softmax, 1), run: bt(I.softmax, 3), cmp: bt(I.softmax, 3) + 6 };
const T_MAX = 4096, RUN = [S2.run + 0.6, S2.cmp];
const tAt = (u) => (u < RUN[0] ? 1 : Math.round(Math.pow(T_MAX, clamp((u - RUN[0]) / (RUN[1] - RUN[0])))));
ev(I.softmax, 0.5, 'whoosh'); ev(I.softmax, S2.m, 'pop'); [0, 1.6, 3.2].forEach((d) => ev(I.softmax, S2.m + 0.6 + d, 'click')); evRun(I.softmax, RUN[0], RUN[1], 0.6, 'tick'); ev(I.softmax, S2.cmp, 'land');
function s2(u) {
  header(I.softmax, u);
  const L = 96, R = 1010, CW = 800, a0 = fade(u, 0.3), FS = 46, NS = 38;
  line(960, 290, 960, 1000, C.lg, 2, a0);
  lab('full attention', L, 300, { color: C.purple, alpha: a0, size: 30 }); lab('线性 attention', R, 300, { color: C.tealD, alpha: a0, size: 30 });
  // 第一行：输出式。左边带 softmax；右边去掉 softmax 后按项移动成 S_t q_t
  const r1 = 420, af = fade(u, S2.f);
  const lw = mrow(['o_t = Σ_i '], L, r1, FS, af)[0][1];
  const nm = [['exp(', C.ink], ['q_t^T', QC], ['k_i', KC], [')', C.ink]], dn = [['Σ_j exp(', C.ink], ['q_t^T', QC], ['k_j', KC], [')', C.ink]];
  const wid = (ps) => ps.reduce((a, [t]) => a + mw(t, NS), 0), fw = Math.max(wid(nm), wid(dn));
  const np = mrow(nm, lw + (fw - wid(nm)) / 2, r1 - 24, NS, af); line(lw, r1 - 12, lw + fw, r1 - 12, C.ink, 2.5, af); const dp = mrow(dn, lw + (fw - wid(dn)) / 2, r1 + 32, NS, af);
  mrow([' ', ['v_i', VC]], lw + fw + 4, r1, FS, af);
  const ea = fade(u, S2.m), sx = mw('Σ_j ', NS);
  sr(np[0][0] - 6, r1 - 60, np[3][1] - np[0][0] + 12, 48, C.red, 2.5, ea); sr(dp[0][0] + sx - 6, r1 - 4, dp[3][1] - dp[0][0] - sx + 12, 48, C.red, 2.5, ea);
  txt('q 位于 exp 之内，无法提到求和之外', L, r1 + 100, { size: 26, weight: 700, color: C.red, alpha: ea });
  const seq = [[MF0, 0], [MF1, S2.m + 0.6], [MF2, S2.m + 2.2], [MF3, S2.m + 3.8]];
  let j = 0; while (j < seq.length - 2 && u >= seq[j + 1][1] + 0.9) j++;
  morph(seq[j][0], seq[j + 1][0], clamp((u - seq[j + 1][1]) / 0.9), R, r1, FS, af);
  const tips = ['去掉 softmax，权重为点积', '点积是标量：移到 v_i 之后', 'q 与求和下标 i 无关：提到求和之外', ''];
  const ti = u >= S2.m + 3.8 ? 3 : u >= S2.m + 2.2 ? 2 : u >= S2.m + 0.6 ? 1 : 0;
  if (tips[ti]) txt(tips[ti], R, r1 + 100, { size: 26, weight: 700, color: C.gray, alpha: af });
  const sa = fade(u, S2.m + 4.4);
  mrow([['S_t', C.tealD], ' = Σ_i ', ['v_i', VC], ['k_i^T', KC]], R, r1 + 104, 38, sa);
  txt('与 q 无关，token 到达时逐项累加', R + 300, r1 + 98, { size: 24, weight: 700, color: C.tealD, alpha: sa });
  // 第二行：生成一个 token 要读什么。上下文从 1 涨到 4096
  const r3 = 640, ar = fade(u, S2.run), t = tAt(u), T = (u - RUN[0]) % 0.6 / 0.6;
  txt(`上下文 ${t.toLocaleString('en-US')} 个 token`, L + CW, r3, { size: 26, weight: 800, align: 'right', color: C.ink, alpha: ar });
  txt('生成一个 token：读取全部 K、V', L, r3, { size: 26, weight: 700, alpha: ar });
  const len = Math.max(3, CW * t / T_MAX);
  [['K', KC], ['V', VC]].forEach(([nm2, col], c) => {
    const y = r3 + 30 + c * 46;
    mtxt(nm2, L, y + 28, 32, { color: col, alpha: ar });
    rr(L + 40, y, CW - 40, 34, C.paper2, ar); rr(L + 40, y, Math.max(3, (CW - 40) * t / T_MAX), 34, col, ar);
    if (u > RUN[0] && u < S2.cmp + 4) rr(L + 40 + (CW - 40) * t / T_MAX * T - 6, y - 4, 12, 42, C.yel, ar * 0.9);   // 每一步从头扫到尾
  });
  txt('生成一个 token：只读写 S', R, r3, { size: 26, weight: 700, alpha: ar });
  const SQ3 = 120, pulse = u > RUN[0] && u < S2.cmp + 4 ? 1 - T : 0;
  heat('sum40', SUM4, R, r3 + 24, SQ3, SQ3, { scale: S_SCALE, alpha: ar });
  sr(R - 4, r3 + 20, SQ3 + 8, SQ3 + 8, C.yel, 4, ar * pulse);
  mrow([['S_t', C.tealD], ' = ', ['S_{t−1}', C.tealD], ' + ', ['v_t', VC], ['k_t^T', KC]], R + 150, r3 + 70, 34, ar);
  mrow(['o_t = ', ['S_t', C.tealD], ' ', ['q_t', QC]], R + 150, r3 + 124, 34, ar);
  // 第三行：每一步读的数，两列同一个刻度（整条 = 4096 × 256）
  const r4 = 880, am = fade(u, S2.cmp), FULL = T_MAX * 256;
  [[L, t * 256, C.purple, `${(t * 256).toLocaleString('en-US')} 个数`, 't × 256（K、V 各 128 维）'], [R, D * D, C.teal, '16,384 个数', '128 × 128，不随上下文变']].forEach(([x, v, col, val, nt]) => {
    txt('每一步读取的数值个数', x, r4, { size: 24, weight: 700, alpha: am }); txt(val, x + CW, r4, { size: 26, weight: 800, align: 'right', color: col === C.teal ? C.tealD : col, alpha: am });
    rr(x, r4 + 14, CW, 30, C.paper2, am); rr(x, r4 + 14, Math.max(3, CW * v / FULL), 30, col, am);
    note(nt, x, r4 + 74, { alpha: am });
  });
}

// ================= 3 真实尺度 =================
// Qwen3.8-27B 的一层，256K 上下文：full attention 每步读的 KV 1 GiB 画成整页紫色；同一层 48 个 head 的 state 3 MiB 画成右下角的小方块，面积按真实比例
const CP = { x: 96, y: 300, w: 1280, h: 600 }, STATE_SIDE = Math.sqrt(CP.w * CP.h * 3 / 1024);
ev(I.scale, 0.5, 'whoosh'); evRun(I.scale, 0.8, 4.6, 0.14, 'tick'); ev(I.scale, bt(I.scale, 1), 'pop');
function s3(u) {
  header(I.scale, u);
  const prog = p(u, 0.8, 4.6), rows = 64, hh = CP.h / rows, full = Math.floor(prog * rows), fr = prog * rows - full;
  rr(CP.x, CP.y, CP.w, CP.h, GHOST, 0.6);
  rr(CP.x, CP.y, CP.w, hh * full, C.purple); if (full < rows) rr(CP.x, CP.y + hh * full, CP.w * fr, hh, C.purple);
  for (let r = 1; r <= full; r++) rr(CP.x, CP.y + r * hh - 0.6, CP.w, 1.2, '#9a6ff7', 0.5);
  lab('full attention 的一层', MX, 330, { color: C.purple });
  note(`上下文 ${Math.round(256 * prog)}K 个 token`, MX, 366);
  txt(`每步读 KV：${prog < 1 ? Math.round(prog * 1024) + ' MiB' : '1 GiB'}`, MX, 416, { size: 32, weight: 700, color: C.purple });
  const sa = fade(u, bt(I.scale, 1), 0.6), sx = CP.x + CP.w + 24, sy = CP.y + CP.h - STATE_SIDE;
  heat('sum40', SUM4, sx, sy, STATE_SIDE, STATE_SIDE, { scale: S_SCALE, alpha: sa, frame: false }); sr(sx, sy, STATE_SIDE, STATE_SIDE, C.tealD, 2, sa);
  line(sx + STATE_SIDE + 8, sy + STATE_SIDE / 2, MX + 40 + 30, sy + STATE_SIDE / 2, C.tealD, 2, sa);
  lab('同一层改用线性 attention', MX + 82, sy - 20, { color: C.tealD, alpha: sa });
  txt('每步读写 state：3 MiB', MX + 82, sy + 30, { size: 30, weight: 700, color: C.tealD, alpha: sa });
  note('与上下文长度无关', MX + 82, sy + 66, { alpha: sa });
  note('面积按真实比例：48 个 head × 128 × 128 × FP32 = 3 MiB；4 个 KV head × 256 维 × K、V × BF16 × 256K = 1 GiB', CP.x, CP.y + CP.h + 46, { alpha: fade(u, bt(I.scale, 1) + 1) });
}

// ================= 叠加的代价 =================
// 左：2 维的简化，key 是单位圆上的箭头；右：128 维随机单位 key 的模拟
const CAP = [16, 128, 1024].map((n) => {
  const S = new Float32Array(D * D); for (let i = 0; i < n; i++) addOuter(S, unit(5000 + i), unit(7000 + i));
  const r = matvec(S, unit(7000)); return { n, S, r, cos: cosine(r, unit(5000)) };
});
const SP = { add: bt(I.superpose, 1), sim: bt(I.superpose, 2) };
ev(I.superpose, 0.5, 'whoosh'); ev(I.superpose, 1.4, 'pop'); ev(I.superpose, 2.4, 'pop'); ev(I.superpose, 3.6, 'land'); ev(I.superpose, SP.add, 'pop'); ev(I.superpose, SP.add + 1.6, 'land'); CAP.forEach((_, i) => ev(I.superpose, SP.sim + 0.4 + i * 1.2, 'land'));
function sSuperpose(u) {
  header(I.superpose, u);
  const a0 = fade(u, 0.4), ad = fade(u, SP.add), grow = p(u, SP.add, SP.add + 1.0), U = 110;
  line(960, 290, 960, 1000, C.lg, 2, a0);
  lab('3 维（简化）', 96, 300, { alpha: a0, size: 28 });
  const V3 = '#b0479a';
  mrow(['S = ', ['v_1', C.amber], ['k_1^T', C.ink], ' + ', ['v_2', C.green], ['k_2^T', C.ink], ' + ', ['v_3', V3], ['k_3^T', C.ink]], 96, 370, 34, a0);
  mrow([' + ', ['v_4', C.blue], ['k_4^T', C.ink]], 96 + mw('S = v_1k_1^T + v_2k_2^T + v_3k_3^T', 34) + 6, 370, 34, ad);
  // 单位球：绕竖直轴缓慢旋转，再俯视一点
  const CX = 280, CY = 650, R = 170, th = -0.55 + 0.35 * Math.sin(u / 22 * Math.PI), ph = 0.38;
  const prj = ([x, y, z]) => { const x1 = x * Math.cos(th) + z * Math.sin(th), z1 = -x * Math.sin(th) + z * Math.cos(th); const y2 = y * Math.cos(ph) - z1 * Math.sin(ph); return [CX + R * x1, CY - R * y2]; };
  g.globalAlpha = a0; g.strokeStyle = C.lg; g.lineWidth = 2; g.beginPath(); g.arc(CX, CY, R, 0, 7); g.stroke();
  g.beginPath(); for (let i = 0; i <= 64; i++) { const t = i / 64 * 2 * Math.PI, [x, y] = prj([Math.cos(t), 0, Math.sin(t)]); i ? g.lineTo(x, y) : g.moveTo(x, y); } g.stroke(); g.globalAlpha = 1;
  const key = (v, lbl, a, col = C.ink) => { const [x, y] = prj(v), [ox, oy] = prj([0, 0, 0]); arrow(ox, oy, x, y, col, a, 4); const dx = x - ox, dy = y - oy, n = Math.hypot(dx, dy) || 1; mtxt(lbl, x + 26 * dx / n - 12, y + 26 * dy / n + 12, 30, { alpha: a, color: col }); };
  key([1, 0, 0], 'k_1', fade(u, 1.2)); key([0, 1, 0], 'k_2', fade(u, 1.8)); key([0, 0, 1], 'k_3', fade(u, 2.4));
  const s3 = 1 / Math.sqrt(3), c = s3 * grow;
  // k₄ 所在立方体的棱（虚线），表明它是 (1, 1, 1) 方向的对角线
  [[[c, c, c], [c, c, 0]], [[c, c, c], [c, 0, c]], [[c, c, c], [0, c, c]], [[c, c, 0], [c, 0, 0]], [[c, c, 0], [0, c, 0]], [[c, 0, c], [c, 0, 0]], [[c, 0, c], [0, 0, c]], [[0, c, c], [0, c, 0]], [[0, c, c], [0, 0, c]]]
    .forEach(([p0, p1]) => { const [x0, y0] = prj(p0), [x1, y1] = prj(p1); line(x0, y0, x1, y1, C.blue, 1.5, ad * 0.6, [5, 5]); });
  key([c, c, c], 'k_4', ad, C.blue);
  note('三个 key 两两垂直；k₄ 与每个坐标轴成 54.7°', 96, CY + R + 60, { alpha: ad, size: 21 });
  note('key 均为单位向量（L2 归一化）', 96, CY + R + 94, { alpha: fade(u, 2.4), size: 21 });
  // 读出：用 k₁、k₂、k₃ 各读一次
  const BX = 520, a3 = fade(u, 3.4), seg = (x, y, len, col, l, a) => { rr(x, y, len * U - 3, 38, col, a); if (l) mtxt(l, x + len * U / 2, y + 28, 24, { align: 'center', color: '#fff', alpha: a }); };
  [['k_1', C.amber, 'v_1', 470], ['k_2', C.green, 'v_2', 590], ['k_3', V3, 'v_3', 710]].forEach(([k, col, v, y]) => {
    txt('用', BX, y - 14, { size: 24, weight: 600, alpha: a3 }); mtxt(k, BX + 32, y - 14, 28, { alpha: a3 }); txt('读出', BX + 32 + mw(k, 28) + 8, y - 14, { size: 24, weight: 600, alpha: a3 });
    seg(BX, y, 1, col, v, a3);
    const m = 0.58 * grow; if (u > SP.add) seg(BX + U, y, m, C.blue, '', ad);
    mtxt(u > SP.add + 1.0 ? `${v} + 0.58 v_4` : v, BX + U * (1 + (u > SP.add ? m : 0)) + 12, y + 28, 26, { alpha: a3, color: u > SP.add + 1.0 ? C.red : C.ink });
  });
  note('3 维至多 3 个相互正交的方向', BX, 820, { alpha: ad });
  // 右：128 维的模拟
  const RX = 1000, sa = fade(u, SP.sim);
  lab('128 维', RX, 300, { alpha: sa, size: 28 });
  note('写入 n 个随机单位 key，再以 k₁ 为 query 读出', RX, 340, { alpha: sa });
  CAP.forEach(({ n, r, cos }, i) => {
    const ry = 420 + i * 160, a = fade(u, SP.sim + 0.4 + i * 1.2);
    lab(`n = ${n}`, RX, ry + 40, { alpha: a });
    note('原值 v₁', RX + 150, ry + 22, { alpha: a, size: 20 }); vec('v500', unit(5000), RX + 260, ry, 420, 26, { scale: 0.18, alpha: a });
    note('读出 S k₁', RX + 150, ry + 76, { alpha: a, size: 20 }); vec('r' + n, r, RX + 260, ry + 54, 420, 26, { scale: 0.18, alpha: a });
    txt(cos.toFixed(2), RX + 700, ry + 52, { size: 30, weight: 800, color: cos < 0.5 ? C.red : C.ink, alpha: a });
  });
  note('右列数字：读出与 v₁ 的余弦相似度', RX, 900, { alpha: fade(u, SP.sim + 3.0) });
}


// ================= 实际的 key =================
// 实测：Qwen3.5-0.8B（data-keys.js）。上：第 17 层 head 6 中，第一个“钥匙”的 key 与每个 token 的 key 的余弦；下：288 个 GDN head 各自的平均 |余弦|，与同维度随机单位向量对比。
const KD = KEY_DATA, S_K = { row: 0.6, rep: bt(I.realkeys, 1), all: bt(I.realkeys, 2) };
ev(I.realkeys, 0.5, 'whoosh'); evRun(I.realkeys, 1.0, 3.6, 0.08, 'tick'); ev(I.realkeys, S_K.rep, 'pop'); evRun(I.realkeys, S_K.all + 0.4, S_K.all + 2.4, 0.06, 'tick');
function sRealKeys(u) {
  header(I.realkeys, u);
  const a0 = fade(u, 0.3), same = (i) => KD.tokens[i] === KD.tokens[KD.src_tok] && i !== KD.src_tok;
  AT_TOK.forEach((t, i) => {
    rr(cx0(i) + 3, 300, cw(i) - 6, 40, i === KD.src_tok || same(i) ? C.yel : '#e2dccd', a0 * (i === KD.src_tok || same(i) ? 0.7 : 1));
    txt(t, cx0(i) + cw(i) / 2, 329, { size: TOK_FS, weight: i === KD.src_tok ? 800 : 600, align: 'center', color: C.ink, alpha: a0 });
  });
  txt(`第 ${KD.layer} 层 · head ${KD.head}`, 310, 322, { size: 22, weight: 700, align: 'right', alpha: a0 });
  // 余弦条
  const BY = 660, BH = 270, ab = fade(u, S_K.row);
  note('与第一个“钥匙”的 key', 310, 420, { align: 'right', alpha: ab }); note('的余弦相似度', 310, 450, { align: 'right', alpha: ab });
  line(330, BY, 1824, BY, C.ink, 2, ab);
  KD.row.forEach((v, i) => {
    if (i === KD.src_tok) return;
    const grow = p(u, S_K.row + 0.4 + i * 0.08, S_K.row + 0.9 + i * 0.08), hh = BH * v * grow;
    rr(cx0(i) + 6, BY - Math.max(hh, 0), cw(i) - 12, Math.abs(hh), same(i) ? C.blue : mixc(C.blue, C.paper, 0.6), ab);
    if (same(i)) txt(v.toFixed(2), cx0(i) + cw(i) / 2, BY - hh - 12, { size: 24, weight: 800, align: 'center', color: C.blue, alpha: fade(u, S_K.rep) });
  });
  txt('自身', cx0(KD.src_tok) + cw(KD.src_tok) / 2, BY - 12, { size: 18, weight: 500, align: 'center', color: C.gray, alpha: ab });
  const ry = BY - BH * KD.random_mean_abs;
  line(330, ry, 1824, ry, C.red, 2, fade(u, S_K.rep), [8, 6]);
  txt(`红色虚线：随机单位向量的平均 |余弦| ${KD.random_mean_abs.toFixed(2)}`, 330, BY + 34, { size: 20, weight: 700, color: C.red, alpha: fade(u, S_K.rep) });
  line(330, BY - BH, 1824, BY - BH, C.lg, 1.5, ab, [4, 6]); note('1', 322, BY - BH + 7, { align: 'right', alpha: ab }); note('0', 322, BY + 7, { align: 'right', alpha: ab });
  // 288 个 head 的平均 |余弦|
  const aa = fade(u, S_K.all), SX0 = 330, SX1 = 1824, xs = (v) => SX0 + (SX1 - SX0) * v, SY = 860, hs = KD.heads_mean_abs;
  lab('全部 288 个 GDN head', 96, 820, { alpha: aa, size: 24 }); note('每个点是一个 head：', 96, 852, { alpha: aa, size: 19 }); note('key 两两 |余弦| 的平均', 96, 878, { alpha: aa, size: 19 });
  line(SX0, SY + 60, SX1, SY + 60, C.ink, 2, aa);
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => txt(String(v), xs(v), SY + 88, { size: 18, weight: 500, align: 'center', color: C.gray, alpha: aa }));
  const r = rng(5), shown = Math.floor(clamp((u - S_K.all - 0.4) / 2) * hs.length);
  g.globalAlpha = aa; g.fillStyle = C.blue;
  hs.forEach((v, i) => { const jy = r(); if (i < shown) { g.beginPath(); g.arc(xs(v), SY + 6 + jy * 44, 5, 0, 7); g.fill(); } });
  g.globalAlpha = 1;
  const med = [...hs].sort((a, b) => a - b)[hs.length >> 1], ma = fade(u, S_K.all + 2.6);
  line(xs(med), SY - 6, xs(med), SY + 60, C.blue, 3, ma); txt(`中位数 ${med.toFixed(2)}`, xs(med) + 10, SY - 8, { size: 22, weight: 800, color: C.blue, alpha: ma });
  line(xs(KD.random_mean_abs), SY - 6, xs(KD.random_mean_abs), SY + 60, C.red, 3, ma, [6, 4]); txt('随机 0.07', xs(KD.random_mean_abs) + 10, SY - 8, { size: 22, weight: 800, color: C.red, alpha: ma });
}

// ================= 6 Delta rule =================
const U6 = 200;
const ROWS6 = [
  { name: '写入前', rule: '已有两条关联', k: [[C.amber, 1, 'v_1']], kr: 'v_1', k2: [[C.green, 1, 'u']], k2r: 'u' },
  { name: '直接相加', rule: 'S + v₂kᵀ', k: [[C.amber, 1, 'v_1'], [C.blue, 1, 'v_2']], kr: 'v_1 + v_2', k2: [[C.green, 1, 'u']], k2r: 'u', bad: 0 },
  { name: '先 decay 再相加', rule: 'αS + v₂kᵀ', k: [[C.amber, 0.5, ''], [C.blue, 1, 'v_2']], kr: '0.5 v_1 + v_2', k2: [[C.green, 0.5, '']], k2r: '0.5 u', bad: 1 },
  { name: 'Delta rule', rule: 'S + β(v₂ − Sk)kᵀ', k: [[C.blue, 1, 'v_2']], kr: 'v_2', k2: [[C.green, 1, 'u']], k2r: 'u', focus: true },
];
ev(I.delta, 0.5, 'whoosh'); ev(I.delta, bt(I.delta, 1), 'pop'); ev(I.delta, bt(I.delta, 1) + 1.6, 'pop'); ev(I.delta, bt(I.delta, 2), 'land'); ev(I.delta, bt(I.delta, 3), 'click');
function s6(u) {
  header(I.delta, u);
  const C1 = 560, C2 = 1120, Y0 = 340, PITCH = 124, BH = 46;
  note('用 k 读出', C1, Y0 - 20, { alpha: fade(u, 0.4) }); note('用 k′ 读出（无关的另一条）', C2, Y0 - 20, { alpha: fade(u, 0.4) });
  const starts = [0.5, bt(I.delta, 1), bt(I.delta, 1) + 1.6, bt(I.delta, 2)];
  ROWS6.forEach((r, n) => {
    const y = Y0 + n * PITCH, a = fade(u, starts[n]);
    txt(r.name, 500, y + 22, { size: 28, weight: 700, align: 'right', alpha: a, color: r.focus ? C.tealD : C.ink });
    txt(r.rule, 500, y + 54, { size: 20, weight: 500, align: 'right', alpha: a, color: C.gray, font: MONO });
    const bar = (x, segs, tail) => {
      let cx = x;
      for (const [col, len, l] of segs) { rr(cx, y, len * U6 - 3, BH, col, a); if (l) mtxt(l, cx + len * U6 / 2, y + 33, 28, { align: 'center', color: '#fff', alpha: a }); cx += len * U6; }
      mtxt(tail, cx + 14, y + 33, 30, { alpha: a }); return cx;
    };
    const e1 = bar(C1, r.k, r.kr); bar(C2, r.k2, r.k2r);
    if (r.focus) sr(C1 - 8, y - 8, C2 + U6 + 90 - C1, BH + 16, C.tealD, 3, fade(u, bt(I.delta, 2) + 0.6));
    if (n === 0) line(160, y + BH + 22, 1500, y + BH + 22, C.lg, 2, a);
    if (r.bad === 0) txt('新旧 value 叠加', C1, y + BH + 30, { size: 22, weight: 700, color: C.red, alpha: fade(u, starts[n] + 0.8) });
    if (r.bad === 1) txt('无关的关联同样减弱', C2, y + BH + 30, { size: 22, weight: 700, color: C.red, alpha: fade(u, starts[n] + 0.8) });
  });
  note('k 与 k′ 正交；α = 0.5，β = 1', C1, Y0 + 4 * PITCH - 10, { alpha: fade(u, bt(I.delta, 2)) });
  // Gated DeltaNet
  const a3 = fade(u, bt(I.delta, 3)), fy = 940;
  lab('Gated DeltaNet', 96, fy - 46, { color: C.tealD, alpha: a3 });
  mrow(['S_t = ', ['α_t', C.red], ' S_{t−1} + ', ['β_t (v_t − α_t S_{t−1} k_t) k_t^T', C.tealD]], 96, fy + 8, 40, a3);
  note('先衰减，再写入差值', 860, fy, { alpha: a3, size: 24 });
}

// ================= 7 两种都要 =================
const NIAH = [
  { task: '重复的填充文本里找 pass-key', ctx: '8K', s: [98.8, 30.4, 91.8], low: 1, why: '只有 decay：pass-key 已衰减' },
  { task: '真实文章里找一个数字', ctx: '4K', s: [18.6, 56.2, 92.2], low: 0, why: '只有 delta rule：无关内容无法清除' },
];
const NM = [['DeltaNet', C.blue, '只有 delta rule'], ['Mamba2', C.amber, '只有 decay'], ['Gated DeltaNet', C.teal, '两者都有']];
ev(I.niah, 0.5, 'whoosh'); [0, 0.3, 0.6].forEach((d) => { ev(I.niah, 0.8 + d, 'pop'); ev(I.niah, bt(I.niah, 1) + d, 'pop'); }); ev(I.niah, bt(I.niah, 2), 'click');
function s7(u) {
  header(I.niah, u);
  const X0 = 400, BW = 820, BH = 44;
  NIAH.forEach((gp, n) => {
    const y0 = 330 + n * 300, st = n ? bt(I.niah, 1) : 0.5, a = fade(u, st);
    txt(gp.task, 96, y0, { size: 28, weight: 700, alpha: a }); note(`上下文 ${gp.ctx}`, 96 + tw(gp.task, 28, 700) + 18, y0, { alpha: a });
    gp.s.forEach((v, i) => {
      const y = y0 + 26 + i * (BH + 12), grow = p(u, st + 0.3 + i * 0.3, st + 1.2 + i * 0.3);
      txt(NM[i][0], X0 - 16, y + 32, { size: 24, weight: 700, align: 'right', alpha: a });
      rr(X0, y, BW, BH, C.paper2, a); rr(X0, y, BW * v / 100 * grow, BH, NM[i][1], a);
      txt((v * grow).toFixed(1), X0 + BW * v / 100 * grow + 12, y + 32, { size: 26, weight: 700, alpha: a, color: i === gp.low ? C.red : C.ink });
    });
    txt(gp.why, X0 + BW + 120, y0 + 26 + gp.low * (BH + 12) + 32, { size: 24, weight: 700, color: C.red, alpha: fade(u, st + 2.2) });
  });
  const a2 = fade(u, bt(I.niah, 2));
  NM.forEach(([nme, col, d], i) => { rr(96 + i * 560, 890, 24, 24, col, a2); txt(`${nme}：${d}`, 132 + i * 560, 910, { size: 22, weight: 500, color: C.ink2, alpha: a2 }); });
  note('准确率（%），1.3B 模型', 96, 956, { alpha: a2 });
}

// ================= α 由 token 决定 =================
// 实测：Qwen3.5-0.8B 读三句话（data.js，scripts/linear-attention/alpha_probe.py 生成）。选三个 GDN head：
// 每步几乎清空的、只在标点处衰减的、从不衰减的。下方是“钥匙”（第 3 个 token）写入后在各 head 中的保留比例 ∏α（只计衰减，不含 delta rule 的覆盖）。
const AD = ALPHA_DATA, AT_TOK = AD.tokens, KEY_TOK = 2;
const AH = [[0, 3, '每步几乎清空', '只看当前 token：接近局部特征'], [20, 8, '在标点处衰减', '按句子记忆：句内保持，标点处清掉一部分'], [12, 12, '几乎不衰减', '长期记忆：保留到段末，适合全文层面的信息']].map(([L, h, d, f]) => ({ L, h, d, f, a: AD.alpha[AD.gdn_layers.indexOf(L)][h] }));
AH.forEach((r) => { let k = 1; r.keep = AT_TOK.map((_, t) => (t <= KEY_TOK ? (t === KEY_TOK ? 1 : null) : (k *= r.a[t]))); });
const AHC = [C.amber, C.blue, C.tealD];
const S_A = { heat: 0.8, curve: [1, 2, 3].map((j) => bt(I.alpha, j)), pos: bt(I.alpha, 4), kda: bt(I.alpha, 5) };
ev(I.alpha, 0.5, 'whoosh'); AH.forEach((_, i) => ev(I.alpha, S_A.heat + 0.8 + i * 0.8, 'land')); S_A.curve.forEach((t) => { ev(I.alpha, t, 'click'); ev(I.alpha, t + 2.2, 'pop'); });
// 列宽按 token 文字宽度分配（标点给最小宽度），热图、标点竖带与保留曲线共用同一组列边界
const TOK_FS = 23;
const TOK_COLS = (() => {
  const X0 = 330, W = 1824 - X0, ws = AT_TOK.map((t) => (/[，。]/.test(t) ? 58 : tw(t, TOK_FS, 600) + 22)), k = W / ws.reduce((a, b) => a + b, 0);
  let x = X0; return ws.map((w) => { const c = [x, w * k]; x += w * k; return c; });
})();
const cx0 = (i) => TOK_COLS[i][0], cw = (i) => TOK_COLS[i][1];
function sAlpha(u) {
  header(I.alpha, u);
  const X0 = 330, a0 = fade(u, 0.3), punct = (t) => /[，。]/.test(t);
  // 标点处的竖带，贯穿上下两部分
  AT_TOK.forEach((t, i) => { if (punct(t)) rr(cx0(i), 296, cw(i), 700, '#e9e2d2', fade(u, S_A.curve[1]) * 0.9); });
  // 分词：每个 token 一个浅色块，块间留缝
  AT_TOK.forEach((t, i) => {
    rr(cx0(i) + 3, 300, cw(i) - 6, 40, i === KEY_TOK ? C.yel : '#e2dccd', a0 * (i === KEY_TOK ? 0.7 : 1));
    txt(t, cx0(i) + cw(i) / 2, 329, { size: TOK_FS, weight: i === KEY_TOK ? 800 : 600, align: 'center', color: C.ink, alpha: a0 });
  });
  // 热图：每行一个 head，格子颜色为 α（0 为底色，1 为深 teal）
  const RY = 356, RH = 54;
  AH.forEach((r, n) => {
    const y = RY + n * (RH + 18), a = fade(u, S_A.heat + 0.8 + n * 0.8);
    txt(`第 ${r.L + 1} 层 · head ${r.h + 1}`, X0 - 20, y + 24, { size: 22, weight: 700, align: 'right', alpha: a, color: AHC[n] });
    note(r.d, X0 - 20, y + 50, { align: 'right', alpha: a, size: 19 });
    r.a.forEach((v, i) => rr(cx0(i) + 3, y, cw(i) - 6, RH, mixc(C.paper, C.tealD, clamp(v)), a));
    if (n === 1) r.a.forEach((v, i) => { if (punct(AT_TOK[i]) && v < 0.97) txt(v.toFixed(2), cx0(i) + cw(i) / 2, y + RH / 2 + 8, { size: 17, weight: 800, align: 'center', color: '#fff', alpha: a }); });
    if (n !== 1) txt(`α ≈ ${(r.a.reduce((x, y2) => x + y2, 0) / r.a.length).toFixed(2)}`, cx0(4) + cw(4) / 2, y + RH / 2 + 8, { size: 20, weight: 800, align: 'center', color: n ? '#fff' : C.ink, alpha: a });
  });
  // 图例
  const la = fade(u, S_A.heat + 3);
  note('α：0', 96, 584, { alpha: la, size: 19 }); for (let i = 0; i < 10; i++) rr(150 + i * 12, 568, 12, 20, mixc(C.paper, C.tealD, i / 9), la); note('1', 278, 584, { alpha: la, size: 19 });
  // 保留曲线
  const ka = fade(u, S_A.curve[0]), PY0 = 975, PH = 270, py = (v) => PY0 - v * PH;
  lab('“钥匙”写入后，在各 head 中还剩的比例', 96, 640, { alpha: ka, size: 24 });
  note('只计衰减 α 的连乘；曲线旁的功能为直观理解', 1824, 640, { align: 'right', alpha: ka });
  line(X0, PY0, 1824, PY0, C.ink, 2, ka); line(X0, py(1), 1824, py(1), C.lg, 1.5, ka, [6, 6]);
  note('1', X0 - 12, py(1) + 7, { align: 'right', alpha: ka }); note('0', X0 - 12, PY0 + 7, { align: 'right', alpha: ka });
  const last = AT_TOK.length - 1, LBL = [[4, PY0 - 18], [9, null], [8, null]];
  AH.forEach((r, n) => {
    const st = S_A.curve[n]; if (u < st) return;
    const prog = clamp((u - st - 0.2) / 2.0) * (last - KEY_TOK);
    g.globalAlpha = 1; g.strokeStyle = AHC[n]; g.lineWidth = 5; g.beginPath();
    for (let t = KEY_TOK; t <= KEY_TOK + Math.floor(prog) + 1 && t <= last; t++) {
      const x0 = cx0(t), y = py(r.keep[t]) + (n - 1) * 3;
      if (t === KEY_TOK) g.moveTo(x0 + cw(t) / 2, y); else { g.lineTo(x0, py(r.keep[t - 1]) + (n - 1) * 3); g.lineTo(x0, y); }
      g.lineTo(x0 + cw(t), y);
    }
    g.stroke();
    const fa = fade(u, st + 2.2), end = r.keep[last];
    txt(end.toFixed(2), 1824 + 8, py(end) + 8 - (n === 2 ? 12 : 0), { size: 22, weight: 800, color: AHC[n], alpha: fa });
    // 直观理解：标在曲线旁
    const [ti, yy] = LBL[n], ly = yy ?? (n === 1 ? py(r.keep[ti]) + 84 : py(1) - 16);
    txt(r.f, cx0(ti), ly, { size: 24, weight: 800, color: AHC[n], alpha: fa });
  });
  // 正在讲的 head：热图那一行加框
  const cur = S_A.curve.filter((t) => u >= t).length - 1;
  if (cur >= 0 && u < S_A.pos) sr(X0 - 2, 356 + cur * 72 - 4, 1824 - X0 + 4, 62, AHC[cur], 3);
  // 位置信息：盖在热图上的两列对照
  const pa = fade(u, S_A.pos, 0.6);
  if (pa > 0) {
    rr(80, 348, 1760, 222, C.paper, pa);
    const box = (x, w, col, title, lines) => { sr(x, 362, w, 192, col, 3, pa); txt(title, x + 24, 404, { size: 26, weight: 800, color: col, alpha: pa }); lines.forEach((l, i) => txt(l, x + 24, 446 + i * 36, { size: 23, weight: 600, color: C.ink2, alpha: pa })); };
    box(96, 820, C.purple, 'full attention 层：RoPE', ['按位置把 q、k 旋转一个角度', '点积 qᵀk 只取决于两个 token 的相对距离', 'Qwen3.5 只对 1/4 的 head 维度做 RoPE']);
    box(944, 880, C.tealD, 'GDN 层：不用 RoPE', ['越早写入的内容乘过的 α 越多，剩得越少', 'short convolution 提供相邻 3 个 token 的顺序', '下方的保留曲线就是位置信息']);
    arrow(1260, 556, 1260, 745, C.tealD, pa, 3);
  }
  [['KDA：同一个 head 内，', 780], ['每个 key channel', 810], ['各有一个 α', 840]].forEach(([t, y]) => txt(t, 96, y, { size: 22, weight: 700, color: C.tealD, alpha: fade(u, S_A.kda) }));
}

// ================= 9 训练怎么并行 =================
const NCH = 64, ZA = 30;
ev(I.chunk, 0.5, 'whoosh'); evRun(I.chunk, 1.0, 4.0, 0.06, 'tick'); ev(I.chunk, bt(I.chunk, 1), 'land'); ev(I.chunk, bt(I.chunk, 1) + 1.2, 'pop'); ev(I.chunk, bt(I.chunk, 1) + 2.0, 'pop'); ev(I.chunk, bt(I.chunk, 2), 'click');
function s9(u) {
  header(I.chunk, u);
  const X0 = 96, SW = 1280, P = SW / NCH, Y0 = 320, CH = 30, a0 = fade(u, 0.4), b1 = bt(I.chunk, 1), cut = p(u, b1, b1 + 0.8);
  // 一条 4K token：先是 4096 个格，再切成 64 块
  txt('4096 个 token', X0, Y0 - 14, { size: 24, weight: 700, alpha: a0 });
  rr(X0, Y0, SW * p(u, 0.6, 1.6), CH, C.lg, a0 * (1 - cut));
  const walk = lin(u, 1.0, 4.0);
  if (cut < 1) { rr(X0 + SW * walk - 6, Y0 - 6, 12, CH + 12, C.teal, a0 * (1 - cut) * (u > 1 ? 1 : 0)); }
  for (let i = 0; i < NCH; i++) rr(X0 + i * P, Y0, P - 3, CH, i === ZA || i === ZA + 1 ? C.blue : C.lg, cut);
  note(`逐个 token 传 state：串行 4096 步`, X0 + SW, Y0 - 14, { align: 'right', alpha: a0 * (1 - cut) });
  note('每 64 个一块，共 64 块', X0 + SW, Y0 - 14, { align: 'right', alpha: cut });
  // 放大两块
  const za = fade(u, b1 + 0.8), YS = 470, SS = 120, CS = 190;
  const xs = [X0 + 40, X0 + 460, X0 + 880];
  line(X0 + ZA * P, Y0 + CH + 4, xs[0], YS - 20, C.gray, 1.5, za, [5, 5]); line(X0 + (ZA + 2) * P, Y0 + CH + 4, xs[2] + SS, YS - 20, C.gray, 1.5, za, [5, 5]);
  xs.forEach((x, i) => { heat('sum40', SUM4, x, YS, SS, SS, { scale: S_SCALE, alpha: za }); mtxt(['S_{[n]}', 'S_{[n+1]}', 'S_{[n+2]}'][i], x + SS / 2, YS + SS + 36, 28, { align: 'center', color: C.tealD, alpha: za }); });
  for (let i = 0; i < 2; i++) {
    const cx = (xs[i] + SS + xs[i + 1]) / 2 - CS / 2, cy = YS - 35, a = fade(u, b1 + 1.2 + i * 0.8);
    rr(cx, cy, CS, CS, GHOST, a);
    g.globalAlpha = a; g.fillStyle = C.blue; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx, cy + CS); g.lineTo(cx + CS, cy + CS); g.closePath(); g.fill(); g.globalAlpha = 1;
    txt('64 × 64', cx + CS - 8, cy + 30, { size: 20, weight: 500, color: C.gray, align: 'right', alpha: a });
    txt(`chunk ${i ? 'n+1' : 'n'}`, cx + CS * 0.36, cy + CS - 16, { size: 22, weight: 700, color: '#fff', align: 'center', alpha: a });
    arrow(xs[i] + SS + 4, YS + SS / 2, cx - 6, YS + SS / 2, C.tealD, a, 3);
    arrow(cx + CS + 6, YS + SS / 2, xs[i + 1] - 4, YS + SS / 2, C.tealD, a, 3);
  }
  note('块内：下三角矩阵乘，在 Tensor Core 上并行计算', xs[0] + SS + 10, YS + CS + 30, { alpha: fade(u, b1 + 2.4) });
  // 右侧：串行步数
  const ma = fade(u, b1 + 1.0);
  txt('串行的步数', MX, 360, { size: 26, weight: 700, alpha: a0 });
  meter(420, '逐个 token', [[4096, C.lg]], '4096', { alpha: a0, full: 4096 });
  meter(540, '分块', [[64, C.teal]], '64', { alpha: ma, full: 4096, color: C.tealD });
  // vLLM 的两处调用
  code(96, 740, 1728, [
    '{g:# prefill：整段按 chunk 计算}',
    'self.{t:chunk_gated_delta_rule}(q=query_non_spec, k=key_non_spec, v=value_non_spec, g=g_non_spec, beta=beta_non_spec,',
    '    initial_state=initial_state, output_final_state=True, ...)',
    '{g:# decode：每个 token 一次递推}',
    '{t:fused_recurrent_gated_delta_rule_packed_decode}(mixed_qkv=mixed_qkv_non_spec, ..., initial_state=ssm_state, ...)',
  ], { size: 21, alpha: fade(u, bt(I.chunk, 2)), title: 'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py' });
}

// ================= 10 精确检索 =================
// 层排布：线性 attention 层是一个小方块（固定大小的 state），full attention 层是一根高条（随上下文增长的 KV）
const STR = [
  { name: 'Qwen3.8-27B', sub: '48 层 GDN，16 层 full attention', n: 64, full: (i) => i % 4 === 3 },
  { name: 'Kimi K3', sub: '69 层 KDA，24 层 MLA', n: 93, full: (i) => i % 4 === 3 || i === 92 },
  { name: 'MiniMax-Text-01', sub: '70 层 lightning attention，10 层 full attention', n: 80, full: (i) => i % 8 === 7 },
];
function strip(r, x, y, w, hFull, hLin, alpha = 1, grow = 1) {
  const P = w / r.n, cw = Math.max(3, P * 0.72);
  for (let i = 0; i < r.n; i++) {
    const a = alpha * clamp((grow * (r.n + 3) - i) / 3);
    if (r.full(i)) rr(x + i * P, y - hFull, cw, hFull, C.purple, a); else rr(x + i * P, y - hLin, cw, hLin, C.teal, a);
  }
}
ev(I.hybrid, 0.5, 'whoosh'); evRun(I.hybrid, 0.8, 2.6, 0.04, 'tick'); ev(I.hybrid, bt(I.hybrid, 1), 'land'); ev(I.hybrid, bt(I.hybrid, 1) + 0.8, 'land'); ev(I.hybrid, bt(I.hybrid, 2), 'click');
function s10(u) {
  header(I.hybrid, u);
  const a0 = fade(u, 0.4), X0 = 96, SW = 1240;
  STR.forEach((r, n) => {
    const st = n ? bt(I.hybrid, 1) + (n - 1) * 0.8 : 0.5, a = fade(u, st), y = n ? 720 + (n - 1) * 170 : 500, hF = n ? 80 : 150, hL = n ? 10 : 16;
    txt(r.name, X0, y - hF - 22, { size: 26, weight: 700, alpha: a }); note(r.sub, X0 + tw(r.name, 26, 700) + 16, y - hF - 22, { alpha: a });
    strip(r, X0, y, SW, hF, hL, a, p(u, st, st + 1.6));
  });
  const la = fade(u, 1.6);
  rr(X0, 534, 22, 22, C.purple, la); txt('full attention 层：KV 随上下文增长，负责精确检索', X0 + 34, 553, { size: 22, weight: 700, color: C.purple, alpha: la });
  rr(X0 + 640, 540, 14, 14, C.teal, la); txt('线性 attention 层：固定大小的 state', X0 + 666, 553, { size: 22, weight: 700, color: C.tealD, alpha: la });
  const a2 = fade(u, bt(I.hybrid, 2));
  txt('九项评测平均（25B-A3B）', MX, 360, { size: 24, weight: 700, alpha: a2 });
  meter(420, '全部 full attention', [[49.87, C.purple]], '49.87', { alpha: a2, full: 60 });
  meter(540, 'GDN 3:1 混合', [[53.81 * p(u, bt(I.hybrid, 2) + 0.3, bt(I.hybrid, 2) + 1.3), C.teal]], '53.81', { alpha: a2, full: 60, color: C.tealD });
}

// ================= 显存与推理系统 =================
// 上：Qwen3.8-27B 每个请求 64 层合计的缓存，与一个 GDN 层在短上下文下 state 与 KV 的比较；下：Kimi K3 的 state 副本
const KV_TOK = 4 * KiB, ST = 3 * MiB;
const SEQ = 6144, CKPT = 512, SHARE = 2800, HIT = 2560;
ev(I.memory, 0.5, 'whoosh'); ev(I.memory, 1.0, 'land'); ev(I.memory, 1.8, 'land'); ev(I.memory, bt(I.memory, 1), 'pop'); evRun(I.memory, bt(I.memory, 2) + 0.4, bt(I.memory, 2) + 2.8, 0.2, 'tick'); ev(I.memory, bt(I.memory, 2) + 3.6, 'click');
function sMemory(u) {
  header(I.memory, u);
  // 左上：总量
  const a1 = fade(u, 0.5), MXL = 96, MWL = 760;
  txt('每个请求，64 层合计（256K 上下文）', MXL, 320, { size: 26, weight: 700, alpha: a1 });
  meter(380, '3:1 混合', [[16 * p(u, 1.0, 1.8), C.purple], [48 * 3 / 1024, C.teal]], '16.1 GiB', { alpha: a1, full: 64, color: C.tealD, x: MXL, w: MWL, note: '16 层 KV 各 1 GiB，48 层 state 各 3 MiB' });
  meter(500, '全部 full attention', [[64 * p(u, 1.8, 2.6), C.purple]], '64 GiB', { alpha: fade(u, 1.8), full: 64, x: MXL, w: MWL });
  // 右上：一个 GDN 层，上下文 0 到 2K
  const b1 = bt(I.memory, 1), a2 = fade(u, b1), ZX = 1000, ZY = 320, ZW = 824, ZH = 250, xs = (l) => ZX + 20 + (l / 2048) * (ZW - 40), ys = (b) => ZY + ZH - 30 - (b / (8 * MiB)) * (ZH - 60);
  rr(ZX, ZY, ZW, ZH, '#fbf9f4', a2); sr(ZX, ZY, ZW, ZH, C.ink, 2, a2);
  txt('一个 GDN 层：上下文 0 到 2K', ZX, ZY - 14, { size: 22, weight: 700, alpha: a2 });
  line(xs(0), ys(ST), xs(2048), ys(ST), C.teal, 5, a2);
  const lp = p(u, b1 + 0.3, b1 + 2.3);
  line(xs(0), ys(0), xs(2048 * lp), ys(2048 * lp * KV_TOK), C.purple, 3, a2, [8, 6]);
  const ca = fade(u, b1 + 1.4);
  g.globalAlpha = ca; g.fillStyle = C.ink; g.beginPath(); g.arc(xs(768), ys(ST), 7, 0, 7); g.fill(); g.globalAlpha = 1;
  txt('768 个 token', xs(768), ys(ST) + 38, { size: 22, weight: 700, align: 'center', alpha: ca });
  txt('state 3 MiB', xs(2048), ys(ST) + 30, { size: 20, weight: 700, align: 'right', color: C.tealD, alpha: a2 });
  txt('KV（若用 full attention）', xs(1300) - 120, ys(1300 * KV_TOK) - 30, { size: 20, weight: 700, color: C.purple, alpha: ca });
  note('Kimi K3：约 5.5K 个 token', ZX, ZY + ZH + 34, { alpha: ca });
  // 下：prefix caching 的 state 副本
  const b2 = bt(I.memory, 2), a3 = fade(u, b2), X0 = 300, SW = 1524, xq = (t) => X0 + SW * t / SEQ, Y = 720;
  line(96, Y - 70, 1824, Y - 70, C.lg, 2, a3);
  lab('复用前缀（Kimi K3）', 96, Y - 24, { color: C.tealD, alpha: a3 });
  txt('请求 A', X0 - 20, Y + 64, { size: 22, weight: 700, align: 'right', alpha: a3 });
  rr(X0, Y + 36, SW * p(u, b2, b2 + 0.8), 36, C.lg, a3);
  for (let c = CKPT; c <= SEQ; c += CKPT) { const a = fade(u, b2 + 0.4 + (c / SEQ) * 2.4, 0.25); rr(xq(c) - 11, Y - 6, 22, 22, C.teal, a); line(xq(c), Y + 16, xq(c), Y + 36, C.tealD, 2, a); }
  note('每 512 个 token 保存一份 state 副本', X0 + SW, Y - 24, { align: 'right', alpha: fade(u, b2 + 0.8) });
  const ba = fade(u, b2 + 3.0);
  txt('请求 B', X0 - 20, Y + 144, { size: 22, weight: 700, align: 'right', alpha: ba });
  rr(X0, Y + 116, xq(HIT) - X0, 36, C.teal, ba * 0.45); rr(xq(HIT), Y + 116, xq(SEQ) - xq(HIT), 36, C.blue, ba * 0.5);
  line(xq(SHARE), Y + 104, xq(SHARE), Y + 164, C.ink, 2, ba, [5, 4]);
  txt('前 2800 个 token 相同', xq(SHARE) + 12, Y + 196, { size: 22, weight: 700, alpha: ba });
  const ha = fade(u, b2 + 3.6);
  sr(xq(HIT) - 13, Y - 8, 26, 26, C.ink, 3, ha);
  txt('从 2560 处继续计算', xq(HIT) - 12, Y + 196, { size: 22, weight: 700, align: 'right', color: C.tealD, alpha: ha });
}


// ================= 13 两条路线 =================
const MM = [
  { name: 'MiniMax-Text-01', when: '2025 年 1 月', sub: '7 层 lightning attention 配 1 层 full attention' },
  { name: 'MiniMax-M2', when: '2025 年 10 月', sub: '全部 full attention' },
  { name: 'MiniMax-M3', when: '', sub: 'GQA 加按 block 选择的稀疏 attention' },
];
ev(I.routes, 0.5, 'whoosh'); ev(I.routes, 1.0, 'land'); ev(I.routes, 2.2, 'land'); ev(I.routes, 3.4, 'land'); ev(I.routes, bt(I.routes, 1), 'pop'); ev(I.routes, bt(I.routes, 2), 'click');
function s13(u) {
  header(I.routes, u);
  const X0 = 96, SW = 1000, H = 44;
  MM.forEach((m, n) => {
    const y = 360 + n * 150, a = fade(u, 1.0 + n * 1.2);
    txt(m.name, X0, y, { size: 26, weight: 700, alpha: a }); note(`${m.when}${m.when ? '，' : ''}${m.sub}`, X0 + tw(m.name, 26, 700) + 16, y, { alpha: a });
    if (n === 0) { const P = SW / 80; for (let i = 0; i < 80; i++) rr(X0 + i * P, y + 18, P * 0.72, H, i % 8 === 7 ? C.purple : C.teal, a); }
    else if (n === 1) rr(X0, y + 18, SW, H, C.purple, a);
    else { rr(X0, y + 18, SW, H, C.blue, a); }
    if (n) arrow(X0 + SW / 2, y - 64, X0 + SW / 2, y - 34, C.gray, a, 3);
  });
  const a1 = fade(u, bt(I.routes, 1));
  rr(X0, 790, SW, 140, '#fbf9f4', a1); sr(X0, 790, SW, 140, C.lg, 2, a1);
  txt('MiniMax 的理由', X0 + 24, 832, { size: 24, weight: 700, color: C.red, alpha: a1 });
  txt('规模扩大后，混合模型在多跳推理上出现明显缺陷', X0 + 24, 872, { size: 24, weight: 500, color: C.ink2, alpha: a1 });
  txt('state 对低精度敏感，prefix cache 与投机解码缺少成熟做法', X0 + 24, 908, { size: 24, weight: 500, color: C.ink2, alpha: a1 });
  // 右侧：各家的选择
  const a2 = fade(u, bt(I.routes, 2)), RX = 1240;
  const col = (x, title, c, items, a) => { rr(x, 330, 260, 6, c, a); txt(title, x, 376, { size: 26, weight: 700, color: c, alpha: a }); items.forEach((s, i) => txt(s, x, 430 + i * 46, { size: 26, weight: 600, color: C.ink2, alpha: a })); };
  col(RX, '线性 attention 混合', C.tealD, ['Kimi K3', 'Qwen3.8'], a2);
  col(RX + 310, '稀疏 attention', C.blue, ['DeepSeek V3.2 / V4', 'GLM-5', 'MiniMax-M3'], fade(u, bt(I.routes, 2) + 0.5));
  note('稀疏：逐 token 保存 KV，每一步只读取选出的部分', RX, 640, { alpha: fade(u, bt(I.routes, 2) + 0.9), size: 20 });
}

// ---------- 字幕 ----------
// 配音稿逐句显示在画面底部：按标点切成不超过 26 字的短句；时长按字数估算（配音合成后改用实测时长），不越过下一句的起点
function cues(text, lim = 26) {
  const out = [];
  for (const p of (text.match(/[^，。：；？！]+[，。：；？！]?/g) || []).map((x) => x.trim()).filter(Boolean)) {
    const t = p.replace(/[，。；]$/, '');
    if (out.length && out[out.length - 1].length + t.length + 1 <= lim && !/[？！]$/.test(out[out.length - 1])) out[out.length - 1] += (out[out.length - 1].endsWith('：') ? '' : '，') + t;
    else out.push(t);
  }
  return out;
}
const est = (t) => (t.match(/[一-鿿]/g) || []).length / 4.6 + (t.match(/[A-Za-z0-9][A-Za-z0-9.,%\-]*/g) || []).length * 0.35;
const SUBS = SCRIPT.map((sc) => sc.beats.flatMap(([t0, , text], j) => {
  if (!text) return [];
  const cs = cues(text), end = j + 1 < sc.beats.length ? sc.beats[j + 1][0] - 0.1 : sc.dur - 0.1, total = cs.reduce((a, c) => a + est(c), 0);
  const scale = Math.min(1, (end - t0) / Math.max(0.1, total)); let t = t0;
  return cs.map((c) => { const d = est(c) * scale; const r = [t, t + d, c]; t += d; return r; });
}));
function subtitle(k, u) {
  const c = SUBS[k].find(([a, b]) => u >= a && u < b); if (!c) return;
  const w = tw(c[2], 34, 600);
  rr(960 - w / 2 - 20, 1012, w + 40, 52, C.paper, 0.9);
  txt(c[2], 960, 1050, { size: 34, weight: 600, align: 'center', color: C.ink });
}

const SCENES = { open: s1, read: sRead, softmax: s2, scale: s3, superpose: sSuperpose, realkeys: sRealKeys, delta: s6, niah: s7, alpha: sAlpha, chunk: s9, hybrid: s10, memory: sMemory, routes: s13 };
function draw(t) {
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  rr(0, 0, W, H, C.paper);
  SCENES[SCRIPT[k].id](t - AT[k]);
  subtitle(k, t - AT[k]);
}
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
