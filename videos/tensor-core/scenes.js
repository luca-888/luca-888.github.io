// Tensor Core 影片的各幕。每幕一个 draw(u)，u 是幕内时间（秒）；节拍起点取自 script.js，音效事件登记进 EV。
// 贯穿全片的图形：乘加方块。一个小方块是一次乘加（a × b + c），矩阵乘法是一堆方块；全片只看一条指令、一个时钟取走多少个方块。
// 方块数量都按真实个数画。颜色：A blue，B yellow，结果 purple，有用的计算 teal，指令开销 red。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const EV = SCRIPT.map(() => []);                      // 每幕的音效：[幕内时间, 种类]
const ev = (k, u, kind) => EV[k].push([u, kind]);
const evRun = (k, a, b, step, kind) => { for (let u = a; u < b - 1e-6; u += step) ev(k, u, kind); };
const NAME = 'Tensor Core';
const BL = { top: '#7d97ea', left: C.blue, right: '#1a3ca6' }, YE = { top: '#f8d867', left: C.yel, right: '#c99a0c' }, PU = { top: '#a77ef8', left: C.purple, right: '#5b25c9' };
const RED = { top: '#f07a76', left: C.red, right: '#b3221e' };

// 页眉：左上角是开场缩上来的片名，下面是这一幕的标签与标题，右上角是出处
function header(k, u) {
  const s = SCRIPT[k], a = po(u, 0, 0.5);
  txt(NAME, 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 76, { size: 26, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 226 + (1 - a) * 16, 56, 700, a);
}
// 两条进度条的赛跑：上面一次走完，下面循环 n 遍（按峰值的真实比例）
function race(x, y, w, u0, u1, u, n, alpha = 1, labels = true) {
  const k = lin(u, u0, u1), laps = Math.min(n, Math.floor(k * n)), frac = k >= 1 ? 1 : (k * n) % 1;
  const rows = [['CUDA core', 'FP32 · 每秒 75 万亿次运算', C.red, k, k >= 1 ? '1 遍' : ''],
    ['Tensor Core', 'BF16 · 每秒 2250 万亿次运算', C.teal, frac, `${laps} 遍`]];
  rows.forEach(([name, rate, col, f, cnt], r) => {
    const yy = y + r * 150;
    if (labels) { txt(name, x, yy - 14, { size: 36, weight: 800, color: r ? C.tealD : C.ink, alpha }); txt(rate, x + tw(name, 36, 800) + 20, yy - 14, { size: 26, weight: 500, color: C.gray, alpha }); }
    rr(x, yy, w, 56, C.paper2, alpha); rr(x, yy, w * f, 56, col, alpha);
    txt(cnt, x + w + 24, yy + 42, { size: 36, weight: 800, color: r ? C.tealD : C.ink, alpha: alpha * (cnt ? 1 : 0), font: ZH });
  });
}

// ================= 1 开场 =================
// HGX B200（8 卡合计，脚注：除 FP4 外为 sparse）换算到每卡 dense：FP32 75 TFLOPS，BF16 2250 TFLOPS，比值 30
const S1 = { out: [4.6, 5.8], race: [bt(0, 2) + 0.2, bt(0, 2) + 6.0], x: bt(0, 3) + 0.1 };
ev(0, 0.25, 'land'); evRun(0, 2.4, 4.2, 0.12, 'tick'); ev(0, S1.out[0], 'whoosh'); ev(0, bt(0, 1) + 1.2, 'pop'); ev(0, bt(0, 1) + 1.6, 'pop');
for (let i = 1; i <= 30; i++) ev(0, S1.race[0] + (S1.race[1] - S1.race[0]) * i / 30, 'tick');
ev(0, S1.race[1], 'land'); ev(0, S1.x, 'pop');
function s1(u) {
  const out = p(u, ...S1.out), k = po(u, 0.15, 1.0);
  const size = mix(168, 28, out);
  txt(NAME, mix(96, 96, out), mix(420, 76, out) + (1 - k) * 40, { size, weight: mix(800, 700, out), color: out < 0.5 ? C.ink : C.ink2, alpha: k });
  txt('同一块芯片上，矩阵乘法快 30 倍的原因', 100, 520, { size: 52, weight: 700, color: C.tealD, alpha: po(u, 1.2, 1.8) * (1 - out) });
  // 芯片：先在右侧大图逐层画出，148 个 SM 依次点亮，一格变绿；随后缩到右下
  const m = p(u, ...S1.out), CX = mix(1160, 1330, m), CY = mix(170, 330, m), CW = mix(660, 494, m);
  const lit = Math.floor(148 * lin(u, 2.4, 4.2)), litSet = new Set(B200_ON.slice(0, lit));
  chip(CX, CY, CW, { build: lin(u, 0.6, 2.6), off: B200_OFF, hot: po(u, 4.0, 4.4), paint: (n) => (litSet.has(n) ? null : ['#4a483f', 1]) });
  txt('B200 · 2 个 die · 启用 148 个 SM', CX + CW / 2, CY + CW * 0.92 + 44, { size: 26, color: C.gray, align: 'center', alpha: po(u, 3.6, 4.2) });
  if (m > 0.02) {
    const [hx, hy] = chipHotXY(CX, CY, CW);
    txt('一个 SM：128 个 CUDA core + 4 个 Tensor Core', CX + CW / 2, CY - 24, { size: 26, weight: 700, color: C.tealD, align: 'center', alpha: po(u, bt(0, 1) + 0.4, bt(0, 1) + 1.0) });
    line(hx, hy, hx, CY - 12, C.tealD, 2, po(u, bt(0, 1) + 0.4, bt(0, 1) + 1.0));
  }
  // 同一道矩阵乘法交给两种计算单元：两条进度按峰值的真实比例推进
  const a = po(u, bt(0, 1) + 1.0, bt(0, 1) + 1.6);
  txt('同一个矩阵乘法', 96, 340, { size: 30, weight: 600, color: C.ink2, alpha: a });
  race(96, 450, 860, ...S1.race, u, 30, a);
  const x = po(u, S1.x, S1.x + 0.4);
  txt('30 倍', 96 + 860 + 24, 760, { size: 64, weight: 800, color: C.tealD, alpha: x });
  placeholder(96, 800, 1120, 150, '实测待补：B200 上的实际用时', ['4096 阶矩阵乘法，FP32（关闭 TF32）与 BF16 各跑一次，两条进度条改用实测用时', '需要：Modal B200 × 1，单次运行约 $1'], po(u, S1.x + 0.6, S1.x + 1.0));
}

// ================= 2 矩阵乘法 =================
// 4 × 4 的 A、B；跟 C 的第 2 行第 3 列（下标 1, 2）。方块堆的坐标：i = 列 q，j = 行 r，k = 求和下标
const M2 = { c: 64, AX: 150, AY: 600, BX: 450, BY: 300, CX: 450, CY: 600, R: 1, Q: 2 };
const SK = { ox: 1430, oy: 560, s: 62 };                              // 方块堆（后角在地面上的位置）
const S2 = { pair: [bt(1, 1) + 0.4, 1.3], fill: [bt(1, 2) + 0.4, bt(1, 2) + 3.4] };
ev(1, 0.5, 'pop'); ev(1, 0.9, 'pop'); ev(1, 1.3, 'pop');
for (let t = 0; t < 4; t++) ev(1, S2.pair[0] + t * S2.pair[1] + 0.6, 'land');
for (let e = 1; e < 16; e++) ev(1, S2.fill[0] + (S2.fill[1] - S2.fill[0]) * e / 15, 'tick');
const ORDER = Array.from({ length: 16 }, (_, e) => e).filter((e) => e !== M2.R * 4 + M2.Q);
// 第 e 个元素（行 r、列 q）的方块数，已落下的个数
function cubesOf(u, r, q) {
  if (r === M2.R && q === M2.Q) return Math.min(4, Math.max(0, Math.floor((u - S2.pair[0] - 0.6) / S2.pair[1]) + 1));
  const idx = ORDER.indexOf(r * 4 + q);
  return u >= S2.fill[0] + (S2.fill[1] - S2.fill[0]) * (idx + 1) / 15 ? 4 : 0;
}
function cubeStack(ox, oy, s, n, fn, alpha = 1) {        // 4 × 4 × 4 的格位，fn(r, q) 返回该列已有的方块数与配色
  for (let sum = 0; sum <= 2 * (n - 1); sum++) for (let q = 0; q < n; q++) {
    const r = sum - q; if (r < 0 || r >= n) continue;
    const [cnt, set] = fn(r, q); for (let k = 0; k < cnt; k++) box(ox, oy, s, q, r, k, 1, 1, 1, set, alpha, C.paper);
  }
}
function s2(u) {
  const { c, AX, AY, BX, BY, CX, CY, R, Q: QQ } = M2, aA = po(u, 0.5, 1.0), aB = po(u, 0.9, 1.4), aC = po(u, 1.3, 1.8);
  const pk = Math.floor((u - S2.pair[0]) / S2.pair[1]), inPair = u >= S2.pair[0] && pk < 4;
  cells(AX, AY, 4, 4, c, '#b9c6f0', aA, (r, q) => (r === R && u > S2.pair[0] - 0.4 ? (inPair && q === pk ? C.blue : '#8ea3e6') : null));
  cells(BX, BY, 4, 4, c, '#f6e3a0', aB, (r, q) => (q === QQ && u > S2.pair[0] - 0.4 ? (inPair && r === pk ? C.yel : '#f1cf62') : null));
  cells(CX, CY, 4, 4, c, '#d6c4fb', aC, (r, q) => (r === R && q === QQ && u > S2.pair[0] - 0.4 ? C.purple : (cubesOf(u, r, q) === 4 ? '#b49cf2' : null)));
  txt('A', AX - 40, AY + 140, { size: 48, weight: 800, color: C.blue, alpha: aA }); txt('B', BX + 270, BY + 140, { size: 48, weight: 800, color: C.yelD, alpha: aB });
  txt('C', CX + 270, CY + 140, { size: 48, weight: 800, color: C.purple, alpha: aC });
  // 求和式：当前这一对高亮
  const f = po(u, S2.pair[0] - 0.4, S2.pair[0]);
  if (f > 0) {
    let x = 96; const y = 940, sz = 28;
    txt('C[1][2] =', x, y, { size: sz, weight: 700, color: C.purple, font: MONO, alpha: f }); x += tw('C[1][2] =', sz, 700, MONO) + 14;
    for (let t = 0; t < 4; t++) {
      const on = inPair && t === pk, done = u >= S2.pair[0] + t * S2.pair[1] + 0.6, al = f * (done || on ? 1 : 0.3);
      const term = `A[1][${t}]·B[${t}][2]`; txt(term, x, y, { size: sz, weight: on ? 800 : 600, color: on ? C.ink : C.ink2, font: MONO, alpha: al }); x += tw(term, sz, 700, MONO) + 8;
      if (t < 3) { txt('+', x, y, { size: sz, color: C.gray, font: MONO, alpha: f }); x += tw('+', sz, 600, MONO) + 8; }
    }
  }
  // 方块堆：每一对相乘再累加落下一个方块
  const st = po(u, S2.pair[0], S2.pair[0] + 0.4);
  box(SK.ox, SK.oy, SK.s, 0, 0, 0, 4, 4, 4, SETS.ghost, st * 0.6);
  cubeStack(SK.ox, SK.oy, SK.s, 4, (r, q) => [cubesOf(u, r, q), r === R && q === QQ ? SETS.teal : SETS.teal], st);
  // 正在落下的方块
  if (inPair) {
    const fall = po(u - S2.pair[0] - pk * S2.pair[1], 0, 0.6);
    if (fall < 1) box(SK.ox, SK.oy - (1 - fall) * 160, SK.s, QQ, R, pk, 1, 1, 1, SETS.teal, fall, C.paper);
  }
  txt('一个方块 = 一次乘加', 1430, 870, { size: 34, weight: 800, color: C.tealD, align: 'center', alpha: po(u, S2.pair[0] + 1.0, S2.pair[0] + 1.5) * (1 - po(u, S2.fill[1], S2.fill[1] + 0.3)) });
  txt('a × b + c', 1430, 910, { size: 28, color: C.gray, align: 'center', font: MONO, alpha: po(u, S2.pair[0] + 1.0, S2.pair[0] + 1.5) * (1 - po(u, S2.fill[1], S2.fill[1] + 0.3)) });
  txt('4 × 4 矩阵乘法 = 64 次乘加', 1430, 870, { size: 34, weight: 800, align: 'center', alpha: po(u, S2.fill[1] + 0.2, S2.fill[1] + 0.6) });
}

// ================= 3 一次一个 =================
// FFMA：每条指令先走取指令、译码、读寄存器，再做一次乘加。能量（Dally，45 nm，FP16 乘加）：乘加 1.5 pJ，开销 30 pJ
const STAGES = [['取指令', C.red], ['译码', C.red], ['读寄存器', C.red], ['乘加', C.teal]];
const S3 = { slow: [0.9, 1.8], fast: [bt(2, 2) + 0.3, bt(2, 2) + 4.6], bars: bt(2, 1) + 0.2 };
// 第 n 条指令进入各阶段的时间：前 3 条慢放，其余 61 条在 fast 区间匀速通过
const instT = (n) => (n < 3 ? S3.slow[0] + n * S3.slow[1] : S3.fast[0] + (S3.fast[1] - S3.fast[0]) * (n - 3) / 61);
const instLen = (n) => (n < 3 ? S3.slow[1] : (S3.fast[1] - S3.fast[0]) / 61 * 4);
for (let n = 0; n < 3; n++) for (let s = 0; s < 4; s++) ev(2, instT(n) + instLen(n) * s / 4, s === 3 ? 'pop' : 'click');
for (let n = 3; n < 64; n++) ev(2, instT(n) + instLen(n), 'tick');
ev(2, S3.bars, 'land');
const doneAt = (u) => { let d = 0; for (let n = 0; n < 64; n++) if (u >= instT(n) + instLen(n)) d++; return d; };
function s3(u) {
  const a = po(u, 0.3, 0.8), X = 96, Y = 330, BW = 230, GAP = 30;
  STAGES.forEach(([name, col], s) => {
    const x = X + s * (BW + GAP);
    // 当前阶段亮起：看最近一条正在流水的指令
    let on = 0; for (let n = 0; n < 64; n++) { const t0 = instT(n) + instLen(n) * s / 4; if (u >= t0 && u < t0 + instLen(n) / 4 + 0.05) on = 1; }
    rr(x, Y, BW, 110, on ? col : C.paper2, a); txt(name, x + BW / 2, Y + 68, { size: 34, weight: 800, align: 'center', color: on ? '#fff' : (s === 3 ? C.tealD : C.red), alpha: a });
    if (s < 3) arrow(x + BW + 2, Y + 55, x + BW + GAP - 2, Y + 55, a, C.ink, a * 0.5, 3);
  });
  // 指令标签沿流水线移动（只画慢放的前 3 条）
  for (let n = 0; n < 3; n++) {
    const k = (u - instT(n)) / instLen(n); if (k < 0 || k > 1.1) continue;
    const x = X + clamp(k, 0, 0.999) * 4 * (BW + GAP) - 10;
    rr(x, Y - 62, 150, 46, C.ink, 1 - clamp((k - 1) * 10)); txt(`FFMA #${n + 1}`, x + 75, Y - 30, { size: 26, weight: 700, color: '#fff', align: 'center', font: MONO, alpha: 1 - clamp((k - 1) * 10) });
  }
  txt('FFMA：fused multiply-add，一条指令一次乘加', X, Y + 170, { size: 30, weight: 600, color: C.ink2, alpha: po(u, 0.8, 1.3) });
  // 能量条：乘加 1.5，开销 30，每 pJ 24 px
  const b = po(u, S3.bars, S3.bars + 0.5), PX = 24, EY = 610;
  txt('每条指令的能耗', X, EY - 20, { size: 30, weight: 700, alpha: b });
  rr(X + 200, EY + 10, 1.5 * PX, 52, C.teal, b); txt('乘加 1.5 pJ', X + 200 + 1.5 * PX + 16, EY + 48, { size: 28, weight: 700, color: C.tealD, alpha: b });
  rr(X + 200, EY + 80, 30 * PX * lin(u, S3.bars + 0.4, S3.bars + 1.4), 52, C.red, b);
  txt('取指令、译码、读寄存器 30 pJ', X + 200 + 16, EY + 116, { size: 28, weight: 700, color: '#fff', alpha: po(u, S3.bars + 1.2, S3.bars + 1.6) });
  txt('乘加', X, EY + 48, { size: 26, color: C.gray, alpha: b }); txt('开销', X, EY + 118, { size: 26, color: C.gray, alpha: b });
  // 累计：已付开销
  const d = doneAt(u), f = po(u, S3.fast[0], S3.fast[0] + 0.3);
  txt(`累计开销：${d} × 30 pJ = ${d * 30} pJ`, X, EY + 220, { size: 32, weight: 800, color: C.red, alpha: f, font: ZH });
  txt(`乘加本身：${d} × 1.5 pJ = ${(d * 1.5).toFixed(1)} pJ`, X, EY + 268, { size: 28, weight: 600, color: C.tealD, alpha: f, font: ZH });
  // 方块堆：64 个格位，每完成一条指令变一个 teal
  box(SK.ox, SK.oy, SK.s, 0, 0, 0, 4, 4, 4, SETS.ghost, a * 0.6);
  let n = 0; cubeStack(SK.ox, SK.oy, SK.s, 4, (r, q) => { const c = clamp(d - (r * 4 + q) * 4, 0, 4); n += c; return [c, SETS.teal]; }, a);
  txt(`已完成 ${d} / 64`, 1430, 960, { size: 34, weight: 800, align: 'center', alpha: a, font: ZH });
}

// ================= 4 一次一整块 =================
// Volta 的 Tensor Core 每个时钟做一次 4 × 4 × 4（64 次乘加）。开销与乘加之比（Dally）：HFMA 2000%，HMMA 22%
const S4 = { go: 0.9, take: 2.0, name: bt(3, 1) + 0.2, bars: bt(3, 2) + 0.2 };
ev(3, S4.go, 'click'); ev(3, S4.go + 0.3, 'click'); ev(3, S4.go + 0.6, 'click'); ev(3, S4.take, 'land'); ev(3, S4.name, 'pop'); ev(3, S4.bars, 'land'); ev(3, S4.bars + 1.2, 'land');
function s4(u) {
  const a = po(u, 0.2, 0.6), X = 96, Y = 330, BW = 230, GAP = 30;
  const stg = [['取指令', C.red], ['译码', C.red], ['读寄存器', C.red], ['64 次乘加', C.teal]];
  stg.forEach(([name, col], s) => {
    const x = X + s * (BW + GAP), t0 = S4.go + s * 0.3, on = u >= t0 && u < t0 + 0.3 + (s === 3 ? 1.2 : 0);
    rr(x, Y, BW, 110, on ? col : C.paper2, a); txt(name, x + BW / 2, Y + 68, { size: 34, weight: 800, align: 'center', color: on ? '#fff' : (s === 3 ? C.tealD : C.red), alpha: a });
    if (s < 3) arrow(x + BW + 2, Y + 55, x + BW + GAP - 2, Y + 55, a, C.ink, a * 0.5, 3);
  });
  const k = (u - S4.go) / 1.2;
  if (k > 0 && k < 1.1) { const x = X + clamp(k, 0, 0.999) * 4 * (BW + GAP) - 10; rr(x, Y - 62, 150, 46, C.ink, 1 - clamp((k - 1) * 10)); txt('HMMA', x + 75, Y - 30, { size: 26, weight: 700, color: '#fff', align: 'center', font: MONO, alpha: 1 - clamp((k - 1) * 10) }); }
  // 整块一次取走
  const t = po(u, S4.take, S4.take + 0.5);
  box(SK.ox, SK.oy, SK.s, 0, 0, 0, 4, 4, 4, SETS.ghost, a * 0.6 * (1 - t));
  if (t > 0) { const drop = (1 - t) * 120; cubeStack(SK.ox, SK.oy - drop, SK.s, 4, () => [4, SETS.teal], t); }
  txt('1 条指令 · 1 个时钟 · 64 次乘加', 1430, 960, { size: 32, weight: 800, align: 'center', color: C.tealD, alpha: po(u, S4.take + 0.4, S4.take + 0.8) });
  // 名字出现
  const nm = po(u, S4.name, S4.name + 0.5);
  txt('Tensor Core', X, 560, { size: 80, weight: 800, color: C.tealD, alpha: nm });
  txt('2017 · Volta 架构首次出现 · 每个 Tensor Core 每个时钟 64 次乘加', X, 612, { size: 28, weight: 600, color: C.ink2, alpha: nm });
  // 开销与乘加的比例：乘加记作 1 份（40 px）
  const b = po(u, S4.bars, S4.bars + 0.5), b2 = po(u, S4.bars + 1.2, S4.bars + 1.7), U = 36, BX0 = 420, Y1 = 700, Y2 = 820;
  txt('一次一个', X, Y1 + 40, { size: 30, weight: 700, alpha: b }); txt('（FP16 乘加）', X, Y1 + 76, { size: 22, color: C.gray, alpha: b });
  rr(BX0, Y1, U, 52, C.teal, b); rr(BX0 + U + 4, Y1, 20 * U * lin(u, S4.bars + 0.2, S4.bars + 1.0), 52, C.red, b);
  txt('开销 = 乘加的 20 倍', BX0 + U + 20, Y1 + 37, { size: 28, weight: 700, color: '#fff', alpha: po(u, S4.bars + 0.8, S4.bars + 1.1) });
  txt('一整块', X, Y2 + 40, { size: 30, weight: 700, alpha: b2 }); txt('（HMMA）', X, Y2 + 76, { size: 22, color: C.gray, alpha: b2 });
  rr(BX0, Y2, U, 52, C.teal, b2); rr(BX0 + U + 4, Y2, 0.22 * U, 52, C.red, b2);
  txt('开销 = 乘加的 0.22 倍', BX0 + U + 30, Y2 + 37, { size: 28, weight: 700, color: C.red, alpha: b2 });
}

// ================= 5 数据复用 =================
// 4 × 4 × 4 按 k 拆成 4 层外积：第 k 层是 A 的第 k 列（4 个数）与 B 的第 k 行（4 个数）两两相乘，16 个乘法器。
// A 的每个数连到同一行的 4 个乘法器，B 的每个数连到同一列的 4 个；4 层共 64 个乘法器，只读 32 个数。
const G5 = { x: 640, y: 450, c: 92, d: 26 };
const S5 = { wire: [0.8, 3.2], hiA: bt(4, 1) + 0.2, hiB: bt(4, 1) + 2.4, bars: bt(4, 2) + 0.2 };
evRun(4, S5.wire[0], S5.wire[1], 0.15, 'tick'); ev(4, S5.hiA, 'pop'); ev(4, S5.hiB, 'pop'); ev(4, S5.bars, 'land'); ev(4, S5.bars + 0.9, 'land');
function s5(u) {
  const { x, y, c, d } = G5, a = po(u, 0.2, 0.7), w = lin(u, ...S5.wire);
  // 后面 3 层（k = 1..3）作为叠在后面的淡层
  for (let L = 3; L >= 1; L--) for (let r = 0; r < 4; r++) for (let q = 0; q < 4; q++) rr(x + q * c + L * d, y + r * c - L * d, c - 10, c - 10, C.tealL, a * 0.25 * (1 + (3 - L) * 0.2));
  for (let r = 0; r < 4; r++) for (let q = 0; q < 4; q++) {
    const hA = u > S5.hiA && r === 1, hB = u > S5.hiB && q === 2;
    rr(x + q * c, y + r * c, c - 10, c - 10, hA || hB ? C.teal : C.tealL, a); txt('×', x + q * c + (c - 10) / 2, y + r * c + c / 2 + 8, { size: 34, weight: 700, color: '#fff', align: 'center', alpha: a });
  }
  // A 的第 0 列在左，B 的第 0 行在上
  const AXc = x - 180, BYr = y - 170;
  for (let r = 0; r < 4; r++) {
    const hi = u > S5.hiA && r === 1;
    rr(AXc, y + r * c, c - 10, c - 10, hi ? C.blue : '#8ea3e6', a);
    for (let q = 0; q < 4; q++) { const ln = clamp(w * 16 - (r * 4 + q)); line(AXc + c - 10, y + r * c + (c - 10) / 2, mix(AXc + c - 10, x + q * c + 8, ln), y + r * c + (c - 10) / 2, hi ? C.blue : '#8ea3e6', hi ? 4 : 2, a * (hi ? 1 : 0.6)); }
  }
  for (let q = 0; q < 4; q++) {
    const hi = u > S5.hiB && q === 2;
    rr(x + q * c, BYr, c - 10, c - 10, hi ? C.yel : '#f1cf62', a);
    for (let r = 0; r < 4; r++) { const ln = clamp(w * 16 - (q * 4 + r)); line(x + q * c + (c - 10) / 2 + 12, BYr + c - 10, x + q * c + (c - 10) / 2 + 12, mix(BYr + c - 10, y + r * c + 8, ln), hi ? C.yelD : '#e2b93a', hi ? 4 : 2, a * (hi ? 1 : 0.6)); }
  }
  txt('A 的第 k 列', AXc + (c - 10) / 2, y + 4 * c + 30, { size: 24, weight: 700, color: C.blue, align: 'center', alpha: a });
  txt('B 的第 k 行', x - 30, BYr + 52, { size: 24, weight: 700, color: C.yelD, align: 'right', alpha: a });
  txt('16 个乘法器 × 4 层（k = 0, 1, 2, 3）= 64', x, y + 4 * c + 30, { size: 26, weight: 700, color: C.tealD, alpha: a });
  txt('A 的这个数使用 4 次', AXc - 10, y + c - 18, { size: 26, weight: 800, color: C.blue, align: 'right', alpha: po(u, S5.hiA + 0.3, S5.hiA + 0.7) });
  txt('B 的这个数使用 4 次', x + 2 * c + 40, BYr - 20, { size: 26, weight: 800, color: C.yelD, align: 'center', alpha: po(u, S5.hiB + 0.3, S5.hiB + 0.7) });
  // 读进来的数：一次一个 128，一整块 32（每个数 4 px 宽 × 4）
  const b = po(u, S5.bars, S5.bars + 0.5), b2 = po(u, S5.bars + 0.9, S5.bars + 1.4), RX = 1250, PX = 4.4;
  txt('64 次乘加需读取的数', RX, 430, { size: 30, weight: 700, alpha: b });
  txt('一次一个', RX, 500, { size: 28, weight: 700, alpha: b }); rr(RX, 516, 128 * PX * lin(u, S5.bars, S5.bars + 0.8), 52, C.red, b);
  txt('128 个', RX + 128 * PX - 16, 553, { size: 28, weight: 800, color: '#fff', align: 'right', alpha: po(u, S5.bars + 0.6, S5.bars + 0.9) });
  txt('每次乘加 2 个', RX, 600, { size: 22, color: C.gray, alpha: b });
  txt('一整块', RX, 680, { size: 28, weight: 700, alpha: b2 }); rr(RX, 696, 32 * PX, 52, C.teal, b2);
  txt('32 个', RX + 32 * PX + 16, 733, { size: 28, weight: 800, color: C.tealD, alpha: b2 });
  txt('A 16 个 + B 16 个', RX, 780, { size: 22, color: C.gray, alpha: b2 });
}

// ================= 6 一条指令 =================
// CUTLASS SM80_16x8x16_F32BF16BF16F32_TN：每个线程 A 4 个 32 bit 寄存器（8 个 BF16）、B 2 个（4 个）、C / D 各 4 个 float
const CODE = ['asm volatile(', '  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "', '  "{%0,  %1,  %2,  %3},"', '  "{%4,  %5,  %6,  %7},"', '  "{%8,  %9},"', '  "{%10, %11, %12, %13};\\n"'];
const S6 = { grow: [bt(5, 1) + 0.2, bt(5, 1) + 2.2], warp: bt(5, 2) + 0.2, join: bt(5, 2) + 3.0 };
const K6 = { ox: 1560, oy: 420, s: 15 };                 // 16 × 8 × 16 的方块堆：i = N（8），j = M（16），k = K（16）
evRun(5, 0.6, 2.4, 0.18, 'click'); evRun(5, S6.grow[0], S6.grow[1], 0.12, 'tick'); ev(5, S6.grow[1], 'land');
evRun(5, S6.warp, S6.warp + 1.6, 0.05, 'tick'); ev(5, S6.join + 0.6, 'land');
function s6(u) {
  // 代码框
  const a = po(u, 0.3, 0.8), X = 96, Y = 300;
  rr(X, Y, 1060, 290, '#1d1d1f', a);
  txt('cute/arch/mma_sm80.hpp', X + 24, Y + 40, { size: 22, color: '#9a968c', font: MONO, alpha: a });
  CODE.forEach((l, i) => { const on = i === 1; txt(l.slice(0, Math.floor(l.length * lin(u, 0.6 + i * 0.25, 0.9 + i * 0.25))), X + 24, Y + 86 + i * 36, { size: 24, weight: on ? 700 : 500, color: on ? '#7fe0cf' : '#e8e4da', font: MONO, alpha: a }); });
  // 方块堆从指令长出来：2048 个，按真实个数画
  const gr = lin(u, ...S6.grow), n = Math.floor(2048 * gr);
  if (gr > 0) {
    stack(1640, 880, 8, 16, 16, K6.s, { count: n });
    txt(`${n} 次乘加`, 1600, 960, { size: 40, weight: 800, align: 'center', color: C.tealD, font: ZH });
    cube(1250, 560, K6.s, SETS.teal, po(u, S6.grow[1], S6.grow[1] + 0.4));
    txt('1 次', 1250, 600, { size: 24, weight: 700, align: 'center', color: C.gray, alpha: po(u, S6.grow[1], S6.grow[1] + 0.4) });
  }
  // 32 个线程：每个持有 A 8 个、B 4 个、C 4 个数
  const w = po(u, S6.warp, S6.warp + 0.4), j = po(u, S6.join, S6.join + 0.8), LW = 126, LH = 58;
  txt('一个 warp = 32 个线程，各持一部分', X, 650, { size: 30, weight: 700, alpha: w });
  for (let t = 0; t < 32; t++) {
    const x = X + (t % 8) * (LW + 6), y = 676 + Math.floor(t / 8) * (LH + 8), al = w * clamp((u - S6.warp) * 20 - t);
    rr(x, y, LW, LH, '#fff', al); txt(`${t}`, x + 8, y + 22, { size: 16, weight: 600, color: C.gray, font: MONO, alpha: al });
    rr(x + 8, y + 30, 8 * 6, 18, C.blue, al); rr(x + 8 + 50, y + 30, 4 * 6, 18, C.yel, al); rr(x + 8 + 78, y + 30, 4 * 6, 18, C.purple, al);
  }
  [[C.blue, 'A 8 个'], [C.yelD, 'B 4 个'], [C.purple, 'C 4 个']].forEach(([col, s], i) =>
    txt(s, 640 + i * 130, 650, { size: 26, weight: 800, color: col, alpha: w }));
  txt('合计：A 16 × 16，B 16 × 8，C 16 × 8', X, 950, { size: 24, weight: 700, color: C.ink2, alpha: j });
  arrow(X + 8 * (LW + 6) + 24, 800, 1400, 800, j, C.ink, j * 0.5, 3);
  placeholder(1180, 300, 640, 150, '实机材料待补：编译结果', ['nvcc 编译该指令得到的 SASS（HMMA.16816）', '需要：CUDA 工具链，不需要 GPU'], po(u, 1.4, 1.8) * (1 - po(u, S6.grow[0] - 0.3, S6.grow[0])));
}

// ================= 7 大矩阵 =================
// 4096 阶：4096³ ≈ 687 亿次乘加 ÷ 2048 = 33,554,432 条 mma。单位 = 16 个元素：大立方体 256 单位，一条 mma 是 0.5 × 1 × 1 单位（N 8、M 16、K 16）
const S7 = { zoom: [0.6, 4.8], count: bt(6, 1) + 0.2, kloop: [bt(6, 2) + 0.3, bt(6, 2) + 4.8] };
const ANC = [880, 600];
ev(6, S7.zoom[0], 'whoosh'); ev(6, S7.zoom[0] + 1.6, 'whoosh'); ev(6, S7.count, 'pop'); evRun(6, S7.count + 0.2, S7.count + 1.6, 0.1, 'tick'); evRun(6, S7.kloop[0], S7.kloop[1], 0.1, 'tick');
function s7(u) {
  const z = p(u, ...S7.zoom), un = Math.exp(mix(Math.log(240), Math.log(1.15), z));   // 一个单位的像素数
  // 主角块的中心固定在 ANC：位于大立方体前上角（i 255.5..256，j 255..256，k 255..256）
  const [hx, hy] = Q(0, 0, un, 255.75, 255.5, 255.5), ox = ANC[0] - hx, oy = ANC[1] - hy;
  g.save(); g.beginPath(); g.rect(0, 262, W, 730); g.clip();             // 拉远前的大块不盖住页眉与字幕
  box(ox, oy, un, 0, 0, 0, 256, 256, 256, SETS.ghost, po(u, 0.6, 1.4), C.lg);
  // K 方向：这一块结果（16 × 8）要沿 K 走 256 条 mma
  const kl = lin(u, ...S7.kloop), kn = Math.floor(256 * kl);
  if (kn > 0) box(ox, oy, un, 255.5, 255, 256 - kn, 0.5, 1, kn, SETS.teal, 1);
  box(ox, oy, un, 255.5, 255, 255, 0.5, 1, 1, PU, 1, un > 6 ? C.paper : null);
  // 小块缩成一个点之后加圈
  const ring = po(u, S7.zoom[1] - 1.2, S7.zoom[1] - 0.4);
  g.restore();
  if (ring > 0) { g.globalAlpha = ring; g.strokeStyle = C.purple; g.lineWidth = 4; g.beginPath(); g.arc(ANC[0], ANC[1], 22, 0, Math.PI * 2); g.stroke(); g.globalAlpha = 1; }
  txt('一条 mma（16 × 8 × 16）', ANC[0] + 34, ANC[1] - 26, { size: 26, weight: 700, color: C.purple, alpha: ring });
  txt('4096 × 4096 × 4096', ANC[0], ANC[1] + 340, { size: 28, weight: 700, color: C.gray, align: 'center', font: MONO, alpha: po(u, S7.zoom[1] - 0.6, S7.zoom[1]) });
  // 计数
  const c = po(u, S7.count, S7.count + 0.4), RX = 1290;
  txt('4096³ ≈ 687 亿次乘加', RX, 420, { size: 36, weight: 800, alpha: c });
  txt('÷ 2048 次 / 条', RX, 476, { size: 30, weight: 600, color: C.gray, alpha: c });
  const cnt = Math.floor(33554432 * po(u, S7.count + 0.2, S7.count + 1.6));
  txt(`= ${cnt.toLocaleString('en-US')} 条 mma`, RX, 548, { size: 40, weight: 800, color: C.tealD, alpha: c, font: ZH });
  const k = po(u, S7.kloop[0], S7.kloop[0] + 0.4);
  txt('同一块结果沿 K 方向累加', RX, 680, { size: 30, weight: 700, color: C.tealD, alpha: k });
  txt(`K：${Math.max(16, kn * 16)} / 4096 · 第 ${Math.max(1, kn)} / 256 条`, RX, 728, { size: 28, weight: 600, color: C.ink2, alpha: k, font: ZH });
  txt('16 × 8 的结果始终保存在累加器中', RX, 776, { size: 24, color: C.gray, alpha: k });
}

// ================= 8 四代 =================
// 每 SM 每时钟的 FP16 / BF16 dense 乘加：V100 512（白皮书：8 个 × 64），A100 1024，H100 2048，B200 4096（由规格推算）；
// 同一个 SM 里 FP32 core：64、64、128、128。方块边长相同，体积即个数。
const GENS = [['V100', '2017', 512, [8, 8, 8], 64, [4, 4, 4], 'mma.sync', '一个 warp'], ['A100', '2020', 1024, [8, 8, 16], 64, [4, 4, 4], 'mma.sync', '一个 warp'],
  ['H100', '2022', 2048, [16, 8, 16], 128, [8, 4, 4], 'wgmma', '4 个 warp 合作'], ['B200', '2024', 4096, [16, 16, 16], 128, [8, 4, 4], 'tcgen05.mma', '单个线程发起']];
const S8 = { gen: [0.8, 1.6], ins: bt(7, 1) + 0.2, ratio: bt(7, 2) + 0.2 };
GENS.forEach((_, i) => { ev(7, S8.gen[0] + i * S8.gen[1], 'land'); ev(7, S8.ins + i * 0.5, 'click'); }); ev(7, S8.ratio, 'pop');
function s8(u) {
  const s = 9.5, BASE = 820;
  line(96, BASE + 4, 1824, BASE + 4, C.ink, 3, po(u, 0.3, 0.8));
  GENS.forEach(([name, yr, n, [nx, ny, nz], f, [fx, fy, fz], ins, who], i) => {
    const cx = 330 + i * 410, a = po(u, S8.gen[0] + i * S8.gen[1], S8.gen[0] + i * S8.gen[1] + 0.6), grow = p(u, S8.gen[0] + i * S8.gen[1], S8.gen[0] + i * S8.gen[1] + 0.9);
    // Tensor Core 的一堆（teal）与普通 core 的一堆（gray），后角定位使前角落在基线上
    const tz = Math.max(1, Math.round(nz * grow)), tox = cx + 50 - (nx - ny) * s * 0.866 / 2, toy = BASE - (nx + ny) * s * 0.5;
    if (a > 0) box(tox, toy, s, 0, 0, 0, nx, ny, tz, SETS.teal, a, C.paper);
    const fox = cx - 150 - (fx - fy) * s * 0.866 / 2, foy = BASE - (fx + fy) * s * 0.5;
    if (a > 0) box(fox, foy, s, 0, 0, 0, fx, fy, fz, SETS.gray, a, C.paper);
    txt(`${n}`, cx + 50, BASE - (nx + ny) * s * 0.5 - nz * s - 18, { size: 34, weight: 800, color: C.tealD, align: 'center', alpha: a * grow });
    txt(`${f}`, cx - 150, BASE - (fx + fy) * s * 0.5 - fz * s - 14, { size: 26, weight: 700, color: C.gray, align: 'center', alpha: a });
    txt(name, cx, BASE + 56, { size: 36, weight: 800, align: 'center', alpha: a }); txt(yr, cx, BASE + 92, { size: 24, color: C.gray, align: 'center', alpha: a });
    const b = po(u, S8.ins + i * 0.5, S8.ins + i * 0.5 + 0.4);
    txt(ins, cx, BASE + 140, { size: 28, weight: 700, color: C.blue, align: 'center', font: MONO, alpha: b });
    txt(who, cx, BASE + 172, { size: 22, color: C.ink2, align: 'center', alpha: b * (1 - po(u, S8.ratio, S8.ratio + 0.2) * (i === 3 ? 0 : 0)) });
  });
  txt('灰：普通 core', 96, 330, { size: 24, weight: 700, color: C.gray, alpha: po(u, 0.8, 1.3) }); txt('绿：Tensor Core', 96, 362, { size: 24, weight: 700, color: C.tealD, alpha: po(u, 0.8, 1.3) });
  const r = po(u, S8.ratio, S8.ratio + 0.5);
  rr(1500, 300, 324, 150, C.ink, r);
  txt('4096 ÷ 128 = 32 倍', 1662, 360, { size: 32, weight: 800, color: '#fff', align: 'center', alpha: r, font: ZH });
  txt('按规格峰值：30 倍', 1662, 410, { size: 26, weight: 600, color: C.tealL, align: 'center', alpha: r });
}

// ================= 9 代价 =================
// B200：BF16 dense 2250 TFLOPS = 1125 万亿次乘加 / 秒；SFU 的 exp 约 5 万亿次 / 秒（Blackwell Ultra 博客的 Blackwell 一列）
const S9 = { bars: 0.6, thin: bt(8, 1) + 0.2, back: bt(8, 2) + 0.2 };
ev(8, S9.bars, 'land'); ev(8, S9.thin, 'pop'); ev(8, S9.back, 'whoosh');
function s9(u) {
  const back = p(u, S9.back, S9.back + 0.8), a = po(u, S9.bars, S9.bars + 0.5) * (1 - back), X = 420, BW = 1400;
  line(X, 400, X, 640, C.ink, 3, a);
  txt('矩阵乘法', 96, 470, { size: 36, weight: 800, color: C.tealD, alpha: a }); rr(X, 430, BW * lin(u, S9.bars, S9.bars + 1.2), 64, C.teal, a);
  txt('每秒 1125 万亿次乘加', X + BW - 20, 474, { size: 30, weight: 800, color: '#fff', align: 'right', alpha: po(u, S9.bars + 1.0, S9.bars + 1.4) * (1 - back) });
  const t = po(u, S9.thin, S9.thin + 0.4) * (1 - back);
  txt('指数 exp', 96, 590, { size: 36, weight: 800, color: C.red, alpha: t }); rr(X, 550, Math.max(3, BW * 5 / 1125), 64, C.red, t);
  txt('每秒 5 万亿次 · 约 1/225', X + 30, 594, { size: 30, weight: 800, color: C.red, alpha: t });
  txt('softmax、激活函数等逐元素运算由普通 core 与特殊函数单元（SFU）执行', 96, 720, { size: 28, weight: 600, color: C.ink2, alpha: po(u, S9.thin + 1.0, S9.thin + 1.5) * (1 - back) });
  // 回到开头：芯片与两条进度，这次画成方块
  if (back > 0) {
    chip(1330, 330, 494, { off: B200_OFF, hot: 1, alpha: back });
    const r0 = S9.back + 0.6, r1 = SCRIPT[8].dur - 0.4, k = lin(u, r0, r1), S = 26;
    txt('一次一个', 96, 400, { size: 34, weight: 800, alpha: back }); txt('普通 core', 96, 440, { size: 24, color: C.gray, alpha: back });
    const one = Math.floor(16 * k); for (let i = 0; i < one; i++) box(330 + i * 34, 360, S, 0, 0, 0, 1, 1, 1, SETS.gray, back, C.paper);
    // 30 倍的速率：同样时间里 16 × 30 = 480 个方块，约 7.5 个 4 × 4 × 4
    txt('一次一整块', 96, 640, { size: 34, weight: 800, color: C.tealD, alpha: back }); txt('Tensor Core', 96, 680, { size: 24, color: C.gray, alpha: back });
    const many = 480 * k, blocks = Math.floor(many / 64);
    for (let b = 0; b <= Math.min(7, blocks); b++) {
      const cnt = b < blocks ? 64 : Math.floor(many - blocks * 64); if (cnt <= 0) continue;
      stack(350 + b * 122, 760, 4, 4, 4, 13, { count: cnt, alpha: back });
    }
    txt('相同时间内：30 倍的方块', 96, 900, { size: 30, weight: 700, color: C.tealD, alpha: po(u, r1 - 1.6, r1 - 1.0) });
  }
}

const DRAW = [s1, s2, s3, s4, s5, s6, s7, s8, s9];
function draw(t) {
  g.fillStyle = C.paper; g.fillRect(0, 0, W, H);
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  const u = t - AT[k];
  if (k > 0) header(k, u); else txt(SCRIPT[0].src, 1824, 76, { size: 26, weight: 500, color: C.gray, align: 'right', alpha: po(u, S1.out[1], S1.out[1] + 0.5) });
  DRAW[k](u);
  subtitle(SCRIPT[k].beats, VO?.beats?.[k], u);
}
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
