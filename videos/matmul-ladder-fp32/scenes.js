// 矩阵乘法阶梯两集（matmul-ladder-fp32 / matmul-ladder-tc）共用的各幕，两个目录里是同一份文件；按 script.js 的 EPISODE 与各幕 tag 取用。
// 主体是固定的三栏画面（stage）：左代码、中数据布局、右 SM 剖面，底部成绩阶梯；三栏由 frames.js 的逐帧剧本同一时钟驱动。
// 阶梯：每一版一级，8192 阶实测 TFLOPS，对数刻度；蓝 = FP32 线，青 = Tensor Core 线，虚线 = cuBLAS。网格配色同 1.3 讲解页：显存读取 = 橙，片上数据 = 绿，选中的线程 = 蓝，同一 warp = 浅蓝。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const IDX = (tag) => SCRIPT.findIndex((s) => s.tag === tag);                     // 两集共用本文件，按 tag 找幕
const NAME = EPISODE === 'fp32' ? 'H100 矩阵乘法 · 上' : 'H100 矩阵乘法 · 下';
// 音效：[幕内时间, 种类]。同类动作同一个声音：卡片出现 pop，阶梯长一级 land，一串柱子依次出现 tick，引出下一个瓶颈 click，收拢成全貌 whoosh
const EV = SCRIPT.map(() => []);
const ev = (k, u, kind) => EV[k].push([u, kind]);
const evBeats = (k, kind = 'pop') => SCRIPT[k].beats.forEach((b) => ev(k, b[0], kind));

function header(k, u) {
  if (SCRIPT[k].stage) {                                        // 三栏幕用紧凑页眉，给三栏留出高度
    const s = SCRIPT[k], a = po(u, 0, 0.5);
    txt(NAME, 40, 40, { size: 22, weight: 700, color: C.ink2 });
    txt(s.src, 1880, 40, { size: 20, weight: 500, color: C.gray, align: 'right', alpha: a });
    const tw0 = tw(s.tag, 28, 700) + 32; rr(40, 62, tw0, 44, C.ink, a); txt(s.tag, 56, 94, { size: 28, weight: 700, color: '#fff', alpha: a });
    rich(s.head, 40 + tw0 + 22, 98, 38, 700, a); return;
  }
  const s = SCRIPT[k], a = po(u, 0, 0.5);
  txt(NAME, 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 76, { size: 24, weight: 500, color: C.gray, align: 'right', alpha: a });
  const tf = /^[\x00-\x7f]+$/.test(s.tag) ? MONO : ZH;
  rr(96, 100, tw(s.tag, 32, 700, tf) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a, font: tf });
  rich(s.head, 96, 226 + (1 - a) * 16, 52, 700, a);
}
// 多行文字：lines = [[文字, 字号, 颜色, 字重, 字体]]
function tl(x, y, lines, a) {
  let yy = y;
  for (const [s, size = 26, col = C.ink, wt = 700, font = ZH] of lines) { yy += size * 1.15; txt(s, x, yy, { size, weight: wt, color: col, alpha: a, font }); yy += size * 0.4; }
}
function cmpBar(x, y, label, v, sc, col, alpha = 1, grow = 1, valTxt = null, h = 44, lsize = 24) {
  txt(label, x - 16, y + h * 0.7, { size: lsize, weight: 700, color: col, align: 'right', alpha });
  const w = Math.max(3, v * sc * grow); rr(x, y, w, h, col, alpha);
  txt(valTxt ?? String(v), x + w + 12, y + h * 0.72, { size: 26, weight: 800, color: col, alpha: alpha * po(grow, 0.7, 1), font: MONO });
}

// 4 × 4 小格（开场用）：hi(r, q) 返回该格颜色
function cells(x, y, n, m, c, fill, alpha = 1, hi) { for (let r = 0; r < n; r++) for (let q = 0; q < m; q++) rr(x + q * c, y + r * c, c - 6, c - 6, (hi && hi(r, q)) || fill, alpha); }
const mixc = (c0, c1, t) => '#' + [0, 1, 2].map((i) => Math.round(parseInt(c0.substr(1 + i * 2, 2), 16) * (1 - t) + parseInt(c1.substr(1 + i * 2, 2), 16) * t).toString(16).padStart(2, '0')).join('');
// ---------- SM 剖面（沿用 1.3 讲解页的 HW 面板布局与各版状态） ----------
// 坐标取自讲解页的 400 × 296 SVG。warps：16 个字符 = 4 个调度器 × 4 个 warp；r 执行 · w 等显存 · s 等同步 · p 生产者 · c 消费者
const SMU = { fp32: [14, 82, 118, 36, 'FP32 单元', '每拍 128 次 FMA'], tc: [140, 82, 118, 36, 'Tensor Core ×4', '每拍 2048 次乘加'], lsu: [266, 82, 120, 36, 'LSU / TMA', '访存 · 搬整块'],
  reg: [14, 160, 176, 36, '寄存器', '每 SM 65536 个'], smem: [206, 160, 180, 36, '共享内存 / L1', '228 KiB · 每拍 128 B'], l2: [6, 222, 388, 26, 'L2 50 MiB（全卡共享）'], hbm: [6, 262, 388, 26, '显存 HBM · 3.0 TB/s'] };
const SML = { 'hbm-l2': [200, 262, 200, 248], 'l2-reg': [90, 222, 90, 196], 'l2-smem': [300, 222, 300, 196], 'smem-reg': [206, 172, 190, 172], 'reg-smem': [190, 184, 206, 184],
  'reg-fp32': [60, 160, 60, 118], 'reg-tc': [170, 160, 170, 118], 'smem-tc': [230, 160, 230, 118], 'smem-l2': [350, 196, 350, 222] };
function smPanel(x0, y0, s, st, a, u) {
  if (a <= 0.003) return;
  const X = (v) => x0 + v * s, Y = (v) => y0 + v * s;
  rr(X(6), Y(6), 388 * s, 200 * s, '#ebe6db', a); sr(X(6), Y(6), 388 * s, 200 * s, C.lg, 1.5, a);
  txt('一个 SM（H100 有 132 个）', X(14), Y(21), { size: 13 * s * (st.focus ? 1.25 : 1), weight: 600, color: C.gray, alpha: a });
  const W = (st.warps || '................').padEnd(16, '.');
  for (let i = 0; i < 4; i++) {
    rr(X(14 + i * 94), Y(26), 86 * s, 44 * s, '#f7f4ee', a); txt('调度器 ' + i, X(20 + i * 94), Y(43), { size: 13 * s * (st.focus ? 1.2 : 1), weight: 600, color: C.gray, alpha: a });
    for (let j = 0; j < 4; j++) {
      const c = W[i * 4 + j], cx = X(30 + i * 94 + j * 18), cy = Y(58), r = 5.5 * s;
      const col = { r: C.use, w: C.hot, s: C.gray, p: C.thr, c: C.use }[c];
      g.globalAlpha = a; g.beginPath(); g.arc(cx, cy, r, 0, 7);
      if (c === 'c') { g.fillStyle = '#fff'; g.fill(); g.lineWidth = 2.5; g.strokeStyle = C.use; g.stroke(); }
      else { g.fillStyle = col || '#d9d3c5'; g.fill(); }
      g.globalAlpha = 1;
    }
  }
  // st.focus：只保留与这一幕有关的单元，其余淡出；有 focus 时字号放大
  const fs = st.focus ? 1.3 : 1;
  for (const [key, [ux, uy, uw, uh, lab, sub]] of Object.entries(SMU)) {
    const busy = (st.busy || []).includes(key), hot = st.hot === key, ua = st.focus && !st.focus.includes(key) ? a * 0.2 : a;
    rr(X(ux), Y(uy), uw * s, uh * s, hot ? '#f6d8c4' : busy ? '#cfe6d7' : '#f7f4ee', ua);
    sr(X(ux), Y(uy), uw * s, uh * s, hot ? C.hot : busy ? C.use : C.lg, hot || busy ? 2.5 : 1.2, ua);
    const f2 = st.focus && st.focus.includes(key) ? fs : 1;
    txt(lab, X(ux + 6), Y(uy + (sub ? 16 * f2 : uh / 2 + 5)), { size: 14 * s * f2, weight: 700, color: hot ? C.hot : C.ink, alpha: ua });
    if (sub && !st.focus) txt(sub, X(ux + 6), Y(uy + 31), { size: 11.5 * s, weight: 500, color: C.gray, alpha: ua });
  }
  for (const [key, [x1, y1, x2, y2]] of Object.entries(SML)) {
    const on = (st.flows || []).includes(key), hot = st.hot === key, thick = (st.thick || []).includes(key);
    const col = hot ? C.hot : on ? C.use : C.lg, lw = (thick ? 5 : on || hot ? 3 : 1.5) * s * 0.9;
    if (on || hot) {
      g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = lw; g.setLineDash([6 * s, 5 * s]); g.lineDashOffset = -u * 40 * s;
      g.beginPath(); g.moveTo(X(x1), Y(y1)); g.lineTo(X(x2), Y(y2)); g.stroke(); g.setLineDash([]); g.lineDashOffset = 0; g.globalAlpha = 1;
      const an = Math.atan2(y2 - y1, x2 - x1); g.globalAlpha = a; g.fillStyle = col; g.beginPath(); g.moveTo(X(x2), Y(y2));
      g.lineTo(X(x2) - 9 * Math.cos(an - 0.5), Y(y2) - 9 * Math.sin(an - 0.5)); g.lineTo(X(x2) - 9 * Math.cos(an + 0.5), Y(y2) - 9 * Math.sin(an + 0.5)); g.fill(); g.globalAlpha = 1;
    } else line(X(x1), Y(y1), X(x2), Y(y2), col, lw, a * 0.6);
  }
  if (st.note) st.note.forEach((t, i) => txt(t, x0, y0 + 312 * s + i * 26, { size: 19, weight: 700, color: i ? C.ink2 : (st.hot ? C.hot : C.use), alpha: a }));
}

// ================= 上集开场：矩阵乘法是什么 =================
// 4 × 4 的例子：点积视角（跟 C[1][2]），外积视角（A 的第 k 列 ⊗ B 的第 k 行，四层相加），计算量，Linear 层，全片的问题
function sIntro(u) {
  const K = IDX('矩阵乘法'), B_ = (j) => bt(K, j), b = (j) => po(u, B_(j), B_(j) + 0.5);
  const c = 66, AX = 150, AY = 560, BX = 470, BY = 270, CX = 470, CY = 560, R = 1, Q = 2;
  const outer = u >= B_(2), dotK = u < B_(1) ? -1 : Math.min(3, Math.floor((u - B_(1)) / ((B_(2) - B_(1)) / 4.2)));
  const ok = outer ? Math.min(3, Math.floor((u - B_(2)) / ((B_(3) - B_(2)) / 4.4))) : -1;
  const m0 = b(0) * (1 - p(u, B_(4) - 0.2, B_(4) + 0.4));
  // A（蓝）、B（黄）、C（紫）
  cells(AX, AY, 4, 4, c, '#c9d3f3', m0, (r, q) => (outer ? (q === ok ? C.blue : null) : r === R && dotK >= 0 ? (q === dotK ? C.blue : '#8ea3e6') : null));
  cells(BX, BY, 4, 4, c, '#f6e6aa', m0, (r, q) => (outer ? (r === ok ? C.yel : null) : q === Q && dotK >= 0 ? (r === dotK ? C.yel : '#f1cf62') : null));
  cells(CX, CY, 4, 4, c, '#e1d5fb', m0, (r, q) => (outer ? mixc('#e1d5fb', C.purple, (ok + 1) / 4) : r === R && q === Q ? C.purple : null));
  txt('A', AX - 46, AY + 150, { size: 44, weight: 800, color: C.blue, alpha: m0 }); txt('B', BX + 280, BY + 150, { size: 44, weight: 800, color: C.yelD, alpha: m0 }); txt('C', CX + 280, CY + 150, { size: 44, weight: 800, color: C.purple, alpha: m0 });
  const RX = 1000;
  // 点积
  const d = b(0) * (1 - (outer ? 1 : 0)) * (1 - p(u, B_(4) - 0.2, B_(4) + 0.4));
  txt('C[i][j] = Σₖ A[i][k] · B[k][j]', RX, 330, { size: 40, weight: 800, color: C.ink, alpha: d, font: MONO });
  txt('C 的一个元素 = A 的一行 · B 的一列（点积）', RX, 384, { size: 26, weight: 700, color: C.ink2, alpha: d });
  if (dotK >= 0 && !outer) { txt('C[1][2] =', RX, 460, { size: 24, weight: 800, color: C.purple, alpha: d, font: MONO }); for (let k = 0; k <= dotK; k++) txt(`${k ? '+ ' : '  '}A[1][${k}] · B[${k}][2]`, RX + 40, 500 + k * 38, { size: 24, weight: 700, color: k === dotK ? C.ink : C.ink2, alpha: d, font: MONO }); }
  // 外积
  const o = (outer ? b(2) : 0) * (1 - p(u, B_(4) - 0.2, B_(4) + 0.4));
  txt('C = Σₖ A[:, k] ⊗ B[k, :]', RX, 330, { size: 40, weight: 800, color: C.ink, alpha: o, font: MONO });
  txt('A 的一列 ⊗ B 的一行 = 一整张 4 × 4（外积）', RX, 384, { size: 26, weight: 700, color: C.ink2, alpha: o });
  txt(`已累加 ${ok + 1} / 4 层`, RX, 450, { size: 26, weight: 800, color: C.purple, alpha: o });
  txt('寄存器分块会用到这个视角', RX, 500, { size: 22, weight: 600, color: C.gray, alpha: o });
  // 计算量
  const cpt = b(3) * (1 - p(u, B_(4) - 0.2, B_(4) + 0.4));
  tl(RX, 560, [['N³ 次乘加 = 2N³ 次浮点运算', 30, C.ink, 800], ['N = 8192：1.1 × 10¹² FLOP', 34, C.tealD, 800, MONO]], cpt);
  // Linear 层
  const ln = b(4) * (1 - p(u, B_(5) - 0.2, B_(5) + 0.4)), LX = 200, LY = 330;
  rr(LX, LY, 220, 300, C.blue, ln); txt('x', LX + 110, LY + 165, { size: 48, weight: 800, color: '#fff', align: 'center', alpha: ln });
  txt('token × hidden', LX + 110, LY + 340, { size: 22, weight: 700, color: C.ink2, align: 'center', alpha: ln });
  txt('·', LX + 270, LY + 170, { size: 60, weight: 800, alpha: ln });
  rr(LX + 320, LY, 300, 220, C.yel, ln); txt('Wᵀ', LX + 470, LY + 125, { size: 48, weight: 800, color: '#fff', align: 'center', alpha: ln });
  txt('hidden × out', LX + 470, LY + 260, { size: 22, weight: 700, color: C.ink2, align: 'center', alpha: ln });
  txt('=', LX + 670, LY + 170, { size: 60, weight: 800, alpha: ln });
  rr(LX + 720, LY, 300, 300, C.purple, ln); txt('y', LX + 870, LY + 165, { size: 48, weight: 800, color: '#fff', align: 'center', alpha: ln });
  txt('大模型的每个 Linear 层：y = x · Wᵀ，一次矩阵乘法', LX, LY + 420, { size: 30, weight: 800, alpha: ln });
  txt('W 按行存放、沿 K 连续：下集 Tensor Core 版本的布局', LX, LY + 466, { size: 22, weight: 600, color: C.gray, alpha: ln });
  // 全片的问题
  const qa = b(5);
  txt('计算单元为什么总在等待？', W / 2, 520, { size: 72, weight: 800, align: 'center', alpha: qa });
  pill('目标：在 H100 SXM 上手写 8192 阶矩阵乘法，追平 cuBLAS', 560, 660, C.ink, qa, 28);
}


// ================= 复杂度与下界 =================
// 计算量 N³ 与数据量 3N² 随 N 拉开；算术强度上限 N/6 FLOP/B 越过 H100 的平衡点；Strassen 一句带过；I/O 下界（Hong & Kung 1981）
function sComplex(u) {
  const K = IDX('复杂度'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  tl(96, 270, [['时间：N³ 次乘加 = 2N³ FLOP', 26, C.purple, 800], ['8192 阶：1.1 × 10¹² FLOP', 22, C.ink2, 700, MONO], ['数据：A、B、C 共 3N² 个数', 26, C.hot, 800], ['8192 阶（FP32）：805 MB', 22, C.ink2, 700, MONO]], b(0));
  tl(96, 470, [['强度上限 = 2N³ ÷ (3N² × 4 B) = N/6 FLOP/B', 24, C.tealD, 800, MONO], ['8192 阶 ≈ 1365 FLOP/B', 22, C.ink2, 700, MONO]], b(1));
  tl(96, 568, [['naive 实际：0.25 FLOP/B', 24, C.red, 800, MONO], ['全片要做的：把 N/6 的复用兑现出来', 26, C.ink, 800]], b(2));
  const c3 = b(3);
  rr(96, 676, 820, 132, C.paper2, c3);
  tl(116, 674, [['Strassen：7 次乘法，O(N^2.807)', 22, C.ink, 800], ['理论指数 ω < 2.371339（2024）→ 2.371177（2026，AlphaEvolve 参与）', 19, C.ink2, 700], ['误差更大、额外加法与临时矩阵增加访存', 18, C.gray, 600]], c3);
  pill('cuBLAS 实际不用', 730, 704, C.red, c3, 22);
  const c4 = b(4), c5 = b(5);
  rr(96, 822, 820, 160, C.paper2, c4);
  tl(116, 820, [['I/O 下界（Hong & Kung 1981，红蓝卵石游戏）', 22, C.ink, 800], ['快速存储能放 M 个数：至少搬运 Ω(N³ / √M) 个数', 22, C.tealD, 800, MONO]], c4);
  tl(116, 900, [['分块可以达到下界：块越大越省，正方形最优', 22, C.use, 800]], c5);
  // 上图：N³ 与 3N²（对数刻度，N 从 16 到 16384）
  const X0 = 1040, W_ = 760, X = (n) => X0 + (Math.log2(n) - 4) / 10 * W_;
  const T0 = 280, TH = 280, YT = (v) => T0 + TH - Math.log10(v) / 13 * TH, a0 = b(0);
  line(X0, T0 + TH, X0 + W_, T0 + TH, C.ink2, 2, a0); line(X0, T0, X0, T0 + TH, C.ink2, 2, a0);
  txt('个数（对数刻度）', X0, T0 - 12, { size: 18, weight: 700, color: C.gray, alpha: a0 });
  const grow = lin(u, bt(K, 0) + 0.3, bt(K, 0) + 2.3), Ns = [];
  for (let e = 4; e <= 14.001; e += 0.25) Ns.push(2 ** e);
  const curve = (f, col, lab) => { const m = Math.max(2, Math.floor(Ns.length * grow)); g.globalAlpha = a0; g.strokeStyle = col; g.lineWidth = 4; g.beginPath(); Ns.slice(0, m).forEach((n, i) => (i ? g.lineTo(X(n), YT(f(n))) : g.moveTo(X(n), YT(f(n))))); g.stroke(); g.globalAlpha = 1; const n = Ns[m - 1]; txt(lab, X(n) - 8, YT(f(n)) - 12, { size: 20, weight: 800, color: col, align: 'right', alpha: a0 }); };
  curve((n) => n ** 3, C.purple, 'N³ 次乘加'); curve((n) => 3 * n * n, C.hot, '3N² 个数');
  // 下图：算术强度 N/6 与平衡点、naive
  const B0 = 640, BH = 300, YB = (v) => B0 + BH - (Math.log10(v) + 1) / 5 * BH, a1 = b(1);
  line(X0, B0 + BH, X0 + W_, B0 + BH, C.ink2, 2, a1); line(X0, B0, X0, B0 + BH, C.ink2, 2, a1);
  txt('FLOP/B（对数刻度）', X0, B0 - 12, { size: 18, weight: 700, color: C.gray, alpha: a1 });
  [16, 256, 4096, 16384].forEach((n) => txt(String(n), X(n), B0 + BH + 26, { size: 16, weight: 600, color: C.gray, align: 'center', alpha: a1, font: MONO }));
  txt('N', X0 + W_ + 14, B0 + BH + 6, { size: 20, weight: 800, color: C.gray, alpha: a1, font: MONO });
  const gi = lin(u, bt(K, 1) + 0.3, bt(K, 1) + 2.0), m = Math.max(2, Math.floor(Ns.length * gi));
  g.globalAlpha = a1; g.strokeStyle = C.teal; g.lineWidth = 4; g.beginPath(); Ns.slice(0, m).forEach((n, i) => (i ? g.lineTo(X(n), YB(n / 6)) : g.moveTo(X(n), YB(n / 6)))); g.stroke(); g.globalAlpha = 1;
  txt('强度上限 N/6', X(64) + 10, YB(64 / 6) + 30, { size: 20, weight: 800, color: C.tealD, alpha: a1 });
  line(X0, YB(295), X0 + W_, YB(295), C.red, 2.5, a1, [10, 7]);
  txt('H100 平衡点 ≈ 295 FLOP/B（989 TFLOPS ÷ 3.35 TB/s）', X0 + 8, YB(295) - 10, { size: 17, weight: 700, color: C.red, alpha: a1 });
  const cr = po(u, bt(K, 1) + 2.0, bt(K, 1) + 2.6);
  g.globalAlpha = a1 * cr; g.fillStyle = C.tealD; g.beginPath(); g.arc(X(8192), YB(8192 / 6), 7, 0, 7); g.fill(); g.globalAlpha = 1;
  txt('8192 阶 ≈ 1365', X(8192) - 10, YB(8192 / 6) - 14, { size: 18, weight: 800, color: C.tealD, align: 'right', alpha: a1 * cr });
  txt('右侧：计算受限', X(1770) + 12, B0 + 150, { size: 20, weight: 800, color: C.tealD, alpha: a1 * cr });
  line(X(1770), B0, X(1770), B0 + BH, C.tealD, 1.5, a1 * cr, [4, 5]);
  const a2 = b(2);
  line(X0, YB(0.25), X0 + W_, YB(0.25), C.hot, 2.5, a2, [10, 7]);
  txt('naive 实际 0.25 FLOP/B', X0 + W_ - 8, YB(0.25) - 10, { size: 18, weight: 800, color: C.hot, align: 'right', alpha: a2 });
}

// ================= 上集结尾：调参与 FP32 天花板 =================
function sNext(u) {
  const K = IDX('结尾'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5), back = p(u, bt(K, 2), bt(K, 2) + 0.8);
  const a0 = b(0) * (1 - back);
  tl(96, 280, [['autotune：9 组分块参数，每个规模选最快', 22, C.gray, 700], ['8192 阶：dbuf 44.30 → tuned 44.53，只多 0.5%', 32, C.ink, 800]], a0);
  const a1 = b(1) * (1 - back), cs = 9, X = 96, Y = 470;
  txt('每个 SM 每个时钟的乘加次数', X, Y - 20, { size: 24, weight: 700, color: C.gray, alpha: a1 });
  for (let i = 0; i < 128; i++) rr(X + (i % 16) * cs, Y + Math.floor(i / 16) * cs, cs - 2, cs - 2, C.blue, a1);
  txt('FP32 单元 128：本集的天花板', X, Y + 8 * cs + 34, { size: 24, weight: 800, color: C.blue, alpha: a1 });
  for (let i = 0; i < 2048; i++) rr(X + 330 + (i % 64) * cs, Y + Math.floor(i / 64) * cs, cs - 2, cs - 2, C.teal, a1);
  txt('Tensor Core 2048 = 16 倍（下集）', X + 330, Y + 32 * cs + 34, { size: 24, weight: 800, color: C.tealD, alpha: a1 });
  // 回到开场的阶梯：FP32 段全部点亮并标出去除的瓶颈，Tensor Core 段灰显
  if (back > 0) {
    ladder(220, 360, 1580, 440, { upto: 7, ghost: 0.6 * back, alpha: back, names: true, values: 'none', ann: (i) => (i < 7 ? 1 : 0), band: b(3), cub: [back, 0.35 * back], cubLabel: false, size: 0.95 });
    txt('下集：Tensor Core', 1800, 340, { size: 28, weight: 800, color: C.tealD, align: 'right', alpha: b(3) });
    let x = 220; CATS.slice(0, 3).forEach(([s, col]) => { rr(x, 952, 24, 24, col, b(3)); txt(s, x + 32, 972, { size: 22, weight: 700, color: col, alpha: b(3) }); x += tw(s, 22, 700) + 80; });
  }
}

// ================= 下集开场：回顾上集 =================
function sRecap(u) {
  const K = IDX('回顾'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  ladder(220, 330, 1580, 470, { upto: 7, ghost: 0.6, alpha: b(0), names: true, values: 'all', pct: true, cub: [b(0), b(1)], cubLabel: false, size: 0.95, flash: u > bt(K, 2) ? lin(u, bt(K, 2), bt(K, 2) + 2.0) * 12 + 6 : -1 });
  const Y = (v) => 330 + 470 - 470 * Math.log10(v) / 3, sw = 1580 / STEPS.length;
  line(220, Y(67), 220 + 7 * sw, Y(67), C.blue, 3, b(0), [8, 6]); txt('FP32 峰值 ≈ 67：上集的天花板', 228, Y(67) - 12, { size: 22, weight: 800, color: C.blue, alpha: b(0) });
  txt('本集：Tensor Core（BF16 输入、FP32 累加）', 1800, 286, { size: 26, weight: 800, color: C.tealD, align: 'right', alpha: b(1) });
  txt('每 SM 每时钟 2048 次乘加，FP32 单元为 128', 1800, 318, { size: 22, weight: 700, color: C.ink2, align: 'right', alpha: b(1) });
}

// ================= 2 纸面估算 =================
function s2(u) {
  const K = IDX('纸面估算'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  tl(96, 280, [['运算量', 22, C.gray, 700], ['2N³ = 1.1 × 10¹² FLOP', 36, C.ink, 800, MONO]], b(0));
  tl(96, 400, [['最少访存：读 A、B，写 C 各一次', 22, C.gray, 700], ['3N² × 4 B = 805 MB → 0.27 ms', 36, C.hot, 800, MONO]], b(1));
  tl(96, 520, [['按 FP32 峰值 67 TFLOPS 计算', 22, C.gray, 700], ['1.1 × 10¹² ÷ 67 × 10¹² = 16.4 ms', 36, C.blue, 800, MONO]], b(2));
  tl(96, 640, [['无复用：每次乘加读 2 个 float = 8 B', 22, C.gray, 700], ['0.25 FLOP/B × 3.0 TB/s = 0.75 TFLOPS', 36, C.hot, 800, MONO]], b(3));
  // 时间条：访存与计算，同一尺度
  const X = 1060, sc = 560 / 16.4, g2 = po(u, bt(1, 2) + 0.2, bt(1, 2) + 1.2);
  txt('同一尺度的时间', X, 300, { size: 22, weight: 700, color: C.gray, alpha: b(1) });
  cmpBar(X + 110, 320, '访存', 0.27, sc, C.hot, b(1), 1, '0.27 ms');
  cmpBar(X + 110, 376, '计算', 16.4, sc, C.blue, b(2), g2, '16.4 ms');
  // 尺子：对数刻度 0.1 … 1000
  const r = b(3), Y0 = 560, L = (v) => X + 60 + (Math.log10(v) + 1) / 4 * 680;
  txt('尺子 · TFLOPS（对数刻度）', X, Y0 - 90, { size: 22, weight: 700, color: C.gray, alpha: r });
  line(L(0.1), Y0, L(1000), Y0, C.ink, 3, r);
  [0.1, 1, 10, 100, 1000].forEach((v) => { line(L(v), Y0 - 8, L(v), Y0 + 8, C.ink, 2, r); txt(String(v), L(v), Y0 + 34, { size: 18, weight: 600, color: C.gray, align: 'center', alpha: r, font: MONO }); });
  const mk = (v, s, col, al, dy) => { if (al <= 0) return; rr(L(v) - 4, Y0 - 30, 8, 30, col, al); txt(s, L(v), Y0 - 40 - dy, { size: 22, weight: 800, color: col, align: 'center', alpha: al }); };
  mk(0.75, '无复用 0.75', C.hot, r, 0); const r2 = b(4); mk(67, 'FP32 峰值 67', C.blue, r2, 0); mk(989, 'BF16 989', C.tealD, r2, 30);
  // 每 SM 每时钟：128 对 2048
  const cs = 7;
  txt('每个 SM 每个时钟的乘加次数', X, 660, { size: 22, weight: 700, color: C.gray, alpha: r2 });
  for (let i = 0; i < 128; i++) rr(X + (i % 16) * cs, 680 + Math.floor(i / 16) * cs, cs - 1.5, cs - 1.5, C.blue, r2);
  txt('FP32 128', X, 760, { size: 22, weight: 800, color: C.blue, alpha: r2, font: MONO });
  for (let i = 0; i < 2048; i++) rr(X + 180 + (i % 64) * cs, 680 + Math.floor(i / 64) * cs, cs - 1.5, cs - 1.5, C.teal, r2);
  txt('Tensor Core 2048 = 16 倍', X + 180, 942, { size: 22, weight: 800, color: C.tealD, alpha: r2 });
}

// ================= 3 线程层级 =================
// C 矩阵（示意 64 × 64，block 16 × 16 → grid 4 × 4）；选中 blockIdx = (2, 1)，threadIdx = (5, 3)：线性序号 53 → warp 1、lane 21
const PROBE = ['$ /tmp/probe', 'name                 NVIDIA H100 80GB HBM3', 'SM count             132', 'smem per SM          228 KiB', 'registers per SM     65536'];
// 等轴测投影：世界坐标 (x, y, z) 以线程格为单位；视图 V = [边长, ox, oy]，推镜头时在两个视图之间插值
const ISO = (V, x, y, z) => [V[1] + (x - y) * V[0] * 0.866, V[2] + (x + y) * V[0] * 0.5 - z * V[0]];
function isoBox(V, x, y, z, dx, dy, dz, top, left, right, a, edge = null) {
  if (a <= 0.003) return;
  const P = (i, j, k) => ISO(V, i, j, k);
  g.globalAlpha = a;
  poly([P(x, y, z + dz), P(x + dx, y, z + dz), P(x + dx, y + dy, z + dz), P(x, y + dy, z + dz)], top);
  poly([P(x, y + dy, z), P(x + dx, y + dy, z), P(x + dx, y + dy, z + dz), P(x, y + dy, z + dz)], left);
  poly([P(x + dx, y, z), P(x + dx, y + dy, z), P(x + dx, y + dy, z + dz), P(x + dx, y, z + dz)], right);
  if (edge) { g.strokeStyle = edge; g.lineWidth = 1.2; g.beginPath(); [[P(x, y, z + dz), P(x + dx, y, z + dz), P(x + dx, y + dy, z + dz), P(x, y + dy, z + dz), P(x, y, z + dz)]].forEach((ps) => ps.forEach((q, n) => (n ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])))); g.stroke(); }
  g.globalAlpha = 1;
}
const SH = { slab: ['#e9e4d8', '#d4cdbd', '#c5bdab'], sel: ['#cfe3d6', '#a9d0b6', '#8fbf9f'], thr: ['#efe9dc', '#d9d2c2', '#cbc3b1'], warp: ['#c7d7f6', '#9db7ea', '#86a4e0'], me: ['#5b8ef0', C.thr, '#1f4fb8'] };
function isoArrow(V, x0, y0, x1, y1, z, col, label, a) {
  const [ax, ay] = ISO(V, x0, y0, z), [bx, by] = ISO(V, x1, y1, z);
  arrow(ax, ay, bx, by, 1, col, a, 3);
  txt(label, bx + (bx > ax ? 10 : -10), by + 8, { size: 22, weight: 800, color: col, align: bx > ax ? 'left' : 'right', alpha: a, font: MONO });
}
function s3(u) {
  // 节拍映射：map3[原第 n 拍] = 现在的第几拍（-1 表示这一段不出现）；v5 只保留 grid、推镜头到 block（含落到 SM）、warp 抬起
  const K = IDX('线程层级'), MAP3 = SCRIPT[K].map3 || [0, 1, 2, 3, 4, 5], T3 = (n) => (MAP3[n] < 0 ? 1e9 : bt(K, MAP3[n])), b = (j) => po(u, T3(j), T3(j) + 0.5);
  term(96, 280, 620, PROBE, { size: 20, lh: 34, hl: (i) => (i >= 2 ? b(0) : 0), alpha: b(0) });
  txt('SM = Streaming Multiprocessor', 96, 500, { size: 24, weight: 800, color: C.ink, alpha: b(0) });
  // 镜头：先看整个 grid（4 × 4 个 block，每个 block 16 × 16 × 1 个线程），再推到 block (2, 1)
  const BS = 16, GAP = 4, TH = 2.5, SEL = [2, 1];
  const zoom = p(u, T3(2), T3(2) + 1.6), g0 = b(1);
  g.save(); g.beginPath(); g.rect(750, 250, 1170, 820); g.clip();       // 推镜头时只在右侧舞台内作画
  const V0 = [6.2, 1250, 600 - (4 * BS + 3 * GAP) * 0.5 * 6.2], cx = SEL[0] * (BS + GAP) + BS / 2, cy = SEL[1] * (BS + GAP) + BS / 2;
  const s1v = 24, V1 = [s1v, 1130 - (cx - cy) * s1v * 0.866, 560 - (cx + cy) * s1v * 0.5];
  const V = V0.map((v, i) => mix(v, V1[i], zoom));
  const slabA = 1 - p(zoom, 0.45, 0.85), cubeA = p(zoom, 0.45, 0.85);
  for (let by = 0; by < 4; by++) for (let bx = 0; bx < 4; bx++) {
    const sel = bx === SEL[0] && by === SEL[1], x = bx * (BS + GAP), y = by * (BS + GAP);
    const al = g0 * (sel ? slabA : (1 - zoom));
    const sh = sel && u > T3(1) + 1.2 ? SH.sel : SH.slab;
    isoBox(V, x, y, 0, BS, BS, TH, sh[0], sh[1], sh[2], al, C.lg);
    if (al > 0.05) {
      g.globalAlpha = al * 0.5; g.strokeStyle = '#bdb5a3'; g.lineWidth = 0.8; g.beginPath();
      for (let i = 1; i < BS; i++) { let [p1, p2] = [ISO(V, x + i, y, TH), ISO(V, x + i, y + BS, TH)]; g.moveTo(...p1); g.lineTo(...p2); [p1, p2] = [ISO(V, x, y + i, TH), ISO(V, x + BS, y + i, TH)]; g.moveTo(...p1); g.lineTo(...p2); }
      g.stroke(); g.globalAlpha = 1;
      const [lx, ly] = ISO(V, x + BS / 2, y + BS / 2, TH); txt(`(${bx},${by})`, lx, ly + 7, { size: 18, weight: 700, color: sel ? C.use : C.ink2, align: 'center', alpha: al, font: MONO });
    }
  }
  const gl = g0 * (1 - zoom);
  isoArrow(V, -2, -3, 30, -3, 0, C.ink2, 'x', gl); isoArrow(V, -3, -2, -3, 30, 0, C.ink2, 'y', gl);
  txt('grid：gridDim = (4, 4, 1)', 1250, 300, { size: 26, weight: 800, align: 'center', alpha: gl, font: MONO });
  txt('每个 block 负责 C 的 16 × 16 格（示意）', 1250, 336, { size: 20, weight: 600, color: C.gray, align: 'center', alpha: gl });
  // 推近后的 block：16 × 16 个线程立方体；warp 1（第 2、3 行，threadIdx.x 连续的 32 个）抬起，选中线程再抬起
  if (cubeA > 0.003) {
    const ox = SEL[0] * (BS + GAP), oy = SEL[1] * (BS + GAP), lift = p(u, T3(3), T3(3) + 1.0), me = p(u, T3(4), T3(4) + 0.9);
    const cubes = [];
    for (let ty = 0; ty < BS; ty++) for (let tx = 0; tx < BS; tx++) {
      const inW = ty === 2 || ty === 3, isMe = tx === 5 && ty === 3;
      cubes.push({ tx, ty, z: (inW ? lift * 2.2 : 0) + (isMe ? me * 2.2 : 0), sh: isMe && me > 0.05 ? SH.me : inW && lift > 0.05 ? SH.warp : SH.thr, up: inW });
    }
    cubes.sort((p1, p2) => (p1.up - p2.up) || (p1.tx + p1.ty - p2.tx - p2.ty));
    cubes.forEach((c) => isoBox(V, ox + c.tx + 0.1, oy + c.ty + 0.1, c.z, 0.8, 0.8, 0.8, c.sh[0], c.sh[1], c.sh[2], g0 * cubeA));
    const ca = g0 * cubeA;
    isoArrow(V, ox, oy - 1.5, ox + 9, oy - 1.5, 0, C.thr, 'threadIdx.x', ca); isoArrow(V, ox - 1.5, oy, ox - 1.5, oy + 9, 0, C.thr, 'threadIdx.y', ca);
    txt('block (2, 1)：blockDim = (16, 16, 1)', 1130, 300, { size: 26, weight: 800, align: 'center', alpha: ca, font: MONO });
    if (lift > 0.05) { const [wx, wy] = ISO(V, ox + 16, oy + 2.5, 3.2); txt('warp 1：threadIdx.x 连续的 32 个线程', wx + 14, wy, { size: 22, weight: 800, color: C.thr, alpha: ca * lift }); txt('16 × 16 的 block 中占两行', wx + 14, wy + 30, { size: 19, weight: 600, color: C.ink2, alpha: ca * lift }); }
    if (me > 0.05) { const [mx, my] = ISO(V, ox + 5.5, oy + 3.5, 5.6); line(mx, my, 1560, 660, C.thr, 2, ca * me); }
  }
  g.restore();
  // SM：block 整体落到一个 SM 上
  const s = b(2), SX = 96, SY = 560;
  rr(SX, SY, 620, 250, C.paper2, s);
  txt('一个 SM', SX + 20, SY + 40, { size: 26, weight: 800, alpha: s });
  rr(SX + 20, SY + 60, 150, 150, '#cfe3d6', s); txt('block (2,1)', SX + 95, SY + 140, { size: 20, weight: 800, color: C.use, align: 'center', alpha: s, font: MONO });
  tl(SX + 200, SY + 50, [['最多 2048 个线程 = 64 个 warp', 22, C.ink2, 700], ['一个 block 最多 1024 个线程', 22, C.ink2, 700], ['共享内存 228 KiB · 寄存器 65536 个', 22, C.use, 700], ['由驻留的 block 分用', 20, C.gray, 600]], s);
  // 选中的线程：逐项标出
  const t = b(4), X = 1560;
  rr(X - 16, 630, 316, 320, '#fbf9f4', t); sr(X - 16, 630, 316, 320, C.thr, 2, t);
  tl(X, 636, [['blockIdx  = (2, 1)', 21, C.ink, 700, MONO], ['threadIdx = (5, 3)', 21, C.thr, 700, MONO], ['序号 3 × 16 + 5 = 53', 19, C.ink2, 600, MONO], ['warp 53 / 32 = 1', 21, C.ink, 800, MONO], ['lane 53 % 32 = 21', 21, C.ink, 800, MONO], ['row = 1 × 16 + 3 = 19', 18, C.gray, 600, MONO], ['col = 2 × 16 + 5 = 37', 18, C.gray, 600, MONO]], t);
  // 合并访存：32 个地址汇成少数几次访问
  const m = b(5), AY = 900, AX = 780;
  txt('一个 warp 发出 32 个读取地址 → 合并为尽量少的内存访问', AX, AY - 18, { size: 22, weight: 800, color: C.hot, alpha: m });
  for (let i = 0; i < 48; i++) rr(AX + i * 15, AY, 12, 28, i >= 16 && i < 32 ? C.hot : C.cell, m);
  sr(AX + 16 * 15 - 3, AY - 3, 16 * 15 + 3, 34, C.ink, 2.5, m);
  txt('一次访问', AX + 24 * 15, AY + 58, { size: 20, weight: 700, color: C.ink, align: 'center', alpha: m });
}


// ---------- 无逐帧剧本的版本用的示意图元 ----------
function perClock(x, y, a, cs = 6) {
  for (let i = 0; i < 128; i++) rr(x + (i % 16) * cs, y + Math.floor(i / 16) * cs, cs - 1, cs - 1, C.blue, a);
  txt('FP32 128', x, y + 8 * cs + 22, { size: 18, weight: 800, color: C.blue, alpha: a, font: MONO });
  for (let i = 0; i < 2048; i++) rr(x + 130 + (i % 64) * cs, y + Math.floor(i / 64) * cs * 0.25, cs - 1, cs * 0.25 - 0.5, C.teal, a);
  txt('Tensor Core 2048（16 倍）', x + 130, y + 8 * cs + 22, { size: 18, weight: 800, color: C.tealD, alpha: a });
}
function wmmaMap(x, y, w, h, a) {
  const s = 2.3, BX = x + 20, BY = y + 10;
  rr(BX, BY, 128 * s, 128 * s, C.cell, a);
  for (let wr = 0; wr < 2; wr++) for (let wc = 0; wc < 4; wc++) sr(BX + wc * 32 * s, BY + wr * 64 * s, 32 * s, 64 * s, C.ink2, 1.5, a);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) rr(BX + (32 + j * 16) * s + 1, BY + i * 16 * s + 1, 16 * s - 2, 16 * s - 2, (i + j) % 2 ? C.teal : C.tealL, a);
  txt('block 128 × 128 · 8 个 warp（2 × 4）', BX, BY + 128 * s + 24, { size: 17, weight: 700, color: C.gray, alpha: a });
  tl(x + 340, y - 4, [['一个 warp：64 × 32', 19, C.tealD, 700], ['= 4 × 2 个 16 × 16 片段', 19, C.ink2, 700], ['C = A·Wᵀ', 22, C.ink, 800, MONO], ['A：M × K，W：N × K', 18, C.ink2, 600, MONO], ['都沿 K 连续（Linear 层）', 18, C.ink2, 600]], a);
}
function pathRow(x, y, items, a, w = 96, gap = 46, size = 17) {
  items.forEach(([s, col, via], i) => {
    const xx = x + i * (w + gap); node(xx, y, w, 44, s, col, a, '', size);
    if (i < items.length - 1) { arrow(xx + w + 3, y + 22, xx + w + gap - 3, y + 22, 1, C.ink2, a, 2.5); if (via) txt(via, xx + w + gap / 2, y - 6, { size: 14, weight: 700, color: C.ink, align: 'center', alpha: a, font: MONO }); }
  });
}
function swzPanels(x, y, a, c = 26) {
  [[false, '原布局', C.red], [true, '块序号 ⊕ 行号', C.use]].forEach(([swz, lab, col], s) => {
    const X = x + s * 260;
    txt(lab, X, y - 8, { size: 18, weight: 800, color: col, alpha: a });
    for (let r = 0; r < 8; r++) for (let q = 0; q < 8; q++) { const lg = swz ? q ^ r : q, on = lg === 0; rr(X + q * c, y + r * c, c - 3, c - 3, on ? col : C.cell, a); txt(String(lg), X + q * c + (c - 3) / 2, y + r * c + 17, { size: 12, weight: 700, color: on ? '#fff' : C.gray, align: 'center', alpha: a, font: MONO }); }
  });
}
function ringMap(x, y, w, h, a, u, t0) {
  const ph = Math.floor(Math.max(0, u - t0) / 1.0);
  ['计算 t', '已到 t+1', '搬运 t+2'].forEach((_, i) => { const role = (i - ph % 3 + 3) % 3; node(x + 20 + i * 172, y + 20, 160, 70, ['计算 t', '已到 t+1', '搬运 t+2'][role], [C.teal, C.use, C.hot][role], a, '', 21); });
  txt('共享内存：3 个缓冲槽轮转', x + 20, y + 124, { size: 19, weight: 700, color: C.ink2, alpha: a });
  const rv = lin(u, t0, t0 + 3), comp = [], cp = [];
  for (let i = 0; i < 6; i++) { comp.push([i * 2, i * 2 + 1.9, C.teal, '计算']); cp.push([i * 2 - 1, i * 2 + 1, C.hot, '搬运']); }
  lane(x + 100, y + 160, 420, 36, 12, comp, rv, a, 'Tensor', C.tealD); lane(x + 100, y + 204, 420, 26, 12, cp, rv, a, 'cp.async', C.hot);
}
function tmaMap(x, y, w, h, a, u, t0) {
  const fill = lin(u, t0 + 0.4, t0 + 2.6), X = x + 20;
  rr(X, y + 30, 14, 14, C.thr, a); arrow(X + 18, y + 37, X + 50, y + 37, 1, C.ink2, a, 2.5);
  node(X + 56, y + 14, 110, 50, 'TMA', C.ink, a, '', 24); arrow(X + 172, y + 37, X + 210, y + 37, 1, C.ink2, a, 2.5);
  rr(X + 216, y + 6, 290, 130, C.cell, a); rr(X + 216, y + 6, 290 * fill, 130, C.use, a);
  txt('A 块 128 × 64 · 16 KB', X + 361, y + 160, { size: 17, weight: 700, color: C.ink2, align: 'center', alpha: a });
  const got = Math.round(16384 * fill), done = fill >= 1;
  rr(X + 216, y + 176, 290, 40, done ? C.use : C.paper, a); txt(`mbarrier ${got} / 16384 B`, X + 228, y + 203, { size: 17, weight: 800, color: done ? '#fff' : C.ink2, alpha: a, font: MONO });
  for (let i = 0; i < 4; i++) node(X + i * 50, y + 250, 44, 40, `w${i}`, C.thr, a, '', 16);
  txt('warpgroup', X, y + 312, { size: 17, weight: 700, color: C.thr, alpha: a });
  node(X + 216, y + 250, 290, 50, 'wgmma → Tensor Core', C.teal, a, '', 20);
  arrow(X + 361, y + 218, X + 361, y + 246, 1, C.use, a, 3); txt('直接读共享内存', X + 372, y + 240, { size: 15, weight: 700, color: C.use, alpha: a });
}
function tiles(x, y, a, grouped, n = 132, cw = 13, ch = 4.6) {
  for (let m = 0; m < 64; m++) for (let q = 0; q < 32; q++) {
    const on = grouped ? m < 16 && q * 16 + m < n : m * 32 + q < n;
    rr(x + q * cw, y + m * ch, cw - 2, ch - 1, on ? C.teal : C.cell, a);
  }
  const ar = grouped ? 16 : 5, wc = grouped ? 9 : 32;
  rr(x - 22, y, 12, 64 * ch, C.cell, a); rr(x - 22, y, 12, ar * ch, C.hot, a);
  rr(x, y - 20, 32 * cw, 12, C.cell, a); rr(x, y - 20, wc * cw, 12, C.hot, a);
  txt('A', x - 22, y + 64 * ch + 20, { size: 16, weight: 800, color: C.hot, alpha: a }); txt('W', x - 30, y - 10, { size: 16, weight: 800, color: C.hot, alpha: a });
}
function clusterMap(x, y, w, h, a, u, t0) {
  const cl = p(u, t0 + 0.8, t0 + 1.8), X = x + 200;
  node(x + 20, y + 80, 120, 90, 'L2', C.ink2, a, '', 28);
  rr(X, y + 10, 220, 40, C.hot, a); txt('W 块', X + 110, y + 37, { size: 20, weight: 800, color: '#fff', align: 'center', alpha: a });
  rr(X, y + 70, 220, 70, C.teal, a); txt('tile 0 · SM 0', X + 110, y + 112, { size: 19, weight: 800, color: '#fff', align: 'center', alpha: a });
  rr(X, y + 150, 220, 70, C.teal, a); txt('tile 1 · SM 1', X + 110, y + 192, { size: 19, weight: 800, color: '#fff', align: 'center', alpha: a });
  arrow(x + 145, y + 110, X - 6, y + 105, 1, C.hot, a * (1 - cl), 7); arrow(x + 145, y + 140, X - 6, y + 185, 1, C.hot, a * (1 - cl), 7);
  arrow(x + 145, y + 110, X - 6, y + 105, 1, C.use, a * cl, 3.5); arrow(x + 145, y + 140, X - 6, y + 185, 1, C.use, a * cl, 3.5);
  line(X + 226, y + 105, X + 250, y + 105, C.use, 3, a * cl); line(X + 250, y + 105, X + 250, y + 185, C.use, 3, a * cl); line(X + 226, y + 185, X + 250, y + 185, C.use, 3, a * cl);
  txt(cl < 0.5 ? '各从 L2 读一整份' : '各读一半，TMA 多播给两边', x + 20, y + 270, { size: 20, weight: 800, color: cl < 0.5 ? C.hot : C.use, alpha: a });
  txt('W 从 L2 读出：2 份 → 1 份；每个 block 每步 48 KB → 32 KB', x + 20, y + 304, { size: 19, weight: 700, color: C.ink2, alpha: a * cl });
}

// ================= 三栏画面：左代码、中数据布局、右 SM 剖面；底部成绩阶梯 =================
// 逐帧剧本来自 frames.js（由 extract.mjs 从 1.3 讲解页交互版导出）：每帧的代码行、格子状态、硬件状态天然对齐。
// 版本节拍的第 4 项 x = { v: 版本名, f: [起帧, 止帧] 或省略（无逐帧剧本时只显示改动行）, end: 'act1' 等（本拍画整幕的阶梯） }
const COL = { code: [40, 140, 724, 728], lay: [780, 140, 552, 728], hw: [1348, 140, 532, 728] };
// 左栏的精简代码：保留真实的变量名与关键行（索引计算、循环、读写），去掉样板，长行按语法断开；
// map 把讲解页源码的行号（帧里的 lines）映射到精简代码的行号，高亮与中栏、右栏同一帧对齐
const SIMPLE = {
  naive_T: { src: '精简自 matmul.cu 的 matmul_naive_T', lines: ['__global__ void matmul_naive_T(', '    const float* A, const float* B,', '    float* C, int N) {', '  int row = blockIdx.x * blockDim.x', '          + threadIdx.x;', '  int col = blockIdx.y * blockDim.y', '          + threadIdx.y;', '  if (row >= N || col >= N) return;', '  float sum = 0;', '  for (int k = 0; k < N; k++)', '    sum += A[row * N + k]', '         * B[k * N + col];', '  C[row * N + col] = sum;', '}'],
    map: { 0: [0, 1, 2], 1: [3, 4], 2: [5, 6], 3: [7], 4: [8], 5: [9, 10, 11], 6: [12], 7: [13] } },
  naive: { src: '精简自 matmul.cu 的 matmul_naive', lines: ['__global__ void matmul_naive(', '    const float* A, const float* B,', '    float* C, int N) {', '  int row = blockIdx.y * blockDim.y', '          + threadIdx.y;', '  int col = blockIdx.x * blockDim.x', '          + threadIdx.x;', '  if (row >= N || col >= N) return;', '  float sum = 0;', '  for (int k = 0; k < N; k++)', '    sum += A[row * N + k]', '         * B[k * N + col];', '  C[row * N + col] = sum;', '}'],
    map: { 0: [0, 1, 2], 1: [3, 4], 2: [5, 6], 3: [7], 4: [8], 5: [9, 10, 11], 6: [12], 7: [13] } },
};
const simpleIdx = (vn, list) => [...new Set((list || []).flatMap((i) => SIMPLE[vn].map[i] || []))];
const STEP_OF = Object.fromEntries(STEPS.map((s, i) => [s[0], i]));
const VNAME = { naive_T: 'naive_T（对照）', smem32: 'smem32', reg8x8: 'reg8x8', tuned: 'tuned' };
// 某一时刻的版本与帧：落在哪一拍，拍内按时间匀速走过 [起帧, 止帧]
function vstate(k, u) {
  const S = SCRIPT[k], xs = S.xs; let j = 0; S.beats.forEach((b, i) => { if (u >= b[0]) j = i; });
  let v = null; for (let i = j; i >= 0; i--) if (xs[i] && xs[i].v) { v = xs[i].v; break; }
  let f = xs[j] && xs[j].f;
  if (!f) for (let i = j - 1; i >= 0; i--) { const x = xs[i]; if (x && x.v && x.v !== v) break; if (x && x.f) { f = [x.f[1], x.f[1]]; break; } }
  const t0 = S.beats[j][0], t1 = j + 1 < S.beats.length ? S.beats[j + 1][0] : S.dur;
  const d = FR[v], n = d ? d.frames.length : 0;
  let fi = -1;
  let ff = 0;
  if (n && f) { const [a, b] = f, pos = (u - t0) / Math.max(0.01, t1 - t0) * (b - a + 1); fi = Math.min(b, a + Math.floor(pos)); ff = a + Math.floor(pos) > b ? 1 : pos - Math.floor(pos); }
  let pv = null, vStart = 0;                      // 本版从哪一拍开始；上一个版本（换版本时代码淡入淡出）
  for (let i = j; i >= 0; i--) { const x = xs[i]; if (x && x.v && x.v !== v) { pv = x.v; break; } vStart = S.beats[i][0]; }
  return { x: { v, f, end: xs[j] && xs[j].end }, d, f: fi >= 0 ? d.frames[fi] : null, fi, ff, t0, t1, j, pv, vStart };
}
// ---------- 左：完整 kernel 代码，当前行高亮，改动行标 + ----------
function codePanel(vn, f, a, u, hl = null) {
  const d = FR[vn]; if (!d || a <= 0.003) return;
  if (SIMPLE[vn]) {                                                // 精简代码：22 px，不截断
    const S = SIMPLE[vn], [x, y, w, h] = COL.code, size = 23, lh = 38, top = y + 70;
    rr(x, y, w, h, '#fbf9f4', a); sr(x, y, w, h, C.lg, 1.5, a);
    txt(vn === 'naive_T' ? 'naive_T（对照组）' : vn, x + 18, y + 38, { size: 24, weight: 800, color: vn === 'naive_T' ? C.red : C.blue, alpha: a, font: MONO });
    const cur = new Set(hl || simpleIdx(vn, f ? f.lines : [])), chg = new Set(simpleIdx(vn, d.changed));
    S.lines.forEach((ln, i) => {
      const yy = top + i * lh;
      if (cur.has(i)) rr(x + 6, yy - 4, w - 12, lh, '#f6d8c4', a);
      if (chg.has(i)) txt('+', x + 14, yy + 24, { size, weight: 800, color: C.use, alpha: a, font: MONO });
      txt(String(i + 1), x + 64, yy + 24, { size: 17, weight: 500, color: C.gray, align: 'right', alpha: a, font: MONO });
      txt(ln, x + 80, yy + 24, { size, weight: cur.has(i) ? 700 : 500, color: cur.has(i) || chg.has(i) ? C.ink : C.ink2, alpha: a, font: MONO });
    });
    txt(S.src, x + w - 16, y + h - 16, { size: 17, weight: 600, color: C.gray, align: 'right', alpha: a });
    return;
  }
  const [x, y, w, h] = COL.code, size = 15, lh = 22, top = y + 44, vis = Math.floor((h - 52) / lh);
  rr(x, y, w, h, '#fbf9f4', a); sr(x, y, w, h, C.lg, 1.5, a);
  txt(vn === 'naive_T' ? 'matmul.cu · matmul_naive_T' : (d.line === 'fp32' ? 'matmul.cu · ' : 'matmul_tc.cu · ') + vn, x + 16, y + 30, { size: 18, weight: 700, color: C.gray, alpha: a, font: MONO });
  const cur = new Set(f ? f.lines : d.changed.slice(0, 6)), chg = new Set(d.changed);
  const focus = f && f.lines.length ? f.lines[0] : d.changed.length ? d.changed[0] : 0;
  const start = clamp(Math.round(focus - vis * 0.35), 0, Math.max(0, d.code.length - vis));
  for (let i = start; i < Math.min(d.code.length, start + vis); i++) {
    const yy = top + (i - start) * lh;
    if (cur.has(i)) rr(x + 4, yy - 2, w - 8, lh, f ? '#f6d8c4' : 'rgba(47,138,87,0.14)', a);
    if (chg.has(i)) txt('+', x + 12, yy + 15, { size, weight: 800, color: C.use, alpha: a, font: MONO });
    txt(String(i + 1), x + 56, yy + 15, { size: 13, weight: 500, color: C.gray, align: 'right', alpha: a, font: MONO });
    let s = d.code[i]; const max = w - 80; if (tw(s, size, 500, MONO) > max) { while (s.length && tw(s + '…', size, 500, MONO) > max) s = s.slice(0, -1); s += '…'; }
    txt(s, x + 66, yy + 15, { size, weight: cur.has(i) ? 700 : 500, color: cur.has(i) || chg.has(i) ? C.ink : C.ink2, alpha: a, font: MONO });
  }
  if (start > 0) txt('⋮', x + w / 2, top - 6, { size: 16, color: C.gray, align: 'center', alpha: a });
  if (start + vis < d.code.length) txt(`⋮ 共 ${d.code.length} 行`, x + w - 16, y + h - 8, { size: 14, weight: 600, color: C.gray, align: 'right', alpha: a });
}
// naive_T → naive 的 diff 动画：两条语句放大，轴字母 x / y 从旧值淡变到新值
function diffCard(k, a) {
  if (a <= 0.003) return;
  const [x, y, w, h] = COL.code, size = 27, X0 = x + 30;
  rr(x, y, w, h, '#fbf9f4', 1); sr(x, y, w, h, C.lg, 1.5, 1);
  txt('naive_T → naive：只改两行', x + 18, y + 40, { size: 26, weight: 800, alpha: 1 });
  const rows = [['int row = blockIdx.', 'x', 'y', ' * blockDim.', 'x', 'y', ''], ['          + threadIdx.', 'x', 'y', ';', '', '', ''], ['int col = blockIdx.', 'y', 'x', ' * blockDim.', 'y', 'x', ''], ['          + threadIdx.', 'y', 'x', ';', '', '', '']];
  rows.forEach((r, i) => {
    const yy = y + 150 + i * 62 + (i >= 2 ? 60 : 0); let xx = X0;
    const seg = (s, col, al = 1) => { txt(s, xx, yy, { size, weight: 600, color: col, alpha: al, font: MONO }); };
    for (let q = 0; q < r.length; q += 3) {
      if (!r[q] && !r[q + 1]) break;
      seg(r[q], C.ink2); xx += tw(r[q], size, 600, MONO);
      if (r[q + 1]) {
        const cw = tw('x', size, 600, MONO);
        rr(xx - 2, yy - size + 2, cw + 4, size + 8, mixc('#f4c7c3', '#c8e6d2', a), 1);
        seg(r[q + 1], C.red, 1 - a); seg(r[q + 2], C.use, a); xx += cw;
      }
    }
  });
  txt('同一 warp 的线程 threadIdx.x 连续：', x + 18, y + 560, { size: 22, weight: 700, color: C.gray });
  txt(a < 0.5 ? 'naive_T：相邻线程 → 相邻的行' : 'naive：相邻线程 → 相邻的列', x + 18, y + 600, { size: 24, weight: 800, color: a < 0.5 ? C.red : C.use });
}
// naive / naive_T 的中栏：完整画出缩小的 A、B、C（N = 16），带外框、尺寸与对应关系的引导线；
// 讲一个线程时 A 的整行与 B 的整列贯穿高亮并连到 C 的那一格，讲 warp 时三者同时高亮；地址条注明是哪几行在内存中的样子。
// 几何与讲解页 framesNaive 一致：block 为 C 的第 4–7 行、第 8–11 列；naive 的 warp 为第 5 行的 4 个线程（选中 C[5][10]），naive_T 的 warp 为第 9 列的 4 个线程（选中 C[6][9]）
function naivePanel(vn, f, fi, mode, a, ff = 0) {
  // 中栏：缩小的示意 N = 16，block 取整个 C；一个 warp = lane 0–31（threadIdx.x 0–15，threadIdx.y 0、1），与真实的 16 × 16 block 中一个 warp 的形状相同。
  // naive_T：row = threadIdx.x、col = threadIdx.y → warp 占 C 的两列；读 A[0…15][k]（16 个扇区）与 B[k][0, 1]（1 个扇区）
  // naive：row = threadIdx.y、col = threadIdx.x → warp 占 C 的两行；读 B[k][0…15]（2 个扇区）与 A[0, 1][k]（2 个地址各广播给 16 个线程）
  // 下方内存带按 32 字节扇区画格，读到的元素从矩阵落到扇区里；地址按真实 N = 4096 标注，相隔 16 KB 的扇区之间用折断符号省略
  if (a <= 0.003) return;
  const [px, py, pw, ph] = COL.lay, c = 10, n = 16, Wm = n * c;
  rr(px, py, pw, ph, '#fbf9f4', a); sr(px, py, pw, ph, C.lg, 1.5, a);
  txt('缩小的示意：N = 16（真实 N = 4096），block 取整个 C', px + 16, py + 30, { size: 18, weight: 800, color: C.gray, alpha: a });
  const BX = px + 300, BY = py + 62, AX = px + 110, AY = py + 252, CX = BX, CY = AY;
  const T = vn === 'naive_T', k = fi >= 0 && fi < 16 ? fi : -1, thread = mode === 'thread';
  const me = T ? [5, 0] : [0, 5];                                   // 选中线程 lane 5 在 C 中的 [行, 列]
  const rowsA = thread ? [me[0]] : T ? [...Array(16).keys()] : [0, 1], colsB = thread ? [me[1]] : T ? [0, 1] : [...Array(16).keys()];
  const inWarp = (r, q) => (T ? q <= 1 : r <= 1);
  const cell = (x, y, col) => rr(x + 0.7, y + 0.7, c - 1.4, c - 1.4, col, a);
  for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) {
    const inR = rowsA.includes(r), inC = colsB.includes(q);
    cell(AX + q * c, AY + r * c, inR ? (q === k ? C.hot : q < k ? '#e8a27d' : C.hotL) : '#e8e4da');
    cell(BX + q * c, BY + r * c, inC ? (r === k ? C.hot : r < k ? '#e8a27d' : C.hotL) : '#e8e4da');
    const isMe = r === me[0] && q === me[1];
    cell(CX + q * c, CY + r * c, isMe ? C.thr : !thread && inWarp(r, q) ? C.warp : '#d6e2f7');
  }
  [[AX, AY], [BX, BY], [CX, CY]].forEach(([x, y]) => sr(x, y, Wm, Wm, C.ink2, 1.5, a));
  txt('A：M × K', AX, AY - 10, { size: 18, weight: 800, color: C.hot, alpha: a, font: MONO });
  txt('B：K × N', BX, BY - 10, { size: 18, weight: 800, color: C.hot, alpha: a, font: MONO });
  txt('C：M × N', CX + Wm + 10, CY + Wm, { size: 18, weight: 800, color: C.thr, alpha: a, font: MONO });
  line(AX + Wm, AY, CX, CY, C.lg, 1.2, a, [5, 5]); line(AX + Wm, AY + Wm, CX, CY + Wm, C.lg, 1.2, a, [5, 5]);
  line(BX, BY + Wm, CX, CY, C.lg, 1.2, a, [5, 5]); line(BX + Wm, BY + Wm, CX + Wm, CY, C.lg, 1.2, a, [5, 5]);
  if (k >= 0) txt(`k = ${k}`, BX + Wm, BY - 10, { size: 18, weight: 800, color: C.hot, align: 'right', alpha: a, font: MONO });
  const [mx, my] = [CX + me[1] * c + c / 2, CY + me[0] * c + c / 2];
  line(AX + Wm, AY + me[0] * c + c / 2, mx, my, C.thr, 2, a); line(BX + me[1] * c + c / 2, BY + Wm, mx, my, C.thr, 2, a);
  txt(thread ? `一个线程：C[${me[0]}][${me[1]}] = A 的第 ${me[0]} 行 · B 的第 ${me[1]} 列` : T ? '一个 warp（32 个线程）：C 的两列，相邻线程对应相邻的行' : '一个 warp（32 个线程）：C 的两行，相邻线程对应相邻的列',
    px + 16, py + 446, { size: 18, weight: 800, color: C.thr, alpha: a });
  if (thread || k < 0) { if (k < 0) txt('写回 C：这一步不读 A、B', px + 16, py + 490, { size: 18, weight: 700, color: C.gray, alpha: a }); return; }
  // 内存带：按示意矩阵的比例画整块 A 与 B（N = 16：一行 16 个 float = 64 字节 = 2 个扇区，整块 32 个扇区），每格一个 32 字节扇区；
  // 读到的扇区标橙，读到的元素从矩阵落进对应的扇区
  const fall = ease(clamp(ff * 2.2));
  const counter = T ? 'A 16 次 + B 1 次 = 17 次访存' : 'B 2 次 + A 2 次 = 4 次访存';
  rr(px + 12, py + 458, pw - 24, 40, T ? '#f6d8c4' : '#cfe6d7', a);
  txt(`这一步 k：${counter}`, px + 24, py + 486, { size: 22, weight: 800, color: T ? C.hot : C.use, alpha: a });
  const drops = [], x0 = px + 16, SW = (pw - 32) / 32, half = k >= 8 ? 1 : 0;
  const band = (y, label, lit) => {
    txt(label, x0, y - 8, { size: 17, weight: 800, color: C.ink, alpha: a });
    for (let i = 0; i < 32; i++) { rr(x0 + i * SW + 0.5, y, SW - 1, 34, lit.includes(i) ? C.hot : '#e8e4da', lit.includes(i) ? a * (0.25 + 0.75 * fall) : a); }
    for (let r = 0; r <= 16; r++) line(x0 + r * 2 * SW, y - 3, x0 + r * 2 * SW, y + 37, C.ink2, r % 4 ? 1 : 2, a);
    for (let r = 0; r < 16; r++) txt(String(r), x0 + (r + 0.5) * 2 * SW, y + 52, { size: 12, weight: 600, color: C.gray, align: 'center', alpha: a, font: MONO });
  };
  const AYb = py + 532, BYb = py + 620;
  const litA = T ? [...Array(16).keys()].map((r) => r * 2 + half) : [half, 2 + half];
  const litB = T ? [k * 2] : [k * 2, k * 2 + 1];
  band(AYb, 'A 在显存中（行主序，每格一个 32 字节扇区，格下为行号）', litA);
  band(BYb, 'B 在显存中', litB);
  txt(T ? 'A：每隔一格读一格，16 次；B：1 次' : 'A：第 0、1 行各 1 次；B：第 k 行连续 2 格，2 次', x0, py + 698, { size: 17, weight: 800, color: T ? C.hot : C.use, alpha: a });
  txt('真实 N = 4096 时，一行 = 16 KB，相邻两行相隔 16 KB', x0, py + 720, { size: 15, weight: 600, color: C.gray, alpha: a });
  (T ? [...Array(16).keys()] : [0, 1]).forEach((r) => drops.push([AX + k * c + c / 2, AY + r * c + c / 2, x0 + (r * 2 + half + 0.5) * SW, AYb + 17]));
  (T ? [0, 1] : [...Array(16).keys()]).forEach((q) => drops.push([BX + q * c + c / 2, BY + k * c + c / 2, x0 + (k * 2 + (q >= 8 ? 1 : 0) + 0.5) * SW, BYb + 17]));
  if (fall < 1) drops.forEach(([x0, y0, x1, y1]) => rr(mix(x0, x1, fall) - 4, mix(y0, y1, fall) - 4, 8, 8, C.hot, a * (1 - fall * 0.3)));
}
// ---------- 中：数据布局（讲解页 #lay 的版式，按 1.4 倍画在画布上） ----------
const CELLC = { blk: '#d6e2f7', warp: C.warp, thr: C.thr, load: C.hot, use: C.use };
function gridL(X, Y, s, x, y, rows, cols, cs, list, a) {
  const st = {}; const ord = { blk: 0, warp: 1, use: 2, load: 3, thr: 4 };
  (list || []).slice().sort((p, q) => ord[p[2]] - ord[q[2]]).forEach(([r, c, k]) => { st[r * cols + c] = k; });
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const k = st[r * cols + c];
    rr(X(x + c * cs), Y(y + r * cs), (cs - 1.5) * s, (cs - 1.5) * s, k ? CELLC[k] : '#e8e4da', a);
    if (k === 'blk') sr(X(x + c * cs), Y(y + r * cs), (cs - 1.5) * s, (cs - 1.5) * s, C.thr, 0.6, a);
  }
}
function layPanel(vn, f, a, fi = -1, mode = null, ff = 0) {
  if (a <= 0.003) return;
  if (vn === 'naive' || vn === 'naive_T') return naivePanel(vn, f, fi, mode, a, ff);
  const d = FR[vn], [px, py, pw, ph] = COL.lay, s = 1.34, ox = px + 4, oy = py + 8;
  const X = (v) => ox + v * s, Y = (v) => oy + v * s;
  rr(px, py, pw, ph, '#fbf9f4', a); sr(px, py, pw, ph, C.lg, 1.5, a);
  const T = (x, y, str, sub = false) => txt(str, X(x), Y(y), { size: sub ? 15.5 : 16.5, weight: sub ? 500 : 600, color: sub ? C.gray : C.ink, alpha: a, font: sub ? ZH : MONO });
  const tc = d.line === 'tc', N = 16, CS = 11;
  gridL(X, Y, s, 214, 22, N, N, CS, f ? f.B : [], a); T(214, 16, tc ? 'Wᵀ（W 是 N×K，沿 K 连续）' : 'B');
  gridL(X, Y, s, 30, 214, N, N, CS, f ? f.A : [], a); T(30, 208, 'A');
  gridL(X, Y, s, 214, 214, N, N, CS, f ? f.C : [], a); T(214, 214 + N * CS + 16, tc ? 'C = A · Wᵀ' : 'C = A × B');
  const ex = (k) => {
    let l = f ? f.ex[k] : [];
    if (l === undefined && vn === 'reg8x8' && f) l = k === 'rA' ? (f.reg ? [[0, 0, 'use'], [1, 0, 'use']] : []) : k === 'rB' ? (f.reg ? [[0, 0, 'use'], [0, 1, 'use']] : []) : k === 'acc' ? [[0, 0, 'thr'], [0, 1, 'thr'], [1, 0, 'thr'], [1, 1, 'thr']] : [];
    return l || [];
  };
  const G = (k, x, y, r, c, cs) => gridL(X, Y, s, x, y, r, c, cs, ex(k), a);
  if (vn === 'smem32') { T(30, 18, '共享内存', true); G('As', 30, 40, 4, 4, 15); T(30, 34, 'As 4×4'); G('Bs', 118, 40, 4, 4, 15); T(118, 34, 'Bs 4×4'); T(30, 128, '寄存器：sum（每线程 1 个）', true); }
  else if (vn === 'reg8x8') { T(30, 18, '共享内存', true); G('As', 30, 40, 8, 2, 12); T(30, 34, 'As 8×2'); G('Bs', 80, 40, 2, 8, 12); T(80, 34, 'Bs 2×8'); T(80, 84, '选中线程的寄存器', true); G('rA', 80, 92, 2, 1, 14); T(98, 104, 'rA', true); G('rB', 120, 92, 1, 2, 14); T(152, 104, 'rB', true); G('acc', 120, 112, 2, 2, 14); T(152, 128, 'acc 2×2', true); }
  else if (vn === 'vec') { T(30, 18, '共享内存（A 块转置存放）', true); G('As', 30, 40, 4, 8, 12); T(30, 34, 'As[k][m] 4×8'); G('Bs', 30, 110, 4, 8, 12); T(30, 104, 'Bs[k][n] 4×8'); T(140, 18, '寄存器', true); G('rA', 140, 40, 1, 4, 12); T(140, 34, 'rA'); G('rB', 140, 72, 1, 4, 12); T(140, 66, 'rB'); G('acc', 140, 104, 4, 4, 12); T(140, 98, 'acc 4×4'); }
  else if (vn === 'warptile') { T(20, 18, '共享内存（示意 BK = 2）', true); G('As', 20, 40, 2, 16, 10); T(20, 34, 'As[k][m] 2×16'); G('Bs', 20, 92, 2, 16, 10); T(20, 86, 'Bs[k][n] 2×16'); T(20, 140, '深蓝 = 选中线程读的两段', true); }
  else if (vn === 'dbuf') { T(20, 22, '缓冲 0'); G('As0', 20, 28, 2, 16, 10); G('Bs0', 20, 52, 2, 16, 10); T(20, 96, '缓冲 1'); G('As1', 20, 102, 2, 16, 10); G('Bs1', 20, 126, 2, 16, 10); T(20, 168, '寄存器 pa / pb（预取）', true); G('pa', 20, 174, 1, 8, 10); }
  else if (vn === 'wgmma_ws') { T(20, 22, '共享内存环形缓冲（4 个槽）'); G('slots', 20, 34, 1, 4, 42); [0, 1, 2, 3].forEach((i) => T(30 + i * 42, 92, '槽 ' + i, true)); T(20, 122, '生产者：1 个线程发 TMA → 写槽', true); T(20, 140, '消费者 ×2：wgmma ← 读槽', true); T(20, 158, '橙 搬入 · 浅蓝 已满 · 绿 正在算', true); }
  else if (vn === 'wgmma_l2') { T(20, 40, 'L2 占用（示意，一格 = 20）'); G('l2', 20, 50, 1, 15, 12); line(X(139), Y(44), X(139), Y(68), C.gray, 1.5, a); T(110, 84, 'L2 容量', true); T(20, 112, '绿 放得下 · 橙 溢出，要回显存', true); }
  else { T(30, 40, '没有共享内存', true); T(30, 58, '每个线程直接从显存读', true); T(30, 76, '寄存器：sum（1 个）', true); }
  // 地址条与计数
  const sp = f && f.strip;
  if (!sp) T(8, 432, '内存地址条：这一帧不读显存，数据都在片上', true);
  else {
    T(8, 432, sp.label); const w = 384 / sp.n;
    for (let i = 0; i < sp.n; i++) rr(X(8 + i * w), Y(440), (w - (sp.n > 32 ? 0.8 : 1.5)) * s, 16 * s, sp.hits.includes(i) ? C.hot : '#e8e4da', a);
    T(8, 472, `地址 ${sp.base} 起 · ${sp.kind}`, true);
  }
  if (vn === 'reg8x8' && fi >= 0 && fi <= 2) {                    // 为什么是正方形：同样 16 格的三种形状
    const bx = px + 300, by = py + 470, rows = [[1, 16], [2, 8], [4, 4]];
    rr(bx - 12, by - 36, 286, 236, '#fff', a);
    txt('一个线程的小块（16 格）', bx, by - 10, { size: 16, weight: 700, color: C.gray, alpha: a });
    txt('形状   读几个数   每读 1 个做几次', bx, by + 18, { size: 15, weight: 700, color: C.gray, alpha: a });
    rows.forEach(([h_, w_], i) => { const on = i === fi, al = a * (i <= fi ? 1 : 0.3); if (on) rr(bx - 6, by + 28 + i * 34, 274, 30, '#f6d8c4', a);
      txt(`${h_} × ${w_}`, bx, by + 50 + i * 34, { size: 19, weight: 800, alpha: al, font: MONO }); txt(String(h_ + w_), bx + 96, by + 50 + i * 34, { size: 19, weight: 800, color: C.hot, alpha: al, font: MONO }); txt((16 / (h_ + w_)).toFixed(2), bx + 196, by + 50 + i * 34, { size: 19, weight: 800, color: C.use, alpha: al, font: MONO }); });
    txt('真实 8 × 8：读 16 个做 64 次（4 次）', bx, by + 152, { size: 15, weight: 700, color: C.ink2, alpha: a * (fi === 2 ? 1 : 0) });
    txt('1 × 64：读 65 个（0.98 次）', bx, by + 176, { size: 15, weight: 700, color: C.ink2, alpha: a * (fi === 2 ? 1 : 0) });
  }
  if (f && f.cnt) { const c = f.cnt; txt(`这个 block 累计从显存读 ${c.loads} 个数，做 ${c.fma} 次乘加`, px + 14, py + ph - 46, { size: 18, weight: 700, color: C.ink2, alpha: a }); txt(`每读 1 个数做 ${c.loads ? (c.fma / c.loads).toFixed(2) : 0} 次乘加`, px + 14, py + ph - 18, { size: 18, weight: 800, color: C.ink, alpha: a }); }
}
// 没有逐帧剧本的版本：中栏画专用的示意图（与上一稿同一套图元）
const MAPS = {
  tuned: (x, y, w, h, a) => tl(x + 20, y + 20, [['autotune：9 组 <BM, BN, BK, WM, WN>', 22, C.ink, 800], ['每个规模选最快的一组', 20, C.ink2, 700], ['8192 阶：dbuf 44.30 → 44.53', 22, C.blue, 800, MONO]], a),
  wmma: (x, y, w, h, a, u, t0) => { perClock(x + 20, y + 10, a * (1 - po(u, t0 + 2.6, t0 + 3.2))); if (u > t0 + 2.6) wmmaMap(x, y, w, h, po(u, t0 + 2.6, t0 + 3.2) * a); },
  mma: (x, y, w, h, a) => { pathRow(x + 10, y + 26, [['显存', C.hot, 'cp.async'], ['共享内存', C.use, 'ldmatrix'], ['寄存器', C.use, 'mma.sync'], ['Tensor', C.teal]], a, 100, 44); swzPanels(x + 20, y + 140, a, 25); },
  mma_async: ringMap,
  wgmma: tmaMap,
  wgmma_pipe: (x, y, w, h, a) => { const c1 = [], c2 = []; for (let i = 0; i < 8; i++) { c1.push([1 + i * 1.9, 2.8 + i * 1.9, C.teal, '计算']); c2.push([1.9 + i * 1.9, 3.7 + i * 1.9, C.teal, '计算']); } lane(x + 130, y + 40, 400, 34, 16, c1, 1, a, '消费者 1', C.tealD); lane(x + 130, y + 84, 400, 34, 16, c2, 1, a, '消费者 2', C.tealD); txt('两个消费者交替，Tensor Core 无空档', x + 130, y + 160, { size: 19, weight: 800, color: C.use, alpha: a }); },
  wgmma_persist: (x, y, w, h, a, u, t0) => { tiles(x + 60, y + 34, a, true, 264); const k = Math.floor(Math.max(0, u - t0) / 0.8) % 3; txt('CTA 0 依次领取 tile 0、132、264', x + 60, y + 340, { size: 18, weight: 700, color: C.thr, alpha: a }); const id = [0, 132, 264][k], m = id % 16, q = Math.floor(id / 16); sr(x + 60 + q * 13 - 2, y + 34 + m * 4.6 - 2, 15, 8, C.thr, 3, a); },
  wgmma_cluster: clusterMap,
  wgmma_tstore: (x, y, w, h, a, u, t0) => { const rv = lin(u, t0, t0 + 3); lane(x + 140, y + 30, 390, 40, 12, [[0, 4, C.teal, '计算'], [4.3, 8.3, C.teal, '计算'], [8.6, 12, C.teal, '计算']], rv, a, 'Tensor Core', C.tealD); lane(x + 140, y + 80, 390, 30, 12, [[4.3, 5.5, C.use, 'TMA'], [8.6, 9.8, C.use, 'TMA']], rv, a, 'TMA 写回', C.use); txt('流水线 4 → 3 级，腾出 64 KB 存放结果', x + 20, y + 170, { size: 19, weight: 700, color: C.ink2, alpha: a }); },
};
function mapPanel(vn, a, u, t0) {
  if (a <= 0.003) return;
  const [px, py, pw, ph] = COL.lay;
  rr(px, py, pw, ph, '#fbf9f4', a); sr(px, py, pw, ph, C.lg, 1.5, a);
  if (MAPS[vn]) MAPS[vn](px + 12, py + 40, pw - 24, ph - 60, a, u, t0);
}
// 讲解页 hwState 的说明改为书面语体（视频画面标注的语体要求），并对齐文章的推算数字；按原文开头匹配
const NOTE_FIX = [
  ['一个 warp 读 A 涉及', '一个 warp 读 A 涉及 16 个不同的行，LSU 拆成 16 次访存。\n多数 warp 等待数据，FP32 单元几乎空闲。'],
  ['warp 读 B 的 16 个连续', 'warp 读 B 的 16 个连续 float，合并为 2 个扇区；A 的两个地址各广播给 16 个线程。\n每次乘加仍从 L1/L2 读取 2 个数，warp 多数时间在等待。'],
  ['每次 FMA 读 2 个共享内存数', '每次乘加读取 2 个共享内存数：SM 每时钟只供给 32 个 float，却能做 128 次乘加。\n瓶颈由显存转移到共享内存带宽。'],
  ['搬块：', '搬块：显存 → 寄存器 → 共享内存，随后 __syncthreads。\n此时 FP32 单元空闲，warp 等待数据或同步。'],
  ['写回 C', '写回 C：寄存器 → 显存。'],
  ['读 16 个数做 64 次 FMA', '读取 16 个数完成 64 次乘加，共享内存不再是瓶颈。\n16 条标量 LDS 占用发射槽，读 B 仍有 bank 冲突。'],
  ['LDS.128 一次读 4 个数', 'LDS.128 一次读取 4 个数：每步访存指令由 16 条减为 4 条。\n剩余：一个 warp 内 2 路 bank 冲突（推算）。'],
  ['block → warp → 线程', 'block → warp → 线程：一个 warp 负责一块，线程的小块彼此相隔。\n一个 warp 读共享内存的地址因此首尾相连。'],
  ['一个 warp 读一段连续的共享内存', '一个 warp 读取一段连续的共享内存：32 个 bank 各一次，无冲突。\n剩余：每块“读显存 → 等待 → 计算”，延迟未被掩盖。'],
  ['同一时刻：下一块的读', '下一块的读取已经发出，FP32 单元同时计算本块。\n显存延迟被计算掩盖；ncu：FMA 管线 68.3%（cuBLAS 79.8%）。'],
  ['把寄存器里预取的下一块', '寄存器中预取的下一块写入另一份共享内存，每块只同步一次。'],
  ['调参只多 0.5%', '调参只多 0.5%：FP32 单元的峰值约 67 TFLOPS。\n继续提高须更换计算单元。'],
  ['计算换到 Tensor Core', '计算换到 Tensor Core；片段从未换位的共享内存读取，bank 冲突严重。\n搬运经过寄存器，搬完才计算，Tensor Core 多数时间等待。'],
  ['cp.async 直接 L2', 'cp.async 由 L2 直接写入共享内存；异或换位后 ldmatrix 无冲突。\n剩余：每块搬完 → 等齐 → 计算，搬运延迟全部暴露。'],
  ['3 级环形缓冲', '三级环形缓冲：搬运第 t+2 块与计算第 t 块重叠。\n剩余：操作数须先由 ldmatrix 读入寄存器，Tensor Core 未被用满。'],
  ['TMA 一条指令搬整块', 'TMA 一条指令搬运整块，wgmma 直接从共享内存读取 A 与 W。\n剩余：每块计算完须等齐并同步，搬运与计算相互等待。'],
  ['1 个生产者 warpgroup', (n) => '1 个生产者 warpgroup（1 个线程发出 TMA）与 2 个消费者 warpgroup 执行 wgmma。\n' + (/开场/.test(n) ? '开始时，消费者在 mbarrier 上等待第一个槽位填满。' : '搬运与计算同时进行：TMA 写入一个槽，Tensor Core 读取另一个槽。')],
  ['分组发 tile', '分组发放：同时使用的 A、W 放得进 L2，读取多在 L2 命中。\n显存流量下降，Tensor Core 持续工作。'],
  ['按行发 tile', '按行发放：同时运行的 CTA 需要整个 W（128 MB），L2 放不下。\n每一波都从显存重读，显存通路饱和，Tensor Core 等待。'],
  ['消费者只等上一批', '消费者只等待上一批 wgmma 即归还槽位，速度几乎不变。\n两个消费者 warpgroup 已互相填补空档。'],
  ['常驻 kernel', '常驻 kernel：132 个 CTA 循环领取 tile，写回上一块时生产者已搬运下一块。\n单独加上慢约 10%，原因未查；集群与 TMA 写回以其为前提。'],
  ['相邻两个 CTA 组成集群', '相邻两个 CTA 组成集群，W 块各读一半，TMA 多播给双方。\n每个 SM 从 L2 读取的 W 减半；L2 流量约 8.7 → 5.9 TB/s（推算）。'],
  ['结果先写共享内存', '结果先写入共享内存，1 个线程发出 TMA 整块写回，消费者随即计算下一块。\nncu：Tensor 管线 92.9%（cuBLAS 97.0%）。'],
];
const fixNote = (n) => { if (!n) return ''; for (const [k, r] of NOTE_FIX) if (n.startsWith(k)) return typeof r === 'function' ? r(n) : r; return n; };
// ---------- 右：SM 剖面（讲解页 HW 面板的版式与 hwState 的状态），按 1.38 倍 ----------
function hwPanel(st, a, u) {
  if (a <= 0.003) return;
  const [px, py, pw, ph] = COL.hw;
  rr(px, py, pw, ph, '#fbf9f4', a); sr(px, py, pw, ph, C.lg, 1.5, a);
  txt('一个 SM 的剖面', px + 16, py + 30, { size: 18, weight: 700, color: C.gray, alpha: a });
  smPanel(px + 6, py + 44, 1.28, { ...st, note: null }, a, u);
  const notes = fixNote(st.note).split('\n').filter(Boolean);
  notes.forEach((s, i) => wrapTxt(s, px + 18, py + 470 + i * (st.focus ? 96 : 80), pw - 36, st.focus ? 25 : 22, i ? C.ink2 : (st.hot ? C.hot : C.use), a));
  const lg = [['执行', C.use], ['等显存', C.hot], ['等同步', C.gray], ['生产者', C.thr]];
  let lx = px + 18; lg.forEach(([s, col]) => { g.globalAlpha = a; g.fillStyle = col; g.beginPath(); g.arc(lx + 7, py + ph - 24, 7, 0, 7); g.fill(); g.globalAlpha = 1; txt(s, lx + 20, py + ph - 17, { size: 17, weight: 600, color: C.ink2, alpha: a }); lx += tw(s, 17, 600) + 48; });
}
function wrapTxt(s, x, y, w, size, col, a) {
  // 按词换行：连续的英文、数字与符号算一个词，中文逐字
  const toks = s.match(/[A-Za-z0-9_.\-/→×≈%()（）]+|./g) || [];
  let line_ = '', yy = y;
  for (const t of toks) { if (line_ && tw(line_ + t, size, 700) > w) { txt(line_, x, yy, { size, weight: 700, color: col, alpha: a }); line_ = t.trimStart(); yy += size * 1.4; } else line_ += t; }
  if (line_) txt(line_, x, yy, { size, weight: 700, color: col, alpha: a });
}
// ---------- 底部：成绩阶梯（当前版本点亮） ----------
function bottomLadder(vn, a) {
  const i = vn === 'naive_T' ? -1 : STEP_OF[vn === 'tuned' ? 'dbuf' : vn];
  rr(40, 878, 1840, 112, '#fbf9f4', a);
  if (vn === 'naive_T' || vn === 'naive') txt(vn === 'naive_T' ? 'naive_T（对照组）：0.96 TFLOPS（4096 阶）' : '4096 阶：naive_T 0.96 → naive 5.39，5.6 倍', 1860, 906, { size: 20, weight: 800, color: vn === 'naive_T' ? C.red : C.blue, align: 'right', alpha: a });
  txt('8192 阶 · TFLOPS', 56, 902, { size: 16, weight: 700, color: C.gray, alpha: a });
  ladder(250, 886, 1600, 96, { upto: (i ?? -1) + 1, ghost: 0.5, alpha: a, values: 'last', axis: false, size: 0.75 });
}
// 赛跑条：naive_T 与 naive 跑同一个 4096 阶矩阵乘法，按实测用时等比（143 ms 对 25.5 ms，约 5.6 倍）
function raceBars(u, t0) {
  rr(40, 878, 1840, 112, '#fbf9f4', 1);
  const LT = 4.6, X = 420, Wd = 1240, k = lin(u, t0 + 0.3, t0 + 0.3 + LT), kN = lin(u, t0 + 0.3, t0 + 0.3 + LT * 25.5 / 143.2);
  txt('4096 阶', 56, 918, { size: 18, weight: 800, color: C.gray }); txt('按实测用时等比', 56, 944, { size: 16, weight: 700, color: C.gray });
  // 统一时间轴：条长 = 用时；naive 在 25.5 ms 处停住，naive_T 跑到 143 ms
  [['naive_T', k, C.red, '143 ms', 902, 1], ['naive', kN, C.blue, '25.5 ms', 946, 25.5 / 143.2]].forEach(([nm, f, col, ms, y, len]) => {
    txt(nm, X - 16, y + 22, { size: 22, weight: 800, color: col, align: 'right', font: MONO });
    rr(X, y, Wd, 30, '#e8e4da'); rr(X, y, Wd * len * f, 30, col);
    if (f >= 1) txt(ms, X + Wd * len + 12, y + 24, { size: 22, weight: 800, color: col, font: MONO });
  });
  if (k >= 1) txt('5.6 倍', 1860, 948, { size: 30, weight: 800, color: C.ink, align: 'right' });
}
// ---------- 整幕结束时的完整阶梯 ----------
function actLadder(upto, a, u, t0) {
  if (a <= 0.003) return;
  rr(40, 140, 1840, 850, C.paper, a);
  const up = Math.min(upto, 1 + (upto - 1) * lin(u, t0, t0 + 1.8));
  ladder(200, 220, 1600, 520, { upto: up, ghost: 0.55, alpha: a, names: true, values: 'all', pct: true, cub: [a, upto > 7 ? a : 0.3 * a], cubLabel: false, size: 0.95 });
}
// 三栏幕：画面全部由 script.js 的节拍（x）驱动
function stage(k) {
  if (k < 0) return null;
  return (u) => {
    const st = vstate(k, u), vn = st.x.v, d = FR[vn], fade = po(u, st.vStart, st.vStart + 0.6);
    const endA = st.x.end ? po(u, st.t0, st.t0 + 0.6) : 0;
    const xj = SCRIPT[k].xs[st.j] || {};
    if (xj.diff) diffCard(k, p(u, st.t0 + 0.8, st.t1 - 1.2));
    else { if (st.pv && fade < 1) codePanel(st.pv, null, 1 - fade, u); codePanel(vn, st.f, st.pv ? fade : 1, u, xj.hl || null); }
    if (d && d.frames.length) layPanel(vn, st.f, 1, st.fi, xj.mode, st.ff); else mapPanel(vn, 1, u, st.vStart);
    const hw = st.f ? st.f.hw : d.hw0;
    hwPanel(vn === 'naive' || vn === 'naive_T' ? { ...hw, focus: ['lsu', 'smem', 'l2', 'hbm'] } : hw, 1, u);   // 合并访存一幕只保留与访存有关的单元
    if (xj.race) raceBars(u, st.t0); else bottomLadder(vn, 1);
    if (endA > 0) actLadder(st.x.end, endA, u, st.t0);
  };
}

// ================= 20 各规模 =================
// 原样输出（节选）：ladder-final-20261021-121144.txt
const BYN = [[256, 2.57, 2.55], [512, 16.64, 20.46], [1024, 106.69, 136.12], [2048, 520.22, 483.23], [4096, 726.48, 743.33], [8192, 777.63, 781.86]];
const OUT_SOAK = ['$ /build/matmul_tc soak cublas,wgmma_l2,wgmma_cluster,wgmma_tstore 4096,8192', '⋮',
  'cublas_bf16          8192  1245 次  平均 1.614 ms   681.14 TFLOPS  SM   1368 MHz  功耗   697 W  (245 个采样)',
  'wgmma_tstore         8192  1250 次  平均 1.611 ms   682.34 TFLOPS  SM   1360 MHz  功耗   699 W  (246 个采样)'];
function s20(u) {
  const K = IDX('各规模'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5), X = 200, Y = 300, H = 420, Wd = 960, Yv = (v) => Y + H - H * Math.log10(v) / 3;
  const a = b(0);
  [1, 10, 100, 1000].forEach((v) => { line(X, Yv(v), X + Wd, Yv(v), C.lg, 1.5, a); txt(String(v), X - 12, Yv(v) + 7, { size: 18, weight: 600, color: C.gray, align: 'right', alpha: a, font: MONO }); });
  txt('TFLOPS（对数刻度）', X, Y - 20, { size: 20, weight: 700, color: C.gray, alpha: a });
  BYN.forEach(([n, m, c], i) => {
    const gx = X + 20 + i * 158, gr = po(u, bt(K, 0) + 0.2 + i * 0.15, bt(K, 0) + 1.0 + i * 0.15), r = m / c * 100;
    const hi = (i === 1 || i === 2) ? b(1) : i === 3 ? b(2) : i === 5 ? b(3) : 0;
    rr(gx, Yv(1 + (m - 1) * gr), 56, Y + H - Yv(1 + (m - 1) * gr), C.teal, a); rr(gx + 62, Yv(1 + (c - 1) * gr), 56, Y + H - Yv(1 + (c - 1) * gr), C.ink2, a);
    txt(String(n), gx + 59, Y + H + 30, { size: 22, weight: 700, color: C.ink, align: 'center', alpha: a, font: MONO });
    const col = r < 95 ? C.red : r > 101 ? C.tealD : C.ink;
    txt((r >= 99 && r < 100 ? r.toFixed(1) : Math.round(r)) + '%', gx + 59, Yv(Math.max(m, c)) - 14, { size: 22, weight: 800, color: col, align: 'center', alpha: a * gr, font: MONO });
    if (hi > 0) sr(gx - 8, Yv(Math.max(m, c)) - 44, 134, Y + H - Yv(Math.max(m, c)) + 52, col, 3, hi);
  });
  rr(X, Y + H + 48, 22, 22, C.teal, a); txt('最终版（wgmma_tstore）', X + 30, Y + H + 66, { size: 20, weight: 700, alpha: a });
  rr(X + 300, Y + H + 48, 22, 22, C.ink2, a); txt('cuBLAS BF16（同布局）', X + 330, Y + H + 66, { size: 20, weight: 700, alpha: a });
  const RX = 1230;
  tl(RX, 280, [['256 阶：两者都约 0.013 ms', 20, C.gray, 700]], a);
  tl(RX, 330, [['512、1024 阶落后', 26, C.red, 800], ['1024 阶只有 32 个 tile（128 × 256）', 20, C.ink2, 700], ['132 个 SM 中 100 个空闲', 20, C.ink2, 700]], b(1));
  tl(RX, 490, [['2048 阶 +7.7%', 26, C.tealD, 800], ['520.2 对 483.2', 20, C.ink2, 700, MONO]], b(2));
  tl(RX, 600, [['8192 阶：短时 99.5%', 26, C.ink, 800], ['连续 2 秒：682.3 对 681.1', 22, C.ink, 800, MONO], ['两者都在 700 W 功耗上限', 22, C.red, 800]], b(3));
  term(96, 836, 1728, OUT_SOAK, { size: 16, lh: 30, hl: (i) => (i >= 2 ? b(3) : 0), alpha: b(3) });
}

// ================= 21 未完成 =================
function s21(u) {
  const K = IDX('未完成'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  rr(96, 270, 1728, 64, C.paper2, b(0));
  txt('没有做的：', 116, 312, { size: 24, weight: 800, color: C.ink, alpha: b(0) });
  txt('Hilbert 曲线调度 tile（H100 worklog 中约 +1%） · FP32 线剩余的 14% · 小矩阵的分块与 split-K', 250, 312, { size: 22, weight: 700, color: C.ink2, alpha: b(0) });
  const up = 1 + 16 * lin(u, bt(K, 1) + 0.2, bt(K, 2) - 0.4), la = b(1);
  ladder(220, 400, 1580, 420, { upto: Math.min(17, up), ghost: 0.6 * la, alpha: la, names: true, values: u > bt(K, 3) ? 'ends' : 'none', ann: (i) => clamp(up - i), band: b(2), cub: [b(3), b(3)], cubLabel: false, size: 0.95 });
  const c = b(2); let x = 220;
  CATS.forEach(([s, col]) => { rr(x, 952, 24, 24, col, c); txt(s, x + 32, 972, { size: 22, weight: 700, color: col, alpha: c }); x += tw(s, 22, 700) + 80; });
  const d = b(3); line(x, 964, x + 56, 964, C.ink, 2.5, d, [10, 7]); txt('cuBLAS', x + 70, 972, { size: 22, weight: 700, color: C.ink, alpha: d });
}



// ================= v5：嵌套方块 + 仪表 + 四拍台阶 =================
// 嵌套方块：世界坐标中 C 为单位正方形，按 8 × 8 分成 block 的块；选中的 block 块内按 16 × 16 分给线程，选中线程的块再画成 8 × 8 格。
// 镜头的层级 z：0 = 整个 C，1 = 一个 block 的块，2 = 一个线程的块；z 可以是小数（推镜头）。
const NB = { box: [96, 268, 700, 700], blk: [3, 2], thr: [6, 5] };
const nbRect = (z) => {
  const bx = NB.blk[0] / 8, by = NB.blk[1] / 8, tx = bx + NB.thr[0] / 128, ty = by + NB.thr[1] / 128;
  const R = [[0.5, 0.5, 1.08], [bx + 1 / 16, by + 1 / 16, 0.125 * 1.5], [tx + 1 / 256, ty + 1 / 256, (1 / 128) * 2.6]];
  const i = Math.min(1, Math.floor(z)), f = z - i, A = R[i], B_ = R[Math.min(2, i + 1)];
  return [mix(A[0], B_[0], f), mix(A[1], B_[1], f), Math.exp(mix(Math.log(A[2]), Math.log(B_[2]), f))];
};
function nested(z, a, { lab = ['C：8192 × 8192', 'block 的块', '线程的块：8 × 8'], blkOn = 0, thrOn = 0, cells = 0, hotLayer = -1, tag = true } = {}) {
  if (a <= 0.003) return;
  const [X, Y, S] = NB.box, [cx, cy, sz] = nbRect(z), P = (wx, wy) => [X + (wx - cx + sz / 2) / sz * S, Y + (wy - cy + sz / 2) / sz * S], L = (w) => w / sz * S;
  g.save(); g.beginPath(); g.rect(X, Y, S, S); g.clip();
  rr(X, Y, S, S, '#fbf9f4', a);
  const [c0x, c0y] = P(0, 0); rr(c0x, c0y, L(1), L(1), '#e8e4da', a);
  for (let i = 1; i < 8; i++) { const [lx, ly] = P(i / 8, 0), [mx, my] = P(0, i / 8); line(lx, ly, lx, ly + L(1), '#cfc8b8', 1.2, a); line(mx, my, mx + L(1), my, '#cfc8b8', 1.2, a); }
  sr(c0x, c0y, L(1), L(1), C.ink2, 2.5, a);
  const bx = NB.blk[0] / 8, by = NB.blk[1] / 8, [bX, bY] = P(bx, by);
  if (blkOn > 0) {
    rr(bX, bY, L(0.125), L(0.125), '#cfe3d6', a * blkOn);
    const tl_ = clamp((z - 0.4) / 0.6);
    for (let i = 1; i < 16; i++) { const [lx, ly] = P(bx + i / 128, by), [mx, my] = P(bx, by + i / 128); line(lx, ly, lx, ly + L(0.125), '#a9cdb6', 1, a * blkOn * tl_); line(mx, my, mx + L(0.125), my, '#a9cdb6', 1, a * blkOn * tl_); }
    sr(bX, bY, L(0.125), L(0.125), C.use, 4, a * blkOn);
  }
  if (thrOn > 0) {
    const tx = bx + NB.thr[0] / 128, ty = by + NB.thr[1] / 128, [tX, tY] = P(tx, ty), w = L(1 / 128);
    rr(tX, tY, w, w, '#9db7ea', a * thrOn);
    if (cells > 0) for (let i = 1; i < 8; i++) { line(tX + i * w / 8, tY, tX + i * w / 8, tY + w, '#fbf9f4', 1.5, a * cells); line(tX, tY + i * w / 8, tX + w, tY + i * w / 8, '#fbf9f4', 1.5, a * cells); }
    sr(tX, tY, w, w, C.thr, 4, a * thrOn);
  }
  g.restore();
  sr(X, Y, S, S, C.lg, 1.5, a);
  // 当前层级的标注（贴在框的左上角）
  const lv = Math.round(z), names = [['显存', C.ink2], ['共享内存', C.use], ['寄存器', C.thr]];
  rr(X + 12, Y + 12, tw(lab[lv], 24, 800) + (tag ? 150 : 28), 44, '#fbf9f4', a * 0.92);
  txt(lab[lv], X + 24, Y + 44, { size: 24, weight: 800, color: names[lv][1], alpha: a });
  if (tag) txt(names[lv][0], X + 36 + tw(lab[lv], 24, 800), Y + 44, { size: 20, weight: 700, color: C.gray, alpha: a });
}
// 存储层次与瓶颈标记：hot 为小数时标记在两层之间移动
const LAYERS = [['显存', C.hot], ['共享内存', C.use], ['寄存器', C.thr], ['指令 / 单元', C.ink2]];
function layerBar(hot, a) {
  if (a <= 0.003) return;
  const X = 812, Y = 300;
  txt('瓶颈在哪一层', X, Y - 14, { size: 18, weight: 700, color: C.gray, alpha: a });
  LAYERS.forEach(([s, col], i) => { rr(X, Y + i * 74, 130, 56, '#fbf9f4', a); sr(X, Y + i * 74, 130, 56, col, 2, a); txt(s, X + 65, Y + i * 74 + 36, { size: 20, weight: 800, color: col, align: 'center', alpha: a }); });
  if (hot >= 0) { const yy = Y + hot * 74; sr(X - 6, yy - 6, 142, 68, C.hot, 4, a); txt('◀', X + 150, yy + 37, { size: 24, weight: 800, color: C.hot, alpha: a }); }
}
// 仪表：从显存读取的算术强度（FLOP/B），对数刻度 0.1 … 1000；pts = [[值, 名字, 颜色, 不透明度]]
function gauge(pts, a, { line300 = 0 } = {}) {
  if (a <= 0.003) return;
  const X = 1660, Y0 = 300, H_ = 600, Yv = (v) => Y0 + H_ - (Math.log10(v) + 1) / 4 * H_;
  txt('算术强度', X - 10, Y0 - 40, { size: 20, weight: 800, alpha: a }); txt('FLOP / B（显存）', X - 10, Y0 - 14, { size: 16, weight: 600, color: C.gray, alpha: a });
  rr(X + 40, Y0, 18, H_, '#e8e4da', a);
  [0.1, 1, 10, 100, 1000].forEach((v) => { line(X + 34, Yv(v), X + 64, Yv(v), C.gray, 1.5, a); txt(String(v), X + 30, Yv(v) + 6, { size: 15, weight: 600, color: C.gray, align: 'right', alpha: a, font: MONO }); });
  line(X + 20, Yv(20), X + 164, Yv(20), C.blue, 3, a, [8, 5]); txt('FP32 要求 20', X + 70, Yv(20) + 22, { size: 17, weight: 800, color: C.blue, alpha: a });
  if (line300 > 0) { line(X + 20, Yv(295), X + 164, Yv(295), C.tealD, 3, a * line300, [8, 5]); txt('Tensor Core ≈ 300', X + 70, Yv(295) - 10, { size: 17, weight: 800, color: C.tealD, alpha: a * line300 }); }
  pts.forEach(([v, name, col, al], i) => { if (al <= 0.003) return; g.globalAlpha = a * al; g.fillStyle = col; g.beginPath(); g.arc(X + 49, Yv(v), 9, 0, 7); g.fill(); g.globalAlpha = 1; txt(`${name} ${v}`, X + 66, Yv(v) + 6 + (i % 2 ? 0 : 0), { size: 16, weight: 800, color: col, alpha: a * al, font: MONO }); });
}
// 右侧内容区（x 980 … 1600）
const CP = [980, 268, 620];
function codeCard(y, title, lines, a, src, hi = -1) {
  if (a <= 0.003) return y;
  const [x, , w] = CP, h = 60 + lines.length * 40 + (src ? 30 : 0);
  rr(x, y, w, h, '#fbf9f4', a); sr(x, y, w, h, C.lg, 1.5, a);
  txt(title, x + 16, y + 34, { size: 20, weight: 800, color: C.gray, alpha: a });
  lines.forEach(([m, s2], i) => { const yy = y + 52 + i * 40; if (m === '+') rr(x + 8, yy, w - 16, 36, 'rgba(47,138,87,0.14)', a); if (i === hi || (hi % 1 && Math.abs(i - hi) < 1)) { rr(x + 8, yy, w - 16, 36, 'rgba(35,80,216,0.16)', a); rr(x + 2, yy, 6, 36, C.blue, a); } txt(m, x + 18, yy + 26, { size: 22, weight: 800, color: C.use, alpha: a, font: MONO }); txt(s2, x + 42, yy + 26, { size: 22, weight: 600, color: C.ink, alpha: a, font: MONO }); });
  if (src) txt(src, x + w - 14, y + h - 12, { size: 15, weight: 600, color: C.gray, align: 'right', alpha: a });
  return y + h + 16;
}
function lines_(y, rows, a) { if (a <= 0.003) return y; let yy = y; rows.forEach(([s2, size = 22, col = C.ink, wt = 700, font = ZH]) => { yy += size * 1.25; txt(s2, CP[0] + 6, yy, { size, weight: wt, color: col, alpha: a, font }); yy += size * 0.35; }); return yy + 10; }
// 预测与实测并排的两根条（同一刻度）
function predBars(y, pred, meas, a, { max = 70, unit = 'TFLOPS', note = '', labels = ['预测', '实测'] } = {}) {
  if (a <= 0.003) return;
  const X = CP[0] + 110, W_ = 400, sc = W_ / max;
  txt(unit + '（8192 阶）', CP[0] + 6, y, { size: 17, weight: 700, color: C.gray, alpha: a });
  [[labels[0], pred, '#9a9284'], [labels[1], meas, C.blue]].forEach(([nm, vals, col], i) => {
    const yy = y + 16 + i * 64;
    txt(nm, X - 14, yy + 34, { size: 22, weight: 800, color: col, align: 'right', alpha: a });
    (typeof vals[0] === 'number' ? [vals] : vals).forEach((v, j, arr) => { const hh = arr.length > 1 ? 22 : 44; rr(X, yy + j * 24, v[0] * sc, hh, col, a); txt(v[1], X + v[0] * sc + 10, yy + j * 24 + hh * 0.75, { size: arr.length > 1 ? 18 : 24, weight: 800, color: col, alpha: a, font: MONO }); });
  });
  if (note) txt(note, CP[0] + 6, y + 156, { size: 19, weight: 800, color: C.red, alpha: a });
}
// 共享内存的 32 个 bank：ok 为 32 个地址落在 32 个 bank（一个周期），conflict 为两个线程落在同一 bank 的不同地址（两个周期）
function bankFig(x, y, a, mode) {
  if (a <= 0.003) return;
  const cw = 18, by = y + 50, BH = mode === 'ok' ? 34 : 64;
  txt('warp 的 32 个线程', x, y + 6, { size: 16, weight: 700, color: C.gray, alpha: a });
  for (let i = 0; i < 32; i++) {
    const tg = mode === 'ok' ? i : i % 16, row = mode === 'ok' ? 0 : Math.floor(i / 16), cx = x + i * cw + cw / 2, bx = x + tg * cw + cw / 2;
    line(cx, y + 22, bx, by + row * 34, mode === 'ok' ? C.use : (row ? C.hot : C.use), 1.2, a * 0.7);
    g.globalAlpha = a; g.fillStyle = C.ink2; g.beginPath(); g.arc(cx, y + 18, 4, 0, 7); g.fill(); g.globalAlpha = 1;
  }
  for (let bk = 0; bk < 32; bk++) {
    sr(x + bk * cw, by, cw - 2, BH, C.lg, 1, a);
    const hit0 = mode === 'ok' || bk < 16, hit1 = mode !== 'ok' && bk < 16;
    if (hit0) rr(x + bk * cw + 1, by + 2, cw - 4, 28, C.use, a * 0.8);
    if (hit1) rr(x + bk * cw + 1, by + 34, cw - 4, 28, C.hot, a * 0.8);
  }
  txt('bank 0', x, by + BH + 18, { size: 14, weight: 700, color: C.gray, alpha: a, font: MONO }); txt('bank 31', x + 32 * cw, by + BH + 18, { size: 14, weight: 700, color: C.gray, alpha: a, font: MONO, align: 'right' });
  txt(mode === 'ok' ? '32 个 bank × 每周期 4 B = 128 B = 32 个 float：一个周期读完' : '两个线程落在同一 bank 的不同地址：分两个周期（bank 冲突）', x, by + BH + 44, { size: 18, weight: 800, color: mode === 'ok' ? C.use : C.hot, alpha: a });
}
// 一步的指令条：读取（橙）与乘加（绿）
function instrStrip(x, y, nLd, nF, a, cap, cs = 26, per = 20) {
  if (a <= 0.003) return;
  for (let i = 0; i < nLd + nF; i++) rr(x + (i % per) * cs, y + Math.floor(i / per) * cs, cs - 4, cs - 4, i < nLd ? C.hot : C.use, a * (i < nLd ? 1 : 0.55));
  txt(cap, x, y + Math.ceil((nLd + nF) / per) * cs + 22, { size: cs < 20 ? 16 : 18, weight: 800, color: C.ink2, alpha: a });
}
// As 转置：一列（隔 BM 个数）→ 一行（连续，两个 float4）
function transposeFig(x, y, a) {
  if (a <= 0.003) return;
  const c = 16;
  txt('As[m][k]：要的 8 个数是一列', x, y, { size: 17, weight: 800, color: C.hot, alpha: a });
  for (let r = 0; r < 10; r++) for (let k = 0; k < 8; k++) { sr(x + k * c, y + 14 + r * c, c - 2, c - 2, C.lg, 1, a); if (k === 2 && r < 8) rr(x + k * c, y + 14 + r * c, c - 2, c - 2, C.hot, a); }
  txt('→', x + 250, y + 100, { size: 34, weight: 800, color: C.gray, alpha: a });
  const x2 = x + 320;
  txt('As[k][m]：变成一行，2 个 float4', x2, y, { size: 17, weight: 800, color: C.thr, alpha: a });
  for (let k = 0; k < 8; k++) for (let m = 0; m < 14; m++) { sr(x2 + m * c, y + 14 + k * c, c - 2, c - 2, C.lg, 1, a); if (k === 2 && m < 8) rr(x2 + m * c, y + 14 + k * c, c - 2, c - 2, C.thr, a); }
  sr(x2 - 2, y + 14 + 2 * c - 2, 4 * c + 2, c + 2, C.ink, 2.5, a); sr(x2 + 4 * c - 2, y + 14 + 2 * c - 2, 4 * c + 2, c + 2, C.ink, 2.5, a);
}
// warp 分块：block 128 × 128 → 8 个 warp（64 × 32）→ 一个线程的 2 × 2 个 4 × 4
function warpFig(x, y, a) {
  if (a <= 0.003) return;
  const s = 2;
  txt('block 128 × 128：8 个 warp', x, y, { size: 17, weight: 800, color: C.use, alpha: a });
  for (let wr = 0; wr < 2; wr++) for (let wc = 0; wc < 4; wc++) { const hl = wr === 0 && wc === 1; rr(x + wc * 32 * s, y + 14 + wr * 64 * s, 32 * s - 3, 64 * s - 3, hl ? '#cfe3d6' : '#efebe2', a); sr(x + wc * 32 * s, y + 14 + wr * 64 * s, 32 * s - 3, 64 * s - 3, hl ? C.use : C.lg, hl ? 3 : 1, a); }
  const x2 = x + 300, q = 4;
  txt('一个 warp 64 × 32：线程 8 × 4', x2, y, { size: 17, weight: 800, color: C.thr, alpha: a });
  rr(x2, y + 14, 32 * q, 64 * q, '#e6edf9', a); sr(x2, y + 14, 32 * q, 64 * q, C.thr, 2, a);
  for (const r0 of [0, 32]) for (const c0 of [0, 16]) rr(x2 + c0 * q, y + 14 + r0 * q, 4 * q, 4 * q, C.thr, a);
  txt('线程 0 的 2 × 2 个 4 × 4 小块', x2 + 32 * q + 14, y + 40, { size: 16, weight: 700, color: C.thr, alpha: a });
  txt('行隔 32、列隔 16', x2 + 32 * q + 14, y + 64, { size: 16, weight: 700, color: C.gray, alpha: a });
}
// 时间线：before 为“读 → 等 → 算”交替，after 为读与算两轨重叠
function timeline(y, mode, a, u, t0) {
  if (a <= 0.003) return;
  const X = CP[0] + 90, Wd = 520, rv = lin(u, t0, t0 + 2.5);
  if (mode === 0) {
    const segs = []; for (let i = 0; i < 3; i++) segs.push([i * 4, i * 4 + 1.5, C.hot, '读'], [i * 4 + 1.5, i * 4 + 2.3, C.gray, '等'], [i * 4 + 2.3, i * 4 + 4, C.use, '算']);
    lane(X, y, Wd, 40, 12, segs, rv, a, '改前', C.ink);
  } else {
    const comp = [], ld = []; for (let i = 0; i < 6; i++) { comp.push([i * 2, i * 2 + 1.9, C.use, '算 t']); ld.push([i * 2, i * 2 + 1.5, C.hot, '读 t+1']); }
    lane(X, y, Wd, 40, 12, comp, rv, a, '计算', C.use); lane(X, y + 48, Wd, 30, 12, ld, rv, a, '读取', C.hot);
  }
}
// ---------- 第 2 幕：尺子 ----------
function sRuler(u) {
  const K = IDX('尺子'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  tl(96, 270, [['计算：N³ 次乘加 = 2N³ FLOP', 28, C.purple, 800], ['数据：A、B、C 共 3N² 个数', 28, C.hot, 800]], b(0));
  tl(96, 400, [['强度上限 = 2N³ ÷ (3N² × 4 B) = N/6 FLOP/B', 24, C.tealD, 800, MONO], ['8192 阶 ≈ 1365 FLOP/B', 24, C.ink2, 700, MONO]], b(1));
  tl(96, 520, [['FP32 单元要跑满：', 24, C.gray, 700], ['67 TFLOPS ÷ 3.35 TB/s ≈ 20 FLOP/B', 30, C.blue, 800, MONO]], b(2));
  tl(96, 660, [['naive：每次乘加从显存读 8 B，0.25 FLOP/B', 26, C.red, 800], ['全片要做的：把复用兑现出来', 30, C.ink, 800]], b(3));
  // 曲线（复用“复杂度”一幕的上图）
  const X0 = 960, W_ = 620, X = (n) => X0 + (Math.log2(n) - 4) / 10 * W_, T0 = 320, TH = 420, YT = (v) => T0 + TH - Math.log10(v) / 13 * TH, a0 = b(0);
  line(X0, T0 + TH, X0 + W_, T0 + TH, C.ink2, 2, a0); line(X0, T0, X0, T0 + TH, C.ink2, 2, a0);
  txt('个数（对数刻度），N = 16 … 16384', X0, T0 - 12, { size: 17, weight: 700, color: C.gray, alpha: a0 });
  const grow = lin(u, bt(K, 0) + 0.3, bt(K, 0) + 2.0), Ns = []; for (let e = 4; e <= 14.001; e += 0.25) Ns.push(2 ** e);
  [[(n) => n ** 3, C.purple, 'N³'], [(n) => 3 * n * n, C.hot, '3N²']].forEach(([f, col, lab]) => { const m = Math.max(2, Math.floor(Ns.length * grow)); g.globalAlpha = a0; g.strokeStyle = col; g.lineWidth = 4; g.beginPath(); Ns.slice(0, m).forEach((n, i) => (i ? g.lineTo(X(n), YT(f(n))) : g.moveTo(X(n), YT(f(n))))); g.stroke(); g.globalAlpha = 1; const n = Ns[m - 1]; txt(lab, X(n) - 8, YT(f(n)) - 12, { size: 22, weight: 800, color: col, align: 'right', alpha: a0, font: MONO }); });
  gauge([[0.25, 'naive', C.red, b(3)]], b(2));
}
// ---------- 第 4 幕：naive ----------
// 主画面是 A、B、C 与点积动画；相邻 4 个线程读 B 同一行的相邻 4 个数（合并访问）。用时对比不占主画面：
// 点积动画走完后在左上角空处叠一条小进度条（按实测用时等比）与数字 5.6 倍，约 4 秒后收起（docs/video.md「结果图不单独占一幕」）
function sNaive(u) {
  const K = IDX('naive'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  const pb = b(2);
  view0(1 - pb, 'dot', u, bt(K, 0) + 0.3, { warp: b(0) }); view0(pb, 'block', u, 0);
  bnTag(120, 598, '同一 block 反复读取这条 A 带、B 带', pb);
  layerBar(b(2) > 0.5 ? 0 : -1, b(0));
  const a1 = b(0), y = 300;
  txt('naive 与对照组 naive_T 只差两行', CP[0], y, { size: 22, weight: 800, color: C.gray, alpha: a1 });
  [['naive_T', 'x', C.red], ['naive', 'y', C.blue]].forEach(([nm, ax, col], i) => {
    const yy = y + 50 + i * 110;
    txt(nm, CP[0], yy, { size: 22, weight: 800, color: col, alpha: a1, font: MONO });
    txt(`int row = … + threadIdx.${ax};`, CP[0] + 20, yy + 36, { size: 22, weight: 600, alpha: a1, font: MONO });
    txt(`int col = … + threadIdx.${ax === 'x' ? 'y' : 'x'};`, CP[0] + 20, yy + 68, { size: 22, weight: 600, alpha: a1, font: MONO });
  });
  lines_(560, [['naive：相邻线程 → C 的相邻列 → B 的相邻地址', 22, C.blue, 800], ['naive_T：相邻线程 → C 的相邻行 → A 相隔 N 个数', 22, C.red, 800]], b(1));
  probCard(680, 'naive', [['每次乘加从显存读 2 个数（8 B）→ 0.25 FLOP/B', 22, C.ink, 800], ['同一 block 的线程各自重复读取同一条 A 带、B 带', 22, C.ink, 800]], b(2));
  // 叠在左上角的小进度条：统一时间轴，条长 = 用时
  const t0 = bt(K, 1) + 0.2, LT = 1.8, tc = t0 + 4.8, a2 = po(u, t0, t0 + 0.3) * (1 - p(u, tc, tc + 0.6));
  if (a2 > 0.003) {
    const k = lin(u, t0 + 0.3, t0 + 0.3 + LT), kN = lin(u, t0 + 0.3, t0 + 0.3 + LT * 25.5 / 143.2), X = 116, Y = 384, W_ = 300;
    rr(X - 14, Y - 32, 340, 206, '#fbf9f4', a2 * 0.95); sr(X - 14, Y - 32, 340, 206, C.lg, 1.5, a2);
    txt('4096 阶 · 用时', X, Y - 6, { size: 17, weight: 700, color: C.gray, alpha: a2 });
    [['naive_T', k, C.red, '143 ms', Y + 8, 1], ['naive', kN, C.blue, '25.5 ms', Y + 52, 25.5 / 143.2]].forEach(([nm, f, col, ms, yy, len]) => {
      rr(X, yy, W_, 16, '#e8e4da', a2); rr(X, yy, W_ * len * f, 16, col, a2);
      txt(nm, X, yy + 36, { size: 17, weight: 800, color: col, alpha: a2, font: MONO });
      if (f >= 1) txt(ms, X + W_, yy + 36, { size: 17, weight: 800, color: col, alpha: a2, font: MONO, align: 'right' });
    });
    txt('5.6 倍', X, Y + 150, { size: 40, weight: 800, color: C.ink, alpha: a2 * po(u, t0 + 0.3 + LT, t0 + 0.6 + LT) });
  }
  gauge([[0.25, 'naive', C.red, 1]], b(0));
}
// ---------- 问题与改动的视觉标记 ----------
// 问题（上一版的瓶颈）：橙色“瓶颈”标签贴在旧版画面上 + 右栏橙框问题卡，只在每幕结尾出现；改动：页眉的版本名标题卡与“改动 · 版本名”代码卡。
function bnTag(x, y, text, a) {
  if (a <= 0.003) return;
  const w0 = tw('瓶颈', 20, 800) + 20, w1 = tw(text, 19, 800) + 20;
  rr(x, y - 26, w0, 34, C.hot, a); txt('瓶颈', x + 10, y - 2, { size: 20, weight: 800, color: '#fff', alpha: a });
  rr(x + w0, y - 26, w1, 34, '#fbf9f4', a * 0.95); sr(x + w0, y - 26, w1, 34, C.hot, 2, a); txt(text, x + w0 + 10, y - 3, { size: 19, weight: 800, color: C.hot, alpha: a });
}
function probCard(y, ver, rows, a, h = null) {
  if (a <= 0.003) return y;
  const [x, , w] = CP, hh = h ?? 64 + rows.length * 38;
  rr(x, y, w, hh, '#fbf3ec', a); sr(x, y, w, hh, C.hot, 3, a);
  rr(x, y, tw('瓶颈', 20, 800) + 24, 38, C.hot, a); txt('瓶颈', x + 12, y + 27, { size: 20, weight: 800, color: '#fff', alpha: a });
  txt(`${ver} 的问题`, x + tw('瓶颈', 20, 800) + 38, y + 27, { size: 20, weight: 800, color: C.hot, alpha: a, font: /^[\x00-\x7f]+$/.test(ver) ? MONO : ZH });
  let yy = y + 50; rows.forEach(([s2, size = 22, col = C.ink, wt = 800]) => { yy += size * 1.25; txt(s2, x + 18, yy, { size, weight: wt, color: col, alpha: a }); yy += size * 0.4; });
  return y + hh;
}
// 共享内存的 32 个 bank：32 个竖列，地址按 4 B 依次轮流落入各列（第 r 行第 b 列 = 地址 128r + 4b）。
// mode：fill = 地址逐个落入；ok = 一个 warp 的 32 个线程落在 32 个 bank（一个周期）；conflict = 两个线程落在同一 bank 的不同地址（两个周期）
function bank32(x, y, w, a, mode, u, t0, { caption = true, rh = 30 } = {}) {
  if (a <= 0.003) return;
  const cw = w / 32, top = y + 74, rows = 4, RD = 2;   // RD：一个 warp 读取的那一行
  for (let bk = 0; bk < 32; bk++) txt(String(bk), x + bk * cw + cw / 2, top - 8, { size: 12, weight: 700, color: C.gray, align: 'center', alpha: a, font: MONO });
  txt('bank', x - 10, top - 8, { size: 14, weight: 800, color: C.gray, align: 'right', alpha: a });
  for (let r = 0; r < rows; r++) txt(`${128 * r} – ${128 * r + 124}`, x - 10, top + r * rh + rh * 0.62, { size: 14, weight: 700, color: C.ink2, align: 'right', alpha: a * (mode === 'fill' ? clamp(lin(u, t0, t0 + 3.2) * 4 - r) : 1), font: MONO });
  const nIn = mode === 'fill' ? Math.floor(lin(u, t0, t0 + 3.2) * 128) : 128;
  for (let r = 0; r < rows; r++) for (let bk = 0; bk < 32; bk++) {
    const i = r * 32 + bk, X = x + bk * cw, Y = top + r * rh;
    sr(X + 1, Y, cw - 2, rh - 3, C.lg, 1, a);
    if (i < nIn) { const fresh = mode === 'fill' && i >= nIn - 3; rr(X + 2, Y + 1, cw - 4, rh - 5, fresh ? C.use : AC.sm, a); }
  }
  for (let bk = 0; bk < 32; bk++) sr(x + bk * cw, top - 2, cw, rows * rh + 2, '#b8b0a0', 1, a * 0.6);
  // 地址标注：第 0 行前几个与最后一个、第 1 行第一个
  const lab_ = (i, s2) => { if (i >= nIn) return; const r = Math.floor(i / 32), bk = i % 32; txt(s2, x + bk * cw + cw / 2, top + r * rh + rh * 0.62, { size: cw < 17 ? 10 : 12, weight: 700, color: C.ink2, align: 'center', alpha: a, font: MONO }); };
  [[0, '0'], [1, '4'], [2, '8'], [3, '12'], [31, '124']].forEach(([i, s2]) => lab_(i, s2));
  if (mode === 'fill') { if (caption) txt('左侧为每一行的字节地址：每 4 B 换下一个 bank，128 B 后回到 bank 0', x, top + rows * rh + 30, { size: 18, weight: 800, color: C.ink2, alpha: a }); return; }
  // 一个 warp 的 32 个线程（上方一排点），连线到要读的格子
  const k = lin(u, t0, t0 + 0.8);
  for (let i = 0; i < 32; i++) {
    const cx = x + i * cw + cw / 2, tg = mode === 'ok' ? i : i % 16, row = mode === 'ok' ? RD : (i < 16 ? RD : RD + 1), bx = x + tg * cw + cw / 2, by = top + row * rh + 4;
    g.globalAlpha = a; g.fillStyle = C.thr; g.beginPath(); g.arc(cx, y + 16, 4, 0, 7); g.fill(); g.globalAlpha = 1;
    line(cx, y + 22, mix(cx, bx, k), mix(y + 22, by, k), mode === 'ok' || i < 16 ? C.thr : C.hot, 1.3, a * 0.8);
    if (k >= 1) rr(x + tg * cw + 2, top + row * rh + 1, cw - 4, rh - 5, mode === 'ok' || i < 16 ? C.thr : C.hot, a * 0.85);
  }
  txt('warp 的 32 个线程', x, y + 2, { size: 13, weight: 700, color: C.thr, alpha: a });
  if (caption) txt(mode === 'ok' ? '32 个地址落在 32 个 bank：32 个窗口同时取数，1 个周期读完' : '两个线程落在同一个 bank 的不同地址：分 2 个周期（bank 冲突）', x, top + rows * rh + 30, { size: 18, weight: 800, color: mode === 'ok' ? C.thr : C.hot, alpha: a * k });
}
// 一个 SM 的 128 个 FP32 乘加单元（8 × 16 个小灯），lit 个亮起
function lamps(x, y, lit, a, d = 24) {
  if (a <= 0.003) return;
  for (let i = 0; i < 128; i++) { const on = i < lit; g.globalAlpha = a; g.fillStyle = on ? C.use : '#d8d3c6'; g.beginPath(); g.arc(x + (i % 16) * d + d / 2, y + Math.floor(i / 16) * d + d / 2, d * 0.38, 0, 7); g.fill(); g.globalAlpha = 1; }
}
// ---------- 台阶幕：改动 → 推算 → 实测 → 收尾（这一版的问题，贴在这一版的画面上） ----------
function sStep(u) {
  const S = SCRIPT[CUR_K], k = CUR_K, b = (j) => po(u, bt(k, j), bt(k, j) + 0.5);
  const id = S.step, T = (j) => bt(k, j), last = S.beats.length - 1, pr = b(last), rc = 1 - pr;   // pr：收尾的问题拍；rc：右栏的改动内容随之收起
  const ver = S.tag;
  if (id === 'smem') {
    let hi = -1;
    const bz = p(u, T(3), T(3) + 1.2), back = p(u, T(8), T(8) + 0.8);
    const SMF = [L1.As[0] - 40, L1.As[1] - 66, 340];
    if (bz <= 0.001) zoomTo(p(u, T(0), T(0) + 1.4), FOCUS0(), (al) => view0(al, 'block', u, 0), (al) => { hi = view1(al, 'k0', u, T(0) + 1.4); });
    else if (back <= 0.001) zoomTo(bz, SMF, (al) => view1(al, 'k0', u, T(0) + 1.4), (al) => {
      lab('共享内存：32 个 bank', 110, 330, C.use, al, 'left', 24);
      lab('每个 bank 每个时钟周期取出 4 B（一个 float）', 110, 362, C.ink2, al, 'left', 19);
      bank32(200, 410, 590, al, u < T(4) ? 'fill' : 'ok', u, u < T(4) ? T(3) + 1.2 : T(4) + 0.3, { rh: 42 });
      if (u >= T(4)) lab('这一行是 Bs[k][0 … 31]：32 个连续的 float', 200, 724, C.thr, al * b(4), 'left', 19);
    });
    else { view1(back, 'thread', u, T(8)); bnTag(136, 726, '共享内存：每次乘加读 2 个数', back); }
    lab('左图示意 4 步；8192 阶、T = 32 时 k0 共走 K ÷ T = 256 步', 106, 930, C.ink2, b(1) * (1 - bz), 'left', 19);
    layerBar(mix(0, 1, p(u, T(2) + 0.3, T(2) + 1.1)), 1);
    const r1 = rc * (1 - b(3));
    let y = codeCard(300, '改动 · matmul_smem', [['+', 'As[ty][tx] = A[row * N + k0 + tx];'], ['+', 'Bs[ty][tx] = B[(k0 + ty) * N + col];'], ['+', '__syncthreads();'], [' ', 'for k: acc += As[ty][k] * Bs[k][tx];']], b(0) * r1, '精简自 matmul.cu 的 matmul_smem', hi);
    lines_(y, [['显存每个数被块内 T 个线程使用', 20, C.ink2, 700], ['T/4 = 8 FLOP/B（T = 32），仍在 20 的线下', 22, C.hot, 800]], b(2) * r1);
    // 推算：共享内存每周期供 32 个数 → 16 次乘加 → 128 个单元亮 16 个
    const r2 = b(3) * rc;
    let yy = lines_(270, [['32 个 bank × 4 B = 128 B = 32 个 float / 周期', 22, C.use, 800]], r2);
    yy = lines_(yy - 6, [['一个 warp 读一次：1 个周期', 22, C.thr, 800]], b(4) * rc);
    const a5 = b(5) * rc;
    if (a5 > 0.003) {
      txt('一个 SM 的 128 个 FP32 乘加单元', CP[0] + 6, 420, { size: 19, weight: 800, color: C.ink2, alpha: a5 });
      lamps(CP[0] + 6, 434, Math.round(16 * lin(u, T(5) + 0.4, T(5) + 1.4)), a5);
      txt('As、Bs 各要 1 个数：32 个 float ÷ 2 = 16 次乘加', CP[0] + 6, 658, { size: 21, weight: 800, color: C.use, alpha: a5 });
    }
    lines_(668, [['16 ÷ 128 = 1/8', 26, C.ink, 800, MONO], ['67 × 1/8 ≈ 8.4 TFLOPS', 26, C.blue, 800, MONO]], b(6) * rc);
    predBars(830, [8.4, '8.4'], [[8.22, '8.2（T = 16）'], [9.22, '9.2（T = 32）']], b(7) * rc, { max: 20 });
    probCard(300, '共享内存分块', [['每次乘加仍从共享内存读取 2 个数', 24, C.ink, 800], ['共享内存每周期只供 32 个 float', 21, C.ink2, 700], ['→ 128 个 FP32 单元只有 16 个在工作', 21, C.ink2, 700]], pr);
    gauge([[0.25, 'naive', C.red, 1], [8, 'smem', C.use, b(2)]], 1);
  } else if (id === 'reg') {
    let hi = -1;
    zoomTo(p(u, T(0), T(0) + 1.4), FOCUS1(), (al) => view1(al, 'thread', u, 0), (al) => { const r = view2(al, u, T(0) + 1.4); hi = r === 0 ? 0.5 : 2; });
    layerBar(mix(1, 3, p(u, T(4) + 0.3, T(4) + 1.1)), 1);
    outerRecall(110, 300, b(1) * (1 - b(2)));
    bnTag(110, 330, '读取指令占五分之一', pr);
    let y = codeCard(300, '改动 · matmul_reg', [['+', 'rA[i] = As[(tRow * TM + i) * BK + k];'], ['+', 'rB[j] = Bs[k * BN + tCol * TN + j];'], ['+', 'acc[i * TN + j] += rA[i] * rB[j];']], b(0) * rc, '精简自 matmul.cu 的 matmul_reg', u > T(0) + 1.4 ? hi : -1);
    const a3 = b(2) * rc;
    if (a3 > 0.003) {
      txt('推算：同样 64 格', CP[0] + 6, y + 18, { size: 18, weight: 700, color: C.gray, alpha: a3 });
      [[1, 64], [2, 32], [4, 16], [8, 8]].forEach(([m, n2], i) => { const yy = y + 48 + i * 30; txt(`${m} × ${n2}`, CP[0] + 6, yy, { size: 21, weight: 800, alpha: a3, font: MONO }); txt(`读 ${m + n2} 个`, CP[0] + 130, yy, { size: 21, weight: 800, color: C.hot, alpha: a3 }); txt(`每读 1 个做 ${(64 / (m + n2)).toFixed(i ? 1 : 2)} 次`, CP[0] + 260, yy, { size: 21, weight: 800, color: C.use, alpha: a3 }); });
      txt('块越大越省、正方形最优（I/O 下界，Hong & Kung 1981）', CP[0] + 6, y + 180, { size: 19, weight: 800, color: C.tealD, alpha: a3 });
      txt('共享内存读取 ÷ 8 → 上限回到 FP32 峰值 67', CP[0] + 6, y + 210, { size: 21, weight: 800, color: C.thr, alpha: a3 });
    }
    predBars(842, [67, '67'], [24.7, '24.7'], b(3) * rc, { max: 70, note: '离上限 67 还远' });
    const yp = probCard(300, '寄存器分块', [['每一步：16 条读取指令 + 64 条乘加指令', 24, C.ink, 800], ['读取占五分之一，与乘加争用发射', 21, C.ink2, 700]], pr);
    if (pr > 0.003) instrStrip(CP[0] + 18, yp + 24, 16, 64, pr, '一个线程一步的指令：橙 = 读取，绿 = 乘加', 26, 20);
    gauge([[0.25, 'naive', C.red, 1], [8, 'smem', C.use, 1], [32, 'reg', C.thr, b(2)]], 1);
  } else if (id === 'vec') {
    layerBar(mix(3, 1, p(u, T(3) + 0.3, T(3) + 1.1)), 1);
    const r2 = view2(1, u, 0, { vec: b(0) }), hi = r2 === 0 ? 2.5 : -1;
    const sa = 1 - pr;
    txt('每一步的指令（一个线程）', 110, 300, { size: 18, weight: 800, color: C.gray, alpha: b(0) * sa });
    instrStrip(110, 314, 16, 64, b(0) * sa, '改前：16 条读取 + 64 条乘加', 13, 20);
    instrStrip(110, 404, 4, 64, b(1) * sa, '改后：4 条读取 + 64 条乘加', 13, 20);
    bnTag(110, 330, '读 Bs：bank 冲突', pr);
    const y = codeCard(300, '改动 · matmul_vec', [['+', 'float4 a = *(float4*)&A[...];'], ['+', 'As[(c + i) * BM + r] = a.x … a.w;'], ['+', 'rA = *(float4*)&As[k * BM + …];'], ['+', 'rB = *(float4*)&Bs[k * BN + …];']], b(0) * rc, '精简自 matmul.cu 的 matmul_vec', hi);
    lines_(y, [['一条指令读 4 个相邻的数', 18, C.gray, 700], ['每步读取指令 16 → 4 条', 24, C.thr, 800]], b(1) * rc);
    predBars(830, [24.7, '24.7'], [35.3, '35.3'], b(2) * rc, { max: 60, labels: ['改前', '实测'] });
    const yp = probCard(300, '向量化读取', [['读 Bs 时两个线程落在同一个 bank 的不同地址', 22, C.ink, 800], ['同一个 bank 每周期只能取 4 B：分 2 个周期', 21, C.ink2, 700]], pr);
    if (pr > 0.003) bank32(CP[0] + 110, yp + 30, 490, pr, 'conflict', u, T(3) + 0.6);
    gauge([[0.25, 'naive', C.red, 1], [8, 'smem', C.use, 1], [32, 'reg', C.thr, 1]], 1);
  } else if (id === 'warp') {
    const sw = p(u, T(3), T(3) + 0.8);
    layerBar(mix(1, 0, p(u, T(3) + 0.3, T(3) + 1.1)), 1);
    viewW(1 - sw, b(1));
    if (b(1) * (1 - sw) > 0.003) { g.save(); g.translate(200, 852); g.scale(0.56, 0.56); bank32(0, 0, 900, b(1) * (1 - sw), 'ok', u, T(1) + 0.3); g.restore(); }
    view1(sw, 'seq', u, T(3));
    bnTag(136, 726, '每一块：读完、同步，才能计算', sw);
    let y = codeCard(300, '改动 · matmul_warptile', [['+', 'int wRow = warp / (BN / WN), wCol = ...;'], ['+', 'int lRow = lane / 4, lCol = lane % 4;'], ['+', 'rB = Bs[k*BN + wCol*WN + ni*16 + lCol*4];']], b(0) * rc, '精简自 matmul.cu 的 matmul_warptile', b(1) > 0.5 ? 2 : -1);
    lines_(y, [['block → warp → 线程，三层分块', 18, C.gray, 700], ['地址首尾相连：一个周期读完', 24, C.use, 800]], b(1) * rc);
    predBars(830, [35.3, '35.3'], [37.0, '37.0'], b(2) * rc, { max: 60, labels: ['改前', '实测'] });
    const yp = probCard(300, 'warp 分块', [['每一块先读完、同步，才能开始计算', 24, C.ink, 800], ['读取期间计算单元空闲', 21, C.ink2, 700]], pr);
    timeline(yp + 30, 0, pr, u, T(3));
    gauge([[0.25, 'naive', C.red, 1], [8, 'smem', C.use, 1], [32, 'reg', C.thr, 1]], 1);
  } else if (id === 'overlap') {
    const hi = view1(1, 'dbuf', u, T(0));
    layerBar(mix(0, 3, p(u, T(3) + 0.3, T(3) + 1.1)), 1);
    bnTag(136, 726, '指令发射效率', pr);
    let y = codeCard(300, '改动 · matmul_dbuf', [['+', 'if (t + 1 < T) fetch((t + 1) * BK);'], ['+', 'if (t + 1 < T) stash(cur ^ 1);']], b(0) * rc, '精简自 matmul.cu 的 matmul_dbuf', hi);
    if (b(1) * rc > 0.003) { txt('计算当前块时读取下一块：等待被覆盖，预测接近 cuBLAS', CP[0] + 6, y + 10, { size: 19, weight: 800, color: C.use, alpha: b(1) * rc }); timeline(y + 24, 1, b(1) * rc, u, T(1)); }
    predBars(830, [51.7, '≈ cuBLAS 51.7'], [44.3, '44.3（86%）'], b(2) * rc, { max: 60 });
    const yp = probCard(300, '双缓冲', [['FMA 管线忙碌的时间比例低于 cuBLAS', 24, C.ink, 800], ['差距在指令发射的效率上', 21, C.ink2, 700]], pr);
    predBars(yp + 40, [79.8, '79.8%'], [68.3, '68.3%'], pr, { max: 100, unit: 'ncu：FMA 管线忙碌比例', labels: ['cuBLAS', 'dbuf'] });
    gauge([[0.25, 'naive', C.red, 1], [8, 'smem', C.use, 1], [32, 'reg', C.thr, 1]], 1);
  }
}
// ---------- A / B / C 经典排布（左侧面板）：B 在上、A 在左、C 在右下；每一层是上一层某个块放大后的样子 ----------
const AC = { aL: '#c9d3f3', aM: '#8ea3e6', bL: '#f6e6aa', bM: '#f1cf62', cL: '#e1d5fb', sm: '#e3efe6' };
const PANEL = [92, 276, 700, 736];
function gl(x, y, w, h, nx, ny, col, a, lw = 1) { for (let i = 1; i < nx; i++) line(x + i * w / nx, y, x + i * w / nx, y + h, col, lw, a); for (let j = 1; j < ny; j++) line(x, y + j * h / ny, x + w, y + j * h / ny, col, lw, a); }
function mat(x, y, w, h, fill, nx, ny, a, stroke = C.ink2) { rr(x, y, w, h, fill, a); gl(x, y, w, h, nx, ny, '#fbf9f4', a, 1.2); sr(x, y, w, h, stroke, 2, a); }
const lab = (s, x, y, col, a, align = 'left', size = 20) => txt(s, x, y, { size, weight: 800, color: col, alpha: a, align });
const cyc = (u, t0, P, n) => { const t = Math.max(0, u - t0), s = Math.floor(t / P) % n; return [s, (t % P) / P]; };
// 放大过渡：旧视图绕焦点放大并淡出，新视图从 0.85 倍放大到 1 并淡入
function zoomTo(f, focus, drawOld, drawNew) {
  if (f <= 0.001) { drawOld(1); return; }
  if (f >= 0.999) { drawNew(1); return; }
  const e = ease(f), [fx, fy, fw] = focus, cx = fx + fw / 2, cy = fy + fw / 2, sc = mix(1, 520 / fw, e);
  g.save(); g.beginPath(); g.rect(...PANEL); g.clip();
  g.save(); g.translate(PANEL[0] + PANEL[2] / 2 - (PANEL[0] + PANEL[2] / 2 - cx) * (1 - e), PANEL[1] + PANEL[3] / 2 - (PANEL[1] + PANEL[3] / 2 - cy) * (1 - e)); g.scale(sc, sc); g.translate(-cx, -cy); drawOld(1 - clamp(e * 1.6)); g.restore();
  const s2 = mix(0.85, 1, e), mx = PANEL[0] + PANEL[2] / 2, my = PANEL[1] + PANEL[3] / 2;
  g.save(); g.translate(mx, my); g.scale(s2, s2); g.translate(-mx, -my); drawNew(clamp((e - 0.35) / 0.65)); g.restore();
  g.restore();
}
// 第 0 层：整个矩阵（16 × 16 格示意 8192 阶）
const L0 = { s: 300, A: [120, 640], B: [444, 316], C: [444, 640], n: 16 };
const L0BLK = [1, 2];                                   // 选中的 block 块（4 × 4 格）
function view0(a, mode, u, t0, { warp = 0 } = {}) {
  if (a <= 0.003) return -1;
  const { s, A, B, C: Cc, n } = L0, cs = s / n;
  mat(A[0], A[1], s, s, AC.aL, n, n, a); mat(B[0], B[1], s, s, AC.bL, n, n, a); mat(Cc[0], Cc[1], s, s, AC.cL, n, n, a);
  lab('A  M × K', A[0], A[1] - 12, C.blue, a); lab('B  K × N', B[0] - 14, B[1] + 24, C.yelD, a, 'right'); lab('C  M × N', Cc[0] + s, Cc[1] + s + 30, C.purple, a, 'right');
  if (mode === 'dot') {
    const i = 5, j = 10, [k] = cyc(u, t0, 0.22, n);
    rr(A[0], A[1] + i * cs, s, cs, AC.aM, a); rr(A[0] + k * cs, A[1] + i * cs, cs, cs, C.blue, a);
    rr(B[0] + j * cs, B[1], cs, s, AC.bM, a); rr(B[0] + j * cs, B[1] + k * cs, cs, cs, C.yel, a);
    rr(Cc[0] + j * cs, Cc[1] + i * cs, cs, cs, mixc(AC.cL, C.purple, (k + 1) / n), a); sr(Cc[0] + j * cs, Cc[1] + i * cs, cs, cs, C.purple, 3, a);
    lab(`一个线程 = C 的一格：A 的一行 · B 的一列，k = ${k}`, A[0], 990, C.ink2, a, 'left', 19);
    if (warp > 0.003) {   // 相邻 4 个线程（C 的第 i 行、第 j … j+3 列）：同一步读 B 第 k 行的相邻 4 个数，A 的同一个数广播
      for (let d = 1; d < 4; d++) { sr(Cc[0] + (j + d) * cs, Cc[1] + i * cs, cs, cs, C.purple, 2, a * warp); rr(B[0] + (j + d) * cs, B[1] + k * cs, cs, cs, C.yel, a * warp * 0.8); }
      sr(B[0] + j * cs - 2, B[1] + k * cs - 2, 4 * cs + 4, cs + 4, C.ink, 2.5, a * warp);
      lab('相邻 4 个线程：B 的相邻 4 个数', B[0] - 14, 586, C.ink2, a * warp, 'right', 18);
      lab('一次合并访问', B[0] - 14, 610, C.use, a * warp, 'right', 18);
    }
    return -1;
  }
  const [bi, bj] = L0BLK, T = 4;
  rr(A[0], A[1] + bi * T * cs, s, T * cs, AC.aM, a); rr(B[0] + bj * T * cs, B[1], T * cs, s, AC.bM, a);
  rr(Cc[0] + bj * T * cs, Cc[1] + bi * T * cs, T * cs, T * cs, '#cfe3d6', a); sr(Cc[0] + bj * T * cs, Cc[1] + bi * T * cs, T * cs, T * cs, C.use, 4, a);
  lab('block 的块 T × T：块内线程都读这条 A 带、B 带', A[0], 990, C.use, a, 'left', 19);
  return -1;
}
const FOCUS0 = () => { const cs = L0.s / L0.n; return [L0.C[0] + L0BLK[1] * 4 * cs, L0.C[1] + L0BLK[0] * 4 * cs, 4 * cs]; };
// 第 1 层：block 的块。A 带（T 行 × K，4 个 T × T 小块）、B 带、C 块，以及共享内存里的 As、Bs
const L1 = { t: 110, A: [106, 742], B: [560, 290], C: [560, 742], As: [176, 486], Bs: [376, 486] };
function view1(a, mode, u, t0) {
  if (a <= 0.003) return -1;
  const { t, A, B, C: Cc, As, Bs } = L1;
  for (let i = 0; i < 4; i++) { mat(A[0] + i * t, A[1], t, t, AC.aL, 4, 4, a); mat(B[0], B[1] + i * t, t, t, AC.bL, 4, 4, a); }
  lab('A 带：T 行 × K', A[0], A[1] + t + 30, C.blue, a); lab('B 带：K × T', B[0] - 14, B[1] + 24, C.yelD, a, 'right'); lab('C 块 T × T', Cc[0] + t, Cc[1] + t + 30, C.purple, a, 'right');
  rr(As[0] - 40, As[1] - 66, 340, 226, AC.sm, a); sr(As[0] - 40, As[1] - 66, 340, 226, C.use, 2.5, a); lab('共享内存', As[0] - 28, As[1] - 34, C.use, a);
  const slot = (x, y, fill, nm, al = 1) => { rr(x, y, t, t, fill, a * al); sr(x, y, t, t, C.ink2, 1.5, a * al); lab(nm, x + t / 2, y + t + 24, C.ink2, a * al, 'center', 18); };
  let hi = -1, cfill = AC.cL;
  if (mode === 'thread') {
    slot(As[0], As[1], AC.aM, 'As'); slot(Bs[0], Bs[1], AC.bM, 'Bs');
    rr(Cc[0], Cc[1], t, t, mixc(AC.cL, C.purple, 0.35), a); gl(Cc[0], Cc[1], t, t, 16, 16, '#fbf9f4', a, 0.8);
    const [k] = cyc(u, t0, 0.3, 16), r = 4, c = 9, q = t / 16;
    rr(Cc[0] + c * q, Cc[1] + r * q, q, q, C.thr, a); sr(Cc[0] + c * q - 2, Cc[1] + r * q - 2, q + 4, q + 4, C.thr, 2, a);
    rr(As[0] + k * q, As[1] + r * q, q, q, C.blue, a); rr(Bs[0] + c * q, Bs[1] + k * q, q, q, C.yel, a);
    sr(As[0], As[1] + r * q, t, q, C.blue, 1.5, a); sr(Bs[0] + c * q, Bs[1], q, t, C.yelD, 1.5, a);
    lab('每次乘加：从 As、Bs 各读 1 个数', As[0] - 40, As[1] + 190, C.hot, a, 'left', 19);
    sr(Cc[0], Cc[1], t, t, C.purple, 3, a);
    return -1;
  }
  const P = mode === 'dbuf' ? 1.3 : 1.7, [st, ph] = cyc(u, t0, P, 4);
  const lerp = (p0, p1, f) => [mix(p0[0], p1[0], f), mix(p0[1], p1[1], f)];
  // 已读完的小块变灰，当前一对高亮
  for (let i = 0; i < st; i++) { rr(A[0] + i * t, A[1], t, t, '#d8d3c6', a * 0.8); rr(B[0], B[1] + i * t, t, t, '#d8d3c6', a * 0.8); }
  sr(A[0] + st * t, A[1], t, t, C.blue, 4, a); sr(B[0], B[1] + st * t, t, t, C.yelD, 4, a);
  let done = st;
  if (mode === 'k0' || mode === 'seq') {
    const fs = mode === 'seq' ? 0.35 : 0.4, ws = mode === 'seq' ? 0.6 : 0.5, f = clamp(ph / fs);
    const pa = lerp([A[0] + st * t, A[1]], As, ease(f)), pb = lerp([B[0], B[1] + st * t], Bs, ease(f));
    slot(As[0], As[1], AC.aL, 'As', 0.5); slot(Bs[0], Bs[1], AC.bL, 'Bs', 0.5);
    rr(pa[0], pa[1], t, t, C.blue, a * 0.75); sr(pa[0], pa[1], t, t, C.blue, 3, a); rr(pb[0], pb[1], t, t, C.yel, a * 0.75); sr(pb[0], pb[1], t, t, C.yelD, 3, a);
    const cf = clamp((ph - ws) / (1 - ws)); done = st + cf;
    if (ph < fs) hi = mode === 'seq' ? 0 : 0.5; else if (ph < ws) hi = mode === 'seq' ? -2 : 2; else hi = mode === 'seq' ? -3 : 3;
    if (mode === 'seq' && ph >= fs && ph < ws) lab('等', As[0] + 140, As[1] + 66, C.gray, a, 'center', 40);
  } else {
    // 双缓冲：算这一对（两个槽里靠前的一份）的同时，下一对已经在路上，进入另一份
    slot(As[0] + 14, As[1] + 14, AC.aL, '', 0.6); slot(Bs[0] + 14, Bs[1] + 14, AC.bL, '', 0.6);
    rr(As[0], As[1], t, t, C.blue, a * 0.75); sr(As[0], As[1], t, t, C.blue, 3, a); rr(Bs[0], Bs[1], t, t, C.yel, a * 0.75); sr(Bs[0], Bs[1], t, t, C.yelD, 3, a);
    lab('As', As[0] + t / 2, As[1] + t + 38, C.ink2, a, 'center', 18); lab('Bs', Bs[0] + t / 2, Bs[1] + t + 38, C.ink2, a, 'center', 18);
    if (st < 3) {
      const f = ease(clamp(ph / 0.8)), n = st + 1;
      const pa = lerp([A[0] + n * t, A[1]], [As[0] + 14, As[1] + 14], f), pb = lerp([B[0], B[1] + n * t], [Bs[0] + 14, Bs[1] + 14], f);
      g.setLineDash([8, 6]); sr(pa[0], pa[1], t, t, C.blue, 3, a); sr(pb[0], pb[1], t, t, C.yelD, 3, a); g.setLineDash([]);
      rr(pa[0], pa[1], t, t, C.blue, a * 0.25); rr(pb[0], pb[1], t, t, C.yel, a * 0.3);
      lab('下一对已在路上', As[0] - 40, As[1] + 190, C.hot, a, 'left', 19);
    }
    done = st + ph; hi = ph < 0.8 ? 0 : 1;
  }
  cfill = mixc(AC.cL, C.purple, clamp(done / 4));
  rr(Cc[0], Cc[1], t, t, cfill, a); sr(Cc[0], Cc[1], t, t, C.purple, 3, a);
  lab(`k0：第 ${st + 1} / 4 对`, As[0] + 300, As[1] - 34, C.ink2, a, 'right', 19);
  return hi;
}
const FOCUS1 = () => { const q = L1.t / 16; return [L1.C[0] + 9 * q, L1.C[1] + 4 * q, q]; };
// 第 2 层：一个线程。As 中本线程的 8 行、Bs 中的 8 列，寄存器 rA、rB 与 8 × 8 的 acc；外积随 k 前进
const L2 = { c: 30, As: [110, 600], rA: [370, 600], Bs: [420, 300], rB: [420, 550], acc: [420, 600] };
function view2(a, u, t0, { vec = 0 } = {}) {
  if (a <= 0.003) return -1;
  const { c, As, rA, Bs, rB, acc } = L2, S = 8 * c, [k, ph] = cyc(u, t0, 0.95, 8);
  mat(As[0], As[1], S, S, AC.aL, 8, 8, a); mat(Bs[0], Bs[1], S, S, AC.bL, 8, 8, a);
  lab('As：本线程的 8 行', As[0], As[1] - 14, C.blue, a); lab('Bs：本线程的 8 列', Bs[0] - 14, Bs[1] + S - 30, C.yelD, a, 'right');
  rr(As[0] + k * c, As[1], c, S, AC.aM, a); sr(As[0] + k * c, As[1], c, S, C.blue, 3, a);
  rr(Bs[0], Bs[1] + k * c, S, c, AC.bM, a); sr(Bs[0], Bs[1] + k * c, S, c, C.yelD, 3, a);
  const f = ease(clamp(ph / 0.35));
  rr(rA[0], rA[1], c, S, '#eef1fb', a); sr(rA[0], rA[1], c, S, C.thr, 2, a); rr(rB[0], rB[1], S, c, '#fbf4dc', a); sr(rB[0], rB[1], S, c, C.thr, 2, a);
  const ax = mix(As[0] + k * c, rA[0], f), by = mix(Bs[1] + k * c, rB[1], f);
  rr(ax, rA[1], c, S, C.blue, a * 0.8); rr(rB[0], by, S, c, C.yel, a * 0.85);
  lab('rA', rA[0] + c / 2, rA[1] + S + 26, C.thr, a, 'center'); lab('rB', rB[0] - 12, rB[1] + 22, C.thr, a, 'right');
  const done = (k + clamp((ph - 0.35) / 0.65)) / 8, pulse = ph > 0.35 ? Math.sin(Math.PI * clamp((ph - 0.35) / 0.4)) : 0;
  rr(acc[0], acc[1], S, S, mixc('#e6edf9', C.thr, done * 0.85), a); gl(acc[0], acc[1], S, S, 8, 8, '#fbf9f4', a, 1.5);
  if (pulse > 0) rr(acc[0], acc[1], S, S, '#ffffff', a * 0.35 * pulse);
  sr(acc[0], acc[1], S, S, C.thr, 3, a);
  lab('acc 8 × 8 = rA ⊗ rB', acc[0] + S, acc[1] + S + 28, C.thr, a, 'right');
  lab(`k = ${k} / 8`, As[0], As[1] + S + 28, C.ink2, a);
  if (vec > 0.003) {
    for (const o of [0, 4]) { sr(rA[0] - 3, rA[1] + o * c - 3, c + 6, 4 * c + 6, C.ink, 4, a * vec); sr(rB[0] + o * c - 3, rB[1] - 3, 4 * c + 6, c + 6, C.ink, 4, a * vec); }
    lab('rA、rB 各用 2 个 float4 读入', As[0], As[1] + S + 56, C.ink, a * vec, 'left', 19);
    lab('float4：一条指令读 4 个相邻的数', As[0], As[1] + S + 82, C.ink, a * vec, 'left', 19);
  }
  return ph < 0.35 ? 0 : 1;
}
// warp 分块：As（128 × BK）、Bs（BK × 128）与 C 块 128 × 128；8 个 warp 各 64 × 32，线程 0 的 2 × 2 个 4 × 4 小块
function viewW(a, hl) {
  if (a <= 0.003) return;
  const X = 400, Y = 512, S = 300, q = S / 128, As = [X - 96, Y], Bs = [X, Y - 96];
  mat(As[0], As[1], 80, S, AC.aL, 1, 8, a); mat(Bs[0], Bs[1], S, 80, AC.bL, 8, 1, a); rr(X, Y, S, S, AC.cL, a);
  lab('As', As[0] + 40, As[1] + S + 26, C.blue, a, 'center'); lab('Bs', Bs[0] - 12, Bs[1] + 48, C.yelD, a, 'right');
  for (let wr = 0; wr < 2; wr++) for (let wc = 0; wc < 4; wc++) { const on = wr === 0 && wc === 1; rr(X + wc * 32 * q, Y + wr * 64 * q, 32 * q, 64 * q, on ? '#cfe3d6' : AC.cL, a); sr(X + wc * 32 * q, Y + wr * 64 * q, 32 * q, 64 * q, on ? C.use : '#fbf9f4', on ? 4 : 2, a); }
  const wx = X + 32 * q, wy = Y;
  rr(As[0], wy, 80, 64 * q, AC.aM, a * hl); rr(wx, Bs[1], 32 * q, 80, AC.bM, a * hl);
  for (const r0 of [0, 32]) { rr(As[0], wy + r0 * q, 80, 4 * q, C.blue, a * hl); for (const c0 of [0, 16]) rr(wx + c0 * q, wy + r0 * q, 4 * q, 4 * q, C.thr, a * hl); }
  for (const c0 of [0, 16]) rr(wx + c0 * q, Bs[1], 4 * q, 80, C.yel, a * hl);
  sr(X, Y, S, S, C.purple, 3, a);
  lab('C 块 128 × 128：8 个 warp，各 64 × 32', 110, 316, C.purple, a, 'left', 19);
  lab('线程 0 的 2 × 2 个 4 × 4 小块（行隔 32、列隔 16）', 110, 346, C.thr, a * hl, 'left', 19);
}
// 回指开场的外积视角：A 的一列 ⊗ B 的一行 = 一整张 4 × 4（同开场的配色）
function outerRecall(x, y, a) {
  if (a <= 0.003) return;
  const c = 22;
  rr(x - 12, y - 30, 300, 182, '#fbf9f4', a * 0.95); sr(x - 12, y - 30, 300, 182, C.lg, 1.5, a);
  lab('开场的外积视角', x, y - 4, C.ink2, a, 'left', 18);
  for (let r = 0; r < 4; r++) rr(x, y + 14 + r * c, c - 3, c - 3, C.blue, a);
  txt('⊗', x + 40, y + 70, { size: 26, weight: 800, alpha: a });
  for (let q = 0; q < 4; q++) rr(x + 70 + q * c, y + 14, c - 3, c - 3, C.yel, a);
  txt('=', x + 168, y + 70, { size: 26, weight: 800, alpha: a });
  for (let r = 0; r < 4; r++) for (let q = 0; q < 4; q++) rr(x + 190 + q * c, y + 14 + r * c, c - 3, c - 3, C.purple, a * 0.75);
  lab('rA ⊗ rB 是它的 8 × 8 版本', x, y + 136, C.thr, a, 'left', 17);
}
let CUR_K = 0;
// ---------- 第 10 幕：调参与 FP32 封顶 ----------
function sCeil(u) {
  const K = IDX('tuned'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  const z = 2 - 2 * p(u, bt(K, 2), bt(K, 2) + 1.6);
  nested(z, 1, { lab: ['三层：C → block 的块 → 线程的块', 'block 的块：128 × 128', '线程的块：8 × 8'], blkOn: 1, thrOn: 1, cells: 1, tag: false });
  layerBar(3, 1);
  let y = lines_(270, [['调参：dbuf 44.30 → tuned 44.53，只多 0.5%', 24, C.ink, 800]], b(0));
  const a1 = b(1), cs = 9;
  if (a1 > 0.003) {
    txt('每 SM 每个时钟周期的乘加次数', CP[0] + 6, y + 20, { size: 18, weight: 700, color: C.gray, alpha: a1 });
    for (let i = 0; i < 128; i++) rr(CP[0] + 6 + (i % 16) * cs, y + 36 + Math.floor(i / 16) * cs, cs - 2, cs - 2, C.blue, a1);
    txt('FP32 128：到顶', CP[0] + 6, y + 36 + 8 * cs + 30, { size: 22, weight: 800, color: C.blue, alpha: a1 });
    for (let i = 0; i < 2048; i++) rr(CP[0] + 190 + (i % 64) * 6.5, y + 36 + Math.floor(i / 64) * 6.5, 5, 5, C.teal, a1 * 0.9);
    txt('Tensor Core 2048（下集）', CP[0] + 190, y + 36 + 32 * 6.5 + 30, { size: 22, weight: 800, color: C.tealD, alpha: a1 });
  }
  if (b(2) > 0.003) { txt('8192 阶 · TFLOPS', CP[0] + 6, 778, { size: 17, weight: 700, color: C.gray, alpha: b(2) }); ladder(CP[0] + 20, 800, 560 * (STEPS.length - 1) / 6, 150, { upto: 7, ghost: 0, skip: [1], alpha: b(2), values: 'all', axis: false, size: 0.8 }); }
  gauge([[0.25, 'naive', C.red, 1], [8, 'smem', C.use, 1], [32, 'reg', C.thr, 1]], 1, { line300: b(3) });
}

const BY_TAG = { '尺子': sRuler, naive: sNaive, '天花板': sCeil, tuned: sCeil, '矩阵乘法': sIntro, '复杂度': sComplex, '纸面估算': s2, '线程层级': s3, '结尾': sNext, '回顾': sRecap, '各规模': s20, '未完成': s21 };
const DRAW = SCRIPT.map((s, k) => (s.kind === 'step' ? (u) => { CUR_K = k; sStep(u); } : s.stage ? stage(k) : BY_TAG[s.tag]));
function draw(t) {
  g.fillStyle = C.paper; g.fillRect(0, 0, W, H);
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  const u = t - AT[k];
  header(k, u);
  DRAW[k](u);
  subtitle(SCRIPT[k].beats, VO?.beats?.[k], u);
}
// 音效：每拍开始 pop；三栏幕换版本 whoosh、整幕阶梯 tick 串；阶梯长一级 land
const evAt = (tag, f) => { const k = IDX(tag); if (k >= 0) f(k); };
SCRIPT.forEach((s, k) => {
  if (!s.stage) return;
  let last = null;
  s.beats.forEach((b, j) => { const x = s.xs[j] || {}; ev(k, b[0], x.v && x.v !== last && last ? 'whoosh' : 'pop'); if (x.v) last = x.v; if (x.end) for (let i = 1; i <= 8; i++) ev(k, b[0] + 1.8 * i / 8, 'tick'); });
});
evAt('矩阵乘法', (k) => evBeats(k)); evAt('尺子', (k) => evBeats(k)); evAt('naive', (k) => evBeats(k)); evAt('tuned', (k) => evBeats(k)); SCRIPT.forEach((s, k) => { if (s.kind === 'step') evBeats(k); }); evAt('复杂度', (k) => evBeats(k)); evAt('纸面估算', (k) => evBeats(k)); evAt('线程层级', (k) => evBeats(k));
evAt('结尾', (k) => { evBeats(k); ev(k, bt(k, 2), 'whoosh'); });
evAt('回顾', (k) => { evBeats(k); for (let i = 7; i < 17; i++) ev(k, bt(k, 2) + (i - 6) * 0.17, 'tick'); });
evAt('各规模', (k) => { evBeats(k); for (let i = 0; i < 6; i++) ev(k, bt(k, 0) + 0.6 + i * 0.15, 'tick'); });
evAt('未完成', (k) => { evBeats(k); for (let i = 1; i <= 16; i++) ev(k, bt(k, 1) + 0.2 + (bt(k, 2) - 0.6 - bt(k, 1)) * i / 16, 'tick'); });
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
