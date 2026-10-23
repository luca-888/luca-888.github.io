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
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a, font: /^[\x00-\x7f]+$/.test(s.tag) ? MONO : ZH });
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
  txt('一个 SM（H100 有 132 个）', X(14), Y(21), { size: 13 * s, weight: 600, color: C.gray, alpha: a });
  const W = (st.warps || '................').padEnd(16, '.');
  for (let i = 0; i < 4; i++) {
    rr(X(14 + i * 94), Y(26), 86 * s, 44 * s, '#f7f4ee', a); txt('调度器 ' + i, X(20 + i * 94), Y(43), { size: 13 * s, weight: 600, color: C.gray, alpha: a });
    for (let j = 0; j < 4; j++) {
      const c = W[i * 4 + j], cx = X(30 + i * 94 + j * 18), cy = Y(58), r = 5.5 * s;
      const col = { r: C.use, w: C.hot, s: C.gray, p: C.thr, c: C.use }[c];
      g.globalAlpha = a; g.beginPath(); g.arc(cx, cy, r, 0, 7);
      if (c === 'c') { g.fillStyle = '#fff'; g.fill(); g.lineWidth = 2.5; g.strokeStyle = C.use; g.stroke(); }
      else { g.fillStyle = col || '#d9d3c5'; g.fill(); }
      g.globalAlpha = 1;
    }
  }
  for (const [key, [ux, uy, uw, uh, lab, sub]] of Object.entries(SMU)) {
    const busy = (st.busy || []).includes(key), hot = st.hot === key;
    rr(X(ux), Y(uy), uw * s, uh * s, hot ? '#f6d8c4' : busy ? '#cfe6d7' : '#f7f4ee', a);
    sr(X(ux), Y(uy), uw * s, uh * s, hot ? C.hot : busy ? C.use : C.lg, hot || busy ? 2.5 : 1.2, a);
    txt(lab, X(ux + 6), Y(uy + (sub ? 16 : uh / 2 + 5)), { size: 14 * s, weight: 700, color: hot ? C.hot : C.ink, alpha: a });
    if (sub) txt(sub, X(ux + 6), Y(uy + 31), { size: 11.5 * s, weight: 500, color: C.gray, alpha: a });
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
function s3(u) {
  const K = IDX('线程层级'), b = (j) => po(u, bt(K, j), bt(K, j) + 0.5);
  term(96, 280, 620, PROBE, { size: 20, lh: 34, hl: (i) => (i >= 2 ? b(0) : 0), alpha: b(0) });
  txt('SM = Streaming Multiprocessor', 96, 500, { size: 24, weight: 800, color: C.ink, alpha: b(0) });
  // grid：C 上 4 × 4 个 block（总览，每格 5 px），右侧放大 block (2, 1)
  const GX = 820, GY = 310, c = 5, bw = 16 * c, g = b(1);
  txt('grid：C 上 4 × 4 个 block（示意）', GX, GY - 18, { size: 22, weight: 700, color: C.gray, alpha: g });
  for (let by = 0; by < 4; by++) for (let bx = 0; bx < 4; bx++) {
    const on = bx === 2 && by === 1, X0 = GX + bx * (bw + 6), Y0 = GY + by * (bw + 6);
    grid(X0, Y0, 16, c, () => (on ? (u > bt(K, 2) ? '#cfe3d6' : C.warp) : null), g, 1);
    txt(`(${bx},${by})`, X0 + bw / 2, Y0 + bw / 2 + 7, { size: 18, weight: 700, color: on ? C.ink : C.gray, align: 'center', alpha: g, font: MONO });
  }
  const SBX = GX + 2 * (bw + 6), SBY = GY + (bw + 6);
  sr(SBX - 3, SBY - 3, bw + 6, bw + 6, C.ink, 3, g);
  // 放大的 block：16 × 16 个线程；warp 1 = 第 2、3 行
  const z = po(u, bt(K, 1) + 0.8, bt(K, 1) + 1.4), ZX = 1210, ZY = 310, zc = 17;
  line(SBX + bw + 3, SBY, ZX, ZY, C.lg, 1.5, z); line(SBX + bw + 3, SBY + bw, ZX, ZY + 16 * zc, C.lg, 1.5, z);
  grid(ZX, ZY, 16, zc, (r, q) => {
    if (u > bt(K, 4) && r === 3 && q === 5) return C.thr;
    if (u > bt(K, 3) && (r === 2 || r === 3)) return C.warp;
    return u > bt(K, 2) ? '#cfe3d6' : null;
  }, z, 2);
  txt('block (2, 1) 的 256 个线程', ZX, ZY + 16 * zc + 28, { size: 19, weight: 700, color: C.ink2, alpha: z });
  txt('threadIdx.x →', ZX + 130, ZY - 10, { size: 16, weight: 700, color: C.gray, alpha: z, font: MONO });
  const wl = b(3);
  line(ZX + 16 * zc + 6, ZY + 2 * zc, ZX + 16 * zc + 6, ZY + 4 * zc - 2, C.thr, 4, wl);
  txt('warp 1', ZX + 16 * zc + 14, ZY + 3 * zc + 7, { size: 17, weight: 800, color: C.thr, alpha: wl, font: MONO });
  // SM：block 整体落到一个 SM 上
  const s = b(2), SX = 96, SY = 560;
  rr(SX, SY, 620, 250, C.paper2, s);
  txt('一个 SM', SX + 20, SY + 40, { size: 26, weight: 800, alpha: s });
  rr(SX + 20, SY + 60, 150, 150, '#cfe3d6', s); txt('block (2,1)', SX + 95, SY + 140, { size: 20, weight: 800, color: C.use, align: 'center', alpha: s, font: MONO });
  arrow(SBX, SBY + bw, SX + 600, SY + 14, s, C.use, s, 3);
  tl(SX + 200, SY + 50, [['最多 2048 个线程 = 64 个 warp', 22, C.ink2, 700], ['一个 block 最多 1024 个线程', 22, C.ink2, 700], ['共享内存 228 KiB · 寄存器 65536 个', 22, C.use, 700], ['由驻留的 block 分用', 20, C.gray, 600]], s);
  // warp 与选中的线程
  const w = b(3), X = 1600;
  tl(X, 290, [['warp：threadIdx.x', 22, C.thr, 800], ['连续的 32 个线程', 22, C.thr, 800], ['16 × 16 的 block 中占两行', 19, C.ink2, 600]], w);
  const t = b(4);
  tl(X, 410, [['blockIdx  = (2, 1)', 22, C.ink, 700, MONO], ['threadIdx = (5, 3)', 22, C.thr, 700, MONO], ['序号 3 × 16 + 5 = 53', 20, C.ink2, 600, MONO], ['warp 53 / 32 = 1', 22, C.ink, 800, MONO], ['lane 53 % 32 = 21', 22, C.ink, 800, MONO], ['row = 1 × 16 + 3 = 19', 18, C.gray, 600, MONO], ['col = 2 × 16 + 5 = 37', 18, C.gray, 600, MONO]], t);
  // 合并访存：32 个地址汇成少数几次访问
  const m = b(5), AY = 900, AX = 820;
  txt('一个 warp 发出 32 个读取地址 → 合并为尽量少的内存访问', AX, AY - 18, { size: 22, weight: 800, color: C.hot, alpha: m });
  for (let i = 0; i < 64; i++) rr(AX + i * 15, AY, 12, 28, i >= 16 && i < 32 ? C.hot : C.cell, m);
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
const COL = { code: [40, 140, 660, 728], lay: [716, 140, 580, 728], hw: [1312, 140, 568, 728] };
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
  if (n && f) { const [a, b] = f; fi = Math.min(b, a + Math.floor((u - t0) / Math.max(0.01, t1 - t0) * (b - a + 1))); }
  let pv = null, vStart = 0;                      // 本版从哪一拍开始；上一个版本（换版本时代码淡入淡出）
  for (let i = j; i >= 0; i--) { const x = xs[i]; if (x && x.v && x.v !== v) { pv = x.v; break; } vStart = S.beats[i][0]; }
  return { x: { v, f, end: xs[j] && xs[j].end }, d, f: fi >= 0 ? d.frames[fi] : null, fi, t0, t1, j, pv, vStart };
}
// ---------- 左：完整 kernel 代码，当前行高亮，改动行标 + ----------
function codePanel(vn, f, a, u) {
  const d = FR[vn]; if (!d || a <= 0.003) return;
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
function layPanel(vn, f, a, fi = -1) {
  if (a <= 0.003) return;
  const d = FR[vn], [px, py, pw, ph] = COL.lay, s = 1.38, ox = px + 4, oy = py + 8;
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
  if (f && f.cnt) { const c = f.cnt; txt(`这个 block 累计从显存读 ${c.loads} 个数 · 做 ${c.fma} 次乘加 · 每读 1 个做 ${c.loads ? (c.fma / c.loads).toFixed(2) : 0} 次`, px + 14, py + ph - 18, { size: 17, weight: 700, color: C.ink2, alpha: a }); }
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
  ['warp 读 B 的 16 个连续', 'warp 读 B 的 16 个连续 float，合并为 2 个扇区；A 的一个数广播。\n每次乘加仍从 L1/L2 读取 2 个数，warp 多数时间在等待。'],
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
  smPanel(px + 6, py + 44, 1.38, { ...st, note: null }, a, u);
  const notes = fixNote(st.note).split('\n').filter(Boolean);
  notes.forEach((s, i) => wrapTxt(s, px + 18, py + 500 + i * 70, pw - 36, 21, i ? C.ink2 : (st.hot ? C.hot : C.use), a));
  const lg = [['执行', C.use], ['等显存', C.hot], ['等同步', C.gray], ['生产者', C.thr]];
  let lx = px + 18; lg.forEach(([s, col]) => { g.globalAlpha = a; g.fillStyle = col; g.beginPath(); g.arc(lx + 7, py + ph - 24, 7, 0, 7); g.fill(); g.globalAlpha = 1; txt(s, lx + 20, py + ph - 17, { size: 17, weight: 600, color: C.ink2, alpha: a }); lx += tw(s, 17, 600) + 48; });
}
function wrapTxt(s, x, y, w, size, col, a) {
  let line_ = '', yy = y;
  for (const ch of s) { if (tw(line_ + ch, size, 700) > w) { txt(line_, x, yy, { size, weight: 700, color: col, alpha: a }); line_ = ''; yy += size * 1.4; } line_ += ch; }
  if (line_) txt(line_, x, yy, { size, weight: 700, color: col, alpha: a });
}
// ---------- 底部：成绩阶梯（当前版本点亮） ----------
function bottomLadder(vn, a) {
  const i = vn === 'naive_T' ? -1 : STEP_OF[vn === 'tuned' ? 'dbuf' : vn];
  if (vn === 'naive_T') txt('naive_T（对照组）：0.96 TFLOPS（4096 阶）', 1850, 902, { size: 18, weight: 800, color: C.red, align: 'right', alpha: a });
  rr(40, 878, 1840, 112, '#fbf9f4', a);
  txt('8192 阶 · TFLOPS', 56, 902, { size: 16, weight: 700, color: C.gray, alpha: a });
  ladder(250, 886, 1600, 96, { upto: (i ?? -1) + 1, ghost: 0.5, alpha: a, values: 'last', axis: false, size: 0.75 });
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
    if (st.pv && fade < 1) codePanel(st.pv, null, 1 - fade, u);
    codePanel(vn, st.f, st.pv ? fade : 1, u);
    if (d && d.frames.length) layPanel(vn, st.f, 1, st.fi); else mapPanel(vn, 1, u, st.vStart);
    hwPanel(st.f ? st.f.hw : d.hw0, 1, u);
    bottomLadder(vn, 1);
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


const BY_TAG = { '矩阵乘法': sIntro, '纸面估算': s2, '线程层级': s3, '结尾': sNext, '回顾': sRecap, '各规模': s20, '未完成': s21 };
const DRAW = SCRIPT.map((s, k) => (s.stage ? stage(k) : BY_TAG[s.tag]));
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
evAt('矩阵乘法', (k) => evBeats(k)); evAt('纸面估算', (k) => evBeats(k)); evAt('线程层级', (k) => evBeats(k));
evAt('结尾', (k) => { evBeats(k); ev(k, bt(k, 2), 'whoosh'); });
evAt('回顾', (k) => { evBeats(k); for (let i = 7; i < 17; i++) ev(k, bt(k, 2) + (i - 6) * 0.17, 'tick'); });
evAt('各规模', (k) => { evBeats(k); for (let i = 0; i < 6; i++) ev(k, bt(k, 0) + 0.6 + i * 0.15, 'tick'); });
evAt('未完成', (k) => { evBeats(k); for (let i = 1; i <= 16; i++) ev(k, bt(k, 1) + 0.2 + (bt(k, 2) - 0.6 - bt(k, 1)) * i / 16, 'tick'); });
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
