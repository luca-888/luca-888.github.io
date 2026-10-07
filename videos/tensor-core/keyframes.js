// Tensor Core 影片的关键帧：每一幕结束时的画面（静态）。动画在分镜确认后按这些画面展开。
// 出处见 docs/tensor-core-notes.md；画面上标“示意”的为示意数据。
Object.assign(KC, { blueL: '#cdd6f6', yelL: '#fbeaa8', purpleL: '#ddd0fd', tealL: '#bfe6df', redL: '#f7c9c7' });

// 页眉：章节、问题标签、回答标题、右上角出处
function header(f) {
  if (f.chap) txt('Tensor Core · ' + f.chap, 96, 76, { size: 28, weight: 700, color: KC.ink2, base: 'alphabetic' });
  if (f.src) txt(f.src, 1824, 76, { size: 26, weight: 500, color: KC.gray, align: 'right', base: 'alphabetic' });
  if (f.tag) {
    g.font = `700 32px ${ZH}`; const w = g.measureText(f.tag).width + 36;
    rr(96, 100, w, 50, 0, f.col || KC.ink, 1);
    txt(f.tag, 114, 136, { size: 32, weight: 700, color: '#fff', base: 'alphabetic' });
  }
  if (f.head) rich(f.head, 96, 226, 60);
  if (f.note) rich(f.note, 96, 1020, 28, KC.gray, 500);
}
const label = (s, x, y, o = {}) => txt(s, x, y, { size: 28, weight: 700, color: KC.ink2, ...o });

const KF = [];
const kf = (f) => KF.push(f);

// ---------- 1 封面 ----------
kf({
  draw() {
    txt('Tensor Core', 96, 330, { size: 150, weight: 800, font: HN });
    rich('一条指令，算完一整块矩阵乘', 100, 450, 64);
    txt('从 CUDA core 讲到 Blackwell', 100, 530, { size: 36, weight: 500, color: KC.gray });
    // 左：一次乘加；右：一条 mma 的 16 × 8 × 16
    cube(1000, 700, 1, 1, 1, 30, { d: 1 });
    label('1 次乘加', 1015, 790, { align: 'center' });
    label('一条 FFMA', 1015, 830, { align: 'center', color: KC.gray, weight: 500 });
    cube(1460, 330, 16, 8, 16, 30, { d: 1 });
    label('2048 次乘加', 1580, 860, { align: 'center' });
    label('一条 mma（16 × 8 × 16）', 1580, 900, { align: 'center', color: KC.gray, weight: 500 });
  },
});

// ---------- 2 矩阵乘 ----------
kf({
  chap: '一 · 矩阵乘', tag: '模型里算得最多的是什么', col: KC.ink,
  head: '矩阵乘：每个输出是一行乘一列，{p:4 次乘加}',
  note: '示意尺寸。模型里的 linear 层都是这种运算，矩阵大得多，例如 4096 × 4096。',
  draw() {
    const s = 80, y = 330;
    grid(200, y, 4, 4, s, (i) => (i === 1 ? COL.a : KC.blueL));
    txt('×', 590, y + 160, { size: 70, align: 'center', color: KC.gray });
    grid(660, y, 4, 4, s, (i, j) => (j === 2 ? COL.b : KC.yelL));
    txt('=', 1050, y + 160, { size: 70, align: 'center', color: KC.gray });
    grid(1120, y, 4, 4, s, (i, j) => (i === 1 && j === 2 ? COL.c : KC.purpleL));
    box(1120 + 2 * s, y + s, s, s, KC.ink, 5);
    label('A', 360, y - 30, { align: 'center' }); label('B', 820, y - 30, { align: 'center' }); label('C', 1280, y - 30, { align: 'center' });
    // 跟踪 C[1][2]
    const yy = 790;
    rich('C[1][2] =', 200, yy, 40, KC.ink, 700);
    for (let k = 0; k < 4; k++) {
      const x = 420 + k * 330;
      rich(`{b:A[1][${k}]}·{y:B[${k}][2]}`, x, yy, 40, KC.ink, 700);
      if (k < 3) txt('+', x + 290, yy - 12, { size: 40, color: KC.gray, align: 'center' });
      badge(x + 100, yy + 50, k + 1);
    }
    rich('每一项是一次 {t:乘加（FMA）}：乘完立刻加进累加值。16 个输出 × 4 = {p:64 次乘加}', 200, 920, 34, KC.ink2, 500);
  },
});

// ---------- 3 CUDA core ----------
kf({
  chap: '二 · CUDA core', tag: 'GPU 原来怎么算', col: KC.ink,
  head: 'CUDA core：一条指令只做 {r:一次乘加}',
  note: '一个线程算 C[1][2]：4 条 FFMA。warp 里 32 个线程同时执行同一条指令，每个线程仍然只做 1 次。',
  draw() {
    const code = ['FFMA  c, a10, b02, c', 'FFMA  c, a11, b12, c', 'FFMA  c, a12, b22, c', 'FFMA  c, a13, b32, c'];
    code.forEach((s, i) => { badge(120, 340 + i * 90, i + 1); txt(s, 165, 342 + i * 90, { size: 34, font: MONO, weight: 600 }); });
    // 一条指令经过的步骤：只有“乘加”是要的计算
    const st = [['取指令', COL.ovh], ['译码', COL.ovh], ['读 3 个操作数', COL.ovh], ['乘加', COL.math], ['写回', COL.ovh]];
    let x = 720;
    label('每一条都要走一遍：', 720, 310);
    st.forEach(([s, c], i) => {
      const w = i === 2 ? 250 : 180;
      block(x, 350, w, 90, c, s, 1, { size: 30 });
      if (i < st.length - 1) arrow(x + w + 4, 395, x + w + 34, 395, 1, KC.ink2, 4, 1, 12);
      x += w + 40;
    });
    // 64 次乘加 = 64 次完整的流程
    label('4 × 4 × 4 的 64 次乘加 = 64 遍', 720, 520);
    for (let i = 0; i < 64; i++) {
      const cx = 720 + (i % 16) * 68, cy = 560 + Math.floor(i / 16) * 82;
      rr(cx, cy, 58, 70, 0, KC.redL); rr(cx + 22, cy, 14, 70, 0, COL.math);
    }
    rich('{r:红}：指令和操作数的开销    {t:绿}：乘加本身', 720, 925, 30, KC.ink2, 500);
  },
});

// ---------- 4 能量 ----------
kf({
  chap: '二 · CUDA core', tag: '能量花在哪', col: COL.ovh,
  head: '一次乘加 {t:1.5 pJ}，取指令和操作数 {r:30 pJ}',
  src: 'Dally, MLSys 2021（45 nm 下的估算）',
  note: '像寄快递：每一单的手续费差不多，一单只寄一件最亏，装满一箱才划算。手续费并不严格固定，HMMA 一单约 24 pJ。',
  draw() {
    const S = 8.6, x0 = 330;
    const row = (y, name, sub, m, o, txtM, txtO) => {
      label(name, 96, y + 30, { size: 40 }); txt(sub, 96, y + 78, { size: 26, color: KC.gray, weight: 500 });
      const wm = hbar(x0, y, m, S, 90, COL.math); hbar(x0 + wm, y, o, S, 90, KC.redL); hatchRect(x0 + wm, y, o * S, 90, 0.5);
      if (wm > 200) txt(txtM, x0 + wm / 2, y + 125, { size: 28, weight: 700, color: KC.tealD, align: 'center' });
      else txt(txtM, x0, y - 26, { size: 28, weight: 700, color: KC.tealD });
      txt(txtO, x0 + wm + o * S + 20, y + 45, { size: 28, weight: 700, color: KC.red });
    };
    row(360, 'HFMA', '一次乘加', 1.5, 30, '计算 1.5 pJ', '开销 30 pJ：计算的 20 倍');
    row(620, 'HMMA', '一条 Tensor Core 指令', 110, 110 * 0.22, '计算 110 pJ', '开销 22%');
    line(x0, 300, x0, 860, KC.ink, 3);
    rich('开销 = 取指令 + 译码 + 读操作数。指令做的事越多，开销摊得越薄。', 330, 930, 34, KC.ink2, 500);
  },
});

// ---------- 5 Tensor Core ----------
kf({
  chap: '三 · Tensor Core', tag: '怎样摊薄开销', col: COL.math,
  head: 'Tensor Core：每个时钟做完一个 {p:4 × 4 × 4} 矩阵乘',
  src: 'NVIDIA Volta 白皮书（V100）',
  note: '立方体的每一小格是一次乘加：左面是 A，顶面是 B，正面是累加结果 C。',
  draw() {
    cube(370, 420, 4, 4, 4, 80, { d: 1 });
    label('一个 Tensor Core', 530, 800, { align: 'center' });
    label('每个时钟 64 次乘加', 530, 845, { align: 'center', color: KC.gray, weight: 500 });
    // V100 的一个 SM：64 个 FP32 core 对 8 个 Tensor Core，每格是每个时钟的一次乘加
    const sq = 13, gap = 3;
    label('V100 的一个 SM，每个时钟：', 900, 330);
    const dots = (x, y, n, col) => { for (let i = 0; i < n; i++) rr(x + (i % 8) * (sq + gap), y + Math.floor(i / 8) * (sq + gap), sq, sq, 0, col); };
    dots(900, 390, 64, COL.math);
    label('64 个 FP32 core', 900, 550); label('64 次乘加', 900, 590, { color: KC.gray, weight: 500 });
    for (let t = 0; t < 8; t++) dots(1180 + (t % 4) * 140, 390 + Math.floor(t / 4) * 150, 64, COL.c);
    label('8 个 Tensor Core', 1180, 720); label('512 次乘加，是 FP32 core 的 8 倍', 1180, 760, { color: KC.gray, weight: 500 });
  },
});

// ---------- 6 数据复用 ----------
kf({
  chap: '三 · Tensor Core', tag: '为什么一条指令能做这么多', col: COL.math,
  head: '矩阵越大，每个读进来的数{p:被用的次数}越多',
  note: 'm × n × k 的矩阵乘读 m·k + k·n 个数，做 m·n·k 次乘加。A 的每个数要乘 B 的一整行，用 n 次。',
  draw() {
    const S = 0.5, x0 = 760;
    const rows = [['一次乘加', [1, 1, 1], 2, 1], ['4 × 4 × 4', [4, 4, 4], 32, 64], ['16 × 8 × 16', [16, 8, 16], 384, 2048]];
    label('读进来的数（A 与 B）', x0, 310, { color: KC.blue }); label('乘加次数', x0 + 420, 310, { color: KC.purple });
    rows.forEach(([name, [m, n, k], inp, fma], i) => {
      const y = 380 + i * 200, s = Math.min(36, 130 / Math.max(m, k));
      cube(560 - n * s / 2 + k * s * 0.22, y + 60 - m * s / 2 + k * s * 0.15, m, n, k, s, { d: 1, lines: m < 16 });
      label(name, 96, y + 30, { size: 36 });
      hbar(x0, y, inp, S, 44, COL.a); txt(String(inp), x0 + inp * S + 14, y + 22, { size: 28, weight: 700, color: KC.blue });
      hbar(x0, y + 54, fma, S, 44, COL.c); txt(String(fma), x0 + fma * S + 14, y + 76, { size: 28, weight: 700, color: KC.purple });
      const r = fma / inp;
      rich(`每个数用 {p:${r < 1 ? '0.5' : r.toFixed(r % 1 ? 1 : 0)}} 次`, 96, y + 95, 30, KC.ink2, 700);
    });
  },
});

// ---------- 7 mma.m16n8k16 ----------
// 寄存器分布（PTX ISA，m16n8k16 的 .bf16 输入、.f32 累加）：groupID = lane >> 2，tig = lane % 4
const laneA = (r, c) => (r % 8) * 4 + Math.floor((c % 8) / 2);
const laneB = (kk, n) => n * 4 + Math.floor((kk % 8) / 2);
const laneC = (r, c) => (r % 8) * 4 + Math.floor(c / 2);
kf({
  chap: '四 · 一条 warp 指令', tag: '程序怎样用 Tensor Core', col: COL.c,
  head: '一个 warp 的 32 个线程各拿一小块，{p:一条指令}算 16 × 8 × 16',
  src: 'PTX ISA 9.4 · mma.m16n8k16',
  note: '每格写的是存着它的线程编号。线程 0 拿 A 的 8 个数、B 的 4 个数，结果 C 的 4 个数也留在它的寄存器里。',
  draw() {
    const s = 27, y = 330;
    const cell = (x, yy, rows, cols, lane, solid, light) => {
      grid(x, yy, rows, cols, s, (i, j) => (lane(i, j) === 0 ? solid : light), { gap: 2 });
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const l = lane(i, j);
        txt(String(l), x + j * s + s / 2, yy + i * s + s / 2 + 1, { size: 12, align: 'center', font: HN, weight: l === 0 ? 700 : 500, color: l === 0 ? '#fff' : KC.ink2 });
      }
    };
    cell(150, y, 16, 16, laneA, COL.a, KC.blueL);
    txt('×', 625, y + 216, { size: 60, align: 'center', color: KC.gray });
    cell(680, y, 16, 8, laneB, COL.b, KC.yelL);
    txt('=', 960, y + 216, { size: 60, align: 'center', color: KC.gray });
    cell(1010, y, 16, 8, laneC, COL.c, KC.purpleL);
    label('A  16 × 16', 150, y - 28); label('B  16 × 8', 680, y - 28); label('C  16 × 8', 1010, y - 28);
    const x = 1340;
    rich('线程 0：{b:A 8 个}、{y:B 4 个}、{p:C 4 个}', x, 400, 34, KC.ink, 700);
    rich('32 个线程合起来正好是整块', x, 460, 30, KC.ink2, 500);
    line(x, 520, 1824, 520, KC.lg, 2);
    rich('一条指令 = 16 × 8 × 16', x, 590, 34, KC.ink, 700);
    rich('= {p:2048 次乘加}', x, 650, 44, KC.ink, 800);
    rich('用 FFMA 要 64 条 warp 指令', x, 710, 30, KC.ink2, 500);
    txt('mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32', 150, 830, { size: 26, font: MONO, weight: 600, color: KC.ink2 });
  },
});

// ---------- 8 分块 ----------
kf({
  chap: '五 · 大矩阵怎么算', tag: '整个矩阵怎样交给 Tensor Core', col: KC.ink,
  head: '切成块：每个 C 块沿 K 方向{p:一步步累加}',
  src: '示意：CUTLASS 的分块层级',
  note: '一个 SM 负责一个 C 块；每一步从显存取 A、B 各一块到 shared memory，再分给 warp 做 mma。',
  draw() {
    const T = 40, n = 8, cx = 500, cy = 630;           // 8 × 8 块的示意
    grid(cx, cy - n * T - 30, n, n, T, () => KC.paper2, { gap: 3 });        // B：K × N
    grid(cx - n * T - 30, cy, n, n, T, () => KC.paper2, { gap: 3 });    // A：M × K（转置示意）
    grid(cx, cy, n, n, T, () => KC.paper2, { gap: 3 });
    // 跟踪 C 的第 (2, 5) 块：A 的第 2 行块 × B 的第 5 列块，当前走到 K 的第 3 步
    const bi = 2, bj = 5, ks = 3;
    for (let k = 0; k < n; k++) {
      rr(cx - n * T - 30 + k * T + 1.5, cy + bi * T + 1.5, T - 3, T - 3, 0, k === ks ? COL.a : KC.blueL);
      rr(cx + bj * T + 1.5, cy - n * T - 30 + k * T + 1.5, T - 3, T - 3, 0, k === ks ? COL.b : KC.yelL);
    }
    rr(cx + bj * T + 1.5, cy + bi * T + 1.5, T - 3, T - 3, 0, COL.c);
    label('B', cx + n * T + 30, cy - n * T / 2 - 30); label('A', cx - n * T - 60, cy + n * T / 2, { align: 'right' });
    label('C', cx + n * T + 30, cy + n * T / 2);
    arrow(cx - n * T - 30 + ks * T + T / 2, cy + bi * T + T / 2, cx + bj * T + 8, cy + bi * T + T / 2, 1, KC.ink, 3, 0.8, 14);
    arrow(cx + bj * T + T / 2, cy - n * T - 30 + ks * T + T / 2, cx + bj * T + T / 2, cy + bi * T + 8, 1, KC.ink, 3, 0.8, 14);
    // 右：存储层级
    const x = 1220, lv = [['显存（HBM）', '整个 A、B、C', KC.ink2], ['shared memory', '一个 SM：A、B 各一块', KC.gray], ['寄存器', '每个 warp：一小片', KC.gray], ['Tensor Core', 'mma：16 × 8 × 16', COL.math]];
    lv.forEach(([a, b, c], i) => {
      const y = 330 + i * 160;
      block(x, y, 300, 100, c, a, 1, { size: 30 }); txt(b, x + 330, y + 50, { size: 28, weight: 500, color: KC.ink2 });
      if (i < 3) arrow(x + 150, y + 104, x + 150, y + 156, 1, KC.ink, 4, 1, 14);
    });
    badge(x - 40, 330 + 130, 1); badge(x - 40, 330 + 290, 2); badge(x - 40, 330 + 450, 3);
  },
});

// ---------- 9 每代翻倍 ----------
kf({
  chap: '六 · 一代比一代快', tag: '后来的 GPU 怎么变', col: KC.ink,
  head: '每个 SM 每个时钟的乘加次数，{p:每代翻倍}',
  src: 'NVIDIA V100 / A100 / H100 白皮书',
  note: 'dense FP16 / BF16。B200 一行由整卡 2.25 PFLOPS、148 个 SM 与 boost clock 推算。',
  draw() {
    const S = 0.28, x0 = 400;
    const gens = [['V100', 2017, 512, 64], ['A100', 2020, 1024, 64], ['H100', 2022, 2048, 128], ['B200', 2024, 4096, 128]];
    gens.forEach(([n, yr, tc, fp], i) => {
      const y = 320 + i * 160;
      label(n, 96, y + 30, { size: 40 }); txt(String(yr), 96, y + 76, { size: 26, color: KC.gray, weight: 500 });
      hbar(x0, y, tc, S, 56, COL.c); txt(`${tc}`, x0 + tc * S + 16, y + 30, { size: 32, weight: 700, color: KC.purple });
      hbar(x0, y + 66, fp, S, 30, COL.math); txt(`${fp}`, x0 + fp * S + 14, y + 83, { size: 26, weight: 700, color: KC.tealD });
      txt(`${tc / fp} 倍`, 1824, y + 46, { size: 34, weight: 700, align: 'right' });
    });
    rich('{p:Tensor Core}    {t:FP32 core}', x0, 975, 30, KC.ink2, 700);
  },
});

// ---------- 10 数据怎么进来 ----------
kf({
  chap: '六 · 一代比一代快', tag: '算得这么快，数据跟得上吗', col: COL.ovh,
  head: '数据不再绕道寄存器，{t:直接从 shared memory 进 Tensor Core}',
  src: 'PTX ISA 9.4；Colfax（2024）；SemiAnalysis（2025）',
  note: '每个 SM 的寄存器共 256 KB。Blackwell 另加 256 KB 的 Tensor Memory（128 行 × 512 列 × 4 B），专放累加结果。',
  draw() {
    const cols = [
      ['A100 · mma.sync', '一个 warp（32 线程）一起发', { A: 'reg', B: 'reg', C: 'reg' }],
      ['H100 · wgmma', '4 个 warp（128 线程）一起发，异步', { A: 'smem', B: 'smem', C: 'reg' }],
      ['B200 · tcgen05.mma', '1 个线程发，异步', { A: 'smem', B: 'smem', C: 'tmem' }],
    ];
    cols.forEach(([name, issue, loc], i) => {
      const x = 110 + i * 590, w = 520;
      label(name, x, 310, { size: 34 }); txt(issue, x, 352, { size: 26, color: KC.gray, weight: 500 });
      const Y = { smem: 400, reg: 560, tc: 720, tmem: 880 };
      rr(x, Y.smem, w, 110, 0, KC.paper2); label('shared memory', x + 20, Y.smem + 30, { size: 24, color: KC.gray });
      rr(x, Y.reg, w, 110, 0, KC.paper2); label('寄存器', x + 20, Y.reg + 30, { size: 24, color: KC.gray });
      block(x, Y.tc, w, 110, COL.math, 'Tensor Core', 1, { size: 32 });
      if (loc.C === 'tmem') { rr(x, Y.tmem, w, 90, 0, KC.purpleL); label('Tensor Memory', x + 20, Y.tmem + 28, { size: 24, color: KC.purple }); }
      // 三个操作数的位置与走向
      const ops = [['A', COL.a, 70], ['B', COL.b, 210], ['C', COL.c, 350]];
      for (const [k, c, ox] of ops) {
        const where = k === 'A' || k === 'B' ? (loc[k] === 'reg' ? 'reg' : 'smem') : loc.C;
        const yy = where === 'tmem' ? Y.tmem + 22 : Y[where] + 50;
        block(x + ox, yy, 100, 54, c, k, 1, { size: 28, color: k === 'B' ? KC.ink : '#fff' });
        if (k !== 'C') { if (where === 'reg') { block(x + ox, Y.smem + 50, 100, 54, k === 'A' ? KC.blueL : KC.yelL, k, 1, { size: 28, color: KC.ink2 }); arrow(x + ox + 50, Y.smem + 108, x + ox + 50, Y.reg + 46, 1, KC.ink2, 3, 1, 10); } arrow(x + ox + 50, yy + 58, x + ox + 50, Y.tc - 4, 1, KC.ink, 4, 1, 12); }
        else if (where === 'tmem') arrow(x + ox + 50, Y.tc + 114, x + ox + 50, yy - 4, 1, KC.ink, 4, 1, 12);
        else arrow(x + ox + 50, yy + 58, x + ox + 50, Y.tc - 4, 1, KC.ink, 4, 1, 12);
      }
    });
  },
});

// ---------- 11 精度 ----------
kf({
  chap: '七 · 更少的 bit', tag: '还能更快吗', col: KC.ink,
  head: '位数减半，{p:吞吐翻倍}',
  src: 'NVIDIA H100、HGX B200 规格（dense）',
  note: '同一块 GPU 的 FP32 只走 CUDA core。FP4 是 Blackwell 才有的格式。单位 TFLOPS。',
  draw() {
    const S = 0.135, x0 = 470;
    const rows = [['FP32', 67, 75, true], ['TF32', 495, 1125], ['BF16', 989, 2250], ['FP8', 1979, 4500], ['FP4', null, 9000]];
    label('H100', x0, 300, { color: KC.blue }); label('B200', x0 + 160, 300, { color: KC.purple });
    rows.forEach(([n, h, b, cuda], i) => {
      const y = 340 + i * 128;
      label(n, 96, y + 40, { size: 40 }); if (cuda) txt('CUDA core', 220, y + 44, { size: 26, color: KC.gray, weight: 500 });
      if (h) { hbar(x0, y, h, S, 40, COL.a); txt(h.toLocaleString(), x0 + h * S + 14, y + 20, { size: 26, weight: 700, color: KC.blue }); }
      else txt('—', x0 + 6, y + 20, { size: 26, weight: 700, color: KC.gray });
      hbar(x0, y + 48, b, S, 40, COL.c); txt(b.toLocaleString(), x0 + b * S + 14, y + 68, { size: 26, weight: 700, color: KC.purple });
    });
  },
});

// ---------- 12 实测（预留） ----------
kf({
  chap: '七 · 更少的 bit', tag: '实测（预留）', col: KC.gray,
  head: '同一个矩阵乘，换成 Tensor Core 要多久',
  note: '预留：H100 上一个 8192 × 8192 × 8192 的矩阵乘，FP32（CUDA core）与 TF32 / BF16 / FP8（Tensor Core）各跑一次，按真实用时播放。',
  draw() {
    ['FP32 · CUDA core', 'TF32', 'BF16', 'FP8'].forEach((n, i) => {
      const y = 340 + i * 150;
      label(n, 96, y + 40, { size: 36 });
      box(520, y, 1300, 80, KC.gray, 3, 1, [12, 10]);
      txt('待测', 550, y + 40, { size: 28, color: KC.gray, weight: 700 });
    });
  },
});

// ---------- 13 exp ----------
kf({
  chap: '八 · 代价', tag: '矩阵乘快了，别的呢', col: COL.ovh,
  head: '矩阵乘之外的运算成了瓶颈：{r:exp 慢 256 倍}',
  src: 'FlashAttention-3（Shah et al., 2024）；NVIDIA Blackwell Ultra 技术博客',
  note: 'attention 的 softmax 要对每个分数算一次 exp。exp 的计算量远少于矩阵乘，但吞吐低 256 倍，仍能占到矩阵乘一半的时间。',
  draw() {
    const S = 1.3, x0 = 420;
    label('H100', 96, 300, { color: KC.gray });
    label('矩阵乘', 96, 380, { size: 40 }); hbar(x0, 350, 989, S, 70, COL.math); txt('989 TFLOPS', x0 + 989 * S - 20, 385, { size: 30, weight: 700, color: '#fff', align: 'right' });
    label('exp', 96, 500, { size: 40 }); hbar(x0, 470, 3.9, S, 70, COL.ovh); txt('3.9 TFLOPS', x0 + 30, 505, { size: 30, weight: 700, color: KC.red });
    line(x0, 330, x0, 560, KC.ink, 3);
    rich('① FlashAttention-3：让一块的 exp 和下一块的矩阵乘{t:同时进行}', 96, 700, 36, KC.ink, 700);
    rich('② Blackwell Ultra：attention 用到的 SFU 指令{t:吞吐翻倍}', 96, 790, 36, KC.ink, 700);
  },
});

// ---------- 14 取舍 ----------
kf({
  chap: '八 · 代价', tag: '芯片面积给了谁', col: KC.ink,
  head: 'Tensor Core 越来越偏向 AI 用的{p:低精度}',
  src: 'NVIDIA HGX B200 / B300 规格（8 卡，dense）',
  note: '每一行以 B200 为 1 画长度。B300 把 FP64 与 INT8 的吞吐大幅降低，换来更多的 FP4。',
  draw() {
    const x0 = 520, L = 700;
    const rows = [['FP4', '72 PFLOPS', '108 PFLOPS', 1.5], ['INT8', '72 POPS', '3 POPS', 3 / 72], ['FP64', '296 TFLOPS', '10 TFLOPS', 10 / 296]];
    label('B200', x0, 300, { color: KC.blue }); label('B300', x0 + 160, 300, { color: KC.purple });
    rows.forEach(([n, a, b, r], i) => {
      const y = 350 + i * 190;
      label(n, 96, y + 50, { size: 44 });
      hbar(x0, y, 1, L, 50, COL.a); txt(a, x0 + L + 16, y + 25, { size: 28, weight: 700, color: KC.blue });
      hbar(x0, y + 60, r, L, 50, COL.c); txt(b, x0 + r * L + 16, y + 85, { size: 28, weight: 700, color: KC.purple });
    });
  },
});

// ---------- 15 结尾 ----------
kf({
  chap: '结尾', tag: '回到开头', col: KC.ink,
  head: '同样 2048 次乘加：{r:64 条}指令对 {p:1 条}',
  draw() {
    for (let i = 0; i < 64; i++) {
      const cx = 140 + (i % 8) * 78, cy = 330 + Math.floor(i / 8) * 76;
      rr(cx, cy, 66, 64, 0, KC.redL); rr(cx + 26, cy, 14, 64, 0, COL.math);
    }
    label('CUDA core：64 条 warp 级 FFMA', 140, 975);
    cube(1400, 400, 16, 8, 16, 30, { d: 1 });
    label('Tensor Core：1 条 mma', 1520, 975, { align: 'center' });
  },
});
