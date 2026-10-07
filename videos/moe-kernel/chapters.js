// 第二至七章与结尾。依赖 core.js 与 film.html 里的配色、rich / box / arrowR / tokenTag / header。
// 各章用本地时间 u = t - 起点；场景标题登记进 SC，由 header() 统一绘制。

const DURATION = 460;
// 第一章在“切成 128 个 expert”之后插入一段（MoE 层的位置与一个 expert 的结构）；其后所有内容整体后移 INS_D 秒
const INS_T = 20.8, INS_D = 30;
SC.push(
  { r: [45, 75], tag: '逐个 expert', col: KC.ink, head: '最直接的写法：每个 expert 单独算，{r:231 个}小 kernel', src: '示意代码；路由为示意数据' },
  { r: [75, 91], tag: 'Tensor Core', col: KC.ink, head: '每个 kernel 又太小：Tensor Core 一次至少算 {r:16 行}', src: 'BF16 mma 指令形状 m16n8k16' },
  { r: [91, 105], tag: '同步', col: KC.ink, head: '形状由路由决定：{r:CPU 要等} GPU 算完路由才能 launch', src: '' },
  { r: [105, 125], tag: '第一步 · 展平', col: KC.purple, head: '每个 token 选 3 个 expert，展平成 {p:12 个}（token, expert）对', src: '例子改编自 vLLM moe_align_block_size 的源码注释' },
  { r: [125, 147], tag: '第二步 · 排序', col: KC.purple, head: '按 expert 排序：同一个 expert 的{t:排到一起}', src: '例子改编自 vLLM moe_align_block_size 的源码注释' },
  { r: [147, 167], tag: '第三步 · 补齐', col: KC.purple, head: '每段补齐到 block 的整数倍（这里 {r:block = 4}）', src: '例子改编自 vLLM moe_align_block_size 的源码注释' },
  { r: [167, 195], tag: '排队结果', col: KC.purple, head: '排完得到{b:三张表}，kernel 只看这三张表', src: '例子改编自 vLLM moe_align_block_size 的源码注释' },
  { r: [195, 225], tag: 'fused_moe_kernel', col: KC.teal, head: '每个 program 先查表：属于{t:哪个 expert}、取{p:哪几行}', src: 'vLLM v0.31.0 · fused_moe.py' },
  { r: [225, 243], tag: 'gather', col: KC.teal, head: 'token 不用先搬到一起：读 A 时{p:按编号直接取}', src: 'vLLM v0.31.0 · fused_moe.py' },
  { r: [243, 265], tag: '一次 launch', col: KC.teal, head: '所有 expert {r:一次 launch} 算完，CPU 也不用等', src: 'vLLM v0.31.0 · fused_moe.py' },
  { r: [265, 315], tag: '流水线', col: KC.blue, head: '一个 MoE 层：两次 {t:grouped GEMM}，中间一次激活', src: 'vLLM v0.31.0 · fused_experts_impl；16 个 token 为例' },
  { r: [315, 343], tag: '读多少权重', col: KC.red, head: 'token 越多，要读的 expert 越多：{r:64 个 token} 就几乎读满', src: '按均匀路由估算；每层、BF16' },
  { r: [343, 365], tag: '每个权重用几次', col: KC.red, head: 'decode 时每个权重只用一两次：瓶颈在{b:读显存}', src: '按均匀路由估算；补齐行数为示意路由' },
  { r: [365, 385], tag: 'tile 分档', col: KC.red, head: 'vLLM 按 token 数换配置：token 少用{t:小 tile}', src: 'vLLM v0.31.0 · get_default_config（BF16 默认值）' },
  { r: [385, 409], tag: 'DeepGEMM', col: KC.ink, head: 'DeepGEMM：prefill 用 {t:contiguous}，decode 用 {b:masked}', src: 'DeepGEMM README' },
  { r: [409, 428], tag: 'MegaBlocks', col: KC.ink, head: 'MegaBlocks：不丢 token，也不按 {r:capacity} 补齐', src: 'MegaBlocks（Gale et al., 2022）' },
  { r: [428, 445], tag: 'vLLM 后端', col: KC.ink, head: 'vLLM 按硬件选 kernel：Hopper 上{t:默认 Triton}', src: 'vLLM v0.31.0 · oracle/unquantized.py（BF16）' },
);

// ---------- 共用小工具 ----------
const EXC = [KC.blue, KC.teal, KC.yel, KC.red];                  // 第三、四章例子里 4 个 expert 的颜色
const onC = (c) => (c === KC.yel ? KC.ink : '#fff');
const HP = {};
function hatchC(col, bg = KC.paper2) {
  const k = col + bg; if (HP[k]) return HP[k];
  const c = document.createElement('canvas'); c.width = c.height = 14; const q = c.getContext('2d');
  q.fillStyle = bg; q.fillRect(0, 0, 14, 14); q.strokeStyle = col; q.lineWidth = 3;
  q.beginPath(); for (let o = -14; o <= 28; o += 14) { q.moveTo(o, 14); q.lineTo(o + 14, 0); } q.stroke();
  return (HP[k] = g.createPattern(c, 'repeat'));
}
function hrect(x, y, w, h, col, a, bg) { if (a <= 0.003 || w <= 0) return; g.globalAlpha = a; g.fillStyle = hatchC(col, bg); g.fillRect(x, y, w, h); g.globalAlpha = 1; }
function cell(x, y, w, h, fill, label, a, { size = 26, font = HN, color } = {}) {
  rr(x, y, w, h, 0, fill, a);
  if (label !== undefined) txt(String(label), x + w / 2, y + h / 2 + 1, { size, weight: 700, color: color || onC(fill), align: 'center', font, alpha: a });
}
function note(t, a, b, s, size = 34) { const k = win(t, a, b, 0.4); if (k > 0) rich(s, 96, 992, size, KC.ink, 700, k); }
function small(s, x, y, a, { size = 26, color = KC.gray, align = 'left', weight = 600, font = ZH } = {}) { txt(s, x, y, { size, color, align, weight, font, alpha: a }); }
function mono(s, x, y, a, { size = 24, color = KC.ink, align = 'left', weight = 600 } = {}) { txt(s, x, y, { size, color, align, weight, font: '"SF Mono",Menlo,monospace', alpha: a }); }
function arrow(x0, y0, x1, y1, k, col, a = 1, lw = 5) {
  if (k <= 0 || a <= 0.003) return;
  const ang = Math.atan2(y1 - y0, x1 - x0), x = mix(x0, x1, k), y = mix(y0, y1, k);
  line(x0, y0, x - 16 * Math.cos(ang), y - 16 * Math.sin(ang), col, lw, a);
  g.globalAlpha = a; g.fillStyle = col; g.beginPath(); g.moveTo(x, y);
  g.lineTo(x - 22 * Math.cos(ang - 0.5), y - 22 * Math.sin(ang - 0.5)); g.lineTo(x - 22 * Math.cos(ang + 0.5), y - 22 * Math.sin(ang + 0.5)); g.fill(); g.globalAlpha = 1;
}
const ACTIVE = counts.filter((c) => c > 0).length;                // 示意路由下 16 个 token 激活的 expert 数

// ================= 第二章：逐个 expert 算（45–105） =================
const CODE = [
  ['for e in range(128):', KC.ink],
  ['    idx = 分到 e 的 token', KC.ink],
  ['    if len(idx) == 0: continue', KC.gray],
  ['    h = silu(x[idx] @ Wg[e]) * (x[idx] @ Wu[e])', KC.ink],
  ['    y[idx] += g[idx, e] * (h @ Wd[e])', KC.ink],
];
const TWO = counts.indexOf(2);                                      // 一个恰好分到 2 个 token 的 expert
function ch2(t) {
  const u = t - 45; if (u < 0 || u > 60.5) return;
  const out = 1 - p(u, 59.4, 60);
  // 2a 代码与时间轴
  const a1 = p(u, 0.2, 0.8) * (1 - p(u, 29.4, 30));
  if (a1 > 0) {
    rr(96, 300, 860, 300, 0, KC.paper2, a1); rr(96, 300, 8, 300, 0, KC.ink, a1);
    small('示意代码', 124, 330, a1);
    CODE.forEach(([l, c], i) => mono(l, 124, 380 + i * 46, a1 * p(u, 0.8 + i * 0.4, 1.3 + i * 0.4), { size: 25, color: c }));
    const L0 = 1020, L1 = 1824, n = ACTIVE * 3, step = (L1 - L0) / n;
    small('GPU', L0, 330, a1, { color: KC.ink, weight: 700 });
    line(L0, 440, L1, 440, KC.lg, 2, a1);
    const shown = Math.floor(n * lin(u, 4, 11));
    for (let i = 0; i < shown; i++) rr(L0 + i * step, 370, Math.max(1.2, step * 0.55), 70, 0, i % 3 === 2 ? KC.tealD : KC.teal, a1);
    rich(`${ACTIVE} 个 expert × 3 次 GEMM = {r:${n} 个 kernel}`, L0, 520, 36, KC.ink, 700, a1 * p(u, 10, 10.8));
    small('还没算 gather、scatter 和激活', L0, 570, a1 * p(u, 11.5, 12.2));
    note(u, 14, 29.6, '每个 kernel 只算这个 expert 分到的那一两个 token');
  }
  // 2b Tensor Core 的 16 行
  const a2 = p(u, 30, 30.6) * (1 - p(u, 45.4, 46));
  if (a2 > 0) {
    small(`expert ${TWO} 分到 2 个 token`, 96, 330, a2, { color: KC.ink, weight: 700, size: 30 });
    tokenTag(96, 400, a2, 'token a', 150); tokenTag(96, 460, a2, 'token b', 150);
    arrowR(270, 430, 600, p(u, 31.5, 32.3), KC.ink, a2);
    const TX0 = 640, TY0 = 300, CS = 26;
    for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
      const x = TX0 + c * CS, y = TY0 + r * CS, k = p(u, 32 + r * 0.05, 32.4 + r * 0.05);
      if (r < 2) rr(x + 1, y + 1, CS - 2, CS - 2, 0, KC.purple, a2 * k);
      else { g.globalAlpha = a2 * k; g.fillStyle = hatchC(KC.red); g.fillRect(x + 1, y + 1, CS - 2, CS - 2); g.globalAlpha = 1; }
    }
    box(TX0, TY0, 16 * CS, 16 * CS, KC.ink, 4, a2 * p(u, 32, 32.5));
    small('Tensor Core 的一个 tile：16 行', TX0, TY0 + 16 * CS + 34, a2 * p(u, 33, 33.6), { color: KC.ink });
    rich('有用的只有 {p:2 行}', 1150, 400, 44, KC.ink, 700, a2 * p(u, 34, 34.6));
    rich('其余 {r:14 行}也要算，结果丢掉', 1150, 470, 44, KC.ink, 700, a2 * p(u, 35, 35.6));
    small('BF16 的 mma 指令一次算 16 × 8 × 16', 1150, 540, a2 * p(u, 36, 36.6), { size: 28 });
    note(u, 38, 45.6, `示意路由下，${counts.filter((c) => c > 0 && c <= 2).length} 个被选中的 expert 只分到 1–2 个 token`);
  }
  // 2c CPU 等 GPU
  const a3 = p(u, 46, 46.6) * out;
  if (a3 > 0) {
    const X0 = 300, Y_CPU = 380, Y_GPU = 560, S = 70;            // S：每秒（示意）对应的像素
    small('CPU', 96, Y_CPU + 30, a3, { color: KC.ink, weight: 700, size: 32 });
    small('GPU', 96, Y_GPU + 30, a3, { color: KC.ink, weight: 700, size: 32 });
    line(X0, Y_CPU + 60, 1824, Y_CPU + 60, KC.lg, 2, a3); line(X0, Y_GPU + 60, 1824, Y_GPU + 60, KC.lg, 2, a3);
    const k1 = p(u, 47, 48.2);
    rr(X0, Y_GPU, 300 * k1, 60, 0, KC.blue, a3);
    txt('router', X0 + 150, Y_GPU + 30, { size: 26, weight: 700, color: '#fff', align: 'center', font: HN, alpha: a3 * k1 });
    hrect(X0, Y_CPU, 300 * k1, 60, KC.red, a3);
    small('CPU 等待', X0 + 150, Y_CPU - 22, a3 * k1, { color: KC.red, align: 'center', weight: 700 });
    arrow(X0 + 300, Y_GPU - 4, X0 + 340, Y_CPU + 64, p(u, 48.4, 49.0), KC.ink, a3);
    small('路由结果拷回 CPU', X0 + 360, (Y_CPU + Y_GPU) / 2 + 30, a3 * p(u, 48.8, 49.4), { color: KC.ink });
    const n = Math.floor(30 * lin(u, 49.4, 52));
    for (let i = 0; i < n; i++) {
      rr(X0 + 360 + i * 38, Y_CPU + 14, 6, 32, 0, KC.ink, a3);
      rr(X0 + 380 + i * 38, Y_GPU, 22, 60, 0, KC.teal, a3);
    }
    small('按每个 expert 的 token 数逐个 launch', X0 + 360, Y_CPU - 22, a3 * p(u, 50, 50.6), { color: KC.ink });
    small('kernel 的个数与形状取决于路由，CUDA Graph 也录不下来', 96, 760, a3 * p(u, 52, 52.6), { size: 32, color: KC.ink });
    note(u, 54, 60, '三个问题：kernel {r:太多}、每个{r:太小}、CPU {r:要等}', 40);
  }
}

// ================= 第三章：按 expert 排队（105–195） =================
const TK = [[1, 2, 3], [0, 1, 3], [0, 2, 3], [0, 1, 2]];              // 4 个 token，top-3，4 个 expert
const FLAT = TK.flat();
const SORTED = [0, 1, 2, 3].flatMap((e) => FLAT.map((v, i) => [v, i]).filter(([v]) => v === e).map(([, i]) => i));
const SW = 92, SH = 70, ROWY = 650;
const flatX = (pos) => 360 + pos * 100;
const groupX = (gi) => 116 + gi * 432;
const sortedX = (pos) => { const k = SORTED.indexOf(pos), e = FLAT[pos], j = k - SORTED.findIndex((q) => FLAT[q] === e); return groupX(e) + j * 100; };
function ch3(t) {
  const u = t - 105; if (u < 0 || u > 90.5) return;
  const A = p(u, 0, 0.5) * (1 - p(u, 89.4, 90));
  // 例子的 topk_ids 表
  const ma = A * (1 - p(u, 20, 21));
  if (ma > 0) {
    small('topk_ids：4 个 token，各选 3 个 expert', 360, 300, ma, { color: KC.ink, weight: 700 });
    TK.forEach((row, i) => {
      const y = 340 + i * 60;
      tokenTag(360, y + 24, ma * p(u, 0.4 + i * 0.15, 0.8 + i * 0.15), 't' + i, 90);
      row.forEach((e, j) => { if (u < 3 + (i * 3 + j) * 0.15) cell(470 + j * 100, y, SW, 48, EXC[e], 'E' + e, ma * p(u, 0.6 + i * 0.15, 1 + i * 0.15)); });
    });
    // 图例
    [0, 1, 2, 3].forEach((e) => { cell(1250 + e * 140, 340, 120, 48, EXC[e], 'expert ' + e, ma * p(u, 1, 1.6), { size: 22 }); });
  }
  // 12 个槽：从表里飞到展平的一行，再排序到各组
  const sortK = p(u, 22, 26);
  for (let pos = 0; pos < 12; pos++) {
    const i = Math.floor(pos / 3), j = pos % 3, e = FLAT[pos];
    const fly = p(u, 3 + pos * 0.15, 3.8 + pos * 0.15); if (fly <= 0) continue;
    const sx = mix(470 + j * 100, flatX(pos), fly), sy = mix(340 + i * 60, ROWY, fly);
    const x = mix(sx, sortedX(pos), sortK), y = sy;
    const sh = mix(48, SH, fly);
    cell(x, y, SW, sh, EXC[e], fly < 0.9 ? 'E' + e : pos, A, { size: fly < 0.9 ? 26 : 32 });
    small('t' + Math.floor(pos / 3), x + SW / 2, ROWY + SH + 28, A * p(u, 9, 9.6), { align: 'center', color: KC.purple, weight: 700 });
  }
  small('槽里的数字是位置 p；下面是它来自的 token（p // 3）', 360, 610, A * p(u, 8, 8.6) * (1 - p(u, 21, 21.6)), { color: KC.ink });
  note(u, 10, 21.6, '第 p 个对来自 token p // 3：位置 {p:7} 来自 token {p:2}');
  // 分组标签
  [0, 1, 2, 3].forEach((e) => small('expert ' + e, groupX(e), 610, A * p(u, 25, 25.6) * (1 - p(u, 63, 63.6)), { color: EXC[e] === KC.yel ? KC.yelD : EXC[e], weight: 700 }));
  note(u, 27, 43.6, '只排编号，token 的数据不动');
  // 补齐位
  const padA = A * p(u, 44, 45);
  [0, 1, 2, 3].forEach((e) => {
    const x = groupX(e) + 300;
    if (padA > 0) { hrect(x, ROWY, SW, SH, KC.gray, padA); txt('12', x + SW / 2, ROWY + SH / 2 + 1, { size: 32, weight: 700, color: KC.ink2, align: 'center', font: HN, alpha: padA }); }
    small('补齐', x + SW / 2, ROWY + SH + 28, padA, { align: 'center', color: KC.gray });
    // block 括号
    const ba = A * p(u, 47 + e * 0.2, 47.6 + e * 0.2);
    line(groupX(e), ROWY + SH + 54, groupX(e) + 392, ROWY + SH + 54, KC.ink, 3, ba);
    line(groupX(e), ROWY + SH + 44, groupX(e), ROWY + SH + 54, KC.ink, 3, ba); line(groupX(e) + 392, ROWY + SH + 44, groupX(e) + 392, ROWY + SH + 54, KC.ink, 3, ba);
    small('block ' + e, groupX(e) + 196, ROWY + SH + 84, ba, { align: 'center', color: KC.ink, weight: 700 });
  });
  note(u, 50, 63.6, '补齐位记作 {r:12}（等于对的总数），kernel 里会被跳过');
  // 三张表
  const ta = A * p(u, 64, 64.8);
  if (ta > 0) {
    mono('sorted_token_ids', 116, 610, ta, { size: 28, weight: 700 });
    small('（上面这一行，共 16 个）', 420, 610, ta);
    [0, 1, 2, 3].forEach((e) => cell(groupX(e) + 136, 860, 120, 52, EXC[e], e, A * p(u, 66 + e * 0.15, 66.5 + e * 0.15), { size: 30 }));
    mono('expert_ids', 116, 944, A * p(u, 66, 66.5), { size: 28, weight: 700 });
    small('（上面每个 block 一个）', 330, 944, A * p(u, 66, 66.5));
    rich('num_tokens_post_padded = {r:16}', 1240, 610, 30, KC.ink, 700, A * p(u, 68, 68.6));
    note(u, 70, 81.6, '三张表都由一个 kernel 在 GPU 上算出，CPU 不用读回任何东西');
    note(u, 82, 90, '真实模型：128 个 expert、top-8；token 数 ≤ 32 时 block = 16');
  }
}

// ================= 第四章：一次 kernel 算完（195–265） =================
const SLOT = [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12];
const rowY = (r) => 330 + r * 30 + Math.floor(r / 4) * 14;
const CX0 = 660, TW = 180;
function grid4(u, a, all) {
  // sorted_token_ids 竖列
  SLOT.forEach((v, r) => {
    const e = Math.floor(r / 4), y = rowY(r);
    if (v === 12) { hrect(560, y, 70, 26, KC.gray, a); txt('12', 595, y + 14, { size: 18, weight: 700, color: KC.ink2, align: 'center', font: HN, alpha: a }); }
    else cell(560, y, 70, 26, EXC[e], v, a, { size: 18 });
  });
  small('sorted_token_ids', 560, 300, a, { color: KC.ink, weight: 700, size: 22 });
  // C：16 行 × 3 列 tile
  for (let r = 0; r < 16; r++) for (let c = 0; c < 3; c++) {
    const x = CX0 + c * (TW + 10), y = rowY(r), e = Math.floor(r / 4), pad = SLOT[r] === 12;
    rr(x, y, TW, 26, 0, KC.paper2, a);
    const fill = all ? p(u, all + c * 0.1, all + 0.5 + c * 0.1) : 0;
    if (fill > 0) pad ? hrect(x, y, TW, 26, KC.gray, a * fill) : rr(x, y, TW, 26, 0, EXC[e], a * fill);
  }
  for (let b = 0; b < 4; b++) for (let c = 0; c < 3; c++) box(CX0 + c * (TW + 10), rowY(b * 4), TW, rowY(b * 4 + 3) + 26 - rowY(b * 4), KC.ink, 1.5, a * 0.6);
  small('C（输出）：每个框是一个 program', CX0 + 150, 300, a, { color: KC.ink, weight: 700, size: 22 });
}
function ch4(t) {
  const u = t - 195; if (u < 0 || u > 70.5) return;
  const A = p(u, 0, 0.6) * (1 - p(u, 69.4, 70));
  const gA = A * (1 - p(u, 29.4, 30)) + A * p(u, 43.4, 44);
  if (gA > 0) grid4(u, gA, u > 44 ? 46 : 0);
  // 4a：一个 program 的完整路径
  const a1 = A * (1 - p(u, 29.4, 30));
  if (a1 > 0) {
    // x 的 4 行
    small('x：4 个 token × 2048', 96, 300, a1, { color: KC.ink, weight: 700, size: 22 });
    for (let i = 0; i < 4; i++) { rr(96, 340 + i * 70, 300, 44, 0, KC.purple, a1); txt('x[t' + i + ']', 246, 362 + i * 70, { size: 24, weight: 700, color: '#fff', align: 'center', font: HN, alpha: a1 }); }
    // 权重
    small('权重：每个 expert 一份', 1420, 300, a1, { color: KC.ink, weight: 700, size: 22 });
    for (let e = 0; e < 4; e++) cell(1420, 330 + e * 110, 400, 90, EXC[e], 'W[E' + e + ']', a1, { size: 30 });
    const focus = p(u, 3, 3.6);
    const fx = CX0, fy = rowY(4), fh = rowY(7) + 26 - fy;
    box(fx - 4, fy - 4, TW + 8, fh + 8, KC.ink, 5, a1 * focus);
    mono('program (pid_m = 1, pid_n = 0)：', 560, 862, a1 * focus, { size: 26, weight: 700 });
    // ① 查 expert
    const s1 = p(u, 6, 6.6);
    box(1416, 326 + 110, 408, 98, KC.ink, 5, a1 * s1);
    arrow(fx + TW + 10, fy + fh / 2, 1410, 330 + 110 + 45, p(u, 6.4, 7.6), KC.ink, a1);
    rich('① expert_ids[1] = {t:1} → 读 W[E1] 的一块', 560, 912, 28, KC.ink, 700, a1 * p(u, 7, 7.6));
    // ② 按编号取行
    const rows = [[4, 0], [5, 1], [6, 3]];
    rows.forEach(([r, tok], i) => curve(396, 362 + tok * 70, 556, rowY(r) + 13, p(u, 11 + i * 0.3, 12.2 + i * 0.3), KC.purple, 3, a1));
    const px = 560, py = rowY(7);
    line(px - 6, py + 2, px + 76, py + 24, KC.red, 4, a1 * p(u, 13, 13.4)); line(px + 76, py + 2, px - 6, py + 24, KC.red, 4, a1 * p(u, 13, 13.4));
    rich('② 第 4–7 行：0、4、10 → token {p:0、1、3}，12 跳过', 560, 954, 28, KC.ink, 700, a1 * p(u, 12, 12.6));
    // ③ 计算
    for (let r = 4; r <= 6; r++) rr(CX0, rowY(r), TW * p(u, 17 + (r - 4) * 0.2, 18 + (r - 4) * 0.2), 26, 0, EXC[1], a1);
    hrect(CX0, rowY(7), TW * p(u, 17.6, 18.4), 26, KC.gray, a1);
    rich('③ 3 行 × W[E1] 的一块，写回 C', 560, 996, 28, KC.ink, 700, a1 * p(u, 17, 17.6));
  }
  // 4b：先搬再算 vs 边读边取
  const a2 = A * p(u, 30, 30.6) * (1 - p(u, 42.8, 43.4));
  if (a2 > 0) {
    const L = [[400, '先搬再算'], [650, '边读边取（vLLM Triton）']];
    L.forEach(([y, name], li) => {
      small(name, 96, y - 60, a2, { color: KC.ink, weight: 700, size: 32 });
      rr(96, y - 30, 220, 60, 0, KC.purple, a2); txt('x', 206, y, { size: 28, weight: 700, color: '#fff', align: 'center', font: HN, alpha: a2 });
      if (li === 0) {
        arrowR(330, y, 520, p(u, 31.5, 32.2), KC.ink, a2);
        const ka = a2 * p(u, 32, 32.8);
        rr(530, y - 30, 440, 60, 0, KC.purple, ka); hrect(530, y + 30, 440, 14, KC.red, ka);
        txt('复制一份：按 expert 排好的 x', 750, y, { size: 24, weight: 700, color: '#fff', align: 'center', font: ZH, alpha: ka });
        small('多写一遍、再读一遍', 530, y + 76, ka, { color: KC.red, weight: 700 });
        arrowR(984, y, 1180, p(u, 33, 33.6), KC.ink, a2);
        cell(1190, y - 30, 300, 60, KC.teal, 'GEMM', a2 * p(u, 33.4, 34));
      } else {
        arrowR(330, y, 1180, p(u, 35, 36.2), KC.purple, a2);
        cell(1190, y - 30, 300, 60, KC.teal, 'GEMM', a2 * p(u, 35.6, 36.2));
        small('kernel 读 A 时按 sorted_token_ids // top_k 直接定位到 x 的行', 330, y + 76, a2 * p(u, 36.4, 37), { color: KC.ink });
      }
    });
    note(u, 38, 43.2, '少了一次复制：中间不需要一份排好序的 x');
  }
  // 4c：一次 launch，grid 按最坏情况开
  const a3 = A * p(u, 44, 44.6);
  if (a3 > 0) {
    const X = 1300;
    small('grid 按最坏情况开', X, 330, a3, { color: KC.ink, weight: 700, size: 30 });
    mono('12 + 4 × (4 − 1) = 24 行', X, 380, a3 * p(u, 48, 48.6), { size: 26 });
    small('→ 6 个 block；实际只有 4 个', X, 420, a3 * p(u, 48.6, 49.2), { color: KC.ink });
    for (let b = 0; b < 6; b++) {
      const real = b < 4, y = 470 + b * 46, k = p(u, 49.4 + b * 0.12, 49.8 + b * 0.12);
      if (real) cell(X, y, 200, 36, EXC[b], 'block ' + b, a3 * k, { size: 20 });
      else { hrect(X, y, 200, 36, KC.gray, a3 * k); small('进来就 return', X + 220, y + 18, a3 * k, { color: KC.red, weight: 700, size: 24 }); }
    }
    small('CPU 不用知道每个 expert 分到几个', X, 780, a3 * p(u, 52, 52.6), { color: KC.ink, weight: 700, size: 28 });
    small('→ 整个 MoE 层能录进 CUDA Graph', X, 822, a3 * p(u, 52.6, 53.2), { color: KC.ink, size: 28 });
    note(u, 56, 70, `逐个 expert：示意路由下 {r:${ACTIVE * 3} 个 kernel}；fused：每次 GEMM {t:1 个 kernel}`);
  }
}

// ================= 第五章：整层流水线（265–315） =================
const STEPS = [
  { name: 'router', col: KC.blue, k: ['gate GEMM', '+ topk_softmax'], io: ['[16, 2048]', '→ ids、权重', '  [16, 8]'] },
  { name: '排队', col: KC.ink, k: ['moe_align_', 'block_size'], io: ['→ sorted_', '  token_ids', '→ expert_ids'] },
  { name: 'GEMM1', col: KC.teal, k: ['fused_moe_kernel', 'w13 = gate + up'], io: ['[16, 2048]', '→ [16, 8, 1536]'] },
  { name: '激活', col: KC.purple, k: ['silu_and_mul'], io: ['[128, 1536]', '→ [128, 768]'] },
  { name: 'GEMM2', col: KC.teal, k: ['fused_moe_kernel', 'w2，× router 权重'], io: ['[128, 768]', '→ [16, 8, 2048]'] },
  { name: '求和', col: KC.ink, k: ['moe_sum'], io: ['[16, 8, 2048]', '→ [16, 2048]'] },
];
function ch5(t) {
  const u = t - 265; if (u < 0 || u > 50.5) return;
  const A = p(u, 0, 0.5) * (1 - p(u, 49.4, 50));
  const BW = 262, GAP = 31, Y = 360, BH = 330;
  STEPS.forEach((s, i) => {
    const x = 96 + i * (BW + GAP), k = A * p(u, 1 + i * 1.8, 1.8 + i * 1.8);
    if (k <= 0) return;
    rr(x, Y, BW, BH, 0, KC.paper2, k); rr(x, Y, BW, 60, 0, s.col, k);
    txt(`${i + 1}  ${s.name}`, x + 18, Y + 31, { size: 30, weight: 700, color: '#fff', font: ZH, alpha: k });
    s.k.forEach((l, j) => mono(l, x + 18, Y + 98 + j * 32, k, { size: 22, color: KC.ink2 }));
    s.io.forEach((l, j) => mono(l, x + 18, Y + 196 + j * 38, k, { size: 22, weight: 700 }));
    if (i > 0) { g.globalAlpha = k; g.fillStyle = KC.ink; g.beginPath(); g.moveTo(x - 6, Y + 165); g.lineTo(x - 24, Y + 152); g.lineTo(x - 24, Y + 178); g.fill(); g.globalAlpha = 1; }
  });
  // 两次 GEMM 是同一个 kernel
  const b = A * p(u, 13, 13.8);
  if (b > 0) {
    const x2 = 96 + 2 * (BW + GAP) + BW / 2, x4 = 96 + 4 * (BW + GAP) + BW / 2;
    line(x2, Y + BH + 10, x2, Y + BH + 50, KC.red, 4, b); line(x4, Y + BH + 10, x4, Y + BH + 50, KC.red, 4, b);
    line(x2, Y + BH + 50, x4, Y + BH + 50, KC.red, 4, b);
    small('同一个 fused_moe_kernel，各调用一次', (x2 + x4) / 2, Y + BH + 86, b, { align: 'center', color: KC.red, weight: 700, size: 28 });
  }
  note(u, 18, 33.6, 'gate 与 up 合成一次 GEMM（N = 2 × 768）；router 权重在第二次 GEMM 里乘上');
  note(u, 34, 50, 'BF16 下不需要输入量化；FP8 等格式在两次 GEMM 前各多一步量化');
}

// ================= 第六章：batch 决定瓶颈（315–385） =================
const NS = [1, 2, 4, 8, 16, 32, 64, 128];
const actN = (n) => E * (1 - Math.pow(1 - K / E, n));                 // 均匀路由下激活 expert 数的期望
const MB = 9.44;                                                      // 一个 expert 的 BF16 权重（MB）
function ch6(t) {
  const u = t - 315; if (u < 0 || u > 70.5) return;
  const A = p(u, 0, 0.5) * (1 - p(u, 69.4, 70));
  // 6a 柱状图
  const a1 = A * (1 - p(u, 27.4, 28));
  if (a1 > 0) {
    const X0 = 170, BASE = 820, HMAX = 440, BW = 84, STEP = 118;
    line(X0 - 20, BASE - HMAX, X0 + NS.length * STEP, BASE - HMAX, KC.gray, 2, a1, [8, 8]);
    small('全部 128 个 · 1.21 GB', X0 + NS.length * STEP + 10, BASE - HMAX, a1);
    line(X0 - 20, BASE, X0 + NS.length * STEP, BASE, KC.ink, 3, a1);
    NS.forEach((n, i) => {
      const v = actN(n), h = (v / E) * HMAX * p(u, 1 + i * 0.4, 1.8 + i * 0.4), x = X0 + i * STEP;
      rr(x, BASE - h, BW, h, 0, n === 64 ? KC.red : KC.teal, a1);
      txt(v < 10 ? v.toFixed(0) : Math.round(v), x + BW / 2, BASE - h - 22, { size: 26, weight: 700, color: KC.ink, align: 'center', font: HN, alpha: a1 * p(u, 1.6 + i * 0.4, 2 + i * 0.4) });
      txt(String(n), x + BW / 2, BASE + 32, { size: 26, weight: 700, color: KC.ink, align: 'center', font: HN, alpha: a1 });
    });
    small('一次前向的 token 数', X0, BASE + 74, a1);
    small('激活的 expert 数', X0, 330, a1, { color: KC.ink, weight: 700 });
    const R = 1220;
    [[1, '1 个 token'], [16, '16 个 token'], [64, '64 个 token']].forEach(([n, l], i) => {
      const k = a1 * p(u, 6 + i * 1.2, 6.6 + i * 1.2), y = 470 + i * 130, v = actN(n);
      small(l, R, y, k, { color: KC.ink, weight: 700, size: 30 });
      rich(`{${n === 64 ? 'r' : 't'}:${Math.round(v)} 个 expert} · 读 ${v * MB < 1000 ? Math.round(v * MB) + ' MB' : (v * MB / 1000).toFixed(2) + ' GB'}`, R, y + 52, 40, KC.ink, 700, k);
    });
    note(u, 12, 27.6, '一个 expert 的权重 BF16 约 9.4 MB；token 一多，几乎每个 expert 都要整块读一遍');
  }
  // 6b 每个权重用几次
  const a2 = A * p(u, 28, 28.6) * (1 - p(u, 49.4, 50));
  if (a2 > 0) {
    small('decode · 16 个 token', 96, 330, a2, { color: KC.ink, weight: 700, size: 32 });
    big2((128 / actN(16)).toFixed(1), 96, 520, KC.red, a2 * p(u, 29, 29.6));
    small('每个激活 expert 平均分到的 token 数', 96, 570, a2 * p(u, 29.4, 30));
    small('prefill · 4096 个 token', 980, 330, a2 * p(u, 31, 31.6), { color: KC.ink, weight: 700, size: 32 });
    big2('256', 980, 520, KC.teal, a2 * p(u, 31.4, 32));
    small('4096 × 8 / 128', 980, 570, a2 * p(u, 31.8, 32.4));
    // 补齐账本
    const la = a2 * p(u, 36, 36.8), LY = 690, LW = 1728, rows = ACTIVE * 16;
    small(`示意路由下 16 个 token：${ACTIVE} 个 expert × 每段补齐到 16 = ${rows} 行`, 96, LY - 30, la, { color: KC.ink, weight: 700 });
    rr(96, LY, LW * 128 / rows, 80, 0, KC.purple, la);
    hrect(96 + LW * 128 / rows + 4, LY, LW * (1 - 128 / rows) - 4, 80, KC.gray, la);
    txt('128 行真 token', 96 + 12, LY + 41, { size: 24, weight: 700, color: '#fff', font: ZH, alpha: la });
    rich(`补齐 {r:${rows - 128} 行}`, 96 + LW * 128 / rows + 4, LY + 124, 30, KC.ink, 700, la);
    note(u, 39, 49.6, 'decode 时瓶颈不在算力：补齐的行多算一些，影响不大；要省的是读权重');
  }
  // 6c 分档表
  const a3 = A * p(u, 50, 50.6);
  if (a3 > 0) {
    const ROWS = [['≤ 4', '不排队：每个 block 只算一对', '—'], ['≤ 32', '排队', '16'], ['≤ 96', '排队', '32'], ['≤ 512', '排队', '64'], ['> 512', '排队', '128']];
    const X = [96, 400, 1100], Y0 = 330;
    ['token 数', '分配方式', 'BLOCK_SIZE_M'].forEach((h, i) => small(h, X[i], Y0, a3, { color: KC.ink, weight: 700, size: 30 }));
    line(96, Y0 + 30, 1500, Y0 + 30, KC.ink, 3, a3);
    ROWS.forEach((r, i) => {
      const k = a3 * p(u, 51 + i * 0.3, 51.5 + i * 0.3), y = Y0 + 84 + i * 72;
      r.forEach((c, j) => j === 1 ? small(c, X[j], y, k, { color: KC.ink, size: 32, weight: 600 }) : txt(c, X[j], y, { size: 34, weight: 700, color: i === 0 ? KC.red : KC.ink, font: HN, alpha: k }));
      line(96, y + 36, 1500, y + 36, KC.lg, 2, k);
    });
    note(u, 54, 70, '≤ 4 来自 token 数 × 8 × 4 ≤ 128；有调优配置文件时按文件，这里是 BF16 默认值');
  }
}
function big2(s, x, y, col, a) { txt(s, x, y, { size: 200, weight: 700, color: col, font: HN, alpha: a, base: 'alphabetic', spacing: -6 }); }

// ================= 第七章：别的系统怎么做（385–445） =================
function ch7(t) {
  const u = t - 385; if (u < 0 || u > 60.5) return;
  const A = p(u, 0, 0.5) * (1 - p(u, 59.4, 60));
  // 7a DeepGEMM 两种布局
  const a1 = A * (1 - p(u, 23.4, 24));
  if (a1 > 0) {
    const segs = [5, 2, 7, 3];                                          // 示意：各 expert 的 token 数
    small('contiguous：训练前向、prefill', 96, 330, a1, { color: KC.ink, weight: 700, size: 32 });
    small('各 expert 的 token 首尾相接，每段对齐到 M block', 96, 376, a1, { size: 26 });
    let x = 96;
    segs.forEach((n, e) => {
      const k = a1 * p(u, 1 + e * 0.3, 1.6 + e * 0.3), al = Math.ceil(n / 4) * 4;
      rr(x, 420, n * 22, 70, 0, EXC[e], k); hrect(x + n * 22, 420, (al - n) * 22, 70, KC.gray, k);
      txt('E' + e, x + 12, 456, { size: 22, weight: 700, color: onC(EXC[e]), font: HN, alpha: k });
      x += al * 22 + 6;
    });
    small('masked：开 CUDA Graph 的 decode', 980, 330, a1 * p(u, 6, 6.6), { color: KC.ink, weight: 700, size: 32 });
    small('每个 expert 一段固定长度，mask 标出有效行数', 980, 376, a1 * p(u, 6, 6.6), { size: 26 });
    segs.forEach((n, e) => {
      const k = a1 * p(u, 7 + e * 0.3, 7.6 + e * 0.3), y = 420 + e * 84;
      rr(980, y, 8 * 70, 64, 0, KC.paper2, k); rr(980, y, n * 70, 64, 0, EXC[e], k);
      txt('E' + e, 996, y + 33, { size: 22, weight: 700, color: onC(EXC[e]), font: HN, alpha: k });
      small(`有效 ${n} 行`, 980 + 8 * 70 + 20, y + 33, k, { color: KC.ink });
    });
    small('每段长度按上限开：CPU 不知道每个 expert 分到几个', 980, 790, a1 * p(u, 10, 10.6), { color: KC.ink });
    note(u, 12, 23.6, 'DeepGEMM 只对 M 轴分组：各 expert 的 N、K 相同（示意数字）');
  }
  // 7b MegaBlocks
  const a2 = A * p(u, 24, 24.6) * (1 - p(u, 42.4, 43));
  if (a2 > 0) {
    const segs = [6, 1, 4, 3], CAP = 4, CW = 70;
    small('按 capacity：每个 expert 固定 4 个位置', 96, 330, a2, { color: KC.ink, weight: 700, size: 32 });
    segs.forEach((n, e) => {
      const y = 400 + e * 92, k = a2 * p(u, 25 + e * 0.3, 25.6 + e * 0.3);
      for (let i = 0; i < Math.max(n, CAP); i++) {
        const x = 96 + i * (CW + 8);
        if (i < Math.min(n, CAP)) cell(x, y, CW, 64, EXC[e], i === 0 ? 'E' + e : '', k, { size: 22 });
        else if (i < CAP) hrect(x, y, CW, 64, KC.gray, k);
        else { rr(x, y, CW, 64, 0, KC.paper2, k); line(x + 10, y + 10, x + CW - 10, y + 54, KC.red, 5, k); line(x + CW - 10, y + 10, x + 10, y + 54, KC.red, 5, k); }
      }
      box(96 - 4, y - 4, CAP * (CW + 8), 72, KC.ink, 2, k * 0.6);
    });
    small('超出的丢掉，不足的补齐', 96, 790, a2 * p(u, 28, 28.6), { color: KC.red, weight: 700, size: 28 });
    small('MegaBlocks：block-sparse，按实际数量算', 980, 330, a2 * p(u, 30, 30.6), { color: KC.ink, weight: 700, size: 32 });
    segs.forEach((n, e) => {
      const y = 400 + e * 92, k = a2 * p(u, 31 + e * 0.3, 31.6 + e * 0.3);
      for (let i = 0; i < n; i++) cell(980 + i * (CW + 8), y, CW, 64, EXC[e], i === 0 ? 'E' + e : '', k, { size: 22 });
    });
    note(u, 34, 42.6, '论文摘要：以前要么丢 token，要么把 padding 算进计算和显存');
  }
  // 7c vLLM 后端顺序
  const a3 = A * p(u, 43, 43.6);
  if (a3 > 0) {
    const COLS = [['Hopper（SM90）', ['Triton', 'Batched Triton', 'FlashInfer TRTLLM', 'FlashInfer CUTLASS']],
      ['其他 NVIDIA GPU', ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'Triton', 'Batched Triton']]];
    COLS.forEach(([h, list], ci) => {
      const x = 96 + ci * 880;
      small(h, x, 330, a3 * p(u, 43.4 + ci * 0.8, 44 + ci * 0.8), { color: KC.ink, weight: 700, size: 34 });
      list.forEach((n, i) => {
        const k = a3 * p(u, 44 + ci * 0.8 + i * 0.2, 44.5 + ci * 0.8 + i * 0.2), y = 380 + i * 92;
        const col = n.startsWith('Triton') ? KC.teal : n.startsWith('Batched') ? KC.tealD : KC.blue;
        rr(x, y, 640, 72, 0, i === 0 ? col : KC.paper2, k);
        txt(`${i + 1}. ${n}`, x + 24, y + 37, { size: 30, weight: 700, color: i === 0 ? '#fff' : KC.ink2, font: HN, alpha: k });
      });
    });
    note(u, 47, 60, '按顺序取第一个支持当前模型与硬件的；源码注释：Hopper 上 FlashInfer 比 Triton 慢');
  }
}

// ================= 结尾：回到开幕的网格（445–460） =================
function ending(t) {
  const u = t - 445; if (u < 0) return;
  const A = p(u, 0, 0.8) * (1 - p(u, 14, 15));
  txt('MoE Kernel', 90, 196, { size: 168, weight: 700, color: KC.ink, font: HN, alpha: A, base: 'alphabetic' });
  rich('排队 → 补齐 → {t:一次算完}', 96, 290, 48, KC.ink2, 700, A * p(u, 1, 1.8));
  const flow = p(u, 4, 8);
  // 网格里的 16 个 token 分派，流向一整行排好的 block
  const order = [...Array(E).keys()].filter((e) => counts[e] > 0);
  const segW = 1728 / order.length;
  for (let e = 0; e < E; e++) {
    const [x, y] = ocell(e), c = counts[e];
    if (!c) { rr(x, y, OC, OC, 0, KC.paper2, A * (1 - flow)); continue; }
    const i = order.indexOf(e), tx = 96 + i * segW, ty = 480;
    const cx = mix(x, tx, flow), cy = mix(y, ty, flow), w = mix(OC, segW - 2, flow), h = mix(OC, 380, flow);
    rr(cx, cy, w, h, 0, KC.teal, A);
    rr(cx, cy, w, h * (c / 16) * flow, 0, KC.purple, A);              // 真 token 占一段的 c / 16
    if (flow > 0.95) hrect(cx, cy + h * (c / 16), w, h * (1 - c / 16), KC.gray, A * p(u, 7.6, 8.2));
    if (flow < 0.05) tokenDots(x, y, c, A);
  }
  small(`示意路由下 16 个 token：${ACTIVE} 个 expert 各一段，紫色是真 token，斜线是补齐到 16 行`, 96, 440, A * p(u, 8.4, 9), { color: KC.ink, weight: 700 });
  const kA = A * p(u, 9.4, 10.2);
  rr(96, 900, 1728, 80, 0, KC.ink, kA);
  txt('fused_moe_kernel · 一次 launch', 960, 941, { size: 34, weight: 700, color: '#fff', align: 'center', font: HN, alpha: kA });
}

// ================= 插入段：MoE 层与一个 expert（第一章内，20.8 s 起 30 s） =================
function sceneHead(u, a, b, tag, col, head, src) {
  const k = win(u, a, b, 0.45); if (k <= 0) return;
  const ent = po(u, a, a + 0.6);
  g.font = `700 32px ${ZH}`; const w = g.measureText(tag).width + 36;
  rr(96, 100, w, 50, 0, col, k);
  txt(tag, 114, 136, { size: 32, weight: 700, color: '#fff', font: ZH, alpha: k, base: 'alphabetic' });
  g.save(); g.translate(0, (1 - ent) * 18); rich(head, 96, 226, 60, KC.ink, 700, k * ent); g.restore();
  txt(src, 1824, 76, { size: 26, weight: 500, color: KC.gray, font: ZH, align: 'right', alpha: k, base: 'alphabetic' });
}
function vbar(x, y, w, h, col, a, label, sub, side = 'below') {
  rr(x, y, w, h, 0, col, a);
  if (label) small(label, x + w / 2, side === 'below' ? y + h + 30 : y - 40, a, { align: 'center', color: KC.ink, weight: 700, size: 24 });
  if (sub) small(sub, x + w / 2, side === 'below' ? y + h + 60 : y - 12, a, { align: 'center', size: 22 });
}
function insertScene(u) {
  const A = p(u, 0.3, 0.9) * (1 - p(u, INS_D - 0.6, INS_D));
  txt('MoE Kernel · 一 · MoE 层算什么', 96, 76, { size: 28, weight: 700, color: KC.ink2, font: ZH, alpha: A, base: 'alphabetic' });
  sceneHead(u, 0.3, 12, 'MoE 层', KC.teal, 'MoE 层替换 Transformer 层里的 {t:FFN}，48 层每层一个', '来源：Qwen3-30B-A3B config.json');
  sceneHead(u, 12, INS_D, 'expert', KC.teal, '一个 expert：{t:3 次 GEMM}，中间一次 SwiGLU', 'Qwen3-30B-A3B · hidden_act = silu');
  // A：一个 Transformer 层
  const a1 = A * (1 - p(u, 11.4, 12));
  if (a1 > 0) {
    const CY = 600;
    tokenTag(96, CY, a1 * p(u, 0.8, 1.3), 'token', 150);
    small('2048 维', 171, CY + 60, a1 * p(u, 0.8, 1.3), { align: 'center' });
    arrowR(256, CY, 330, p(u, 1.3, 1.8), KC.ink, a1);
    const k2 = a1 * p(u, 1.6, 2.2);
    rr(340, CY - 70, 300, 140, 0, KC.ink, k2);
    txt('attention', 490, CY, { size: 34, weight: 700, color: '#fff', align: 'center', font: HN, alpha: k2 });
    arrowR(650, CY, 724, p(u, 2.2, 2.7), KC.ink, a1);
    const k3 = a1 * p(u, 2.6, 3.4);
    rr(734, CY - 140, 760, 280, 0, KC.paper2, k3); box(734, CY - 140, 760, 280, KC.teal, 6, k3);
    txt('MoE 层', 760, CY - 104, { size: 30, weight: 700, color: KC.teal, font: ZH, alpha: k3 });
    rr(770, CY - 40, 170, 80, 0, KC.blue, k3 * p(u, 3.4, 3.9));
    txt('router', 855, CY, { size: 28, weight: 700, color: '#fff', align: 'center', font: HN, alpha: k3 * p(u, 3.4, 3.9) });
    arrowR(944, CY, 1010, p(u, 3.9, 4.3), KC.ink, k3);
    for (let e = 0; e < E; e++) {
      const x = 1020 + (e % 16) * 28, y = CY - 90 + Math.floor(e / 16) * 22, k = k3 * p(u, 4.0 + e * 0.008, 4.4 + e * 0.008);
      rr(x, y, 24, 18, 0, KC.teal, k);
    }
    small('128 个 expert，选 8 个', 1020 + 224, CY + 110, k3 * p(u, 4.8, 5.4), { align: 'center', color: KC.ink });
    arrowR(1504, CY, 1580, p(u, 5.2, 5.7), KC.ink, a1);
    const k4 = a1 * p(u, 5.5, 6.0);
    rr(1590, CY - 50, 234, 100, 0, KC.paper2, k4);
    txt('下一层', 1707, CY, { size: 30, weight: 700, color: KC.ink2, align: 'center', font: ZH, alpha: k4 });
    // 一个 Transformer 层的括号
    const b = a1 * p(u, 6.4, 7.2);
    line(340, CY - 190, 1494, CY - 190, KC.ink, 3, b); line(340, CY - 190, 340, CY - 170, KC.ink, 3, b); line(1494, CY - 190, 1494, CY - 170, KC.ink, 3, b);
    rich('一个 Transformer 层，重复 {t:48 层}', 340, CY - 210, 32, KC.ink, 700, b);
    small('原来这里是一个 FFN', 1114, CY + 200, a1 * p(u, 7.6, 8.2), { align: 'center', color: KC.red, weight: 700, size: 28 });
    note(u, 8.4, 11.8, 'Qwen3-30B-A3B 的 48 层每层都是 MoE；attention 之外的计算都在这里');
  }
  // B：一个 expert 的 SwiGLU
  const a2 = A * p(u, 12, 12.6);
  if (a2 > 0) {
    const S = 240 / 2048;                                              // 每维对应的像素
    const L = 2048 * S, M = 768 * S;
    const on = (t0) => a2 * p(u, t0, t0 + 0.5);
    g.save(); g.translate(0, -40);                                    // 整张图上移，给底部说明留空
    vbar(120, 490, 40, L, KC.purple, on(12.4), 'x', '2048');
    // gate 与 up
    rr(320, 340, M, L, 0, KC.teal, on(13.0)); small('W_gate', 320 + M / 2, 320, on(13.0), { align: 'center', color: KC.ink, weight: 700, size: 24 });
    small('2048 × 768', 320 + M / 2, 340 + L + 28, on(13.0), { align: 'center', size: 22 });
    curve(164, 610, 316, 460, p(u, 12.9, 13.4), KC.ink, 3, a2);
    rr(320, 680, M, L, 0, KC.teal, on(14.6)); small('W_up', 320 + M / 2, 660, on(14.6), { align: 'center', color: KC.ink, weight: 700, size: 24 });
    small('2048 × 768', 320 + M / 2, 680 + L + 28, on(14.6), { align: 'center', size: 22 });
    curve(164, 610, 316, 800, p(u, 14.5, 15.0), KC.ink, 3, a2);
    arrowR(416, 460, 494, p(u, 13.6, 14.0), KC.ink, a2); vbar(500, 415, 40, M, KC.tealD, on(14.0), 'g', '768', 'above');
    arrowR(416, 800, 494, p(u, 15.2, 15.6), KC.ink, a2); vbar(500, 755, 40, M, KC.tealD, on(15.4), 'u', '768');
    // SwiGLU
    curve(544, 460, 636, 610, p(u, 16.2, 16.7), KC.ink, 3, a2); curve(544, 800, 636, 610, p(u, 16.2, 16.7), KC.ink, 3, a2);
    rr(640, 565, 210, 90, 0, KC.purple, on(16.4));
    txt('silu(g) × u', 745, 610, { size: 28, weight: 700, color: '#fff', align: 'center', font: HN, alpha: on(16.4) });
    small('逐元素', 745, 690, on(16.6), { align: 'center' });
    arrowR(856, 610, 914, p(u, 17.2, 17.6), KC.ink, a2); vbar(920, 565, 40, M, KC.tealD, on(17.4), 'h', '768', 'above');
    // down
    arrowR(966, 610, 1024, p(u, 17.9, 18.3), KC.ink, a2);
    rr(1030, 565, L, M, 0, KC.teal, on(18.2)); small('W_down', 1030 + L / 2, 545, on(18.2), { align: 'center', color: KC.ink, weight: 700, size: 24 });
    small('768 × 2048', 1030 + L / 2, 565 + M + 28, on(18.2), { align: 'center', size: 22 });
    arrowR(1276, 610, 1334, p(u, 18.8, 19.2), KC.ink, a2); vbar(1340, 490, 40, L, KC.purple, on(19.0), '输出', '2048');
    g.restore();
    // 步骤
    [[13.0, '① gate GEMM', '2048 → 768'], [14.6, '② up GEMM', '2048 → 768'], [16.4, '③ silu(g) × u', '逐元素'], [18.2, '④ down GEMM', '768 → 2048']].forEach(([t0, n, d], i) => {
      const k = on(t0), y = 400 + i * 90;
      rich(n.includes('GEMM') ? `{t:${n}}` : `{p:${n}}`, 1470, y, 32, KC.ink, 700, k);
      small(d, 1470, y + 36, k, { size: 24 });
    });
    note(u, 21, INS_D - 0.2, '一个 expert = {t:3 次 GEMM} + 1 次逐元素运算；router 选中的 8 个 expert 都这样算');
  }
}

start((t) => {
  // 插入段期间，主线停在 INS_T；插入段之后，主线时间后移 INS_D
  const tt = t < INS_T ? t : t < INS_T + INS_D ? INS_T : t - INS_D;
  g.fillStyle = KC.paper; g.fillRect(0, 0, W, H);
  if (tt < 10) opening(tt);
  header(tt);
  chapter(tt);
  ch2(tt); ch3(tt); ch4(tt); ch5(tt); ch6(tt); ch7(tt); ending(tt);
  if (t >= INS_T && t < INS_T + INS_D) {
    const u = t - INS_T, cover = p(u, 0, 0.5) * (1 - p(u, INS_D - 0.5, INS_D));
    g.globalAlpha = cover; g.fillStyle = KC.paper; g.fillRect(0, 0, W, H); g.globalAlpha = 1;
    insertScene(u);
  }
}, DURATION + INS_D);
