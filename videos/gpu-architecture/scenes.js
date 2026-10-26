// GPU 架构影片的各幕。每幕一个 draw(u)，u 是幕内时间（秒）；节拍起点取自 script.js，音效事件登记进 EV。
// 贯穿全片的图形：结果矩阵里的一个 128 × 256 块（紫色）。全片跟着它从整道题里分出来、落到一个 SM、数据一层层搬进来。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const EV = SCRIPT.map(() => []);                      // 每幕的音效：[幕内时间, 种类]
const ev = (k, u, kind) => EV[k].push([u, kind]);
const evRun = (k, a, b, step, kind) => { for (let u = a; u < b - 1e-6; u += step) ev(k, u, kind); };
const NAME = 'GPU 架构';
// 存储层级的颜色：离计算单元越远越偏金；A blue、B yellow、结果 purple、Tensor Core teal 沿用系列
const MEM = { hbm: '#d4b352', l2: '#b39a72', smem: '#6f8a9e', reg: '#3a3a3a', tmem: '#c9b6f5' };
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const HERO_SM = () => B200_ON[37];          // 主角块（第 37 块）第 1 轮落在的 SM
// 占位框：待实测的画面，写明将放什么、需要什么硬件，不填数字
function placeholder(x, y, w, h, lines, alpha = 1) {
  if (alpha <= 0.003) return;
  rr(x, y, w, h, C.paper2, alpha * 0.6); g.globalAlpha = alpha; g.setLineDash([10, 8]); g.strokeStyle = C.gray; g.lineWidth = 2; g.strokeRect(x, y, w, h); g.setLineDash([]); g.globalAlpha = 1;
  txt('占位 · 待实测', x + 24, y + 44, { size: 26, weight: 800, color: C.gray, alpha });
  lines.forEach((l, i) => txt(l, x + 24, y + 86 + i * 34, { size: 24, color: C.gray, alpha }));
}
// 引线标注：从文字左侧连到所指的点
function callout(px, py, x, y, title, sub, alpha, col = C.ink) {
  if (alpha <= 0.003) return;
  line(px, py, x - 16, y - 10, col, 2, alpha * 0.7); g.globalAlpha = alpha; g.fillStyle = col; g.beginPath(); g.arc(px, py, 6, 0, 7); g.fill(); g.globalAlpha = 1;
  txt(title, x, y, { size: 32, weight: 800, color: col, alpha }); if (sub) txt(sub, x, y + 38, { size: 24, color: C.ink2, alpha });
}

// 页眉：左上角是开场缩上来的片名，下面是这一幕的标签与标题，右上角是出处
function header(k, u) {
  const s = SCRIPT[k], a = po(u, 0, 0.5);
  txt(NAME, 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 76, { size: 26, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 226 + (1 - a) * 16, 56, 700, a);
}

// ================= 1 开场 =================
// 速率：B200 dense BF16 2.25 PFLOPS，HBM3e 8 TB/s，比值 281 FLOP/B；每次乘加回内存取两个 BF16（4 字节）只够 0.5 FLOP/B
const S1 = { out: [4.6, 5.8], move: [5.4, 6.6], byte: [bt(0, 2) + 0.2, bt(0, 2) + 1.4], fill: [bt(0, 2) + 1.4, bt(0, 2) + 4.6], bars: bt(0, 3) + 0.2 };
ev(0, 0.25, 'land'); evRun(0, 2.6, 4.4, 0.12, 'tick'); ev(0, S1.out[0], 'whoosh');
ev(0, bt(0, 1) + 1.4, 'pop'); ev(0, bt(0, 1) + 2.2, 'pop');
ev(0, S1.byte[0], 'click'); ev(0, S1.byte[1], 'land'); evRun(0, S1.fill[0], S1.fill[1], 0.1, 'tick'); ev(0, S1.fill[1], 'pop');
ev(0, S1.bars, 'land'); ev(0, S1.bars + 1.6, 'click'); ev(0, S1.bars + 2.6, 'pop');
function s1(u) {
  const out = p(u, ...S1.out), k = po(u, 0.15, 1.0);
  // 大标题：开场居左，随后缩成左上角的片名
  const size = mix(220, 28, out);
  txt(NAME, mix(84, 96, out), mix(420, 76, out) + (1 - k) * 40, { size, weight: mix(800, 700, out), color: out < 0.5 ? C.ink : C.ink2, alpha: k });
  txt('以 B200 为例：数据怎样送到 Tensor Core', 96, 530, { size: 48, weight: 700, color: C.tealD, alpha: po(u, 1.2, 1.8) * (1 - out) });
  // 芯片：先在右侧逐层画出，148 格依次点亮；随后缩小移到中间
  const m = p(u, ...S1.move), CX = mix(1110, 750, m), CY = mix(140, 300, m), CW = mix(714, 420, m);
  const lit = Math.floor(148 * lin(u, 2.6, 4.4)), litSet = new Set(B200_ON.slice(0, lit));
  chip(CX, CY, CW, { build: lin(u, 0.7, 2.8), off: B200_OFF, paint: (n) => (litSet.has(n) ? null : ['#4a483f', 1]) });
  txt('B200 · 2 个 die · 160 个 SM 中启用 148 个（暗格为关闭，位置示意）', 1824, 836, { size: 24, color: C.gray, align: 'right', alpha: po(u, 3.6, 4.2) * (1 - out) });

  // 两个速率：左边 HBM 送数据，右边 Tensor Core 消耗数据
  const b = po(u, bt(0, 1) + 1.2, bt(0, 1) + 1.8), c = po(u, bt(0, 1) + 2.0, bt(0, 1) + 2.6);
  rr(96, 360, 300, 200, MEM.hbm, b); txt('HBM', 246, 450, { size: 48, weight: 800, align: 'center', alpha: b }); txt('显存', 246, 500, { size: 30, align: 'center', color: C.ink2, alpha: b });
  txt('每秒送来 8 TB', 246, 610, { size: 36, weight: 700, align: 'center', alpha: b });
  rr(396, 446, 354, 28, C.paper2, b);                                    // 管道
  const by = lin(u, ...S1.byte);
  if (u > S1.byte[0] && u < S1.byte[1] + 0.3) rr(mix(396, 900, by), 448, 24, 24, C.blue, 1 - lin(u, S1.byte[1], S1.byte[1] + 0.3));
  txt('1 字节', mix(396, 900, by) + 12, 436, { size: 26, weight: 700, color: C.blue, align: 'center', alpha: po(u, S1.byte[0], S1.byte[0] + 0.3) * (1 - lin(u, S1.byte[1], S1.byte[1] + 0.3)) });
  txt('Tensor Core', 1560, 300, { size: 36, weight: 700, align: 'center', color: C.tealD, alpha: c });
  txt('每秒 2250 万亿次运算', 1560, 346, { size: 30, align: 'center', alpha: c });
  // 281 个运算格：每行 20 个，一格一次运算（FLOP），和左边的一个字节一样大
  const G = 22, GX = 1340, GY = 370, nOp = Math.floor(281 * lin(u, ...S1.fill));
  for (let i = 0; i < 281; i++) rr(GX + (i % 20) * G, GY + Math.floor(i / 20) * G, G - 4, G - 4, i < nOp ? C.teal : C.paper2, c);
  arrow(1180, 460, 1300, 460, c, C.ink, c * 0.6);
  const f = po(u, S1.fill[1], S1.fill[1] + 0.4);
  txt('1 字节要用 281 次', 1560, 724, { size: 40, weight: 800, align: 'center', color: C.tealD, alpha: f });
  txt('2250 万亿 ÷ 8 万亿 ≈ 281', 1560, 758, { size: 26, align: 'center', color: C.gray, alpha: f, font: MONO });

  // 两条对着同一条基线：需要 281，每次回内存取数只有 0.5
  const d = po(u, S1.bars, S1.bars + 0.5), X = 520, BW = 1300;
  rr(0, 772, W, 228, C.paper, d);                                       // 盖住上半幕下沿的标注，留出两条
  line(X, 784, X, 940, C.ink, 3, d);
  txt('需要', 96, 830, { size: 34, weight: 700, alpha: d }); rr(X, 794, BW * lin(u, S1.bars, S1.bars + 1.2), 52, C.teal, d);
  txt('281 次 / 字节', X + BW - 16, 831, { size: 30, weight: 700, color: '#fff', align: 'right', alpha: po(u, S1.bars + 1.1, S1.bars + 1.5) });
  const e = po(u, S1.bars + 1.6, S1.bars + 2.0);
  txt('每次运算回内存取数', 96, 920, { size: 34, weight: 700, alpha: e }); rr(X, 884, Math.max(3, BW * 0.5 / 281), 52, C.red, e);
  txt('0.5 次 / 字节', X + 24, 921, { size: 30, color: C.red, weight: 700, alpha: e });
  txt('一次乘加是 2 次运算，要取 2 个 BF16，共 4 字节', X + 24, 966, { size: 24, color: C.gray, alpha: e });
  txt('差 562 倍', 1824, 921, { size: 40, weight: 800, color: C.red, align: 'right', alpha: po(u, S1.bars + 2.6, S1.bars + 3.0) });
}

// ================= 2 切块 =================
// C = A × B，三个都是 4096 × 4096；C 按 128 × 256 切成 32 行 × 16 列 = 512 块，主角是第 2 行第 5 列（序号 37）
const MAT = { s: 320, AX: 96, AY: 610, BX: 460, BY: 272 }, TR = 32, TC = 16, HERO = 2 * TC + 5;
const tileXY = (i) => [MAT.BX + (i % TC) * MAT.s / TC, MAT.AY + Math.floor(i / TC) * MAT.s / TR, MAT.s / TC, MAT.s / TR];
// 每轮最多 148 块；同一行的 16 块一起出发，各行相隔 0.055 s；飞完之后停一下再开始下一轮
const S2 = { cut: [bt(1, 1), bt(1, 1) + 1.4], hero: bt(1, 1) + 1.8, strips: bt(1, 1) + 2.8, waves: [bt(1, 2) + 0.3, 1.75], fly: 0.55 };
const S2delay = (slot) => 0.055 * Math.floor(slot / 16);
const CHIP2 = [1064, 196, 760];
ev(1, 0.6, 'pop'); ev(1, 1.0, 'pop'); ev(1, 1.4, 'pop'); evRun(1, S2.cut[0], S2.cut[1], 0.1, 'tick'); ev(1, S2.hero, 'land'); ev(1, S2.strips, 'click'); ev(1, S2.strips + 0.5, 'click');
for (let w = 0; w < 4; w++) { ev(1, S2.waves[0] + w * S2.waves[1], 'whoosh'); ev(1, S2.waves[0] + w * S2.waves[1] + S2.fly + 0.4, 'land'); }
function s2(u) {
  const { s, AX, AY, BX, BY } = MAT, aA = po(u, 0.5, 1.0), aB = po(u, 0.9, 1.4), aC = po(u, 1.3, 1.8);
  // 三个矩阵：B 在上，A 在左，C 在右下，A 的行与 B 的列对着 C 的行与列
  rr(AX, AY, s, s, '#b9c6f0', aA); txt('A', AX + s / 2, AY + s / 2 + 40, { size: 110, weight: 800, color: C.blue, align: 'center', alpha: aA * 0.35 });
  txt('4096 × 4096', AX + s / 2, AY + s + 40, { size: 24, color: C.gray, align: 'center', alpha: aA, font: MONO });
  rr(BX, BY, s, s, '#f6e3a0', aB); txt('B', BX + s / 2 + 60, BY + s / 2 + 40, { size: 110, weight: 800, color: C.yelD, align: 'center', alpha: aB * 0.35 });
  rr(BX, AY, s, s, '#d6c4fb', aC); txt('C', BX + s / 2, AY + s / 2 + 40, { size: 110, weight: 800, color: C.purple, align: 'center', alpha: aC * 0.35 * (1 - po(u, S2.cut[0], S2.cut[1])) });
  // 切块：横线 32 条、竖线 16 条依次画出
  const cut = lin(u, ...S2.cut);
  for (let r = 1; r < TR; r++) line(BX, AY + r * s / TR, BX + s, AY + r * s / TR, C.paper, 1.2, clamp(cut * TR - r + 1) * aC);
  for (let q = 1; q < TC; q++) line(BX + q * s / TC, AY, BX + q * s / TC, AY + s, C.paper, 1.6, clamp(cut * TC - q + 1) * aC);
  txt('32 × 16 = 512 块，每块 128 × 256', BX + s / 2, AY + s + 40, { size: 24, color: C.gray, align: 'center', alpha: po(u, S2.cut[1], S2.cut[1] + 0.4), font: MONO });
  // 已经算完的块在 C 上变暗
  const [w0, ws] = S2.waves, wave = Math.min(3, Math.floor((u - w0) / ws)), wu = u - w0 - wave * ws;
  const landed = (slot) => wu > S2delay(slot) + S2.fly;
  const done = wave < 0 ? 0 : Math.min(512, wave * 148 + B200_ON.filter((_, slot) => landed(slot)).length);
  for (let i = 0; i < done; i++) { if (i === HERO) continue; const [tx, ty, tw_, th] = tileXY(i); rr(tx, ty, tw_ - 1, th - 1, '#8d7ad0', 0.55); }
  // 主角块与它用到的 A 横条、B 竖条
  const h = po(u, S2.hero, S2.hero + 0.4), st = po(u, S2.strips, S2.strips + 0.5), [hx, hy, hw, hh] = tileXY(HERO);
  rr(AX, hy, s, hh, C.blue, st); rr(hx, BY, hw, s, C.yel, st);
  line(AX + s, hy + hh / 2, hx, hy + hh / 2, C.blue, 2, st * 0.6, [6, 6]); line(hx + hw / 2, BY + s, hx + hw / 2, hy, C.yelD, 2, st * 0.6, [6, 6]);
  rr(hx, hy, hw, hh, C.purple, h); sr(hx - 3, hy - 3, hw + 6, hh + 6, C.ink, 3, h);
  txt('A 的横条：128 × 4096', AX, hy - 12, { size: 22, color: C.blue, weight: 700, alpha: st });
  txt('这一块', hx + hw + 12, hy + hh + 6, { size: 24, color: C.purple, weight: 800, alpha: h });
  txt('B 的竖条', hx + hw + 10, BY + 30, { size: 22, color: C.yelD, weight: 700, alpha: st });
  txt('4096 × 256', hx + hw + 10, BY + 58, { size: 22, color: C.yelD, weight: 600, alpha: st, font: MONO });
  // 芯片：148 个 SM，块按轮飞过去；每轮最多 148 块，第 4 轮只剩 68 块
  const [X, Y, Wc] = CHIP2, a = po(u, bt(1, 2) - 0.4, bt(1, 2) + 0.2);
  chip(X, Y, Wc, { alpha: a, detail: false, off: B200_OFF, paint: (n) => {
    const slot = B200_ON.indexOf(n); if (slot < 0 || wave < 0) return null;
    const tile = wave * 148 + slot; if (tile >= 512) return ['#4a483f', 1];
    return landed(slot) ? [tile === HERO ? C.purple : '#b49cf2', 1] : null;
  } });
  if (wave >= 0) for (let slot = 0; slot < 148; slot++) {               // 正在飞的块：同一行一起走
    const tile = wave * 148 + slot; if (tile >= 512) break;
    const t0 = S2delay(slot), k = p(wu, t0, t0 + S2.fly); if (k <= 0 || k >= 1) continue;
    const [tx, ty, tw_, th] = tileXY(tile), [cx, cy, cw, ch] = chipCell(X, Y, Wc, B200_ON[slot]);
    rr(mix(tx, cx, k), mix(ty, cy, k), mix(tw_, cw, k), mix(th, ch, k), tile === HERO ? C.purple : '#9a7ff0', 0.9);
  }
  txt(`第 ${clamp(wave + 1, 1, 4)} 轮 · 已分配 ${Math.min(512, done)} / 512 块`, X + Wc / 2, Y + Wc * 0.92 + 48, { size: 30, weight: 700, align: 'center', alpha: a * (wave >= 0 ? 1 : 0) });
  txt('第 4 轮只有 68 块，80 个 SM 空闲', X + Wc / 2, Y + Wc * 0.92 + 88, { size: 26, color: C.red, weight: 600, align: 'center', alpha: po(u, w0 + 3 * ws + 1.0, w0 + 3 * ws + 1.4) });
}

// ================= 3 芯片 =================
const S3 = { x: 96, y: 270, w: 700 };
ev(2, 0.9, 'pop'); ev(2, 2.3, 'pop'); ev(2, 3.7, 'pop'); ev(2, bt(2, 1), 'click'); ev(2, bt(2, 1) + 0.7, 'land'); ev(2, bt(2, 2), 'pop');
function s3(u) {
  const { x, y, w } = S3, uu = w / 100, hs = HERO_SM(), h = po(u, bt(2, 1) + 0.6, bt(2, 1) + 1.0);
  chip(x, y, w, { off: B200_OFF, alpha: po(u, 0, 0.5), paint: (n) => (n === hs ? [C.purple, h] : null) });
  // 主角块所在的 GPC：20 格框出
  const gpc = Math.floor(hs / 20), c0 = chipCell(x, y, w, gpc * 20), c1 = chipCell(x, y, w, gpc * 20 + 19), gb = po(u, bt(2, 1), bt(2, 1) + 0.5);
  sr(c0[0] - 6, c0[1] - 6, c1[0] + c1[2] - c0[0] + 12, c1[1] + c1[3] - c0[1] + 12, C.purple, 4, gb);
  const [hx, hy, hw, hh] = chipCell(x, y, w, hs);
  callout(x + 71.7 * uu, y + 18 * uu, 920, 330, 'HBM3e × 8 组', '共 180 GB，每秒送 8 TB', po(u, 0.8, 1.3));
  callout(x + 79.5 * uu, y + 40 * uu, 920, 450, '2 个 die', '208B 晶体管，TSMC 4NP', po(u, 2.2, 2.7));
  callout(x + 50 * uu, y + 64 * uu, 920, 570, 'die 间互连 NV-HBI', '每秒 10 TB，软件看到的是一块 GPU', po(u, 3.6, 4.1));
  callout(hx + hw / 2, hy + hh / 2, 920, 690, 'GPC × 8，每个 20 个 SM', 'B200 启用 148 个 SM；紫色这一格算主角块', gb, C.purple);
  placeholder(920, 790, 904, 170, ['B200 上运行 deviceQuery：SM 数、L2 容量、shared memory 上限', '需要：Modal 上的 B200 一台，约 1 分钟'], po(u, bt(2, 2), bt(2, 2) + 0.5));
}

// ================= 4 一个 SM =================
// 每 SM：128 个 FP32 core、4 个 Tensor Core、寄存器 256 KB、L1 / shared memory 256 KB（shared 最多 228 KB）、TMEM 256 KB
// 右边按面积 = 容量：边长 22 px 代表 256 KB；L2 126 MB 边长约 494 px；HBM 180 GB 边长约 1.9 万 px
const S4 = { U: 22, l2: [bt(3, 2) + 1.0, bt(3, 2) + 2.4], hbm: [bt(3, 2) + 3.0, bt(3, 2) + 3.8] };
for (let q = 0; q < 4; q++) ev(3, 0.6 + q * 0.35, 'pop'); ev(3, bt(3, 1), 'land'); ev(3, bt(3, 1) + 0.7, 'land');
ev(3, bt(3, 2), 'click'); ev(3, S4.l2[0], 'whoosh'); ev(3, S4.hbm[0], 'whoosh');
function s4(u) {
  const a = po(u, 0.2, 0.6);
  sr(100, 268, 840, 694, C.ink, 3, a);
  for (let q = 0; q < 4; q++) {
    const px = 120 + (q % 2) * 410, py = 288 + Math.floor(q / 2) * 228, al = po(u, 0.6 + q * 0.35, 1.0 + q * 0.35);
    rr(px, py, 390, 210, C.paper2, al); txt(`分区 ${q}`, px + 16, py + 34, { size: 24, weight: 700, alpha: al });
    for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) rr(px + 16 + c * 22, py + 52 + r * 22, 18, 18, C.gray, al);
    txt('32 个 FP32 core', px + 16, py + 172, { size: 20, color: C.ink2, alpha: al });
    rr(px + 210, py + 48, 164, 72, C.teal, al); txt('Tensor Core', px + 292, py + 93, { size: 22, weight: 700, color: '#fff', align: 'center', alpha: al });
    rr(px + 210, py + 130, 164, 60, MEM.reg, al); txt('寄存器 64 KB', px + 292, py + 168, { size: 22, weight: 700, color: '#fff', align: 'center', alpha: al });
  }
  const b = po(u, bt(3, 1), bt(3, 1) + 0.5), c = po(u, bt(3, 1) + 0.7, bt(3, 1) + 1.2);
  rr(120, 758, 800, 72, MEM.smem, b); txt('L1 / shared memory 256 KB（shared 最多 228 KB）', 520, 803, { size: 24, weight: 700, color: '#fff', align: 'center', alpha: b });
  rr(120, 846, 800, 96, MEM.tmem, c); txt('Tensor Memory（TMEM）256 KB', 520, 886, { size: 26, weight: 800, color: C.purple, align: 'center', alpha: c });
  txt('128 lane × 512 列 × 4 字节，Blackwell 新增', 520, 922, { size: 22, color: C.ink2, align: 'center', alpha: c });
  // 右边：同一比例下的容量
  const d = po(u, bt(3, 2), bt(3, 2) + 0.5), U = S4.U, RX = 1000;
  txt('同一比例：面积 = 容量', RX, 300, { size: 26, weight: 700, color: C.ink2, alpha: d });
  [['寄存器 256 KB', 1, MEM.reg], ['shared memory 228 KB', 228 / 256, MEM.smem], ['TMEM 256 KB', 1, MEM.tmem]].forEach(([l, f, col], i) => {
    const sz = U * Math.sqrt(f), yy = 334 + i * 40; rr(RX, yy + (U - sz), sz, sz, col, d); txt(l, RX + 40, yy + 20, { size: 22, alpha: d });
  });
  txt('（以上为一个 SM）', RX, 476, { size: 20, color: C.gray, alpha: d });
  const l2 = U * Math.sqrt(126 * 1024 / 256) * Math.sqrt(po(u, ...S4.l2)), LY = 490;
  rr(RX, LY, l2, l2, MEM.l2, d); txt('L2 126 MB · 全芯片', RX + 24, LY + 50, { size: 30, weight: 800, color: '#fff', alpha: po(u, S4.l2[1] - 0.4, S4.l2[1]) });
  const hb = po(u, ...S4.hbm), HX = mix(W, 1520, hb);
  rr(HX, LY, W - HX, H - LY, MEM.hbm, hb);
  txt('HBM 180 GB', HX + 24, LY + 50, { size: 30, weight: 800, alpha: hb });
  txt('同一比例下边长约', HX + 24, LY + 92, { size: 22, alpha: hb }); txt('1.9 万像素，画面', HX + 24, LY + 124, { size: 22, alpha: hb }); txt('只露出一角', HX + 24, LY + 156, { size: 22, alpha: hb });
}

// ================= 5 线程与硬件 =================
const S5 = { run: bt(4, 2) + 0.4, per: 1.1, seq: [5, 22, 41, 13, 58, 30] };
evRun(4, 0.5, 2.0, 0.09, 'tick'); ev(4, 2.6, 'land'); ev(4, bt(4, 1), 'click'); for (let i = 0; i < 4; i++) ev(4, bt(4, 1) + 0.6 + i * 0.4, 'pop');
for (let i = 0; i < S5.seq.length; i++) ev(4, S5.run + i * S5.per, 'click');
function s5(u) {
  const RY = (i) => 320 + i * 72;
  for (let i = 0; i < 4; i++) {
    const n = i === 0 ? Math.floor(32 * lin(u, 0.5, 2.0)) : (u > 2.4 + i * 0.25 ? 32 : 0), al = i === 0 ? 1 : po(u, 2.4 + i * 0.25, 2.7 + i * 0.25);
    for (let j = 0; j < n; j++) rr(96 + j * 16, RY(i), 12, 44, C.blue, al * (i === 0 ? 1 : 0.85));
    txt(`warp ${i}`, 620, RY(i) + 32, { size: 26, weight: 700, color: C.blue, alpha: i === 0 ? po(u, 2.0, 2.3) : al });
  }
  txt('1 根竖条 = 1 个线程', 96, 300, { size: 22, color: C.gray, alpha: po(u, 0.5, 0.9) });
  txt('4 个 warp = 1 个 warpgroup = 128 个线程', 96, 650, { size: 28, weight: 700, alpha: po(u, 3.4, 3.8) });
  // 4 个 warp ↔ 4 个分区 ↔ TMEM 的 4 段 lane
  for (let i = 0; i < 4; i++) {
    const al = po(u, bt(4, 1) + 0.6 + i * 0.4, bt(4, 1) + 0.9 + i * 0.4), y = RY(i);
    arrow(735, y + 22, 790, y + 22, al, C.ink, al, 3);
    rr(800, y, 170, 44, C.paper2, al); txt(`分区 ${i}`, 885, y + 31, { size: 22, weight: 700, align: 'center', alpha: al });
    arrow(975, y + 22, 1020, y + 22, al, C.ink, al, 3);
    rr(1030, y, 220, 44, MEM.tmem, al); txt(`TMEM lane ${i * 32}–${i * 32 + 31}`, 1140, y + 31, { size: 20, weight: 700, color: C.purple, align: 'center', alpha: al });
  }
  txt('每个 warp 只能读写 TMEM 的 32 个 lane', 800, 650, { size: 26, weight: 700, color: C.purple, alpha: po(u, bt(4, 1) + 2.4, bt(4, 1) + 2.8) });
  // 右边：64 个 warp 槽，一个等数据就换另一个
  const GX = 1330, GY = 300, CS = 60, a = po(u, bt(4, 2), bt(4, 2) + 0.5), k = Math.floor((u - S5.run) / S5.per);
  for (let n = 0; n < 64; n++) {
    let col = C.lg;
    S5.seq.forEach((w, i) => { if (w === n && k >= i) col = k === i ? C.teal : k - i <= 3 ? C.red : C.lg; });
    rr(GX + (n % 8) * CS, GY + Math.floor(n / 8) * CS, CS - 8, CS - 8, col, a);
  }
  txt('一个 SM 最多驻留 64 个 warp', GX, 810, { size: 26, weight: 700, alpha: a });
  rr(GX, 836, 24, 24, C.teal, a); txt('执行', GX + 34, 857, { size: 22, alpha: a }); rr(GX + 110, 836, 24, 24, C.red, a); txt('等数据', GX + 144, 857, { size: 22, alpha: a });
  rr(GX + 250, 836, 24, 24, C.lg, a); txt('就绪', GX + 284, 857, { size: 22, alpha: a });
}

// ================= 6 跟一个块走一遍 =================
// 每段 K = 64：A 128 × 64（16 KB）、B 256 × 64（32 KB）；4 个缓冲共 192 KB；结果 128 × 256 FP32 = 128 KB，占 TMEM 一半
const B6 = { hbm: [96, 420, 180, 240], l2: [350, 450, 150, 180], smem: [580, 360, 480, 340], tc: [1110, 460, 200, 140], tmem: [1360, 360, 464, 340], reg: [1360, 760, 464, 64] };
const slot6 = (i) => [600 + i * 114, 420, 100, 240];
const S6 = { fill: (i) => bt(5, 1) + 0.4 + i * 0.9, trip: 0.8, T0: bt(5, 2) + 0.2, PER: 1.3, NS: 5, fast: [bt(5, 3) + 0.1, bt(5, 3) + 1.9], epi: [bt(5, 3) + 2.2, bt(5, 3) + 4.6] };
const ctr = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
function along(pts, k) {                                   // 折线上的位置，k 从 0 到 1
  const seg = pts.length - 1, f = clamp(k) * seg, i = Math.min(seg - 1, Math.floor(f)), t = f - i;
  return [mix(pts[i][0], pts[i + 1][0], t), mix(pts[i][1], pts[i + 1][1], t)];
}
for (let i = 0; i < 4; i++) { ev(5, S6.fill(i), 'click'); ev(5, S6.fill(i) + S6.trip, 'land'); }
for (let s = 0; s < S6.NS; s++) { ev(5, S6.T0 + s * S6.PER, 'pop'); ev(5, S6.T0 + s * S6.PER + 0.6, 'click'); }
evRun(5, S6.fast[0], S6.fast[1], 0.06, 'tick'); ev(5, S6.epi[0], 'whoosh'); ev(5, S6.epi[1], 'land');
function slotFull6(i, u) {
  let f = u >= S6.fill(i) + S6.trip ? 1 : 0;
  for (let s = i; s < S6.NS; s += 4) { const tc = S6.T0 + s * S6.PER; if (u >= tc) f = 1 - lin(u, tc, tc + 0.6); if (u >= tc + 1.4) f = 1; }
  if (u >= S6.fast[0]) f = 1; if (u >= S6.fast[1] - 0.2) f = 1 - lin(u, S6.fast[1] - 0.2, S6.fast[1] + 0.2);
  return f;
}
function done6(u) {
  if (u < S6.T0) return 0; if (u < S6.fast[0]) return Math.min(S6.NS, Math.floor((u - S6.T0 - 0.6) / S6.PER) + 1);
  return Math.min(64, S6.NS + Math.floor(59 * lin(u, ...S6.fast)));
}
function s6(u) {
  const a = po(u, 0.5, 1.2), lab = (b, t, col = '#fff') => txt(t, b[0] + b[2] / 2, b[1] + 44, { size: 28, weight: 800, color: col, align: 'center', alpha: a });
  rr(...B6.hbm, MEM.hbm, a); lab(B6.hbm, 'HBM', C.ink); rr(...B6.l2, MEM.l2, a); lab(B6.l2, 'L2');
  rr(...B6.smem, C.paper2, a); sr(...B6.smem, MEM.smem, 4, a); txt('shared memory', B6.smem[0] + 16, B6.smem[1] + 40, { size: 26, weight: 800, color: MEM.smem, alpha: a });
  rr(...B6.tc, C.teal, a); lab(B6.tc, 'Tensor Core');
  rr(...B6.tmem, C.paper2, a); sr(...B6.tmem, C.purple, 4, a); txt('TMEM', B6.tmem[0] + 16, B6.tmem[1] + 40, { size: 26, weight: 800, color: C.purple, alpha: a });
  const y0 = 540; arrow(276, y0, 346, y0, a, C.ink, a, 3); arrow(500, y0, 576, y0, a, C.ink, a, 3); arrow(1060, 530, 1106, 530, a, C.ink, a, 3); arrow(1310, 530, 1356, 530, a, C.ink, a, 3);
  // 4 个缓冲：A 16 KB 在上，B 32 KB 在下，高度按字节数
  for (let i = 0; i < 4; i++) {
    const [x, y, w, h] = slot6(i), f = slotFull6(i, u);
    sr(x, y, w, h, MEM.smem, 2, a * 0.6); rr(x, y, w, h / 3, C.blue, a * f); rr(x, y + h / 3, w, h * 2 / 3, C.yel, a * f);
    txt(`缓冲 ${i}`, x + w / 2, y + h + 30, { size: 20, color: C.ink2, align: 'center', alpha: a });
  }
  txt('4 个缓冲 × 48 KB（A 16 KB + B 32 KB）= 192 KB', B6.smem[0], 760, { size: 22, color: C.ink2, alpha: po(u, bt(5, 1) + 3.6, bt(5, 1) + 4.0) });
  // TMA：一个线程发起
  const t = po(u, bt(5, 1), bt(5, 1) + 0.4);
  rr(350, 664, 10, 34, C.blue, t); txt('TMA：1 个线程发起', 370, 690, { size: 22, weight: 700, alpha: t });
  // 搬运的数据包：HBM → L2 → 缓冲
  const loadPath = (i) => { const [x, y, w, h] = slot6(i); return [ctr(B6.hbm), ctr(B6.l2), [x + w / 2, y + h / 2]]; };
  const pkt = (pt, al = 1) => { rr(pt[0] - 16, pt[1] - 20, 32, 13, C.blue, al); rr(pt[0] - 16, pt[1] - 5, 32, 26, C.yel, al); };
  for (let i = 0; i < 4; i++) { const k = lin(u, S6.fill(i), S6.fill(i) + S6.trip); if (k > 0 && k < 1) pkt(along(loadPath(i), k)); }
  for (let s = 0; s < S6.NS; s++) {
    const tc = S6.T0 + s * S6.PER, i = s % 4, [x, y, w, h] = slot6(i);
    const kc = lin(u, tc, tc + 0.6); if (kc > 0 && kc < 1) { const pt = along([[x + w / 2, y + h / 2], ctr(B6.tc), [1480, 540]], kc); rr(pt[0] - 14, pt[1] - 14, 28, 28, C.purple, 0.9); }
    if (kc > 0 && kc < 1) sr(x - 5, y - 5, w + 10, h + 10, C.teal, 5, 1);
    const kr = lin(u, tc + 0.6, tc + 1.4); if (kr > 0 && kr < 1) pkt(along(loadPath(i), kr));
  }
  // TMEM：512 列中的 256 列放主角块的结果，颜色随累加的段数加深
  const n = done6(u), [mx, my] = [B6.tmem[0] + 16, B6.tmem[1] + 64];
  rr(mx, my, 216, 250, C.purple, a * (0.12 + 0.88 * n / 64)); sr(mx, my, 216, 250, C.purple, 2, a);
  txt('结果 128 × 256', mx + 108, my + 120, { size: 24, weight: 800, color: n > 30 ? '#fff' : C.purple, align: 'center', alpha: a });
  txt('FP32 · 128 KB', mx + 108, my + 152, { size: 22, color: n > 30 ? '#fff' : C.purple, align: 'center', alpha: a });
  rr(mx + 228, my, 204, 250, '#ece6f8', a); txt('另一半空着', mx + 330, my + 120, { size: 22, color: C.gray, align: 'center', alpha: a }); txt('可放下一块的结果', mx + 330, my + 152, { size: 20, color: C.gray, align: 'center', alpha: a });
  txt(`第 ${n} / 64 段`, ctr(B6.tc)[0], B6.tc[1] - 22, { size: 30, weight: 800, align: 'center', alpha: po(u, S6.T0 - 0.3, S6.T0) });
  // 收尾：结果读回寄存器，写回 HBM
  const e = po(u, S6.epi[0] - 0.4, S6.epi[0]);
  rr(...B6.reg, MEM.reg, e); txt('寄存器', B6.reg[0] + B6.reg[2] / 2, B6.reg[1] + 43, { size: 26, weight: 800, color: '#fff', align: 'center', alpha: e });
  const back = [[1592, 700], [1592, 760], [1592, 824], [1592, 880], [186, 880], [186, 664]];
  for (let i = 0; i + 1 < back.length; i++) if (i !== 1) line(back[i][0], back[i][1], back[i + 1][0], back[i + 1][1], C.purple, 3, e * 0.6, [8, 6]);
  const ke = lin(u, ...S6.epi); if (ke > 0 && ke < 1) { const pt = along(back, ke); rr(pt[0] - 20, pt[1] - 12, 40, 24, C.purple, 1); }
  txt('写回 C 的这一块', 600, 870, { size: 24, weight: 700, color: C.purple, alpha: e });
}

// ================= 7 复用在哪发生 =================
// 块内：一段读 48 KB，做 128 × 256 × 64 = 2,097,152 次乘加 = 4,194,304 次运算 → 85 次 / 字节
// 块间（理想）：2 × 4096³ ÷（A、B 67 MB + C 34 MB）≈ 1365 次 / 字节；A、B 共 67 MB < L2 126 MB
const S7 = { X: 600, BW: 1200, rows: [700, 780, 860] };
const bar7 = (v) => S7.BW * v / 1365;
ev(6, 0.5, 'pop'); ev(6, 2.6, 'land'); ev(6, bt(6, 1), 'pop'); ev(6, bt(6, 1) + 1.6, 'click'); ev(6, bt(6, 2), 'whoosh'); ev(6, bt(6, 2) + 1.4, 'land');
function s7(u) {
  // 左上：一段的计算
  const a = po(u, 0.5, 1.0);
  rr(200, 300, 256, 64, C.yel, a); rr(120, 380, 64, 128, C.blue, a); rr(200, 380, 256, 128, C.purple, a);
  txt('B 一段 64 × 256 · 32 KB', 200, 292, { size: 20, weight: 700, color: C.yelD, alpha: a });
  txt('A 一段', 120, 532, { size: 20, weight: 700, color: C.blue, alpha: a }); txt('128 × 64 · 16 KB', 120, 558, { size: 20, color: C.blue, alpha: a });
  txt('128 × 256 个结果', 328, 452, { size: 22, weight: 700, color: '#fff', align: 'center', alpha: a });
  const a2 = po(u, 1.6, 2.0);
  txt('读进 48 KB', 500, 400, { size: 26, weight: 800, alpha: a2 });
  txt('每个结果累加 64 次：', 500, 440, { size: 22, alpha: a2 }); txt('210 万次乘加 = 419 万次运算', 500, 472, { size: 22, weight: 700, alpha: a2 });
  txt('每字节 85 次', 500, 520, { size: 30, weight: 800, color: C.tealD, alpha: po(u, 2.4, 2.8) });
  // 右上：相邻的块共用横条与竖条
  const b = po(u, bt(6, 1), bt(6, 1) + 0.5), GX = 1180, GY = 300, cw = 12, chh = 10, hr = 2, hc = 5;
  for (let r = 0; r < TR; r++) for (let q = 0; q < TC; q++) {
    const inRow = r === hr && po(u, bt(6, 1) + 0.8, bt(6, 1) + 1.2), inCol = q === hc && po(u, bt(6, 1) + 1.6, bt(6, 1) + 2.0);
    rr(GX + q * cw, GY + r * chh, cw - 2, chh - 2, r === hr && q === hc ? C.purple : inRow ? '#7d93e0' : inCol ? '#e8c75a' : '#d6c4fb', b);
  }
  rr(GX - 24, GY + hr * chh, 16, chh - 2, C.blue, b * po(u, bt(6, 1) + 0.8, bt(6, 1) + 1.2)); rr(GX + hc * cw, GY - 24, cw - 2, 16, C.yel, b * po(u, bt(6, 1) + 1.6, bt(6, 1) + 2.0));
  txt('同一行 16 块共用 A 的横条', 1400, 340, { size: 24, weight: 700, color: C.blue, alpha: po(u, bt(6, 1) + 0.8, bt(6, 1) + 1.2) });
  txt('同一列 32 块共用 B 的竖条', 1400, 380, { size: 24, weight: 700, color: C.yelD, alpha: po(u, bt(6, 1) + 1.6, bt(6, 1) + 2.0) });
  txt('A、B 共 67 MB，L2 有 126 MB', 1400, 440, { size: 24, alpha: po(u, bt(6, 1) + 2.6, bt(6, 1) + 3.0) });
  txt('理想情况下各从显存读一次', 1400, 476, { size: 24, alpha: po(u, bt(6, 1) + 2.6, bt(6, 1) + 3.0) });
  // 下方：三条对着同一条基线，竖线是需要的 281
  const { X, rows } = S7, rowsDef = [['每次回内存取数', 0.5, C.red, 0], ['块内复用 · shared memory', 85, C.teal, 2.4], ['块间复用 · L2', 1365, C.purple, bt(6, 2)]];
  const base = po(u, 0.3, 0.8);
  line(X, rows[0] - 10, X, rows[2] + 60, C.ink, 3, base);
  rowsDef.forEach(([l, v, col, t0], i) => {
    const al = i === 0 ? base : po(u, t0, t0 + 0.4), k = i === 0 ? 1 : p(u, t0, t0 + 1.2), y = rows[i];
    txt(l, 96, y + 36, { size: 28, weight: 700, alpha: al }); rr(X, y, Math.max(3, bar7(v) * k), 50, col, al);
    txt(`${v} 次 / 字节`, X + Math.max(3, bar7(v) * k) + 14, y + 36, { size: 26, weight: 800, color: col, alpha: al * (i < 2 ? 1 : 0) });
  });
  txt('1365 次 / 字节', X + S7.BW - 16, rows[2] + 36, { size: 28, weight: 800, color: '#fff', align: 'right', alpha: po(u, bt(6, 2) + 1.1, bt(6, 2) + 1.4) });
  const nx = X + bar7(281); line(nx, rows[0] - 30, nx, rows[2] + 66, C.red, 3, base, [10, 8]);
  txt('需要 281', nx + 10, rows[0] - 14, { size: 24, weight: 800, color: C.red, alpha: base });
  txt('2 × 4096³ 次运算 ÷（读 A、B 67 MB + 写 C 34 MB）≈ 1365', X, rows[2] + 96, { size: 22, color: C.gray, alpha: po(u, bt(6, 2) + 1.4, bt(6, 2) + 1.8) });
}

// ================= 8 两个 SM 合作 =================
// cta_group::2：一对 SM 合算 M = 256、N = 256。每个 SM 存 A 的 128 行（16 KB）与 B 的一半（16 KB），每段 32 KB；
// 每个 SM 的运算量不变（128 × 256 × 64 次乘加），每字节运算次数 = 4,194,304 ÷ 32,768 = 128
const P8 = { A: [120, 420, 102, 410], B: [250, 290, 410, 102], C: [250, 420, 410, 410] };
ev(7, 0.5, 'pop'); ev(7, 1.4, 'land'); ev(7, bt(7, 1), 'click'); ev(7, bt(7, 1) + 1.2, 'whoosh'); ev(7, bt(7, 1) + 2.2, 'land'); ev(7, bt(7, 2), 'pop'); ev(7, bt(7, 2) + 1.2, 'land');
function s8(u) {
  const a = po(u, 0.5, 1.0), sp = po(u, 1.4, 1.9);
  const [ax, ay, aw, ah] = P8.A, [bx, by, bw, bh] = P8.B, [cx, cy, cw, ch] = P8.C;
  // SM 0 的部分实色，SM 1 的部分浅色
  rr(ax, ay, aw, ah / 2, C.blue, a); rr(ax, ay + ah / 2, aw, ah / 2, '#7d93e0', a);
  rr(bx, by, bw / 2, bh, C.yel, a); rr(bx + bw / 2, by, bw / 2, bh, '#f6e3a0', a);
  rr(cx, cy, cw, ch / 2, C.purple, a); rr(cx, cy + ch / 2, cw, ch / 2, '#b49cf2', a);
  line(ax - 10, ay + ah / 2, cx + cw + 10, cy + ch / 2, C.ink, 4, sp); line(bx + bw / 2, by - 10, bx + bw / 2, by + bh + 4, C.ink, 4, sp);
  txt('SM 0', cx + cw / 2, cy + ch / 4 + 12, { size: 36, weight: 800, color: '#fff', align: 'center', alpha: sp });
  txt('SM 1', cx + cw / 2, cy + ch * 3 / 4 + 12, { size: 36, weight: 800, color: '#fff', align: 'center', alpha: sp });
  txt('256 × 256 的块', cx + cw / 2, cy + ch + 40, { size: 24, weight: 700, color: C.purple, align: 'center', alpha: a });
  txt('B 的一半在 SM 0', bx + bw / 4, by - 14, { size: 20, weight: 700, color: C.yelD, align: 'center', alpha: sp });
  txt('一半在 SM 1', bx + bw * 3 / 4, by - 14, { size: 20, weight: 700, color: C.yelD, align: 'center', alpha: sp });
  // SM 0 从 SM 1 读 B 的另一半
  const r = po(u, bt(7, 1) + 1.0, bt(7, 1) + 2.0);
  if (r > 0) { g.globalAlpha = r; g.strokeStyle = C.ink; g.lineWidth = 4; g.setLineDash([10, 8]); g.beginPath(); g.moveTo(bx + bw * 3 / 4, by + bh); g.quadraticCurveTo(bx + bw * 0.9, cy + 60, bx + bw * 0.62, cy + ch / 4 - 30); g.stroke(); g.setLineDash([]); g.globalAlpha = 1; }
  txt('SM 0 直接读 SM 1 的', 690, 470, { size: 22, weight: 700, alpha: r }); txt('shared memory', 690, 500, { size: 22, weight: 700, alpha: r });
  // 右边：每个 SM 每段搬的字节
  const KB = 15, RX = 960, b = po(u, bt(7, 1), bt(7, 1) + 0.5), c = po(u, bt(7, 1) + 2.2, bt(7, 1) + 2.7);
  txt('每个 SM 每段搬的数据', RX, 320, { size: 26, weight: 800, alpha: b });
  txt('一个 SM 算 128 × 256', RX, 372, { size: 22, alpha: b }); rr(RX, 386, 16 * KB, 44, C.blue, b); rr(RX + 16 * KB, 386, 32 * KB, 44, C.yel, b);
  txt('48 KB', RX + 48 * KB + 14, 418, { size: 26, weight: 800, alpha: b });
  txt('两个 SM 合算 256 × 256', RX, 482, { size: 22, alpha: c }); rr(RX, 496, 16 * KB, 44, C.blue, c); rr(RX + 16 * KB, 496, 16 * KB, 44, C.yel, c);
  txt('32 KB', RX + 32 * KB + 14, 528, { size: 26, weight: 800, color: C.tealD, alpha: c });
  // 每字节运算次数：85 → 128
  const d = po(u, bt(7, 2), bt(7, 2) + 0.5), F = 4.4, k = p(u, bt(7, 2) + 0.4, bt(7, 2) + 1.2);
  txt('每字节运算次数', RX, 630, { size: 26, weight: 800, alpha: d });
  rr(RX, 650, 85 * F, 40, C.teal, d); txt('85', RX + 85 * F + 12, 680, { size: 26, weight: 800, alpha: d });
  rr(RX, 704, mix(85, 128, k) * F, 40, C.tealD, d); txt('128', RX + mix(85, 128, k) * F + 12, 734, { size: 26, weight: 800, color: C.tealD, alpha: d });
  txt('cluster 最多 16 个 CTA；TMA 还能把同一份数据同时送给 cluster 里的多个 SM', RX, 820, { size: 22, color: C.gray, alpha: po(u, bt(7, 2) + 1.6, bt(7, 2) + 2.0) });
}

// ================= 9 与 Hopper 对照 =================
const R9 = [['die', 1, 2, ''], ['SM', 132, 148, ''], ['L2', 50, 126, ' MB'], ['BF16 算力', 989, 2250, ' TFLOPS'], ['HBM 带宽', 3.35, 8, ' TB/s'], ['每字节需要的运算', 295, 281, ' 次']];
for (let i = 0; i < 5; i++) ev(8, 0.6 + i * 0.8, 'pop'); ev(8, bt(8, 1), 'land'); ev(8, bt(8, 2), 'click'); ev(8, bt(8, 2) + 0.6, 'click');
function s9(u) {
  const X = 560, BW = 900, a = po(u, 0.3, 0.7);
  rr(1560, 262, 24, 24, C.gray, a); txt('H100', 1594, 283, { size: 24, weight: 700, alpha: a });
  rr(1680, 262, 24, 24, C.teal, a); txt('B200', 1714, 283, { size: 24, weight: 700, alpha: a });
  R9.forEach(([l, h, b, unit], i) => {
    const t0 = i < 5 ? 0.6 + i * 0.8 : bt(8, 1), al = po(u, t0, t0 + 0.4), k = p(u, t0, t0 + 0.9), y = 300 + i * 88, mx = Math.max(h, b);
    if (i === 5) rr(80, y - 16, 1760, 84, '#e8f3f0', al);
    txt(l, 96, y + 36, { size: 28, weight: 700, alpha: al });
    rr(X, y, BW * h / mx * k, 24, C.gray, al); txt(`${h}${unit}`, X + BW * h / mx * k + 12, y + 21, { size: 22, weight: 700, color: C.ink2, alpha: al });
    rr(X, y + 30, BW * b / mx * k, 24, C.teal, al); txt(`${b}${unit}`, X + BW * b / mx * k + 12, y + 51, { size: 22, weight: 800, color: C.tealD, alpha: al });
  });
  txt('989 ÷ 3.35 ≈ 295，2250 ÷ 8 ≈ 281', X, 300 + 5 * 88 + 86, { size: 20, color: C.gray, font: MONO, alpha: po(u, bt(8, 1) + 1.0, bt(8, 1) + 1.4) });
  const c = po(u, bt(8, 2), bt(8, 2) + 0.4), d = po(u, bt(8, 2) + 0.6, bt(8, 2) + 1.0), y = 880;
  txt('累加结果在哪', 96, y + 36, { size: 28, weight: 700, alpha: c });
  rr(X, y, 24, 24, C.gray, c); txt('寄存器 · wgmma，128 个线程一起发起', X + 36, y + 21, { size: 24, weight: 700, color: C.ink2, alpha: c });
  rr(X, y + 34, 24, 24, C.teal, d); txt('TMEM · tcgen05.mma，1 个线程发起', X + 36, y + 55, { size: 24, weight: 800, color: C.tealD, alpha: d });
}

// ================= 10 结尾 =================
// 回到开场的画面：HBM、芯片、281 格；需要 281，做到 1365（理想情况）
const S10 = { list: (i) => bt(9, 1) + i * 0.5, bars: bt(9, 1) + 2.4 };
evRun(9, 0.6, 3.4, 0.2, 'tick'); for (let i = 0; i < 4; i++) ev(9, S10.list(i), 'pop'); ev(9, S10.bars, 'land'); ev(9, S10.bars + 1.2, 'land'); ev(9, bt(9, 2), 'pop');
function s10(u) {
  const a = po(u, 0.2, 0.7);
  rr(96, 360, 300, 200, MEM.hbm, a); txt('HBM', 246, 450, { size: 48, weight: 800, align: 'center', alpha: a }); txt('每秒送来 8 TB', 246, 610, { size: 36, weight: 700, align: 'center', alpha: a });
  rr(396, 446, 354, 28, C.paper2, a);
  const lit = Math.floor((u - 0.6) / 0.7);                     // 一轮轮亮起
  chip(750, 300, 420, { alpha: a, detail: false, off: B200_OFF, paint: (n) => (u > 0.6 && B200_ON.includes(n) && (lit % 2 === 0) ? ['#b49cf2', 1] : null) });
  for (let i = 0; i < 281; i++) rr(1340 + (i % 20) * 22, 370 + Math.floor(i / 20) * 22, 18, 18, C.teal, a);
  arrow(1180, 460, 1300, 460, a, C.ink, a * 0.6);
  txt('Tensor Core · 每秒 2250 万亿次运算', 1560, 340, { size: 26, weight: 700, color: C.tealD, align: 'center', alpha: a });
  // 四层复用
  [['块内复用', 'shared memory'], ['结果留在', 'TMEM'], ['两个 SM', '共用 B'], ['块间复用', 'L2']].forEach(([l1, l2], i) => {
    const al = po(u, S10.list(i), S10.list(i) + 0.3), x = 96 + i * 230;
    rr(x, 700, 214, 92, C.paper2, al); txt(l1, x + 18, 738, { size: 24, weight: 800, alpha: al }); txt(l2, x + 18, 772, { size: 22, color: C.ink2, alpha: al });
  });
  // 需要 281，做到 1365
  const X = 96, BW = 1000, b = po(u, S10.bars, S10.bars + 0.4), y = 830;
  line(X, y - 6, X, y + 120, C.ink, 3, b);
  rr(X, y, BW * 281 / 1365, 44, C.red, b); txt('需要 281', X + BW * 281 / 1365 + 14, y + 33, { size: 26, weight: 800, color: C.red, alpha: b });
  const k = p(u, S10.bars + 1.0, S10.bars + 2.0);
  rr(X, y + 60, BW * k, 44, C.purple, po(u, S10.bars + 1.0, S10.bars + 1.3)); txt('做到 1365（理想情况）', X + BW * k - 14, y + 93, { size: 26, weight: 800, color: '#fff', align: 'right', alpha: po(u, S10.bars + 1.8, S10.bars + 2.1) });
  placeholder(1180, 716, 644, 190, ['实测：B200 上 4096³ BF16 矩阵乘法', '的用时与 TFLOPS', '需要：Modal 上的 B200 一台，约 1 分钟'], po(u, bt(9, 2), bt(9, 2) + 0.5));
}

const DRAW = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10];
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
