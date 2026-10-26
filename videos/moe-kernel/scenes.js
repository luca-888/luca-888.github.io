// MoE Kernel 影片的各幕。每幕一个 draw(u)，u 是幕内时间（秒）；节拍起点取自 script.js，音效事件登记进 EV。
// 贯穿全片的图形：128 个 expert 的网格（8 行 × 16 列，编号按行）。开场与结尾都回到它。
// 数据：DATA（data.js，H100 实测）。颜色：token blue，expert / 有用的计算 teal，补齐与空闲 gray，开销 red，权重 yellow。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const EV = SCRIPT.map(() => []);
const ev = (k, u, kind) => EV[k].push([u, kind]);
const evRun = (k, a, b, step, kind) => { for (let u = a; u < b - 1e-6; u += step) ev(k, u, kind); };
const NAME = 'MoE Kernel';
const D = DATA, TOP = 8, NT = D.ids.length;                   // 16 个 token，每个选 8 个 expert
const CNT = Array(128).fill(0); D.ids.flat().forEach((e) => CNT[e]++);
const ACTIVE = CNT.map((c, e) => (c ? e : -1)).filter((e) => e >= 0);   // 65 个
const fmt = (x, d = 0) => x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

function header(k, u) {
  const s = SCRIPT[k], a = po(u, 0, 0.5);
  txt(NAME, 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 76, { size: 26, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 226 + (1 - a) * 16, 56, 700, a);
}
// 128 个 expert 的网格：fill(e) 返回 [颜色, 透明度]，没有返回则画空格
function grid(x, y, c, fill, alpha = 1, label = true) {
  for (let e = 0; e < 128; e++) {
    const gx = x + (e % 16) * c, gy = y + Math.floor(e / 16) * c, f = fill(e);
    rr(gx, gy, c - 6, c - 6, C.paper2, alpha);
    if (f) rr(gx, gy, c - 6, c - 6, f[0], alpha * f[1]);
    if (label && c >= 40) txt(String(e), gx + (c - 6) / 2, gy + (c - 6) / 2 + 7, { size: 18, weight: 600, color: f && f[1] > 0.5 ? '#fff' : C.gray, align: 'center', font: MONO, alpha });
  }
}
const cellXY = (x, y, c, e) => [x + (e % 16) * c + (c - 6) / 2, y + Math.floor(e / 16) * c + (c - 6) / 2];
const tealK = (k) => mixc('#cfeee8', C.tealD, clamp(k));       // 分到的 token 越多颜色越深

// ================= 1 开场 =================
const G1 = { x: 830, y: 330, c: 62 };
const S1 = { out: [4.6, 5.8], one: bt(0, 1) + 0.4, many: [bt(0, 2) + 0.4, bt(0, 2) + 5.6] };
ev(0, 0.25, 'land'); evRun(0, 1.0, 3.0, 0.1, 'tick'); ev(0, S1.out[0], 'whoosh'); ev(0, S1.one, 'pop');
for (let t = 1; t < NT; t++) ev(0, S1.many[0] + (S1.many[1] - S1.many[0]) * t / NT, 'click');
function s1(u) {
  const out = p(u, ...S1.out), k = po(u, 0.15, 1.0);
  txt(NAME, 96, mix(420, 76, out) + (1 - k) * 40, { size: mix(130, 28, out), weight: mix(800, 700, out), color: out < 0.5 ? C.ink : C.ink2, alpha: k });
  txt('一个 MoE 层在 GPU 上的计算过程', 100, 500, { size: 38, weight: 700, color: C.tealD, alpha: po(u, 1.2, 1.8) * (1 - out) });
  const { x, y, c } = G1, n = Math.floor(128 * lin(u, 1.0, 3.0));
  // 当前 token：开头一个，随后 16 个依次进来；累计的颜色按分到的 token 数加深
  const [m0, m1] = S1.many, tt = u < m0 ? 0 : Math.min(NT - 1, Math.floor((u - m0) / (m1 - m0) * NT));
  const cum = Array(128).fill(0); if (u >= m0) for (let t = 0; t <= tt; t++) D.ids[t].forEach((e) => cum[e]++);
  const cur = u >= S1.one ? new Set(D.ids[tt]) : new Set();
  grid(x, y, c, (e) => (e >= n ? [C.paper, 1] : cur.has(e) ? [C.teal, 1] : cum[e] ? [tealK(0.15 + cum[e] / 10), 0.8] : null), 1);
  txt('128 个 expert', x + 8 * c - 3, y + 8 * c + 34, { size: 30, weight: 700, color: C.ink2, align: 'center', alpha: po(u, 2.6, 3.2) });
  // token：左侧一个蓝色块，连线到它的 8 个 expert
  const a = po(u, S1.one, S1.one + 0.4) * (1 - out * 0) * (u >= S1.out[1] ? 1 : 0);
  if (a > 0) {
    const TX = 140, TY = 560;
    rr(TX, TY, 300, 90, C.blue, a);
    txt(u < m0 ? '一个 token' : `token ${tt + 1} / 16`, TX + 150, TY + 58, { size: 34, weight: 800, color: '#fff', align: 'center', alpha: a, font: ZH });
    D.ids[tt].forEach((e) => { const [ex, ey] = cellXY(x, y, c, e); line(TX + 300, TY + 45, ex - 20, ey, C.blue, 2.5, a * 0.6); });
    txt('激活 8 个', TX + 150, TY + 130, { size: 30, weight: 700, color: C.blue, align: 'center', alpha: a });
  }
  txt(`累计激活 ${cum.filter(Boolean).length} 个 expert`, x + 8 * c - 3, y + 8 * c + 74, { size: 28, weight: 700, color: C.tealD, align: 'center', alpha: po(u, m0 + 0.3, m0 + 0.7), font: ZH });
}

// ================= 2 一个 expert =================
// SwiGLU：x（2048）→ gate、up（各 2048 × 768）→ silu(g) ⊙ u（768）→ down（768 × 2048）→ 2048。按 0.21 px / 维画出真实比例
const SC = 0.21, L2 = 2048 * SC, L7 = 768 * SC;      // 430、161 px
const S2 = { draw: [0.5, 2.6], flow: [bt(1, 1) + 0.2, bt(1, 1) + 5.8], size: bt(1, 2) + 0.2 };
ev(1, 0.6, 'pop'); ev(1, 1.2, 'pop'); ev(1, 1.8, 'pop'); evRun(1, S2.flow[0], S2.flow[1], 1.4, 'click'); ev(1, S2.size, 'land');
function vecV(x, y, h, col, a, lab) { rr(x, y, 26, h, col, a); if (lab) txt(lab, x + 13, y + h + 32, { size: 22, weight: 700, color: C.ink2, align: 'center', alpha: a }); }
function s2(u) {
  const Y = 330, f = lin(u, ...S2.flow), a = (i) => po(u, S2.draw[0] + i * 0.4, S2.draw[0] + i * 0.4 + 0.5);
  const hot = (a0, b0) => (f > a0 && f <= b0 ? 1 : 0);
  // x → gate / up
  vecV(110, Y, L2, hot(0, 0.25) ? C.blue : '#8ea3e6', a(0), 'x · 2048');
  rr(200, Y, L7, L2, hot(0, 0.25) ? C.yel : '#f6e3a0', a(1)); txt('gate', 200 + L7 / 2, Y + L2 / 2, { size: 30, weight: 800, color: C.yelD, align: 'center', alpha: a(1) });
  txt('2048 × 768', 200 + L7 / 2, Y + L2 + 32, { size: 20, color: C.gray, align: 'center', font: MONO, alpha: a(1) });
  rr(390, Y, L7, L2, hot(0, 0.25) ? C.yel : '#f6e3a0', a(2)); txt('up', 390 + L7 / 2, Y + L2 / 2, { size: 30, weight: 800, color: C.yelD, align: 'center', alpha: a(2) });
  txt('2048 × 768', 390 + L7 / 2, Y + L2 + 32, { size: 20, color: C.gray, align: 'center', font: MONO, alpha: a(2) });
  arrow(570, Y + 120, 640, Y + 120, a(2), C.ink, a(2) * 0.5, 3);
  // 两个 768 维的中间结果，逐元素相乘
  vecV(660, Y + 40, L7, hot(0.25, 0.5) ? C.purple : '#c9b6f7', a(3), ''); vecV(700, Y + 40, L7, hot(0.25, 0.5) ? C.purple : '#c9b6f7', a(3), '');
  txt('silu(g) ⊙ u', 700, Y + 40 + L7 + 34, { size: 22, weight: 700, color: C.purple, align: 'center', alpha: a(3) });
  txt('768', 700, Y + 24, { size: 20, color: C.gray, align: 'center', font: MONO, alpha: a(3) });
  arrow(745, Y + 120, 815, Y + 120, a(3), C.ink, a(3) * 0.5, 3);
  vecV(830, Y + 40, L7, hot(0.5, 0.75) ? C.purple : '#c9b6f7', a(4), '');
  // down：768 → 2048
  rr(880, Y + 40, L2, L7, hot(0.5, 0.75) ? C.yel : '#f6e3a0', a(4)); txt('down', 880 + L2 / 2, Y + 40 + L7 / 2 + 10, { size: 30, weight: 800, color: C.yelD, align: 'center', alpha: a(4) });
  txt('768 × 2048', 880 + L2 / 2, Y + 40 + L7 + 32, { size: 20, color: C.gray, align: 'center', font: MONO, alpha: a(4) });
  arrow(1325, Y + 120, 1395, Y + 120, a(4), C.ink, a(4) * 0.5, 3);
  vecV(1410, Y, L2, hot(0.75, 1.01) ? C.tealD : C.tealL, a(5), '输出 · 2048');
  // 流动的高亮：一段跑过
  if (f > 0 && f < 1) { const xs = [123, 480, 700, 1100, 1423], i = Math.min(3, Math.floor(f * 4)), k = f * 4 - i; g.globalAlpha = 0.9; g.fillStyle = C.ink; g.beginPath(); g.arc(mix(xs[i], xs[i + 1], k), Y - 30, 10, 0, 7); g.fill(); g.globalAlpha = 1; }
  // 大小
  const s = po(u, S2.size, S2.size + 0.5), RX = 1520;
  txt('一个 expert', RX, 380, { size: 28, weight: 700, alpha: s }); txt('9.44 MB', RX, 430, { size: 44, weight: 800, color: C.yelD, alpha: s });
  txt('一层 128 个', RX, 510, { size: 28, weight: 700, alpha: s }); txt('1.21 GB', RX, 560, { size: 44, weight: 800, color: C.yelD, alpha: s });
  for (let e = 0; e < 128; e++) rr(RX + (e % 16) * 18, 600 + Math.floor(e / 16) * 18, 14, 14, D.ids[0].includes(e) ? C.teal : C.paper2, s);
  txt('每个 token 读取 8 个', RX, 780, { size: 26, weight: 700, color: C.tealD, alpha: po(u, S2.size + 0.6, S2.size + 1.0) });
}

// ================= 3 router =================
const P0 = D.probs0, PMAX = Math.max(...P0), TOP0 = D.ids[0], W0 = D.w[0];
const S3 = { bars: [0.5, 2.5], pick: bt(2, 1) + 0.2, sum: bt(2, 2) + 0.2 };
evRun(2, S3.bars[0], S3.bars[1], 0.12, 'tick'); ev(2, S3.pick, 'pop'); ev(2, S3.sum, 'land');
function s3(u) {
  const X = 110, BW = 13, BASE = 720, HM = 340, a = po(u, 0.3, 0.8), pk = po(u, S3.pick, S3.pick + 0.6);
  line(X, BASE + 2, X + 128 * BW, BASE + 2, C.ink, 2, a);
  for (let e = 0; e < 128; e++) {
    const h = HM * P0[e] / PMAX * clamp(lin(u, ...S3.bars) * 128 - e), top = TOP0.includes(e);
    rr(X + e * BW, BASE - h, BW - 3, h, top ? mixc('#9aa3a0', C.teal, pk) : C.gray, top ? 1 : mix(1, 0.3, pk));
  }
  txt('expert 0', X, BASE + 34, { size: 20, color: C.gray, font: MONO, alpha: a }); txt('127', X + 128 * BW - 10, BASE + 34, { size: 20, color: C.gray, font: MONO, alpha: a, align: 'right' });
  txt('router 分数（softmax 后）', X, 330, { size: 26, weight: 700, color: C.ink2, alpha: a });
  // 前 8 个：重新归一化后的权重
  // 标注按横向位置错开：与左边相邻标注太近时抬高一行
  let lastX = -99, lift = 0;
  [...TOP0.map((e, i) => [e, i])].sort((p1, p2) => p1[0] - p2[0]).forEach(([e, i]) => {
    const h = HM * P0[e] / PMAX, lx = X + e * BW + 5; lift = lx - lastX < 70 ? lift + 1 : 0; lastX = lx;
    txt(W0[i].toFixed(2), lx, BASE - HM - 16 - lift * 26, { size: 20, weight: 800, color: C.tealD, align: 'center', alpha: pk, font: MONO });
  });
  txt('前 8 个，重新归一化后和为 1', X + 640, BASE + 34, { size: 26, weight: 700, color: C.tealD, alpha: pk });
  // 加权求和：8 段按权重排成一条
  const s = po(u, S3.sum, S3.sum + 0.6); let x = X;
  txt('输出 =', X, 840, { size: 28, weight: 700, alpha: s });
  const TW = 1400;
  TOP0.forEach((e, i) => { const w = TW * W0[i] * clamp(lin(u, S3.sum, S3.sum + 1.6) * 8 - i); rr(x + 110, 810, w - 3, 44, mixc(C.tealL, C.tealD, i / 7), s); if (w > 60) txt(`e${e}`, x + 110 + w / 2, 840, { size: 20, weight: 700, color: '#fff', align: 'center', font: MONO, alpha: s }); x += TW * W0[i]; });
  txt('8 个 expert 的输出按权重加权求和', X + 110, 900, { size: 24, color: C.gray, alpha: s });
}

// ================= 4 一批 token =================
const M4 = { x: 120, y: 320, cw: 11, rh: 24 };
const S4 = { dots: [0.5, 3.5], cols: bt(3, 1) + 0.2, hist: bt(3, 2) + 0.2 };
for (let t = 0; t < NT; t++) ev(3, S4.dots[0] + t * (S4.dots[1] - S4.dots[0]) / NT, 'tick'); ev(3, S4.cols, 'land'); ev(3, S4.hist, 'pop');
function s4(u) {
  const { x, y, cw, rh } = M4, a = po(u, 0.3, 0.7), rows = clamp(lin(u, ...S4.dots) * NT, 0, NT), c = po(u, S4.cols, S4.cols + 0.5);
  for (let e = 0; e < 128; e++) rr(x + e * cw, y, cw - 3, NT * rh, CNT[e] ? mixc(C.paper2, '#cfeee8', c) : mixc(C.paper2, '#e2dccd', c), a);
  for (let t = 0; t < NT; t++) {
    const al = clamp(rows - t); if (al <= 0) continue;
    txt(`${t + 1}`, x - 14, y + t * rh + 18, { size: 16, color: C.gray, align: 'right', font: MONO, alpha: al });
    D.ids[t].forEach((e) => rr(x + e * cw, y + t * rh + 3, cw - 3, rh - 6, C.blue, al));
  }
  txt('16 个 token（每行一个）', x, y - 16, { size: 24, weight: 700, color: C.blue, alpha: a });
  txt('128 个 expert', x + 128 * cw, y - 16, { size: 24, weight: 700, color: C.gray, align: 'right', alpha: a });
  txt('65 个被激活', x + 128 * cw + 30, y + 120, { size: 32, weight: 800, color: C.tealD, alpha: c }); txt('63 个空闲', x + 128 * cw + 30, y + 170, { size: 32, weight: 800, color: C.gray, alpha: c });
  // 每列的 token 数：向下的柱
  const h = po(u, S4.hist, S4.hist + 0.8), HY = y + NT * rh + 20;
  for (let e = 0; e < 128; e++) if (CNT[e]) rr(x + e * cw, HY, cw - 3, CNT[e] * 16 * h, CNT[e] <= 2 ? C.teal : C.tealD, h);
  txt('每个 expert 分配到的 token 数', x, HY + 200, { size: 24, weight: 700, color: C.ink2, alpha: h });
  const t2 = po(u, S4.hist + 1.0, S4.hist + 1.5);
  txt('1 个：32　2 个：19　3 个：7　4 个：3　5 个：3　10 个：1', x + 560, HY + 200, { size: 22, weight: 600, color: C.ink2, alpha: t2 });
  txt('51 个仅分配到 1 到 2 个', x + 128 * cw + 30, HY + 60, { size: 32, weight: 800, color: C.tealD, alpha: t2 });
}

// ================= 5 逐个 expert =================
const LOOP = ['for expert_idx in expert_hit:', '    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])', '    current_state = hidden_states[token_idx]',
  '    gate, up = linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)', '    current_hidden_states = self.act_fn(gate) * up',
  '    current_hidden_states = linear(current_hidden_states, self.down_proj[expert_idx])', '    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]',
  '    final_hidden_states.index_add_(0, token_idx, current_hidden_states)'];
const TL = D.loop16, TSPAN = TL.at(-1)[1] + TL.at(-1)[2], BUSY = TL.filter((k) => k[0] !== 'Memcpy DtoH').reduce((a, k) => a + k[2], 0);
const S5 = { tl: [bt(4, 1) + 0.2, bt(4, 1) + 6.0], end: bt(4, 2) + 0.2 };
evRun(4, S5.tl[0], S5.tl[1], 0.09, 'tick'); ev(4, S5.end, 'land');
function s5(u) {
  const a = po(u, 0.3, 0.8), X = 96, Y = 290;
  rr(X, Y, 1728, 330, '#1d1d1f', a);
  txt('transformers · modeling_qwen3_moe.py（节选，linear 即 nn.functional.linear）', X + 24, Y + 38, { size: 20, color: '#9a968c', font: MONO, alpha: a });
  LOOP.forEach((l, i) => txt(l, X + 24, Y + 80 + i * 32, { size: 22, weight: i === 0 ? 700 : 500, color: i === 0 ? '#7fe0cf' : (i === 1 ? '#f2a19e' : '#e8e4da'), font: MONO, alpha: a }));
  // 时间线：profiler 记录的一次调用，三条泳道
  const TX = 220, TW = 1600, TY = 680, k = lin(u, ...S5.tl), show = TSPAN * k, b = po(u, S5.tl[0] - 0.4, S5.tl[0]);
  const lanes = [['矩阵乘法', C.teal], ['其他小 kernel', C.gray], ['拷回 CPU', C.red]];
  lanes.forEach(([n, col], i) => { txt(n, TX - 16, TY + i * 56 + 32, { size: 22, weight: 700, color: col === C.gray ? C.ink2 : col, align: 'right', alpha: b }); rr(TX, TY + i * 56, TW, 44, C.paper2, b); });
  for (const [kind, t0, dt] of TL) {
    if (t0 > show) break;
    const lane = kind === 'GEMM' ? 0 : kind === 'Memcpy DtoH' ? 2 : 1;
    rr(TX + TW * t0 / TSPAN, TY + lane * 56, Math.max(1.5, TW * dt / TSPAN), 44, lanes[lane][1], b);
  }
  txt('0', TX, TY + 200, { size: 20, color: C.gray, font: MONO, alpha: b }); txt(`${(TSPAN / 1000).toFixed(1)} ms（profiler 记录的一次调用）`, TX + TW, TY + 200, { size: 20, color: C.gray, align: 'right', font: MONO, alpha: b });
  const e = po(u, S5.end, S5.end + 0.5);
  txt(`821 个 kernel · 326 次拷回 CPU · GPU 忙碌 ${(BUSY / 1000).toFixed(1)} ms`, TX, TY + 246, { size: 28, weight: 800, color: C.red, alpha: e, font: ZH });
  txt('一层 17.6 ms（50 次中位数）', TX + TW, TY + 246, { size: 28, weight: 800, align: 'right', alpha: e });
}

// ================= 6 按 expert 排队 =================
// 展平：第 t 个 token 的第 k 个选择是第 t·8 + k 行；按 expert 稳定排序；每个 expert 的段补齐到 16 行（vLLM moe_align_block_size，BLOCK_SIZE_M = 16）
const PAIRS = D.ids.flatMap((ids, t) => ids.map((e, k) => ({ row: t * TOP + k, e })));
const SORTED = [...PAIRS].sort((p1, p2) => p1.e - p2.e || p1.row - p2.row);
const SPOS = new Map(D.align16.sorted_token_ids.filter((v) => v < NT * TOP).map((v, i) => [v, i]));   // 排序后的位置取 vLLM 的实际输出
// 每段的行取 vLLM 实际输出的 sorted_token_ids（段内顺序由原子操作决定）
const GROUPS = ACTIVE.map((e, i) => ({ e, rows: D.align16.sorted_token_ids.slice(i * 16, i * 16 + 16).filter((v) => v < NT * TOP) }));
const hue = (e) => `hsl(${(e * 137.5) % 360}, 55%, 52%)`;           // 颜色区分 expert 编号
const S6 = { flat: [0.5, 2.5], sort: [bt(5, 1) + 0.3, bt(5, 1) + 3.5], pad: [bt(5, 2) + 0.2, bt(5, 2) + 2.6], tab: bt(5, 3) + 0.2 };
evRun(5, S6.flat[0], S6.flat[1], 0.1, 'tick'); ev(5, S6.sort[0], 'whoosh'); ev(5, S6.sort[1], 'land'); evRun(5, S6.pad[0], S6.pad[1], 0.12, 'tick'); ev(5, S6.tab, 'pop'); ev(5, S6.tab + 0.6, 'pop');
function s6(u) {
  const X = 120, CW = 11, Y1 = 330, Y2 = 450, a = po(u, 0.3, 0.7), fl = lin(u, ...S6.flat), so = p(u, ...S6.sort);
  txt('按 token 顺序', X, Y1 - 14, { size: 22, weight: 700, color: C.ink2, alpha: a * (1 - so) });
  txt('按 expert 排序', X, Y2 - 14, { size: 22, weight: 700, color: C.ink2, alpha: po(u, S6.sort[0], S6.sort[0] + 0.4) });
  for (let t = 0; t < NT; t++) txt(`${t + 1}`, X + (t * TOP + 4) * CW, Y1 + 62, { size: 15, color: C.gray, align: 'center', font: MONO, alpha: a * (1 - so) });
  for (const q of PAIRS) {
    if (q.row >= 128 * fl) continue;
    const xx = mix(X + q.row * CW, X + SPOS.get(q.row) * CW, so), yy = mix(Y1, Y2, so);
    rr(xx, yy, CW - 2, 40, hue(q.e), 1);
  }
  txt('颜色 = expert 编号', X + 128 * CW + 24, Y2 + 28, { size: 22, color: C.gray, alpha: a });
  // 补齐：1040 行，每行 1.6 px；真实行着色，补齐行灰
  const pd = lin(u, ...S6.pad), PW = 1.35, Y3 = 580;
  if (pd > 0) {
    txt('每段补齐到 16 行', X, Y3 - 14, { size: 22, weight: 700, color: C.ink2, alpha: po(u, S6.pad[0], S6.pad[0] + 0.4) });
    GROUPS.forEach((gp, i) => { if (i / 65 > pd) return; const gx = X + i * 16 * PW; rr(gx, Y3, 16 * PW - 0.6, 40, C.lg, 1); rr(gx, Y3, gp.rows.length * PW, 40, hue(gp.e), 1); });
    txt('65 段 × 16 = 1040 行，其中有效 128 行，其余为补齐', X + 600, Y3 - 14, { size: 24, weight: 700, color: C.ink2, alpha: po(u, S6.pad[1] - 0.4, S6.pad[1]) });
    // 放大：前 4 段，每格写出 sorted_token_ids 的值（补齐位 = 128）
    const z = po(u, S6.pad[0] + 0.6, S6.pad[0] + 1.2), ZX = X, ZY = 690, ZC = 34;
    line(X, Y3 + 44, ZX, ZY - 4, C.gray, 1.5, z * 0.6); line(X + 3 * 16 * PW, Y3 + 44, ZX + 3 * 16 * ZC, ZY - 4, C.gray, 1.5, z * 0.6);
    GROUPS.slice(0, 3).forEach((gp, i) => {
      for (let j = 0; j < 16; j++) {
        const real = j < gp.rows.length, xx = ZX + (i * 16 + j) * ZC;
        rr(xx, ZY, ZC - 3, 46, real ? hue(gp.e) : C.lg, z);
        txt(real ? String(gp.rows[j]) : '128', xx + (ZC - 3) / 2, ZY + 30, { size: 15, weight: 700, color: real ? '#fff' : C.gray, align: 'center', font: MONO, alpha: z });
      }
      txt(`expert ${gp.e}`, ZX + i * 16 * ZC + 8 * ZC, ZY + 80, { size: 20, weight: 700, color: C.ink2, align: 'center', alpha: z });
    });
  }
  // 两张表
  const tb = po(u, S6.tab, S6.tab + 0.5), tb2 = po(u, S6.tab + 0.6, S6.tab + 1.1), TY = 870;
  const sids = D.align16.sorted_token_ids.slice(0, 10).join(', '), eids = D.align16.expert_ids.slice(0, 12).join(', ');
  txt(`sorted_token_ids = [${sids}, …]`, X, TY, { size: 22, weight: 700, color: C.ink, font: MONO, alpha: tb });
  txt('每行对应哪个 token 的第几个选择（值 ÷ 8 = token 序号，128 表示补齐）', X, TY + 28, { size: 20, color: C.gray, alpha: tb });
  txt(`expert_ids = [${eids}, …]`, X + 1000, TY, { size: 22, weight: 700, color: C.ink, font: MONO, alpha: tb2 });
  txt('每段所属的 expert', X + 1000, TY + 28, { size: 20, color: C.gray, alpha: tb2 });
}

// ================= 7 一次 kernel =================
// GEMM1：输出 1040 行 × 1536 列，tile 16 × 64 → 65 × 24 个 program。主角：分到 10 个 token 的那个 expert 的段
const HERO_G = GROUPS.findIndex((gp) => gp.rows.length === 10), HERO = GROUPS[HERO_G];
const KCODE = ['offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)', 'token_mask = offs_token < num_valid_tokens', 'off_experts = tl.load(expert_ids_ptr + pid_m)',
  'a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + …)', 'b_ptrs = b_ptr + off_experts * stride_be + (…)'];
const S7 = { one: 0.6, look: bt(6, 1) + 0.2, mask: bt(6, 2) + 0.2, all: bt(6, 3) + 0.2 };
ev(6, S7.one, 'pop'); ev(6, S7.look, 'click'); ev(6, S7.look + 1.6, 'click'); ev(6, S7.look + 3.0, 'land'); ev(6, S7.mask, 'pop'); ev(6, S7.all, 'whoosh'); ev(6, S7.all + 0.8, 'land');
function s7(u) {
  const GX = 520, GY = 300, CWg = 24, RH = 9, a = po(u, 0.3, 0.7), all = lin(u, S7.all, S7.all + 1.0);
  // program 网格
  for (let r = 0; r < 65; r++) for (let q = 0; q < 24; q++) {
    const hero = r === HERO_G && q === 5, on = all > 0 && (r + q * 0.3) / 72 < all;
    rr(GX + q * CWg, GY + r * RH, CWg - 2, RH - 2, hero ? C.purple : on ? '#b49cf2' : C.paper2, a);
  }
  txt('输出：65 段 × 24 块，每块由一个 program 计算', GX, GY - 16, { size: 22, weight: 700, color: C.ink2, alpha: a });
  const [hx, hy] = [GX + 5 * CWg, GY + HERO_G * RH];
  sr(hx - 3, hy - 3, CWg + 4, RH + 4, C.ink, 2, po(u, S7.one, S7.one + 0.4));
  // 放大的这一块：16 行，前 10 行真实，后 6 行补齐
  const z = po(u, S7.one + 0.2, S7.one + 0.8), ZX = 1180, ZY = 330, ZR = 28;
  line(hx + CWg, hy, ZX, ZY, C.ink, 1.5, z * 0.4); line(hx + CWg, hy + RH, ZX, ZY + 16 * ZR, C.ink, 1.5, z * 0.4);
  const m = po(u, S7.mask, S7.mask + 0.5);
  for (let j = 0; j < 16; j++) {
    const real = j < HERO.rows.length;
    rr(ZX, ZY + j * ZR, 200, ZR - 4, real ? C.purple : mixc('#d6c4fb', C.lg, m), z);
    txt(real ? `行 ${HERO.rows[j]} → token ${Math.floor(HERO.rows[j] / TOP) + 1}` : (m > 0.5 ? '屏蔽' : '补齐'), ZX + 12, ZY + j * ZR + 19, { size: 16, weight: 700, color: real ? '#fff' : C.gray, font: ZH, alpha: z });
  }
  txt(`expert ${HERO.e} · 第 6 块`, ZX, ZY - 16, { size: 22, weight: 700, color: C.purple, alpha: z });
  // 查表：expert_ids → 权重；sorted_token_ids → 输入行
  const l1 = po(u, S7.look, S7.look + 0.6), l2 = po(u, S7.look + 1.4, S7.look + 2.0), WX = 1440, WY = 360;
  txt('128 个 expert 的权重', WX, WY - 20, { size: 22, weight: 700, color: C.yelD, alpha: l1 });
  for (let e = 0; e < 128; e++) rr(WX + (e % 32) * 12, WY + Math.floor(e / 32) * 30, 9, 26, e === HERO.e ? C.yel : '#f3e6bd', l1);
  txt(`expert_ids[pid_m] = ${HERO.e}`, WX, WY + 150, { size: 22, weight: 700, color: C.ink, font: MONO, alpha: l1 });
  arrow(WX + 140, WY + 165, ZX + 210, ZY + 60, l1, C.yelD, l1 * 0.7, 3);
  // 输入 A：16 个 token 的行，按下标 gather，不搬动数据
  const AX = 96, AY = 330;
  txt('输入：16 个 token', AX, AY - 16, { size: 22, weight: 700, color: C.blue, alpha: a });
  const used = new Set(HERO.rows.map((r) => Math.floor(r / TOP)));
  for (let t = 0; t < NT; t++) {
    rr(AX, AY + t * 30, 300, 24, used.has(t) && l2 > 0 ? C.blue : '#b9c6f0', a);
    if (used.has(t)) line(AX + 300, AY + t * 30 + 12, ZX, ZY + HERO.rows.findIndex((r) => Math.floor(r / TOP) === t) * ZR + 12, C.blue, 1.5, l2 * 0.5);
  }
  txt('按 sorted_token_ids ÷ 8 读取行', AX, AY + 16 * 30 + 30, { size: 22, weight: 700, color: C.blue, alpha: l2 });
  // 源码原文
  const c = po(u, S7.look + 2.6, S7.look + 3.2);
  rr(96, 860, 1728, 0, C.ink, 0);
  KCODE.forEach((l, i) => txt(l, 1080, 820 + i * 30, { size: 17, weight: 600, color: C.ink2, font: MONO, alpha: c }));
  txt(`其余 64 段同时计算：所有 expert 在一次启动中完成`, GX, GY + 65 * RH + 40, { size: 26, weight: 800, color: C.purple, alpha: po(u, S7.all + 0.6, S7.all + 1.0) });
}

// ================= 8 整层 =================
const F16 = D.fused16, FSPAN = F16.at(-1)[1] + F16.at(-1)[2];
const KCOL = { copy: C.gray, align: C.blue, count_sort: C.blue, fused_moe_kernel: C.teal, silu_and_mul: C.purple, moe_sum: C.purple };
const S8 = { tl: [0.5, 3.5], cmp: bt(7, 1) + 0.2 };
F16.forEach(([, t0], i) => ev(7, S8.tl[0] + (S8.tl[1] - S8.tl[0]) * t0 / FSPAN, i === 3 || i === 5 ? 'land' : 'tick')); ev(7, S8.cmp, 'whoosh'); ev(7, S8.cmp + 1.6, 'pop');
function s8(u) {
  const X = 140, TW = 1640, Y = 380, a = po(u, 0.3, 0.7), show = FSPAN * lin(u, ...S8.tl), c = p(u, S8.cmp, S8.cmp + 0.8);
  // 上：7 个 kernel 的真实时间线（µs），相邻的标注上下错开
  rr(X, Y, TW, 70, C.paper2, a * (1 - c));
  F16.forEach(([n, t0, dt], i) => {
    if (t0 > show) return;
    const x = X + TW * t0 / FSPAN, w = Math.max(3, TW * dt / FSPAN), al = a * (1 - c);
    rr(x, Y, w, 70, KCOL[n], al);
    const up = i % 2 === 0, big = dt > 50;
    txt(n === 'fused_moe_kernel' ? `矩阵乘法 ${i === 3 ? 1 : 2}` : n, big ? x + w / 2 : x, big ? Y + 44 : (up ? Y - 40 : Y + 110), { size: big ? 26 : 20, weight: 700, color: big ? '#fff' : C.ink2, align: big ? 'center' : 'left', font: big ? ZH : MONO, alpha: al });
    txt(`${dt.toFixed(1)} µs`, big ? x + w / 2 : x, big ? Y + 106 : (up ? Y - 14 : Y + 136), { size: 18, color: C.gray, align: big ? 'center' : 'left', font: MONO, alpha: al });
  });
  txt('0', X, Y + 180, { size: 20, color: C.gray, font: MONO, alpha: a * (1 - c) }); txt(`${(FSPAN / 1000).toFixed(2)} ms（profiler 记录，含 kernel 之间的间隔）`, X + TW, Y + 180, { size: 20, color: C.gray, align: 'right', font: MONO, alpha: a * (1 - c) });
  txt('两次矩阵乘法：212 µs，占 GPU 时间的 95%', X, Y + 250, { size: 30, weight: 800, color: C.tealD, alpha: po(u, S8.tl[1], S8.tl[1] + 0.5) * (1 - c) });
  // 下：同一时间轴上的两种写法（中位数）
  if (c > 0) {
    const MS = TW / 17.58, T = D.timing[16];
    txt('逐个 expert 循环', X, 430, { size: 32, weight: 800, alpha: c }); rr(X, 450, T.loop * MS * lin(u, S8.cmp + 0.4, S8.cmp + 1.6), 70, C.red, c);
    txt(`${T.loop.toFixed(1)} ms`, X + T.loop * MS - 16, 498, { size: 30, weight: 800, color: '#fff', align: 'right', alpha: po(u, S8.cmp + 1.4, S8.cmp + 1.8) });
    txt('一次 fused kernel', X, 640, { size: 32, weight: 800, color: C.tealD, alpha: c }); rr(X, 660, T.fused * MS, 70, C.teal, c);
    txt(`${T.fused.toFixed(2)} ms`, X + T.fused * MS + 16, 708, { size: 30, weight: 800, color: C.tealD, alpha: c });
    txt(`约 ${Math.round(T.loop / T.fused)} 倍`, X + TW, 708, { size: 48, weight: 800, color: C.tealD, align: 'right', alpha: po(u, S8.cmp + 1.6, S8.cmp + 2.0) });
    txt('16 个 token，H100，单层，50 次中位数', X, 800, { size: 22, color: C.gray, alpha: c });
  }
}

// ================= 9 瓶颈 =================
// 16 个 token：两次矩阵乘法共 212 µs，读 65 个 expert 的权重 65 × 9.44 = 613 MB → 2.9 TB/s；H100 显存带宽规格 3.35 TB/s。
// 计算量 2 × 128 行 × (2048 × 1536 + 768 × 2048) = 1.21 GFLOP → 5.7 TFLOPS，BF16 dense 规格 989 TFLOPS 的 0.6%
const S9 = { flow: [0.5, 4.5], bw: bt(8, 1) + 0.2, use: bt(8, 2) + 0.2 };
evRun(8, S9.flow[0], S9.flow[1], 0.0615, 'tick'); ev(8, S9.bw, 'land'); ev(8, S9.bw + 0.8, 'land'); ev(8, S9.use, 'pop');
function s9(u) {
  const a = po(u, 0.3, 0.7), f = lin(u, ...S9.flow), n = Math.floor(65 * f);
  // 65 个 expert 的权重块从左边（显存）流向右边（芯片）
  txt('显存 HBM', 150, 330, { size: 28, weight: 800, color: C.yelD, alpha: a }); txt('芯片', 1320, 330, { size: 28, weight: 800, color: C.tealD, alpha: a });
  rr(96, 350, 420, 300, '#f6e3a0', a); rr(1240, 350, 300, 300, '#cfeee8', a);
  for (let i = 0; i < 65; i++) {
    const k = clamp(f * 65 - i, 0, 1), x = mix(110 + (i % 13) * 31, 1255 + (i % 13) * 21, ease(k)), y = 368 + Math.floor(i / 13) * 56;
    rr(x, y, k > 0 && k < 1 ? 28 : (k >= 1 ? 18 : 26), 46, k > 0 && k < 1 ? C.yel : (k >= 1 ? C.yelD : '#e9c95a'), a);
  }
  txt(`已读取 ${fmt(Math.min(613, n * 9.44), 0)} MB / 613 MB`, 96, 700, { size: 28, weight: 700, alpha: a, font: ZH });
  // 带宽对比
  const b = po(u, S9.bw, S9.bw + 0.5), b2 = po(u, S9.bw + 0.8, S9.bw + 1.3), BX = 1600, BH = 300 / 3.35;
  txt('带宽', BX, 330, { size: 26, weight: 700, alpha: b });
  rr(BX, 650 - 3.35 * BH, 80, 3.35 * BH, C.paper2, b2); sr(BX, 650 - 3.35 * BH, 80, 3.35 * BH, C.ink, 2, b2); txt('规格 3.35', BX + 90, 650 - 3.35 * BH + 20, { size: 20, color: C.ink2, alpha: b2 });
  rr(BX, 650 - 2.9 * BH * b, 80, 2.9 * BH * b, C.yelD, b); txt('实际 2.9 TB/s', BX + 90, 650 - 2.9 * BH + 40, { size: 22, weight: 800, color: C.yelD, alpha: b });
  txt('613 MB ÷ 212 µs', BX - 20, 700, { size: 22, color: C.gray, font: ZH, alpha: b });
  // Tensor Core 利用率与每个权重的使用次数
  const c = po(u, S9.use, S9.use + 0.5);
  txt('每个权重平均与约 2 个 token 相乘（128 行 ÷ 65 个 expert）', 96, 800, { size: 30, weight: 800, color: C.ink, alpha: c });
  txt('Tensor Core：5.7 TFLOPS，为 BF16 规格 989 的 0.6%', 96, 860, { size: 30, weight: 800, color: C.red, alpha: po(u, S9.use + 1.0, S9.use + 1.5) });
}

// ================= 10 batch 变大 =================
const NS = [1, 4, 16, 64, 256, 1024, 4096];
const S10 = { act: [0.5, 3.5], time: [bt(9, 1) + 0.2, bt(9, 1) + 3.2], tile: bt(9, 2) + 0.2 };
evRun(10 - 1, S10.act[0], S10.act[1], 0.25, 'tick'); evRun(9, S10.time[0], S10.time[1], 0.4, 'tick'); ev(9, S10.tile, 'pop');
function chart(x, y, w, h, title, ymax, yticks, pts, k, col, alpha, unit) {
  txt(title, x, y - 24, { size: 26, weight: 800, alpha });
  line(x, y + h, x + w, y + h, C.ink, 2, alpha); line(x, y, x, y + h, C.ink, 2, alpha);
  const X = (n) => x + w * Math.log2(n) / 12, Y = (v) => y + h - h * v / ymax;
  [1, 16, 256, 4096].forEach((n) => txt(String(n), X(n), y + h + 30, { size: 20, color: C.gray, align: 'center', font: MONO, alpha }));
  yticks.forEach((v) => { line(x, Y(v), x + w, Y(v), C.lg, 1, alpha); txt(String(v), x - 10, Y(v) + 7, { size: 18, color: C.gray, align: 'right', font: MONO, alpha }); });
  txt('token 数', x + w, y + h + 62, { size: 20, color: C.gray, align: 'right', alpha });
  const m = Math.floor(pts.length * k + 0.999);
  g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = 4; g.beginPath(); pts.slice(0, m).forEach(([n, v], i) => (i ? g.lineTo(X(n), Y(v)) : g.moveTo(X(n), Y(v)))); g.stroke(); g.globalAlpha = 1;
  pts.slice(0, m).forEach(([n, v]) => { g.globalAlpha = alpha; g.fillStyle = col; g.beginPath(); g.arc(X(n), Y(v), 7, 0, 7); g.fill(); g.globalAlpha = 1; txt(unit(v), X(n), Y(v) - 16, { size: 18, weight: 700, color: col, align: 'center', font: MONO, alpha }); });
}
function s10(u) {
  const a = po(u, 0.3, 0.7);
  const act = [...Object.entries(D.active.decode).map(([n, v]) => [+n, v]), ...[[256, D.active.prefill[256]], [1024, D.active.prefill[1024]], [4096, D.active.prefill[4096]]]];
  chart(160, 330, 700, 360, '激活的 expert 数', 128, [32, 64, 96, 128], act, lin(u, ...S10.act), C.tealD, a, (v) => v.toFixed(0));
  const tm = NS.map((n) => [n, D.timing[n].fused]), b = po(u, S10.time[0] - 0.3, S10.time[0]);
  chart(1060, 330, 700, 360, '单层耗时（ms）', 1.2, [0.4, 0.8, 1.2], tm, lin(u, ...S10.time), C.blue, b, (v) => v.toFixed(2));
  // 每个 expert 平均的行数与 tile 行数
  const t = po(u, S10.tile, S10.tile + 0.5), Y = 820;
  txt('token 数', 96, Y, { size: 22, weight: 700, color: C.gray, alpha: t }); txt('每个 expert 的行数', 96, Y + 46, { size: 22, weight: 700, alpha: t }); txt('tile 行数', 96, Y + 92, { size: 22, weight: 700, color: C.purple, alpha: t });
  NS.forEach((n, i) => {
    const x = 380 + i * 205, rows = n * TOP / D.timing[n].active, bm = D.config[n][0], al = t * clamp((u - S10.tile) * 4 - i * 0.6);
    txt(String(n), x, Y, { size: 22, color: C.gray, align: 'center', font: MONO, alpha: al });
    txt(rows < 10 ? rows.toFixed(1) : rows.toFixed(0), x, Y + 46, { size: 26, weight: 800, align: 'center', font: MONO, alpha: al });
    txt(String(bm), x, Y + 92, { size: 26, weight: 800, color: C.purple, align: 'center', font: MONO, alpha: al });
  });
}

// ================= 11 各家做法 =================
const WAYS = [['vLLM · Triton', '排序后只补齐下标', '按下标读取行，不拷贝 token'], ['transformers', 'sort + 精确偏移', 'CUTLASS grouped GEMM，不补齐'],
  ['DeepGEMM', 'contiguous：每段对齐到 128', 'masked：decode 按掩码只计算有效行'], ['MegaBlocks', 'block-sparse，128 × 128 块', '不丢弃 token（训练）']];
const S11 = { cards: 0.5, b200: bt(10, 1) + 0.2, back: bt(10, 2) + 0.2 };
WAYS.forEach((_, i) => ev(10, S11.cards + i * 0.5, 'pop')); ev(10, S11.b200, 'land'); ev(10, S11.back, 'whoosh'); evRun(10, S11.back + 0.6, S11.back + 2.6, 0.2, 'tick');
function s11(u) {
  const back = p(u, S11.back, S11.back + 0.8);
  WAYS.forEach(([n, l1, l2], i) => {
    const a = po(u, S11.cards + i * 0.5, S11.cards + i * 0.5 + 0.4) * (1 - back), x = 96 + i * 440;
    rr(x, 300, 410, 250, '#fff', a); rr(x, 300, 410, 8, C.teal, a);
    txt(n, x + 24, 360, { size: 32, weight: 800, alpha: a }); txt(l1, x + 24, 420, { size: 24, weight: 600, color: C.ink2, alpha: a }); txt(l2, x + 24, 460, { size: 22, color: C.gray, alpha: a });
    // 每张卡片下面画一条示意：彩色为真实行，灰为补齐
    for (let j = 0; j < 6; j++) { const real = [5, 2, 7, 3, 4, 1][j], pad = i === 1 ? real : (i === 2 ? 8 : 8); rr(x + 24 + j * 62, 500, pad * 7, 20, C.lg, a); rr(x + 24 + j * 62, 500, real * 7, 20, hue(j * 23), a); }
  });
  const b = po(u, S11.b200, S11.b200 + 0.5) * (1 - back);
  txt('B200 上的默认实现：FlashInfer TRT-LLM（路由融合在 kernel 内）', 96, 640, { size: 30, weight: 800, color: C.tealD, alpha: b });
  txt('H100 上 vLLM 与 SGLang 默认使用上述 Triton kernel', 96, 684, { size: 24, color: C.ink2, alpha: b });
  placeholder(96, 720, 1100, 150, '实测待补：B200 上的单层耗时', ['同一层、16 与 4096 个 token：FlashInfer TRT-LLM 与 Triton fused MoE 的 kernel 列表与耗时', '需要：Modal B200 × 1，约 $1'], b);
  // 回到开头的网格：65 个 expert 亮起，排成 65 段一次算完
  if (back > 0) {
    const { x, y, c } = G1, n = Math.floor(65 * lin(u, S11.back + 0.6, S11.back + 2.6)), lit = new Set(ACTIVE.slice(0, n));
    grid(x, y - 30, c, (e) => (lit.has(e) ? [tealK(0.3 + CNT[e] / 8), 1] : null), back);
    txt('16 个 token → 65 个 expert → 1 次启动', x + 8 * c - 3, y - 30 + 8 * c + 40, { size: 30, weight: 800, color: C.tealD, align: 'center', alpha: po(u, S11.back + 2.4, S11.back + 2.9) });
  }
}

const DRAW = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11];
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
