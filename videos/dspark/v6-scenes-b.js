// DSpark v6 后半：表有多大 → 效果 → 验证的代价 → 值多少 → 概率从哪来 → 验几个 → 回到开头

// ---------- 7 这张表有多大 ----------
// 按真实比例：151,936 × 151,936 的方块，旁边两条宽 256 的矩阵（同一比例下不到 1 像素）
const VOC = 129280, RANK = 256;   // DeepSeek-V4-Flash 的 checkpoint 配置（vocab_size、dspark_markov_rank）；V4-Pro 的秩为 512
function sTable() {
  frame('transition bias 的规模', K.purple, '词表 × 词表{r:无法存储}：分解为两个低秩矩阵', 'DeepSeek-V4-Flash：词表 129,280 · 秩 256（checkpoint config）· 耗时：论文，Qwen3-4B');
  const x = 96, y = 280, Z = 520, g = S(.6, 1.0), thin = Math.max(2, Z * RANK / VOC);
  R(x, y, Z * g, Z * g, '#e9e0fd'); if (g > .05) box(x, y, Z * g, Z * g, 3, K.purple);
  IN(1.4, () => { TX('129,280 × 129,280', x + Z / 2, y + Z / 2 - 10, font(700, 40, HN), K.purple, 'center'); TX('≈ 167 亿个参数', x + Z / 2, y + Z / 2 + 48, font(700, 36), K.purple, 'center'); }, .5, 0);
  // 两条细矩阵
  const w1 = S(3.4, .8), w2 = S(4.0, .8);
  R(x + Z + 40, y, thin, Z * w1, K.purple); R(x + Z + 80, y, Z * w2, thin, K.purple);
  IN(3.6, () => TX('W₁：129,280 × 256', x + Z + 58, y + Z - 10, font(700, 26, HN), K.purple), .5, 0);
  IN(4.2, () => TX('W₂：256 × 129,280', x + Z + 80, y + 36, font(700, 26, HN), K.purple), .5, 0);
  IN(4.8, () => TX('按同一比例，宽度不足 1 像素', x + Z + 80, y + 76, font(600, 26), K.ink2), .5, 0);
  IN(5.6, () => RICH('两个矩阵共 6,619 万个参数，为完整矩阵的 {p:0.40%}', x + Z + 80, y + 140, font(700, 32), K.ink), .5, 0);
  // 查一次：当然 → W₁ 的一行（256 个数）→ × W₂ → 对每个词的加分
  const py = 540, ax = x + Z + 80;
  IN(6.8, () => chip(ax, py, 120, 70, 'word', '当然', 30), .4, 0);
  arrowP(ax + 128, py + 35, ax + 200, py + 35, S(7.2, .4), K.purple, 4);
  IN(7.4, () => { for (let i = 0; i < 16; i++) R(ax + 210 + i * 17, py + 10, 14, 50, i % 3 ? '#a988f8' : K.purple); TX('W₁ 的一行：256 维', ax + 210, py + 100, font(600, 26), K.purple); }, .4, 0);
  arrowP(ax + 490, py + 35, ax + 560, py + 35, S(8.0, .4), K.purple, 4);
  IN(8.2, () => { R(ax + 570, py, 100, 70, K.purple); TX('× W₂', ax + 620, py + 46, font(700, 30, HN), '#fff', 'center'); }, .4, 0);
  arrowP(ax + 678, py + 35, ax + 740, py + 35, S(8.6, .4), K.purple, 4);
  IN(8.8, () => { const bx = ax + 750; for (let i = 0; i < 28; i++) { const hv = i === 6 ? 56 : i === 17 ? -40 : Math.sin(i * 2.7) * 12; if (hv >= 0) R(bx + i * 12, py + 35 - hv, 8, hv + 1, i === 6 ? K.purple : K.lg); else R(bx + i * 12, py + 35, 8, -hv, i === 17 ? K.red : K.lg); }
    TX('可以 +1.1', bx + 76, py - 32, font(700, 26), K.purple, 'center'); TX('问题 −1.1', bx + 208, py + 106, font(700, 26), K.red, 'center'); TX('词表上的 logit bias', bx, py + 146, font(600, 26), K.purple); }, .4, 0);
  // 底部：与第 3、4 幕同一条时间条；Markov head 的串行部分只在 draft 段末尾多出一小段
  const tb = 330, by = 870;
  IN(8.8, () => { TX('一轮耗时', 96, by + 42, font(700, 30), K.ink); for (let l = 0; l < 3; l++) R(tb + 1, by + l * 20, 208, 18, K.purple); R(tb + 210, by, 8 * S(9.2, .5), 60, K.red);
    R(tb + 220, by, 618, 60, K.blue); TX('验证：target 一次 forward', tb + 240, by + 40, font(700, 28), '#fff');
    RICH('draft：3 层 × 1 次 {r:+ Markov head，约 +1%}', tb, by - 14, font(700, 28), K.purple); }, .5, 0);
}

// ---------- 8 效果 ----------
// 各位置的条件接受率（论文 Fig. 2 聊天子图，Qwen3-4B，按图读数；0.53、0.72、0.63 为正文数字），每轮收下数（Qwen3-8B，γ = 7）
const CURVES = [['自回归（EAGLE-3）', [.53, .59, .64, .68, .71, .73, .74], K.gray, 1.0], ['并行（DFlash）', [.72, .65, .63, .66, .68, .69, .69], K.ink, 3.0], ['并行 + Markov head（DSpark）', [.735, .74, .74, .745, .76, .765, .76], K.purple, 5.4]];
function sCurve() {
  frame('效果', K.teal, '第 1 位接受率高，{t:后续位置不下降}', '论文 Fig. 2 聊天，Qwen3-4B（按图读数）· 接受长度：论文表格，Qwen3-8B');
  const x0 = 210, x1 = 1000, yt = 330, yb = 800, px = (k) => x0 + (k - 1) / 6 * (x1 - x0), py = (v) => yb - (v - .5) / .3 * (yb - yt);
  IN(.4, () => { line(x0 - 20, yb, x1 + 20, yb, K.ink, 3); [.5, .6, .7, .8].forEach((v) => { TX(v.toFixed(1), x0 - 34, py(v) + 9, font(600, 26, HN), K.ink2, 'right'); if (v > .5) line(x0 - 20, py(v), x1 + 20, py(v), K.paper2, 2); });
    for (let k = 1; k <= 7; k++) TX(String(k), px(k), yb + 38, font(600, 26, HN), K.ink2, 'center'); TX('位置', (x0 + x1) / 2, yb + 76, font(600, 26), K.ink2, 'center');
    TX('条件接受率（Qwen3-4B，聊天）', x0 - 20, yt - 22, font(600, 26), K.ink2); }, .5, 0);
  CURVES.forEach(([name, vs, c, t0]) => { const p = clamp((T - t0) / 1.6) * 6; if (T < t0) return;
    const pts = []; for (let k = 0; k <= Math.floor(p); k++) pts.push([px(k + 1), py(vs[k])]); if (p < 6) { const k = Math.floor(p), f = p - k; pts.push([px(k + 1 + f), py(mix(vs[k], vs[k + 1], f))]); }
    poly(pts, c, 6); pts.slice(0, Math.floor(p) + 1).forEach(([xx, yy]) => dot(xx, yy, 8, c));
    IN(t0 + 1.6, () => TX(name, x1 + 30, py(vs[6]) + (c === K.purple ? -14 : c === K.gray ? 2 : 22), font(700, 26), c), .4, 0);
    IN(t0 + .2, () => { const [lx, ly, al] = c === K.gray ? [px(1) + 16, py(vs[0]) + 34, 'left'] : c === K.purple ? [px(1) + 16, py(vs[0]) - 18, 'left'] : [px(1) - 6, py(vs[0]) + 74, 'left']; TX(vs[0].toFixed(2).replace('0.73', '0.74'), lx, ly, font(700, 26, HN), c, al); }, .4, 0); });
  // 每轮收下的 token：三根柱子对着同一条基线
  const bx = 1440, base = 800, U = 120, bars = [['EAGLE-3', 2.66, K.gray], ['DFlash', 3.11, K.ink], ['DSpark', 3.72, K.purple]];
  IN(7.6, () => { TX('每轮接受的 token 数', bx - 20, yt - 58, font(600, 26), K.ink2); TX('Qwen3-8B · MT-Bench · γ = 7', bx - 20, yt - 22, font(600, 26), K.ink2); line(bx - 20, base, 1824, base, K.ink, 3);
    [1, 2, 3].forEach((v) => { line(bx - 20, base - v * U, 1824, base - v * U, K.paper2, 2); TX(String(v), bx - 30, base - v * U + 8, font(600, 26, HN), K.ink2, 'right'); }); }, .5, 0);
  bars.forEach(([n, v, c], i) => { const g = S(8.0 + i * .4, .8), x = bx + i * 130; R(x, base - v * U * g, 96, v * U * g, c);
    IN(8.4 + i * .4, () => { TX(v.toFixed(2), x + 48, base - v * U - 14, font(700, 30, HN), c, 'center'); TX(n, x + 48, base + 36, font(700, 26, HN), K.ink2, 'center'); }, .4, 0); });
}

// ---------- 9 验证的代价随负载变 ----------
// 一步耗时随这一步的 token 数：先平后陡（示意）。空闲点与繁忙点各多放同样多的 token
const STEP = (b) => 20 + Math.max(0, b - 64) * .22;
function sCost() {
  frame('是否验证全部 draft token', K.blue, '低负载时多验证几乎无代价，高负载时{r:每个都占用算力}', '曲线为示意');
  const x0 = 200, x1 = 1060, yt = 330, yb = 800, bx = (b) => x0 + b / 256 * (x1 - x0), ty = (t) => yb - (t - 10) / 50 * (yb - yt);
  IN(.4, () => { line(x0, yb, x1 + 20, yb, K.ink, 3); line(x0, yb, x0, yt - 20, K.ink, 3);
    TX('本步送入 target 的 token 数（所有请求合计）', x0, yb + 46, font(600, 26), K.ink2); TX('单步耗时', x0 + 14, yt - 4, font(600, 26), K.ink2); }, .5, 0);
  const cp = S(1.0, 1.4), pts = []; for (let b = 0; b <= 256 * cp; b += 4) pts.push([bx(b), ty(STEP(b))]); if (pts.length > 1) poly(pts, K.blue, 6);
  const mark = (b0, b1, t0, col, txt, dy) => { IN(t0, () => { dot(bx(b0), ty(STEP(b0)), 10, col); }, .3, 0);
    const p = S(t0 + .5, .7), b = mix(b0, b1, p); if (p > 0) { line(bx(b0), ty(STEP(b0)), bx(b), ty(STEP(b0)), col, 4, [8, 6]); line(bx(b), ty(STEP(b0)), bx(b), ty(STEP(b)), col, 6); dot(bx(b), ty(STEP(b)), 10, col); }
    IN(t0 + 1.3, () => TX(txt, dy > 0 && b0 < 100 ? bx(b0) - 10 : bx(b0) - 24, ty(STEP(b0)) + (b0 < 100 ? dy : -40), font(700, 28), col, b0 < 100 ? 'left' : 'right'), .4, 0); };
  mark(16, 40, 2.8, K.teal, '低负载：增加一批 token，耗时几乎不变', 56);
  IN(4.2, () => TX('耗时主要在读取权重', x0 + 14, ty(STEP(0)) - 26, font(600, 26), K.teal), .5, 0);
  mark(176, 200, 5.0, K.red, '高负载：增加同样数量，耗时随之上升', 56);
  // 右：固定猜 5 个时，每个请求都验满
  const rx = 1180, rows = [['A', 'oooox'], ['B', 'oox--'], ['C', 'x----']], cs = 84, cg = 96;
  IN(8.0, () => { TX('每个请求固定验证 5 个', rx, 364, font(700, 30), K.ink);
    rows.forEach(([n, s], i) => { const y = 396 + i * 108; TX(n, rx, y + 56, font(700, 36, HN), K.ink); chip(rx + 50, y, cs, cs, 'ctx', null);
      s.split('').forEach((ch, j) => { const st = T < 9.0 ? 'guess' : ch === 'o' ? 'ok' : ch === 'x' ? 'bad' : 'void'; POP(8.2 + i * .1 + j * .05, rx + 146 + j * cg, y, cs, cs, () => chip(rx + 146 + j * cg, y, cs, cs, st, null, 34), .3); }); }); }, .5, 0);
  IN(8.6, () => { R(rx, 728, 26, 26, K.ink); TX('上一个已确定的 token', rx + 38, 750, font(600, 26), K.ink2); }, .5, 0);
  IN(9.6, () => TX('红色与斜纹：无效计算', rx, 800, font(700, 30), K.red), .5, 0);
}

// ---------- 10 一个 token 值多少 ----------
// 请求 B 的 4 个 draft token，各自猜对的概率（前面都对时）0.80 0.70 0.65 0.60；被收下的概率一路乘下去
const CB = [.80, .70, .65, .60], SB = (() => { let p = 1; return CB.map((c) => (p *= c)); })(), BW = ['我', '来', '查', '一下'];
function sValue() {
  frame('单个 token 的期望收益', K.purple, '前面各位均被接受，它才有效：{p:条件接受率逐位相乘}', '数值为示意');
  const cx = (i) => 380 + i * 300, base = 840, U = 400;
  IN(.4, () => { TX('请求 B', 96, 344, font(700, 34), K.ink); TX('4 个 draft token', 96, 386, font(600, 26), K.ink2); }, .4, 0);
  BW.forEach((w, i) => IN(.6 + i * .12, () => { chip(cx(i), 300, 200, 76, 'word', w, 32); RICH(`条件接受率：{p:${CB[i].toFixed(2)}}`, cx(i) + 100, 420, font(700, 26), K.ink2, 'center'); }, .4, 0));
  IN(1.6, () => { TX('存活概率', 96, base - U * .8 + 10, font(700, 28), K.ink); line(cx(0) - 30, base, cx(3) + 230, base, K.ink, 3); }, .5, 0);
  // 第 k 根柱子先等于上一根，再按 c_k 缩短
  SB.forEach((s, i) => { const t0 = 2.4 + i * 1.6, prev = i ? SB[i - 1] : 1, g = P(t0, .5), sh = S(t0 + .5, .8), v = mix(prev, s, sh), x = cx(i) + 20;
    if (g <= 0) return; R(x, base - U * v * g, 160, U * v * g, K.purple);
    if (sh > 0 && i) HR(x, base - U * prev, 160, U * (prev - v), K.lg, K.paper);
    if (sh > .5) TX(s.toFixed(2), x + 80, base - U * s - 16, font(700, 34, HN), K.purple, 'center');
    if (i) IN(t0 + .3, () => { line(cx(i - 1) + 180, base - U * prev, x, base - U * prev, K.gray, 2, [6, 6]); TX(`× ${CB[i].toFixed(2)}`, x + 80, base - U * prev - 54, font(700, 26, HN), K.ink2, 'center'); }, .3, 0); });
  IN(9.0, () => RICH('4 个合计：期望接受 {p:1.94} 个', 1824, 490, font(700, 30), K.ink, 'right'), .5, 0);
  IN(9.2, () => TX(SB.map((v) => v.toFixed(2)).join(' + ') + ' = 1.94', 1824, 530, font(600, 26, HN), K.ink2, 'right'), .5, 0);   // 合计的算法：四个存活概率相加
}

// ---------- 11 概率从哪来 ----------
// 被接受的概率 = drafter 与 target 两个分布的重叠（Σ min）。target 的分布要验了才有，所以让 drafter 自己估
const CAND = ['可以', '问题', '呢', '的'], QD = [.90, .10, 0, 0], PT = [.80, .05, .10, .05];
function sConf() {
  frame('验证前如何得到条件接受率', K.purple, '由 drafter 自行估计：额外输出{p:一个 0 到 1 之间的值}', '分布为示意 · 训练方法：论文');
  const x = 220, U = 700, hide = S(6.4, .6);
  IN(.4, () => { TX('第 2 位（前一个 token 为“当然”）：drafter 与 target 的分布', 96, 316, font(700, 28), K.ink);
    R(1100, 296, 26, 26, K.purple); TX('drafter 的分布', 1136, 318, font(600, 26), K.ink2); R(1330, 296, 26, 26, K.blue); TX('target 的分布', 1366, 318, font(600, 26), K.ink2); }, .5, 0);
  CAND.forEach((w, i) => { const y = 356 + i * 92;
    IN(.8 + i * .1, () => { TX(w, 96, y + 52, font(700, 32), K.ink); R(x, y, U * QD[i], 34, K.purple); TX(QD[i] ? QD[i].toFixed(2) : '0', x + U * QD[i] + 10, y + 28, font(700, 26, HN), K.purple); }, .4, 0);
    IN(1.6 + i * .1, () => { R(x, y + 40, U * PT[i], 34, hide > .5 ? K.lg : K.blue); TX(hide > .5 ? '?' : PT[i].toFixed(2), x + U * PT[i] + 10, y + 68, font(700, 26, HN), hide > .5 ? K.red : K.blue); }, .4, 0);
    if (Math.min(QD[i], PT[i]) > 0) IN(3.4 + i * .15, () => box(x - 3, y - 3, U * Math.min(QD[i], PT[i]) + 6, 80, 5, K.teal), .3, 0); });
  IN(4.6, () => RICH('条件接受率 = 两个分布的{t:重叠部分}：0.80 + 0.05 + 0 + 0 = {t:0.85}', 96, 760, font(700, 30), K.ink), .5, 0);
  IN(6.4, () => TX('但 target 的分布只有在验证时才能得到', 96, 812, font(700, 30), K.red), .5, 0);
  // 右：drafter 多一个小层，自己估
  const rx = 1180;
  IN(7.6, () => { TX('drafter 在该位置的 hidden state', rx, 430, font(600, 26), K.ink2); for (let i = 0; i < 12; i++) R(rx + i * 22, 444, 18, 44, i % 4 ? '#a988f8' : K.purple);
    TX('前一个 token', rx, 560, font(600, 26), K.ink2); chip(rx, 574, 120, 60, 'word', '当然', 28); }, .5, 0);
  arrowP(rx + 270, 466, rx + 360, 520, S(8.2, .4), K.purple, 4); arrowP(rx + 130, 604, rx + 360, 556, S(8.3, .4), K.purple, 4);
  IN(8.6, () => { R(rx + 366, 494, 170, 90, K.purple); TX('线性层', rx + 451, 532, font(700, 26), '#fff', 'center'); TX('+ sigmoid', rx + 451, 568, font(700, 26, HN), '#fff', 'center'); }, .4, 0);
  arrowP(rx + 540, 539, rx + 580, 539, S(9.0, .3), K.purple, 4);
  IN(9.2, () => TX('0.88', rx + 590, 552, font(700, 40, HN), K.purple), .4, 0);
  IN(10.0, () => RICH('训练时以真实重叠（{t:0.85}）为标签', rx, 680, font(700, 28), K.ink), .5, 0);
  IN(11.2, () => RICH('这一层称为 {p:confidence head}', rx, 740, font(700, 34), K.ink), .5, 0);
}

// ---------- 12 验几个 ----------
// 三个请求各 4 个 token（被收下的概率）。所有 token 按概率排成一队，依次加入，算“期望收下 ÷ 一步耗时”，到峰值停下。
// 耗时（示意）：空闲 20 + 0.4B，繁忙 5 + 2B（B 为这一步的 token 数，含每个请求自己的 1 个）。贪心结果：空闲 10 个，繁忙 5 个
const REQ = [['A', '代码', [.95, .88, .81, .73]], ['B', '聊天', SB.map((v) => +v.toFixed(2))], ['C', '难预测', [.50, .25, .13, .06]]];
const QUEUE = REQ.flatMap(([n, , vs], r) => vs.map((v, j) => ({ r, j, v }))).sort((a, b) => b.v - a.v);
function thr(cost) { let tau = 3; const out = [tau / cost(3)]; QUEUE.forEach((q, k) => { tau += q.v; out.push(tau / cost(4 + k)); }); const mx = Math.max(...out); return { rel: out.map((v) => v / mx), peak: out.indexOf(mx) }; }
const IDLE = thr((b) => 20 + .4 * b), BUSY = thr((b) => 5 + 2 * b);
const Q0 = 3.0, I0 = 6.0, DT = .42, B0 = 13.4;
function sSched() {
  frame('本步验证多少个', K.blue, '所有请求的 token 统一排序，{b:验证至吞吐不再提升}', '数值与耗时曲线为示意');
  const busy = T >= B0, M = busy ? BUSY : IDLE, t0 = busy ? B0 + .6 : I0, k = clampR(Math.floor((T - t0) / DT) + 1, 0, M.peak), stopped = T > t0 + M.peak * DT + .3;
  const taken = (q) => QUEUE.indexOf(q) < k, rowX = (j) => 360 + j * 132, rowY = (r) => 300 + r * 86, qX = (i) => 96 + i * 140, qY = 680;
  REQ.forEach(([n, d, vs], r) => IN(.4 + r * .15, () => { TX(`请求 ${n}`, 96, rowY(r) + 34, font(700, 30), K.ink); TX(d, 96, rowY(r) + 66, font(600, 26), K.ink2);
    vs.forEach((v, j) => { const q = QUEUE.find((u) => u.r === r && u.j === j), st = T < I0 ? 'guess' : taken(q) ? 'blue' : stopped ? 'void' : 'guess'; chip(rowX(j), rowY(r), 120, 66, st, v.toFixed(2), 28); }); }, .4, 0));
  IN(Q0 - .2, () => TX('按存活概率排序', 96, qY - 22, font(700, 26), K.ink), .4, 0);
  QUEUE.forEach((q, i) => { const p = S(Q0 + i * .08, .7); if (p <= 0) return; const x = mix(rowX(q.j), qX(i), p), y = mix(rowY(q.r), qY, p), st = T < I0 ? 'guess' : i < k ? 'blue' : stopped ? 'void' : 'guess';
    const hh = mix(66, 88, p); chip(x, y, 124, hh, st, null); TX(q.v.toFixed(2), x + 62, y + 36, font(700, 28, HN), st === 'blue' ? '#fff' : st === 'void' ? K.gray : K.purple, 'center'); A(p, () => TX('请求 ' + REQ[q.r][0], x + 62, y + 72, font(700, 24), st === 'blue' ? '#fff' : K.gray, 'center')); });
  // 指针
  if (T > t0 && !stopped) { const xx = qX(Math.max(0, k - 1)) + 124 + 8; line(xx, qY - 8, xx, qY + 96, K.ink, 5); }
  // 右上：期望收下 ÷ 耗时
  const cx0 = 1060, cx1 = 1800, cyT = 300, cyB = 560, gx = (i) => cx0 + i / 12 * (cx1 - cx0), gy = (v) => cyB - (v - .3) / .7 * (cyB - cyT);
  IN(I0 - .8, () => { line(cx0, cyB, cx1, cyB, K.ink, 3); TX('期望接受数 ÷ 单步耗时', cx0, cyT - 14, font(600, 26), K.ink2); TX('加入的 token 数', cx1, cyB + 38, font(600, 26), K.ink2, 'right'); }, .4, 0);
  [[IDLE, I0, K.teal, '低负载'], [BUSY, B0 + .6, K.red, '高负载']].forEach(([m, s0, c, nm]) => { if (T < s0) return; const kk = clampR(Math.floor((T - s0) / DT) + 1, 0, 12), pts = m.rel.slice(0, kk + 1).map((v, i) => [gx(i), gy(v)]);
    A(m === IDLE && busy ? .35 : 1, () => { if (pts.length > 1) poly(pts, c, 5); pts.forEach(([x, y]) => dot(x, y, 6, c));
      if (T > s0 + m.peak * DT + .3) { dot(gx(m.peak), gy(1), 12, c); TX(`${nm}：峰值在 ${m.peak} 个`, gx(m.peak), gy(1) + (nm === '低负载' ? -22 : 44), font(700, 26), c, 'center'); } }); });
  IN(I0 + IDLE.peak * DT + .6, () => RICH('{t:低负载}：验证 10 个（A 4、B 4、C 2）', 96, 812, font(700, 32), K.ink), .4, 0);
  IN(B0 + .6 + BUSY.peak * DT + 1.4, () => TX('线上：每个请求平均验证 4–6 个，MTP-1 固定 2 个', 1824, 862, font(600, 28), K.ink2, 'right'), .5, 0);
  IN(B0 + .6 + BUSY.peak * DT + .6, () => RICH('{r:高负载}：验证 5 个（A 4、B 1、C 0）', 96, 862, font(700, 32), K.ink), .4, 0);
}

// ---------- 13（B）总结：式子的三项各由哪个部件负责，再回到开头的两个窗口 ----------
// 挂在配音节拍上：前三句各点亮一列（与第 2 幕的 ① ② ③ 同位置、同颜色），第四句两个窗口重新输出并给出结果。
const EB = beatTimes(13, [.6, .4, .4, .5]), ENDB_DUR = EB[3].e + 2.4;
function sEndB() {
  const b = (i, f) => at(EB, i, f);
  frame('总结', K.ink2, '吞吐相同时，V4-Flash 单用户快 {b:60%–85%}，V4-Pro 快 {b:57%–78%}', null, .15, b(3, .15));
  formula(330, 350, 76, .3, '每个 token 的耗时');
  [[500, K.purple, '① draft 更快', '并行 backbone', '一次 forward 生成 5 个 draft token'],
   [1000, K.teal, '② 接受更多', 'Markov head', '每轮接受 3.11 → 3.72 个（Qwen3）'],
   [1500, K.blue, '③ 减少无效验证', 'confidence head + 调度器', '线上每个请求验证 4–6 个']].forEach(([x, col, head, part, res], i) => IN(b(i, .15), () => {
    TX(head, x, 520, font(700, 36), col, 'center'); X.font = font(700, 28, HN); const w = X.measureText(part).width + 44;
    cell(x - w / 2, 542, w, 54, col, part, 28, '#fff', HN); TX(res, x, 640, font(600, 28), K.ink, 'center'); }, .5, 0));
  const el = clamp((T - b(3)) / 2.5) * 2.5;
  [0, 1].forEach((i) => IN(b(3) - .4, () => chatWin([96, 972][i], 684, 852, 256, ['MTP-1', 'DSpark'][i], TPS[i], TPS[i] * el, [K.ink2, K.purple][i], 20), .5, 0));
}
const ENDB_EV = (() => { const b = (i, f) => at(EB, i, f); return [[.1, 'click'], [b(0, .15), 'pop'], [b(1, .15), 'pop'], [b(2, .15), 'pop'], [b(3) - .4, 'whoosh'], ...Array.from({ length: 10 }, (_, k) => [b(3) + k * .25, 'tick']), [b(3) + 2.6, 'land']]; })();

// ---------- 13 回到开头 ----------
function sEnd() {
  frame('总结', K.ink2, '吞吐相同时，单用户生成速度提升 {b:60%–85%}', '论文 Fig. 7：V4-Flash / V4-Pro 线上流量');
  const el = clamp((T - 1.0) / 2.0) * 2.0;
  [0, 1].forEach((i) => IN(.3, () => chatWin([96, 972][i], 290, 852, 330, ['MTP-1', 'DSpark'][i], TPS[i], TPS[i] * el, [K.ink2, K.purple][i], 20), .5, 0));
  const M = formula(560, 708, 60, 3.8, '每个 token 的耗时');
  fLabel(560, 880, 5.0, K.purple, '一次 forward + Markov head', 'draft 耗时仅增加约 1%', 30);
  fLabel(1000, 880, 5.8, K.teal, '第 1 位高，后续不下降', '每轮接受更多', 30);
  fLabel(1440, 880, 6.6, K.blue, '按存活概率与负载验证', '每个请求 4–6 个', 30);
}
