// 第 1–4 幕：logits 有多大 → loss 需要什么 → 梯度原地写回 → 按 token 分块。数字全部由形状推出（见 script.md）
const NV = 151936, NH = 2048, NT = 16384, CH = 256, NC = NT / CH, BLK = 32768;
const LOGITS_BF16 = NT * NV * 2, LOGITS_FP32 = NT * NV * 4, HID = NT * NH * 2, CHUNK_LOGITS = CH * NV * 2;

// ---------- 1 logits 有多大 ----------
// 横轴按真实比例：logits 一行 1110 px 对应 151936 个数，hidden state 一行 2048 个数只有 15 px。
// 第三句把一行堆成 16384 行（两块同高），第四句在右侧的显存柱上依次叠上 BF16 logits、FP32 副本与梯度
const B1 = beatTimes(1, [.5, .6, .6, .6]), S1_DUR = B1[3].e + 1.4;
function s1() {
  const b = (i, f) => at(B1, i, f), LX = 400, LW = 900, hw = LW * NH / NV, y0 = 330;
  IN(b(1, .7), () => head(0, 'logits 比 hidden state 长 {y:74 倍}'));
  const grow = S(b(2), 1.2), rows = mix(56, 520, grow);            // 一行 → 16384 行
  const lmA = 1 - S(b(2) - .3, .5);
  // 一个 token 的 hidden state
  IN(b(0), () => { R(LX - 200, y0, hw, rows, K.blue); TX('hidden state', LX - 200, y0 - 22, font(700, 28, HN), K.blue); }, .5, 0);
  // lm_head：2048 × 151936，按同一比例画成高 15 px 的细长矩阵
  A(lmA * P(b(0, .15), .5), () => { R(LX, y0 + 160, LW, hw, K.ink); TX('lm_head：2048 × 151936', LX, y0 + 220, font(700, 28, HN), K.ink);
    arrowP(LX - 200 + hw + 8, y0 + 28, LX - 12, y0 + 28, P(b(0, .3), .5), K.ink2, 4); TX('× lm_head', LX - 100, y0 + 12, font(700, 24, HN), K.ink2, 'center'); });
  // logits：一行 151936 个分数
  IN(b(0, .55), () => { R(LX, y0, LW, rows, K.yel); TX('logits', LX, y0 - 22, font(700, 28, HN), K.yelD); }, .5, 0);
  // 74 份 hidden state 宽度并排铺满一行 logits
  const tiles = clamp((T - b(1, .25)) / 1.6) * 74;
  if (tiles > 0 && grow < .02) for (let i = 0; i < Math.floor(tiles); i++) R(LX + i * hw + 1, y0 + 64, hw - 2, 14, K.blue);
  IN(b(1, .25), () => { if (grow < .5) TX(`${Math.floor(tiles)} 份 hidden state 的长度`, LX, y0 + 116, font(700, 28), K.blue); }, .4, 0);
  IN(b(1), () => { TX('2048', LX - 200, y0 + rows + 40, font(700, 28, HN), K.blue); TX('151936', LX + LW, y0 + rows + 40, font(700, 28, HN), K.yelD, 'right'); }, .4, 0);
  // 16384 行
  IN(b(2, .3), () => { span(LX + LW + 26, LX + LW + 26, y0, K.ink, 3); line(LX + LW + 26, y0, LX + LW + 26, y0 + rows, K.ink2, 3);
    X.save(); X.translate(LX + LW + 50, y0 + rows / 2); X.rotate(Math.PI / 2); X.font = font(700, 28); X.fillStyle = K.ink2; X.textAlign = 'center'; X.fillText('16384 个 token', 0, 0); X.restore();
    TX('64 MB', LX - 200 + hw / 2, y0 + rows + 84, font(700, 34, HN), K.blue, 'center');
    TX(GB(LOGITS_BF16), LX + LW / 2, y0 + rows / 2 + 16, font(700, 56, HN), K.ink, 'center'); }, .5, 0);
  // 右：这一层的显存（柱高按 GB 计）
  const MX = 1420, MW = 150, MB0 = 860, U = 26;
  // 实测峰值 27.82 GB（eager，16384 token）= BF16 logits + FP32 副本 + 其余（log_softmax 输出与梯度，FP32）
  const EAGER_PEAK = window.DATA.micro['eager/16384'].gb * 2 ** 30;
  const segs = [[LOGITS_BF16, K.yel, 'logits（BF16）', b(2, .3)], [LOGITS_FP32, 'h' + K.yel, 'FP32 副本', b(3, .1)], [EAGER_PEAK - LOGITS_BF16 - LOGITS_FP32, K.red, 'log_softmax 输出与梯度', b(3, .55)]];
  IN(b(2, .3), () => { line(MX - 20, MB0, MX + MW + 20, MB0, K.ink, 3); TX('这一层的显存', MX + MW / 2, MB0 + 44, font(700, 28), K.ink, 'center'); }, .4, 0);
  let top = MB0, tot = 0;
  segs.forEach(([bytes, col, name, t0]) => { const p = P(t0, .6); if (p <= 0) return; const h = bytes / 2 ** 30 * U * p; top -= h; tot += bytes * p;
    if (col[0] === 'h') HR(MX, top, MW, h, K.yel, K.paper); else R(MX, top, MW, h, col);
    A(p, () => { TX(name, MX + MW + 16, top + h / 2 - 4, font(700, 24), col === K.red ? K.red : K.yelD); TX(GB(bytes), MX + MW + 16, top + h / 2 + 26, font(700, 24, HN), K.ink2); }); });
  IN(b(3, .8), () => TX(`峰值 ${GB(tot)}`, MX + MW / 2, top - 18, font(700, 32, HN), K.ink, 'center'), .4, 0);
}

// ---------- 2 一个 token 的 loss 需要什么 ----------
// 一行 logits 用 48 根柱子示意（真实一行有 151936 个数），正确词为第 30 根。loss = log-sum-exp − 正确词的 logit
const NB = 48, YI = 30, LG = Array.from({ length: NB }, (_, i) => { const r = Math.sin(i * 12.9898) * 43758.5453; return 1.6 + 3.2 * (r - Math.floor(r)) + (i === YI ? 2.6 : 0); });
const LSE = Math.log(LG.reduce((a, v) => a + Math.exp(v), 0)), SM = LG.map((v) => Math.exp(v - LSE));
const GMAX = Math.max(...SM.filter((_, i) => i !== YI));
const B2 = beatTimes(2, [.5, .5, .5, .6]), S2_DUR = B2[3].e + 1.4;
function s2() {
  const b = (i, f) => at(B2, i, f), x0 = 160, bw = 16, gap = 4, bx = (i) => x0 + i * (bw + gap), base = 560, U = 28;
  IN(b(1, .2), () => head(0, 'loss 只需要{t:两个数}，梯度用完即丢'));
  const toGrad = S(b(2), 1.0);
  IN(b(0), () => TX('一个 token 的 logits（示意：真实一行有 151936 个数）', x0, 300, font(700, 28), K.ink2), .4, 0);
  LG.forEach((v, i) => IN(b(0) + i * .012, () => {
    const g = SM[i] - (i === YI ? 1 : 0), hL = v * U, hG = Math.max(-170, g / GMAX * 150);   // 梯度柱：softmax 减 one-hot，按最大正值缩放；正确词处约为 −0.5，截断显示
    const h = mix(hL, hG, toGrad), col = toGrad > .5 ? K.red : i === YI ? K.teal : K.yel;
    if (h >= 0) R(bx(i), base - h, bw, h, col); else R(bx(i), base, bw, -h, col);
  }, .3, 0));
  IN(b(0), () => line(x0 - 10, base, bx(NB) + 4, base, K.ink, 3), .3, 0);
  IN(b(0, .3), () => { if (toGrad < .5) TX('正确词', bx(YI) + bw / 2, base - LG[YI] * U - 16, font(700, 26), K.teal, 'center'); }, .3, 0);
  // 公式与两个数
  const fx = 1180;
  IN(b(0, .45), () => RICH('loss = {y:log-sum-exp} − {t:正确词的 logit}', fx, 360, font(700, 36, HN), K.ink), .5, 0);
  IN(b(1), () => { cell(fx, 410, 300, 80, K.yelD, `log-sum-exp ${LSE.toFixed(2)}`, 28); cell(fx + 320, 410, 280, 80, K.teal, `logit ${LG[YI].toFixed(2)}`, 28);
    RICH(`loss = ${LSE.toFixed(2)} − ${LG[YI].toFixed(2)} = {t:${(LSE - LG[YI]).toFixed(2)}}`, fx, 548, font(700, 32, HN), K.ink); }, .5, 0);
  IN(b(2, .2), () => { RICH('梯度 = {r:softmax − one-hot}', fx, 640, font(700, 34, HN), K.ink); TX('与 logits 一样长，正确词处为负', fx, 686, font(600, 26), K.ink2); }, .5, 0);
  IN(b(2, .2), () => TX('变为梯度（纵轴放大）', x0, 300 + 44, font(700, 26), K.red), .4, 0);
  // 两次乘法
  const gy = 820, gw = 760;
  IN(b(3), () => { R(x0, gy, gw, 40, K.red); TX('梯度：151936 个数', x0 + 16, gy + 30, font(700, 26), '#fff'); }, .4, 0);
  arrowP(x0 + gw + 16, gy + 20, fx - 24, gy - 32, S(b(3, .1), .5), K.blue, 4);
  arrowP(x0 + gw + 16, gy + 20, fx - 24, gy + 62, S(b(3, .45), .5), K.ink, 4);
  IN(b(3, .15), () => RICH('× lm_head → {b:hidden state 的梯度 dX}', fx, gy - 22, font(700, 30), K.ink), .4, 0);
  IN(b(3, .5), () => RICH('× hidden state → 累加到 lm_head 的梯度 dW', fx, gy + 72, font(700, 30), K.ink), .4, 0);
  A(S(b(3, .85), .6), () => { HR(x0, gy, gw, 40, K.lg, K.paper2); TX('用完即丢', x0 + 16, gy + 30, font(700, 26), K.gray); });
}

// ---------- 3 梯度原地写回 ----------
// Liger 的 CE kernel：每个 token 一个 program，一行按 BLOCK_SIZE = 32768 分成 5 段（151936 / 32768 ≈ 4.6）。
// 第一遍逐段更新最大值 m 与指数和 d（online softmax），第二遍把 softmax / N 写回同一块显存，正确词位置单独用 FP32 重算
const B3 = beatTimes(3, [.5, .5, .6]), S3_DUR = B3[2].e + 1.6;
function s3() {
  const b = (i, f) => at(B3, i, f), x0 = 96, W = 1728, y = 380, h = 110, segW = W * BLK / NV, nSeg = Math.ceil(NV / BLK);
  IN(b(1, .3), () => head(0, '梯度{r:原地覆盖} logits'));
  IN(b(0), () => TX('一个 token 的一行 logits：151936 个数，kernel 每次读入 32768 个', x0, y - 40, font(700, 28), K.ink2), .4, 0);
  // 第一遍：b(0, .35) 起扫描；第二遍：b(1) 起逐段写回
  const p1 = clamp((T - b(0, .35)) / (B3[0].d * .5)) * nSeg, p2 = clamp((T - b(1, .1)) / (B3[1].d * .7)) * nSeg;
  IN(b(0), () => {
    for (let s = 0; s < nSeg; s++) { const sx = x0 + s * segW, sw = Math.min(segW, x0 + W - sx), red = clamp(p2 - s);
      R(sx, y, sw, h, K.yel); if (red > 0) R(sx, y, sw * red, h, K.red); line(sx, y - 6, sx, y + h + 6, K.paper, 4); }
  }, .5, 0);
  // 扫描框
  const scan = T < b(1, .1) ? p1 : p2, pass = T < b(1, .1) ? 1 : 2;
  if (scan > 0 && scan < nSeg) { const s = Math.floor(scan), sx = x0 + s * segW; box(sx, y - 12, Math.min(segW, x0 + W - sx), h + 24, 6, pass === 1 ? K.ink : K.red);
    TX(`第 ${pass} 遍`, sx + 8, y + h + 50, font(700, 28), pass === 1 ? K.ink : K.red); }
  // 第一遍的读数：m 与 d 随段数更新（示意数值）
  const seen = Math.min(nSeg, Math.ceil(p1)), M = [6.1, 6.8, 6.8, 7.9, 7.9].slice(0, seen), m = M.length ? M[M.length - 1] : 0, D = [1840, 3110, 4460, 2390, 3510].slice(0, seen).reduce((a, v) => a + v, 0);
  IN(b(0, .35), () => { const ry = 620;
    RICH(`最大值 m = {y:${seen ? m.toFixed(1) : '−∞'}}`, x0, ry, font(700, 34, HN), K.ink);
    RICH(`指数和 d = {y:${seen ? (D / 1e3).toFixed(2) + 'k' : '0'}}`, x0 + 420, ry, font(700, 34, HN), K.ink);
    TX('m 变大时，d 先按 exp(m旧 − m新) 缩小再累加', x0 + 860, ry, font(600, 28), K.ink2); }, .4, 0);
  IN(b(0, .9), () => RICH('读完：log-sum-exp = m + log d，{t:loss = log-sum-exp − 正确词的 logit}', x0, 684, font(700, 32, HN), K.ink), .4, 0);
  IN(b(1, .3), () => RICH('第二遍：{r:softmax / N} 写回同一地址，正确词处再减 1/N', x0, 748, font(700, 32, HN), K.ink), .4, 0);
  // 实测：这一层一步 forward + backward 的峰值（16384 token，H100）
  const my = 826, U = 44, ev = window.DATA.micro['eager/16384'].gb, lc = window.DATA.micro['liger_ce/16384'].gb;
  IN(b(2), () => { TX('eager', x0, my + 34, font(700, 28, HN), K.ink2); R(x0 + 190, my, ev * U, 44, K.ink2); TX(`${ev.toFixed(1)} GB`, x0 + 206 + ev * U, my + 34, font(700, 28, HN), K.ink); }, .4, 0);
  IN(b(2, .2), () => { TX('Liger CE', x0, my + 96, font(700, 28, HN), K.ink2); R(x0 + 190, my + 62, lc * U, 44, K.red); R(x0 + 190, my + 62, LOGITS_BF16 / 2 ** 30 * U, 44, K.red);
    TX(`${lc.toFixed(1)} GB，其中 logits ${GB(LOGITS_BF16)}`, x0 + 206 + lc * U, my + 96, font(700, 28, HN), K.ink); }, .4, 0);
}

// ---------- 4 按 token 分块 ----------
// 纵轴按真实比例：16384 个 token 高 640 px，每块 256 个只有 10 px。完整 logits 只画虚线轮廓，实际只存在当前这一块（黄 → 红）。
// 每块：logits = X_c · Wᵀ → CE kernel 原地变成梯度 G_c → dX_c = G_c · W → dW += G_cᵀ · X_c
const B4 = beatTimes(4, [.5, .6, .5, .5, .6, .6]), S4_DUR = B4[5].e + 1.4;
function s4() {
  const b = (i, f) => at(B4, i, f), y0 = 270, HT = 640, rh = HT / NC, hx = 330, hwid = 16, gx = 440, gw = 820, dxx = 1330, wx = 1440, wy = 300, ww = 380, wh = 180;
  IN(b(0, .3), () => head(0, '按 token 分块：logits 一次只存{p:一块}'));
  // 当前块：第一块从 b(1) 讲到 b(3) 结束；b(4) 起其余 63 块依次走完
  const rest = clamp((T - b(4, .05)) / (B4[4].d * .8)), first = T < b(4, .05), cur = first ? 0 : Math.min(NC - 1, 1 + Math.floor(rest * (NC - 1))), all = rest >= 1;
  const done = first ? (T > b(3, .5) ? 1 : 0) : all ? NC : cur;   // 已完成的块数
  const red0 = clamp((T - b(2)) / (B4[2].d * .45));                  // 第一块从黄变红的进度
  // hidden state 一列（左侧文字说明块数）
  IN(b(0), () => { R(hx, y0, hwid, HT, K.blue); TX('hidden state', hx + hwid / 2, y0 - 22, font(700, 26, HN), K.blue, 'center');
    for (let c = 1; c < NC; c++) line(hx - 6, y0 + c * rh, hx + hwid + 6, y0 + c * rh, K.paper, 1.5); }, .4, 0);
  IN(b(0, .5), () => { TX('16384 个 token', 96, y0 + HT / 2 - 44, font(700, 28), K.ink); TX('切成 64 块', 96, y0 + HT / 2 + 4, font(700, 32), K.purple); TX('每块 256 个', 96, y0 + HT / 2 + 46, font(600, 26), K.ink2); }, .4, 0);
  // 完整 logits 的轮廓（不存在）
  IN(b(0, .2), () => { X.setLineDash([10, 8]); box(gx, y0, gw, HT, 3, K.lg); X.setLineDash([]);
    TX(`完整 logits ${GB(LOGITS_BF16)}：不再整块存在`, gx + gw / 2, y0 + HT - 40, font(700, 30), K.lg, 'center'); }, .4, 0);
  // 当前块：按真实比例只有 10 px 高
  if (T > b(1, .2) && !all) {
    const cy = y0 + cur * rh, fill = first ? S(b(1, .3), .8) : 1, red = first ? red0 : clamp((rest * (NC - 1)) % 1 * 2);
    box(hx - 5, cy - 2, hwid + 10, rh + 4, 4, K.purple);
    R(gx, cy, gw * fill, rh, K.yel); if (red > 0) R(gx, cy, gw * red, rh, K.red);
    if (first) arrowP(hx + hwid + 8, cy + rh / 2, gx - 8, cy + rh / 2, S(b(1, .15), .4), K.purple, 4);
  }
  // 放大：第一块 256 × 151936，跟着它走完三步
  const zA = first ? P(b(1, .35), .5) : 1 - P(b(4, .05), .4), zy = 380, zh = 150;
  A(zA, () => { X.setLineDash([6, 6]); line(gx, y0 + rh, gx, zy, K.gray, 2); line(gx + gw, y0 + rh, gx + gw, zy, K.gray, 2); X.setLineDash([]);
    R(gx, zy, gw, zh, K.yel); if (red0 > 0) R(gx, zy, gw * red0, zh, K.red);
    TX('放大：第一块的 logits', gx, zy - 14, font(700, 26), K.ink2);
    TX(red0 >= 1 ? '同一块显存，已变成梯度' : `256 × 151936，${MB(CHUNK_LOGITS)}`, gx + gw / 2, zy + zh / 2 + 14, font(700, 38, HN), red0 > .5 ? '#fff' : K.ink, 'center');
    IN(b(2, .1), () => TX('CE kernel 原地写回', gx + gw * Math.min(red0, 1), zy + zh + 36, font(700, 26), K.red, red0 > .7 ? 'right' : 'left'), .3, 0); });
  // 右：dX（逐块填满）与 dW（逐块累加）
  IN(b(2, .5), () => { box(dxx, y0, hwid, HT, 2, K.blue); TX('dX', dxx + hwid / 2, y0 - 22, font(700, 26, HN), K.blue, 'center'); R(dxx, y0, hwid, Math.max(done, T > b(2, .7) ? 1 : 0) * rh, K.blue); }, .4, 0);
  if (first && T > b(2, .5)) { arrowP(gx + gw + 8, zy + 40, dxx - 10, y0 + rh / 2 + 4, S(b(2, .5), .4), K.blue, 4);
    IN(b(2, .55), () => TX('× lm_head → dX', gx + gw - 12, y0 + 60, font(700, 26, HN), K.blue, 'right'), .3, 0); }
  const acc = T < b(3) ? 0 : Math.max(done, T > b(3, .3) ? 1 : 0) / NC;
  IN(b(3), () => { box(wx, wy, ww, wh, 3, K.ink); R(wx, wy + wh * (1 - acc), ww, wh * acc, K.ink2);
    TX('dW：lm_head 的梯度', wx, wy - 18, font(700, 26), K.ink); TX(`已累加 ${Math.round(acc * NC)} / 64 块`, wx + ww / 2, wy + wh + 44, font(700, 28, HN), K.ink, 'center'); }, .4, 0);
  if (first && T > b(3)) { arrowP(gx + gw + 8, zy + zh - 30, wx - 10, wy + wh - 30, S(b(3, .1), .5), K.ink, 4);
    IN(b(3, .15), () => TX('× 这 256 个 hidden state → 累加进 dW', gx + gw - 12, zy + zh + 84, font(700, 26), K.ink, 'right'), .3, 0); }
  // 块大小的来历
  IN(b(4, .3), () => RICH(`每块 token 数 = 16384 ÷ ⌈151936 ÷ 2048⌉ ≈ 219，取 2 的幂：{p:256}；每块 logits ${MB(CHUNK_LOGITS)}，全部 hidden state ${MB(HID)}`, 96, 952, font(700, 28, HN), K.ink), .4, 0);
  // backward
  IN(b(5, .1), () => { R(wx - 10, 600, ww + 20, 150, K.paper2); TX('backward', wx + 6, 642, font(700, 30, HN), K.purple);
    TX('dX、dW 已在 forward 中算好', wx + 6, 686, font(600, 26), K.ink); TX('只乘传回的系数（为 1 时跳过）', wx + 6, 726, font(600, 26), K.ink); }, .4, 0);
}
