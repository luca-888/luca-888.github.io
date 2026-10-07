// 第 0、5–8 幕：H100 实测（data.js，由 data.py 从 public/measurements/fused-linear-cross-entropy/ 生成）
const D = window.DATA;

// ---------- 显存时间线（第 0 幕与第 8 幕共用） ----------
// Qwen3-1.7B 一步训练（16384 token，gradient checkpointing，AdamW），横轴为这一步内的时间，纵轴为已分配显存
const TL = { x: 200, y: 290, w: 1560, h: 560 };
const TL_MAX = Math.ceil(Math.max(D.timeline.eager.peak_gb, D.timeline.flce.peak_gb) / 10) * 10;
const tlX = (t, end) => TL.x + t / end * TL.w, tlY = (g) => TL.y + TL.h - g / TL_MAX * TL.h;
// 尖峰：第一次 ≥ 2 GB 的分配（logits）之前的显存为起点，曲线回落到起点以下为止
function spikeOf(m) {
  const tl = D.timeline[m], big = tl.big.find((a) => a.gb >= 2), pts = tl.points;
  if (!big) return null;
  const i0 = Math.max(0, pts.findIndex((p) => p[0] >= big.t) - 1), pre = pts[i0][1];
  let i1 = pts.findIndex((p, i) => i > i0 && p[1] === tl.peak_gb); const ip = i1;
  while (i1 < pts.length - 1 && pts[i1][1] > pre + .2 * (tl.peak_gb - pre)) i1++;
  return { t0: pts[i0][0], t1: pts[i1][0], pre, peak: tl.peak_gb, tp: pts[ip][0] };
}
function axes(end) {
  line(TL.x, TL.y + TL.h, TL.x + TL.w, TL.y + TL.h, K.ink, 3); line(TL.x, TL.y - 10, TL.x, TL.y + TL.h, K.ink, 3);
  for (let g = 0; g <= TL_MAX; g += 10) { const y = tlY(g); if (g) line(TL.x, y, TL.x + TL.w, y, K.paper2, 2); TX(`${g}`, TL.x - 16, y + 9, font(600, 26, HN), K.ink2, 'right'); }
  TX('显存（GB）', TL.x - 16, TL.y - 30, font(700, 26), K.ink2);
  TX('一步训练中的时间 →', TL.x + TL.w, TL.y + TL.h + 44, font(600, 26), K.ink2, 'right');
  const sp = spikeOf('eager'); if (sp) { const xs = tlX(sp.t0, D.timeline.eager.end_s);
    TX('forward', (TL.x + xs) / 2, TL.y + TL.h + 44, font(700, 26, HN), K.ink2, 'center'); TX('backward（含重计算）', xs + 24, TL.y + TL.h + 44, font(700, 26, HN), K.ink2); }
}
function curve(m, p, col, lw = 4) {
  const tl = D.timeline[m], end = tl.end_s, pts = tl.points, n = Math.floor(pts.length * clamp(p));
  if (n < 2) return; X.strokeStyle = col; X.lineWidth = lw; X.lineJoin = 'round'; X.beginPath();
  for (let i = 0; i < n; i++) { const x = tlX(pts[i][0], end), y = tlY(pts[i][1]); i ? X.lineTo(x, y) : X.moveTo(x, y); }
  X.stroke();
}

const B0 = beatTimes(0, [.8, .6, .5]), S0_DUR = B0[2].e + 1.2;
function s0() {
  const b = (i, f) => at(B0, i, f), e = D.timeline.eager, sp = spikeOf('eager'), end = e.end_s;
  IN(.1, () => { TX('Fused Linear Cross Entropy', 96, 150, font(700, 64, HN), K.ink); TX('Liger Kernel · Qwen3-1.7B · H100', 96, 206, font(600, 32, HN), K.ink2); }, .5, 0);
  IN(.3, () => axes(end), .5, 0);
  // 参数与优化器状态：这一步开始时已经在显存里
  IN(b(1), () => { line(TL.x, tlY(e.start_gb), TL.x + TL.w, tlY(e.start_gb), K.ink2, 2, [8, 6]);
    TX(`参数 + 优化器状态 ${e.start_gb.toFixed(1)} GB`, TL.x + TL.w, tlY(e.start_gb) + 34, font(700, 26), K.ink2, 'right'); }, .4, 0);
  curve('eager', clamp((T - b(0, .1)) / (B0[0].d * .9)), K.ink, 4);
  if (!sp) return;
  // 尖峰高度
  IN(b(1, .3), () => { const x = tlX(sp.tp, end) + 30; span(sp.pre, sp.peak, 0); line(x, tlY(sp.pre), x, tlY(sp.peak), K.red, 4); line(x - 10, tlY(sp.pre), x + 10, tlY(sp.pre), K.red, 4); line(x - 10, tlY(sp.peak), x + 10, tlY(sp.peak), K.red, 4);
    TX(`+${(sp.peak - sp.pre).toFixed(1)} GB`, x + 24, (tlY(sp.pre) + tlY(sp.peak)) / 2 + 14, font(700, 44, HN), K.red); }, .4, 0);
  // 尖峰区间：最后一层
  IN(b(2), () => { X.save(); X.globalAlpha *= .22; R(tlX(sp.t0, end), TL.y, tlX(sp.t1, end) - tlX(sp.t0, end), TL.h, K.yel); X.restore();
    TX('最后一层：lm_head + cross-entropy', tlX(sp.t1, end) + 200, TL.y + 40, font(700, 30), K.yelD); }, .4, 0);
}

const B8 = beatTimes(8, [.5]), S8_DUR = B8[0].e + 2.4;
function s8() {
  const b = (i, f) => at(B8, i, f), e = D.timeline.eager, f = D.timeline.flce, end = Math.max(e.end_s, f.end_s);
  head(0, '尖峰消失');
  axes(end);
  // 两条曲线各按自己的时长画；横轴取较长者，FLCE 慢的部分直接可见
  const sc = (m) => { const tl = D.timeline[m]; return tl.end_s / end; };
  X.save(); X.translate(TL.x, 0); X.scale(sc('eager'), 1); X.translate(-TL.x, 0); A(1 - .55 * P(b(0, .2), .6), () => curve('eager', 1, K.ink, 4 / sc('eager'))); X.restore();
  X.save(); X.translate(TL.x, 0); X.scale(sc('flce'), 1); X.translate(-TL.x, 0); curve('flce', clamp((T - b(0, .2)) / 2.4), K.purple, 5 / sc('flce')); X.restore();
  IN(b(0, .3), () => { TX(`eager 峰值 ${e.peak_gb.toFixed(1)} GB`, 1760, tlY(e.peak_gb) - 16, font(700, 30, HN), K.ink2, 'right'); }, .4, 0);
  IN(b(0, .7), () => { TX(`FLCE 峰值 ${f.peak_gb.toFixed(1)} GB`, 1760, tlY(f.peak_gb) - 16, font(700, 34, HN), K.purple, 'right'); }, .4, 0);
}

// ---------- 通用折线图 ----------
// 横轴为 token 数（4K…64K，按 2 的幂等距），纵轴线性；series: [[名字, 颜色, [[i, 值] …]]]
const TOKS = [4096, 8192, 16384, 32768, 65536], TK = ['4K', '8K', '16K', '32K', '64K'];
function lineChart(x, y, w, h, ymax, ystep, unit, series, p = 1) {
  const px = (i) => x + i / (TOKS.length - 1) * w, py = (v) => y + h - v / ymax * h;
  line(x, y + h, x + w + 10, y + h, K.ink, 3); line(x, y - 10, x, y + h, K.ink, 3);
  for (let v = 0; v <= ymax; v += ystep) { if (v) line(x, py(v), x + w, py(v), K.paper2, 2); TX(`${v}`, x - 12, py(v) + 9, font(600, 24, HN), K.ink2, 'right'); }
  TX(unit, x - 12, y - 28, font(700, 24), K.ink2);
  TK.forEach((t, i) => TX(t, px(i), y + h + 36, font(600, 24, HN), K.ink2, 'center'));
  TX('token 数', x + w, y + h + 72, font(600, 24), K.ink2, 'right');
  series.forEach(([name, col, pts], k) => { const n = Math.min(pts.length, 1 + (pts.length - 1) * clamp(p)); if (n < 1) return;
    const vis = pts.slice(0, Math.ceil(n)); poly(vis.map(([i, v]) => [px(i), py(v)]), col, 5); vis.forEach(([i, v]) => dot(px(i), py(v), 8, col)); });
  return { px, py };
}
const M = (m, t) => D.micro[`${m}/${t}`];

// ---------- 5 显存与时间 ----------
const B5 = beatTimes(5, [.5, .6, .6, .6]), S5_DUR = B5[3].e + 1.4;
function s5() {
  const b = (i, f) => at(B5, i, f), v1 = 1 - P(b(2), .5), v2 = win(b(2), b(3)), v3 = P(b(3), .5);
  IN(.1, () => head(0, '显存几乎不变，时间取决于{p:块数}'));
  // 第一、二句：左图峰值显存，右图时间
  A(v1, () => {
    const mem = (m) => TOKS.map((t, i) => [i, M(m, t)]).filter(([, d]) => d && d.gb != null).map(([i, d]) => [i, d.gb]);
    IN(b(0), () => { const c = lineChart(160, 300, 700, 520, 80, 20, '峰值显存（GB）', [['eager', K.ink, mem('eager')], ['FLCE', K.purple, mem('flce')]], clamp((T - b(0, .2)) / 2));
      line(160, c.py(80), 860, c.py(80), K.red, 2, [8, 6]); TX('H100 80 GB', 860, c.py(80) - 12, font(700, 24, HN), K.red, 'right');
      IN(b(0, .45), () => { TX('OOM', c.px(4), c.py(80) + 44, font(700, 34, HN), K.red, 'center'); TX('eager', c.px(3) - 20, c.py(M('eager', 32768).gb) - 10, font(700, 28, HN), K.ink, 'right'); }, .3, 0);
      IN(b(0, .7), () => TX(`FLCE ${M('flce', 65536).gb.toFixed(1)} GB`, c.px(4), c.py(M('flce', 65536).gb) - 22, font(700, 28, HN), K.purple, 'right'), .3, 0); }, .4, 0);
    IN(b(1), () => { const ms = (m) => TOKS.map((t, i) => [i, M(m, t).ms]).filter(([, v]) => v != null);
      const c = lineChart(1080, 300, 700, 520, 250, 50, '时间（毫秒）', [['eager', K.ink, ms('eager')], ['FLCE', K.purple, ms('flce')]], clamp((T - b(1, .1)) / 2));
      IN(b(1, .55), () => { RICH(`{p:${M('flce', 4096).ms.toFixed(0)}} 对 ${M('eager', 4096).ms.toFixed(0)} ms`, c.px(0) + 20, c.py(M('flce', 4096).ms) - 24, font(700, 28, HN), K.ink); }, .3, 0);
      IN(b(1, .3), () => TX(`16K：${M('eager', 16384).ms.toFixed(0)} 对 ${M('flce', 16384).ms.toFixed(0)} ms`, c.px(2) + 20, c.py(M('eager', 16384).ms) + 50, font(700, 26, HN), K.ink2), .3, 0); }, .4, 0);
  });
  // 第三句：16384 token 一步的 kernel 时间构成（1 ms = 11 px）
  A(v2, () => {
    const U = 11, x0 = 300, ek = D.kernels.eager.by_cat, fk = D.kernels.flce.top;
    const t = (frag) => fk.filter(([n]) => n.includes(frag)).reduce((a, r) => a + r[2], 0);
    const fl = [['dW 累加 ×64', t('badd'), K.ink], ['logits ×64', t('bz_coopA_TNT'), K.yelD], ['dX ×64', t('splitK_NNT') + t('splitKreduce'), K.blue], ['CE kernel ×64', t('liger_cross_entropy'), K.red]];
    const eg = [['矩阵乘 ×3', ek.gemm, K.ink2], ['softmax 前后向', ek.softmax, K.yel], ['类型转换等', ek.other, K.lg]];
    const bar = (y, name, parts, t0) => IN(t0, () => { TX(name, x0 - 24, y + 52, font(700, 30, HN), K.ink, 'right'); let x = x0;
      parts.forEach(([n, ms, c]) => { R(x, y, ms * U - 3, 84, c); if (ms * U > 120) TX(n, x + 12, y + 36, font(700, 22), c === K.yel || c === K.lg ? K.ink : '#fff'); if (ms * U > 80) TX(`${ms.toFixed(1)}`, x + 12, y + 70, font(700, 24, HN), c === K.yel || c === K.lg ? K.ink : '#fff'); x += ms * U; });
    }, .4, 0);
    TX('16384 个 token，一步 forward + backward 的 GPU 时间（毫秒）', 96, 300, font(700, 28), K.ink2);
    bar(380, 'eager', eg, b(2));
    bar(540, 'FLCE', fl, b(2, .15));
    IN(b(2, .45), () => { const w = t('badd') * U; box(x0 - 4, 536, w + 4, 92, 5, K.red); TX('每块都把整个 dW 读出、加上、写回', x0, 690, font(700, 30), K.red); }, .3, 0);
  });
  // 第四句：块大小扫描（16384 token）
  A(v3, () => {
    const C = ['1', '2', '4', '8', '16', '32', '128'], CS = [256, 512, 1024, 2048, 4096, 8192, 16384], n = C.length, x = 260, w = 1300, y = 320, h = 460;
    const px = (i) => x + i / (n - 1) * w, pt = (v) => y + h - v / 100 * h, pg = (v) => y + h - v / 6 * h;
    line(x - 40, y + h, x + w + 40, y + h, K.ink, 3);
    CS.forEach((c, i) => { TX(`${c}`, px(i), y + h + 36, font(600, 24, HN), K.ink2, 'center'); TX(`${16384 / c} 块`, px(i), y + h + 66, font(600, 22), K.gray, 'center'); });
    TX('每块 token 数', x + w + 40, y + h + 100, font(600, 24), K.ink2, 'right');
    C.forEach((c, i) => { const d = D.sweep[c]; R(px(i) - 40, pg(d.gb), 80, y + h - pg(d.gb), '#c9b6fb'); TX(`${d.gb.toFixed(1)} GB`, px(i), y + h - 14, font(700, 22, HN), K.purple, 'center'); });
    poly(C.map((c, i) => [px(i), pt(D.sweep[c].ms)]), K.ink, 5); C.forEach((c, i) => dot(px(i), pt(D.sweep[c].ms), 9, K.ink));
    C.forEach((c, i) => TX(`${D.sweep[c].ms.toFixed(0)} ms`, px(i), pt(D.sweep[c].ms) - 20, font(700, 26, HN), K.ink, 'center'));
    const e = M('eager', 16384).ms; line(x - 40, pt(e), x + w + 40, pt(e), K.red, 2, [8, 6]); TX(`eager ${e.toFixed(0)} ms`, x + w + 40, pt(e) - 12, font(700, 24, HN), K.red, 'right');
    TX('柱：峰值显存　线：时间', x - 40, y - 30, font(600, 24), K.ink2);
    IN(b(3, .3), () => box(px(3) - 56, y - 12, 112, h + 90, 5, K.purple), .3, 0);
  });
}

// ---------- 6 整步训练 ----------
const SEQS = [4096, 8192, 16384, 32768, 65536, 131072], SK = ['4K', '8K', '16K', '32K', '64K', '128K'];
const B6 = beatTimes(6, [.5]), S6_DUR = B6[0].e + 1.4;
function s6() {
  const b = (i, f) => at(B6, i, f), x = 220, w = 1520, y = 300, h = 520, U = h / 80, gw = w / SEQS.length, run = (m, s) => D.e2e.runs[`${m}/${s}`];
  IN(.1, () => head(0, '同一张卡，序列长度{p:4 倍}'));
  IN(.3, () => { line(x, y + h, x + w, y + h, K.ink, 3); line(x, y + h - 80 * U, x + w, y + h - 80 * U, K.red, 2, [8, 6]); TX('H100 80 GB', x + 8, y + h - 80 * U + 30, font(700, 24, HN), K.red);
    TX('Qwen3-1.7B 一步训练的峰值显存（GB）', x, y - 30, font(700, 26), K.ink2);
    SK.forEach((k, i) => TX(`${k} token`, x + gw * (i + .5), y + h + 40, font(600, 24, HN), K.ink2, 'center'));
    const ly = y - 52; R(x + w - 520, ly, 24, 24, K.ink2); TX('eager', x + w - 486, ly + 21, font(700, 24, HN), K.ink2); R(x + w - 380, ly, 24, 24, K.purple); TX('FLCE', x + w - 346, ly + 21, font(700, 24, HN), K.purple);
    HR(x + w - 250, ly, 24, 24, K.lg, K.paper2); TX('参数与优化器状态', x + w - 216, ly + 21, font(600, 22), K.ink2); }, .4, 0);
  const base = D.timeline.eager.start_gb;
  SEQS.forEach((sq, i) => [['eager', K.ink2, 0], ['flce', K.purple, 1]].forEach(([m, col, j]) => {
    const r = run(m, sq), bx = x + gw * i + gw * .18 + j * gw * .33, bw = gw * .3, t0 = b(0, .05 + i * .07 + j * .3);
    IN(t0, () => { if (r.oom) { TX('OOM', bx + bw / 2, y + h - 80 * U + 44, font(700, 30, HN), K.red, 'center'); return; }
      const p = P(t0, .6); R(bx, y + h - r.gb * U * p, bw, r.gb * U * p, col); HR(bx, y + h - base * U, bw, base * U, K.lg, col === K.purple ? '#d9ccfb' : K.paper2);
      TX(r.gb.toFixed(0), bx + bw / 2, y + h - r.gb * U * p - 10, font(700, 22, HN), col, 'center'); }, .3, 0); }));
  IN(b(0, .8), () => RICH(`16K token 每步：eager {g:${run('eager', 16384).s.toFixed(2)} s}，FLCE {p:${run('flce', 16384).s.toFixed(2)} s}`, 96, 950, font(700, 30, HN), K.ink), .4, 0);
}

// ---------- 7 同类做法 ----------
const B7 = beatTimes(7, [.5, .6, .6, .6]), S7_DUR = B7[3].e + 1.4;
function s7() {
  const b = (i, f) => at(B7, i, f), v1 = 1 - P(b(1), .5), v2 = win(b(1), b(3)), v3 = P(b(3), .5);
  A(1 - v3, () => head(0, 'Cut Cross Entropy：logits {t:不写入显存}')); A(v3, () => head(0, 'Liger 的 Hopper 实现：logits {p:只写一次}'));
  // 第一句：logits 分块只在 SRAM 里；backward 跳过可忽略的块（块的取舍为示意）
  A(v1, () => {
    const gx = 160, gy = 330, cw = 52, nC = 24, nR = 8;
    IN(b(0), () => { X.setLineDash([10, 8]); box(gx, gy, cw * nC, cw * nR, 3, K.lg); X.setLineDash([]); TX('logits：从不完整写入显存', gx, gy - 20, font(700, 28), K.gray); }, .4, 0);
    const k = Math.floor(clamp((T - b(0, .15)) / (B7[0].d * .8)) * nC * nR) % (nC * nR), kc = k % nC, kr = Math.floor(k / nC);
    IN(b(0, .15), () => { R(gx + kc * cw + 2, gy + kr * cw + 2, cw - 4, cw - 4, K.teal); box(gx + kc * cw - 4, gy + kr * cw - 4, cw + 8, cw + 8, 4, K.teal);
      TX('一次只算一小块，放在 SRAM 里累加 log-sum-exp', gx, gy + cw * nR + 50, font(700, 28), K.teal); }, .4, 0);
    IN(b(0, .6), () => { const ox = 1520; TX('backward', ox, gy - 20, font(700, 28, HN), K.red);
      for (let r = 0; r < nR; r++) for (let c = 0; c < 6; c++) { const keep = (r * 7 + c * 3) % 11 === 0 || c === r % 6 && r % 3 === 0; R(ox + c * 48, gy + r * cw, 44, cw - 6, keep ? K.red : K.paper2); }
      TX('灰色：softmax 太小，跳过', ox, gy + cw * nR + 50, font(700, 26), K.ink2); TX('（示意）', ox, gy + cw * nR + 88, font(600, 22), K.gray); }, .4, 0);
  });
  // 第二、三句：16384 token 的几种实现，显存与时间对照；两条批评
  A(v2, () => {
    const rows = [['eager', M('eager', 16384), K.ink2], ['Liger CE', M('liger_ce', 16384), K.red], ['FLCE（64 块）', M('flce', 16384), K.purple], ['FLCE（8 块）', D.sweep['8'], '#a988f8'], ['Cut Cross Entropy', M('cce', 16384), K.teal]];
    const y0 = 330, rh = 76, mx = 440, tx = 1180;
    TX('16384 个 token', 96, y0 - 30, font(700, 26), K.ink2); TX('峰值显存', mx, y0 - 30, font(700, 26), K.ink2); TX('时间', tx, y0 - 30, font(700, 26), K.ink2);
    rows.forEach(([n, d, c], i) => IN(b(1) + i * .12, () => { const y = y0 + i * rh; TX(n, mx - 24, y + 40, font(700, 28, HN), i === 4 ? K.teal : K.ink, 'right');
      R(mx, y + 8, Math.max(4, d.gb * 20), 44, c); TX(`${d.gb.toFixed(d.gb < 2 ? 2 : 1)} GB`, mx + Math.max(4, d.gb * 20) + 12, y + 42, font(700, 26, HN), K.ink);
      R(tx, y + 8, d.ms * 6, 44, c); TX(`${d.ms.toFixed(0)} ms`, tx + d.ms * 6 + 12, y + 42, font(700, 26, HN), K.ink); }, .3, 0));
    IN(b(2), () => { const y = 760; R(96, y, 820, 130, K.paper2); TX('块多省显存，块少才快', 120, y + 52, font(700, 32), K.ink); TX('上表 64 块与 8 块的差别', 120, y + 100, font(600, 26), K.ink2); }, .4, 0);
    IN(b(2, .45), () => { const y = 760; R(960, y, 864, 130, K.paper2); TX('loss 与梯度一起在 forward 里算', 984, y + 52, font(700, 32), K.ink); TX('对 loss 的任何变换都要写进 kernel', 984, y + 100, font(600, 26), K.ink2); }, .4, 0);
  });
  // 第四句：Liger 的 Hopper 实现（另一组形状：4096 token、hidden 4096、词表 128256）
  A(v3, () => {
    const S = D.sm90, y0 = 400, rh = 110, x0 = 560, U = 12;
    TX('4096 个 token · hidden size 4096 · 词表 128256 · H100', 96, 330, font(700, 28, HN), K.ink2);
    [['eager', S.eager, K.ink2], ['FLCE（Triton）', S.flce, K.purple], ['FLCE（Hopper 实现）', S.flce_sm90, K.purple], ['Cut Cross Entropy', S.cce, K.teal]].forEach(([n, d, c], i) => IN(b(3, .05) + i * .15, () => {
      const y = y0 + i * rh; TX(n, x0 - 24, y + 50, font(700, 30, HN), K.ink, 'right'); R(x0, y + 10, d.ms * U, 60, c); TX(`${d.ms.toFixed(1)} ms · ${d.gb.toFixed(1)} GB`, x0 + d.ms * U + 14, y + 52, font(700, 28, HN), K.ink); }, .3, 0));
    IN(b(3, .5), () => TX('Hopper 实现：logits 只写一次显存，原地变成梯度', 96, 900, font(700, 30), K.purple), .4, 0);
  });
}
// 音效：每句配音开头一声轻响（film.html 里按节拍生成）
const EV = {};
