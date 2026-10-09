// KV Cache 压缩影片的各幕。每幕的时间由配音节拍决定（beats.js / vo.js），b(i, f) 为第 i 句进行到 f 处的时刻。
// 全片一个配置：80 层、64 个 query head、head dim 128、BF16（与文章开头的表相同）。出处见 docs/attention-kv-compression-notes.md。
// 语义色：key blue，value yel，latent purple，RoPE key teal，query 与输出 ink，装不下与重复 red
const KL = { blue: '#cdd6f6', yel: '#fbeaa8', purple: '#ddd0fd', teal: '#bfe6df', red: '#f7c9c7' };
const lab = (s, x, y, size = 28, col = K.ink2, a = 'left', w = 700, f = ZH) => TX(s, x, y, font(w, size, f), col, a);
const src = (s) => TX(s, 1824, 76, font(500, 26), K.gray, 'right');
function badge(x, y, n) { dot(x, y, 22, K.ink); TX(String(n), x, y + 9, font(700, 26, HN), '#fff', 'center'); }
// 两行字的色块
function cell2(x, y, w, h, col, label, sub, tc = '#fff', size = 30) {
  R(x, y, w, h, col); TX(label, x + w / 2, y + h / 2 + (sub ? -4 : size * .36), font(700, size), tc, 'center');
  if (sub) TX(sub, x + w / 2, y + h / 2 + size * .95, font(500, size * .74), tc, 'center');
}
function hatchT(col) { const k = 'T' + col; if (PAT[k]) return PAT[k]; const g = 16, c = document.createElement('canvas'); c.width = c.height = g; const q = c.getContext('2d');
  q.strokeStyle = col; q.lineWidth = 5; q.beginPath(); for (let o = -g; o <= g * 2; o += g) { q.moveTo(o, g); q.lineTo(o + g, 0); } q.stroke(); return (PAT[k] = X.createPattern(c, 'repeat')); }
const HT = (x, y, w, h, col = K.red) => { X.fillStyle = hatchT(col); X.fillRect(x, y, w, h); };
function dash(x, y, w, h, col = K.ink, lw = 4) { X.setLineDash([14, 10]); box(x, y, w, h, lw, col); X.setLineDash([]); }
function vtext(s, x, y, f, col) { X.save(); X.translate(x, y); X.rotate(-Math.PI / 2); X.font = f; X.fillStyle = col; X.textAlign = 'center'; X.textBaseline = 'alphabetic'; X.fillText(s, 0, 0); X.restore(); }

// 缓存瓦片：一个 token 在一层里的缓存。宽 w 对应 key 128 维 + value 128 维，每个 KV head 一行，面积与元素数成正比
function tile(x, y, w, rows, rh, light) {
  const kc = light ? KL.blue : K.blue, vc = light ? KL.yel : K.yel, g = rh >= 5 ? 1 : 0;
  for (let i = 0; i < rows; i++) { R(x, y + i * rh, w / 2 - 1, rh - g, kc); R(x + w / 2 + 1, y + i * rh, w / 2 - 1, rh - g, vc); }
}
// 一条 128K token 请求的 KV cache：三根柱子对着同一条 80 GB 线，U 为每 GiB 的像素。g 为各柱的生长进度，lim 为 80 GB 线与超出部分斜线的透明度
function gbBars(x, base, U, which = 3, g = [1, 1, 1], lim = 1) {
  const bw = 150, gap = 110, items = [['MHA', 320], ['GQA-8', 40], ['MLA', 11.25]].slice(0, which);
  const x1 = x + items.length * (bw + gap) - gap + (which === 1 ? 250 : 30);
  items.forEach(([name, gb], i) => {
    if (g[i] <= .003) return;
    const bx = x + i * (bw + gap), h = gb * U * g[i];
    if (i < 2) { R(bx, base - h / 2, bw, h / 2, K.blue); R(bx, base - h, bw, h / 2, K.yel); }
    else { R(bx, base - h * 8 / 9, bw, h * 8 / 9, K.purple); R(bx, base - h, bw, h / 9, K.teal); }
    if (gb > 80 && h > 80 * U) A(lim, () => HT(bx, base - h, bw, h - 80 * U));
    const v = gb * g[i];
    A(Math.min(1, g[i] * 4), () => { TX(`${gb === 11.25 ? v.toFixed(1) : Math.round(v)} GiB`, bx + bw / 2, base - h - 18, font(700, 34, HN), gb > 80 && lim > .5 ? K.red : K.ink, 'center');
      TX(name, bx + bw / 2, base + 46, font(700, 32, HN), K.ink, 'center'); });
  });
  line(x - 30, base, x1, base, K.ink, 3);
  A(lim, () => { line(x - 30, base - 80 * U, x1, base - 80 * U, K.ink, 4, [14, 10]); TX('一张 H100：80 GB', x1, base - 80 * U - 14, font(700, 26), K.ink, 'right'); });
}
const EV = {};
const tail = (B) => B[B.length - 1].e + 1.3;

// ---------- 0 封面 ----------
const B0 = beatTimes(0, [.9, .6]), S0_DUR = tail(B0);
function s0() {
  const b = (i, f) => at(B0, i, f);
  IN(.1, () => TX('KV Cache 压缩', 96, 400, font(800, 128), K.ink), .6);
  IN(.35, () => RICH('{b:MQA}、{b:GQA} 与 {p:MLA}', 100, 520, font(700, 76), K.ink), .6);
  IN(.6, () => lab('一条 128K token 的请求，缓存要占多少显存', 100, 610, 36, K.gray, 'left', 500), .5);
  A(P(.3, .5), () => gbBars(1150, 900, 1.9, 3, [S(.5, 1.6), S(b(1, .55), .7), S(b(1, .8), .7)], P(1.8, .5)));
}

// ---------- 1 KV cache 从哪来 ----------
const B1 = beatTimes(1, [.6, .5, .5, .6]), S1_DUR = tail(B1);
function s1() {
  const b = (i, f) => at(B1, i, f);
  IN(b(3, .1), () => head(0, 'decode 每一步都要读所有历史 token 的 {b:key} 与 {y:value}'));
  const tx = (j) => 620 + j * 240, w = 190, wt = ['0.1', '0.6', '0.2', '0.1'];
  IN(b(0, .1), () => lab('历史 token', tx(0), 262, 26, K.gray), .4, 0);
  for (let j = 0; j < 4; j++) {
    IN(b(0, .1) + j * .12, () => cell(tx(j), 280, w, 60, K.paper2, `token ${j + 1}`, 26, K.ink2), .4);
    IN(b(1, .3) + j * .1, () => { arrow(tx(j) + w / 2, 344, tx(j) + w / 2, 382, K.gray, 3); cell(tx(j), 386, w, 70, K.blue, 'key', 30); }, .4);
    IN(b(1, .75) + j * .1, () => { line(tx(j) + w / 2, 456, tx(j) + w / 2, 500, K.ink, 3); cell(tx(j) + w / 2 - 50, 500, 100, 50, K.ink, wt[j], 28, '#fff', HN); }, .4, 0);
    IN(b(2, .05) + j * .1, () => cell(tx(j), 596, w, 70, K.yel, 'value', 30, K.ink), .4);
    IN(b(2, .4) + j * .08, () => line(tx(j) + w / 2, 550, tx(j) + w / 2, 596, K.ink, 3), .3, 0);
  }
  IN(b(0, .45), () => { lab('当前 token', 140, 262, 26, K.gray); cell(140, 280, 260, 60, K.paper2, 'token 5', 26, K.ink2); }, .4);
  IN(b(1, .05), () => { arrow(270, 344, 270, 382, K.gray, 3); cell(140, 386, 260, 70, K.ink, 'query', 30); }, .4);
  arrowP(408, 421, tx(0) - 10, 421, P(b(1, .5), .5), K.ink, 5); IN(b(1, .5), () => lab('点积', 505, 404, 26, K.ink2, 'center'), .4, 0);
  IN(b(1, .75), () => lab('softmax 后的权重', 580, 534, 26, K.ink2, 'right'), .4, 0);
  IN(b(2, .4), () => lab('按权重求和', 580, 640, 26, K.ink2, 'right'), .4, 0);
  arrowP(tx(3) + w + 10, 631, 1620, 631, P(b(2, .6), .5), K.ink, 5);
  IN(b(2, .7), () => cell(1630, 596, 190, 70, K.ink, '输出', 30), .4);
  IN(b(3, .0), () => RICH('{b:key}、{y:value} 只由历史 token 决定：算一次，存下来，以后每一步都读', 140, 790, font(700, 34), K.ink), .5);
  IN(b(3, .7), () => RICH('这份缓存就是 {p:KV cache}。query 与权重每一步重新计算，不缓存', 140, 850, font(700, 34), K.ink), .5);
}

// ---------- 2 有多大 ----------
const B2 = beatTimes(2, [.6, .5, .5, .5]), S2_DUR = tail(B2);
function s2() {
  const b = (i, f) => at(B2, i, f);
  IN(b(2, .7), () => head(0, '每个 head 各存一份：一条请求要 {r:320 GiB}'));
  const x = 230, y = 300, w = 512, rh = 8, rows = Math.max(1, Math.round(mix(1, 64, S(b(0, .5), 2.2))));
  IN(.3, () => { tile(x, y, w, rows, rh);
    lab('key · 128 维', x + w / 4, y - 18, 26, K.blue, 'center'); lab('value · 128 维', x + w * 3 / 4, y - 18, 26, K.yelD, 'center');
    box(x - 4, y - 3, w + 8, rh + 5, 3, K.ink); lab('← 刚才那个 head', x + w + 18, y + 12, 24, K.ink2); }, .5);
  IN(b(0, .5), () => vtext('64 个 head，每个一行', x - 26, y + 256, font(700, 26), K.ink2), .5, 0);
  IN(b(1, .3), () => lab('一个 token 在一层里的缓存', x, y + 512 + 44, 28, K.ink), .5, 0);
  const cx = 900, ts = [b(1, .5), b(1, .82), b(2, .05), b(2, .35)];
  [['2 × 64 × 128', '16384 个数', K.ink2], ['BF16，每个数 2 字节', '32 KiB', K.ink], ['× 80 层', '2560 KiB', K.ink], ['× 128K 个 token', '320 GiB', K.red]].forEach(([a, c, col], i) =>
    IN(ts[i], () => { lab(a, cx, 350 + i * 130, 28, K.gray, 'left', 500); TX(c, cx, 404 + i * 130, font(700, 52, HN), col); }, .5));
  A(P(b(2, .5), .4), () => gbBars(1420, 812, 1.6, 1, [S(b(2, .55), 1.4)], P(b(3, .1), .5)));
  IN(b(3, .55), () => lab('4 张才装得下', 1495, 906, 28, K.red, 'center'), .4, 0);
}

// ---------- 3 decode 的瓶颈 ----------
const B3 = beatTimes(3, [.6, .5, .5]), S3_DUR = tail(B3);
function s3() {
  const b = (i, f) => at(B3, i, f);
  IN(b(1, .1), () => head(0, '每读入一个数只做{r:一次乘加}：算力在等显存'));
  IN(b(1, .1), () => src('Zadouri et al., 2025'), .4, 0);
  IN(b(0, .1), () => cell2(96, 300, 420, 120, K.paper2, '显存里的 KV cache', null, K.ink, 32), .5);
  arrowP(526, 360, 880, 360, P(b(0, .5), .6), K.ink, 5); IN(b(0, .5), () => lab('读 1 个数 = 2 字节', 703, 340, 26, K.ink2, 'center'), .4, 0);
  IN(b(0, .75), () => cell2(890, 300, 460, 120, K.teal, '与 query 做 1 次乘加', '= 2 FLOPs', '#fff', 32), .5);
  IN(b(1, .1), () => RICH('→ {r:1 FLOP / 字节}', 1390, 374, font(700, 40), K.ink), .5, 0);
  const x0 = 520, Sx = 4.2;
  IN(b(1, .15), () => { lab('每读 1 字节', 96, 560, 28, K.gray, 'left', 500); line(x0, 600, x0, 870, K.ink, 3);
    lab('MHA decode', 96, 660, 38, K.ink); lab('实际做的运算', 96, 702, 26, K.gray, 'left', 500);
    R(x0, 630, 5, 70, K.red); lab('1 FLOP', x0 + 22, 678, 34, K.red, 'left', 700, HN); }, .5, 0);
  IN(b(1, .5), () => { lab('H100', 96, 800, 38, K.ink); lab('能做的运算', 96, 842, 26, K.gray, 'left', 500); }, .5, 0);
  const gw = 295 * Sx * S(b(1, .55), 1.6);
  if (gw > 1) { R(x0, 770, gw, 70, K.teal); if (gw > 300) lab(`${Math.round(gw / Sx)} FLOPs`, x0 + gw - 20, 818, 34, '#fff', 'right', 700, HN); }
  note(b(2, .35), 'decode 的耗时取决于读了多少字节：缓存越小，这一步越快。', 940);
}

// ---------- 4 MQA ----------
const B4 = beatTimes(4, [.6, .5, .5, .5]), S4_DUR = tail(B4);
function s4() {
  const b = (i, f) => at(B4, i, f);
  IN(b(1, .45), () => head(0, '所有 query head 读{b:同一组} key 与 value：16384 → {p:256} 个数'));
  IN(b(3, .1), () => src('Shazeer, 2019'), .4, 0);
  IN(b(1, .45), () => { tile(140, 300, 512, 64, 8, true); lab('MHA：64 组', 396, 860, 30, K.gray, 'center'); }, .6);
  const qx = (i) => 900 + i * 14, sx = 1092, sy = 640;
  IN(b(0, .2), () => lab('64 个 query head，一个不少', 900, 300, 28, K.ink2), .4, 0);
  for (let i = 0; i < 64; i++) {
    const lp = P(b(0, .7) + i * .02, .5); if (lp > 0) line(qx(i) + 5, 374, mix(qx(i) + 5, sx + 256 + (i - 31.5) * 6, lp), mix(374, sy - 6, lp), K.gray, 1.5);
    IN(b(0, .2) + i * .012, () => R(qx(i), 320, 10, 50, K.ink), .3, 8);
  }
  IN(b(0, .55), () => { tile(sx, sy, 512, 1, 8); lab('key · 128 维', sx + 128, sy + 52, 26, K.blue, 'center'); lab('value · 128 维', sx + 384, sy + 52, 26, K.yelD, 'center'); }, .5);
  IN(b(1, .05), () => RICH('{p:Multi-Query Attention（MQA）}：1 组，256 个数', 900, 780, font(700, 34), K.ink), .5);
  IN(b(2, .1), () => RICH('各 head 的 query 不同，权重不同；取到的内容来自同一个 value', 900, 836, font(500, 28), K.ink2), .5);
  arrowP(672, 556, 860, 620, P(b(1, .7), .5), K.ink, 5); IN(b(1, .7), () => lab('缩小到 1/64', 766, 550, 28, K.ink, 'center'), .4, 0);
  note(b(3, .4), '翻译模型的 decoder，每个 token 的耗时：MHA {r:46 µs}，MQA {t:3.8 µs}', 940);
}

// ---------- 5 GQA ----------
const B5 = beatTimes(5, [.6, .5, .5, .5]), S5_DUR = tail(B5);
function s5() {
  const b = (i, f) => at(B5, i, f);
  IN(b(1, .3), () => head(0, '分成 8 组，每组共享一份：{p:2048} 个数'));
  IN(b(2, .1), () => src('Ainslie et al., 2023；DeepSeek-V2 附录'), .4, 0);
  const gx = (k) => 110 + k * 96;
  IN(b(0, .2), () => lab('64 个 query head，每 8 个一组', 110, 300, 28, K.ink2), .4, 0);
  for (let k = 0; k < 8; k++) {
    IN(b(0, .2) + k * .08, () => { for (let i = 0; i < 8; i++) R(gx(k) + i * 10.5, 320, 8, 50, K.ink); }, .4, 8);
    const lp = P(b(0, .5) + k * .08, .5); if (lp > 0) for (let i = 0; i < 8; i++) line(gx(k) + i * 10.5 + 4, 372, mix(gx(k) + i * 10.5 + 4, gx(k) + 40, lp), mix(372, 494, lp), K.gray, 1.5);
    IN(b(0, .55) + k * .08, () => { R(gx(k), 500, 39, 60, K.blue); R(gx(k) + 41, 500, 39, 60, K.yel); }, .4);
  }
  IN(b(0, .8), () => lab('8 个 KV head：每个是一组 key 与 value', 110, 610, 28, K.ink2), .4, 0);
  IN(b(1, .05), () => RICH('{p:Grouped-Query Attention（GQA）}', 110, 720, font(700, 36), K.ink), .5);
  IN(b(1, .7), () => RICH('Llama-3.1-70B：64 个 query head、8 个 KV head', 110, 776, font(500, 28), K.ink2), .5);
  // 右：三种形式的缓存与 MMLU
  const x = 1010, rows = [['MHA', 64, '32 KiB', 45.2, .55], ['GQA-8', 8, '4 KiB', 41.2, .72], ['MQA', 1, '0.5 KiB', 37.9, .88]];
  IN(b(2, .05), () => lab('每 token 每层的缓存', x, 300, 26, K.gray, 'left', 500), .4, 0);
  IN(b(2, .4), () => lab('MMLU（DeepSeek 的三个 7B 模型）', x + 330, 300, 26, K.gray, 'left', 500), .4, 0);
  let y = 330;
  rows.forEach(([n, r, kib, m, f], i) => { const yy = y;
    IN(b(2, .05) + i * .2, () => { tile(x, yy, 160, r, 2.5); lab(n, x + 180, yy + 30, 32, K.ink, 'left', 700, HN); lab(kib, x + 180, yy + 64, 24, K.gray, 'left', 500, HN); }, .5);
    const gp = S(b(2, f), .8); if (gp > 0) { R(x + 330, yy, m * 9 * gp, 56, K.ink2); A(gp, () => lab((m * gp).toFixed(1), x + 330 + m * 9 * gp + 14, yy + 40, 32, K.ink, 'left', 700, HN)); }
    y += Math.max(r * 2.5, 70) + 40; });
  note(b(3, .1), '另一组 876M 模型的实验里，四种形式的下游准确率相差不到 1 个点', 940);
}

// ---------- 6 MLA：latent ----------
const B6 = beatTimes(6, [.6, .5, .5, .5, .5]), S6_DUR = tail(B6);
function s6() {
  const b = (i, f) => at(B6, i, f);
  IN(b(0, .5), () => head(0, '不共享，改存{p:中间结果}：512 维的 latent'));
  IN(b(0, .5), () => src('DeepSeek-V2, 2024'), .4, 0);
  const row = (y, name) => { lab(name, 96, y + 56, 40, K.ink, 'left', 700, HN); cell2(240, y, 210, 96, K.paper2, 'hidden state', null, K.ink2, 28); };
  // MHA
  IN(b(1, .0), () => row(300, 'MHA'), .5);
  arrowP(458, 348, 552, 348, P(b(1, .25), .4), K.ink, 4);
  IN(b(1, .3), () => cell2(560, 300, 250, 96, K.gray, '投影', '每个 head 一份'), .5);
  arrowP(818, 348, 932, 348, P(b(1, .5), .4), K.ink, 4);
  IN(b(1, .55), () => { cell(940, 300, 520, 46, K.blue, '64 个 key · 8192 维', 26); cell(940, 350, 520, 46, K.yel, '64 个 value · 8192 维', 26, K.ink); }, .5);
  IN(b(1, .8), () => { dash(930, 290, 540, 116); lab('缓存这一步', 1486, 358, 28, K.ink); }, .4, 0);
  // MLA
  IN(b(2, .0), () => row(580, 'MLA'), .5);
  arrowP(458, 628, 532, 628, P(b(2, .4), .4), K.ink, 4);
  IN(b(2, .45), () => cell2(540, 580, 230, 96, K.gray, '下投影', '所有 head 共用'), .5);
  arrowP(778, 628, 852, 628, P(b(2, .7), .4), K.ink, 4);
  IN(b(2, .75), () => cell2(860, 580, 150, 96, K.purple, 'latent', '512 维'), .5);
  arrowP(1028, 628, 1092, 628, P(b(3, .05), .4), K.ink, 4);
  IN(b(3, .1), () => cell2(1100, 580, 230, 96, K.gray, '上投影', '每个 head 一份'), .5);
  arrowP(1338, 628, 1402, 628, P(b(3, .55), .4), K.ink, 4);
  IN(b(3, .6), () => { cell(1410, 580, 410, 46, KL.blue, '64 个 key', 26, K.ink2); cell(1410, 630, 410, 46, KL.yel, '64 个 value', 26, K.ink2); }, .5);
  IN(b(4, .0), () => { dash(850, 570, 170, 116); lab('缓存这一步', 935, 724, 28, K.ink, 'center'); lab('用到时才还原，不缓存', 1615, 724, 28, K.gray, 'center'); }, .4, 0);
  IN(b(4, .2), () => RICH('64 个 head 的 key 与 value 共 16384 维，全部由这 {p:512 维}决定', 96, 850, font(700, 34), K.ink), .5);
  IN(b(4, .72), () => RICH('上投影矩阵是模型权重，对所有 token 相同，不占 KV cache', 96, 906, font(500, 28), K.ink2), .5);
}

// ---------- 7 矩阵吸收 ----------
const B7 = beatTimes(7, [.6, .5, .5, .5, .5]), S7_DUR = tail(B7);
function s7() {
  const b = (i, f) => at(B7, i, f);
  IN(b(2, .1), () => head(0, '要的是点积，不是 key 本身：上投影{t:挪到 query 一侧}'));
  const U = 1.5, pill = (x, y) => { cell(x, y, 110, 50, K.ink, '点积', 26); };
  // 展开
  IN(b(0, .05), () => { lab('展开', 96, 352, 40, K.ink); cell(420, 300, 512 * U, 80, K.purple, 'latent · 512 维（缓存）', 28); }, .5);
  arrowP(420 + 512 * U + 8, 340, 1380, 340, P(b(0, .3), .5), K.red, 5);
  IN(b(0, .3), () => { lab('× 上投影', 1284, 322, 26, K.red, 'center'); RICH('{r:每个历史 token 做一次}', 96, 400, font(700, 26), K.ink); }, .4, 0);
  IN(b(0, .45), () => cell(1390, 300, 128 * U, 80, K.blue, 'key · 128', 26), .5);
  IN(b(1, .05), () => { cell(1390, 430, 128 * U, 80, K.ink, 'query · 128', 26);
    line(1582, 470, 1640, 470, K.ink, 3); line(1640, 340, 1640, 470, K.ink, 3); line(1582, 340, 1640, 340, K.ink, 3); line(1640, 405, 1660, 405, K.ink, 3);
    pill(1660, 380); lab('分数', 1664, 474, 28, K.ink); }, .5, 0);
  IN(b(1, .55), () => RICH('query · (上投影 × latent) = (上投影ᵀ × query) · latent', 420, 880, font(700, 32, HN), K.ink2), .5);
  IN(b(2, .0), () => line(96, 560, 1824, 560, K.lg, 2), .4, 0);
  // 吸收
  IN(b(2, .0), () => { lab('吸收', 96, 662, 40, K.ink); cell(420, 610, 128 * U, 80, K.ink, 'query · 128', 26); }, .5);
  arrowP(420 + 128 * U + 8, 650, 742, 650, P(b(2, .1), .5), K.teal, 5);
  IN(b(2, .1), () => lab('× 上投影', 680, 632, 26, K.teal, 'center'), .4, 0);
  IN(b(2, .2), () => cell(752, 610, 512 * U, 80, K.ink, '乘过上投影的 query · 512 维', 28), .5);
  IN(b(2, .5), () => cell(752, 740, 512 * U, 80, K.purple, 'latent · 512 维（缓存），直接读', 28), .5);
  IN(b(2, .65), () => { line(1520, 650, 1640, 650, K.ink, 3); line(1520, 780, 1640, 780, K.ink, 3); line(1640, 650, 1640, 780, K.ink, 3); line(1640, 715, 1660, 715, K.ink, 3);
    pill(1660, 690); lab('同一个分数', 1664, 784, 28, K.ink); }, .5, 0);
  IN(b(2, .65), () => RICH('{t:每一步只做一次}', 96, 710, font(700, 26), K.ink), .4, 0);
  note(b(4, .1), 'value 一侧同样处理：先对 latent 加权求和，最后只乘一次上投影', 944);
}

// ---------- 8 Decoupled RoPE ----------
const B8 = beatTimes(8, [.6, .5, .5, .5]), S8_DUR = tail(B8);
function s8() {
  const b = (i, f) => at(B8, i, f);
  IN(b(2, .05), () => head(0, '位置编码挡在中间：另存 {t:64 维}带位置的 key'));
  IN(b(2, .05), () => src('DeepSeek-V2, 2024'), .4, 0);
  // 上：旋转夹在 query 与上投影之间
  const cw = 250, xs = [330, 620, 910, 1200];
  IN(b(0, .1), () => { lab('直接加 RoPE', 96, 342, 30, K.ink);
    cell(xs[0], 290, cw, 80, K.ink, 'query', 28); cell(xs[2], 290, cw, 80, K.gray, '上投影', 28); cell(xs[3], 290, cw, 80, K.purple, 'latent', 28);
    lab('×', xs[2] + cw + 20, 342, 34, K.gray, 'center'); }, .5);
  IN(b(0, .45), () => { cell2(xs[1], 290, cw, 80, K.red, 'RoPE 的旋转', '随两个 token 的距离变化', '#fff', 26);
    lab('×', xs[0] + cw + 20, 342, 34, K.gray, 'center'); lab('×', xs[1] + cw + 20, 342, 34, K.gray, 'center'); }, .5);
  IN(b(1, .25), () => RICH('{r:旋转夹在 query 与上投影之间}，上投影无法提前乘进 query', 330, 420, font(700, 28), K.ink), .5);
  IN(b(2, .0), () => line(96, 470, 1824, 470, K.lg, 2), .4, 0);
  // 下：两段对齐
  const U = 1.5, x0 = 330, xr = x0 + 512 * U + 20;
  IN(b(2, .0), () => lab('拆成两段', 96, 650, 30, K.ink), .4, 0);
  IN(b(2, .15), () => { lab('query 一侧，每个 head 一份', x0, 524, 26, K.gray, 'left', 500); cell(x0, 540, 512 * U, 76, K.ink, '乘过上投影的 query · 512 维，不带位置', 26);
    cell(x0, 720, 512 * U, 76, K.purple, 'latent · 512 维，不带位置', 26); lab('缓存一侧，所有 head 共享', x0, 834, 26, K.gray, 'left', 500); }, .5);
  IN(b(2, .28), () => { line(x0 + 384, 616, x0 + 384, 720, K.ink, 3); cell(x0 + 384 - 55, 644, 110, 48, K.ink, '点积', 26); }, .4, 0);
  IN(b(2, .4), () => { cell(xr, 720, 64 * U, 76, K.teal, '64', 26, '#fff', HN); RICH('{t:带 RoPE 的 key}', xr + 110, 768, font(700, 28), K.ink);
    cell(xr, 540, 64 * U, 76, K.ink, '64', 26, '#fff', HN); lab('带 RoPE 的 query', xr + 110, 588, 28, K.ink2); }, .5);
  IN(b(2, .8), () => { line(xr + 48, 616, xr + 48, 720, K.ink, 3); cell(xr + 48 - 55, 644, 110, 48, K.ink, '点积', 26); lab('+', (x0 + 384 + xr + 48) / 2, 680, 44, K.ink, 'center'); }, .4, 0);
  IN(b(3, .2), () => RICH('每 token 每层缓存：{p:512} + {t:64} = 576 个数', x0, 910, font(700, 36), K.ink), .5);
}

// ---------- 9 吸收后的形态 ----------
const B9 = beatTimes(9, [.6, .5, .5, .5]), S9_DUR = tail(B9);
function s9() {
  const b = (i, f) => at(B9, i, f);
  IN(b(0, .1), () => head(0, 'decode 时 64 个 head 读同一份 {p:576 维}'));
  IN(b(3, .1), () => src('DeepSeek-V2, 2024'), .4, 0);
  const w = 360, rh = w / 64, base = 720, a = w * w / 16384, xs = (i) => 110 + i * 440;
  const items = [['MHA', '16384 个数', '32 KiB', '320 GiB', b(1, .42)], ['GQA-8', '2048 个数', '4 KiB', '40 GiB', b(1, .62)], ['MLA', '576 个数', '1.1 KiB', '11.3 GiB', b(0, .1)], ['MQA', '256 个数', '0.5 KiB', '5 GiB', b(1, .88)]];
  IN(items[0][4], () => tile(xs(0), base - w, w, 64, rh), .6); IN(items[1][4], () => tile(xs(1), base - 8 * rh, w, 8, rh), .6); IN(items[3][4], () => tile(xs(3), base - rh, w, 1, rh), .6);
  IN(items[2][4], () => { R(xs(2), base - 512 * a / w, w, 512 * a / w, K.purple); R(xs(2), base - 512 * a / w - rh - 1, 64 * a / rh, rh, K.teal); }, .6);
  IN(b(0, .1), () => line(96, base + 2, 1824, base + 2, K.ink, 3), .4, 0);
  const t3 = [b(2, .35), b(2, .6), b(2, .8), b(2, .95)];
  items.forEach(([n, e, k, gbs, t0], i) => {
    IN(t0, () => { lab(n, xs(i), base + 56, 38, K.ink, 'left', 700, HN); lab(`${e} · ${k}`, xs(i), base + 100, 26, K.ink2, 'left', 500); }, .5, 0);
    IN(t3[i], () => RICH(`128K 请求：{${i === 0 ? 'r' : i === 2 ? 'p' : 'g'}:${gbs}}`, xs(i), base + 144, font(700, 28), K.ink2), .5, 0);
  });
  IN(b(1, .15), () => lab('每 token 每层的缓存，面积按元素数', 110, 290, 26, K.gray, 'left', 500), .4, 0);
  IN(b(0, .3), () => RICH('latent {p:512} + RoPE key {t:64}', xs(2), 560, font(700, 28), K.ink), .5);
  IN(b(0, .65), () => lab('就是只有一个 KV head 的 MQA', xs(2), 604, 26, K.ink2, 'left', 500), .5);
  IN(b(3, .35), () => RICH('约 250B 的模型上，MMLU：MLA {p:59.0}，MHA 57.5', xs(1) + 200, 380, font(700, 32), K.ink), .5);
}

// ---------- 10 用计算换读取 ----------
const B10 = beatTimes(10, [.6, .5, .5, .5]), S10_DUR = tail(B10);
function s10() {
  const b = (i, f) => at(B10, i, f);
  IN(b(0, .35), () => head(0, '读得少了，每个字节上的{t:运算}多了'));
  IN(b(0, .35), () => src('Zadouri et al., 2025；Kimi K2, 2025'), .4, 0);
  const x0 = 420, Sx = 4.2, rows = [['MHA', 1, K.red, .0], ['GQA-8', 8, K.blue, .2], ['MQA', 64, K.blue, .4], ['MLA', 128, K.purple, .58]];
  IN(b(0, .6), () => { lab('decode 时每读 1 字节做的运算（FLOPs）', x0, 290, 26, K.gray, 'left', 500); line(x0, 310, x0, 660, K.ink, 3); }, .4, 0);
  rows.forEach(([n, v, c, f], i) => { const y = 320 + i * 86, gp = S(b(1, f), .7);
    IN(b(1, f), () => lab(n, 96, y + 42, 36, K.ink, 'left', 700, HN), .4, 0);
    if (gp > 0) { R(x0, y, Math.max(5, v * Sx * gp), 60, c); A(gp, () => lab(String(Math.max(1, Math.round(v * gp))), x0 + v * Sx * gp + 16, y + 42, 32, c, 'left', 700, HN)); } });
  IN(b(1, .8), () => { line(x0 + 295 * Sx, 300, x0 + 295 * Sx, 660, K.teal, 4, [14, 10]); lab('H100 能做 295', x0 + 295 * Sx - 14, 344, 28, K.teal, 'right'); }, .4, 0);
  IN(b(2, .0), () => line(96, 700, 1824, 700, K.lg, 2), .4, 0);
  IN(b(2, .1), () => RICH('MLA 的每个 head 都要和 576 维做点积：{r:head 越多，算得越多}', 96, 764, font(700, 32), K.ink), .5);
  IN(b(3, .12), () => { lab('64 个 head', 96, 842, 30, K.ink); R(420, 808, 400, 50, K.purple); lab('推理的 FLOPs', 836, 842, 26, K.gray, 'left', 500); }, .5, 0);
  const g2 = S(b(3, .22), .9);
  IN(b(3, .2), () => { lab('128 个 head', 96, 912, 30, K.ink); R(420, 878, 400 + 332 * g2, 50, K.purple); if (g2 > .01) HT(820, 878, 332 * g2, 50, K.paper); }, .4, 0);
  IN(b(3, .38), () => lab('+83%', 420 + 732 + 16, 914, 32, K.red, 'left', 700, HN), .4, 0);
  IN(b(3, .5), () => RICH('validation loss 只降低 0.5%–1.2%', 1300, 914, font(700, 28), K.ink2), .4, 0);
}

// ---------- 11 TP 下的 latent 复制 ----------
const B11 = beatTimes(11, [.6, .5, .5, .5]), S11_DUR = tail(B11);
function s11() {
  const b = (i, f) => at(B11, i, f);
  IN(b(1, .45), () => head(0, 'latent 不能按 head 切：8 张卡{r:各存一份}'));
  IN(b(1, .45), () => src('Zadouri et al., 2025；SGLang v0.4'), .4, 0);
  const x0 = 470, cw = 150, ch = 124, gx = (i) => x0 + i * 170;
  const rowsY = [290, 500, 710], t0 = [b(0, .5), b(1, .05), b(2, .1)], tt = [b(0, .85), b(1, .75), b(2, .7)];
  const names = [['GQA-8', '按 head 分给 8 张卡', '{t:合计 1 份}'], ['MLA', '按 head 分给 8 张卡', '{r:合计 8 份}'], ['MLA', '按请求分给 8 张卡', '{t:合计 1 份}']];
  IN(b(0, .25), () => { for (let r = 0; r < 3; r++) for (let i = 0; i < 8; i++) R(gx(i), rowsY[r], cw, ch, K.paper2);
    for (let i = 0; i < 8; i++) lab(`GPU ${i + 1}`, gx(i) + cw / 2, 270, 22, K.gray, 'center', 500); }, .5, 0);
  rowsY.forEach((y, r) => {
    IN(t0[r], () => { lab(names[r][0], 96, y + 40, 38, K.ink, 'left', 700, HN); lab(names[r][1], 96, y + 80, 26, K.gray, 'left', 500); }, .5, 0);
    IN(tt[r], () => RICH(names[r][2], 96, y + 120, font(700, 28), K.ink), .4, 0);
    for (let i = 0; i < 8; i++) IN(t0[r] + .1 + i * .07, () => {
      const ih = 96, ix = gx(i) + 12, iw = cw - 24, by = y + ch - 14;
      if (r === 0) { R(ix, by - ih / 8, iw / 2 - 1, ih / 8, K.blue); R(ix + iw / 2 + 1, by - ih / 8, iw / 2 - 1, ih / 8, K.yel); lab(`head ${i + 1}`, gx(i) + cw / 2, y + 44, 24, K.ink2, 'center', 600); }
      if (r === 1) { R(ix, by - ih, iw, ih, K.purple); if (i) A(P(b(1, .6) + i * .05, .4), () => HT(ix, by - ih, iw, ih, K.red)); lab('完整', gx(i) + cw / 2, y + 72, 26, '#fff', 'center'); }
      if (r === 2) { R(ix, by - ih / 8, iw, ih / 8, K.purple); lab('1/8 的请求', gx(i) + cw / 2, y + 44, 24, K.ink2, 'center', 600); }
    }, .4, 10);
  });
  note(b(3, .1), 'SGLang 的 DP attention：8 张 H100 上 decode 吞吐为原来的 {t:1.9 倍}', 904);
}

// ---------- 12 DeepSeek-V4 ----------
const B12 = beatTimes(12, [1.6, .6, .5, .5]), S12_DUR = tail(B12);
function s12() {
  const b = (i, f) => at(B12, i, f);
  IN(b(1, .05), () => head(0, 'DeepSeek-V4：去掉上投影，直接训练{p:一个 512 维的 KV head}'));
  IN(b(1, .05), () => src('DeepSeek-V4, 2026'), .4, 0);
  const U = 1.3;
  IN(.4, () => { lab('MLA', 96, 352, 38, K.ink, 'left', 700, HN);
    cell(230, 300, 512 * U, 80, K.purple, 'latent · 512', 26); cell(230 + 512 * U + 6, 300, 64 * U, 80, K.teal, '64', 24, '#fff', HN);
    arrow(996, 340, 1052, 340, K.ink, 4); cell(1060, 300, 190, 80, K.lg, '上投影', 28);
    arrow(1258, 340, 1314, 340, K.ink, 4);
    cell(1322, 300, 150, 38, KL.blue, 'key', 24, K.ink2); cell(1322, 342, 150, 38, KL.yel, 'value', 24, K.ink2); }, .5);
  const c1 = P(b(0, .45), .3), c2 = P(b(0, .45) + .25, .3);
  if (c1 > 0) line(1050, 290, 1050 + 210 * c1, 290 + 100 * c1, K.red, 6); if (c2 > 0) line(1050, 390, 1050 + 210 * c2, 390 - 100 * c2, K.red, 6);
  IN(b(1, .0), () => { lab('V4', 96, 552, 38, K.ink, 'left', 700, HN); R(230, 500, 448 * U, 80, K.purple); R(230 + 448 * U, 500, 64 * U, 80, K.purple); }, .5);
  IN(b(1, .3), () => TX('一个 512 维向量：既是 key，也是 value', 230 + 224 * U, 550, font(700, 26), '#fff', 'center'), .4, 0);
  IN(b(1, .5), () => { R(230 + 448 * U, 500, 64 * U, 80, K.teal); lab('RoPE 作用在最后 64 维', 230 + 512 * U + 18, 550, 26, K.teal); }, .4, 0);
  IN(b(1, .68), () => RICH('{p:Shared-KV MQA}', 1322, 552, font(700, 34), K.ink), .5);
  // 沿序列方向压缩
  IN(b(2, .05), () => { lab('再沿序列方向压缩', 96, 720, 30, K.ink); lab('16 个 token', 1220, 722, 26, K.gray, 'left', 500); }, .4, 0);
  for (let i = 0; i < 16; i++) IN(b(2, .05) + i * .03, () => R(420 + i * 44 + Math.floor(i / 4) * 16, 690, 40, 44, K.paper2), .3, 8);
  for (let k = 0; k < 4; k++) { const cx = 420 + k * 192 + 86; arrowP(cx, 740, cx, 786, P(b(2, .4) + k * .1, .4), K.gray, 3); IN(b(2, .45) + k * .1, () => R(cx - 20, 792, 40, 44, K.purple), .4, 0); }
  IN(b(2, .6), () => lab('4 个 KV entry（另一些层每 128 个 token 合成 1 个）', 1220, 824, 26, K.gray, 'left', 500), .4, 0);
  note(b(3, .3), '1M token 的上下文下，V4 的 KV cache 约为 GQA-8 的 {p:2%}', 930);
}

// ---------- 13 各模型 ----------
const B13 = beatTimes(13, [.6, .5, .5]), S13_DUR = tail(B13);
function s13() {
  const b = (i, f) => at(B13, i, f);
  IN(.3, () => head(0, '现在的模型：每个 token 的 KV cache'));
  IN(.3, () => src('各模型的 config.json，BF16，只计 full attention 层'), .4, 0);
  const x0 = 520, Sx = 3.6;
  const rows = [['Llama-3.1-70B', 'GQA-8', 320, K.blue, b(0, .45)], ['Qwen3-235B-A22B', 'GQA-4', 188, K.blue, b(0, .55)], ['MiniMax-M3', 'GQA-4', 120, K.blue, b(0, .65)],
    ['GLM-5', 'MLA', 87.75, K.purple, b(1, .0)], ['DeepSeek-V3、Kimi-K2', 'MLA', 68.6, K.purple, b(1, .15)], ['Kimi-K3', '24 层 MLA', 27, K.purple, b(1, .5)], ['DeepSeek-V4-Pro', 'Shared-KV MQA', 7.7, K.teal, b(2, .0)]];
  IN(b(0, .3), () => line(x0, 270, x0, 890, K.ink, 3), .4, 0);
  rows.forEach(([n, f, v, c, t0], i) => { const y = 280 + i * 88, gp = S(t0, .8);
    IN(t0, () => { lab(n, 96, y + 30, 32, K.ink, 'left', 700, HN); lab(f, 96, y + 62, 24, K.gray, 'left', 500, HN); }, .4, 0);
    if (gp > 0) { R(x0, y, v * Sx * gp, 56, c); A(gp, () => lab(`${v >= 100 ? Math.round(v * gp) : (v * gp).toFixed(1)} KiB`, x0 + v * Sx * gp + 16, y + 40, 30, c, 'left', 700, HN)); } });
  IN(b(0, .45), () => RICH('{b:GQA}    {p:MLA}    {t:Shared-KV MQA}', 1300, 860, font(700, 28), K.ink2), .4, 0);
}

// ---------- 14 结尾 ----------
const B14 = beatTimes(14, [.6, .5, .5]), S14_DUR = tail(B14) + 1.2;
function s14() {
  const b = (i, f) => at(B14, i, f);
  IN(b(2, .2), () => head(0, '同一条 128K token 的请求：{r:320} → {b:40} → {p:11.3} GiB'));
  A(P(.3, .5), () => gbBars(1150, 880, 1.7, 3, [S(.4, 1.2), S(b(0, .75), .8), S(b(1, .45), .8)], P(1.2, .5)));
  const y = 380;
  IN(b(0, .35), () => RICH('{b:MQA、GQA}：多个 query head 共享 KV head', 96, y, font(700, 36), K.ink), .5);
  IN(b(1, .0), () => RICH('{p:MLA}：只缓存 latent，decode 时把上投影吸收进 query', 96, y + 110, font(700, 36), K.ink), .5);
  IN(b(1, .6), () => lab('代价是计算，以及 TP 下的重复', 96, y + 160, 28, K.gray, 'left', 500), .5);
  IN(b(2, .0), () => RICH('{t:Shared-KV MQA}：直接训练吸收后的形态', 96, y + 270, font(700, 36), K.ink), .5);
}
