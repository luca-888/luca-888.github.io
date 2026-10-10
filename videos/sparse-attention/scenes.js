// 稀疏 Attention 影片的十一幕。每幕一个 draw(u)，u 是幕内时间（秒）；节拍起点取自 script.js，音效事件登记进 EV。
// 贯穿全片的图形：128K 个格子铺成的一页（core.js 的 page），一格是一个 token 的 KV entry；每一幕看这一页上哪些被读、哪些留着、哪些已释放。
// 颜色与文章的自绘图一致：blue 被读取的未压缩 KV，purple 被读取的压缩 / 选中 entry，teal indexer key，灰褐色留在缓存但不读，浅底色已释放，red sink。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const EV = SCRIPT.map(() => []);                      // 每幕的音效：[幕内时间, 种类]
const ev = (k, u, kind) => EV[k].push([u, kind]);
const evRun = (k, a, b, step, kind) => { for (let u = a; u < b - 1e-6; u += step) ev(k, u, kind); };
const fade = (u, a, d = 0.45) => po(u, a, a + d);
const fmtN = (n) => (n >= 10000 ? `${Math.round(n / 1024)}K` : String(Math.round(n)));
const KEPT = '#b9b2a1', GHOST = '#e4ded0';
loadImg('sinkFig', '../../src/assets/sparse-attention/streamingllm-attention.svg');

// 页眉：左上角是片名，下面是这一幕的标签与标题，右上角是出处
function header(k, u) {
  const s = SCRIPT[k], a = fade(u, 0, 0.5);
  txt('稀疏 Attention', 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 76, { size: 26, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 226 + (1 - a) * 16, 56, 700, a);
}
const cap = (s, alpha = 1, x = PAGE.x, y = PAGE.y - 14) => rich(s, x, y, 24, 500, alpha, C.gray);
// 右下角正在生成的 token
function query(alpha = 1, label = '') {
  rr(PR + 8, PB - 20, 20, 20, C.yel, alpha); sr(PR + 8, PB - 20, 20, 20, C.ink, 2, alpha);
  if (label) txt(label, MX, PB - 2, { size: 24, weight: 700, color: C.yelD, alpha });
}
// 从页的底边往下引到底部一行的标注
const below = (px, s, alpha = 1, color = C.blue) => callout(px, PB - 1, px, 984, s, { size: 26, color, alpha });
const note = (s, y, o = {}) => txt(s, MX, y, { size: 22, weight: 500, color: C.ink2, ...o });
const kept = (o) => page('kept', () => KEPT, o), ghost = (o) => page('ghost', () => GHOST, o), full = (o) => page('full', () => C.blue, o);
const SEL = pick(2048, N, 7);                                  // DSA 选中的 2048 个 token（位置为示意）
const SEL_CSA = pick(1024, 32768, 11);                         // CSA 选中的 1024 个压缩 entry（位置为示意）
const dsa = (o) => page('dsa', (i) => (SEL.has(i) ? C.purple : KEPT), o);

// ================= 1 开场 =================
// 大标题；out 从 0 到 1 时缩成左上角的片名
function bigTitle(out, alpha = 1) {
  const size = mix(132, 28, out), w1 = tw('稀疏', 132, 800) + 36, w2 = tw('稀疏 ', 28, 700);
  txt('稀疏', mix(90, 96, out), mix(196, 76, out), { size, weight: mix(800, 700, out), color: out > 0.5 ? C.ink2 : C.ink, alpha });
  txt('Attention', mix(90 + w1, 96 + w2, out), mix(196, 76, out), { size, weight: mix(800, 700, out), color: mixc(C.blue, C.ink2, out), alpha });
}
const S1 = { sweeps: [[bt(0, 1) + 0.5, bt(0, 1) + 2.4], [bt(0, 1) + 2.9, bt(0, 1) + 3.9], [bt(0, 1) + 4.3, bt(0, 1) + 4.9]], out: [SCRIPT[0].dur - 1.5, SCRIPT[0].dur - 0.3] };
ev(0, 0.2, 'land'); evRun(0, 0.9, 3.0, 0.14, 'tick'); ev(0, bt(0, 1), 'pop'); S1.sweeps.forEach(([a, b]) => { evRun(0, a, b, 0.1, 'tick'); ev(0, b, 'click'); }); ev(0, S1.out[0], 'whoosh');
function s1(u) {
  const out = p(u, ...S1.out), keep = 1 - p(u, S1.out[0] - 0.2, S1.out[0] + 0.5), b1 = bt(0, 1);
  bigTitle(out, fade(u, 0.15, 0.8));
  txt('每生成一个 token，要把前面的全部读一遍', 96, 264, { size: 38, weight: 700, color: C.ink2, alpha: fade(u, 0.9) * keep });
  kept({ r1: ROWS * p(u, 0.8, 3.2) });
  // 三次扫读：一次比一次快，扫完的蓝色在下一次开始前褪去，最后一次留住
  let si = 0; while (si < 2 && u >= S1.sweeps[si + 1][0] - 0.25) si++;
  const [a, b] = S1.sweeps[si], prog = lin(u, a, b), hold = si === 2 ? 1 : 1 - lin(u, S1.sweeps[si + 1][0] - 0.25, S1.sweeps[si + 1][0]);
  full({ r1: ROWS * prog, alpha: hold });
  query(fade(u, b1, 0.3), 'query：正在生成的 token');
  rr(MX, 318, 22, 22, KEPT, fade(u, 1.4)); txt('一格 = 一个 token 的 KV', MX + 34, 338, { size: 26, weight: 700, alpha: fade(u, 1.4) });
  note('512 列 × 256 行 = 128K 个 token', 380, { alpha: fade(u, 2.2) }); note('按顺序从左上排到右下', 410, { alpha: fade(u, 2.2) });
  const ra = fade(u, b1 + 0.2);
  txt(`第 ${si + 1} 步`, MX, 470, { size: 24, weight: 700, color: C.yelD, alpha: ra });
  meter(530, '每步读取', [[N * prog * hold, C.blue]], `${fmtN(N * prog)} 个`, { alpha: ra });
  meter(650, '缓存', [[N, KEPT]], '128K 个', { alpha: fade(u, 3.2) });
  txt('每一步，都把这一页从头读到尾', 96, 1000, { size: 30, weight: 700, color: C.ink2, alpha: fade(u, bt(0, 2)) * keep });
}

// ================= 2 读回来的东西 =================
const HOT = [[58, 120, 180], [100, 300, 90], [131, 300, 200], [190, 200, 140], [215, 40, 110]];   // 行、列、长度：示意的几处高权重
const WT = (() => {
  const w = new Float32Array(N), sp = pick(600, N, 3);
  for (const i of sp) w[i] = 0.35;
  for (const [r, c, len] of HOT) for (let j = 0; j < len; j++) w[r * COLS + c + j] = 0.7 + 0.3 * Math.sin(j * 0.7);
  for (let i = N - 5000; i < N; i++) w[i] = Math.max(w[i], Math.exp(-(N - 1 - i) / 900));
  w[0] = 1; return w;
})();
const weights = (o) => page('weights', (i) => mixc(GHOST, C.blue, Math.round(WT[i] * 12) / 12), o);
ev(1, 0.5, 'whoosh'); [0, 1.3, 2.6].forEach((d) => ev(1, bt(1, 1) + d, 'pop')); ev(1, bt(1, 2), 'click');
function s2(u) {
  header(1, u);
  const b1 = bt(1, 1);
  weights(); full({ alpha: 1 - p(u, 0.6, 3.6) }); query();
  callout(PAGE.x + 1, PAGE.y + 1, PAGE.x + 44, PAGE.y - 22, '第一个 token', { size: 24, color: C.blue, alpha: fade(u, b1) });
  callout(PAGE.x + 300 * 2.5, PAGE.y + 58 * 2.5 + 1, MX - 8, PAGE.y + 58 * 2.5 + 1, '零星几处', { size: 24, color: C.blue, alpha: fade(u, b1 + 1.3) });
  below(PR - 320, '最近的一段', fade(u, b1 + 2.6));
  const la = fade(u, 1.2);
  txt('一个 head 给每个 token 的权重', MX, 338, { size: 26, weight: 700, alpha: la });
  for (let j = 0; j < 12; j++) rr(MX + j * 32, 356, 32, 22, mixc(GHOST, C.blue, j / 11), la);
  note('接近 0', 404, { color: C.gray, alpha: la }); txt('大', MX + MW, 404, { size: 22, weight: 500, color: C.gray, align: 'right', alpha: la });
  note('示意图：按论文里的常见形态画', 620, { color: C.gray, alpha: la });
  txt('哪几个有用，读之前并不知道', 96, 1000, { size: 30, weight: 700, color: C.ink2, alpha: fade(u, bt(1, 2)) });
}

// ================= 3 按位置选 =================
const S3 = { clear: [bt(2, 0) + 0.8, bt(2, 0) + 4.0], rel: [bt(2, 1) + 0.3, bt(2, 1) + 2.4] };
evRun(2, S3.clear[0], S3.clear[1], 0.12, 'tick'); ev(2, S3.clear[1], 'land'); ev(2, S3.rel[0], 'whoosh'); ev(2, S3.rel[1], 'click'); ev(2, bt(2, 2), 'pop');
function s3(u) {
  header(2, u);
  const pre = p(u, 0, 0.7), clear = p(u, ...S3.clear), rel = p(u, ...S3.rel), r = 254 * clear, b2 = bt(2, 2);
  weights({ alpha: 1 - pre });
  kept({ r1: r, alpha: pre * (1 - rel) }); ghost({ r1: r, alpha: rel }); full({ r0: r, alpha: pre });
  query();
  meter(330, '每步读取', [[mix(N, 1024, clear), C.blue]], `${fmtN(mix(N, 1024, clear))} 个`, { alpha: pre });
  meter(450, '缓存', [[mix(N, 1024, rel), KEPT]], `${fmtN(mix(N, 1024, rel))} 个`, { alpha: pre });
  txt('整条 = 128K 个', MX + MW, 536, { size: 22, weight: 500, color: C.gray, align: 'right', alpha: pre });
  below(PR - 320, '最近 1024 个 = 最后两行', fade(u, S3.clear[1]));
  const ba = fade(u, S3.rel[1] - 0.4);
  line(PR + 24, PAGE.y, PR + 24, PB - 30, C.gray, 2, ba); line(PR + 12, PAGE.y, PR + 24, PAGE.y, C.gray, 2, ba); line(PR + 12, PB - 30, PR + 24, PB - 30, C.gray, 2, ba);
  callout(PR + 24, 770, MX - 10, 770, '窗口之外：已释放', { size: 24, color: C.ink2, alpha: ba }); txt('这一层读不到', MX, 812, { size: 24, weight: 700, color: C.red, alpha: fade(u, b2 + 0.8) });
  txt('sliding window attention', MX, 628, { size: 32, weight: 700, color: C.blue, alpha: fade(u, b2) });
  note('只读最近 W 个 token，W 称为窗口', 668, { alpha: fade(u, b2 + 0.3) });
}

// ================= 4 远处的内容 =================
const S4 = { y0: 300, pitch: 11.4, sq: [bt(3, 0), bt(3, 0) + 1.6], row: (i) => bt(3, 0) + 1.7 + i * 0.03, glob: (n) => bt(3, 1) + 0.2 + n * 0.3 };
ev(3, S4.sq[0], 'whoosh'); evRun(3, S4.row(0), S4.row(61), 0.09, 'tick'); for (let n = 0; n < 10; n++) ev(3, S4.glob(n), 'click'); ev(3, bt(3, 2) + 0.3, 'land'); ev(3, bt(3, 2) + 1.4, 'pop');
function s4(u) {
  header(3, u);
  const sq = p(u, ...S4.sq), stub = PW * 1024 / N, b2 = bt(3, 2);
  cap('62 层，自上而下 · 一条 = 一层的 128K 个 token（刚才的一页压扁）', fade(u, S4.sq[1]), PAGE.x, 286);
  if (sq < 1) { const h = mix(PH, 8, sq), y = mix(PAGE.y, S4.y0, sq); ghost({ y, h, r1: 254 }); full({ y, h, r0: 254 }); }
  for (let i = 0; i < 62; i++) {
    const y = S4.y0 + i * S4.pitch, isG = i % 6 === 5, a = i === 0 ? (sq >= 1 ? 1 : 0) : fade(u, S4.row(i), 0.25), gk = isG ? po(u, S4.glob((i - 5) / 6), S4.glob((i - 5) / 6) + 0.6) : 0, bw = mix(stub, PW, gk);
    rr(PAGE.x, y, PW, 8, GHOST, a); rr(PR - bw, y, bw, 8, C.blue, a);
    if (isG) txt('global', PR + 12, y + 9, { size: 20, weight: 700, color: C.blue, alpha: gk });
  }
  const x = 1480, w = 344, g1 = p(u, b2 + 0.3, b2 + 1.2), g2 = p(u, b2 + 1.4, b2 + 2.2);
  txt('128K 上下文的 KV cache', x, 330, { size: 26, weight: 700, alpha: fade(u, b2) });
  meter(400, '62 层都读全部', [[N * g1, C.blue]], '100%', { x, w, alpha: fade(u, b2 + 0.2) });
  meter(520, '5 local : 1 global', [[N * 0.168 * g2, C.blue]], '17%', { x, w, note: '10 × 128K + 52 × 1024 个 entry', alpha: fade(u, b2 + 1.3) });
  const la = fade(u, S4.row(61)), ga = fade(u, bt(3, 1) + 1.0);
  txt('local 层：窗口 1024', x, 690, { size: 24, weight: 700, color: C.ink2, alpha: la }); txt('只剩右端一小段', x, 724, { size: 22, weight: 500, color: C.ink2, alpha: la });
  txt('global 层：读全部', x, 784, { size: 24, weight: 700, color: C.ink2, alpha: ga }); txt('远处的内容从这里进来', x, 818, { size: 22, weight: 500, color: C.ink2, alpha: ga });
  ['Gemma 3 的消融：5 : 1 与 1 : 1', '的 perplexity 相差不到 0.03'].forEach((s, j) => txt(s, x, 900 + j * 32, { size: 22, weight: 500, color: C.gray, alpha: fade(u, b2 + 2.6) }));
}

// ================= 5、6 attention sink =================
// 一条权重条：segs 为 [占比, 颜色, 条内文字]，合计为 1；grow 只画出左边的一部分
function wbar(x, y, w, h, segs, { size = 26, grow = 1, alpha = 1 } = {}) {
  let at = x;
  for (const [f, col, label] of segs) {
    const bw = w * f, vis = clamp((x + w * grow - at) / bw);
    if (bw > 3) rr(at, y, (bw - 3) * vis, h, col, alpha);
    if (label && bw > tw(label, size, 700) + 16) txt(label, at + bw / 2, y + h / 2 + size * 0.36, { size, weight: 700, color: '#fff', align: 'center', alpha: alpha * (vis >= 1 ? 1 : 0) });
    at += bw;
  }
}
const B4 = (f) => [0, 1, 2, 3].map(() => [f / 4, C.blue, '']);
// 原图里第一个 token 的那一列（画面坐标）：Layer 2、9、16、23、31
const SINK_COLS = [[992, 311, 27, 358], [1417, 312, 15, 164], [1632, 312, 15, 164], [1417, 511, 15, 159], [1632, 511, 15, 159]];
const S5 = { box: (j) => bt(4, 1) + j * 0.35, grow: [bt(4, 1) + 2.0, bt(4, 1) + 3.6] };
ev(4, bt(4, 0), 'whoosh'); SINK_COLS.forEach((_, j) => ev(4, S5.box(j), 'click')); evRun(4, S5.grow[0], S5.grow[1], 0.12, 'tick'); ev(4, S5.grow[1], 'land'); ev(4, bt(4, 2), 'pop');
function s5(u) {
  header(4, u);
  const im = IMG.sinkFig, iw = 1728, ih = im.naturalWidth ? iw * im.naturalHeight / im.naturalWidth : 392, fa = fade(u, bt(4, 0), 0.7), b2 = bt(4, 2);
  rr(96, 290, iw, ih, '#fff', fa); if (im.naturalWidth) { g.globalAlpha = fa; g.drawImage(im, 96, 290, iw, ih); g.globalAlpha = 1; }
  SINK_COLS.forEach(([x, y, w, h], j) => sr(x, y, w, h, C.ink, 4, fade(u, S5.box(j), 0.25)));
  txt('框出的最左一列是第一个 token；颜色越红，attention 越大', 96, 724, { size: 24, weight: 500, color: C.gray, alpha: fade(u, S5.box(0)) });
  const y = 820, grow = p(u, ...S5.grow);
  txt('一个 head 的权重（示意）', 96, y - 18, { size: 24, weight: 500, color: C.gray, alpha: fade(u, S5.grow[0]) });
  wbar(96, y, PW, 64, [[0.8, C.red, '第一个 token'], ...B4(0.2)], { grow });
  const ba = fade(u, b2);
  line(96, y + 80, PR, y + 80, C.ink, 2, ba); line(96, y + 72, 96, y + 88, C.ink, 2, ba); line(PR, y + 72, PR, y + 88, C.ink, 2, ba);
  rr(PAGE.x + PW / 2 - 70, y + 66, 140, 30, C.paper, ba); txt('合计 = 1', PAGE.x + PW / 2, y + 90, { size: 26, weight: 700, align: 'center', alpha: ba });
  txt('它的 value 接近 0：权重放在这里，等于什么都没读', 96, y + 140, { size: 26, weight: 700, color: C.red, alpha: fade(u, b2 + 1.2) });
  txt('真正读的', PR, y + 140, { size: 26, weight: 700, color: C.blue, align: 'right', alpha: fade(u, S5.grow[1]) });
  txt('像一张必须投完的选票：', MX, y + 106, { size: 24, weight: 500, color: C.ink2, alpha: fade(u, b2 + 2.2) });
  txt('没有想选的，就投弃权', MX, y + 140, { size: 24, weight: 500, color: C.ink2, alpha: fade(u, b2 + 2.2) });
  txt('attention sink', MX, y + 6, { size: 40, weight: 700, color: C.red, alpha: fade(u, b2 + 4.2) });
  txt('接住多余权重的位置', MX, y + 46, { size: 24, weight: 500, color: C.ink2, alpha: fade(u, b2 + 4.2) });
}
const S6 = { rel: [bt(5, 0) + 0.8, bt(5, 0) + 2.8], sq: [bt(5, 1) + 0.4, bt(5, 1) + 2.4] };
ev(5, S6.rel[0], 'whoosh'); ev(5, S6.rel[1], 'click'); evRun(5, S6.sq[0], S6.sq[1], 0.14, 'tick'); ev(5, S6.sq[1] + 0.2, 'land'); ev(5, bt(5, 2) + 0.2, 'pop');
function s6(u) {
  header(5, u);
  const rel = p(u, ...S6.rel), f = p(u, ...S6.sq), b1 = bt(5, 1), b2 = bt(5, 2);
  kept({ r1: 254, alpha: 1 - rel }); ghost({ r1: 254, alpha: rel }); full({ r0: 254 }); query();
  rr(PAGE.x - 4, PAGE.y - 4, 14, 14, C.red, 1 - rel); sr(PAGE.x - 4, PAGE.y - 4, 14, 14, C.red, 3);
  callout(PAGE.x + 10, PAGE.y + 10, PAGE.x + 70, PAGE.y + 56, '第一个 token', { size: 26, color: C.red, alpha: fade(u, 0.3) });
  txt('：已释放', PAGE.x + 80 + tw('第一个 token', 26, 700), PAGE.y + 56 + 9, { size: 26, weight: 700, color: C.red, alpha: fade(u, S6.rel[1] - 0.3) });
  below(PR - 320, '窗口：最近的 token', fade(u, 0.5));
  const row = (y, a, label, segs, value, vcol, va, n1) => {
    txt(label, MX, y, { size: 26, weight: 700, alpha: a }); wbar(MX, y + 16, MW, 48, segs, { size: 22, alpha: a });
    txt(value, MX, y + 100, { size: 26, weight: 700, color: vcol, alpha: va }); note(n1, y + 134, { color: C.gray, alpha: va });
  };
  row(330, fade(u, 0.4), '另外留住开头 4 个 token', [[0.8, C.red, '开头的 token'], ...B4(0.2)], 'perplexity 5.40', C.ink, fade(u, 0.9), '多余的权重有地方放');
  const red = 0.8 * (1 - f);
  row(530, fade(u, b1), '只留最近的 token', [[red, C.red, ''], ...B4(1 - red)], 'perplexity 5158', C.red, fade(u, S6.sq[1] + 0.2), '窗口里的合计：0.20 变为 1');
  row(730, fade(u, b2), 'sink logit：每个 head 一个数', [[0.8, '#f08c88', 'sink logit'], ...B4(0.2)], 'gpt-oss、DeepSeek-V4', C.ink, fade(u, b2 + 1.0), '不在缓存里，窗口滑不走');
}

// ================= 7 按内容选 =================
const S7 = { slab: [bt(6, 0) + 3.4, bt(6, 0) + 4.6], scan: [bt(6, 1) + 0.3, bt(6, 1) + 2.5], sel: [bt(6, 1) + 2.5, bt(6, 1) + 3.7] };
ev(6, S7.slab[0], 'whoosh'); ev(6, S7.slab[1], 'land'); evRun(6, S7.scan[0], S7.scan[1], 0.1, 'tick'); ev(6, S7.sel[0] + 0.3, 'pop'); ev(6, bt(6, 2) + 0.2, 'click'); ev(6, bt(6, 2) + 1.6, 'click');
function s7(u) {
  header(6, u);
  const pre = p(u, 0.1, 1.1), sl = p(u, ...S7.slab), scan = lin(u, ...S7.scan), sel = p(u, ...S7.sel), b2 = bt(6, 2);
  ghost({ r1: 254, alpha: 1 - pre }); full({ r0: 254, alpha: 1 - pre }); kept({ alpha: pre }); dsa({ alpha: sel }); query();
  cap('128K 个 entry，每个 576 维', pre * (1 - sel)); cap('128K 个 entry，每个 576 维 · {p:紫色：选中的 2048 个，2048 / 128K = 1/64}（位置为示意）', sel);
  const sy = PB + 16 + (1 - sl) * 120, sh = 84;
  rr(PAGE.x, sy, PW, sh, C.teal, sl);
  if (scan > 0 && scan < 1) { rr(PAGE.x, sy, PW * scan, sh, '#3fc0ae', 0.55); rr(PAGE.x + PW * scan - 6, sy, 6, sh, '#fff', 0.9); }
  txt('indexer key：每个 token 另存一个，128 维', PAGE.x + 24, sy + 38, { size: 26, weight: 700, color: '#fff', alpha: sl });
  txt('每一步全部扫一遍，打分，取分数最高的 2048 个', PAGE.x + 24, sy + 70, { size: 22, weight: 500, color: '#fff', alpha: fade(u, S7.scan[0]) });
  arrow(PR + 46, sy + sh / 2, PR + 46, PB - 60, C.tealD, sel); line(PR, sy + sh / 2, PR + 46, sy + sh / 2, C.tealD, 4, sel);
  const n = mix(N, 2048, sel), ta = fade(u, b2);
  meter(330, '每步读取', [[n, mixc(C.blue, C.purple, sel)], ...(ta > 0 ? [[N * ta, C.teal, 128 / 576]] : [])], `${fmtN(n)} 个`, { alpha: pre, note: ta > 0.5 ? '另扫一遍 128K 个小 key' : '' });
  const ca = fade(u, b2 + 1.0);
  meter(480, '缓存', [[N, KEPT], ...(ca > 0 ? [[N * ca, C.teal, 128 / 576]] : [])], '128K 个', { alpha: pre, note: ca > 0.5 ? '另存 128K 个小 key，比原来大' : '' });
  const ka = fade(u, b2 + 2.2);
  rr(MX, 640, MW, 160, '#1d1d1f', ka);
  txt('DeepSeek-V3.2 · config.json', MX + 20, 674, { size: 18, weight: 500, color: '#9a968c', font: MONO, alpha: ka });
  ['"index_head_dim": 128,', '"index_n_heads": 64,', '"index_topk": 2048,'].forEach((s, j) => txt(s, MX + 20, 712 + j * 32, { size: 22, weight: 500, color: '#f1ede4', font: MONO, alpha: ka }));
  const na = fade(u, S7.sel[1]);
  txt('DeepSeek Sparse Attention', MX, 852, { size: 28, weight: 700, color: C.purple, alpha: na });
  note('简称 DSA，打分的模块叫 indexer', 886, { alpha: na });
  note('继续训练后 MMLU-Pro 85.0 对 85.0', 918, { color: C.gray, alpha: fade(u, b2 + 3.2) });
}

// ================= 8 存储也要降 =================
// 压缩后的一页：256 × 128 = 32K 个 entry，下面右端是最近 128 个未压缩的 token
const CSA = { x: PR - 640, y: PB - 328 };
function csaPage(x, y, sel = 1, win = 1, w = 640, h = 320) {
  page('csa0', () => KEPT, { x, y, cols: 256, rows: 128, w, h });
  page('csa', (i) => (SEL_CSA.has(i) ? C.purple : KEPT), { x, y, cols: 256, rows: 128, w, h, alpha: sel });
  page('w128', () => C.blue, { x: x + 640 - 320, y: y + 320 + 4, cols: 128, rows: 1, alpha: win });
}
const S8 = { shrink: [bt(7, 0) + 2.8, bt(7, 0) + 5.0], sel: [bt(7, 1) + 0.2, bt(7, 1) + 1.4], win: bt(7, 1) + 1.9 };
ev(7, bt(7, 0) + 0.4, 'pop'); ev(7, bt(7, 0) + 1.6, 'click'); ev(7, bt(7, 0) + 1.9, 'click'); ev(7, S8.shrink[0], 'whoosh'); ev(7, S8.shrink[1], 'land'); ev(7, S8.sel[0] + 0.3, 'pop'); ev(7, S8.win, 'click'); ev(7, bt(7, 2), 'pop');
function s8(u) {
  header(7, u);
  const pre = p(u, 0, 0.7), s = p(u, ...S8.shrink), sel = p(u, ...S8.sel), win = fade(u, S8.win), b0 = bt(7, 0);
  ghost({ alpha: s }); sr(PAGE.x - 2, PAGE.y - 2, PW + 4, PH + 4, C.gray, 2, s, [10, 8]);
  const x = mix(PAGE.x, CSA.x, s), y = mix(PAGE.y, CSA.y, s), w = mix(PW, 640, s), h = mix(PH, 320, s);
  if (s > 0) rr(x - 10, y - 12, w + 10, h + 12 + (PB - y - h), C.paper);
  if (s < 1) { kept({ x, y, w, h }); dsa({ x, y, w, h, alpha: 1 - pre }); } else csaPage(CSA.x, CSA.y, sel, win);
  cap('虚线：原来 128K 个 token 的位置，不再缓存 · 右下：32K 个压缩 entry，{p:紫色是选中的 1024 个}（位置为示意）', sel);
  // 放大：4 个 token 合成 1 个 entry
  const ix = 160, iy = 356, ia = fade(u, b0 + 0.4), ea = fade(u, b0 + 1.6);
  rr(ix - 24, iy - 26, 500, 232, C.paper, ia); sr(ix - 24, iy - 26, 500, 232, C.ink, 2, ia);
  txt('放大：每 4 个 token → 1 个 entry', ix, iy + 14, { size: 26, weight: 700, alpha: ia });
  for (let j = 0; j < 8; j++) rr(ix + j * 56 + (j >= 4 ? 12 : 0), iy + 44, 48, 48, KEPT, ia);
  [0, 1].forEach((e) => { const ex = ix + e * 236 + 108, a = fade(u, b0 + 1.6 + e * 0.3); arrow(ex, iy + 98, ex, iy + 128, C.ink, a, 3); rr(ex - 24, iy + 134, 48, 48, '#8f8872', a); txt(`entry ${e}`, ex + 36, iy + 168, { size: 22, weight: 500, color: C.gray, alpha: a }); });
  line(ix + 476, iy + 206, CSA.x, CSA.y, C.ink, 2, s >= 1 ? ea : 0, [6, 6]);
  query();
  below(PR - 160, '最近 128 个 token：未压缩，直接读', fade(u, S8.win + 0.3));
  meter(330, '每步读取', [[mix(2048, 1024, sel), C.purple], [128 * win, C.blue]], sel > 0.5 ? (win > 0.5 ? '1024 + 128 个' : '1024 个') : '2048 个');
  meter(450, '缓存', [[mix(N, 32768, s), KEPT], [128 * win, C.blue]], `${fmtN(mix(N, 32768, s))}${win > 0.5 ? ' + 128' : ''} 个`);
  txt('整条 = 128K 个', MX + MW, 536, { size: 22, weight: 500, color: C.gray, align: 'right' });
  const na = fade(u, bt(7, 2));
  txt('Compressed Sparse Attention', MX, 628, { size: 26, weight: 700, color: C.purple, alpha: na });
  note('简称 CSA：先压缩，再选 top-1024', 664, { alpha: na });
}

// ================= 9 看得细与看得全 =================
const S9 = { move: [bt(8, 0), bt(8, 0) + 1.4], hca: bt(8, 0) + 2.6, tick: (j) => bt(8, 1) + 0.5 + j * 0.06, fill: [bt(8, 1) + 2.2, bt(8, 1) + 3.4], layer: (i) => bt(8, 2) + 0.3 + i * 0.035 };
ev(8, S9.move[0], 'whoosh'); ev(8, S9.hca, 'pop'); evRun(8, S9.tick(0), S9.tick(16), 0.12, 'tick'); ev(8, S9.fill[0], 'whoosh'); ev(8, S9.fill[1], 'land'); evRun(8, S9.layer(0), S9.layer(61), 0.1, 'tick'); ev(8, bt(8, 2) + 2.8, 'click'); ev(8, bt(8, 2) + 3.4, 'click');
const COVER = (() => { const r = rng(5); return Array.from({ length: 16 }, () => Math.floor(r() * (PW - 3))); })();
function s9(u) {
  header(8, u);
  const m = p(u, ...S9.move), ha = fade(u, S9.hca), b1 = bt(8, 1), b2 = bt(8, 2);
  ghost({ alpha: 1 - m }); sr(PAGE.x - 2, PAGE.y - 2, PW + 4, PH + 4, C.gray, 2, 1 - m, [10, 8]);
  const cx = mix(CSA.x, PAGE.x, m), cy = mix(CSA.y, 300, m);
  rr(cx - 10, cy - 12, 660, 350, C.paper, 1 - m); csaPage(cx, cy);
  cap('{p:CSA}：4 个压成 1 个 · 32K 个 entry，读其中 1024 个', fade(u, S9.move[1]), PAGE.x, 286);
  const hx = 860, hy = 300 + 320 - 80, hs = 80 * po(u, S9.hca, S9.hca + 0.5);
  page('hca', () => C.purpleL, { x: hx + 40 - hs / 2, y: hy + 40 - hs / 2, cols: 32, rows: 32, w: hs, h: hs, alpha: ha });
  page('w128', () => C.blue, { x: hx, y: hy + 84, cols: 128, rows: 1, alpha: fade(u, S9.hca + 0.6) });
  rich('{p:HCA}：128 个压成 1 个', hx + 104, hy + 34, 28, 700, fade(u, S9.hca + 0.3)); txt('1024 个 entry，全部读', hx + 104, hy + 70, { size: 24, weight: 500, color: C.ink2, alpha: fade(u, S9.hca + 0.3) });
  // 覆盖
  txt('一步读到的内容，来自上下文的多大一部分', 96, 690, { size: 26, weight: 700, alpha: fade(u, b1) });
  const ca = fade(u, b1 + 0.3), fl = p(u, ...S9.fill), fa = fade(u, S9.fill[0] - 0.3);
  txt('CSA：1024 个 entry，来自 4096 个 token，是上下文的 1/32', 96, 730, { size: 22, weight: 500, color: C.ink2, alpha: ca }); rr(96, 740, PW, 24, GHOST, ca);
  COVER.forEach((x, j) => rr(96 + x, 740, 2.5, 24, C.purple, fade(u, S9.tick(j), 0.2)));
  txt('HCA：1024 个 entry，来自全部 128K 个 token，每个 entry 混合 128 个', 96, 800, { size: 22, weight: 500, color: C.ink2, alpha: fa }); rr(96, 810, PW, 24, GHOST, fa); rr(96, 810, PW * fl, 24, C.purpleL);
  // 61 层
  txt('DeepSeek-V4-Pro 的 61 层', 96, 890, { size: 26, weight: 700, alpha: fade(u, b2) });
  txt('"compress_ratios": [128, 128, 4, 128, 4, 128, 4, …]', PR, 890, { size: 20, weight: 500, color: C.gray, font: MONO, align: 'right', alpha: fade(u, b2 + 1.0) });
  const lp = PW / 61, wa = p(u, b2 + 2.8, b2 + 3.3), sa = p(u, b2 + 3.4, b2 + 3.9);
  for (let i = 0; i < 61; i++) { const hca = i < 2 || i % 2 === 1; rr(96 + i * lp, 904, lp - 4, 36, hca ? C.purpleL : C.purple, fade(u, S9.layer(i), 0.2)); }
  rr(96, 944, PW * wa - 4, 5, C.blue); rr(96, 953, PW * sa - 4, 5, C.red);
  const ga = fade(u, b2 + 2.4);
  rr(96, 984, 20, 20, C.purple, ga); txt('CSA × 30', 124, 1002, { size: 22, weight: 500, color: C.ink2, alpha: ga }); rr(250, 984, 20, 20, C.purpleL, ga); txt('HCA × 31', 278, 1002, { size: 22, weight: 500, color: C.ink2, alpha: ga });
  rr(410, 991, 20, 5, C.blue, wa); txt('每层：128 token 的窗口', 438, 1002, { size: 22, weight: 500, color: C.ink2, alpha: wa }); rr(720, 991, 20, 5, C.red, sa); txt('每个 head 一个 sink logit', 748, 1002, { size: 22, weight: 500, color: C.ink2, alpha: sa });
  // 右栏：HCA 的读取与缓存
  const ma = fade(u, S9.hca + 0.9);
  meter(330, 'HCA 读取', [[1024, C.purpleL], [128, C.blue]], '1024 + 128 个', { alpha: ma });
  meter(450, 'HCA 缓存', [[1024, C.purpleL], [128, C.blue]], '1024 + 128 个', { alpha: ma });
  txt('整条 = 128K 个', MX + MW, 536, { size: 22, weight: 500, color: C.gray, align: 'right', alpha: ma });
  txt('Heavily Compressed Attention', MX, 628, { size: 25, weight: 700, color: C.purple, alpha: fade(u, S9.hca + 1.6) }); note('简称 HCA：压缩后全部读', 664, { alpha: fade(u, S9.hca + 1.6) });
}

// ================= 10 1M 上下文 =================
const SEL_1M = (() => { const r = rng(21), s = new Set(); while (s.size < 4096) { const e = Math.floor(r() * 262144) * 4; for (let j = 0; j < 4; j++) s.add(e + j); } return s; })();
const MRCR = [['8K', 0.90], ['16K', 0.85], ['32K', 0.94], ['64K', 0.90], ['128K', 0.92], ['256K', 0.82], ['512K', 0.66], ['1M', 0.59]];
const S10 = { zoom: [bt(9, 0), bt(9, 0) + 2.2], gqa: [bt(9, 1) + 0.2, bt(9, 1) + 1.4], v4: bt(9, 1) + 1.9, pt: (j) => bt(9, 2) + 0.5 + j * 0.3 };
ev(9, S10.zoom[0], 'whoosh'); ev(9, S10.zoom[1], 'land'); evRun(9, S10.gqa[0], S10.gqa[1], 0.12, 'tick'); ev(9, S10.v4, 'pop'); MRCR.forEach((_, j) => ev(9, S10.pt(j), 'click'));
const page1m = (o) => page('1m', (i) => (i >= 1048576 - 128 ? C.blue : SEL_1M.has(i) ? C.purple : '#d3ccbc'), { cols: 2048, rows: 512, pitch: 0.75, S: 2, ...o });
function s10(u) {
  header(9, u);
  const X = 96, Y = 296, PWm = 1536, PHm = 384, o = p(u, ...S10.zoom), big = p(u, S10.zoom[0] + 0.9, S10.zoom[0] + 2.6), b1 = bt(9, 1), b2 = bt(9, 2);
  page1m({ x: X, y: Y, alpha: big });
  // 上一幕的那一页收成 1M 的最上面一条
  const ry = mix(300, Y, o), rw = mix(640, PWm, o), rh = mix(320, PHm / 8, o);
  if (o < 1) { csaPage(X, ry, 1, 0, rw, rh); rr(X, ry, rw, rh, C.paper, o); page1m({ x: X, y: Y, alpha: big * o, r1: 64 }); }
  sr(X - 2, ry - 2, rw + 4, rh + 4, C.ink, 2);
  callout(X + PWm + 2, Y + PHm / 16, X + PWm + 26, Y + PHm / 16, '128K', { size: 24, alpha: fade(u, S10.zoom[1]) });
  txt('刚才的一页', X + PWm + 36, Y + PHm / 16 + 40, { size: 22, weight: 500, color: C.gray, alpha: fade(u, S10.zoom[1]) });
  rich('1M 个 token · {p:紫色：CSA 一步读到的 1024 个 entry}（位置为示意），个数与 128K 时相同', X, Y + PHm + 36, 24, 500, fade(u, S10.zoom[1] + 0.4), C.gray);
  const bw = 900, ga = fade(u, b1), va = fade(u, S10.v4);
  txt('1M 上下文的 KV cache', X, 786, { size: 28, weight: 700, alpha: ga });
  txt('BF16 GQA-8，head dim 128', X, 838, { size: 24, weight: 500, color: C.ink2, alpha: ga }); rr(X, 852, bw * p(u, ...S10.gqa), 36, KEPT);
  txt('DeepSeek-V4', X, 934, { size: 24, weight: 500, color: C.ink2, alpha: va }); rr(X, 948, bw, 36, GHOST, va); rr(X, 948, bw * 0.02, 36, C.purple, va);
  txt('约 2%', X + bw * 0.02 + 14, 976, { size: 28, weight: 700, color: C.purple, alpha: fade(u, S10.v4 + 0.4) });
  // MRCR
  const cx = 1150, cy = 1000, cw = 640, ch = 170, px = (j) => cx + 30 + j * (cw - 60) / 7, py = (v) => cy - v * ch, ca = fade(u, b2);
  txt('从长文里找回 8 处内容的准确度（MRCR 8-needle，V4-Pro）', cx - 30, 786, { size: 24, weight: 700, alpha: ca });
  line(cx, cy, cx + cw, cy, C.ink, 2, ca); line(cx, py(1), cx + cw, py(1), C.lg, 2, ca, [6, 6]); txt('1.0', cx - 10, py(1) + 8, { size: 20, weight: 500, color: C.gray, align: 'right', alpha: ca }); txt('0', cx - 10, cy + 6, { size: 20, weight: 500, color: C.gray, align: 'right', alpha: ca });
  const tip = clamp((u - S10.pt(0)) / 0.3, 0, 7);
  if (u > S10.pt(0)) {
    g.strokeStyle = C.purple; g.lineWidth = 4; g.beginPath();
    for (let j = 0; j <= Math.floor(tip); j++) (j ? g.lineTo(px(j), py(MRCR[j][1])) : g.moveTo(px(j), py(MRCR[j][1])));
    const j0 = Math.floor(tip), fr = tip - j0; if (j0 < 7) g.lineTo(mix(px(j0), px(j0 + 1), fr), mix(py(MRCR[j0][1]), py(MRCR[j0 + 1][1]), fr));
    g.stroke();
  }
  MRCR.forEach(([n, v], j) => {
    const hi = n === '128K' || n === '1M', a = fade(u, S10.pt(j), 0.2);
    g.globalAlpha = a; g.fillStyle = C.purple; g.beginPath(); g.arc(px(j), py(v), 7, 0, 7); g.fill(); g.globalAlpha = 1;
    txt(n, px(j), cy + 30, { size: 20, weight: hi ? 700 : 500, color: hi ? C.ink : C.gray, align: 'center', alpha: ca });
    if (hi) txt(v.toFixed(2), px(j) + (n === '1M' ? 14 : -34), py(v) + 38, { size: 26, weight: 700, color: n === '1M' ? C.red : C.ink, align: 'center', alpha: a });
  });
}

// ================= 11 结尾 =================
const SEL_END = (() => { const s = new Set(); for (const e of SEL_CSA) for (let j = 0; j < 4; j++) s.add(e * 4 + j); return s; })();
ev(10, 0.2, 'whoosh'); ev(10, 1.3, 'land'); ev(10, 2.4, 'click'); ev(10, 3.4, 'pop');
function s11(u) {
  const back = p(u, 0.2, 1.4);
  bigTitle(1 - back);
  txt('每一步，只读其中一小部分', 96, 264, { size: 38, weight: 700, color: C.ink2, alpha: fade(u, 1.4) });
  page('end', (i) => (i >= N - 128 ? C.blue : SEL_END.has(i) ? C.purple : GHOST), { alpha: back }); query(back, 'query：正在生成的 token');
  txt('每步读取', MX, 440, { size: 28, weight: 700, alpha: fade(u, 2.2) });
  meter(500, '开场', [[N * p(u, 2.4, 3.2), C.blue]], '128K 个', { alpha: fade(u, 2.2) });
  meter(620, '现在', [[1024, C.purple], [128, C.blue]], '1024 + 128 个', { alpha: fade(u, 3.4) });
  txt('上下文再长，这一条也不变', MX, 750, { size: 26, weight: 700, color: C.purple, alpha: fade(u, 4.4) });
}

const SCENES = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11];
function draw(t) {
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  rr(0, 0, W, H, C.paper);
  SCENES[k](t - AT[k]);
}
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
