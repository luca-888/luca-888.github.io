// DSpark v6 前半：换了什么 → 快在哪 → 一个个写 → 一次写完 → 各抽各的 → 查表（脚本见 script.md）

// ---------- 1 封面 → 两个聊天窗口 ----------
// 封面：左边是名字与关键概念的大字，右边是一轮的整体结构（target → drafter 三个部件 → 调度器 → 回到 target 验证）。
// 转场时封面淡出，两个聊天窗口淡入成左右两栏，再按真实速度同时输出 2.5 秒
// 节拍：封面一句，第 1 幕一句。封面停到第一句讲完；转场后两个窗口在第 1 幕第一句开始时输出
const MORPH = 1.0, RUN = 2.5, OB = beatTimes(0, [.8, 1.6]), CV = OB[0].e + .4, STR = OB[1].s, OPEN_DUR = OB[1].e + 1.2;
BEAT_AT[0] = OB;
function coverMap() {
  const x0 = 1010, w = 814, cx = (i) => 1290 + i * 104, cw = 92, WD = ['可以', '，', '我', '来', '看'];
  // target
  const c = (f) => at(OB, 0, f);
  IN(.5, () => { R(x0, 250, w, 96, K.blue); TX('target：DeepSeek-V4', x0 + 28, 310, font(700, 36, HN), '#fff'); TX('模型不变', x0 + w - 28, 310, font(700, 28), '#fff', 'right'); }, .5, 0);
  IN(.8, () => { line(1300, 346, 1300, 414, K.blue, 4, [8, 6]); TX('hidden state', 1316, 390, font(700, 24), K.blue); }, .4, 0);
  // drafter：三个部件
  IN(.8, () => { box(x0, 420, w, 364, 4, K.purple); R(x0, 420, 190, 40, K.purple); TX('drafter', x0 + 18, 450, font(700, 26, HN), '#fff'); }, .5, 0);
  const row = (t0, y, name, sub, draw) => IN(t0, () => { TX(name, x0 + 28, y + 42, font(700, 30, HN), K.purple); draw(); }, .5, 0);
  row(.9, 476, '并行 backbone', '一次生成多个 draft', () => { R(cx(0), 476, cx(4) + cw - cx(0), 60, K.purple); TX('一次 forward 生成 5 个 draft token', (cx(0) + cx(4) + cw) / 2, 516, font(700, 26), '#fff', 'center'); });
  row(1.0, 584, 'Markov head', '按前一个 token 修正', () => { WD.forEach((wd, i) => { cell(cx(i), 584, cw, 60, K.purple, wd, 28);
    if (i) { X.strokeStyle = K.purple; X.lineWidth = 3; X.beginPath(); X.moveTo(cx(i - 1) + cw / 2 + 10, 580); X.quadraticCurveTo(cx(i) - 6, 548, cx(i) + cw / 2 - 10, 580); X.stroke(); } }); });
  row(1.1, 680, 'confidence head', '估计接受概率', () => { [60, 48, 36, 24, 12].forEach((h, i) => R(cx(i) + 16, 752 - h, cw - 32, h, '#a988f8')); line(cx(0), 752, cx(4) + cw, 752, K.purple, 2); });
  // 调度器
  IN(1.2, () => { arrow(1500, 784, 1500, 826, K.purple, 4);
    TX('调度器', x0 + 28, 874, font(700, 30), K.blue);
    WD.forEach((wd, i) => i < 3 ? cell(cx(i), 832, cw, 60, K.blue, wd, 28) : HR(cx(i), 832, cw, 60, K.lg, K.paper2)); }, .5, 0);
  // 回到 target 验证
  IN(1.3, () => { poly([[x0, 862], [972, 862], [972, 298]], K.blue, 4); arrow(972, 298, x0 - 2, 298, K.blue, 4);
    X.save(); X.translate(950, 580); X.rotate(-Math.PI / 2); X.font = font(700, 24); X.fillStyle = K.blue; X.textAlign = 'center'; X.fillText('target 一次 forward 验证', 0, 0); X.restore(); }, .5, 0);
}
const TPS = [80, 128];
function sOpen() {
  const m = S(CV, MORPH);
  A(1 - clamp(m * 2.5), () => {
    TX('DeepSeek V4 · speculative decoding', 96, 200, font(700, 40, HN), K.ink2);
    TX('DSpark', 84, 420, font(700, 220, HN), K.ink);
    IN(.4, () => RICH('{p:并行生成}，{b:按需验证}', 96, 570, font(800, 88), K.ink));
    IN(1.0, () => RICH('单用户生成速度提升 {b:60%–85%}', 96, 680, font(700, 48), K.ink));
    IN(1.4, () => TX('仅替换 drafter，模型不变', 96, 752, font(600, 34), K.ink2));
    coverMap();
  });
  A(clamp(m * 2 - 1), () => frame('改动了什么', K.blue, '模型不变，单用户生成速度提升 {b:60%–85%}', '论文 Fig. 7：V4-Flash 线上流量 · 回答文字为示意', CV + .6, at(OB, 1, .45)));
  const el = clamp((T - STR) / RUN) * RUN;
  [0, 1].forEach((i) => {
    const cx = 1000, cy = [240, 600][i], cw = 824, ch = 320, mx = [96, 972][i], my = 300, mw = 852, mh = 540;
    const x = mix(cx, mx, m), y = mix(cy, my, m), w = mix(cw, mw, m), h = mix(ch, mh, m);
    const n = T < STR ? [56, 90][i] * (1 - m) : TPS[i] * el;
    A(clamp(m * 2 - .6), () => chatWin(x, y, w, h, ['MTP-1', 'DSpark'][i], TPS[i], n, [K.ink2, K.purple][i], mix(20, 24, m)));
    IN(STR + RUN + .3 + i * .3, () => RICH(`2.5 秒：{${i ? 'p' : 'g'}:${Math.floor(TPS[i] * RUN)} 个字}`, mx, 892, font(700, 40), K.ink), .5, 0);
  });
}

// ---------- 1（B）开场：同一个回答，两个窗口按真实速度输出到写完 ----------
// 不停在静态封面上：标题与两个窗口同时出现，第一句配音开始时两边一起输出同一个回答（320 个字）。DSpark 2.5 秒写完，MTP-1 4.0 秒，
// 窗口下方的进度条与计时对着同一条基线。第二句给出不变的部分（target、输出分布），第三句给出改动（每轮 draft token 1 → 5，按负载验证）。
const OBB = beatTimes(0, [.6, .5, .5]), OPENB_DUR = OBB[2].e + 1.0, RACE_N = 320;
function sOpenB() {
  const b = (i, f) => at(OBB, i, f), t = Math.max(0, T - b(0)), wy = 300, wh = 400;
  IN(.05, () => { TX('DSpark', 88, 232, font(700, 150, HN), K.ink); TX('DeepSeek-V4 线上的 speculative decoding', 724, 150, font(700, 36, HN), K.ink2); }, .4, 0);
  IN(b(0, .55), () => RICH('吞吐相同，单用户快 {b:60%–85%}', 720, 234, font(700, 64), K.ink), .5, 0);
  [0, 1].forEach((i) => { const mx = [96, 972][i], col = [K.ink2, K.purple][i], end = RACE_N / TPS[i], ti = Math.min(t, end), done = t >= end;
    IN(.15, () => chatWin(mx, wy, 852, wh, ['MTP-1', 'DSpark'][i], TPS[i], TPS[i] * ti, col, 24), .4, 0);
    IN(.15, () => { R(mx, wy + wh + 16, 852, 18, K.paper2); R(mx, wy + wh + 16, 852 * ti / end, 18, col); }, .4, 0);
    if (T >= b(0)) RICH(`{${i ? 'p' : 'g'}:${ti.toFixed(1)} 秒}${done ? ' · 写完' : ''}`, mx, wy + wh + 78, font(700, 40, HN), K.ink);
    // 每轮的 draft token：MTP-1 为 1 个；DSpark 为 5 个，其中一部分送去验证
    IN(b(2, .1 + i * .3), () => { const cy = 884, cw = 60, n = i ? 5 : 1;
      for (let k = 0; k < n; k++) { if (i && k >= 3) HR(mx + k * (cw + 8), cy, cw, 50, K.lg, K.paper2); else R(mx + k * (cw + 8), cy, cw, 50, K.purple); }
      TX(i ? '每轮 5 个 draft token，按负载验证其中几个' : '每轮 1 个 draft token', mx + n * (cw + 8) + 12, cy + 36, font(700, 30), K.ink); }, .5, 0); });
  IN(b(1, .1), () => RICH('{b:target：DeepSeek-V4，不变} · 输出分布不变 · {p:只换 drafter}', 960, 848, font(700, 34), K.ink, 'center'), .5, 0);
}
// 信息流封面（4:3，1440 × 1080，画在画布左侧 1440 px 内）：开场第一页的内容重新摆放，不另行设计。
// 顶部 12% 不放字，左右边距 86 px，右上角约 1/8 宽留给播放图标。两个窗口停在输出 1.75 秒处
function coverFeed() {
  R(0, 0, 1920, 1080, K.paper); const t = 1.75, wy = 640, wh = 318, ww = 626;
  TX('DeepSeek V4 · speculative decoding', 90, 192, font(700, 40, HN), K.ink2);
  TX('DSpark', 80, 376, font(700, 190, HN), K.ink);
  RICH('吞吐相同，单用户快 {b:60%–85%}', 88, 500, font(700, 84), K.ink);
  RICH('{p:并行生成} draft token，{b:按负载验证}', 90, 576, font(700, 42), K.ink2);
  [0, 1].forEach((i) => { const mx = [86, 728][i], col = [K.ink2, K.purple][i];
    chatWin(mx, wy, ww, wh, ['MTP-1', 'DSpark'][i], TPS[i], TPS[i] * t, col, 20);
    R(mx, wy + wh + 14, ww, 16, K.paper2); R(mx, wy + wh + 14, ww * t / (RACE_N / TPS[i]), 16, col); });
}
const OPENB_EV = (() => { const b = (i, f) => at(OBB, i, f); return [[.1, 'pop'], ...Array.from({ length: 16 }, (_, k) => [b(0) + k * .25, 'tick']), [b(0) + 2.5, 'land'], [b(0) + 4.0, 'land'], [b(1, .1), 'pop'], [b(2, .1), 'pop'], [b(2, .4), 'pop']]; })();

// ---------- 1b 全貌：一轮从上到下走一遍 ----------
// 六行：target → drafter 的三个部件 → 调度器 → target 验证。每行左边是部件名，中间是同一组 5 个 draft token 在这一步的样子，右边一句标注。
// 之后各幕顶部的结构条（partBar）就是这里四个部件的缩略
const MB = beatTimes(1, [.5, .4, .4, .4, .4, .4]), MAP_DUR = MB[5].e + 1.8;
BEAT_AT[1] = MB;
const MAP_P = [.92, .81, .66, .41, .25];   // 示意数值
function sMap() {
  const b = (i, f) => at(MB, i, f), cw = 160, rh = 70, C = (c) => 400 + c * 180, Y = (r) => 250 + r * 108, lx = 150, ax = 1500, gx = C(0) + cw / 2;
  frame('全貌', K.ink, 'DSpark 的一轮：{p:drafter 的三个部件}加{b:调度器}', '概率为示意', .15, b(5, .5));
  const lab = (r, s, col) => TX(s, lx, Y(r) + 45, font(700, 30, HN), col);
  const note = (r, l1, l2) => { TX(l1, ax, Y(r) + (l2 ? 30 : 46), font(600, 28), K.ink); if (l2) TX(l2, ax, Y(r) + 66, font(600, 28), K.ink2); };
  const down = (r, t0, col) => arrowP(gx, Y(r) + rh + 6, gx, Y(r + 1) - 6, S(t0, .3), col, 4);
  // drafter 的三行放在同一条淡紫底带里；骨架：六行的部件名与空格子先以淡色全部出现，字幕讲到哪一行，哪一行点亮
  const NAMES = ['target', '并行 backbone', 'Markov head', 'confidence head', '调度器', 'target 验证'], SLOTS = [0, 5, 5, 5, 5, 3];
  IN(.3, () => { R(100, Y(1) - 16, 1724, 2 * 108 + rh + 32, '#ebe4fa');
    X.save(); X.translate(128, Y(2) + rh / 2); X.rotate(-Math.PI / 2); X.font = font(700, 24, HN); X.fillStyle = K.purple; X.textAlign = 'center'; X.fillText('drafter', 0, 8); X.restore(); }, .5, 0);
  IN(.3, () => { NAMES.forEach((n, r) => { TX(n, lx, Y(r) + 45, font(700, 30, HN), K.lg);
      if (!r) box(C(0) + 2, Y(0) + 2, C(5) + cw - C(0) - 4, rh - 4, 3, K.lg); if (r === 1) box(C(0) + 2, Y(1) + 2, cw - 4, rh - 4, 3, K.lg);
      for (let c = 1; c <= SLOTS[r]; c++) box(C(c) + 2, Y(r) + 2, cw - 4, rh - 4, 3, K.lg);
      if (r < 5) arrow(gx, Y(r) + rh + 6, gx, Y(r + 1) - 6, K.lg, 4); }); }, .6, 0);
  IN(b(0, .3), () => { lab(0, 'target', K.blue); R(C(0), Y(0), C(5) + cw - C(0), rh, K.blue); TX('DeepSeek-V4：算完上文', C(0) + 24, Y(0) + 46, font(700, 30), '#fff'); note(0, '输出 hidden state', '模型不变'); }, .5, 0);
  down(0, b(1) - .2, K.blue);
  IN(b(1), () => { lab(1, '并行 backbone', K.purple); chip(C(0), Y(1), cw, rh, 'ctx', '当然', 32); for (let c = 1; c <= 5; c++) chip(C(c), Y(1), cw, rh, 'guess', null); note(1, '一次 forward', '5 个位置的预测'); }, .5, 0);
  down(1, b(2) - .2, K.purple);
  IN(b(2), () => { lab(2, 'Markov head', K.purple); note(2, '逐位采样', '参考前一个 token'); }, .5, 0);
  WORDS.forEach((w, i) => { const t0 = b(2, .15 + i * .14); POP(t0, C(i + 1), Y(2), cw, rh, () => chip(C(i + 1), Y(2), cw, rh, 'word', w, 32), .35);
    if (i && T > t0 - .15) A(P(t0 - .15, .3), () => { X.strokeStyle = K.purple; X.lineWidth = 4; X.beginPath(); X.moveTo(C(i) + cw / 2 + 20, Y(2) - 2); X.quadraticCurveTo(C(i) + cw + 10, Y(2) - 30, C(i + 1) + cw / 2 - 20, Y(2) - 2); X.stroke(); }); });
  down(2, b(3) - .2, K.purple);
  IN(b(3), () => { lab(3, 'confidence head', K.purple); note(3, '估计每个 token', '被接受的概率'); }, .5, 0);
  MAP_P.forEach((v, i) => IN(b(3, .1 + i * .1), () => { R(C(i + 1), Y(3), cw, rh, K.paper2); R(C(i + 1), Y(3), cw * v * S(b(3, .1 + i * .1), .5), rh, '#a988f8'); TX(v.toFixed(2), C(i + 1) + cw / 2, Y(3) + 46, font(700, 30, HN), K.ink, 'center'); }, .3, 0));
  down(3, b(4) - .2, K.blue);
  IN(b(4), () => { lab(4, '调度器', K.blue); note(4, '按概率与负载', '选出验证哪些'); }, .5, 0);
  WORDS.forEach((w, i) => IN(b(4, .2 + i * .08), () => (i < 3 ? chip(C(i + 1), Y(4), cw, rh, 'blue', w, 32) : chip(C(i + 1), Y(4), cw, rh, 'void', '不验证', 26)), .3, 0));
  down(4, b(5) - .2, K.blue);
  const fix = b(5, .72);   // 第 3 位被拒绝后，换成 target 自己采样的 token（蓝色，同第 2 幕的“target 补的”）
  IN(b(5), () => { lab(5, 'target 验证', K.blue); note(5, '接受 2 个', T >= fix ? '第 3 个换成 target 的 token' : '第 3 个被拒绝'); }, .5, 0);
  [['可以', 'ok'], ['，', 'ok']].forEach(([w, st], i) => POP(b(5, .12 + i * .1), C(i + 1), Y(5), cw, rh, () => chip(C(i + 1), Y(5), cw, rh, st, w, 32), .35));
  if (T < fix) POP(b(5, .34), C(3), Y(5), cw, rh, () => chip(C(3), Y(5), cw, rh, 'bad', '我', 32), .35); else POP(fix, C(3), Y(5), cw, rh, () => chip(C(3), Y(5), cw, rh, 'blue', '帮', 32), .4);
}
const MAP_EV = (() => { const b = (i, f) => at(MB, i, f); return [[.1, 'click'], [b(0, .3), 'pop'], [b(1), 'pop'], ...[0, 1, 2, 3, 4].map((i) => [b(2, .15 + i * .14), 'tick']), [b(3), 'pop'], [b(4), 'pop'], [b(4, .5), 'whoosh'], [b(5, .12), 'land'], [b(5, .34), 'err'], [b(5, .72), 'pop']]; })();

// ---------- 2 一轮的时间：式子拆成三段 ----------
// 一轮 = 猜（紫）+ 验证（蓝）；MTP-1 猜 1 个，验证后收下“猜对的 + target 自己补的一个”
const RB = { x: 330, y: 372, h: 64, bw: 70, vw: 620 };
// 式子（分式）：t = (T_draft + T_verify) / τ，记号与文章一致。lead 是式子前的中文说明。
// 返回各项的位置：draft / verify 为分子两项的中心与顶部，tau 为分母的中心、右端与基线
const MI = (fs) => `italic 400 ${fs}px "Times New Roman", "STIX Two Text", serif`, MR = (fs) => `400 ${fs}px "Times New Roman", "STIX Two Text", serif`, MS = (fs) => `400 ${Math.round(fs * .42)}px "Times New Roman", "STIX Two Text", serif`;
function mathW(items, fs) { return items.reduce((w, [s, k]) => { X.font = k === 'i' ? MI(fs) : k === 's' ? MS(fs) : MR(fs); return w + X.measureText(s).width + (k === 's' ? 2 : 0); }, 0); }
function mathTX(items, x, y, fs, col) { let cx = x; const pos = []; for (const [s, k, c] of items) { const f = k === 'i' ? MI(fs) : k === 's' ? MS(fs) : MR(fs); X.font = f; const w = X.measureText(s).width;
  X.fillStyle = c || col; X.textAlign = 'left'; X.textBaseline = 'alphabetic'; X.fillText(s, cx + (k === 's' ? 2 : 0), k === 's' ? y + fs * .14 : y); pos.push([cx, w + (k === 's' ? 2 : 0)]); cx += w + (k === 's' ? 2 : 0); } return pos; }
function formula(x, fy, fs, t0, lead) {
  const num = [['T', 'i', K.purple], ['draft', 's', K.purple], [' + ', 'r', K.ink], ['T', 'i', K.blue], ['verify', 's', K.blue]], den = [['τ', 'i', K.teal]];
  X.font = font(700, fs * .58); const lw = lead ? X.measureText(lead).width + fs * .45 : 0;
  const lhs = [['t', 'i', K.ink], [' = ', 'r', K.ink]], lhsW = mathW(lhs, fs), nw = mathW(num, fs), dw = mathW(den, fs), fw = Math.max(nw, dw) + fs * .3;
  const fx = x + lw + lhsW, nx = fx + (fw - nw) / 2, dx = fx + (fw - dw) / 2, ny = fy - fs * .26, dy = fy + fs * .92;
  IN(t0, () => { if (lead) TX(lead, x, fy + fs * .2, font(700, fs * .58), K.ink); mathTX(lhs, x + lw, fy + fs * .3, fs, K.ink); }, .4, 0);
  IN(t0 + .2, () => { mathTX(num, nx, ny, fs, K.ink); line(fx, fy, fx + fw, fy, K.ink, Math.max(2, fs / 22)); }, .4, 0);
  IN(t0 + .4, () => mathTX(den, dx, dy, fs, K.ink), .4, 0);
  X.font = MI(fs); const tw = X.measureText('T').width; X.font = MS(fs); const sd = X.measureText('draft').width + 2;
  X.font = MR(fs); const pw = X.measureText(' + ').width;
  return { draft: { x: nx + (tw + sd) / 2, top: ny - fs * .78 }, verify: { x: nx + tw + sd + pw + (nw - tw - sd - pw) / 2, top: ny - fs * .78 }, tau: { x: dx + dw / 2, right: dx + dw, y: dy - fs * .3 }, right: fx + fw };
}
// 式子下面的三条注解：从左到右为 ① ② ③，颜色对应分子两项与分母
function fLabel(lx, ly, t0, col, l1, l2, fs = 32) {
  IN(t0, () => { TX(l1, lx, ly, font(700, fs), col, 'center'); if (l2) TX(l2, lx, ly + fs + 10, font(600, fs - 4), K.ink2, 'center'); }, .5, 0);
}
// 记号的含义：彩色记号（与式子同一字体）+ “：” + 说明，整体以 lx 居中
function symDef(lx, ly, t0, items, text, fs = 46) {
  IN(t0, () => { const f = font(600, 30), sw = mathW(items, fs); X.font = f; const w = sw + X.measureText('：' + text).width;
    mathTX(items, lx - w / 2, ly, fs, K.ink); TX('：' + text, lx - w / 2 + sw, ly - 2, f, K.ink); }, .5, 0);
}
const RT = beatTimes(2, [.5, .5, .5, .5]), ROUND_DUR = RT[3].e + 1.2;
BEAT_AT[2] = RT;
function sRound() {
  const b = (i, f) => at(RT, i, f);
  frame('提速来自哪里', K.blue, '一轮耗时分摊到接受的 token 上：提速有{b:三种途径}', null, .15, b(3));
  IN(b(0, .3), () => { TX('MTP-1', 96, RB.y + 44, font(700, 40, HN), K.ink); }, .5, 0);
  const pd = S(b(0, .5), .5), pv = S(b(0, .6), .9);
  roundBar(RB.x, RB.y, RB.h, 1, RB.bw, RB.vw, pd, pv);
  IN(b(0, .5), () => TX('draft 1 个', RB.x, RB.y - 14, font(700, 26), K.purple), .4, 0);
  // 这一轮的 token：同一次 forward 的两种结果。draft 被接受时，下一位的分布已经算出，顺带采一个 bonus token；被拒绝时 target 在 draft 位补上自己的 token
  const tx = 1070, th = 56, r1 = 352, r2 = 420, c2 = tx + 140, c3 = tx + 262;
  IN(b(1), () => { chip(tx, r1, 130, th, 'ctx', '…当然', 28); chip(c2, r1, 110, th, T > b(1, .3) ? 'ok' : 'guess', '可以', 28); }, .4, 0);
  POP(b(1, .4), c3, r1, 80, th, () => chip(c3, r1, 80, th, 'blue', '，', 28), .4);
  IN(b(1, .5), () => RICH('{b:bonus} · {t:产出 2 个}', c3 + 100, r1 + 40, font(700, 30), K.ink), .5, 0);
  IN(b(1, .78), () => { chip(tx, r2, 130, th, 'ctx', '…当然', 28); chip(c2, r2, 110, th, 'bad', '可以', 28); chip(c3, r2, 80, th, 'blue', '是', 28);
    RICH('{b:target 补的} · {t:产出 1 个}', c3 + 100, r2 + 40, font(700, 30), K.ink); }, .5, 0);
  // 式子与三段的来历
  const fy = 690, M = formula(330, fy, 76, b(2), '每个 token 的耗时');
  arrowP(RB.x + RB.bw / 2, RB.y + RB.h + 8, M.draft.x, M.draft.top - 6, S(b(2, .35), .6), K.purple, 4);
  arrowP(RB.x + RB.bw + RB.vw / 2, RB.y + RB.h + 8, M.verify.x, M.verify.top - 6, S(b(2, .45), .6), K.blue, 4);
  IN(b(2, .6), () => { line(1790, r1 + 4, 1800, r1 + 4, K.teal, 4); line(1800, r1 + 4, 1800, r2 + th - 4, K.teal, 4); line(1790, r2 + th - 4, 1800, r2 + th - 4, K.teal, 4); }, .3, 0);
  { const p = S(b(2, .65), .8), sx = 1800, sy = r2 + th - 4, ex = M.right + 60; if (p > .02) { const a1 = Math.min(1, p * 2), a2 = clamp(p * 2 - 1);   // 先向下，再向左指到分母
    line(sx, sy, sx, mix(sy, fy + 46, a1), K.teal, 4); if (a1 >= 1) { line(sx, fy + 46, mix(sx, ex, a2), fy + 46, K.teal, 4); if (a2 > .02) arrow(ex + 1, fy + 46, mix(ex, M.tau.right + 14, a2), fy + 46, K.teal, 4); } } }
  // 记号的含义随式子出现；三种途径在下一拍出现，各自对着所改变的那一项
  symDef(500, 846, b(2, .5), [['T', 'i', K.purple], ['draft', 's', K.purple]], '生成 draft 的耗时');
  symDef(1000, 846, b(2, .8), [['τ', 'i', K.teal]], '每轮平均接受的 token 数');
  symDef(1500, 846, b(2, .65), [['T', 'i', K.blue], ['verify', 's', K.blue]], '验证的耗时');
  fLabel(500, 900, b(3, .2), K.purple, '① draft 更快', '并行 backbone');
  fLabel(1000, 900, b(3, .4), K.teal, '② 接受更多', 'backbone + Markov head');
  fLabel(1500, 900, b(3, .65), K.blue, '③ 减少无效验证', 'confidence head + 调度器');
}
// 第 2 幕的音效（与画面同一套节拍）
const ROUND_EV = (() => { const b = (i, f) => at(RT, i, f); return [[.1, 'click'], [b(0, .5), 'pop'], [b(0, .6), 'whoosh'], [b(1, .3), 'land'], [b(1, .4), 'pop'], [b(1, .78), 'err'], [b(2), 'whoosh'], [b(2, .35), 'tick'], [b(2, .45), 'tick'], [b(2, .65), 'tick'], [b(3, .2), 'pop'], [b(3, .4), 'pop'], [b(3, .65), 'pop']]; })();

// ---------- 3 一个个写 ----------
// 1 层的小 drafter 跑 5 次，每次先读自己刚写的那个；时间条的猜那一段一格格变长
const WORDS = ['可以', '，', '我', '来', '看'], R0 = 1.4, RDT = 1.25;
const SX = (i) => 300 + i * 150;
function sAR() {
  frame('生成 5 个 draft token 的直接做法', K.purple, '自回归生成：5 个 draft token 需要{p:运行 5 次}', '0.53：论文 Fig. 2 聊天，Qwen3-4B · 时间条为示意');
  const sy = 320, sh = 76, sw = 130, by = 500, bh = 96;
  IN(.4, () => chip(96, sy, 180, sh, 'ctx', '…当然'), .4, 0);
  WORDS.forEach((w, i) => IN(.6 + i * .08, () => chip(SX(i), sy, sw, sh, T > R0 + i * RDT + .9 ? 'word' : 'guess', T > R0 + i * RDT + .9 ? w : null), .4, 0));
  // drafter 在第 i 次时停在第 i 个空位下方
  const k = clampR(Math.floor((T - R0 + .3) / RDT), 0, 4), move = S(R0 + k * RDT - .3, .3), bx = mix(SX(Math.max(0, k - 1)), SX(k), k ? move : 1);
  if (T < R0 + 5 * RDT + .4) {
    A(P(.9, .5), () => { const t0 = R0 + k * RDT, act = win(t0 + .35, t0 + .9, .1);
      R(bx, by, sw, bh, act > 0 ? K.purple : '#cdb8fb'); TX('drafter', bx + sw / 2, by + 42, font(700, 28, HN), '#fff', 'center'); TX('1 层', bx + sw / 2, by + 78, font(700, 26), '#fff', 'center');
      TX(`第 ${k + 1} 次`, bx + sw / 2, by + bh + 40, font(700, 28), K.purple, 'center'); TX('读取上一次的输出', bx + sw + 18, by + 58, font(700, 26), K.ink);
      const px = k ? SX(k - 1) + sw / 2 : 186;
      arrowP(px, sy + sh + 6, bx + sw / 2 - 20, by - 6, S(t0, .35), K.ink, 4);
      arrowP(bx + sw / 2 + 20, by - 6, SX(k) + sw / 2 + 20, sy + sh + 6, S(t0 + .5, .35), K.purple, 4); });
  }
  // 时间条
  const ty = 760, done = clamp((T - R0 - .9) / (5 * RDT));
  IN(1.0, () => { TX('一轮耗时', 96, ty + 42, font(700, 30), K.ink); roundBar(330, ty, 60, 5, 70, 620, done, S(R0 + 5 * RDT + .2, .8)); }, .5, 0);
  IN(1.4, () => TX(`draft：运行 ${Math.min(5, Math.ceil(done * 5 - .01))} 次`, 330, ty - 14, font(700, 26), K.purple), .4, 0);
  // 第 1 个猜对的比例
  const mx = 1180, my = 330;
  IN(9.2, () => { TX('运行次数多，drafter 规模受限', mx, my + 30, font(700, 32), K.ink); TX('第 1 位接受率（聊天）', mx, my + 90, font(600, 28), K.ink2);
    R(mx, my + 110, 600, 50, K.paper2); R(mx, my + 110, 600 * .53 * S(9.6, .8), 50, K.purple); TX('0.53', mx + 600 * .53 + 16, my + 148, font(700, 40, HN), K.red); }, .5, 0);
  // 回扣第 2 幕的式子：T_draft 变大；τ 比 MTP-1 高，但受第 1 位接受率限制，升幅有限
  if (T > 10.8) { const F = formula(1400, 640, 60, 10.8); IN(11.6, () => { TX('↑', F.draft.x + 46, F.draft.top + 30, font(800, 44, HN), K.red, 'center'); TX('升幅有限', F.tau.right + 18, F.tau.y + 22, font(700, 28), K.red); }, .4, 0); }
}

// ---------- 3（重排，与第 4 幕同一版面）一个个写 ----------
// 时刻与 sAR 相同（R0、RDT、9.2、10.8）。主图：与第 4 幕同一网格的 token 行，1 层的 drafter 逐位运行并留下痕迹，共 5 次；
// 底部时间条与第 4 幕同一位置，draft 段一格格变长；右下角的第 1 位接受率也在第 4 幕的同一位置
function sARB() {
  frame('生成 5 个 draft token 的直接做法', K.purple, '自回归生成：5 个 draft token 需要{p:运行 5 次}', '0.53：论文 Fig. 2 聊天，Qwen3-4B · 时间条为示意');
  const cw = 170, C = (c) => 350 + c * 190, ty = 300, th = 76, dy = 500, dh = 110, ax = 1512;
  IN(.4, () => { TX('输出', 320, ty + 50, font(700, 28), K.ink, 'right'); chip(C(0), ty, cw, th, 'ctx', '当然', 34); }, .4, 0);
  WORDS.forEach((w, i) => IN(.6 + i * .08, () => { const done = T > R0 + i * RDT + .9; chip(C(i + 1), ty, cw, th, done ? 'word' : 'guess', done ? w : null, 34); }, .4, 0));
  // drafter：第 k 次运行停在第 k 个空位下方，运行过的留下浅色痕迹
  IN(.9, () => TX('drafter', 320, dy + 64, font(700, 28, HN), K.ink, 'right'), .4, 0);
  for (let k = 0; k < 5; k++) { const t0 = R0 + k * RDT, x = C(k + 1); if (T < t0 - .3) continue;
    const act = T < t0 + RDT - .3 || k === 4 && T < t0 + 1.2;
    A(P(t0 - .3, .3), () => { R(x, dy, cw, dh, act ? K.purple : '#cdb8fb'); TX('1 层', x + cw / 2, dy + 50, font(700, 30), '#fff', 'center'); TX(`第 ${k + 1} 次`, x + cw / 2, dy + 90, font(700, 26), '#fff', 'center'); });
    arrowP(C(k) + cw / 2 + 30, ty + th + 8, x + cw / 2 - 40, dy - 8, S(t0, .35), act ? K.ink : K.lg, 4);
    arrowP(x + cw / 2 + 20, dy - 8, x + cw / 2 + 20, ty + th + 8, S(t0 + .5, .35), act ? K.purple : '#cdb8fb', 5); }
  IN(R0 + .2, () => { TX('每次读取', ax, dy + 44, font(600, 28), K.ink); TX('上一次的输出', ax, dy + 82, font(600, 28), K.ink2); }, .5, 0);
  // 底部时间条：与第 4 幕同一位置
  const bx = 330, by = 870, bh = 60, done = clamp((T - R0 - .9) / (5 * RDT)), n = Math.min(5, Math.ceil(done * 5 - .01));
  IN(1.0, () => { TX('一轮耗时', 96, by + 42, font(700, 30), K.ink);
    for (let i = 0; i < 5; i++) { const p = clamp(done * 5 - i); if (p > 0) { R(bx + i * 70 + 1, by, 68 * p, bh, '#a988f8'); if (p > .6) TX('1 层', bx + i * 70 + 35, by + 40, font(700, 24), '#fff', 'center'); } }
    TX(`draft：1 层 × ${Math.max(1, n)} 次`, bx, by - 14, font(700, 28), K.purple);
    const pv = S(R0 + 5 * RDT + .2, .8); if (pv > 0) { R(bx + 352, by, 618 * pv, bh, K.blue); if (pv > .6) TX('验证：target 一次 forward', bx + 372, by + 40, font(700, 28), '#fff'); } }, .5, 0);
  // 右下：第 1 位接受率（第 4 幕在它下面加上“并行”一行）
  const rx = 1350;
  IN(9.2, () => { TX('第 1 位接受率（Qwen3 实验）', rx, by - 14, font(700, 28), K.ink); TX('自回归', rx, by + 23, font(700, 26), K.ink2); R(rx + 96, by, 250 * .53 * S(9.6, .8), 26, '#a988f8'); TX('0.53', rx + 96 + 250 * .53 + 12, by + 24, font(700, 30, HN), K.red); }, .5, 0);
  // 回扣第 2 幕的式子：T_draft 变大；τ 比 MTP-1 高，但受第 1 位接受率限制，升幅有限
  if (T > 10.8) { const F = formula(1180, 690, 64, 10.8); IN(11.6, () => { TX('↑', F.draft.x + 50, F.draft.top + 32, font(800, 46, HN), K.red, 'center'); TX('升幅有限', F.tau.right + 18, F.tau.y + 22, font(700, 28), K.red); }, .4, 0); }
}

// ---------- 4 一次写完 ----------
// 上一个确定的 token 加 5 个空位，一起穿过 5 层（每层里这些位置互相看得见，也读 target 的中间状态），一次 forward 得到 5 个位置的打分
const PCX = (c) => 380 + c * 118;
// 画面事件的时刻（幕内时间）。默认值为 v6 的固定时间；film-v7.html 按配音节拍改写，让每个对象在讲到它的那句出现
const PAR_T = { inp: .5, mask: .5, maskDt: 0, cap: .7, up: 1.0, lay: .9, tgt: 1.2, ltxt: 1.6, scan: 2.2, out: 3.9, r1: 5.4, ar: 5.8, par: 6.4, acc: 7.2, more: 9.0 };
function sPar() {
  const t = PAR_T;
  frame('能否一次生成全部', K.purple, '一次 forward，{p:5 个 mask 位置}同时生成', '0.72：论文 Fig. 2 聊天，Qwen3-4B · 时间条为示意');
  const iy = 800, oy = 310, ch = 66, cw = 100, L0 = 420, LH = 54, LG = 10, LB = L0 + 5 * (LH + LG) - LG, lx = PCX(0) - 28;
  // 输入：上一个已确定的 token + 5 个 mask（占位 token）；每个位置一条向上的箭头进入 drafter
  IN(t.inp, () => { TX('输入', lx, iy + 42, font(700, 24), K.ink, 'right'); chip(PCX(0), iy, cw, ch, 'ctx', '当然', 30); }, .5, 0);
  for (let c = 1; c <= 5; c++) IN(t.mask + (c - 1) * t.maskDt, () => { R(PCX(c), iy, cw, ch, K.paper); box(PCX(c) + 2, iy + 2, cw - 4, ch - 4, 3, K.lg); TX('mask', PCX(c) + cw / 2, iy + 44, font(700, 26, HN), K.gray, 'center'); }, .4, 0);
  IN(t.cap, () => RICH('{p:mask}：占位 token，该位置等待预测；“当然”：上一个确定的 token', PCX(0) - 14, iy + ch + 40, font(600, 24), K.ink), .5, 0);
  for (let c = 0; c <= 5; c++) arrowP(PCX(c) + cw / 2, iy - 8, PCX(c) + cw / 2, LB + 8, S(t.up + c * .05, .3), c ? K.gray : K.ink, 4);
  // drafter 的 5 层；扫描高亮自下而上走一遍。每层做的事写在第 1 层里
  const sc = S(t.scan, 1.8), cur = 4 - Math.min(4, Math.floor(sc * 5));
  for (let l = 0; l < 5; l++) { const y = L0 + l * (LH + LG);
    IN(t.lay + (4 - l) * .1, () => { const hot = T > t.scan && T < t.scan + 1.9 && l === cur;
      R(PCX(0) - 14, y, PCX(5) + cw + 14 - PCX(0) + 14, LH, hot ? K.purple : '#ddd0fa');
      TX(`第 ${5 - l} 层`, lx, y + 36, font(700, 24), K.purple, 'right');
      if (l === 4 && T > t.ltxt) A(P(t.ltxt, .5), () => TX('每一层：各位置读取 hidden state，并相互可见（无 causal mask）', (PCX(0) - 14 + PCX(5) + cw + 14) / 2, y + 36, font(700, 23), hot ? '#fff' : K.ink, 'center')); }, .4, 0); }
  // target 的中间状态注入每一层
  IN(t.tgt, () => { R(96, 470, 170, 200, K.blue); ['target', '上文的', 'hidden state'].forEach((s, i) => TX(s, 181, 530 + i * 40, font(700, i ? 24 : 28, i ? ZH : HN), '#fff', 'center'));
    for (let l = 0; l < 5; l++) line(266, 570, PCX(0) - 110, L0 + l * (LH + LG) + LH / 2, K.blue, 2, [6, 6]); }, .5, 0);
  // 输出：5 个 mask 位置各得到一个 draft token
  for (let c = 1; c <= 5; c++) { arrowP(PCX(c) + cw / 2, L0 - 8, PCX(c) + cw / 2, oy + ch + 8, S(t.out + c * .04, .3), K.purple, 4);
    POP(t.out + .1 + c * .04, PCX(c), oy, cw, ch, () => chip(PCX(c), oy, cw, ch, 'word', WORDS[c - 1], 30), .4); }
  IN(t.out + .1, () => TX('输出', lx, oy + 42, font(700, 24), K.ink, 'right'), .5, 0);
  IN(t.out + .5, () => TX('每个 mask 位置输出一组 token 概率，各采样一个 draft token', PCX(0) - 14, oy - 18, font(600, 24), K.purple), .5, 0);
  // 右：同样的时间，两种写法能用多少层
  const rx = 1200, bw = 84, nGuess = T > t.more ? 7 : 5;
  IN(t.r1, () => TX('层越多预测越准，耗时也越长。相同耗时：', rx, 360, font(700, 30), K.ink), .5, 0);
  IN(t.ar, () => { TX('自回归', rx, 420, font(700, 28), K.ink); for (let i = 0; i < nGuess; i++) { const p = i < 5 ? 1 : S(t.more + (i - 5) * .3, .4); if (p > 0) { R(rx + i * bw, 436, (bw - 6) * p, 56, '#a988f8'); TX('1 层', rx + i * bw + (bw - 6) / 2, 474, font(700, 22), '#fff', 'center'); } } }, .5, 0);
  IN(t.par, () => { TX('并行', rx, 556, font(700, 28), K.ink); for (let l = 0; l < 5; l++) R(rx, 572 + l * 11.2, 5 * bw - 6, 9, K.purple); TX('5 层 × 1 次', rx + 5 * bw + 14, 612, font(700, 26), K.purple); }, .5, 0);
  IN(t.more + .4, () => TX(`再增加 2 个：自回归多运行两次，并行耗时不变`, rx, 680, font(700, 26), K.ink2), .5, 0);
  IN(t.acc, () => { TX('第 1 位接受率', rx, 760, font(600, 28), K.ink2);
    [['自回归', .53, '#a988f8', K.red], ['并行', .72, K.purple, K.teal]].forEach(([n, v, c, tc], i) => { const y = 780 + i * 66; TX(n, rx, y + 40, font(700, 24), K.ink2); R(rx + 120, y, 420, 50, K.paper2); R(rx + 120, y, 420 * v * S(t.acc + .2 + i * .5, .7), 50, c); TX(v.toFixed(2), rx + 120 + 420 * v + 12, y + 38, font(700, 34, HN), tc); }); }, .5, 0);
}

// ---------- 4（样稿 B）一次写完：一个主图 + 贯穿的时间条 ----------
// 与 sPar 同一套时刻（PAR_T）。主图铺满中间：输入行 → 3 层（每层每个位置一格）→ 输出行；说明只留名词标注，放在主图右侧。
// 底部是第 2、3 幕的那条时间条：draft 段从“1 层 × 5 次”缩短为“3 层 × 1 次”（DeepSeek-V4 线上的 backbone 为 3 层 MoE），右边是第 1 位接受率（论文只在 Qwen3 上有这组对比）
function sParB() {
  const t = PAR_T, cw = 170, C = (c) => 350 + c * 190, ih = 76, oy = 250, iy = 740, SX0 = 336, SW = 1148, NL = 3, L0 = 370, LH = 100, LG = 15, LB = L0 + NL * (LH + LG) - LG, ax = 1512;
  frame('能否一次生成全部', K.purple, '一次 forward，{p:5 个 mask 位置}同时生成', '0.53、0.72：论文 Fig. 2 聊天，Qwen3-4B · 时间条为示意');
  // 输入行
  IN(t.inp, () => { TX('输入', SX0 - 16, iy + 50, font(700, 28), K.ink, 'right'); chip(C(0), iy, cw, ih, 'ctx', '当然', 34); }, .5, 0);
  for (let c = 1; c <= 5; c++) IN(t.mask + (c - 1) * t.maskDt, () => { R(C(c), iy, cw, ih, K.paper); box(C(c) + 2, iy + 2, cw - 4, ih - 4, 4, K.lg); TX('mask', C(c) + cw / 2, iy + 50, font(700, 32, HN), K.gray, 'center'); }, .4, 0);
  IN(t.cap, () => { RICH('{p:mask}：占位 token', ax, iy + 30, font(700, 28), K.ink); TX('该位置等待预测', ax, iy + 68, font(600, 28), K.ink2); }, .5, 0);
  for (let c = 0; c <= 5; c++) arrowP(C(c) + cw / 2, iy - 6, C(c) + cw / 2, LB + 6, S(t.up + c * .05, .3), c ? K.gray : K.ink, 5);
  // 5 层：每层每个位置一格；扫描时这一层的格子变实色
  const sc = S(t.scan, 1.8), cur = NL - 1 - Math.min(NL - 1, Math.floor(sc * NL)), scanning = T > t.scan && T < t.scan + 1.9, done = T >= t.scan + 1.9;
  for (let l = 0; l < NL; l++) { const y = L0 + l * (LH + LG);
    IN(t.lay + (NL - 1 - l) * .1, () => { const hot = scanning && l === cur, passed = done || (scanning && l > cur);
      R(SX0, y, SW, LH, '#e9e0fd');
      for (let c = 0; c <= 5; c++) R(C(c) + 10, y + 14, cw - 20, LH - 28, hot ? (c ? K.purple : K.ink) : passed ? (c ? '#a988f8' : K.gray) : (c ? '#cdb8fb' : K.lg));
      if (hot) for (let c = 0; c < 5; c++) line(C(c) + cw - 8, y + LH / 2, C(c + 1) + 8, y + LH / 2, K.purple, 4); }, .4, 0); }
  IN(t.lay + .3, () => { line(ax - 16, L0, ax - 16, LB, K.purple, 4); line(ax - 28, L0, ax - 16, L0, K.purple, 4); line(ax - 28, LB, ax - 16, LB, K.purple, 4);
    TX('drafter：3 层 MoE', ax, (L0 + LB) / 2 - 18, font(700, 34), K.purple); }, .5, 0);
  IN(t.ltxt, () => { TX('各位置相互可见', ax, (L0 + LB) / 2 + 30, font(600, 28), K.ink); TX('无 causal mask', ax, (L0 + LB) / 2 + 68, font(600, 28), K.ink2); }, .5, 0);
  // target 的 hidden state 进入每一层
  IN(t.tgt, () => { R(96, L0, 194, LB - L0, K.blue); TX('target', 193, (L0 + LB) / 2 - 8, font(700, 34, HN), '#fff', 'center'); TX('hidden state', 193, (L0 + LB) / 2 + 32, font(700, 28, HN), '#fff', 'center'); }, .5, 0);
  for (let l = 0; l < NL; l++) arrowP(292, L0 + l * (LH + LG) + LH / 2, SX0 - 2, L0 + l * (LH + LG) + LH / 2, S(t.tgt + .3 + (NL - 1 - l) * .06, .3), K.blue, 4);
  // 输出行
  for (let c = 1; c <= 5; c++) { arrowP(C(c) + cw / 2, L0 - 6, C(c) + cw / 2, oy + ih + 6, S(t.out + c * .04, .3), K.purple, 5);
    POP(t.out + .1 + c * .04, C(c), oy, cw, ih, () => chip(C(c), oy, cw, ih, 'word', WORDS[c - 1], 34), .4); }
  IN(t.out + .1, () => { TX('输出', SX0 - 16, oy + 50, font(700, 28), K.ink, 'right'); TX('5 个 draft token', ax, oy + 50, font(700, 28), K.purple); }, .5, 0);
  // 底部：贯穿的时间条。draft 段先是自回归的 1 层 × 5 次，再变成等宽的 5 层 × 1 次
  const bx = 330, by = 870, bh = 60, VW = 620, m = S(t.par, .8), DW = mix(350, 210, m);   // 自回归 1 层 × 5 次 = 5 格；并行 3 层 × 1 次 = 3 格
  IN(.3, () => { TX('一轮耗时', 96, by + 42, font(700, 30), K.ink);
    A(1 - m, () => { for (let i = 0; i < 5; i++) { R(bx + i * 70 * DW / 350 + 1, by, 70 * DW / 350 - 2, bh, '#a988f8'); TX('1 层', bx + (i * 70 + 35) * DW / 350, by + 40, font(700, 24), '#fff', 'center'); } TX('draft：1 层 × 5 次', bx, by - 14, font(700, 28), K.purple); });
    A(m, () => { for (let l = 0; l < 3; l++) R(bx + 1, by + l * 20, DW - 2, 18, K.purple); TX('draft：3 层 × 1 次', bx, by - 14, font(700, 28), K.purple); });
    R(bx + DW + 2, by, VW - 2, bh, K.blue); TX('验证：target 一次 forward', bx + DW + 22, by + 40, font(700, 28), '#fff'); }, .5, 0);
  // 第 1 位接受率
  const rx = 1350;
  IN(t.acc, () => { TX('第 1 位接受率（Qwen3 实验）', rx, by - 14, font(700, 28), K.ink);
    [['自回归', .53, '#a988f8', K.red], ['并行', .72, K.purple, K.teal]].forEach(([n, v, c, tc], i) => { const y = by + i * 34, w = 250 * v * S(t.acc + .2 + i * .5, .7);
      TX(n, rx, y + 23, font(700, 26), K.ink2); R(rx + 96, y, w, 26, c); TX(v.toFixed(2), rx + 96 + 250 * v + 12, y + 24, font(700, 30, HN), tc); }); }, .5, 0);
}

// ---------- 5 各抽各的 ----------
// 开头那个问题：target 会说“当然可以”或“没问题”，各一半。第 1、2 位各自的概率都对，但各抽一次就可能拼出“当然问题”
const BL = K.blue, YE = K.yel;
function distRows(x, y, rows, pick, t0) {   // rows: [词, 概率, 颜色]；pick 为被抽中的行
  rows.forEach(([w, p, c], i) => IN(t0 + i * .1, () => { const yy = y + i * 58;
    TX(w, x, yy + 36, font(700, 30), K.ink); R(x + 76, yy + 6, 120 * p * 2, 40, c); TX(`${Math.round(p * 100)}%`, x + 76 + 240 * p + 10, yy + 38, font(700, 26, HN), K.ink2);
    if (pick === i) box(x - 10, yy - 2, 300, 56, 4, K.purple); }, .4, 0));
}
function sIndep() {
  frame('第 2 位能否看到第 1 位的结果', K.red, '并行的代价：各位置{r:独立采样}', '例子取自论文 §3.1 · 概率为示意 · 0.72、0.65：论文 Fig. 2 聊天（按图读数）');
  // 来历：开头那个问题
  IN(.5, () => { R(96, 290, 470, 60, K.paper2); TX('用户：能帮我查一下吗？', 116, 330, font(600, 28), K.ink2); }, .5, 0);
  IN(1.0, () => { TX('target 的两种回答', 96, 410, font(700, 28), K.ink);
    R(96, 430, 320, 62, BL); TX('当然可以…', 116, 472, font(700, 32), '#fff'); TX('50%', 430, 472, font(700, 28, HN), K.ink2);
    R(96, 506, 320, 62, YE); TX('没问题…', 116, 548, font(700, 32), K.ink); TX('50%', 430, 548, font(700, 28, HN), K.ink2); }, .5, 0);
  // 两个位置与各自的概率
  const s1 = 640, s2 = 960, sy = 300, sw = 240, sh = 86, s3 = [1280, 1440, 1600];
  const p1 = T > 6.0 ? 0 : -1, p2 = T > 7.0 ? 1 : -1, ver = T > 8.2;
  IN(1.8, () => { chip(s1, sy, sw, sh, ver ? 'ok' : p1 === 0 ? 'word' : 'guess', p1 === 0 ? '当然' : '第 1 位', 32); chip(s2, sy, sw, sh, ver ? 'bad' : p2 === 1 ? 'word' : 'guess', p2 === 1 ? '问题' : '第 2 位', 32);
    s3.forEach((x) => chip(x, sy, 140, sh, T > 8.8 ? 'void' : 'guess', T > 8.8 ? '丢弃' : null, 28)); }, .5, 0);
  distRows(s1 + 10, 430, [['当然', .5, BL], ['没', .5, YE]], p1 === 0 ? 0 : -1, 2.2);
  distRows(s2 + 10, 430, [['可以', .5, BL], ['问题', .5, YE]], p2 === 1 ? 1 : -1, 2.5);
  IN(3.4, () => TX('每个位置的概率均正确', s1, 600, font(700, 30), K.teal), .5, 0);
  IN(4.4, () => { line(s1 + sw + 8, sy + sh / 2, s2 - 8, sy + sh / 2, K.red, 3, [8, 6]); TX('✗', (s1 + sw + s2) / 2, sy + sh / 2 - 10, font(700, 34), K.red, 'center'); TX('互相看不到采样结果', (s1 + sw + s2) / 2, sy + sh + 4 + 28, font(700, 26), K.red, 'center'); }, .5, 0);
  IN(8.8, () => TX('验证在第一个错误处停止，其后全部丢弃', 1280, sy + sh + 42, font(600, 26), K.ink2), .5, 0);
  // 四种组合
  // 把拼接错误换算成每个位置的接受率，再与实测上下对齐：两组都是第 2 位低于第 1 位
  const ax = 640, gy = 660, BW = 520, bar = (y, n, v, c, tc, t0) => { TX(n, ax, y + 30, font(700, 30), K.ink); R(ax + 120, y, BW * v * S(t0, .6), 36, c); TX(`${Math.round(v * 100)}%`, ax + 120 + BW * v + 14, y + 31, font(700, 32, HN), tc); };
  IN(10.2, () => { TX('这个例子的接受率', ax, gy - 18, font(700, 28), K.ink2); bar(gy, '第 1 位', 1, K.purple, K.ink, 10.3); bar(gy + 46, '第 2 位', .5, K.red, K.red, 10.6); }, .5, 0);
  IN(11.0, () => TX('少的一半：拼接错误', ax + 120 + BW * .5 + 110, gy + 76, font(700, 28), K.red), .5, 0);
  IN(11.6, () => { TX('DFlash 实测（Qwen3，聊天）', ax, gy + 136, font(700, 28), K.ink2); bar(gy + 154, '第 1 位', .72, K.purple, K.ink, 11.7); bar(gy + 200, '第 2 位', .65, K.red, K.red, 12.0); }, .5, 0);
}

// ---------- 6（重画）查表：接着第 5 幕的例子，查一行，概率条当场变化 ----------
// 左：第 1 位采到的 token 与 backbone 给第 2 位的概率（各占一半，同第 5 幕）；中：加分表，每个 token 一行；右：查表之后第 2 位的概率与采样结果。
// 先走“当然”这一支，再走“没”这一支，两支都接得上；最后一行是 6 个 token 逐位查表
const MKB = beatTimes(6, [.5, .4, .4, .4]), MK_DUR = MKB[MKB.length - 1].e + 1.8, MK_NO = () => 1e9;   // 四句：缺什么 → 表 → 查“当然”一行 → 逐位查表与名字。“没”这一支（MK_NO）不入片
function sMarkovB() {
  const b = (i, f) => at(MKB, i, f), br = T >= MK_NO() ? 1 : 0, adj = br ? S(MK_NO(0, .45), .8) : S(b(2, .5), .8), pOk = br ? mix(.5, .1, adj) : mix(.5, .9, adj);
  frame('第 2 位缺少的信息', K.purple, '第 2 位只缺少一条信息：{p:第 1 位的采样结果}', '加分与概率为示意', .15, b(0, .3));
  const head = (s, x, y) => TX(s, x, y, font(700, 28), K.ink2);
  const bars = (x, y, p) => [['可以', p, BL], ['问题', 1 - p, YE]].forEach(([w, v, c], i) => { const yy = y + i * 70; TX(w, x, yy + 38, font(700, 34), K.ink); R(x + 96, yy, 300 * v, 52, c); TX(`${Math.round(v * 100)}%`, x + 96 + 300 * v + 12, yy + 39, font(700, 32, HN), K.ink); });
  // 左：第 1 位与查表之前的第 2 位
  const c1 = [96, 300, 220, 80];
  IN(b(0), () => { head('第 1 位采到', 96, 280); if (!br) chip(...c1, 'word', '当然', 36); }, .5, 0);
  if (br) POP(MK_NO(), ...c1, () => chip(...c1, 'word', '没', 36), .4);
  IN(b(0, .5), () => { head('第 2 位：base logits 对应的概率', 96, 452); bars(96, 472, .5); }, .5, 0);
  // 中：加分表
  const tx = 640, lw = 150, cw = 190, rh = 70, ty = 330, ROWS = [['当然', [1, -1]], ['没', [-1, 1]], ['…', [0, 0]]], hot = T >= b(3) ? -1 : T >= MK_NO(0, .2) ? 1 : T >= b(2, .2) ? 0 : -1;
  IN(b(1), () => { TX(T >= b(3, .5) ? 'Markov head：每个 token 一行' : 'transition bias：每个 token 一行', tx, 280, font(700, 28), T >= b(3, .5) ? K.purple : K.ink2);
    ['“可以”的 logit', '“问题”的 logit'].forEach((c, j) => TX(c, tx + lw + j * (cw + 10) + cw / 2, ty - 14, font(700, 24), K.ink2, 'center'));
    ROWS.forEach(([r, vs], i) => { const y = ty + i * (rh + 10), dim = hot >= 0 && hot !== i; A(dim ? .3 : 1, () => { R(tx, y, lw - 10, rh, K.paper2); TX(r, tx + (lw - 10) / 2, y + 46, font(700, 32), K.ink, 'center');
      vs.forEach((v, j) => { const x = tx + lw + j * (cw + 10); if (!v) { R(x, y, cw, rh, K.paper2); TX('…', x + cw / 2, y + 44, font(700, 30), K.gray, 'center'); } else cell(x, y, cw, rh, v > 0 ? K.teal : K.red, v > 0 ? '↑ +1.1' : '↓ −1.1', 32, '#fff', HN); }); });
      if (hot === i) box(tx - 6, y - 6, lw + 2 * cw + 10 + 12, rh + 12, 5, K.purple); }); }, .5, 0);
  // 查哪一行：从第 1 位指到那一行；再从那一行指到右侧的概率
  const rowY = (i) => ty + i * (rh + 10) + rh / 2, t1 = br ? MK_NO(0, .15) : b(2, .15);
  if (T < b(3)) { arrowP(c1[0] + c1[2] + 10, c1[1] + c1[3] / 2, tx - 14, rowY(br), S(t1, .4), K.purple, 5); arrowP(tx + lw + 2 * cw + 24, rowY(br), 1290, 520, S(t1 + .5, .4), K.purple, 5); }
  // 右：查表之后的第 2 位
  const c2 = [1300, 300, 220, 80], t2 = br ? MK_NO(0, .75) : b(2, .82);
  IN(b(2, .35), () => { head('第 2 位：加上 bias 之后', 1300, 452); bars(1300, 472, pOk); }, .5, 0);
  IN(b(2, .8), () => head('第 2 位采到', 1300, 280), .4, 0);
  POP(t2, ...c2, () => chip(...c2, 'word', br ? '问题' : '可以', 36), .4);
  // 两支的结果
  POP(b(2, .92), 640, 650, 300, 76, () => chip(640, 650, 300, 76, 'ok', '当然可以', 34), .4);
  POP(MK_NO(0, .88), 960, 650, 300, 76, () => chip(960, 650, 300, 76, 'ok', '没问题', 34), .4);
  IN(b(2, .97), () => TX('不再出现“当然问题”', 960, 698, font(700, 30), K.teal), .5, 0);
  // 之后每一位都查前一位的那一行
  const W6 = ['当然', ...WORDS], qx = (i) => 640 + i * 170, qy = 800;
  IN(b(3) - .2, () => TX('每一位加上前一位对应的一行', 96, qy + 48, font(700, 30), K.ink), .5, 0);
  W6.forEach((w, i) => { const t0 = b(3, .08 + i * .1); POP(t0, qx(i), qy, 150, 72, () => chip(qx(i), qy, 150, 72, i ? 'word' : 'ctx', w, 32), .35);
    if (i && T > t0 - .1) A(P(t0 - .1, .3), () => { X.strokeStyle = K.purple; X.lineWidth = 4; X.beginPath(); X.moveTo(qx(i - 1) + 95, qy - 4); X.quadraticCurveTo(qx(i) - 10, qy - 34, qx(i) + 55, qy - 4); X.stroke(); }); });
}
const MK_EV = (() => { const b = (i, f) => at(MKB, i, f); return [[.1, 'click'], [b(0), 'pop'], [b(1), 'whoosh'], [b(2, .2), 'tick'], [b(2, .5), 'whoosh'], [b(2, .82), 'land'], [b(2, .92), 'pop'], ...[0, 1, 2, 3, 4, 5].map((i) => [b(3, .08 + i * .1), 'tick'])]; })();

// ---------- 6 查表 ----------
// 第 2 位缺的只是“第 1 位抽到了什么”。保留大网络的分数，按前一个 token 查一行加分：可以 +1.1、问题 −1.1，50% 变 90%
const TBL = { cols: ['可以', '问题', '呢', '…'], rows: [['当然', ['+1.1', '−1.1', '+0.2', '…']], ['没', ['−1.2', '+1.3', '−0.1', '…']], ['可以', ['−0.3', '−0.8', '+0.4', '…']], ['…', ['…', '…', '…', '…']]] };
function sMarkov() {
  frame('第 2 位缺少的信息', K.purple, '第 2 位只缺少一条信息：{p:前一个 token 的采样结果}', '分数为示意');
  const sy = 330, sw = 180, sh = 76, SXM = (i) => 96 + i * 224, filled = (i) => i === 0 || T > [0, 10.0, 11.6, 12.6, 13.6][i];
  WORDS.slice(0, 4).forEach((w, j) => { const i = j + 1; IN(.4 + i * .06, () => chip(SXM(i), sy, sw, sh, filled(i) ? 'word' : 'guess', filled(i) ? w : `第 ${i + 1} 位`, 30), .4, 0); });
  IN(.3, () => chip(SXM(0), sy, sw, sh, 'word', '当然', 30), .4, 0);
  // 被否定的办法
  IN(2.4, () => { R(1240, 330, 584, 76, K.paper2); TX('重新运行 backbone，使其看到“当然”？', 1262, 378, font(700, 26), K.ink2); if (T > 3.4) { line(1252, 368, 1812, 368, K.red, 4); TX('即退回自回归', 1812, 434, font(700, 24), K.red, 'right'); } }, .5, 0);
  // 加分表：在第 1 位正下方
  const tx = 96, ty = 470, lw = 110, cw = 118, rh = 58;
  IN(4.4, () => { TX('一张加分表：行为前一个 token，列为下一个 token', tx + 130, ty - 22, font(700, 26), K.ink);
    TBL.cols.forEach((c, j) => TX(c, tx + lw + j * cw + cw / 2, ty + 36, font(700, 26), K.ink2, 'center'));
    TBL.rows.forEach(([r, vals], i) => { const y = ty + 52 + i * rh, hot = i === 0 && T > 6.0; R(tx, y, lw + 4 * cw, rh - 6, hot ? '#e9e0fd' : K.paper2);
      TX(r, tx + 20, y + 36, font(700, 26), hot ? K.purple : K.ink); vals.forEach((v, j) => TX(v, tx + lw + j * cw + cw / 2, y + 36, font(700, 26, HN), hot && j < 2 ? K.purple : K.ink2, 'center')); }); }, .5, 0);
  IN(6.0, () => { box(tx - 4, ty + 48, lw + 4 * cw + 8, rh + 2, 4, K.purple); }, .3, 0);
  arrowP(SXM(0) + sw / 2, sy + sh + 6, SXM(0) + sw / 2, ty + 46, S(5.6, .4), K.purple, 4);
  // 第 2 位的分数
  const px = 900, b0 = 1010, U = 150, adj = S(7.8, 1.0), y1 = 490, y2 = 584;
  IN(1.0, () => { TX('第 2 位的 logits（backbone 一次给出）', px, 450, font(700, 28), K.ink);
    [[y1, '可以', 1], [y2, '问题', -1]].forEach(([y, w, sg]) => { TX(w, px, y + 40, font(700, 30), K.ink); R(b0, y, 2.0 * U, 56, K.lg);
      if (sg > 0) R(b0 + 2 * U, y, 1.1 * U * adj, 56, K.purple); else HR(b0 + 2 * U - 1.1 * U * adj, y, 1.1 * U * adj, 56, K.red, K.paper);
      const sc = 2 + sg * 1.1 * adj; TX(sc.toFixed(1), b0 + (sg > 0 ? sc : 2) * U + 14, y + 40, font(700, 30, HN), K.ink2);
      const pr = 1 / (1 + Math.exp(-sg * 2.2 * adj)); TX(`${Math.round(pr * 100)}%`, 1780, y + 42, font(700, 40, HN), sg > 0 && adj > .5 ? K.purple : K.ink2, 'right'); }); }, .5, 0);
  IN(1.0, () => TX('概率', 1780, 478, font(600, 26), K.ink2, 'right'), .5, 0);
  arrowP(tx + lw + 4 * cw + 10, ty + 78, b0 + 2 * U - 10, y1 + 28, S(7.0, .6), K.purple, 3);
  arrowP(tx + lw + 4 * cw + 10, ty + 78, b0 + 2 * U - 10, y2 + 28, S(7.2, .6), K.red, 3);
  // 之后每一位都查前一位那一行
  [2, 3, 4].forEach((i) => { const t0 = [0, 0, 11.0, 12.0, 13.0][i], hx0 = SXM(i - 1) + sw / 2, hx1 = SXM(i) + sw / 2, p = S(t0, .5);
    if (p > 0) { X.strokeStyle = K.purple; X.lineWidth = 3; X.beginPath(); const steps = 24; for (let s = 0; s <= steps * p; s++) { const f = s / steps, xx = mix(hx0, hx1, f), yy = sy - 8 - Math.sin(Math.PI * f) * 26; s ? X.lineTo(xx, yy) : X.moveTo(xx, yy); } X.stroke();
      A(p, () => TX('查一行', (hx0 + hx1) / 2, sy - 42, font(700, 24), K.purple, 'center')); } });
  IN(14.4, () => RICH('这张表称为 {p:Markov head}：只依赖前一个 token', px, 760, font(700, 34), K.ink), .5, 0);
}
