// EP 通信与 PP 影片的画面：每幕一个函数，u 为幕内时间（秒）。文案与节拍在 script.js，真实 trace 在 data.js（data.py 生成）。
// 贯穿全片的图形是集群图：机器是竖长卡片，8 张 GPU 一行一张；机器内 NVLink 为左侧粗竖条，网卡在右侧，经上方的网络横线互连。
// 线宽按真实带宽比例：NVLink 160 GB/s 对网卡 50 GB/s，16 : 5。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const I = Object.fromEntries(SCRIPT.map((s, i) => [s.id, i]));
const EV = SCRIPT.map(() => []);
const ev = (k, u, kind) => EV[k].push([u, kind]);
const fade = (u, a, d = 0.45) => po(u, a, a + d);
const FILM = 'EP 通信与 PP';
const MB = [C.yel, '#7cbf3c'];                      // 两个 micro-batch 的颜色（与 DeepSeek 原图一致）
const NVC = '#4a4a46', IBC = '#9a927f';               // NVLink 与网络的线色

loadImg('normal', '../../src/assets/ep-pp-communication/deepep-normal.png');
loadImg('heat', '../../src/assets/ep-pp-communication/lmsys-eplb-stat.png');
loadImg('pfill', '../../src/assets/ep-pp-communication/profile-prefill.jpg');
loadImg('pdec', '../../src/assets/ep-pp-communication/profile-decode.jpg');

function header(k, u) {
  const s = SCRIPT[k], a = fade(u, 0, 0.5);
  txt(FILM, 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 76, { size: 22, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 222 + (1 - a) * 16, 54, 700, a);
}
const note = (s, x, y, o = {}) => txt(s, x, y, { size: 22, weight: 500, color: C.gray, ...o });
const lab = (s, x, y, o = {}) => txt(s, x, y, { size: 26, weight: 700, ...o });
function img(name, x, y, w, alpha = 1) {
  const im = IMG[name]; if (!im || !im.naturalWidth || alpha <= 0.003) return 0;
  const h = w * im.naturalHeight / im.naturalWidth;
  g.globalAlpha = alpha; g.drawImage(im, x, y, w, h); g.globalAlpha = 1; sr(x, y, w, h, C.lg, 2, alpha);
  return h;
}
function rrect(x, y, w, h, r, fill, alpha = 1, stroke) {
  if (alpha <= 0.003 || w <= 0) return;
  r = Math.min(r, w / 2, h / 2); g.globalAlpha = alpha; g.beginPath(); g.roundRect(x, y, w, h, r);
  if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = 2; g.stroke(); } g.globalAlpha = 1;
}
function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function polyAt(pts, k) {
  const L = []; let tot = 0;
  for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); L.push(d); tot += d; }
  let s = clamp(k) * tot;
  for (let i = 0; i < L.length; i++) { if (s <= L[i] || i === L.length - 1) { const f = L[i] ? s / L[i] : 0; return [mix(pts[i][0], pts[i + 1][0], clamp(f)), mix(pts[i][1], pts[i + 1][1], clamp(f))]; } s -= L[i]; }
  return pts[pts.length - 1];
}
function polyline(pts, col, w, alpha = 1, dash) {
  if (alpha <= 0.003) return;
  g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = w; g.setLineDash(dash || []); g.lineJoin = 'round';
  g.beginPath(); pts.forEach((q, i) => (i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]))); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
}
function chip(x, y, col, alpha = 1, s = 1) { rrect(x - 19 * s, y - 7 * s, 38 * s, 14 * s, 4, col, alpha, C.ink); }

// ---------- 4 台机器的集群图（EP32） ----------
const CL = { x0: 170, pitch: 440, w: 310, y: 350, bus: 316, row0: 404, rp: 58, rh: 48 };
const cardX = (n) => CL.x0 + n * CL.pitch;
const riserX = (n) => cardX(n) + CL.w + 36;
const cell = (n, i) => ({ x: cardX(n) + 44, y: CL.row0 + i * CL.rp, w: 250, h: CL.rh });
const slot = (n, i, s) => ({ x: cardX(n) + 44 + 68 + s * 19, y: CL.row0 + i * CL.rp + 16, s: 15 });
const SRC = [0, 5];
const TGT = [[0, 1, 3], [0, 6, 0], [1, 2, 1], [1, 2, 6], [1, 7, 4], [3, 0, 2], [3, 4, 7], [3, 5, 5]];
function cluster(alpha = 1, { dimCells = 0, redundant = 0 } = {}) {
  if (alpha <= 0.003) return;
  line(riserX(0) - 20, CL.bus, riserX(3), CL.bus, IBC, 5, alpha);
  txt('机器之间的网络（每张 GPU 一张网卡，50 GB/s）', riserX(0), CL.bus - 14, { size: 20, weight: 600, color: C.gray, alpha });
  for (let n = 0; n < 4; n++) {
    const x = cardX(n);
    rr(x, CL.y, CL.w, 520, '#faf7f0', alpha); sr(x, CL.y, CL.w, 520, C.lg, 2, alpha);
    txt(`机器 ${n}`, x + 18, CL.y + 36, { size: 24, weight: 700, alpha });
    txt('8 × H800', x + CL.w - 16, CL.y + 36, { size: 18, weight: 500, color: C.gray, align: 'right', alpha });
    rr(x + 16, CL.row0, 16, 7 * CL.rp + CL.rh, NVC, alpha * 0.85);
    line(riserX(n), CL.bus, riserX(n), CL.row0 + 7 * CL.rp + CL.rh / 2, IBC, 5, alpha);
    for (let i = 0; i < 8; i++) {
      const c = cell(n, i);
      line(c.x + c.w, c.y + c.h / 2, riserX(n), c.y + c.h / 2, IBC, 5, alpha);
      line(x + 32, c.y + c.h / 2, c.x, c.y + c.h / 2, NVC, 6, alpha * 0.85);
      rr(c.x, c.y, c.w, c.h, C.paper2, alpha * (1 - dimCells * 0.5));
      txt(`GPU ${i}`, c.x + 10, c.y + 31, { size: 18, weight: 600, color: C.ink2, alpha });
      for (let s = 0; s < 9; s++) { const q = slot(n, i, s); if (s < 8) rr(q.x, q.y, q.s, q.s, '#bdb5a3', alpha); else sr(q.x + 1, q.y + 1, q.s - 2, q.s - 2, redundant ? C.red : '#bdb5a3', 2, alpha, [3, 2]); }
    }
  }
  txt('NVLink 160 GB/s', cardX(0) + 8, CL.y + 548, { size: 18, weight: 600, color: NVC, alpha });
}
// 远程路径：GPU 右侧 → 本机网卡竖线 → 网络 → 目标机竖线 → 目标 GPU 右侧；本地路径：沿 NVLink 竖条
function pathRemote(n0, i0, n1, i1, lane = 0) {
  const a = cell(n0, i0), b = cell(n1, i1), ya = a.y + a.h / 2, yb = b.y + b.h / 2;
  return [[a.x + a.w, ya], [riserX(n0) - 6 + lane * 4, ya], [riserX(n0) - 6 + lane * 4, CL.bus + 4 + lane * 3], [riserX(n1) + 6 - lane * 4, CL.bus + 4 + lane * 3], [riserX(n1) + 6 - lane * 4, yb], [b.x + b.w, yb]];
}
function pathLocal(n, i0, i1) { const a = cell(n, i0), b = cell(n, i1); return [[a.x, a.y + a.h / 2], [cardX(n) + 24, a.y + a.h / 2], [cardX(n) + 24, b.y + b.h / 2], [b.x, b.y + b.h / 2]]; }
const pathTo = (n0, i0, n1, i1, lane) => (n0 === n1 ? pathLocal(n0, i0, i1) : pathRemote(n0, i0, n1, i1, lane));
const slotC = (n, i, s) => { const q = slot(n, i, s); return [q.x + q.s / 2, q.y + q.s / 2]; };
function srcToken(alpha = 1, col = C.blue) { const c = cell(...SRC); rr(c.x - 2, c.y - 2, c.w + 4, c.h + 4, col, alpha * 0.18); sr(c.x - 2, c.y - 2, c.w + 4, c.h + 4, col, 3, alpha); }
function targets(alpha, col = C.blue) { for (const [n, i, s] of TGT) { const q = slot(n, i, s); rr(q.x - 2, q.y - 2, q.s + 4, q.s + 4, col, alpha); } }

// ---------- 开场 ----------
function bigTitle(out, alpha = 1) {
  const size = mix(104, 28, out);
  txt(FILM, 96, mix(240, 76, out), { size, weight: mix(800, 700, out), color: out > 0.5 ? C.ink2 : C.ink, alpha });
  txt('跨机推理时网络上传输的数据', 100, 330, { size: 40, weight: 600, color: C.ink2, alpha: alpha * (1 - out) });
}
const S1 = { fill: [1.0, 5.6], out: [11.6, 13.0] };
ev(I.open, 0.3, 'land'); ev(I.open, 4.4, 'pop'); ev(I.open, 7.0, 'land'); ev(I.open, S1.out[0], 'whoosh');
function machine(x, y, label, fillGB, alpha, over = false) {
  rr(x, y, 300, 560, '#faf7f0', alpha); sr(x, y, 300, 560, C.lg, 2, alpha);
  txt(label, x + 18, y + 38, { size: 26, weight: 700, alpha }); txt('8 × H800，每张 80 GB', x + 18, y + 68, { size: 18, weight: 500, color: C.gray, alpha });
  for (let i = 0; i < 8; i++) {
    const yy = y + 90 + i * 57, f = clamp((fillGB - i * 80) / 80);
    rr(x + 18, yy, 264, 46, C.paper2, alpha); rr(x + 18, yy, 264 * f, 46, over ? C.red : C.purple, alpha * 0.85);
    txt(`GPU ${i}`, x + 30, yy + 30, { size: 18, weight: 600, color: f > 0.2 ? '#fff' : C.ink2, alpha });
  }
}
function s1(u) {
  const out = p(u, ...S1.out), keep = 1 - out, a = fade(u, 0.1, 0.6);
  bigTitle(out, a);
  const gb = 689 * lin(u, ...S1.fill), m1 = Math.min(gb, 640), m2 = Math.max(0, gb - 640);
  machine(1000, 330, '机器 0', m1, keep * fade(u, 0.3));
  txt(`${Math.round(gb)} / 689 GB`, 1000, 950, { size: 34, weight: 700, color: gb > 640 ? C.red : C.ink, alpha: keep * fade(u, 0.8) });
  note('DeepSeek-V3 的 FP8 权重；一台机器共 640 GB', 1000, 990, { alpha: keep * fade(u, 0.8) });
  const a2 = keep * fade(u, 4.4);
  if (a2 > 0) {
    machine(1460, 330, '机器 1', m2, a2, true);
    line(1300, 320, 1460, 320, IBC, 5, a2); txt('网卡', 1380, 306, { size: 20, weight: 600, color: C.gray, align: 'center', alpha: a2 });
  }
}

// ---------- 按 expert 切 ----------
const S2 = { hi: bt(I.route, 1), go: bt(I.route, 2), back: bt(I.route, 2) + 3.4, all: bt(I.route, 3) };
ev(I.route, 0.4, 'land'); ev(I.route, S2.hi, 'pop'); TGT.forEach((_, j) => ev(I.route, S2.go + 0.1 * j, 'tick')); TGT.forEach((_, j) => ev(I.route, S2.back + 0.1 * j, 'click')); ev(I.route, S2.all, 'whoosh');
const FLOW = (() => { const r = rng(7), hot = Array.from({ length: 32 }, () => 0.5 + r() * r() * 1.6); return Array.from({ length: 32 }, (_, i) => Array.from({ length: 32 }, (_, j) => hot[j] * (0.55 + r() * 0.9))); })();
const FLOWMAX = Math.max(...FLOW.flat());
function sRoute(u) {
  header(I.route, u);
  const sc = mix(1, 0.62, p(u, S2.all, S2.all + 1.2));
  g.save(); g.translate(CL.x0, 300); g.scale(sc, sc); g.translate(-CL.x0, -300);
  cluster(fade(u, 0.2));
  srcToken(fade(u, 1.5));
  targets(fade(u, S2.hi) * (1 - 0.6 * fade(u, S2.go + 1.8)), C.blue);
  // dispatch：8 份同时出发
  TGT.forEach(([n, i, s], j) => {
    const path = [...pathTo(SRC[0], SRC[1], n, i, j % 3), slotC(n, i, s)], k = p(u, S2.go + 0.1 * j, S2.go + 1.6 + 0.1 * j);
    polyline(path, C.blue, 3, fade(u, S2.go) * (1 - fade(u, S2.back)) * 0.7);
    if (k > 0 && k < 1) { const [x, y] = polyAt(path, k); chip(x, y, C.blue); }
    const q = slot(n, i, s), comp = fade(u, S2.go + 1.7) * (1 - fade(u, S2.back + 1.5));
    rr(q.x - 2, q.y - 2, q.s + 4, q.s + 4, C.yel, comp);
    const kb = p(u, S2.back + 0.1 * j, S2.back + 1.6 + 0.1 * j), back = [...path].reverse();
    polyline(back, C.purple, 3, fade(u, S2.back) * (1 - fade(u, S2.all)) * 0.7);
    if (kb > 0 && kb < 1) { const [x, y] = polyAt(back, kb); chip(x, y, C.purple); }
  });
  const c = cell(...SRC), aSum = fade(u, S2.back + 1.9);
  txt('Σ 加权求和', c.x + c.w - 8, c.y + 31, { size: 18, weight: 700, color: C.purple, align: 'right', alpha: aSum * (1 - fade(u, S2.all)) });
  // 其余所有卡同时收发
  if (u > S2.all) { const r = rng(11); for (let m = 0; m < 70; m++) { const n0 = (r() * 4) | 0, i0 = (r() * 8) | 0, n1 = (r() * 4) | 0, i1 = (r() * 8) | 0, ph = r(); const k = ((u - S2.all) * 0.55 + ph) % 1; const [x, y] = polyAt(pathTo(n0, i0, n1, i1, m % 3), k); chip(x, y, m % 2 ? C.blue : C.purple, fade(u, S2.all + 0.3) * 0.75, 0.7); } }
  g.restore();
  const aL = fade(u, S2.go + 0.3) * (1 - fade(u, S2.all));
  lab('dispatch：发往 expert 所在的卡', 170, 960, { color: C.blue, alpha: aL }); lab('combine：结果发回，加权求和', 900, 960, { color: C.purple, alpha: fade(u, S2.back) * (1 - fade(u, S2.all)) });
  note('router 为这个 token 选出 8 个 expert（示意）', 170, 994, { alpha: fade(u, S2.hi) * (1 - fade(u, S2.all)) });
  // 流量矩阵
  const aM = fade(u, S2.all + 1.0);
  if (aM > 0) {
    const mx = 1290, my = 380, cs = 15;
    for (let i = 0; i < 32; i++) for (let j = 0; j < 32; j++) rr(mx + j * cs, my + i * cs, cs - 1, cs - 1, mixc(C.paper2, C.blue, FLOW[i][j] / FLOWMAX), aM);
    txt('发送方 GPU ↓   接收方 GPU →', mx, my - 16, { size: 20, weight: 600, color: C.ink2, alpha: aM });
    txt('all-to-all', mx, my + 32 * cs + 46, { size: 34, weight: 800, color: C.blue, alpha: fade(u, S2.all + 2.6) });
    note('每格是一对卡之间的 token 数，每层每步都在变（示意）', mx, my + 32 * cs + 82, { alpha: aM, size: 20 });
    lab('每个 MoE 层两次，DeepSeek-V3 有 58 个 MoE 层', 170, 760, { alpha: fade(u, S2.all + 3.2), size: 26 });
  }
}

// ---------- 要传多少 ----------
const S3 = { tok: 0.5, pipe: bt(I.volume, 1), tr: bt(I.volume, 2), way: bt(I.volume, 3) };
ev(I.volume, 0.6, 'land'); ev(I.volume, 2.6, 'pop'); ev(I.volume, 4.4, 'pop'); ev(I.volume, S3.pipe, 'whoosh'); ev(I.volume, S3.tr, 'land'); ev(I.volume, S3.way, 'click');
const busy = (items, s) => items.filter((x) => x.s === s).reduce((a, x) => a + x.d, 0);
function sVolume(u) {
  header(I.volume, u);
  // token 的拆解
  const a0 = fade(u, S3.tok);
  lab('一个 token：7168 个数', 96, 320, { alpha: a0 });
  rr(96, 340, 860, 26, C.blue, a0);
  const a1 = fade(u, 2.4);
  lab('去程 FP8：每个数 1 字节，≈ 7 KB × 8 份 = {57 KB}'.replace('{57 KB}', '57 KB'), 96, 420, { alpha: a1, color: C.blue });
  for (let j = 0; j < 8; j++) rr(96 + j * 108, 436, 100, 14, C.blue, a1 * fade(u, 2.6 + j * 0.08));
  const a2 = fade(u, 4.2);
  lab('回程 BF16：每个数 2 字节，× 8 份 = 115 KB', 96, 500, { alpha: a2, color: C.purple });
  for (let j = 0; j < 8; j++) rr(96 + j * 108, 516, 100, 28, C.purple, a2 * fade(u, 4.4 + j * 0.08));
  // 两条管道
  const ap = fade(u, S3.pipe);
  lab('每张卡的带宽', 1100, 320, { alpha: ap });
  rr(1100, 350, 720, 64, NVC, ap * 0.9); txt('NVLink（机器内）160 GB/s', 1120, 392, { size: 24, weight: 700, color: '#fff', alpha: ap });
  rr(1100, 452, 720, 20, IBC, ap); txt('网卡（机器之间）50 GB/s', 1100, 508, { size: 24, weight: 700, color: C.ink2, alpha: ap });
  note('线宽按带宽的真实比例', 1100, 546, { alpha: ap });
  // 真实 trace：一层两个 micro-batch，两条 stream 的累计时长
  const at = fade(u, S3.tr);
  if (at > 0) {
    const comp = TRACE.stats.prefill.layer_compute_ms, comm = TRACE.stats.prefill.layer_comm_ms, px = 700 / 34;
    lab('prefill 每层（两个 micro-batch，各 8192 个 token），kernel 时长之和', 96, 640, { alpha: at });
    const k = p(u, S3.tr + 0.3, S3.tr + 2.3);
    rr(300, 670, comp * px * k, 52, C.yel, at); txt('计算', 96, 706, { size: 26, weight: 700, alpha: at }); txt(`${comp.toFixed(1)} ms`, 310 + comp * px * k, 706, { size: 26, weight: 700, alpha: at * k });
    rr(300, 744, comm * px * k, 52, C.blue, at); txt('通信', 96, 780, { size: 26, weight: 700, alpha: at }); txt(`${comm.toFixed(1)} ms`, 310 + comm * px * k, 780, { size: 26, weight: 700, alpha: at * k });
    note('通信：dispatch、combine 与数量交换；计算：attention、expert 与其余 kernel', 96, 836, { alpha: at });
  }
  const aw = fade(u, S3.way);
  lab('两个优化方向', 1100, 640, { alpha: aw, color: C.red });
  lab('① 减少经过慢速链路的数据', 1100, 700, { alpha: aw }); lab('② 通信与计算重叠', 1100, 760, { alpha: fade(u, S3.way + 0.6) });
}

// ---------- 少走网卡 ----------
const S4 = { naive: 0.5, fwd: bt(I.forward, 1), bw: bt(I.forward, 2), name: bt(I.forward, 3) };
const REMOTE = TGT.filter(([n]) => n !== SRC[0]);
ev(I.forward, 0.4, 'land'); REMOTE.forEach((_, j) => ev(I.forward, 1.6 + 0.35 * j, 'tick')); ev(I.forward, S4.fwd + 0.6, 'land'); ev(I.forward, S4.fwd + 2.6, 'land'); ev(I.forward, S4.bw, 'pop'); ev(I.forward, S4.name, 'click');
function sForward(u) {
  header(I.forward, u);
  cluster(fade(u, 0.1)); srcToken(1); targets(0.85);
  const off = fade(u, S4.fwd - 0.4);
  // 直接发送：6 份各自经过网卡
  REMOTE.forEach(([n, i, s], j) => {
    const path = [...pathRemote(SRC[0], SRC[1], n, i, j % 4), slotC(n, i, s)], k = p(u, 1.5 + 0.35 * j, 3.3 + 0.35 * j);
    polyline(path, C.blue, 3, fade(u, 1.5 + 0.35 * j) * (1 - off) * 0.55);
    if (k > 0 && k < 1) { const [x, y] = polyAt(path, k); chip(x, y, C.blue, 1 - off); }
  });
  // 转发：每台机器一份，到同编号的 GPU 5，再沿 NVLink 分发
  const hubs = [1, 3];
  hubs.forEach((n, j) => {
    const path = pathRemote(SRC[0], SRC[1], n, 5, j), k = p(u, S4.fwd + 0.3, S4.fwd + 2.2), a = fade(u, S4.fwd);
    polyline(path, C.blue, 5, a * 0.8);
    if (k > 0 && k < 1) { const [x, y] = polyAt(path, k); chip(x, y, C.blue); }
    const hub = cell(n, 5); sr(hub.x - 2, hub.y - 2, hub.w + 4, hub.h + 4, C.blue, 3, fade(u, S4.fwd + 2.2));
    TGT.filter(([m, i]) => m === n && i !== 5).forEach(([m, i, s], q) => {
      const lp = [...pathLocal(n, 5, i), slotC(n, i, s)], kk = p(u, S4.fwd + 2.4 + 0.15 * q, S4.fwd + 3.6 + 0.15 * q);
      polyline(lp, C.blue, 3, fade(u, S4.fwd + 2.4) * 0.8);
      if (kk > 0 && kk < 1) { const [x, y] = polyAt(lp, kk); chip(x, y, C.blue, 1, 0.8); }
    });
  });
  TGT.filter(([n]) => n === SRC[0]).forEach(([n, i, s]) => polyline([...pathLocal(n, SRC[1], i), slotC(n, i, s)], C.blue, 3, fade(u, S4.fwd + 2.4) * 0.8));
  // 计数
  const nIB = u < S4.fwd + 1 ? 6 : 2;
  rr(1250, 905, 570, 70, '#fbf9f4', fade(u, 1.2)); sr(1250, 905, 570, 70, C.lg, 2, fade(u, 1.2));
  txt('经过网卡的份数', 1272, 950, { size: 26, weight: 700, alpha: fade(u, 1.2) });
  txt(String(nIB), 1790, 954, { size: 44, weight: 800, align: 'right', color: nIB === 2 ? C.tealD : C.red, alpha: fade(u, 1.2) });
  if (u > S4.fwd + 1) txt('6 →', 1740, 954, { size: 30, weight: 700, align: 'right', color: C.gray, alpha: fade(u, S4.fwd + 1) });
  const ab = fade(u, S4.bw);
  txt('实测（H800）：机器内 153 GB/s，机器之间 43–58 GB/s', 170, 940, { size: 24, weight: 700, alpha: ab });
  note('DeepEP V1，4096 token / batch，每张卡的瓶颈带宽', 170, 974, { alpha: ab });
  lab('normal kernel：训练与 prefill', 830, 940, { color: C.blue, alpha: fade(u, S4.name) });
}

// ---------- 转发的代价：先报数 ----------
const S5 = { img: 0.5, box: bt(I.notify, 1), pre: bt(I.notify, 2), dec: bt(I.notify, 3) };
ev(I.notify, 0.5, 'land'); ev(I.notify, S5.box, 'pop'); ev(I.notify, S5.pre, 'land'); ev(I.notify, S5.dec, 'land'); ev(I.notify, S5.dec + 3, 'click');
function sNotify(u) {
  header(I.notify, u);
  const IX = 96, IY = 268, IW = 1500, a = fade(u, S5.img), h = img('normal', IX, IY, IW, a);
  note('DeepEP 原图：normal kernel 的 CPU 与 GPU 时序', IX + IW, IY + h + 30, { alpha: a, size: 20, align: 'right' });
  const ab = fade(u, S5.box);
  if (h) { sr(IX + IW * 0.135, IY + h * 0.47, IW * 0.132, h * 0.24, C.red, 5, ab); txt('等待数量，再分配显存', IX + IW * 0.135, IY + h * 0.47 + h * 0.24 + 34, { size: 24, weight: 700, color: C.red, alpha: ab }); }
  // 同一个毫秒刻度：prefill 一层 vs decode 一步
  const X = 300, PX = 1500 / 96, y1 = 740, y2 = 860;
  const a1 = fade(u, S5.pre);
  txt('prefill 一层', 96, y1 + 34, { size: 24, weight: 700, alpha: a1 });
  rr(X, y1, 34 * PX, 48, C.yel, a1); rr(X, y1, 0.9 * PX, 48, C.red, a1);
  txt('约 34 ms，获知数量约 0.9 ms', X + 34 * PX + 16, y1 + 34, { size: 24, weight: 600, color: C.ink2, alpha: a1 });
  const a2 = fade(u, S5.dec);
  if (a2 > 0) {
    txt('decode 一步', 96, y2 + 34, { size: 24, weight: 700, alpha: a2 });
    const brk = p(u, S5.dec + 3, S5.dec + 4.5), st = TRACE.step, n = st.length, shown = Math.floor(n * lin(u, S5.dec, S5.dec + 2));
    g.globalAlpha = a2; g.fillStyle = C.teal;
    for (let i = 0; i < shown; i++) { const lay = Math.floor(st[i] / (96 / 61)); g.fillRect(X + st[i] * PX + brk * lay * 4 * 0, y2, 1, 48); }
    g.globalAlpha = 1;
    sr(X - 4, y2 - 4, 95.5 * PX + 8, 56, C.tealD, 3, a2 * (1 - brk));
    for (let L = 1; L < 61; L++) rr(X + L * 96 / 61 * PX - 2, y2 - 6, 5, 60, C.red, brk * a2);
    txt(`约 96 ms，${n} 个 kernel：CUDA graph 一次提交`, X, y2 + 88, { size: 22, weight: 600, color: C.tealD, alpha: a2 * (1 - brk) });
    txt('每层等待数量：CUDA graph 无法使用，只能逐个 launch（示意）', X, y2 + 88, { size: 22, weight: 700, color: C.red, alpha: brk * a2 });
    note('同一刻度；kernel 起点取自 DeepSeek 的 decode trace', X + 820, y2 + 88, { alpha: a2 * (1 - brk), size: 20 });
  }
}

// ---------- decode：直接发送 ----------
const C16 = { x0: 100, pitch: 107, w: 98, y: 300, row0: 336, rp: 34, rh: 28 };
const c16 = (n, i) => ({ x: C16.x0 + n * C16.pitch + 6, y: C16.row0 + i * C16.rp, w: C16.w - 12, h: C16.rh });
const SRC16 = [0, 5], TGT16 = [[2, 1], [2, 6], [7, 3], [7, 5], [7, 0], [11, 2], [14, 4], [14, 7]];
function cluster16(alpha = 1, hi = []) {
  for (let n = 0; n < 16; n++) {
    const x = C16.x0 + n * C16.pitch, h = hi.includes(n);
    rr(x, C16.y, C16.w, 310, '#faf7f0', alpha); sr(x, C16.y, C16.w, 310, h ? C.blue : C.lg, h ? 3 : 2, alpha);
    txt(`机器 ${n}`, x + C16.w / 2, C16.y + 26, { size: 15, weight: 700, align: 'center', alpha });
    for (let i = 0; i < 8; i++) { const c = c16(n, i); rr(c.x, c.y, c.w, c.h, C.paper2, alpha); rr(c.x + c.w - 36, c.y + 7, 14, 14, '#bdb5a3', alpha); rr(c.x + c.w - 18, c.y + 7, 14, 14, '#bdb5a3', alpha); }
  }
}
const S6 = { cl: 0.5, go: bt(I.lowlat, 1), buf: bt(I.lowlat, 2), fill: bt(I.lowlat, 3), sc: bt(I.lowlat, 4) };
ev(I.lowlat, 0.5, 'whoosh'); TGT16.forEach((_, j) => ev(I.lowlat, S6.go + 0.6 + 0.08 * j, 'tick')); ev(I.lowlat, S6.buf, 'land'); ev(I.lowlat, S6.fill, 'pop'); ev(I.lowlat, S6.sc, 'land');
const BUF = (() => { const r = rng(5), f = new Uint8Array(2 * 128 * 128); for (let e = 0; e < 2; e++) { let c = 0; for (let i = 0; i < 128 * 128 && c < 512; i++) if (r() < 0.034) { f[e * 16384 + i] = 1; c++; } } return f; })();
function sLowlat(u) {
  header(I.lowlat, u);
  const a = fade(u, S6.cl), tg = [...new Set(TGT16.map(([n]) => n))];
  cluster16(a, u > S6.go ? tg : []);
  const s = c16(...SRC16); sr(s.x - 2, s.y - 2, s.w + 4, s.h + 4, C.blue, 3, a);
  TGT16.forEach(([n, i], j) => {
    const t = c16(n, i), x0 = s.x + s.w / 2, y0 = s.y, x1 = t.x + t.w / 2, y1 = t.y, k = p(u, S6.go + 0.1 * j, S6.go + 1.7 + 0.1 * j), top = 290 - 8 * j;
    const pts = []; for (let q = 0; q <= 24; q++) { const f = q / 24; pts.push([mix(x0, x1, f), (1 - f) * (1 - f) * y0 + 2 * f * (1 - f) * (top - 120) + f * f * y1]); }
    polyline(pts.slice(0, Math.max(2, Math.ceil(24 * k) + 1)), C.blue, 2.5, fade(u, S6.go) * 0.7);
    if (k > 0 && k < 1) { const [x, y] = polyAt(pts, k); chip(x, y, C.blue, 1, 0.6); }
    rr(t.x - 1, t.y - 1, t.w + 2, t.h + 2, C.blue, fade(u, S6.go + 1.7 + 0.1 * j) * 0.35);
  });
  note('8 份落在 4 台机器上：DeepSeek-V3 限制每个 token 最多发往 4 台机器', 100, 640, { alpha: fade(u, S6.go + 2) });
  // 接收区：每个 expert 预留 128 × 128 个位置
  const ab = fade(u, S6.buf), BX = 100, BY = 700, cs = 2;
  if (ab > 0) {
    lab('一张卡的接收区（2 个 expert）', BX, 690, { alpha: ab, size: 24 });
    for (let e = 0; e < 2; e++) {
      const x = BX + e * 290; rr(x, BY + 10, 128 * cs, 128 * cs, C.paper2, ab); sr(x, BY + 10, 128 * cs, 128 * cs, C.ink, 2, ab);
      const fk = fade(u, S6.fill, 1.2);
      if (fk > 0) { g.globalAlpha = fk; g.fillStyle = C.blue; for (let i = 0; i < 16384; i++) if (BUF[e * 16384 + i]) g.fillRect(x + (i % 128) * cs, BY + 10 + ((i / 128) | 0) * cs, cs, cs); g.globalAlpha = 1; }
      note(`expert ${e}：128 卡 × 128 token`, x, BY + 10 + 128 * cs + 26, { alpha: ab, size: 18 });
    }
    txt('实际收到约 3%', BX + 400, 690, { size: 24, weight: 700, color: C.blue, alpha: fade(u, S6.fill + 0.8) });
  }
  const ac = fade(u, S6.buf + 0.6);
  code(760, 680, 1060, ['{b:buffer.low_latency_dispatch}(x, topk_idx,', '    num_max_dispatch_tokens_per_rank, num_experts,', '    use_fp8={t:True}, return_recv_hook={t:True})'], { size: 22, alpha: ac, title: 'DeepEP V1 · deep_ep/buffer.py' });
  // 同一刻度：通信 vs 这一层的计算
  const as = fade(u, S6.sc);
  if (as > 0) {
    const comp = busy(TRACE.decode, 7) / 2 - 0, X = 980, PX = 0.95;
    const cmp = TRACE.decode.filter((x) => x.k !== 'dispatch' && x.k !== 'combine').reduce((a, x) => a + x.d, 0) / 2;
    txt('一个 micro-batch 一层的计算', 760, 892, { size: 22, weight: 700, alpha: as }); rr(X + 100, 868, cmp * PX, 34, C.yel, as); txt(`${(cmp / 1000).toFixed(2)} ms`, X + 110 + cmp * PX, 894, { size: 22, weight: 700, alpha: as });
    txt('dispatch + combine', 760, 946, { size: 22, weight: 700, alpha: as }); rr(X + 100, 922, 192 * PX, 34, C.blue, as); rr(X + 100 + 192 * PX, 922, 369 * PX, 34, C.purple, as);
    txt('192 + 369 µs', X + 110 + 561 * PX, 948, { size: 22, weight: 700, alpha: as });
    note('EP128，每卡 128 个 token；计算取自 decode trace', 760, 984, { alpha: as, size: 18 });
  }
}

// ---------- 重叠 · prefill（真实 trace） ----------
// micro-batch 的归属按 trace 中的依赖关系推断：dispatch 之后的 expert GEMM 属于同一个 micro-batch
const mbPre = (x) => {
  if (x.s === 16) return x.t < 6000 ? 0 : x.t < 13800 ? 1 : x.t < 24150 ? 0 : x.t < 33900 ? 1 : 0;
  return x.t < 5990 ? 1 : x.t < 13900 ? 0 : x.t < 21940 ? 1 : x.t < 32000 ? 0 : 1;
};
const KINDNAME = { attn: 'attention', moe: 'expert', dispatch: 'dispatch', combine: 'combine', other: '' };
const S7 = { lanes: 0.5, col: bt(I.overlap1, 1), sm: bt(I.overlap1, 2), busy: bt(I.overlap1, 3) };
ev(I.overlap1, 0.5, 'land'); ev(I.overlap1, S7.col, 'whoosh'); ev(I.overlap1, S7.sm, 'pop'); ev(I.overlap1, S7.busy, 'click');
function union(items) { const iv = items.map((x) => [x.t, x.t + x.d]).sort((a, b) => a[0] - b[0]); let s = 0, cur = null; for (const [a, b] of iv) { if (!cur || a > cur[1]) { if (cur) s += cur[1] - cur[0]; cur = [a, b]; } else cur[1] = Math.max(cur[1], b); } if (cur) s += cur[1] - cur[0]; return s; }
function sOverlap1(u) {
  header(I.overlap1, u);
  const X = 300, Wt = 1380, T = 34000, PX = Wt / T, lanes = [[7, 330, '计算', '108 个 SM'], [16, 470, '通信', '24 个 SM']];
  const reveal = lin(u, S7.lanes, S7.lanes + 3), col = fade(u, S7.col, 1);
  lanes.forEach(([s, y, name, sm]) => {
    const a = fade(u, S7.lanes);
    lab(name, 96, y + 52, { alpha: a }); txt(sm, 96, y + 86, { size: 22, weight: 600, color: s === 16 ? C.blue : C.ink2, alpha: fade(u, S7.sm) });
    rr(X, y, Wt, 110, '#faf7f0', a); sr(X, y, Wt, 110, C.lg, 2, a);
    for (const x of TRACE.prefill) {
      if (x.s !== s || x.t > T * reveal) continue;
      const base = x.k === 'dispatch' ? C.blue : x.k === 'combine' ? C.purple : x.k === 'attn' ? C.teal : x.k === 'moe' ? C.yel : C.lg;
      const fill = mixc(base, MB[mbPre(x)], col * (x.k === 'other' ? 0 : 1)), w = Math.max(1, Math.min(x.d, T - x.t) * PX);
      rr(X + x.t * PX, y + 8, w, 94, fill, a);
      if (w > 90 && KINDNAME[x.k]) txt(KINDNAME[x.k], X + x.t * PX + w / 2, y + 64, { size: 20, weight: 700, align: 'center', color: C.ink, alpha: a });
    }
  });
  // 刻度
  const at = fade(u, S7.lanes);
  for (let ms = 0; ms <= 34; ms += 2) { line(X + ms * 1000 * PX, 590, X + ms * 1000 * PX, 600, C.gray, 2, at); txt(`${ms}`, X + ms * 1000 * PX, 624, { size: 18, weight: 500, color: C.gray, align: 'center', alpha: at }); }
  txt('ms', X + Wt + 18, 624, { size: 18, weight: 500, color: C.gray, alpha: at });
  // 图例
  const ac = fade(u, S7.col);
  rr(X, 650, 26, 26, MB[0], ac); txt('micro-batch 0', X + 36, 672, { size: 22, weight: 700, alpha: ac });
  rr(X + 240, 650, 26, 26, MB[1], ac); txt('micro-batch 1', X + 276, 672, { size: 22, weight: 700, alpha: ac });
  const ab = fade(u, S7.busy);
  if (ab > 0) {
    const pc = (s) => Math.round(100 * (s === 7 ? TRACE.stats.prefill.layer_compute_ms : TRACE.stats.prefill.layer_comm_ms) / 34.48);
    txt(`忙碌 ${pc(7)}%`, X + Wt + 20, 392, { size: 24, weight: 800, color: C.tealD, alpha: ab }); txt(`忙碌 ${pc(16)}%`, X + Wt + 20, 532, { size: 24, weight: 800, color: C.tealD, alpha: ab });
  }
  const ai = fade(u, S7.col + 1.5), h = img('pfill', X, 720, 1000, ai);
  note('DeepSeek 原图：同一套重叠方式', X + 1020, 760, { alpha: ai, size: 20 });
}

// ---------- 重叠 · decode（真实 trace） ----------
// 归属按 DeepSeek 原图的 5 级流水推断：dispatch 接收之后是 MLP，同一 micro-batch 的 combine 发出；另一半在此期间做 attention
const MBD = [[0, 600, 0], [600, 656, 1], [656, 723, 0], [723, 910, 1], [910, 1300, 0], [1300, 1350, 0], [1350, 1418, 1], [1418, 1e9, 0]];
const mbDec = (x) => { if (x.t >= 205 && x.t < 600 && x.k === 'attn') return 1; if (x.t >= 900 && x.t < 1290 && x.k === 'attn') return 0; for (const [a, b, m] of MBD) if (x.t >= a && x.t < b) return x.k === 'attn' && x.t >= 1350 ? 1 : m; return 0; };
const COMMS = (() => { const c = TRACE.decode.filter((x) => x.k === 'dispatch' || x.k === 'combine'); return c; })();
// 在网络上的区间：发出 kernel 结束 → 对应的接收 kernel 开始（窗口内成对出现的）
const FLIGHT = [[228, 581, 'combine', 0], [619, 723, 'dispatch', 1], [922, 1274, 'combine', 1], [1313, 1418, 'dispatch', 0]];
const S8 = { lane: 0.5, send: bt(I.overlap2, 1), fly: bt(I.overlap2, 2), recv: bt(I.overlap2, 3), name: bt(I.overlap2, 4) };
ev(I.overlap2, 0.5, 'land'); ev(I.overlap2, S8.send, 'tick'); ev(I.overlap2, S8.send + 0.4, 'tick'); ev(I.overlap2, S8.fly, 'whoosh'); ev(I.overlap2, S8.recv, 'click'); ev(I.overlap2, S8.name, 'pop');
function sOverlap2(u) {
  header(I.overlap2, u);
  const X = 160, Wt = 1620, T = 1460, PX = Wt / T, Y = 350, a = fade(u, S8.lane), mbk = fade(u, S8.lane + 1.5);
  lab('计算', 60, Y + 52, { alpha: a, size: 24 }); txt('132 个 SM', 60, Y + 84, { size: 18, weight: 600, color: C.ink2, alpha: a });
  rr(X, Y, Wt, 120, '#faf7f0', a); sr(X, Y, Wt, 120, C.lg, 2, a);
  const hiComm = fade(u, S8.send);
  for (const x of TRACE.decode) {
    const comm = x.k === 'dispatch' || x.k === 'combine', base = x.k === 'dispatch' ? C.blue : x.k === 'combine' ? C.purple : x.k === 'attn' ? C.teal : x.k === 'moe' ? C.yel : C.lg;
    const fill = comm ? base : mixc(base, MB[mbDec(x)], mbk * (x.k === 'other' ? 0 : 1)), w = Math.max(2, Math.min(x.d, T - x.t) * PX);
    rr(X + x.t * PX, Y + 8, w, 104, fill, a);
    if (comm) sr(X + x.t * PX - 1, Y + 4, w + 2, 112, C.ink, 2, hiComm * a);
    if (x.n === 'FlashMLA') txt('FlashMLA 243 µs', X + x.t * PX + w / 2, Y + 68, { size: 20, weight: 700, align: 'center', alpha: a });
  }
  for (let us = 0; us <= 1400; us += 200) { line(X + us * PX, Y + 130, X + us * PX, Y + 140, C.gray, 2, a); txt(`${us}`, X + us * PX, Y + 164, { size: 18, weight: 500, color: C.gray, align: 'center', alpha: a }); }
  txt('µs', X + Wt + 14, Y + 164, { size: 18, weight: 500, color: C.gray, alpha: a });
  // 在网络上
  FLIGHT.forEach(([s, e, kind, m], j) => {
    const k = p(u, S8.fly + 0.3 * j, S8.fly + 1.3 + 0.3 * j), y = Y + 205 + (j % 2) * 56, col = kind === 'dispatch' ? C.blue : C.purple;
    if (k <= 0) return;
    line(X + s * PX, Y + 120, X + s * PX, y, col, 2, 0.7);
    line(X + s * PX, y, X + mix(s, e, k) * PX, y, col, 5, 1, [10, 7]);
    if (k > 0.99) arrow(X + e * PX - 30, y, X + e * PX, y, col, fade(u, S8.recv), 4);
    if (k > 0.99) line(X + e * PX, y, X + e * PX, Y + 120, col, 2, fade(u, S8.recv) * 0.7);
    const lt = `${kind}，micro-batch ${m}，传输中`, right = j === FLIGHT.length - 1;
    txt(lt, right ? X + e * PX - 8 : X + s * PX + 8, y - 12, { size: 18, weight: 700, color: col, alpha: k, align: right ? 'right' : 'left' });
  });
  const ar = fade(u, S8.recv);
  txt(`这一层两个 micro-batch 共 ${(1418 / 1000).toFixed(2)} ms，stream 忙碌 ${Math.round(100 * union(TRACE.decode.filter((x) => x.t + x.d <= 1418)) / 1418)}%`, X, Y + 350, { size: 26, weight: 700, color: C.tealD, alpha: ar });
  rr(X + 960, Y + 328, 26, 26, MB[0], mbk); txt('micro-batch 0', X + 996, Y + 350, { size: 22, weight: 700, alpha: mbk });
  rr(X + 1200, Y + 328, 26, 26, MB[1], mbk); txt('micro-batch 1', X + 1236, Y + 350, { size: 22, weight: 700, alpha: mbk });
  note('框出的是 dispatch / combine 的发出与接收 kernel', X, Y + 390, { alpha: hiComm, size: 20 });
  const an = fade(u, S8.name);
  if (an > 0) {
    rr(X, 790, 760, 150, '#fbf9f4', an); sr(X, 790, 760, 150, C.lg, 2, an);
    txt('框架中的名称', X + 22, 830, { size: 22, weight: 700, color: C.gray, alpha: an });
    txt('SGLang：two-batch overlap', X + 22, 870, { size: 24, weight: 700, alpha: an }); txt('vLLM：DBO（dual batch overlap）', X + 22, 908, { size: 24, weight: 700, alpha: an });
    txt('每卡 32 个 token 时，吞吐 −27%', X + 820, 870, { size: 26, weight: 800, color: C.red, alpha: fade(u, S8.name + 0.6) });
    note('SGLang 实测（H100，decode）：batch 拆成两半后 GEMM 变小', X + 820, 906, { alpha: fade(u, S8.name + 0.6), size: 20 });
  }
}

// ---------- 负载不均与 EPLB ----------
// 示意：256 个 expert 的负载按热图的偏斜构造（少数 expert 明显更热）；EPLB 按 DeepSeek 的做法：复制负载 / 副本数最大的 expert，再均衡装卡
// 摆放按历史统计（LOAD）计算，柱高按当前这一批的负载（NOW：历史值上下浮动）评估，所以重排后也不会完全齐平
const LOAD = (() => { const r = rng(3); return Array.from({ length: 256 }, (_, e) => { const z = -Math.log(1 - r()); return 0.5 + z * 0.55 + (e < 24 ? 0.8 * r() : 0); }); })();
const NOW = (() => { const r = rng(17); return LOAD.map((l) => l * (0.7 + 0.6 * r())); })();
const GPU_BEFORE = Array.from({ length: 32 }, (_, gi) => NOW.slice(gi * 8, gi * 8 + 8).reduce((a, b) => a + b, 0));
const GPU_AFTER = (() => {
  const cnt = LOAD.map(() => 1);
  for (let k = 0; k < 32; k++) { let b = 0; for (let e = 1; e < 256; e++) if (LOAD[e] / cnt[e] > LOAD[b] / cnt[b]) b = e; cnt[b]++; }
  const items = []; LOAD.forEach((l, e) => { for (let c = 0; c < cnt[e]; c++) items.push([l / cnt[e], NOW[e] / cnt[e]]); });
  items.sort((a, b) => b[0] - a[0]); const est = new Array(32).fill(0), load = new Array(32).fill(0), n = new Array(32).fill(0);
  for (const [h, now] of items) { let b = -1; for (let gi = 0; gi < 32; gi++) if (n[gi] < 9 && (b < 0 || est[gi] < est[b])) b = gi; est[b] += h; load[b] += now; n[b]++; }
  return load;
})();
const S9 = { heat: 0.5, bars: bt(I.eplb, 1), fix: bt(I.eplb, 2), slot: bt(I.eplb, 3) };
ev(I.eplb, 0.5, 'land'); ev(I.eplb, S9.bars, 'land'); ev(I.eplb, S9.fix, 'whoosh'); ev(I.eplb, S9.fix + 2.2, 'land'); ev(I.eplb, S9.slot, 'pop');
function sEplb(u) {
  header(I.eplb, u);
  const ah = fade(u, S9.heat), h = img('heat', 70, 280, 860, ah);
  note('SGLang 实测：DeepSeek-V3 各层 256 个 expert 收到的 token 比例', 90, 280 + h + 30, { alpha: ah, size: 20 });
  const ab = fade(u, S9.bars), k = p(u, S9.fix + 0.5, S9.fix + 3), BX = 1010, BY = 860, BW = 25, mx = Math.max(...GPU_BEFORE);
  if (ab > 0) {
    lab('32 张卡这一层的负载', BX, 320, { alpha: ab });
    const cur = GPU_BEFORE.map((v, i) => mix(v, GPU_AFTER[i], k)), top = Math.max(...cur), mean = cur.reduce((a, b) => a + b, 0) / 32;
    cur.forEach((v, i) => rr(BX + i * BW, BY - 440 * v / mx, BW - 4, 440 * v / mx, v === top ? C.red : C.blue, ab));
    line(BX - 10, BY - 440 * top / mx, BX + 32 * BW + 10, BY - 440 * top / mx, C.red, 3, ab, [8, 6]);
    txt('其余卡等待这张卡', BX + 32 * BW - 4, BY - 440 * top / mx - 12, { size: 22, weight: 700, color: C.red, align: 'right', alpha: ab });
    line(BX - 10, BY, BX + 32 * BW + 10, BY, C.ink, 2, ab);
    txt(`平均 / 最大 = ${(mean / top).toFixed(2)}`, BX, BY + 44, { size: 26, weight: 800, color: k > 0.5 ? C.tealD : C.red, alpha: ab });
    note('示意：负载按热图的偏斜构造；副本按历史统计摆放，柱高是当前这一批', BX, BY + 80, { alpha: ab, size: 18 });
  }
  const af = fade(u, S9.fix);
  txt('复制最热的 32 个 expert → 256 + 32 = 288 个位置，每卡 9 个', BX, 380, { size: 22, weight: 700, color: C.tealD, alpha: af });
  const as = fade(u, S9.slot);
  if (as > 0) {
    for (let s = 0; s < 9; s++) { const x = BX + s * 30; rr(x, 400, 24, 24, s < 8 ? '#bdb5a3' : C.tealD, as); }
    txt('第 9 格：副本', BX + 280, 420, { size: 20, weight: 700, color: C.tealD, alpha: as });
    txt('SGLang 实测（96 × H100）：EPLB 后 prefill 吞吐 1.49 倍，decode 2.54 倍', 90, 960, { size: 24, weight: 700, alpha: fade(u, S9.slot + 1.2) });
  }
}

// ---------- 2026 年 ----------
const S10 = { t0: 0.5, v2: bt(I.now, 1), pipe: bt(I.now, 2), lim: bt(I.now, 3) };
ev(I.now, 0.5, 'land'); ev(I.now, S10.v2, 'pop'); ev(I.now, S10.v2 + 2, 'pop'); ev(I.now, S10.pipe, 'whoosh'); ev(I.now, S10.lim, 'land');
function sNow(u) {
  header(I.now, u);
  const a = fade(u, S10.t0), Y = 330;
  line(140, Y, 1780, Y, C.ink, 3, a);
  const pts = [[220, '2025-02', 'DeepEP 开源', '两种 kernel：normal / low-latency', S10.t0], [820, '2026-04', 'DeepEP V2', '合并为同一套 dispatch / combine；SM 用量至多降至 1/4；不再支持 0 SM 的模式', S10.v2], [1420, '2026-09', 'DeepEP V2.5', '删除 V1 的接口与 NVSHMEM 依赖', S10.v2 + 2]];
  for (const [x, d, t, s, at] of pts) {
    const aa = fade(u, at); g.globalAlpha = aa; g.fillStyle = C.blue; g.beginPath(); g.arc(x, Y, 11, 0, 7); g.fill(); g.globalAlpha = 1;
    txt(d, x, Y - 24, { size: 22, weight: 600, color: C.gray, align: 'center', alpha: aa }); txt(t, x, Y + 50, { size: 28, weight: 800, align: 'center', alpha: aa });
    const words = s.split('；'); words.forEach((w, i) => txt(w, x, Y + 88 + i * 32, { size: 21, weight: 500, color: C.ink2, align: 'center', alpha: aa }));
  }
  const ap = fade(u, S10.pipe), PX = 0.12, X = 520;
  const P = [['网卡（CX7）', 50, IBC], ['H800 NVLink', 160, NVC], ['GB200 NVL72 NVLink', 900, C.tealD]];
  let y = 560;
  P.forEach(([n, bw, col], i) => { const aa = fade(u, S10.pipe + 0.4 * i); txt(n, 140, y + bw * PX / 2 + 9, { size: 24, weight: 700, alpha: aa }); rr(X, y, 1100, bw * PX, col, aa); txt(`${bw} GB/s`, X + 1120, y + bw * PX / 2 + 9, { size: 24, weight: 700, alpha: aa }); y += bw * PX + 34; });
  note('每张卡每个方向；NVL72 的 72 张卡在同一个 NVLink 域里', 140, y + 6, { alpha: ap, size: 20 });
  const al = fade(u, S10.lim);
  if (al > 0) {
    const L = 1150 / 14.76, y2 = y + 50;
    txt('decode 每 token 的通信下限（DeepSeek 估算）', 140, y2, { size: 22, weight: 700, alpha: al });
    rr(X, y2 + 16, 14.76 * L * p(u, S10.lim, S10.lim + 1.5), 30, C.red, al); txt('14.76 ms', X + 14.76 * L + 14, y2 + 40, { size: 22, weight: 700, alpha: al });
    rr(X, y2 + 56, 0.82 * L, 30, C.tealD, al); txt('0.82 ms', X + 0.82 * L + 14, y2 + 80, { size: 22, weight: 700, alpha: al });
  }
}

// ---------- 按层切 ----------
const STG = ['层 1–16', '层 17–31', '层 32–46', '层 47–61'];
const S11 = { st: 0.5, go: bt(I.pp, 1), cmp: bt(I.pp, 2) };
ev(I.pp, 0.5, 'land'); [0, 1, 2].forEach((j) => ev(I.pp, S11.go + 1.2 + j * 1.6, 'tick')); ev(I.pp, S11.cmp, 'land');
function stageCards(alpha, hot = -1) {
  for (let n = 0; n < 4; n++) {
    const x = cardX(n);
    rr(x, 320, CL.w, 330, n === hot ? '#eef7f4' : '#faf7f0', alpha); sr(x, 320, CL.w, 330, n === hot ? C.tealD : C.lg, n === hot ? 3 : 2, alpha);
    txt(`机器 ${n}`, x + 18, 356, { size: 24, weight: 700, alpha }); txt(STG[n], x + CL.w / 2, 470, { size: 40, weight: 800, align: 'center', color: C.tealD, alpha });
    note('机器内 8 张卡用 TP', x + CL.w / 2, 600, { alpha, align: 'center', size: 18 });
  }
}
function sPp(u) {
  header(I.pp, u);
  const a = fade(u, S11.st), k = lin(u, S11.go, S11.go + 5.5), pos = k * 3, cur = Math.min(3, Math.floor(pos + 0.0001));
  stageCards(a, u > S11.go ? cur : -1);
  for (let j = 0; j < 3; j++) {
    const x0 = cardX(j) + CL.w, x1 = cardX(j + 1), ya = 500, done = fade(u, S11.go + 0.6 + j * 1.6);
    arrow(x0 + 8, ya, x1 - 8, ya, C.tealD, a, 5); txt('28 KB', (x0 + x1) / 2, ya - 22, { size: 22, weight: 800, align: 'center', color: C.tealD, alpha: done });
  }
  if (u > S11.go && k < 1) { const n = Math.floor(pos), f = pos - n, x = n < 3 ? mix(cardX(n) + CL.w / 2, cardX(n + 1) + CL.w / 2, clamp((f - 0.5) * 2)) : cardX(3) + CL.w / 2; chip(x, 540, C.tealD, 1, 1.4); }
  note('每过一个边界：hidden state 与 residual，2 × 7168 × 2 字节', CL.x0, 700, { alpha: fade(u, S11.go + 1), size: 22 });
  const ac = fade(u, S11.cmp);
  if (ac > 0) {
    const X = 520, L = 1250 / 9.98, kk = p(u, S11.cmp, S11.cmp + 1.6);
    txt('每个 token 每一步', 170, 780, { size: 24, weight: 700, alpha: ac });
    txt('EP', 170, 846, { size: 28, weight: 800, color: C.blue, alpha: ac }); rr(X, 820, 9.98 * L * kk, 36, C.blue, ac); txt('约 10 MB', X + 9.98 * L * kk + 14, 848, { size: 24, weight: 700, alpha: ac });
    txt('PP', 170, 916, { size: 28, weight: 800, color: C.tealD, alpha: ac }); rr(X, 890, Math.max(2, 0.086 * L), 36, C.tealD, ac); txt('约 86 KB', X + 0.086 * L + 14, 918, { size: 24, weight: 700, alpha: ac });
    note('EP：58 个 MoE 层 × (57 + 115) KB，按 top-8 不去重的上限；PP：3 个边界 × 28 KB', 170, 966, { alpha: ac, size: 20 });
  }
}

// ---------- 气泡 ----------
const S12 = { one: 0.5, four: bt(I.bubble, 1), follow: bt(I.bubble, 2), concl: bt(I.bubble, 3) };
ev(I.bubble, 0.5, 'land'); ev(I.bubble, S12.four, 'whoosh'); ev(I.bubble, S12.follow, 'pop'); ev(I.bubble, S12.concl, 'click');
const BC = [C.blue, C.purple, C.amber, C.green];
function sBubble(u) {
  header(I.bubble, u);
  const X = 330, Y = 320, UW = 150, RH = 74, a = fade(u, S12.one), N = 9;
  for (let s = 0; s < 4; s++) { txt(`机器 ${s}`, 96, Y + s * 92 + 48, { size: 24, weight: 700, alpha: a }); rr(X, Y + s * 92, N * UW, RH, '#faf7f0', a); sr(X, Y + s * 92, N * UW, RH, C.lg, 2, a); }
  const many = fade(u, S12.four), nb = many > 0 ? 4 : 1, tNow = u < S12.four ? lin(u, S12.one + 0.5, S12.one + 4.5) * 4 : 4 + lin(u, S12.four, S12.four + 4) * 5;
  // 一个 batch 时：气泡
  for (let s = 0; s < 4; s++) for (let c = 0; c < 4; c++) if (c !== s && c < tNow) rr(X + c * UW + 3, Y + s * 92 + 3, UW - 6, RH - 6, C.red, 0.25 * a * (1 - many));
  for (let b = 0; b < nb; b++) for (let s = 0; s < 4; s++) {
    const c = s + (u < S12.four ? 0 : b), st = u < S12.four ? c : c;
    if (u >= S12.four && b === 0 && c >= 4) continue;
    const show = u < S12.four ? clamp(tNow - st) : clamp((tNow - 4) * 1.6 - st + 0.4);
    if (show <= 0) continue;
    const bb = u < S12.four ? 0 : b;
    rr(X + c * UW + 3, Y + s * 92 + 3, (UW - 6) * show, RH - 6, BC[bb], (u < S12.four ? a : many) * 0.9);
    if (show > 0.6) txt(`batch ${bb + 1}`, X + c * UW + UW / 2, Y + s * 92 + 46, { size: 20, weight: 700, color: '#fff', align: 'center' });
  }
  // 稳态：之后的列依次由 4 个 batch 轮流占满（同一个 batch 的下一步重新从机器 0 开始）
  if (u >= S12.four) for (let c = 4; c < N; c++) for (let s = 0; s < 4; s++) {
    const bb = ((c - s) % 4 + 4) % 4, show = clamp((tNow - 4) * 1.6 - c + 0.4); if (show <= 0 || c - s < 4) continue;
    rr(X + c * UW + 3, Y + s * 92 + 3, (UW - 6) * show, RH - 6, BC[bb], many * 0.9);
  }
  txt('利用率 1/4', X + N * UW + 20, Y + 190, { size: 28, weight: 800, color: C.red, alpha: a * (1 - many) * fade(u, 3) });
  txt('填满', X + N * UW + 20, Y + 190, { size: 28, weight: 800, color: C.tealD, alpha: fade(u, S12.four + 3.5) });
  code(X, 720, 1120, ['{g:# PP requires PP-size concurrent batches to fill the pipeline.}'], { size: 22, alpha: fade(u, S12.four + 1), title: 'vLLM · vllm/config/vllm.py' });
  // 跟住一个 token
  const af = fade(u, S12.follow);
  if (af > 0) { const pts = [0, 1, 2, 3].map((s) => [X + (s + 1) * UW + UW / 2, Y + s * 92 + RH / 2]); polyline(pts, C.ink, 5, af); pts.forEach(([x, y]) => { g.globalAlpha = af; g.fillStyle = C.ink; g.beginPath(); g.arc(x, y, 9, 0, 7); g.fill(); g.globalAlpha = 1; });
    txt('batch 2 的一个 token：仍要依次经过 4 段、3 次传输', X, 870, { size: 24, weight: 700, alpha: af }); }
  const ac = fade(u, S12.concl);
  txt('提高：吞吐与容量（每段只存储所负责各层的权重与 KV）', X, 920, { size: 24, weight: 700, color: C.tealD, alpha: ac });
  txt('不变：单个 token 的延迟；每个 batch 只有 1/4 的请求', X, 962, { size: 24, weight: 700, color: C.red, alpha: fade(u, S12.concl + 0.6) });
}

// ---------- 推理里的 PP ----------
const S13 = { c1: 0.5, c2: bt(I.where, 1), c3: bt(I.where, 2) };
ev(I.where, 0.5, 'land'); ev(I.where, S13.c2, 'land'); for (let j = 0; j < 8; j++) ev(I.where, S13.c2 + 0.6 + j * 0.5, 'tick'); ev(I.where, S13.c3, 'land');
function sWhere(u) {
  header(I.where, u);
  const COL = [[96, S13.c1, '机器之间只有慢速网络'], [700, S13.c2, '超长 prompt 的 prefill'], [1304, S13.c3, '大 MoE 的 decode']];
  COL.forEach(([x, t, title], i) => { const a = fade(u, t); rr(x, 290, 540, 700, i === 2 ? '#f3f1fb' : '#faf7f0', a); sr(x, 290, 540, 700, C.lg, 2, a); txt(title, x + 24, 340, { size: 28, weight: 800, alpha: a }); });
  // 第一栏
  const a1 = fade(u, S13.c1);
  for (let m = 0; m < 2; m++) { const y = 390 + m * 230; rr(126, y, 480, 150, '#fff', a1); sr(126, y, 480, 150, C.lg, 2, a1); txt(`机器 ${m}`, 146, y + 36, { size: 22, weight: 700, alpha: a1 }); txt('TP 8：机器内 NVLink', 366, y + 92, { size: 24, weight: 700, color: NVC, align: 'center', alpha: a1 }); }
  arrow(366, 545, 366, 615, C.tealD, a1, 5); txt('PP：只传输 hidden state', 390, 588, { size: 22, weight: 700, color: C.tealD, alpha: a1 });
  note('vLLM 文档：TP = 每台机器的卡数，PP = 机器数', 126, 830, { alpha: a1, size: 19 });
  note('没有 NVLink 的卡（如 L40S）也用 PP 代替 TP', 126, 862, { alpha: a1, size: 19 });
  // 第二栏：chunk 流水
  const a2 = fade(u, S13.c2), X2 = 730, Y2 = 390;
  for (let s = 0; s < 4; s++) { rr(X2 + 70, Y2 + s * 52, 400, 40, '#fff', a2); txt(`段 ${s}`, X2, Y2 + s * 52 + 28, { size: 20, weight: 700, alpha: a2 }); }
  const tc = lin(u, S13.c2 + 0.5, S13.c2 + 5) * 11;
  for (let c = 0; c < 8; c++) for (let s = 0; s < 4; s++) { const f = clamp(tc - c - s); if (f > 0) rr(X2 + 70 + (c + s) * 36 + 2, Y2 + s * 52 + 4, 32 * f, 32, BC[c % 4], a2 * 0.9); }
  note('128K 的 prompt 切成 chunk，依次流入', X2, Y2 + 240, { alpha: a2, size: 19 });
  const ab = fade(u, S13.c2 + 4), L = 400 / 48.5;
  txt('TTFT', X2, 720, { size: 24, weight: 800, alpha: ab });
  txt('TP8', X2, 768, { size: 20, weight: 700, alpha: ab }); rr(X2 + 90, 748, 48.5 * L, 28, C.red, ab); txt('48.5 s', X2 + 96 + 48.5 * L - 74, 769, { size: 20, weight: 700, color: '#fff', alpha: ab });
  txt('PP4 TP8', X2, 816, { size: 20, weight: 700, alpha: ab }); rr(X2 + 90, 796, 15.5 * L, 28, C.tealD, ab); txt('15.5 s', X2 + 100 + 15.5 * L, 817, { size: 20, weight: 700, alpha: ab });
  note('SGLang 实测：H20，DeepSeek-V3.1，128K 输入', X2, 862, { alpha: ab, size: 19 });
  note('吞吐：PP4 TP8 为 TP8 的 3.31 倍，TP32 为 2.54 倍', X2, 894, { alpha: ab, size: 19 });
  // 第三栏
  const a3 = fade(u, S13.c3), X3 = 1334;
  const lk = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], ctr = (m) => [X3 + (m % 2) * 240 + 110, 390 + Math.floor(m / 2) * 190 + 75];
  lk.forEach(([i, j]) => { const [x0, y0] = ctr(i), [x1, y1] = ctr(j); line(x0, y0, x1, y1, C.blue, 4, a3 * 0.6, [8, 6]); });
  for (let m = 0; m < 4; m++) { const x = X3 + (m % 2) * 240, y = 390 + Math.floor(m / 2) * 190; rr(x, y, 220, 150, '#fff', a3); sr(x, y, 220, 150, C.lg, 2, a3); txt(`机器 ${m}`, x + 14, y + 30, { size: 20, weight: 700, alpha: a3 }); txt('DP attention', x + 110, y + 82, { size: 20, weight: 700, color: C.tealD, align: 'center', alpha: a3 }); txt('EP', x + 110, y + 118, { size: 22, weight: 800, color: C.blue, align: 'center', alpha: a3 }); }
  note('decode 每步每个请求 1 个 token，', X3, 830, { alpha: a3, size: 19 }); note('流水线难以填满；跨机也用 EP', X3, 862, { alpha: a3, size: 19 });
  note('DeepSeek、SGLang、Kimi K2 的 decode 部署', X3, 894, { alpha: a3, size: 19 });
}

// ---------- 结尾 ----------
const S14 = { ep: 0.5, pp: bt(I.end, 1), title: 12.5 };
ev(I.end, 0.5, 'whoosh'); ev(I.end, S14.pp, 'land'); ev(I.end, S14.title, 'land');
function sEnd(u) {
  const out = p(u, S14.title, S14.title + 1.2), keep = 1 - out, a = fade(u, 0.2);
  txt(FILM, 96, 76, { size: 28, weight: 700, color: C.ink2, alpha: keep });
  const sc = mix(1, 0.66, out);
  g.save(); g.translate(mix(0, 1830 - CL.x0 - 1660 * 0.66 + 0, out), mix(0, 120, out)); g.translate(CL.x0, 300); g.scale(sc, sc); g.translate(-CL.x0, -300);
  cluster(a);
  const r = rng(21), aE = fade(u, S14.ep);
  for (let m = 0; m < 90; m++) { const n0 = (r() * 4) | 0, i0 = (r() * 8) | 0, n1 = (r() * 4) | 0, i1 = (r() * 8) | 0, ph = r(); const k = ((u * 0.5) + ph) % 1; const [x, y] = polyAt(pathTo(n0, i0, n1, i1, m % 3), k); chip(x, y, m % 2 ? C.blue : C.purple, aE * 0.8, 0.6); }
  const aP = fade(u, S14.pp);
  if (aP > 0) { const k = ((u - S14.pp) * 0.35) % 1, x = mix(cardX(0) + CL.w / 2, cardX(3) + CL.w / 2, k); rrect(x - 50, 888, 100, 30, C.tealD, aP); for (let j = 0; j < 3; j++) arrow(cardX(j) + CL.w + 10, 903, cardX(j + 1) - 10, 903, C.tealD, aP * 0.6, 3);
  }
  g.restore();
  lab('EP：每个 MoE 层两次 all-to-all', 170, 960, { color: C.blue, alpha: fade(u, S14.ep) * keep });
  lab('PP：每段之间每步一次', 980, 960, { color: C.tealD, alpha: aP * keep });
  bigTitle(1 - out, fade(u, S14.title));
}

// ---------- 字幕 ----------
// 配音稿逐句显示在画面底部：按标点切成不超过 26 字的短句。
// 时长：有 vo.js（vo.py 合成后生成，文案一致）时用实测时长，否则按字数估算；都不越过下一句的起点。
function cues(text, lim = 26) {
  const out = [];
  for (const p of (text.match(/[^，。：；？！]+[，。：；？！]?/g) || []).map((x) => x.trim()).filter(Boolean)) {
    const t = p.replace(/[，。；]$/, '');
    if (out.length && out[out.length - 1].length + t.length + 1 <= lim && !/[？！]$/.test(out[out.length - 1])) out[out.length - 1] += (out[out.length - 1].endsWith('：') ? '' : '，') + t;
    else out.push(t);
  }
  return out;
}
const est = (t) => (t.match(/[一-鿿]/g) || []).length / 4.6 + (t.match(/[A-Za-z0-9][A-Za-z0-9.,%\-]*/g) || []).length * 0.35;
const voOf = (k, j) => { const v = typeof VO !== 'undefined' ? VO.beats[k]?.[j] : null; return v && v.text === SCRIPT[k].beats[j][2] ? v : null; };
const SUBS = SCRIPT.map((sc, k) => sc.beats.flatMap(([t0, , text], j) => {
  if (!text) return [];
  const cs = cues(text), end = j + 1 < sc.beats.length ? sc.beats[j + 1][0] - 0.1 : sc.dur - 0.1, v = voOf(k, j);
  const total = cs.reduce((a, c) => a + est(c), 0), spoken = v ? v.dur : total;
  const scale = Math.min(spoken / Math.max(0.1, total), (end - t0) / Math.max(0.1, total)); let t = t0;
  return cs.map((c) => { const d = est(c) * scale; const r = [t, t + d, c]; t += d; return r; });
}));
function subtitle(k, u) {
  const c = SUBS[k].find(([a, b]) => u >= a && u < b); if (!c) return;
  const w = tw(c[2], 34, 600);
  rr(960 - w / 2 - 20, 1012, w + 40, 52, C.paper, 0.92);
  txt(c[2], 960, 1050, { size: 34, weight: 600, align: 'center', color: C.ink });
}

const SCENES = { open: s1, route: sRoute, volume: sVolume, forward: sForward, notify: sNotify, lowlat: sLowlat, overlap1: sOverlap1, overlap2: sOverlap2, eplb: sEplb, now: sNow, pp: sPp, bubble: sBubble, where: sWhere, end: sEnd };
function draw(t) {
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  rr(0, 0, W, H, C.paper);
  SCENES[SCRIPT[k].id](t - AT[k]);
  subtitle(k, t - AT[k]);
}
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
