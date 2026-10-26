// KV cache 分层存储影片的各幕（顺序与 id 见 script.js）。每幕一个 draw(u)，u 是幕内时间（秒）；节拍起点取自 script.js，音效事件登记进 EV。
// 贯穿全片的图形：一台 8 × H100 服务器的剖面（machine）：上层 8 张 GPU 的显存，每张卡 42 格（一格 = 一个会话在这张卡上的 1.25 GiB），
// 中层主机内存 150 格（一格 = 一个会话的 10 GiB），下层本地 SSD；运行例子是一个 32K token 的会话，紫色，8 张卡各存八分之一。
// 颜色：紫 = 这个会话的 KV，灰 = 其他请求，蓝 = 显存，青 = 主机内存，黄 = SSD，红 = 网络 / 重算。数字的估算过程见 docs/kv-cache-tiering-notes.md。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const I = Object.fromEntries(SCRIPT.map((s, i) => [s.id, i]));
const EV = SCRIPT.map(() => []);
const ev = (k, u, kind) => EV[k].push([u, kind]);
const fade = (u, a, d = 0.45) => po(u, a, a + d);
const OTHER = '#c9c1af', EMPTY = '#ebe6da', DRAMC = C.tealD, SSDC = C.yelD;
const PANEL = '#f8f5ee';

function header(k, u, a0 = 0) {
  const s = SCRIPT[k], a = fade(u, a0, 0.5);
  txt('KV cache 的分层存储与传输', 96, 76, { size: 28, weight: 700, color: C.ink2, alpha: k === 0 ? fade(u, 3.6, 0.6) : 1 });
  txt(s.src, 1824, 76, { size: 22, weight: 500, color: C.gray, align: 'right', alpha: a });
  rr(96, 100, tw(s.tag, 32, 700) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a });
  rich(s.head, 96, 226 + (1 - a) * 16, 52, 700, a);
}
const note = (s, x, y, o = {}) => txt(s, x, y, { size: 22, weight: 500, color: C.gray, ...o });
const lab = (s, x, y, o = {}) => txt(s, x, y, { size: 26, weight: 700, ...o });
// 占位：将来放实测的地方，虚线框写明放什么、需要什么硬件
function placeholder(x, y, w, lines, alpha = 1) {
  if (alpha <= 0.003) return;
  const h = 46 + lines.length * 30;
  rr(x, y, w, h, '#fff7e6', alpha); sr(x, y, w, h, C.amber, 2.5, alpha, [10, 7]);
  rr(x, y, 64, 30, C.amber, alpha); txt('占位', x + 32, y + 22, { size: 18, weight: 700, color: '#fff', align: 'center', alpha });
  lines.forEach((s, i) => txt(s, x + 18, y + 64 + i * 30, { size: 20, weight: 500, color: C.ink2, alpha }));
}
function tag(s, x, y, col, alpha = 1, size = 20) {
  const w = tw(s, size, 700) + 20; rr(x, y - size - 4, w, size + 14, col, alpha);
  txt(s, x + 10, y, { size, weight: 700, color: '#fff', alpha }); return w;
}
// 一个计时器：秒数按真实速率走
const fmtT = (s) => (s < 0.1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`);
// 管道：粗细按带宽；flow 为 0..1 时在管道里画移动的亮点
function pipe(x0, y0, x1, y1, w, col, alpha = 1, flow = -1) {
  line(x0, y0, x1, y1, mixc(col, '#ffffff', 0.55), w, alpha);
  if (flow >= 0 && alpha > 0.003) {
    const n = 4; for (let i = 0; i < n; i++) { const f = (flow + i / n) % 1; const x = mix(x0, x1, f), y = mix(y0, y1, f); rr(x - w / 2 - 1, y - w / 2 - 1, w + 2, w + 2, col, alpha); }
  }
}

// ================= 服务器剖面 =================
// 本地坐标 1000 × 600；scenes 里用 at(x, y, s) 平移缩放。cells(i, c)：第 i 张卡第 c 格的颜色（0..41）；dram(c)：内存第 c 格（0..149）。
const GX = (i) => 24 + i * 120, GY = 104, GW = 108, GH = 150;
const gcell = (i, c) => [GX(i) + 9 + (c % 6) * 15.5, GY + 10 + Math.floor(c / 6) * 19.4, 13, 16];
const DR = { x: 24, y: 330, w: 952, h: 110 }, SD = { x: 24, y: 470, w: 952, h: 100 };
const dcell = (c) => [300 + (c % 30) * 22.3, DR.y + 12 + Math.floor(c / 30) * 18, 19, 15];
const SI = 15, DI = 37;   // 会话在每张卡上的格、在内存里的格
function at(x, y, s, fn) { g.save(); g.translate(x, y); g.scale(s, s); fn(); g.restore(); }
function machine({ alpha = 1, cells = () => EMPTY, dram = () => EMPTY, tiers = [1, 1, 1], pcie = 1, flowP = -1, flowS = -1, title = '一台 8 × H100 服务器' } = {}) {
  if (alpha <= 0.003) return;
  rr(0, 0, 1000, 600, '#f7f4ec', alpha); sr(0, 0, 1000, 600, C.ink2, 2, alpha);
  txt(title, 24, 44, { size: 26, weight: 700, alpha });
  const a0 = alpha * tiers[0], a1 = alpha * tiers[1], a2 = alpha * tiers[2];
  txt('显存 HBM · 每卡 80 GB', 24, 90, { size: 21, weight: 700, color: C.blue, alpha: a0 });
  for (let i = 0; i < 8; i++) {
    rr(GX(i), GY, GW, GH, '#fff', a0); sr(GX(i), GY, GW, GH, C.blue, 2.5, a0);
    for (let c = 0; c < 42; c++) { const [x, y, w, h] = gcell(i, c); rr(x, y, w, h, cells(i, c), a0); }
    txt(`GPU ${i}`, GX(i) + GW / 2, GY + GH + 22, { size: 16, weight: 600, color: C.gray, align: 'center', alpha: a0 });
  }
  // PCIe：每张卡一条，粗细按带宽
  for (let i = 0; i < 8; i++) pipe(GX(i) + GW / 2, GY + GH + 30, GX(i) + GW / 2, DR.y, 7, C.teal, alpha * pcie * Math.min(tiers[0], tiers[1]), flowP);
  rr(DR.x, DR.y, DR.w, DR.h, '#fff', a1); sr(DR.x, DR.y, DR.w, DR.h, DRAMC, 2.5, a1);
  txt('主机内存 DRAM', DR.x + 16, DR.y + 40, { size: 22, weight: 700, color: DRAMC, alpha: a1 });
  txt('2 TB · 150 格', DR.x + 16, DR.y + 72, { size: 19, weight: 500, color: C.gray, alpha: a1 });
  for (let c = 0; c < 150; c++) { const [x, y, w, h] = dcell(c); rr(x, y, w, h, dram(c), a1); }
  pipe(500, DR.y + DR.h, 500, SD.y, 3, C.yel, alpha * pcie * Math.min(tiers[1], tiers[2]), flowS);
  rr(SD.x, SD.y, SD.w, SD.h, '#fff', a2); sr(SD.x, SD.y, SD.w, SD.h, SSDC, 2.5, a2);
  txt('本地 SSD（NVMe）', SD.x + 16, SD.y + 40, { size: 22, weight: 700, color: SSDC, alpha: a2 });
  txt('8 × 3.84 TB · 约 2,800 格', SD.x + 16, SD.y + 72, { size: 19, weight: 500, color: C.gray, alpha: a2 });
  for (let r = 0; r < 4; r++) for (let c = 0; c < 60; c++) rr(300 + c * 11.1, SD.y + 14 + r * 19, 9, 15, mixc('#ffffff', C.yel, 0.25), a2);
  txt('…', 968, SD.y + 62, { size: 26, color: SSDC, align: 'right', alpha: a2 });
}
// 一格在全局坐标里的中心（machine 画在 (mx, my)，缩放 ms）
const gpos = (mx, my, ms, i, c = SI) => { const [x, y, w, h] = gcell(i, c); return [mx + (x + w / 2) * ms, my + (y + h / 2) * ms]; };
const dpos = (mx, my, ms, c = DI) => { const [x, y, w, h] = dcell(c); return [mx + (x + w / 2) * ms, my + (y + h / 2) * ms]; };
// 一个移动中的小方块（KV 的一片）
const chip = (x, y, s, col, alpha = 1) => rr(x - s / 2, y - s / 2, s, s, col, alpha);
// 会话的 8 片从显存移到内存（k = 0..1，反向时从内存回到显存）
function slicesMove(mx, my, ms, k, up = false, alpha = 1) {
  if (k <= 0 || k >= 1) return;
  const [dx, dy] = dpos(mx, my, ms);
  for (let i = 0; i < 8; i++) {
    const [gx, gy] = gpos(mx, my, ms, i), kk = ease(clamp(k * 1.15 - i * 0.02));
    const [x0, y0, x1, y1] = up ? [dx, dy, gx, gy] : [gx, gy, dx, dy];
    const mid = up ? kk : kk;
    const x = mix(x0, x1, mid), y = mix(y0, y1, mid);
    chip(x, y, 14 * ms + 4, C.purple, alpha);
  }
}
// 会话格：8 条竖纹（8 张卡的切片）
function sessionDram(c, alpha) { return alpha > 0.5 ? C.purple : EMPTY; }

// ================= 1 开场 =================
const M1 = { x: 96, y: 300, s: 1 };
function sOpen(u) {
  const k = I.open;
  // 封面：大标题，3.6 s 后缩到左上角
  const sh = p(u, 3.6, 4.6);
  if (sh < 1) {
    const size = mix(80, 28, sh), x = mix(960, 96, sh), y = mix(190, 76, sh);
    txt('KV cache 的分层存储与传输', x, y, { size, weight: 700, align: sh < 0.5 ? 'center' : 'left', alpha: 1 - sh * 0.999 * (sh > 0.5 ? 1 : 0) });
    txt('存放位置 · 传输时机 · 传输方式', 960, 262, { size: 36, weight: 600, color: C.gray, align: 'center', alpha: 1 - p(u, 3.2, 3.8) });
  }
  if (u > 3.6) header(k, u, 4.4);
  // 灰色请求：开始 22 格，5.5 s 起逐格填满；会话在 8.2 s 被淘汰，它的格随后被新请求占用
  const fillN = Math.round(mix(22, 42, lin(u, 5.8, 9.6)));
  const evict = lin(u, 8.2, 8.9), taken = u > 9.0;
  const order = [...Array(42).keys()].filter((c) => c !== SI);
  const cells = (i, c) => {
    if (c === SI) return taken ? OTHER : mixc(C.purple, '#ffffff', evict * 0.8);
    const r = order.indexOf(c); return r < fillN - 1 ? OTHER : EMPTY;
  };
  at(M1.x, M1.y, M1.s, () => machine({ cells, tiers: [1, 0.25, 0.25], pcie: 0.25 }));
  // 会话的标注
  const [cx, cy] = gpos(M1.x, M1.y, M1.s, 7);
  if (u < 9) {
    const a = fade(u, 1.0) * (1 - lin(u, 8.4, 9));
    sr(gpos(M1.x, M1.y, 1, 0)[0] - 10, cy - 13, cx - gpos(M1.x, M1.y, 1, 0)[0] + 20, 26, C.purple, 3, a);
    callout(cx + 12, cy, 1180, cy, '本例会话：32K token，10 GiB', { color: C.purple, alpha: a });
    note('8 张卡各存八分之一：1.25 GiB', 1190, cy + 40, { alpha: a });
    note('每格 = 一个 32K 会话在一张卡上的 KV', 1190, cy + 72, { alpha: a * fade(u, 2.2) });
  }
  if (u > 5.5 && u < 10.5) {
    const a = fade(u, 5.8) * (1 - lin(u, 10, 10.5));
    lab('新请求持续到达', 1190, 420, { color: C.ink2, alpha: a });
    if (u > 8.2) { lab('会话被淘汰', 1190, 470, { color: C.red, alpha: fade(u, 8.3) * a }); }
  }
  // 重算：按真实速率走 1.7 s
  if (u > 10.5) {
    const a = fade(u, 10.6), t = clamp(u - 11.2, 0, 1.7);
    lab('会话恢复：重新 prefill 32K token', 1190, 400, { alpha: a });
    rr(1190, 430, 630, 40, C.paper2, a); rr(1190, 430, 630 * t / 1.7, 40, C.red, a);
    txt(fmtT(t), 1820, 512, { size: 34, weight: 700, color: C.red, align: 'right', alpha: a });
    note('估算：6.7 PFLOP ÷（8 × 989 TFLOPS × 50%）', 1190, 512, { alpha: a, size: 20 });
    placeholder(1190, 580, 630, ['8 × H100 上实测 Llama-3.3-70B 的 32K prefill 时间', '替换 1.7 s 的估算；需要 Modal 8 × H100'], fade(u, 13.2));
  }
}
ev(I.open, 0.6, 'pop'); ev(I.open, 5.8, 'whoosh'); ev(I.open, 8.3, 'click'); ev(I.open, 11.2, 'tick'); ev(I.open, 12.9, 'land');

// ================= 2 四层存储（真实容量） =================
// 一格 = 一个 10 GiB 会话；每行 42 格，显存 42 格、内存 150 格、SSD 约 2,800 格。宽度固定，高度按容量；镜头从 1 拉远到 0.15。
const ST = { x: 160, top: 330, w: 900, row: 60 };
const TIERS = [
  { name: '显存 HBM', cap: '8 × 80 GB，扣除权重约 425 GiB', n: 42, col: C.blue },
  { name: '主机内存 DRAM', cap: '2 TB，留给 KV 约 1.5 TiB', n: 150, col: DRAMC },
  { name: '本地 SSD', cap: '8 × 3.84 TB NVMe，约 28 TiB', n: 2800, col: SSDC },
];
function sTiers(u) {
  const k = I.tiers; header(k, u);
  const zoom = mix(1, 0.14, p(u, 6.2, 10.5)), aStack = 1 - p(u, 17.3, 18.1);
  if (aStack > 0.003) {
    let y = ST.top, lastL = -99; const cw = ST.w / 42;
    TIERS.forEach((t, j) => {
      const show = j === 0 ? fade(u, 0.5) : fade(u, j === 1 ? 5.6 : 7.2, 0.8);
      const rows = Math.ceil(t.n / 42), h = rows * ST.row * zoom, a = show * aStack;
      if (a > 0.003 && y < 1000) {
        const hh = Math.min(h, 1000 - y);
        rr(ST.x, y, ST.w, hh, mixc('#ffffff', t.col, 0.18), a);
        // 格子：缩小后格线太密，只在格高 ≥ 6 px 时画
        g.save(); g.beginPath(); g.rect(ST.x, y, ST.w, hh); g.clip();
        const ch = ST.row * zoom;
        for (let c = 0; c < t.n && c < 42 * 80; c++) {
          const r = Math.floor(c / 42), x = ST.x + (c % 42) * cw, yy = y + r * ch;
          if (yy > 1000) break;
          const col = j === 0 && c === SI ? C.purple : mixc('#ffffff', t.col, 0.55);
          rr(x + 1.5, yy + Math.min(1.5, ch * 0.1), cw - 3, Math.max(0.6, ch - Math.min(3, ch * 0.2)), col, a);
        }
        g.restore();
        sr(ST.x, y, ST.w, hh, t.col, 2.5, a);
        const ly = Math.max(lastL + 72, Math.min(y + Math.min(h, 600) / 2 + 10, 940)); lastL = ly;
        lab(`${t.name}`, ST.x + ST.w + 36, ly, { color: t.col, alpha: a });
        note(`${t.cap} · ${t.n.toLocaleString()} 个会话`, ST.x + ST.w + 36, ly + 32, { alpha: a, color: C.ink2 });
      }
      y += h + 10;
    });
    if (zoom < 0.95) note('高度按容量绘制：显存仅占顶部一条', ST.x, ST.top - 18, { alpha: fade(u, 9.5) * aStack });
    // 共享池
    const ap = fade(u, 12.6) * aStack;
    if (ap > 0.003) {
      rr(1096, 640, 728, 236, PANEL, ap); sr(1096, 640, 728, 236, C.red, 2.5, ap);
      lab('跨机器的共享池', 1120, 684, { color: C.red, alpha: ap });
      txt('单机内存约 1.5 TiB，约 500 万 token', 1120, 728, { size: 22, weight: 600, color: C.ink2, alpha: ap });
      txt('Mooncake 的 trace：缓存 5000 万 token', 1120, 768, { size: 22, weight: 600, color: C.ink2, alpha: fade(u, 13.4) * aStack });
      txt('才接近最高命中率，约 15 TiB', 1120, 802, { size: 22, weight: 600, color: C.ink2, alpha: fade(u, 13.4) * aStack });
      txt('→ 约 10 台机器的内存组成一个池', 1120, 846, { size: 22, weight: 700, color: C.red, alpha: fade(u, 14.6) * aStack });
    }
  }
  // 示意图：剖面 + 管道粗细按带宽
  const am = p(u, 17.6, 18.6);
  if (am > 0.003) {
    at(96, 300, 1, () => machine({ alpha: am, cells: (i, c) => (c === SI ? C.purple : c < 30 ? OTHER : EMPTY), dram: (c) => (c < 60 ? OTHER : EMPTY), flowP: (u * 0.6) % 1, flowS: (u * 0.15) % 1 }));
    // 网卡：8 条红线从机箱右侧通往共享池
    for (let i = 0; i < 8; i++) pipe(1096, 410 + i * 9, 1240, 410 + i * 9, 5, C.red, am);
    rr(1240, 380, 584, 110, '#fff', am); sr(1240, 380, 584, 110, C.red, 2.5, am);
    lab('共享池（其他机器的内存）', 1264, 424, { color: C.red, alpha: am });
    note('经 RDMA 网络', 1264, 460, { alpha: am });
    const rows = [['显存', '每卡 3.35 TB/s', C.blue], ['PCIe Gen5', '每卡单向约 50 GB/s', C.tealD], ['RDMA 网卡', '每卡 400 Gb/s = 50 GB/s', C.red], ['NVMe RAID', '约 25 GB/s', SSDC]];
    rows.forEach(([a, b, col], i) => { const y = 560 + i * 52, aa = am * fade(u, 18.4 + i * 0.35); rr(1240, y - 24, 10, 32, col, aa); txt(a, 1266, y, { size: 24, weight: 700, color: col, alpha: aa }); txt(b, 1824, y, { size: 24, weight: 600, color: C.ink2, align: 'right', alpha: aa }); });
    note('管道粗细按带宽；剖面层高不按容量', 1240, 790, { alpha: am * fade(u, 19.8) });
  }
}
ev(I.tiers, 0.5, 'pop'); ev(I.tiers, 5.6, 'pop'); ev(I.tiers, 6.4, 'whoosh'); ev(I.tiers, 12.6, 'pop'); ev(I.tiers, 17.6, 'whoosh');

// ================= 3 搬一次要多久 =================
// 横道按真实速率增长：0..2 s 对应 520..1800 px
const L3 = { x: 540, pps: 620 };
const LANES = [
  { name: '内存 → 显存', sub: 'PCIe，8 × 50 GB/s', t: 0.027, col: DRAMC, bw: 400 },
  { name: '远端内存 → 显存', sub: 'RDMA，实测 190 GB/s', t: 0.057, col: C.red, bw: 190 },
  { name: '本地 SSD → 显存', sub: 'NVMe RAID，约 25 GB/s', t: 0.43, col: SSDC, bw: 25 },
  { name: '对象存储 → 显存', sub: 'S3 Express，约 1 GB/s', t: 10.7, col: C.gray, bw: 1 },
  { name: '重算', sub: '32K token prefill', t: 1.7, col: C.red, re: true },
];
const bwX = (b) => L3.x + Math.log10(b) / 3 * 1240;
function sCost(u) {
  const k = I.cost; header(k, u);
  const t = Math.max(0, u - 1.2), a0 = fade(u, 0.4);
  // 时间轴
  for (let s = 0; s <= 2; s += 0.5) { const x = L3.x + s * L3.pps; line(x, 300, x, 760, C.lg, 1.5, a0); note(`${s} s`, x, 290, { align: 'center', alpha: a0 }); }
  LANES.forEach((ln, i) => {
    const y = 330 + i * 86, a = a0;
    txt(ln.name, L3.x - 30, y + 26, { size: 25, weight: 700, align: 'right', alpha: a, color: ln.re ? C.red : C.ink });
    note(ln.sub, L3.x - 30, y + 56, { align: 'right', alpha: a, size: 19 });
    const d = Math.min(t, ln.t), wpx = Math.min(d * L3.pps, 2.08 * L3.pps);
    rr(L3.x, y, Math.max(wpx, t > 0 ? 3 : 0), 44, ln.re ? mixc(C.red, '#ffffff', 0.25) : ln.col, a);
    if (t >= ln.t && ln.t < 2.1) txt(fmtT(ln.t), L3.x + wpx + 14, y + 33, { size: 26, weight: 700, color: ln.re ? C.red : ln.col, alpha: a });
    if (ln.t > 2.1 && t > 2.08) { txt('约 11 s →', L3.x + 2.08 * L3.pps - 14, y + 32, { size: 26, weight: 700, color: '#fff', align: 'right', alpha: a }); }
    if (!ln.re && ln.t < 2 && t < ln.t) txt(fmtT(d), L3.x + wpx + 14, y + 33, { size: 22, weight: 600, color: C.gray, alpha: a });
  });
  // 重算的红线
  const ar = fade(u, bt(k, 1));
  line(L3.x + 1.7 * L3.pps, 312, L3.x + 1.7 * L3.pps, 760, C.red, 3, ar, [10, 8]);
  // 带宽轴与交叉点
  const ab = fade(u, bt(k, 2)), y0 = 880;
  if (ab > 0.003) {
    rr(L3.x, y0 - 30, bwX(6.3) - L3.x, 60, mixc('#ffffff', C.red, 0.12), ab);
    rr(bwX(6.3), y0 - 30, bwX(1000) - bwX(6.3), 60, mixc('#ffffff', C.teal, 0.14), ab);
    line(L3.x, y0, bwX(1000), y0, C.ink2, 2, ab);
    [1, 10, 100, 1000].forEach((b) => { line(bwX(b), y0 - 8, bwX(b), y0 + 8, C.ink2, 2, ab); note(`${b} GB/s`, bwX(b), y0 + 52, { align: 'center', alpha: ab, size: 19 }); });
    note('重算更快', L3.x + 12, y0 - 6, { alpha: ab, color: C.red, size: 20 });
    note('读回更快', bwX(1000) - 12, y0 - 6, { alpha: ab, color: C.tealD, size: 20, align: 'right' });
    LANES.filter((l) => l.bw).forEach((l) => { const x = bwX(l.bw); rr(x - 7, y0 - 7, 14, 14, l.col, ab); });
    line(bwX(6.3), y0 - 40, bwX(6.3), y0 + 30, C.red, 4, ab);
    txt('10 GiB ÷ 1.7 s ≈ 6.3 GB/s', bwX(6.3) - 40, y0 - 50, { size: 24, weight: 700, color: C.red, align: 'center', alpha: ab });
    txt('交叉点', L3.x - 30, y0 + 8, { size: 25, weight: 700, align: 'right', alpha: ab });
    const am = fade(u, bt(k, 3));
    [[6, 'A800'], [19, 'H800']].forEach(([b, n]) => { const x = bwX(b); line(x, y0 - 30, x, y0 + 30, C.ink, 3, am); txt(n, x, y0 + 56, { size: 19, weight: 700, align: 'center', alpha: am }); });
    note('Mooncake 论文：70B、8K 前缀的交叉点，8 × A800 为 6 GB/s，8 × H800 为 19 GB/s', 1824, y0 + 96, { alpha: am, size: 20, color: C.ink2, align: 'right' });
  }
}
ev(I.cost, 1.2, 'whoosh'); ev(I.cost, 1.23, 'land'); ev(I.cost, 1.26, 'land'); ev(I.cost, 1.63, 'land'); ev(I.cost, 2.9, 'click'); ev(I.cost, 5.0, 'pop'); ev(I.cost, 11.5, 'pop'); ev(I.cost, 18, 'pop');

// ================= 4 何时写下去 =================
// 两栏：同一个会话逐 block 生成（示意 16 个 block）；左栏立即写，右栏淘汰时再写
const NB = 16, bT = (j) => 0.8 + j * 0.38;
function writePanel(u, x, mode) {
  const k = I.write, ev8 = bt(k, 1), w = 820;
  const hy = 420, dy = 600, bw = 44, gap = 4.5;
  const bx = (j) => x + 20 + j * (bw + gap);
  txt('显存', x, hy - 14, { size: 21, weight: 700, color: C.blue });
  txt('内存', x, dy - 14, { size: 21, weight: 700, color: DRAMC });
  rr(x, hy, w, 70, '#fff'); sr(x, hy, w, 70, C.blue, 2.5);
  rr(x, dy, w, 70, '#fff'); sr(x, dy, w, 70, DRAMC, 2.5);
  // 计算条：一直在走；右栏淘汰时停下等待拷贝
  const stallA = ev8 + 0.3, stallB = ev8 + 0.3 + 3.2;
  const stall = mode === 'back' && u > stallA && u < stallB;
  txt('计算', x, hy - 52, { size: 19, weight: 600, color: C.gray });
  rr(x + 56, hy - 66, w - 56, 18, C.paper2);
  const prog = (uu) => (mode === 'back' ? Math.min(uu, stallA) + Math.max(0, uu - stallB) : uu);
  rr(x + 56, hy - 66, (w - 56) * clamp(prog(u) / 22), 18, C.blue);
  if (mode === 'back' && u > stallA) { const s0 = x + 56 + (w - 56) * stallA / 22, s1 = Math.min(u, stallB) - stallA; rr(s0, hy - 66, (w - 56) * s1 / 22 + 1, 18, C.red); }
  const evicted = u > ev8;
  for (let j = 0; j < NB; j++) {
    const f = lin(u, bT(j), bT(j) + 0.35);
    // 显存中的 block
    let aH = f;
    if (evicted) aH = mode === 'through' ? 1 - lin(u, ev8, ev8 + 0.4) : 1 - lin(u, ev8 + 0.3 + j * 0.2, ev8 + 0.5 + j * 0.2);
    rr(bx(j), hy + 12, bw * (u < bT(j) + 0.35 ? f : 1), 46, C.purple, aH);
    // 拷贝：立即写在写满后 0.3 s 出发；淘汰时再写在淘汰后逐个出发
    const go = mode === 'through' ? bT(j) + 0.45 : ev8 + 0.3 + j * 0.2, kk = p(u, go, go + 0.5);
    if (kk > 0 && kk < 1) chip(bx(j) + bw / 2, mix(hy + 35, dy + 35, kk), 22, C.purple);
    if (kk >= 1) rr(bx(j), dy + 12, bw, 46, C.purple);
  }
  pipe(x + w + 26, hy + 35, x + w + 26, dy + 35, 7, C.teal, 1, mode === 'through' ? (u < 8 ? (u * 1.2) % 1 : -1) : (stall ? (u * 1.2) % 1 : -1));
  if (evicted) {
    if (mode === 'through') tag('淘汰：副本已存在，直接释放', x + 20, hy + 120, C.tealD, fade(u, ev8 + 0.2));
    else { tag('淘汰：拷贝完成后才能释放', x + 20, hy + 120, C.red, fade(u, ev8 + 0.2)); }
  }
}
function sWrite(u) {
  const k = I.write; header(k, u);
  lab('立即写入（write-through）', 96, 300, { color: C.ink }); note('vLLM OffloadingConnector · SGLang 默认', 96, 332);
  lab('淘汰时写入（write-back）', 1004, 300, { color: C.ink }); note('SGLang write_back · LMCache lazy offload', 1004, 332);
  writePanel(u, 96, 'through'); writePanel(u, 1004, 'back');
  const a3 = fade(u, bt(k, 2));
  if (a3 > 0.003) { sr(90, 590, 832, 82, C.amber, 3, a3); txt('每个 block：一次 PCIe 拷贝 + 一份内存', 96, 712, { size: 23, weight: 700, color: C.amber, alpha: a3 }); }
  code(96, 740, 820, ['vllm serve meta-llama/Llama-3.3-70B-Instruct -tp 8 \\', '  {t:--kv-offloading-size} 1536 --kv-offloading-backend native'], { size: 19, alpha: fade(u, 3), title: 'CPU 层 1.5 TiB（vLLM）' });
  const a4 = fade(u, bt(k, 3));
  if (a4 > 0.003) {
    rr(1004, 740, 820, 210, PANEL, a4); sr(1004, 740, 820, 210, C.lg, 2, a4);
    rich('{t:内存层须大于显存}：否则副本被反复覆盖（churn）', 1026, 790, 23, 700, a4);
    note('vLLM、Dynamo 文档均有同样的说明', 1026, 824, { alpha: a4, size: 20 });
    rich('{y:SSD 只写重复命中的 block}：写入量决定寿命', 1026, 880, 23, 700, fade(u, bt(k, 3) + 1.2));
    note('Dynamo FrequencyFilter；SGLang write_through_selective（第二次命中才写入）', 1026, 914, { alpha: fade(u, bt(k, 3) + 1.2), size: 20 });
  }
}
for (let j = 0; j < NB; j++) ev(I.write, 0.8 + j * 0.38 + 0.95, 'tick');
ev(I.write, 8.0, 'click'); ev(I.write, 13.5, 'pop'); ev(I.write, 17, 'pop');

// ================= 5 何时读回来 =================
function step(x, y, w, s, col, a, mark = '') {
  rr(x, y, w, 58, '#fff', a); sr(x, y, w, 58, col, 2.5, a);
  txt(s, x + 18, y + 38, { size: 22, weight: 700, color: col, alpha: a });
  if (mark) txt(mark, x + w - 18, y + 40, { size: 30, weight: 700, color: mark === '✓' ? C.tealD : C.red, align: 'right', alpha: a });
}
// 甘特图：L 层，每层加载 lt、计算 ct（毫秒）；overlap 时第 i 层的计算等第 i 层加载完、第 i-1 层算完
function gantt(x, y, pxms, L, lt, ct, overlap, a) {
  let tl = 0, tc = 0;
  for (let i = 0; i < L; i++) {
    const l0 = overlap ? tl : i * lt, l1 = l0 + lt; tl = l1;
    const c0 = overlap ? Math.max(l1, tc) : L * lt + i * ct, c1 = c0 + ct; tc = c1;
    rr(x + l0 * pxms, y, Math.max(1, lt * pxms - (overlap ? 0.6 : 0)), 22, i % 2 ? C.teal : C.tealD, a);
    rr(x + c0 * pxms, y + 28, Math.max(1, ct * pxms - 0.6), 22, i % 2 ? C.blue : '#1a3fae', a);
  }
  return tc;
}
function sRead(u) {
  const k = I.read; header(k, u);
  // 左：请求与调度
  const a0 = fade(u, 0.4);
  rr(96, 290, 600, 64, C.ink, a0); txt('第 2 轮：32K 历史 + 1K 新输入', 120, 332, { size: 24, weight: 700, color: '#fff', alpha: a0 });
  step(96, 380, 290, '查显存', C.blue, fade(u, 1.4), '✗');
  step(406, 380, 290, '查内存', DRAMC, fade(u, 2.8), '✓');
  const a1 = fade(u, bt(k, 1)), loaded = u > bt(k, 1) + 2.6;
  if (a1 > 0.003) {
    rr(96, 466, 600, 58, loaded ? mixc('#ffffff', C.blue, 0.15) : '#fff4e0', a1); sr(96, 466, 600, 58, loaded ? C.blue : C.amber, 2.5, a1);
    txt(loaded ? 'RUNNING：参与调度' : 'WAITING_FOR_REMOTE_KVS', 116, 503, { size: 22, weight: 700, color: loaded ? C.blue : C.amber, font: MONO, alpha: a1 });
    code(96, 546, 600, ['{g:# vllm/v1/core/sched/scheduler.py}', 'request.status = RequestStatus.{a:WAITING_FOR_REMOTE_KVS}'], { size: 17, alpha: a1 });
  }
  // 右：显存与内存，8 条 PCIe
  const rx = 800, ry = 290, ra = fade(u, 0.6);
  rr(rx, ry, 1024, 96, '#fff', ra); sr(rx, ry, 1024, 96, C.blue, 2.5, ra); txt('显存', rx + 14, ry + 30, { size: 19, weight: 700, color: C.blue, alpha: ra });
  rr(rx, ry + 170, 1024, 70, '#fff', ra); sr(rx, ry + 170, 1024, 70, DRAMC, 2.5, ra); txt('内存', rx + 14, ry + 196, { size: 19, weight: 700, color: DRAMC, alpha: ra });
  rr(rx, ry + 300, 1024, 60, '#fff', ra); sr(rx, ry + 300, 1024, 60, SSDC, 2.5, ra); txt('SSD', rx + 14, ry + 326, { size: 19, weight: 700, color: SSDC, alpha: ra });
  const kL = p(u, bt(k, 1) + 0.4, bt(k, 1) + 2.4);
  for (let i = 0; i < 8; i++) {
    const gx = rx + 110 + i * 112;
    rr(gx, ry + 40, 96, 44, EMPTY, ra); txt(`GPU ${i}`, gx + 48, ry + 30, { size: 15, weight: 600, color: C.gray, align: 'center', alpha: ra });
    pipe(gx + 48, ry + 96, gx + 48, ry + 170, 7, C.teal, ra, kL > 0 && kL < 1 ? (u * 1.5) % 1 : -1);
    const dx = rx + 480, dyy = ry + 205;
    if (kL <= 0) rr(dx - 8 + i * 2, dyy - 14, 4, 28, C.purple, ra);
    if (kL > 0 && kL < 1) chip(mix(dx + i * 2, gx + 48, kL), mix(dyy, ry + 62, kL), 16, C.purple, ra);
    if (kL >= 1) rr(gx + 4, ry + 44, 88, 36, C.purple, ra);
  }
  if (u > bt(k, 1) && u < bt(k, 1) + 3) note('8 × 1.25 GiB 并行，约 27 ms（慢放）', rx + 1024, ry + 140, { align: 'right', alpha: fade(u, bt(k, 1) + 0.2) });
  // 预取：另一个请求的 KV 在 SSD
  const a2 = fade(u, bt(k, 2));
  if (a2 > 0.003) {
    const kp = p(u, bt(k, 2) + 0.6, bt(k, 2) + 3.2);
    rr(rx + 700, ry + 312, 60, 36, C.amber, a2 * (1 - kp)); chip(rx + 730, mix(ry + 330, ry + 205, kp), 26, C.amber, kp > 0 && kp < 1 ? a2 : 0);
    if (kp >= 1) rr(rx + 700, ry + 187, 60, 36, C.amber, a2);
    pipe(rx + 800, ry + 300, rx + 800, ry + 240, 3, C.yel, a2, kp > 0 && kp < 1 ? (u * 0.7) % 1 : -1);
    tag('请求 B 排队期间：从 SSD 预取到内存', rx + 200, ry + 405, C.amber, a2);
  }
  // 下：甘特图（毫秒）
  const a3 = fade(u, bt(k, 3)), gx0 = 380, pxms = 14.2, gy = 760;
  if (a3 > 0.003) {
    const ct = mix(58, 7.3, p(u, bt(k, 4), bt(k, 4) + 1.5)) / 80, lt = 27 / 80;
    for (let s = 0; s <= 100; s += 20) { line(gx0 + s * pxms, gy - 10, gx0 + s * pxms, gy + 200, C.lg, 1.5, a3); note(`${s} ms`, gx0 + s * pxms, gy - 18, { align: 'center', alpha: a3, size: 18 }); }
    txt('先加载再计算', gx0 - 24, gy + 36, { size: 22, weight: 700, align: 'right', alpha: a3 });
    const e1 = gantt(gx0, gy + 4, pxms, 80, lt, ct, false, a3);
    txt('逐层重叠', gx0 - 24, gy + 136, { size: 22, weight: 700, align: 'right', alpha: a3 });
    const e2 = gantt(gx0, gy + 104, pxms, 80, lt, ct, true, a3);
    txt(`${e1.toFixed(0)} ms`, gx0 + e1 * pxms + 12, gy + 48, { size: 22, weight: 700, alpha: a3 });
    txt(`${e2.toFixed(0)} ms`, gx0 + e2 * pxms + 12, gy + 148, { size: 22, weight: 700, color: C.blue, alpha: a3 });
    rr(96, gy + 176, 16, 16, C.teal, a3); note('加载（80 层）', 120, gy + 192, { alpha: a3, size: 19 });
    rr(270, gy + 176, 16, 16, C.blue, a3); note(u > bt(k, 4) ? '计算：128 个新 token' : '计算：1K 个新 token', 294, gy + 192, { alpha: a3, size: 19 });
    note('估算；SGLang：layer_transfer_counter.wait_until(layer_id)', 1824, gy + 192, { align: 'right', alpha: a3, size: 19 });
  }
}
ev(I.read, 0.4, 'pop'); ev(I.read, 1.4, 'click'); ev(I.read, 2.8, 'click'); ev(I.read, 5.9, 'whoosh'); ev(I.read, 8.1, 'land'); ev(I.read, 11.1, 'whoosh'); ev(I.read, 13.7, 'land'); ev(I.read, 15, 'pop'); ev(I.read, 20.5, 'pop');

// ================= 6 传输的单位 =================
// 消息大小与吞吐：LMCache 论文引用的 RCCL 数据（8 × 400G 网卡）
const TPUT = [[64 * 1024, 4], [1024 * 1024, 30], [16 * 1024 * 1024, 49], [64 * 1024 * 1024, 49]];
const CH = { x: 960, y: 300, w: 820, h: 330 };
const sizeX = (b) => CH.x + (Math.log2(b) - 12) / (26 - 12) * CH.w;   // 4 KiB .. 64 MiB
const tpY = (v) => CH.y + CH.h - v / 50 * CH.h;
function sUnit(u) {
  const k = I.unit; header(k, u);
  // 左：一个 block 的大小
  const a0 = fade(u, 0.4);
  rr(96, 300, 760, 330, PANEL, a0); sr(96, 300, 760, 330, C.lg, 2, a0);
  lab('一个 block、一层、一张卡', 120, 344, { alpha: a0 });
  const fl = ['16 token', '× 1 个 KV head', '× 128 维', '× K、V', '× 2 B'];
  let fx = 120; fl.forEach((s, i) => { txt(s, fx, 410, { size: 24, weight: 600, color: C.ink2, alpha: fade(u, 0.8 + i * 0.35) }); fx += tw(s, 24, 600) + 18; });
  rich('= {p:8 KiB}', 120, 470, 40, 700, fade(u, 2.8));
  note('TP = 8 时每张卡存 1 个 KV head；整段会话有 2,048 个 block × 80 层', 120, 520, { alpha: fade(u, 3.4), size: 20 });
  note('逐个发送：每卡 16 万条消息', 120, 556, { alpha: fade(u, 3.8), size: 20 });
  for (let i = 0; i < 24; i++) { const f = ((u * 0.9) + i / 24) % 1; chip(130 + f * 690, 600, 8, C.purple, fade(u, 4) * (u < 12 ? 1 : 1 - lin(u, 12, 12.5))); }
  // 右：吞吐曲线
  const a1 = fade(u, bt(k, 1));
  if (a1 > 0.003) {
    line(CH.x, CH.y + CH.h, CH.x + CH.w, CH.y + CH.h, C.ink2, 2, a1); line(CH.x, CH.y, CH.x, CH.y + CH.h, C.ink2, 2, a1);
    [[4096, '4 KiB'], [65536, '64 KiB'], [1 << 20, '1 MiB'], [16 << 20, '16 MiB'], [64 << 20, '64 MiB']].forEach(([b, s]) => note(s, sizeX(b), CH.y + CH.h + 30, { align: 'center', alpha: a1, size: 18 }));
    [0, 25, 50].forEach((v) => note(`${v} GB/s`, CH.x - 10, tpY(v) + 6, { align: 'right', alpha: a1, size: 18 }));
    const kk = p(u, bt(k, 1), bt(k, 1) + 2);
    g.globalAlpha = a1; g.strokeStyle = C.blue; g.lineWidth = 4; g.beginPath();
    TPUT.forEach(([b, v], i) => { const x = sizeX(b), y = tpY(v); if (x > CH.x + CH.w * kk + sizeX(65536) - CH.x) return; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.globalAlpha = 1;
    TPUT.slice(0, 3).forEach(([b, v], i) => { rr(sizeX(b) - 7, tpY(v) - 7, 14, 14, C.blue, a1 * fade(u, bt(k, 1) + i * 0.6)); txt(`${v} GB/s`, sizeX(b) + 14, tpY(v) - 12, { size: 21, weight: 700, color: C.blue, alpha: a1 * fade(u, bt(k, 1) + i * 0.6) }); });
    line(sizeX(8192), CH.y + 20, sizeX(8192), CH.y + CH.h, C.purple, 3, a1, [8, 6]);
    txt('8 KiB', sizeX(8192) + 8, CH.y + 40, { size: 21, weight: 700, color: C.purple, alpha: a1 });
    note('LMCache 论文引用的 RCCL 数据，8 × 400G 网卡', CH.x + CH.w, CH.y + CH.h + 62, { align: 'right', alpha: a1, size: 19 });
    const a2 = fade(u, bt(k, 2) + 2.5);
    line(sizeX(10 << 20), CH.y + 20, sizeX(10 << 20), CH.y + CH.h, C.tealD, 3, a2, [8, 6]);
    txt('10 MiB', sizeX(10 << 20) - 8, CH.y + CH.h - 20, { size: 21, weight: 700, color: C.tealD, align: 'right', alpha: a2 });
  }
  // 下：布局
  const a3 = fade(u, bt(k, 2)), ly = 750, kk = p(u, bt(k, 2) + 1, bt(k, 2) + 3.5);
  if (a3 > 0.003) {
    lab('layer_first', 96, ly, { alpha: a3, color: C.ink2 }); txt('(2, L, tokens, H, D)', 96, ly + 34, { size: 20, weight: 500, font: MONO, color: C.gray, alpha: a3 });
    lab('page_first', 1060, ly, { alpha: a3, color: C.tealD }); txt('(2, tokens, L, H, D)', 1060, ly + 34, { size: 20, weight: 500, font: MONO, color: C.gray, alpha: a3 });
    // 左：12 行代表 80 层，每行中这一页的片段在同一列
    for (let r = 0; r < 10; r++) {
      const y = ly + 56 + r * 18;
      rr(400, y, 560, 12, EMPTY, a3);
      const x0 = 400 + 260, x1 = 1060 + (r * 52), y1 = ly + 112;
      rr(mix(x0, x1, kk), mix(y, y1, kk), mix(40, 52, kk), 12, C.purple, a3);
    }
    txt('⋮ 80 层', 400 + 280, ly + 56 + 10 * 18 + 22, { size: 18, weight: 600, color: C.gray, align: 'center', alpha: a3 * (1 - kk) });
    sr(1060, ly + 104, 520, 28, C.tealD, 2.5, a3 * kk);
    note('一页的 80 层合并为一段，整块写入下层', 1060, ly + 170, { alpha: a3 * kk, size: 20 });
    rich('LMCache：{t:256 token 一个 chunk}，每卡 {t:10 MiB}', 1060, ly + 212, 23, 700, fade(u, bt(k, 2) + 2.5));
    arrow(980, ly + 120, 1040, ly + 120, C.ink2, a3 * fade(u, bt(k, 2) + 0.8), 3);
  }
}
ev(I.unit, 0.4, 'pop'); ev(I.unit, 2.8, 'land'); ev(I.unit, 6.0, 'pop'); ev(I.unit, 12, 'pop'); ev(I.unit, 13, 'whoosh'); ev(I.unit, 15.5, 'land');

// ================= 7 跨实例复用 =================
const MA = { x: 96, y: 340, s: 0.5 }, MB = { x: 640, y: 340, s: 0.5 };
function sShare(u) {
  const k = I.share; header(k, u);
  const kPull = p(u, bt(k, 2) + 1.2, bt(k, 2) + 3.4);
  at(MA.x, MA.y, MA.s, () => machine({ title: '实例 A', cells: (i, c) => (c === SI ? C.purple : c < 30 ? OTHER : EMPTY), dram: (c) => (c < 70 ? OTHER : EMPTY) }));
  at(MB.x, MB.y, MB.s, () => machine({ title: '实例 B', alpha: fade(u, 0.4), cells: (i, c) => (c === SI && kPull >= 1 ? C.purple : c < 26 ? OTHER : EMPTY), dram: (c) => (c < 50 ? OTHER : EMPTY) }));
  // 共享池
  const py = 700, pa = fade(u, 0.6);
  rr(96, py, 1044, 90, mixc('#ffffff', C.red, 0.1), pa); sr(96, py, 1044, 90, C.red, 2.5, pa);
  lab('共享池：各机内存经 RDMA 互联', 120, py + 38, { color: C.red, alpha: pa }); note('Mooncake Store · LMCache 远端 · SGLang L3', 120, py + 72, { alpha: pa, size: 19 });
  for (let c = 0; c < 26; c++) rr(640 + c * 18, py + 22, 14, 46, c === 9 ? C.purple : OTHER, pa);
  // 请求落到 B
  const ar = fade(u, 0.8);
  if (u < bt(k, 2)) tag('第 3 轮请求 → 实例 B：显存中无此 KV', 640, 322, C.ink, ar);
  // 全局索引
  const ia = fade(u, bt(k, 1)), ix = 1220, iy = 300;
  if (ia > 0.003) {
    rr(ix, iy, 604, 330, PANEL, ia); sr(ix, iy, 604, 330, C.ink2, 2, ia);
    lab('全局索引：prefix hash → 位置', ix + 20, iy + 40, { alpha: ia });
    const rows = [['h(块 1)', 'A 显存 · 池'], ['h(块 2)', 'A 显存 · 池'], ['…', ''], ['h(块 2048)', 'A 显存 · 池']];
    rows.forEach(([h, loc], i) => { const y = iy + 90 + i * 52, a = ia * fade(u, bt(k, 1) + 0.6 + i * 0.3); if (h !== '…') { rr(ix + 20, y - 28, 180, 38, C.purple, a); txt(h, ix + 110, y, { size: 20, weight: 700, color: '#fff', align: 'center', font: MONO, alpha: a }); txt(loc, ix + 230, y, { size: 22, weight: 600, color: C.ink2, alpha: a }); } else txt('⋮', ix + 110, y, { size: 24, color: C.gray, align: 'center', alpha: a }); });
    note('实例上报 BlockStored / BlockRemoved 事件', ix + 20, iy + 312, { alpha: ia, size: 19 });
    // 事件从 A、B 飞向索引
    for (let j = 0; j < 4; j++) { const f = lin(u, bt(k, 1) + j * 0.5, bt(k, 1) + j * 0.5 + 0.9); if (f > 0 && f < 1) { const sx = j % 2 ? 1140 : 600, sy = 300; chip(mix(sx, ix, f), mix(sy, iy + 160, f) - Math.sin(f * 3.14) * 40, 12, C.ink2); } }
    code(ix, iy + 350, 604, ['{g:# Dynamo KV router 的代价}', 'prefill_blocks − device_overlap', '  − 0.75·host_overlap − 0.25·disk_overlap'], { size: 18, alpha: fade(u, bt(k, 2)) });
  }
  // 两条路
  const a2 = fade(u, bt(k, 2));
  if (a2 > 0.003) {
    g.globalAlpha = a2; g.strokeStyle = C.blue; g.lineWidth = 4; g.setLineDash([12, 8]); g.beginPath(); g.moveTo(890, 336); g.quadraticCurveTo(620, 262, 360, 336); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    txt('① 路由回 A：显存命中', 500, 296, { size: 22, weight: 700, color: C.blue, alpha: a2 });
    pipe(1170, py, 1170, MB.y + 90, 8, C.red, a2, kPull > 0 && kPull < 1 ? (u * 1.3) % 1 : -1); pipe(1170, MB.y + 90, 1142, MB.y + 90, 8, C.red, a2);
    if (kPull > 0 && kPull < 1) for (let i = 0; i < 8; i++) chip(mix(803, gpos(MB.x, MB.y, MB.s, i)[0], kPull), mix(py + 45, gpos(MB.x, MB.y, MB.s, i)[1], kPull), 12, C.purple);
    txt('② B 从池中拉取：10 GiB ÷ 190 GB/s ≈ 57 ms', 1140, py - 22, { size: 21, weight: 700, color: C.red, align: 'right', alpha: fade(u, bt(k, 2) + 0.8) });
  }
  // 命中率
  const a3 = fade(u, bt(k, 3));
  if (a3 > 0.003) {
    const by = 860;
    lab('Mooncake：全局池与单机缓存的命中率', 96, by, { alpha: a3 });
    rr(96, by + 22, 300, 40, C.gray, a3); txt('单机缓存 1×', 110, by + 50, { size: 21, weight: 700, color: '#fff', alpha: a3 });
    rr(96, by + 76, 300 * 2.36 * p(u, bt(k, 3) + 0.2, bt(k, 3) + 1.2), 40, C.red, a3); txt('全局池 最高 2.36×', 110, by + 104, { size: 21, weight: 700, color: '#fff', alpha: a3 });
    note('FAST\'25 §5.3.2，回放 trace', 830, by + 104, { alpha: a3, size: 19 });
  }
}
ev(I.share, 0.8, 'pop'); ev(I.share, 6, 'tick'); ev(I.share, 6.5, 'tick'); ev(I.share, 7, 'tick'); ev(I.share, 12, 'pop'); ev(I.share, 13.2, 'whoosh'); ev(I.share, 15.4, 'land'); ev(I.share, 18.7, 'pop');

// ================= 8 PD 分离 =================
// 两台机器的数据路径：GPU → PCIe 交换机 → 网卡 → 网络 → 网卡 → PCIe 交换机 → GPU；CPU 内存被跳过
function pdNode(x, name, col, u, kv, a) {
  rr(x, 300, 660, 470, '#f7f4ec', a); sr(x, 300, 660, 470, C.ink2, 2, a);
  lab(name, x + 20, 338, { color: col, alpha: a });
  for (let i = 0; i < 8; i++) {
    const gx = x + 24 + i * 78;
    rr(gx, 370, 68, 80, '#fff', a); sr(gx, 370, 68, 80, C.blue, 2, a);
    txt(`GPU ${i}`, gx + 34, 392, { size: 14, weight: 600, color: C.gray, align: 'center', alpha: a });
    if (kv > 0) rr(gx + 8, 402, 52, 38, C.purple, a * kv);
    rr(gx + 4, 500, 60, 34, mixc('#ffffff', C.ink2, 0.12), a); txt('PCIe', gx + 34, 523, { size: 14, weight: 600, color: C.ink2, align: 'center', alpha: a });
    rr(gx + 4, 580, 60, 34, mixc('#ffffff', C.red, 0.2), a); txt('NIC', gx + 34, 603, { size: 14, weight: 700, color: C.red, align: 'center', alpha: a });
  }
  rr(x + 24, 660, 612, 86, mixc('#ffffff', C.gray, 0.15), a); txt('CPU 与主机内存', x + 44, 712, { size: 22, weight: 700, color: C.gray, alpha: a });
}
function sPD(u) {
  const k = I.pd; header(k, u);
  const kT = p(u, bt(k, 1) + 2.2, bt(k, 2) + 3.5), a = fade(u, 0.3);
  pdNode(96, 'prefill 实例 P', C.ink, u, kT < 1 ? 1 : 1, a);
  pdNode(1164, 'decode 实例 D', C.ink, u, kT >= 1 ? 1 : 0, a);
  // 网络
  rr(756, 570, 408, 54, mixc('#ffffff', C.red, 0.12), a); txt('RDMA 网络', 960, 558, { size: 22, weight: 700, color: C.red, align: 'center', alpha: a });
  // router 与位置信息
  const a0 = fade(u, bt(k, 0) + 0.3);
  rr(860, 300, 200, 60, C.ink, a0); txt('router', 960, 340, { size: 24, weight: 700, color: '#fff', align: 'center', alpha: a0 });
  const km = p(u, 1.5, 4.5);
  if (km > 0 && km < 1) { const x = km < 0.5 ? mix(700, 960, km * 2) : mix(960, 1220, km * 2 - 1), y = km < 0.5 ? mix(420, 330, km * 2) : mix(330, 420, km * 2 - 1); tag('remote_block_ids', x - 80, y, C.purple, 1, 17); }
  if (km >= 1 && u < bt(k, 1) + 2) tag('remote_block_ids', 1180, 470, C.purple, 1 - lin(u, bt(k, 1) + 1.5, bt(k, 1) + 2), 17);
  // READ 请求
  const ar = fade(u, bt(k, 1));
  if (ar > 0.003 && u < bt(k, 2) + 4) { arrow(1180, 470, 760, 470, C.ink2, ar * (1 - lin(u, bt(k, 2) + 3, bt(k, 2) + 4)), 3); txt('NIXL READ', 970, 456, { size: 22, weight: 700, font: MONO, alpha: ar * (1 - lin(u, bt(k, 2) + 3, bt(k, 2) + 4)), align: 'center' }); }
  // 数据路径：每张卡一条（第 3 节拍前只点亮 GPU 0）
  const all = fade(u, bt(k, 2));
  for (let i = 0; i < 8; i++) {
    const aa = i === 0 ? ar : ar * all;
    if (aa <= 0.003) continue;
    const xs = 96 + 24 + i * 78 + 34, xd = 1164 + 24 + i * 78 + 34, yy = 604 - 0;
    const path = [[xs, 450], [xs, 500], [xs, 580], [xs, 597 + i * 2], [xd, 597 + i * 2], [xd, 580], [xd, 500], [xd, 450]];
    g.globalAlpha = aa * 0.55; g.strokeStyle = C.purple; g.lineWidth = 3; g.beginPath(); path.forEach(([x, y], j) => (j ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke(); g.globalAlpha = 1;
    if (kT > 0 && kT < 1) {
      const L = path.slice(1).reduce((s, q, j) => s + Math.hypot(q[0] - path[j][0], q[1] - path[j][1]), 0);
      for (let m = 0; m < 3; m++) {
        let d = ((kT * 3 + m / 3 + i * 0.05) % 1) * L, j = 0;
        while (j < path.length - 1) { const sl = Math.hypot(path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1]); if (d <= sl) break; d -= sl; j++; }
        if (j < path.length - 1) { const sl = Math.hypot(path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1]) || 1, f = d / sl; chip(mix(path[j][0], path[j + 1][0], f), mix(path[j][1], path[j + 1][1], f), 12, C.purple, aa); }
      }
    }
  }
  if (ar > 0.003) { line(120, 650, 736, 750, C.red, 4, ar * 0.9); txt('不经过', 640, 712, { size: 22, weight: 700, color: C.red, alpha: ar }); line(1188, 650, 1804, 750, C.red, 4, ar * 0.9); txt('不经过', 1708, 712, { size: 22, weight: 700, color: C.red, alpha: ar }); }
  if (u > bt(k, 1) && u < bt(k, 2) + 4) note('GPUDirect RDMA：网卡直接读写显存', 960, 680, { align: 'center', alpha: ar, size: 20 });
  // 实测表
  const a3 = fade(u, bt(k, 3));
  if (a3 > 0.003) {
    const ty = 790; rr(96, ty, 1728, 204, PANEL, a3); sr(96, ty, 1728, 204, C.lg, 2, a3);
    const cols = [['', 120], ['prompt', 640], ['KV', 840], ['传输时间', 1040], ['带宽', 1260], ['占 TTFT', 1540]];
    cols.forEach(([s, x]) => note(s, x, ty + 36, { alpha: a3, size: 19 }));
    const row = (y, cells, col, aa) => cells.forEach((s, j) => txt(s, cols[j][1], y, { size: j ? 24 : 20, weight: j ? 700 : 600, color: j ? col : C.ink2, alpha: aa }));
    row(ty + 80, ['Mooncake 文档实测：Qwen3-8B，H800，RDMA', '32K token', '4.50 GB', '31.65 ms', '142 GB/s', '4.2%'], C.ink, a3);
    row(ty + 126, ['本片的例子（估算）：70B，RDMA', '32K token', '10 GiB', '57 ms', '190 GB/s', '约 3%'], C.purple, fade(u, bt(k, 3) + 1));
    row(ty + 172, ['同一 NVL72 机柜内经 NVLink（估算）', '32K token', '10 GiB', '约 12 ms', '每卡约 900 GB/s', '<1%'], C.blue, fade(u, bt(k, 3) + 2));
  }
}
ev(I.pd, 1.5, 'whoosh'); ev(I.pd, 6.5, 'click'); ev(I.pd, 8.7, 'whoosh'); ev(I.pd, 12.5, 'pop'); ev(I.pd, 16, 'land'); ev(I.pd, 16.5, 'pop');

// ================= 9 逐层还是整块 =================
const G9 = { x: 380, pps: 560 };   // 0..1.8 s
function lane9(y, name, sub, kind, a, u) {
  txt(name, G9.x - 24, y + 22, { size: 23, weight: 700, align: 'right', alpha: a });
  note(sub, G9.x - 24, y + 52, { align: 'right', alpha: a, size: 18 });
  const T = 1.7, grow = clamp((u - 0.6) / 3) * 1.9;
  const draw = (t0, t1, yy, col) => { const a0 = Math.min(t0, grow), a1 = Math.min(t1, grow); if (a1 > a0) rr(G9.x + a0 * G9.pps, yy, Math.max(1.2, (a1 - a0) * G9.pps), 22, col, a); };
  if (kind === 'layer') for (let i = 0; i < 80; i++) { const c1 = (i + 1) * T / 80; draw(i * T / 80, c1 - 0.0015, y, C.blue); draw(c1, c1 + 0.057 / 80, y + 28, C.red); }
  if (kind === 'chunk') for (let i = 0; i < 4; i++) { const c1 = (i + 1) * T / 4; draw(i * T / 4, c1 - 0.004, y, C.blue); draw(c1, c1 + 0.057 / 4, y + 28, C.red); }
  if (kind === 'whole') { draw(0, T, y, C.blue); draw(T, T + 0.057, y + 28, C.red); }
}
function sLayer(u) {
  const k = I.layer; header(k, u);
  const a = fade(u, 0.3);
  for (let s = 0; s <= 1.8001; s += 0.3) { line(G9.x + s * G9.pps, 310, G9.x + s * G9.pps, 700, C.lg, 1.5, a); note(`${s.toFixed(1)} s`, G9.x + s * G9.pps, 300, { align: 'center', alpha: a, size: 18 }); }
  lane9(340, '逐层传', 'Mooncake、Splitwise 论文', 'layer', a, u);
  lane9(470, '按 chunk 传', 'SGLang（8K token 一个 chunk）', 'chunk', fade(u, bt(k, 1)), u);
  lane9(600, '整块传', 'vLLM 的 NIXL、Mooncake connector', 'whole', fade(u, bt(k, 1) + 0.6), u);
  rr(G9.x, 720, 16, 16, C.blue, a); note('prefill 计算', G9.x + 24, 736, { alpha: a, size: 19 }); rr(G9.x + 180, 720, 16, 16, C.red, a); note('KV 传输（57 ms）', G9.x + 204, 736, { alpha: a, size: 19 });
  // 放大末端
  const az = fade(u, bt(k, 2)), zx = 1460, zw = 364, t0 = 1.66, t1 = 1.80, zp = zw / (t1 - t0);
  if (az > 0.003) {
    rr(zx, 310, zw, 400, '#fff', az); sr(zx, 310, zw, 400, C.ink2, 2, az);
    sr(G9.x + t0 * G9.pps, 320, (t1 - t0) * G9.pps, 380, C.ink2, 2, az, [6, 5]);
    note('放大 1.66–1.80 s', zx + 12, 340, { alpha: az, size: 18 });
    const ends = [[340, 1.7 + 0.057 / 80], [470, 1.7 + 0.057 / 4], [600, 1.757]];
    ends.forEach(([y, e], i) => {
      rr(zx, y, (1.7 - t0) * zp, 22, C.blue, az);
      rr(zx + (1.7 - t0) * zp, y + 28, (e - 1.7) * zp, 22, C.red, az);
      txt(`+${((e - 1.7) * 1000).toFixed(e - 1.7 < 0.005 ? 1 : 0)} ms`, zx + (e - t0) * zp + 8, y + 46, { size: 20, weight: 700, color: C.red, alpha: az });
    });
  }
  code(96, 790, 1080, ['{g:# vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py}', 'def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):', '    """{r:NixlConnector does not do layerwise saving.}"""'], { size: 19, alpha: fade(u, bt(k, 1) + 1) });
  const an = fade(u, bt(k, 2) + 1.5);
  if (an > 0.003) { rr(1220, 790, 604, 128, PANEL, an); sr(1220, 790, 604, 128, C.lg, 2, an); txt('分析：传输只占 prefill 的约 3%，', 1244, 836, { size: 22, weight: 700, alpha: an }); txt('逐层的消息小、效率低，chunk 级重叠已足够', 1244, 874, { size: 22, weight: 700, alpha: an }); note('LMCache 的 layerwise 模式仍保留此做法（默认关闭）', 1244, 906, { alpha: an, size: 18 }); }
}
ev(I.layer, 0.6, 'whoosh'); ev(I.layer, 6, 'pop'); ev(I.layer, 7, 'click'); ev(I.layer, 11.5, 'pop');

// ================= 10 回到开场 =================
function sEnd(u) {
  const k = I.end; header(k, u);
  const kUp = p(u, bt(k, 1) + 0.3, bt(k, 1) + 1.8), freed = u > bt(k, 1);
  const order = [...Array(42).keys()].filter((c) => c !== SI);
  const cells = (i, c) => (c === SI ? (kUp >= 1 ? C.purple : freed ? EMPTY : OTHER) : order.indexOf(c) < 41 ? OTHER : EMPTY);
  at(M1.x, M1.y, M1.s, () => machine({ cells, dram: (c) => (c === DI ? (kUp > 0 ? (kUp >= 1 ? EMPTY : C.purple) : C.purple) : c < 30 ? OTHER : EMPTY), tiers: [1, 1, 0.25], flowP: kUp > 0 && kUp < 1 ? (u * 1.4) % 1 : -1 }));
  if (u > bt(k, 1)) slicesMove(M1.x, M1.y, M1.s, kUp, true);
  const [dx, dy] = dpos(M1.x, M1.y, M1.s);
  if (kUp <= 0) callout(dx + 10, dy, 1180, dy - 60, '会话的副本留在内存', { color: C.purple, alpha: fade(u, 0.8) });
  // 两条同一尺度的条：读回与重算，按真实速率走
  const ab = fade(u, bt(k, 1)), t = clamp(u - bt(k, 1) - 0.3, 0, 1.7);
  if (ab > 0.003) {
    lab('读回', 1190, 360, { color: DRAMC, alpha: ab }); rr(1190, 376, 630, 36, C.paper2, ab); rr(1190, 376, 630 * Math.min(t, 0.027) / 1.7, 36, DRAMC, ab);
    if (t >= 0.027) txt('27 ms', 1220, 404, { size: 26, weight: 700, color: DRAMC, alpha: ab });
    lab('重算', 1190, 460, { color: C.red, alpha: ab }); rr(1190, 476, 630, 36, C.paper2, ab); rr(1190, 476, 630 * t / 1.7, 36, mixc(C.red, '#ffffff', 0.25), ab);
    txt(fmtT(t), 1820, 548, { size: 26, weight: 700, color: C.red, align: 'right', alpha: ab });
  }
  const a3 = fade(u, bt(k, 2));
  if (a3 > 0.003) {
    lab('边界', 1190, 600, { alpha: a3 });
    const bs = [['显存容量足够时', 'offload 只增加开销'], ['链路带宽过低时', '重算更快：LMCache 在 32 Gbps 下 256K token 以上读回才更快'], ['GPU 越快、KV 越大', '交叉点越高']];
    bs.forEach(([a, b], i) => { const y = 642 + i * 72, aa = fade(u, bt(k, 2) + 0.5 + i * 1.2); rr(1190, y - 22, 8, 54, C.ink, aa); txt(a, 1214, y, { size: 22, weight: 700, alpha: aa }); txt(b, 1214, y + 30, { size: 19, weight: 500, color: C.ink2, alpha: aa }); });
  }
  placeholder(1190, 870, 634, ['4090 实测 Qwen3-8B、16K 会话：GPU 命中、CPU 读回、', '重算三种 TTFT 与 vLLM 日志；需要 4090 服务器'], fade(u, 12));
}
ev(I.end, 0.5, 'pop'); ev(I.end, 5.3, 'whoosh'); ev(I.end, 5.35, 'land'); ev(I.end, 7.0, 'land'); ev(I.end, 10.5, 'pop'); ev(I.end, 11.7, 'pop'); ev(I.end, 12.9, 'pop');

// ---------- 字幕 ----------
// 配音稿逐句显示在画面底部：按标点切成不超过 26 字的短句。时长：有 vo.js（配音已合成）时用每句的实测时长，否则按字数估算；不越过下一节拍。
// 接口：VO.beats[幕][句] = { text, dur, head, file, cs? }，cs（可选）为各短句在这句录音里的起点比例，缺省时按字数分配。
function cues(text, lim = 26) {
  const out = [];
  for (const p0 of (text.match(/[^，。：；？！]+[，。：；？！]?/g) || []).map((x) => x.trim()).filter(Boolean)) {
    const t = p0.replace(/[，。；]$/, '');
    if (out.length && out[out.length - 1].length + t.length + 1 <= lim && !/[？！]$/.test(out[out.length - 1])) out[out.length - 1] += (out[out.length - 1].endsWith('：') ? '' : '，') + t;
    else out.push(t);
  }
  return out;
}
const est = (t) => (t.match(/[一-鿿]/g) || []).length / 4.6 + (t.match(/[A-Za-z0-9][A-Za-z0-9.,%\-]*/g) || []).length * 0.35;
const SUBS = SCRIPT.map((sc, k) => sc.beats.flatMap(([t0, , text], j) => {
  if (!text) return [];
  const cs = cues(text), end = j + 1 < sc.beats.length ? sc.beats[j + 1][0] - 0.1 : sc.dur - 0.1;
  const v = typeof VO !== 'undefined' && VO.beats[k] && VO.beats[k].find((b) => b.text === text && b.file);
  if (v) {   // 实测：一句的总时长按各短句的起点比例（没有则按字数）分配
    const tot = cs.reduce((a, c) => a + est(c), 0); let acc = 0;
    const starts = v.cs && v.cs.length === cs.length ? v.cs : cs.map((c) => { const s = acc / tot; acc += est(c); return s; });
    return cs.map((c, i) => [t0 + starts[i] * v.dur, Math.min(end, t0 + (i + 1 < cs.length ? starts[i + 1] : 1) * v.dur + (i + 1 < cs.length ? 0 : 0.3)), c]);
  }
  const total = cs.reduce((a, c) => a + est(c), 0), scale = Math.min(1, (end - t0) / Math.max(0.1, total)); let t = t0;
  return cs.map((c) => { const d = est(c) * scale; const r = [t, t + d, c]; t += d; return r; });
}));
function subtitle(k, u) {
  const c = SUBS[k].find(([a, b]) => u >= a && u < b); if (!c) return;
  const w = tw(c[2], 34, 600);
  rr(960 - w / 2 - 20, 1012, w + 40, 52, C.paper, 0.92);
  txt(c[2], 960, 1050, { size: 34, weight: 600, align: 'center', color: C.ink });
}

const SCENES = { open: sOpen, tiers: sTiers, cost: sCost, write: sWrite, read: sRead, unit: sUnit, share: sShare, pd: sPD, layer: sLayer, end: sEnd };
function draw(t) {
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  rr(0, 0, W, H, C.paper);
  SCENES[SCRIPT[k].id](t - AT[k]);
  subtitle(k, t - AT[k]);
}
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
