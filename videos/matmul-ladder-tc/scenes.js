// 矩阵乘法阶梯 · 下集（Tensor Core）v2 的各幕画面。只用于本集（上集的 scenes.js 不再共用）。
// 贯穿全片的对象：
//   硬件塔 tower()：显存 → L2 → 共享内存 → 寄存器 → Tensor Core 五层立体板，数据沿真实路径以小方块流动；瓶颈层 / 通路标橙。
//   完整矩阵 mats()：Wᵀ 在上、A 在左、C 在右下，按真实比例标出一个 block 的块，再用放大镜放大。
//   tile 图 tileMap()：8192 阶的 C 按 128 × 256 切成 64 × 32 个 tile，一轮同时运行的 132 个 tile 用紫色标出。
//   角落小条 prog()：8192 阶 TFLOPS，线性刻度，cuBLAS BF16 781.9 为虚线。
// 配色沿用上集：A = 蓝、Wᵀ = 黄、C = 紫；片上数据 = 绿；瓶颈 = 橙；Tensor Core = 青。
const AT = []; let DUR = 0; SCRIPT.forEach((s) => { AT.push(DUR); DUR += s.dur; });
const bt = (k, j) => SCRIPT[k].beats[j][0];
const NAME = 'H100 矩阵乘法 · 下';
const EV = SCRIPT.map(() => []);
const ev = (k, u, kind) => EV[k].push([u, kind]);
let K = 0, U = 0;                                             // 当前幕与幕内时间
const T0 = (j) => bt(K, j);
const fin = (j, d = 0.6, off = 0) => po(U, T0(j) + off, T0(j) + off + d);     // 第 j 拍开始后淡入
const out = (j, d = 0.5, off = 0) => 1 - p(U, T0(j) + off, T0(j) + off + d);  // 第 j 拍开始后淡出
const L2C = '#5a6f9c';
const PV = { fp32: 44.53, wmma: 93.29, mma: 269.41, mma3: 308.77, wgmma: 484.21, ws: 479.74, l2: 740.05, cl: 757.26, ts: 777.63, cub: 781.86 };

// ---------- 页眉 ----------
function header(k, u) {
  const s = SCRIPT[k], a = po(u, 0, 0.5);
  txt(NAME, 96, 76, { size: 28, weight: 700, color: C.ink2 });
  txt(s.src, 1824, 64, { size: 21, weight: 500, color: C.gray, align: 'right', alpha: a });
  const tf = /^[\x00-\x7f]+$/.test(s.tag) ? MONO : ZH;
  rr(96, 100, tw(s.tag, 32, 700, tf) + 36, 50, C.ink, a); txt(s.tag, 114, 136, { size: 32, weight: 700, color: '#fff', alpha: a, font: tf });
  rich(s.head, 96, 226 + (1 - a) * 16, 52, 700, a);
}
const lab = (s, x, y, col = C.ink2, a = 1, align = 'left', size = 20, wt = 700, font = ZH) => txt(s, x, y, { size, weight: wt, color: col, alpha: a, align, font });

// ---------- 角落小条：8192 阶 TFLOPS ----------
// from → to 在 tg 起 1.2 s 内增长；note 为增长后浮出的倍数
function prog(from, to, tg, { alpha = 1, note = '', fp32 = false } = {}) {
  if (alpha <= 0.003) return;
  const x = 1300, y = 102, w = 524, h = 26, X = (v) => x + w * v / 800;
  const k = tg == null ? 1 : po(U, tg, tg + 1.2), v = mix(from, to, k);
  lab('8192 阶 · BF16 · TFLOPS', x, y - 10, C.gray, alpha, 'left', 18);
  rr(x, y, w, h, C.paper2, alpha);
  if (from > 0 && k < 1) rr(x, y, X(from) - x, h, C.teal, alpha * 0.3);
  if (v > 0) rr(x, y, X(v) - x, h, C.teal, alpha);
  if (fp32) { rr(x, y, X(PV.fp32) - x, h, C.blue, alpha); }
  line(X(PV.cub), y - 6, X(PV.cub), y + h + 6, C.ink, 2.5, alpha, [5, 4]);
  lab('cuBLAS 781.9', X(PV.cub), y + h + 24, C.ink, alpha, 'right', 17, 700);
  const vs = fp32 ? '44.5（FP32）' : v >= 777 ? '777.6' : v < 60 ? v.toFixed(1) : String(Math.round(v));
  if (v > 0 || fp32) {
    const vx = fp32 ? X(PV.fp32) : X(v), inside = vx - x > 120 && !fp32;
    lab(vs, inside ? vx - 10 : vx + 10, y + 21, inside ? '#fff' : C.tealD, alpha, inside ? 'right' : 'left', 21, 800, MONO);
  }
  if (note && k > 0.6) lab(note, x, y + h + 24, C.tealD, alpha * po(U, tg + 0.8, tg + 1.4), 'left', 18, 800);
}

// ---------- 完整矩阵：Wᵀ 在上、A 在左、C 在右下 ----------
// tile = {m, n, h, w}（0..1 的比例）；kp 为当前 k 块的位置（0..1），bk 为 k 块的宽度
function mats(x, y, S, { tile = null, kp = -1, bk = 64 / 8192, alpha = 1, gap = 12, labels = true, minPx = 6, size = '8192' } = {}) {
  if (alpha <= 0.003) return {};
  const wx = x + S + gap, wy = y, ax = x, ay = y + S + gap, cx = wx, cy = ay;
  rr(wx, wy, S, S, '#f3e6b8', alpha); rr(ax, ay, S, S, '#d9e2f7', alpha); rr(cx, cy, S, S, '#e7defa', alpha);
  let T = null;
  if (tile) {
    const th = Math.max(minPx, tile.h * S), tww = Math.max(minPx, tile.w * S);
    rr(ax, ay + tile.m * S, S, th, '#9db4ee', alpha); rr(wx + tile.n * S, wy, tww, S, '#ecd27c', alpha);
    if (kp >= 0) { const kw = Math.max(4, bk * S); rr(ax + kp * S, ay + tile.m * S, kw, th, C.blue, alpha); rr(wx + tile.n * S, wy + kp * S, tww, kw, C.yelD, alpha); }
    rr(cx + tile.n * S, cy + tile.m * S, tww, th, C.purple, alpha);
    T = { x: cx + tile.n * S, y: cy + tile.m * S, w: tww, h: th };
  }
  if (labels) {
    lab('Wᵀ', wx + S + 10, wy + 24, C.yelD, alpha, 'left', 22, 800);
    lab('A', ax - 10, ay + 24, C.blue, alpha, 'right', 22, 800);
    lab('C', cx + S + 10, cy + 24, C.purple, alpha, 'left', 22, 800);
    lab(`${size} × ${size}`, cx + S / 2, cy + S + 26, C.gray, alpha, 'center', 17, 600);
  }
  return { wx, wy, ax, ay, cx, cy, S, T };
}
// 放大镜：从小矩形 r 引两条线到大框 R
function lens(r, R, a, col = C.purple) {
  if (a <= 0.003 || !r) return;
  line(r.x + r.w, r.y, R.x, R.y, col, 1.5, a * 0.7, [5, 4]); line(r.x + r.w, r.y + r.h, R.x, R.y + R.h, col, 1.5, a * 0.7, [5, 4]);
  sr(r.x - 2, r.y - 2, r.w + 4, r.h + 4, col, 2, a);
}

// ---------- 硬件塔 ----------
const LV = [['Tensor Core', C.teal, '每 SM 4 个'], ['寄存器', C.thr, '每 SM 256 KB'], ['共享内存', C.use, '每 SM 228 KB'], ['L2', L2C, '50 MB · 全卡共享'], ['显存 HBM', '#9a6a3a', '80 GB · 3.35 TB/s']];
// st：path 为经过的层（如 [4,3,1,2,1,0]）；hop 为各跳的标注；hot = {lv:[层], hop:[跳]}；xpath 为划掉的旧路径；flow 为流动速度系数；
//     dataCol(i) 为第 i 个小方块的颜色；busy 为亮绿的层；note(lv) 为层右侧的说明
function towerGeo(x, y, s, gk = 1) { const w = 400 * s, h = 42 * s, gap = 100 * s * gk; return { w, h, gap, Y: (i) => y + i * gap, dx: 22 * s, dy: -14 * s }; }
// st.lanes / st.xl：主路径与划掉的旧路径占用的横向范围（占板宽的比例）；st.tags = {层: 文字}，在该层下方贴“瓶颈”标签（透明度 st.tagA）
function tower(x, y, s, st, a) {
  if (a <= 0.003) return null;
  const G = towerGeo(x, y, s, st.gap || 1), { w, h, Y } = G, tags = st.tags || {};
  const smy = Y(0) - 34 * s, smh = Y(2) + h + (tags[2] ? 68 : 32) * s - smy;
  g.globalAlpha = a; g.strokeStyle = C.gray; g.lineWidth = 1.5; g.setLineDash([7, 5]); g.strokeRect(x - 14 * s, smy, w + 50 * s, smh); g.setLineDash([]); g.globalAlpha = 1;
  lab(st.smLabel || '一个 SM（共 132 个）', x - 8 * s, smy - 8 * s, C.gray, a, 'left', Math.max(16, 17 * s), 700);
  for (let i = 0; i < 5; i++) {
    const hot = st.hot?.lv?.includes(i), busy = st.busy?.includes(i), dim = st.dim?.includes(i);
    const yy = Y(i), fa = a * (dim ? 0.35 : 1);
    g.globalAlpha = fa; poly([[x, yy], [x + G.dx, yy + G.dy], [x + w + G.dx, yy + G.dy], [x + w, yy]], hot ? '#f3cdb3' : busy ? '#cfe6d7' : '#e4ddcd');
    g.globalAlpha = fa; poly([[x + w, yy], [x + w + G.dx, yy + G.dy], [x + w + G.dx, yy + G.dy + h], [x + w, yy + h]], hot ? '#e7b593' : '#d6cfbf'); g.globalAlpha = 1;
    rr(x, yy, w, h, hot ? '#fbe6d8' : busy ? '#e3f1e7' : '#f8f5ee', fa);
    sr(x, yy, w, h, hot ? C.hot : busy ? C.use : C.lg, hot || busy ? 2.5 : 1.5, fa);
    rr(x + 10 * s, yy + 11 * s, 8 * s, h - 22 * s, LV[i][1], fa);
    lab(LV[i][0], x + 26 * s, yy + h / 2 + 7 * Math.max(0.8, s), hot ? C.hot : C.ink, fa, 'left', Math.max(17, 21 * s), 800);
    const ta = tags[i] ? (st.tagA ?? 1) : 0;
    if (!st.noSub) lab(LV[i][2], x + 26 * s, yy + h + 20 * s, C.gray, fa * 0.9, 'left', Math.max(14, 14.5 * s), 600);
    if (ta > 0) bnTag(x + 8 * s, yy + h + (st.noSub ? 32 : 52) * s, tags[i], a * ta, 17);
    const nt = st.note && st.note(i); if (nt) { const [s2, col = C.ink2, na = 1] = Array.isArray(nt) ? nt : [nt]; lab(s2, x + w + 40 * s, yy + h / 2 + 8 * s, col, a * na, 'left', Math.max(16, 19 * s), 700); }
  }
  const laneX = (i, n, lr) => { const x0 = x + w * lr[0], x1 = x + w * lr[1]; return n > 1 ? mix(x0, x1, i / (n - 1)) : (x0 + x1) / 2; };
  const P = (path, lr) => {                                // 路径的折线：每一跳一条竖线，相邻两跳在层内横向相接
    const n = path.length - 1, pts = [];
    for (let i = 0; i < n; i++) {
      const lx = laneX(i, n, lr), a0 = path[i], b0 = path[i + 1], yb = a0 > b0 ? Y(b0) + h - 4 * s : Y(b0) + 4 * s;
      pts.push([lx, Y(a0) + h / 2], [lx, yb]);
      if (i < n - 1) pts.push([lx, Y(b0) + h / 2]);
    }
    return pts;
  };
  const draw = (path, mode, hops, hotHops, lr) => {
    const n = path.length - 1; let prev = null;
    for (let i = 0; i < n; i++) {
      const lx = laneX(i, n, lr), a0 = path[i], b0 = path[i + 1], up = a0 > b0;
      const ya = Y(a0) + h / 2, yb = up ? Y(b0) + h - 2 * s : Y(b0) + 2 * s;
      const hot = hotHops.includes(i), col = mode === 'x' ? C.gray : hot ? C.hot : st.pathCol || C.use;
      if (prev !== null) line(prev, Y(a0) + h / 2, lx, Y(a0) + h / 2, col, 5 * s, a, mode === 'x' ? [8, 6] : null);
      for (let L = Math.min(a0, b0) + 1; L < Math.max(a0, b0); L++) rr(lx - 7 * s, Y(L) - 2, 14 * s, h + 4, '#fbf9f4', a);   // 越过的层：从它前面经过
      if (mode === 'x') line(lx, ya, lx, yb, col, 4 * s, a, [8, 6]); else arrow(lx, ya, lx, yb, 1, col, a, 5 * s);
      if (mode === 'x') { const my = (ya + yb) / 2; line(lx - 11, my - 11, lx + 11, my + 11, C.red, 4, a); line(lx - 11, my + 11, lx + 11, my - 11, C.red, 4, a); }
      if (hops[i]) {
        const fs = Math.max(15, 17 * s), my = up ? Y(a0) + G.dy - 8 : Y(b0) + G.dy - 8, tx = lx - 9 * s, wv = tw(hops[i], fs, 700, MONO);
        rr(tx - wv - 4, my - fs, wv + 8, fs + 8, 'rgba(241,237,228,0.9)', a);
        lab(hops[i], tx, my, mode === 'x' ? C.gray : hot ? C.hot : C.ink2, a, 'right', fs, 700, MONO);
      }
      prev = lx;
    }
  };
  if (st.xpath) draw(st.xpath, 'x', st.xhop || [], [], st.xl || [0.3, 0.4]);
  const lr = st.lanes || [0.48, 0.94];
  if (st.path) draw(st.path, 'ok', st.hop || [], st.hot?.hop || [], lr);
  if (st.path && st.flow !== 0) {                          // 流动的小方块
    const pts = P(st.path, lr); let L = 0; const seg = [];
    for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(d); L += d; }
    const N = st.nDots || 7, sp = 150 * s * (st.flow || 1);
    for (let j = 0; j < N; j++) {
      let d = ((U * sp + j * L / N) % L + L) % L, i = 0;
      while (i < seg.length - 1 && d > seg[i]) { d -= seg[i]; i++; }
      const k = seg[i] ? d / seg[i] : 0, px = mix(pts[i][0], pts[i + 1][0], k), py = mix(pts[i][1], pts[i + 1][1], k);
      const hop = Math.floor(i / 3), big = st.blockHop === hop, qw = (big ? 40 : 13) * s, qh = (big ? 24 : 13) * s;
      let pa = a; for (let L = 0; L < 5; L++) if (!st.path.includes(L) && py > Y(L) - 2 && py < Y(L) + h + 2) pa = a * 0.15;
      const col = st.dataCol ? st.dataCol(j) : j % 2 ? C.yel : C.blue;
      rr(px - qw / 2, py - qh / 2, qw, qh, col, pa); if (big) for (let r = 0; r < 3; r++) rr(px - qw / 2 + ((r * 3) % 5) * qw / 5, py - qh / 2 + r * qh / 3, qw / 5, qh / 3, '#ffffff88', pa); sr(px - qw / 2, py - qh / 2, qw, qh, '#fff', 1.5, pa * 0.8);
    }
  }
  return G;
}
// 塔上某层右侧的瓶颈标签
function bnTag(x, y, text, a, z = 20) {
  if (a <= 0.003) return;
  const w0 = tw('瓶颈', z, 800) + 20, w1 = tw(text, z, 800) + 20, hh = z + 16;
  rr(x, y - z - 6, w0, hh, C.hot, a); txt('瓶颈', x + 10, y - 1, { size: z, weight: 800, color: '#fff', alpha: a });
  rr(x + w0, y - z - 6, w1, hh, '#fbf3ec', a); sr(x + w0, y - z - 6, w1, hh, C.hot, 2, a); txt(text, x + w0 + 10, y - 1, { size: z, weight: 800, color: C.hot, alpha: a });
  return w0 + w1;
}

// ---------- 代码框：只放改动的几行 ----------
// lines = [[文字, '+' 或 ''], …]；hl 为高亮行号（可为数组）
function codeCard(x, y, w, title, lines, { hl = -1, alpha = 1, size = 22, lh = 40 } = {}) {
  if (alpha <= 0.003) return y;
  const hh = 54 + lines.length * lh + 12, hls = Array.isArray(hl) ? hl : [hl];
  rr(x, y, w, hh, '#fbf9f4', alpha); sr(x, y, w, hh, C.lg, 1.5, alpha);
  lab(title, x + 18, y + 34, C.gray, alpha, 'left', 18, 700, MONO);
  let fs = size; for (const [s] of lines) while (tw(s, fs, 600, MONO) > w - 70 && fs > 14) fs -= 0.5;
  lines.forEach(([s, m], i) => {
    const yy = y + 50 + i * lh;
    if (hls.includes(i)) { rr(x + 6, yy, w - 12, lh - 4, '#dbe5f6', alpha); rr(x + 6, yy, 6, lh - 4, C.blue, alpha); }
    else if (m === '+') rr(x + 6, yy, w - 12, lh - 4, '#e3efe6', alpha);
    if (m === '+') lab('+', x + 22, yy + lh * 0.68, C.use, alpha, 'left', fs, 800, MONO);
    lab(s, x + 46, yy + lh * 0.68, C.ink, alpha, 'left', fs, 600, MONO);
  });
  return y + hh;
}
// 一行说明卡（问题卡）：橙框
function probCard(x, y, w, ver, rows, a) {
  if (a <= 0.003) return y;
  const hh = 58 + rows.length * 36;
  rr(x, y, w, hh, '#fbf3ec', a); sr(x, y, w, hh, C.hot, 2.5, a);
  rr(x, y, tw('瓶颈', 20, 800) + 24, 36, C.hot, a); lab('瓶颈', x + 12, y + 26, '#fff', a, 'left', 20, 800);
  lab(`${ver} 的问题`, x + tw('瓶颈', 20, 800) + 36, y + 26, C.hot, a, 'left', 20, 800, /^[\x00-\x7f]+$/.test(ver) ? MONO : ZH);
  rows.forEach(([s, col = C.ink], i) => lab(s, x + 18, y + 72 + i * 36, col, a, 'left', 21, 700));
  return y + hh;
}

// ---------- 时间线 ----------
// rows = [[名称, segs, 颜色]]；segs = [[起, 止, 颜色, 文字]]；T 为总时长；reveal 0..1
function tline(x, y, w, rows, T, reveal, a, { lh = 40, gap = 12, title = '', ns = 19 } = {}) {
  if (a <= 0.003) return y;
  if (title) lab(title, x - 150, y - 12, C.gray, a, 'left', 18, 700);
  rows.forEach(([name, segs, col], i) => lane(x, y + i * (lh + gap), w, lh, T, segs, reveal, a, name, col || C.ink, ns));
  return y + rows.length * (lh + gap);
}
const IDLE = '#bdb6a6';

// ---------- 小图元 ----------
function dots(x, y, cols, rows, cs, col, a, lit = 1e9, r = 0.36) {
  if (a <= 0.003) return;
  g.globalAlpha = a;
  for (let i = 0; i < cols * rows; i++) { g.fillStyle = i < lit ? col : '#d9d3c5'; g.beginPath(); g.arc(x + (i % cols) * cs + cs / 2, y + Math.floor(i / cols) * cs + cs / 2, cs * r, 0, 7); g.fill(); }
  g.globalAlpha = 1;
}
// 等轴测长方体（m 行 × n 列 × k 深），每单位 u 像素
function ibox(x, y, m, n, k, u, cols, a, { grid = 0 } = {}) {
  if (a <= 0.003) return;
  const ax = 0.866 * u, ay = 0.5 * u;
  const P = (i, j, z) => [x + (j - z) * ax, y + (j + z) * ay - i * u];   // j 沿 n，z 沿 k，i 向上为 m
  const [top, left, right] = cols;
  g.globalAlpha = a;
  poly([P(m, 0, 0), P(m, n, 0), P(m, n, k), P(m, 0, k)], top);
  poly([P(0, 0, k), P(m, 0, k), P(m, n, k), P(0, n, k)], left);
  poly([P(0, n, 0), P(m, n, 0), P(m, n, k), P(0, n, k)], right);
  g.globalAlpha = 1;
  if (grid) {
    for (let j = 1; j < n; j += grid) { const [x0, y0] = P(0, j, k), [x1, y1] = P(m, j, k); line(x0, y0, x1, y1, 'rgba(255,255,255,0.6)', 1, a); }
    for (let i = 1; i < m; i += grid) { const [x0, y0] = P(i, 0, k), [x1, y1] = P(i, n, k); line(x0, y0, x1, y1, 'rgba(255,255,255,0.6)', 1, a); }
  }
}
const TEALS = ['#5fd0c0', '#0fa391', '#0b7d70'], GRAYS = ['#e6e0d3', '#cfc8b8', '#bdb6a6'];

// ---------- tile 图：8192 阶的 C 切成 64 × 32 个 128 × 256 的 tile ----------
// order：'row' 按行发放；'grp' 16 行一组、组内按列；wave 为第几轮（可为小数，表示正在推进）；hiW / hiA 为是否点亮 Wᵀ、A 被用到的部分
const TM = 64, TN = 32, NSM = 132;
const tileOf = (id, order) => order === 'row' ? [Math.floor(id / TN), id % TN] : (() => { const per = 16 * TN, first = Math.floor(id / per) * 16, r = id % per; return [first + r % 16, Math.floor(r / 16)]; })();
function tileMap(x, y, cw, ch, order, wave, a, { strip = 64, hiW = 1, hiA = 1, pair = null, waveCol = C.purple, label = true } = {}) {
  if (a <= 0.003) return;
  const W0 = TN * cw, H0 = TM * ch, ax = x - strip - 10, wy = y - strip - 10;
  rr(ax, y, strip, H0, '#d9e2f7', a); rr(x, wy, W0, strip, '#f3e6b8', a); rr(x, y, W0, H0, '#e7defa', a);
  const w0 = Math.floor(wave), f = wave - w0, usedR = new Set(), usedC = new Set();
  for (let id = 0; id < TM * TN; id++) {
    const [m, n] = tileOf(id, order), wv = Math.floor(id / NSM);
    let col = null;
    if (wv < w0) col = '#cbb8f2';
    else if (wv === w0) { col = waveCol; usedR.add(m); usedC.add(n); }
    else if (wv === w0 + 1 && f > 0 && (id - (w0 + 1) * NSM) < f * NSM) col = waveCol;
    if (col) rr(x + n * cw + 0.5, y + m * ch + 0.5, cw - 1, ch - 1, col, a);
  }
  for (const m of usedR) rr(ax, y + m * ch, strip, ch, C.blue, a * hiA);
  for (const n of usedC) rr(x + n * cw, wy, cw, strip, C.yelD, a * hiW * 0.9);
  if (pair) for (const [m, n] of pair) sr(x + n * cw - 1, y + m * ch - 1, cw + 2, ch + 2, C.ink, 3, a);
  if (label) {
    lab('A', ax + strip / 2, y + H0 + 26, C.blue, a, 'center', 20, 800);
    lab('Wᵀ', x + W0 + 12, wy + strip / 2 + 8, C.yelD, a, 'left', 20, 800);
    lab('C：64 × 32 个 tile（每个 128 × 256）', x, y + H0 + 26, C.purple, a, 'left', 19, 700);
    lab('A、Wᵀ 沿 K 方向压缩显示', x, y + H0 + 52, C.gray, a, 'left', 16, 600);
  }
  return { rows: usedR.size, cols: usedC.size };
}
// 工作集条：A 与 W 的 MB 数，L2 50 MB 刻线
function wsetBar(x, y, w, A, Wm, a, { max = 140, title = '同时运行的 132 个 tile 需要的数据' } = {}) {
  if (a <= 0.003) return;
  const X = (v) => x + w * v / max;
  lab(title, x, y - 14, C.ink2, a, 'left', 20, 700);
  rr(x, y, w, 40, C.paper2, a);
  rr(x, y, X(A) - x, 40, C.blue, a); rr(X(A), y, X(A + Wm) - X(A), 40, C.yel, a);
  if (X(A) - x > 70) lab(`A ${Math.round(A)}`, x + 8, y + 28, '#fff', a, 'left', 18, 800);
  if (X(A + Wm) - X(A) > 70) lab(`W ${Math.round(Wm)}`, X(A) + 8, y + 28, C.ink, a, 'left', 18, 800);
  lab(`${Math.round(A + Wm)} MB`, X(A + Wm) + 10, y + 28, A + Wm > 50 ? C.hot : C.use, a, 'left', 22, 800, MONO);
  line(X(50), y - 10, X(50), y - 2, C.ink, 3, a); line(X(50), y + 42, X(50), y + 54, C.ink, 3, a); lab('L2 50 MB', X(50), y + 74, C.ink, a, 'center', 18, 800);
}

// =====================================================================
// 第 1 幕：换用 Tensor Core
// =====================================================================
// 一个完整 SM 的缩略：4 个处理块，部件画成色块（按 H100 白皮书的 SM 图）；hi 为高亮的处理块，fpHi 为 FP32 色条的亮度
const PB = { sch: '#cfc8b8', reg: '#cdbff0', int: '#a9a497', fp32: C.blue, fp64: '#a9bfee', tc: C.teal, ldst: '#e9c46a', sfu: '#c4b1ef' };
function smThumb(x, y, a, { hi = -1, fpHi = 0 } = {}) {
  if (a <= 0.003) return null;
  const bw = 124, bh = 158, gp = 10, W0 = 2 * bw + gp + 24, H0 = 26 + 2 * bh + gp + 56;
  rr(x, y, W0, H0, '#e3f1ee', a); sr(x, y, W0, H0, C.teal, 2, a);
  rr(x + 12, y + 8, W0 - 24, 10, PB.sch, a * 0.7);
  const boxes = [];
  for (let k = 0; k < 4; k++) {
    const bx = x + 12 + (k % 2) * (bw + gp), by = y + 26 + Math.floor(k / 2) * (bh + gp), iw = bw - 14;
    rr(bx, by, bw, bh, '#fbf9f4', a); sr(bx, by, bw, bh, k === hi ? C.teal : C.lg, k === hi ? 3 : 1.5, a);
    const ch = (yy, hh, col, al = 1, xx = 0, ww = iw) => rr(bx + 7 + xx, by + yy, ww, hh, col, a * al);
    ch(7, 8, PB.sch, 0.6); ch(18, 11, PB.sch); ch(32, 12, PB.reg);
    ch(48, 9, PB.int); ch(60, 20, PB.fp32, 0.35 + 0.65 * fpHi); ch(83, 9, PB.fp64);
    ch(96, 32, PB.tc); ch(132, 10, PB.ldst, 1, 0, iw * 0.55); ch(132, 10, PB.sfu, 1, iw * 0.55 + 4, iw * 0.45 - 4);
    boxes.push({ x: bx, y: by, w: bw, h: bh });
  }
  const by2 = y + 26 + 2 * bh + gp + 8;
  rr(x + 12, by2, W0 - 24, 12, PB.sch, a); rr(x + 12, by2 + 16, W0 - 24, 22, '#bfe0cc', a);
  lab('L1 / 共享内存 256 KB', x + W0 / 2, by2 + 33, '#2f6b4a', a, 'center', 15, 800);
  lab('一个 SM（共 132 个）', x, y - 12, C.ink2, a, 'left', 20, 800);
  return { boxes, W0, H0 };
}
function sIntro() {
  const a0 = fin(0), a1 = fin(1), a2 = fin(2), a3 = fin(3), a4 = fin(4);
  // 左：完整 SM 的缩略
  const T = smThumb(96, 320, a0, { hi: a1 > 0.01 ? 0 : -1, fpHi: 1 - a1 * 0.7 });
  if (a0 > 0.003 && a1 < 0.5) { lab('FP32 单元：已用满', 96, 820, C.blue, a0 * (1 - a1), 'left', 20, 800); lab('上集最好 44.5 TFLOPS，为 cuBLAS FP32 的 86%', 96, 848, C.gray, a0 * (1 - a1), 'left', 16, 600); }
  // 中：放大处理块 0
  const Z = { x: 452, y: 300, w: 712, h: 560 };
  if (T && a1 > 0.003) {
    const b = T.boxes[0];
    line(b.x + b.w, b.y, Z.x, Z.y, C.teal, 1.5, a1 * 0.7, [5, 4]); line(b.x + b.w, b.y + b.h, Z.x, Z.y + Z.h, C.teal, 1.5, a1 * 0.7, [5, 4]);
    smThumb(96, 320, a0, { hi: 0, fpHi: 1 - a1 * 0.7 });          // 重画一次，盖住引线
    rr(Z.x, Z.y, Z.w, Z.h, '#fbf9f4', a1); sr(Z.x, Z.y, Z.w, Z.h, C.teal, 3, a1);
    lab('处理块 0（共 4 个）', Z.x + 20, Z.y + 38, C.ink, a1, 'left', 22, 800);
    [['L0 指令缓存', 0.6], ['warp 调度器', 1], ['分发单元', 0.6]].forEach(([t, o], i) => { rr(Z.x + 20 + i * 158, Z.y + 56, 150, 30, PB.sch, a1 * o); lab(t, Z.x + 95 + i * 158, Z.y + 77, C.ink2, a1, 'center', 16, 700); });
    rr(Z.x + 494, Z.y + 56, 176, 30, PB.reg, a1); lab('寄存器 64 KB', Z.x + 582, Z.y + 77, '#5b3fb0', a1, 'center', 16, 800);
    // FP32：32 个单元，每格一个点
    const cx0 = Z.x + 220, cs = 12;
    lab('FP32 单元 × 32', Z.x + 20, Z.y + 132, C.blue, a1, 'left', 20, 800);
    lab('每个每周期 1 次', Z.x + 20, Z.y + 158, C.gray, a1, 'left', 16, 600);
    const n32 = Math.round(32 * lin(U, T0(1) + 0.6, T0(1) + 2.0));
    for (let i = 0; i < 32; i++) { rr(cx0 + i * cs, Z.y + 124, cs - 2, cs - 2, '#d9e2f7', a1); if (i < n32) { g.globalAlpha = a1; g.fillStyle = C.blue; g.beginPath(); g.arc(cx0 + i * cs + cs / 2 - 1, Z.y + 124 + cs / 2 - 1, 3.6, 0, 7); g.fill(); g.globalAlpha = 1; } }
    lab('32 次', cx0 + 32 * cs + 14, Z.y + 136, C.blue, a1, 'left', 22, 800, MONO);
    // Tensor Core：1 个单元，框内 16 × 32 个点，点距与 FP32 相同
    if (a2 > 0.003) {
      lab('Tensor Core × 1', Z.x + 20, Z.y + 232, C.tealD, a2, 'left', 20, 800);
      lab('每周期 512 次', Z.x + 20, Z.y + 258, C.gray, a2, 'left', 16, 600);
      rr(cx0 - 6, Z.y + 190, 32 * cs + 10, 16 * cs + 10, '#d3ede8', a2); sr(cx0 - 6, Z.y + 190, 32 * cs + 10, 16 * cs + 10, C.teal, 2.5, a2);
      dots(cx0 - 1, Z.y + 195, 32, 16, cs, C.teal, a2, 512 * lin(U, T0(2) + 0.3, T0(2) + 2.4), 0.3);
      lab('512 次', cx0 + 32 * cs + 14, Z.y + 284, C.tealD, a2, 'left', 22, 800, MONO);
      lab('16 倍', cx0 + 32 * cs + 14, Z.y + 312, C.tealD, a2, 'left', 18, 800);
      lab('另有 INT32 × 16、FP64 × 16、LD/ST × 8、SFU × 4', Z.x + 20, Z.y + 440, C.gray, a2, 'left', 16, 600);
      lab('整个 SM 每周期：FP32 128 次，Tensor Core 2048 次', Z.x + 20, Z.y + 500, C.ink, a2, 'left', 22, 800);
      lab('输入 BF16（bfloat16，16 位浮点），累加 FP32', Z.x + 20, Z.y + 532, C.gray, a2, 'left', 16, 600);
    }
  }
  // 右：一条 wmma 指令（H100：16 × 16 × 16），高亮 C 的一个元素的计算路径
  const x1 = 1188;
  if (a3 > 0.003) {
    lab('一条指令：wmma 16 × 16 × 16', x1, 312, C.ink2, a3, 'left', 22, 800);
    rr(x1, 330, 636, 330, '#fbf9f4', a3); sr(x1, 330, 636, 330, C.lg, 1.5, a3);
    const FS = 96, uu = FS / 16, fy = 380, hk = lin(U, T0(3) + 0.8, T0(3) + 1.6);
    const frag = (fx, base, strong, mode) => {
      rr(fx, fy, FS, FS, base, a3);
      if (mode === 'row') rr(fx, fy, FS, uu, strong, a3 * hk);
      if (mode === 'col') rr(fx + 15 * uu, fy, uu, FS, strong, a3 * hk);
      if (mode === 'cell') { rr(fx + 15 * uu, fy, uu, uu, strong, a3 * hk); sr(fx + 15 * uu - 4, fy - 4, uu + 8, uu + 8, strong, 2.5, a3 * hk); }
    };
    frag(x1 + 24, '#d9e2f7', C.blue, 'row'); lab('×', x1 + 136, fy + 58, C.ink, a3, 'center', 26, 800);
    frag(x1 + 152, '#f3e6b8', C.yelD, 'col'); lab('+', x1 + 264, fy + 58, C.ink, a3, 'center', 26, 800);
    frag(x1 + 280, '#e7defa', C.purple, 'cell'); lab('=', x1 + 394, fy + 58, C.ink, a3, 'center', 26, 800);
    lab('A 16×16', x1 + 72, fy + FS + 28, C.blue, a3, 'center', 16, 800); lab('B 16×16', x1 + 200, fy + FS + 28, C.yelD, a3, 'center', 16, 800); lab('C 16×16', x1 + 328, fy + FS + 28, C.purple, a3, 'center', 16, 800);
    const u = 4.6, ox = x1 + 500, oy = fy + 128;
    ibox(ox, oy, 16, 16, 16, u, ['#bfe6df', '#9fd6cc', '#7fc4b8'], a3);
    const ax = 0.866 * u, ay = 0.5 * u;
    ibox(ox + 15 * ax, oy + 15 * ay - 15 * u, 1, 1, 16, u, ['#b8a6ef', '#9a82e6', '#7d63d6'], a3 * hk);
    lab('C 的每个元素沿 K 累加 16 次', x1 + 24, 600, C.tealD, a3, 'left', 21, 800);
    lab('256 个元素 × 16 = 4096 次乘加；一个 Tensor Core 约 8 个周期', x1 + 24, 632, C.gray, a3, 'left', 17, 700);
  }
  // 右下：完整矩阵与目标
  if (a4 > 0.003) {
    mats(1250, 720, 80, { alpha: a4, labels: false });
    lab('Wᵀ', 1250 + 2 * 80 + 12 + 10, 744, C.yelD, a4, 'left', 18, 800); lab('A', 1240, 836, C.blue, a4, 'right', 18, 800); lab('C', 1250 + 2 * 80 + 12 + 10, 836, C.purple, a4, 'left', 18, 800);
    lab('C = A · Wᵀ（Linear 层）', 1470, 760, C.ink, a4, 'left', 21, 800);
    lab('A：activation，M × K', 1470, 794, C.blue, a4, 'left', 17, 700);
    lab('W：权重，N × K，按行存放', 1470, 822, C.yelD, a4, 'left', 17, 700);
    lab('A、W 都沿 K 连续', 1470, 850, C.gray, a4, 'left', 17, 600);
  }
  prog(0, 0, null, { alpha: a4 });
}

// =====================================================================
// 第 2 幕：wmma 的分工（block → warp → 片段；块的大小；128 × 32 的 K 块）
// =====================================================================
function bank64(x, y, a, rounds) {
  // 64 字节一行的共享内存：每 128 字节的一排 bank 放两行；8 行的第 1 块（16 字节）分别落在 bank 4–7 与 20–23
  if (a <= 0.003) return;
  const cw = 15, rh = 28;
  lab('As 每行 32 个 BF16 = 64 字节，一排 32 个 bank 放两行', x - 40, y - 16, C.ink2, a, 'left', 18, 800);
  for (let b = 0; b < 32; b += 4) lab(String(b), x + b * cw + 1, y + 14, C.gray, a, 'left', 15, 700, MONO);
  for (let L = 0; L < 4; L++) {
    const yy = y + 24 + L * (rh + 6);
    lab(`行 ${2 * L} | ${2 * L + 1}`, x - 10, yy + 20, C.gray, a, 'right', 15, 600);
    for (let b = 0; b < 32; b++) {
      const hit = (b >= 4 && b < 8) || (b >= 20 && b < 24), cur = hit && Math.floor(rounds) > L;
      rr(x + b * cw, yy, cw - 2, rh, hit ? (cur ? C.hot : '#f0c6a8') : b < 16 ? '#d9e2f7' : '#e4ebfa', a);
    }
  }
  const yb = y + 24 + 4 * (rh + 6) + 6;
  for (const b0 of [4, 20]) { line(x + b0 * cw, yb, x + (b0 + 4) * cw - 2, yb, C.hot, 3, a); lab('4 个请求', x + (b0 + 2) * cw, yb + 22, C.hot, a, 'center', 16, 800); }
  if (rounds > 0) lab(rounds >= 4 ? '读取 8 行的同一块：4 轮（4 路冲突）' : `读取 8 行的同一块：第 ${Math.max(1, Math.ceil(rounds))} 轮`, x - 40, yb + 58, C.hot, a, 'left', 20, 800);
}
// 循环阶梯的颜色：每一层循环与它在画面上管辖的区域同色
const LC = { grid: '#7d5fd6', warp: '#3C3489', k0: '#d9822b', kk: '#2e9b6a', ij: '#0b7d70' };
function loopTag(s, x, y, col, a, align = 'left') {
  if (a <= 0.003) return;
  const w = tw(s, 16, 800, MONO) + 16, x0 = align === 'right' ? x - w : x;
  rr(x0, y - 20, w, 28, col, a); lab(s, x0 + 8, y, '#fff', a, 'left', 16, 800, MONO);
}
// 代码阶梯：由内向外点亮，lit 为已点亮的层数（从最内层数起），cur 为当前层
const LORD = ['grid', 'warp', 'k0', 'kk', 'ij'];
function codeLadder(x, y, w, lit, cur, a) {
  if (a <= 0.003) return;
  const rows = [
    ['grid', 0, 'kernel<<<(64, 64), 256>>>', '4096 个 block'],
    ['warp', 0, 'warp = tid / 32 → (wRow, wCol)', '8 个 warp'],
    ['k0', 0, 'for (k0 = 0; k0 < K; k0 += 32)', '256 次'],
    ['k0', 1, 'As, Ws ← A, W 的 32 列', ''],
    ['kk', 1, 'for (kk = 0; kk < 32; kk += 16)', '2 次'],
    ['ij', 2, 'for i<4, j<2: mma_sync(c[i][j], a[i], b[j])', '8 条'],
  ];
  const lh = 46, hh = 52 + rows.length * lh + 10;
  rr(x, y, w, hh, '#fbf9f4', a); sr(x, y, w, hh, C.lg, 1.5, a);
  lab('matmul_tc.cu · mm_wmma（精简）', x + 18, y + 34, C.gray, a, 'left', 17, 700, MONO);
  rows.forEach(([k, ind, code, cnt], i) => {
    const yy = y + 50 + i * lh, lv = LORD.indexOf(k), on = lv >= 5 - lit, now = k === cur, col = LC[k];
    if (now) rr(x + 6, yy, w - 12, lh - 6, '#efe9dd', a);
    rr(x + 12, yy + 8, 8, lh - 22, col, a * (on ? 1 : 0.25));
    let fs = 17; while (tw(code, fs, 600, MONO) > w - 150 - ind * 22 && fs > 13) fs -= 0.5;
    lab(code, x + 34 + ind * 22, yy + 27, on ? C.ink : C.gray, a * (on ? 1 : 0.55), 'left', fs, 600, MONO);
    if (cnt && on) lab(cnt, x + w - 16, yy + 27, col, a, 'right', 16, 800);
  });
}
// ---------- 迭代空间长方体：M × N × K，每格一次乘加；正面 = C（M × N），侧面 = A（M × K），顶面 = Wᵀ（K × N）----------
// 等轴测：i 向上（M），j 向右下（N），z 向左下（K，z = k 一面朝向观众左前方）。su 为 M、N 方向每单位像素，sk 为 K 方向每单位像素。
const MMC = { c: '#b9a8ef', a: '#9db8ef', w: '#ecd68e', cD: '#7d5fd6', aD: '#2f5fd0', wD: '#b8860b' };
function mmGeo(m, n, k, su, sk) {
  const P = (i, j, z) => [(j * su - z * sk) * 0.866, (j * su + z * sk) * 0.5 - i * su];
  const pts = [[0, 0, 0], [m, 0, 0], [0, n, 0], [m, n, 0], [0, 0, k], [m, 0, k], [0, n, k], [m, n, k]].map(([i, j, z]) => P(i, j, z));
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return { P, x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}
// cx, cy：长方体外接框的中心；g = { m, n, k } 为面上格线的间距（单位）；返回把 (i, j, z) 映射到屏幕的函数
function mmBox(cx, cy, m, n, k, su, sk, a, { grid = null, alpha = 1, edge = C.ink2 } = {}) {
  const G = mmGeo(m, n, k, su, sk), ox = cx - (G.x0 + G.x1) / 2, oy = cy - (G.y0 + G.y1) / 2;
  const P = (i, j, z) => { const q = G.P(i, j, z); return [ox + q[0], oy + q[1]]; };
  if (a <= 0.003) return P;
  const face = (pts, col) => { g.globalAlpha = a * alpha; poly(pts, col); g.globalAlpha = 1; };
  face([P(m, 0, 0), P(m, n, 0), P(m, n, k), P(m, 0, k)], MMC.w);       // 顶面：Wᵀ
  face([P(0, 0, k), P(m, 0, k), P(m, n, k), P(0, n, k)], MMC.c);       // 正面：C
  face([P(0, n, 0), P(m, n, 0), P(m, n, k), P(0, n, k)], MMC.a);       // 侧面：A
  if (grid) {
    const L = (p, q) => line(p[0], p[1], q[0], q[1], 'rgba(255,255,255,0.85)', 1.2, a);
    for (let j = grid.n; j < n - 1e-6; j += grid.n) { L(P(0, j, k), P(m, j, k)); L(P(m, j, 0), P(m, j, k)); }
    for (let i = grid.m; i < m - 1e-6; i += grid.m) { L(P(i, 0, k), P(i, n, k)); L(P(i, n, 0), P(i, n, k)); }
    if (grid.k) for (let z = grid.k; z < k - 1e-6; z += grid.k) { L(P(0, n, z), P(m, n, z)); L(P(m, 0, z), P(m, n, z)); }
  }
  const E = (p, q) => line(p[0], p[1], q[0], q[1], edge, 1.6, a);
  E(P(m, 0, 0), P(m, n, 0)); E(P(m, n, 0), P(m, n, k)); E(P(m, n, k), P(m, 0, k)); E(P(m, 0, k), P(m, 0, 0));
  E(P(0, 0, k), P(m, 0, k)); E(P(0, n, k), P(m, n, k)); E(P(0, 0, k), P(0, n, k)); E(P(0, n, 0), P(0, n, k)); E(P(0, n, 0), P(m, n, 0));
  return P;
}
// 在长方体的某个面上，涂一块子区域（i0..i1, j0..j1, z0..z1 中只用该面的两个维度）
function mmPatch(P, face, r, col, a) {
  if (a <= 0.003) return;
  const [i0, i1, j0, j1, z0, z1] = r; let pts;
  if (face === 'c') pts = [P(i0, j0, z1), P(i1, j0, z1), P(i1, j1, z1), P(i0, j1, z1)];
  if (face === 'a') pts = [P(i0, j1, z0), P(i1, j1, z0), P(i1, j1, z1), P(i0, j1, z1)];
  if (face === 'w') pts = [P(i1, j0, z0), P(i1, j1, z0), P(i1, j1, z1), P(i1, j0, z1)];
  g.globalAlpha = a; poly(pts, col); g.globalAlpha = 1;
}
function mmDims(P, m, n, k, a, { lm = `M ${m}`, ln = `N ${n}`, lk = `K ${k}` } = {}) {
  if (a <= 0.003) return;
  const pm = P(m / 2, n, 0), pn = P(0, n / 2, k), pk = P(0, n, k / 2);
  lab(lm, pm[0] + 14, pm[1] - 20, MMC.cD, a, 'left', 18, 800, MONO);
  lab(ln, pn[0] - 6, pn[1] + 34, MMC.cD, a, 'right', 18, 800, MONO);
  lab(lk, pk[0] + 12, pk[1] + 30, MMC.aD, a, 'left', 18, 800, MONO);
}
// 三面板：显存层（block）→ 共享内存层（warp）→ 寄存器层（片段）。每个面板同一种排法：A 在左、Wᵀ 在上、C 在右下；
// 上一层高亮的块是下一层的底色；面板之间用虚线放大连接；每个面板下方是这一层的循环代码，最下方是 k0、kk 的作用范围条。
const LV3 = { blk: '#7d5fd6', blkL: '#e7defa', wp: '#0fa391', wpL: '#cdeee7', fr: '#e0679a', frL: '#f8d6e3' };
function panelHead(x, y, l1, l2, l3, a, l4 = '') {
  if (a <= 0.003) return;
  lab(l1, x, y, '#c2412d', a, 'left', 19, 800); lab(l2, x, y + 26, '#2f5fd0', a, 'left', 17, 800); lab(l3, x, y + 50, '#1f7a4f', a, 'left', 17, 800);
  if (l4) lab(l4, x + tw(l3, 17, 800) + 12, y + 50, C.gray, a, 'left', 15, 700);
}
function panelCode(x, y, rows, a, litN) {
  if (a <= 0.003) return;
  rows.forEach(([t, col, cnt], i) => {
    const on = i < litN, yy = y + i * 34;
    rr(x, yy - 18, 6, 24, col, a * (on ? 1 : 0.25));
    lab(t, x + 16, yy, on ? C.ink : C.gray, a * (on ? 1 : 0.5), 'left', 16, 600, MONO);
    if (cnt && on) lab(cnt, x + 16 + tw(t, 16, 600, MONO) + 12, yy, col, a, 'left', 15, 800);
  });
}
function kArrow(x0, y0, x1, y1, label, col, a) {
  if (a <= 0.003) return;
  arrow(x0, y0, x1, y1, 1, col, a, 3);
  lab(label, (x0 + x1) / 2 + (y0 === y1 ? 0 : 10), (y0 + y1) / 2 + (y0 === y1 ? 22 : 6), col, a, y0 === y1 ? 'center' : 'left', 16, 800, MONO);
}
function sWmma() {
  const a0 = fin(0), a1 = fin(1), a2 = fin(2), a3 = fin(3), a4 = fin(4), a5 = fin(5);
  // ---------------- 面板 1：显存层 ----------------
  const X1 = 80, Y = 370, S = 180, gp = 10, ax = X1, wy = Y, wx = X1 + S + gp, cy = Y + S + gp;
  panelHead(X1, 296, 'GPU：一组 block', '显存', '每个 block 输出 BM × BN', a0, '本例 128 × 128');
  if (a0 > 0.003) {
    rr(ax, cy, S, S, '#d9e2f7', a0); rr(wx, wy, S, S, '#f3e6b8', a0); rr(wx, cy, S, S, LV3.blkL, a0);
    lab('A', ax - 10, cy + 22, C.blue, a0, 'right', 20, 800); lab('Wᵀ', wx + S + 10, wy + 22, C.yelD, a0, 'left', 20, 800); lab('C', wx + S + 10, cy + 22, LV3.blk, a0, 'left', 20, 800);
  }
  const bt = 0.38, bl = 0.62, bs = 18;                                         // block 在 C 中的位置（放大显示）
  const bx = wx + bl * S, by = cy + bt * S;
  if (a1 > 0.003) {
    rr(ax, by, S, bs, C.blue, a1 * 0.5); rr(bx, wy, bs, S, C.yelD, a1 * 0.5);
    rr(bx, by, bs, bs, LV3.blk, a1);
  }
  if (a2 > 0.003) {
    const pr = Math.min(1, Math.max(0, (U - T0(2) - 0.4) / 4.5)), kw = 10, kx = ax + pr * (S - kw), ky = wy + pr * (S - kw);
    rr(kx, by, kw, bs, '#d9822b', a2); rr(bx, ky, bs, kw, '#d9822b', a2);
    kArrow(ax + 20, by + bs + 16, ax + S - 20, by + bs + 16, 'k0', '#d9822b', a2);
    kArrow(bx - 16, wy + 20, bx - 16, wy + S - 20, '', '#d9822b', a2);
  }
  // ---------------- 面板 2：共享内存层 ----------------
  const X2 = 720, B = 256, u2 = B / 128, aw = 32 * u2, A2x = X2, W2y = Y, C2x = X2 + aw + gp, C2y = Y + aw + gp;
  panelHead(X2, 296, 'block：若干 warp', '共享内存', '每个 warp 输出 WM × WN', a3, '本例 64 × 32');
  if (a3 > 0.003) {
    line(bx + bs, by, C2x, C2y, LV3.blk, 1.5, a3 * 0.8, [5, 4]); line(bx + bs, by + bs, C2x, C2y + B, LV3.blk, 1.5, a3 * 0.8, [5, 4]);
    rr(A2x, C2y, aw, B, '#9db4ee', a3); rr(C2x, W2y, B, aw, '#ecd27c', a3); rr(C2x, C2y, B, B, LV3.blkL, a3);
    // w1 用到的 A 上半、Wᵀ 第二列
    rr(A2x, C2y, aw, B / 2, C.blue, a3 * 0.85); rr(C2x + B / 4, W2y, B / 4, aw, C.yelD, a3 * 0.85);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) {
      const me = i === 0 && j === 1;
      if (me) rr(C2x + j * B / 4, C2y + i * B / 2, B / 4, B / 2, LV3.wp, a3);
      sr(C2x + j * B / 4, C2y + i * B / 2, B / 4, B / 2, LV3.blk, me ? 3 : 1.2, a3);
      lab(`w${i * 4 + j}`, C2x + j * B / 4 + 8, C2y + i * B / 2 + 22, me ? '#fff' : LV3.blk, a3, 'left', 15, 800, MONO);
    }
    lab('As：BM × BK', A2x + aw / 2, C2y + B + 22, C.blue, a3, 'center', 14, 800); lab('Ws：BK × BN', C2x + B + 10, W2y + aw / 2 + 6, C.yelD, a3, 'left', 14, 800);
  }
  if (a4 > 0.003) {
    const h = Math.floor(Math.max(0, U - T0(4) - 0.3) / 1.4) % 2, hw = aw / 2;
    sr(A2x + h * hw, C2y, hw, B, '#2e9b6a', 3.5, a4); sr(C2x, W2y + h * hw, B, hw, '#2e9b6a', 3.5, a4);
    arrow(A2x + 4, W2y + aw - 12, A2x + aw - 4, W2y + aw - 12, 1, '#2e9b6a', a4, 3);
    const an = a4 * out(5), nx = C2x + B + 24, ny = C2y + 60;          // 为什么一次搬 32 列（面板 3 出现前淡出）
    if (an > 0.003) {
      lab('段长 BK 的取舍', nx, ny, C.ink, an, 'left', 18, 800);
      lab('BK 越大：搬运与同步越少', nx, ny + 32, '#d9822b', an, 'left', 16, 800);
      lab('BK 越大：共享内存占用越多', nx, ny + 60, C.gray, an, 'left', 16, 800);
      lab('kk 再按指令的 K 切开', nx, ny + 96, '#2e9b6a', an, 'left', 16, 800);
    } lab('kk', A2x + aw / 2, W2y + aw - 26, '#2e9b6a', a4, 'center', 16, 800, MONO);
  }
  // ---------------- 面板 3：寄存器层 ----------------
  const X3 = 1330, F = 62, A3x = X3, B3y = Y + 10, C3x = X3 + F + gp, C3y = B3y + F + gp;
  panelHead(X3, 296, 'warp → Tensor Core', '寄存器', '每条指令 16 × 16 × 16', a5, '形状固定');
  if (a5 > 0.003) {
    line(C2x + B / 4 + B / 4, C2y, C3x, C3y, LV3.wp, 1.5, a5 * 0.8, [5, 4]); line(C2x + B / 2, C2y + B / 2, C3x, C3y + 4 * F, LV3.wp, 1.5, a5 * 0.8, [5, 4]);
    const st = Math.max(0, Math.floor((U - T0(5) - 0.4) / 1.0)), sI = st % 8, done = Math.min(8, st), ci = Math.floor(sI / 2), cj = sI % 2;
    for (let i = 0; i < 4; i++) rr(A3x + 2, C3y + i * F + 2, F - 4, F - 4, C.blue, a5 * (i === ci ? 1 : 0.35));
    for (let j = 0; j < 2; j++) rr(C3x + j * F + 2, B3y + 2, F - 4, F - 4, C.yelD, a5 * (j === cj ? 1 : 0.35));
    for (let f = 0; f < 8; f++) {
      const i = Math.floor(f / 2), j = f % 2, now = f === sI;
      rr(C3x + j * F + 2, C3y + i * F + 2, F - 4, F - 4, now ? LV3.fr : f < done ? '#ef9fbf' : LV3.wpL, a5);
      if (now) sr(C3x + j * F, C3y + i * F, F, F, C.ink, 3, a5);
    }
    sr(C3x, C3y, 2 * F, 4 * F, LV3.wp, 3, a5);
    kArrow(C3x + 2 * F + 18, C3y + 10, C3x + 2 * F + 18, C3y + 4 * F - 10, 'i', '#0b7d70', a5);
    kArrow(C3x + 8, C3y + 4 * F + 18, C3x + 2 * F - 8, C3y + 4 * F + 18, 'j', '#0b7d70', a5);
    lab(`c[${ci}][${cj}] += a[${ci}] × b[${cj}]`, X3 + 230, C3y + 60, C.ink, a5, 'left', 19, 800, MONO);
    lab('a、b 在片段之间复用', X3 + 230, C3y + 96, C.ink2, a5, 'left', 17, 700);
    lab('a 16 × 16', A3x + F / 2, C3y + 4 * F + 22, C.blue, a5, 'center', 14, 800);
  }
  // ---------------- 下方：各层的循环代码与作用范围条 ----------------
  const CY = 790;
  panelCode(X1, CY, [['kernel<<<grid, block>>>', LV3.blk, '(M/BM)×(N/BN) 个'], ['for (k0 = 0; k0 < K; k0 += BK)', '#d9822b', 'K/BK 次'], ['  As, Ws ← A, W 的 BK 列', '#d9822b', '']], a0, U >= T0(2) ? 3 : U >= T0(1) ? 1 : 0);
  panelCode(X2, CY, [['warp → (wRow, wCol)', LV3.wp, ''], ['for (kk = 0; kk < BK; kk += 16)', '#2e9b6a', 'BK/16 次'], ['  a[i], b[j] ← As, Ws', '#2e9b6a', '']], a3, U >= T0(4) ? 3 : 1);
  panelCode(X3, CY, [['for i < WM/16, j < WN/16:', LV3.fr, ''], ['  mma_sync(c[i][j], a[i], b[j])', LV3.fr, '']], a5, 2);
  // 作用范围条：k0 包住面板 1 的循环体以及面板 2、3；kk 包住面板 2 的循环体与面板 3
  const sy = 912;
  if (a2 > 0.003) { const x1 = U >= T0(5) ? 1840 : U >= T0(3) ? 1290 : 640; rr(X1, sy, x1 - X1, 8, '#d9822b', a2); lab('k0 循环的范围（K / BK 次）', X1, sy + 34, '#d9822b', a2, 'left', 16, 800); }
  if (a4 > 0.003) { const x1 = U >= T0(5) ? 1840 : 1290; rr(X2, sy + 46, x1 - X2, 8, '#2e9b6a', a4); lab('kk 循环的范围（BK / 16 次）', X2, sy + 80, '#2e9b6a', a4, 'left', 16, 800); }
  // 可调参数卡片（本例取值）
  const a6 = fin(6);
  if (a6 > 0.003) {
    const px = 1560, py = 610, w = 270, h = 150;
    rr(px, py, w, h, '#fbf9f4', a6); sr(px, py, w, h, C.lg, 1.5, a6);
    lab('可调参数（本例）', px + 14, py + 28, C.ink, a6, 'left', 17, 800);
    [['BM × BN', '128 × 128', LV3.blk], ['WM × WN', '64 × 32', LV3.wp], ['BK', '32', '#d9822b'], ['指令', '16 × 16 × 16，固定', LV3.fr]].forEach(([k, v, col], r) => {
      lab(k, px + 14, py + 58 + r * 26, col, a6, 'left', 15, 800, MONO); lab(v, px + 110, py + 58 + r * 26, C.ink2, a6, 'left', 15, 700);
    });
  }
}
// 第 2 幕（续）：wmma 的数据通路与 bank 冲突
function sWmmaPath() {
  const a0 = fin(0), a1 = fin(1), tg = fin(3, 0.5, 0.4);
  tower(220, 360, 1.05, { gap: 1.12, path: [4, 3, 1, 2, 1, 0], hop: ['', '读入', '写入', '读回', 'mma'], lanes: [0.52, 0.97], hot: tg > 0.5 ? { lv: [2], hop: [1, 2, 3] } : null, busy: [0], flow: U < T0(1) ? 1 : 0,
    tags: { 1: '寄存器中转', 2: 'bank 冲突' }, tagA: tg }, a0);
  codeCard(1080, 290, 744, 'matmul_tc.cu · mm_wmma（精简）', [
    ['*(int4*)&As[…] = *(int4*)&A[…];        // 显存 → 寄存器 → 共享内存', ''],
    ['wmma::load_matrix_sync(a[i], &As[…], BK); // 共享内存 → 寄存器', '+'],
  ], { alpha: a0, hl: U < T0(1) ? [0, 1] : -1, size: 18 });
  bank64(1180, 600, a1, lin(U, T0(3) + 0.6, T0(3) + 3.2) * 4);
  prog(PV.fp32, PV.wmma, T0(2), { note: '上集的 2.1 倍 · cuBLAS 的 12%' });
}

// =====================================================================
// 第 3 幕：换位存放与 mma.sync
// =====================================================================
const HUE = ['#eab49b', '#efd27c', '#bfdc9f', '#94d3c4', '#a3bdee', '#c4b1ef', '#eeb0cf', '#d2c9b4'];
function swzGrid(x, y, a, q, read) {
  // 8 行 × 8 块（每块 16 字节）；q 为换位进度；read 为读取第 2 块时已完成的轮数（-1 不读）
  if (a <= 0.003) return;
  const cw = 92, ch = 46;
  for (let b = 0; b < 8; b++) lab(`${4 * b}–${4 * b + 3}`, x + b * cw + cw / 2, y - 12, C.gray, a, 'center', 15, 600, MONO);
  lab('bank', x - 14, y - 12, C.gray, a, 'right', 15, 700);
  const cnt = Array(8).fill(0);
  for (let r = 0; r < 8; r++) {
    lab(`行 ${r}`, x - 14, y + r * ch + 30, C.gray, a, 'right', 16, 600);
    for (let c = 0; c < 8; c++) {
      const pos = mix(c, c ^ r, q), xx = x + pos * cw, yy = y + r * ch, isRead = read >= 0 && c === 2;
      rr(xx + 2, yy + 2, cw - 4, ch - 4, HUE[c], a * (isRead ? 1 : read >= 0 ? 0.5 : 1));
      lab(String(c), xx + cw / 2, yy + 30, C.ink2, a, 'center', 18, 700, MONO);
      if (isRead) {
        const done = q > 0.5 ? read >= 1 : read > r;
        sr(xx + 3, yy + 3, cw - 6, ch - 6, done ? C.ink : C.gray, done ? 4 : 2, a);
        if (done) cnt[Math.round(pos)]++;
      }
    }
  }
  if (read >= 0) {
    const yb = y + 8 * ch + 8;
    for (let b = 0; b < 8; b++) { const n = cnt[b]; rr(x + b * cw + 10, yb, cw - 20, 6 + n * 5, n > 1 ? C.hot : n ? C.use : C.paper2, a); if (n) lab(String(n), x + b * cw + cw / 2, yb + 28 + n * 5, n > 1 ? C.hot : C.use, a, 'center', 18, 800, MONO); }
  }
}
// 第二段统一模板：左侧固定的硬件塔（位置同第 3 幕），右侧一张机制图与 1–2 行关键代码；四拍：改动、塔上路径变化、实测、瓶颈
const SEG = { tx: 220, ty: 360, ts: 1.05, rx: 1000 };
function segTower(st, a) {
  const flowOn = U >= T0(1) && U < T0(2), focus = 0.4 + 0.6 * fin(1) - 0.35 * fin(2) + 0.35 * fin(3);
  return tower(SEG.tx, SEG.ty, SEG.ts, { gap: 1.12, lanes: [0.52, 0.97], ...st, flow: flowOn ? (st.flow || 1) : 0 }, a * focus);
}
function sMma() {
  const a0 = fin(0), a1 = fin(1), tg = fin(3, 0.5, 0.3);
  // 塔：第 0 拍仍是上一版的路径（经寄存器折返），第 1 拍换成 cp.async 直达共享内存
  if (a1 < 0.5) segTower({ path: [4, 3, 1, 2, 1, 0], hop: ['', '读入', '写入', '读回', 'mma'], busy: [0], flow: 1, dim: [] }, 1);
  else segTower({ path: [4, 3, 2, 1, 0], hop: ['', 'cp.async', 'ldmatrix', 'mma.sync'], xpath: [3, 1, 2], xhop: ['', ''], xl: [0.38, 0.46], busy: [0], flow: 1,
    hot: tg > 0.5 ? { lv: [0] } : null, tags: { 0: '每块等拷贝完成' }, tagA: tg }, 1);
  // 右：换位网格（不换位读 8 轮 → 换位 → 1 轮）
  const t = U - T0(0), q = p(U, T0(0) + 4.2, T0(0) + 5.4);
  const read = t < 0.6 ? -1 : t < 4.0 ? lin(U, T0(0) + 0.6, T0(0) + 3.8) * 8 : t < 5.6 ? -1 : 1;
  const gx = SEG.rx + 80, gy = 340;
  swzGrid(gx, gy, a0 * (1 - fin(3)), q, read);
  if (a0 > 0.003 && fin(3) < 0.99) {
    const ok = t >= 5.6;
    lab(read < 0 && !ok ? '共享内存的 8 行，每行 8 个 16 字节的块' : ok ? '换位后：8 个地址分散到 8 组 bank，一轮读完' : `不换位：同一块的 8 个地址落在同一组 bank，第 ${Math.max(1, Math.ceil(read))} 轮`, gx - 70, gy + 8 * 46 + 80, ok ? C.use : read >= 0 ? C.hot : C.ink2, a0 * (1 - fin(3)), 'left', 21, 800);
    codeCard(gx - 70, gy + 8 * 46 + 104, 814, 'mm_mma（精简）', [['swz(r, c) = r * 128 + ((c ^ (r & 7)) << 4);', '+'], ['cp_async16(sa + swz(r, c), A + …);', '+']], { alpha: a0 * (1 - fin(3)), size: 19, lh: 36 });
  }
  // 瓶颈：拷贝、等待、计算交替
  const a3 = fin(3);
  if (a3 > 0.003) {
    const T = 12, rev = lin(U, T0(3) + 0.2, T0(3) + 2.2), L = [], Cc = [];
    for (let b = 0; b < 3; b++) { const t0 = b * 4; L.push([t0, t0 + 2, C.hot, '拷贝']); Cc.push([t0, t0 + 2.4, IDLE, '等待'], [t0 + 2.4, t0 + 4, C.teal, '计算']); }
    lab('每一块：发出拷贝 → 等待完成 → 同步 → 计算', SEG.rx + 10, 470, C.ink2, a3, 'left', 20, 800);
    tline(SEG.rx + 150, 500, 640, [['拷贝', L, C.hot], ['计算', Cc, C.tealD]], T, rev, a3, { lh: 38, gap: 10, ns: 17 });
  }
  prog(PV.wmma, PV.mma, T0(2), { note: '上一步的 2.9 倍' });
}

// ---------- 三级缓冲 ----------
const SLOT = [['计算中', C.teal, '#fff'], ['已到', C.use, '#fff'], ['在途', '#f0c6a8', C.hot]];
function ring3(cx, cy, R, a, t) {
  if (a <= 0.003) return;
  for (let s = 0; s < 3; s++) {
    const kblk = ((s - t) % 3 + 3) % 3, role = SLOT[kblk], a0 = -Math.PI / 2 + s * 2 * Math.PI / 3 + 0.04, a1 = a0 + 2 * Math.PI / 3 - 0.08;
    g.globalAlpha = a; g.fillStyle = role[1]; g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, a0, a1); g.closePath(); g.fill(); g.globalAlpha = 1;
    const m = (a0 + a1) / 2;
    lab(`槽位 ${s}`, cx + Math.cos(m) * R * 0.6, cy + Math.sin(m) * R * 0.6 - 6, role[2], a, 'center', 17, 800);
    lab(role[0], cx + Math.cos(m) * R * 0.6, cy + Math.sin(m) * R * 0.6 + 16, role[2], a, 'center', 15, 700);
  }
  g.globalAlpha = a; g.fillStyle = C.paper; g.beginPath(); g.arc(cx, cy, R * 0.2, 0, 7); g.fill(); g.globalAlpha = 1;
}
function sMma3() {
  const a0 = fin(0), a1 = fin(1), tg = fin(3, 0.5, 0.3), t = Math.max(0, Math.floor((Math.min(U, T0(1)) - T0(0) - 0.6) / 1.4));
  segTower({ path: [4, 3, 2, 1, 0], hop: ['', 'cp.async', 'ldmatrix', 'mma.sync'], busy: a1 > 0.5 ? [0, 2] : [0], flow: a1 > 0.5 ? 1.4 : 1, nDots: a1 > 0.5 ? 10 : 7,
    hot: tg > 0.5 ? { hop: [3] } : null, tags: { 1: 'mma.sync 指令太小' }, tagA: tg }, 1);
  ring3(SEG.rx + 170, 470, 130, a0, t);
  lab('共享内存：3 个槽位', SEG.rx + 40, 316, C.ink2, a0, 'left', 19, 800);
  codeCard(SEG.rx + 360, 330, 464, 'mm_mma<…, 3>（精简）', [['cp_async_wait<STAGES - 2>();', '+'], ['load((t + 2) % STAGES, …);', '+']], { alpha: a0, size: 18, lh: 36 });
  // 时间线：改前与改后
  const T = 12, rev = lin(U, T0(0) + 0.6, T0(0) + 3.0), bC = [], nL = [], nC = [];
  for (let b = 0; b < 3; b++) bC.push([b * 4, b * 4 + 2.4, '#8f887a', '等待'], [b * 4 + 2.4, b * 4 + 4, '#2f8f80', '计算']);
  for (let b = 0; b < 8; b++) nL.push([b * 1.6, b * 1.6 + 1.4, C.hot, '拷贝']);
  nC.push([0, 3.2, IDLE, '']); for (let b = 0; b < 6; b++) nC.push([3.2 + b * 1.6, 3.2 + b * 1.6 + 1.5, C.teal, '计算']);
  lab('改前', SEG.rx + 100, 712, C.gray, a0, 'right', 18, 700);
  tline(SEG.rx + 230, 690, 590, [['计算', bC, C.gray]], T, 1, a0 * 0.8, { lh: 30, gap: 0, ns: 16 });
  lab('改后', SEG.rx + 100, 786, C.ink2, a0, 'right', 18, 800);
  tline(SEG.rx + 230, 762, 590, [['拷贝', nL, C.hot], ['计算', nC, C.tealD]], T, rev, a0, { lh: 32, gap: 8, ns: 16 });
  prog(PV.mma, PV.mma3, T0(2), { note: '+15%' });
}

// ---------- TMA 与 wgmma ----------
function sWgmma() {
  const a0 = fin(0), a1 = fin(1), tg = fin(3, 0.5, 0.4);
  if (a1 < 0.5) segTower({ path: [4, 3, 2, 1, 0], hop: ['', 'cp.async', 'ldmatrix', 'mma.sync'], busy: [0, 2], flow: 1.2 }, 1);
  else {
    const G = segTower({ path: [4, 3, 2, 0], hop: ['', 'TMA', 'wgmma'], xpath: [2, 1, 0], xhop: ['', ''], xl: [0.38, 0.46], blockHop: 1, busy: [0, 2], flow: 1, nDots: 5,
      hot: tg > 0.5 ? { lv: [0] } : null, tags: { 0: '两块之间空闲' }, tagA: tg }, 1);
    if (G) { rr(SEG.tx + 250, G.Y(1) + 12, 20, 20, C.purple, a1); lab('累加结果', SEG.tx + 276, G.Y(1) + 30, C.purple, a1, 'left', 17, 800); }
  }
  // 右：TMA 一个线程搬整块；两条指令的计算量按体积并排
  const aM = a0 * (1 - fin(3));
  if (aM > 0.003) {
    lab('TMA：一个线程发出一条指令，搬运整个矩形块', SEG.rx + 20, 330, C.ink2, aM, 'left', 20, 800);
    const fly = p(U, T0(0) + 0.6, T0(0) + 2.2), bx = SEG.rx + 60 + fly * 300;
    g.globalAlpha = aM; g.fillStyle = C.thr; g.beginPath(); g.arc(SEG.rx + 36, 380, 8, 0, 7); g.fill(); g.globalAlpha = 1;
    rr(bx, 360, 120, 44, C.blue, aM); for (let r = 0; r < 3; r++) rr(bx + ((r * 3) % 7) * 16, 362 + r * 14, 16, 12, '#c8d6f7', aM);
    lab('L2', SEG.rx + 60, 430, L2C, aM, 'left', 16, 800); lab('共享内存', SEG.rx + 400, 430, C.use, aM, 'left', 16, 800);
    lab('一条指令的计算量（体积按比例）', SEG.rx + 20, 500, C.ink2, aM, 'left', 20, 800);
    ibox(SEG.rx + 120, 600, 16, 128, 64, 0.62, TEALS, aM);
    lab('wgmma', SEG.rx + 260, 600, C.tealD, aM, 'left', 19, 800);
    ibox(SEG.rx + 560, 600, 16, 8, 16, 0.62, TEALS, aM);
    lab('mma.sync', SEG.rx + 590, 600, C.tealD, aM, 'left', 19, 800);
    codeCard(SEG.rx + 20, 700, 804, 'mm_wgmma（精简）', [['if (tid == 0) tma_load(sa, &tmA, &full[s], …);', '+'], ['wgmma_n128(d, desc(sa), desc(sw));', '+']], { alpha: aM, size: 18, lh: 36 });
  }
  // 瓶颈：同一个 warpgroup 既搬运又计算
  const a3 = fin(3);
  if (a3 > 0.003) {
    const T = 14, rev = lin(U, T0(3) + 0.2, T0(3) + 2.0), L = [], Cc = [];
    for (let b = 0; b < 3; b++) { const t0 = b * 4.6; L.push([t0, t0 + 0.6, C.thr, '']); Cc.push([t0, t0 + 0.9, IDLE, ''], [t0 + 0.9, t0 + 3.3, C.teal, 'wgmma'], [t0 + 3.3, t0 + 4.6, IDLE, '同步']); }
    lab('同一个 warpgroup：发出 TMA → 等待 → 计算 → 全体同步 → 再发出', SEG.rx + 10, 470, C.ink2, a3, 'left', 19, 800);
    tline(SEG.rx + 180, 500, 620, [['TMA', L, C.thr], ['Tensor Core', Cc, C.tealD]], T, rev, a3, { lh: 38, gap: 10, ns: 17 });
  }
  prog(PV.mma3, PV.wgmma, T0(2), { note: 'cuBLAS 的 62%' });
}

// ---------- warp 特化 ----------
function slots4(x, y, a, t) {
  if (a <= 0.003) return;
  const sw = 110, gap = 18;
  for (let s = 0; s < 4; s++) {
    const ph = ((t * 0.7 - s * 0.25) % 1 + 1) % 1, xx = x + s * (sw + gap);
    const st = ph < 0.35 ? 'w' : ph < 0.45 ? 'f' : ph < 0.85 ? 'r' : 'e';
    rr(xx, y, sw, 80, C.paper2, a);
    if (st === 'w') rr(xx, y + 80 * (1 - ph / 0.35), sw, 80 * ph / 0.35, '#f0c6a8', a);
    if (st === 'f') rr(xx, y, sw, 80, C.use, a);
    if (st === 'r') rr(xx, y, sw, 80, '#cfe6d7', a);
    sr(xx, y, sw, 80, C.lg, 1.5, a);
    lab(`槽 ${s}`, xx + sw / 2, y + 36, C.ink2, a, 'center', 17, 800);
    lab({ w: '写入中', f: '已到齐', r: '读取中', e: '已空出' }[st], xx + sw / 2, y + 62, C.ink2, a, 'center', 14, 700);
    rr(xx + 2, y - 30, 50, 22, st === 'f' || st === 'r' ? C.use : '#d9d3c5', a); lab('full', xx + 27, y - 13, '#fff', a, 'center', 15, 800, MONO);
    rr(xx + 56, y - 30, 54, 22, st === 'e' || st === 'w' ? C.thr : '#d9d3c5', a); lab('empty', xx + 83, y - 13, '#fff', a, 'center', 15, 800, MONO);
  }
}
function wgBox(x, y, w, title, col, a, lit = 128) {
  if (a <= 0.003) return;
  rr(x, y, w, 86, '#fbf9f4', a); sr(x, y, w, 86, col, 2.5, a);
  lab(title, x + 12, y + 26, col, a, 'left', 17, 800);
  for (let r = 0; r < 4; r++) dots(x + 12, y + 36 + r * 11, 32, 1, (w - 24) / 32, col, a, Math.max(0, lit - r * 32), 0.32);
}
function sWs() {
  const a0 = fin(0), a1 = fin(1), a2 = fin(2), tg = fin(3, 0.5, 0.3);
  segTower({ path: [4, 3, 2, 0], hop: ['', 'TMA', 'wgmma'], busy: a1 > 0.5 ? [0, 2] : [2], flow: a1 > 0.5 ? 1.4 : 1, nDots: a1 > 0.5 ? 9 : 5,
    hot: tg > 0.5 ? { lv: [3], hop: [0] } : null, tags: { 3: 'W 放不下 L2' }, tagA: tg }, 1);
  const x = SEG.rx + 10;
  wgBox(x, 300, 800, '0 号 warpgroup：生产者（一个线程发出 TMA）', C.thr, a0, 1);
  slots4(x + 4, 440, a0, Math.min(U, T0(2)));
  wgBox(x, 548, 392, '1 号：消费者', C.use, a0); wgBox(x + 408, 548, 392, '2 号：消费者', C.use, a0);
  // 第 1 拍：两个消费者交替，Tensor Core 一轨连续
  const aT = a1 * (1 - a2);
  if (aT > 0.003) {
    const T = 12, rev = lin(U, T0(1), T0(1) + 1.6), c1 = [], c2 = [], tc = [];
    for (let b = 0; b < 6; b++) { c1.push([b * 2, b * 2 + 1, C.teal, 'wgmma'], [b * 2 + 1, b * 2 + 2, IDLE, '']); c2.push([b * 2 + 1, b * 2 + 2, C.teal, 'wgmma'], [b * 2, b * 2 + 1, IDLE, '']); tc.push([b * 2, b * 2 + 2, C.teal, '']); }
    tline(x + 150, 700, 640, [['消费者 1', c1, C.use], ['消费者 2', c2, C.use], ['Tensor Core', tc, C.tealD]], T, rev, aT, { lh: 30, gap: 8, ns: 16 });
  }
  // 第 2 拍起：两个规模的实测
  if (a2 > 0.003) {
    const mx = x + 150, my = 712, X = (v) => mx + 560 * v / 800, gk = po(U, T0(2) + 0.3, T0(2) + 1.4);
    [['4096 阶', 481.44, 634.69], ['8192 阶', 484.21, 479.74]].forEach(([n, v0, v1], i) => {
      const yy = my + i * 70;
      lab(n, mx - 14, yy + 32, C.ink2, a2, 'right', 18, 800);
      rr(mx, yy, X(v0) - mx, 20, '#9fd6c9', a2);
      rr(mx, yy + 24, (X(v1) - mx) * gk, 22, C.teal, a2);
      lab(i === 0 ? '提高约三分之一' : '没有变化', mx + (X(v1) - mx) * gk + 10, yy + 40, i === 0 ? C.tealD : C.hot, a2 * gk, 'left', 17, 800);
    });
    lab('浅：上一步　深：warp 特化', mx, my + 150, C.gray, a2, 'left', 15, 700);
  }
  prog(PV.wgmma, PV.ws, T0(2), { note: '8192 阶几乎不变' });
}

// =====================================================================
// 第 7 幕：tile 分组发放
// =====================================================================
function sL2() {
  const sw = p(U, T0(0) + 1.0, T0(0) + 2.4);
  const wave = U > T0(2) + 0.4 ? (U > T0(2) + 1.6 ? 2 : 1) : 0;
  const order = sw < 0.5 ? 'row' : 'grp';
  tileMap(190, 440, 16, 7.2, order, wave, 1);
  if (wave > 0) { rr(470, 306, 18, 18, C.purple, 1); lab('正在运行的一轮', 494, 322, C.ink2, 1, 'left', 16, 700); rr(630, 306, 18, 18, '#cbb8f2', 1); lab('已完成', 654, 322, C.ink2, 1, 'left', 16, 700); }
  lab(order === 'row' ? '改前：按行发放' : '改后：16 行一组，组内按列', 110, 300, C.purple, 1, 'left', 20, 800);
  lab(order === 'row' ? '一轮覆盖约 4 行 × 全部 32 列' : '一轮覆盖 16 行 × 约 8 列', 110, 330, C.purple, 1, 'left', 18, 700);
  codeCard(1290, 290, 534, 'wgmma_l2 · tile_mn（精简）', [
    ['m = first + r % 16;', '+'],
    ['n = r / 16;            // 组内按列', '+'],
  ], { alpha: 1, size: 20, hl: U < T0(1) ? [0, 1] : -1 });
  // 工作集
  const a1 = fin(1), k = order === 'grp' ? p(U, T0(1) + 0.3, T0(1) + 1.6) : 0;
  wsetBar(800, 470, 440, mix(8, 32, k), mix(128, 33, k), 1);
  if (a1 > 0.003) {
    lab('减为一半（仍大于 L2）', 800, 590, C.ink2, a1, 'left', 18, 800);
    lab('每块 W 同时被 16 个 tile 读取（按行时约 4 个）', 800, 618, C.yelD, a1, 'left', 18, 800);
  }
  // 硬件塔：显存 → L2 由橙转绿
  const a2 = fin(2), good = a2 > 0.5, aT = 1 - fin(4, 0.4);
  tower(860, 660, 0.66, { path: [4, 3, 2, 0], noSub: true, flow: 0.8, hot: good ? null : { lv: [3, 4], hop: [0] }, busy: good ? [3] : [], nDots: 6 }, aT);
  if (a2 > 0.003) lab('同一块 W 读入 L2 一次，被 16 个 tile 使用', 1160, 660 + 3 * 66 + 30, C.use, a2 * aT, 'left', 18, 800);
  // 收尾：每个 SM 各自从 L2 读整块 W
  const a4 = fin(4, 0.5, 0.3);
  if (a4 > 0.003) {
    const fx = 800, fy = 680;
    lab('132 个 SM 各自从 L2 读取 A 块与整块 W', fx, fy, C.ink2, a4, 'left', 18, 800);
    lab('L2 → SM 约 8.7 TB/s（推算），三分之二是 W', fx, fy + 30, C.hot, a4, 'left', 21, 800);
    rr(fx, fy + 236, 1020, 40, '#e8e3f2', a4); sr(fx, fy + 236, 1020, 40, L2C, 2, a4); lab('L2', fx + 14, fy + 264, L2C, a4, 'left', 20, 800);
    for (let i = 0; i < 8; i++) {
      const sx = fx + 24 + i * 126, pair = i === 2 || i === 3;
      rr(sx, fy + 56, 100, 62, '#fbf9f4', a4); sr(sx, fy + 56, 100, 62, pair ? C.hot : C.lg, pair ? 3 : 1.5, a4);
      lab(`SM ${i}`, sx + 50, fy + 82, C.ink2, a4, 'center', 16, 800); if (pair) lab('同一块 W', sx + 50, fy + 106, C.hot, a4, 'center', 14, 800);
      line(sx + 34, fy + 234, sx + 34, fy + 120, C.yel, 4, a4); line(sx + 66, fy + 234, sx + 66, fy + 120, '#9db4ee', 4, a4);
      const yy = fy + 222 - ((U * 110 + i * 37) % 100);
      rr(sx + 28, yy, 12, 12, C.yelD, a4); rr(sx + 60, Math.max(fy + 122, yy - 30), 12, 12, C.blue, a4);
    }
    lab('…', fx + 1012, fy + 96, C.gray, a4, 'left', 24, 800);
    bnTag(fx + 600, fy + 2, '各 block 重复读取 W', a4 * fin(4, 0.4, 1.4));
  }
  prog(PV.ws, PV.l2, T0(3), { note: '1.54 倍 · cuBLAS 的 95%' });
}

// =====================================================================
// 第 8 幕：集群与 TMA 多播
// =====================================================================
function cLocal(x, y, a) {
  // C 的局部：6 行 × 4 列 tile，上下相邻的两个 tile 用同一块 W
  if (a <= 0.003) return;
  const cw = 92, ch = 46;
  rr(x, y - 70, 4 * cw, 56, '#f3e6b8', a); rr(x + cw, y - 70, cw, 56, C.yelD, a);
  rr(x - 70, y, 56, 6 * ch, '#d9e2f7', a); rr(x - 70, y + 2 * ch, 56, 2 * ch, C.blue, a);
  for (let r = 0; r < 6; r++) for (let q = 0; q < 4; q++) { const on = q === 1 && (r === 2 || r === 3); rr(x + q * cw + 1, y + r * ch + 1, cw - 2, ch - 2, on ? C.purple : '#e7defa', a); }
  sr(x + cw - 2, y + 2 * ch - 2, cw + 4, 2 * ch + 4, C.ink, 3, a);
  lab('block 0', x + cw * 1.5, y + 2 * ch + 30, '#fff', a, 'center', 16, 800); lab('block 1', x + cw * 1.5, y + 3 * ch + 30, '#fff', a, 'center', 16, 800);
  lab('Wᵀ 同一块', x + cw * 1.5, y - 34, '#fff', a, 'center', 17, 800);
  lab('C 的局部（每格一个 128 × 256 的 tile）', x - 70, y + 6 * ch + 26, C.purple, a, 'left', 16, 700);
}
function sCluster() {
  const a0 = fin(0);
  cLocal(180, 410, a0);
  // 常驻 kernel：132 个 block 循环领取 tile（12 × 11 个小格，当前一轮依次点亮）
  const ap = fin(1);
  if (ap > 0.003) {
    lab('常驻 kernel：只启动 132 个 block', 110, 778, C.ink2, ap, 'left', 17, 800);
    const rnd = Math.floor(Math.max(0, U - T0(1)) / 1.2);
    for (let i = 0; i < 132; i++) rr(110 + (i % 22) * 21, 792 + Math.floor(i / 22) * 13, 18, 10, (i + rnd * 7) % 22 < 11 ? C.purple : '#cbb8f2', ap);
    lab(`各自循环领取 tile：第 ${Math.min(15, rnd + 1)} 轮（共约 15.5 轮）`, 110, 900, C.purple, ap, 'left', 16, 700);
    lab('单独加上常驻：740 → 657（−11%）', 110, 926, C.hot, ap, 'left', 16, 800);
  }
  lab('集群：上下相邻、用同一块 W 的两个 tile', 110, 300, C.ink2, a0, 'left', 19, 800);
  // 两个 SM 与 L2
  const X0 = 640, Y0 = 330, a1 = fin(2), half = p(U, T0(2) + 0.2, T0(2) + 1.2), HC = [C.yel, '#d9a300'];
  g.globalAlpha = a0; g.strokeStyle = C.ink2; g.lineWidth = 2; g.setLineDash([10, 6]); g.strokeRect(X0 - 16, Y0 - 16, 620, 270); g.setLineDash([]); g.globalAlpha = 1;
  lab('线程块集群（thread block cluster）：2 个 block 在相邻的 SM', X0 - 10, Y0 - 26, C.ink2, a0, 'left', 18, 800);
  for (let b = 0; b < 2; b++) {
    const sx = X0 + b * 310;
    rr(sx, Y0, 280, 238, '#fbf9f4', a0); sr(sx, Y0, 280, 238, C.lg, 1.5, a0);
    lab(`block ${b}（SM ${b}）`, sx + 14, Y0 + 30, C.ink2, a0, 'left', 18, 800);
    rr(sx + 14, Y0 + 46, 252, 178, '#e3f1e7', a0); lab('共享内存', sx + 24, Y0 + 70, C.use, a0, 'left', 16, 800);
    rr(sx + 24, Y0 + 84, 70, 128, '#9db4ee', a0); lab('A', sx + 59, Y0 + 154, '#fff', a0, 'center', 18, 800);
    for (let h = 0; h < 2; h++) { rr(sx + 108 + h * 76, Y0 + 84, 70, 128, half > 0.5 || a1 < 0.01 ? HC[h] : '#f3e6b8', a0); lab(h ? 'W 后半' : 'W 前半', sx + 143 + h * 76, Y0 + 154, C.ink2, a0, 'center', 15, 800); }
  }
  const ly = Y0 + 420;
  rr(X0 - 16, ly, 620, 80, '#e8e3f2', a0); sr(X0 - 16, ly, 620, 80, L2C, 2, a0); lab('L2', X0, ly + 48, L2C, a0, 'left', 22, 800);
  rr(X0 + 230, ly + 14, 70, 52, HC[0], a0); rr(X0 + 304, ly + 14, 70, 52, HC[1], a0); lab('W 块（256 行）', X0 + 384, ly + 46, C.yelD, a0, 'left', 16, 800);
  if (a1 > 0.003) {
    for (let h = 0; h < 2; h++) for (let b = 0; b < 2; b++) {
      const x0 = X0 + 265 + h * 74, y0 = ly + 12, x1 = X0 + b * 310 + 143 + h * 76, y1 = Y0 + 216, col = h ? '#b8860b' : C.yelD;
      arrow(x0, y0, x1, y1, half, col, a1, 3.5);
      if (half >= 1) { const t = (U * 0.7 + h * 0.5) % 1; rr(mix(x0, x1, t) - 7, mix(y0, y1, t) - 7, 14, 14, col, a1); }
    }
    lab('block 0 发出前半、block 1 发出后半', X0 + 430, ly - 50, C.yelD, a1, 'left', 16, 800);
    lab('TMA 多播：从 L2 读一次，写入两个 block', X0 + 430, ly - 24, C.yelD, a1, 'left', 16, 800);
  }
  codeCard(1290, 290, 534, 'wgmma_cluster（精简）', [
    ['tma_load(sa, &tmA, &full[s], k, bm);', ''],
    ['tma_load_mc(sw + rank * half, &tmW,', '+'],
    ['   &full[s], k, bn + rank * 128, 0b11);', '+'],
  ], { alpha: a0, size: 19, hl: U > T0(2) && U < T0(3) ? [1, 2] : -1 });
  // 读出量
  const a2 = fin(3), nx = 1290, ny = 520, k = p(U, T0(3) + 0.3, T0(3) + 1.5);
  if (a2 > 0.003) {
    lab('每个 block 每一步从 L2 读取', nx, ny, C.ink2, a2, 'left', 19, 800);
    const X = (v) => nx + 420 * v / 48;
    rr(nx, ny + 16, X(16) - nx, 32, C.blue, a2); rr(X(16), ny + 16, (X(48) - X(16)) * mix(1, 0.5, k), 32, C.yel, a2);
    lab(`${Math.round(mix(48, 32, k))} KB`, X(16) + (X(48) - X(16)) * mix(1, 0.5, k) + 10, ny + 42, C.ink, a2, 'left', 22, 800, MONO);
    lab('A 16 KB + W 32 KB → A 16 KB + W 16 KB', nx, ny + 76, C.gray, a2, 'left', 16, 700);
    lab(`L2 → SM：${mix(8.7, 5.9, k).toFixed(1)} TB/s（推算）`, nx, ny + 112, C.use, a2, 'left', 21, 800);
  }
  // 收尾：零散写回
  const a3 = fin(4), T = 12, rev = lin(U, T0(4) + 0.2, T0(4) + 2.0);
  if (a3 > 0.003) {
    const tx = 780;
    const cs = [[0, 4.5, C.teal, 'tile 1 计算'], [4.5, 6, '#c9a27c', '写回'], [6, 10.5, C.teal, 'tile 2 计算'], [10.5, 12, '#c9a27c', '写回']];
    const tc = [[0, 4.5, C.teal, ''], [4.5, 6, IDLE, '空闲'], [6, 10.5, C.teal, ''], [10.5, 12, IDLE, '空闲']];
    tline(tx, 862, 680, [['消费者', cs, C.use], ['Tensor Core', tc, C.tealD]], T, rev, a3, { lh: 32, gap: 8, ns: 17 });
    lab('写回：每个线程向显存零散写入 4 字节', tx, 850, '#9a6a3a', a3, 'left', 16, 800);
    lab('ncu：Tensor 管线 92.9%，cuBLAS 97.0%', 1500, 880, C.ink2, a3, 'left', 17, 800);
    bnTag(1500, 940, '写回期间空闲', a3 * fin(4, 0.4, 1.6));
  }
  prog(PV.l2, PV.cl, T0(3), { note: '常驻 + 集群 + 多播 · cuBLAS 的 96.9%' });
}

// =====================================================================
// 第 9 幕：TMA 写回
// =====================================================================
function sTstore() {
  const a0 = fin(0);
  tower(150, 330, 1.0, { gap: 1.12, path: [0, 1, 2, 3, 4], hop: ['', '换位写入', 'TMA 写回', ''], lanes: [0.62, 0.96], xpath: [1, 4], xhop: ['旧：零散写回'], xl: [0.47, 0.47], noSub: true, busy: [0, 2], pathCol: C.purple,
    dataCol: () => C.purple, flow: 0.9, note: (i) => i === 2 ? ['结果 64 KB', C.purple, a0] : null }, 1);
  // 写回的目的地：完整的 C（8192 × 8192），当前 tile 闪动
  rr(640, 740, 130, 130, '#e7defa', a0); const tk = Math.floor(U / 1.2) % 6; rr(640 + 130 * (12 + tk) / 32, 740 + 130 * 20 / 64, 130 / 32 + 2, 130 / 64 + 2, C.purple, a0); sr(640 + 130 * (12 + tk) / 32 - 3, 740 + 130 * 20 / 64 - 3, 130 / 32 + 8, 130 / 64 + 8, C.purple, 2, a0);
  lab('C（显存）', 640, 896, C.purple, a0, 'left', 17, 800); lab('每次写回一个 tile', 640, 922, C.gray, a0, 'left', 15, 700);
  arrow(600, 820, 636, 820, 1, C.purple, a0, 3);
  codeCard(820, 290, 1004, 'matmul_tc.cu · wgmma_tstore（精简）', [
    ['*(bf162*)(cs + swz(r, j)) = cvt(d[…]);   // 换位写入共享内存', '+'],
    ['fence_proxy_async();', '+'],
    ['if (tid % 128 == 0) tma_store(&tmC, cs + q * 8192, …);', '+'],
  ], { alpha: a0, size: 20, hl: U > T0(0) + 1 && U < T0(1) ? [0, 2] : -1 });
  const a1 = fin(1), T = 12, rev = lin(U, T0(1) + 0.1, T0(1) + 1.8);
  if (a1 > 0.003) {
    const cs = [[0, 4.5, C.teal, 'tile 1 计算'], [4.5, 5.0, C.purple, ''], [5.0, 9.5, C.teal, 'tile 2 计算'], [9.5, 10, C.purple, ''], [10, 12, C.teal, '']];
    const ts = [[4.6, 6.4, '#c9a27c', 'TMA 写回'], [9.6, 11.4, '#c9a27c', 'TMA 写回']];
    tline(980, 560, 840, [['消费者', cs, C.use], ['TMA', ts, '#9a6a3a'], ['Tensor Core', [[0, 12, C.teal, '']], C.tealD]], T, rev, a1, { lh: 34, gap: 8, ns: 17 });
    lab('时间线为示意', 1820, 720, C.gray, a1, 'right', 15, 600);
    lab('紫色：换位写入共享内存；写回与下一块的计算重叠', 980, 740, C.use, a1, 'left', 19, 800);
    const bx = 980, by = 810, X = (v) => bx + 840 * v / 228;
    lab('共享内存（每 SM 228 KB）：3 级缓冲 × 48 KB + 结果 64 KB = 208 KB', bx, by - 14, C.ink2, a1, 'left', 18, 700);
    rr(bx, by, 840, 36, C.paper2, a1);
    for (let i = 0; i < 3; i++) { rr(X(i * 48) + 1, by, X(48) - bx - 2, 36, C.use, a1); lab(`槽 ${i}`, X(i * 48) + 10, by + 25, '#fff', a1, 'left', 16, 800); }
    rr(X(144) + 1, by, X(64) - bx - 2, 36, C.purple, a1); lab('结果 64 KB', X(144) + 10, by + 25, '#fff', a1, 'left', 16, 800);
    lab('缓冲由 4 级减为 3 级', bx, by + 66, C.gray, a1, 'left', 17, 700);
  }
  prog(PV.cl, PV.ts, T0(2), { note: 'cuBLAS 的 99.5%' });
}

// =====================================================================
// 第 10 幕：功耗墙与回顾
// =====================================================================
function sEnd() {
  const a0 = fin(0), bx = 420, by = 330, BW = 500, X = (v) => bx + BW * v / 1000, k = po(U, T0(0) + 0.3, T0(0) + 1.6);
  lab('8192 阶 · TFLOPS（同一刻度）', 96, by - 16, C.ink2, a0, 'left', 20, 800);
  const rows = [['规格峰值（1830 MHz）', 989, C.lg, '989'], ['1360 MHz 时的峰值', 735, '#9fd6c9', '735'], ['手写版（连续 2 s，699 W）', 682.34, C.teal, '682.3'], ['cuBLAS（连续 2 s，697 W）', 681.14, C.ink2, '681.1']];
  rows.forEach(([n, v, col, s], i) => {
    const yy = by + i * 60, kk = i === 0 ? 1 : k;
    lab(n, bx - 14, yy + 28, C.ink2, a0, 'right', 18, 700);
    rr(bx, yy, (X(v) - bx) * kk, 38, col, a0);
    lab(s, i >= 2 ? bx + (X(v) - bx) * kk - 10 : bx + (X(v) - bx) * kk + 10, yy + 27, i >= 2 ? '#fff' : C.ink, a0, i >= 2 ? 'right' : 'left', 20, 800, MONO);
  });
  const a1 = fin(1);
  if (a1 > 0.003) { line(X(735), by + 46, X(735), by + 230, C.tealD, 2.5, a1, [6, 4]); lab('两者都是 735 的 93%', X(735) - 4, by + 262, C.tealD, a1, 'right', 22, 800); }
  // 补充实测（2026-10-22）：右侧面板，回顾开始时淡出
  const aS = fin(2) * (1 - fin(4, 0.5)), aT = fin(3) * (1 - fin(4, 0.5));
  if (aS > 0.003) {
    const px = 1000, py = 300;
    lab('补充实测（Modal H100，3 次重复取中位数）', px, py, C.ink2, aS, 'left', 20, 800);
    const sc = 30;                                         // 每 1% = 30 px，零线在每列中间
    ['2048 阶', '4096 阶', '8192 阶'].forEach((m, q) => lab(m, px + 480 + q * 120, py + 40, C.gray, aS, 'center', 16, 700));
    const rowsS = [['Hilbert 曲线调度', [-0.21, 0.14, -0.2], C.gray, '无增益：L2 分组已足够紧凑'], ['首块 wgmma 直接覆盖累加器', [0.39, 0.61, 1.41], C.teal, '省去清零；不带集群时 +1.1% 到 +3.1%']];
    rowsS.forEach(([n, vs, col, note], r) => {
      const y0 = py + 60 + r * 86;
      lab(n, px, y0 + 22, C.ink, aS, 'left', 19, 800); lab(note, px, y0 + 50, col === C.gray ? C.ink2 : C.tealD, aS, 'left', 15, 700);
      vs.forEach((v, q) => {
        const cx = px + 480 + q * 120;
        line(cx, y0 + 2, cx, y0 + 34, C.ink, 1.5, aS);
        rr(v >= 0 ? cx : cx + v * sc, y0 + 8, Math.abs(v) * sc, 20, col, aS);
        lab(`${v > 0 ? '+' : ''}${v.toFixed(1)}%`, cx, y0 + 54, col === C.gray ? C.ink2 : C.tealD, aS, 'center', 15, 800, MONO);
      });
    });
  }
  if (aT > 0.003) {
    const px = 1000, py = 560, X = (v) => px + 230 + 400 * v / 800;
    lab('4096 阶：不带集群 + 微调 对 cuBLAS（同一刻度）', px, py, C.ink2, aT, 'left', 19, 800);
    [['短时计时', 766.27, 753.24, '+1.7%', C.tealD], ['连续 2 s', 667.10, 682.00, '−2.2%', C.hot]].forEach(([n, a, b, d, dc], r) => {
      const yy = py + 20 + r * 92;
      lab(n, px, yy + 42, C.ink2, aT, 'left', 18, 800);
      rr(px + 230, yy, X(a) - px - 230, 30, C.teal, aT); lab(`手写 ${a.toFixed(0)}`, X(a) + 8, yy + 22, C.tealD, aT, 'left', 16, 800, MONO);
      rr(px + 230, yy + 36, X(b) - px - 230, 30, C.ink2, aT); lab(`cuBLAS ${b.toFixed(0)}`, X(b) + 8, yy + 58, C.ink, aT, 'left', 16, 800, MONO);
      lab(d, px + 120, yy + 42, dc, aT, 'left', 20, 800, MONO);
    });
    // 功耗表：连续运行时的平均功耗，700 W 上限
    const wx = px + 230, wy = py + 222, PW = (v) => wx + 400 * v / 750;
    lab('功耗', px, wy + 30, C.ink2, aT, 'left', 18, 800);
    rr(wx, wy, PW(698) - wx, 18, C.hot, aT); lab('手写 698 W', PW(698) + 8, wy + 16, C.hot, aT, 'left', 15, 800);
    rr(wx, wy + 24, PW(552) - wx, 18, C.ink2, aT); lab('cuBLAS 552 W', PW(552) - 8, wy + 39, '#fff', aT, 'right', 15, 800);
    line(PW(700), wy - 8, PW(700), wy + 50, C.red, 2.5, aT, [5, 4]); lab('700 W 上限', PW(700), wy - 12, C.red, aT, 'center', 14, 800);
    lab('4096 阶带集群 734，不带 752：集群在这一规模反而拖慢', px, wy + 78, C.gray, aT, 'left', 15, 700);
  }
  // 首尾呼应：2048 个点，93% 点亮
  const a2 = fin(4);
  dots(96, 610, 64, 32, 7.0, C.teal, a2, Math.round(2048 * 0.93 * lin(U, T0(4) + 0.3, T0(4) + 2.2)));
  lab('每个 SM 每周期 2048 次乘加：持续运行时用到 93%', 96, 868, C.tealD, a2, 'left', 20, 800);
  // 硬件塔：每一层去除的瓶颈
  const notes = ['空闲等待：第 3、5、8 步', '寄存器中转：第 2、4 步', 'bank 冲突：第 2 步', 'L2 的复用与读出量：第 6、7 步', ''];
  tower(1060, 350, 0.8, { path: [4, 3, 2, 0], noSub: true, flow: 1.1, busy: [0, 1, 2, 3, 4], lanes: [0.6, 0.94],
    note: (i) => notes[i] ? [notes[i], C.ink2, po(U, T0(4) + 0.6 + i * 0.3, T0(4) + 1.1 + i * 0.3)] : null }, a2);
  lab('本集在各层去除的瓶颈', 1060, 286, C.ink2, a2, 'left', 20, 800);
  // 结尾的阶梯：本集 8 级（总结，按线性刻度）
  if (a2 > 0.003) {
    const sx = 1060, sy = 900, Y = (v) => sy - 150 * v / 800, vs = [PV.wmma, PV.mma, PV.mma3, PV.wgmma, PV.ws, PV.l2, PV.cl, PV.ts];
    vs.forEach((v, i) => { const kk = po(U, T0(4) + 0.4 + i * 0.15, T0(4) + 0.9 + i * 0.15); rr(sx + i * 92, Y(v * kk), 70, sy - Y(v * kk), C.teal, a2); lab(fmtT(v), sx + i * 92 + 35, v > 600 ? Y(v * kk) + 26 : Y(v * kk) - 8, v > 600 ? '#fff' : C.tealD, a2 * kk, 'center', 16, 800, MONO); lab(`${i + 1}`, sx + i * 92 + 35, sy + 22, C.gray, a2, 'center', 15, 700); });
    line(sx - 6, Y(PV.cub), sx + 8 * 92, Y(PV.cub), C.ink, 2, a2, [6, 4]); lab('cuBLAS 781.9', sx - 10, Y(PV.cub) - 8, C.ink, a2, 'left', 15, 700);
    lab('第 N 步', sx - 10, sy + 22, C.gray, a2, 'right', 15, 700);
  }
  prog(0, PV.ts, null, { alpha: 1 - a2 });
}

const BY_KIND = { intro: sIntro, end: sEnd };
const BY_TAG = { mm_wmma: sWmma, mm_mma: sMma, 'mm_mma<3级>': sMma3, mm_wgmma: sWgmma, mm_wgmma_ws: sWs, wgmma_l2: sL2, wgmma_cluster: sCluster, wgmma_tstore: sTstore };
const BY_FN = { wmmaPath: sWmmaPath };
const DRAW = SCRIPT.map((s) => BY_FN[s.fn] || BY_TAG[s.tag] || BY_KIND[s.kind]);
function draw(t) {
  g.fillStyle = C.paper; g.fillRect(0, 0, W, H);
  let k = AT.length - 1; while (k > 0 && t < AT[k]) k--;
  K = k; U = t - AT[k];
  header(k, U);
  DRAW[k]();
  subtitle(SCRIPT[k].beats, typeof VO !== 'undefined' && VO ? VO.beats?.[k] : null, U);
}
// 音效：每拍开始 pop；角落小条增长 land
SCRIPT.forEach((s, k) => { s.beats.forEach((b, j) => ev(k, b[0], s.stages[j] === 3 ? 'land' : 'pop')); });
// 关键帧：每幕的“机制”一帧（stages 中最后一个 2 拍结束前）与幕尾一帧
window.KEYS = SCRIPT.map((s, k) => {
  const j = s.stages.lastIndexOf(2), mid = j >= 0 && j + 1 < s.beats.length ? s.beats[j + 1][0] - 0.15 : s.kind === 'end' ? s.beats[4][0] - 0.15 : s.beats.length > 3 ? s.beats[3][0] - 0.15 : null;
  return { mid, end: s.dur - 0.05 };
});
window.SCENE_AT = AT; window.EVENTS = EV;
start(draw, DUR);
