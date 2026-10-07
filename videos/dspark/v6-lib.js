// DSpark v6 的公共图元：配色、缓动、文字与形状，与 videos/llm-quantization/film.html 同一套写法。
const K = { paper: '#f1ede4', paper2: '#e6e0d3', ink: '#141414', ink2: '#3a3a3a', gray: '#8a8474', lg: '#b9b2a2',
  red: '#e5322d', blue: '#2350d8', yel: '#f4c21b', yelD: '#9a7500', purple: '#7c3ff5', teal: '#0fa391' };
const ZH = '"PingFang SC", "Helvetica Neue", sans-serif', HN = '"Helvetica Neue", "PingFang SC", sans-serif';
const font = (w, s, f = ZH) => `${w} ${s}px ${f}`;
let X, T = 0;
const clamp = (x) => Math.min(1, Math.max(0, x));
const clampR = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (x) => 1 - Math.pow(1 - clamp(x), 3);
const io = (x) => { x = clamp(x); return x < .5 ? 4 * x ** 3 : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const P = (t0, d = .7) => ease((T - t0) / d);                    // 缓出
const S = (t0, d = 1) => io((T - t0) / d);                       // 缓入缓出
const mix = (a, b, p) => a + (b - a) * p;
const R = (x, y, w, h, c) => { if (w > 0 && h > 0) { X.fillStyle = c; X.fillRect(x, y, w, h); } };
function TX(s, x, y, f, c, a = 'left') { X.font = f; X.fillStyle = c; X.textAlign = a; X.textBaseline = 'alphabetic'; const available = a === 'center' ? Math.min(x, 1920 - x) * 2 - 32 : a === 'right' ? x - 72 : 1824 - x; X.fillText(s, x, y, Math.max(40, available)); }
function A(a, fn) { if (a <= .003) return; X.save(); X.globalAlpha *= Math.min(1, a); fn(); X.restore(); }
function IN(t0, fn, d = .7, dy = 20) { const p = P(t0, d); if (p <= 0) return; X.save(); X.globalAlpha *= p; X.translate(0, (1 - p) * dy); fn(); X.restore(); }
function POP(t0, x, y, w, h, fn, d = .5) { const p = P(t0, d); if (p <= 0) return; X.save(); X.globalAlpha *= Math.min(1, p * 1.5); X.translate(x + w / 2, y + h / 2); X.scale(.86 + .14 * p, .86 + .14 * p); X.translate(-x - w / 2, -y - h / 2); fn(); X.restore(); }
const win = (a, b, f = .3) => Math.min(P(a, f), 1 - P(b, f));   // a 时淡入、b 时淡出
function RICH(s, x, y, f, base, align = 'left') {
  const ACC = { b: K.blue, r: K.red, y: K.yelD, p: K.purple, t: K.teal, g: K.gray };
  const parts = []; const re = /\{(\w):([^}]*)\}/g; let last = 0, m;
  while ((m = re.exec(s))) { if (m.index > last) parts.push([s.slice(last, m.index), null]); parts.push([m[2], m[1]]); last = re.lastIndex; }
  if (last < s.length) parts.push([s.slice(last), null]);
  X.font = f; const w = parts.reduce((a, [t]) => a + X.measureText(t).width, 0); let cx = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  X.textAlign = 'left'; X.textBaseline = 'alphabetic';
  X.save(); const limit = align === 'center' ? Math.min(x, 1920 - x) * 2 - 32 : align === 'right' ? x - 72 : 1824 - x; if (w > limit) { X.translate(cx, 0); X.scale(limit / w, 1); cx = 0; }
  for (const [t, k] of parts) { X.fillStyle = k ? ACC[k] : base; X.fillText(t, cx, y); cx += X.measureText(t).width; } X.restore();
}
const PAT = {};
function hatch(col, bg) { const k = col + bg; if (PAT[k]) return PAT[k]; const g = 14, c = document.createElement('canvas'); c.width = c.height = g; const q = c.getContext('2d');
  q.fillStyle = bg; q.fillRect(0, 0, g, g); q.strokeStyle = col; q.lineWidth = 4; q.beginPath(); for (let o = -g; o <= g * 2; o += g) { q.moveTo(o, g); q.lineTo(o + g, 0); } q.stroke(); return (PAT[k] = X.createPattern(c, 'repeat')); }
const HR = (x, y, w, h, col = K.lg, bg = K.paper2) => { if (w > 0 && h > 0) { X.fillStyle = hatch(col, bg); X.fillRect(x, y, w, h); } };
const box = (x, y, w, h, lw = 4, col = K.ink) => { X.strokeStyle = col; X.lineWidth = lw; X.strokeRect(x, y, w, h); };
function line(x1, y1, x2, y2, col = K.ink, lw = 3, dash) { X.strokeStyle = col; X.lineWidth = lw; X.setLineDash(dash || []); X.beginPath(); X.moveTo(x1, y1); X.lineTo(x2, y2); X.stroke(); X.setLineDash([]); }
function poly(pts, col = K.ink, lw = 4) { X.strokeStyle = col; X.lineWidth = lw; X.lineJoin = 'round'; X.beginPath(); pts.forEach(([x, y], i) => (i ? X.lineTo(x, y) : X.moveTo(x, y))); X.stroke(); }
function arrow(x1, y1, x2, y2, col = K.ink, lw = 5) { const a = Math.atan2(y2 - y1, x2 - x1), hl = lw * 3.4, hw = lw * 2.2; if (Math.hypot(x2 - x1, y2 - y1) < hl) return;
  line(x1, y1, x2 - Math.cos(a) * hl * .9, y2 - Math.sin(a) * hl * .9, col, lw);
  X.fillStyle = col; X.beginPath(); X.moveTo(x2, y2); X.lineTo(x2 - Math.cos(a) * hl + Math.sin(a) * hw, y2 - Math.sin(a) * hl - Math.cos(a) * hw); X.lineTo(x2 - Math.cos(a) * hl - Math.sin(a) * hw, y2 - Math.sin(a) * hl + Math.cos(a) * hw); X.fill(); }
// 画到一半的箭头：p 从 0 到 1
const arrowP = (x1, y1, x2, y2, p, col, lw) => { if (p > .02) arrow(x1, y1, mix(x1, x2, p), mix(y1, y2, p), col, lw); };
function cell(x, y, w, h, col, label, size = 30, tc = '#fff', fam = ZH) { R(x, y, w, h, col); if (label != null) TX(label, x + w / 2, y + h / 2 + size * .36, font(700, size, fam), tc, 'center'); }
function span(x1, x2, y, col = K.ink, lw = 4) { line(x1, y, x2, y, col, lw); line(x1, y - 10, x1, y + 10, col, lw); line(x2, y - 10, x2, y + 10, col, lw); }
const dot = (x, y, r, c) => { X.fillStyle = c; X.beginPath(); X.arc(x, y, r, 0, 7); X.fill(); };
// 节拍：第 k 幕的每句配音一个节拍，pre[i] 为第 i 句之前留给画面的空档。返回每句的起止（幕内时间）；at(B, i, f) 为第 i 句进行到 f 处的时刻
function beatTimes(k, pre = []) { let t = 0; return (window.BEATS[k] || []).map((b, i) => { t += pre[i] ?? .4; const s = t; t += b.dur; return { s, e: t, d: b.dur }; }); }
const at = (B, i, f = 0) => B[i].s + B[i].d * f;
window.BEAT_AT = {};
// 把一幕原有的动画时间对齐到配音节拍：anchors 为 [原时间, 节拍序号]，表示原动画在这一刻对应这句配音的开头。
// 动画按原速播放，先到就停住等配音；一段配音比动画短时最多加速到 1.25 倍，其余由推后节拍补足。oldEnd 为原动画最后一个变化之后的时刻
function beatWarp(k, anchors, oldEnd, pre = []) {
  const G = window.BEATS[k], MAXR = 1.25, A = [[0, 0]], B = []; let t = 0, lo = 0, ln = 0;
  G.forEach((b, i) => { t += pre[i] ?? .4; const an = anchors.find((a) => a[1] === i);
    if (an) { t = Math.max(t, ln + (an[0] - lo) / MAXR); A.push([an[0], t]); lo = an[0]; ln = t; }
    B.push({ s: t, e: t + b.dur, d: b.dur }); t += b.dur; });
  const end = Math.max(t + 1.2, ln + (oldEnd - lo) / MAXR + .6); A.push([oldEnd, end]);
  return { B, A, dur: end };
}
function warpT(A, t) { for (let i = 0; i < A.length - 1; i++) { const [o0, n0] = A[i], [o1, n1] = A[i + 1]; if (t < n1) { const r = Math.max(1, (o1 - o0) / Math.max(1e-6, n1 - n0)); return Math.min(o1, o0 + (t - n0) * r); } } return A[A.length - 1][0]; }
function unwarpT(A, o) { for (let i = 0; i < A.length - 1; i++) { const [o0, n0] = A[i], [o1, n1] = A[i + 1]; if (o < o1) { const r = Math.max(1, (o1 - o0) / Math.max(1e-6, n1 - n0)); return n0 + (o - o0) / r; } } return A[A.length - 1][1]; }
// 每幕只有一个标题（这一幕的结论），在配音讲到结论时出现。tag 是这一幕要回答的问题，只留在代码里，不画在画面上；
// 顶部的系列名只在封面出现（见 sOpen）。src（出处与“示意”说明）同样只留在代码与 script.md 里
const SERIES = 'DSpark · DeepSeek V4 的 speculative decoding';
// 顶部的结构条：四个部件按一轮里的顺序排开，当前这一幕讲的部件为实色。tag（这一幕的问题）决定高亮哪些
const PARTS = [['并行 backbone', K.purple], ['Markov head', K.purple], ['confidence head', K.purple], ['调度器', K.blue]];
const PART_OF = { '提速来自哪里': [], '生成 5 个 draft token 的直接做法': [0], '能否一次生成全部': [0], '第 2 位能否看到第 1 位的结果': [1], '第 2 位缺少的信息': [1], 'transition bias 的规模': [1], '效果': [0, 1],
  '是否验证全部 draft token': [2, 3], '单个 token 的期望收益': [2], '验证前如何得到条件接受率': [2], '本步验证多少个': [3], '总结': [0, 1, 2, 3] };
function partBar(on) { let x = 96; const y = 44, h = 46, f = font(700, 24, HN);
  PARTS.forEach(([n, c], i) => { X.font = f; const w = X.measureText(n).width + 36, a = on.includes(i);
    if (a) R(x, y, w, h, c); else box(x + 1, y + 1, w - 2, h - 2, 2, K.lg);
    TX(n, x + w / 2, y + 31, f, a ? '#fff' : K.gray, 'center'); x += w;
    if (i < PARTS.length - 1) { arrow(x + 8, y + h / 2, x + 34, y + h / 2, K.lg, 3); x += 42; } }); }
function frame(tag, col, head, src, t0 = .15, th) {
  if (PART_OF[tag]) IN(t0, () => partBar(PART_OF[tag]), .5, 0);
  IN(th ?? t0 + .15, () => RICH(head, 96, 170, font(700, 60), K.ink));
}
// 底部两行：结论（深色）与补充（灰色）
const foot1 = (t0, s) => IN(t0, () => RICH(s, 96, 972, font(700, 38), K.ink));
const foot2 = (t0, s) => IN(t0, () => RICH(s, 96, 1030, font(600, 34), K.ink2));

// ---------- token 色块 ----------
// 状态：ctx 已确定（黑），guess 猜的（紫框空心），word 猜的且已填字（紫底），ok 接受（青 + ✓），bad 拒绝（红 + ✗），void 作废（灰斜纹），blue 送去验证
function chip(x, y, w, h, st, label, size = 34) {
  if (st === 'ctx') return cell(x, y, w, h, K.ink, label, size);
  if (st === 'blue') return cell(x, y, w, h, K.blue, label, size);
  if (st === 'word') return cell(x, y, w, h, K.purple, label, size);
  if (st === 'ok') return cell(x, y, w, h, K.teal, label == null ? '✓' : label + ' ✓', size);
  if (st === 'bad') return cell(x, y, w, h, K.red, label == null ? '✗' : label + ' ✗', size);
  if (st === 'void') { HR(x, y, w, h, K.lg, K.paper2); if (label) TX(label, x + w / 2, y + h / 2 + size * .36, font(700, size), K.gray, 'center'); return; }
  R(x, y, w, h, K.paper); box(x + 2.5, y + 2.5, w - 5, h - 5, 5, K.purple); if (label) TX(label, x + w / 2, y + h / 2 + size * .36, font(700, size), K.purple, 'center');
}

// ---------- 聊天窗口 ----------
// 同一个问题，按真实速度输出（每个字算一个 token）。n 为已输出的字数
const ASK = '能帮我查一下这个函数为什么慢吗？';
const REPLY = '当然可以。这个函数慢，主要是因为它在循环里反复查询数据库：每处理一条记录，就发一次请求，一千条记录就是一千次往返。可以先把所有 id 收集起来，用一条查询批量取回，再在内存里按 id 建一个字典，循环里直接查字典。这样数据库往返从一千次降到一次。另外，循环里每次都在拼接字符串，字符串不可变，每次拼接都会复制一遍，记录多了就是平方级的开销，改成先放进列表、最后 join 一次就好。还有一处排序写在了循环内部，其实只需要在循环结束后排一次。这三处改完，大概能快一个数量级。如果还想再快，可以用性能分析工具看一下热点，确认时间具体花在哪一行，再决定要不要加缓存或者改用并发请求。改完之后记得补一个测试，用一千条和十万条数据各跑一次，确认结果一致、耗时符合预期。需要的话，我可以直接把改好的代码写出来。';
function chatWin(x, y, w, h, name, tps, n, col, fs = 24) {
  R(x, y, w, h, '#fbf9f4'); box(x, y, w, h, 4, K.ink); R(x, y, w, 58, col);
  TX(name, x + 22, y + 40, font(700, 30, HN), '#fff'); TX(`${tps} tok/s`, x + w - 22, y + 40, font(700, 28, HN), '#fff', 'right');
  X.save(); X.beginPath(); X.rect(x, y + 58, w, h - 58); X.clip();
  const cpl = Math.floor((w - 44) / fs), lh = fs * 1.42, s = REPLY.slice(0, Math.min(REPLY.length, Math.floor(n)));
  const maxL = Math.floor((h - 78 - fs - 22 - 22 - fs - 16) / lh), up = Math.max(0, Math.floor(s.length / cpl) - maxL) * lh, y0 = y + 78 + fs + 22 + 22 + fs - up;
  X.font = font(600, fs); const aw = X.measureText(ASK).width + 32; R(x + w - 22 - aw, y + 78 - up, aw, fs + 22, K.paper2); TX(ASK, x + w - 22 - aw + 16, y + 78 + fs + 4 - up, font(600, fs), K.ink2);
  X.fillStyle = K.ink; X.textAlign = 'left';
  for (let l = 0; l * cpl < s.length; l++) X.fillText(s.slice(l * cpl, (l + 1) * cpl), x + 22, y0 + l * lh);
  if (n < REPLY.length && n > 0) { const k = s.length; R(x + 22 + (k % cpl) * fs + 2, y0 + Math.floor(k / cpl) * lh - fs + 2, 12, fs + 2, col); }
  X.restore();
}

// ---------- 一轮的时间条 ----------
// 紫色是猜（blocks 个小格，每格 bw 宽），蓝色是验证。返回条的右端
function roundBar(x, y, h, blocks, bw, vw, pDraft = 1, pVer = 1, label = true) {
  for (let i = 0; i < blocks; i++) { const p = clamp(pDraft * blocks - i); if (p > 0) R(x + i * bw + 1, y, (bw - 2) * p, h, K.purple); }
  const dx = x + blocks * bw; R(dx + 2, y, (vw - 2) * pVer, h, K.blue);
  if (label && pVer > .6) TX('验证：target 一次 forward', dx + 20, y + h / 2 + 11, font(700, 28), '#fff');
  return dx + vw;
}
