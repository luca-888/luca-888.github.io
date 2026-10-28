// 全片骨架：存储层级图（地图）。显存 → L2 → shared memory → 寄存器 → Tensor Core 自下而上，外加寄存器 → 显存的写回。
// 宽度表示共用范围（不表示容量）：显存、L2 全芯片共用，最宽；shared memory 一个 SM 共用，在 SM 框里占一整条；
// 寄存器与 Tensor Core 每个分区各一份，画成 4 份并排。各层用语义色实心块（frame.md）。
// 各幕都用它重建同一张图，只加本幕的动作。
//
//   const M = HFMap(document.getElementById("s01-map"), { x: 460, y: 180 });
//   M.layer.smem          —— 一层（div）；寄存器、Tensor Core 是第 0 个分区（数据通道经过的那一份）
//   M.part.reg / part.tc  —— 4 个分区各自的寄存器 / Tensor Core（数组）
//   M.all(id)             —— 某层的全部元素（含 4 个分区）
//   M.link.l2             —— 进入某层的箭头（显存 → L2 记为 l2，L2 → shared memory 记为 smem……），SVG <path>，dataset.len 便于逐段画出
//   M.wb                  —— 写回箭头（寄存器 → 显存）
//   M.box                 —— SM 虚线框
//   M.at(id)              —— 数据通道上某层的中心点 {x, y}（地图坐标），放数据块用
//   M.el                  —— 整张图的外层 div：整体缩放、移动（如缩到左上角）都作用在它上面
// 动画辅助（都写进传入的时间线）：
//   HFMap.build(tl, M, t, ids, gap)  从下往上依次出现各层（先画进入它的箭头）
//   HFMap.lit(tl, M, ids, t)          点亮：加黑描边并略放大，其余不动
(function () {
  const W = 1000, H = 760;
  // 几何（地图坐标）：x、宽、y、高
  const G = {
    hbm: { x: 0, w: 1000, y: 630, h: 116 },
    l2: { x: 70, w: 860, y: 494, h: 92 },
    smem: { x: 170, w: 660, y: 326, h: 84 },
    reg: { y: 190, h: 64 },
    tc: { y: 36, h: 118 },
    box: { x: 140, w: 720, y: 0, h: 440 },
  };
  const PX = (k) => 182 + k * 162, PW = 148;   // 4 个分区
  const LANE = PX(0) + PW / 2;                  // 数据通道：经过第 0 个分区
  const ORDER = ["hbm", "l2", "smem", "reg", "tc"];
  const NS = "http://www.w3.org/2000/svg";
  const div = (p, css, html) => { const e = document.createElement("div"); e.style.cssText = css; if (html !== undefined) e.innerHTML = html; p.appendChild(e); return e; };
  const block = (p, x, y, w, h, bg, html, fs = 36) => div(p, `position:absolute;left:${x}px;top:${y}px;width:${w}px;height:${h}px;background:${bg};border-radius:4px;color:#fff;display:flex;align-items:center;justify-content:center;text-align:center;font-size:${fs}px;font-weight:800;line-height:1.15;white-space:nowrap;outline:0 solid var(--ink)`, html);

  window.HFMap = function (root, o = {}) {
    const el = div(root, `position:absolute;left:${o.x ?? 460}px;top:${o.y ?? 180}px;width:${W}px;height:${H}px;transform-origin:0 0`);
    el.setAttribute("data-layout-allow-overflow", "");
    const box = div(el, `position:absolute;left:${G.box.x}px;top:${G.box.y}px;width:${G.box.w}px;height:${G.box.h}px;border:3px dashed var(--gray);border-radius:8px`);
    const boxL = div(el, `position:absolute;left:${G.box.x + 4}px;top:${G.box.y - 48}px;font-size:32px;font-weight:800;color:var(--note);white-space:nowrap`, "一个 SM");
    const layer = {}, part = { reg: [], tc: [] };
    layer.hbm = block(el, G.hbm.x, G.hbm.y, G.hbm.w, G.hbm.h, "var(--hot)", "显存 <span style=\"font-size:28px;font-weight:600;margin-left:14px\">全芯片共用</span>", 40);
    layer.l2 = block(el, G.l2.x, G.l2.y, G.l2.w, G.l2.h, "var(--ink2)", "L2 缓存 <span style=\"font-size:28px;font-weight:600;margin-left:14px\">全芯片共用</span>", 38);
    layer.smem = block(el, G.smem.x, G.smem.y, G.smem.w, G.smem.h, "var(--use)", "shared memory <span style=\"font-size:28px;font-weight:600;margin-left:14px\">一个 SM 共用</span>", 34);
    for (let k = 0; k < 4; k++) {
      part.reg.push(block(el, PX(k), G.reg.y, PW, G.reg.h, "var(--ink2)", "<span>寄存器</span>", 30));
      part.tc.push(block(el, PX(k), G.tc.y, PW, G.tc.h, "var(--teal)", "<span>Tensor<br>Core</span>", 30));
    }
    layer.reg = part.reg[0]; layer.tc = part.tc[0];
    const partL = div(el, `position:absolute;left:${G.box.x + G.box.w - 220}px;top:${G.box.y - 44}px;width:220px;text-align:right;font-size:28px;font-weight:700;color:var(--note);white-space:nowrap`, "每个分区各一份");
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible";
    el.appendChild(svg);
    const path = (d, col, w) => {
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", d); p.setAttribute("fill", "none"); p.setAttribute("stroke", col); p.setAttribute("stroke-width", w);
      p.setAttribute("stroke-linecap", "butt"); p.setAttribute("stroke-linejoin", "round");
      svg.appendChild(p);
      const len = p.getTotalLength(); p.dataset.len = len; p.style.strokeDasharray = len;
      return p;
    };
    const head = (x, y, dir, col) => {
      const p = document.createElementNS(NS, "polygon"), s = 13;
      const pts = dir === "up" ? [[x, y - s], [x - s, y + 4], [x + s, y + 4]] : [[x, y + s], [x - s, y - 4], [x + s, y - 4]];
      p.setAttribute("points", pts.map((q) => q.join(",")).join(" ")); p.setAttribute("fill", col);
      svg.appendChild(p); return p;
    };
    // 进入某层的箭头：沿数据通道，从下一层的上沿到这一层的下沿
    const link = {}, linkHead = {};
    for (let i = 1; i < ORDER.length; i++) {
      const a = G[ORDER[i - 1]], b = G[ORDER[i]], y0 = a.y - 4, y1 = b.y + b.h + 14;
      link[ORDER[i]] = path(`M ${LANE} ${y0} L ${LANE} ${y1}`, "var(--ink)", 6);
      linkHead[ORDER[i]] = head(LANE, y1, "up", "var(--ink)");
    }
    // 写回：最右一个分区的寄存器 → 出 SM 框 → 沿图右侧下行 → 进入显存
    const rx = PX(3) + PW, wx = 966;
    const wb = path(`M ${rx + 6} ${G.reg.y + G.reg.h / 2} H ${wx} V ${G.hbm.y - 18}`, "var(--purple)", 6);
    const wbHead = head(wx, G.hbm.y - 14, "down", "var(--purple)");
    const wbL = div(el, `position:absolute;left:${wx - 92}px;top:${(G.reg.y + G.hbm.y) / 2 - 30}px;font-size:32px;font-weight:800;color:var(--purple-t);white-space:nowrap`, "写回");
    // wgmma 起：shared memory 直接进 Tensor Core（绕过寄存器，从第 0 个分区左侧上行）
    const bx = G.box.x + 20;
    const direct = path(`M ${PX(0) + 20} ${G.smem.y - 4} L ${bx} ${G.smem.y - 30} L ${bx} ${G.tc.y + G.tc.h / 2} L ${PX(0) - 16} ${G.tc.y + G.tc.h / 2}`, "var(--teal)", 7);
    const directHead = (() => { const p = document.createElementNS(NS, "polygon"), x = PX(0) - 2, y = G.tc.y + G.tc.h / 2, s = 13; p.setAttribute("points", `${x},${y} ${x - 17},${y - s} ${x - 17},${y + s}`); p.setAttribute("fill", "var(--teal)"); svg.appendChild(p); return p; })();
    const at = (id) => ({ x: LANE, y: G[id].y + G[id].h / 2 });
    const all = (id) => (part[id] ? part[id] : [layer[id]]);
    gsap.set([box, boxL, partL, wbL, layer.hbm, layer.l2, layer.smem, ...part.reg, ...part.tc, ...Object.values(linkHead), wbHead, directHead], { opacity: 0 });
    [...Object.values(link), wb, direct].forEach((p) => gsap.set(p, { strokeDashoffset: +p.dataset.len }));
    const subs = [...el.querySelectorAll("span"), boxL, wbL];   // 小地图时藏起的副标签与分区里的小字
    // 小地图时的大字：显存、L2、shared memory 的层名放大；寄存器、Tensor Core 各用一个横跨 4 格的标签（缩小后仍不小于 28 px）
    const rowW = PX(3) + PW - PX(0);
    const rowL = ["reg", "tc"].map((id) => div(el, `position:absolute;left:${PX(0)}px;top:${G[id].y}px;width:${rowW}px;height:${G[id].h}px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:68px;font-weight:800;white-space:nowrap;pointer-events:none`, id === "reg" ? "寄存器" : "Tensor Core"));
    gsap.set(rowL, { opacity: 0 });
    const names = [layer.hbm, layer.l2, layer.smem], nameFs = [40, 38, 34];
    return { el, box, boxL, partL, layer, part, all, link, linkHead, wb, wbHead, wbL, direct, directHead, svg, at, subs, rowL, names, nameFs, W, H, G, PX, PW, LANE };
  };

  HFMap.ORDER = ORDER;
  HFMap.build = function (tl, M, t, ids = ORDER, gap = 0.9) {
    ids.forEach((id, i) => {
      const s = t + i * gap;
      if (M.link[id]) {
        tl.fromTo(M.link[id], { strokeDashoffset: +M.link[id].dataset.len }, { strokeDashoffset: 0, duration: 0.3, ease: "power1.inOut" }, s);
        tl.fromTo(M.linkHead[id], { opacity: 0 }, { opacity: 1, duration: 0.1 }, s + 0.25);
      }
      M.all(id).forEach((e, k) => tl.fromTo(e, { opacity: 0, y: 24 }, { opacity: 1, y: 0, duration: 0.55, ease: "expo.out" }, s + (M.link[id] ? 0.25 : 0) + k * 0.08));
    });
    return tl;
  };
  HFMap.lit = function (tl, M, ids, t, on = true) {
    ids.forEach((id) => M.all(id).forEach((e) => tl.fromTo(e, { outlineWidth: on ? 0 : 5, scale: on ? 1 : 1.03 }, { outlineWidth: on ? 5 : 0, scale: on ? 1.03 : 1, duration: 0.3, ease: "power2.out" }, t)));
    return tl;
  };

  // —— 静态状态：幕开头直接把地图摆成“前面各步做完”的样子（不播动画）——
  // opts.direct：wgmma 之后（s04 起），shared memory → 寄存器 → Tensor Core 两段变淡，直达箭头可见
  // opts.mini：缩到左上角作小地图（x, y, scale 可覆盖）
  HFMap.MINI = { x: 40, y: 200, scale: 0.42 };
  HFMap.state = function (M, opts = {}) {
    gsap.set([M.box, M.boxL, M.partL, M.wbL, M.layer.hbm, M.layer.l2, M.layer.smem, ...M.part.reg, ...M.part.tc, ...Object.values(M.linkHead), M.wbHead], { opacity: 1 });
    [...Object.values(M.link), M.wb].forEach((p) => gsap.set(p, { strokeDashoffset: 0 }));
    if (opts.direct) {
      gsap.set([M.link.reg, M.link.tc, M.linkHead.reg, M.linkHead.tc], { opacity: 0.18 });
      gsap.set(M.direct, { strokeDashoffset: 0 }); gsap.set(M.directHead, { opacity: 1 });
    }
    if (opts.mini) { const m = { ...HFMap.MINI, ...(opts.mini === true ? {} : opts.mini) }; gsap.set(M.el, { x: m.x - parseFloat(M.el.style.left), y: m.y - parseFloat(M.el.style.top), scale: m.scale }); gsap.set([M.partL, ...M.subs], { opacity: 0 }); gsap.set(M.rowL, { opacity: 1 }); gsap.set(M.names, { fontSize: 68 }); }
    return M;
  };
  // 整张图缩到左上角（或从左上角放大回原处）
  HFMap.toMini = function (tl, M, t, d = 0.9, back = false, rowOp = [1, 1]) {
    const m = HFMap.MINI, dx = m.x - parseFloat(M.el.style.left), dy = m.y - parseFloat(M.el.style.top);
    const A = { x: 0, y: 0, scale: 1 }, Z = { x: dx, y: dy, scale: m.scale };
    tl.fromTo(M.el, back ? Z : A, { ...(back ? A : Z), duration: d, ease: "power2.inOut" }, t);
    tl.fromTo([M.partL, ...M.subs], { opacity: back ? 0 : 1 }, { opacity: back ? 1 : 0, duration: 0.3 }, t);
    M.rowL.forEach((e, i) => tl.fromTo(e, { opacity: back ? rowOp[i] : 0 }, { opacity: back ? 0 : rowOp[i], duration: 0.3 }, t + (back ? 0 : d * 0.5)));
    M.names.forEach((e, i) => tl.fromTo(e, { fontSize: back ? 68 : M.nameFs[i] }, { fontSize: back ? M.nameFs[i] : 68, duration: d, ease: "power2.inOut" }, t));
    return tl;
  };
  // 点亮一段：本段相关的层保持原样并加描边，箭头变粗、变 teal；其余层与箭头变淡。seg：
  //   "hbm-l2"、"l2-smem"、"smem-reg"、"smem-tc"（直达）、"wb"（写回）、"sm"（整个 SM 内部）
  const SEG = {
    "hbm-l2": { layers: ["hbm", "l2"], links: ["l2"] },
    "l2-smem": { layers: ["l2", "smem"], links: ["smem"] },
    "smem-reg": { layers: ["smem", "reg"], links: ["reg"] },
    "smem-tc": { layers: ["smem", "tc"], links: [], direct: true },
    "wb": { layers: ["reg", "hbm"], links: [], wb: true },
    "sm": { layers: ["smem", "reg", "tc"], links: ["reg", "tc"] },
  };
  HFMap.focus = function (tl, M, seg, t, col = "#c4561d") {
    const S = SEG[seg], others = ORDER.filter((id) => !S.layers.includes(id));
    others.forEach((id) => M.all(id).forEach((e) => tl.fromTo(e, { opacity: 1 }, { opacity: 0.28, duration: 0.4 }, t)));
    const rows = { reg: M.rowL[0], tc: M.rowL[1] };
    others.forEach((id) => { if (rows[id] && parseFloat(gsap.getProperty(rows[id], "opacity")) > 0) tl.fromTo(rows[id], { opacity: 1 }, { opacity: 0, duration: 0.3 }, t); });
    S.layers.forEach((id) => M.all(id).forEach((e) => tl.fromTo(e, { outlineWidth: 0 }, { outlineWidth: 5, duration: 0.3 }, t)));
    const lines = [...S.links.map((k) => M.link[k]), ...(S.direct ? [M.direct] : []), ...(S.wb ? [M.wb] : [])];
    const heads = [...S.links.map((k) => M.linkHead[k]), ...(S.direct ? [M.directHead] : []), ...(S.wb ? [M.wbHead] : [])];
    lines.forEach((p) => tl.fromTo(p, { strokeWidth: 6 }, { strokeWidth: 12, stroke: col, duration: 0.3 }, t));
    heads.forEach((h) => tl.fromTo(h, { scale: 1 }, { scale: 1.6, fill: col, transformOrigin: "50% 50%", duration: 0.3 }, t));
    return tl;
  };
})();
