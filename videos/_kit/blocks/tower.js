// 系列母题“硬件塔”：显存 HBM → L2 → 共享内存 → 寄存器 → Tensor Core 五层薄板自下而上，SM 内三层虚线框住。
// 新建视频时复制到 assets/kit/tower.js，由 index.html 引入。用法（在幕的脚本里，建一次，之后只动画它返回的元素）：
//   const T = kitTower(document.getElementById("s03-tower"), { layers: [...可选，覆盖默认层...] });
//   T.layer.smem                      —— 某一层的 DOM 元素（可淡入、标橙）
//   T.route("s03-new", ["l2", "smem", "reg", "tc"], { color: "var(--use)" })  —— 沿层画一条路径，返回 <path>，
//       用 tl.fromTo(path, { strokeDashoffset: path.dataset.len }, { strokeDashoffset: 0 }) 逐段画出
//   T.at("smem")                      —— 某层右侧锚点 {x, y}（容器坐标），放数据方块、标签用
// 容器需要定宽定高（默认 520 × 600），布局全部按容器尺寸计算，确定性，无随机。
(function () {
  const DEFAULT = [
    { id: "tc", name: "Tensor Core", note: "每 SM 4 个", color: "var(--teal)" },
    { id: "reg", name: "寄存器", note: "每 SM 256 KB", color: "var(--purple)" },
    { id: "smem", name: "共享内存", note: "每 SM 228 KB", color: "var(--use)" },
    { id: "l2", name: "L2", note: "50 MB · 全卡共享", color: "var(--blue)" },
    { id: "hbm", name: "显存 HBM", note: "80 GB · 3.35 TB/s", color: "var(--hot)" },
  ];
  window.kitTower = function (el, opts = {}) {
    const layers = opts.layers || DEFAULT, inSM = opts.inSM ?? 3;
    const W = el.clientWidth || 520, H = el.clientHeight || 600;
    const n = layers.length, slabH = 64, gap = (H - n * slabH) / n, depth = 14;
    el.style.position = el.style.position || "absolute";
    const pos = {}, layer = {};
    layers.forEach((L, i) => {
      const y = i * (slabH + gap) + gap / 2 + (i >= inSM ? 18 : 0);
      const d = document.createElement("div");
      d.className = "kit-ly";
      d.id = `${el.id}-${L.id}`;
      d.style.cssText = `position:absolute;left:${depth}px;top:${y}px;width:${W - depth * 2 - 150}px;height:${slabH}px;background:var(--paper2);border:2px solid var(--line);box-shadow:${depth}px -${depth / 2}px 0 -2px var(--cell);display:flex;align-items:center;gap:12px;padding-left:12px;`;
      d.innerHTML = `<i style="width:8px;height:40px;background:${L.color};display:block"></i><div><div style="font-size:26px;font-weight:700;color:var(--ink)">${L.name}</div><div style="font-size:20px;font-weight:500;color:var(--ink2)">${L.note || ""}</div></div>`;
      el.appendChild(d);
      layer[L.id] = d;
      pos[L.id] = { x: W - depth - 150, y: y + slabH / 2, cx: (W - 150) / 2, top: y, bottom: y + slabH };
    });
    if (inSM > 0) {   // SM 框：虚线框住上面 inSM 层
      const top = pos[layers[0].id].top - 30, bottom = pos[layers[inSM - 1].id].bottom + 10;
      const b = document.createElement("div");
      b.id = `${el.id}-sm`;
      b.setAttribute("data-layout-allow-overflow", "");   // 标签故意放在框上沿外
      b.style.cssText = `position:absolute;left:0;top:${top}px;width:${W - 130}px;height:${bottom - top}px;border:2px dashed var(--gray);`;
      b.innerHTML = `<span style="position:absolute;left:8px;top:-30px;font-size:20px;font-weight:600;color:var(--note)">一个 SM（共 132 个）</span>`;
      el.appendChild(b);
    }
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible";
    el.appendChild(svg);
    const at = (id) => ({ x: pos[id].x, y: pos[id].y });
    // 路径沿右侧走廊上下：每到一层先水平伸进该层再回到走廊，表示“经过这一层”
    function route(id, ids, { color = "var(--use)", width = 5, lane = 0 } = {}) {
      const x = pos[ids[0]].x + 40 + lane * 18;
      let d = `M ${pos[ids[0]].x} ${pos[ids[0]].y} H ${x}`;
      for (const k of ids.slice(1)) d += ` V ${pos[k].y} H ${pos[k].x} H ${x}`;
      d = d.replace(/ H [\d.]+$/, "");
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.id = id; p.setAttribute("d", d); p.setAttribute("fill", "none");
      p.setAttribute("stroke", color); p.setAttribute("stroke-width", width); p.setAttribute("stroke-linejoin", "round");
      svg.appendChild(p);
      const len = p.getTotalLength ? p.getTotalLength() : 1000;
      p.dataset.len = len; p.style.strokeDasharray = len; p.style.strokeDashoffset = 0;
      return p;
    }
    return { layer, at, route, svg };
  };
})();
