// 全片常驻的页眉与速度条，以及各幕通用的小工具。各幕：const U = HFKit("s03", tl);
//   U.mk(parent, css, html)                  建 div
//   U.appear / U.leave / U.ft                常用补间（写进本幕时间线）
//   HFHeader(root, step, title)              页眉：左上角“第 N 步”色块 + 标题（优化方式）；.show(tl, t)、.instant()
//   HFSpeed(root, v)                         速度条：右上角，cuBLAS = 100%；.show(tl, t)、.to(tl, v, t)、.shake(tl, t)、.instant()
// 速度条各步的取值（8192 阶，cuBLAS = 100%，见 EVIDENCE.md / CODE.md）：
//   起点 wmma 0.119；swizzle 0.26；cp.async + pipeline 0.395；wgmma 0.619；warp specialization 0.614（8192 阶不变）；
//   tile 分组 0.946；cluster 多播 0.968；TMA store 0.995
(function () {
  window.HFSPEED = { s01: 0.119, s02: 0.26, s03: 0.395, s04: 0.619, s05: 0.614, s06: 0.946, s07: 0.968, s08: 0.995 };
  const div = (p, css, html) => { const e = document.createElement("div"); e.style.cssText = css; if (html !== undefined) e.innerHTML = html; p.appendChild(e); return e; };

  window.HFKit = function (id, tl) {
    const ft = (el, a, b, t) => tl.fromTo(el, a, b, t);
    return {
      $: (k) => document.getElementById(id + "-" + k),
      mk: div, ft,
      ARRIVE: "expo.out", CAM: "power2.inOut", OUT: "power3.in",
      appear: (el, t, d = 0.6, y = 20) => ft(el, { opacity: 0, y }, { opacity: 1, y: 0, duration: d, ease: "expo.out" }, t),
      fade: (el, t, d = 0.4, to = 1) => ft(el, { opacity: 0 }, { opacity: to, duration: d, ease: "power1.out" }, t),
      leave: (el, t, d = 0.3, from = 1) => ft(el, { opacity: from }, { opacity: 0, duration: d, ease: "power3.in" }, t),
      pop: (el, t, d = 0.5) => ft(el, { opacity: 0, scale: 0.8 }, { opacity: 1, scale: 1, duration: d, ease: "back.out(1.6)" }, t),
      draw: (p, t, d = 0.5) => ft(p, { strokeDashoffset: +p.dataset.len }, { strokeDashoffset: 0, duration: d, ease: "power1.inOut" }, t),
      // 数据块沿一组折线点移动（地图坐标或画面坐标），结果确定
      travel: (el, pts, t, d, ease = "none") => {
        const seg = [0];
        for (let i = 1; i < pts.length; i++) seg.push(seg[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
        const L = seg[seg.length - 1], kf = {};
        pts.forEach((p, i) => { kf[((seg[i] / L) * 100).toFixed(2) + "%"] = { x: p[0], y: p[1] }; });
        tl.set(el, { xPercent: -50, yPercent: -50, x: pts[0][0], y: pts[0][1] }, t - 0.01);
        tl.to(el, { keyframes: { ...kf, easeEach: "none" }, duration: d, ease }, t);
      },
      // SVG 路径（画面坐标），带 dataset.len，初始藏起
      path: (svg, d, col, w = 6, dash) => {
        const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
        p.setAttribute("d", d); p.setAttribute("fill", "none"); p.setAttribute("stroke", col); p.setAttribute("stroke-width", w);
        p.setAttribute("stroke-linecap", "butt"); p.setAttribute("stroke-linejoin", "round");
        svg.appendChild(p);
        const len = p.getTotalLength(); p.dataset.len = len;
        if (dash) p.setAttribute("stroke-dasharray", dash); else { p.style.strokeDasharray = len; p.style.strokeDashoffset = len; }
        return p;
      },
      svg: (parent) => {
        const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        s.setAttribute("width", 1920); s.setAttribute("height", 1080);
        s.style.cssText = "position:absolute;left:0;top:0;overflow:visible";
        parent.appendChild(s); return s;
      },
    };
  };

  // —— 页眉：左上角“第 N 步”色块 + 标题 ——
  window.HFHeader = function (root, step, title) {
    const h = div(root, "position:absolute;left:64px;top:44px;display:flex;align-items:center;gap:20px;white-space:nowrap");
    const chip = div(h, "padding:6px 16px 8px;background:var(--tealD);color:#fff;font-size:34px;font-weight:800;border-radius:3px", step);
    const t = div(h, "font-size:44px;font-weight:800;color:var(--ink)", title);
    gsap.set([chip, t], { opacity: 0 });
    return {
      el: h,
      show(tl, at) { tl.fromTo(chip, { opacity: 0, x: -16 }, { opacity: 1, x: 0, duration: 0.4, ease: "expo.out" }, at); tl.fromTo(t, { opacity: 0, filter: "blur(8px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.3, ease: "power1.out" }, at + 0.1); return this; },
      instant() { gsap.set([chip, t], { opacity: 1 }); return this; },
    };
  };

  // —— 速度条：右上角；满格 = cuBLAS ——
  window.HFSpeed = function (root, v = 0.119) {
    const W = 520;
    const s = div(root, "position:absolute;right:64px;top:40px;width:" + (W + 150) + "px;height:110px");
    const lbl = div(s, "position:absolute;left:0;top:0;font-size:30px;font-weight:700;color:var(--ink2);white-space:nowrap", "速度");
    const track = div(s, `position:absolute;left:0;top:46px;width:${W}px;height:30px;background:var(--paper2);border-radius:3px`);
    const fill = div(track, `position:absolute;left:0;top:0;width:${W}px;height:30px;background:var(--teal);border-radius:3px;transform-origin:0 50%`);
    const full = div(s, `position:absolute;left:${W - 2}px;top:36px;width:4px;height:50px;background:var(--ink)`);
    const fullL = div(s, `position:absolute;left:${W + 14}px;top:44px;font-size:30px;font-weight:800;color:var(--ink);white-space:nowrap`, "cuBLAS");
    gsap.set(fill, { scaleX: v }); gsap.set(s, { opacity: 0 });
    let cur = v;
    return {
      el: s, fill, full, fullL,
      show(tl, at, d = 0.5) { tl.fromTo(s, { opacity: 0 }, { opacity: 1, duration: d }, at); return this; },
      instant() { gsap.set(s, { opacity: 1 }); return this; },
      to(tl, nv, at, d = 1.0) { tl.fromTo(fill, { scaleX: cur }, { scaleX: nv, duration: d, ease: "power2.inOut" }, at); cur = nv; return this; },
      // 末段放大镜：前进很小时（如 0.946 → 0.968），在速度条下方把满格附近 [lo, 1] 放大显示，前进看得出来
      zoomTo(tl, nv, at, lo = 0.9) {
        const from = cur;
        this.to(tl, nv, at, 1.0);
        const z = div(s, `position:absolute;left:${W * lo}px;top:96px;width:${W * (1 - lo)}px;height:2px;background:var(--ink2)`);
        const box = div(root, `position:absolute;right:${64 + 150}px;top:128px;width:${W}px;height:34px;background:var(--paper2);border:3px solid var(--ink2);border-radius:3px`);
        const zf = div(box, `position:absolute;left:0;top:0;width:100%;height:100%;background:var(--teal);transform-origin:0 50%`);
        const zl = div(root, `position:absolute;right:${64 + 150 + W + 16}px;top:126px;font-size:30px;font-weight:700;color:var(--ink2);white-space:nowrap`, "放大末端");
        const zt = div(box, "position:absolute;right:-4px;top:-8px;width:4px;height:46px;background:var(--ink)");
        const k = (v) => Math.max(0, (v - lo) / (1 - lo));
        gsap.set([z, box, zl], { opacity: 0 }); gsap.set(zf, { scaleX: k(from) });
        tl.fromTo([z, box, zl], { opacity: 0 }, { opacity: 1, duration: 0.3 }, at - 0.4);
        tl.fromTo(zf, { scaleX: k(from) }, { scaleX: k(nv), duration: 1.0, ease: "power2.inOut" }, at);
        tl.fromTo([z, box, zl], { opacity: 1 }, { opacity: 0, duration: 0.35 }, at + 2.6);
        return this;
      },
      shake(tl, at) { tl.fromTo(fill, { x: 0 }, { keyframes: [{ x: 6, duration: 0.08 }, { x: -5, duration: 0.08 }, { x: 3, duration: 0.08 }, { x: 0, duration: 0.08 }] }, at); return this; },
      mark(tl, at) { tl.fromTo([full, fullL], { scale: 1 }, { scale: 1.25, duration: 0.25, ease: "power2.out", transformOrigin: "0 50%" }, at); tl.fromTo(fullL, { color: "#141414" }, { color: "#0b7d70", duration: 0.25 }, at); tl.fromTo([full, fullL], { scale: 1.25 }, { scale: 1, duration: 0.3, transformOrigin: "0 50%" }, at + 0.4); return this; },
    };
  };
})();
