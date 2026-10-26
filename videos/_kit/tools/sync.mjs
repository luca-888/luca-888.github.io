// node videos/_kit/tools/sync.mjs <视频目录>
// 按 script.json（文案与节拍）和 assets/vo/vo.json（配音实测时长，可缺）重写 HyperFrames 项目里由文案决定的部分：
//   index.html 的 kit:hosts 区域（各幕宿主、字幕宿主、逐句配音 <audio>、总时长）
//   compositions/captions.html（整份生成，字幕按标点切成短句）
//   compositions/<幕>.html 的 kit:header 区域（页眉标记 + BEAT / DUR 常量 + 进度条动画）；幕文件不存在时从模板新建
// 机制画面写在幕文件 kit 区域之外，用 B.<拍的 id>（或 BEAT[i]）定时，改文案或重录配音后重跑本脚本即可整体重排。
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const KIT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dir = resolve(process.argv[2] || '.');
const S = JSON.parse(readFileSync(join(dir, 'script.json'), 'utf8'));
const voFile = join(dir, 'assets/vo/vo.json');
const VO = existsSync(voFile) ? JSON.parse(readFileSync(voFile, 'utf8')).lines : {};
const cfg = { gap: 0.5, minBeat: 2.6, hold: 2.2, tail: 1.0, lead: 0.5, ...(S.timing || {}) };

// 未录音时按字数估计朗读时长（与旧版 script.js 的 SPK 相同，语速 1.15）
const est = (x) => (x.match(/[一-鿿]/g) || []).length * 0.21 + (x.match(/[A-Za-z0-9][A-Za-z0-9._\-]*/g) || []).length * 0.32 + (x.match(/[，；：、]/g) || []).length * 0.15;
const r2 = (x) => Math.round(x * 100) / 100;
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
// 标题里 {…} 是强调词，{b:…} 等指定语义色（见 brand.css 的 .acc-*）
const rich = (s) => esc(s).replace(/\{(?:([bpyogr]):)?([^}]*)\}/g, (_, c, t) => `<span class="${c ? 'acc-' + c : 'acc'}">${t}</span>`);

// —— 排节拍 ——
let t0 = 0, missing = 0;
const scenes = S.scenes.map((sc) => {
  let t = cfg.lead;
  const beats = sc.beats.map((b) => {
    const rec = b.vo ? VO[b.vo] : null;
    if (b.vo && !rec) missing++;
    const say = b.vo ? (rec ? rec.dur : est(b.vo) + 0.4) : 0;
    const start = b.t ?? t;
    t = start + (b.vo ? Math.max(cfg.minBeat, say + cfg.gap) : (b.hold ?? cfg.hold)) + (b.extra || 0);
    return { ...b, start: r2(start), say: r2(say), rec };
  });
  const dur = r2(sc.dur ?? t - cfg.gap + cfg.tail);
  const out = { ...sc, beats, dur, abs: r2(t0) };
  t0 += dur;
  return out;
});
const total = r2(t0);

// —— 区域替换 ——
function region(text, name, body, file) {
  const re = new RegExp(`(<!-- kit:${name} -->|/\\* kit:${name} \\*/)[\\s\\S]*?(<!-- /kit:${name} -->|/\\* /kit:${name} \\*/)`);
  if (!re.test(text)) throw new Error(`${file} 里没有 kit:${name} 区域`);
  return text.replace(re, (_, a, z) => `${a}\n${body}\n${z}`);
}

// —— index.html ——
const indexPath = join(dir, 'index.html');
let index = readFileSync(indexPath, 'utf8');
const hosts = [];
for (const sc of scenes)
  hosts.push(`      <div id="${sc.id}" data-composition-id="${sc.id}" data-composition-src="compositions/${sc.id}.html" data-start="${sc.abs}" data-duration="${sc.dur}" data-track-index="1" data-track-kind="graphics" data-width="1920" data-height="1080"></div>`);
hosts.push(`      <div id="captions" data-composition-id="captions" data-composition-src="compositions/captions.html" data-start="0" data-duration="${total}" data-track-index="2" data-track-kind="captions" data-width="1920" data-height="1080"></div>`);
for (const sc of scenes)
  sc.beats.forEach((b, i) => {
    if (!b.rec) return;
    const head = b.rec.head || 0;   // 录音开头的静音：音频提前这么多开始，人声正好落在节拍上
    hosts.push(`      <audio id="vo-${sc.id}-${i}" class="clip" src="${b.rec.file}" data-start="${r2(sc.abs + b.start - head)}" data-duration="${r2(b.rec.len ?? b.rec.dur + head)}" data-track-index="3" data-volume="1"></audio>`);
  });
index = region(index, 'hosts', hosts.join('\n'), 'index.html');
index = index.replace(/(id="root"[^>]*?data-duration=")[\d.]+(")/, `$1${total}$2`);
writeFileSync(indexPath, index);

// —— 字幕：按句号、分号切；过长的再按逗号切；时长按估计的朗读量分配 ——
function pieces(text, dur) {
  let parts = text.split(/(?<=[。；！？])/).map((s) => s.trim()).filter(Boolean);
  parts = parts.flatMap((p) => {
    if (p.length <= 26) return [p];
    const out = []; let cur = '';
    for (const seg of p.split(/(?<=[，：])/)) { if (cur && (cur + seg).length > 26) { out.push(cur); cur = seg; } else cur += seg; }
    if (cur) out.push(cur);
    return out;
  });
  const w = parts.map((p) => Math.max(est(p), 0.3)), sw = w.reduce((a, b) => a + b, 0);
  let t = 0;
  return parts.map((p, i) => { const d = (dur * w[i]) / sw; const o = { text: p.replace(/[。；，：]$/, ''), t: r2(t), d: r2(d) }; t += d; return o; });
}
const caps = [];
for (const sc of scenes)
  sc.beats.forEach((b, i) => {
    if (!b.vo) return;
    const ps = pieces(b.vo, b.say);
    const next = sc.beats.slice(i + 1).find((x) => x.vo);
    const until = next ? next.start : sc.dur;
    ps.forEach((p, j) => {
      const start = sc.abs + b.start + p.t;
      const end = j === ps.length - 1 ? sc.abs + Math.max(b.start + b.say + 0.3, Math.min(until, b.start + b.say + 1.2)) : start + p.d;
      caps.push(`    <div id="cap-${sc.id}-${i}-${j}" class="clip kit-sub" data-start="${r2(start)}" data-duration="${r2(end - start)}" data-track-index="1"><span>${esc(p.text)}</span></div>`);
    });
  });
writeFileSync(join(dir, 'compositions/captions.html'), `<template>
  <style>
    #root { position: absolute; inset: 0; }
  </style>
  <!-- 由 videos/_kit/tools/sync.mjs 按 script.json 生成，不要手改：改 script.json 的配音稿后重跑 sync -->
  <div id="root" data-composition-id="captions" data-width="1920" data-height="1080" data-layout-allow-caption-zone>
${caps.join('\n')}
  </div>
  <script>
    window.__timelines["captions"] = gsap.timeline({ paused: true }).set({}, {}, ${total});
  </script>
</template>
`);

// —— 每幕的页眉区域 ——
const tpl = readFileSync(join(KIT, 'template/compositions/scene.html'), 'utf8');
mkdirSync(join(dir, 'compositions'), { recursive: true });
for (const sc of scenes) {
  const f = join(dir, `compositions/${sc.id}.html`);
  let html = existsSync(f) ? readFileSync(f, 'utf8') : tpl.replaceAll('__S__', sc.id);
  const m = sc.meter;
  const meter = m ? `
      <div id="${sc.id}-meter" class="kit-meter">
        <div class="lbl">${esc(m.label)}</div>
        <div class="track"><div id="${sc.id}-fill" class="fill"></div><div id="${sc.id}-num" class="num"></div></div>
        <div class="tick"></div>
        <div class="ref">${esc(m.refLabel || '')} ${esc(m.ref)}</div>
        ${m.note ? `<div id="${sc.id}-mnote" class="note">${esc(m.note)}</div>` : ''}
      </div>` : '';
  const markup = `    <div id="${sc.id}-hd" class="clip kit-hd" data-start="0" data-duration="${sc.dur}" data-track-index="0">
      <div class="kit-series">${esc(S.series)}</div>
      ${sc.chip ? `<div class="kit-chip">${esc(sc.chip)}</div>` : ''}
      ${sc.title ? `<div class="kit-title">${rich(sc.title)}</div>` : ''}
      ${sc.source ? `<div class="kit-src">${esc(sc.source)}</div>` : ''}${meter}
    </div>`;
  const at = m ? (typeof m.at === 'string' ? `B[${JSON.stringify(m.at)}]` : Number.isInteger(m.at) ? `BEAT[${m.at}]` : (m.at ?? 0)) : 0;
  const named = Object.fromEntries(sc.beats.filter((b) => b.id).map((b) => [b.id, b.start]));
  const js = `    const B = ${JSON.stringify(named)};   // 有 id 的拍的起点（幕内秒）：动画优先用 B.名字，插拍、删拍都不会错位
    const BEAT = ${JSON.stringify(sc.beats.map((b) => b.start))};   // 第 i 拍的起点，按序号
    const DUR = ${sc.dur};
    const tl = gsap.timeline({ paused: true });
    tl.set({}, {}, DUR);${m ? `
    {
      const from = ${m.from ?? m.to}, to = ${m.to}, ref = ${m.ref}, W = 508;
      const fill = document.getElementById("${sc.id}-fill"), num = document.getElementById("${sc.id}-num");
      fill.style.width = W + "px";
      const show = (v) => { num.textContent = String(Math.round(v)); num.style.width = Math.max((v / ref) * W, 40) + "px"; };
      const st = { v: from }; show(from);
      tl.fromTo(fill, { scaleX: from / ref }, { scaleX: to / ref, duration: 1.2, ease: "power2.inOut" }, ${at});
      tl.fromTo(st, { v: from }, { v: to, duration: 1.2, ease: "power2.inOut", onUpdate: () => show(st.v) }, ${at});${m.note ? `
      tl.fromTo("#${sc.id}-mnote", { opacity: 0 }, { opacity: 1, duration: 0.3 }, ${at} + 1.1);` : ''}
    }` : ''}`;
  html = region(html, 'header', markup, `${sc.id}.html`);
  html = region(html, 'beats', js, `${sc.id}.html`);
  // 标了 data-kit-full 的部件贯穿整幕：时长跟着幕长走
  html = html.replace(/<[^>]*\bdata-kit-full\b[^>]*>/g, (tag) => tag.replace(/data-duration="[\d.]+"/, `data-duration="${sc.dur}"`));
  writeFileSync(f, html);
}

console.log(`${scenes.length} 幕，共 ${total} s；字幕 ${caps.length} 条` + (missing ? `；${missing} 句未录音，按字数估计时长排节拍` : '；全部按配音实测时长排节拍'));
for (const sc of scenes) console.log(`  ${sc.id}  ${sc.abs.toFixed(1)}–${(sc.abs + sc.dur).toFixed(1)} s  节拍 ${sc.beats.map((b) => b.start).join(' ')}`);
