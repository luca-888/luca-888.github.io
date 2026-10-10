// 分镜审阅页与逐幕预览视频，各片的 render.mjs 共用（页面版式来自 tensor-core/render.mjs）。
//   const kit = reviewKit({ page, title, fps, scenes });   在影片目录下运行，page 已打开 film.html（1920 × 1080 视口）
//   await kit.clips({ only, force })   每幕一段带配音与字幕的视频 build/clips/sNN.mp4（1280 宽），没改过的幕跳过；每渲染完一幕更新一次分镜页
//   await kit.storyboard()             build/storyboard.html：左边目录，逐幕列出视频（没有就放关键帧）与配音稿
// scenes[i]：{ tag, chap, head, srcNote, at, dur, key, beats: [{ s, e, text, file, head }] }
//   at / dur 为这一幕在全片里的起点与时长；beats 的 s / e 为幕内时间，file 为这一句的录音（没有则留空），head 为录音开头的静音；
//   key 为这一幕的内容指纹，变了才重渲；hold 为关键帧距幕尾的秒数（默认 0.01，幕尾有淡出时加大）。页面需提供 window.renderAt(t, fps, samples)。
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

export function reviewKit({ page, title, fps, scenes }) {
const clip = { x: 0, y: 0, width: 1920, height: 1080 };
const pad = (i) => String(i + 1).padStart(2, '0');
const esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const clipFile = (i) => `build/clips/s${pad(i)}.mp4`;
const fresh = (i) => existsSync(clipFile(i)) && existsSync(clipFile(i) + '.key') && readFileSync(clipFile(i) + '.key', 'utf8') === scenes[i].key;

// 配音稿的标记：数字加粗，英文术语换色，校对时一眼能看到要核对的地方
const mark = (t) => esc(t).replace(/(\d+(?:\.\d+)?(?:\/\d+)?(?: × \d+)*%?)|([A-Za-z][A-Za-z0-9_.\-]*(?: [A-Za-z][A-Za-z0-9_.\-]*)*)/g, (m, num, en) => (num ? `<b>${num}</b>` : `<span class="en">${en}</span>`));
const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
// 页眉里的 {r:…} {p:…} {t:…} 沿用画面上的强调色
const headHtml = (h) => esc(h).replace(/\{(\w):([^}]*)\}/g, '<span class="c-$1">$2</span>');
// 分镜页：左边是各幕的目录，右边逐幕列出视频（没有就放关键帧）与配音稿；宽屏时视频与配音稿并排
async function storyboard() {
  mkdirSync('build/posters', { recursive: true });
  let nav = '', body = '';
  const anyVoice = scenes.some((s) => s.beats.some((b) => b.file));   // 无配音的片子不提示“配音待录”
  for (let i = 0; i < scenes.length; i++) {
    const s = scenes[i], n = s.beats.length, voiced = s.beats.filter((b) => b.file).length;
    await page.evaluate((t) => window.renderAt(t, 60, 1), s.at + s.dur - (s.hold ?? 0.01));
    await page.screenshot({ path: `build/posters/s${pad(i)}.jpg`, clip, type: 'jpeg', quality: 80 });
    const img = `posters/s${pad(i)}.jpg?v=${s.key}`, title = s.head ? headHtml(s.head) : esc(s.tag);
    const media = fresh(i)
      ? `<video controls preload="none" poster="${img}" src="clips/s${pad(i)}.mp4?v=${s.key}"></video>`
      : `<img src="${img}" alt="第 ${i + 1} 幕">`;
    const chips = (fresh(i) ? '' : `<span class="chip warn">${existsSync(clipFile(i)) ? '画面改过，视频待重渲' : '视频待渲染'}</span>`)
      + (voiced === n || !anyVoice ? '' : `<span class="chip warn">${voiced ? `配音 ${voiced}/${n} 句` : '配音待录'}</span>`);
    nav += `<li><a href="#s${i + 1}"><img src="${img}" alt="" loading="lazy"><span class="nt"><b>${i + 1}</b>${esc(s.tag)}</span><span class="nd">${clock(s.at)}${fresh(i) && (voiced === n || !anyVoice) ? '' : ' <i></i>'}</span></a></li>`;
    body += `<section id="s${i + 1}"><header><span class="num">${pad(i)}</span><div class="ttl">${s.head ? `<p class="tag">${esc(s.chap ? s.chap + ' · ' + s.tag : s.tag)}</p>` : ''}<h2>${title}</h2></div>`
      + `<p class="meta">${clock(s.at)} – ${clock(s.at + s.dur)}<span>${s.dur.toFixed(1)} s · ${n} 句</span>${chips}</p></header>`
      + `<div class="body"><div class="media">${media}</div><ol class="script">`
      + s.beats.map((b, j) => `<li data-s="${b.s.toFixed(2)}" data-e="${(j + 1 < n ? s.beats[j + 1].s : s.dur).toFixed(2)}"><span class="tc">${clock(b.s)}</span><span class="tx">${mark(b.text)}${b.sub ? `<em>字幕　${esc(b.sub)}</em>` : ''}</span><span class="len">${(b.e - b.s).toFixed(1)} s</span></li>`).join('')
      + `</ol></div>${s.srcNote ? `<p class="src">${esc(s.srcNote)}</p>` : ''}</section>\n`;
  }
  const total = scenes.reduce((a, s) => a + s.dur, 0), done = scenes.filter((_, i) => fresh(i)).length;
  const beats = scenes.flatMap((s) => s.beats), voicedAll = beats.filter((b) => b.file).length;
  writeFileSync('build/storyboard.html', `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>${esc(title)} · 分镜</title>
<style>
  :root { --ink: #141414; --mute: #7a7a72; --line: #e3e3e3; --soft: #f6f5f1; --accent: #567891; --warn: #b4541a; --side: 236px; }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; scroll-padding-top: 16px; }
  body { margin: 0; background: #fff; color: var(--ink); font: 16px/1.6 "PingFang SC", "Helvetica Neue", sans-serif; }
  p, h1, h2, ol { margin: 0; padding: 0; } ol { list-style: none; }
  .mono, .tc, .len, .num, .nd, .meta { font-family: "SF Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }

  /* 目录 */
  aside { position: fixed; inset: 0 auto 0 0; width: var(--side); padding: 24px 12px 16px 16px; border-right: 1px solid var(--line); overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
  aside h1 { font-size: 20px; line-height: 1.2; } aside h1 small { display: block; margin-top: 4px; font-size: 13px; font-weight: 500; color: var(--mute); }
  .stats { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-size: 13px; color: var(--mute); } .stats b { color: var(--ink); font-weight: 600; }
  aside li a { display: grid; grid-template-columns: 64px 1fr; grid-template-rows: auto auto; column-gap: 10px; align-items: center; padding: 6px; border-radius: 6px; border-left: 3px solid transparent; color: inherit; text-decoration: none; }
  aside li a:hover { background: var(--soft); } aside li a.on { background: var(--soft); border-left-color: var(--accent); }
  aside li img { grid-row: 1 / 3; width: 64px; aspect-ratio: 16 / 9; border: 1px solid var(--line); border-radius: 3px; }
  .nt { font-size: 14px; line-height: 1.3; align-self: end; } .nt b { margin-right: 6px; color: var(--accent); }
  .nd { font-size: 12px; color: var(--mute); align-self: start; } .nd i { display: inline-block; width: 6px; height: 6px; margin-left: 4px; border-radius: 50%; background: var(--warn); }
  .opts { margin-top: auto; font-size: 13px; color: var(--mute); display: grid; gap: 6px; } .opts label { color: var(--ink); cursor: pointer; }
  kbd { font: 12px "SF Mono", Menlo, monospace; padding: 1px 5px; border: 1px solid var(--line); border-radius: 4px; background: var(--soft); }

  /* 各幕 */
  main { margin-left: var(--side); padding: 8px 32px 120px; max-width: 1760px; }
  section { padding: 28px 0 32px; border-bottom: 1px solid var(--line); }
  header { display: grid; grid-template-columns: auto 1fr auto; gap: 14px; align-items: end; margin-bottom: 14px; }
  .num { font-size: 30px; font-weight: 600; line-height: 1; color: var(--accent); }
  .tag { font-size: 13px; color: var(--mute); } h2 { font-size: 22px; line-height: 1.35; }
  .c-p { color: #7b3ff2; } .c-t { color: #0f8f7c; } .c-r { color: #dc2f2a; } .c-b { color: #2350d8; } .c-y { color: #9a7500; } .c-g { color: var(--mute); }
  .meta { font-size: 13px; color: var(--mute); text-align: right; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
  .chip { font: 12px/1 "PingFang SC", sans-serif; padding: 5px 8px; border-radius: 999px; } .chip.warn { color: var(--warn); background: #fbefe4; }
  .body { display: grid; gap: 16px; }
  .media { width: min(100%, calc((100vh - 230px) * 16 / 9)); }
  img, video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #f1ede4; } .media > * { border: 1px solid var(--line); border-radius: 4px; }
  .script li { display: grid; grid-template-columns: 40px 1fr 48px; gap: 12px; align-items: baseline; padding: 10px 12px; border-left: 3px solid transparent; border-radius: 0 6px 6px 0; font-size: 17px; line-height: 1.75; }
  section:has(video) .script li { cursor: pointer; } .script li:hover { background: var(--soft); }
  .script li.on { background: #eef2f5; border-left-color: var(--accent); }
  .tc, .len { font-size: 12px; color: var(--mute); } .len { text-align: right; }
  .tx em { display: block; margin-top: 2px; font-style: normal; font-size: 14px; color: var(--warn); }
  .tx b { font-weight: 700; } .tx .en { color: var(--accent); font-weight: 600; }
  .src { margin-top: 12px; font-size: 13px; color: var(--mute); }
  .src::before { content: "来源　"; color: var(--ink); }

  /* 宽屏：视频与配音稿并排，一屏看完一幕 */
  @media (min-width: 1500px) {
    .body { grid-template-columns: minmax(0, 1.8fr) minmax(340px, 1fr); gap: 24px; align-items: start; }
    .media { width: 100%; }
  }
  /* 窄窗口：目录收成顶部的一行 */
  @media (max-width: 1000px) {
    aside { position: sticky; inset: 0 0 auto 0; z-index: 2; width: auto; flex-direction: row; align-items: center; padding: 8px 16px; border-right: 0; border-bottom: 1px solid var(--line); background: #fff; }
    aside h1 { font-size: 15px; white-space: nowrap; } aside h1 small, .stats, .opts span, aside li img, .nd { display: none; }
    aside ol { display: flex; gap: 2px; overflow-x: auto; scrollbar-width: none; flex: 1; } aside li a { display: block; padding: 4px 8px; border-left: 0; border-bottom: 3px solid transparent; white-space: nowrap; }
    aside li a.on { border-bottom-color: var(--accent); } .opts { margin: 0; white-space: nowrap; }
    main { margin-left: 0; padding: 0 16px 80px; } html { scroll-padding-top: 60px; }
    header { grid-template-columns: auto 1fr; } .meta { grid-column: 1 / 3; justify-content: flex-start; }
    .media { width: min(100%, calc((100vh - 300px) * 16 / 9)); }
  }
</style></head><body>
<aside>
  <h1>${esc(title)}<small>分镜 · ${fps} fps 预览</small></h1>
  <p class="stats"><span>全片</span><b>${scenes.length} 幕 · ${clock(total)}</b><span>视频</span><b>${done} / ${scenes.length} 幕</b><span>配音</span><b>${anyVoice ? `${voicedAll} / ${beats.length} 句` : '无'}</b></p>
  <ol>${nav}</ol>
  <p class="opts"><label><input type="checkbox" id="chain"> 连续播放</label><span><kbd>J</kbd> <kbd>K</kbd> 换幕　<kbd>空格</kbd> 播放</span><span>点一行字幕，跳到视频里这一句</span></p>
</aside>
<main>
${body}</main>
<script>
const secs = [...document.querySelectorAll('section')], links = [...document.querySelectorAll('aside li a')], chain = document.getElementById('chain');
let cur = 0;
const go = (i, play) => { const s = secs[i]; if (!s) return; s.scrollIntoView(); const v = s.querySelector('video'); if (play && v) { v.currentTime = 0; v.play(); } };
try { chain.checked = localStorage.getItem('chain') === '1'; chain.onchange = () => localStorage.setItem('chain', chain.checked ? '1' : '0'); } catch (e) {}
secs.forEach((sec, i) => {
  const v = sec.querySelector('video'), rows = [...sec.querySelectorAll('.script li')];
  if (!v) return;
  // 点一句跳到这一句；播放时高亮当前句
  // 视频还没加载（preload="none"）时先等元数据，再定位，否则部分浏览器会从头播放
  rows.forEach((li) => li.addEventListener('click', () => {
    const seek = () => { v.currentTime = Math.max(0, li.dataset.s - 0.15); v.play(); };
    if (v.readyState >= 1) seek(); else { v.addEventListener('loadedmetadata', seek, { once: true }); v.load(); }
  }));
  v.addEventListener('timeupdate', () => rows.forEach((li) => li.classList.toggle('on', v.currentTime >= li.dataset.s - 0.2 && v.currentTime < li.dataset.e - 0.2)));
  v.addEventListener('play', () => document.querySelectorAll('video').forEach((o) => o !== v && o.pause()));
  v.addEventListener('ended', () => { rows.forEach((li) => li.classList.remove('on')); if (chain.checked) go(i + 1, true); });
});
// 目录跟随滚动：视口上部所在的那一幕
const sync = () => { const y = innerHeight * 0.35; cur = Math.max(0, secs.findLastIndex((s) => s.getBoundingClientRect().top <= y)); links.forEach((a, i) => a.classList.toggle('on', i === cur)); };
addEventListener('scroll', sync, { passive: true }); sync();
addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|VIDEO/.test(e.target.tagName)) return;
  if (e.key === 'j') go(cur + 1); else if (e.key === 'k') go(cur - 1);
  else if (e.key === ' ') { const v = secs[cur].querySelector('video'); if (v) { e.preventDefault(); v.paused ? v.play() : v.pause(); } }
});
</script>
</body></html>
`);
}

// 一幕的视频：逐帧从画布取图送给 ffmpeg，各句录音按节拍的起点放进音轨
async function renderClip(i) {
  const s = scenes[i], n = Math.ceil(s.dur * fps), voiced = s.beats.filter((b) => b.file);
  mkdirSync('build/clips', { recursive: true });
  const args = ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-'];
  voiced.forEach((b) => args.push('-i', b.file));
  if (voiced.length) {
    const f = voiced.map((b, q) => `[${q + 1}:a]adelay=${Math.max(0, Math.round((b.s - b.head) * 1000))}:all=1[a${q}]`).join(';')
      + ';' + voiced.map((_, q) => `[a${q}]`).join('') + `amix=inputs=${voiced.length}:normalize=0[a]`;
    args.push('-filter_complex', f, '-map', '0:v', '-map', '[a]', '-c:a', 'aac', '-b:a', '160k');
  }
  args.push('-vf', 'scale=1280:-2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', '-t', s.dur.toFixed(3), '-movflags', '+faststart', clipFile(i));
  const ff = spawn('ffmpeg', args, { stdio: ['pipe', 'inherit', 'inherit'] });
  const closed = new Promise((r) => ff.on('close', r));
  for (let a = 0; a < n; a += 12) {
    const frames = await page.evaluate((at, a, b, fps) => {
      const out = [], cv = document.getElementById('cv');
      for (let f = a; f < b; f++) { window.renderAt(at + f / fps, fps, 1); out.push(cv.toDataURL('image/jpeg', 0.92).split(',')[1]); }
      return out;
    }, s.at, a, Math.min(n, a + 12), fps);
    for (const f of frames) if (!ff.stdin.write(Buffer.from(f, 'base64'))) await new Promise((r) => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  if (await closed) throw new Error(`ffmpeg failed on scene ${i + 1}`);
  writeFileSync(clipFile(i) + '.key', s.key);
}

async function clips({ only, force } = {}) {
  for (let i = 0; i < scenes.length; i++) {
    if (only && !only.includes(i)) continue;
    if (fresh(i) && !force) { console.log(`第 ${i + 1} 幕：未变，跳过`); continue; }
    const t0 = Date.now();
    await renderClip(i); await storyboard();
    console.log(`第 ${i + 1} 幕：${scenes[i].dur.toFixed(1)} s，渲染用时 ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
}
return { storyboard, clips };
}
