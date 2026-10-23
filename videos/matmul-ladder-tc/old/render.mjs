// 逐幕预览：node render.mjs [--fps 12] [--only 1,2] [--force] [--silent]（--silent：分镜阶段的无声预览）（在本目录运行）
//   -> build/clips/sNN.mp4（每幕一段，带配音、字幕与音效，1280 宽）、build/audio/sNN.wav（每幕音轨，成片 final.mjs 复用）、build/storyboard.html（分镜页）、build/matmul-ladder-tc.preview.mp4（各幕拼接）
// node render.mjs --keyframes     -> keyframes/sNN.png（每幕讲完时的一帧，1920 × 1080）
// node render.mjs --stills        -> build/stills/sNN-<秒>.png（每幕的节拍处与幕尾各一帧，检查画面用）
// 预览默认 12 fps；成片用 node final.mjs（../lib/render-frames.mjs，60 fps）。
import puppeteer from 'puppeteer-core';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { reviewKit } from '../lib/storyboard.mjs';

const argv = process.argv.slice(2), val = (k) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : undefined; };
const fps = Number(val('--fps')) || 12, only = val('--only')?.split(',').map((n) => Number(n) - 1), force = argv.includes('--force');
mkdirSync('build/clips', { recursive: true }); mkdirSync('build/audio', { recursive: true });
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars', '--allow-file-access-from-files'] });
const page = await browser.newPage();
const errs = []; page.on('pageerror', (e) => errs.push(e.message));
await page.setViewport({ width: 1920, height: 1080 });
await page.goto('file://' + resolve('film.html'));
await page.evaluate(async () => { await document.fonts.ready; });
const { script, at, vo, events } = await page.evaluate(() => ({ script: SCRIPT, at: window.SCENE_AT, vo: typeof VO === 'undefined' ? null : VO, events: window.EVENTS }));
const pad = (i) => String(i + 1).padStart(2, '0');
// 内容指纹按幕计算：共用文件 + 这一幕的文案，改一幕只重渲这一幕（scenes.js 改动仍会让全部重渲）
const common = ['core.js', 'scenes.js', 'frames.js', 'vo.js', 'render.mjs'].map((f) => (existsSync(f) ? readFileSync(f) : '')).join('|');
const scenes = script.map((s, k) => ({
  tag: s.tag, chap: '', head: s.head || 'H100 矩阵乘法 · 下（Tensor Core）', srcNote: s.src, at: at[k], dur: s.dur, hold: 0.05,
  key: createHash('sha1').update(common + JSON.stringify(s) + fps + argv.includes('--silent')).digest('hex').slice(0, 12),
  beats: s.beats.map(([t, action, text], j) => { const v = vo?.beats[k]?.[j]; return { s: t, e: t + (v?.dur ?? 3), text, act: action, file: v && v.text === text && existsSync(v.file) ? v.file : '', head: v?.head ?? 0 }; }),
}));
const shot = async (t, path) => { await page.evaluate((t) => window.renderAt(t), t); await page.screenshot({ path, clip: { x: 0, y: 0, width: 1920, height: 1080 } }); };

if (argv.includes('--stills') || argv.includes('--keyframes')) {
  const dir = argv.includes('--stills') ? 'build/stills' : 'keyframes'; mkdirSync(dir, { recursive: true });
  for (let k = 0; k < scenes.length; k++) {
    if (only && !only.includes(k)) continue;
    if (dir === 'keyframes') { await shot(at[k] + scenes[k].dur - 0.05, `${dir}/s${pad(k)}.png`); continue; }
    const ts = [...new Set([...script[k].beats.slice(1).map((b) => b[0] - 0.05), ...(val('--at')?.split(',').map(Number) ?? []), scenes[k].dur - 0.05])];
    for (const u of ts) await shot(at[k] + u, `${dir}/s${pad(k)}-${u.toFixed(1)}.png`);
  }
  if (errs.length) console.log('page errors:', errs);
  await browser.close(); process.exit(0);
}

// 音效：柔和起音、偏中低频、整体低通加轻微空间感；同类事件过密时抽稀，连发时逐个减弱（做法与 tensor-core 相同）
function sfx(evs, dur, file) {
  const A = (hz, dec, d, v, extra = '') => `aevalsrc='(1-exp(-t*600))*exp(-${dec}*t)*(sin(2*PI*${hz}*t)${extra})':d=${d},volume=${v}`;
  const src = {
    tick: A(880, 55, .09, .10, '+0.3*sin(2*PI*1760*t)'),
    click: A('(520+120*exp(-60*t))', 40, .12, .16, '+0.25*sin(2*PI*1040*t)'),
    pop: A('(392+140*exp(-35*t))', 18, .22, .20),
    land: A('(260+180*exp(-45*t))', 20, .22, .26),
    whoosh: 'anoisesrc=d=0.45:c=brown:a=0.8,bandpass=f=900:w=1400,afade=t=in:d=0.2:curve=qsin,afade=t=out:st=0.2:d=0.25:curve=qsin,volume=0.45',
  };
  const GAP = { tick: .09, click: .08, pop: .08, land: .07 }, last = {}, run = {}, kept = [];
  for (const [t, k] of [...evs].filter(([t]) => t < dur).sort((a, b) => a[0] - b[0])) {
    if (GAP[k] && t - (last[k] ?? -9) < GAP[k]) continue;
    run[k] = t - (last[k] ?? -9) < .5 ? (run[k] || 0) + 1 : 0;
    last[k] = t; kept.push([t, k, Math.max(.45, 1 - run[k] * .06)]);
  }
  const args = ['-y', '-loglevel', 'error'];
  kept.forEach(([, k]) => args.push('-f', 'lavfi', '-i', src[k]));
  const filt = kept.map(([t, , v], i) => `[${i}]aformat=sample_rates=48000:channel_layouts=stereo,volume=${v.toFixed(2)},adelay=${Math.round(t * 1000)}:all=1[a${i}]`).join(';')
    + ';' + kept.map((_, i) => `[a${i}]`).join('') + `amix=inputs=${kept.length}:normalize=0,highpass=f=45,lowpass=f=5200,aecho=0.8:0.6:45|90:0.12|0.06,volume=4dB,alimiter=limit=0.7,apad=whole_dur=${dur}[o]`;
  execFileSync('ffmpeg', [...args, '-filter_complex', filt, '-map', '[o]', '-t', String(dur), file]);
}
// 一幕的音轨：配音响度归一到 -16 LUFS，音效降 9 dB 垫在下面
function audio(k) {
  const s = scenes[k], fx = `build/audio/fx-${pad(k)}.wav`, out = `build/audio/s${pad(k)}.wav`, vs = s.beats.filter((b) => b.file);
  if (argv.includes('--silent')) { execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-t', String(s.dur), out]); return out; }   // 分镜预览：无声
  sfx(events[k], s.dur, fx);
  if (!vs.length) { execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', fx, '-af', 'volume=-9dB', out]); return out; }
  const args = ['-y', '-loglevel', 'error', '-i', fx]; vs.forEach((b) => args.push('-i', b.file));
  const filt = vs.map((b, i) => `[${i + 1}]adelay=${Math.round(Math.max(0, b.s - b.head) * 1000)}:all=1[v${i}]`).join(';') + ';' + vs.map((_, i) => `[v${i}]`).join('')
    + `amix=inputs=${vs.length}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[vo];[0]volume=-9dB[fx];[vo][fx]amix=inputs=2:normalize=0,alimiter=limit=0.9,apad=whole_dur=${s.dur}[o]`;
  execFileSync('ffmpeg', [...args, '-filter_complex', filt, '-map', '[o]', '-t', String(s.dur), out]);
  return out;
}
const clipFile = (k) => `build/clips/s${pad(k)}.mp4`;
async function renderClip(k) {
  const s = scenes[k], n = Math.ceil(s.dur * fps), wav = audio(k);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-', '-i', wav, '-map', '0:v', '-map', '1:a',
    '-vf', 'scale=1280:-2', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-t', s.dur.toFixed(3), '-movflags', '+faststart', clipFile(k)], { stdio: ['pipe', 'inherit', 'inherit'] });
  const closed = new Promise((r) => ff.on('close', r));
  for (let a = 0; a < n; a += 12) {
    const frames = await page.evaluate((t0, a, b, fps) => {
      const out = [], cv = document.getElementById('cv');
      for (let f = a; f < b; f++) { window.renderAt(t0 + f / fps); out.push(cv.toDataURL('image/jpeg', 0.92).split(',')[1]); }
      return out;
    }, s.at, a, Math.min(n, a + 12), fps);
    for (const f of frames) if (!ff.stdin.write(Buffer.from(f, 'base64'))) await new Promise((r) => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  if (await closed) throw new Error(`ffmpeg failed on scene ${k + 1}`);
  writeFileSync(clipFile(k) + '.key', s.key);
}
const kit = reviewKit({ page, title: 'H100 矩阵乘法 · 下（Tensor Core）', fps, scenes });
for (let k = 0; k < scenes.length; k++) {
  if (only && !only.includes(k)) continue;
  const ok = existsSync(clipFile(k)) && existsSync(clipFile(k) + '.key') && readFileSync(clipFile(k) + '.key', 'utf8') === scenes[k].key;
  if (ok && !force) { console.log(`第 ${k + 1} 幕：未变，跳过`); continue; }
  const t0 = Date.now(); await renderClip(k);
  console.log(`第 ${k + 1} 幕：${scenes[k].dur.toFixed(1)} s，渲染用时 ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}
await kit.storyboard();
const fresh = (k) => existsSync(`build/clips/s${pad(k)}.mp4.key`) && readFileSync(`build/clips/s${pad(k)}.mp4.key`, 'utf8') === scenes[k].key;
if (scenes.every((_, k) => fresh(k))) {
  writeFileSync('build/clips/concat.txt', scenes.map((_, k) => `file 's${pad(k)}.mp4'\n`).join(''));
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'build/clips/concat.txt', '-c', 'copy', '-movflags', '+faststart', 'build/matmul-ladder-tc.preview.mp4']);
  console.log('→ build/matmul-ladder-tc.preview.mp4');
}
if (errs.length) console.log('page errors:', errs);
console.log('→ build/storyboard.html');
await browser.close(); process.exit(0);
