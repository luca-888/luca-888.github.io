// node render.mjs build/attention-kv-compression.mp4 [--fps N] [--draft] [--only a-b] [--workers N]  -> 带配音与音效的影片（并行分段渲染，未变的段沿用缓存）；加 --silent 不合成音效
// node render.mjs --audio build/attention-kv-compression.mp4   -> 只重做音效并替换已有视频的音轨
// node render.mjs --stills 3 12.5 40    -> build/s-<秒>.png（时间轴上的静帧）
import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';
import { mkdirSync, copyFileSync, renameSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseOpts, outPath, renderSilent } from '../lib/render-frames.mjs';

const HTML = process.env.FILM || 'film.html';
const [out, ...rest] = process.argv.slice(2);
const opts = parseOpts(process.argv, 60);
mkdirSync('build', { recursive: true });
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars', '--allow-file-access-from-files'] });
const page = await browser.newPage();
const errs = []; page.on('pageerror', (e) => errs.push(e.message));
await page.setViewport({ width: 1920, height: 1080 });
await page.goto('file://' + resolve(HTML));
await page.evaluate(() => document.fonts.ready);
const clip = { x: 0, y: 0, width: 1920, height: 1080 };
const dur = await page.evaluate(() => window.DURATION);
// 配音：vo.js 里每句录音放到它的节拍开头（幕起点 + 节拍起点 − 录音开头的静音）
const voice = await page.evaluate(() => (window.VO ? window.VO.beats.flatMap((g, k) => g.map((v, j) => { const b = window.BEATS[k][j], B = window.BEAT_AT[k]; return b.vo === v && B ? [window.sceneStarts[k] + B[j].s - v.head, v.file] : null; })) : []).filter(Boolean));
const events = (await page.evaluate(() => window.EVENTS || [])).filter(([t]) => t < dur);
const shot = async (t, path) => { await page.evaluate((x) => window.renderAt(x, 60, 1), t); await page.screenshot({ path, clip }); };

if (out === '--stills') {
  for (const t of rest.map(Number)) await shot(t, `build/s-${t}.png`);
} else if (out === '--keyframes') {   // 每幕结束时的画面 -> build/keyframes/kNN.png
  mkdirSync('build/keyframes', { recursive: true });
  const ends = await page.evaluate(() => window.sceneEnds);
  console.log('scene starts', (await page.evaluate(() => window.sceneStarts)).map((x) => x.toFixed(1)).join(' '), 'duration', dur.toFixed(1));
  for (let i = 0; i < ends.length; i++) await shot(ends[i], `build/keyframes/k${String(i).padStart(2, '0')}.png`);
}
if (errs.length) console.log('page errors:', errs);
await browser.close();
// 有配音时：音效降 9 dB 垫在下面，配音响度归一到 -16 LUFS
function mixVoice() {
  const vs = voice.filter(([at]) => at < dur); if (!vs.length) return 'build/sfx.wav';
  const args = ['-y', '-loglevel', 'error', '-i', 'build/sfx.wav']; vs.forEach(([, f]) => args.push('-i', f));
  const filt = vs.map(([at], i) => `[${i + 1}]adelay=${Math.round(Math.max(0, at) * 1000)}:all=1[v${i}]`).join(';') + ';' + vs.map((_, i) => `[v${i}]`).join('')
    + `amix=inputs=${vs.length}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[vo];[0]volume=-9dB[fx];[vo][fx]amix=inputs=2:normalize=0,alimiter=limit=0.9,apad=whole_dur=${dur}[o]`;
  execFileSync('ffmpeg', [...args, '-filter_complex', filt, '-map', '[o]', '-t', String(dur), 'build/mix.wav']); return 'build/mix.wav';
}
const mux = (video, file) => execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', video, '-i', mixVoice(), '-map', '0:v', '-map', '1:a',
  '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', file]);
if (out === '--audio') {
  const video = rest[0], tmp = video.replace(/\.mp4$/, '.tmp.mp4');
  sfx(events, dur, 'build/sfx.wav'); mux(video, tmp); renameSync(tmp, video);
} else if (out && !out.startsWith('--')) {
  const silent = await renderSilent(HTML, opts, dur), file = outPath(out, opts);
  if (process.argv.includes('--silent')) copyFileSync(silent, file); else { sfx(events, dur, 'build/sfx.wav'); mux(silent, file); }
  console.log(`duration ${dur.toFixed(1)} s -> ${file}`);
}
process.exitCode = 0; setTimeout(() => process.exit(), 200).unref();   // 渲染用的浏览器进程偶尔不退出，这里主动结束

// 合成音效：柔和起音（避免爆音）、偏中低频、整体低通 + 轻微空间感；同类事件过密时抽稀、并随序号递减音量。（与 vllm-distributed/render.mjs 相同）
function sfx(events, dur, file) {
  const A = (hz, dec, d, v, extra = '') => `aevalsrc='(1-exp(-t*600))*exp(-${dec}*t)*(sin(2*PI*${hz}*t)${extra})':d=${d},volume=${v}`;
  const src = {
    tick: A(880, 55, .09, .10, '+0.3*sin(2*PI*1760*t)'),
    click: A('(520+120*exp(-60*t))', 40, .12, .16, '+0.25*sin(2*PI*1040*t)'),
    pop: A('(392+140*exp(-35*t))', 18, .22, .20),
    land: A('(260+180*exp(-45*t))', 20, .22, .26),
    err: `aevalsrc='(1-exp(-t*300))*(sin(2*PI*330*t)*exp(-9*t)*lt(t,0.18)+sin(2*PI*247*(t-0.16))*exp(-7*(t-0.16))*gte(t,0.16))':d=0.6,volume=0.22`,
    whoosh: 'anoisesrc=d=0.45:c=brown:a=0.8,bandpass=f=900:w=1400,afade=t=in:d=0.2:curve=qsin,afade=t=out:st=0.2:d=0.25:curve=qsin,volume=0.45',
  };
  const GAP = { tick: .09, click: .08, pop: .08, land: .07 };
  const last = {}, run = {}, kept = [];
  for (const [at, k] of [...events].sort((a, b) => a[0] - b[0])) {
    if (GAP[k] && at - (last[k] ?? -9) < GAP[k]) continue;
    run[k] = at - (last[k] ?? -9) < .5 ? (run[k] || 0) + 1 : 0;   // 连发时逐个减弱
    last[k] = at; kept.push([at, k, Math.max(.45, 1 - run[k] * .06)]);
  }
  const args = ['-y', '-loglevel', 'error'];
  kept.forEach(([, k]) => args.push('-f', 'lavfi', '-i', src[k]));
  const filt = kept.map(([at, , g], i) => `[${i}]aformat=sample_rates=48000:channel_layouts=stereo,volume=${g.toFixed(2)},adelay=${Math.round(at * 1000)}:all=1[a${i}]`).join(';')
    + ';' + kept.map((_, i) => `[a${i}]`).join('') + `amix=inputs=${kept.length}:normalize=0,highpass=f=45,lowpass=f=5200,aecho=0.8:0.6:45|90:0.12|0.06,volume=4dB,alimiter=limit=0.7,apad=whole_dur=${dur}[o]`;
  args.push('-filter_complex', filt, '-map', '[o]', '-t', String(dur), file);
  execFileSync('ffmpeg', args);
}
