// node film.mjs build/jev.mp4 [--fps N] [--draft] [--only a-b] [--workers N]  -> 带音效的影片（并行分段渲染，未变的段沿用缓存）；加 --silent 不合成音效
// node film.mjs --audio build/jev.mp4   -> 只重做音效并替换已有视频的音轨
// node film.mjs --stills 3 12.5 40      -> build/s-<秒>.png（时间轴上的静帧）
// node film.mjs --keyframes             -> keyframes/kNN.png（每幕结束时的画面）与每 6 帧一张的总览 keyframes/sheet-N.png
import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';
import { mkdirSync, copyFileSync, renameSync, existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseOpts, outPath, renderSilent } from '../lib/render-frames.mjs';

const HTML = 'film.html';
const [out, ...rest] = process.argv.slice(2);
const opts = parseOpts(process.argv, 60);
mkdirSync('build', { recursive: true });
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars', '--allow-file-access-from-files'] });
const page = await browser.newPage();
const errs = []; page.on('pageerror', (e) => errs.push(e.message));
await page.setViewport({ width: 1080, height: 1920 });
await page.goto('file://' + resolve(HTML));
await page.evaluate(() => document.fonts.ready);
const clip = { x: 0, y: 0, width: 1080, height: 1920 };
const dur = await page.evaluate(() => window.DURATION);
const events = await page.evaluate(() => window.EVENTS || []);
const shot = async (t, path) => { await page.evaluate((x) => window.renderAt(x, 60, 1), t); await page.screenshot({ path, clip }); };

if (out === '--cover') {   // node film.mjs --cover -> build/cover-1080x1920.png 与按信息流 3:4 裁切后的 build/cover-3x4.png
  await page.evaluate(() => window.renderCover()); await page.screenshot({ path: 'build/cover-1080x1920.png', clip });
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', 'build/cover-1080x1920.png', '-vf', 'crop=1080:1440:0:240', 'build/cover-3x4.png']);
} else if (out === '--stills') {
  for (const t of rest.map(Number)) await shot(t, `build/s-${t}.png`);
} else if (out === '--keyframes') {
  mkdirSync('keyframes', { recursive: true });
  const ends = await page.evaluate(() => window.sceneEnds), names = ends.map((_, i) => `keyframes/k${String(i + 1).padStart(2, '0')}.png`);
  for (let i = 0; i < ends.length; i++) await shot(ends[i], names[i]);
  for (let s = 0; s * 6 < names.length; s++) {
    const ids = names.slice(s * 6, s * 6 + 6), n = Math.max(2, ids.length), ins = (ids.length < 2 ? [ids[0], ids[0]] : ids).flatMap((f) => ['-i', f]);
    const layout = ['0_0', 'w0_0', 'w0+w1_0', '0_h0', 'w0_h0', 'w0+w1_h0'].slice(0, n).join('|');
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...ins, '-filter_complex', `xstack=inputs=${n}:layout=${layout}:fill=white,scale=2880:-1`, `keyframes/sheet-${s + 1}.png`]);
  }
}
if (errs.length) console.log('page errors:', errs);
await browser.close();
// 有配音时间轴（timing.js，由 vo.py 生成）时，把各段配音按起点叠到音效上，音效压低
function mixdown() { if (!existsSync('timing.js')) return 'build/sfx.wav';
  const tm = JSON.parse(readFileSync('timing.js', 'utf8').replace(/^window\.TIMING = /, '').replace(/;\s*$/, ''));
  // 背景音乐：目录里有 bgm.mp3（外部做好的音乐，如 Suno 导出）就用它，统一到较低的响度并首尾淡入淡出；否则用 bgm.py 生成的 build/bgm.wav。NO_BGM=1 时不混入
  const ext = existsSync('bgm.mp3'), bgm = (ext || existsSync('build/bgm.wav')) && !process.env.NO_BGM, len = tm.duration + 1;
  const args = ['-y', '-loglevel', 'error', '-i', 'build/sfx.wav', '-i', tm.voice]; if (bgm) args.push('-stream_loop', '-1', '-t', String(len), '-i', ext ? 'bgm.mp3' : 'build/bgm.wav');
  const music = ext ? `[2]aformat=sample_rates=48000:channel_layouts=stereo,loudnorm=I=-31:TP=-9,afade=t=in:d=0.6,afade=t=out:st=${(len - 2).toFixed(2)}:d=2[m]` : '[2]volume=0.16[m]';
  // 配音是一整条原始录音，只统一响度，不做其他处理
  const f = '[0]volume=0.45[s];[1]aformat=sample_rates=48000:channel_layouts=stereo,loudnorm=I=-16:TP=-1.5[v];'
    + (bgm ? music + ';[s][v][m]amix=inputs=3:normalize=0:duration=longest,alimiter=limit=0.9[o]' : '[s][v]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.9[o]');
  execFileSync('ffmpeg', [...args, '-filter_complex', f, '-map', '[o]', '-ar', '48000', 'build/mix.wav']); return 'build/mix.wav'; }
const mux = (video, file) => execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', video, '-i', mixdown(), '-map', '0:v', '-map', '1:a',
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
