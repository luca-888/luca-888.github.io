// node render.mjs film.html build/vllm-distributed.mp4   -> video (并行分段渲染，可选 --draft / --fps N / --workers N / --only a-b / --force)
// node render.mjs film.html --stills 1 4 9                  -> build/s-<t>.png
// node render.mjs film.html --audio build/vllm-distributed.mp4 -> 只重做音效并替换已有视频的音轨
import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';
import { renameSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseOpts, outPath, renderSilent } from '../lib/render-frames.mjs';

const [html, out, ...rest] = process.argv.slice(2);
const opts = parseOpts(process.argv, 60);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars'],
});
const page = await browser.newPage();
const errs = []; page.on('pageerror', (e) => errs.push(e.message));
await page.setViewport({ width: 1920, height: 1080 });
await page.goto('file://' + resolve(html));
await page.evaluate(async () => { await document.fonts.ready; if (window.READY) await window.READY; });
const clip = { x: 0, y: 0, width: 1920, height: 1080 };
const dur = await page.evaluate(() => window.DURATION);
const events = await page.evaluate(() => window.EVENTS || []);

if (out === '--stills') {
  for (const t of rest.map(Number)) {
    await page.evaluate((x) => window.renderAt(x), t);
    await page.screenshot({ path: `build/s-${t}.png`, clip });
  }
} else if (out === '--audio') {
  const video = rest[0], tmp = video.replace(/\.mp4$/, '.tmp.mp4');
  sfx(events, dur, 'build/sfx.wav');
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', video, '-i', 'build/sfx.wav', '-map', '0:v', '-map', '1:a',
    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', tmp]);
  renameSync(tmp, video);
} else {
  sfx(events, dur, 'build/sfx.wav');
  const silent = await renderSilent(html, opts, dur);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', 'build/sfx.wav', '-map', '0:v', '-map', '1:a',
    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', outPath(out, opts)]);
}
if (errs.length) console.log('page errors:', errs);
await browser.close();

// 合成音效：柔和起音（避免爆音）、偏中低频、整体低通 + 轻微空间感；同类事件过密时抽稀、并随序号递减音量。
function sfx(events, dur, file) {
  const A = (hz, dec, d, v, extra = '') => `aevalsrc='(1-exp(-t*600))*exp(-${dec}*t)*(sin(2*PI*${hz}*t)${extra})':d=${d},volume=${v}`;
  const src = {
    tick: A(880, 55, .09, .10, '+0.3*sin(2*PI*1760*t)'),                                   // 轻木鱼
    click: A('(520+120*exp(-60*t))', 40, .12, .16, '+0.25*sin(2*PI*1040*t)'),
    pop: A('(392+140*exp(-35*t))', 18, .22, .20),
    land: A('(260+180*exp(-45*t))', 20, .22, .26),
    // 鼓点：起音放慢（约 8 ms）、去掉大幅下扫、短衰减，只留闷的一声
    thud: `aevalsrc='(1-exp(-t*120))*exp(-11*t)*(sin(2*PI*(92+18*exp(-25*t))*t)+0.25*sin(2*PI*184*t)*exp(-9*t))':d=0.4,lowpass=f=360,volume=0.26`,
    hit: `aevalsrc='(1-exp(-t*100))*exp(-8*t)*(sin(2*PI*(78+16*exp(-20*t))*t)+0.2*sin(2*PI*196*t)*exp(-10*t))':d=0.5,lowpass=f=360,volume=0.28`,
    err: `aevalsrc='(1-exp(-t*300))*(sin(2*PI*330*t)*exp(-9*t)*lt(t,0.18)+sin(2*PI*247*(t-0.16))*exp(-7*(t-0.16))*gte(t,0.16))':d=0.6,volume=0.22`, // 两声下行
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
