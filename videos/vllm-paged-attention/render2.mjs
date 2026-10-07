// node render2.mjs sample.html build/sample.mp4          -> video (并行分段渲染，可选 --draft / --fps N / --workers N / --only a-b / --force)
// node render2.mjs sample.html --stills 1 4 9            -> build/s-<t>.png
import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';
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
await page.evaluate(() => document.fonts.ready);
const clip = { x: 0, y: 0, width: 1920, height: 1080 };

if (out === '--stills') {
  for (const t of rest.map(Number)) {
    await page.evaluate((x) => window.renderAt(x), t);
    await page.screenshot({ path: `build/s-${t}.png`, clip });
  }
} else {
  const dur = await page.evaluate(() => window.DURATION);
  const events = await page.evaluate(() => window.EVENTS || []);
  sfx(events, dur, 'build/sfx.wav');
  const silent = await renderSilent(html, opts, dur);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', 'build/sfx.wav', '-map', '0:v', '-map', '1:a',
    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', outPath(out, opts)]);
}
if (errs.length) console.log('page errors:', errs);
await browser.close();

// synthesized sound effects placed at event times
function sfx(events, dur, file) {
  const src = {
    whoosh: 'anoisesrc=d=0.4:c=pink:a=0.6,highpass=f=700,lowpass=f=5000,afade=t=in:d=0.18,afade=t=out:st=0.18:d=0.22,volume=0.28',
    land: "aevalsrc='0.6*sin(2*PI*(420+600*exp(-40*t))*t)*exp(-18*t)':d=0.25,volume=0.5",
    click: "aevalsrc='0.7*sin(2*PI*(900+1500*exp(-80*t))*t)*exp(-60*t)':d=0.08,volume=0.45",
    tick: "aevalsrc='0.5*sin(2*PI*2100*t)*exp(-120*t)':d=0.04,volume=0.3",
    pop: "aevalsrc='0.5*sin(2*PI*(660+300*exp(-30*t))*t)*exp(-28*t)':d=0.15,volume=0.4",
    thud: "aevalsrc='0.9*sin(2*PI*(55+110*exp(-30*t))*t)*exp(-9*t)':d=0.5,volume=0.9",
    err: "aevalsrc='0.35*(sin(2*PI*140*t)+0.6*sin(2*PI*147*t))*exp(-4*t)*(0.6+0.4*sin(2*PI*30*t))':d=0.5,volume=0.6",
    alloc: "aevalsrc='0.4*(sin(2*PI*660*t)+sin(2*PI*990*t))*exp(-6*t)':d=0.6,volume=0.4",
    hit: "aevalsrc='0.8*sin(2*PI*(60+140*exp(-25*t))*t)*exp(-7*t)':d=0.8,volume=0.8",
  };
  const args = ['-y', '-loglevel', 'error'];
  events.forEach(([, k]) => args.push('-f', 'lavfi', '-i', src[k]));
  const filt = events.map(([at], i) => `[${i}]aformat=sample_rates=48000:channel_layouts=stereo,adelay=${Math.round(at * 1000)}:all=1[a${i}]`).join(';')
    + ';' + events.map((_, i) => `[a${i}]`).join('') + `amix=inputs=${events.length}:normalize=0,alimiter=limit=0.9,apad=whole_dur=${dur}[o]`;
  args.push('-filter_complex', filt, '-map', '[o]', '-t', String(dur), file);
  execFileSync('ffmpeg', args);
}
