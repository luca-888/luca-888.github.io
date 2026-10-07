// node render.mjs film.html --keyframes   -> keyframes/kNN.png 与总览 keyframes/sheet.png
// 动画定稿后再加整片渲染（沿用 ../lib/render-frames.mjs）。
import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const [html, mode] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars'],
});
const page = await browser.newPage();
const errs = []; page.on('pageerror', (e) => errs.push(e.message));
await page.setViewport({ width: 1920, height: 1080 });
await page.goto('file://' + resolve(html));
await page.evaluate(async () => { await document.fonts.ready; });
const clip = { x: 0, y: 0, width: 1920, height: 1080 };
if (mode === '--keyframes') {
  mkdirSync('keyframes', { recursive: true });
  const n = await page.evaluate(() => window.DURATION);
  const files = [];
  for (let i = 0; i < n; i++) {
    await page.evaluate((x) => window.renderAt(x + 0.5), i);
    const f = `keyframes/k${String(i + 1).padStart(2, '0')}.png`;
    await page.screenshot({ path: f, clip }); files.push(f);
  }
  // 总览：每行 3 张，缩到 640 宽
  const rows = Math.ceil(files.length / 3);
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...files.flatMap((f) => ['-i', f]), '-filter_complex',
    files.map((_, i) => `[${i}]scale=640:-1[s${i}]`).join(';') + ';' + files.map((_, i) => `[s${i}]`).join('') +
    `xstack=inputs=${files.length}:layout=` + files.map((_, i) => `${(i % 3) * 640}_${Math.floor(i / 3) * 360}`).join('|') + `:fill=white`,
    'keyframes/sheet.png']);
  console.log(files.length, 'keyframes,', rows, 'rows');
}
if (errs.length) console.log('page errors:', errs);
await browser.close();
process.exit(0);
