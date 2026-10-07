// node clips.mjs [幕名 ...] [--fps N]  -> build/clips/<幕名>.mp4（每幕一个片段，供 review.html 逐幕审阅；默认 12 fps，审阅用）
// 幕名为 k01、k01b、k02 … k13；不带幕名时切全部。各幕取自 film-v7.html（画面对齐到配音节拍）；K01 含封面，K01b 是全貌。
// 带幕名时只重渲这些幕所在的时间段，其余沿用缓存，几十秒内完成；改动影响了多幕或要出整片时不带幕名运行。
import puppeteer from 'puppeteer-core';
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const KN = ['k01', 'k01b', 'k02', 'k03', 'k04', 'k05', 'k06', 'k07', 'k08', 'k09', 'k10', 'k11', 'k12', 'k13'];
const args = process.argv.slice(2), fi = args.indexOf('--fps'), fps = fi < 0 ? '12' : args[fi + 1];
const norm = (a) => (/^\d+$/.test(a) ? 'k' + a.padStart(2, '0') : a.toLowerCase());
const want = args.filter((a, i) => i !== fi && i !== fi + 1 || fi < 0).map(norm).filter((a) => KN.includes(a));
const html = 'film-v7.html', full = 'build/clips/_film-v7.mp4';
mkdirSync('build/clips', { recursive: true });
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--allow-file-access-from-files'] });
const page = await browser.newPage(); await page.goto('file://' + resolve(html));
const sc = await page.evaluate(() => SCENES.map((s) => [s.at, s.at + s.dur])); await browser.close();
// 指定幕名时只渲染这些幕所在的时间段（--only），其余段沿用缓存：整片文件里别的幕可能是旧画面，但这里只切指定的幕。不带幕名时整片重渲
const render = (extra) => execFileSync('node', ['render.mjs', full, '--fps', fps, ...extra], { env: { ...process.env, FILM: html }, stdio: ['ignore', 'ignore', 'inherit'] });
if (!want.length) render([]); else for (const name of want) { const [a, b] = sc[KN.indexOf(name)]; render(['--only', `${Math.max(0, a - 1).toFixed(1)}-${(b + 1).toFixed(1)}`]); }
KN.forEach((name, i) => { if (want.length && !want.includes(name)) return; const [a, b] = sc[i], out = `build/clips/${name}.mp4`;
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', String(a), '-to', String(b), '-i', full, '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', fps,
    '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', out]);
  console.log(`${out}  ${(b - a).toFixed(1)} s`); });
process.exit(0);
