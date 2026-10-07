// 渲染成片：node render.mjs [--draft] [--force] [--only a-b]（在本目录运行），输出 build/moe-kernel.mp4（无配音）
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { parseOpts, outPath, renderSilent } from '../lib/render-frames.mjs';

const DUR = 490;
const o = parseOpts(process.argv, 60);
mkdirSync('build', { recursive: true });
const silent = await renderSilent('film.html', o, DUR);
const out = outPath('build/moe-kernel.mp4', o);
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-c', 'copy', '-movflags', '+faststart', out]);
console.log('→', out);
process.exit(0);   // puppeteer 偶尔留下句柄，让 node 不退出
