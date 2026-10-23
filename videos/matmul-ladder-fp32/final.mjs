// 成片：node final.mjs [--fps 60]（在本目录运行；先跑 python3 vo.py 与 node render.mjs，逐幕的音轨 build/audio/sNN.wav 由 render.mjs 生成）
//   -> build/matmul-ladder-fp32.mp4（1920 × 1080，画面用 ../lib/render-frames.mjs 分段渲染，音轨为各幕音轨顺序拼接）
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { parseOpts, renderSilent } from '../lib/render-frames.mjs';

const o = parseOpts(process.argv.slice(2), 60);
// script.js 按 vo.js 里的实测时长排节拍，先载入 vo.js
const script = new Function('var VO, VODUR;' + (existsSync('vo.js') ? readFileSync('vo.js', 'utf8') : '') + readFileSync('script.js', 'utf8') + '; return SCRIPT;')();
const dur = script.reduce((a, s) => a + s.dur, 0);
const wavs = script.map((_, k) => `build/audio/s${String(k + 1).padStart(2, '0')}.wav`);
const missing = wavs.filter((f) => !existsSync(f)); if (missing.length) { console.error('缺少音轨，先运行 node render.mjs：', missing.join(' ')); process.exit(1); }
const silent = await renderSilent('film.html', o, dur);
// 各幕音轨补齐或截到这一幕的精确时长再拼接，避免逐幕的细小误差累积成音画不同步
const exact = wavs.map((f, k) => { const g = f.replace('/s', '/x'); execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', f, '-af', `apad=whole_dur=${script[k].dur},atrim=end=${script[k].dur}`, '-ar', '48000', '-ac', '2', g]); return g; });
writeFileSync('build/audio/concat.txt', exact.map((f) => `file '${f.replace('build/audio/', '')}'\n`).join(''));
const out = 'build/matmul-ladder-fp32.mp4';
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-f', 'concat', '-safe', '0', '-i', 'build/audio/concat.txt', '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', dur.toFixed(3), '-movflags', '+faststart', out]);
exact.forEach((f) => rmSync(f)); rmSync(silent); rmSync('build/audio/concat.txt');
console.log('→', out);
process.exit(0);
