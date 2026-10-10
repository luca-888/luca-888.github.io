// 成片：node final.mjs [--fps 60]（在本目录运行；先跑 node render.mjs，逐幕的音轨 build/audio/sNN.wav 由它生成）
//   -> build/sparse-attention.mp4（1920 × 1080，画面用 ../lib/render-frames.mjs 分段渲染，音轨为各幕音轨顺序拼接）
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { parseOpts, renderSilent } from '../lib/render-frames.mjs';

const o = parseOpts(process.argv.slice(2), 60);
const script = new Function(readFileSync('script.js', 'utf8') + '; return SCRIPT;')();
const dur = script.reduce((a, s) => a + s.dur, 0);
const wavs = script.map((_, k) => `build/audio/s${String(k + 1).padStart(2, '0')}.wav`);
const missing = wavs.filter((f) => !existsSync(f)); if (missing.length) { console.error('缺少音轨，先运行 node render.mjs：', missing.join(' ')); process.exit(1); }
const silent = await renderSilent('film.html', o, dur);
writeFileSync('build/audio/concat.txt', wavs.map((f) => `file '${f.replace('build/audio/', '')}'\n`).join(''));
const out = 'build/sparse-attention.mp4';
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-f', 'concat', '-safe', '0', '-i', 'build/audio/concat.txt', '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', dur.toFixed(3), '-movflags', '+faststart', out]);
console.log('→', out);
process.exit(0);
