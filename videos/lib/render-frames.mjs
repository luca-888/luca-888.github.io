// 并行、分段、可增量的画面渲染：把 canvas#cv 的每一帧编码成无声 mp4。
//   - 直接从 canvas 取帧（toDataURL），不走 page.screenshot（后者约 43 ms/帧）
//   - 多个浏览器并行渲染不同时间段
//   - 每 SEG 秒一段；先用探针帧的像素哈希判断这一段是否变化，未变的段直接复用缓存
// 选项（命令行）：--fps N  --crf N  --workers N  --draft  --force  --only a-b（秒，只重渲这个区间，其余段沿用缓存）
import puppeteer from 'puppeteer-core';
import { spawn, execFileSync } from 'node:child_process';
import { cpus } from 'node:os';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, basename } from 'node:path';

const SEG = 5;          // 每段秒数
const PROBES = 20;      // 每段的探针帧数（约 0.25 s 一帧）
const TOL = 8;          // 缩略图单格灰阶差超过此值视为内容变化
const VERSION = 2;      // 编码参数变化时递增，使旧缓存失效

export function parseOpts(argv, defaultFps) {
  const val = (k) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : undefined; };
  const only = val('--only')?.split('-').map(Number);
  const draft = argv.includes('--draft');
  return {
    draft, force: argv.includes('--force'), only,
    fps: Number(val('--fps')) || (draft ? 30 : defaultFps),
    crf: Number(val('--crf')) || 16,                // 正式输出的 x264 画质（越小越清晰）
    workers: Number(val('--workers')) || Math.max(1, Math.min(4, Math.floor(cpus().length / 2))),
  };
}

// 草稿输出加 .draft 后缀，避免覆盖正式成片
export const outPath = (out, o) => (o.draft ? out.replace(/\.mp4$/, '.draft.mp4') : out);

async function openPage(html) {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true, args: ['--force-color-profile=srgb', '--hide-scrollbars', '--allow-file-access-from-files'],
  });
  const page = await browser.newPage();
  const errs = []; page.on('pageerror', (e) => errs.push(e.message));
  await page.setViewport({ width: 1920, height: 1080 });
  await page.goto('file://' + resolve(html));
  await page.evaluate(async () => { await document.fonts.ready; if (window.READY) await window.READY; if (window.ASSETS) await window.ASSETS; });
  return { browser, page, errs };
}

// 返回 build/silent.mp4（草稿为 silent.draft.mp4）的路径
export async function renderSilent(html, o, dur) {
  const mode = o.draft ? 'draft' : 'full';
  const dir = `build/segs/${basename(html, '.html')}-${mode}-${o.fps}`;
  mkdirSync(dir, { recursive: true });
  const segFrames = SEG * o.fps, total = Math.ceil(dur * o.fps), nSeg = Math.ceil(total / segFrames);
  const samples = o.draft ? 3 : 5;
  const enc = o.draft
    ? ['-vf', "scale='if(gt(iw,ih),1280,720)':-2", '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23']
    : ['-c:v', 'libx264', '-preset', 'slow', '-crf', String(o.crf), '-profile:v', 'high'];
  const segFile = (s) => `${dir}/seg-${String(s).padStart(3, '0')}.mp4`;
  const keyFile = (s) => `${segFile(s)}.key`;
  const wanted = (s) => !o.only || (s + 1) * SEG > o.only[0] && s * SEG < (o.only[1] ?? Infinity);

  const t0 = Date.now(); let done = 0, reused = 0;
  const queue = Array.from({ length: nSeg }, (_, s) => s);
  const errs = [];
  const worker = async () => {
    const { browser, page, errs: e } = await openPage(html); errs.push(...e);
    for (let s; (s = queue.shift()) !== undefined;) {
      const first = s * segFrames, count = Math.min(segFrames, total - first);
      const have = existsSync(segFile(s)) && existsSync(keyFile(s));
      if (!wanted(s) && have) { reused++; continue; }        // --only 之外的段沿用缓存
      // 探针：在这一段内均匀取帧，缩成 128x72 灰度缩略图；与缓存的缩略图逐格比较。
      // 不用精确哈希，因为 canvas 光栅化偶有亚像素级抖动；真实内容变化会让某些格子差出几十个灰阶。
      const times = Array.from({ length: PROBES }, (_, p) => (first + ((p + 0.5) / PROBES) * count) / o.fps);
      const thumbs = Buffer.from(await page.evaluate((ts, fps) => {
        const cv = document.querySelector('canvas'), sm = document.createElement('canvas'); sm.width = 128; sm.height = 72;
        const sc = sm.getContext('2d', { willReadFrequently: true }), out = new Uint8Array(ts.length * 128 * 72);
        ts.forEach((t, k) => {
          window.renderAt(t, fps, 2);
          sc.drawImage(cv, 0, 0, 128, 72);
          const d = sc.getImageData(0, 0, 128, 72).data;
          for (let i = 0; i < 128 * 72; i++) out[k * 9216 + i] = (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
        });
        let bin = ''; for (let i = 0; i < out.length; i += 8192) bin += String.fromCharCode(...out.subarray(i, i + 8192));
        return btoa(bin);
      }, times, o.fps), 'base64');
      const key = `${VERSION}|${mode}|${o.fps}|${o.crf}|${samples}|${count}`;
      if (!o.force && have) {
        const old = readFileSync(keyFile(s));
        const nl = old.indexOf('\n'), same = old.subarray(0, nl).toString() === key && old.length - nl - 1 === thumbs.length;
        let worst = 0;
        if (same) for (let i = 0; i < thumbs.length; i++) worst = Math.max(worst, Math.abs(old[nl + 1 + i] - thumbs[i]));
        if (same && worst <= TOL) { reused++; continue; }
        if (process.env.DEBUG_RENDER) console.log(`segment ${s}: changed (max diff ${worst})`);
      }

      const tmp = segFile(s).replace(/\.mp4$/, '.tmp.mp4');
      const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(o.fps), '-c:v', 'mjpeg', '-i', '-',
        ...enc, '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', tmp], { stdio: ['pipe', 'inherit', 'inherit'] });
      const closed = new Promise((r) => ff.on('close', r));
      for (let i = 0; i < count; i++) {
        const b64 = await page.evaluate((t, fps, n) => {
          window.renderAt(t, fps, n);
          return document.querySelector('canvas').toDataURL('image/jpeg', 0.95).slice(23);
        }, (first + i) / o.fps, o.fps, samples);
        if (!ff.stdin.write(Buffer.from(b64, 'base64'))) await new Promise((r) => ff.stdin.once('drain', r));
      }
      ff.stdin.end();
      if (await closed) throw new Error(`ffmpeg failed on segment ${s}`);
      execFileSync('mv', [tmp, segFile(s)]);
      writeFileSync(keyFile(s), Buffer.concat([Buffer.from(key + '\n'), thumbs]));
      done++;
      console.log(`segment ${s + 1}/${nSeg} rendered  (${done} rendered, ${reused} reused, ${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    }
    await browser.close();
  };
  await Promise.all(Array.from({ length: o.workers }, worker));
  if (errs.length) { console.log('page errors:', errs); process.exit(1); }

  const list = `${dir}/list.txt`, silent = o.draft ? 'build/silent.draft.mp4' : 'build/silent.mp4';
  writeFileSync(list, Array.from({ length: nSeg }, (_, s) => `file '${resolve(segFile(s))}'`).join('\n'));
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', silent]);
  console.log(`frames: ${done} segments rendered, ${reused} reused, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  return silent;
}
