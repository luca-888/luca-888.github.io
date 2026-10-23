// 从 1.3 讲解页（交互版）导出逐帧剧本：node extract.mjs <讲解页.html>  -> frames.js（两集共用，复制到 ../matmul-ladder-tc/）
// 讲解页的脚本包在 IIFE 里，先在临时副本的末尾暴露内部对象 window.__X，再逐版、逐帧取出：
//   每版：代码行、相对上一版的改动行、实测；每帧：A/B/C 与片上网格的格子状态、地址条、计数、当前代码行、hwState(v, f) 的硬件状态
import puppeteer from 'puppeteer-core';
import { readFileSync, writeFileSync, mkdtempSync, copyFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';

const src = process.argv[2];
const dir = mkdtempSync(join(tmpdir(), 'mmx-'));
const html = readFileSync(src, 'utf8').replace('selectVersion(1);\n})();', 'window.__X = { V, selectVersion, hwState, frames: () => frames };\nselectVersion(1);\n})();');
writeFileSync(join(dir, 'page.html'), html);
try { copyFileSync(join(dirname(src), 'page.css'), join(dir, 'page.css')); } catch {}
const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
const page = await browser.newPage();
await page.goto('file://' + join(dir, 'page.html'), { waitUntil: 'load' });
const data = await page.evaluate(() => {
  const out = {};
  __X.V.forEach((v, i) => {
    __X.selectVersion(i);
    const fr = __X.frames().map((f) => {
      const ex = {};
      for (const [k, val] of Object.entries(f)) if (Array.isArray(val) && !['A', 'B', 'C', 'lines'].includes(k)) ex[k] = val;
      return { A: f.A || [], B: f.B || [], C: f.C || [], ex, strip: f.strip || null, cnt: f.cnt || null, lines: f.lines || [], reg: !!f.reg, skip: !!f.skip, hw: __X.hwState(v, f) };
    });
    out[v.name] = { line: v.line, tflops: v.tflops, code: v.code || [], changed: v.changed || [], frames: fr, hw0: __X.hwState(v, null) };
  });
  return out;
});
await browser.close(); rmSync(dir, { recursive: true });
writeFileSync('frames.js', '// 由 extract.mjs 从 1.3 讲解页（交互版）导出的逐帧剧本，两集共用；不要手改\nconst FR = ' + JSON.stringify(data) + ';\n');
console.log(Object.entries(data).map(([k, v]) => `${k}:${v.frames.length}`).join(' '));
