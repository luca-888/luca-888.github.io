// 用法：先 npm run dev，再 npm run xhs -- <slug>
// 输出到 xhs/<slug>/：01.png… 正文页、outline.json 标题清单、<slug>-xhs.zip（仅含正文页）。
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import puppeteer from 'puppeteer-core'

const slug = process.argv[2]
const base = process.env.XHS_URL ?? 'http://localhost:5173'
const chrome = process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
if (!slug) throw new Error('用法：npm run xhs -- <slug>')
await fetch(base).catch(() => { throw new Error(`无法访问 ${base}，请先运行 npm run dev`) })

const out = path.join('xhs', slug)
fs.mkdirSync(out, { recursive: true })
for (const file of fs.readdirSync(out)) if (/^\d{2}\.png$/.test(file)) fs.rmSync(path.join(out, file))

const browser = await puppeteer.launch({ executablePath: chrome, defaultViewport: { width: 1200, height: 1500, deviceScaleFactor: 1 } })
try {
  const page = await browser.newPage()
  page.on('pageerror', error => console.error('页面错误：', error.message))
  await page.goto(`${base}/xhs.html?slug=${slug}`, { waitUntil: 'load', timeout: 120_000 })
  await page.waitForFunction('window.__xhs?.ready', { timeout: 120_000 })
  const result = await page.evaluate(() => window.__xhs)
  if (!result.textIntact) throw new Error('分页前后正文文字不一致，可能丢失或重复了内容，未导出')
  const sections = await page.$$('.xhs-page')
  const files = []
  for (const [index, section] of sections.entries()) {
    const file = path.join(out, `${String(index + 1).padStart(2, '0')}.png`)
    await section.screenshot({ path: file })
    files.push(file)
  }
  fs.writeFileSync(path.join(out, 'outline.json'), JSON.stringify(result, null, 2))

  const zip = path.join(out, `${slug}-xhs.zip`)
  fs.rmSync(zip, { force: true })
  execFileSync('zip', ['-jq', zip, ...files])

  console.log(`正文 ${files.length} 页 → ${out}`)
  for (const h of result.headings.filter(h => h.level === 2)) console.log(`  p${String(h.page).padStart(2)}  ${h.text}`)
  if (result.shrunk.length) console.log('缩小过的块：', result.shrunk.map(s => `p${s.page} ×${s.scale}`).join('，'))
  console.log(`ZIP：${zip}`)
} finally {
  await browser.close()
}
