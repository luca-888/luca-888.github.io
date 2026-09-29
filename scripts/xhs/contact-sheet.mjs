// 生成整套页面的缩略总览：xhs/<slug>/preview.png
import fs from 'node:fs'
import path from 'node:path'
import puppeteer from 'puppeteer-core'

const slug = process.argv[2]
const dir = path.resolve('xhs', slug)
const files = fs.readdirSync(dir).filter(file => /^\d{2}\.png$/.test(file)).sort()
const cells = files.map(file => `<figure><img src="file://${path.join(dir, file)}"><figcaption>${file}</figcaption></figure>`).join('')
const html = `<style>body{margin:0;padding:16px;background:#e9ebee;font:14px sans-serif}main{display:grid;grid-template-columns:repeat(5,216px);gap:16px}figure{margin:0}img{width:216px;height:288px;display:block;box-shadow:0 0 0 1px #ccc}figcaption{text-align:center;color:#666;margin-top:4px}</style><main>${cells}</main>`
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--allow-file-access-from-files'] })
const page = await browser.newPage()
await page.setViewport({ width: 5 * 216 + 4 * 16 + 32, height: 800 })
const tmp = path.join(dir, 'preview.html')
fs.writeFileSync(tmp, html)
await page.goto(`file://${tmp}`, { waitUntil: 'load' })
await page.screenshot({ path: path.join(dir, 'preview.png'), fullPage: true })
fs.rmSync(tmp)
await browser.close()
console.log(`总览：${path.join('xhs', slug, 'preview.png')}`)
