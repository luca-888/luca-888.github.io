// 用法：node covers/matmul-ladder-tc/render-4x3.mjs，导出同目录的 cover-4x3.png（4:3 横版，小红书横版视频封面）
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'
const dir = path.dirname(fileURLToPath(import.meta.url))
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' })
const page = await browser.newPage()
await page.setViewport({ width: 1440, height: 1080, deviceScaleFactor: 1 })
await page.goto(`file://${path.join(dir, 'cover-4x3.html')}`, { waitUntil: 'load' })
await page.screenshot({ path: path.join(dir, 'cover-4x3.png') })
await browser.close()
