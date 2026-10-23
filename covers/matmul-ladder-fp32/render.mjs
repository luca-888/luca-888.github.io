// 用法：node covers/matmul-ladder-fp32/render.mjs，导出同目录的 cover-2048x1536.png
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'
const dir = path.dirname(fileURLToPath(import.meta.url))
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' })
const page = await browser.newPage()
await page.setViewport({ width: 2048, height: 1536, deviceScaleFactor: 1 })
await page.goto(`file://${path.join(dir, 'cover.html')}`, { waitUntil: 'load' })
await page.screenshot({ path: path.join(dir, 'cover-2048x1536.png') })
await browser.close()
