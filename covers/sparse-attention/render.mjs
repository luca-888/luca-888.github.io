// 用法：node covers/sparse-attention/render.mjs，导出同目录的 portrait-1080x1920.png
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'

const dir = path.dirname(fileURLToPath(import.meta.url))
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' })
const page = await browser.newPage()
await page.setViewport({ width: 1080, height: 1920, deviceScaleFactor: 1 })
await page.goto(`file://${path.join(dir, 'cover.html')}`, { waitUntil: 'load' })
await page.screenshot({ path: path.join(dir, 'portrait-1080x1920.png') })
await browser.close()
