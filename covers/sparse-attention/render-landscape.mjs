// 用法：node covers/sparse-attention/render-landscape.mjs，导出同目录的 landscape-1920x1440.png
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'

const dir = path.dirname(fileURLToPath(import.meta.url))
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' })
const page = await browser.newPage()
await page.setViewport({ width: 1920, height: 1440, deviceScaleFactor: 1 })
await page.goto(`file://${path.join(dir, 'cover-landscape.html')}`, { waitUntil: 'load' })
await page.screenshot({ path: path.join(dir, 'landscape-1920x1440.png') })
await browser.close()
