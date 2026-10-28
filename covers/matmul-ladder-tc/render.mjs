// 用法：node covers/matmul-ladder-tc/render.mjs，导出同目录的 cover-1920x1080.png（16:9 横版）
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer-core'
const dir = path.dirname(fileURLToPath(import.meta.url))
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' })
const page = await browser.newPage()
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 })
await page.goto(`file://${path.join(dir, 'cover.html')}`, { waitUntil: 'load' })
await page.screenshot({ path: path.join(dir, 'cover-1920x1080.png') })
await browser.close()
