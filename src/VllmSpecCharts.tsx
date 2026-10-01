import { useEffect, useRef, type ReactNode } from 'react'
import { init, use, type ComposeOption } from 'echarts/core'
import { LineChart, type LineSeriesOption } from 'echarts/charts'
import { GridComponent, LegendComponent, MarkLineComponent, TitleComponent, TooltipComponent, type GridComponentOption, type LegendComponentOption, type MarkLineComponentOption, type TitleComponentOption, type TooltipComponentOption } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'
import { useCompact } from './figure-kit'
import { useDarkTheme } from './theme'
import summary from './data/vllm-spec-summary.json'
import './figure-kit.css'
import './VllmSpecCharts.css'

use([LineChart, GridComponent, LegendComponent, MarkLineComponent, TitleComponent, TooltipComponent, SVGRenderer])

type Option = ComposeOption<LineSeriesOption | GridComponentOption | LegendComponentOption | MarkLineComponentOption | TitleComponentOption | TooltipComponentOption>
type Cell = { cfg: string; task: string; temp: number; c: number; al?: number; pos?: number[]; speedup?: number; kv_limited: boolean }
const cells = summary.cells as Cell[]
const find = (cfg: string, task: string, c: number) => cells.find(x => x.cfg === cfg && x.task === task && x.temp === 0 && x.c === c)

// 语义色与全文一致：draft model purple，EAGLE-3 teal，n-gram amber。
const DRAFTERS = [
  { key: 'draft', name: 'draft model', hue: '--fk-purple' },
  { key: 'eagle3', name: 'EAGLE-3', hue: '--fk-teal' },
  { key: 'ngram', name: 'n-gram', hue: '--fk-amber' },
]
const TASKS = [{ key: 'chat', name: '聊天' }, { key: 'summ', name: '摘要' }]

type UI = { text: string; muted: string; border: string; surface: string; compact: boolean; fs: number; lw: number }

// 紧凑模式（小红书导出）：两个子图上下排，字号、线宽放大到接近正文。
function Chart({ build, height, label, caption }: { build: (color: (v: string) => string, ui: UI) => Option; height: number; label: string; caption: ReactNode }) {
  const dark = useDarkTheme()
  const compact = useCompact()
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const styles = getComputedStyle(ref.current!)
    const get = (n: string) => styles.getPropertyValue(n).trim()
    const chart = init(ref.current!, undefined, { renderer: 'svg' })
    chart.setOption(build(get, { text: get('--color-text'), muted: get('--color-muted'), border: get('--color-border'), surface: get('--color-surface'), compact, fs: compact ? 24 : 12, lw: compact ? 2 : 1 }))
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(ref.current!)
    return () => { resize.disconnect(); chart.dispose() }
  }, [build, dark, compact])
  return <figure className="spec-chart-figure">
    <div ref={ref} className="spec-chart" style={{ height: compact ? 1010 : height }} role="img" aria-label={label} />
    <figcaption>{caption}</figcaption>
  </figure>
}

// 两个子图的位置：桌面左右并排，紧凑模式上下排。right 留给曲线末端的直接标注。
const gridsFor = (ui: UI, right = 16, bottom = 56) => ui.compact
  ? [{ left: 90, right: right > 16 ? 140 : 30, top: 80, height: 310 }, { left: 90, right: right > 16 ? 140 : 30, top: 570, height: 310 }]
  : [{ left: 52, right: right > 16 ? '58%' : '53%', top: 58, bottom }, { left: '55%', right, top: 58, bottom }]

function panels(ui: UI, xData: string[], yName: string, yMin: number, yMax: number, titles = TASKS.map(t => t.name)) {
  return {
    grid: gridsFor(ui),
    title: titles.map((t, i) => ({ text: t, left: ui.compact ? 90 : i === 0 ? 52 : '55%', top: ui.compact ? (i === 0 ? 4 : 494) : 4, textStyle: { color: ui.text, fontSize: ui.compact ? 28 : 13, fontWeight: 600 as const } })),
    xAxis: TASKS.map((_, i) => ({ gridIndex: i, type: 'category' as const, data: xData, axisLine: { lineStyle: { color: ui.border } }, axisTick: { show: false }, axisLabel: { color: ui.muted, fontSize: ui.fs } })),
    yAxis: TASKS.map((_, i) => ({ gridIndex: i, type: 'value' as const, name: i === 0 || ui.compact ? yName : '', nameTextStyle: { color: ui.muted, fontSize: ui.fs, align: 'left' as const }, min: yMin, max: yMax, splitLine: { lineStyle: { color: ui.border, type: 'dashed' as const } }, axisLabel: { color: ui.muted, fontSize: ui.fs } })),
  }
}
const legendFor = (ui: UI) => ({ data: DRAFTERS.map(d => d.name), bottom: 0, textStyle: { color: ui.text, fontSize: ui.fs }, itemWidth: ui.compact ? 34 : 16, itemHeight: ui.compact ? 14 : 8, itemGap: ui.compact ? 30 : 10 })

// 图 3：公式给出的加速比随 γ 的变化；α 用 blue 的三档深浅，圆点为每条曲线的最优 γ
const GAMMAS = Array.from({ length: 16 }, (_, i) => i + 1)
const COSTS = [{ c: 0.05, name: 'c = 0.05（轻量 draft head）' }, { c: 0.2, name: 'c = 0.2（小模型）' }]
const ALPHAS = [{ a: 0.9, opacity: 1 }, { a: 0.7, opacity: 0.6 }, { a: 0.5, opacity: 0.32 }]
const speedup = (a: number, g: number, c: number) => (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))

export function SpecGammaChart() {
  const build = (color: (v: string) => string, ui: UI): Option => {
    const blue = color('--fk-blue')
    const series: LineSeriesOption[] = []
    COSTS.forEach(({ c }, gi) => ALPHAS.forEach(({ a, opacity }) => {
      const ys = GAMMAS.map(g => speedup(a, g, c))
      const best = ys.indexOf(Math.max(...ys))
      series.push({
        name: `α = ${a}`, type: 'line', xAxisIndex: gi, yAxisIndex: gi, symbol: 'circle', showAllSymbol: true,
        data: ys.map((y, i) => ({ value: +y.toFixed(2), symbolSize: i === best ? 9 * ui.lw : 0 })),
        lineStyle: { color: blue, width: 2 * ui.lw, opacity }, itemStyle: { color: blue, opacity },
        endLabel: { show: true, formatter: `α = ${a}`, color: ui.text, fontSize: ui.fs, distance: 6 },
        markLine: a === 0.9 ? { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: ui.muted, type: 'solid', width: 1 }, data: [{ yAxis: 1 }] } : undefined,
      })
    }))
    const base = panels(ui, GAMMAS.map(String), '加速比', 0, 5, COSTS.map(x => x.name))
    return {
      animation: false,
      ...base,
      grid: gridsFor(ui, 64, 40),
      xAxis: base.xAxis.map(x => ({ ...x, axisLabel: { ...x.axisLabel, interval: (i: number) => i % 2 === 1 }, name: 'γ', nameLocation: 'middle' as const, nameGap: ui.compact ? 44 : 26, nameTextStyle: { color: ui.muted, fontSize: ui.fs } })),
      series,
    }
  }
  return <Chart build={build} height={300} label="加速比随 γ 的变化" caption="图 3　按第三节的公式计算，不是实测。横轴为每轮猜的 token 数 γ，纵轴为加速比；圆点为每条曲线的最优 γ，灰线为 1×。" />
}

// 图 4：γ=5 时各位置的累计接受概率，虚线为 i.i.d. 假设下的预测 α₁^i
export function SpecPositionChart() {
  const build = (color: (v: string) => string, ui: UI): Option => {
    const series: LineSeriesOption[] = []
    TASKS.forEach((t, gi) => DRAFTERS.forEach(d => {
      const cell = find(`${d.key}-k5`, t.key, 1)!
      const pos = cell.pos!
      const c = color(d.hue)
      series.push({ name: d.name, type: 'line', xAxisIndex: gi, yAxisIndex: gi, data: pos.map(v => +v.toFixed(3)), symbol: 'circle', symbolSize: 7 * ui.lw, lineStyle: { color: c, width: 2 * ui.lw }, itemStyle: { color: c } })
      series.push({ name: `${d.name} i.i.d.`, type: 'line', xAxisIndex: gi, yAxisIndex: gi, data: pos.map((_, i) => +(pos[0] ** (i + 1)).toFixed(3)), symbol: 'none', silent: true, itemStyle: { color: c }, lineStyle: { color: c, width: 1.2 * ui.lw, type: 'dashed', opacity: 0.8 } })
    }))
    return {
      animation: false,
      ...panels(ui, ['1', '2', '3', '4', '5'], '累计接受概率', 0, 0.8),
      legend: legendFor(ui),
      tooltip: { trigger: 'axis', confine: true, backgroundColor: ui.surface, borderColor: ui.border, textStyle: { color: ui.text, fontSize: 12 }, extraCssText: 'box-shadow: none;' },
      series,
    }
  }
  return <Chart build={build} height={316} label="γ=5 时各位置的累计接受概率" caption="图 4　横轴为 draft token 的位置，纵轴为“前 i 个 draft token 全部被接受”的概率（T=0，并发 1，γ=5）。实线为实测，虚线为 i.i.d. 假设下用第 1 位实测值算出的 α₁ⁱ。" />
}

// 图 5：γ=3 时相对配对 baseline 的加速比随并发的变化
export function SpecSpeedupChart() {
  const build = (color: (v: string) => string, ui: UI): Option => {
    const conc = [1, 4, 16, 32]
    const series: LineSeriesOption[] = []
    TASKS.forEach((t, gi) => DRAFTERS.forEach(d => {
      const c = color(d.hue)
      series.push({
        name: d.name, type: 'line', xAxisIndex: gi, yAxisIndex: gi, symbol: 'circle', symbolSize: 7 * ui.lw,
        data: conc.map(n => { const cell = find(`${d.key}-k3`, t.key, n); return cell && !cell.kv_limited ? +cell.speedup!.toFixed(2) : null }),
        lineStyle: { color: c, width: 2 * ui.lw }, itemStyle: { color: c },
        markLine: d.key === 'draft' ? { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: ui.muted, type: 'solid', width: 1 }, data: [{ yAxis: 1 }] } : undefined,
      })
    }))
    return {
      animation: false,
      ...panels(ui, conc.map(String), '相对配对 baseline 的加速比', 0.8, 1.8),
      legend: legendFor(ui),
      tooltip: { trigger: 'axis', confine: true, backgroundColor: ui.surface, borderColor: ui.border, textStyle: { color: ui.text, fontSize: 12 }, extraCssText: 'box-shadow: none;' },
      series,
    }
  }
  return <Chart build={build} height={316} label="γ=3 时加速比随并发的变化" caption="图 5　横轴为并发数，纵轴为吞吐相对配对 baseline 的倍数（T=0，γ=3，Qwen3-8B，RTX 4090）；灰线为 1×。draft model 在摘要并发 32 处因 KV 容量不足未绘出。" />
}
