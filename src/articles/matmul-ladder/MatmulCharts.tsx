import { useEffect, useRef, type ReactNode } from 'react'
import { init, use, type ComposeOption } from 'echarts/core'
import { BarChart, LineChart, type BarSeriesOption, type LineSeriesOption } from 'echarts/charts'
import { GridComponent, LegendComponent, MarkLineComponent, type GridComponentOption, type LegendComponentOption, type MarkLineComponentOption } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'
import data from '../../data/matmul-ladder.json'
import { useCompact } from '../../figure-kit'
import '../../figure-kit.css'
import './matmul-ladder.css'

use([BarChart, LineChart, GridComponent, LegendComponent, MarkLineComponent, SVGRenderer])

type Option = ComposeOption<BarSeriesOption | LineSeriesOption | GridComponentOption | LegendComponentOption | MarkLineComponentOption>
type Row = { name: string; line: string; tflops: Record<string, number>; side?: boolean }
// compact：小红书导出（图宽 880px），字号与条宽按约 1.8 倍放大。
type UI = { get: (v: string) => string; text: string; muted: string; border: string; compact: boolean }

const ladder = (data.ladder as Row[]).filter(r => !r.side)
const base = data.baselines as Record<string, { tflops: Record<string, number> }>
const CUBLAS_FP32 = base.cublas_fp32.tflops['8192']
const CUBLAS_BF16 = base.cublas_bf16.tflops['8192']
const LABEL: Record<string, string> = { mma_async: 'mma pipeline', smem16: 'smem T=16', smem32: 'smem T=32' }

function Chart({ build, height, compactHeight, label, caption }: { build: (ui: UI) => Option; height: number; compactHeight?: number; label: string; caption: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  const compact = useCompact()
  useEffect(() => {
    const styles = getComputedStyle(ref.current!)
    const get = (n: string) => styles.getPropertyValue(n).trim()
    const chart = init(ref.current!, undefined, { renderer: 'svg' })
    chart.setOption(build({ get, text: get('--color-text'), muted: get('--color-muted'), border: get('--color-border'), compact }))
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(ref.current!)
    return () => { resize.disconnect(); chart.dispose() }
  }, [build, compact])
  return <figure className="ml-chart-figure">
    <div ref={ref} className="ml-chart" style={{ height: compact && compactHeight ? compactHeight : height }} role="img" aria-label={label} />
    <figcaption>{caption}</figcaption>
  </figure>
}

// 成绩阶梯：8192 阶，每版一级，对数横轴；FP32 路线 blue，Tensor Core 路线 teal，变慢的一级 red。
const buildLadder = (focus: 'fp32' | 'tc') => (ui: UI): Option => {
  const gray = '#c3c8d0'
  const blue = focus === 'fp32' ? ui.get('--fk-blue') : gray, teal = focus === 'tc' ? ui.get('--fk-teal') : gray, red = focus === 'tc' ? ui.get('--fk-red') : gray
  const fs = ui.compact ? 21 : 12, fl = ui.compact ? 20 : 11
  return {
    grid: ui.compact ? { left: 190, right: 150, top: 44, bottom: 80 } : { left: 110, right: 92, top: 30, bottom: 46 },
    xAxis: { type: 'log', min: 1, max: 1000, logBase: 10, name: 'TFLOPS（对数刻度）', nameLocation: 'middle', nameGap: ui.compact ? 46 : 26, nameTextStyle: { color: ui.muted, fontSize: fs },
      axisLabel: { color: ui.muted, fontSize: fs }, splitLine: { lineStyle: { color: ui.border, type: 'dashed' } }, axisLine: { lineStyle: { color: ui.border } } },
    yAxis: { type: 'category', inverse: true, data: ladder.map(r => LABEL[r.name] ?? r.name), axisTick: { show: false }, axisLine: { lineStyle: { color: ui.border } },
      axisLabel: { color: ui.text, fontSize: fs, fontFamily: 'SFMono-Regular, Consolas, monospace' } },
    series: [{
      type: 'bar', barWidth: ui.compact ? 24 : 13,
      data: ladder.map(r => {
        const v = r.tflops['8192'], ref = r.line === 'fp32' ? CUBLAS_FP32 : CUBLAS_BF16
        const color = r.name === 'wgmma_persist' ? red : r.line === 'fp32' ? blue : teal
        return { value: v, itemStyle: { color, borderRadius: [0, 2, 2, 0] }, label: { show: true, position: 'right', color: ui.text, fontSize: fl, formatter: `${v.toFixed(1)} · ${(v / ref * 100).toFixed(1)}%` } }
      }),
      markLine: { silent: true, symbol: 'none', data: [
        { xAxis: CUBLAS_FP32, lineStyle: { color: blue, type: 'dashed', width: 1 }, label: { formatter: `cuBLAS FP32 ${CUBLAS_FP32}`, color: blue, fontSize: fl, position: 'start' } },
        { xAxis: CUBLAS_BF16, lineStyle: { color: teal, type: 'dashed', width: 1 }, label: { formatter: `cuBLAS BF16 ${CUBLAS_BF16}`, color: teal, fontSize: fl, position: 'start' } },
      ] },
    }],
  }
}
const LADDER_FP32 = buildLadder('fp32'), LADDER_TC = buildLadder('tc')

export function LadderChart({ focus }: { focus: 'fp32' | 'tc' }) {
  return <Chart build={focus === 'fp32' ? LADDER_FP32 : LADDER_TC} height={520} compactHeight={900} label="8192 阶各版本的 TFLOPS 阶梯，对数刻度"
    caption={focus === 'fp32'
      ? 'H100 SXM，8192 阶，短时计时的中位数，对数横轴。蓝色为本篇的 FP32 路线，百分比相对 cuBLAS SGEMM；灰色为下篇的 Tensor Core 路线，百分比相对同布局的 cuBLAS BF16。'
      : '8192 阶，短时计时。百分比相对同布局的 cuBLAS BF16（灰色的 FP32 各版相对 cuBLAS SGEMM），红色为变慢的一版。'} />
}

// 各规模：最终版与 cuBLAS BF16，对数纵轴，柱上标注两者之比。
const NS = ['256', '512', '1024', '2048', '4096', '8192']
const final = ladder.find(r => r.name === 'wgmma_tstore')!
function buildSizes(ui: UI): Option {
  const teal = ui.get('--fk-teal'), gray = ui.get('--fk-gray')
  return {
    grid: { left: 56, right: 20, top: 40, bottom: 48 },
    legend: { data: ['最终版 wgmma_tstore', 'cuBLAS BF16'], bottom: 0, textStyle: { color: ui.text, fontSize: 12 }, itemWidth: 14, itemHeight: 9 },
    xAxis: { type: 'category', data: NS.map(n => `${n} 阶`), axisTick: { show: false }, axisLine: { lineStyle: { color: ui.border } }, axisLabel: { color: ui.muted, fontSize: 12 } },
    yAxis: { type: 'log', min: 1, max: 1000, name: 'TFLOPS', nameTextStyle: { color: ui.muted, fontSize: 12, align: 'left' }, axisLabel: { color: ui.muted, fontSize: 12 }, splitLine: { lineStyle: { color: ui.border, type: 'dashed' } } },
    series: [
      { name: '最终版 wgmma_tstore', type: 'bar', barGap: '8%', barWidth: 26, itemStyle: { color: teal },
        data: NS.map(n => ({ value: final.tflops[n], label: { show: true, position: 'top', color: ui.text, fontSize: 11, formatter: `${(final.tflops[n] / base.cublas_bf16.tflops[n] * 100).toFixed(1)}%` } })) },
      { name: 'cuBLAS BF16', type: 'bar', barWidth: 26, itemStyle: { color: gray }, data: NS.map(n => base.cublas_bf16.tflops[n]) },
    ],
  }
}

export function SizesChart() {
  return <Chart build={buildSizes} height={330} label="256 到 8192 阶最终版与 cuBLAS BF16 的 TFLOPS 对比"
    caption="短时计时；柱上为最终版 ÷ cuBLAS BF16。" />
}

// 强度上限 N/6 随阶数增长，与 H100 的两个拐点对比；naive 的实际强度为 0.25。
const IN = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
function buildIntensity(ui: UI): Option {
  const blue = ui.get('--fk-blue'), teal = ui.get('--fk-teal'), amber = ui.get('--fk-amber')
  const line = (y: number, color: string, text: string) => ({ yAxis: y, lineStyle: { color, type: 'dashed' as const, width: 1 }, label: { formatter: text, color, fontSize: 11, position: 'insideStartTop' as const } })
  return {
    grid: { left: 60, right: 30, top: 20, bottom: 46 },
    xAxis: { type: 'log', logBase: 2, min: 64, max: 8192, name: '阶数 N（对数刻度）', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: ui.muted, fontSize: 12 },
      axisLabel: { color: ui.muted, fontSize: 12 }, axisLine: { lineStyle: { color: ui.border } }, splitLine: { show: false } },
    yAxis: { type: 'log', min: 0.1, max: 10000, name: 'FLOP/byte', nameTextStyle: { color: ui.muted, fontSize: 12, align: 'left' }, axisLabel: { color: ui.muted, fontSize: 12 }, splitLine: { lineStyle: { color: ui.border, type: 'dashed' } } },
    series: [{
      type: 'line', data: IN.map(n => [n, n / 6]), symbolSize: 6, lineStyle: { color: blue, width: 2 }, itemStyle: { color: blue },
      endLabel: { show: true, formatter: '上限 N/6：1365', color: blue, fontSize: 11, offset: [-110, -14] },
      markLine: { silent: true, symbol: 'none', data: [line(295, teal, 'BF16 拐点 295'), line(20, blue, 'FP32 拐点 20'), line(0.25, amber, 'naive 无复用 0.25')] },
    }],
  }
}

export function IntensityChart() {
  return <Chart build={buildIntensity} height={280} label="arithmetic intensity 上限 N/6 随阶数的变化与 H100 拐点"
    caption="FP32 数据、每个数只读写一次时的强度上限 N/6，与 H100 的拐点（规格算力 ÷ 3.35 TB/s）对比。N 超过约 1800 时，上限也越过 BF16 的拐点。" />
}
