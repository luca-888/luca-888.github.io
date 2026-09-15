import { useEffect, useId, useRef, useState } from 'react'
import { init, use, graphic, type EChartsType, type ComposeOption } from 'echarts/core'
import { CustomChart, type CustomSeriesOption } from 'echarts/charts'
import { GridComponent, DataZoomComponent, TooltipComponent, type GridComponentOption, type DataZoomComponentOption, type TooltipComponentOption } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'
import trace from '../content/data/rmsnorm-eager-timeline.json'
import './RmsNormTimeline.css'

use([CustomChart, GridComponent, DataZoomComponent, TooltipComponent, SVGRenderer])

type Direction = 'forward' | 'backward'
type Option = ComposeOption<CustomSeriesOption | GridComponentOption | DataZoomComponentOption | TooltipComponentOption>
const categories = {
  cast: { label: '类型转换', color: '#bda382' },
  elementwise: { label: '逐元素计算', color: '#8faabe' },
  reduce: { label: '归约', color: '#89a793' },
}

export function RmsNormTimeline() {
  const id = useId()
  const container = useRef<HTMLDivElement>(null)
  const chart = useRef<EChartsType | null>(null)
  const [direction, setDirection] = useState<Direction>('forward')
  const [selected, setSelected] = useState(0)
  const { kernels, spanUs } = trace.directions[direction]
  const kernel = kernels[selected]

  useEffect(() => {
    const styles = getComputedStyle(container.current!)
    const colors = {
      text: styles.getPropertyValue('--color-text').trim(),
      muted: styles.getPropertyValue('--color-muted').trim(),
      border: styles.getPropertyValue('--color-border').trim(),
      surface: styles.getPropertyValue('--color-surface').trim(),
    }
    const instance = init(container.current!, undefined, { renderer: 'svg' })
    chart.current = instance
    const option: Option = {
      animation: false,
      grid: { left: 48, right: 18, top: 12, bottom: 62 },
      xAxis: {
        type: 'value', min: 0, max: spanUs, splitNumber: 5,
        axisLabel: { color: colors.muted, fontSize: 12, fontFamily: 'Consolas, "Liberation Mono", monospace', formatter: value => `${Math.round(value)} μs`, hideOverlap: true },
        axisLine: { show: true, lineStyle: { color: colors.border } },
        axisTick: { show: false }, splitLine: { lineStyle: { color: '#edf0f2', type: 'dashed' } },
      },
      yAxis: {
        type: 'category', data: ['GPU'], axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { color: colors.muted, fontSize: 12 },
      },
      tooltip: {
        trigger: 'item', confine: true, padding: 10, borderColor: colors.border,
        backgroundColor: '#fff', borderRadius: 4, extraCssText: 'box-shadow: none;',
        formatter: params => {
          const item = kernels[(Array.isArray(params) ? params[0] : params).dataIndex]
          const tip = document.createElement('div')
          tip.className = 'kernel-tooltip'
          const code = document.createElement('strong')
          code.textContent = `#${item.index}  ${item.code}`
          const timing = document.createElement('div')
          timing.textContent = `${item.durationUs.toFixed(2)} μs · ${item.cpuOp}`
          const name = document.createElement('div')
          name.className = 'kernel-tooltip-name'
          name.textContent = item.name
          tip.append(code, timing, name)
          return tip
        },
      },
      dataZoom: [{
        type: 'slider', xAxisIndex: 0, filterMode: 'weakFilter', left: 48, right: 18, bottom: 8,
        height: 16, showDetail: false, showDataShadow: false, brushSelect: false,
        borderColor: colors.border, backgroundColor: colors.surface, fillerColor: '#dce5ec80',
        handleStyle: { color: '#b5c4cf', borderColor: '#93a8b8' },
      }],
      series: [{
        type: 'custom', clip: true,
        dimensions: ['lane', 'start', 'end', 'index'], encode: { x: [1, 2], y: 0 },
        data: kernels.map((item, index) => [0, item.startUs, item.startUs + item.durationUs, index]),
        renderItem: (params, api) => {
          const index = Number(api.value(3))
          const item = kernels[index]
          const start = api.coord([api.value(1), 0])
          const end = api.coord([api.value(2), 0])
          const bounds = params.coordSys as { type: string; x: number; y: number; width: number; height: number }
          const shape = graphic.clipRectByRect({ x: start[0], y: start[1] - 15, width: end[0] - start[0], height: 30 }, bounds)
          if (!shape) return
          return {
            type: 'group', children: [
              {
                type: 'rect', shape,
                style: { fill: categories[item.category as keyof typeof categories].color },
                emphasis: { style: { fill: categories[item.category as keyof typeof categories].color, stroke: colors.text, lineWidth: 1.5 } },
              },
              ...(shape.width > 22 ? [{
                type: 'text' as const, silent: true,
                style: { x: shape.x + shape.width / 2, y: shape.y + 15, text: String(item.index), fill: colors.text, font: '12px Consolas, "Liberation Mono", monospace', align: 'center' as const, verticalAlign: 'middle' as const },
              }] : []),
            ],
          }
        },
      }],
    }
    instance.setOption(option)
    instance.on('mouseover', { seriesIndex: 0 }, event => setSelected(event.dataIndex))
    const resize = new ResizeObserver(() => instance.resize())
    resize.observe(container.current!)
    return () => {
      resize.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [direction, kernels, spanUs])

  useEffect(() => {
    chart.current?.dispatchAction({ type: 'downplay', seriesIndex: 0 })
    chart.current?.dispatchAction({ type: 'highlight', seriesIndex: 0, dataIndex: selected })
  }, [selected, direction])

  return (
    <figure className="kernel-timeline" aria-labelledby={`${id}-title`}>
      <header className="kernel-timeline-header">
        <div>
          <h4 id={`${id}-title`}>Eager kernel 时间线</h4>
          <span>1024 × 4096 · BF16 · {kernels.length} 个 kernel</span>
        </div>
        <div className="kernel-direction" role="group" aria-label="计算方向">
          {(['forward', 'backward'] as const).map(mode => <button key={mode} type="button" aria-pressed={direction === mode} onClick={() => { setSelected(0); setDirection(mode) }}>{mode === 'forward' ? 'Forward' : 'Backward'}</button>)}
        </div>
      </header>
      <div className="kernel-legend">
        {Object.entries(categories).map(([key, item]) => <span key={key}><i style={{ background: item.color }} />{item.label}</span>)}
      </div>
      <div className="kernel-chart" ref={container} role="img" aria-label={`${direction === 'forward' ? 'Forward' : 'Backward'} 单次 GPU trace，${kernels.length} 个 kernel，时间跨度 ${spanUs.toFixed(2)} 微秒。下方可逐个选择 kernel。`} />
      <div className="kernel-selection">
        <label htmlFor={`${id}-kernel`}>Kernel</label>
        <select id={`${id}-kernel`} value={selected} onChange={event => setSelected(Number(event.target.value))}>
          {kernels.map((item, index) => <option key={item.index} value={index}>#{item.index} · {item.cpuOp.replace('aten::', '')}</option>)}
        </select>
        <code>{kernel.code}</code>
        <strong>{kernel.durationUs.toFixed(2)} <span>μs</span></strong>
      </div>
      <figcaption>单次 trace · 色块宽度表示执行时间，空隙表示 kernel 间隔。拖动滑块缩放，悬停查看详情。</figcaption>
    </figure>
  )
}
