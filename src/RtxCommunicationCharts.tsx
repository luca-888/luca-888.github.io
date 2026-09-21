import { useDarkTheme } from './theme'
import { useEffect, useId, useRef, useState } from 'react'
import { init, use, type ComposeOption } from 'echarts/core'
import { CustomChart, HeatmapChart, LineChart, type CustomSeriesOption, type HeatmapSeriesOption, type LineSeriesOption } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent, VisualMapComponent, type GridComponentOption, type LegendComponentOption, type TooltipComponentOption, type VisualMapComponentOption } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'
import './RtxCommunicationCharts.css'

use([CustomChart, HeatmapChart, LineChart, GridComponent, LegendComponent, TooltipComponent, VisualMapComponent, SVGRenderer])

type Matrix = (number | null)[][] | null
type Metric = 'bandwidth' | 'latency'
type Operation = 'all_reduce' | 'all_gather'
type Option = ComposeOption<CustomSeriesOption | HeatmapSeriesOption | LineSeriesOption | GridComponentOption | LegendComponentOption | TooltipComponentOption | VisualMapComponentOption>
type Colors = { heading: string; text: string; muted: string; border: string; surface: string; accent: string; secondary: string }

export type RtxCommunicationData = {
  devices: { id: number; bus_id: string }[]
  path_labels: (string | null)[][] | null
  bandwidth_gbs: Matrix
  remote_latency_ns: Matrix
  collectives: {
    group: string
    operation: string
    gpu_indices: number[]
    rows: { size_bytes: number; out_of_place: { time_us: number | null } }[]
  }[]
}

const mono = '"SFMono-Regular", Consolas, "Liberation Mono", monospace'
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0
const groupLabel = (indices: number[]) => `GPU [${indices.join(', ')}]`

function formatBytes(bytes: number) {
  const unit = bytes >= 2 ** 30 ? 'GiB' : bytes >= 2 ** 20 ? 'MiB' : bytes >= 2 ** 10 ? 'KiB' : 'B'
  const divisor = { B: 1, KiB: 2 ** 10, MiB: 2 ** 20, GiB: 2 ** 30 }[unit]
  return `${Number((bytes / divisor).toFixed(2))} ${unit}`
}

function tooltip(lines: string[]) {
  const box = document.createElement('div')
  lines.forEach(line => {
    const item = document.createElement('div')
    item.textContent = line
    box.append(item)
  })
  return box
}

function Chart({ option, height, label, className = '' }: {
  option: (colors: Colors) => Option
  height: number
  label: string
  className?: string
}) {
  const dark = useDarkTheme()
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const styles = getComputedStyle(ref.current!)
    const color = (name: string) => styles.getPropertyValue(`--color-${name}`).trim()
    const colors = { heading: color('heading'), text: color('text'), muted: color('muted'), border: color('border'), surface: color('surface'), accent: color('accent'), secondary: color('secondary') }
    const chart = init(ref.current!, undefined, { renderer: 'svg' })
    chart.setOption(option(colors))
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(ref.current!)
    return () => { resize.disconnect(); chart.dispose() }
  }, [option, dark])
  return <div ref={ref} className={`rtx-communication-chart ${className}`} style={{ height }} role="img" aria-label={label} />
}

export function RtxPairCommunication({ devices, path_labels, bandwidth_gbs, remote_latency_ns, bandwidthOnly = false }: Pick<RtxCommunicationData, 'devices' | 'path_labels' | 'bandwidth_gbs' | 'remote_latency_ns'> & { bandwidthOnly?: boolean }) {
  const id = useId()
  const [metric, setMetric] = useState<Metric>('bandwidth')
  const matrix = metric === 'bandwidth' ? bandwidth_gbs : remote_latency_ns
  const title = metric === 'bandwidth' ? 'GPU 两两单向带宽' : 'GPU 远端内存访问延迟'
  const unit = metric === 'bandwidth' ? 'GB/s' : 'ns'
  const decimals = metric === 'bandwidth' ? 2 : 1
  const direction = metric === 'bandwidth' ? '行：source（源 GPU） · 列：destination（目标 GPU）' : '行：requester（发起访问的 GPU） · 列：owner（内存所属 GPU）'
  const cells = devices.flatMap((source, row) => devices.map((destination, column) => {
    const value = matrix?.[row]?.[column]
    return { row, column, source, destination, value: row !== column && finite(value) ? value : null }
  }))
  const measured = cells.filter(cell => cell.value !== null)
  const missing = cells.filter(cell => cell.value === null)
  const absentPairs = missing.filter(cell => cell.row !== cell.column).length
  const labels = devices.map(device => `GPU ${device.id}`)
  const option = (colors: Colors): Option => ({
    animation: false,
    grid: { left: 70, right: 26, top: 42, bottom: 62 },
    xAxis: { type: 'category', data: labels, position: 'top', axisLine: { show: false }, axisTick: { show: false }, axisLabel: { interval: 0, fontSize: 12, color: colors.muted } },
    yAxis: { type: 'category', data: labels, inverse: true, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { fontSize: 12, color: colors.muted } },
    visualMap: {
      type: 'continuous', seriesIndex: 0, min: metric === 'bandwidth' && measured.length ? Math.floor(Math.min(...measured.map(cell => cell.value!))) : 0, max: Math.ceil(Math.max(...measured.map(cell => cell.value!))) || 1,
      orient: 'horizontal', left: 'center', bottom: 8, itemWidth: 10, itemHeight: 170,
      precision: 1, textStyle: { color: colors.muted, fontSize: 11 }, calculable: false,
      inRange: { color: metric === 'bandwidth' ? ['#f3d4a4', '#e5edf2', '#24648d'] : ['#f0f5f2', '#91ad9d'] },
    },
    tooltip: {
      trigger: 'item', confine: true, borderColor: colors.border, backgroundColor: colors.surface, borderRadius: 4,
      textStyle: { color: colors.text, fontSize: 12 }, extraCssText: 'box-shadow: none;',
      formatter: params => {
        const item = Array.isArray(params) ? params[0] : params
        const [column, row] = item.value as number[]
        const cell = cells[row * devices.length + column]
        return tooltip([
          `GPU ${cell.source.id} → GPU ${cell.destination.id}`,
          cell.row === cell.column ? '同一设备，不纳入远端通信比较' : cell.value === null ? '未取得测量结果' : `${title}：${cell.value.toFixed(decimals)} ${unit}`,
          `拓扑：${path_labels?.[row]?.[column] ?? '未取得'}`,
          `${cell.source.bus_id} → ${cell.destination.bus_id}`,
        ])
      },
    },
    series: [
      {
        type: 'heatmap', data: measured.map(cell => ({ value: [cell.column, cell.row, cell.value!], label: { color: metric === 'bandwidth' && cell.value! >= 48 ? '#fff' : '#24282b' } })),
        label: { show: true, color: '#24282b', fontSize: 12, fontFamily: mono, formatter: item => Number((item.value as number[])[2]).toFixed(decimals) },
        itemStyle: { borderColor: colors.surface, borderWidth: 2 },
        emphasis: { itemStyle: { borderColor: colors.accent, borderWidth: 1 } },
      },
      {
        type: 'custom', data: missing.map(cell => [cell.column, cell.row]), encode: { x: 0, y: 1 },
        renderItem: (_, api) => {
          const column = Number(api.value(0))
          const row = Number(api.value(1))
          const [x, y] = api.coord([column, row])
          const [width, height] = api.size!([1, 1]) as number[]
          return { type: 'group', children: [
            { type: 'rect', shape: { x: x - width / 2 + 1, y: y - Math.abs(height) / 2 + 1, width: width - 2, height: Math.abs(height) - 2 }, style: { fill: colors.surface } },
            { type: 'text', style: { x, y, text: row === column ? '—' : '未取得', fill: colors.muted, font: '12px sans-serif', align: 'center', verticalAlign: 'middle' } },
          ] }
        },
      },
    ],
  })

  return <figure className="rtx-communication" aria-labelledby={`${id}-title`}>
    <div className="rtx-communication-heading">
      <h4 id={`${id}-title`}>{title} <span>{unit}</span></h4>
      {!bandwidthOnly && <div className="rtx-communication-switch" role="group" aria-label="选择 GPU 通信指标">
        <button type="button" aria-pressed={metric === 'bandwidth'} onClick={() => setMetric('bandwidth')}>单向带宽</button>
        <button type="button" aria-pressed={metric === 'latency'} onClick={() => setMetric('latency')}>远端访问延迟</button>
      </div>}
    </div>
    <p className="rtx-communication-note">{direction}</p>
    {!bandwidthOnly && (measured.length ? <Chart option={option} height={Math.max(300, devices.length * 44 + 104)} label={`${title}热力图，${direction}。完整数值见下方数据表。`} className="rtx-pair-chart" />
      : <p className="rtx-communication-empty" role="status">{devices.length ? `尚未取得${title}结果。` : '尚未取得实机 GPU 清单和通信结果。'}</p>)}
    {devices.length > 0 && <div className="rtx-communication-details">
      <div className="table-scroll" tabIndex={0} role="region" aria-label={`${title}完整矩阵`}>
        <table className="rtx-matrix-table">
          <caption>{title}（{unit}）· 数值下方为拓扑标记</caption>
          <thead><tr><th scope="col">{metric === 'bandwidth' ? 'source / destination' : 'requester / owner'}</th>{devices.map(device => <th scope="col" key={device.id}>{`GPU ${device.id}`}</th>)}</tr></thead>
          <tbody>{devices.map((device, row) => <tr key={device.id}>
            <th scope="row">GPU {device.id}</th>
            {devices.map((target, column) => {
              const cell = cells[row * devices.length + column]
              return <td key={target.id} className={metric === 'bandwidth' && cell.value !== null ? (cell.value >= 48 ? 'rtx-bandwidth-high' : 'rtx-bandwidth-low') : undefined}>
                {row === column ? '—' : cell.value === null ? <span className="rtx-communication-missing">未取得</span> : cell.value.toFixed(decimals)}
                {row !== column && <small>{path_labels?.[row]?.[column] ?? '路径未取得'}</small>}
              </td>
            })}
          </tr>)}</tbody>
        </table>
      </div>
      <div className="rtx-device-mapping">{devices.map(device => <span key={device.id}>GPU {device.id}<code>{device.bus_id}</code></span>)}</div>
    </div>}
    <figcaption>对角线表示同一设备，不参与远端比较。{absentPairs > 0 ? `${absentPairs} 个方向尚未取得结果。` : ''}{metric === 'latency' ? '延迟表示 GPU 访问对端显存的时间，不是整次数据复制耗时。' : '每格为一个源到目标方向的带宽，不是双向带宽之和。'}</figcaption>
  </figure>
}

export function RtxCollectiveCommunication({ collectives }: Pick<RtxCommunicationData, 'collectives'>) {
  const id = useId()
  const [operation, setOperation] = useState<Operation>('all_reduce')
  const title = operation === 'all_reduce' ? 'AllReduce' : 'AllGather'
  const groups = collectives.filter(result => result.operation === operation)
  const sizes = [...new Set(groups.flatMap(group => group.rows.filter(row => row.size_bytes > 0 && finite(row.out_of_place.time_us)).map(row => row.size_bytes)))].sort((a, b) => a - b)
  const results = groups.map(group => ({
    ...group,
    values: new Map(group.rows.filter(row => row.size_bytes > 0 && finite(row.out_of_place.time_us)).map(row => [row.size_bytes, row.out_of_place.time_us!])),
  }))
  const available = results.filter(group => group.values.size > 0)
  const option = (colors: Colors): Option => ({
    animation: false,
    color: [colors.accent, colors.secondary, '#927b97', '#9b8068', '#7792a5', '#809b8b'],
    grid: { left: 65, right: 28, top: 66, bottom: 54 },
    legend: { type: 'scroll', left: 12, right: 12, top: 8, icon: 'roundRect', itemWidth: 16, itemHeight: 3, textStyle: { color: colors.text, fontSize: 12 }, pageIconColor: colors.accent, pageTextStyle: { color: colors.muted } },
    xAxis: {
      type: 'log', logBase: 4, min: sizes.length > 1 ? sizes[0] : undefined, max: sizes.length > 1 ? sizes[sizes.length - 1] : undefined,
      axisLabel: { formatter: formatBytes, hideOverlap: true, color: colors.muted, fontSize: 11, fontFamily: mono },
      axisLine: { lineStyle: { color: colors.border } }, axisTick: { show: false }, splitLine: { show: false },
    },
    yAxis: { type: 'value', min: 0, name: '耗时（μs）', nameTextStyle: { color: colors.muted, fontSize: 12 }, axisLabel: { color: colors.muted, fontSize: 11, fontFamily: mono }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: colors.border } } },
    tooltip: {
      trigger: 'axis', confine: true, borderColor: colors.border, backgroundColor: colors.surface, borderRadius: 4,
      textStyle: { color: colors.text, fontSize: 12 }, extraCssText: 'box-shadow: none;',
      formatter: params => {
        const points = Array.isArray(params) ? params : [params]
        const size = Number((points[0].value as number[])[0])
        return tooltip([
          `${title} · ${formatBytes(size)}`,
          ...results.map(group => `${groupLabel(group.gpu_indices)}：${group.values.has(size) ? `${group.values.get(size)!.toFixed(2)} μs` : '未取得'}`),
        ])
      },
    },
    series: available.map((group, index) => ({
      type: 'line', name: groupLabel(group.gpu_indices), id: group.group,
      data: sizes.map(size => [size, group.values.get(size) ?? null]), connectNulls: false,
      symbol: ['circle', 'rect', 'diamond', 'triangle'][index % 4], symbolSize: 6,
      lineStyle: { width: 1.6, type: index < 6 ? 'solid' : 'dashed' }, emphasis: { focus: 'series' },
    })),
  })

  return <figure className="rtx-communication" aria-labelledby={`${id}-title`}>
    <div className="rtx-communication-heading">
      <h4 id={`${id}-title`}>NCCL {title}</h4>
      <div className="rtx-communication-switch" role="group" aria-label="选择 collective">
        <button type="button" aria-pressed={operation === 'all_reduce'} onClick={() => setOperation('all_reduce')}>AllReduce</button>
        <button type="button" aria-pressed={operation === 'all_gather'} onClick={() => setOperation('all_gather')}>AllGather</button>
      </div>
    </div>
    <p className="rtx-communication-note">横轴：{operation === 'all_reduce' ? '每 rank 的数据量' : '每 rank 的接收总量'} · 纵轴：collective 耗时</p>
    {available.length ? <Chart option={option} height={350} label={`${title} 数据量与耗时曲线，按实际 GPU 编号分组。完整数值见下方数据表。`} />
      : <p className="rtx-communication-empty" role="status">尚未取得通过正确性检查的 {title} 结果。</p>}
    {results.length > 0 && <details className="rtx-communication-details">
      <summary>查看 {title} 完整数值</summary>
      <div className="table-scroll" tabIndex={0} role="region" aria-label={`${title} 耗时数据`}>
        <table className="rtx-collective-table">
          <caption>{title} 耗时（μs）</caption>
          <thead><tr><th scope="col">数据量</th>{results.map(group => <th scope="col" key={group.group}>{groupLabel(group.gpu_indices)}</th>)}</tr></thead>
          <tbody>{sizes.map(size => <tr key={size}>
            <th scope="row">{formatBytes(size)}</th>
            {results.map(group => <td key={group.group}>{group.values.has(size) ? group.values.get(size)!.toFixed(2) : <span className="rtx-communication-missing">未取得</span>}</td>)}
          </tr>)}</tbody>
        </table>
      </div>
    </details>}
    <figcaption>图例列出各组实际 GPU 编号；比较拓扑影响时应先比较相同卡数的组。曲线使用 out-of-place 结果。{operation === 'all_gather' ? 'AllGather 的横轴不是单个 rank 的发送量。' : ''}{results.length > available.length ? '部分分组尚未取得结果，见数据表。' : ''}</figcaption>
  </figure>
}

export function RtxCommunicationCharts(data: RtxCommunicationData) {
  return <div className="rtx-communication-charts">
    <RtxPairCommunication {...data} />
    <RtxCollectiveCommunication collectives={data.collectives} />
  </div>
}
