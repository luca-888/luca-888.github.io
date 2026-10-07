import { useEffect, useRef } from 'react'
import { init, use, type ComposeOption } from 'echarts/core'
import { LineChart, type LineSeriesOption } from 'echarts/charts'
import { GridComponent, TitleComponent, TooltipComponent, type GridComponentOption, type TitleComponentOption, type TooltipComponentOption } from 'echarts/components'
import { SVGRenderer } from 'echarts/renderers'
import { Cell, Figure, Legend, Link, TexLabel, type Fill } from '../../figure-kit'
import data from '../../data/jev.json'
import '../vllm-speculative-decoding/VllmSpecCharts.css'

use([LineChart, GridComponent, TitleComponent, TooltipComponent, SVGRenderer])

// 语义色：teal 为 state，blue 为问题 1（及一般的问题 branch），amber 为问题 2，purple 为 <decide> 与读出，gray 为被屏蔽。
const example = data.probe.example
const OUT = example.usage.output_tokens
const mono = (s: string) => <tspan className="fk-mono">{s}</tspan>

// 图 1：同一份答案，LLM 要 prefill 加逐 token decode，Jev 只做 prefill 加一次读出
export function JevTwoPaths() {
  const X0 = 20
  const PW = 190
  const DX = X0 + PW + 14
  const STEP = (740 - DX) / OUT
  return (
    <Figure
      title="生成 JSON 与直接读出概率的计算步骤"
      desc={`LLM 先 prefill，再串行 decode 约 ${OUT} 个 token，每个 token 一次 forward。Jev 做完 prefill 后直接读出三个分布。`}
      height={196}
      below={<Legend items={[['teal', 'prefill：所有输入 token 一次 forward'], ['blue', 'decode：每个输出 token 一次 forward'], ['purple', '读出概率']]} />}
      caption={`${OUT} 是 Jev 对示例请求返回的 output_tokens，即同一份答案写成 JSON 的 token 数。方块宽度不代表耗时。`}
    >{() => <>
      <text x={X0} y="14" className="fk-strong" textAnchor="start">LLM 生成 JSON</text>
      <Cell x={X0} y={28} w={PW} h={40} rx={6} fill="teal" label="state + 3 个问题" />
      {Array.from({ length: OUT }, (_, i) => <rect key={i} x={DX + i * STEP} y={28} width={STEP - 1.6} height={40} rx="1.5" className="fk-fill-blue" />)}
      <text x={DX} y="84" className="fk-muted" textAnchor="start">{OUT} 个 token，一个接一个：后一个要等前一个选定</text>

      <text x={X0} y="124" className="fk-strong" textAnchor="start">Jev</text>
      <Cell x={X0} y={138} w={PW} h={40} rx={6} fill="teal" label="state + 3 个问题" />
      <Cell x={DX} y={138} w={64} h={40} rx={6} fill="purple" label="读出" />
      <text x={DX + 76} y="158" className="fk-muted" textAnchor="start">3 个分布同时得到，没有 decode</text>
    </>}</Figure>
  )
}

// 图 2：一条序列里的 attention mask。行是读的一方，列是被读的一方。
export function JevMask() {
  // [标签, 所属段：0 = state，1 / 2 = 问题]
  const toks: [string, number][] = [
    ['s', 0], ['s', 0], ['s', 0],
    ['问', 1], ['A', 1], ['B', 1], ['决', 1],
    ['问', 2], ['A', 2], ['B', 2], ['决', 2],
  ]
  const hue = (seg: number): Fill => seg === 0 ? 'teal' : seg === 1 ? 'blue' : 'amber'
  const C = 30
  const X0 = 250
  const Y0 = 78
  const n = toks.length
  const pos = toks.map(([, seg], i) => seg === 0 ? i : 3 + (i - 3) % 4)
  const allowed = (i: number, j: number) => j <= i && (toks[j][1] === 0 || toks[j][1] === toks[i][1])
  return (
    <Figure
      title="state 与两个问题排成一条序列时的 attention mask"
      desc="11 个 token：3 个 state token，问题 1 与问题 2 各 4 个。每个 token 可以读 state 和自己所在问题里更早的 token；问题 2 读不到问题 1。两个问题的 position ID 都从 3 开始。"
      height={470}
      below={<Legend items={[['teal', '读 state'], ['blue', '问题 1 读自己'], ['amber', '问题 2 读自己'], ['gray', '别的问题：屏蔽'], ['free', '后面的 token：屏蔽']]} />}
      caption="行是读的一方，列是被读的一方。“问”是问题的指令，A、B 是两个选项，“决”是 <decide>。描边的方块是两个问题之间被屏蔽的部分。"
    >{() => <>
      <text x={X0 + n * C / 2} y="12" className="fk-strong">被读的 token</text>
      {toks.map(([label, seg], j) => <Cell key={j} x={X0 + j * C + 1} y={28} w={C - 2} h={C - 2} fill={hue(seg)} label={label} />)}
      <text x={X0 - 60} y={Y0 + n * C / 2} className="fk-strong" transform={`rotate(-90 ${X0 - 60} ${Y0 + n * C / 2})`}>读的 token</text>
      {toks.map(([label, seg], i) => <Cell key={i} x={X0 - C - 10} y={Y0 + i * C + 1} w={C - 2} h={C - 2} fill={hue(seg)} label={label} />)}

      {toks.map((_, i) => toks.map(([, seg], j) => {
        const fill: Fill = allowed(i, j) ? hue(seg) : j < i ? 'gray' : 'free'
        return <rect key={`${i}-${j}`} x={X0 + j * C + 1} y={Y0 + i * C + 1} width={C - 2} height={C - 2} rx="3" className={`fk-fill-${fill}`} />
      }))}
      <rect x={X0 + 3 * C - 1} y={Y0 + 7 * C - 1} width={4 * C + 2} height={4 * C + 2} rx="4" className="fk-focus" />

      <text x={X0 - 18} y={Y0 + n * C + 26} className="fk-strong" textAnchor="end">position ID</text>
      {pos.map((p, j) => <text key={j} x={X0 + j * C + C / 2} y={Y0 + n * C + 26} className={`fk-mono ${toks[j][1] === 2 ? 'fk-text-purple' : ''}`} style={toks[j][1] === 2 ? { fontWeight: 700 } : undefined}>{p}</text>)}
      <text x={X0 + n * C + 14} y={Y0 + 1.5 * C} className="fk-muted" textAnchor="start">state：只读自己</text>
      <text x={X0 + n * C + 14} y={Y0 + 5 * C} className="fk-muted" textAnchor="start">问题 1：state + 自己</text>
      <text x={X0 + n * C + 14} y={Y0 + 9 * C} className="fk-muted" textAnchor="start">问题 2：state + 自己</text>
      <text x={X0 + n * C + 14} y={Y0 + n * C + 26} className="fk-muted" textAnchor="start">问题 2 也从 3 开始</text>
    </>}</Figure>
  )
}

// 图 3：pointer head 从 <decide> 与各 </opt> 的 hidden state 读出分布
export function JevPointer() {
  const score = example.answers.frustration
  const levels = ['Calm', 'Frustrated', 'Very angry']
  const probs = levels.map((_, i) => score.probabilities[String(i) as '0' | '1' | '2'])
  const ox = [282, 396, 510] // 每个选项的起点
  const OW = 78
  const CW = 28
  const close = ox.map(x => x + OW + 2 + CW / 2) // </opt> 的中心
  const DX = 700
  const BASE = 318
  return (
    <Figure
      title="pointer head 的读出"
      desc="一个问题的 token 依次是指令、三个选项和 <decide>。每个选项末尾的 </opt> 与 <decide> 的 hidden state 各过一个线性层，做点积得到三个 logit，softmax 后是三个选项的概率。"
      height={340}
      caption="结构来自 Kev 源码；三根柱子是 Jev 对示例工单“客户有多不满”这个 Score 问题返回的概率。"
    >{arrow => <>
      <Cell x={10} y={24} w={70} h={36} rx={6} fill="teal" label="state" />
      <Cell x={84} y={24} w={190} h={36} rx={6} fill="blue" label={<>{mono('<q>')} How frustrated…?</>} />
      {ox.map((x, i) => <g key={i}>
        <Cell x={x} y={24} w={OW} h={36} rx={6} fill="blue" label={levels[i]} small />
        <Cell x={x + OW + 2} y={24} w={CW} h={36} rx={6} fill="blue" label="/o" focus />
      </g>)}
      <Cell x={DX - 28} y={24} w={56} h={36} rx={6} fill="purple" label="decide" small focus />
      <text x={10} y="84" className="fk-muted" textAnchor="start">最后一层的 hidden state</text>

      {close.map((x, i) => <g key={i}>
        <Link d={`M${x} 64 V110`} hue="blue" arrow={arrow} />
        <rect x={x - 26} y={110} width={52} height={28} rx="5" className="fk-fill-blue" />
        <text x={x} y={124.5} className="fk-cell-text">k(·)</text>
        <Link d={`M${x} 138 V172`} hue="blue" arrow={arrow} />
        <circle cx={x} cy={184} r="12" className="fk-fill-purple" />
        <text x={x} y={184.5} className="fk-cell-text" style={{ fontSize: 16 }}>·</text>
        <Link d={`M${x} 196 V${BASE - 96}`} hue="purple" arrow={arrow} />
        <rect x={x - 22} y={BASE - Math.max(probs[i] * 78, 2)} width={44} height={Math.max(probs[i] * 78, 2)} rx="2" className="fk-fill-purple" />
        <text x={x} y={BASE - Math.max(probs[i] * 78, 2) - 10} className="fk-stat">{probs[i].toFixed(2)}</text>
        <text x={x} y={BASE + 14} className="fk-muted">{levels[i]}</text>
      </g>)}
      <line x1={close[0] - 40} y1={BASE} x2={close[2] + 40} y2={BASE} className="fk-rule" />

      <Link d={`M${DX} 64 V110`} hue="purple" arrow={arrow} />
      <rect x={DX - 26} y={110} width={52} height={28} rx="5" className="fk-fill-purple" />
      <text x={DX} y={124.5} className="fk-cell-text">q(·)</text>
      <Link d={`M${DX} 138 V184 H${close[2] + 12}`} hue="purple" arrow={arrow} />
      <Link d={`M${close[2] - 12} 184 H${close[1] + 12}`} hue="purple" arrow={arrow} />
      <Link d={`M${close[1] - 12} 184 H${close[0] + 12}`} hue="purple" arrow={arrow} />

      <TexLabel x={150} y={150} w={300} h={46} source={String.raw`z_i=\frac{k(h_{\text{/o}_i})\cdot q(h_{\text{decide}})}{\sqrt{256}}`} />
      <TexLabel x={150} y={212} w={300} h={40} source={String.raw`p=\operatorname{softmax}(z/T)`} />
      <text x={150} y={262} className="fk-muted">Score 的返回值是档位的期望：</text>
      <text x={150} y={282} className="fk-stat">0×{probs[0].toFixed(2)} + 1×{probs[1].toFixed(2)} + 2×{probs[2].toFixed(2)} = {score.score.toFixed(2)}</text>
    </>}</Figure>
  )
}

type Option = ComposeOption<LineSeriesOption | GridComponentOption | TitleComponentOption | TooltipComponentOption>

// 图 4：Jev 服务端耗时随 state 长度与问题数的变化（Archer Hume 的测量）
export function JevLatency() {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const styles = getComputedStyle(ref.current!)
    const get = (n: string) => styles.getPropertyValue(n).trim()
    const text = get('--color-text'), muted = get('--color-muted'), border = get('--color-border')
    const axis = (gridIndex: number, name: string, max: number) => ({
      gridIndex, type: 'value' as const, name, nameLocation: 'middle' as const, nameGap: 28, min: 0, max,
      nameTextStyle: { color: muted, fontSize: 12 }, axisLine: { lineStyle: { color: border } }, axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 12, formatter: (v: number) => v >= 1000 ? `${v / 1000}k` : String(v) }, splitLine: { show: false },
    })
    const yAxis = (gridIndex: number) => ({
      gridIndex, type: 'value' as const, name: gridIndex === 0 ? '服务端耗时（ms）' : '', min: 0, max: 700,
      nameTextStyle: { color: muted, fontSize: 12, align: 'left' as const }, axisLabel: { color: muted, fontSize: 12 }, splitLine: { lineStyle: { color: border, type: 'dashed' as const } },
    })
    const line = (i: number, points: number[][], color: string): LineSeriesOption => ({
      type: 'line', xAxisIndex: i, yAxisIndex: i, data: points, symbol: 'circle', symbolSize: 6, lineStyle: { color, width: 2 }, itemStyle: { color },
    })
    const title = (t: string, left: string | number) => ({ text: t, left, top: 4, textStyle: { color: text, fontSize: 13, fontWeight: 600 as const } })
    const option: Option = {
      animation: false,
      tooltip: { trigger: 'axis' },
      title: [title('state 变长，1 个问题', 52), title('问题变多，state 约 360 tokens', '55%')],
      grid: [{ left: 52, right: '53%', top: 58, bottom: 44 }, { left: '55%', right: 16, top: 58, bottom: 44 }],
      xAxis: [axis(0, 'state 的 token 数', 30000), axis(1, '问题数', 1500)],
      yAxis: [yAxis(0), yAxis(1)],
      series: [line(0, data.hume.state, get('--fk-teal')), line(1, data.hume.questions, get('--fk-blue'))],
    }
    const chart = init(ref.current!, undefined, { renderer: 'svg' })
    chart.setOption(option)
    const resize = new ResizeObserver(() => chart.resize())
    resize.observe(ref.current!)
    return () => { resize.disconnect(); chart.dispose() }
  }, [])
  return <figure className="spec-chart-figure">
    <div ref={ref} className="spec-chart" style={{ height: 300 }} role="img" aria-label="Jev 服务端耗时：state 从 360 增加到 29,835 tokens，耗时从 57.5 ms 增加到 218 ms；问题从 1 个增加到 1,500 个，耗时从 86.5 ms 增加到 610 ms，100 个以内基本不变。" />
    <figcaption>数据来自 Archer Hume 对 jev-1.13.0 的测量：响应头里的上游服务耗时，每个点是 8 次请求的中位数。</figcaption>
  </figure>
}
