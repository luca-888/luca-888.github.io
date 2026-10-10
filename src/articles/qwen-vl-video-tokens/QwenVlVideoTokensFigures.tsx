import type { ReactNode } from 'react'
import { Cell, Figure, Legend, Link } from '../../figure-kit'

// 语义色：视频帧与 video token 为 blue，第二个视频为 teal，时间戳文本为 amber，
// vision_start / vision_end 为 gray，示例路径（第 1 组）用 purple 描边与箭头。

// 一段 1920×1080、30 fps、10 s 视频在 Qwen3-VL-8B 官方配置下的数据（按 transformers 源码公式计算）。
const TOTAL = 300
const SAMPLED = [0, 16, 31, 47, 63, 79, 94, 110, 126, 142, 157, 173, 189, 205, 220, 236, 252, 268, 283, 299]
const STAMPS = ['0.3', '1.3', '2.4', '3.4', '4.5', '5.5', '6.6', '7.6', '8.7', '9.7']

// 图 1：一段视频从帧到 token 序列的完整路径
export function VideoPath() {
  const X0 = 130
  const W = 610
  const fx = (i: number) => X0 + (i / (TOTAL - 1)) * W
  const GW = 55
  const gx = (k: number) => X0 + k * 61
  // 缩放后的帧按 1/8 绘制：1472×832 → 184×104；patch 为 2 px，合并后的 token 为 4 px。
  const FW = 184
  const FH = 104
  const ZY = 196
  const grid = (x: number, step: number, width: number) => <g>
    {Array.from({ length: FW / step - 1 }, (_, i) => <line key={`v${i}`} x1={x + (i + 1) * step} y1={ZY} x2={x + (i + 1) * step} y2={ZY + FH} stroke="#fff" strokeOpacity=".55" strokeWidth={width} />)}
    {Array.from({ length: FH / step - 1 }, (_, i) => <line key={`h${i}`} x1={x} y1={ZY + (i + 1) * step} x2={x + FW} y2={ZY + (i + 1) * step} stroke="#fff" strokeOpacity=".55" strokeWidth={width} />)}
  </g>
  const TY = 384
  return (
    <Figure
      title="一段 10 s、1080p、30 fps 视频在 Qwen3-VL 中变成 token 的路径"
      desc="300 帧中按 2 fps 采样 20 帧，相邻两帧成一组，共 10 组，每组带一个时间戳。第 1 组由第 0 帧和第 16 帧组成，缩放到 1472×832，切成 52×92 个 patch，2×2 合并为 26×46 = 1196 个 token。token 序列中每组为时间戳文本、vision_start、1196 个 video_pad、vision_end。"
      height={420}
      below={<Legend items={[['blue', '视频帧 / video token'], ['amber', '时间戳文本'], ['gray', 'vs、ve：<|vision_start|>、<|vision_end|>']]} />}
      caption="按 Qwen3-VL-8B 的官方 processor 配置与 transformers 源码公式计算。紫色描边为第 1 组的路径；橙色小方块是一个 token 对应的 2×2 个 patch。"
    >{arrow => <>
      {/* 原视频与采样帧 */}
      <text x="0" y="40" className="fk-strong" textAnchor="start">原视频</text>
      <text x="0" y="58" className="fk-muted" textAnchor="start">300 帧 · 10 s</text>
      <rect x={X0} y={30} width={W} height={24} rx="3" className="fk-fill-void" />
      {SAMPLED.map(i => <rect key={i} x={fx(i) - 3} y={30} width={6} height={24} rx="1" className="fk-fill-blue" />)}
      <rect x={fx(0) - 5} y={28} width={fx(16) - fx(0) + 10} height={28} rx="3" className="fk-focus" />
      <text x={fx(0)} y={16} className="fk-muted fk-small" textAnchor="start">第 0、16 帧</text>
      <text x={fx(299)} y={16} className="fk-muted fk-small" textAnchor="end">第 299 帧</text>

      {/* 10 组 */}
      <text x="0" y="114" className="fk-strong" textAnchor="start">10 组</text>
      <text x="0" y="132" className="fk-muted" textAnchor="start">每组 2 帧</text>
      {STAMPS.map((s, k) => {
        const mid = (fx(SAMPLED[2 * k]) + fx(SAMPLED[2 * k + 1])) / 2
        return <g key={k}>
          {k > 0 && <path d={`M${mid} 54 L${gx(k) + GW / 2} 100`} className="fk-link fk-line-muted" style={{ strokeWidth: 1 }} />}
          <Cell x={gx(k)} y={100} w={GW} h={42} fill="blue" label={`组 ${k + 1}`} sub={`${s} s`} focus={k === 0} />
        </g>
      })}
      <Link d={`M${(fx(0) + fx(16)) / 2} 56 L${gx(0) + GW / 2} 100`} hue="purple" arrow={arrow} focus />

      {/* 第 1 组的放大 */}
      <Link d={`M${gx(0) + GW / 2} 144 V166 H${96} V${ZY - 8}`} hue="purple" arrow={arrow} focus />
      <rect x={8} y={ZY - 8} width={FW} height={FH} rx="2" className="fk-fill-blue" opacity=".45" />
      <Cell x={0} y={ZY} w={FW} h={FH} rx={2} fill="blue" label="第 0、16 帧" sub="1472 × 832" fs={13} />
      <text x={FW / 2} y={ZY + FH + 18} className="fk-muted">缩放：宽高为 32 的倍数</text>

      <Link d={`M${FW + 16} ${ZY + FH / 2} H300`} arrow={arrow} />
      <text x={248} y={ZY + FH / 2 - 26} className="fk-muted fk-small">切 patch</text>
      <text x={248} y={ZY + FH / 2 - 12} className="fk-muted fk-small">16×16 像素 × 2 帧</text>
      <rect x={300} y={ZY} width={FW} height={FH} rx="2" className="fk-fill-blue" />
      {grid(300, 2, 0.5)}
      <rect x={300} y={ZY} width={4} height={4} className="fk-fill-amber" />
      <rect x={298} y={ZY - 2} width={8} height={8} className="fk-focus" />
      <text x={300 + FW / 2} y={ZY + FH + 18} className="fk-muted">52 × 92 = 4784 个 patch</text>

      <Link d={`M${300 + FW + 8} ${ZY + FH / 2} H556`} arrow={arrow} />
      <text x={524} y={ZY + FH / 2 - 12} className="fk-muted fk-small">2×2 合并</text>
      <rect x={556} y={ZY} width={FW} height={FH} rx="2" className="fk-fill-blue" />
      {grid(556, 4, 0.6)}
      <rect x={556} y={ZY} width={4} height={4} className="fk-fill-amber" />
      <rect x={554} y={ZY - 2} width={8} height={8} className="fk-focus" />
      <text x={556 + FW / 2} y={ZY + FH + 18} className="fk-muted">26 × 46 = 1196 个 token</text>

      {/* token 序列 */}
      <Link d={`M${556 + FW / 2} ${ZY + FH + 30} V350 H248 V${TY}`} hue="purple" arrow={arrow} focus />
      <Cell x={0} y={TY} w={100} h={34} fill="amber" label="<0.3 seconds>" focus />
      <Cell x={104} y={TY} w={30} h={34} fill="gray" label="vs" />
      <Cell x={138} y={TY} w={220} h={34} fill="blue" label="1196 × <|video_pad|>" focus />
      <Cell x={362} y={TY} w={30} h={34} fill="gray" label="ve" />
      <Cell x={404} y={TY} w={100} h={34} fill="amber" label="<1.3 seconds>" />
      <Cell x={508} y={TY} w={30} h={34} fill="gray" label="vs" />
      <Cell x={542} y={TY} w={130} h={34} fill="blue" label="1196 × …" />
      <Cell x={676} y={TY} w={30} h={34} fill="gray" label="ve" />
      <text x={712} y={TY + 17} className="fk-strong" textAnchor="start">… ×10</text>
    </>}</Figure>
  )
}

// 图 2：一条消息中两个视频时，第二个视频的展开（LlamaFactory 默认配置，按源码公式计算的时间戳）
const STAMPS_A = ['0.2', '1.3', '2.3', '3.4', '4.4', '5.5', '6.5', '7.6', '8.6', '9.7']
const STAMPS_B = ['0.2', '1.3', '2.3', '3.3', '4.4', '5.4', '6.4', '7.5', '8.5', '9.5', '10.6', '11.6', '12.6', '13.7', '14.7']

export function MultiVideo() {
  const X0 = 172
  const CW = 29
  const cx = (k: number) => X0 + k * (CW + 3)
  const strip = (y: number, stamps: string[], fill: 'blue' | 'teal', missing = 0) => <g>
    {stamps.map((s, k) => <Cell key={k} x={cx(k)} y={y} w={CW} h={30} fill={fill} label={s} fs={10} />)}
    {Array.from({ length: missing }, (_, k) => <Cell key={`m${k}`} x={cx(stamps.length + k)} y={y} w={CW} h={30} fill="empty" />)}
  </g>
  const row = (y: number, title: string, sub: string) => <g>
    <text x="0" y={y + 7} className="fk-strong" textAnchor="start">{title}</text>
    <text x="0" y={y + 24} className="fk-muted fk-small" textAnchor="start">{sub}</text>
  </g>
  const AX = cx(15) + 10
  return (
    <Figure
      title="一条消息中有两个视频时，第二个视频展开的组数与时间戳"
      desc="消息为“比较 A 和 B”。A 为 10 s，展开为 10 组；B 为 15 s，应展开为 15 组。修复前 B 的组数取第一个视频的 10，时间戳取消息下标 0 对应的 A 的元数据；修复后按出现顺序取第 2 项，得到 15 组和 B 自己的时间戳。"
      height={240}
      below={<Legend items={[['blue', '来自视频 A 的组数与时间戳'], ['teal', '来自视频 B'], ['empty', '缺少的组']]} />}
      caption="LlamaFactory 默认配置下，同一条消息中两段 1080p、30 fps 的视频，A 为 10 s，B 为 15 s。每格是一组，格中数字为时间戳（秒）。"
    >{() => <>
      <rect x="0" y="0" width="760" height="46" rx="10" className="fk-band-gray" />
      <text x="16" y="23" className="fk-strong" textAnchor="start">消息 0（user）：比较</text>
      <Cell x={168} y={9} w={92} h={28} fill="blue" label="<video> A" />
      <text x="272" y="23" className="fk-strong" textAnchor="start">和</text>
      <Cell x={294} y={9} w={92} h={28} fill="teal" label="<video> B" />
      <text x="744" y="23" className="fk-muted" textAnchor="end">计数器：A 为第 0 个，B 为第 1 个</text>

      {row(72, 'A', '第 0 个 <video>')}
      {strip(70, STAMPS_A, 'blue')}
      <text x={cx(10) + 10} y={85} className="fk-muted" textAnchor="start">10 组，修复前后相同</text>

      {row(132, 'B · 修复前', '组数、时间戳均取 [0]')}
      {strip(130, STAMPS_A, 'blue', 5)}
      <text x={AX} y={137} className="fk-strong" textAnchor="start">10 组</text>
      <text x={(cx(10) + cx(15) - 3) / 2} y={174} className="fk-muted fk-small">缺 5 组，forward 时报错</text>

      {row(202, 'B · 修复后', '三项均取 [1]')}
      {strip(200, STAMPS_B, 'teal')}
      <text x={AX} y={215} className="fk-strong" textAnchor="start">15 组</text>
    </>}</Figure>
  )
}

// 实测与复现的占位
function Placeholder({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ border: '1.5px dashed var(--color-border)', borderRadius: 8, padding: '16px 20px', color: 'var(--color-muted)', margin: '1.5em 0' }}>
      <strong>{title}</strong>{children}
    </div>
  )
}

export const MeasureFastPathCap = () => <Placeholder title="对照待补。">
  在 <code>cap_pixels_per_frame</code> 关闭与开启两种设置下，<code>video_max_pixels</code> 分别取 256 × 256 与 1920 × 1080，比较元数据路径与解码路径算出的 <code>video_grid_thw</code>。只需 CPU。这里将放一张两条路径的对照表。
</Placeholder>

export const MeasureGenerate = () => <Placeholder title="复现待补。">
  在 LlamaFactory 当前主分支上，用含视频的 Qwen3-VL 样本运行 predict（调用 <code>generate()</code>），确认是否因 <code>video_metadata</code> 报错，并对比应用 #10509 之后的结果。需要能加载 Qwen3-VL-2B 的 GPU（如一张 RTX 4090），或 CPU 加随机初始化的小模型。这里将放报错信息与修复前后的对照。
</Placeholder>

export const MeasureTrainInfer = () => <Placeholder title="对照待补。">
  同一组公开视频分别经 LlamaFactory 预处理与 vLLM 输入处理，比较两边的采样帧下标、时间戳与视频 token 数。只需 CPU。这里将放一张对照表和差异的统计。
</Placeholder>
