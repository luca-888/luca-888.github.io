import type { ReactNode } from 'react'
import { Band, Cell, Figure, Legend, Link, tex } from '../../figure-kit'

// 语义色：blue 为 K，amber 为 V，purple 为 MLA 的 latent（同时充当 K 与 V），red 为 RoPE key，gray 为 query head。
const sub = (base: string, s: string): ReactNode => <>{base}<tspan dy="4" style={{ fontSize: '0.75em' }}>{s}</tspan></>

// 图 1：128 个 query head、head dim 128 时，四种形式每层每 token 缓存的元素数，面积按真实比例
const H = 128
const D = 128
const X0 = 160
const W = 540
const RH = 15 // K、V 各一行的高度；MHA 的 K 行恰好铺满 W
const PXR = W / (H * D) // 每行每个元素的宽度
const GAP = 2

type Scheme = { name: string; note: string; groups: number; cache: number; mla?: boolean }
const SCHEMES: Scheme[] = [
  { name: 'MHA', note: '每个 query head 一组 K/V', groups: H, cache: 2 * H * D },
  { name: 'GQA-8', note: '每 16 个 query head 共享一组', groups: 8, cache: 2 * 8 * D },
  { name: 'MQA', note: '全部 query head 共享一组', groups: 1, cache: 2 * D },
  { name: 'MLA', note: '全部 query head 共享一个 latent', groups: 1, cache: 512 + 64, mla: true },
]

export function KvCachePerLayer() {
  const TOPS = [0, 100, 234, 346] // 各行起点：MHA 的 K/V 紧贴 query head，不需要汇聚区
  return (
    <Figure
      title="四种 attention 形式每层每 token 的 KV cache"
      desc="128 个 query head、head dim 128 时，每层每 token 的缓存元素数：MHA 32768，GQA-8 2048，MQA 256，MLA 576（512 维 latent 加 64 维 RoPE key）。"
      height={TOPS[3] + 110}
      below={<Legend items={[['gray', 'query head'], ['blue', 'K'], ['amber', 'V'], ['purple', 'latent，同时充当 K 与 V'], ['red', 'RoPE key']]} />}
      caption="缓存块的面积与元素数成正比，四行同一比例：MHA 的 K、V 各铺满一行，MQA 只有 MHA 的 1/128。MLA 只缓存一个低维向量（latent）和一个 RoPE key，维度取 DeepSeek-V3 的配置：latent 512 维，RoPE key 64 维；为便于比较，四行都按每个 head 的 key、value 为 128 维计算。"
    >{() => SCHEMES.map((s, r) => {
      const y = TOPS[r]
      const qy = y + 4, qh = 20, cy = y + 64
      const span = W / s.groups
      const kvW = D * PXR // 一组 K（或 V）的宽度
      return <g key={s.name}>
        <text x={X0 - 16} y={qy + 10} className="fk-strong" textAnchor="end">{s.name}</text>
        <text x={X0 - 16} y={qy + 30} className="fk-muted fk-small" textAnchor="end">{s.note}</text>
        {s.groups === 8
          ? Array.from({ length: 8 }, (_, g) => <Cell key={g} x={X0 + g * span + GAP} y={qy} w={span - 2 * GAP} h={qh} fill="gray" label="16" small />)
          : <Cell x={X0} y={qy} w={W} h={qh} fill="gray" label="128 个 query head" small />}
        {s.groups === H
          ? <>
              <Cell x={X0} y={cy - 26} w={W} h={RH} fill="blue" label="K：128 个 head × 128 维" small />
              <Cell x={X0} y={cy - 26 + RH + GAP} w={W} h={RH} fill="amber" label="V：128 个 head × 128 维" small />
            </>
          : Array.from({ length: s.groups }, (_, g) => {
              const mid = X0 + (g + 0.5) * span
              const bw = s.mla ? 512 / 2 * PXR : kvW
              const bx = mid - bw / 2
              return <g key={g}>
                <polygon points={`${X0 + g * span + GAP},${qy + qh + 2} ${X0 + (g + 1) * span - GAP},${qy + qh + 2} ${bx + bw + 1},${cy - 3} ${bx - 1},${cy - 3}`} className="fk-band-gray" />
                {s.mla
                  ? <>
                      <rect x={bx} y={cy} width={bw} height={2 * RH + GAP} className="fk-fill-purple" />
                      <rect x={bx + bw} y={cy} width={64 / 2 * PXR} height={2 * RH + GAP} className="fk-fill-red" />
                    </>
                  : <>
                      <rect x={bx} y={cy} width={bw} height={RH} className="fk-fill-blue" />
                      <rect x={bx} y={cy + RH + GAP} width={bw} height={RH} className="fk-fill-amber" />
                    </>}
              </g>
            })}
        {s.groups === 1 && (s.mla
          ? <>
              <text x={X0 + W / 2 + 16} y={cy + 7} className="fk-muted fk-text-purple" textAnchor="start">latent 512 维</text>
              <text x={X0 + W / 2 + 16} y={cy + RH + 9} className="fk-muted" textAnchor="start">+ RoPE key 64 维</text>
            </>
          : <>
              <text x={X0 + W / 2 + 12} y={cy + RH / 2} className="fk-muted" textAnchor="start">K 128 维</text>
              <text x={X0 + W / 2 + 12} y={cy + RH * 1.5 + GAP} className="fk-muted" textAnchor="start">V 128 维</text>
            </>)}
        {s.groups === 8 && <text x={X0 + W / 2} y={cy + 2 * RH + 16} className="fk-muted">每组 K、V 各 128 维，共 8 组</text>}
        <text x={X0 + W + 14} y={(s.groups === H ? cy - 26 : cy) + RH + 1} className="fk-stat" textAnchor="start">{s.cache}</text>
      </g>
    })}</Figure>
  )
}

// 图 2：一个 query head 与一个历史 token 的分数，两种算法的计算顺序
export function MlaAbsorb() {
  const CW = 84, CH = 44
  return (
    <Figure
      title="MLA 的矩阵吸收"
      desc="左侧先用上投影矩阵把缓存的 512 维 latent 还原成 128 维 key，再与 query 点积；右侧先把 query 乘以上投影矩阵的转置，得到 512 维向量，再直接与缓存的 latent 点积。两者结果相同。"
      height={318}
      below={<div className="fk-tex" style={{ marginTop: 8 }} dangerouslySetInnerHTML={tex(String.raw`q_i^\top\big(W^{UK}_i\,c_j\big)\;=\;\big(W^{UK\top}_i\,q_i\big)^{\!\top} c_j`)} />}
      caption="i 是 query head，j 是历史 token；W 的上标 UK 表示 key 的上投影，每个 head 一份，形状为 128 × 512。展开时每个历史 token 都要乘一次 W；吸收后每个 head 每步只乘一次，缓存的 latent 原样参与点积。value 一侧同理：先对 latent 加权求和，再乘 value 的上投影。"
    >{arrow => <>
      <Band x={8} y={4} w={364} h={266} hue="gray" title="展开：先还原 key" />
      <Band x={388} y={4} w={364} h={266} hue="purple" title="吸收：先变换 query" />

      {/* 左：c_j → W^UK → k → · q */}
      <Cell x={30} y={56} w={CW} h={CH} rx={6} fill="purple" label={sub('c', 'j')} sub="512 · 缓存" focus />
      <Cell x={150} y={56} w={CW} h={CH} rx={6} fill="gray" label={<>× W<tspan dy="-5" style={{ fontSize: '0.7em' }}>UK</tspan></>} sub="每个 token 一次" />
      <Cell x={270} y={56} w={CW} h={CH} rx={6} fill="blue" label={sub('k', 'j,i')} sub="128" />
      <Link d={`M${30 + CW} 78 H150`} arrow={arrow} hue="purple" focus />
      <Link d={`M${150 + CW} 78 H270`} arrow={arrow} hue="purple" focus />
      <Cell x={150} y={156} w={CW} h={CH} rx={6} fill="gray" label={sub('q', 'i')} sub="128" />
      <circle cx={312} cy={178} r={16} className="fk-fill-ink" />
      <text x={312} y={178} className="fk-cell-text">·</text>
      <Link d={`M312 ${56 + CH} V162`} arrow={arrow} hue="purple" focus />
      <Link d={`M${150 + CW} 178 H296`} arrow={arrow} />
      <text x={190} y={236} className="fk-muted">点积在 128 维上；历史有 L 个 token，</text>
      <text x={190} y={254} className="fk-muted">就要做 L 次 512 × 128 的还原</text>

      {/* 右：q_i → W^UK⊤ → q̃ ; c_j 直接参与 */}
      <Cell x={410} y={156} w={CW} h={CH} rx={6} fill="gray" label={sub('q', 'i')} sub="128" />
      <Cell x={530} y={156} w={CW} h={CH} rx={6} fill="gray" label={<>× W<tspan dy="-5" style={{ fontSize: '0.7em' }}>UK⊤</tspan></>} sub="每步一次" />
      <Cell x={650} y={156} w={CW} h={CH} rx={6} fill="purple" label={<>q̃<tspan dy="4" style={{ fontSize: '0.75em' }}>i</tspan></>} sub="512" />
      <Link d={`M${410 + CW} 178 H530`} arrow={arrow} />
      <Link d={`M${530 + CW} 178 H650`} arrow={arrow} />
      <Cell x={530} y={56} w={CW} h={CH} rx={6} fill="purple" label={sub('c', 'j')} sub="512 · 缓存" focus />
      <circle cx={692} cy={78} r={16} className="fk-fill-ink" />
      <text x={692} y={78} className="fk-cell-text">·</text>
      <Link d={`M${530 + CW} 78 H676`} arrow={arrow} hue="purple" focus />
      <Link d={`M692 156 V94`} arrow={arrow} />
      <text x={570} y={236} className="fk-muted">点积在 512 维上；不生成 key，</text>
      <text x={570} y={254} className="fk-muted">缓存的 latent 直接读入</text>
    </>}</Figure>
  )
}

// 图 3：真实模型每 token 的 KV cache（BF16，只计 attention 层的 KV；数据为官方 config.json 计算，2026-10-08 读取）
// DeepSeek-V4-Pro：61 层中 30 层每 4 个 token 一个 512 维 entry，31 层每 128 个 token 一个，平均到每个 token 为 7928 B。
type ModelRow = { name: string; kind: string; kib: number; fill: 'teal' | 'purple' | 'ink'; note?: string }
const MODELS: ModelRow[] = [
  { name: 'Llama-3.1-70B', kind: 'GQA · 80 层 × 2048', kib: 320, fill: 'teal' },
  { name: 'Qwen3-235B-A22B', kind: 'GQA · 94 层 × 1024', kib: 188, fill: 'teal' },
  { name: 'MiniMax-M3', kind: 'GQA · 60 层 × 1024', kib: 120, fill: 'teal' },
  { name: 'GLM-5', kind: 'MLA · 78 层 × 576', kib: 87.75, fill: 'purple' },
  { name: 'DeepSeek-V3', kind: 'MLA · 61 层 × 576', kib: 68.625, fill: 'purple' },
  { name: 'Kimi-K2', kind: 'MLA · 61 层 × 576', kib: 68.625, fill: 'purple' },
  { name: 'Kimi-K3', kind: 'MLA · 24 层 × 576', kib: 27, fill: 'purple', note: '另 69 层为线性 attention' },
  { name: 'DeepSeek-V4-Pro', kind: 'shared-KV MQA · 61 层 × 512', kib: 7928 / 1024, fill: 'ink', note: '每 4 或 128 个 token 一个 entry' },
]

export function KvCacheModels() {
  const X0 = 210, PX = 1.4, BH = 26, PITCH = 40
  const rows = MODELS.length
  const fmt = (v: number) => (v >= 10 ? (Number.isInteger(v) ? String(v) : v.toFixed(1)) : v.toFixed(1))
  return (
    <Figure
      title="真实模型每 token 的 KV cache"
      desc={`BF16 下每 token 的 KV cache：${MODELS.map(m => `${m.name} ${fmt(m.kib)} KiB`).join('，')}。`}
      height={rows * PITCH + 10}
      below={<Legend items={[['teal', 'GQA'], ['purple', 'MLA'], ['ink', 'shared-KV MQA']]} />}
      caption="由各模型官方 config.json 计算，统一按 BF16（每个元素 2 字节），不含 DSA indexer 的 key 与线性 attention 的固定大小状态。DeepSeek-V4-Pro 按压缩后的 entry 平均到每个 token，不含每层固定 128 个 token 的 sliding window；它实际以 FP8 与 BF16 混合存储，约为图中的 56%。右侧为一条 128K token 的请求所需的 KV cache。"
    >{() => <>
      {MODELS.map((m, i) => {
        const y = 4 + i * PITCH
        return <g key={m.name}>
          <text x={X0 - 12} y={y + BH / 2 - 7} className="fk-strong" textAnchor="end">{m.name}</text>
          <text x={X0 - 12} y={y + BH / 2 + 9} className="fk-muted fk-small" textAnchor="end">{m.kind}</text>
          <Cell x={X0} y={y} w={Math.max(m.kib * PX, 3)} h={BH} rx={3} fill={m.fill} />
          <text x={X0 + Math.max(m.kib * PX, 3) + 10} y={y + BH / 2} className="fk-stat" textAnchor="start">
            {fmt(m.kib)} KiB<tspan className="fk-muted" dx="10">128K：{fmt(m.kib / 8)} GiB{m.note ? `，${m.note}` : ''}</tspan>
          </text>
        </g>
      })}
    </>}</Figure>
  )
}
