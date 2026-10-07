import { Cell, Figure, Legend, Link, type Fill } from '../../figure-kit'

// 语义色（三张图一致）：已缓存 / 运行中的 block blue，命中复用 teal，需要新算 purple，
// 空闲但仍可命中 green，被淘汰 red，未写满或空闲无 hash 为描边。

const TW = 26 // token 格宽
const TG = 3 // 同一 block 内 token 间距
const BG = 16 // block 间距
const BLOCK_W = 4 * TW + 3 * TG
const tokenX = (x0: number, i: number) => x0 + Math.floor(i / 4) * (BLOCK_W + BG) + (i % 4) * (TW + TG)
const blockX = (x0: number, b: number) => x0 + b * (BLOCK_W + BG)
const blockMid = (x0: number, b: number) => blockX(x0, b) + BLOCK_W / 2

// 描边格子的文字用正文色，实心格子用白字。
function Token({ x, y, w = TW, h = 28, fill, label, focus }: { x: number; y: number; w?: number; h?: number; fill: Fill; label: string; focus?: boolean }) {
  const outline = fill === 'free' || fill === 'empty'
  return <g>
    <Cell x={x} y={y} w={w} h={h} fill={fill} label={outline ? undefined : label} focus={focus} />
    {outline && <text x={x + w / 2} y={y + h / 2 + 0.5} className="fk-muted">{label}</text>}
  </g>
}

// 图 1：链式 hash 的逐块查找
const A_TOKENS = 'abcdefghijklmn'.split('')
const B_TOKENS = 'abcdefghijxyz'.split('')
const PHYS = [3, 7, 1]
const HX = 112

export function PrefixHashChain() {
  const sub = ['₀', '₁', '₂']
  return (
    <Figure
      title="新请求按链式 hash 逐块查找已缓存的 block"
      desc="block 大小为 4。请求 A 有 14 个 token，前三个 block 写满，各自的 hash h0、h1、h2 依次以前一个 hash 为输入，登记到 hash 表中，最后一块只有 2 个 token，不缓存。请求 B 与 A 的前 10 个 token 相同：前两块的 hash 与 h0、h1 相同，命中；第三块只有前两个 token 相同，hash 不同，查找在这里停止，i、j 也要重算。"
      height={272}
      below={<Legend items={[['blue', 'A 已缓存的 block'], ['teal', 'B 命中复用'], ['purple', 'B 需要计算'], ['empty', '未写满，不缓存']]} />}
      caption="构造示例：block 大小为 4，字母代表 token。表中的 #3、#7、#1 是物理 block 编号，B 命中后直接引用这两个物理 block。"
    >{arrow => <>
      <text x="0" y="72" className="fk-strong" textAnchor="start">请求 A</text>
      <text x="0" y="148" className="fk-strong" textAnchor="start">hash 表</text>
      <text x="0" y="224" className="fk-strong" textAnchor="start">请求 B</text>

      {/* A：三块写满并登记，h 链相连 */}
      {[0, 1, 2].map(b => <text key={b} x={blockMid(HX, b)} y="30" className="fk-strong">h{sub[b]}</text>)}
      {[0, 1].map(b => <Link key={b} d={`M${blockMid(HX, b) + 16} 30 H${blockMid(HX, b + 1) - 16}`} arrow={arrow} />)}
      <text x={blockMid(HX, 3)} y="30" className="fk-muted">未写满</text>
      {A_TOKENS.map((t, i) => <Token key={i} x={tokenX(HX, i)} y={58} fill={i < 12 ? 'blue' : 'empty'} label={t} />)}
      {[0, 1, 2].map(b => <Link key={b} d={`M${blockMid(HX, b)} 90 V128`} arrow={arrow} />)}

      {PHYS.map((id, b) => <g key={b}>
        <rect x={blockX(HX, b) + 10} y={132} width={BLOCK_W - 20} height={30} rx={3} className="fk-fill-free" />
        <text x={blockMid(HX, b)} y={147.5}>h{sub[b]} → #{id}</text>
      </g>)}

      {/* B：逐块查表，前两块命中，第三块分叉后停止 */}
      {[0, 1].map(b => <Link key={b} d={`M${blockMid(HX, b)} 190 V166`} hue="teal" arrow={arrow} focus />)}
      <Link d={`M${blockMid(HX, 2)} 190 V166`} arrow={arrow} />
      <text x={blockMid(HX, 0) + 30} y="179" className="fk-text-teal fk-small" textAnchor="start">命中</text>
      <text x={blockMid(HX, 1) + 30} y="179" className="fk-text-teal fk-small" textAnchor="start">命中</text>
      <text x={blockMid(HX, 2) + 10} y="179" className="fk-muted" textAnchor="start">h₂′ ≠ h₂，停止</text>
      {B_TOKENS.map((t, i) => <Token key={i} x={tokenX(HX, i)} y={210} fill={i < 8 ? 'teal' : 'purple'} label={t} focus={i === 8 || i === 9} />)}
      <text x={tokenX(HX, 8) + TW + TG / 2} y="256" className="fk-muted fk-small">与 A 相同，仍要重算</text>

      <text x="630" y="72" className="fk-muted" textAnchor="start">写满即算 hash</text>
      <text x="630" y="147" className="fk-muted" textAnchor="start">hash → 物理 block</text>
      <text x="630" y="216" className="fk-stat" textAnchor="start">命中 8 个 token</text>
      <text x="630" y="236" className="fk-muted" textAnchor="start">计算 5 个 token</text>
    </>}</Figure>
  )
}

// 图 2：free queue 的归还与淘汰顺序
type Blk = [number, Fill]
type Moment = { label: string; notes: string[]; tables: [string, Blk[]][]; queue: Blk[]; queueNote?: string }

const MOMENTS: Moment[] = [
  { label: '① A、B 运行', notes: ['B 与 A 共享 0、1'],
    tables: [['A', [[0, 'blue'], [1, 'blue'], [2, 'blue'], [3, 'blue']]], ['B', [[0, 'blue'], [1, 'blue'], [4, 'blue'], [5, 'blue']]]],
    queue: [[6, 'free'], [7, 'free']] },
  { label: '② A 结束', notes: ['逆序归还 3、2', '0、1 仍被 B 引用'],
    tables: [['B', [[0, 'blue'], [1, 'blue'], [4, 'blue'], [5, 'blue']]]],
    queue: [[3, 'free'], [6, 'free'], [7, 'free'], [2, 'green']] },
  { label: '③ B 结束', notes: ['逆序归还 5、4、1、0'],
    tables: [],
    queue: [[5, 'free'], [3, 'free'], [6, 'free'], [7, 'free'], [2, 'green'], [4, 'green'], [1, 'green'], [0, 'green']] },
  { label: '④ C 命中', notes: ['prompt 共 32 token（8 个 block）', '前 12 个与 A 相同'],
    tables: [['C', [[0, 'teal'], [1, 'teal'], [2, 'teal']]]],
    queue: [[5, 'free'], [3, 'free'], [6, 'free'], [7, 'free'], [4, 'green']], queueNote: '0、1、2 从队列中摘出' },
  { label: '⑤ C 分配', notes: ['还需 5 个 block', '从队首依次取走'],
    tables: [['C', [[0, 'teal'], [1, 'teal'], [2, 'teal'], [5, 'purple'], [3, 'purple'], [6, 'purple'], [7, 'purple'], [4, 'red']]]],
    queue: [], queueNote: '（队列已空）' },
]

const ROW_H = 78
const TABLE_X = 204
const QUEUE_X = 480
const CW = 26
const CGAP = 4

export function PrefixFreeQueue() {
  return (
    <Figure
      title="请求结束后 block 回到 free queue 的顺序，以及新请求如何命中与淘汰"
      desc="block 大小为 4，共 8 个 block。A 占 0 到 3，B 与 A 共享 0、1，另占 4、5，3 和 5 未写满。A 结束时逆序归还：3 没有 hash，放到队首；2 有 hash，放到队尾；0、1 仍被 B 引用。B 结束时 5 放到队首，4、1、0 依次放到队尾。C 的 prompt 共 32 个 token（8 个 block），前 12 个与 A 相同，先命中 0、1、2，把它们从队列中摘出；再从队首取走 5 个 block，队列变空：5、3、6、7 没有 hash，4 带着 B 的 hash，被淘汰后分配给 C。"
      height={30 + ROW_H * MOMENTS.length}
      below={<Legend items={[['blue', '运行中'], ['green', '空闲，仍可命中'], ['free', '空闲，无 hash'], ['teal', '命中，摘出队列'], ['purple', '重新分配'], ['red', '淘汰后重用']]} />}
      caption="构造示例：block 大小为 4，块中数字为物理 block 编号。队列从左（队首，先被分配）到右（队尾）。按 vLLM v0.30.0 的 free_blocks 规则绘制：没有 hash 的 block 放队首，有 hash 的放队尾。"
    >{() => <>
      <text x={TABLE_X} y="10" className="fk-muted" textAnchor="start">block table</text>
      <text x={QUEUE_X} y="10" className="fk-muted" textAnchor="start">free queue：队首 → 队尾</text>
      {MOMENTS.map((m, row) => {
        const y0 = 30 + row * ROW_H
        return <g key={m.label}>
          <text x="0" y={y0 + 12} className="fk-title" textAnchor="start">{m.label}</text>
          {m.notes.map((note, i) => <text key={note} x="0" y={y0 + 34 + i * 18} className="fk-muted" textAnchor="start">{note}</text>)}
          {m.tables.length === 0 && <text x={TABLE_X} y={y0 + 14} className="fk-muted" textAnchor="start">（无运行中的请求）</text>}
          {m.tables.map(([name, blocks], line) => {
            const y = y0 + line * 34
            return <g key={name}>
              <text x={TABLE_X - 22} y={y + 14} className="fk-strong">{name}</text>
              {blocks.map(([id, fill], i) => <Token key={i} x={TABLE_X + i * (CW + CGAP)} y={y} fill={fill} label={String(id)} />)}
            </g>
          })}
          {m.queue.map(([id, fill], i) => <Token key={i} x={QUEUE_X + i * (CW + CGAP)} y={y0} fill={fill} label={String(id)} />)}
          {m.queueNote && <text x={QUEUE_X} y={y0 + (m.queue.length ? 46 : 18)} className="fk-muted" textAnchor="start">{m.queueNote}</text>}
        </g>
      })}
    </>}</Figure>
  )
}

// 图 3：同一组请求在 block hash 表与 radix tree 下的命中
const R1 = 'SSSSSSaaa'.split('')
const R2 = 'SSSSSSbbbb'.split('')
const SW = 22
const SGAP = 2
const SBG = 12
const smallX = (x0: number, i: number) => x0 + i * (SW + SGAP) + Math.floor(i / 4) * (SBG - SGAP)

export function PrefixBlockVsRadix() {
  const rx = 440 // radix tree 起点
  const edgeX = (i: number) => rx + 28 + i * (SW + SGAP)
  const nodeX = edgeX(6) + 12
  const branchX = (i: number) => nodeX + 26 + i * (SW + SGAP)
  return (
    <Figure
      title="同一组请求在 block hash 表与 radix tree 下的命中长度"
      desc="R1 与 R2 共享 6 个 token 的 system prompt S，之后分别是 aaa 与 bbbb。按 block 大小 4 匹配时，第二块分别是 SSaa 与 SSbb，hash 不同，R2 只命中第一块的 4 个 token。radix tree 中，SSSSSS 是一条边，在它的末尾分叉出 aaa 与 bbbb 两条边，R2 命中 6 个 token。"
      height={236}
      below={<Legend items={[['blue', 'R1 已缓存'], ['teal', 'R2 命中复用'], ['purple', 'R2 需要计算']]} />}
      caption="构造示例：R1 先运行并已缓存，R2 随后到达。左侧 block 大小为 4；右侧为 page 大小 1 的 radix tree，R2 插入时在节点处分叉。"
    >{() => <>
      <text x="0" y="12" className="fk-title" textAnchor="start">vLLM：按 block 匹配</text>
      <text x="0" y="62" className="fk-strong" textAnchor="start">R1</text>
      <text x="0" y="132" className="fk-strong" textAnchor="start">R2</text>
      {R1.map((t, i) => <Token key={i} x={smallX(34, i)} y={48} w={SW} fill={i < 8 ? 'blue' : 'empty'} label={t} />)}
      {R2.map((t, i) => <Token key={i} x={smallX(34, i)} y={118} w={SW} fill={i < 4 ? 'teal' : 'purple'} label={t} focus={i === 4 || i === 5} />)}
      <text x={smallX(34, 6)} y="96" className="fk-muted fk-small">SSaa ≠ SSbb</text>
      <text x="0" y="186" className="fk-stat" textAnchor="start">R2 命中 4 / 10 个 token</text>
      <text x="0" y="206" className="fk-muted" textAnchor="start">第二块中的 S S 也要重算</text>

      <text x={rx} y="12" className="fk-title" textAnchor="start">SGLang：radix tree</text>
      <circle cx={rx + 10} cy={96} r={6} className="fk-fill-gray" />
      <Link d={`M${rx + 16} 96 H${edgeX(0) - 2}`} />
      {Array.from({ length: 6 }, (_, i) => <Token key={i} x={edgeX(i)} y={82} w={SW} fill="teal" label="S" />)}
      <circle cx={nodeX} cy={96} r={6} className="fk-fill-gray" />
      <Link d={`M${edgeX(5) + SW + 2} 96 H${nodeX - 6}`} />
      <Link d={`M${nodeX + 4} 91 L${branchX(0) - 4} 58`} />
      <Link d={`M${nodeX + 4} 101 L${branchX(0) - 4} 134`} hue="purple" focus />
      {'aaa'.split('').map((t, i) => <Token key={i} x={branchX(i)} y={44} w={SW} fill="blue" label={t} />)}
      {'bbbb'.split('').map((t, i) => <Token key={i} x={branchX(i)} y={120} w={SW} fill="purple" label={t} />)}
      <text x={branchX(3) + 6} y="58" className="fk-muted" textAnchor="start">R1</text>
      <text x={branchX(4) + 6} y="134" className="fk-muted" textAnchor="start">R2</text>
      <text x={rx} y="186" className="fk-stat" textAnchor="start">R2 命中 6 / 10 个 token</text>
      <text x={rx} y="206" className="fk-muted" textAnchor="start">在第 6 个 token 后分叉，插入新边 bbbb</text>
    </>}</Figure>
  )
}
