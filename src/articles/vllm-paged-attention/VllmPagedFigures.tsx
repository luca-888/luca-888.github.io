import { Cell, Figure, Legend, Link, Steps, TexLabel, type Fill } from '../../figure-kit'

// 语义色：已写入 green、预留 amber、内部碎片 red、外部碎片 gray；请求 A / R1 blue，B / R2 purple，共享前缀 teal。
type Slot = { fill: Fill; label?: string }

// 16 × 8 个 slot；每 4 个 slot 之间多留 4px，标出分页方案中的 block 边界。
const slotX = (x0: number, col: number) => x0 + col * 20 + Math.floor(col / 4) * 4
const slotY = (row: number) => 50 + row * 22

function contiguousSlots(): Slot[] {
  const slots: Slot[] = Array.from({ length: 128 }, () => ({ fill: 'gray' }))
  // [请求, 起点, 已写入, 之后会写]，每个请求预留 32 个 slot
  for (const [req, start, written, future] of [['A', 0, 10, 4], ['B', 48, 5, 3], ['C', 80, 20, 6]] as const) {
    for (let i = 0; i < 32; i++) slots[start + i] = i < written ? { fill: 'green', label: req } : { fill: i < written + future ? 'amber' : 'red' }
  }
  return slots
}

function pagedSlots(): Slot[] {
  const slots: Slot[] = Array.from({ length: 128 }, () => ({ fill: 'free' }))
  // [请求, 物理 block 编号, 写入的 token 数]
  const blocks: [string, number, number][] = [
    ['A', 7, 4], ['A', 1, 4], ['A', 13, 2],
    ['B', 20, 4], ['B', 9, 1],
    ['C', 3, 4], ['C', 26, 4], ['C', 16, 4], ['C', 30, 4], ['C', 11, 4],
  ]
  for (const [req, id, used] of blocks) {
    for (let i = 0; i < 4; i++) slots[id * 4 + i] = i < used ? { fill: 'green', label: req } : { fill: 'amber' }
  }
  return slots
}

function SlotGrid({ x0, slots }: { x0: number; slots: Slot[] }) {
  return <g>{slots.map((slot, index) =>
    <Cell key={index} x={slotX(x0, index % 16)} y={slotY(Math.floor(index / 16))} w={18} h={18} rx={2} fill={slot.fill} label={slot.label} small />)}</g>
}

export function VllmAllocationCompare() {
  return (
    <Figure
      title="同一组请求在连续预留与分页下的显存占用"
      desc="显存共 128 个 token slot。连续预留时 A、B、C 各按最大长度 32 预留一段，只写入 35 个 slot，其余为预留、内部碎片和两段各 16 个 slot 的外部碎片，新请求找不到 32 个连续 slot。分页时 block 为 4 个 slot，三个请求共占 10 个不相邻的 block，余下 22 个 block 空闲。"
      height={300}
      below={<Legend items={[['green', '已写入的 KV'], ['amber', '预留，之后会写'], ['red', '内部碎片'], ['gray', '外部碎片'], ['free', '空闲 block']]} />}
      caption="构造示例：三个请求 A、B、C 分别已生成到 10、5、20 个 token。左右两侧 slot 相同，只改变分配方式。"
    >{() => <>
      <text x="24" y="26" className="fk-title" textAnchor="start">连续预留</text>
      <text x="100" y="26" className="fk-muted" textAnchor="start">按最大长度 32 预留</text>
      <text x="406" y="26" className="fk-title" textAnchor="start">分页</text>
      <text x="446" y="26" className="fk-muted" textAnchor="start">block = 4 个 slot，按需分配</text>
      <SlotGrid x0={24} slots={contiguousSlots()} />
      <SlotGrid x0={406} slots={pagedSlots()} />
      <text x="24" y="246" className="fk-stat" textAnchor="start">写入 35 / 128 个 slot</text>
      <text x="24" y="266" className="fk-muted" textAnchor="start">空闲 32 个，但最长连续只有 16：新请求放不下</text>
      <text x="406" y="246" className="fk-stat" textAnchor="start">占用 10 个 block，写入 35 / 40 个 slot</text>
      <text x="406" y="266" className="fk-muted" textAnchor="start">空闲 22 个 block，任何请求都能用</text>
    </>}</Figure>
  )
}

// 物理显存画成一条线性地址：32 个 slot，每 4 个一组为一个 block。
const B = 4
const physX = (slot: number) => 84 + slot * 19 + Math.floor(slot / B) * 6
const logicalX = (x0: number, block: number, i = 0) => x0 + block * 124 + i * 28

export function VllmBlockTable() {
  const requests = [
    { name: 'A', x0: 84, table: [7, 1, 3], length: 10, hue: 'blue' as const },
    { name: 'B', x0: 494, table: [4, 6], length: 5, hue: 'purple' as const },
  ]
  const owner = new Map<number, { pos: number; hue: Fill; focus: boolean }>()
  for (const { name, table, length, hue } of requests) {
    table.forEach((block, logical) => {
      for (let i = 0; i < B; i++) {
        const pos = logical * B + i
        if (pos < length) owner.set(block * B + i, { pos, hue, focus: name === 'A' && pos === 9 })
      }
    })
  }
  return (
    <Figure
      title="block table 把逻辑 block 翻译成物理地址"
      desc="请求 A 有 10 个 token，三个逻辑 block 经 block table 映射到物理 block 7、1、3；请求 B 有 5 个 token，映射到物理 block 4、6。逻辑上连续的 token 在物理显存中分散存放。A 的位置 9 位于逻辑 block 2 的偏移 1，查表得到物理 block 3，对应 slot 13。"
      height={300}
      below={<Steps items={[
        ['token 位置', String.raw`\text{pos} = 9`],
        ['逻辑 block 与偏移', String.raw`\lfloor 9/4 \rfloor = 2,\ \ 9 \bmod 4 = 1`],
        ['查 block table', String.raw`\texttt{block\_table}[2] = 3`],
        ['物理 slot', String.raw`3 \times 4 + 1 = 13`],
      ]} />}
      caption="构造示例，B = 4。逻辑上连续的 token 在物理显存中分散存放；描边为请求 A 本步新生成的 token 及其翻译路径。"
    >{arrow => <>
      <text x="20" y="53" className="fk-muted" textAnchor="start">逻辑 block</text>
      <text x="20" y="128" className="fk-muted" textAnchor="start">block table</text>
      <text x="20" y="239" className="fk-muted" textAnchor="start">物理显存</text>
      {requests.map(({ name, x0, table, length, hue }) => <g key={name}>
        <text x={x0} y="22" className={`fk-title fk-text-${hue}`} textAnchor="start">请求 {name}</text>
        {table.map((block, logical) => {
          const center = logicalX(x0, logical) + 54
          const focus = name === 'A' && logical === 2
          const target = physX(block * B) + 36
          return <g key={logical}>
            {Array.from({ length: B }, (_, i) => {
              const pos = logical * B + i
              return <Cell key={i} x={logicalX(x0, logical, i)} y={40} w={24} h={26} fill={pos < length ? hue : 'empty'}
                label={pos < length ? pos : undefined} focus={name === 'A' && pos === 9} />
            })}
            <text x={center} y="80" className="fk-muted fk-small">逻辑 {logical}</text>
            <Link d={`M${center} 88 V104`} hue={hue} arrow={arrow} />
            <rect x={center - 22} y="112" width="44" height="30" rx="4" className={`fk-outline fk-line-${hue} ${focus ? 'fk-outline-focus' : ''}`} />
            <text x={center} y="127.5" className="fk-mono fk-strong">{block}</text>
            <Link d={`M${center} 146 C${center} 185 ${target} 175 ${target} 216`} hue={hue} arrow={arrow} focus={focus} />
          </g>
        })}
      </g>)}
      {Array.from({ length: 32 }, (_, slot) => {
        const cell = owner.get(slot)
        return <Cell key={slot} x={physX(slot)} y={226} w={17} h={26} rx={2} fill={cell?.hue ?? 'empty'} label={cell?.pos} small focus={cell?.focus} />
      })}
      {Array.from({ length: 8 }, (_, block) => <text key={block} x={physX(block * B) + 36} y="266" className="fk-mono fk-muted fk-small">#{block}</text>)}
      <text x={physX(13) + 8.5} y="288" className="fk-mono fk-strong fk-small">slot 13</text>
    </>}</Figure>
  )
}

export function VllmPrefixSharing() {
  const pitch = 55
  const blockX = (block: number) => 70 + block * (4 * pitch + 9)
  const cellX = (block: number, i: number) => blockX(block) + i * pitch
  const center = (block: number) => blockX(block) + 2 * pitch - 3
  const shared = [['You', 'are', 'a', 'math'], ['tutor', '.', 'Be', 'brief']]
  const r1 = ['What', 'is', 'KV', 'cache']
  const r2 = ['Why', 'use', 'paging']
  const hashes = [String.raw`h_0 = H(\varnothing,\ t_{0:4})`, String.raw`h_1 = H(h_0,\ t_{4:8})`, String.raw`h_2 = H(h_1,\ t_{8:12})`]
  return (
    <Figure
      title="两个请求共享 system prompt 的 block"
      desc="R1 与 R2 的前 8 个 token 相同，对应两个写满的 block，只存一份，物理 block 5 和 2 的 ref_cnt 为 2，hash 为 h0、h1。R1 之后写满了自己的 block 3，hash h2 由 h1 与本块 token 算出。R2 的第三个 block 只写了 3 个 token，没有 hash，不参与缓存。"
      height={250}
      caption="构造示例，B = 4。hash 链式包含之前的全部 token：R2 若之后写满第三个 block，会得到与 R1 不同的 hash，因为 token 不同。"
    >{arrow => <>
      {hashes.map((hash, block) => <g key={block}>
        <TexLabel x={center(block)} y={25} source={hash} hue={block === 2 ? 'blue' : 'teal'} />
        {block > 0 && <Link d={`M${blockX(block) - 40} 25 H${blockX(block) + 10}`} arrow={arrow} />}
      </g>)}
      <text x="20" y="75" className="fk-title fk-text-blue" textAnchor="start">R1</text>
      <text x="20" y="147" className="fk-title fk-text-purple" textAnchor="start">R2</text>
      {shared.map((tokens, block) => tokens.map((token, i) => <Cell key={`${block}-${i}`} x={cellX(block, i)} y={56} w={52} h={110} rx={4} fill="teal" label={token} />))}
      {r1.map((token, i) => <Cell key={i} x={cellX(2, i)} y={56} w={52} h={38} rx={4} fill="blue" label={token} />)}
      {Array.from({ length: 4 }, (_, i) => <Cell key={i} x={cellX(2, i)} y={128} w={52} h={38} rx={4} fill={i < r2.length ? 'purple' : 'empty'} label={r2[i]} />)}
      <text x={center(2)} y="111" className="fk-mono fk-small fk-muted">物理 #3 · ref_cnt 1</text>
      <text x={center(0)} y="186" className="fk-mono fk-small fk-strong">物理 #5 · ref_cnt 2</text>
      <text x={center(1)} y="186" className="fk-mono fk-small fk-strong">物理 #2 · ref_cnt 2</text>
      <text x={center(2)} y="186" className="fk-mono fk-small fk-muted">物理 #6 · ref_cnt 1</text>
      <path d={`M${blockX(0) + 4} 196 V200 H${blockX(1) + 4 * pitch - 7} V196`} className="fk-link fk-line-teal" />
      <text x={blockX(0) + 4 * pitch + 2} y="210" className="fk-text-teal fk-small">两个请求的 block table 都指向这两个 block，只存一份</text>
      <text x={center(2)} y="210" className="fk-muted fk-small">未写满，没有 hash，不缓存</text>
    </>}</Figure>
  )
}
