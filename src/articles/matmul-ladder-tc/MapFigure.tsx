import { Badge, Cell, Figure, Link, type Fill } from '../../figure-kit'

// 存储层级图（同视频第六版的地图）：显存 → L2 → shared memory → 寄存器 → Tensor Core，结果再从寄存器写回显存；七步各标在作用的一层。
// focus 为某一步时只高亮该步，其余步骤淡出；不传时七步全部标出。
type Node = 'hbm' | 'l2' | 'smem' | 'reg' | 'tc'
const Y = 74, H = 46, W = 118
const NODES: Record<Node, [number, string, Fill]> = {
  hbm: [16, '显存', 'amber'],
  l2: [164, 'L2', 'amber'],
  smem: [326, 'shared memory', 'green'],
  reg: [474, '寄存器', 'purple'],
  tc: [622, 'Tensor Core', 'teal'],
}
const x = (n: Node) => NODES[n][0]
const mid = (a: Node, b: Node) => (x(a) + W + x(b)) / 2

// [步骤号, 标签, 编号的 x, y]；标签接在编号右侧
const STEPS: [number, string, number, number][] = [
  [1, 'swizzle', x('smem') + W - 4, Y - 16],
  [2, 'cp.async、pipeline', x('l2') + W - 22, Y - 16],
  [3, 'wgmma', x('reg') + 22, 22],
  [4, 'warp specialization', x('smem') + 104, Y + H + 26],
  [5, 'tile 分组', x('hbm') + W - 26, Y - 16],
  [6, 'cluster 多播', x('l2') + W - 22, Y + H + 26],
  [7, 'TMA store', x('l2') + 60, 206],
]

export function MapFigure() {
  const wbY = 186
  return <Figure title="存储层级与七步优化" desc="数据从显存经 L2、shared memory、寄存器送到 Tensor Core，结果再从寄存器写回显存。七步各作用于一层：1 swizzle、2 cp.async 与 pipeline、3 wgmma、4 warp specialization 在 SM 内部；5 tile 分组、6 cluster 多播、7 TMA store 在 SM 之外。" height={226}
    caption="灰色底为一个 SM，青色箭头为 wgmma 直接从 shared memory 读操作数，紫色箭头为结果写回。编号为本文七步的顺序。">
    {arrow => <>
      <rect x={308} y={40} width={446} height={126} rx={12} className="fk-band-gray" />
      <text x={744} y={156} textAnchor="end" className="fk-strong">一个 SM</text>
      <Link d={`M ${x('hbm') + W} ${Y + H / 2} H ${x('l2')}`} hue="blue" arrow={arrow} />
      <Link d={`M ${x('l2') + W} ${Y + H / 2} H ${x('smem')}`} hue="blue" arrow={arrow} />
      <Link d={`M ${x('smem') + W} ${Y + H / 2} H ${x('reg')}`} hue="blue" arrow={arrow} />
      <Link d={`M ${x('reg') + W} ${Y + H / 2} H ${x('tc')}`} hue="blue" arrow={arrow} />
      {/* wgmma：操作数从 shared memory 直接进 Tensor Core */}
      <Link d={`M ${x('smem') + W / 2} ${Y} V ${Y - 40} H ${x('tc') + W / 2} V ${Y}`} hue="teal" arrow={arrow} />
      {/* 写回：寄存器 → 显存 */}
      <Link d={`M ${x('reg') + W / 2} ${Y + H} V ${wbY} H ${x('hbm') + W / 2} V ${Y + H}`} hue="purple" arrow={arrow} />
      {(Object.keys(NODES) as Node[]).map(k => <Cell key={k} x={x(k)} y={Y} w={W} h={H} fill={NODES[k][2]} label={NODES[k][1]} />)}
      {STEPS.map(([n, t, cx, cy]) => <g key={n}>
        <Badge x={cx} y={cy} n={n} />
        <text x={cx + 13} y={cy} textAnchor="start" className="fk-small fk-strong">{t}</text>
      </g>)}
    </>}
  </Figure>
}
