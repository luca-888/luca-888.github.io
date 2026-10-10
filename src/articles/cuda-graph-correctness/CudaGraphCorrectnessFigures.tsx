import { Band, Cell, Figure, Legend, Link } from '../../figure-kit'

// 语义色：录进图、replay 时执行的 GPU 工作 blue；capture 时直接执行、不进图 red；图外 eager 执行 teal；
// 持久 buffer 与旧数据 gray；本批图像 embedding amber；图中缺失的步骤用虚线空格；图的边界用 purple 底带。

// 图 1：RMSNorm backward 修复前后，哪些操作进了图
export function CaptureMiss() {
  const lane = (y: number, text: string) => <text x="16" y={y} className="fk-muted" textAnchor="start">{text}</text>
  return (
    <Figure
      title="RMSNorm backward 的 capture 与 replay：修复前后"
      desc="修复前 capture 时，backward kernel 发到 stream 0，当下执行但不进图，只有 dw_partial.sum(0) 被录进图。修复前 replay 时，图中只有 sum，读取 buffer 中的旧数据，dX 不被写入。修复后 kernel 与 sum 都在图中，按本批 dy 计算。"
      height={376}
      below={<Legend items={[['blue', '录进图，replay 时执行'], ['red', 'capture 时直接执行，不进图'], ['empty', '图中缺失'], ['gray', 'buffer 中的旧数据'], ['teal', '按本批 dy 写入的结果']]} />}
      caption="自上而下：修复前的 capture、修复前的 replay、修复后的 replay。每段内横向为执行顺序，右列是 sum 读取的 buffer。"
    >{arrow => <>
      <Band x={0} y={0} w={760} h={124} hue="gray" title="修复前 · capture" />
      {lane(58, 'capture stream')}
      {lane(98, 'stream 0')}
      <Cell x={380} y={42} w={170} h={32} fill="blue" label="dw_partial.sum(0)" />
      <Cell x={160} y={82} w={180} h={32} fill="red" label="backward kernel" />
      <text x="590" y="58" className="fk-muted" textAnchor="start">在 capture stream 上，录进图</text>
      <text x="590" y="98" className="fk-muted" textAnchor="start">当下执行，不进图</text>

      <Band x={0} y={138} w={760} h={110} hue="gray" title="修复前 · replay（写入新 dy）" />
      {lane(186, 'capture stream')}
      <Cell x={160} y={170} w={180} h={32} fill="empty" label="backward kernel 不在图中" />
      <Cell x={380} y={170} w={170} h={32} fill="blue" label="dw_partial.sum(0)" />
      <Cell x={590} y={172} w={170} h={28} fill="gray" label="dw_partial：旧数据" />
      <Cell x={590} y={208} w={170} h={28} fill="gray" label="dX：未写入" />
      <Link d="M590 186 H550" arrow={arrow} />

      <Band x={0} y={262} w={760} h={110} hue="blue" title="修复后 · replay" />
      {lane(310, 'capture stream')}
      <Cell x={160} y={294} w={180} h={32} fill="blue" label="backward kernel" />
      <Cell x={380} y={294} w={170} h={32} fill="blue" label="dw_partial.sum(0)" />
      <Cell x={590} y={296} w={170} h={28} fill="teal" label="dw_partial：本批结果" />
      <Cell x={590} y={332} w={170} h={28} fill="teal" label="dX：本批结果" />
      <Link d="M340 310 H380" hue="blue" arrow={arrow} />
      <Link d="M590 310 H550" arrow={arrow} />
    </>}</Figure>
  )
}

// 图 2：EAGLE draft prefill 中，embedding 合并落在 FULL 与 PIECEWISE 图边界的哪一侧
export function GraphBoundary() {
  const row = (y0: number, full: boolean, arrow: string) => <g>
    <Cell x={16} y={y0} w={130} h={44} rx={6} fill="gray" label="input_ids" sub="持久 buffer" />
    <Cell x={16} y={y0 + 56} w={130} h={44} rx={6} fill="amber" label="图像 embedding" sub="本批 encoder 输出" />
    <Cell x={200} y={y0 + 6} w={150} h={84} rx={8} fill={full ? 'blue' : 'teal'} label="embedding 合并" sub={full ? '只录了文本查表' : 'eager，带本批图像'} fs={14} />
    <Cell x={400} y={y0 + 6} w={150} h={84} rx={8} fill="blue" label="draft forward" fs={14} />
    <Cell x={600} y={y0 + 6} w={144} h={84} rx={8} fill={full ? 'blue' : 'teal'} label="logits + 采样" sub={full ? undefined : 'eager'} fs={14} />
    <Link d={`M146 ${y0 + 22} H200`} arrow={arrow} />
    {full
      ? <><path d={`M146 ${y0 + 78} H164`} className="fk-link fk-line-muted" /><Cell x={166} y={y0 + 69} w={18} h={18} fill="red" label="✕" /></>
      : <Link d={`M146 ${y0 + 78} H200`} arrow={arrow} />}
    <Link d={`M350 ${y0 + 48} H400`} hue="blue" arrow={arrow} />
    <Link d={`M550 ${y0 + 48} H600`} hue="blue" arrow={arrow} />
  </g>
  return (
    <Figure
      title="EAGLE draft prefill 的图边界：FULL 与 PIECEWISE"
      desc="FULL 模式把 embedding 合并、draft forward、logits 与采样录成一张图，capture 与 replay 都不传入图像 embedding，合并步骤只有文本查表。PIECEWISE 模式只录 draft forward 中的分段，embedding 合并在图外以 eager 执行，读取本批图像 embedding。"
      height={350}
      below={<Legend items={[['blue', '图内节点'], ['teal', '图外 eager 执行'], ['gray', '持久 buffer'], ['amber', '本批图像 embedding']]} />}
      caption="紫色底带是图的边界。上：FULL；下：PIECEWISE。"
    >{arrow => <>
      <Band x={184} y={26} w={572} h={128} hue="purple" title="FULL：整个 draft prefill 一张图" />
      {row(50, true, arrow)}
      <text x="200" y="166" className="fk-muted fk-small" textAnchor="start">capture 与 replay 都不传入图像 embedding</text>
      <line x1="0" y1="180" x2="760" y2="180" className="fk-rule" />
      <Band x={384} y={194} w={182} h={146} hue="purple" title="PIECEWISE 分段图" />
      {row(240, false, arrow)}
    </>}</Figure>
  )
}
