import './RmsNormTriton.css'

export function RmsNormTritonGraph() {
  return <figure className="triton-reduction">
    <svg viewBox="0 0 820 282" role="img" aria-labelledby="triton-reduction-title triton-reduction-desc">
      <title id="triton-reduction-title">4096 × 8192：融合 dx 与 dgamma，共享输入读取</title>
      <desc id="triton-reduction-desc">compile 的两个 kernel 分别读取 x、g。手写 Triton 的第一个 kernel 用 128 个 program、每个处理 32 行，计算 dx 并累加 dgamma partial；第二个 kernel 汇总 4 MiB 的 FP32 partial，得到 dgamma。</desc>
      <defs><marker id="triton-reduction-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M1 1 L7 4 L1 7" fill="none" stroke="#98a2aa" strokeWidth="1.2" />
      </marker></defs>
      <text x="8" y="68" className="reduction-label">compile</text>
      <g><rect x="104" y="18" width="84" height="96" rx="4" className="reduction-input" />
        <text x="146" y="62">x, g</text><text x="146" y="80" className="reduction-detail">分别读取</text></g>
      <path d="M188 41 H238 M188 91 H238" className="reduction-edge" />
      <g><rect x="245" y="18" width="210" height="44" rx="4" className="reduction-main" />
        <text x="350" y="45">kernel 0 · dgamma</text></g>
      <g><rect x="245" y="72" width="210" height="44" rx="4" className="reduction-main" />
        <text x="350" y="99">kernel 1 · dx</text></g>
      <path d="M455 40 H733 M455 94 H733" className="reduction-edge" />
      <g><rect x="740" y="21" width="72" height="38" rx="4" className="reduction-output" />
        <text x="776" y="45">dgamma</text></g>
      <g><rect x="740" y="75" width="72" height="38" rx="4" className="reduction-output" />
        <text x="776" y="99">dx</text></g>
      <text x="8" y="212" className="reduction-label">Triton</text>
      <g><rect x="104" y="177" width="84" height="72" rx="4" className="reduction-input" />
        <text x="146" y="208">x, g</text><text x="146" y="227" className="reduction-detail">共享读取</text></g>
      <path d="M188 211 H238" className="reduction-edge" />
      <g><rect x="245" y="163" width="210" height="101" rx="4" className="reduction-main" />
        <text x="350" y="186">kernel 0 · 融合梯度</text>
        <text x="350" y="212">dx + dgamma partial</text>
        <text x="350" y="235" className="reduction-detail">128 programs · 每组 32 行</text>
        <text x="350" y="251" className="reduction-detail">每行处理完整的 8192 列</text></g>
      <path d="M455 188 H733 M455 245 H478" className="reduction-edge" />
      <g><rect x="740" y="169" width="72" height="38" rx="4" className="reduction-output" />
        <text x="776" y="193">dx</text></g>
      <g><rect x="485" y="223" width="140" height="46" rx="4" className="reduction-partial" />
        <text x="555" y="241">FP32 partial</text>
        <text x="555" y="258" className="reduction-detail">128 × 8192 · 4 MiB</text></g>
      <path d="M625 246 H643 M712 246 H733" className="reduction-edge" />
      <g><rect x="650" y="223" width="62" height="46" rx="4" className="reduction-main" />
        <text x="681" y="241">归约</text><text x="681" y="258" className="reduction-detail">kernel 1</text></g>
      <g><rect x="740" y="227" width="72" height="38" rx="4" className="reduction-output" />
        <text x="776" y="251">dgamma</text></g>
    </svg>
    <figcaption>两种实现均为 2 个 kernel；融合后，第二个 kernel 只读取 partial。</figcaption>
  </figure>
}
