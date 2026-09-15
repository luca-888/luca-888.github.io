import { useId } from 'react'
import './RmsNormCompile.css'

export function RmsNormCompileGraph() {
  const id = useId()
  const artifacts = `${import.meta.env.BASE_URL}measurements/rmsnorm-compile/4096x8192/`

  return (
    <figure className="compile-fusion">
      <svg viewBox="0 0 852 298" role="img" aria-labelledby={`${id}-title ${id}-description`}>
        <title id={`${id}-title`}>4096 × 8192：torch.compile 的融合结构</title>
        <desc id={`${id}-description`}>
          Forward 用一个 kernel 计算并返回 y 和 r。Backward 的第一个 kernel 沿 M 归约得到 dgamma，
          第二个 kernel 沿 N 计算 dr，再得到 dx。完整的 FP32 中间张量不再写入显存。
        </desc>
        <defs>
          <marker id={`${id}-arrow`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M1 1 L7 4 L1 7" fill="none" stroke="#98a2aa" strokeWidth="1.2" />
          </marker>
        </defs>
        <text x="18" y="19" className="fusion-heading">Forward · 1 kernel</text>
        <text x="749" y="19" className="fusion-detail fusion-code">M = 4096 · N = 8192</text>
        <g>
          <rect x="18" y="42" width="130" height="64" rx="4" fill="#edf2f5" stroke="#a9bece" />
          <text x="83" y="69">x, gamma</text>
          <text x="83" y="90" className="fusion-detail">BF16 输入</text>
          <rect x="180" y="42" width="516" height="64" rx="4" fill="#edf3ee" stroke="#a4bcac" />
          <text x="438" y="69">K0 · s → r → x_hat → y</text>
          <text x="438" y="90" className="fusion-detail">类型转换、沿 N 归约与缩放融合 · FP32 中间计算</text>
          <rect x="738" y="42" width="100" height="64" rx="4" fill="#edf3ee" stroke="#a4bcac" />
          <text x="788" y="69">y, r</text>
          <text x="788" y="90" className="fusion-detail">BF16 / FP32</text>
        </g>
        <text x="18" y="139" className="fusion-heading">Backward · 2 kernels</text>
        <g>
          <rect x="18" y="189" width="130" height="64" rx="4" fill="#edf2f5" stroke="#a9bece" />
          <text x="83" y="216">x, gamma, g</text>
          <text x="83" y="237">r</text>
          <rect x="180" y="154" width="516" height="58" rx="4" fill="#f5ebed" stroke="#c8aaae" />
          <text x="438" y="178">K0 · g * x_hat → dgamma</text>
          <text x="438" y="199" className="fusion-detail">沿 M 归约 · 汇总所有 token 的贡献</text>
          <rect x="738" y="154" width="100" height="58" rx="4" fill="#f5ebed" stroke="#c8aaae" />
          <text x="788" y="178">dgamma</text>
          <text x="788" y="199" className="fusion-detail">BF16 · [N]</text>
          <rect x="180" y="230" width="516" height="58" rx="4" fill="#edf2f5" stroke="#a9bece" />
          <text x="438" y="254">K1 · dx_hat → dr → dx</text>
          <text x="438" y="275" className="fusion-detail">沿 N 归约 · 合并两条梯度路径</text>
          <rect x="738" y="230" width="100" height="58" rx="4" fill="#edf2f5" stroke="#a9bece" />
          <text x="788" y="254">dx</text>
          <text x="788" y="275" className="fusion-detail">BF16 · [M,N]</text>
        </g>
        {[
          'M148 74 H176', 'M696 74 H734',
          'M148 221 H162 V183 H176', 'M162 221 V259 H176',
          'M696 183 H734', 'M696 259 H734',
        ].map(path => <path key={path} d={path} className="fusion-edge" markerEnd={`url(#${id}-arrow)`} />)}
      </svg>
      <figcaption>
        <span>连线表示数据依赖；Backward 按 K0 → K1 启动。</span>
        <span>生成源码：<a href={`${artifacts}forward.py`} download>Forward</a><a href={`${artifacts}backward.py`} download>Backward</a><a href={`${import.meta.env.BASE_URL}measurements/rmsnorm-compile.zip`} download>完整产物</a></span>
      </figcaption>
    </figure>
  )
}
