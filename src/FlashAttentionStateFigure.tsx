import { useId } from 'react'
import { renderToString } from 'katex'
import './FlashAttentionStateFigure.css'

const source = 'https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/csrc/flash_attn/src/'
const mathLabels = {
  scores: String.raw`QK^{\top}`,
  weightedValues: String.raw`\widetilde{P}V`,
  weights: String.raw`\widetilde{P}`,
  output: String.raw`O=U/\ell`,
}
const mathHtml = Object.fromEntries(Object.entries(mathLabels).map(([key, tex]) =>
  [key, { __html: renderToString(tex, { output: 'html', throwOnError: true }) }],
))

export function FlashAttentionStateFlow() {
  const id = useId()
  const arrow = `url(#${id}-arrow)`

  return <figure className="fa-state-figure" id="fa-forward-flow">
    <div className="fa-state-heading">
      <p>一块 Q，两次矩阵乘法</p>
      <span>每轮处理一块 K/V，中间结果留在片上</span>
    </div>
    <div className="fa-state-scroll">
      <svg width="600" height="368" viewBox="0 0 600 368" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>QK 点积、Online Softmax 与指数权重乘 V 的前向流程</title>
        <desc id={`${id}-desc`}>第一次矩阵乘法 gemm 将 QK 点积累加到 FP32 的 acc_s。
          Online Softmax 原地把 acc_s 转成指数权重，更新 row_max 和线程局部 row_sum，后续轮次先缩放旧 acc_o。
          权重转换为 BF16 的 rP 后，第二次矩阵乘法 gemm_rs 与 V 相乘，累加到 FP32 的 acc_o。
          row_max、row_sum、acc_o 跨 K/V tile 保留；遍历完成后合并 row_sum，归一化 acc_o 并写回 BF16 输出。</desc>
        <defs>
          <marker id={`${id}-arrow`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M1 1 L7 4 L1 7" fill="none" stroke="currentColor" strokeWidth="1.2" />
          </marker>
        </defs>

        <rect x="4" y="12" width="168" height="160" rx="4" className="fa-state-dot-product" />
        <rect x="428" y="12" width="168" height="160" rx="4" className="fa-state-weighted-sum" />
        <text x="20" y="37" className="fa-state-step">01　计算分数</text>
        <text x="230" y="37" className="fa-state-step">02　更新权重</text>
        <text x="444" y="37" className="fa-state-step">03　累加输出</text>

        <foreignObject x="16" y="48" width="144" height="64" aria-hidden="true">
          <div className="fa-state-math fa-state-operation" dangerouslySetInnerHTML={mathHtml.scores} />
        </foreignObject>
        <text x="88" y="125" textAnchor="middle" className="fa-state-code">gemm</text>
        <text x="88" y="152" textAnchor="middle" className="fa-state-code">acc_s · FP32</text>

        <text x="300" y="82" textAnchor="middle" className="fa-state-softmax">Online</text>
        <text x="300" y="110" textAnchor="middle" className="fa-state-softmax">Softmax</text>
        <text x="300" y="152" textAnchor="middle">acc_s 原地更新</text>

        <foreignObject x="440" y="48" width="144" height="64" aria-hidden="true">
          <div className="fa-state-math fa-state-operation" dangerouslySetInnerHTML={mathHtml.weightedValues} />
        </foreignObject>
        <text x="512" y="125" textAnchor="middle" className="fa-state-code">gemm_rs</text>
        <text x="512" y="152" textAnchor="middle" className="fa-state-code">acc_o · FP32</text>

        <g fill="none" className="fa-state-arrow" markerEnd={arrow}>
          <path d="M181 90 H221" />
          <path d="M379 90 H419" />
        </g>
        <text x="201" y="112" textAnchor="middle" className="fa-state-small">FP32</text>
        <text x="399" y="73" textAnchor="middle" className="fa-state-code">rP</text>
        <text x="399" y="112" textAnchor="middle" className="fa-state-small">BF16</text>
        <foreignObject x="4" y="178" width="592" height="36" aria-hidden="true">
          <div className="fa-state-explanation"><span dangerouslySetInnerHTML={mathHtml.weights} /> 是未归一化权重；转为 BF16 的 rP 后参与第二次乘法</div>
        </foreignObject>

        <rect x="4" y="222" width="592" height="62" rx="4" className="fa-state-carry" />
        <text x="16" y="246" className="fa-state-label">跨轮保留</text>
        <text x="16" y="268" className="fa-state-small">全部 FP32</text>
        <text x="174" y="247" textAnchor="middle" className="fa-state-code fa-state-persistent">row_max</text>
        <text x="174" y="270" textAnchor="middle" className="fa-state-small">未缩放分数的最大值</text>
        <text x="345" y="247" textAnchor="middle" className="fa-state-code fa-state-persistent">row_sum</text>
        <text x="345" y="270" textAnchor="middle" className="fa-state-small">每个线程的局部指数和</text>
        <text x="530" y="247" textAnchor="middle" className="fa-state-code fa-state-persistent">acc_o</text>
        <text x="530" y="270" textAnchor="middle" className="fa-state-small">输出累加值 U</text>

        <path d="M4 293 H596" className="fa-state-rule" />
        <text x="4" y="329" className="fa-state-label">遍历结束</text>
        <foreignObject x="98" y="304" width="164" height="48" aria-hidden="true">
          <div className="fa-state-math fa-state-result" dangerouslySetInnerHTML={mathHtml.output} />
        </foreignObject>
        <path d="M260 326 H301" fill="none" className="fa-state-arrow" markerEnd={arrow} />
        <text x="320" y="323" className="fa-state-label">转换并写回 BF16</text>
        <text x="320" y="345" className="fa-state-small">先以 4 lane 合并分母，再做 FP32 归一化</text>
      </svg>
    </div>
    <div className="fa-state-details">
      <p><strong>数据位置</strong>Q/K/V 经 shared memory 装载到寄存器；分数、权重和累计值留在寄存器中。</p>
      <p><strong>跨块更新</strong>第一块建立初始状态；后续先换算旧 <code>row_sum</code> 和 <code>acc_o</code>，再累加新块。</p>
    </div>
    <figcaption>依据 <a href={`${source}flash_fwd_kernel.h#L377-L436`}>前向主循环</a>与 <a href={`${source}softmax.h#L136-L185`}>Softmax 实现</a>绘制，展示核心计算与跨块状态。</figcaption>
  </figure>
}
