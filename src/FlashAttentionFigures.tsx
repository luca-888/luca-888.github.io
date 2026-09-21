import tiling from './assets/flash-attention/fa1-tiling-online-softmax.png'
import blockParallelism from './assets/flash-attention/fa2-thread-block-parallelism.png'
import fa1Warps from './assets/flash-attention/fa1-warp-partitioning.png'
import fa2Warps from './assets/flash-attention/fa2-warp-partitioning.png'
import './FlashAttentionFigures.css'

const paper = 'https://arxiv.org/abs/2307.08691v1'

export function FlashAttentionTiling() {
  return <figure className="fa-source-figure">
    <a href={tiling} target="_blank" rel="noreferrer" aria-label="查看分块计算官方原图">
      <img src={tiling} width={5844} height={3018} loading="lazy" alt="Q 分别与两个 K tile 相乘，指数权重在片上计算，并结合 V 更新输出，避免将完整中间矩阵写入 HBM。" />
    </a>
    <figcaption>来源：Tri Dao，<a href={paper}>FlashAttention-2，Figure 1</a>。
      图中 A 为指数权重，省略了缩放与减去行最大值；正文用 U 表示输出累加值。</figcaption>
  </figure>
}

export function FlashAttentionBlocks() {
  return <figure className="fa-source-figure fa-block-figure">
    <a href={blockParallelism} target="_blank" rel="noreferrer" aria-label="查看 CTA 划分官方原图">
      <img src={blockParallelism} width={2298} height={1268} loading="lazy" alt="Causal attention 中，前向 CTA 负责不同 Q tile，反向 CTA 负责不同 K tile。" />
    </a>
    <figcaption>来源：Tri Dao，<a href={paper}>FlashAttention-2，Figure 2</a>。左图为本文讨论的前向 Q tile，右图为反向 K tile。</figcaption>
  </figure>
}

export function FlashAttentionWarps() {
  return <figure className="fa-source-figure">
    <div className="fa-warp-comparison">
      <div>
        <p className="fa-figure-label">FlashAttention · 划分 K/V</p>
        <a href={fa1Warps} target="_blank" rel="noreferrer" aria-label="查看 FlashAttention warp 划分官方原图">
          <img src={fa1Warps} width={3710} height={2194} loading="lazy" alt="FA1：warp 共享访问 Q，分别处理 K/V 的不同片段。" />
        </a>
      </div>
      <div>
        <p className="fa-figure-label">FlashAttention-2 · 划分 Q</p>
        <a href={fa2Warps} target="_blank" rel="noreferrer" aria-label="查看 FlashAttention-2 warp 划分官方原图">
          <img src={fa2Warps} width={3100} height={2186} loading="lazy" alt="FA2：warp 分别处理 Q 的不同行，共享访问 K/V。" />
        </a>
      </div>
    </div>
    <figcaption>来源：Tri Dao，<a href={paper}>FlashAttention-2，Figure 3(a–b)</a>。蓝色表示 warp 共同访问，橙色表示分配给不同 warp。</figcaption>
  </figure>
}
