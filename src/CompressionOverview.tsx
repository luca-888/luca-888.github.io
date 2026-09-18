import overview from './assets/compression-harness/overview-friendly.png'

export function CompressionOverview() {
  return (
    <figure className="compression-overview">
      <img
        src={overview}
        width={1536}
        height={1024}
        alt="能力的来源与发挥：左侧的文本、图片和代码通过训练形成模型；右侧的 Harness 将模型与上下文、工具和验证连接成反馈循环，产出任务结果。"
      />
      <figcaption>训练从样本中学习规律；Harness 组织上下文、工具与反馈，让模型参与完成任务。左侧漏斗为学习过程的示意。</figcaption>
    </figure>
  )
}
