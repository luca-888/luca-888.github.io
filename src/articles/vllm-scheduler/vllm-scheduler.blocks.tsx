import type { Components } from 'react-markdown'
import { VllmBatchingSlots, VllmFlattenedBatch, VllmSchedulingTimelines, VllmStepBudget } from './VllmSchedulerFigures'
import { PaperFigure } from '../../figure-kit'
import source from '../../../content/vllm-scheduler.md?raw'
import linearTime from '../../assets/vllm-scheduler/linear-time-vs-tokens.svg'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const vllmScheduler = {
  title: 'Vllm',
  subtitle: 'Continuous Batching 与调度器',
  source,
  components: {
    p({ children }) {
      if (children === '::vllm-batching-slots::') return <VllmBatchingSlots />
      if (children === '::vllm-flattened-batch::') return <VllmFlattenedBatch />
      if (children === '::vllm-timelines::') return <VllmSchedulingTimelines />
      if (children === '::vllm-step-budget::') return <VllmStepBudget />
      if (children === '::vllm-linear-time::') return <PaperFigure src={linearTime} maxWidth={520}
        alt="LLaMA2-70B 线性层耗时随 batch 内 token 数变化：128 到 512 个 token 之间几乎持平，之后随 token 数上升"
        caption={<>来源：Agrawal et al., <a href="https://arxiv.org/abs/2403.02310">Sarathi-Serve</a>，线性层耗时与 batch 内 token 数的关系。LLaMA2-70B，A100，TP-2 与 TP-4；token 少时耗时由权重读取决定，越过拐点后线性增长。</>} />
      return <p>{children}</p>
    },
  } satisfies Components,
}
