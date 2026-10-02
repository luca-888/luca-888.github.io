import type { Components } from 'react-markdown'
import { vllmCpuOverhead } from '../vllm-cpu-overhead.blocks'
import { vllmDistributed } from '../vllm-distributed.blocks'
import { vllmPagedAttention } from '../vllm-paged-attention.blocks'
import { vllmPrefixCaching } from '../vllm-prefix-caching.blocks'
import { vllmScheduler } from '../vllm-scheduler.blocks'
import { vllmSpeculativeDecoding } from '../vllm-speculative-decoding.blocks'

export type XhsArticle = { title: string; subtitle?: string; source: string; components?: Components }

// 可导出为小红书正文图片的文章：slug → 正文与图表映射。
export const articles: Record<string, XhsArticle> = {
  'vllm-paged-attention': vllmPagedAttention,
  'vllm-scheduler': vllmScheduler,
  'vllm-prefix-caching': vllmPrefixCaching,
  'vllm-cpu-overhead': vllmCpuOverhead,
  'vllm-speculative-decoding': vllmSpeculativeDecoding,
  'vllm-distributed': vllmDistributed,
}
