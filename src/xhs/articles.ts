import type { Components } from 'react-markdown'
import { attentionKvCompression } from '../articles/attention-kv-compression/attention-kv-compression.blocks'
import { sparseAttention } from '../articles/sparse-attention/sparse-attention.blocks'
import { vllmCpuOverhead } from '../articles/vllm-cpu-overhead/vllm-cpu-overhead.blocks'
import { vllmDistributed } from '../articles/vllm-distributed/vllm-distributed.blocks'
import { vllmPagedAttention } from '../articles/vllm-paged-attention/vllm-paged-attention.blocks'
import { vllmPrefixCaching } from '../articles/vllm-prefix-caching/vllm-prefix-caching.blocks'
import { vllmScheduler } from '../articles/vllm-scheduler/vllm-scheduler.blocks'
import { vllmSpeculativeDecoding } from '../articles/vllm-speculative-decoding/vllm-speculative-decoding.blocks'

export type XhsArticle = { title: string; subtitle?: string; source: string; components?: Components }

// 可导出为小红书正文图片的文章：slug → 正文与图表映射。
export const articles: Record<string, XhsArticle> = {
  'vllm-paged-attention': vllmPagedAttention,
  'vllm-scheduler': vllmScheduler,
  'vllm-prefix-caching': vllmPrefixCaching,
  'vllm-cpu-overhead': vllmCpuOverhead,
  'vllm-speculative-decoding': vllmSpeculativeDecoding,
  'vllm-distributed': vllmDistributed,
  'attention-kv-compression': attentionKvCompression,
  'sparse-attention': sparseAttention,
}
