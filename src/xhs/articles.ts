import type { Components } from 'react-markdown'
import { vllmPagedAttention } from '../vllm-paged-attention.blocks'

export type XhsArticle = { title: string; subtitle?: string; source: string; components?: Components }

// 可导出为小红书正文图片的文章：slug → 正文与图表映射。
export const articles: Record<string, XhsArticle> = {
  'vllm-paged-attention': vllmPagedAttention,
}
