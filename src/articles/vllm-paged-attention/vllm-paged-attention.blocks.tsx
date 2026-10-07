import type { Components } from 'react-markdown'
import { VllmAllocationCompare, VllmBlockTable, VllmPrefixSharing } from './VllmPagedFigures'
import { PaperFigure } from '../../figure-kit'
import source from '../../../content/vllm-paged-attention.md?raw'
import memoryBreakdown from '../../assets/vllm-paged-attention/memory-breakdown.svg'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const vllmPagedAttention = {
  title: 'Vllm',
  subtitle: 'PagedAttention：KV Cache 的分页管理',
  source,
  components: {
    p({ children }) {
      if (children === '::vllm-allocation-compare::') return <VllmAllocationCompare />
      if (children === '::vllm-block-table::') return <VllmBlockTable />
      if (children === '::vllm-prefix-sharing::') return <VllmPrefixSharing />
      if (children === '::vllm-paper-breakdown::') return <PaperFigure src={memoryBreakdown}
        alt="Orca 三种变体与 vLLM 的 KV cache 组成：Token states 分别为 20.4%、26.8%、38.2% 和 96.3%"
        caption={<>来源：Kwon et al., <a href="https://arxiv.org/abs/2309.06180">PagedAttention</a>，Figure 2。OPT-13B，§6.2 实验中的平均占比；绿色为真实 token 的 KV，其余为三种浪费。</>} />
      return <p>{children}</p>
    },
  } satisfies Components,
}
