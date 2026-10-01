import type { Components } from 'react-markdown'
import { PrefixBlockVsRadix, PrefixFreeQueue, PrefixHashChain } from './VllmPrefixFigures'
import source from '../content/vllm-prefix-caching.md?raw'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const vllmPrefixCaching = {
  title: 'Vllm',
  subtitle: 'Prefix Caching：跨请求复用 KV Cache',
  source,
  components: {
    p({ children }) {
      if (children === '::prefix-hash-chain::') return <PrefixHashChain />
      if (children === '::prefix-free-queue::') return <PrefixFreeQueue />
      if (children === '::prefix-block-vs-radix::') return <PrefixBlockVsRadix />
      return <p>{children}</p>
    },
  } satisfies Components,
}
