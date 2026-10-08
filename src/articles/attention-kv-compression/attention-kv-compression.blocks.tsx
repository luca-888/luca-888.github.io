import type { Components } from 'react-markdown'
import { KvCacheModels, KvCachePerLayer, MlaAbsorb } from './KvCompressionFigures'
import source from '../../../content/attention-kv-compression.md?raw'

// 正文、标题与图表占位映射。
export const attentionKvCompression = {
  title: 'KV Cache 压缩',
  subtitle: 'MQA、GQA 与 MLA',
  source,
  components: {
    p({ children }) {
      if (children === '::kv-cache-per-layer::') return <KvCachePerLayer />
      if (children === '::mla-absorb::') return <MlaAbsorb />
      if (children === '::kv-cache-models::') return <KvCacheModels />
      return <p>{children}</p>
    },
  } satisfies Components,
}
