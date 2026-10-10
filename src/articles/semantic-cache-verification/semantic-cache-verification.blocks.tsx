import type { Components } from 'react-markdown'
import { ScvFailurePolicy, ScvLookupPath, ScvMeasurePlaceholder, ScvSimilarity } from './ScvFigures'
import source from '../../../content/semantic-cache-verification.md?raw'

// 正文、标题与图表占位映射。
export const semanticCacheVerification = {
  title: 'Semantic Cache 的命中校验',
  subtitle: '误命中、校验器故障与 fail-closed',
  source,
  components: {
    p({ children }) {
      if (children === '::scv-similarity::') return <ScvSimilarity />
      if (children === '::scv-lookup-path::') return <ScvLookupPath />
      if (children === '::scv-failure-policy::') return <ScvFailurePolicy />
      if (children === '::scv-measure-placeholder::') return <ScvMeasurePlaceholder />
      return <p>{children}</p>
    },
  } satisfies Components,
}
