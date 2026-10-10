import type { Components } from 'react-markdown'
import { OffsetCheckpointPlaceholder, OffsetDataflow, OffsetRulers, OffsetTracePlaceholder } from './Bf16WeightOffsetFigures'
import source from '../../../content/bf16-weight-offset.md?raw'

// 正文、标题与图表占位映射。
export const bf16WeightOffset = {
  title: 'Gemma RMSNorm 的 1 + w',
  subtitle: 'BF16 舍入与 weight offset 的相加位置',
  source,
  components: {
    p({ children }) {
      if (children === '::offset-rulers::') return <OffsetRulers />
      if (children === '::offset-dataflow::') return <OffsetDataflow />
      if (children === '::offset-checkpoint-placeholder::') return <OffsetCheckpointPlaceholder />
      if (children === '::offset-trace-placeholder::') return <OffsetTracePlaceholder />
      return <p>{children}</p>
    },
  } satisfies Components,
}
