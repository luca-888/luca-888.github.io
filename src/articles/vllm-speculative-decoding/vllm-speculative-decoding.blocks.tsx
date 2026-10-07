import type { Components } from 'react-markdown'
import { SpecOverlap, SpecVerifyPath } from './VllmSpecFigures'
import { SpecGammaChart, SpecPositionChart, SpecSpeedupChart } from './VllmSpecCharts'
import source from '../../../content/vllm-speculative-decoding.md?raw'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const vllmSpeculativeDecoding = {
  title: 'Vllm',
  subtitle: 'Speculative Decoding：猜多个，验一次',
  source,
  components: {
    p({ children }) {
      if (children === '::spec-verify-path::') return <SpecVerifyPath />
      if (children === '::spec-overlap::') return <SpecOverlap />
      if (children === '::spec-gamma::') return <SpecGammaChart />
      if (children === '::spec-position::') return <SpecPositionChart />
      if (children === '::spec-speedup::') return <SpecSpeedupChart />
      return <p>{children}</p>
    },
  } satisfies Components,
}
