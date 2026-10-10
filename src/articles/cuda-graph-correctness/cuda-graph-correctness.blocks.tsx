import type { Components } from 'react-markdown'
import { CaptureMiss, GraphBoundary } from './CudaGraphCorrectnessFigures'
import source from '../../../content/cuda-graph-correctness.md?raw'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const cudaGraphCorrectness = {
  title: 'CUDA Graph 的正确性',
  subtitle: '捕获遗漏与过期输入',
  source,
  components: {
    p({ children }) {
      if (children === '::capture-miss::') return <CaptureMiss />
      if (children === '::graph-boundary::') return <GraphBoundary />
      return <p>{children}</p>
    },
  } satisfies Components,
}
