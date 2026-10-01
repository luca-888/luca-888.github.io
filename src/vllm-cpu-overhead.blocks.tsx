import type { Components } from 'react-markdown'
import { EngineCudaGraphDispatch, EngineMeasured, EnginePersistentBatch, EngineProcessPath, EngineStepTimeline } from './VllmCpuOverheadFigures'
import source from '../content/vllm-cpu-overhead.md?raw'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const vllmCpuOverhead = {
  title: 'Vllm',
  subtitle: 'CPU Overhead：让 GPU 不等 CPU',
  source,
  components: {
    p({ children }) {
      if (children === '::engine-step-timeline::') return <EngineStepTimeline />
      if (children === '::engine-process-path::') return <EngineProcessPath />
      if (children === '::engine-persistent-batch::') return <EnginePersistentBatch />
      if (children === '::engine-cudagraph-dispatch::') return <EngineCudaGraphDispatch />
      if (children === '::engine-measured::') return <EngineMeasured />
      return <p>{children}</p>
    },
  } satisfies Components,
}
