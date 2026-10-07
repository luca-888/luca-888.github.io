import type { Components } from 'react-markdown'
import { DistEpPath, DistKvTpDp, DistLedger, DistPdFlow, DistTpLayer, DistV4Anatomy } from './VllmDistributedFigures'
import source from '../../../content/vllm-distributed.md?raw'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const vllmDistributed = {
  title: '分布式推理',
  subtitle: 'TP、DP + EP 与 PD 分离',
  source,
  components: {
    p({ children }) {
      if (children === '::dist-v4-anatomy::') return <DistV4Anatomy />
      if (children === '::dist-ledger::') return <DistLedger />
      if (children === '::dist-tp-layer::') return <DistTpLayer />
      if (children === '::dist-kv-tp-dp::') return <DistKvTpDp />
      if (children === '::dist-ep-path::') return <DistEpPath />
      if (children === '::dist-pd-flow::') return <DistPdFlow />
      return <p>{children}</p>
    },
  } satisfies Components,
}
