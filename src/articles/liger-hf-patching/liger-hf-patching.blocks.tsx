import type { Components } from 'react-markdown'
import { LigerBindingScope, LigerCoverage, LigerLogits, LigerMeasureAudit, LigerMeasureTrain, LigerShiftLabels } from './LigerHfPatchingFigures'
import source from '../../../content/liger-hf-patching.md?raw'

// 正文、标题与图表占位映射。
export const ligerHfPatching = {
  title: 'Liger-Kernel 的模型接入',
  subtitle: '类替换、实例 patch 与静默回退',
  source,
  components: {
    p({ children }) {
      if (children === '::liger-binding-scope::') return <LigerBindingScope />
      if (children === '::liger-coverage::') return <LigerCoverage />
      if (children === '::liger-measure-train::') return <LigerMeasureTrain />
      if (children === '::liger-measure-audit::') return <LigerMeasureAudit />
      if (children === '::liger-logits::') return <LigerLogits />
      if (children === '::liger-shift-labels::') return <LigerShiftLabels />
      return <p>{children}</p>
    },
  } satisfies Components,
}
