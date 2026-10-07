import type { Components } from 'react-markdown'
import { DsparkCycle, DsparkDraft, DsparkMarkov, DsparkMtp, DsparkSchedule } from './DsparkFigures'
import { PaperFigure } from '../../figure-kit'
import source from '../../../content/dspark.md?raw'
import positionAccept from '../../assets/dspark/position-cond-accept.svg'
import onlineService from '../../assets/dspark/online-service.svg'
import onlineBudget from '../../assets/dspark/online-service-tradeoff.svg'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const dspark = {
  title: 'DSpark',
  subtitle: 'DeepSeek V4 的 Speculative Decoding：并行生成 draft，按把握验证',
  source,
  components: {
    p({ children }) {
      if (children === '::dspark-mtp::') return <DsparkMtp />
      if (children === '::dspark-cycle::') return <DsparkCycle />
      if (children === '::dspark-draft::') return <DsparkDraft />
      if (children === '::dspark-markov::') return <DsparkMarkov />
      if (children === '::dspark-schedule::') return <DsparkSchedule />
      if (children === '::dspark-position::') return <PaperFigure src={positionAccept} maxWidth={760}
        alt="三种 drafter 在数学、代码、聊天三类任务上各位置的条件接受率：DFlash 第 1 位高、之后下降，Eagle3 第 1 位低、之后上升，DSpark 各位置都最高且平稳"
        caption={<>来源：Cheng et al., <a href="https://arxiv.org/abs/2607.05147">DSpark</a>，各位置的条件接受率。Target 为 Qwen3-4B，每类任务取三个 benchmark 的平均。</>} />
      if (children === '::dspark-online::') return <PaperFigure src={onlineService} maxWidth={760}
        alt="DeepSeek-V4-Flash 与 V4-Pro 线上流量下，每张卡的吞吐与每个用户生成速度的关系：DSpark 的曲线整体在 MTP 的右上方"
        caption={<>来源：Cheng et al., <a href="https://arxiv.org/abs/2607.05147">DSpark</a>，线上流量下吞吐与每个用户生成速度的关系。图中 TPS 指每个用户每秒收到的 token 数；散点为线上采样，实线为拟合。</>} />
      if (children === '::dspark-budget::') return <PaperFigure src={onlineBudget} maxWidth={760}
        alt="DeepSeek-V4-Flash 与 V4-Pro 线上流量下，吞吐与每个请求的验证长度随并发的变化：MTP 固定验 2 个 token，DSpark 低并发时验 5 到 6 个，并发越高越少"
        caption={<>来源：Cheng et al., <a href="https://arxiv.org/abs/2607.05147">DSpark</a>，吞吐（上排）与每个请求的平均验证长度（下排）随并发的变化。验证长度含 target 自己的输出位，MTP-1 固定为 2。</>} />
      return <p>{children}</p>
    },
  } satisfies Components,
}
