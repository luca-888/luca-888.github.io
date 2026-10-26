import type { Components } from 'react-markdown'
import { PaperFigure } from '../../figure-kit'
import { PdAdmissionOrder, PdDecisions, PdLocalOrRemote, PdMeasurePlaceholder, PdPickP, PdRatio } from './PdSchedulingFigures'
import source from '../../../content/pd-scheduling.md?raw'
import mooncakeLoad from '../../assets/pd-scheduling/mooncake-load_example.svg'
import fixedRatios from '../../assets/pd-scheduling/mitra-fixed-ratios.svg'

// 正文、标题与图表占位映射。
export const pdScheduling = {
  title: 'PD 分离的调度',
  subtitle: '路由、接纳与比例',
  source,
  components: {
    p({ children }) {
      if (children === '::pd-decisions::') return <PdDecisions />
      if (children === '::pd-pick-p::') return <PdPickP />
      if (children === '::pd-ratio::') return <PdRatio />
      if (children === '::pd-local-or-remote::') return <PdLocalOrRemote />
      if (children === '::pd-admission-order::') return <PdAdmissionOrder />
      if (children === '::pd-measure-placeholder::') return <PdMeasurePlaceholder />
      if (children === '::pd-mooncake-load::') return <PaperFigure src={mooncakeLoad} maxWidth={680}
        alt="Mooncake 集群 20 分钟内 prefill 与 decode 实例的负载曲线，两者交替涨落"
        caption={<>来源：Qin et al., <a href="https://arxiv.org/abs/2407.00079">Mooncake</a>。20 台机器的线上集群采用 early rejection、尚未按预测接收时，20 分钟内 prefill（绿）与 decode（橙）实例的负载。</>} />
      if (children === '::pd-fixed-ratios::') return <PaperFigure src={fixedRatios} maxWidth={680}
        alt="DeepSeek-R1 在 ISL 16384、OSL 2048 下，固定 P:D 比例 0.5、3.5 与动态比例、合并部署的吞吐与交互性曲线"
        caption={<>来源：Mitra et al., <a href="https://arxiv.org/abs/2506.05508">Beyond the Buzz</a>。DeepSeek-R1，输入 16384、输出 2048 个 token，NVIDIA 的仿真器。横轴为每个用户的 token 速率（越往右延迟要求越严），纵轴为每张 GPU 的吞吐，均已归一化。</>} />
      return <p>{children}</p>
    },
  } satisfies Components,
}
