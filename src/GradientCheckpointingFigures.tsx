import ac from './assets/gradient-checkpointing/pytorch-checkpoint.png'
import sac from './assets/gradient-checkpointing/pytorch-selective.png'
import officialBenchmark from './assets/gradient-checkpointing/pytorch-memory-budget-benchmark.png'
import './GradientCheckpointing.css'

const source = 'https://pytorch.org/blog/activation-checkpointing-techniques/'

export function GcOfficialFigure({ selective = false }: { selective?: boolean }) {
  const url = selective ? sac : ac
  return <figure className="gc-official">
    <a href={url} target="_blank" rel="noreferrer" aria-label="查看 PyTorch 官方原图"><img src={url} alt={selective ? 'SAC 在 checkpoint 区域中额外保存指定算子的结果，以减少 backward 时的重算。' : '普通执行保存多个中间 tensor，checkpoint 区域通过保留边界输入和 backward 重算减少长期保存的 activation。'} loading="lazy" /></a>
    <figcaption>来源：<a href={source}>PyTorch 官方技术博客</a>。{selective ? '红色算子的结果被保存。' : '黑色框标出 checkpoint 区域。'}</figcaption>
  </figure>
}

export function GcOfficialBenchmark() {
  return <figure className="gc-official">
    <a href={officialBenchmark} target="_blank" rel="noreferrer" aria-label="查看 PyTorch 官方 benchmark 原图"><img src={officialBenchmark} alt="PyTorch 官方 Transformer 结果：横轴为 Memory Budget，纵轴为 Speed；放宽保存预算时速度提高。" loading="lazy" width="1600" height="1045" /></a>
    <figcaption>来源：<a href={source}>PyTorch · Memory Budget API</a>。原图未注明 Speed 的单位、完整模型与硬件配置；memory budget 表示相对保存预算，不是整卡峰值显存比例。</figcaption>
  </figure>
}
