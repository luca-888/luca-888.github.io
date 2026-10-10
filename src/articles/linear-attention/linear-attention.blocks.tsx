import type { Components } from 'react-markdown'
import { Chunkwise, DeltaWrite, HybridLayers, LayerStrips, NiahBars, SpecCard, StateStep, StateVsKv, Tex } from './LinearAttentionFigures'
import './linear-attention.css'
import { PaperFigure } from '../../figure-kit'
import source from '../../../content/linear-attention.md?raw'
import qwenGdn from '../../assets/linear-attention/qwen-gdn-module.svg'
import k3PrefixCache from '../../assets/linear-attention/k3-prefix-cache.svg'

// 正文、标题与图表占位映射。
export const linearAttention = {
  title: '线性 Attention',
  subtitle: 'Gated DeltaNet、KDA 与混合架构',
  source,
  components: {
    p({ children }) {
      if (children === '::state-step::') return <StateStep />
      if (children === '::niah-bars::') return <NiahBars />
      if (children === '::layer-strips::') return <LayerStrips />
      if (children === '::delta-write::') return <DeltaWrite />
      if (children === '::chunkwise::') return <Chunkwise />
      if (children === '::hybrid-layers::') return <HybridLayers />
      if (children === '::state-vs-kv::') return <StateVsKv />
      if (children === '::spec-linear::') return <SpecCard items={[
        ['更新', <Tex src="S_t = S_{t-1} + v_t k_t^\top" />],
        ['写入', <>直接相加<small>同一个 key 的 value 叠加</small></>],
        ['遗忘', <>无<small>串扰随上下文累积</small></>],
        ['代表', <>Linear Transformer<small>Katharopoulos 等，2020</small></>],
      ]} />
      if (children === '::spec-decay::') return <SpecCard items={[
        ['更新', <Tex src="S_t = \alpha_t S_{t-1} + v_t k_t^\top" />],
        ['写入', <>直接相加</>],
        ['遗忘', <>整体乘 α<sub>t</sub><small>每个 head 一个，或每个 channel 一个</small></>],
        ['代表', <>Mamba2、GLA</>],
      ]} />
      if (children === '::spec-gdn::') return <SpecCard items={[
        ['更新', <Tex src="S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top" />],
        ['写入', <>只写差值<small>同一个 key 的 value 被替换</small></>],
        ['遗忘', <>整体乘 α<sub>t</sub><small>每个 head 一个</small></>],
        ['代表', <>Qwen3-Next、Qwen3.5–3.8</>],
      ]} />
      if (children === '::spec-kda::') return <SpecCard items={[
        ['更新', <Tex src="S_t = S_{t-1}\,\mathrm{Diag}(\alpha_t)(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top" />],
        ['写入', <>只写差值</>],
        ['遗忘', <>逐 key channel<small>每个 head 128 个系数</small></>],
        ['代表', <>Kimi Linear、Kimi K3</>],
      ]} />
      if (children === '::qwen-gdn-module::') return <PaperFigure src={qwenGdn} maxWidth={340}
        alt="Gated DeltaNet 层：输入经三个线性层得到 q、k、v，各经 short convolution 与 SiLU，q 与 k 再做 L2 归一化；另两个线性层给出 α 与 β；五者进入 gated delta rule，输出经 zero-centered RMSNorm 后与 sigmoid 输出 gate 相乘，再经线性层输出"
        caption={<>来源：Qwen Team, <a href="https://arxiv.org/abs/2608.30320">On the Design of Qwen3.8-Next Architecture</a>。α 为 decay，β 为写入强度。</>} />
      if (children === '::k3-prefix-cache::') return <PaperFigure src={k3PrefixCache} maxWidth={700}
        alt="一个 6144 token 的物理 block 分为 12 个 512 token 的 hash block，前 5 个已缓存 MLA KV；下方一排圆点表示各 hash block 末尾是否保存了 KDA checkpoint，第 3 个为灰色实心点，第 5 个为橙色实心点，即命中边界 B = 2560；命中后恢复 B 处的 KDA checkpoint，对未满的 MLA block 做 copy-on-write，从第 B 个 token 继续 prefill"
        caption={<>来源：Kimi Team, <a href="https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf">Kimi K3 Technical Report</a> Figure 12。空心圆为没有保存 checkpoint 的边界，实心点为已保存的 checkpoint，橙色为本次命中的位置。</>} />
      return <p>{children}</p>
    },
  } satisfies Components,
}
