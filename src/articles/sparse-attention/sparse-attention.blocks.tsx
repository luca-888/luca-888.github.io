import type { Components } from 'react-markdown'
import { DsaTwoStage, LocalGlobalLayers, NsaReadSet, ReadVsContext, SinkSoftmax, SparseReadSet, SpecCard, V4ReadSet } from './SparseAttentionFigures'
import './sparse-attention.css'
import { PaperFigure } from '../../figure-kit'
import source from '../../../content/sparse-attention.md?raw'
import mistralSwa from '../../assets/sparse-attention/mistral-swa.svg'
import sinkAttention from '../../assets/sparse-attention/streamingllm-attention.svg'
import sinkScheme from '../../assets/sparse-attention/streamingllm-scheme.svg'

// 正文、标题与图表占位映射。
export const sparseAttention = {
  title: '稀疏 Attention',
  subtitle: 'Sliding Window、Attention Sink、DSA 与 CSA',
  source,
  components: {
    p({ children }) {
      if (children === '::sparse-read-set::') return <SparseReadSet />
      if (children === '::sink-softmax::') return <SinkSoftmax />
      if (children === '::dsa-two-stage::') return <DsaTwoStage />
      if (children === '::v4-read-set::') return <V4ReadSet />
      if (children === '::nsa-read-set::') return <NsaReadSet />
      if (children === '::read-vs-context::') return <ReadVsContext />
      if (children === '::local-global-layers::') return <LocalGlobalLayers />
      if (children === '::spec-swa::') return <SpecCard items={[
        ['读取哪些', '最近 W 个 token'],
        ['每步读取', <>W<small>与上下文长度无关</small></>],
        ['缓存', <>W<small>窗口外的 KV 释放</small></>],
        ['代表模型', <>Gemma 3<small>与 full attention 层交替</small></>],
      ]} />
      if (children === '::spec-nsa::') return <SpecCard items={[
        ['读取哪些', <>压缩 entry、16 个 block、窗口<small>三个分支</small></>],
        ['每步读取', <>L / 16 + 1536<small>64K 上下文时为 5632</small></>],
        ['缓存', <>全部 token<small>另存压缩 entry</small></>],
        ['代表模型', <>27B 实验模型<small>NSA 论文</small></>],
      ]} />
      if (children === '::spec-dsa::') return <SpecCard items={[
        ['读取哪些', <>index score 最高的 token<small>由内容决定</small></>],
        ['每步读取', <>2048<small>indexer 另读全部 indexer key</small></>],
        ['缓存', <>全部 token<small>另存 indexer key</small></>],
        ['代表模型', <>DeepSeek-V3.2、GLM-5</>],
      ]} />
      if (children === '::spec-csa::') return <SpecCard items={[
        ['读取哪些', <>压缩 entry 加窗口<small>CSA 选 top-k，HCA 读全部</small></>],
        ['每步读取', <>CSA 1152<small>HCA 为 L / 128 + 128</small></>],
        ['缓存', <>CSA L / 4，HCA L / 128<small>另存 128 个窗口 KV</small></>],
        ['代表模型', <>DeepSeek-V4-Pro、V4-Flash</>],
      ]} />
      if (children === '::mistral-swa::') return <PaperFigure src={mistralSwa} maxWidth={700}
        alt="左：full attention 的 mask，每个 token 可读取它之前的全部 token；中：sliding window attention 的 mask，窗口为 3，每个 token 只读最近 3 个；右：每经过一层，信息向后传递一个窗口的距离"
        caption={<>来源：Jiang et al., <a href="https://arxiv.org/abs/2310.06825">Mistral 7B</a>。示例的窗口为 3；mask 中 1 表示可读取，0 表示不可读取。</>} />
      if (children === '::sink-attention::') return <PaperFigure src={sinkAttention} maxWidth={760}
        alt="Llama-2-7B 第 0、1、2、9、16、23、31 层的平均 attention logits：第 0、1 层集中在对角线附近，第 2 层起第一列的数值明显高于其余各列"
        caption={<>来源：Xiao et al., <a href="https://arxiv.org/abs/2309.17453">Efficient Streaming Language Models with Attention Sinks</a>。256 个长度为 16 的句子上的平均值，第一列对应第一个 token。</>} />
      if (children === '::sink-scheme::') return <PaperFigure src={sinkScheme} maxWidth={760}
        alt="四种处理长文本的方式：dense attention 的 PPL 为 5641；window attention 丢弃开头 token 后 PPL 为 5158；每步重算最近 L 个 token 的 PPL 为 5.43，但复杂度为 O(TL²)；StreamingLLM 保留开头的 sink token 与最近的 token，PPL 为 5.40"
        caption={<>来源：Xiao et al., <a href="https://arxiv.org/abs/2309.17453">Efficient Streaming Language Models with Attention Sinks</a>。模型训练时的长度为 L，预测第 T 个 token（T ≫ L）；黄色为保留的开头 token。</>} />
      return <p>{children}</p>
    },
  } satisfies Components,
}
