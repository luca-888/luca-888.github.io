import type { Components } from 'react-markdown'
import { QuantCodesToTicks, QuantGptqColumns, QuantMarlinDequant, QuantMarlinSpeedup, QuantMemoryBudget, QuantNvfp4Levels, QuantRulers, QuantSmoothChannels, QuantWeightHist } from './LlmQuantizationFigures'
import { PaperFigure } from '../../figure-kit'
import source from '../../../content/llm-quantization.md?raw'
import pictureBits from '../../assets/llm-quantization/picture-bits.png'
import pictureFp8 from '../../assets/llm-quantization/picture-fp8.png'
import pictureGroups from '../../assets/llm-quantization/picture-groups.png'
import pictureDiffusion from '../../assets/llm-quantization/picture-diffusion.png'

const ANALOGY = '以灰度图类比，每个像素的亮度对应一个权重。图片取自本站视频，各结果均按对应格式实际计算。'

// 正文、标题与图表占位映射。
export const llmQuantization = {
  title: 'LLM 量化',
  subtitle: 'FP8、INT4、NVFP4 与 Marlin Kernel',
  source,
  components: {
    p({ children }) {
      if (children === '::quant-codes::') return <QuantCodesToTicks />
      if (children === '::quant-rulers::') return <QuantRulers />
      if (children === '::quant-picture-bits::') return <PaperFigure src={pictureBits} maxWidth={760} alt="同一张球的灰度图按 16、8、4 bit 存：8 bit 与原图看不出区别，4 bit 出现色阶断层" caption={ANALOGY} />
      if (children === '::quant-picture-fp8::') return <PaperFigure src={pictureFp8} maxWidth={760} alt="球旁加一盏亮度为 100 的灯：INT8 下球身只用到 3 级亮度，FP8 下用到 79 级" caption={ANALOGY} />
      if (children === '::quant-picture-groups::') return <PaperFigure src={pictureGroups} maxWidth={760} alt="带灯的灰度图量化到 4 bit：INT4 每 128 个像素一组时灯所在的整块区域受影响，NVFP4 每 16 个像素一组时只影响灯周围的小块区域" caption={ANALOGY} />
      if (children === '::quant-picture-diffusion::') return <PaperFigure src={pictureDiffusion} maxWidth={760} alt="4 bit 下各像素独立取最近的刻度时出现色阶断层；把误差分给相邻像素后，同样 16 级亮度下断层消失" caption={ANALOGY} />
      if (children === '::quant-weight-hist::') return <QuantWeightHist />
      if (children === '::quant-gptq-columns::') return <QuantGptqColumns />
      if (children === '::quant-smooth-channels::') return <QuantSmoothChannels />
      if (children === '::quant-nvfp4-levels::') return <QuantNvfp4Levels />
      if (children === '::quant-marlin-dequant::') return <QuantMarlinDequant />
      if (children === '::quant-marlin-speedup::') return <QuantMarlinSpeedup />
      if (children === '::quant-memory-budget::') return <QuantMemoryBudget />
      return <p>{children}</p>
    },
  } satisfies Components,
}
