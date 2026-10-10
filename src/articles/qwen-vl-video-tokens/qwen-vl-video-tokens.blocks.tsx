import type { Components } from 'react-markdown'
import { MeasureFastPathCap, MeasureGenerate, MeasureTrainInfer, MultiVideo, VideoPath } from './QwenVlVideoTokensFigures'
import source from '../../../content/qwen-vl-video-tokens.md?raw'

// 正文、标题与图表占位映射。
export const qwenVlVideoTokens = {
  title: 'Qwen-VL 的视频 token',
  subtitle: '采样、grid 与时间戳',
  source,
  components: {
    p({ children }) {
      if (children === '::video-path::') return <VideoPath />
      if (children === '::multi-video::') return <MultiVideo />
      if (children === '::measure-fast-path-cap::') return <MeasureFastPathCap />
      if (children === '::measure-generate::') return <MeasureGenerate />
      if (children === '::measure-train-infer::') return <MeasureTrainInfer />
      return <p>{children}</p>
    },
  } satisfies Components,
}
