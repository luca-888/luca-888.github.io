import type { Components } from 'react-markdown'
import { JevLatency, JevMask, JevPointer, JevTwoPaths } from './JevFigures'
import source from '../../../content/jev.md?raw'

// 网页与小红书导出共用的正文、标题与图表占位映射。
export const jev = {
  title: 'Jev',
  subtitle: '不生成文本的模型怎么做决定',
  source,
  components: {
    p({ children }) {
      if (children === '::jev-two-paths::') return <JevTwoPaths />
      if (children === '::jev-mask::') return <JevMask />
      if (children === '::jev-pointer::') return <JevPointer />
      if (children === '::jev-latency::') return <JevLatency />
      return <p>{children}</p>
    },
  } satisfies Components,
}
