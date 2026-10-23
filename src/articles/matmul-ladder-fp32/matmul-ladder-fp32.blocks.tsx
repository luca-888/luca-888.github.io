import type { Components } from 'react-markdown'
import { MatmulStill, matmulComponents } from '../matmul-ladder/blocks'
import source from '../../../content/matmul-ladder-fp32.md?raw'

// 正文里的 ![说明](../src/assets/matmul-ladder-fp32/x.png) 是视频画面的静帧（videos/matmul-ladder-fp32，render.mjs --stills 导出后裁剪）；说明文字作图注。
const stills = import.meta.glob('../../assets/matmul-ladder-fp32/*.png', { eager: true, import: 'default' }) as Record<string, string>
const still = (src = '') => stills[src.replace(/^\.\.\/src\//, '../../')] ?? src
// 静帧按 2 倍像素导出；显示宽度取视频里的 CSS 宽度（像素宽 ÷ 2），字号与视频原画面相同，retina 上不虚。
const CSS_WIDTH: Record<string, number> = { matrix: 727, grid: 866, 'block-warp': 1102, naive: 693, smem: 616, bank: 731, eighth: 519, 'reg-issue': 673, ceiling: 948 }
const cssWidth = (src: string) => CSS_WIDTH[src.replace(/^.*\//, '').replace(/\.png$/, '')] ?? 760

const components = {
  ...matmulComponents,
  p(props) {
    const { node, children } = props
    const only = node?.children.length === 1 ? node.children[0] : null
    if (only?.type === 'element' && only.tagName === 'img') {
      const alt = String(only.properties.alt ?? '')
      const src = String(only.properties.src ?? '')
      return <MatmulStill src={still(src)} alt={alt} width={cssWidth(src)} />
    }
    return matmulComponents.p({ children })
  },
} satisfies Components

export const matmulLadderFp32 = {
  title: '矩阵乘法优化（上）：FP32 路线',
  subtitle: 'H100 上从朴素写法到双缓冲',
  source,
  components,
}
