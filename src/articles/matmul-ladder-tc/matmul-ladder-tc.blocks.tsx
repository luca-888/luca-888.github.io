import type { Components } from 'react-markdown'
import { MatmulStill, matmulComponents } from '../matmul-ladder/blocks'
import source from '../../../content/matmul-ladder-tc.md?raw'
import sizes from '../../assets/matmul-ladder-tc/sizes.json'

// 正文里的 ![说明](../src/assets/matmul-ladder-tc/x.png) 是视频画面的静帧（videos/matmul-ladder-tc，render.mjs --stills --scale 2 导出后裁剪）；说明文字作图注。
// sizes.json 记录每张图的 CSS 宽度（2 倍图像素宽 ÷ 2），图按这一宽度封顶显示，不放大、字号与视频画面一致。
const stills = import.meta.glob('../../assets/matmul-ladder-tc/*.png', { eager: true, import: 'default' }) as Record<string, string>
const still = (src = '') => stills[src.replace(/^\.\.\/src\//, '../../')] ?? src

const components = {
  ...matmulComponents,
  p(props) {
    const { node, children } = props
    const only = node?.children.length === 1 ? node.children[0] : null
    if (only?.type === 'element' && only.tagName === 'img') {
      const alt = String(only.properties.alt ?? ''), src = String(only.properties.src ?? '')
      const w = (sizes as Record<string, number>)[src.replace(/^.*\//, '').replace(/\.png$/, '')]
      return <MatmulStill src={still(src)} alt={alt} width={w ?? 900} />
    }
    return matmulComponents.p({ children })
  },
} satisfies Components

export const matmulLadderTc = {
  title: '矩阵乘法优化（下）：Tensor Core 路线追平 cuBLAS',
  subtitle: 'H100 上的 wmma、mma.sync、TMA 与 wgmma',
  source,
  components,
}
