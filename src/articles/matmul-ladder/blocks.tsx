// 系列两篇（matmul-ladder-fp32 / matmul-ladder-tc）共用的占位映射；::ml-hw:<版本>:: 为该版本的 SM 剖面（旧），::ml-tower:<版本>:: 为硬件塔画法的剖面与数据通路。
import type { ComponentType } from 'react'
import type { Components } from 'react-markdown'
import { IntensityChart, LadderChart, SizesChart } from './MatmulCharts'
import { BankFigure, ClusterFigure, ClusterFigure2, CoalesceFigure2, EighthFigure, HierarchyFigure2, HierarchyFigure3D, OuterFigure, PathFigure, RegTileFigure2, SmemTileFigure, SwizzleFigure2, TcRateFigure, WgmmaSizeFigure, WmmaTileFigure, WmmaPathFigure, WmmaInstrFigure, WsL2Figure, PowerFigure, SummaryFigure, HierarchyHopperFigure, WsFigure, CoalesceFigure, HierarchyFigure, HwProfile, HW, RegTileFigure, SwizzleFigure, TileOrderFigure, TileOrderFigure2, TimelineFp32, TimelineMma, TimelineTstore, TimelineWs, WarptileFigure, TowerFigure, CoalesceFigure3, SmemFigure3D, RegTileFigure3D, WarptileFigure3D, WsFigure2, ClusterFigure3 } from './MatmulFigures'

const FIGS: Record<string, ComponentType> = {
  'ml-ladder:fp32': () => <LadderChart focus="fp32" />,
  'ml-ladder:tc': () => <LadderChart focus="tc" />,
  'ml-sizes': SizesChart,
  'ml-intensity': IntensityChart,
  'ml-hierarchy': HierarchyFigure,
  'ml-coalesce': CoalesceFigure,
  'ml-regtile': RegTileFigure,
  'ml-warptile': WarptileFigure,
  'ml-swizzle': SwizzleFigure,
  'ml-tile-order': TileOrderFigure,
  'ml-tile-order2': TileOrderFigure2,
  'ml-bank': BankFigure,
  'ml-outer': OuterFigure,
  'ml-hierarchy2': HierarchyFigure2,
  'ml-hierarchy3d': HierarchyFigure3D,
  'ml-coalesce2': CoalesceFigure2,
  'ml-smem': SmemTileFigure,
  'ml-eighth': EighthFigure,
  'ml-regtile2': RegTileFigure2,
  'ml-tc-rate': TcRateFigure,
  'ml-wmma-tile': WmmaTileFigure,
  'ml-wmma-instr': WmmaInstrFigure,
  'ml-ws-l2': WsL2Figure,
  'ml-power': PowerFigure,
  'ml-summary': SummaryFigure,
  'ml-hier:wg': () => <HierarchyHopperFigure focus="warpgroup" />,
  'ml-hier:cluster': () => <HierarchyHopperFigure focus="cluster" />,
  'ml-wmma-path': WmmaPathFigure,
  'ml-path:wmma': () => <PathFigure v="wmma" />,
  'ml-path:wgmma': () => <PathFigure v="wgmma" />,
  'ml-path:tstore': () => <PathFigure v="tstore" />,
  'ml-swizzle2': SwizzleFigure2,
  'ml-wgmma-size': WgmmaSizeFigure,
  'ml-ws': WsFigure,
  'ml-cluster2': ClusterFigure2,
  'ml-cluster': ClusterFigure,
  'ml-coalesce3': CoalesceFigure3,
  'ml-smem3d': SmemFigure3D,
  'ml-regtile3d': RegTileFigure3D,
  'ml-warptile3d': WarptileFigure3D,
  'ml-ws2': WsFigure2,
  'ml-cluster3': ClusterFigure3,
  ...Object.fromEntries(['naive', 'smem', 'dbuf', 'wmma', 'mma', 'wgmma', 'tstore'].map(v => [`ml-tower:${v}`, () => <TowerFigure v={v} />])),
  'ml-timeline:fp32': TimelineFp32,
  'ml-timeline:mma': TimelineMma,
  'ml-timeline:ws': TimelineWs,
  'ml-timeline:tstore': TimelineTstore,
}

// 视频静帧卡片（两篇共用）：米色底取自静帧本身，图片已在四周留 24 CSS px 同色边距；width 为图的 CSS 宽度（2 倍图像素宽 ÷ 2），不放大。
export function MatmulStill({ src, alt, width }: { src: string; alt: string; width: number }) {
  return <figure className="ml-still">
    <div className="ml-still-card" style={{ maxWidth: width + 2 * 12 + 2 }}><img src={src} alt={alt} /></div>
    <figcaption>{alt}</figcaption>
  </figure>
}

export const matmulComponents = {
  // 正文里以 > 开头的段落是辅助块：与主线论证无关的背景、历史和旁证。
  blockquote({ children }) {
    return <aside className="ml-aside">{children}</aside>
  },
  p({ children }) {
    const m = typeof children === 'string' ? children.match(/^::(.+)::$/) : null
    if (m) {
      if (m[1].startsWith('ml-hw:') && HW[m[1].slice(6)]) return <div className="ml-fig"><HwProfile v={m[1].slice(6)} /></div>
      const Fig = FIGS[m[1]]
      if (Fig) return <div className="ml-fig"><Fig /></div>
    }
    return <p>{children}</p>
  },
} satisfies Components
