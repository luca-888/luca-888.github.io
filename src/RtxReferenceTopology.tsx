import referenceTopology from './assets/rtx-pro-6000-topology/nvidia-rtx-pro-6000-8-gpu.png'
import './RtxReferenceTopology.css'

export function RtxReferenceTopology() {
  return <figure className="rtx-reference">
    <a href={referenceTopology} target="_blank" rel="noreferrer" aria-label="查看 NVIDIA 八卡参考拓扑原图">
      <img
        src={referenceTopology}
        width={1783}
        height={936}
        alt="NVIDIA RTX PRO 6000 Blackwell Server Edition 八卡参考拓扑：双 CPU、四个 PCIe Switch、八张 GPU、四张计算网卡，以及独立的 DPU 和 NVMe。"
      />
    </a>
    <figcaption>
      NVIDIA 八卡参考设计（2-8-5-200），点击可查看原图。
      来源：<a href="https://dam-cdn.nvd.orangelogic.com/AssetLink/86247t1n184elod6454b4i33e1lrw422.pdf#page=23">NVIDIA Enterprise Reference Architecture 白皮书，Figure 10</a>。
    </figcaption>
  </figure>
}
