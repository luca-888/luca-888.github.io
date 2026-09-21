import './RtxMeasuredTopology.css'

type Device = { id: number; bus_id: string; numa_node: number; pci_ancestors: string[] }

export function RtxMeasuredTopology({ devices }: { devices: Device[] }) {
  const nodes = [...new Set(devices.map(device => device.numa_node))].sort((a, b) => a - b)
  const sides = nodes.map(node => devices.filter(device => device.numa_node === node))
  const commonParent = (pair: Device[]) => pair[0].pci_ancestors.filter(bus => pair.every(device => device.pci_ancestors.includes(bus))).at(-1)
  return <figure className="rtx-measured-topology">
    <svg viewBox="0 0 920 370" role="img" aria-labelledby="rtx-measured-title rtx-measured-description">
      <title id="rtx-measured-title">实机拓扑：四对 GPU 分属两个 NUMA 节点</title>
      <desc id="rtx-measured-description">GPU 0、1 与 GPU 2、3 属于 NUMA 0；GPU 4、5 与 GPU 6、7 属于 NUMA 1。配对内部为 PIX，同侧配对之间为 NODE，两侧之间为 SYS。图中 PCI 地址表示已确认的共同上游，省略中间端口。</desc>
      <path className="rtx-measured-system" d="M235 76 V33 H685 V76" />
      <rect className="rtx-measured-label-bg" x="358" y="15" width="204" height="34" />
      <text className="rtx-measured-path" x="460" y="37">跨 NUMA · SYS</text>
      {sides.map((side, index) => {
        const x = index * 450 + 25
        return <g key={nodes[index]}>
          <rect className="rtx-measured-side" x={x} y="76" width="420" height="258" rx="12" />
          <text className="rtx-measured-heading" x={x + 210} y="111">NUMA {nodes[index]}</text>
          <path className="rtx-measured-branch" d={`M${x + 105} 176 V144 H${x + 315} V176`} />
          <rect className="rtx-measured-label-bg" x={x + 174} y="129" width="72" height="28" />
          <text className="rtx-measured-path" x={x + 210} y="149">NODE</text>
          {[side.slice(0, 2), side.slice(2, 4)].map((pair, pairIndex) => {
            const center = x + 105 + pairIndex * 210
            return <g key={pairIndex}>
              <rect className="rtx-measured-pair" x={center - 96} y="176" width="192" height="138" rx="8" />
              <text className="rtx-measured-small" x={center} y="197">共同上游 PCI 节点</text>
              <text className="rtx-measured-bus" x={center} y="218">{commonParent(pair)}</text>
              <path className="rtx-measured-branch" d={`M${center - 46} 254 V233 H${center + 46} V254`} />
              <rect className="rtx-measured-label-bg" x={center - 22} y="221" width="44" height="25" />
              <text className="rtx-measured-path" x={center} y="240">PIX</text>
              {pair.map((device, i) => <g key={device.id}>
                <rect className="rtx-measured-gpu" x={center - 88 + i * 92} y="254" width="84" height="30" rx="4" />
                <text className="rtx-measured-gpu-name" x={center - 46 + i * 92} y="275">GPU {device.id}</text>
                <text className="rtx-measured-bus" x={center - 46 + i * 92} y="301">{device.bus_id.slice(5)}</text>
              </g>)}
            </g>
          })}
        </g>
      })}
    </svg>
    <figcaption>依据实机设备树与拓扑矩阵绘制，省略中间 PCIe 端口。数字为 PCI 设备地址，GPU 下方省略了共同前缀 0000:。</figcaption>
  </figure>
}
