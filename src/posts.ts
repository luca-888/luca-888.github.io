export type Post = {
  slug: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
} & (
  | { status: 'draft'; publishedAt?: never }
  | { status: 'published'; publishedAt: string }
);

// 日期仅在文章正式发布时填写，格式为 YYYY-MM-DD。
export const posts: Post[] = [
  {
    slug: 'rtx-pro-6000-topology',
    title: 'RTX PRO 6000 多卡拓扑：连接结构与通信路径',
    description: '以八卡参考设计为例，介绍 PCIe 连接、NUMA 内存组织、GPU 通信路径与任务分组。',
    category: 'GPU 系统',
    tags: ['RTX PRO 6000', 'PCIe', 'NUMA', 'P2P', 'NCCL', 'GPU 拓扑'],
    status: 'published',
    publishedAt: '2026-09-21',
  },
  {
    slug: 'compression-harness',
    title: '压缩即智能：从大模型到 Harness',
    description: '从训练与能力涌现，到上下文、推理和 Harness，讨论模型能力如何转化为任务表现。',
    category: '大模型',
    tags: ['LLM', 'Scaling laws', '上下文', 'Agent', 'Harness'],
    status: 'published',
    publishedAt: '2026-09-18',
  },
  {
    slug: 'cuda-graph',
    title: 'CUDA Graph：计算与调度的解耦',
    description: '通过四节点 diamond 和真实 Nsight trace，理解 CUDA Graph 的执行原理、加速来源与应用边界。',
    category: 'GPU 编程',
    tags: ['CUDA Graph', 'CUDA', 'PyTorch', '性能优化', 'GPU'],
    status: 'published',
    publishedAt: '2026-09-16',
  },
  {
    slug: 'rmsnorm',
    title: 'RMSNorm：计算原理与 GPU 算子优化',
    description: '从计算图与 PyTorch 参考实现出发，分析 eager、torch.compile 与手写 Triton 的执行结构、性能和显存开销。',
    category: '算子优化',
    tags: ['RMSNorm', '机器学习', '性能优化', 'Triton', 'GPU'],
    status: 'published',
    publishedAt: '2026-09-15',
  },
];

export const orderedPosts = [...posts].sort((a, b) =>
  (b.publishedAt ?? '').localeCompare(a.publishedAt ?? ''),
);
