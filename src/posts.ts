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
    slug: 'cuda-graph',
    title: 'CUDA Graph：执行原理与优化实践',
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
