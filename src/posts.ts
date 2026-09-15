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
    slug: 'rmsnorm',
    title: 'RMSNorm：数学推导与 GPU 算子优化',
    description: '从数学公式到 GPU 算子优化：前向与反向推导、Triton 实现，以及性能与精度的取舍。',
    category: '算子优化',
    tags: ['RMSNorm', '机器学习', '性能优化', 'Triton', 'GPU'],
    status: 'draft',
  },
];

export const orderedPosts = [...posts].sort((a, b) =>
  (b.publishedAt ?? '').localeCompare(a.publishedAt ?? ''),
);
