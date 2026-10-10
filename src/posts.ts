export type Post = {
  slug: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
  // B 站 BV 号；成片不进仓库，上传后填写。源文件在 videos/<slug>/。
  video?: string;
} & (
  | { status: 'draft'; publishedAt?: never }
  | { status: 'published'; publishedAt: string }
);

// 日期仅在文章正式发布时填写，格式为 YYYY-MM-DD。
export const posts: Post[] = [
  {
    slug: 'bf16-weight-offset',
    title: 'Gemma RMSNorm 的 1 + w：BF16 舍入与 weight offset 的相加位置',
    description: 'Gemma 的 RMSNorm 把缩放系数写成 1 + w。weight 以 BF16 存储时，在 BF16 中加 1 会舍掉 w 的低位：BF16 的 ulp 与舍入、误差怎样进入 Y 与 dX，以及把 offset 移进 kernel、在 FP32 中相加的做法。',
    category: '算子优化',
    tags: ['RMSNorm', 'Gemma', 'BF16', '数值精度', 'CuTe DSL', 'Liger Kernel'],
    status: 'draft',
  },
  {
    slug: 'linear-attention',
    title: '线性 Attention：Gated DeltaNet、KDA 与混合架构',
    description: '把历史写进固定大小的矩阵 state：decay gate 与 delta rule 决定忘什么、怎么写，chunkwise 计算让训练并行，Gated DeltaNet 与 KDA 以 3:1 与 full attention 混合，以及 state 在推理系统中的代价与反对意见。',
    category: '推理系统',
    tags: ['Attention', 'Linear attention', 'Gated DeltaNet', 'KDA', 'Qwen', 'Kimi', 'KV cache'],
    status: 'draft',
  },
  {
    slug: 'sparse-attention',
    title: '稀疏 Attention：Sliding Window、Attention Sink、DSA 与 CSA',
    description: '每个 query 只读一部分历史 token：sliding window 与 local / global 层交替，attention sink 的成因与显式 sink，NSA、DSA 的学习式选择，DeepSeek V4 先压缩再选择的 CSA 与 HCA。',
    category: '推理系统',
    tags: ['Attention', 'Sparse attention', 'Sliding window', 'Attention sink', 'NSA', 'DSA', 'DeepSeek', 'KV cache'],
    status: 'published',
    publishedAt: '2026-10-10',
  },
  {
    slug: 'attention-kv-compression',
    title: 'KV Cache 压缩：MQA、GQA 与 MLA',
    description: 'decode 每步都要把 KV cache 读一遍。MQA、GQA 让多个 query head 共享 KV head，MLA 只缓存低秩 latent 并借矩阵吸收在 decode 时做 MQA，DeepSeek V4 直接训练 key 与 value 共用的单个 KV head。',
    category: '推理系统',
    tags: ['Attention', 'KV cache', 'MQA', 'GQA', 'MLA', 'DeepSeek', 'vLLM', 'LLM serving'],
    status: 'draft',
  },
  {
    slug: 'dspark',
    title: 'DSpark：DeepSeek V4 的 Speculative Decoding',
    description: 'DeepSeek V4 线上的 speculative decoding：并行 backbone 加一个很小的 Markov head 生成 draft，confidence head 估计每个 token 的存活概率，调度器按负载决定每个请求验几个。',
    category: '推理系统',
    tags: ['DSpark', 'DeepSeek V4', 'Speculative decoding', 'DFlash', 'Adaptive verification', 'vLLM', 'LLM serving'],
    status: 'draft',
  },
  {
    slug: 'vllm-distributed',
    title: '分布式推理：TP、DP + EP 与 PD 分离',
    description: '一张卡不够时怎么办：稠密模型用 TP，MoE 模型用 DP attention 加 EP，实例之间做 PD 分离。每种并行切的是什么、通信发生在哪、适用于什么模型。',
    category: '推理系统',
    tags: ['vLLM', 'Tensor parallelism', 'Expert parallelism', 'MoE', 'PD disaggregation', 'LLM serving'],
    status: 'published',
    publishedAt: '2026-10-03',
  },
  {
    slug: 'jev',
    title: 'Jev：不生成文本的模型怎么做决定',
    description: '用一条工单和三个问题，拆解 decision model 怎样在一次 forward 里共享 state、隔离问题，并从 hidden state 直接读出概率。',
    category: '推理系统',
    tags: ['Jev', 'Kev', 'Decision model', 'Attention mask', 'Pointer head', 'Calibration', 'Prefix caching'],
    status: 'draft',
  },
  {
    slug: 'llm-quantization',
    title: 'LLM 量化：FP8、INT4、NVFP4 与 Marlin Kernel',
    description: '以 Qwen3.8-27B 为例，从一个数怎么用 4 bit 存讲起：16 个刻度放在哪（FP8、分组与 NVFP4），每个数落到哪个刻度（GPTQ），以及省下的字节换来的显存与速度。',
    category: '推理系统',
    tags: ['Quantization', 'FP8', 'INT4', 'NVFP4', 'Marlin', 'GPTQ', 'AWQ', 'vLLM', 'LLM serving'],
    status: 'draft',
  },
  {
    slug: 'vllm-speculative-decoding',
    title: 'Vllm · Speculative Decoding：猜多个，验一次',
    description: '从拒绝采样出发，理解 speculative decoding 为什么不改变输出分布，以及 draft model、n-gram 与 EAGLE-3 在 vLLM 中的接受率与加速比。',
    category: '推理系统',
    tags: ['vLLM', 'Speculative decoding', 'EAGLE', 'Rejection sampling', 'LLM serving'],
    status: 'published',
    publishedAt: '2026-10-02',
  },
  {
    slug: 'vllm-cpu-overhead',
    title: 'Vllm · CPU Overhead：让 GPU 不等 CPU',
    description: '从每步 decode 的 CPU overhead 出发，理解 vLLM V1 如何用进程拆分、async scheduling、Model Runner V2 与 CUDA Graph 让 GPU 不等 CPU。',
    category: '推理系统',
    tags: ['vLLM', 'Async scheduling', 'Model Runner V2', 'CUDA Graph', 'LLM serving'],
    status: 'published',
    publishedAt: '2026-10-01',
  },
  {
    slug: 'vllm-prefix-caching',
    title: 'Vllm · Prefix Caching：跨请求复用 KV Cache',
    description: '从链式 hash、free queue 与 radix tree 出发，理解 vLLM 与 SGLang 如何跨请求复用 KV cache，以及路由、分层存储与侧信道带来的边界。',
    category: '推理系统',
    tags: ['vLLM', 'Prefix caching', 'KV cache', 'SGLang', 'RadixAttention', 'LLM serving'],
    status: 'published',
    publishedAt: '2026-10-01',
  },
  {
    slug: 'vllm-scheduler',
    title: 'Vllm · Continuous Batching 与调度器',
    description: '从 token 数与算力的关系出发，理解 iteration-level scheduling、chunked prefill 与 V1 的 token budget，并对照 vLLM v0.30.0 的实现。',
    category: '推理系统',
    tags: ['vLLM', 'Continuous batching', 'Chunked prefill', 'Scheduler', 'LLM serving'],
    status: 'published',
    publishedAt: '2026-09-30',
  },
  {
    slug: 'vllm-paged-attention',
    title: 'Vllm · PagedAttention：KV Cache 的分页管理',
    description: '从 KV cache 的三种浪费出发，理解 block table、按需分配与前缀共享，并对照 vLLM v0.30.0 的实现。',
    category: '推理系统',
    tags: ['vLLM', 'PagedAttention', 'KV cache', 'Prefix caching', 'LLM serving'],
    status: 'published',
    publishedAt: '2026-09-29',
  },
  {
    slug: 'kl-nll-ce',
    title: '从 MLE 到 KL 散度：理解 NLL 与交叉熵',
    description: '从分类器的 loss 与编码长度出发，理解 NLL、CE 和 KL 的联系，以及它们在 SFT、蒸馏与 GRPO 中的作用。',
    category: '机器学习基础',
    tags: ['KL', 'NLL', 'CE', '交叉熵', 'MLE', '信息论', 'SFT'],
    status: 'published',
    publishedAt: '2026-09-23',
  },
  {
    slug: 'gradient-checkpointing',
    title: 'Gradient Checkpointing：原理、实现与自动重计算策略',
    description: '从 activation 生命周期到 PyTorch 两种实现、SAC 与自动重计算策略，结合 PyTorch 官方结果理解速度与显存的取舍。',
    category: '训练优化',
    tags: ['Gradient Checkpointing', 'PyTorch', 'Autograd', 'SAC', 'torch.compile', '显存优化'],
    status: 'published',
    publishedAt: '2026-09-22',
  },
  {
    slug: 'flash-attention',
    title: 'FlashAttention-2：从 Online Softmax 到 GPU 并行实现',
    description: '从 Online Softmax 到官方 FlashAttention-2 前向源码，理解分块计算、GPU 并行划分与 causal mask。',
    category: '算子优化',
    tags: ['FlashAttention', 'Attention', 'Online Softmax', 'CUDA', 'CUTLASS', '源码解读'],
    status: 'published',
    publishedAt: '2026-09-21',
  },
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
