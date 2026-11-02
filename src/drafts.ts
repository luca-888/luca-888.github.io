import type { Post } from './posts';

// 未发布的文章，只在 dev 中列出；发布时移到 posts.ts 并填写日期。
export const drafts: Post[] = [
  {
    slug: 'cuda-graph-correctness',
    title: 'CUDA Graph 的正确性：捕获遗漏与过期输入',
    description: 'Replay 等价于 eager 需要两个前提：每个 GPU 操作都录进图，每个随批次变化的输入都在 replay 前写入固定地址。Liger-Kernel 的 RMSNorm backward 漏捕获了 kernel，vLLM 的 EAGLE draft prefill 在 FULL 图中拿不到本批图像 embedding。',
    category: 'GPU 编程',
    tags: ['CUDA Graph', 'Stream capture', 'CuTe DSL', 'Liger-Kernel', 'vLLM', 'EAGLE', 'Speculative decoding'],
    status: 'draft',
  },
  {
    slug: 'qwen-vl-video-tokens',
    title: 'Qwen-VL 的视频 token：采样、grid 与时间戳',
    description: 'Qwen2-VL 与 Qwen3-VL 如何把一段视频变成 prompt 中的 token：帧采样、奇数帧补齐、smart_resize、video_grid_thw 与时间戳，以及不解码视频计算 token 数、多个视频时占位符与元数据的对应。',
    category: '多模态',
    tags: ['Qwen3-VL', 'Qwen2-VL', 'VLM', 'Video', 'LlamaFactory', 'transformers', 'vLLM'],
    status: 'draft',
  },
  {
    slug: 'semantic-cache-verification',
    title: 'Semantic Cache 的命中校验：误命中、校验器故障与 fail-closed',
    description: 'Semantic cache 靠 embedding 相似度命中，意思相反的问题也可能命中。命中校验的做法，以及校验器故障时 fail-open 与 fail-closed 的取舍、请求取消与候选相似度的处理。',
    category: '推理系统',
    tags: ['Semantic cache', 'Semantic Router', 'NLI', 'Fail-closed', 'LLM serving'],
    status: 'draft',
  },
  {
    slug: 'liger-hf-patching',
    title: 'Liger-Kernel 的模型接入：类替换、实例 patch 与静默回退',
    description: 'Liger-Kernel 怎样把 Triton kernel 接入 Hugging Face 模型：构建前替换类与构建后 patch 实例的区别，漏掉的 patch 为什么静默回退到原实现，以及 patched forward 中 logits_to_keep、shift_labels 与 FLCE 的关系。',
    category: '训练优化',
    tags: ['Liger Kernel', 'Hugging Face', 'transformers', 'Monkey patch', 'RMSNorm', 'FLCE', 'Qwen'],
    status: 'draft',
  },
  {
    slug: 'bf16-weight-offset',
    title: 'Gemma RMSNorm 的 1 + w：BF16 舍入与 weight offset 的相加位置',
    description: 'Gemma 的 RMSNorm 把缩放系数写成 1 + w。weight 以 BF16 存储时，在 BF16 中加 1 会舍掉 w 的低位：BF16 的 ulp 与舍入、误差怎样进入 Y 与 dX，以及把 offset 移进 kernel、在 FP32 中相加的做法。',
    category: '算子优化',
    tags: ['RMSNorm', 'Gemma', 'BF16', '数值精度', 'CuTe DSL', 'Liger Kernel'],
    status: 'draft',
  },
  {
    slug: 'pd-scheduling',
    title: 'PD 分离的调度：路由、接纳与比例',
    description: 'PD 分离之后，一个请求要经过四次调度决策：选哪个 P、要不要远端 prefill、D 何时接纳、P 与 D 各几个。Mooncake、SGLang、vLLM、llm-d 与 Dynamo 的做法与取舍。',
    category: '推理系统',
    tags: ['PD disaggregation', 'LLM serving', 'Scheduling', 'KV cache', 'vLLM', 'SGLang', 'Dynamo', 'Mooncake'],
    status: 'draft',
  },
  {
    slug: 'jev',
    title: 'Jev：不生成文本的模型怎么做决定',
    description: '用一条工单和三个问题，拆解 decision model 怎样在一次 forward 里共享 state、隔离问题，并从 hidden state 直接读出概率。',
    category: '推理系统',
    tags: ['Jev', 'Kev', 'Decision model', 'Attention mask', 'Pointer head', 'Calibration', 'Prefix caching'],
    status: 'draft',
  },
];
