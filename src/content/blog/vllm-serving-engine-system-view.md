---
title: "vLLM 的系统视角：从 Transformers.generate 到 LLM Serving Engine"
description: "从 LLM serving 的角度理解 vLLM 如何通过调度、batching、KV cache 管理和执行引擎把单请求生成变成高并发在线服务。"
pubDate: 2026-04-28
tags: ["vLLM", "LLM Serving", "Inference", "Systems", "AI Infra"]
---

很多人第一次接触 vLLM，是因为它可以用 OpenAI-compatible API 把本地模型快速起成服务。但如果只把 vLLM 理解成一个更快的 `generate` wrapper，就会错过它真正解决的问题：LLM serving。

`Transformers.generate` 解决的是“给一个模型和一个输入，生成一段输出”。vLLM 解决的是“在很多请求、不同 prompt 长度、不同输出长度、流式返回和有限 GPU 显存之间，持续安排哪些 token 该被执行”。这篇只建立系统视角，不深入 PagedAttention 和 scheduler 的实现细节。

## 1. LLM 推理和 LLM Serving 不是一回事

单请求推理最关心的是这一次能不能跑通、速度多快、显存够不够。在线 serving 关心的是另一组问题：同时来了很多请求怎么办，长 prompt 会不会拖住短请求，decode 阶段 GPU 利用率为什么上不去，KV cache 如何管理，流式输出如何边生成边返回。

这两个问题的差异可以简化成下面这样：

```text
Transformers.generate:
    one request -> one loop -> one output

vLLM:
    many requests -> scheduler -> batching -> KV cache manager -> model executor -> streaming outputs
```

`Transformers.generate` 很适合研究、离线实验和单请求脚本。它把 generation loop、sampling、stopping criteria、KV cache 等逻辑封装在一次调用里，接口清晰，容易调试。

但在线服务里，瓶颈不只是某一次 forward 有多快。真正的问题通常是：

| Serving 问题 | 为什么单请求 generate 不够 |
| --- | --- |
| 请求排队 | 多个请求同时到达，需要决定谁先执行、谁等待 |
| 动态 batch | 请求长度不同、到达时间不同，不能只做固定 batch |
| KV cache | 每个请求的 cache 会随 token 增长，占用大量 GPU 显存 |
| 流式输出 | 每生成一个或几个 token，就要把结果送回客户端 |
| GPU 利用率 | decode 每步计算很小，容易让 GPU 等内存或等调度 |
| 尾延迟 | 少数长请求或复杂采样会影响整体服务体验 |

所以“能生成”和“能服务”不是一回事。前者是模型能力接口，后者是系统调度问题。

## 2. Prefill / Decode：理解 LLM Serving 的基本执行模型

LLM 自回归生成通常可以拆成两个阶段：prefill 和 decode。

Prefill 处理输入 prompt。模型一次性读入 prompt token，计算每一层 attention 的 key/value，并把它们写入 KV cache。这个阶段输入 token 多，矩阵计算密集，通常更 compute-heavy。

Decode 逐 token 生成。每一步只输入最新 token，但要读取之前所有 token 对应的 KV cache，算出下一个 token 的 logits，再采样。这个阶段每步计算量相对小，但要反复读写模型权重和 KV cache，通常更 memory-bandwidth-heavy。

一个请求的延迟可以粗略写成：

```text
Request latency = waiting time + prefill time + decode time
```

这个公式虽然简单，但很有用。等待时间来自排队和调度；prefill time 受 prompt 长度、batch 规模和计算吞吐影响；decode time 受输出 token 数、KV cache 访问、batching 和采样策略影响。

在真实服务里，不同 workload 的瓶颈不同：

| Workload | 典型瓶颈 |
| --- | --- |
| 短 prompt、长输出聊天 | decode 时间和 decode 吞吐 |
| 长文档总结 | prefill 计算和 KV cache 分配 |
| 多轮 agent | prefix/cache 复用、工具等待、长上下文管理 |
| 高并发 API | 调度策略、batching、尾延迟和显存碎片 |

需要注意的是，prefill/decode 是理解 LLM serving 的基本概念模型，不等于 vLLM 内部永远严格把两者切成两个独立系统。vLLM V1 的 scheduler 更接近统一 token budget：对每个请求动态分配本轮要执行多少 token，让 chunked prefill、prefix caching、speculative decoding 等能力可以放进同一套调度框架。

## 3. vLLM 的核心抽象

理解 vLLM，不需要一开始就读 CUDA kernel。先抓住这些抽象就够了：

| 概念 | 含义 |
| --- | --- |
| Request | 客户端发来的生成任务，包含 prompt、采样参数、停止条件和输出约束 |
| Sequence | 正在生成的一条 token 序列；beam search 或并行采样时，一个 request 可能对应多条 sequence |
| Token | vLLM 调度和执行的最小语义单位；prefill 是处理输入 token，decode 是生成输出 token |
| Batch | 某一轮被一起送进模型执行的 token 集合，不一定等于客户端请求的固定 batch |
| KV cache | 每层 attention 保存的 key/value，中间状态会随序列增长，占据大量 GPU 显存 |
| Scheduler | 决定哪些 request/sequence 在这一轮执行，以及每个请求执行多少 token |
| Worker | 实际持有模型、执行 forward、运行 attention/kernel 的进程或设备执行单元 |
| Sampling | 根据 logits、temperature、top-p、top-k、stop tokens 等规则选择下一个 token |

这些抽象背后的重点是：vLLM 把“请求级别”的服务问题，拆成“token 级别”的执行问题。

客户端看到的是 request：我要问一个问题，生成一段回答。GPU 看到的是 token：这一轮有哪些 token 要算，它们的 KV cache 在哪里，算完后哪些请求结束，哪些请求还要继续进入下一轮。

## 4. 一次请求在 vLLM 中的生命周期

一个请求进入 vLLM 后，可以用下面的流程理解：

```text
Client
  ↓
OpenAI-compatible API Server
  ↓
Tokenizer / Chat Template
  ↓
Request Queue
  ↓
Scheduler
  ↓
KV Cache Allocation
  ↓
Prefill
  ↓
Decode Loop
  ↓
Sampler
  ↓
Detokenizer
  ↓
Streaming Response
```

每一层的职责不同。

API server 负责把外部请求转成 vLLM 能理解的输入。它处理 OpenAI-compatible endpoint、请求参数、流式响应和错误返回。对用户来说，这是最像“服务”的一层。

Tokenizer / chat template 负责把消息结构转成模型 token。chat model 通常不直接吃 `{role, content}`，而是吃模板化后的文本序列。模板处理错了，模型能力再强也会表现异常。

Request queue 接住还没被执行的任务。在线系统里，请求不是整齐同时到达，而是持续涌入；queue 是调度器看到的待处理工作集合。

Scheduler 是 vLLM serving 的大脑。它要在吞吐和延迟之间做取舍：哪些请求可以一起执行，长 prompt 要不要切块，decode 请求是否优先，显存不够时如何处理，什么时候认为一个请求完成。

KV cache allocation 决定请求的中间状态放在哪里。vLLM 最著名的 PagedAttention，就是围绕这个问题展开：KV cache 大、动态、长度不可预测，如果按最大长度预留，会浪费大量显存。

Prefill 执行 prompt 部分，把输入 token 转成 KV cache。Decode loop 则在每一轮执行新 token，读已有 cache，生成下一个 token，再把请求放回调度循环，直到满足停止条件。

Sampler 根据 logits 和采样参数选择 token。Detokenizer 把 token 转回文本。Streaming response 则把增量结果持续返回给客户端，而不是等完整回答生成后一次性返回。

## 5. vLLM 的架构分层

从系统分层看，vLLM 可以粗略分成三层。

| 层 | 主要职责 |
| --- | --- |
| API Layer | OpenAI-compatible server、请求解析、鉴权/参数处理、streaming response |
| Engine Layer | request queue、scheduler、batching、KV cache manager、输出组装 |
| Execution Layer | model executor、GPU worker、attention kernel、sampling kernel、分布式执行 |

API Layer 让 vLLM 像一个服务。它不是核心性能来源，但决定外部系统如何接入、如何流式消费结果、如何兼容 OpenAI SDK 和客户端生态。

Engine Layer 是 vLLM 和普通 `generate` wrapper 拉开差距的地方。它不只是把请求丢给模型，而是持续维护系统状态：哪些请求在等待，哪些请求正在 decode，哪些 KV block 已分配，哪些 token 可以合批执行，哪些输出该返回。

Execution Layer 负责把调度结果变成 GPU 上的实际计算。模型权重、attention kernel、tensor parallel、pipeline parallel、CUDA graph、quantization 等能力都落在这一层或与这一层强相关。

这三层也解释了为什么 vLLM 不是“单个优化点”。PagedAttention 解决 KV cache 管理，scheduler 解决 token 执行顺序，worker 和 kernels 解决 GPU 执行效率，API server 解决在线服务接口。它们组合起来，才是 serving engine。

## 6. 本篇总结

vLLM 的核心价值不是让单个请求极限加速，而是在多请求、多长度、多阶段的 serving 场景下，持续把 GPU 资源调度给最该执行的 token。

用一句话概括：

```text
vLLM = API server + scheduling engine + KV cache manager + model execution runtime
```

这篇建立了四个基本判断：

| 判断 | 含义 |
| --- | --- |
| LLM inference 不等于 LLM serving | 单请求能跑，不代表能支撑高并发在线服务 |
| Prefill / decode 是基本执行模型 | prompt 处理和逐 token 生成的瓶颈不同 |
| KV cache 是 serving 的核心资源 | 显存管理决定能同时服务多少请求 |
| Scheduler 决定系统行为 | 高吞吐、低延迟和公平性都要通过调度折中 |

后续如果继续深入，最应该看两块：KV cache 为什么会成为瓶颈，以及 vLLM scheduler 如何把不同阶段、不同长度、不同状态的请求合并成 GPU 能高效执行的 token batch。

<div class="reference-list">
  <a class="reference-card" href="https://docs.vllm.ai/en/stable/design/arch_overview.html">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>vLLM Architecture Overview</strong>
      <small>docs.vllm.ai</small>
    </span>
  </a>
  <a class="reference-card" href="https://docs.vllm.ai/en/stable/usage/v1_guide.html">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>vLLM V1 Guide</strong>
      <small>docs.vllm.ai</small>
    </span>
  </a>
  <a class="reference-card" href="https://vllm.ai/blog/vllm">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention</strong>
      <small>vllm.ai</small>
    </span>
  </a>
  <a class="reference-card" href="https://arxiv.org/abs/2309.06180">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>Efficient Memory Management for Large Language Model Serving with PagedAttention</strong>
      <small>arxiv.org</small>
    </span>
  </a>
  <a class="reference-card" href="https://huggingface.co/docs/transformers/main/en/main_classes/text_generation">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>Transformers GenerationMixin.generate</strong>
      <small>huggingface.co</small>
    </span>
  </a>
</div>
