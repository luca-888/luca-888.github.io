# CUDA Graph 的两类正确性问题：捕获遗漏与过期输入 文章资料

状态：2026-10-10 调研完成并提交参考方案。同日协调会话“公开有价值的PR汇总”转达确认（称用户委托其协调 5 篇文章），并定下：图 3 不画；图 1、图 2 跳过草图直接精画；M1–M4 不测，正文留占位；待核实项能核实的核实，不能核实的不写进正文；vLLM 案例写成“尚未合并的修复”，可写 main 仍有问题，#43716 自动关闭不写；理解检查题不等用户回答，附参考答案。据此完成正文 `content/cuda-graph-correctness.md`（`status: 'draft'`）、两张图与页面；未 commit。注意：方案确认来自协调会话转达，不是用户在本会话中直接确认。

博客已有 `content/cuda-graph.md`（Definition / Instantiation / Execution、四节点 diamond、多 stream 捕获、4090 上 60.80 → 3.63 μs）。本篇只就地交代出错机制所需的背景，不重复那篇。

## 源码基准

| 仓库 | commit | 日期 | 用途 |
| --- | --- | --- | --- |
| Liger-Kernel（上游，`reference-repos/Liger-Kernel-upstream`） | `d5f2817` | 2026-10-07 | 现状核查；PR #1450 merge commit `ad2436e` |
| vLLM | `c41b263` | 2026-10-10 | 现状核查；PR #43776 base `158289e0` |
| SGLang | `6fc8d9d` | 2026-10-10 | 静态输入 buffer 管理对比 |
| PyTorch | `a6b28b6` | 2026-09-30 | capture 规则、stream 创建 flag |
| CUTLASS（CuTe DSL） | `0b55a2f` | 2026-09-23 | launch 不带 stream 时的默认值、env stream 文档、testing 辅助函数 |

`reference-repos/Liger-Kernel/` 是用户 fork 的完整克隆，没有动；上游另加 `Liger-Kernel-upstream` 与 `cutlass` 到 `sync.sh`。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [linkedin/Liger-Kernel#1450](https://github.com/linkedin/Liger-Kernel/pull/1450) diff、描述、测试 | luca-888（作者本人） | 已读 | 核心 | 二、三 |
| 2 | [vllm-project/vllm#43776](https://github.com/vllm-project/vllm/pull/43776) diff、描述、测试（未合并，有冲突） | luca-888（作者本人） | 已读 | 核心 | 四、五 |
| 3 | [vllm-project/vllm#43716](https://github.com/vllm-project/vllm/issues/43716) 问题报告与最小复现（A100，fake draft model） | glaziermag | 已读（2026-09-30 因不活跃自动关闭） | 核心 | 四 |
| 4 | [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)（Cross-stream Dependencies、Prohibited and Unhandled Operations、Invalidation、Graph update） | NVIDIA | 已读相关节（WebFetch，前 100K 字符） | 核心 | 一、二 |
| 5 | [CUDA Runtime API: Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)（`cudaStreamBeginCapture` mode、`cudaThreadExchangeStreamCaptureMode`） | NVIDIA | 已读相关条目（WebFetch） | 核心 | 二 |
| 6 | PyTorch `docs/source/notes/cuda.md` CUDA Graphs 节（Constraints、multistream）、`aten/src/ATen/cuda/CUDAGraph.cpp` `capture_begin`、`c10/cuda/CUDAStream.cpp` | PyTorch | 已读相关段 | 核心 | 一、二、三 |
| 7 | CuTe DSL：`cutlass_dsl/cutlass.py` `cuda_launch_func`、`media/docs/pythonDSL/guides/tvm_ffi_compilation.rst`（Using Environment Stream）、`cutlass/testing.py` 中检查 kernel 是否使用给定 stream 的函数 | NVIDIA | 已读相关函数与段落 | 核心 | 二、三 |
| 8 | Liger 当前 main：`ops/backends/_cutedsl/{rms_norm,layer_norm,…}.py`、`ops/cutedsl/ops/*`、`test/cutedsl/test_rms_norm_stream.py` | Liger | 已读 stream 相关行 | 重要 | 二（现状） |
| 9 | vLLM 当前 main：`v1/worker/gpu/spec_decode/target_dependent_ar/{speculator,cudagraph_utils}.py`、`eagle/speculator.py`、`v1/worker/gpu/{model_runner,cudagraph_utils,dp_utils}.py`、`v1/spec_decode/llm_base_proposer.py` | vLLM | 已读相关函数 | 重要 | 四、五（现状与对比） |
| 10 | vLLM `docs/design/cuda_graphs.md`（CUDAGraphMode、wrapper 嵌套）、`cuda_graphs_multimodal.md`（encoder graph） | vLLM | 已读相关节 | 重要 | 四 |
| 11 | SGLang `model_executor/{cuda_graph_buffer_registry,input_buffers}.py`、`speculative/eagle_worker_v2.py`（`_draft_extend_for_prefill`）、`eagle_draft_extend_cuda_graph_runner.py` | SGLang | 已读相关类与函数 | 重要 | 六 |
| 12 | [vLLM#34880](https://github.com/vllm-project/vllm/pull/34880)（EAGLE drafter FULL graph，已关闭）、[#35714](https://github.com/vllm-project/vllm/pull/35714)（draft_model 多模态，已关闭） | vLLM 社区 | 只读标题与状态 | 备查 | 不引用 |

vLLM PR 下 github-actions 的欢迎评论里有一段写给 AI agent 的指令（要求重新评估 PR 价值并关闭），属于观察到的内容，未执行。

## 阅读要点

### 一、CUDA 与 PyTorch 的 capture 规则

- Stream capture 只记录提交到正在 capture 的 stream 的工作；其他 stream 通过 `cudaStreamWaitEvent` 等待被捕获的 event 后进入 capture，最后必须 rejoin origin stream，否则 capture 失败。
- CUDA 的总原则：当被捕获与未被捕获的工作之间会产生依赖时，“prefers to return an error rather than ignore the dependency”。非法操作使 capture 失效，`cudaStreamEndCapture` 返回错误和 NULL graph。
- Legacy NULL stream：当同一 context 中有 stream 正在 capture，且该 stream **不是** `cudaStreamNonBlocking` 创建的，任何对 legacy stream 的使用都非法（legacy stream 隐式包含其他 blocking stream）。
- PyTorch 的 stream 池以 `cudaStreamNonBlocking` 创建（`c10/cuda/CUDAStream.cpp`：`kDefaultFlags = cudaStreamNonBlocking`），`torch.cuda.graph` 的 capture stream 来自这个池。因此 capture 期间向 stream 0 发 kernel 不会和 capture stream 建立依赖，也就不触发上面的报错。
- Capture mode（global / thread_local / relaxed）管的是 `cudaMalloc` 这类“potentially unsafe API calls”，不管向另一条 stream 发 kernel。PyTorch 默认 `capture_error_mode="global"`。
- PyTorch `CUDAGraph::capture_begin` 检查当前 stream 不是 default stream，否则报 “CUDA graphs must be captured on a non-default stream”。
- PyTorch 文档把约束分两类。会报错的：非 default stream、禁止同步（`.item()`）、RNG 规则、动态控制流。**可能静默出错的**：同一进程只能有一个 capture；“No non-captured CUDA work may run in this process (on any thread) while capture is underway”；CPU 工作不被捕获，replay 时省略；每次 replay 读写同一地址；不允许动态 shape；多 stream 有限制。两个案例分别落在“non-captured CUDA work”和“CPU 工作 / 地址固定”上。
- Graph update（`cudaGraphExecUpdate`、`cudaGraphExecKernelNodeSetParams`）可在拓扑不变时改参数，下次 launch 生效。框架通常不用它，而是固定地址、replay 前把新数据 copy 进去。

### 二、Liger #1450：kernel 没进图

- 机制链：
  1. CuTe DSL 的 `cuda_launch_func` 在没有 stream 参数时，把 `Int64(0)` cast 成 stream，即 stream handle 0。
  2. 修复前，`_LigerRMSNormCuTeDSLBackward.__call__` 没有 `stream` 参数，`.launch(...)` 不带 `stream=`；`cute.compile` 也没有传 env stream 占位。于是 backward kernel 每次都发到 stream 0。
  3. Eager（默认 stream 上跑）时一切正常，所以只有在非默认 stream 上运行或 capture 时才暴露。
  4. Capture 时，torch 的 capture stream 是 non-blocking，stream 0 上的 launch 合法且不进图：kernel 在 capture 当下真实执行一次（用 capture 时的 dy），写入 `dx` 与 `dw_partial`。随后 host 端的 `dw_partial.sum(dim=0)` 和 dtype 转换是 PyTorch 算子，发在 current stream 上，被捕获。
  5. Replay 时只有 reduction 等后续算子执行：`dW` 由 capture 时留下的 `dw_partial` 求和得到，`dX` 停留在 capture 时的值。结果是 replay 返回“capture 时那次 upstream gradient 的梯度”，与本次 dy 无关。
  6. 另外，capture 时 stream 0 上的 kernel 与 capture stream 之间没有顺序保证，capture 当次的结果本身也可能有竞争。（推论，待核实：是否值得写进正文。）
- 修复：`__call__` 增加 `stream: cuda.CUstream`，`.launch(..., stream=stream)`；`cute.compile` 传 `cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)`，调用时由 TVM FFI 自动填入当前 PyTorch stream，调用端代码不变。CuTe DSL 文档推荐 env stream，理由是调用开销更小且与框架集成更简单。
- 修复前代码注释写着 “CuTe DSL compiled kernels read torch.cuda.current_stream() at launch; the kernel ABI does NOT take a stream positional. Mirrors the LayerNorm sibling's launch call.” 这是错误前提：LayerNorm 的调用端确实不传 stream，但它在 compile 时声明了 env stream。RMSNorm 抄了调用端，漏了声明。当前 LayerNorm 文件里同样的注释仍在（`layer_norm.py:684`），但 LayerNorm 本身是对的。
- FusedAddRMSNorm 共用 `_rms_norm_cutedsl_backward`，带 `ds` epilogue，一并修复。
- 测试：两组不同 upstream gradient，先在默认 stream 上 eager 求参考值，再在 side stream 上 warmup 3 次、capture、逐组 `copy_` 进 static buffer 后 replay，`atol=0, rtol=0` 比较。Triton backend 作为对照（Triton launch 用当前 torch stream）。in-place backward 在图内 `clone()` dy，避免覆盖 static buffer。
- PR 中的测量（可用，标明出自 PR）：H100，PyTorch 2.9.1 / CUDA 12.8 / CuTe DSL 4.6.0 / TVM FFI 0.1.13.post3；修复前 8 个 CuTe 用例全失败、8 个 Triton 对照全通过；修复后 16 个 graph 回归 + 97 个已有 RMSNorm 用例全通过，B200 同。
- 现状（`d5f2817`）：修复仍在；`_cutedsl/` 与 `cutedsl/ops/` 下所有 `.launch(` 都带 `stream=stream`，各 compile 处都用 env stream。回归测试 `test/cutedsl/test_rms_norm_stream.py` 仍在。
- 相关做法：CUTLASS `cutlass/testing.py` 有一个辅助函数，在给定 stream 上 begin capture、跑 kernel、检查图里是否有节点，以此判断 kernel 是否用了传入的 stream；注释称较新驱动上向其他 stream 的 launch 在 capture 期间会直接失败（待核实驱动版本与具体条件）。

### 三、vLLM #43776：回放时用了过期输入

- 路径（PR base `158289e0`，当前代码已搬到 `target_dependent_ar/speculator.py`，逻辑相同）：
  1. `_run_model` 里，`supports_mm_inputs` 为真时调用 `self.model.embed_input_ids(input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)`，写入持久 buffer `self.inputs_embeds`。这一步把图像 token 位置的 embedding 替换成 encoder 输出。
  2. FULL 模式下整个 `_prefill`（embedding 合并 + draft forward + logits + 采样）被录成一张图。Capture 由 `SpeculatorCudaGraphManager.capture` 驱动，`forward_fn` 不带 `mm_inputs`，所以录下的是 `embed_input_ids(input_ids, None, None)`：只有文本 embedding 查表，没有图像特征的 scatter。
  3. Replay 时 `run_fullgraph(desc)` 不接收也不写入 `mm_inputs`。`input_ids` 是持久 buffer，所以文本 embedding 按本批重算；图像占位 token 位置拿到的是占位 token id 的 embedding，本批图像特征从未进入图。
  4. PIECEWISE 与 eager 下，`embed_input_ids` 在 `_run_model` 里、编译区之外以 eager 执行，每批都带 `mm_inputs`，所以没有问题。
- 与 target model 的对照：`model_runner.py` 在 replay 前用 `model_state.prepare_inputs_embeds(...)` 准备好 `inputs_embeds`，注释写明 “we don't need to pass the input tensors, because they are already copied to the CUDA graph input buffers”。V1 proposer（`llm_base_proposer.py` 的 `build_model_inputs_first_pass`）也是在模型调用前 eager 合并、写进 `self.inputs_embeds`。MRV2 draft prefill 的 FULL 边界把合并步骤包进了图，这是两者的差别。
- 后果：draft 看到的图像位置 embedding 错误，draft token 质量下降；最终输出由 target model 验证，正确性受保护（待核实：在 vLLM 当前 rejection sampler 下是否严格无损，以及 acceptance rate 下降幅度）。
- Issue 的最小复现（A100-40GB，torch 2.4.1，fake draft model，强制 `supports_mm_inputs=True`）：eager 下图像 A / B 的 mm 位置 embedding sum 为 48 / −112，FULL replay 两次都是 5.04，与图像无关。issue 提出短期禁用 FULL、长期为 mm embedding 与 `is_mm_embed` mask 建持久 buffer 并在 replay 前更新。
- PR 做法：`CudaGraphManager.dispatch` 增加 `invalid_modes`；`_get_prefill_invalid_cudagraph_modes(mm_inputs)` 只在 `mm_inputs` 非空且有 embedding 时返回 `{FULL}`；不按 `supports_mm_inputs` 判断，因为多模态模型的纯文本 batch 仍可用 FULL。
- DP：`sync_cudagraph_and_dp_padding` 用 all_reduce 收集各 rank 期望的 mode，取 `min` 作为同步结果（NONE=0 < PIECEWISE=1 < FULL=2）；NONE 时全部 eager。否则用 `max(num_tokens)` 重新 `dispatch`，**重新 dispatch 不看同步出的 mode**，可能把已降级为 PIECEWISE 的 rank 选回 FULL。PR 在同步 mode 不是 FULL 时把 FULL 加入 `invalid_modes` 再 dispatch。
- 现状（`c41b263`）：
  - FULL 分支仍是 `run_fullgraph(prefill_batch_desc)`，capture 仍不带 `mm_inputs`；问题仍在。PR 未合并、`needs-rebase`；issue 已因不活跃自动关闭。
  - `dispatch_cg_and_sync_dp` 新增 `dp_sync_state`：draft prefill 复用 target 这一批已达成的 DP 共识，不再单独做 collective。PR 的 DP 部分需要按新结构重新设计（rebase 时处理，正文是否提待定）。
  - `sync_cudagraph_and_dp_padding` 的重新 dispatch 仍不看 `synced_cg_mode`（只处理 NONE）。

### 四、静态输入管理的对比

- PyTorch 手动 capture：调用方负责 static input / output；文档示例在 replay 前 `static_input.copy_(data)`。
- vLLM：持久 buffer（`InputBuffers`、`inputs_embeds`、`hidden_states` 等）在初始化时分配，capture 与 replay 共用地址；replay 前的准备阶段写入本批数据；`SpeculatorCudaGraphManager` 每次 capture 都重建 dummy 输入与 attention metadata，避免共享 buffer 内容与当前 descriptor 不符。
- SGLang：`CudaGraphBufferRegistry` 把“ForwardBatch 字段 → 图内 buffer”声明成 `GraphSlot`，每个 slot 带 padding policy（KEEP_PAD / FILL_SENTINEL / ZERO / FOREACH_COPY / FILL_ONCE）。`positions`、`out_cache_loc`、`req_pool_indices` 等索引语义 buffer 的 padding 区必须清零，“stale content is unsafe to execute”。`input_embeds` slot 由模型在图内写入，replay 前只清 padding（`copy_from_fb=False`）。多个 runner 通过 `share_input_buffer` 共享同名 buffer，安全前提是“filled immediately before each replay”。EAGLE draft prefill（`_draft_extend_for_prefill`）把 `mm_input_embeds` 随 batch 传入 extend forward。
- 共同点：图外准备、图内只读固定地址。区别在于 SGLang 把这份契约集中声明，vLLM 分散在各 runner 的准备代码里。

## 文章主线（2026-10-10 经协调会话确认）

CUDA Graph 的 replay 等价于 eager，需要两个前提：eager 的每个 GPU 操作都录进了图（覆盖）；每个随批变化的输入都放在固定地址、并在 replay 前写入（新鲜）。两个案例各破坏一个前提，而且都不报错：前者因为 stream 0 与 non-blocking capture stream 之间没有依赖，CUDA 没有理由报错；后者因为图在语法上完全合法，只是少了一步。结论落到检测方法（改变输入后与 eager 逐位比较、数图节点）和框架的约定（图外准备、集中声明静态输入）。

## 待核实（2026-10-10 处理结果）

1. **stream handle 0 是 legacy NULL stream 还是 per-thread default stream**：无法从开源代码确认（lowering 在闭源的 MLIR 部分）。正文只写“stream 0（handle 为 0 的默认 stream）”。两种情况下结论相同：legacy stream 与 non-blocking 的 capture stream 没有隐式同步；per-thread default stream 只与 legacy stream 同步。所以正文“不产生依赖、不报错”的推理不依赖这一点。
2. **CUTLASS 注释“较新驱动上 foreign-stream launch 在 capture 期间直接失败”**：注释未给版本，CUDA 文档里也没有找到对应条款。未核实，正文不写。
3. **修复前 capture 当次 backward kernel 是否真的执行**：按 CUDA 语义，向未在 capture 的 stream 发的 launch 是普通 launch，会执行。正文第一章的表格写了这条一般语义；描述案例时只写“replay 读到 buffer 里留下的旧数据，与本次 dy 无关；dX 不会被写入”，不断言旧数据就是 capture 那次 dy 的结果（没有硬件验证，且 capture 当次 kernel 与 capture stream 之间没有顺序）。M1/M2 占位可以回答这个问题。
4. **EAGLE 验证是否无损**：已从源码确认（`spec_decode/rejection_sampler_utils.py`）。greedy 时 `accepted = target_argmax == draft_sampled`，拒绝时写入 target argmax；采样时 `accepted = target_logprob > log(u) + draft_logprob`，拒绝后从 `max(p − q, 0)` 重新采样（`USE_BLOCK_VERIFICATION` 时用 block verification）。Draft logits 由 FULL 图内同一份 embedding 算出，与 draft token 的采样分布一致，所以 bug 只影响 acceptance rate。`SYNTHETIC_MODE`（人为设定接受率）是 benchmark 功能，与此无关。
5. **怎样称呼未合并的 PR**：按协调会话的决定，写成“作者提交的修复……尚未合并”，并写 main 中 FULL 路径仍不传 `mm_inputs`；#43716 自动关闭一事不写。

补充核实：**触发条件**。`CUDAGraphMode` 中 `FULL_AND_PIECEWISE = (FULL, PIECEWISE)`，`decode_mode()` 为 FULL，`mixed_mode()` 为 PIECEWISE。`_init_candidates` 只给 uniform decode 建 FULL descriptor（带 `uniform_token_count`）。draft prefill 的 `uniform_token_count` 来自 `get_uniform_decode_token_count`，要求 `decode_graph_eligible`，而 `model_runner.py` 中该值为 `not has_prefill`（唯一例外是只新增 1 个 token 的 prefill）。所以默认模式下，draft prefill 只有不含 prefill 的 batch 才走 FULL；`cudagraph_mode=FULL` 时 `mixed_mode()` 是 FULL，含图像的混合 batch 也走 FULL。

## 实测占位（本阶段不做）

| # | 内容 | 硬件 | 放在哪一章 |
| --- | --- | --- | --- |
| M1 | Liger 修复前后：capture 得到的 graph 节点数（kernel 节点列表）、replay 梯度与 eager 的最大绝对误差 | H100 或 B200（Liger CuTe backend 的测试环境；是否支持 4090 待核实） | 二、三 |
| M2 | 修复前 capture 期间的 nsys 时间轴：backward kernel 落在 stream 7（默认）而其余算子在 capture stream | 同 M1 | 二 |
| M3 | vLLM：Qwen2.5-VL / Qwen3-VL + EAGLE3 draft，带图像的请求在 FULL（bug）与 PIECEWISE（修复）下的 acceptance length | A100 80GB / H100 | 四 |
| M4（可选） | 同上，draft prefill 耗时 FULL vs PIECEWISE，说明降级的代价 | 同 M3 | 四 |

## 理解检查（供用户自测，附参考答案）

1. **修复前，Liger 的 capture 为什么不报错？如果 capture stream 是 blocking stream，会有什么不同？**
   参考答案：kernel 发到 stream 0，而 PyTorch 的 capture stream 是 `cudaStreamNonBlocking` 创建的，两者没有隐式同步，被捕获与未被捕获的工作之间没有依赖，CUDA 只在两者会产生依赖时报错；capture mode 只管 `cudaMalloc` 这类调用。若 capture stream 是 blocking stream，而 stream 0 是 legacy stream，CUDA 文档规定此时任何对 legacy stream 的使用都非法，capture 会失效，`cudaStreamEndCapture` 返回错误，问题会以报错的形式暴露。

2. **修复前，replay 得到的 dW 和 dX 分别对应哪一次的 dy？测试为什么要用两组不同的 dy、并且 `atol=0`？**
   参考答案：图中只有 `dw_partial.sum(0)` 等后续算子。dW 是对 buffer 中留下的 `dw_partial` 求和，这份数据不是本次 dy 算出来的（最可能来自 capture 当下 kernel 在 stream 0 上的那次执行，但未经硬件验证）；dX 在 replay 中根本不被写入，停留在之前的内容。只用一组 dy 时，replay 结果可能碰巧等于 capture 时留下的结果，测试会漏掉问题；换一组 dy 后，旧数据与正确结果不同。参考值来自同一 backend 的 eager 执行，reduction 顺序相同，所以可以要求逐位一致。

3. **vLLM 的 bug 中，replay 时哪部分 embedding 是新的、哪部分缺失？为什么 PIECEWISE 不受影响？**
   参考答案：`input_ids` 在持久 buffer 中，replay 前已写入本批，所以文本 embedding 查表是按本批重算的；缺失的是把图像占位位置替换成 encoder 输出的那一步，FULL 图录制时 `mm_inputs=None`，replay 时也不传入，占位位置得到的是占位 token 自身的 embedding。PIECEWISE 只录 draft forward 中的可编译分段，embedding 合并在图外以 eager 执行，每批都带本批 `mm_inputs`。

4. **这个 vLLM bug 会让最终输出出错吗？影响的是哪个指标？**
   参考答案：不会。验证是无损的：greedy 只接受等于 target argmax 的 draft token；采样时按 $p(x) > u \cdot q(x)$ 接受，$q$ 是 draft 自己的 logits，与 draft token 出自同一张图，拒绝后从 $\max(p-q,0)$ 重采样，输出分布仍是 target model 的分布。受影响的是 acceptance rate（以及由此带来的吞吐与延迟）。另外，默认 `FULL_AND_PIECEWISE` 下 draft prefill 只有不含 prefill 的 batch 走 FULL，主要在 `cudagraph_mode=FULL` 时触发。

5. **DP 下，为什么一个 rank 降级后其他 rank 也不能用 FULL？PR 修的是哪一步重新 dispatch？**
   参考答案：DP 各 rank 在 MoE 等层要做集合通信，vLLM 要求所有 rank 使用同一种运行模式和同一个 padding 后的 token 数，所以同步时取各 rank 期望模式的最小值。问题出在同步后的第二步：按最大 token 数重新 dispatch 时不看同步出的模式，可能把已降级的 rank（例如含图像、只能用 PIECEWISE 的 rank）重新选回 FULL。PR 在同步结果不是 FULL 时把 FULL 加入 `invalid_modes` 再 dispatch。当前 main 中 draft prefill 改为复用 target 的 DP 共识（`dp_sync_state`），rebase 时这部分要按新结构重做。

## 交付（2026-10-10）

正文 `status: 'draft'`，未 commit、未 push。正文约 6.8K 字（不含参考）；完成成稿自查（统一“持久 buffer”“acceptance rate”“运行模式”的用词，补充开头段的硬件与指代），再由不带写作上下文的 subagent 用 de-ai-edit 改 12 处；按其意见把七个章节标题改为名词短语。在本会话自己起的 dev server（端口 57037）上检查：页面 200，两张图正常显示，无 KaTeX 错误，console 无本页报错。

新增文件：

- `content/cuda-graph-correctness.md`
- `docs/cuda-graph-correctness-notes.md`
- `posts/cuda-graph-correctness/index.html`
- `src/articles/cuda-graph-correctness/cuda-graph-correctness.tsx`
- `src/articles/cuda-graph-correctness/cuda-graph-correctness.blocks.tsx`
- `src/articles/cuda-graph-correctness/CudaGraphCorrectnessFigures.tsx`（图 1 `CaptureMiss`、图 2 `GraphBoundary`）

修改的共享文件：

- `vite.config.ts`：`build.rollupOptions.input` 末尾新增一条 `cudaGraphCorrectness: 'posts/cuda-graph-correctness/index.html'`。
- `src/posts.ts`：`posts` 数组开头新增 `slug: 'cuda-graph-correctness'` 一条（category `GPU 编程`，`status: 'draft'`）。

仓库外的改动（不在本仓库提交范围内）：`~/Documents/ChatGPT/reference-repos/sync.sh` 的 `REPOS` 新增 `Liger-Kernel-upstream` 与 `cutlass` 两行，并新克隆了 `Liger-Kernel-upstream/`。

