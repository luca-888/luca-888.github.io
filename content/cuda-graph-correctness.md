**CUDA Graph 的 replay 与 eager 执行等价，需要两个前提：eager 中的每个 GPU 操作都录进了图，本文称为“覆盖”；每个随批次变化的输入都放在固定地址、并在 replay 前写入，本文称为“新鲜”。** 违反任一前提，replay 通常不报错，只是算出另一份结果。本文用作者提交的两个修复说明这两类问题：Liger-Kernel 的 CuTe DSL RMSNorm backward 漏捕获了核心 kernel，修复前 H100 上 8 个 CUDA Graph 回归用例全部失败；vLLM 的 EAGLE draft prefill 在 FULL 图中缺少本批的图像 embedding，后一个修复尚未合并。

## 一、图中记录的内容

PyTorch 通过 stream capture 构图：`torch.cuda.graph` 进入时切换到一条 side stream 并开始 capture，此后提交到这条 stream 的 kernel 和内存复制被记录为节点，暂不执行。其他 stream 可以通过等待 capture stream 上的 event 加入 capture，结束前必须重新汇合到 capture stream。Replay 按记录重新执行全部节点，每个节点的参数（指针与标量）都是 capture 时的取值。

| eager 中的内容 | capture 时 | replay 时 |
| :--- | :--- | :--- |
| capture stream 上的 kernel、复制 | 记录为节点 | 重新执行 |
| 经 event 加入 capture 的其他 stream 上的工作 | 记录为节点与依赖边 | 重新执行 |
| 未加入 capture 的 stream 上的 kernel | 照常执行，不进图 | 不执行 |
| Python 分支、CPU 上算出的值 | 按当时取值决定录什么 | 不重新计算 |
| kernel 读写的地址 | 固定为 capture 时的地址 | 读写同一地址 |

表中第三行破坏覆盖，最后两行要求新鲜：随批次变化的数据只能通过固定地址的 buffer 进入图，本文称为持久 buffer，调用方在 replay 前把本批数据写进去。

CUDA 对 capture 期间的非法操作有一条总原则：被捕获的工作与未被捕获的工作之间一旦会产生依赖，CUDA 返回错误并让这次 capture 失效，而不是忽略这条依赖。两类问题都落在这条原则之外，PyTorch 文档也把“capture 期间有未被捕获的 CUDA 工作”和“CPU 工作在 replay 时被省略”列在“可能静默出错”一类。

## 二、捕获遗漏：发到其他 stream 的 kernel

PyTorch 要求 capture 在非默认 stream 上进行，`torch.cuda.graph` 使用的 side stream 来自 PyTorch 的 stream 池，池中的 stream 都以 `cudaStreamNonBlocking` 创建。Non-blocking stream 与默认 stream 之间没有隐式同步，所以 capture 期间向 stream 0（handle 为 0 的默认 stream）发 kernel，不会与 capture stream 产生依赖，CUDA 因此不报错。`capture_error_mode` 限制的是 `cudaMalloc` 这类不经过 stream 的调用，对这次 launch 同样不起作用。这个 kernel 在 capture 时照常执行一次，但不进图。

CuTe DSL 是 CUTLASS 的 Python kernel DSL。用它写的 kernel 经 `cute.compile` 编译后，stream 是 launch 的显式参数；`.launch()` 不带 `stream=` 时，生成的代码把 stream 设为 0。Liger-Kernel 的 RMSNorm backward 由一个 CuTe kernel 和一个 PyTorch 算子组成：kernel 读 dy，写出 dX 和按 SM 分块的部分梯度 `dw_partial`（形状 `[sm_count, N]`）；随后 `dw_partial.sum(dim=0)` 得到 dW。修复前，kernel 的 launch 没有指定 stream：

::capture-miss::

图中三段的含义：

- **修复前，capture**：backward kernel 发到 stream 0，在 capture 当下执行，不进图；`sum` 由 PyTorch 发到当前 stream，即 capture stream，被录进图。
- **修复前，replay**：图中只有 `sum`。它读到的 `dw_partial` 是 buffer 里留下的旧数据，与本次 dy 无关；dX 不会被写入。
- **修复后，replay**：kernel 与 `sum` 都在图中，按本批 dy 计算。

Eager 训练默认在 stream 0 上运行，kernel 发到的 stream 0 与 PyTorch 当前 stream 相同，所以这个问题只在 capture 或使用非默认 stream 时出现。FusedAddRMSNorm 共用同一个 backward 函数，同时受影响。

修复把 stream 作为 kernel 的参数，并在编译时声明它取自 TVM FFI 的 environment stream。CuTe DSL 编译出的函数通过 TVM FFI 调用，environment stream 即调用时 PyTorch 的当前 stream，由 TVM FFI 自动填入，调用端代码不变：

```python
@cute.jit
def __call__(self, mX, mW, mdO, mRstd, mdX, mdW, sm_count, stream: cuda.CUstream, mdS=None):
    self.kernel(...).launch(grid=grid, block=block, cluster=cluster, stream=stream)  # 修复前没有 stream=

compiled = cute.compile(kernel, x_cute, weight_cute, dout_cute, rstd_cute, dx_cute, dw_cute, Int32(0),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),  # 修复新增
                        ds_cute, options="--enable-tvm-ffi")
compiled(x_flat, w_kernel, dy_flat, rstd, dx, dw_partial, sm_count, ds_flat)  # 调用端不传 stream
```

修复前的注释写着 CuTe kernel 在 launch 时读取 PyTorch 当前 stream，并说明沿用了 LayerNorm 的调用方式。LayerNorm 的调用端同样不传 stream，但它在编译时声明了 environment stream 参数；RMSNorm 只沿用了调用端的写法。调用端不传 stream 有两种可能：编译时声明了 environment stream，或者 kernel 根本没有 stream 参数。只看调用端无法区分这两种情况。

## 三、漏捕获的检测

漏捕获的 kernel 在 capture 时仍然执行，第一次 replay 的结果可能恰好正确。可靠的检测要让 replay 的输入与 capture 时不同，再与 eager 比较：

| 方法 | 做法 | 能否发现本例 |
| :--- | :--- | :--- |
| 换输入后与 eager 比较 | 准备两组不同的 dy，分别写入持久 buffer 后 replay，与同一 backend 的 eager 结果逐位比较（`atol=0`） | 能 |
| 检查图中节点 | 在给定 stream 上 capture 一次 kernel 调用，检查图中节点数是否大于 0；CUTLASS 的测试工具用这种方法判断 kernel 是否使用了传入的 stream | 能 |
| `capture_error_mode="global"` | 默认设置，只拦截 `cudaMalloc` 这类调用 | 不能 |

Liger 的回归测试采用第一种：对 RMSNorm 的 in-place 与 out-of-place backward、FusedAddRMSNorm 带与不带 residual 梯度，各测 FP32 与 BF16，并用同一套测试跑 Triton backend 作为对照。Triton kernel 在 PyTorch 当前 stream 上 launch，对照组通过，说明测试代码本身没有问题。比较用同一 backend 的 eager 结果，reduction 顺序相同，所以可以要求逐位一致。PR 中的结果：H100 上修复前 8 个 CuTe 用例全部失败、8 个 Triton 对照全部通过；修复后 16 个 graph 回归用例和 97 个已有 RMSNorm 用例在 H100 与 B200 上全部通过。

> **实测占位**：修复前后各 capture 一次 RMSNorm backward，列出图中的 kernel 节点，并给出 replay 梯度与 eager 梯度的最大绝对误差；附修复前 capture 期间的 Nsight Systems 时间轴，显示 backward kernel 落在 stream 0、`sum` 落在 capture stream。需要 H100 或 B200。

## 四、过期输入：图外的本批数据

图中每个 kernel 的指针固定，随批次变化的数据要在 replay 前写进这些地址。vLLM 在初始化时分配持久 buffer（`input_ids`、`positions`、`inputs_embeds` 等），每一步先在图外把本批数据写进去，再 replay。Target model 的 FULL 路径即如此：replay 前准备好 `inputs_embeds`，replay 时不再传任何输入。

EAGLE 是一种 speculative decoding 方法：一个很小的 draft model 读 target model 的 hidden states 和 token embedding，先预测后续几个 token，再由 target model 一次验证。Draft model 每步先做一次 prefill，处理 target model 本步刚确认的 token。对多模态模型，图像在 prompt 中展开为一串占位 token，embedding 合并这一步把这些位置的 embedding 替换为视觉 encoder 的输出：

```python
self.inputs_embeds[:n] = self.model.embed_input_ids(
    input_ids[:n], multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed)
```

vLLM 的 CUDA Graph 有两种运行模式。FULL 把整段计算录成一张图；PIECEWISE 只录模型中可编译的分段，attention 等不兼容的操作与分段之外的代码以 eager 执行。两种模式下，embedding 合并分别落在图边界的两侧：

::graph-boundary::

FULL 模式录的是整个 draft prefill：embedding 合并、draft forward、logits 与采样。Capture 用 dummy batch 进行，调用时不带 `mm_inputs`，录下的合并步骤只有文本 embedding 查表；replay 的入口 `run_fullgraph(desc)` 也不接收 `mm_inputs`。`input_ids` 在持久 buffer 中，文本 embedding 按本批重算，图像特征从未进入图，占位 token 位置得到的是占位 token 本身的 embedding。PIECEWISE 与 eager 下，合并在图外以 eager 执行，每批都带着本批的 `mm_inputs`，不受影响。

问题报告中的最小复现用一个 fake draft model 依次输入图像 A、B：eager 下图像位置 embedding 的和为 48 与 −112；FULL replay 两次都是 5.04，与图像无关。

触发条件取决于 `cudagraph_mode`。默认的 `FULL_AND_PIECEWISE` 下，draft prefill 只有不含 prefill 的 batch 才走 FULL，这类 batch 通常不含图像 token；设为 `FULL` 时，含图像的混合 batch 也走 FULL 图。

这个问题影响 draft 质量，不影响输出正确性。vLLM 的验证是无损的：greedy 时只有 draft token 等于 target 的 argmax 才接受；采样时按 $p(x) > u \cdot q(x)$ 接受，其中 $q$ 是 draft model 自己的 logits，拒绝后从 $\max(p - q, 0)$ 重新采样。Draft logits 与 draft token 出自同一张图、同一份错误的 embedding，验证用的 $q$ 与实际采样的分布一致，输出分布仍是 target model 的分布。代价是图像相关位置的 draft 更容易被拒绝，acceptance rate 下降。

> **实测占位**：Qwen2.5-VL 或 Qwen3-VL 配 EAGLE3 draft，`cudagraph_mode=FULL`，带图像的请求分别在 FULL（修复前）与 PIECEWISE（修复后）下运行，比较 acceptance rate 与 draft prefill 耗时。需要 A100 80GB 或 H100。

修复有两条路线：为图像 embedding 和 `is_mm_embed` mask 建立持久 buffer，replay 前写入，让输入恢复新鲜；或者让这类 batch 不走 FULL。作者提交的修复采用第二条，改动很小：`CudaGraphManager.dispatch` 增加 `invalid_modes` 参数，只有本批确实带图像输入时才排除 FULL，回退到 PIECEWISE，没有 PIECEWISE 图时回退到 eager。判断依据是本批的 `mm_inputs`，而不是模型是否支持多模态，所以多模态模型上的纯文本 batch 仍可使用 FULL。这个修复尚未合并，vLLM main 中 FULL 路径仍不传 `mm_inputs`。

## 五、数据并行下的降级

数据并行（DP）时，各 rank 在 MoE 层等处要做集合通信，vLLM 让所有 rank 使用同一种运行模式和同一个 padding 后的 token 数。同步分两步：先用 all_reduce 收集各 rank 期望的运行模式，取最小值（NONE < PIECEWISE < FULL），结果为 NONE 时全部 eager；否则按各 rank 中最大的 token 数重新 dispatch，选出具体的图。第二步的重新 dispatch 不参考第一步的结果，已经降级的 rank 可能被选回 FULL。

| | rank 0 | rank 1 |
| :--- | :--- | :--- |
| 本批 | 含图像 | 纯文本 |
| 期望的运行模式 | PIECEWISE | FULL |
| 同步结果（取最小值） | PIECEWISE | PIECEWISE |
| 修复前重新 dispatch | FULL（图像特征丢失） | FULL |
| 修复后重新 dispatch | PIECEWISE | PIECEWISE |

修复在同步结果不是 FULL 时，把 FULL 加入重新 dispatch 的 `invalid_modes`，让降级在所有 rank 上保持。

## 六、各框架的持久 buffer 管理

两个案例都说明同一个约定：每批数据在图外准备，写进固定地址；图内只从这些地址读。各系统在“谁负责写、写哪些区域”上做法不同：

| 系统 | 管理方式 |
| :--- | :--- |
| PyTorch `torch.cuda.graph` | 调用方自己分配持久 buffer（PyTorch 文档称 static input），replay 前 `copy_` 写入，输出在下次 replay 前取走 |
| vLLM | 初始化时分配持久 buffer，各 runner 在 replay 前的准备代码中写入；V1 的 EAGLE proposer 在调用模型前以 eager 合并图像 embedding |
| SGLang | `CudaGraphBufferRegistry` 把每个需要进图的 batch 字段登记为一个 slot，每个 slot 声明 padding 策略 |

SGLang 的 padding 策略处理的是另一种过期数据。Batch 被 pad 到 capture 时的尺寸，padding 区也会被图中的 kernel 处理；如果 padding 区留着上一批的 `positions` 或 KV cache 下标，这些位置会读写真实的 cache 槽位。SGLang 规定这类索引 buffer 的 padding 区每次清零，指向槽位 0。vLLM 的 draft prefill 问题违反的是同一个约定：合并步骤落在图边界内侧，本该每批在图外写入的图像 embedding 被录成了 capture 时的样子。

## 七、小结

Replay 只重复 capture 时记录下的 GPU 工作和地址。发到其他 stream 的 kernel 不在记录中，图外的数据不按批次更新，两者在 capture 时都不报错。检测的有效做法是换一组输入再 replay，与 eager 逐位比较；设计上把随批次变化的数据全部放进持久 buffer，并在 replay 前写入。

## 参考

- luca-888. [fix(cutedsl): use the current stream for RMSNorm backward](https://github.com/linkedin/Liger-Kernel/pull/1450). Liger-Kernel #1450, 2026（merge commit `ad2436e`）.
- luca-888. [Disable EAGLE prefill FULL CUDA graph for live multimodal batches](https://github.com/vllm-project/vllm/pull/43776). vLLM #43776, 2026（未合并）.
- glaziermag. [EAGLE FULL prefill CUDA graph drops mm_inputs](https://github.com/vllm-project/vllm/issues/43716). vLLM #43716, 2026.
- NVIDIA. [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) · [CUDA Runtime API: Stream Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html) · [CuTe DSL: TVM FFI compilation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/pythonDSL/guides/tvm_ffi_compilation.rst).
- PyTorch. [CUDA semantics: CUDA Graphs](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs).
- vLLM. [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/).
- 源码：Liger-Kernel `d5f2817`（`ops/backends/_cutedsl/rms_norm.py`、`test/cutedsl/test_rms_norm_stream.py`）；vLLM `c41b263`（`v1/worker/gpu/spec_decode/target_dependent_ar/`、`v1/worker/gpu/{cudagraph_utils,dp_utils,model_runner}.py`、`spec_decode/rejection_sampler_utils.py`）；SGLang `6fc8d9d`（`model_executor/cuda_graph_buffer_registry.py`）；PyTorch `a6b28b6`（`c10/cuda/CUDAStream.cpp`、`aten/src/ATen/cuda/CUDAGraph.cpp`）；CUTLASS `0b55a2f`（`python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py`、`cutlass/testing.py`）。
