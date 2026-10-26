# MoE Kernel 视频：分镜表

16:9 横版，全片约 3 分 18 秒，11 幕。文案与节拍在 `script.js`，实测数据由 `python3 data.py` 从 `public/measurements/moe-kernel/20261008-105322/results.json` 抽成 `data.js`，画面在 `core.js` 与 `scenes.js`，`node render.mjs` 逐幕渲染 12 fps 预览（带字幕与音效）、分镜页 `build/storyboard.html` 与拼接预览 `build/moe-kernel.preview.mp4`。状态：2026-10-20 从头重做，预览已渲染，配音未合成。

## 全片的问题

Qwen3-30B-A3B 的每个 MoE 层有 128 个 expert，一个 token 只用 8 个；一批 token 各选各的，GPU 怎样把这一层算快？

链条：128 选 8（第 1 幕）→ 一个 expert 是 3 次矩阵乘法（第 2 幕）→ router 怎样选、怎样合（第 3 幕）→ 一批 token 分散到 65 个 expert，多数只有一两个 token（第 4 幕）→ 逐个 expert 循环：GPU 空等（第 5 幕）→ 先按 expert 排队、补齐（第 6 幕）→ 一次 kernel 算完所有 expert（第 7 幕）→ 整层 7 个 kernel，快 44 倍（第 8 幕）→ 剩下的时间花在读权重（第 9 幕）→ token 越多，权重用得越满（第 10 幕）→ 各家做法与 B200（第 11 幕，回到开场网格）。

## 贯穿全片的图形

128 个 expert 的网格（8 × 16，编号按行）：开场时 token 点亮它的 8 格，结尾时 16 个 token 的 65 个 expert 依次亮起、一次算完。颜色：token blue，expert 与有用的计算 teal，补齐与空闲 gray，开销 red，权重 yellow。

## 分镜（11 幕）

| 幕 | 时长 | 画面上的问题 | 画面（讲完时） | 数字与出处 |
| --- | --- | --- | --- | --- |
| 1 开场 | 16 s | MoE 层是什么 | 大标题；128 格网格；一个 token 连到 8 格；16 个 token 依次进来，网格累计变色 | 实测路由（请求 0：104、115、68、100、69、119、53、70） |
| 2 一个 expert | 18 s | 一个 expert 算什么 | gate、up（2048 × 768）、down（768 × 2048）按真实比例；一个点流过 SwiGLU；9.44 MB、1.21 GB、一个 token 读 8 个 | config.json |
| 3 router | 16 s | 怎样选、怎样合 | 128 根真实 router 分数；前 8 根变色并标出归一化权重；输出条按权重分成 8 段 | 实测 router 分数 |
| 4 一批 token | 18 s | 一批 token 落在哪 | 16 行 × 128 列的点阵；65 列用到、63 列空闲；每列 token 数的柱；51 个只有 1–2 个 | 实测 |
| 5 逐个 expert | 19 s | 最直接的写法为什么慢 | transformers 循环源码；profiler 时间线三条泳道（矩阵乘法、小 kernel、拷回 CPU）逐段展开；821 个 kernel、326 次拷回、GPU 忙碌 2.2 ms；一层 17.6 ms | 实测 profiler；transformers 源码 |
| 6 按 expert 排队 | 20 s | 怎样让它们一起算 | 128 个选择按 token 顺序 → 按 expert 排序 → 补齐成 1040 行（65 段）；放大前 3 段的 `sorted_token_ids`（补齐位 128）；两张表的真实前缀 | vLLM `moe_align_block_size` 的实际输出 |
| 7 一次 kernel | 20 s | 一次 kernel 里发生什么 | 65 × 24 个 program；放大 expert 115 段的一块：10 行真实、6 行屏蔽；查 `expert_ids` 取权重，按下标 gather 输入行；源码原文；其余段一起亮起 | vLLM `fused_moe_kernel` 源码 |
| 8 整层 | 16 s | 整层花了多少时间 | 7 个 kernel 的真实时间线（两次矩阵乘法 212 µs，占 95%）；同一时间轴：17.6 ms 对 0.40 ms，约 44 倍 | 实测 |
| 9 瓶颈 | 17 s | 剩下的时间花在哪 | 65 块权重从 HBM 流进芯片（613 MB）；带宽柱：实际 2.9 对规格 3.35 TB/s；每个权重约 2 个 token；Tensor Core 只用了 0.6% | 实测 + H100 规格 |
| 10 batch 变大 | 20 s | token 多了会怎样 | 激活 expert 数与单层耗时两条实测曲线（1 到 4096 个 token）；每个 expert 的平均行数 1 → 306，vLLM 的 tile 行数 16 → 128 | 实测；vLLM 默认配置 |
| 11 各家做法 | 18 s | 别家怎么做 | 四张卡：vLLM Triton、transformers、DeepGEMM、MegaBlocks；B200 上默认 FlashInfer TRT-LLM（占位）；回到开场网格，65 格依次亮起 | 源码与论文 |

## 占位

| 幕 | 将放什么 | 需要 |
| --- | --- | --- |
| 11 | B200 上同一层、16 与 4096 个 token：FlashInfer TRT-LLM（vLLM 在 B200 上的默认）与 Triton fused MoE 的 kernel 列表与单层耗时 | Modal B200 × 1，约 $1；可沿用 `scripts/moe-kernel/modal_measure.py` |

其余数字都来自 2026-10-08 的 H100 实测，不需要新测量。

## 简化之处

- 第 6、7 幕只画 GEMM1（w1，N = 1536）；GEMM2 的排布相同。
- 第 8 幕上半部分是 profiler 记录的一次调用（含记录开销，1.19 ms），下半部分的 17.6 ms 与 0.40 ms 是 50 次中位数；两者分开标注。
- 第 10 幕 128 个 token 以内取自独立请求的解码位置，更多 token 取自一条长序列的 prefill，所以 256 个 token 激活的 expert 比 64 个少；画面直接画实测值。
- 第 11 幕的四张卡片只画“补齐还是不补齐”，不展开各自的调度方式。
- 第 2 幕的 SwiGLU 把 gate 与 up 画成两个矩阵；vLLM 与 transformers 实际把它们合并成一个 1536 列的 w1。

## 配音

- 配音稿即 `script.js` 每个节拍的第三项，与底部字幕是同一份文字；配音不念画面上的数字。
- 尚未合成。字幕按估计时长（`core.js` 的 `SPOKEN`）排；运行 `python3 vo.py` 合成后生成 `vo.js`，字幕与音轨换成实测时长，并列出超出节拍的句子。
- 读法写在本片 `vo.py` 的 `EXTRA`：`Qwen3-30B-A3B`、`vLLM`、`SwiGLU`、`FlashInfer`、`TRT-LLM`、`GEMM`；需要试听确认。
