# Qwen-VL 视频 token：调研笔记

## 范围与状态

- 2026-10-10 调研完成并提交参考方案。同日协调 5 篇文章的会话转达：按方案执行、视为确认；两张图跳过草图直接精画；M1 用 PR #10404 的数字，M2–M4 只留占位；理解检查题不等用户回答，参考答案写在本文末尾。正文、图与页面已完成，状态 draft。
- 0.2 s / 0.3 s 的训练与推理差异：再次核对 LlamaFactory `_get_video_sample_indices`（`astype(int32)`）与 vLLM `Qwen3VLVideoBackend.compute_frames_index_to_sample`（`.round()`，vLLM 对 `Qwen3VLVideoProcessor` 自动选用该后端）及两边的时间戳格式 `f"<{t:.1f} seconds>"`，确认成立（`f"{0.25:.1f}"` 为 `0.2`）。
- `cap_pixels_per_frame`：类属性默认 `None`，`_preprocess` 中视为 `False` 并告警，告警称 v5.22 起默认开启。LlamaFactory 解码路径调用 video processor 时不传该参数，沿用实例默认值；元数据路径直接调用 `smart_resize`、不封顶。正文只写“可能不同”，触发条件是每帧像素超过 `min(768·32², longest_edge / 帧数)`，默认 `video_max_pixels` 远低于此。
- 本篇讲 Qwen2-VL / Qwen3-VL（及沿用其 processor 的 Qwen3.5）中一段视频如何变成 prompt 里的 token；LlamaFactory 的三个 PR（#10404、#10518、#10509，作者 luca-888，即本站作者）作为案例，不写成 PR 报告。
- 同期另外四篇（CUDA Graph 正确性、Liger 接入 HF 模型、BF16 下 1+w、semantic cache 校验失败策略）不在本篇范围。
- 本阶段不做实测；文中“按规则计算”的数字是对源码公式的手算（脚本见下文），不是测量。

## 源码版本

| 仓库 | commit | 日期 | 备注 |
| --- | --- | --- | --- |
| LlamaFactory | `1fded91` | 2026-10-10 | 本次新加入 reference-repos 的 `sync.sh` |
| transformers | `536ecc0` | 2026-10-09 | `5.19.0.dev0` |
| vLLM | `c41b2639e2` | 2026-10-10 | |
| SGLang | `6fc8d9da` | 2026-10-10 | |
| QwenLM/Qwen3-VL（含 qwen-vl-utils、qwen-vl-finetune） | `9658872` | 2026-01-30 | 本次新加入 `sync.sh` |

PR 合并 commit：#10404 `8752280d`（2026-05-03 合并）；#10518 `c8a082e0`（2026-07-02 合并）；#10509 未合并（2026-10-10 仍为 open）。

## 参考清单（按重要程度）

| 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| transformers `models/qwen3_vl/{processing,video_processing}_qwen3_vl.py`、`processing_utils.py`（`__call__`、`_process_videos`、`get_text_with_replacements`） | Hugging Face / Qwen | 已读全文与相关段落 | 核心 | 采样、`smart_resize`、`patchify` 补帧、`replace_video_token`、`_calculate_timestamps`、按出现顺序消费 replacement |
| transformers `models/qwen2_vl/{image,video}_processing_qwen2_vl.py`、`processing_qwen2_vl.py` | Hugging Face / Qwen | 已读相关段落 | 核心 | Qwen2-VL 的每帧像素预算、无时间戳的展开 |
| LlamaFactory `data/mm_plugin.py`（`MMPluginMixin`、`Qwen2VLPlugin`、`Qwen3VLPlugin`、`GLM4VPlugin`）、`data/collator.py`、`data/template.py` | hiyouga 等 | 已读相关段落 | 核心 | 现状：fast path、按 placeholder 顺序消费、Qwen3.5 模板用 `qwen3_vl` 插件、`video_metadata` 进入 features |
| [LlamaFactory #10404](https://github.com/hiyouga/LlamaFactory/pull/10404) Optimize Qwen video token metadata preprocessing | luca-888 | 已读描述、diff、review | 核心 | 第 4 节案例；PR 内测量 5313.69 → 11.76 ms/sample |
| [LlamaFactory #10398](https://github.com/hiyouga/LlamaFactory/issues/10398) Repeated video decoding during tokenizer/message preprocessing | luca-888 | 已读全部讨论 | 核心 | 问题来源：50 万视频预处理接近一天、1–2 TB 内存峰值；ffprobe + SQLite 缓存把 180 分钟降到约 3 分钟（作者本地 workaround） |
| [LlamaFactory #10518](https://github.com/hiyouga/LlamaFactory/pull/10518) 多视频 prompt 展开 | luca-888 | 已读描述、diff、讨论 | 核心 | 第 5 节案例；讨论中提到 GLM4V 同类假设 |
| [LlamaFactory #9704](https://github.com/hiyouga/LlamaFactory/issues/9704) qwen3vl 多模态 video sft: index out of range | KYRIE-LI11 | 已读 | 核心 | 多轮视频的报错现象 |
| [LlamaFactory #10509](https://github.com/hiyouga/LlamaFactory/pull/10509) drop Qwen3-VL video metadata before model inputs | luca-888 | 已读描述、diff | 重要 | 第 5 节末：processor 元数据不进模型（未合并） |
| [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)（arXiv 2511.21631） | Qwen Team | 已读 TeX：intro、§Interleaved MRoPE、§Video Timestamp | 核心 | 文本时间戳 `<3.0 seconds>` 的动机：绝对时间位置 id 过大过稀 |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191)（arXiv 2409.12191） | Wang et al. | 已读 TeX：§Model Architecture（Naive Dynamic Resolution、M-RoPE、Unified Image and Video Understanding）、§Infrastructure | 核心 | 2 fps 采样、深度为 2 的 3D 卷积、2×2 合并、每段视频 16384 token 上限、“视频解码是主要瓶颈，改用缓存解码” |
| [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)（arXiv 2502.13923） | Bai et al. | 已读 TeX：intro、§Native Dynamic Resolution and Frame Rate、§MRoPE Aligned to Absolute Time 的开头 | 重要 | 动态 FPS、绝对时间对齐的 MRoPE（`second_per_grid_ts`），作为 Qwen3-VL 时间戳的前身 |
| qwen-vl-utils `vision_process.py`（`smart_nframes`、`fetch_video`） | Qwen Team | 已读相关段落 | 重要 | 官方参考实现：帧数取 2 的倍数、按总像素预算给每帧封顶 |
| [transformers #48071](https://github.com/huggingface/transformers/pull/48071) `cap_pixels_per_frame` | 外部贡献者 | 已读描述 | 重要 | 现状变化：Qwen3-VL video processor 的每帧像素封顶，v5.22 起默认开启；PR 内 vLLM 测量 90 s 1080p 约 184K → 53K token |
| vLLM `model_executor/models/qwen3_vl.py`（`_get_video_second_idx`、`_get_prompt_updates`、`_call_hf_processor`）、`multimodal/video.py`（`Qwen3VLVideoBackend`）、`video_decoders/opencv.py` | vLLM | 已读相关段落 | 重要 | 推理侧：请求处理时按下标读帧；每个视频 item 自带 metadata 与 timestamps；EVS 剪枝时每帧 token 数不同 |
| SGLang `srt/multimodal/processors/qwen_vl.py`（`smart_nframes`、`preprocess_video`、`build_input_ids_with_timestamps`） | SGLang | 已读相关段落 | 重要 | 推理侧：沿用 qwen-vl-utils 规则；`video_idx` 计数器按出现顺序取 grid 与时间戳 |
| qwen-vl-finetune `qwenvl/data/data_processor.py` | Qwen Team | 已读相关段落 | 补充 | 官方微调：lazy dataset，`__getitem__` 中 `apply_chat_template` 一次完成解码与展开，没有单独的 tokenize 阶段 |
| HF Hub `Qwen/Qwen3-VL-8B-Instruct`、`Qwen/Qwen3.5-9B` 的 `video_preprocessor_config.json` | Qwen Team | 已读 | 补充 | 实际配置：`longest_edge` 25165824、`shortest_edge` 4096、patch 16、merge 2、temporal 2；两者相同 |
| TimeMarker（Qwen3-VL 报告引用的 chen2024timemarker） | Chen et al. | 未读 | 补充 | 文本时间戳的来源；若正文提到需先读 |

## 阅读要点

### 规则（transformers 当前 main）

- **采样**：Qwen3-VL `sample_frames`：`num_frames = int(total / fps_src * fps)`，夹在 `[min_frames=4, max_frames=768]` 与 `total` 之间，`np.linspace(0, total-1, n).round()`。Qwen2-VL 版本额外把帧数向下取到 `temporal_patch_size` 的倍数，下标用 `arange(0, total, total/n)`。qwen-vl-utils 的 `smart_nframes` 也取 2 的倍数。
- **补帧**：`patchify` 中 `pad = -num_frames % temporal_patch_size`，复制最后一帧补齐。`grid_t = num_frames / temporal_patch_size`。
- **缩放**：`factor = patch_size * merge_size`（Qwen2-VL 14×2=28，Qwen3-VL 16×2=32）。Qwen2-VL 的 `smart_resize(h, w)` 只约束单帧像素；Qwen3-VL 的 `smart_resize(num_frames, h, w, temporal_factor)` 约束 `t_bar * h_bar * w_bar`，即整段视频的像素预算。#48071 之后可选 `cap_pixels_per_frame`：每帧上限 `min(768 * factor², longest_edge / num_frames)`，下限 `1.05 * shortest_edge`，v5.22 起默认开启。
- **grid 与 token 数**：`video_grid_thw = [t, H/patch, W/patch]`；送入 LLM 的 token 数 = `t * h * w / merge_size²`。
- **时间戳（Qwen3-VL）**：`replace_video_token` 对每个时间组生成 `<{t:.1f} seconds><|vision_start|>` + `frame_seqlen` 个 `<|video_pad|>` + `<|vision_end|>`；`_calculate_timestamps(indices, fps, temporal_patch_size)` 把下标补到组长的倍数，再取每组首尾帧时间的平均。源码中该参数名为 `merge_size`，调用处传的是 `temporal_patch_size`。
- **对应关系**：`ProcessorMixin._process_videos` 对 batch 中第 i 个视频调用 `replace_video_token(video_idx=i)`；`get_text_with_replacements` 按 placeholder 出现顺序消费 replacement 列表。
- **元数据**：`__call__` 末尾在 `return_metadata` 为假时 pop `video_metadata`；Qwen3-VL 默认 `videos_kwargs.return_metadata=True`。
- **Qwen3.5**：`processing_auto` / `video_processing_auto` 把 `qwen3_5`、`qwen3_5_moe`（以及 `qwen4_exp`）映射到 `Qwen3VLProcessor`、`Qwen3VLVideoProcessor`；`Qwen3_5Model.get_rope_index` 注释说明“用时间戳分隔视频”，按 `grid_t` 拆分 `video_grid_thw`。

### LlamaFactory（当前 main）

- 预处理（`dataset.map`）阶段调用 `process_messages()` 展开 token；训练时 collator 再调用 `get_mm_inputs()` 解码出 `pixel_values_videos`。#10404 之前，两处都会解码，前者只为算出 token 数。
- 旧解码路径 `_regularize_videos`：`container.decode` 顺序解码所有帧，只保留采样下标；采样规则 `floor(duration * video_fps)`，夹在 `[1, min(total, video_maxlen)]`，`linspace(...).astype(int32)`（向下取整，与 HF 的 `round` 不同）；奇数帧复制末帧；再按 `video_max_pixels`（默认 256×256）缩小每帧。
- fast path（#10404）：`_get_qwen_video_stream_metadata` 只打开容器读 stream 的 `width / height / average_rate / duration / frames`；复刻四步：采样下标、奇数补齐、`_preprocess_image` 的缩放、模型对应的 `smart_resize`。`duration` 或 `average_rate` 缺失、非路径输入（如帧列表）、有音频时返回 `None`，回退旧路径。图片仍走原路径。
- Qwen3-VL 的时间戳：LlamaFactory 以 `fps=video_fps`、`frames_indices = idx / original_fps * video_fps` 喂给 processor（源码注释 “hack usage when do_sample_frames=False”），使 `idx' / fps` 等于真实秒数。
- #10518 之后：`video_grid = video_grid_thw[num_video_tokens]`、`metadata = video_metadata[num_video_tokens]`，`num_video_tokens` 在全部消息上累计。
- `GLM4VPlugin.process_messages` 仍是 `num_frames = video_grid_thw[0][0]  # hard code for now`，时间戳取第一个视频（#10518 讨论中作者已指出，当前 main 未改）。
- `Qwen3VLPlugin` 未覆盖 `get_mm_inputs`，`_get_mm_inputs` 的结果（含 `video_metadata`）经 collator `features.update(mm_inputs)` 进入模型输入。训练时 forward 接受 `**kwargs` 不报错；`generate()` 的 `_validate_model_kwargs` 只承认 `prepare_inputs_for_generation` 与 forward 的具名参数，会报 unused `video_metadata`（源码推断，未复现）。#10509 修复此问题，未合并。

### 多视频 bug 的三种后果（据旧代码推理）

旧代码：`metadata = video_metadata[idx]`（`idx` 为消息下标），`num_frames = video_grid_thw[0][0]`（第一个视频）。

| 情形 | 后果 |
| --- | --- |
| 多轮对话，第 3 条消息（下标 2）中有视频，共 2 个视频 | `video_metadata[2]` 越界，即 #9704 的 IndexError |
| 同一消息两个视频，帧数不同 | 第二个视频按第一个的组数展开，token 数与视觉特征数不等，forward 时 `Video features and video tokens do not match` |
| 同一消息两个视频，帧数相同（如都长于 `video_maxlen / video_fps` = 64 s，被截到 128 帧） | 不报错，第二个视频的时间戳用的是第一个视频的，静默错误 |

### 推理框架

- vLLM：`Qwen3VLVideoBackend.compute_frames_index_to_sample` 复刻 HF 采样；OpenCV 后端顺序 `grab` 到最大下标、只 `retrieve` 采样帧；元数据（`fps`、`frames_indices`、`do_sample_frames`）随每个视频 item 传递，`_get_video_second_idx` 在调用 HF processor 前算好时间戳；`_get_prompt_updates` 按 `item_idx` 生成每个视频的替换序列；支持 EVS / VidCom2 剪枝，此时每帧 token 数不同。推理请求马上需要像素，解码不可省；不解码的 token 计数只用于 profiling（dummy 输入）。
- SGLang：`preprocess_video` 用 qwen-vl-utils 的 `smart_nframes` 与总像素预算，按下标取帧；`build_input_ids_with_timestamps` 以 `video_idx` 计数器按出现顺序取 `video_grid_thw` 与时间戳。
- Qwen 官方 qwen-vl-finetune：lazy dataset，每个样本在 `__getitem__` 中一次完成解码与展开，没有“只为算长度而解码”的阶段；按长度分桶需要预先写入 `num_tokens`。

### 按规则计算的例子（非实测）

脚本思路：直接实现上文公式（Qwen3-VL `smart_resize`、两种采样、`_calculate_timestamps`）。

| 视频 | 配置 | 采样帧 | 每帧尺寸 | grid | 每组 token | 视频 token |
| --- | --- | --- | --- | --- | --- | --- |
| 1920×1080，30 fps，10 s | HF / vLLM 默认（Qwen3-VL-8B 配置） | 20 | 1472×832 | 10×52×92 | 1196 | 11960 |
| 同上 | 同上，`cap_pixels_per_frame=True` | 20 | 1152×640 | 10×40×72 | 720 | 7200 |
| 同上 | LlamaFactory 默认（`video_max_pixels=65536`） | 20 | 352×192 | 10×12×22 | 66 | 660 |
| 1280×720，30 fps，9.5 s | LlamaFactory 默认 | 19 → 补到 20 | 352×192 | 10×12×22 | 66 | 660 |

时间戳（10 s 视频）：HF 采样下标 `0, 16, 31, …` 得 `<0.3 seconds>, <1.3>, <2.4>, …, <9.7>`；LlamaFactory 采样下标 `0, 15, 31, …`（向下取整）得 `<0.2 seconds>, <1.3>, <2.3>, …, <9.7>`。同一视频在 LlamaFactory 训练与 vLLM 推理中的时间戳文本可能相差 0.1 s，属于本文分析，待验证。

## 分歧与开放问题

- `_calculate_timestamps` 的第三个参数：#10518 改为传 `temporal_patch_size`，与 transformers 当前一致；在公开配置下 `merge_size` 与 `temporal_patch_size` 都等于 2，旧写法结果相同，修改的意义在语义与一致性。
- fast path 与上游规则的同步：`cap_pixels_per_frame` 默认开启后，若用户把 `video_max_pixels` 设得很大，fast path（未考虑封顶）与解码路径算出的 grid 可能不同；LlamaFactory 默认 256×256 下不受影响。
- 训练与推理的采样差异（floor vs round、LlamaFactory 的 `video_max_pixels` 预缩放）是否影响效果，没有公开讨论，正文只陈述差异。

## 待核实

1. `cap_pixels_per_frame=True` 时 fast path 与解码路径的 `video_grid_thw` 是否一致（CPU 可测）。
2. 当前 main 上 Qwen3-VL 视频样本走 `do_predict` / `generate` 是否确实报 unused `video_metadata`（需要能加载 Qwen3-VL 的 GPU，或用 tiny 随机权重模型在 CPU 上复现）。
3. LlamaFactory 与 vLLM 对同一视频的帧下标、时间戳、token 数对照（CPU 可测：只需两边的预处理代码）。
4. `stream.frames` 为 0 或不准确（部分容器）时 fast path 与旧路径是否同样处理。
5. TimeMarker 原文是否需要引用（未读）。

## 实测占位

| 编号 | 内容 | 硬件 | 正文位置 |
| --- | --- | --- | --- |
| M1 | 旧路径与 fast path 的 `process_messages` 耗时（可直接用 #10404 的数字，标明出自 PR；如需本站自测，用公开视频数据集若干条） | CPU（Mac 即可） | 第 4 节 |
| M2 | `cap_pixels_per_frame` 开 / 关 与 fast path 的 grid 一致性 | CPU | 第 4 节末 |
| M3 | LlamaFactory 与 vLLM 预处理同一视频的帧下标、时间戳、token 数 | CPU | 第 6 节 |
| M4（可选） | `generate()` 因 `video_metadata` 报错的复现 | 1 张 24 GB 以上 GPU（如 RTX 4090）跑 Qwen3-VL-2B/8B，或 CPU + tiny 模型 | 第 5 节末 |

## 理解检查（参考答案）

1. **9.5 s、30 fps 的视频在 LlamaFactory 中采样 19 帧，为什么 `video_grid_thw` 的 t 是 10？补上的帧对最后一个时间戳有什么影响？**
   帧数为 ⌊9.5 × 2⌋ = 19，奇数，复制最后一帧补到 20 帧，`temporal_patch_size` 为 2，t = 20 / 2 = 10。`_calculate_timestamps` 同样把下标补成偶数（复制最后一个下标），最后一组的首尾两帧是同一帧，时间戳就是该帧的时刻（9.5 s），而不是两帧的平均。
2. **token 数只需要元数据就能算出，为什么训练时仍要解码？容器缺少 `duration` 时 #10404 的路径怎么处理？**
   token 数只决定 prompt 中占位 token 的个数，模型的视觉编码器需要像素来计算这些位置上的 embedding，所以 collator 仍要解码出 `pixel_values_videos`；省掉的只是预处理阶段为了数个数而做的那次解码。`_get_qwen_video_stream_metadata` 在 `duration` 或 `average_rate` 为空时返回 `None`，整个样本回退到解码路径。
3. **修复前，同一条消息里两个都长于 64 s 的视频（`video_maxlen=128`、`video_fps=2`）会不会报错？哪里不对？**
   不报错。两段都被截为 128 帧、64 组，第二个视频按第一个视频的组数展开，组数恰好相同，每组 token 数又按计数器取了自己的 grid，token 总数与视觉特征一致。错误在时间戳：第二个视频用 `video_metadata[0]`，带着第一个视频的时间戳训练。
4. **`_calculate_timestamps` 的第三个参数为什么应传 `temporal_patch_size`？两者都等于 2 时，修改改变了什么？**
   这个参数决定几帧共用一个时间戳，对应时间方向的组长；`merge_size` 是空间上 2×2 合并的边长，两者语义无关。公开配置下两者都为 2，计算结果不变；修改使插件与 transformers 的 `replace_video_token` 调用一致，将来某个模型的两者取值不同时不会出错。
5. **vLLM 推理同一段视频时，时间戳可能与 LlamaFactory 训练时差 0.1 s，原因在哪？**
   采样下标的取整方式不同：LlamaFactory 用 `np.linspace(...).astype(int32)` 向下取整，transformers 与 vLLM 用 `.round()`。10 s、30 fps 的视频第 2 个采样帧分别是第 15 帧与第 16 帧，第 1 组时间戳为 (0 + 15/30)/2 = 0.25 → `0.2` 与 (0 + 16/30)/2 ≈ 0.27 → `0.3`。帧数公式也略有不同（时长 × fps 与总帧数 / 原始帧率 × fps，后者至少 4 帧）。

## 交付（2026-10-10）

新增：

- `content/qwen-vl-video-tokens.md`（正文，已做成稿自查与 de-ai-edit 改稿）
- `docs/qwen-vl-video-tokens-notes.md`（本文件）
- `posts/qwen-vl-video-tokens/index.html`
- `src/articles/qwen-vl-video-tokens/qwen-vl-video-tokens.tsx`
- `src/articles/qwen-vl-video-tokens/qwen-vl-video-tokens.blocks.tsx`
- `src/articles/qwen-vl-video-tokens/QwenVlVideoTokensFigures.tsx`（图 1 `VideoPath`、图 2 `MultiVideo`，三个实测占位 M2–M4）

修改的共享文件（均为定点插入）：

- `vite.config.ts`：在 `rollupOptions.input` 中 `main` 之后加入一行 `qwenVlVideoTokens: 'posts/qwen-vl-video-tokens/index.html'`。
- `src/posts.ts`：在 `posts` 数组开头加入 `qwen-vl-video-tokens` 一项，`status: 'draft'`，category 为“多模态”。
- `~/Documents/ChatGPT/reference-repos/sync.sh`（不在本仓库）：REPOS 中 `triton` 之后加入 `LlamaFactory` 与 `Qwen3-VL` 两行。

未修改 `docs/terminology.md`。没有实测数据、图片资源与脚本文件；没有 commit。
