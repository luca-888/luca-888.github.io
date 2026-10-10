Qwen2-VL 与 Qwen3-VL 把一段视频转换为 prompt 中的一串 token：按固定帧率采样，相邻两帧合成一个时间组，每组切成 patch 并合并为 token；Qwen3-VL 还在每组前插入一段文本时间戳。token 的数量只取决于视频的时长、帧率、宽高与 processor 配置，与像素内容无关。按 Qwen3-VL-8B 的官方配置计算，一段 10 s、1080p、30 fps 的视频展开为 10 组，每组 1196 个 token，共 11960 个视频 token。本文说明这些规则，并以作者向 LlamaFactory 提交的修复为例，讨论规则在训练框架中的两个用法：不解码视频而算出 token 数，以及多个视频时每个 `<video>` 对应自己的 grid 与时间戳。

## 一、从帧到 grid

transformers 的 `Qwen3VLVideoProcessor` 分四步处理一段视频：

1. **采样**：按 2 fps 取帧，帧数为总帧数除以原始帧率再乘 2，限制在 4 到 768 之间；采样到的帧的下标（下文称采样下标）在第一帧与最后一帧之间均匀分布，四舍五入取整。
2. **补成偶数**：视觉编码器在时间方向的 patch 深度 `temporal_patch_size` 为 2，相邻两帧合成一个时间组，本文简称“组”。帧数为奇数时复制最后一帧。
3. **缩放**：宽高缩放到 `patch_size × merge_size` 的整数倍，Qwen3-VL 为 16 × 2 = 32，Qwen2-VL 为 14 × 2 = 28；总像素限制在配置的范围内。这一步由 `smart_resize` 完成。
4. **切 patch 与合并**：每组切成 16 × 16 像素、深度为 2 帧的 patch，得到 `video_grid_thw = [t, h, w]`，即组数与每组 patch 的行数、列数。进入 LLM 之前，相邻的 2 × 2 个 patch 由一个 MLP 合并为一个 token（`merge_size` 为 2）。

一段视频的 token 数因此为

$$
N_\text{video} = t \cdot \frac{h \cdot w}{\text{merge\_size}^2}
$$

::video-path::

两代模型的缩放规则不同。Qwen2-VL 的 `smart_resize` 只约束单帧的像素数；Qwen3-VL 的版本约束 $t \cdot H \cdot W$，整段视频共享一个像素预算，Qwen3-VL-8B 的配置中为 25165824。视频越长，每帧分到的分辨率越低：同为 1080p，10 s 的视频每帧缩放到 1472 × 832，15 s 的视频缩放到 1216 × 672。官方参考实现 qwen-vl-utils 另给每帧设置上限，每帧最多 768 个 token。transformers 于 2026 年 8 月加入同样的可选参数 `cap_pixels_per_frame`，默认关闭，计划在 v5.22 改为默认开启；开启后，上例的 10 s 视频每帧为 1152 × 640，共 7200 个视频 token。

训练框架还可能在 processor 之前自行缩小每帧。LlamaFactory 的 `video_max_pixels` 默认为 256 × 256，同一段 10 s 视频的每帧先缩小到 341 × 192，再由 `smart_resize` 取整为 352 × 192，共 10 × 66 = 660 个 token。视频 token 数取决于整条流水线的配置，同一模型在不同框架中的结果可以相差一个数量级。

## 二、时间戳

Qwen2-VL 的视频时间只体现在 M-RoPE 的时间维：每组的 temporal position id 加 1，与帧间隔无关。Qwen2.5-VL 把时间维的 position id 对齐到绝对时间，processor 为此输出 `second_per_grid_ts`，即每组跨越的秒数。Qwen3-VL 技术报告指出，按绝对时间分配的 position id 在长视频中过大且稀疏，因此改为在每组前插入文本时间戳，如 `<3.0 seconds>`。

Qwen3-VL 中一组的时间戳是组内首尾两帧时刻的平均值。上例第 1 组由第 0 帧与第 16 帧组成，两帧分别位于 0 s 与 0.53 s，时间戳为 `<0.3 seconds>`。展开后一段视频的结构为：

```text
<0.3 seconds><|vision_start|><|video_pad|> × 1196<|vision_end|><1.3 seconds><|vision_start|> …
```

transformers 中的计算由 `_calculate_timestamps(frames_indices, fps, temporal_patch_size)` 完成：采样下标除以原始帧率得到秒数，帧数补齐到组长的整数倍，再逐组取平均。时间戳因此需要两项元数据，即采样下标与原始帧率，processor 把它们放在 `video_metadata` 中。Qwen3.5 直接复用 Qwen3-VL 的 processor 与 video processor，视频的展开方式相同。

## 三、不解码的 token 计数

LlamaFactory 的数据处理分两个阶段。预处理阶段套用对话模板、展开 `<video>` 并 tokenize，结果缓存后供各个 epoch 复用；训练阶段由 collator 解码视频，生成 `pixel_values_videos`。展开 `<video>` 需要 grid 与时间戳，原实现为此调用训练阶段的同一个函数，把视频完整解码一遍，得到像素后只使用其形状。

grid 与时间戳只依赖总帧数、时长、宽高与原始帧率，这些信息都在视频容器的 stream 头部，因此这次解码可以省去。作者在 LlamaFactory issue #10398 中报告，约 50 万条视频样本的预处理接近一天，内存峰值达到 1–2 TB。Qwen2-VL 论文同样提到，视频解码是训练的主要瓶颈，团队最终采用了缓存解码。

PR #10404 为 Qwen2-VL 与 Qwen3-VL 增加了只读元数据的路径，下文称为元数据路径，原实现称为解码路径。它用 PyAV 打开容器，读取 stream 的宽高、平均帧率、时长与总帧数，再复现解码路径中影响 token 数的四条规则：

| 规则 | 解码路径 | 元数据路径 |
| --- | --- | --- |
| 采样 | 顺序解码全部帧，保留采样下标上的帧 | 由总帧数、时长与 `video_fps` 算出下标 |
| 奇数补齐 | 复制最后一帧 | 帧数加 1 |
| 预缩放 | 按 `video_max_pixels` 缩小每帧图像 | 只计算缩小后的宽高 |
| `smart_resize` | 由 video processor 执行 | 调用同一个函数 |

Qwen3-VL 的时间戳由采样下标与平均帧率构造 `video_metadata`，交给 processor 的 `_calculate_timestamps` 计算。容器缺少时长或帧率、输入为图片序列而非文件、样本含音频时，元数据路径返回空值，回退到解码路径。PR 中对 100 条单视频样本的测量显示，Qwen2-VL 插件每条样本的预处理时间为 5313.69 ms，改为元数据路径后为 11.76 ms，两条路径展开的结果一致；差距来自解码路径解码了全部帧，元数据路径只读取头部。

元数据路径的代价是这些规则必须与上游保持一致，`cap_pixels_per_frame` 即为一例。若 transformers 把它改为默认开启，LlamaFactory 的解码路径会按封顶后的预算缩放，元数据路径复现的则是不封顶的规则。在默认的 `video_max_pixels` 下，每帧的像素远低于上限，两者结果相同；`video_max_pixels` 设得较大时，两条路径算出的 grid 可能不同。

::measure-fast-path-cap::

## 四、多个视频的对应

prompt 中的 `<video>` 与 processor 的输出是两条并行的序列。processor 按视频顺序输出 `video_grid_thw` 与 `video_metadata`，第 i 项属于第 i 个视频；文本中的占位符可能分布在多条消息中，一条消息也可能包含多个。展开时需要一个跨所有消息累计的计数器，第 i 个 `<video>` 取第 i 项。transformers 的 `ProcessorMixin` 即按此实现：先为每个视频生成替换文本，再按占位符出现的顺序逐个替换。

LlamaFactory 的 Qwen3-VL 插件原来用三种下标取这三项信息：组数取第一个视频的 `video_grid_thw[0][0]`，每组的 token 数取计数器对应的视频，时间戳取 `video_metadata[消息下标]`。只有一个视频且位于第一条消息时，三种下标一致。

::multi-video::

三种下标不一致时，后果取决于视频的位置与长度：

| 情形 | 后果 |
| --- | --- |
| 多轮对话，第二个视频位于第 3 条消息（下标 2） | `video_metadata[2]` 越界，报 IndexError（issue #9704） |
| 一条消息中两个视频，组数不同 | 第二个视频的 token 数与视觉特征数不等，forward 时报错 |
| 一条消息中两个视频，组数相同 | 不报错，第二个视频带着第一个视频的时间戳参与训练 |

第三种情形在 LlamaFactory 的默认配置下容易出现：`video_maxlen` 为 128、`video_fps` 为 2，长于 64 s 的视频都被截为 128 帧、64 组。

PR #10518 把三项信息都改为按计数器读取，并修正了时间戳函数的一个参数。`_calculate_timestamps` 的第三个参数决定几帧共用一个时间戳，应为时间方向的组长 `temporal_patch_size`，原实现传入的是空间合并的 `merge_size`。两者在公开配置中都为 2，这一修改不改变计算结果，作用是与 transformers 的调用保持一致。作者在该 PR 的讨论中指出，GLM-4V 插件也只按第一个视频取组数与时间戳；LlamaFactory 的主分支中，这段代码仍未修改。

`video_metadata` 只用于展开占位符，模型并不使用。LlamaFactory 把 processor 的全部输出交给模型：训练时 forward 接受任意关键字参数，不受影响；`generate()` 会检查模型不使用的参数，按源码推断会因 `video_metadata` 报错。作者提交的 PR #10509 在交给模型之前删除该字段，尚未合并。

::measure-generate::

## 五、推理框架的做法

vLLM 与 SGLang 在处理请求时解码视频。推理请求随即需要像素，不存在单独计算长度的阶段，因此没有可以省去的解码。两者都只读取采样到的帧：vLLM 的 OpenCV 后端顺序 grab 到最大的采样下标，只对采样帧执行 retrieve 与颜色转换；SGLang 用 torchcodec 按下标取帧，可多线程并行解码。多个视频的对应方式与 transformers 相同：每个视频作为独立的 item 携带自己的元数据，vLLM 按 item 下标生成替换序列，SGLang 用计数器按出现顺序读取 grid 与时间戳。

Qwen 官方的微调代码 qwen-vl-finetune 采用另一种组织方式：数据集按需加载，每个样本在被取出时一次完成解码与展开，没有预处理阶段。按长度分组采样时，需要事先在数据中写入每条样本的 token 数。

LlamaFactory 与推理框架的采样规则不完全相同。LlamaFactory 的帧数为时长乘以 `video_fps` 后向下取整，采样下标也向下取整；transformers 与 vLLM 的帧数由总帧数与原始帧率算出，至少为 4 帧，采样下标四舍五入。对于一段 10 s、30 fps、共 300 帧的视频，第 2 个采样帧在 LlamaFactory 中是第 15 帧，在 vLLM 中是第 16 帧，第 1 组的时间戳分别为 `<0.2 seconds>` 与 `<0.3 seconds>`。每组的 token 数也不同：LlamaFactory 默认为 66，vLLM 按官方配置为 1196。分辨率可以通过配置对齐，采样下标的取整方式不能。

::measure-train-infer::

## 参考

- Wang et al. [Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191). 2024.
- Bai et al. [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923). 2025.
- Qwen Team. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631). 2025.
- Qwen Team. [qwen-vl-utils](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-utils) 与 [qwen-vl-finetune](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)（commit `9658872`）。
- Qwen Team. [Qwen3-VL-8B-Instruct 的 video processor 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/video_preprocessor_config.json)。
- Hugging Face. [transformers](https://github.com/huggingface/transformers/tree/536ecc007387a50e77603bb5d92100e9b07514cc)（commit `536ecc0`）：`models/qwen3_vl/`、`models/qwen2_vl/`、`processing_utils.py`；[#48071 Add an opt-in per-frame pixel cap to the Qwen3-VL video processor](https://github.com/huggingface/transformers/pull/48071)。
- hiyouga et al. [LlamaFactory](https://github.com/hiyouga/LlamaFactory/tree/1fded918ac4723ac9966363f51ed4500866c5d1d)（commit `1fded91`）：`src/llamafactory/data/mm_plugin.py`、`collator.py`。
- LlamaFactory issue 与 PR：[#10398 Repeated video decoding during tokenizer/message preprocessing](https://github.com/hiyouga/LlamaFactory/issues/10398) · [#10404 Optimize Qwen video token metadata preprocessing](https://github.com/hiyouga/LlamaFactory/pull/10404) · [#9704 qwen3vl 多模态 video sft: index out of range](https://github.com/hiyouga/LlamaFactory/issues/9704) · [#10518 Fixes Qwen3-VL prompt expansion when a message contains multiple videos](https://github.com/hiyouga/LlamaFactory/pull/10518) · [#10509 drop Qwen3-VL video metadata before model inputs](https://github.com/hiyouga/LlamaFactory/pull/10509)。
- vLLM Team. [vLLM](https://github.com/vllm-project/vllm/tree/c41b2639e29c3bc01add1d34bef3032a6d9d8aca)（commit `c41b263`）：`vllm/model_executor/models/qwen3_vl.py`、`vllm/multimodal/video.py`。
- SGLang Team. [SGLang](https://github.com/sgl-project/sglang/tree/6fc8d9da3288e9710a6f9a1f59503cacf8a984f0)（commit `6fc8d9d`）：`python/sglang/srt/multimodal/processors/qwen_vl.py`。
