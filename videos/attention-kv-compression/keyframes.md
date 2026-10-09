# KV Cache 压缩视频：概念与分镜

源文件 `film.html`（浏览器里加 `?play` 播放，`?t=秒` 跳到某一刻），绘图分在 `lib.js`（配色、图元，沿用 Fused Linear Cross Entropy 视频）与 `scenes.js`（15 幕，时间由配音节拍决定）。配音稿在 `script.md`：`python3 subs.py` 生成 `beats.js`，`python3 vo.py --speed 1.05` 合成配音并生成 `vo.js`。渲染：在本目录运行 `node render.mjs build/attention-kv-compression.mp4`，可加 `--draft`、`--only a-b`；`--stills 12 40` 输出单帧，`--keyframes` 输出每幕结束时的画面。成片 1920 × 1080、60 fps、约 8 分 47 秒，MiniMax 配音，底部一行字幕。出处见 `docs/attention-kv-compression-notes.md`。

## 概念

**讲法。** 全片只问一个问题：decode 每一步都要读的那份缓存，能不能更小？先看缓存从哪来（一个 head 的 query 读 4 个历史 token 的 key 与 value），再算它有多大（320 GiB），再看为什么小了就快（每读一个数只做一次乘加）。然后依次给出三种缩小的办法：全部 head 共享一组（MQA）、分组共享（GQA）、不共享而改存中间结果（MLA）。MLA 留下两个问题，各用一幕回答：每步都要还原 key（矩阵吸收）、位置编码挡住吸收（decoupled RoPE）。后半段讲代价（计算、TP 下的重复）、DeepSeek-V4 的做法和各模型的现状，结尾回到开头的三根柱子。名字都在问题出现之后才给。

**全片一个配置。** 80 层、64 个 query head、head dim 128、BF16，一条 128K token 的请求（与文章开头的表相同）。MHA 每层 16384 个数、GQA-8 为 2048、MLA 为 576、MQA 为 256。

**贯穿全片的图形。**

- **缓存瓦片**：一个 token 在一层里的缓存，左半 key、右半 value，每个 KV head 一行，面积与元素数成正比。第 2 幕的 64 行、第 4 幕的 1 行、第 5 幕的 8 行、第 9 幕四种形式并排，都是同一种画法。
- **128K 请求的显存柱**：对着同一条 80 GB 线（一张 H100）。封面、第 2 幕、结尾是同一组柱子。
- **按维度画宽度的条**：第 7、8、12 幕里 128 维与 512 维的块按同一比例画，展开与吸收的差别直接看得出来。

**一次只跟一个例子。** 第 1 幕跟一个 head 的 query 走完打分、softmax、加权求和，第 2 幕在 64 行里把这个 head 标出来；第 7 幕跟一个 query head 与一个历史 token。

**颜色语义**（沿用系列配色）：key blue，value yellow，latent purple，RoPE key teal，query 与输出 ink，装不下与重复 red。

**比喻。** 第 7 幕（配音里说）：查外文资料时，与其把每一页都翻译过来，不如把问题翻译一次。不同之处：矩阵吸收前后的分数严格相等，没有翻译的损失。

## 分镜（15 幕）

| 幕 | 当前的问题 | 回答 | 数据 / 来源 |
| --- | --- | --- | --- |
| 0 | 封面 | 一条 128K token 的请求：MHA 320 GiB 冲出 80 GB 线，GQA-8 40，MLA 11.3 | 公式 |
| 1 | KV cache 是什么 | 一个 head：query 与 4 个历史 token 的 key 点积，softmax，对 value 加权求和；key、value 只由历史 token 决定，算一次存下来 | 权重为示意 |
| 2 | 它有多大 | 64 个 head 各存一份：16384 个数 → 32 KiB → × 80 层 → × 128K token = 320 GiB，4 张 H100 | 公式 |
| 3 | 为什么小了就快 | 每读一个数只做一次乘加：1 FLOP/字节，H100 能做 295 | Zadouri et al. |
| 4 | 能不能只存一组 | MQA：64 个 query head 读同一组 key、value，256 个数；decoder 每 token 46 µs → 3.8 µs | Shazeer |
| 5 | 一组太少呢 | GQA：8 个 KV head，2048 个数；DeepSeek 7B 的 MMLU 为 45.2 / 41.2 / 37.9 | Ainslie et al.；DeepSeek-V2 附录 |
| 6 | 不共享，能不能也存得少 | MLA：下投影到 512 维的 latent，只缓存 latent，key、value 用到时由上投影还原 | DeepSeek-V2 |
| 7 | 每步都要还原 key 吗 | 矩阵吸收：上投影乘到 query 一侧，每步只做一次，分数相同 | DeepSeek-V2 |
| 8 | 位置编码怎么办 | RoPE 的旋转夹在 query 与上投影之间；另存 64 维带位置、所有 head 共享的 key，缓存 512 + 64 = 576 | DeepSeek-V2 |
| 9 | 吸收之后是什么 | 64 个 head 读同一份 576 维，即单个 KV head 的 MQA；四种形式的瓦片并排；约 250B 模型上 MMLU 59.0 对 57.5 | DeepSeek-V2 |
| 10 | 代价之一 | 每读 1 字节的运算：1 / 8 / 64 / 128，对着 H100 的 295；head 从 64 加到 128，decode FLOPs +83%，loss 只降 0.5%–1.2% | Zadouri et al.；Kimi K2 |
| 11 | 代价之二 | TP=8：GQA-8 每张卡一个 KV head，MLA 每张卡一份完整 latent；按请求分（DP attention）后 decode 吞吐为 1.9 倍 | Zadouri et al.；SGLang v0.4 |
| 12 | DeepSeek 后来怎么做 | V4 去掉上投影，直接训练一个 512 维、key 与 value 共用的 KV head，再沿序列方向压缩；1M 上下文下约为 GQA-8 的 2% | DeepSeek-V4 |
| 13 | 现在的模型用什么 | 每 token 的 KV cache：Llama-3.1-70B 320 KiB … DeepSeek-V4-Pro 7.7 KiB | 各模型 config.json |
| 14 | 回到开头 | 同一条请求：320 → 40 → 11.3 GiB，三句小结 | — |

## 文章有、视频不讲的内容

- value 一侧的吸收（先对 latent 加权求和，再乘一次上投影）：只在第 7 幕配音里带一句。
- prefill 用展开、decode 用吸收的两套算法（vLLM）。
- GQA 由 MHA checkpoint 转换（取平均 + 5% 算力继续训练）与 T5-XXL 的数字。
- GLA（把 latent 拆成两个 head）。
- V4 对输出再施加 −t 旋转的细节。

## 已知的限制

- 第 1 幕的权重 0.1 / 0.6 / 0.2 / 0.1 是示意。
- 第 2 幕与封面把 320 GiB 对着“80 GB”的线画，没有区分 GiB 与 GB。
- 配音里 GiB、KiB 读作 GB、KB，字幕仍写 GiB、KiB。
- 第 10 幕的 8、64、128 由 Zadouri et al. 的近似式（GQA 约为 h/h_kv，MQA 约为 h，MLA 约为 2h）代入 h = 64 得到；文章里写的 256 对应 h = 128。
- 第 11 幕每张卡里色块的高度是“占该形式全部缓存的几分之几”，三行之间不比较绝对大小。
- 第 12 幕的 16 → 4 只画了压缩比为 4 的层。
- 第 13 幕只计 full attention 层，不计 GLM-5 的 DSA indexer 与 Kimi-K3 的线性层状态。
