# 线性 Attention 视频：逻辑与分镜

16:9 横版，有配音、字幕和音效，全片约 4 分 12 秒。文案与节拍在 `script.js`（配音稿只此一份，每幕有 id），画面在 `core.js` 与 `scenes.js`（按 id 取幕），实测数据在 `data.js` 与 `data-keys.js`（由 `scripts/linear-attention/alpha_probe.py`、`key_cosine.py` 生成，Modal CPU）；`node render.mjs` 逐幕渲染 12 fps 的预览并生成分镜页 `build/storyboard.html`，`--stills` 只出检查用的静帧。状态（2026-10-13 中间态）：按“挑选与叠加”的主线重排为 13 幕，关键帧与 12 fps 无声预览（带字幕）已出，配音未录。待办：第 3 幕（softmax）按用户要求改为不讲公式、以模型结构图为主体；全片约 4 分 12 秒，可考虑压缩。对应文章 `content/linear-attention.md`，出处见 `docs/linear-attention-notes.md`。

## 主线

先立一个直觉：full attention 逐条保存 k、v，读取时用 softmax 挑出最匹配的一项；线性 attention 把每条“k → v”的关联加进同一个矩阵 S，读取 $Sq$ 得到的是所有 value 按 $q^\top k$ 加权的叠加。后面各幕都从这个区别推出来：

- 叠加读出不能靠竞争挑出目标，key 必须相互正交才不串扰；正交方向的个数等于维数，head 维度即容量（第 5 幕）。
- 写入 β = 1 时要恰好覆盖旧值，key 做 L2 归一化（第 5 幕）。
- 容量固定，必须管理写入：整体衰减 α 与按 key 覆盖的 delta rule；delta rule 中 k 用两次（第 6、7 幕）。
- α 由当前 token 算出，遗忘是学出来的；位置信息也由衰减提供（第 8 幕，实测）。

开场那句话“小明把钥匙放在厨房的抽屉里，然后出门上班了。”贯穿全片：第 2 幕用它的 14 个 key 演示两种读法，第 6、9 幕的实测文本以它开头。结尾幕已按用户要求删去，全片在“两条路线”收束。

## 分镜（13 幕）

| 幕 | id | 时长 | 要建立的直觉 | 讲完时的画面 | 数字与出处 |
| --- | --- | --- | --- | --- | --- |
| 1 开场 | open | 11 s | 一个越存越多，一个尺寸不变 | 大标题；读完的一句话；中间的 state；底部 14 格 KV | 示意 |
| 2 两种读法 | read | 22 s | 挑选与叠加 | 同一个 q 读 14 个 token：softmax 权重“抽屉”占 0.88，读出与 v_抽屉 相似度 1.00；点积权重各项都不小（和为 7.2），读出相似度 0.57 | 示意：key 含公共分量，两两点积约 0.5 |
| 3 去掉 softmax | softmax | 23 s | q 能提到求和之外，每步开销与上下文无关 | 两列公式，右列按项移动成 S_t q_t；上下文从 1 增长到 4096，左侧扫描全部 K、V，右侧只读写 S；每步读取的数值个数 1,048,576 对 16,384 | Katharopoulos et al.；一个 head |
| 4 真实尺度 | scale | 11 s | 换成真实模型差多少 | 一层 KV（1 GiB）铺满整页，同一层 state（3 MiB）为右下角小方块，面积按真实比例 | Qwen3.8-27B config；自行计算 |
| 5 叠加的代价 | superpose | 22 s | key 越接近正交串扰越小，维数即容量 | 左：3 维单位球上 k₁、k₂、k₃ 两两正交，加入与各轴成 54.7° 的 k₄ 后，用任一个读出都混入 0.58 v₄；右：128 维写入 16 / 128 / 1024 个 key 后相似度 0.96 / 0.77 / 0.35 | 左为简化；右为随机单位向量的模拟 |
| 6 实际的 key | realkeys | 20 s | 实际的 key 并不正交 | 第 6 层 head 14：第一个“钥匙”与各 token 的 key 余弦，后两次“钥匙”0.93、0.94，其余 0.1–0.6；288 个 head 的平均 \|余弦\| 中位数 0.41，随机向量 0.07 | 实测：Qwen3.5-0.8B，Modal CPU |
| 7 管理容量 | delta | 20 s | 整体衰减与按 key 覆盖 | 四行：写入前、直接相加、先衰减再相加、delta rule；下方 Gated DeltaNet 更新式 | 示意 α = 0.5，β = 1 |
| 8 两种都要 | niah | 13 s | 两种机制互补 | S-NIAH 两组条：Mamba2 30.4、DeltaNet 18.6，GDN 均在 90 以上 | Gated DeltaNet 论文 |
| 9 α 由 token 决定 | alpha | 40 s | 遗忘是学出来的 | 实测热图：三个 GDN head 在每个 token 上的 α；“钥匙”的保留曲线依次出现：第 1 层 head 4 立即降到 0（只看当前 token），第 21 层 head 9 在标点处阶梯下降到 0.31（按句子记忆），第 13 层 head 13 保持 0.99（长期记忆）；随后两列对照位置信息：full attention 用 RoPE，GDN 由衰减与 short convolution 体现；左侧带出 KDA | 实测：Qwen3.5-0.8B，Modal CPU、FP32；功能为直观理解 |
| 10 训练怎么并行 | chunk | 17 s | 递推也能分块并行 | 64 块的序列与放大的两块；串行步数 4096 对 64；vLLM 的两处调用原文 | vLLM `c41b2639` |
| 11 精确检索 | hybrid | 18 s | 叠加做不到精确取回，留几层 full attention | 三个模型的层排布；49.87 对 53.81 | config.json；Qwen3.8-Next 架构报告 |
| 12 显存与推理系统 | memory | 20 s | 省多少，代价在哪 | 16.1 对 64 GiB；一个 GDN 层的交叉点 768；每 512 个 token 一份 state 副本，从 2560 处继续计算 | 自行计算；Kimi K3 技术报告 |
| 13 两条路线 | routes | 15 s | 反对意见与稀疏路线 | MiniMax 三代结构、理由；各家两栏 | MiniMax-01 论文、M2 博客、config |

## 实测

第 6、9 幕用真实数据：Qwen3.5-0.8B（24 层，18 层 GDN，每层 16 个 head）在本机 CPU 上以 FP32 前向一段三句话的中文文本，记录每个 GDN 层、每个 head 在每个 token 上的 α 与 β，以及 L2 归一化后 key 两两的余弦。脚本 `scripts/linear-attention/alpha_probe.py`、`key_cosine.py`，结果 `public/measurements/linear-attention/20261013-184512/`、`20261013-185805-keys/`。原计划在 4090 上比较 vLLM KV cache 容量的占位已取消。

## 简化之处

- 第 1、2 幕的向量与 state 是示意；第 2 幕的 key 人为加入公共分量，使两两点积约 0.5，以显示 softmax 与点积权重的差别。
- 第 5 幕左侧是 3 维的简化；右侧是独立随机单位向量的模拟，真实 key 之间有相关性。
- 第 6 幕的 α = 0.5、β = 1、k 与 k′ 正交是讲机制用的特例。
- 第 9 幕的保留比例只计衰减 α 的连乘；曲线旁的 head 功能是直观理解，不是严格结论，不含 delta rule 的覆盖。
- 第 11 幕 state 按 FP32、KV 按 BF16 计，conv state 不计。
- 第 12 幕 MiniMax-M2、M3 的条只表示结构类型，不按层数画。

## 字幕

配音逐句烧在画面底部（`scenes.js` 的 `subtitle`），按标点切成不超过 26 字的短句；配音合成前按字数估算时长，合成后改用实测时长。

## 待核实

- 配音里的英文词读法（state、query、key、value、softmax、Gated DeltaNet、KDA、chunk、Tensor Core、pass-key）录音后试听。
