// 线性 Attention 影片的文案与节拍，画面（scenes.js）与渲染（render.mjs）共用。纯数据，不依赖页面。
// 每幕：id 供 scenes.js 引用、tag 标签、head 标题（{t:…} teal、{p:…} purple、{r:…} red、{b:…} blue、{y:…} 深黄 为强调色）、src 出处、dur 时长（秒）、beats 节拍。
// 节拍：[幕内起点（秒）, 画面动作, 配音]。配音只讲“为什么”和“看哪里”，不念画面上的字和数字；出处见 docs/linear-attention-notes.md。
// hold：关键帧距幕尾的秒数（开场取在标题缩上去之前）。
const SCRIPT = [
  { id: 'open', tag: '开场', head: '', src: '', dur: 11, hold: 1.6, beats: [
    [0.6, '大标题；一句话逐个 token 读入，每读入一个，中间 state 的取值更新一次，尺寸不变', '模型逐个读入 token。'],
    [3.0, '底部紫色的 KV 每个 token 一格，随之变长', 'full attention 为每个 token 保存一份 KV。'],
    [6.0, '读完一句：KV 14 份，state 尺寸不变；大标题缩到左上角', '线性 attention 只保存中间这个矩阵：每读入一个 token，矩阵的取值更新一次，尺寸不变。'],
  ] },
  { id: 'read', tag: '两种读法', head: 'full attention {p:挑出一项}，线性 attention 读出{t:叠加}', src: '示意：q、k、v 为构造的向量，14 个 key 两两点积约 0.5', dur: 22, beats: [
    [0.5, '开场那句话的 14 个 token；当前 token 的 q 与“抽屉”的 key 最匹配（示意）', '设当前 token 的 query 与“抽屉”的 key 最匹配，分别用两种方式读取这句话。'],
    [4.0, '第一行：full attention 的 softmax 权重，“抽屉”一项占绝大部分；读出与 v_抽屉 的相似度很高', 'full attention 对每个 key 打分，softmax 让分数最高的一项占主导，读出的基本就是它的 value。'],
    [11.0, '第二行：线性 attention 直接以点积为权重，其余各项也不小；读出是 14 个 value 的叠加，相似度低', '线性 attention 没有 softmax，权重就是点积。分数较低的 key 也按比例计入，读出的是所有 value 的叠加。'],
  ] },
  { id: 'softmax', tag: '去掉 softmax', head: 'q 能提到求和之外，每一步的开销就{t:与上下文长度无关}', src: 'Katharopoulos et al. 2020', dur: 23, beats: [
    [0.5, '两列：左侧 full attention 的输出式；右侧去掉 softmax，权重为点积', '两种输出式并排：左侧带 softmax，右侧权重为点积。'],
    [3.5, '左侧 exp 被红框标出；右侧公式按项移动成 S_t q_t', '左侧的 query 在 exp 之内，无法提出；右侧的点积是标量，query 可以提到求和之外。'],
    [9.0, '下方给出 S_t 的定义', '括号内的求和记作 S，它与 query 无关，可以在 token 到达时逐项累加。'],
    [14.0, '上下文从 1 增长到 4096 个 token：左侧每一步扫描全部 K、V，右侧只读写 S；下方两条读取量在同一刻度上比较', '因此生成一个 token 时，full attention 的读取量随上下文线性增长，线性 attention 的读取量不变。'],
  ] },
  { id: 'scale', tag: '真实尺度', head: '一层、256K 上下文：每一步 {p:1 GiB} 对 {t:3 MiB}', src: 'Qwen3.8-27B：GQA 4 个 KV head、head dim 256；GDN 48 个 head、head dim 128；state 按 FP32', dur: 11, beats: [
    [0.5, '紫色逐行铺满整页：一层 full attention 在 256K 上下文下的 KV，1 GiB', '以 Qwen3.8-27B 的一层、256K 上下文为例。'],
    [5.0, '右下角出现青色小方块：同一层 state 的 3 MiB，面积按真实比例', '线性 attention 每一步只读写右下角的 state。'],
  ] },
  { id: 'superpose', tag: '叠加的代价', head: 'key 越接近正交，{t:串扰越小}', src: '左：3 维的简化；右：随机单位 key 的模拟，d = 128', dur: 22, beats: [
    [0.5, '左：单位球上三个相互正交的 key，各关联一个 value；用各自的 key 读出，恰为自己的 value', '理想情况下，key 相互正交：三个正交的 key，各自读出自己的 value。'],
    [7.0, '加入第四个 key k₄，与三个坐标轴均成 54.7°；用 k₁、k₂、k₃ 读出都混入 0.58 v₄', '3 维空间只有三个正交方向。第四个 key 必然与已有的 key 重叠，读出时混入它的 value，即串扰。'],
    [14.0, '右：128 维，写入 16、128、1024 个 key 后，读出值与原值的相似度', '128 维也一样：写入的 key 远多于 128 个后，串扰迅速增大。head 维度决定了 state 的容量。'],
  ] },
  { id: 'realkeys', tag: '实际的 key', head: '实际的 key {r:并不正交}', src: '实测：Qwen3.5-0.8B，32 个 token，key 经 L2 归一化', dur: 20, beats: [
    [0.5, '第 6 层的一个 head：第一个“钥匙”的 key 与每个 token 的 key 的余弦相似度', '实际模型中，key 并不正交。这是第 6 层的一个 head，第一个“钥匙”与每个 token 的 key 的余弦相似度。'],
    [7.5, '后两次出现的“钥匙”为 0.93 与 0.94；其余 token 在 0.1 到 0.6 之间；虚线是随机向量的 0.07', '在这个 head 中，同一个词再次出现时，key 几乎重合；无关的 token 之间也普遍为正，因为 key 共享一个公共方向。'],
    [14.0, '下方：全部 288 个 head 的平均 |余弦|，中位数 0.41，远高于随机的 0.07', '因此需要管理写入：重复的 key 由 delta rule 覆盖，过时的内容由衰减清除。'],
  ] },
  { id: 'delta', tag: '管理容量', head: '两种办法：{a:整体衰减}与{t:按 key 覆盖}', src: 'Yang et al. 2024；Gated DeltaNet，ICLR 2025；示意 α = 0.5，β = 1', dur: 20, beats: [
    [0.5, '写入前：k 对应 v₁，k′ 对应 u；新 token 要把 k 对应的 value 改为 v₂', '容量有限，就要决定保留什么。设 key k 已关联 v₁，新 token 要把它改为 v₂。'],
    [5.5, '直接相加：k 读出 v₁ + v₂；先衰减再相加：v₁ 与 u 一起减半', '直接相加，新旧 value 叠加。先整体乘衰减系数 α 再相加，旧值变小，无关的关联同样变小。'],
    [12.0, 'delta rule 一行被框出：k 读出 v₂，u 不变', 'delta rule 先用 k 读出旧值，只写入新旧之差：旧值被替换，其他关联不变。'],
    [16.5, '下方给出 Gated DeltaNet 的更新式：先衰减，再写入差值', '两种办法结合，即 Gated DeltaNet。'],
  ] },
  { id: 'niah', tag: '两种都要', head: '只有一种机制，各在一类检索上{r:失败}', src: 'Gated DeltaNet 论文：1.3B 模型，RULER S-NIAH', dur: 13, beats: [
    [0.5, '第一组：找 pass-key，只有衰减的 Mamba2 为 30.4', '只有衰减时，pass-key 在被读取之前已经衰减。'],
    [5.0, '第二组：在文章中找数字，只有 delta rule 的 DeltaNet 为 18.6', '只有 delta rule 时，无关内容持续写入，没有机制清除。'],
    [9.5, '两组中 Gated DeltaNet 均在 90 以上', '两种机制互为补充。'],
  ] },
  { id: 'alpha', tag: 'α 由 token 决定', head: '遗忘是{t:学出来的}：各 head 分工不同', src: '实测：Qwen3.5-0.8B，Modal CPU、FP32 前向', dur: 40, beats: [
    [0.5, '三句话的 32 个 token；下方三行：三个 GDN head 在每个 token 上的 α', 'α 由当前 token 算出。这是 Qwen3.5-0.8B 读这三句话时，三个 head 的 α。'],
    [6.5, '第一条曲线：第 1 层 head 4 中“钥匙”的保留比例，下一个 token 就降到 0；旁注“只看当前 token”', '第一个 head 每一步几乎清空 state，“钥匙”在下一个 token 就消失了。它只处理眼前的内容，作用接近局部特征。'],
    [14.0, '第二条曲线：第 21 层 head 9，句内保持，在逗号和句号处阶梯下降，到段末 0.31；标点竖带亮起；旁注“按句子记忆”', '第二个 head 在句子内部几乎不衰减，遇到逗号和句号才清掉一部分，像是在记当前这句话。'],
    [21.5, '第三条曲线：第 13 层 head 13，始终约 1.00；旁注“长期记忆”', '第三个 head 几乎不衰减，“钥匙”一直保留到段末，适合保存全文层面的信息，例如人物和主题。'],
    [27.5, '热图区域换成两列对照：full attention 用 RoPE 把位置编进 q、k；GDN 不用 RoPE，位置体现在衰减与 short convolution，箭头指向下方的保留曲线', '衰减还提供了位置信息：越早写入的 token，在 state 里剩得越少。full attention 则用 RoPE 把位置编进 query 和 key。'],
    [35.0, '左侧出现一行：KDA 为每个 key channel 设置一个 α', '同一个 token 在不同 head 中保留的时间完全不同。KDA 更进一步，为每个 channel 设置独立的 α。'],
  ] },
  { id: 'chunk', tag: '训练怎么并行', head: '4K 个 token 分成 64 块，state {t:只串行传递 64 次}', src: 'vLLM qwen_gdn_linear_attn.py（c41b2639）', dur: 17, beats: [
    [0.5, '一条 4096 个 token 的序列；逐 token 递推需要串行 4096 步', '训练与 prefill 一次处理数千个 token。逐 token 递推是串行计算，也无法使用 Tensor Core。'],
    [7.0, '分成 64 块；放大其中两块：块内为 64 × 64 的下三角矩阵乘，块之间传递一次 state', '分块计算：块内用矩阵乘并行完成，块之间传递 state。'],
    [12.0, '下方为 vLLM 中的两处调用：prefill 用 chunk kernel，decode 用 recurrent kernel', 'vLLM 中，prefill 调用分块的 kernel，decode 调用逐步递推的 kernel。'],
  ] },
  { id: 'hybrid', tag: '精确检索', head: '每 4 层保留 1 层 {p:full attention}', src: 'config.json；Qwen3.8-Next 架构报告：25B-A3B，九项评测平均', dur: 18, beats: [
    [0.5, 'Qwen3.8-27B 的 64 层：48 层为固定大小的 state，16 层为随上下文增长的 KV', '固定大小的 state 是有损压缩，无法精确检索原文。因此保留少量 full attention 层负责检索。'],
    [10.0, '下方加入 Kimi K3 与 MiniMax-Text-01 的层排布', '各模型的比例相近，约三个线性 attention 层配一个 full attention 层。'],
    [14.0, '右侧两条：全部 full attention 49.87，3:1 混合 53.81', '混合模型的评测结果高于全部使用 full attention 的模型。'],
  ] },
  { id: 'memory', tag: '显存与推理系统', head: '256K：{t:16.1 GiB} 对 {p:64 GiB}；短上下文时 state {r:更大}', src: 'Qwen3.8-27B，KV 按 BF16、state 按 FP32 计；Kimi K3 技术报告', dur: 20, beats: [
    [0.5, '左上：每个请求 64 层合计，混合 16.1 GiB，全部 full attention 64 GiB', '以 Qwen3.8-27B、256K 上下文计，混合模型的缓存约为全部 full attention 的四分之一。'],
    [6.5, '右上放大图：一个 GDN 层，KV 在 768 个 token 处超过 state', '上下文较短时，state 反而大于 KV，超过几百个 token 才开始节省。'],
    [12.0, '下方：每 512 个 token 保存一份 state 副本；新请求前 2800 个 token 相同，从 2560 处继续计算', 'state 原地更新，复用前缀只能从预先保存的副本恢复；投机解码中被拒绝的 token，也要靠重放撤回。'],
  ] },
  { id: 'routes', tag: '两条路线', head: 'MiniMax {r:退回} full attention', src: 'MiniMax-01 论文；MiniMax M2 博客；各模型 config.json', dur: 15, beats: [
    [0.5, 'MiniMax 三代模型的层排布：7:1 混合、全部 full attention、稀疏 attention', 'MiniMax 的路线与此相反。'],
    [4.5, '下方列出 MiniMax 的理由：规模扩大后多跳推理出现缺陷', '模型规模扩大后，其混合模型在多跳推理上表现出明显缺陷。'],
    [9.5, '右侧两栏：线性 attention 混合 Kimi、Qwen；稀疏 attention DeepSeek、GLM、MiniMax', '另一条路线是稀疏 attention：逐 token 保存 KV，每一步只读取其中一部分。'],
  ] },
];
