# 技术图表参考

## 原则

- **先想清楚这张图回答什么问题**，再决定画什么。一张图只承担一个解释任务。
- **图形就是解释**：用面积、长度、位置和连接表示数学量与关系，比例与数值一致。读者应能从图里直接看出“多了什么、少了什么、从哪到哪”。
- **具体例子 + 一条示例路径**：用可数的小例子（几个 token、几个 block），并把其中一条完整路径加粗描边，其余保持完整但不抢眼。
- **颜色有语义**：一种颜色对应一类对象或状态，全文沿用；对照图只改变要比较的因素。
- **直接标注**：标签贴近对象，少用图例；数学表达式用 KaTeX。
- **引用原图**保持原样。

## 本站范例

[vLLM PagedAttention](../content/vllm-paged-attention.md) 的三张自绘图（`src/articles/vllm-paged-attention/VllmPagedFigures.tsx`）是当前认可的风格：

| 图 | 做法 |
| --- | --- |
| 连续预留 vs 分页 | 同一片 slot 网格左右对照，只改分配方式；按状态着色，面积即浪费量 |
| block table 地址翻译 | 逻辑 block → 表项 → 线性物理地址三层，请求各用一种颜色；加粗一个 token 的完整翻译路径，图下用 KaTeX 分步算一遍 |
| prefix 共享 | 共享部分画成跨两行的一块，直接表示“只存一份”；未写满的 block 用虚线空格 |

可复用的图元见 `src/figure-kit.tsx`。

## 外部参考

| 参考 | 借鉴点 |
| --- | --- |
| [colah · Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) | 用矩形的高、宽、面积分别表示概率、编码长度和平均代价，一套图形语言贯穿全文 |
| [Distill · Why Momentum Really Works](https://distill.pub/2017/momentum/) | 从可分析的具体模型出发，参数与轨迹联动，交互服务于比较 |
| [Distill · Circuits](https://distill.pub/2020/circuits/) | 局部例子与整体结构放在同一解释脉络中 |
