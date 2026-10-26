# Tensor Core 视频：画面素材出处

视频没有直接使用外部图片。画面上的一手材料如下，均为原文照录。

| 画面 | 内容 | 出处 |
| --- | --- | --- |
| 第 1、9 幕芯片 | 自绘矢量图，结构参照 NVIDIA newsroom 的 Blackwell 渲染图（未嵌入原图） | NVIDIA Blackwell 官方渲染图 |
| 第 6 幕代码 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 的 inline asm | CUTLASS `include/cute/arch/mma_sm80.hpp` L224–252，commit `0b55a2f6` |
| 第 3、4 幕能量 | HFMA 1.5 pJ、开销 30 pJ（2000%）、HMMA 开销 22% | Bill Dally，Stanford AHA Retreat 2023 keynote 第 12 页 |
