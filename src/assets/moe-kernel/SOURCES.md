# MoE Kernel 视频：画面素材出处

视频没有直接使用外部图片。画面上的一手材料如下。

| 画面 | 内容 | 出处 |
| --- | --- | --- |
| 第 1、3、4、6、7 幕路由 | Qwen3-30B-A3B 第 24 层、16 个请求第 20 步解码的真实 top-8 与 router 分数 | 本站实测 `public/measurements/moe-kernel/20261008-105322/results.json` |
| 第 5、8 幕时间线 | `torch.profiler` 记录的 GPU kernel 列表 | 同上 |
| 第 5 幕代码 | `Qwen3MoeExperts.forward` 的循环（`linear` 为 `nn.functional.linear` 的简写） | transformers `models/qwen3_moe/modeling_qwen3_moe.py` L221–243，commit `536ecc0` |
| 第 7 幕代码 | `fused_moe_kernel` 中取 `sorted_token_ids`、`expert_ids` 与计算指针的几行（省略号为节略） | vLLM `model_executor/layers/fused_moe/fused_moe.py` L408–476，commit `fce64bc5` |
