# llm-quantization 图片来源

四张图都是本站视频 `videos/llm-quantization/film.html` 的画面，由 `node film.mjs --stills 54 85.4 100 137.6` 导出后裁出三联图区域（x 86–1836，y 从 290 起），缩到 1520 px 宽。图是程序生成的类比：把权重想成像素的亮度，每张结果按对应格式真的算出来。

| 文件 | 视频的幕 | 内容 |
| --- | --- | --- |
| `picture-bits.png` | 第 3 幕 | 同一张图按 16 / 8 / 4 bit 存 |
| `picture-fp8.png` | 第 5 幕 | 加一盏亮度 100 的灯，INT8 对 FP8 |
| `picture-groups.png` | 第 6 幕 | 同一张带灯的图压到 4 bit：INT4 每 128 个一组，NVFP4 每 16 个一组 |
| `picture-diffusion.png` | 第 8 幕 | 4 bit 各自取最近的刻度，对比误差分给邻居（Floyd–Steinberg） |
