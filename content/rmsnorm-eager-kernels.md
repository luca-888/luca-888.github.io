`1024×4096` 的 kernel 顺序与对应 ATen 算子；类型转换统一列为 `aten::copy_`。

**Forward · 9 个 kernel**

```text
01  aten::copy_   x → FP32
02  aten::copy_   gamma → FP32
03  aten::pow     x.square()
04  aten::mean    沿 N 求均值 → s
05  aten::add     s + eps
06  aten::rsqrt   rsqrt(s + eps) → r
07  aten::mul     x * r → x_hat
08  aten::mul     x_hat * gamma → y
09  aten::copy_   y → BF16
```

**Backward · 17 个 kernel**

```text
01  aten::copy_   x → FP32
02  aten::copy_   gamma → FP32
03  aten::copy_   g → FP32
04  aten::mul     x * r → x_hat
05  aten::mul     g * gamma → dx_hat
06  aten::mul     dx_hat * x
07  aten::sum     沿 N 求和 → dr
08  aten::mul     r * dx_hat（直接路径贡献）
09  aten::pow     r.pow(3)
10  aten::mul     x * r³
11  aten::mul     上一步结果 * dr
12  aten::div     上一步结果 / N
13  aten::sub     第 08 步 − 第 12 步 → dx（两条路径贡献相加）
14  aten::mul     g * x_hat
15  aten::sum     沿 M 求和 → dgamma
16  aten::copy_   dx → BF16
17  aten::copy_   dgamma → BF16
```

