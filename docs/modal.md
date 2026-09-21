# Modal 调用与验证情况

供本项目 Codex 开展 GPU 实验时读取；不是博客正文，也不从前端调用 Modal。

## 已有能力

2026-09-18 在本机验证：Modal CLI 1.5.5 可调用，当前连接的默认环境应用列表、工作区月账单查询成功。未在此次验证中创建应用、申请 GPU 或执行实验；H100/B200 的可分配性和实验镜像仍需按任务实际验证。动态应用数量、运行状态和费用不固化到仓库。

复用已有 CLI/SDK 即可，无需新建服务或复制 Dashboard 适配器。CLI 查找顺序为 PATH、`~/.local/share/personal-dashboard/runtimes/modal-current/bin/modal`、旧备用目录 `~/.local/share/personal-dashboard/runtimes/modal/bin/modal`。旧版本不保证支持账单命令。

## 资源选择与费用控制

高价值 GPU 或任何多卡实验必须遵守 [AGENTS.md 的实验计划约束](../AGENTS.md#高价值-gpu-与多卡实验)：先写好一次性测试计划，向用户说明预计时间与费用，再申请 GPU。首次探测也包含在计划内；已有硬件授权不等于可以省略计划和估算。

Modal 主要承担短期调试、集中测量和特定架构验证，重点是本地或国内难以获得（用户所称“禁售”）的高级卡，例如 Hopper、Blackwell。这里记录的是资源偏好，不把平台目录等同于账号在任意地区都能申请到资源。

长期、多轮、交互式调试应向用户申请常驻 RTX 4090 等资源。优先在常驻卡完成通用正确性检查、数据准备和代码调试，再把目标架构专有功能、互连拓扑与最终性能测试集中到 Modal；必须依赖目标硬件的部分不得降级替代。

1. 启动前整理完整测试矩阵、输入、依赖和输出格式，尽可能先完成无需目标 GPU 的检查。
2. 尽量同一镜像、一次申请、一次远端调用完成该硬件上的全部已准备任务；跨型号测试分别申请对应设备，但由同一个实验入口组织。
3. 在同一个容器内依次完成设备与拓扑检查、正确性、预热、性能、显存及必要 profiler 采集，复用数据与编译缓存；不把每个小测试单独启动一个容器。
4. 根据任务设置超时、容器上限和 GPU 数量，不设置无必要的常驻容器，不默认 detach、持久部署或自动重复重试。异常记录后有针对性补测，不盲目重跑全部矩阵。
5. 结果及时返回或持久化，测试完成后结束调用，确认不再有运行容器；停止写作和分析期间的 GPU 占用。合并运行是减少重复开销的策略，不保证每种任务都更便宜。

### 执行前计划

计划保存在对应任务的实验记录中，至少包括：

- **资源与任务**：型号、卡数、容器数、并发、测试矩阵、准备情况、结果文件与完成条件；设备/拓扑检查、正确性、性能和必要采集合并安排。
- **时间**：准备与镜像构建、排队、GPU 占用、结果回传各阶段的估计，以及总时长和最大 GPU 占用时间；未知排队时间单独标注。
- **费用**：执行前查询的费率、币种、GPU 与 CPU/内存等费用分项、预计区间及预算上限；说明构建、启动和收尾等开销的假设，不把函数 timeout 直接当成完整计费时长。
- **停止与补测**：超时、最大容器数、重试限制、异常退出和结果保存方式；完成后核查容器退出。补测只覆盖缺失项，并先更新时间和费用估算。

GPU 费用估算采用“单卡每秒费率 × 卡数 × 预计计费秒数”；按各计费资源的实际占用时段另算 CPU、内存等费用。一次申请失败不自动换型号、减卡或循环重试；保留原因，再调整计划。无需目标 GPU 的本地检查与只读状态/费率查询可在申请前完成。

## GPU 配置目录

核对日期：2026-09-18。下表覆盖当日 [Modal 官方 GPU 目录](https://modal.com/docs/guide/gpu) 的全部 14 个参数值；架构特性按 [NVIDIA Compute Capability](https://developer.nvidia.com/cuda/gpus) 区分。表中“最多”是文档标注的单容器、同一物理机 GPU 数量上限，不保证任意数量当时有库存，也不是当前账号已验证配额。

| `gpu` 参数 | 架构 / Compute Capability | 官方单机数量说明 | 选择注意 |
| --- | --- | --- | --- |
| `T4` | Turing / 7.5 | 最多 8 卡 | 老架构对照 |
| `L4` | Ada / 8.9 | 最多 8 卡 | 按实验需要选择 |
| `A10` | Ampere / 8.6 | 最多 4 卡 | 不写成八卡支持 |
| `L40S` | Ada / 8.9 | 最多 8 卡 | 与 Hopper 特性不同 |
| `A100` | Ampere / 8.0 | 最多 8 卡 | 默认 40 GB，可自动升级 80 GB |
| `A100-40GB` | Ampere / 8.0 | A100 系列最多 8 卡 | 明确要求 40 GB |
| `A100-80GB` | Ampere / 8.0 | A100 系列最多 8 卡 | 明确要求 80 GB |
| `RTX-PRO-6000` | Blackwell / 12.0 | 型号已列出，多卡上限未明确 | 八卡仅本机解析通过，待服务端及实机验证 |
| `H100` | Hopper / 9.0 | 最多 8 卡 | 可自动升级为 H200 |
| `H100!` | Hopper / 9.0 | H100 系列最多 8 卡 | 精确 H100，避免自动升级 |
| `H200` | Hopper / 9.0 | 最多 8 卡 | 精确目标型号需实机核对 |
| `B200` | Blackwell / 10.0 | 最多 8 卡 | 不主动接受升级型号 |
| `B200+` | Blackwell / 10.0 或 10.3 | B200/B300 系列最多 8 卡 | 允许分配 B200 或 B300 |
| `B300` | Blackwell Ultra / 10.3 | 最多 8 卡 | 官方要求 CUDA 13.1+ |

数量通过 `:<N>` 后缀表达，省略为单卡。例如 `H100!:2`、`H100!:4`、`H100!:8`、`B200:8`、`A100-80GB:8`、`A10:4`。单机多卡不等于多节点，也不能由卡数推断 NVLink/NVSwitch、PCIe 或 NUMA 拓扑；需要这些结论时在同一次实验中采集 `nvidia-smi topo -m` 和相应通信测量。

`RTX-PRO-6000` 是 Blackwell 型号，不是 RTX 6000 Ada 或 RTX A6000；RTX PRO 6000 的 CC 12.0 与 B200 的 CC 10.0 也不能当作相同指令集或相同编译目标。常驻 RTX 4090 属于 Ada / CC 8.9，未出现在上述 Modal 目录中，应向用户申请其他已有资源。

### 八卡 RTX PRO 6000 的准确状态

2026-09-18 实际申请补充：已登记八卡远端函数并观察到待处理输入，但首次等待 10 分钟、修正监测后的再次等待 30 分钟均未获得设备信息；长等待期间 60 次采样未观察到运行容器。请求结束后已确认应用停止、容器与 backlog 均为 0。**八卡实机分配仍未验证**，不能将排队等同于已可用；详见[实验记录](rtx-pro-6000-experiment.md#第二轮延长-modal-等待)。

```python
@app.function(image=image, gpu="RTX-PRO-6000:8", timeout=300)
def experiment_suite():
    # 放入准备好的完整测试矩阵；不要逐项重新申请八卡。
    ...
```

本机 SDK 1.5.5 已将该字符串解析为型号 `RTX-PRO-6000`、数量 8；这只是客户端语法检查，不是服务端受理、账号配额或库存验证，更不是成功分配。同样对全部 14 个型号/别名及各自 1、2、4、8 卡（A10 至 4 卡）的 55 个代表配置进行了本地解析，未发起任何 GPU 申请。实际运行仍可能被拒绝或排队，不能从 SDK 接受字符串推断全部组合可用。

### 升级与回退

精确对照实验优先固定型号，例如 `H100!`；不默认使用 `B200+`、`gpu="any"` 或 GPU 回退列表。只有当前实验允许硬件变化时，才使用如 `gpu=["H100", "A100-80GB"]` 的有序回退配置，并按实际分配型号分别记录结果。B200+ 可能分配 B300，镜像必须同时兼容两种设备及 B300 的 CUDA 要求。

## 调用配置与执行范围

GPU 配置写入 `@app.function(...)`，与 `image`、`timeout` 一起固定。短期单容器测试可以显式设置 `max_containers=1`、`min_containers=0`，CPU 和内存则按工作负载设置 `cpu`、`memory`；内存单位为 MiB，避免为纯 GPU 测量随意分配大量主机资源。依赖安装放进镜像构建，固定版本，不要在每个测试函数里重复安装。

- 一次性运行：`modal run experiment.py::main`；本地入口用 `.remote()` 调用远端整套实验。
- 只读状态：`modal app list --json`；具体日志命令先用 `modal app logs --help`，不保存带私人标识的原始输出到仓库。
- 脱离终端：`modal run --detach ...`；只在任务明确需要时使用，须跟踪运行状态。
- 持久应用：`modal deploy experiment.py`；用于明确的持久服务需求，不作为短期测试默认入口。
- GPU 数量按单容器配置，容器数量与调用并发单独控制；不能用八个单卡容器代替八卡同机互连实验。

SDK 的完整参数以本机 `modal.App.function` 签名和 [官方函数参考](https://modal.com/docs/reference/modal.App#function) 为准；本说明覆盖 GPU 实验所需配置，不复制全部平台 API。

## 本机调用

```bash
# 本机已验证的 CLI；若 PATH 中已有适用版本，也可直接使用 modal。
MODAL_CLI="$HOME/.local/share/personal-dashboard/runtimes/modal-current/bin/modal"
"$MODAL_CLI" --version
"$MODAL_CLI" app list --json
"$MODAL_CLI" billing summary --for "$(date -u +%Y-%m)" --json
```

命令复用当前 profile；需要切换时显式传入 `--profile PROFILE_NAME`，不要把真实 profile 名写进共享示例。应用环境按命令的 `--env`、`MODAL_ENVIRONMENT`、profile 或工作区默认配置选择，账单范围则是工作区。

凭据通常由 `~/.modal.toml` 管理，也可能来自既有环境变量。不要为检查认证输出配置内容或 token；配置权限应为 600。当前已授权，不需要重新登录。其他机器确需配置时使用官方 `modal token set` 交互输入，不把 secret 写入命令行或仓库。

若相邻 Dashboard 仓库仍在，也可直接读取其统一结果：

```bash
python3 ../personal-dashboard/tools/modal_service.py
```

该适配器每次 CLI 调用超时 30 秒，返回应用列表、运行容器数量和账单状态；账单失败独立标记，不抹掉已成功读取的应用列表。不要将完整输出重定向到博客仓库。

## GPU 实验入口

以下仅展示调用结构，尚未执行；开展实验前沿用 `AGENTS.md` 的硬件确认规则。用户已明确当前实验所用资源时，直接按授权推进。

```python
import modal

app = modal.App("blog-gpu-experiment")
image = modal.Image.debian_slim(python_version="3.11")

@app.function(image=image, gpu="H100!", timeout=300)
def probe():
    import subprocess
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()

@app.local_entrypoint()
def main():
    print(probe.remote())
```

将实验定义保存为任务自己的 Python 文件后，使用下面的命令运行；此步骤会申请远端资源并可能产生费用，不属于只读查询：

```bash
"$MODAL_CLI" run experiment.py::main
```

`H100!` 要求 H100，避免普通 `H100` 请求被自动升级为 H200；B200 使用 `gpu="B200"`；多卡数量按具体实验要求配置，并在执行前确认当前 CLI/API 支持。不要为获得空闲资源擅自使用型号回退列表或 B200+ 升级选项。参考 [官方 GPU 配置](https://modal.com/docs/guide/gpu)。

正式实验需按硬件选择兼容的 CUDA、PyTorch、Triton 镜像与固定依赖版本；上面的最小探测镜像不包含完整实验环境。远端函数通过 `.remote()` 调用；本地数据上传和结果保存按具体实验配置，避免隐式上传整个工作区。一次性实验优先 `modal run`，只有任务需要持久部署时才使用 `modal deploy`。`--detach` 会改变断开后的运行行为，不默认启用。调用语法可用 `modal run --help` 查看。

## 账单与测量边界

执行前查询 `modal billing rates --json` 或 [官方价格](https://modal.com/pricing)，使用当时费率估算，不把价格快照当作固定收费。GPU 估算为单卡每秒单价 × 卡数 × 实际计费秒数，另外计算 CPU、主机内存、存储及适用的其他收费；设置超时不能替代完整预算控制。

- 月账单使用当前 UTC 自然月；应用列表只代表所选环境的当前查询范围，不是完整历史。
- `metered_cost` 是原始用量金额，`billed_cost` 是调整后费用；后者不是已支付金额，也不是历史累计支出。金额为 USD。
- Dashboard 适配器将账单中的额度调整归一化为已使用抵扣 `credits_applied_usd`；这不是剩余额度。接口未提供剩余额度时保持未知，不按固定赠额反推。
- 本次只验证账单汇总，未接入 GPU 逐项费用。权限、网络或 CLI 版本错误不能解释为零费用。
- 实验输出按项目已有测量目录约定保存；发布前剔除账号、应用 ID、凭据及私人账单。正文仍只注明显卡型号，详细环境与测量方法留在经检查的原始记录中。

来源：[Dashboard Modal 手册](../../personal-dashboard/catalog/services/modal.md)、[现有适配器](../../personal-dashboard/tools/modal_service.py)、[官方账单 CLI](https://modal.com/docs/cli/latest/billing)、[官方运行 CLI](https://modal.com/docs/cli/latest/run)。相邻仓库链接仅在本机目录结构一致时可用，换机器以官方文档和本机实际安装为准。
