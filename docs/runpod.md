# Runpod 调用与费用控制

供本项目 Codex 开展实验时读取，不属于博客正文，不从静态网页调用云 API。

## 已验证的本机入口

2026-09-18 已验证官方 CLI 2.14.0 的账户信息、Pods 列表和 GPU 价格/库存目录读取成功。同机八卡 RTX PRO 6000 Server Edition 已在 EUR-IS-2 分配成功，实际报价 $16.72/h；容器因 Docker 镜像拉取失败未就绪，SSH 和实验仍未验证。所有本轮 Pod 已删除并核查列表为空，详见[实验记录](rtx-pro-6000-runpod-experiment.md)。动态余额、资源 ID 与原始响应不保存到仓库。

本机已安装 Runpod 插件和相关技能。开始任务先使用 `runpod` 技能；当前会话有 Runpod MCP 工具时可直接复用，否则使用 `runpodctl` 技能与 CLI。插件已安装、OAuth 已授权、当前会话工具可调用、实际查询成功是不同状态，不应混为一谈。

```bash
RUNPOD_CLI="$HOME/.local/bin/runpodctl"
"$RUNPOD_CLI" version
"$RUNPOD_CLI" user
"$RUNPOD_CLI" pod list --all
"$RUNPOD_CLI" gpu list --include-unavailable
```

CLI 自动使用本机 `~/.runpod/config.toml` 的 `apikey` 配置或 `RUNPOD_API_KEY` 环境变量。现有凭据已配置，不需要重新登录或复制密钥；配置文件权限为 600。不要 cat 配置、打印环境变量或把凭据放进命令参数。账号命令可能返回邮箱等私人字段，只在本机按需读取，不把完整输出保存到项目或发进对话。Codex MCP OAuth 不等于 CLI 认证。

相邻 Dashboard 已接入账户余额和 Pods 只读展示，命令行共用其适配器：

```bash
python3 ../personal-dashboard/tools/runpod_service.py
```

相邻仓库路径仅在本机目录结构一致时适用。适配器不租用 GPU，不执行实验，也不代表全部 Serverless 资源或账单已接入。

## RTX PRO 6000 同机八卡

精确型号为 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，每卡 96GB。`Max-Q`、`Workstation Edition`、RTX 6000 Ada、RTX A6000 和 MIG 24GB/48GB 分片都是不同选择，不能自动替代。GPU 目录显示 available 或 Low 只代表该型号的汇总库存，不证明单个节点可分配 8 卡。

创建前按 [高价值 GPU 与多卡实验](../AGENTS.md#高价值-gpu-与多卡实验) 完成计划与费用说明，并沿用当前任务已授权的硬件、预算和范围。先准备依赖、数据、测试矩阵、输出目录与收尾方式。镜像应按任务固定版本，包含兼容该卡的 CUDA、PyTorch 和必要的通信测试工具；本手册不声称任意 PyTorch 模板都已满足这些依赖。

先用只读命令查看模板与实际参数：

```bash
"$RUNPOD_CLI" template search pytorch
"$RUNPOD_CLI" pod create --help
```

以下为付费创建模板，仅在具体实验获得授权并完成上述计划后执行。`RUNPOD_EXPERIMENT_IMAGE` 必须预先设置为已准备好的固定镜像；不使用未经验证的 latest 镜像。磁盘大小按输入与结果调整，不能把示例当作适合所有任务的配置。

```bash
: "${RUNPOD_EXPERIMENT_IMAGE:?先设置实验镜像及固定版本}"
"$RUNPOD_CLI" pod create \
  --name blog-rtx-pro-6000-suite \
  --gpu-id "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
  --gpu-count 8 \
  --cloud-type SECURE \
  --image "$RUNPOD_EXPERIMENT_IMAGE" \
  --container-disk-in-gb 40 \
  --ports 22/tcp \
  --ssh \
  --wait --wait-timeout 10m
```

这是一个八卡 Pod，不能改成八个单卡 Pod。选择 Community 必须符合任务计划，并按 CLI 要求配置公开 SSH 连接（如 `--public-ip`）；更便宜的目录价不代表当时有同机八卡。创建结果的 Pod ID 只保存在本机任务记录，后续命令使用本次真实 ID，不复用旧记录中的资源 ID。

**等待超时不会删除或停止 Pod。** `--wait` 检查 SSH 服务可达，不等于认证、CUDA、依赖和实验均已就绪。超时或创建结果不明确时，先 `pod list --all` / `pod get` 确认已有资源，再决定后续操作，避免重复创建。

本次 CLI 2.14.0 的 `pod get --include-machine` 实际返回精确 GPU 型号于 `machine.gpuId`，顶层没有 `gpuTypeId`；不能把字段缺失误判为错误型号。`runtimeStatus=initializing / awaiting_container` 只能说明容器未就绪。若 API 日志接口没有返回内容，应同时检查浏览器控制台系统日志；本次拉取失败的具体原因由控制台日志确认。

## 连接、集中执行与回收

```bash
: "${RUNPOD_POD_ID:?设置为本次创建返回的 Pod ID}"
"$RUNPOD_CLI" pod get "$RUNPOD_POD_ID"
"$RUNPOD_CLI" ssh info "$RUNPOD_POD_ID"
```

按 `ssh info` 返回的连接方式与本机密钥连接；SSH 密钥可先用 `runpodctl ssh list-keys` 核对。不要打印私钥或为测试默认生成、上传新密钥。使用 SSH/SCP 仅上传本次需要的脚本和输入，回传结果到仓库外的本机任务目录，不上传整个工作区或本机配置。

进入远端后，先核实设备数量、精确型号与互连，不能从“八卡”推断 NVLink、PCIe 或 NUMA 拓扑：

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
nvidia-smi topo -m
```

本仓库已有 [拓扑测试套件](../scripts/rtx-topology-suite.py)，先检查其依赖、参数与当前实验计划。上传到远端并准备好依赖后，可在同一 Pod 集中执行；下面的时间仅为结构示例，实际以批准的计划为准：

```bash
python3 rtx-topology-suite.py --help
python3 rtx-topology-suite.py --output /workspace/rtx-topology.json --budget-seconds 660
```

套件预算不是云资源自动关机时间。启动前须准备独立于 SSH 连接的停止机制或可靠的本机收尾控制，覆盖正常结束、报错、断线、最大 GPU 占用时间和预算上限；不能只依赖脚本退出。结果保存、回传和完整性检查后，及时执行本次授权范围内的清理：

```bash
# 停止计算；持久存储可能继续收费。
"$RUNPOD_CLI" pod stop "$RUNPOD_POD_ID"
"$RUNPOD_CLI" pod get "$RUNPOD_POD_ID"

# 仅在结果已回传、不再保留本次 Pod 时终止；Pod 本地数据会丢失。
# "$RUNPOD_CLI" pod delete "$RUNPOD_POD_ID"
```

终止后用 `pod list --all` 核查；独立 network volume 不应假定随 Pod 删除，若本次创建了卷需单独核查其保留需求和费用。状态判断使用 `runtimeStatus`；`desiredStatus=RUNNING` 不表示 GPU 或容器已就绪。

## 预算参考

2026-09-18 CLI 查询的 Server Edition 目录单卡单价为 Community $1.69/小时、Secure $2.09/小时；这是价格快照，不是八卡节点的最终报价。八卡计算费用按“单卡时价 × 8 × 占用小时数”估算：

| 八卡占用 | Community | Secure |
| --- | ---: | ---: |
| 30 分钟 | $6.76 | $8.36 |
| 1 小时 | $13.52 | $16.72 |
| 2 小时 | $27.04 | $33.44 |

存储和适用税费等另计；准备、启动、下载和收尾中实际计费的占用也要计入。执行前重新查询实际报价，分开记录排队与计费时间，不把等待超时当作费用封顶。$20–30 可作为约一小时集中调试的初步预算建议，但不是当前任务的消费授权或完成保证。

Runpod 与 Modal 均优先用于短期集中调试及目标架构验证；尽量一次整合已准备的所有测试，测完释放。长期交互调试向用户申请常驻 RTX 4090 等资源，不能用较低配置替代目标架构必需的实测。

来源：[官方 CLI](https://github.com/runpod/runpodctl)、[RTX PRO 6000](https://www.runpod.io/gpu-models/rtx-pro-6000)、[Pods 计费](https://docs.runpod.io/pods/pricing)。命令按本机 CLI 2.14.0 的 help 核对；本轮创建、状态查询与删除已有实际验证，SSH 和测试执行尚未成功。
