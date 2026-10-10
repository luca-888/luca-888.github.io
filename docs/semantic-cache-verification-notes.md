# Semantic cache 的命中校验：调研笔记

## 范围与状态

- 2026-10-10 开始调研并提交参考方案。同日由协调五篇文章的会话转达确认（并非用户在本会话中直接回复）：主线按建议写法（原理 + `b1b0e40` 案例 + 现状一节），三张图都画、跳过草图直接精画，图 2（校验器故障）用演示数据并在图注标明；作者身份正文提一次；两项可选实测不做，正文留占位；理解检查不等用户作答，题目与参考答案写在本文末尾。
- 正文、三张图、页面入口已完成，状态 draft；成稿自查后由不带写作上下文的 subagent 用 `de-ai-edit` 改过一遍（7 句，均为措辞）。
- 本篇只讲 semantic cache 的命中校验与校验失败策略。同批另外四篇（CUDA Graph 正确性、Qwen-VL 视频 token、Liger 接入 HF 模型、BF16 下 1+w 的计算位置）不涉及。
- 本阶段不做实测，占位见文末。

## 源码版本

| 仓库 | commit | 日期 | 说明 |
| --- | --- | --- | --- |
| vllm-project/semantic-router main | `a9d8539` | 2026-10-10 | 现状：`pkg/cache/polarity.go`、`polarity_candidate.go`、`inmemory_cache_search.go`、`website/docs/model-runtime/migrate.md` |
| vllm-project/semantic-router（#3162 合并点） | `b1b0e40` | 2026-09-18 | 案例代码：`pkg/cache/inmemory_cache_polarity.go`、`polarity_nli.go`、`inmemory_cache_search.go`、`pkg/extproc/req_filter_cache.go` |
| zilliztech/GPTCache main | `a74ac65` | 2026-09-22 | `similarity_evaluation/onnx.py`、`processor/post.py`、`adapter/adapter.py` |

- 以上源码用 `gh api .../contents?ref=<commit>` 按 commit 读取。semantic-router 已加入 reference-repos 的 `sync.sh`（“请求路由”一组），2026-10-10 两次浅克隆都因网络中断失败（`fetch-pack: invalid index-pack output`，浅克隆超过 200 MB），之后网络稳定时再运行 `./sync.sh semantic-router`，克隆成功后补记其 commit。

## 参考清单（按重要程度）

| 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| [#3162](https://github.com/vllm-project/semantic-router/pull/3162) Treat semantic-cache NLI verifier failures as cache misses | luca-888（本文作者） | 已读描述、全部 review 与评论、diff | 核心 | 第 4、5 节：fail-closed 的实现、取消、保留相似度、e2e 改动 |
| [#3176](https://github.com/vllm-project/semantic-router/issues/3176) Define the runtime failure policy for semantic-cache NLI verification | luca-888；维护者 Xunzhuo 决定 | 已读 | 核心 | 第 4 节：“只对未校验的候选 fail-closed，请求本身照常可用”；non-goals |
| [#3075](https://github.com/vllm-project/semantic-router/pull/3075) Optional NLI polarity tier for the in-memory semantic cache | yaojiejia | 已读描述、review、文件列表 | 核心 | 第 3、4 节：NLI 层设计、原 fail-open 及理由、E2E 观测 |
| [#2691](https://github.com/vllm-project/semantic-router/issues/2691) Prevent semantic-cache false hits on negated or antonym queries | issue 作者 | 已读 | 核心 | 开头与第 2 节：相似度数字、阈值无法分开两类 |
| [#2751](https://github.com/vllm-project/semantic-router/issues/2751) Optional NLI polarity tier (L2) | WUKUNTAI-0211 | 已读 | 重要 | 第 3 节：NLI 分数（矛盾约 0.96 对 ≤0.002）、约 70 ms/次 CPU |
| [#4512](https://github.com/vllm-project/semantic-router/pull/4512) Built-in model runtime + `migrate.md` | Xunzhuo | 已读描述中相关段落、迁移文档 | 核心（现状） | 第 6 节：NLI 层随 native binding 一起退役 |
| [#3944](https://github.com/vllm-project/semantic-router/pull/3944) Resolve verified v0.4 release regressions | Xunzhuo | 已读缓存相关段落 | 重要（现状） | 第 6 节：词法下限覆盖全部六种后端 |
| [#4148](https://github.com/vllm-project/semantic-router/pull/4148) Flag semantic hits the English negation guard can't judge | 1fanwang | 已读 | 重要（现状） | 第 6 节：`negation_guard: checked / not_applicable` |
| [#2728](https://github.com/vllm-project/semantic-router/pull/2728) guard in-memory semantic cache polarity | — | 已读描述前半 | 补充 | 第 3 节：词法层设计（先跑便宜的一层） |
| [#2473](https://github.com/vllm-project/semantic-router/issues/2473) propagate cancellation and return similarity per lookup | — | 已读 | 补充 | 第 5 节：相似度按请求返回、取消传到底层 |
| [#3036](https://github.com/vllm-project/semantic-router/issues/3036) Epic: response caching safe, measurable, lifecycle-aware | — | 已读 | 补充 | 背景：“embedding 相似度本身不能证明两个请求等价” |
| [vCache: Verified Semantic Prompt Caching](https://arxiv.org/abs/2502.03771)，ICLR 2026 | Schroeder et al.（UC Berkeley 等） | 已读 arXiv HTML v5 主要部分（经摘要工具） | 重要 | 第 2、3 节对比：正确与错误命中的相似度分布重叠（均值 0.84 / 0.85）；每条缓存学习自己的阈值，按概率调用 LLM 校验 |
| GPTCache 源码（见上表） | Zilliz | 已读相关函数 | 重要 | 第 3、4 节对比：ONNX cross-encoder 出异常返回 0（等同拒绝，静默）；`LlmVerifier` 出异常返回 None，回到真实 LLM 调用 |
| [LangCache concepts](https://redis.io/docs/latest/develop/ai/context-engine/langcache/concepts/) | Redis | 已读 | 重要 | 第 3 节对比：只靠阈值（默认 0.85），文档明说放弃“命中必正确”的保证 |
| [Envoy ext_authz](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_authz/v3/ext_authz.proto)、[rate limit](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ratelimit/v3/rate_limit.proto) | Envoy | 已读相关字段 | 重要 | 第 4 节：鉴权默认 fail-closed（`failure_mode_allow` 默认 false），限流 `failure_mode_deny` 为 true 时才拒绝（proto 默认 false，待核实文档措辞） |
| [Kubernetes admission webhook failurePolicy](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/) | Kubernetes | 已读该节 | 重要 | 第 4 节：默认 `Fail`；明确拒绝不受 failure policy 影响（“拒绝”与“故障”分开） |
| [Avoiding fallback in distributed systems](https://builder.aws.com/content/3EuS9Sakq7L3VLQIF3qzfMfke1Y/avoiding-fallback-in-distributed-systems) | Jacob Gabrielson（AWS） | 已读（经摘要工具） | 重要 | 第 4 节反对意见：2001 年缓存失效后回源压垮数据库；回退路径平时不跑、难测试 |
| [Building Secure and Reliable Systems, ch. 8](https://google.github.io/building-secure-and-reliable-systems/raw/ch08.html) | Google | 已读相关节（经摘要工具） | 补充 | 第 4 节：安全关键的检查不应 fail open，否则攻击者只需让它不可用 |

## 阅读要点

### 问题与数字（#2691、#2751、#3075）

- mmbert 384 维 embedding 下，“turn on / off dark mode”余弦 0.915，“enable / disable 2FA”0.823；真正的同义改写“enable 2FA / turn on 2FA”只有 0.673。阈值 0.80 时 12 对反义中 8 对误命中，0.86 时 3 对，0.92 时 0 对但同义改写也全部 miss。
- NLI 层（#2751 提出，#3075 实现）：只对阈值之上的唯一最佳候选跑一次，在缓存锁外；premise 为缓存中的问题、hypothesis 为新问题；contradiction > 0.5 即拒绝。模型 `tasksource/ModernBERT-base-nli`，复用 hallucination explainer。#2751 测得反义对 contradiction 约 0.97–0.997，同义 0.001；单线程 CPU 约 70 ms/次（p99 约 97 ms）。
- #3075 的 E2E：三对反义（相似度 0.93–0.97，contradiction 0.95–0.99）均被拒；“Should I commit / not commit this change?”（相似度 0.97）contradiction 低于 0.5，没拦住，这类交给词法层。
- #3075 的失败策略：模型在查询时出错就照样返回缓存并记 `cache_polarity_nli_skipped`（fail open）。代码注释的理由：“a model hiccup never turns into a cache outage”。配置期缺模型、启动期加载失败则直接报错（这部分始终是 fail-closed）。
- 维护者在 #3162 首轮 review 说 #2751 与 #3075 “explicitly define” fail-open。核对：#2751 正文没有写失败策略，fail-open 出自 #3075 的描述与代码。正文写“#3075 把它定为 fail-open”。

### #3162 的改动（合并点 `b1b0e40`）

- `applyPolarityNLI` 的四种结局：取消（返回 ctx 错误，不计 hit/miss）、校验器缺失或出错（miss，保留候选相似度）、contradiction 超阈值（miss，记 `cache_negation_reject`）、通过（返回给调用方，走命中路径）。
- 取消检查在调用校验器之前和之后各一次。之后那次是新加的：校验器可能因为取消返回错误，也可能在取消后仍返回 `nil`；两种情况都应作为取消处理，而不是记为校验器故障或 miss。测试覆盖 `error=context.Canceled` 与 `error=nil` 两种。
- 校验器故障与 contradiction 拒绝分开记：故障记 `polarity_nli/error` 和 `cache_polarity_nli_skipped`（字段从 `fail_open: true` 改为 `cache_miss: true`），并以普通 miss 计数；不进入 contradiction 相关指标。
- 未校验候选不更新访问信息（`LastAccessAt`、`HitCount`），即不影响 LRU 之类的淘汰。
- ExtProc 层（`performCacheLookup`）：miss 时把 `lookupResult.Similarity` 写入请求上下文，经 `x-vsr-debug` 面暴露为 `x-vsr-cache-similarity`；查询返回错误（含取消）时 span 记 `error / lookup_failed`，不记 hit 也不记 miss。
- e2e：原来断言 miss 的相似度在 [0,1)，理由是“miss 报 1.0 说明命中路径被绕过”；改为 [0,1]，因为校验失败可以让相似度 1.0 的候选也 miss。
- 测试还覆盖“新旧两代 cache 同时服务时，一代的校验器故障不影响另一代”（配置热更新期间，#3768 引入 cache 自有的 verifier）。
- 文档 `stores-and-tools.md`：明确“correctness over cache-hit latency”。
- #3176 的 non-goals：不加重试、熔断器或可配置的失败策略。

### 现状（main `a9d8539`）

- #4512（2026-10-06 合并）把全部 in-process native binding 移除，模型改由独立的 model runtime 服务；NLI 模型（hallucination explainer 与 cache 的 polarity guard）随之退役。`migrate.md`：“Its NLI tier is gone, so there is nothing left to choose”，`polarity_guard` 配置块由 `vllm-sr config migrate` 删除，旧配置启动时被拒。`inmemory_cache_polarity.go`、`polarity_nli.go` 已删除。退役原因是 NLI 模型不在新 runtime 中，与失败策略无关（PR 描述与迁移文档未提失败策略）。
- 现在只有词法层：纯 Go 函数，比较两句的 token 集合，差异不超过 2 个 token 时检查否定词（not、no、never、without、cannot，`n't` 先展开）和 12 组反义词。#3944 把它扩展到 in-memory、Redis、Valkey、Milvus、Qdrant、hybrid 六种后端。
- 与 NLI 层的另一处不同：词法层在扫描候选时就过滤，被拒的候选不终止查找，继续看次优候选；NLI 层只看唯一最佳候选，拒绝即 miss。
- #4148：每个命中带 `negation_guard`：`checked`（两句同词或只差否定词）或 `not_applicable`（有其他词变化，包括非英语）。用于标出词法层无法判断的命中，不改变是否命中。#4146 的复现：德语 `nicht`、中文“不”的否定句与原句相似度 > 0.97，照样命中。
- 结论：本地确定性函数没有“不可用”这一状态，失败策略问题随 NLI 层一起消失；代价转为覆盖面，用 `not_applicable` 标记暴露出来。只要以后再引入远程或模型校验器，同样的问题会重新出现。

### 相关系统

- **GPTCache**：先取 top-k 候选，用 `similarity_evaluation` 打分，过阈值的再交给 post-process。`OnnxModelEvaluation.evaluation` 整体包在 `except Exception: return 0` 中，故障与“判为不相似”无法区分，结果上是 fail-closed 但不留诊断。2025-07 新增的 `LlmVerifier`（#669）让 LLM 回答 yes/no，异常时打印并返回 None，触发真实 LLM 调用，也是 fail-closed。
- **Redis LangCache**：只有相似度阈值（默认 0.85，建议 0.8–0.9）和 attributes 分区；文档明说 semantic caching 放弃“命中必正确”的保证，建议从紧阈值开始。Redis 另在 Hugging Face 发布 LangCache 用的 cross-encoder reranker，未查到服务端集成文档，正文不写。
- **vCache**：不加独立校验器，而是给每条缓存学习一个阈值；以一定概率在“本可命中”时仍调用 LLM，用结果更新该条目的阈值，使错误率不超过用户给的 δ。论文中正确与错误命中的平均相似度为 0.84 与 0.85，最优阈值按条目在 0.71–1.0 之间。
- **fail-open / fail-closed 惯例**：Envoy ext_authz 默认 fail-closed（返回 403），Kubernetes webhook 默认 `Fail`；限流类检查通常 fail-open。Kubernetes 文档区分“webhook 明确拒绝”与“调用 webhook 出错”，failure policy 只管后者，与本文“contradiction 拒绝”和“校验器故障”分开处理相同。
- **反对意见**：AWS 的文章反对回退路径：平时不跑、难测试，且会把负载转移到下游（2001 年缓存失效后 web 服务器直接查库，数据库被压垮）。fail-closed 在校验器故障时把所有本可命中的请求转给上游模型，属于这一类风险；#3176 明确不加熔断。Google 的书给出另一面：安全相关检查 fail-open 时，攻击者只需让检查不可用。

## 分歧与待核实

- “fail-closed”一词：安全语境中指拒绝请求；#3176 特意说明这里只拒绝缓存候选，请求照常转发上游。正文需要就地说明，避免误读。
- Envoy rate limit 的 `failure_mode_deny`：已核对 `api/envoy/extensions/filters/http/ratelimit/v3/rate_limit.proto`，字段为普通 proto3 `bool`（不是 `BoolValue`），默认 false，即限流服务无响应时放行。正文据此写入。
- NLI 的 premise/hypothesis 是陈述句训练的，用在问句对上的效果只有 #2751、#3075 的少量样例，正文不展开、不下一般性结论。
- 克隆完成后核对 main 的 commit 与上表一致或更新。

## 实测占位

正文默认不需要实测，数字取自 issue 与 PR（标明出处）。可选两项，均需用户另行确认：

1. 一次命中与一次 miss 的端到端延迟对比，说明 fail-closed 在校验器故障期间的延迟代价。需要一张能跑上游模型的 GPU（如 4090 上的 vLLM + 8B 模型）与 `b1b0e40` 版本的 router（带 native binding 构建）。
2. 校验器故障期间的上游请求量：可以不实测，按“命中率 × QPS”给演示数据并标明。

## 理解检查

题目供用户自测，参考答案附后。

1. **为什么调高相似度阈值解决不了“开启 / 关闭”的误命中？**
   参考答案：反义句只改动一两个词，相似度很高；同义改写换词更多，相似度反而低。#2691 中 8 对反义句为 0.823–0.915，同义改写 enable 2FA / turn on 2FA 只有 0.673。阈值 0.92 能挡住全部反义句，同义改写也同时 miss；两类的相似度区间交错，任何全局阈值都分不开。
2. **这里的 fail-closed 对用户的这一次请求意味着什么？与 Envoy ext_authz 的 fail-closed 有何不同？**
   参考答案：只丢弃未经校验的缓存候选，请求走普通 miss 路径转发上游模型，用户照样得到回答，只是更慢。ext_authz 的 fail-closed 拒绝请求本身（返回 403），用户得不到服务。
3. **NLI 服务完全不可用 10 分钟，两种策略在监控上分别是什么样子？**
   参考答案：fail-open 时命中率与延迟都正常，只多出 `cache_polarity_nli_skipped` 警告，意思相反的回答混在正常响应里，难以察觉。fail-closed 时命中率掉到接近 0，上游请求量与延迟上升，故障在仪表盘上直接可见。
4. **为什么校验器返回之后还要再检查一次 ctx，而不只看它返回的 err？**
   参考答案：取消后校验器可能返回 `context.Canceled` 这类错误，也可能已经算完、返回 nil。只看 err，前者会被记成校验器故障（误报警、计入 miss），后者会被当作正常结果继续走命中或拒绝的流程，向已经离开的调用方返回结果、并计入统计。再查一次 ctx，可以把两种情况都归为取消：不计 hit，也不计 miss。
5. **当前 main 上的词法层为什么不需要失败策略？代价是什么？**
   参考答案：它是进程内的纯函数，不依赖模型或远程服务，没有“不可用”这一状态。代价是覆盖面：只认英语否定词和 12 组登记反义词，相差超过 2 个 token、无否定词的改写，以及非英语的否定都判断不了；#4148 用 `negation_guard: not_applicable` 把这类命中标出来，但仍然返回缓存。

## 交付（2026-10-10）

未 commit、未 push。

新增：

- `content/semantic-cache-verification.md`：正文。
- `posts/semantic-cache-verification/index.html`：页面入口。
- `src/articles/semantic-cache-verification/semantic-cache-verification.tsx`：入口组件。
- `src/articles/semantic-cache-verification/semantic-cache-verification.blocks.tsx`：标题与图表占位映射。
- `src/articles/semantic-cache-verification/ScvFigures.tsx`：三张自绘图（`ScvSimilarity`、`ScvLookupPath`、`ScvFailurePolicy`）与实测占位 `ScvMeasurePlaceholder`。
- `docs/semantic-cache-verification-notes.md`：本笔记。

修改的共享文件：

- `vite.config.ts`：`build.rollupOptions.input` 增加一条 `semanticCacheVerification: 'posts/semantic-cache-verification/index.html'`。
- `src/posts.ts`：`posts` 数组开头增加 `slug: 'semantic-cache-verification'` 一项，`status: 'draft'`。
- `~/Documents/ChatGPT/reference-repos/sync.sh`（不在本仓库）：`REPOS` 中增加“# 请求路由”分组与 `semantic-router` 一行。

验证：在本会话启动的 dev server（端口 56077，已停止）上用 1280 × 900 视口打开页面，控制台无报错，三张图都正常显示；本篇文件的 `tsc --noEmit` 没有报错。

