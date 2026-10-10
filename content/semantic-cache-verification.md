Semantic cache 按 embedding 相似度复用 LLM 的回答：新问题与某条缓存问题的余弦相似度达到阈值，就直接返回那条缓存的回答，省去一次生成。相似度高不等于意思相同，开启和关闭同一个功能的两个问题，往往比同一件事的两种说法更接近，因此返回之前需要对相似度最高的那条缓存（称为候选）再判断一次，这一步称为命中校验，执行它的组件称为校验器。本文以 vLLM Semantic Router 为例，说明命中校验的做法，以及校验器本身出故障时候选应当怎样处理。在 Semantic Router 的 issue #2691 中，“How to turn on dark mode?”与“How to turn off dark mode?”的相似度为 0.915，高于仓库配置中的阈值 0.80；“enable 2FA”与同义的“turn on 2FA”只有 0.673。

## 一、相似度阈值的局限

Semantic cache 通常用 bi-encoder 把每个问题独立编码成一个向量，相似度反映的是两个问题在话题与用词上的接近程度。否定与反义只改动一两个词，话题和其余用词都不变，相似度因此很高；同义改写换掉的词更多，相似度反而更低。

::scv-similarity::

Issue 列出的 8 对反义句，相似度在 0.823 到 0.915 之间，同义改写为 0.673。阈值为 0.80 时，这 8 对全部误命中；阈值提高到 0.92 时误命中消失，同义改写也不再命中。两类问题的相似度区间交错，任何全局阈值都无法将两者分开。

vCache 论文的实验给出同样的结论：正确命中与错误命中的相似度分布大面积重叠，平均值分别为 0.84 与 0.85。vCache 的做法是放弃全局阈值，为每条缓存在线学习各自的阈值。Redis LangCache 只提供阈值（默认 0.85），文档说明 semantic caching 放弃了“命中即正确”的保证，建议从较严的阈值开始调整。

## 二、命中校验

命中校验发生在相似度达标之后、返回缓存之前，对候选再做一次判断。它只处理少数候选，可以使用比 bi-encoder 更慢、更准的方法。Semantic Router 用过两层校验：

- **词法层**：比较两句的 token 集合。两句相差不超过 2 个 token 时，若差异中只有一侧含否定词（not、no、never、without、cannot，`n't` 先展开为 not），或差异恰好是一组登记过的反义词（enable 与 disable、on 与 off 等 12 组），就拒绝候选。它是纯 Go 函数，耗时可以忽略，只覆盖英语中带明确否定词或登记反义词的情形。
- **NLI 层**：以缓存中的问题为 premise、新问题为 hypothesis，用 NLI 模型 ModernBERT-base-nli 计算 contradiction 概率，超过 0.5 即拒绝。它只对相似度最高的一个候选运行一次，并在缓存锁之外执行。

::scv-lookup-path::

Issue #2751 的测量中，NLI 层对 5 对反义句给出的 contradiction 概率在 0.967 到 0.997 之间，对同义句为 0.001；单线程 CPU 上每次约 70 ms。#2751 据此认为，一次命中省去的是数秒的生成，增加 70 ms 的校验是值得的。

两层各自漏掉的情形不同。PR #3075 的端到端测试中，NLI 层拒绝了“turn on the lights”与“add a contact”这对相似度 0.83 的无关问题；“Should I commit this change?”与“Should I not commit this change?”的相似度为 0.97，NLI 层没有拦住，这一对属于词法层的覆盖范围。

GPTCache 把校验做成可替换的 similarity evaluation：先取 top-k 候选，可以用 ONNX cross-encoder 为每对问题打分，也可以让一个 LLM 判断缓存的回答是否完全对应新问题。vCache 不使用独立的校验器，而是在本可命中的请求上按一定概率照样调用 LLM，用 LLM 的回答更新这条缓存的阈值，使错误率不超过用户给定的上限。

## 三、校验器故障时的两种策略

词法层是本地的确定性函数，不存在“不可用”的状态。NLI 层依赖一个模型，模型可能没有加载、推理出错或者超时。此时候选没有经过校验，有两种处理方式：

- **fail-open**：照常返回缓存的回答，记录一条警告。
- **fail-closed**：把候选当作 miss，请求转发给上游模型，即 router 之后实际生成回答的 LLM。

安全系统中的 fail-closed 指拒绝请求，例如 Envoy 的 ext_authz 在鉴权服务不可达时默认返回 403。这里的 fail-closed 只拒绝缓存候选，请求本身走普通的 miss 路径，用户照样得到回答。

两种策略付出的代价不同。fail-open 的代价是正确性：故障期间校验相当于关闭，意思相反的回答从缓存中以正常响应返回，用户与监控都看不出异常，只有日志里多出警告。fail-closed 的代价是延迟与上游负载：故障期间每个本可命中的请求都要等一次完整生成，命中率在监控上直接下降，故障容易被发现。

::scv-failure-policy::

Semantic Router 最初的 NLI 层（PR #3075）选择 fail-open，代码注释给出的理由是不让模型的偶发故障变成整个缓存的故障。作者提交的修复 PR #3162 改为 fail-closed。维护者在首轮 review 中没有讨论实现，而是指出这是在改变一项已合并的行为约定，要求先单独开 issue，由维护者作出决定。Issue #3176 正式接受之后，PR 才合并。#3176 记录的决定限定了范围：只对未校验的候选 fail-closed，请求照常转发；不加重试与熔断，也不提供可配置的策略。

其他系统的默认值取决于检查保护的对象。Kubernetes 的 admission webhook 默认 `failurePolicy: Fail`，Envoy 的 ext_authz 默认拒绝，两者保护的是访问控制；Envoy 的限流过滤器在限流服务无响应时默认放行（`failure_mode_deny` 默认为 false），限流失效的后果只是多放过一些请求。Kubernetes 文档还区分了“webhook 明确拒绝”与“调用 webhook 出错”，failure policy 只作用于后者。命中校验同样要分开这两种结局：contradiction 超过阈值是校验结果，校验器出错是故障，两者都导致 miss，记录与告警则应当分开。GPTCache 的 ONNX 评估器在出错时返回 0，结果上也是 fail-closed，但故障与“判为不相似”无法区分。

fail-closed 也有反对意见。AWS 的 Jacob Gabrielson 以 Amazon 2001 年的一次故障为例反对回退路径：缓存失效后，web 服务器改为直接查询数据库，数据库被压垮，网站随之停止服务。fail-closed 在校验器故障时同样把本可命中的流量全部转给上游模型，上游容量若是按扣除命中之后的流量规划的，就可能过载。#3176 不加熔断，前提是上游能承受没有缓存时的全部流量；不满足这一前提时，需要在上游一侧限流，或为校验器单独规划容量与告警。另一方面，Google 的《Building Secure and Reliable Systems》指出，安全相关的检查若 fail-open，攻击者只需让检查不可用就能绕过它。校验器在高负载下最容易超时，fail-open 会让正确性随校验器的可用性一起下降。

::scv-measure-placeholder::

## 四、取消与候选相似度

#3162 合并时，NLI 层对一个候选的处理如下：

```go
verdict := c.verifyPolarityNLI(ctx, model, bestEntry.Query, query)
if err := ctxErr(ctx); err != nil {
	return LookupResult{}, true, err // 请求取消：不计 hit，也不计 miss
}
if verdict.Skipped { // 校验器缺失或出错
	c.recordPolaritySkippedMiss(start, bestSimilarity)
	return LookupResult{Similarity: bestSimilarity}, true, nil
}
if !verdict.Reject {
	return LookupResult{}, false, nil // 通过，由调用方按命中返回
}
c.recordPolarityReject(start, model, query, bestEntry.Query, bestSimilarity, threshold, verdict)
return LookupResult{Similarity: bestSimilarity}, true, nil
```

一次查询有五种结局：

| 结局 | 返回 | 计数 | 报告的相似度 | 记录 |
| --- | --- | --- | --- | --- |
| 通过 | 缓存的回答 | hit | 候选的相似度 | `cache_hit` |
| 低于阈值 | 转发上游 | miss | 最佳候选的相似度 | `cache_miss` |
| contradiction 拒绝 | 转发上游 | miss | 候选的相似度 | `cache_negation_reject` |
| 校验器故障 | 转发上游 | miss | 候选的相似度 | `cache_polarity_nli_skipped` |
| 请求取消 | 无 | 均不计 | 无 | `lookup_failed` |

请求取消指客户端断开或请求超时，使请求的 context 结束。此时已经没有调用方在等待回答，转发上游没有意义；记为 miss 会拉低命中率，记为校验器故障则会对一个正常的模型触发告警。因此取消单独返回 context 的错误，不进入任何缓存统计。取消检查放在调用校验器之后：校验器可能因为取消而返回错误，也可能在取消后照常算完、返回 nil。只看校验器的返回值，前一种会被记成故障，后一种会被当作正常结果继续处理。

miss 时保留候选的相似度，用来区分 miss 的原因。相似度为 0.3 的 miss 说明缓存里没有相近的问题，0.95 的 miss 说明有相近的候选，但被校验拒绝或未能校验。Semantic Router 通过调试响应头 `x-vsr-cache-similarity` 报告这个值。原来的端到端测试断言 miss 的相似度小于 1.0，理由是相似度为 1.0 的候选必定命中；改为 fail-closed 之后，校验器故障也能让相似度为 1.0 的候选成为 miss，#3162 因此把断言区间改为 [0, 1]。未校验的候选也不更新访问时间与命中次数，不影响缓存的淘汰顺序。

## 五、现状

2026 年 10 月，Semantic Router 把所有模型从进程内的 native binding（candle、ONNX Runtime 等）迁移到独立的 model runtime 服务（PR #4512）。新 runtime 不提供这个 NLI 模型，NLI 层随之退役：`polarity_guard` 配置在启动时被拒绝，由迁移工具删除。退役与失败策略无关，#3162 加入的 fail-closed 处理随 NLI 层一起删除。

NLI 层退役后，Semantic Router 只保留词法层，PR #3944 已把它扩展到全部六种缓存后端。词法层在扫描候选时过滤，被拒的候选不会终止查找，次优候选只要达到阈值仍可命中；NLI 层只检查相似度最高的一个候选，拒绝即 miss。词法层没有故障状态，因此不需要失败策略，但覆盖面有限。在 issue #4146 的复现中，德语 nicht、中文“不”构成的否定句与原句的相似度都超过 0.97，词法层无法判断。PR #4148 为每个命中加上 `negation_guard` 标记：两句用词相同或只差否定词时为 `checked`，其余为 `not_applicable`。这个标记只标出词法层没有判断过的命中，不改变命中结果。

今后只要引入依赖模型或远程服务的校验器，就需要重新决定它出故障时如何处理候选。

## 参考

- vLLM Semantic Router：
  - [#2691](https://github.com/vllm-project/semantic-router/issues/2691) Prevent semantic-cache false hits on negated or antonym queries；[#2751](https://github.com/vllm-project/semantic-router/issues/2751) Optional NLI polarity tier for in-memory semantic cache negation guard。
  - [#3075](https://github.com/vllm-project/semantic-router/pull/3075) Optional NLI polarity tier for the in-memory semantic cache。
  - [#3176](https://github.com/vllm-project/semantic-router/issues/3176) Define the runtime failure policy for semantic-cache NLI verification；[#3162](https://github.com/vllm-project/semantic-router/pull/3162) Treat semantic-cache NLI verifier failures as cache misses，合并提交 [`b1b0e40`](https://github.com/vllm-project/semantic-router/tree/b1b0e40fc620ed5cfd36e0df15f78227b94fdbf6)：`src/semantic-router/pkg/cache/inmemory_cache_polarity.go`、`pkg/extproc/req_filter_cache.go`。
  - [#3944](https://github.com/vllm-project/semantic-router/pull/3944) Resolve verified v0.4 release regressions；[#4148](https://github.com/vllm-project/semantic-router/pull/4148) Flag semantic hits the English negation guard can't judge；[#4512](https://github.com/vllm-project/semantic-router/pull/4512) Built-in model runtime。
  - main [`a9d8539`](https://github.com/vllm-project/semantic-router/tree/a9d8539b95fe4ef1624ab7f68d116779da517bf0)：`src/semantic-router/pkg/cache/polarity.go`、`website/docs/model-runtime/migrate.md`。
- Luis Gaspar Schroeder, et al. [vCache: Verified Semantic Prompt Caching](https://arxiv.org/abs/2502.03771). ICLR 2026.
- Zilliz. [GPTCache](https://github.com/zilliztech/GPTCache/tree/a74ac65)：`gptcache/similarity_evaluation/onnx.py`、`gptcache/processor/post.py`。
- Redis. [LangCache concepts](https://redis.io/docs/latest/develop/ai/context-engine/langcache/concepts/).
- Envoy. [ext_authz](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_authz/v3/ext_authz.proto) · [rate limit](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ratelimit/v3/rate_limit.proto).
- Kubernetes. [Dynamic Admission Control: Failure policy](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/).
- Jacob Gabrielson. [Avoiding fallback in distributed systems](https://builder.aws.com/content/3EuS9Sakq7L3VLQIF3qzfMfke1Y/avoiding-fallback-in-distributed-systems). Amazon Builders' Library.
- Heather Adkins, et al. [Building Secure and Reliable Systems](https://google.github.io/building-secure-and-reliable-systems/raw/ch08.html), Chapter 8: Design for Resilience. O'Reilly, 2020.
