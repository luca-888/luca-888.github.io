---
title: "Demis Hassabis 访谈笔记：DeepMind、世界模型、科学发现与 AGI"
description: "基于 Lex Fridman Podcast #475 完整 transcript，整理 Demis Hassabis 对 learnable patterns、Veo、AlphaEvolve、AI for science、AGI、Google、能源和未来编程的判断。"
pubDate: 2026-04-27
tags: ["AI", "DeepMind", "AGI", "World Models", "Podcast"]
---

这篇笔记基于 [Lex Fridman Podcast #475 – Demis Hassabis: Future of AI, Simulating Reality, Physics and Video Games](https://lexfridman.com/demis-hassabis-2) 及其 [官方 transcript](https://lexfridman.com/demis-hassabis-2-transcript)。我按完整 transcript 阅读，纯文本统计约 2.8 万英文词。

这期访谈最重要的线索是：DeepMind 不是只把 AI 当成聊天产品，而是把它当成理解自然、模拟现实、加速科学发现的工具。Demis 从 AlphaGo、AlphaFold、Veo、AlphaEvolve 一路讲到 AGI、能源、Google、教育和 consciousness。

## 1. Learnable Patterns：自然界是否可被学习

Demis 在 Nobel Prize lecture 里提出一个很大胆的 conjecture：自然界中能生成或存在的模式，可能可以被 classical learning algorithm 高效发现和建模。访谈开头围绕这个想法展开。

他的逻辑来自 AlphaGo 和 AlphaFold。围棋局面和蛋白质结构空间都极其巨大，暴力枚举不可能；但这些空间并非随机，它们有结构、有约束、有可学习的模式。学习系统如果能抓到这些低维结构，就能引导搜索。

这就是 DeepMind 世界观的核心：AI 不只是拟合数据，而是在高维复杂系统中找到可利用的结构。自然界之所以可学，是因为它经历了长期选择、稳定和演化，不是随机噪声。

## 2. P vs NP、信息与物理

Computation and P vs NP 章节很有理论色彩。Demis 把 P vs NP 看成不仅是计算机科学问题，也可能是物理问题。如果宇宙本质上是信息系统，那么“哪些问题可被有效求解”就和物理现实有关。

他关心的是是否存在一类“自然可学习系统”：它们可能在形式上看起来组合爆炸，但由于现实结构和演化约束，可以通过模型高效近似。蛋白质折叠、围棋搜索、视频物理、生命系统都可能属于这类。

这部分适合和 Stephen Wolfram 一起听。Wolfram 更偏计算宇宙和符号系统，Demis 更偏学习系统如何进入自然科学。

## 3. Veo：视频模型是否在学习现实

Veo 3 章节是这期最接近当前 AI 产品的一段。Demis 谈到视频模型对液体、材质、光照和物理运动的模拟能力。传统图形学和物理引擎需要工程师手写规则，而视频生成模型可能从大量视频中反向学习现实动态。

这不意味着模型已经理解物理定律，但说明它可能学到了某种低维 manifold：物体如何移动、液体如何变形、光如何反射、材料如何相互作用。

这段应该和 Sam Altman 的 Sora 部分一起听。Sora 和 Veo 的意义都不只是娱乐生成，而是视频模型正在成为世界模型的候选路径。

## 4. Video Games：游戏是 AGI 的训练场

Demis 的游戏背景贯穿整期。他早年写游戏和物理引擎，对互动世界、规则、策略和模拟有长期兴趣。游戏对 DeepMind 不只是爱好，而是构建智能系统的实验场。

AlphaGo、AlphaZero 和更广义的游戏环境说明：如果环境有规则、反馈和搜索空间，AI 可以通过自我对弈、规划和学习发现超越人类直觉的策略。游戏把 intelligence 的某些部分压缩到可测试、可迭代的系统里。

这部分和 David Silver 访谈高度相关。Demis 给出 DeepMind 的大方向，David Silver 解释强化学习和 self-play 的机制。

## 5. AlphaEvolve 与 AI Research

AlphaEvolve 章节展示了 DeepMind 对“AI 帮助 AI 研究”的看法。AI 不只是应用在外部任务上，也可以进入算法发现、代码优化、数学推导和科学假设生成。

这和 Demis 对 AGI 的定义有关：真正强的 AI 应该能帮助人类发现新知识，而不是只回答已知问题。AlphaEvolve、AlphaFold、AlphaGo 这些项目共同说明 DeepMind 重视可验证进步：系统要在一个困难领域产生外部可检验的成果。

## 6. 生命、模拟与科学发现

Simulating a biological organism 和 Origin of life 章节把话题推向生物学。Demis 关心 AI 能否模拟复杂生命系统，甚至帮助理解生命起源。

这里的挑战远超蛋白质结构。生命系统涉及多尺度动力学：分子、细胞、组织、器官、生态和演化。Demis 的判断是，AI 可能逐步帮助我们构建这些层次的模型，虽然完整模拟生命仍然非常困难。

这也是 AI for science 的重点：不是简单把模型用于搜索论文，而是让 AI 成为科学建模、实验设计和假设验证的一部分。

## 7. Path to AGI：不止语言模型

Demis 对 AGI 的路径更偏组合式。他承认 scaling laws 和大型模型重要，但也强调 planning、memory、reasoning、world models、tool use、multimodality 和科学发现能力。

他不像一些纯 LLM 路线那样把 token prediction 视为全部，也不像完全反 LLM 的观点那样否认 scaling。DeepMind 的路线更像把多种能力拼成通用问题求解系统：模型需要理解世界、提出计划、搜索方案、验证结果，并在科学和现实环境中产生新发现。

## 8. Scaling、Compute 与 Energy

Scaling laws、Compute、Future of energy 这三段是 DeepMind 对基础设施的看法。Demis 认为 compute 仍然关键，但也强调 AI 可能反过来帮助解决能源问题，例如材料、fusion、grid、气候和科学发现。

这和 Jensen Huang 的能源讨论形成呼应。Jensen 从数据中心需求出发，Demis 从科学突破和资源 abundance 出发。一个讲 AI 需要多少电，一个讲 AI 是否能帮助人类获得更多清洁能源。

## 9. Google 与 AGI 竞赛

Google and the race to AGI 是本期最长、也最现实的一段。Demis 谈到 Google 在过去一年从外界看似落后，到 Gemini、Veo、产品整合和 DeepMind 研究能力重新进入竞争中心。

Google 的优势是基础设施、研究人才、产品分发、数据、TPU 和长期科学项目。劣势是大公司组织复杂，产品整合速度不一定像创业公司。Demis 的任务是在 Google 体系内保持 DeepMind 的研究强度，同时把能力转成产品。

这部分应该和 Sam Altman、Sundar Pichai 一起听。Sam 讲 OpenAI 如何挑战巨头，Sundar 讲 Google 平台如何转型，Demis 讲 DeepMind 如何在 Google 内部推动 AGI 和 AI for science。

## 10. Competition、Talent 与 Future of Programming

Demis 对人才竞争的看法不只是薪酬。他认为真正吸引顶级人才的是使命：能否参与可能改变科学和文明的问题。AGI、AI for science、生命模拟和能源突破都属于这种任务。

Future of programming 章节里，他承认 AI 会深刻改变程序员工作。未来很多代码会由 AI 生成，人类的角色会转向定义问题、审查结果、架构系统和提出更高层目标。但他也不把编程消失简单化，因为软件工程包含大量判断、系统设计和责任。

这和 Cursor Team、Chris Lattner、DHH 的访谈可以连起来，构成 AI 编程线。

## 11. John von Neumann、p(doom) 与 Humanity

后半段进入更哲学的问题：John von Neumann、Maniac、p(doom)、humanity、consciousness、quantum computation、David Foster Wallace、education 和 research。

Demis 的 p(doom) 讨论比较克制。他承认风险真实存在，但更强调国际合作、科学精神和谨慎部署。他不把 AI 风险只看成模型失控，也把人类冲突、地缘政治和治理失败看成重要风险。

Consciousness 部分没有给出确定答案，但延续了他对信息、计算和物理的兴趣。David Foster Wallace 和教育讨论则显示 Demis 并不把智能只看成技术指标，他也关心注意力、意义、学习和人类如何使用新能力。

## 12. 这期的核心结论

Demis Hassabis 这期的价值在于把 AGI 放回科学发现和现实模拟中理解：

| 主题 | 关键结论 |
| --- | --- |
| Nature | 自然系统可能有可学习结构，AI 能在高维空间中引导搜索 |
| Video models | Sora/Veo 这类系统可能成为世界模型的重要路径 |
| Science | AlphaFold、AlphaEvolve 等项目说明 AI 可以产生可验证科学成果 |
| AGI | 通用智能需要 scaling、planning、memory、world models 和工具使用的组合 |
| Google | DeepMind 的优势在研究、基础设施和科学使命，但要持续转化为产品 |

如果只听一遍，建议重点听 Learnable patterns in nature、Veo 3、AlphaEvolve、Path to AGI、Google and the race to AGI、Future of programming 和 Education and research。这些段落共同说明：DeepMind 的 AGI 路线不是只做聊天机器人，而是构建能理解和推进科学的通用问题求解系统。
