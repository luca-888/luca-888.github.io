---
title: "Marc Raibert 访谈笔记：Boston Dynamics、腿足机器人与运动智能"
description: "基于 Lex Fridman Podcast #412 完整 transcript，整理 Marc Raibert 对 Boston Dynamics、BigDog、Spot、Atlas、液压驱动、运动智能和机器人未来的讨论。"
pubDate: 2026-04-28
tags: ["Robotics", "Boston Dynamics", "AI", "Engineering", "Podcast"]
---

这篇笔记基于 [Lex Fridman Podcast #412 – Marc Raibert: Boston Dynamics and the Future of Robotics](https://lexfridman.com/marc-raibert) 及其 [官方 transcript](https://lexfridman.com/marc-raibert-transcript)。我按完整 transcript 阅读，纯文本统计约 1.9 万英文词。

Marc Raibert 这期是理解 Boston Dynamics 精神的核心材料。它不是从大模型、聊天机器人或人形机器人热潮切入，而是从早期机器人、腿足运动、BigDog、Spot、Atlas、液压驱动、Leg Lab 和 athletic intelligence 讲起。它提醒我们：机器人首先是身体，身体首先要会运动。

## 1. Early Robots：机器人从动态平衡开始

Early robots 章节说明 Raibert 早期关心的问题很朴素：机器如何保持平衡、跳跃、落地、恢复动作，并在不完美环境里继续运动。

这和很多人对机器人的想象不同。大众经常先关心机器人是否像人、是否会说话、是否有 AGI；Raibert 的路线先问一个更基础的问题：它能不能像动物一样在真实世界里稳定运动。

这条线贯穿 Boston Dynamics。它的机器人之所以让人印象深，不只是外形，而是运动里有一种动态控制能力：被推、踩到不平地面、做快速动作时，系统还能恢复。

## 2. Legged Robots：腿足机器人为什么难

Legged robots 是这期的技术底层。轮式机器人在平地上效率很高，但真实世界大量空间不是为轮子设计的：楼梯、碎石、草地、门槛、坡道、障碍物都要求更复杂的身体控制。

腿足机器人难在它必须持续处理不稳定。每一步都涉及接触、摩擦、冲击、姿态、力矩和预测。机器人不是在静态环境里摆姿势，而是在每个瞬间决定身体如何把力传给世界。

这也是为什么腿足机器人进展看起来慢。它不是纯软件问题，机械设计、传感器、执行器、控制算法和测试环境都要一起进步。

## 3. Boston Dynamics：Demo 背后的工程文化

Boston Dynamics 的视频常常被当成炫技 demo，但 Raibert 讲得很清楚，视频背后是长期工程文化。一个机器人能做后空翻、跑跳或跳舞，意味着控制、机械、电源、传感器和软件都通过了高强度协调。

Videos 和 dancing robots 章节也很有意思。Raibert 并不把视频只看成营销，它们也是团队设定目标、压迫系统边界、展示能力和吸引人才的方式。

这点对工程团队很有启发：好的 demo 不是虚假包装，而是把复杂系统压缩成一个可观察、可挑战、可传播的目标。

## 4. BigDog 与液压驱动：力量、噪声和真实世界

BigDog 是 Boston Dynamics 历史中最重要的节点之一。它展示了腿足机器人可以在崎岖地形上承载负载、保持平衡，并对外界扰动作出反应。

Hydraulic actuation 章节解释了为什么早期系统偏向液压：液压能提供高功率密度和强力输出，适合粗糙地形和动态动作。但代价也明显：噪声、维护、复杂性和消费级产品化难度。

这能帮助理解 Spot 和 Atlas 后续路线的变化。机器人商业化不仅要“能动”，还要可靠、安静、可维护、成本合理，并能在客户环境中长期运行。

## 5. Natural Movement：自然动作不是装饰

Natural movement 是 Raibert 机器人观的核心。自然动作不是为了让机器人看起来可爱，而是因为自然界里的运动往往代表高效、稳定和可恢复的控制策略。

动物运动不是每一步都精确规划到毫米，而是在身体结构、控制反馈和环境互动之间形成稳定闭环。Raibert 对这种“身体智能”很敏感。

这也是 athletic intelligence 的前置概念。机器人如果要进入真实世界，不能只会执行离散指令，还要能在运动中持续调整。

## 6. Leg Lab 到 AI Institute：从控制到智能

Leg Lab 和 AI Institute 章节把 Raibert 的路线从传统机器人控制推向更广义智能。AI Institute 的目标不是只训练语言模型，而是研究能在物理世界中行动、学习和适应的智能系统。

这里的 AI 和大模型访谈里的 AI 不太一样。对机器人来说，智能要体现在身体上：看见环境、理解可行动性、选择动作、保持平衡、恢复失败，并在现实约束下完成任务。

这期适合和 Yann LeCun 一起听。Yann 强调世界模型和非语言学习，Raibert 则从机器人身体角度说明为什么世界模型不能只存在于文本里。

## 7. Athletic Intelligence：身体能力也是智能

Athletic intelligence 是这期最重要的概念。Raibert 把运动能力视为智能的一部分，而不是低级控制问题。一个优秀运动员不是只靠肌肉，而是具备感知、预测、平衡、节奏、恢复和策略。

机器人也一样。真正有用的机器人必须知道什么时候踩哪里、如何分配力量、如何从错误中恢复、如何在未知地形保持任务目标。

这能纠正一个常见误解：给机器人接上更强 LLM，并不会自动获得身体智能。语言可以给目标，身体必须自己解决运动和接触问题。

## 8. Building a Team 与 Engineering

Building a team、Engineering 和 Hiring 章节透露了 Boston Dynamics 的团队标准。Raibert 需要的是能把物理、机械、控制、软件和实验结合起来的人，而不是只会在单一层面优化的人。

机器人团队的难点在于反馈周期昂贵。软件 bug 可以快速重跑，机器人实验可能损坏硬件、耗费时间，并受环境影响。因此团队必须既有创造力，也有严谨测试和工程纪律。

这和 AI 软件公司很不同。机器人创业不能只靠 demo 融资，最终要面对材料、供应链、现场维护和安全责任。

## 9. Optimus 与机器人未来

Optimus robot 和 future of robotics 章节把话题拉到人形机器人。Raibert 对未来机器人并不悲观，但他的判断更工程化：人形机器人很有吸引力，因为世界为人类身体设计，但真正有用还要跨过稳定性、成本、可靠性、操控和任务泛化。

这和 Elon Musk 的 Optimus 叙事形成对照。Elon 更强调规模、制造和未来劳动力；Raibert 更关注运动能力、控制系统和机器人身体本身。

两者可以一起听。一个回答“为什么大规模机器人值得做”，一个回答“身体智能到底难在哪里”。

## 10. 这期的核心结论

| 主题 | 关键结论 |
| --- | --- |
| 腿足机器人 | 难点在动态平衡、接触、执行器、控制和真实地形 |
| Boston Dynamics | 视频 demo 背后是长期工程、测试和团队文化 |
| BigDog | 展示高功率腿足运动能力，也暴露噪声和产品化约束 |
| Athletic Intelligence | 运动能力本身就是智能，不是语言模型的附属功能 |
| 机器人未来 | 人形机器人有潜力，但必须先解决身体、可靠性和成本 |

如果只听一遍，建议重点听 Legged robots、Boston Dynamics、BigDog、Hydraulic actuation、Natural movement、AI Institute、Athletic intelligence、Engineering、Optimus robot 和 Future of robotics。这期最适合作为机器人线的基础课：在讨论机器人会不会像人之前，先理解机器如何拥有身体。
