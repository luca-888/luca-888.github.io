---
title: "Lex Fridman AI、自动驾驶与科技节目听单"
description: "按 AI 产业、算力、模型公司、AI 编程、自动驾驶、机器人和底层工程整理 30 期高价值 Lex Fridman Podcast。"
pubDate: 2026-04-27
tags: ["AI", "Podcast", "Autonomous Driving", "Robotics", "Programming"]
---

这是一份面向 AI、自动驾驶和科技产业的 Lex Fridman Podcast 听单。排序不是按嘉宾名气，而是按信息密度和覆盖面：先抓 AI 产业、算力、模型公司和编程范式，再补自动驾驶、机器人、芯片、强化学习和互联网平台。

如果时间有限，最建议先听这 10 期：Jensen Huang、Sam Altman、Dario Amodei、Demis Hassabis、DeepSeek/Dylan Patel & Nathan Lambert、Andrej Karpathy、Cursor Team、Chris Lattner、George Hotz、Jim Keller。它们能最快覆盖当下 AI 的几个主轴：算力供给、模型公司战略、开源与闭源竞争、AI 编程、自动驾驶和底层工程。

## 1. Jensen Huang

[官方 Episode：#494 – Jensen Huang: NVIDIA – The $4 Trillion Company & the AI Revolution](https://lexfridman.com/jensen-huang)

这期适合放在第一位，因为它把 AI 产业最底层的约束讲清楚：rack-scale engineering、供应链、HBM、功耗、数据中心和 TSMC。Jensen Huang 讨论的不只是 GPU，而是从芯片、机柜、网络到软件生态的一整套 co-design。

这期的价值在于理解 NVIDIA 为什么不只是卖芯片。AI scaling laws 最终会落到供电、内存、制造能力、系统集成和客户部署速度上；这些约束决定了模型公司能多快扩张，也决定了 AI 数据中心为什么变成国家级基础设施。

## 2. Sam Altman

[官方 Episode：#419 – Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power & AGI](https://lexfridman.com/sam-altman-2)

这期是理解 OpenAI 战略和治理问题的核心材料。话题覆盖 OpenAI board saga、Ilya、Elon Musk lawsuit、Sora、GPT-5、memory、privacy、compute 和 AGI，是模型公司从研究实验室走向平台公司的一个切面。

它最值得听的不是某个产品预测，而是 Sam Altman 对规模、资本、组织和权力的处理方式。OpenAI 的挑战已经不只是训练更强模型，还包括算力融资、产品节奏、治理信任和社会影响。

## 3. Dario Amodei

[官方 Episode：#452 – Dario Amodei: Anthropic CEO on Claude, AGI & the Future of AI & Humanity](https://lexfridman.com/dario-amodei)

这期适合和 Sam Altman 对照听。Dario Amodei 的重点更偏向 scaling laws、安全等级、post-training、Constitutional AI、Claude 的产品性格，以及 Anthropic 对监管和 AGI 时间线的判断。

后半段还加入 Amanda Askell 和 Chris Olah，分别展开 Claude 的行为塑造与 mechanistic interpretability。它能帮助理解 Anthropic 为什么把“模型能力”和“模型可控性”绑定成同一件事。

## 4. Demis Hassabis

[官方 Episode：#475 – Demis Hassabis: Future of AI, Simulating Reality, Physics and Video Games](https://lexfridman.com/demis-hassabis-2)

Demis Hassabis 这一期更像 DeepMind 世界观的浓缩版：从自然界可学习模式、Veo、AlphaEvolve、生命起源、AGI 路径，到模拟现实和科学发现。它不是单纯聊 Gemini，而是在讨论 AI 如何成为探索自然规律的工具。

如果说 OpenAI 的叙事更接近产品和通用智能，DeepMind 的叙事更接近科学、搜索、强化学习和世界模型。这期适合用来理解“AI for science”为什么会成为下一阶段的重要方向。

## 5. Dylan Patel & Nathan Lambert

[官方 Episode：#459 – DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate, and AI Megaclusters](https://lexfridman.com/deepseek-dylan-patel-nathan-lambert)

这是一期高密度产业分析。Dylan Patel 从半导体、GPU 集群、出口管制、TSMC、NVIDIA 和 megacluster 角度解释 AI 竞争；Nathan Lambert 则补上模型训练、RL、开源和 DeepSeek 的技术侧理解。

这期很适合建立“模型能力不是孤立出现的”这个框架。DeepSeek 的成本争议、训练数据、GPU 使用、审查、开源策略和中美竞争，都要放在算力、供应链和人才流动里看。

## 6. Andrej Karpathy

[官方 Episode：#333 – Andrej Karpathy: Tesla AI, Self-Driving, Optimus, Aliens, and AGI](https://lexfridman.com/andrej-karpathy)

Andrej Karpathy 这期横跨神经网络、Transformer、语言模型、Software 2.0、Tesla Vision、Data Engine、自动驾驶和 Optimus。它适合用来理解深度学习工程师如何把模型能力落到真实系统里。

自动驾驶部分尤其重要。Karpathy 把 perception、数据闭环、人类标注、视觉方案和系统迭代放在一起讲，能看出 Tesla AI 的核心并不是某个单点模型，而是持续收集、筛选、训练和部署数据的工程机器。

## 7. Cursor Team

[官方 Episode：#447 – Cursor Team: Future of Programming with AI](https://lexfridman.com/cursor-team)

这期是理解 AI 编程工具最直接的一期。Cursor 团队讨论 code editor、Copilot、Cursor Tab、diff、上下文、prompt engineering、agent、后台运行代码、debugging 和模型选择。

它的重点不是“AI 会不会取代程序员”，而是编程交互界面正在改变：补全、编辑、生成、运行和调试逐渐合并成一个闭环。未来 IDE 的竞争，很大一部分会变成上下文组织能力和执行反馈能力的竞争。

## 8. Chris Lattner

[官方 Episode：#381 – Chris Lattner: Future of Programming and AI](https://lexfridman.com/chris-lattner-3)

Chris Lattner 从 LLVM、Swift、MLIR、TPU、Mojo 和 Modular 的经验出发，讨论编程语言、编译器、类型系统、autotuning、分布式部署和 AI runtime。它适合理解 AI 工程为什么离不开编译器和系统软件。

这期和 Cursor Team 是互补关系。Cursor 代表开发界面变革，Lattner 代表底层执行栈变革：当 AI 工作负载越来越复杂，语言、编译器、runtime 和硬件之间的边界会重新被设计。

## 9. Yann LeCun

[官方 Episode：#416 – Yann Lecun: Meta AI, Open Source, Limits of LLMs, AGI & the Future of AI](https://lexfridman.com/yann-lecun-3)

Yann LeCun 这期适合作为 LLM 热潮中的反方校准。他重点谈 LLM 的局限、JEPA、video prediction、hierarchical planning、reasoning、hallucination、open source 和 Llama。

这期最有价值的是提供另一种 AGI 路线判断：不是把 autoregressive LLM 视为终点，而是强调世界模型、规划、抽象表征和非语言学习。无论是否同意他的判断，这期都能避免只从 scaling 角度理解 AI。

## 10. George Hotz

[官方 Episode：#387 – George Hotz: Tiny Corp, Twitter, AI Safety, Self-Driving, GPT, AGI & God](https://lexfridman.com/george-hotz-3)

George Hotz 这期有很强的 hacker 视角。话题包括 tiny corp、NVIDIA vs AMD、tinybox、self-driving、programming、AI safety、Twitter、prompt engineering 和游戏。

它适合和 Jensen Huang、DeepSeek 那几期放在一起听：一个讲产业级供应链，一个讲地缘和集群，一个讲从个人工程师与开源硬件角度挑战既有栈。Hotz 的价值在于把“能不能自己做出来”这个问题问得很直接。

## 11. Aravind Srinivas

[官方 Episode：#434 – Aravind Srinivas: Perplexity CEO on Future of AI, Search & the Internet](https://lexfridman.com/aravind-srinivas)

Aravind Srinivas 这期围绕 Perplexity、Google Search、RAG、搜索入口、startup 策略和未来互联网展开。它适合理解 AI 原生搜索为什么不是简单的“搜索框加聊天框”。

这期的核心问题是入口权力会不会变化。传统搜索依赖索引、排序和广告系统；AI 搜索要处理答案生成、引用可信度、实时信息、用户意图和商业模式之间的张力。

## 12. Elon Musk & Neuralink Team

[官方 Episode：#438 – Elon Musk: Neuralink and the Future of Humanity](https://lexfridman.com/elon-musk-and-neuralink-team)

这期前半段是 Elon Musk 对 Neuralink、xAI、Optimus、问题求解和人机融合的判断，后半段由 Neuralink 团队解释脑机接口的工程细节。它覆盖 Telepathy、神经接口历史、生物物理、安全、升级和未来能力。

这期适合从“AI 之外的人机接口”角度补齐技术图景。AI 提升机器能力，BCI 则尝试提升人和机器之间的带宽，两者最终会在交互方式上相遇。

## 13. Marc Raibert

[官方 Episode：#412 – Marc Raibert: Boston Dynamics and the Future of Robotics](https://lexfridman.com/marc-raibert)

Marc Raibert 这期是理解 Boston Dynamics 精神的好入口。内容从早期机器人、legged robots、BigDog、液压驱动、自然运动、Leg Lab 到 AI Institute 和 athletic intelligence。

它强调机器人不是“把模型塞进身体”这么简单。稳定行走、动态平衡、机械设计、控制、团队文化和真实世界测试缺一不可；这也是机器人进展看起来慢，但每一步都很难的原因。

## 14. Jim Keller

[官方 Episode：#162 – Jim Keller: The Future of Computing, AI, Life, and Consciousness](https://lexfridman.com/jim-keller-2)

Jim Keller 这期把处理器设计、RISC/CISC、模块化设计、Moore's Law、深度学习硬件、GPU、Tesla Autopilot、Dojo 和神经网络加速放在同一条线上。它适合用来理解 AI 硬件不是单独的芯片问题，而是计算抽象问题。

Keller 的表达方式很适合工程师：先抓第一性原理，再看哪些复杂性是历史包袱，哪些复杂性是真约束。这期能帮助把“算力”从一个口号还原成架构、带宽、缓存、互联和编程模型。

## 15. John Carmack

[官方 Episode：#309 – John Carmack: Doom, Quake, VR, AGI, Programming, Video Games, and Rockets](https://lexfridman.com/john-carmack)

John Carmack 这期很长，但信息密度高。它从编程语言、现代编程、工作方式、id Software、Doom、Quake、VR、火箭、核能一路谈到 AGI。

这期最值得听的是工程判断。Carmack 一直强调可测量进展、直接面对系统复杂性和避免空泛抽象。对于做 AI 工具、系统或产品的人，这种工程审美比具体技术栈更有长期价值。

## 16. Ilya Sutskever

[官方 Episode：#94 – Ilya Sutskever: Deep Learning](https://lexfridman.com/ilya-sutskever)

Ilya Sutskever 这期是早期深度学习路线的经典材料。话题包括 AlexNet、cost functions、RNN、deep double descent、backpropagation、reasoning、long-term memory、language models、GPT-2 和 staged release。

它适合回看 OpenAI 早期研究气质：对深度学习可扩展性的信念非常强，同时又已经开始意识到模型发布、安全和社会影响的问题。放在今天听，很多判断仍然能解释后来的发展路径。

## 17. Andrew Ng

[官方 Episode：#73 – Andrew Ng: Deep Learning, Education, and Real-World AI](https://lexfridman.com/andrew-ng)

Andrew Ng 这期更偏教育和落地。内容包括在线教育早期、Google Brain、deeplearning.ai、unsupervised learning、AI career、是否读 PhD、AI Fund、Landing AI 和 AGI。

这期适合想进入 AI 或做企业 AI 落地的人。Ng 的视角不是只追最前沿模型，而是关注如何学习、如何组建项目、如何把 AI 变成真实业务里的生产力。

## 18. Francois Chollet

[官方 Episode：#120 – Francois Chollet: Measures of Intelligence](https://lexfridman.com/francois-chollet-2)

Francois Chollet 这期围绕智能度量展开，涵盖语言、智能定义、GPT-3、semantic web、自动驾驶、ARC Challenge、generalization、Turing Test 和 Hutter Prize。

它适合作为 AGI 讨论的概念校准。Chollet 关心的是泛化效率和抽象能力，而不是只看模型在已有分布上的表现。听完这期，再看大模型 benchmark，会更容易区分“会做题”和“真的会迁移”。

## 19. David Silver

[官方 Episode：#86 – David Silver: AlphaGo, AlphaZero, and Deep Reinforcement Learning](https://lexfridman.com/david-silver)

David Silver 这期是强化学习路线的代表。内容包括 AlphaGo、围棋规则、RL 个人历程、self-play、Lee Sedol、Kasparov、AlphaZero、reward functions 和创造性。

它适合理解为什么 self-play 曾经被视为通往通用智能的重要路径。相比 LLM 的大规模模仿学习，AlphaGo/AlphaZero 展示的是通过搜索、奖励和自我对弈产生超越人类经验的策略。

## 20. Chris Urmson

[官方 Episode：Chris Urmson: Self-Driving Cars at Aurora, Google, CMU, and DARPA](https://lexfridman.com/chris-urmson)

Chris Urmson 这期是自动驾驶历史线的重要材料。他经历了 CMU、DARPA Grand Challenge、Google Self-Driving Car 和 Aurora，能把自动驾驶从学术挑战、工程竞赛讲到商业化公司。

这期适合和 Karpathy、Elon Musk、Kyle Vogt 对照听。Urmson 的路线更偏系统安全、工程成熟度和长期部署，能补上端到端视觉叙事之外的自动驾驶复杂性。

## 21. Elon Musk

[官方 Episode：Elon Musk: Tesla Autopilot](https://lexfridman.com/elon-musk)

这期是 Tesla Autopilot 早期叙事的代表。它聚焦 Tesla、Autopilot、车队数据、神经网络和自动驾驶目标，是理解 Tesla 为什么坚持大规模真实道路数据路线的起点。

放在今天听，这期的价值不是验证某个时间表，而是观察 Tesla 自动驾驶战略的早期假设：摄像头、数据闭环、车队规模、软件更新和垂直整合。

## 22. Kyle Vogt

[官方 Episode：Kyle Vogt: Cruise Automation](https://lexfridman.com/kyle-vogt)

Kyle Vogt 这期代表 Cruise 和 robotaxi 路线。它从 Cruise、Twitch、自动驾驶公司建设和城市无人车挑战出发，展示另一种不同于 Tesla 的商业化路径。

如果说 Tesla 更强调量产车队和端到端数据，Cruise 路线更强调限定区域、专用运营和 robotaxi 服务。两者对比，能看清自动驾驶不是单一技术问题，而是产品形态、监管和运营系统的组合。

## 23. Mark Zuckerberg

[官方 Episode：#383 – Mark Zuckerberg: Future of AI at Meta, Facebook, Instagram, and WhatsApp](https://lexfridman.com/mark-zuckerberg-2)

Mark Zuckerberg 这期适合理解 Meta 的 AI 战略。内容包括开源运动、下一代模型、Meta AI、bots、内容审查、社交产品、Quest、Apple Vision Pro、AI 风险和 embodied AGI。

它的核心价值在于解释 Meta 为什么愿意推动开源模型。对 Meta 来说，AI 既是基础设施，也是社交、广告、设备和创作者生态的入口；开源则是改变行业权力结构的手段。

## 24. Sundar Pichai

[官方 Episode：#471 – Sundar Pichai: CEO of Google and Alphabet](https://lexfridman.com/sundar-pichai)

Sundar Pichai 这期覆盖 Google 视角的 AI 平台转型：Veo、scaling laws、AGI/ASI、AI mode vs Google Search、Chrome、programming、Android、Google Beam、XR Glasses。

这期适合和 OpenAI、Perplexity、Meta 对照。Google 的问题不是没有技术，而是如何把 AI 嵌入搜索、浏览器、移动系统、生产力工具和硬件，同时不破坏原有商业系统。

## 25. Stephen Wolfram

[官方 Episode：#376 – Stephen Wolfram: ChatGPT and the Nature of Truth, Reality & Computation](https://lexfridman.com/stephen-wolfram-4)

Stephen Wolfram 这期从 WolframAlpha、ChatGPT、计算、现实本质、人类认知、AI 风险、truth、教育、consciousness 和 entropy 出发，把大模型放进更长的计算理论脉络里。

它适合作为技术听单里的哲学和理论补充。Wolfram 的强项是把语言模型、符号系统、计算宇宙和知识表达放到一起讨论，帮助理解 AI 不只是产品变化，也是知识组织方式的变化。

## 补充高价值候选

### DHH

[官方 Episode：#474 – DHH: Future of Programming, AI, Ruby on Rails, Productivity & Parenting](https://lexfridman.com/dhh-david-heinemeier-hansson)

DHH 这期适合补软件工程文化视角。话题包括 Ruby、Rails、dynamic typing、beautiful code、small teams、meetings、离开云、拥有服务器、AI 编程和 vibe coding。

它的价值在于提醒我们：AI 工具之外，软件生产力仍然受团队规模、框架审美、部署方式和组织习惯影响。对长期写代码的人，这期比单纯的工具讨论更接近日常决策。

### ThePrimeagen

[官方 Episode：#461 – ThePrimeagen: Programming, AI, ADHD, Productivity, Addiction, and God](https://lexfridman.com/theprimeagen)

ThePrimeagen 这期更像程序员工作流和人生经验的混合体。技术部分覆盖 debugging、Netflix/Twitch/YouTube infrastructure、语言学习、Python、Bash、FFmpeg、performance、Rust 和项目经验。

它适合补“每天写代码的人如何保持手感”。和 Cursor、Lattner、DHH 一起听，可以得到从 AI IDE、编译器、框架哲学到个人工作方式的完整视角。

### Marc Andreessen

[官方 Episode：#386 – Marc Andreessen: Future of the Internet, Technology, and AI](https://lexfridman.com/marc-andreessen)

Marc Andreessen 这期更偏科技投资和互联网历史。内容包括搜索、LLM training、truth、journalism、AI startups、browser、Netscape、AI 风险、核能、经济、中国和技术演化。

它适合从资本和产业叙事角度理解 AI。Andreessen 的观点鲜明，不一定要全盘接受，但能帮助识别硅谷乐观主义、平台迁移和技术扩散背后的投资逻辑。

### Robert Playter

[官方 Episode：#374 – Robert Playter: Boston Dynamics CEO on Humanoid and Legged Robotics](https://lexfridman.com/robert-playter)

Robert Playter 这期和 Marc Raibert 互补，更偏 Boston Dynamics 的产品和公司管理视角。话题包括 Atlas、DARPA Robotics Challenge、BigDog、Spot、Stretch、机器人入户、Optimus 和 ChatGPT。

这期适合理解机器人从 demo 到产品的难点：硬件可靠性、成本、客户场景、团队管理和公众恐惧。它能把“机器人很酷”拉回“机器人如何真的进入生产环境”。

### Jim Keller

[官方 Episode：Jim Keller: Moore's Law, Microprocessors, Abstractions, and First Principles](https://lexfridman.com/jim-keller)

这期是 Jim Keller 早期访谈，更集中在 Moore's Law、微处理器、抽象和第一性原理。它适合作为 #162 的前置或补充材料。

如果只听一期 Jim Keller，可以先听 #162；如果想更系统地理解处理器工程和抽象层如何影响计算进步，这期也值得补上。

## 一条推荐听法

先听 Jensen Huang、DeepSeek/Dylan Patel & Nathan Lambert 和 Jim Keller，建立算力、芯片、集群和供应链框架。再听 Sam Altman、Dario Amodei、Demis Hassabis、Yann LeCun、Mark Zuckerberg、Sundar Pichai，理解模型公司和平台公司的分歧。然后听 Cursor Team、Chris Lattner、DHH、ThePrimeagen，看 AI 如何改变软件工程。最后补 Karpathy、Elon Musk、Chris Urmson、Kyle Vogt、Marc Raibert、Robert Playter，形成自动驾驶和机器人路线图。

这样听的好处是先从物理约束和产业约束出发，再进入模型能力和产品入口，最后回到真实世界系统。AI 不是单一模型进步，而是算力、数据、组织、产品、工具链和硬件系统共同推进的结果。
