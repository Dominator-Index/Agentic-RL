# Dr.MAS Related Work：从单模型 GRPO 到多智能体协作训练

> 更广的 2025–2026 Agent RL 调研（含 ExGRPO、经验回放和 2026 新方法）见 [05 · 近期论文与 Related Work 调研](./05-2025-2026-Agent-RL近期论文与Related-Work调研.md)。本文保留为 Dr.MAS 引用脉络的专项对照。

整理日期：2026-10-05。以本目录 [DrMAS.pdf](./DrMAS.pdf) 的 §2 Related Work、参考文献和现有 [算法分析](./analysis.md) 为起点，并核对相关论文的 arXiv 页面。下文侧重方法定位与阅读线索，不重复已有数学推导，也不把不同实验设置中的分数直接比较。

**阅读主线：单模型策略优化 → 单 agent 多轮交互与信用分配 → 多 agent 协作训练与稳定性。** GiGPO 是 **Group-in-Group Policy Optimization** 的正式缩写。

## 1. 先区分这些方法在改什么

| 研究方向 | 代表方法 | 主要问题 | 与 Dr.MAS 的联系 |
| --- | --- | --- | --- |
| 单模型策略优化 | GRPO、RLOO、Dr. GRPO、DAPO、GSPO | 如何估计优势、降低开销、控制更新与优化偏差 | Dr.MAS 所建立在其上的优化背景 |
| 单 agent 多轮 RL | GiGPO、ARPO、RAGEN、SimpleTIR | 最终奖励如何指导中间动作；工具交互为何导致训练不稳 | 都处理多轮训练，但主要沿时间和状态维度展开 |
| 协作式多 agent RL | MAGRPO、MHGPO、MAPoRL、ReMA、MATPO、M-GRPO | 如何训练协作、分配奖励、组织不同角色的 rollout | 与 Dr.MAS 最直接相关 |
| 自博弈与角色条件优势 | SPIRAL、MARSHAL | 如何从交互中产生训练信号，并稳定不同角色的更新 | 已涉及角色条件基线或 agent-specific normalization，不能忽略 |
| Agent RL 系统 | MARTI、Agent Lightning、VerlTool、PettingLLMs | 如何把工作流、工具执行、数据与策略训练连接起来 | 对应 Dr.MAS 的多模型训练与资源调度贡献 |

这里的 single-agent 指主要训练一个与环境交互的决策主体；多次采样、调用多个工具或进行多轮推理，本身不等于 multi-agent。反过来，多个逻辑角色也可以共享同一套 LLM 参数，例如 MATPO。

## 2. 单模型策略优化：GRPO 及其相关改进

### 2.1 GRPO / DeepSeekMath（2024）

- **论文**：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)。Dr.MAS 参考文献 [14]。
- **方法**：对同一道题采样一组回答，以组内奖励统计构造相对优势，避免额外学习 value critic，再进行 PPO 风格的裁剪更新。
- **与 Dr.MAS 的关系**：提供基本优化结构。Dr.MAS 研究把这种组内比较用于协作轨迹后，共用奖励统计与不同角色实际训练样本分布之间的错配。

### 2.2 RLOO（2024，LLM 后训练代表论文）

- **论文**：[Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)。对应 [26]；[25] 是更早的 leave-one-out baseline 工作。
- **方法**：使用 REINFORCE 风格更新，对每条回答，以同题其他回答的平均奖励作为 leave-one-out baseline，降低方差，无需学习 critic。
- **与 Dr.MAS 的关系**：都关注如何构造基线，但 RLOO 主要解决同题样本之间的比较，Dr.MAS 关注角色执行样本的统计口径。RLOO 不是完整的多智能体训练方案。

### 2.3 Dr. GRPO（2025）★

- **论文**：[Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)。对应 [27]。
- **方法**：分析 GRPO 中回答长度归一化和问题级奖励标准差归一化带来的优化偏差；去掉这些可变重加权因素，缓解错误回答变长等现象，提升 token 效率。
- **与 Dr.MAS 的关系**：两者都审视归一化，但问题不同。Dr. GRPO 关注不同回答长度、不同难度问题受到怎样的权重；Dr.MAS 关注不同 agent 的优势尺度是否匹配其样本分布。不能把 Dr.MAS 简单理解为 Dr. GRPO 的多智能体版本。

### 2.4 DAPO（2025）

- **论文**：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)。对应 [28]。
- **方法**：组合非对称裁剪（Clip-Higher）、动态采样、token-level policy gradient loss、超长回答奖励处理，改善探索、有效训练样本比例及长输出训练。
- **与 Dr.MAS 的关系**：DAPO 主要改采样、损失聚合和训练配方；Dr.MAS 主要改角色级优势统计。二者处于不同设计维度，组合是否有效仍需实验验证。

### 2.5 GSPO（2025）

- **论文**：[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)。对应 [29]。
- **方法**：基于序列似然构造长度归一化的重要性比率，进行 sequence-level clipping，使更新单位更贴近序列奖励，并改善包括 MoE 在内的训练稳定性。
- **与 Dr.MAS 的关系**：GSPO 改的是重要性比率和裁剪粒度，Dr.MAS 改的是优势所使用的奖励统计。两者虽然都讨论稳定性，但对应不同失稳机制。

作为背景，PPO 原文为 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，对应 [24]。阅读时重点了解 clipped surrogate objective，以及常用 actor–critic 实现如何估计优势即可。

## 3. 单 agent 的多轮训练与工具使用

### 3.1 GiGPO（2025）★

- **论文**：[Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978)。对应 [40]。
- **方法**：构造两级相对优势：外层比较完整 episode；内层使用 anchor state grouping，将不同轨迹中遇到的重复环境状态对齐，比较从同一状态出发的动作，提供更细的步骤级信用信号。无需额外 critic，也不要求为此额外采样 rollout。
- **与 Dr.MAS 的关系**：GiGPO 问的是“同一状态下，哪个动作更值得强化”；Dr.MAS 问的是“对某个角色更新时，应该用什么奖励均值和标准差”。前者沿状态/步骤维度组织比较，后者沿 agent 维度校准尺度。
- **阅读重点**：episode advantage 与 step advantage 如何组合，以及环境状态如何判定为可分组。重复状态是否足够，是理解其适用场景的重要因素。

### 3.2 ARPO（2025）★

- **论文**：[Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849)。对应 [41]。
- **方法**：观察工具反馈后模型的不确定性上升，以熵驱动的自适应 rollout 在高不确定步骤增加探索，并结合优势归因，学习不同工具交互步骤的价值。
- **与 Dr.MAS 的关系**：ARPO 侧重“在哪里分配采样与探索预算，以及如何提供步骤信号”；Dr.MAS 侧重“已有协作 rollout 如何用于稳定的角色更新”。

### 3.3 RAGEN / StarPO（2025）

- **论文**：[RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073)。对应 [39]。
- **方法**：提出 trajectory-level agent RL 框架 StarPO，并分析多轮训练中的 Echo Trap、奖励方差下降与梯度尖峰；稳定化变体结合轨迹过滤、critic 和梯度稳定措施。
- **与 Dr.MAS 的关系**：是多轮 agent RL 训练崩溃的重要背景，但不能把它的所有失稳现象都归结为 Dr.MAS 所分析的跨角色统计错配。

### 3.4 SimpleTIR（2025）★

- **论文**：[SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning](https://arxiv.org/abs/2509.02479)。对应 [37]。
- **方法**：将多轮工具推理的梯度爆炸与工具反馈引发的分布漂移、低概率 token 联系起来；过滤包含 void turns（既无代码块、也无最终答案的轮次）的轨迹以稳定更新。
- **与 Dr.MAS 的关系**：两者都直接研究梯度不稳定，但 SimpleTIR 的主要干预是异常轨迹过滤，Dr.MAS 的主要干预是角色级优势归一化。这是一组很适合并列介绍的稳定性工作。

### 3.5 Search-R1（2025）

- **论文**：[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)。Dr.MAS 参考文献 [63]，不在 §2 的主文献段落中。
- **方法**：让一个 LLM 在推理过程中学习发起搜索、利用检索结果并继续作答；采用最终结果奖励，并对检索返回的 token 做训练掩码。
- **与 Dr.MAS 的关系**：是理解搜索任务的自然单 agent 对照。Search-R1 中搜索工具不是另一套被联合训练的 LLM 策略；Dr.MAS 的搜索案例则划分 Search、Answer、Verifier 等角色。

### 3.6 ToolRL（2025）

- **论文**：[ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)。对应 [36]。
- **方法**：系统研究工具选择与调用任务的奖励类型、尺度、粒度和时间变化，使用 GRPO 训练工具使用能力。
- **与 Dr.MAS 的关系**：主要贡献在奖励设计，适合作为“如何得到有效工具学习信号”的背景；Dr.MAS 主要研究已有奖励如何进入各角色的优势计算。

## 4. 与 Dr.MAS 最直接相关的协作式多 agent RL

### 4.1 MAGRPO（2025）★

- **论文**：[LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.04652)。对应 [45]。
- **方法**：把 LLM 协作建模为 cooperative MARL，将组相对策略优化扩展至多 agent、多轮交互，在写作和代码协作中学习合作。
- **与 Dr.MAS 的关系**：代表“怎样将 GRPO 用于协作训练”的直接前序工作；Dr.MAS 进一步分析共享组统计在角色样本分布不一致时的稳定性问题。
- **命名注意**：MAGRPO、下面的 M-GRPO，以及泛称的 multi-agent GRPO 不是同一个专名。Dr.MAS 实验中的 vanilla GRPO 也不应自动等同于完整复现 MAGRPO。

### 4.2 MHGPO（2025）★

- **论文**：[Heterogeneous Group-Based Reinforcement Learning for LLM-based Multi-Agent Systems（v1）](https://arxiv.org/abs/2506.02718v1)。对应 [46]。
- **方法**：面向多 agent 搜索系统，利用异质 rollout groups 的相对奖励优化系统表现，无需 critic，并比较三种 group rollout 采样策略。
- **与 Dr.MAS 的关系**：MHGPO 重点是组构造、采样与系统级协作优化；Dr.MAS 重点是每个角色的奖励统计与梯度尺度。这里的 heterogeneous group 不能直接等同于“部署不同大小的 LLM”。
- **版本说明**：[当前条目](https://arxiv.org/abs/2506.02718)标题改为 *End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning*；上面的 v1 链接对应 Dr.MAS 引用的旧标题。

### 4.3 MAPoRL（2025）

- **论文**：[MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](https://arxiv.org/abs/2502.18439)。对应 [48]。
- **方法**：多个 LLM 先各自回答，再多轮讨论；verifier 同时评价答案正确性和讨论中的纠错、说服行为，将分数用于协同后训练。
- **与 Dr.MAS 的关系**：强调协作行为需要通过 co-training 学习，而非仅依赖提示词。其讨论奖励与训练范式适合对比 Dr.MAS 的共享结果奖励设定，但不能笼统归为同一种 GRPO 算法。

### 4.4 ReMA（2025）

- **论文**：[ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2503.09501)。对应 [49]。
- **方法**：将元思考、策略监督与具体推理解耦为高层和低层两个 agent，通过 RL 学习协作；进一步研究多轮交互及参数共享。
- **与 Dr.MAS 的关系**：与 Solver–Verifier 一类角色分工有结构上的联系，但 ReMA 侧重诱导 meta-thinking，Dr.MAS 侧重这类角色系统的训练稳定性。

### 4.5 MATPO（2025）★

- **论文**：[Multi-Agent Tool-Integrated Policy Optimization](https://arxiv.org/abs/2510.04678)。对应 [51]。
- **方法**：通过不同角色提示词，让同一个 LLM 承担 planner 和 worker，设计跨角色 rollout 的信用分配并进行 RL 训练，兼顾角色专门化和部署开销。
- **与 Dr.MAS 的关系**：很适合比较“逻辑多 agent、物理单模型”与多个独立模型联合训练。Dr.MAS 支持参数共享与非共享，包括异构模型配置。

### 4.6 M-GRPO / Multi-Agent Deep Research（2025）★

- **论文**：[Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO](https://arxiv.org/abs/2511.13288)。对应 [52]。
- **方法**：面向主 agent 规划、子 agent 执行工具任务的层级系统，计算主/子 agent 的组相对优势，使用轨迹对齐处理变化的子 agent 调用数，并通过解耦训练管线支持不同服务器上的模型。
- **与 Dr.MAS 的关系**：问题设定非常接近，均涉及角色调用频率不同和独立模型训练。M-GRPO 强调层级信用分配、轨迹对齐与训练解耦；Dr.MAS 强调统计错配的梯度分析、角色级归一化与共享资源调度。

### 4.7 CoMAS（2025）

- **论文**：[CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards](https://arxiv.org/abs/2510.08529)。对应 [50]。
- **方法**：从 agent 间讨论产生交互奖励，以 LLM-as-a-judge 评价交互并优化各 agent，使系统通过互动共同进化。
- **与 Dr.MAS 的关系**：CoMAS 主要创新在训练信号来源；Dr.MAS 主要关注共享任务奖励下的优化稳定性。“无需外部监督”也不等于“没有奖励评价器”。

### 4.8 AT-GRPO / PettingLLMs（2025）★

- **论文**：[Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs（v1）](https://arxiv.org/abs/2510.11062v1)。对应 [58]。
- **方法**：按 agent 和 turn 构建训练分组，处理角色与轮次变化导致的上下文差异，并支持单策略与多策略的 on-policy 训练。
- **与 Dr.MAS 的关系**：不仅是系统接口相关工作，其 agent/turn-wise grouping 在算法层面也与 Dr.MAS 直接相关。重点比较“按什么条件形成可比较的样本组”与“在哪个样本集合上统计角色均值、标准差”。
- **版本说明**：[当前条目](https://arxiv.org/abs/2510.11062)标题为 *Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs*；这里保留 Dr.MAS 引用时的标题并链接 v1。

## 5. 不应遗漏：自博弈、角色条件优势与单模型蒸馏

### 5.1 SPIRAL（2025）★

- **论文**：[SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2506.24119)。对应 [42]。
- **方法**：通过零和游戏自博弈，让不断增强的对手形成自动课程；使用 **role-conditioned advantage estimation（RAE）** 稳定多 agent 训练，并研究向通用推理任务的迁移。
- **与 Dr.MAS 的关系**：场景偏竞争式自博弈，但其角色条件优势估计与 Dr.MAS 的动机直接相连。因此 related work 不能只用“游戏自博弈”一句话概括，也不能宣称此前无人考虑角色特定基线。

### 5.2 MARSHAL（2025）★

- **论文**：[MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs](https://arxiv.org/abs/2510.15414)。对应 [43]。
- **方法**：在合作和竞争游戏中进行自博弈，结合 turn-level advantage estimator 与 **agent-specific advantage normalization**，处理多轮信用分配和更新稳定性。
- **与 Dr.MAS 的关系**：这是比较“按 agent 归一化”时必须阅读的近邻。应比较具体的统计范围、奖励结构、优势估计及理论结论；不能仅凭 Dr.MAS 的摘要就认定这一操作本身是首次提出。

### 5.3 AgentFlow / Flow-GRPO（2025）

- **论文**：[In-the-Flow Agentic System Optimization for Effective Planning and Tool Use](https://arxiv.org/abs/2510.05592)。对应 [47]。
- **方法**：通过 planner、executor、verifier、generator 等模块组织多轮工具流程，在真实交互中优化 planner；Flow-GRPO 将最终结果信号用于各轮 planner 决策。
- **与 Dr.MAS 的关系**：工作流有多个模块，但主要训练对象是 planner，不能自动等同于所有角色共同更新。它适合放在“可训练 agentic 系统”与“多模型联合训练”的交界处。

### 5.4 Chain-of-Agents / Agent Foundation Models（2025）

- **论文**：[Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL](https://arxiv.org/abs/2508.13167)。对应 [44]。
- **方法**：先将多 agent 系统的协作轨迹蒸馏到一个模型，再通过 agentic RL 改进，使单模型内部呈现工具 agent 和角色协作式推理。
- **与 Dr.MAS 的关系**：目标偏向把多 agent 能力内化到单模型；Dr.MAS 则保留显式角色及灵活模型分配。此处特指 2025 年这篇论文，避免与其他同名 Chain-of-Agents 工作混淆。

## 6. 系统相关工作：写一小段即可

| 工作 | 论文或官方入口 | 核心定位与 Dr.MAS 的关系 |
| --- | --- | --- |
| veRL / HybridFlow，[15] | [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256) | 通用 LLM RL 训练基础设施，是理解 rollout–training 执行的背景 |
| VerlTool，[55] | [VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use](https://arxiv.org/abs/2509.01055) | 在 veRL 上统一工具接口与异步多轮 rollout，侧重工具交互工程支持 |
| Agent Lightning，[56] | [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680) | 解耦 agent 执行与训练，通过统一接口和 LightningRL 信用分配处理复杂工作流 |
| MARTI，[57] | [论文（OpenReview）](https://openreview.net/forum?id=E7jZqo0A50)；[官方代码](https://github.com/TsinghuaC3I/MARTI) | 集中组织多 agent 交互与奖励，分布式训练策略，支持动态工作流 |
| PettingLLMs / AT-GRPO，[58] | [arXiv](https://arxiv.org/abs/2510.11062v1) | 多 agent on-policy 训练接口，同时有 agent/turn 分组的算法贡献，见 §4.8 |

MARTI 原始框架论文此次未确认独立 arXiv 编号，因此给出官方论文与代码入口。不要把后续 [MARTI-MARS²（2602.07848）](https://arxiv.org/abs/2602.07848) 当作原始 MARTI 的 arXiv 编号。

写系统对比时应标明版本。Dr.MAS 对其他框架的能力评价反映其论文所考察的实现；当前 MARTI 官方项目已经明确列出异构多 agent 训练支持，因而不宜笼统写“现有框架都不支持异构模型”。更具体的比较维度是：多角色编排、共享/独立参数、角色独立优化配置、GPU 资源池，以及动态释放和调度机制。

## 7. 可以直接改写进 related work 的中文段落

**组相对策略优化。** 基于组内比较的强化学习为 LLM 后训练提供了无需额外价值网络的方案。GRPO 在同题回答之间构造相对优势，RLOO 则利用其他样本的奖励构造 leave-one-out baseline。后续工作分别从优化偏差、采样与裁剪策略、重要性比率粒度等角度改进训练：Dr. GRPO 分析长度及奖励标准差归一化带来的偏差，DAPO 结合动态采样和裁剪等训练技术，GSPO 采用序列级重要性比率与裁剪。这些工作为协作式 LLM 训练提供了基础，但各自的主要设计重点并非角色条件下的奖励统计匹配。相关文献：[GRPO](https://arxiv.org/abs/2402.03300)、[RLOO](https://arxiv.org/abs/2402.14740)、[Dr. GRPO](https://arxiv.org/abs/2503.20783)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)。

**多轮 agent 强化学习。** 当 LLM 在环境中持续调用工具时，稀疏结果奖励带来的步骤信用分配以及交互引入的训练不稳定成为重要问题。GiGPO 结合 episode-level 与基于重复状态分组的 step-level 优势；ARPO 根据工具交互后的不确定性调整探索并估计步骤贡献。与此同时，RAGEN 与 SimpleTIR 从多轮训练动力学和工具反馈诱发的异常轨迹等角度研究稳定性，Search-R1 展示了以结果奖励训练自主检索的路径。这些研究主要围绕一个决策主体的交互过程，构成研究多角色协作训练的基础。相关文献：[GiGPO](https://arxiv.org/abs/2505.10978)、[ARPO](https://arxiv.org/abs/2507.19849)、[RAGEN](https://arxiv.org/abs/2504.20073)、[SimpleTIR](https://arxiv.org/abs/2509.02479)、[Search-R1](https://arxiv.org/abs/2503.09516)。

**多 agent 协作与训练稳定性。** MAGRPO、MHGPO 和 MAPoRL 研究通过 RL 学习 LLM 间的协作；ReMA、MATPO 和 M-GRPO 进一步探索层级角色分工、跨角色信用分配及不同调用频率下的训练组织。角色条件训练也已有直接相关研究：SPIRAL 使用角色条件优势估计，MARSHAL 结合轮次级优势与 agent-specific normalization，AT-GRPO 按 agent 和轮次组织比较。在这一背景下，Dr.MAS 的重点是分析协作式组相对训练中共用归一化统计与角色实际奖励分布的错配，建立其与梯度二阶矩放大的联系，并以角色级奖励统计校准优势，同时提供多模型协同训练和资源调度支持。这一定位应与自博弈、信用分配以及不同统计范围下的归一化工作具体比较。相关文献：[MAGRPO](https://arxiv.org/abs/2508.04652)、[MHGPO](https://arxiv.org/abs/2506.02718v1)、[MAPoRL](https://arxiv.org/abs/2502.18439)、[ReMA](https://arxiv.org/abs/2503.09501)、[MATPO](https://arxiv.org/abs/2510.04678)、[M-GRPO](https://arxiv.org/abs/2511.13288)、[SPIRAL](https://arxiv.org/abs/2506.24119)、[MARSHAL](https://arxiv.org/abs/2510.15414)、[AT-GRPO](https://arxiv.org/abs/2510.11062v1)、[Dr.MAS](https://arxiv.org/abs/2602.08847)。

## 8. 推荐阅读顺序与比较问题

1. **Dr. GRPO → GiGPO**：先区分优化偏差与步骤信用分配，避免把所有 GRPO 改进都理解为同一件事。
2. **MAGRPO → MHGPO → M-GRPO**：了解组相对优化如何进入协作系统，尤其关注 group 的定义与采样方式。
3. **SPIRAL → MARSHAL → AT-GRPO**：专门检查角色条件基线、agent-specific normalization 与 agent/turn 分组，这是理解 Dr.MAS 贡献边界的关键。
4. **MATPO → AgentFlow → Dr.MAS**：对照共享模型多角色、主要训练 planner、多个模型共同训练的差异。
5. **SimpleTIR → RAGEN**：补充其他梯度不稳定来源，理解 Dr.MAS 尚未覆盖的问题。

逐篇阅读时，记录五件事即可：**训练哪些参数、采样组如何构成、奖励来自哪里、优势在哪个集合上计算、稳定性或信用分配问题由什么机制解决。** 上述“与 Dr.MAS 的关系”是据方法机制作出的比较性归纳，不代表这些方法已经在统一实验下验证了相互替代或组合效果。
