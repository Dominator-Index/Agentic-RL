# 2025–2026 Agent RL 近期论文与 Related Work 调研

**范围：2025 年至 2026-10-05 的 LLM agent 强化学习方法，以及与之直接相关的单模型 RL 优化方法。** 从 Dr.MAS 出发，但不局限于它的引用列表；补充 ExGRPO 及其经验回放相关工作、GiGPO 之后的步骤信用分配工作，以及 2026 年的多 agent 信用分配和在线训练方法。

本文按问题分类，每篇简述“做什么、和谁相关”。年份原则上采用 arXiv 首次公开年份，2025 年预印本即使在 2026 年发表，也不另算一篇 2026 新论文。主要条目已核对 arXiv 标题和摘要；对用于追溯文献的核心论文，还读取了 related work / 方法章节。这里是代表性调研，不是穷尽目录或统一实验排名。

原先以 Dr.MAS 为中心的逐篇对照见 [04 · Dr.MAS Related Work 论文梳理](./04-DrMAS-Related-Work论文梳理.md)。**这份 05 是范围更广的主阅读清单。**

## 1. 先看全局：近期工作主要解决什么问题

| 问题 | 代表论文 | 阅读时应关注的变量 |
| --- | --- | --- |
| 单模型 RL 更新存在偏差或不稳定 | Dr. GRPO、DAPO、GSPO | 归一化、重要性比率、裁剪、loss 聚合方式 |
| rollout 很贵，历史经验被浪费 | ExGRPO、RePO、ReMix、RLEP、GUI-ARPO | 存什么经验、回放什么、怎样纠正 off-policy 偏差 |
| 最终奖励无法分辨中间动作 | GiGPO、SPA-RL、ARPO、GraphGPO、CW-GRPO | 状态分组、额外分支采样、价值估计、过程评价 |
| 多轮或异步更新发生崩溃 | RAGEN、SimpleTIR、SORL、SAO | 工具反馈、低概率动作、policy lag、更新粒度 |
| 多 agent 成功后不清楚谁有贡献 | M-GRPO、SHARP、C3、CCPO、COSAC | 团队奖励如何变成角色/消息级学习信号 |
| 不同角色的梯度尺度不匹配 | Dr.MAS、SPIRAL、MARSHAL | 奖励统计集合、角色条件基线、agent-specific normalization |
| 数据、环境或反馈应随训练变化 | AgentGym-RL、CoEvolve、RLAnything、OpenClaw-RL | 交互长度课程、任务生成、奖励模型、在线反馈 |
| 多角色/多环境训练难以实现 | Agent Lightning、AgentRL、Agent-R1、MARTI | 轨迹表示、异步管线、模型共享、环境接口和资源配置 |

这几个方向经常能组合，但“可以组合”是方法层面的判断，不等于原论文已验证组合效果。例如：经验回放可能增加 off-policy 偏差；引入步骤信用分配也可能增加 judge 或环境采样成本。

## 2. 单模型优化与经验回放：把 ExGRPO 放进完整脉络

这些工作主要研究 reasoning / RLVR，不应全部称为多轮 agent 算法；但它们构成 agent RL 的直接优化背景。

| 方法 / 首发 | arXiv | 方法简述 | 与其他方法的关系 |
| --- | --- | --- | --- |
| **Dr. GRPO · 2025-03** | [2503.20783](https://arxiv.org/abs/2503.20783) | 去除回答长度和问题级奖励标准差带来的可变重加权，分析长度与难度偏差 | 关注损失与优势的偏差；ExGRPO 的实现沿用其去除相关归一化的处理 |
| **DAPO · 2025-03** | [2503.14476](https://arxiv.org/abs/2503.14476) | 非对称裁剪、动态采样、token-level loss、超长回答处理 | 改训练配方及有效探索，适合与 Dr. GRPO、GSPO 对照 |
| **GSPO · 2025-07** | [2507.18071](https://arxiv.org/abs/2507.18071) | 从 token-level importance ratio 转向长度归一化的序列级比率与裁剪 | 重点是更新粒度；不等同于步骤或角色信用分配 |
| **ExGRPO · 2025-10；2026 修订** | [2510.02245](https://arxiv.org/abs/2510.02245) | 以正确性和熵等指标管理、优先选择经验，用 mixed-policy objective 结合新 rollout 与历史经验 | 核心问题是“什么经验值得重复学习”；比只加一个 replay buffer 更强调经验质量 |
| **RePO · 2025-06** | [2506.09340](https://arxiv.org/abs/2506.09340) | 从回放池检索同题历史输出，扩展用于优化的样本集合，并比较不同 replay 策略 | ExGRPO 的直接 related work；适合比较回放选择策略 |
| **ReMix · 2025-07** | [2507.06892](https://arxiv.org/abs/2507.06892) | mixed-policy proximal policy gradient、提高 update-to-data ratio，配合 KL 约束和 policy reincarnation | 更关注如何有效而稳定地利用 off-policy 数据，与 ExGRPO 的经验价值筛选互补 |
| **RLEP · 2025-07** | [2507.07451](https://arxiv.org/abs/2507.07451) | 先收集经过验证的成功轨迹，再在训练中混合新样本与回放成功样本 | 是“高质量成功经验复用”的直接对照 |
| **TOPR · 2025-03** | [2503.14286](https://arxiv.org/abs/2503.14286) | 使用非对称、渐缩的重要性权重，统一利用正负离策略样本 | 重点是 off-policy 梯度估计及方差控制，而不只是经验存储 |

### ExGRPO 值得重点记住的三点

1. **研究对象主要是单模型推理 RLVR。** 它适合出现在 agent RL 的经验复用背景中，但不能直接写成已解决多 agent 或长程环境信用分配。
2. **有经验不等于经验都有价值。** 它研究任务和轨迹的价值，利用正确性与熵等性质组织回放，避免把 replay 理解成均匀重复训练旧数据。
3. **经验筛选与优化目标是两层设计。** 不仅选样本，还要处理历史策略与当前策略的差异，以及新探索和经验利用之间的平衡。

**从其 related work 实际扩展出的阅读链：** ExGRPO §2 → RePO、ReMix、RLEP、TOPR，以及下文的 GUI-ARPO。这里是“引用与主题关系”，不表示所有箭头都构成算法的直接继承关系。来源：[ExGRPO 全文](https://arxiv.org/html/2510.02245)。

## 3. 单 agent 多轮优化：步骤信用分配与稳定性

| 方法 / 首发 | arXiv | 方法简述 | 与 GiGPO / Dr.MAS 的关系 |
| --- | --- | --- | --- |
| **LOOP · 2025-02** | [2502.01600](https://arxiv.org/abs/2502.01600) | 在 AppWorld 等有状态 API 环境中训练交互 agent；使用无需 value network 的高效 PPO 变体 | GiGPO related work 中的重要长程交互前序工作 |
| **GiGPO · 2025-05** | [2505.10978](https://arxiv.org/abs/2505.10978) | episode-level 分组之外，对 rollout 中重复出现的环境状态构造 step-level group，比较从同一状态采取的动作 | 关注“哪一步好”；Dr.MAS 关注“这个角色的优势如何校准” |
| **SPA-RL · 2025-05** | [2505.20732](https://arxiv.org/abs/2505.20732) | 学习任务进展估计器，把最终结果分解为步骤贡献，并结合动作 grounding 信号 | 与 GiGPO 对照：学习进展模型 vs. 利用采样中的状态分组 |
| **Turn-level reward / credit assignment · 2025-05** | [2505.11821](https://arxiv.org/abs/2505.11821) | 将交互建模为 turn-level MDP，区分 terminal、delayed、per-turn reward，并构造相应 GRPO/PPO | 是理解“整条轨迹一个奖励”与“每轮学习信号”的直接文献；题名随版本调整 |
| **ARPO（Reinforced）· 2025-07** | [2507.19849](https://arxiv.org/abs/2507.19849) | 在工具反馈后不确定性较高的位置自适应增加 rollout 探索，结合步骤优势归因 | 与 GiGPO 都追求局部信用，但 ARPO 还主动改变采样分配 |
| **RAGEN / StarPO · 2025-04** | [2504.20073](https://arxiv.org/abs/2504.20073) | 研究多轮训练的 Echo Trap、奖励方差变化与梯度尖峰，提出稳定化配方 | 与 Dr.MAS 同属训练动力学问题，但失稳机制并不相同 |
| **SimpleTIR · 2025-09** | [2509.02479](https://arxiv.org/abs/2509.02479) | 将工具反馈引起的分布漂移、低概率 token 与梯度爆炸联系起来，过滤含 void turn 的轨迹 | 通过轨迹过滤处理工具交互失稳，而非按角色调整奖励统计 |
| **SORL / SO-PPO / SO-GRPO · 2025-11；2026 修订** | [2511.20718](https://arxiv.org/abs/2511.20718) | turn-level importance sampling 与 clipping-triggered normalization，抑制多轮离策略训练中的不可靠更新 | 将“更新粒度”与“off-policy 稳定性”连接起来，适合在 ExGRPO 后阅读 |

**GiGPO 的 related work 主线：** 长程交互训练（如 LOOP、RAGEN）与无 critic 的组相对优化交汇，进一步需要 finer-grained credit assignment。**SORL 的 related work 主线：** Search-R1、WebAgent-R1 等交互训练 + TOPR、RePO、GUI-ARPO 等离策略方法 + turn-level credit assignment。来源：[GiGPO §2](https://arxiv.org/html/2505.10978)、[SORL §2](https://arxiv.org/html/2511.20718)。

## 4. 搜索、GUI、工具使用：算法如何落到 agent 任务

| 方法 / 首发 | arXiv | 方法简述 | 分类时要注意 |
| --- | --- | --- | --- |
| **Search-R1 · 2025-03** | [2503.09516](https://arxiv.org/abs/2503.09516) | 用结果奖励训练一个模型进行多轮搜索与推理，对检索返回 token 做 mask | 搜索引擎是环境/工具，不是另一套被联合训练的 LLM 策略 |
| **ToolRL · 2025-04** | [2504.13958](https://arxiv.org/abs/2504.13958) | 系统研究工具选择与调用的奖励类型、尺度、粒度和时序 | 主要是 reward design，不能仅按“GRPO 变体”概括 |
| **WebAgent-R1 · 2025-05** | [2505.16421](https://arxiv.org/abs/2505.16421) | 从真实网页交互中异步采样轨迹，以任务成功奖励端到端训练 | 适合研究长程环境交互与在线 rollout 成本 |
| **ARPO（Replay，GUI）· 2025-05** | [2505.16282](https://arxiv.org/abs/2505.16282) | 为 GUI agent 的 GRPO 加成功经验回放，并按基线表现选择训练任务 | **不是** 2507.19849；这是 ExGRPO related work 中的 ARPO |
| **AgentFlow / Flow-GRPO · 2025-10** | [2510.05592](https://arxiv.org/abs/2510.05592) | 组织 planner、executor、verifier、generator，在交互中用最终结果训练 planner | 多模块工作流不自动意味着所有模块共同更新 |
| **Chain-of-Agents / AFM · 2025-08** | [2508.13167](https://arxiv.org/abs/2508.13167) | 将多 agent 协作轨迹蒸馏到单模型，再做 agentic RL | 与显式多模型 co-training 是不同路线 |

**两个 ARPO 的消歧：**

- **Agentic Replay Policy Optimization**：[ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay](https://arxiv.org/abs/2505.16282)，关键词是 **GUI、replay buffer**。
- **Agentic Reinforced Policy Optimization**：[Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849)，关键词是 **工具反馈、熵、自适应探索**。

## 5. 多 agent 协作：2025 年形成的主要方法线

| 方法 / 首发 | arXiv | 方法简述 | 与 Dr.MAS 的关系 |
| --- | --- | --- | --- |
| **MAPoRL · 2025-02** | [2502.18439](https://arxiv.org/abs/2502.18439) | 多个 LLM 先独立回答再讨论，以答案与讨论评价进行 co-training | 强调“合作能力需要被训练”，奖励设计是重要组成 |
| **ReMA · 2025-03** | [2503.09501](https://arxiv.org/abs/2503.09501) | 高层 meta-thinking agent 与低层 reasoning agent 分工并通过 RL 学习协作 | 更偏角色结构与元思考能力，而非统一的稳定性修正 |
| **MARFT · 2025-04** | [2504.16129](https://arxiv.org/abs/2504.16129) | 研究面向 LLM 多 agent 的强化微调建模与框架，处理异步交互、角色和异构结构 | 为多 agent RL 的形式化和实现提供更广背景 |
| **MHGPO · 2025-06** | [2506.02718](https://arxiv.org/abs/2506.02718) | 从异质 rollout groups 估计相对优势，比较 group 采样策略，优化多 agent 搜索 | 核心是组构造与系统级优化；heterogeneous group 不等于异构参数规模 |
| **MAGRPO · 2025-08** | [2508.04652](https://arxiv.org/abs/2508.04652) | 将协作式多 agent、多轮交互纳入 group-relative policy optimization | 是 Dr.MAS 最直接的组相对协作训练背景 |
| **MATPO · 2025-10** | [2510.04678](https://arxiv.org/abs/2510.04678) | 一个 LLM 以不同角色提示词担任 planner/worker，并做跨角色信用分配 | 展示共享参数的多角色训练，区别于多个独立模型 |
| **M-GRPO · 2025-11** | [2511.13288](https://arxiv.org/abs/2511.13288) | 主 agent 与子 agent 的层级优势、可变调用数下的轨迹对齐、解耦模型训练 | 与 Dr.MAS 的角色调用异质性和独立模型训练高度相关 |
| **CoMAS · 2025-10** | [2510.08529](https://arxiv.org/abs/2510.08529) | 从讨论动态构造交互奖励，通过 judge 和 RL 让 agents 共同进化 | 重点是训练信号来源，区别于直接使用任务结果奖励 |
| **AT-GRPO / PettingLLMs · 2025-10** | [2510.11062](https://arxiv.org/abs/2510.11062) | 针对 agent 和 turn 变化组织分组，支持单策略及多策略 on-policy 训练 | 同时是算法近邻与系统近邻；不能只作为基础设施引用 |
| **SPIRAL · 2025-06** | [2506.24119](https://arxiv.org/abs/2506.24119) | 零和游戏自博弈产生课程，使用 role-conditioned advantage estimation | 已关注角色条件基线，是讨论 Dr.MAS 新颖性时应阅读的工作 |
| **MARSHAL · 2025-10** | [2510.15414](https://arxiv.org/abs/2510.15414) | 合作/竞争游戏自博弈，turn-level advantage 与 agent-specific normalization | 与 Dr.MAS 的按 agent 归一化联系直接，应比较统计范围和理论设定 |

**命名提醒：** MAGRPO、M-GRPO、MHGPO 是不同论文的具体方法名。泛称 “multi-agent GRPO” 时，需要说明对应哪个实现与 group 定义。

## 6. 2026 新工作：从“稳定更新”走向“谁真正贡献了结果”

| 方法 / 首发 | arXiv | 方法简述 | 与既有工作的关系 |
| --- | --- | --- | --- |
| **Dr.MAS · 2026-02** | [2602.08847](https://arxiv.org/abs/2602.08847) | 分析全局奖励统计与角色执行样本分布的错配，用 agent-wise normalization 校准梯度尺度，支持多模型联合训练 | 稳定性主线；共享结果奖励经过不同归一化，不代表已识别各角色因果贡献 |
| **SHARP · 2026-02** | [2602.08335](https://arxiv.org/abs/2602.08335) | 结合全局正确性、Shapley 边际贡献及工具过程奖励，并归一化 agent-specific advantages | 同时涉及奖励分解与优势归一化，是 Dr.MAS 的重要近邻 |
| **C3 · 2026-03；2026-09 大幅修订** | [2603.06859](https://arxiv.org/abs/2603.06859) | 在可恢复的共享上下文决策点替换消息，执行后续 continuation，以结果差异估计消息贡献 | 从终局广播转向消息级反事实信用；有效性依赖 trace 是否充分表示可恢复状态 |
| **CCPO / SEPO · 2026-03；后续修订** | [2603.21563](https://arxiv.org/abs/2603.21563) | CCPO 比较去掉某 agent 的反事实结果；SEPO 利用受约束的自评/互评构造角色信号 | 当前版本明确定位于 reward/credit construction，可接 GRPO、GSPO 等优化器 |
| **COSAC · 2026-04** | [2604.17693](https://arxiv.org/abs/2604.17693) | 对顺序协作团队拟合加性奖励分解，用虚拟后续生成估计角色优势，避免额外环境/奖励调用 | 与 C3 比较“执行反事实”和“拟合分解”，重点是偏差、方差与额外成本 |
| **MARS² · 2026-04** | [2604.14564](https://arxiv.org/abs/2604.14564) | 多个独立优化的 agent 在共享搜索树中生成与改进代码，使用路径级组优势和树一致奖励塑形 | 把多 agent 协作与 tree-search RL 结合，重点从固定工作流扩展到结构化探索 |

### 这一组和 Dr.MAS 的核心区别

Dr.MAS 中，各角色可以继承同一团队结果奖励，只是统计和归一化方式不同。**“尺度校准”与“贡献识别”是两个问题。** SHARP、C3、CCPO、COSAC 更直接研究某个 agent 或消息为最终结果带来的增量。因此，写 related work 时可以把它们分别组织成 *stability of multi-agent optimization* 和 *agent/message-level credit assignment*，同时承认两类方法可能重叠。

**实际追溯证据：** C3 当前版本 §7 将 MARFT、MAGRPO、Dr.MAS、SHARP、CCPO、COSAC、AT-GRPO 等放在相邻文献中讨论；MARS² 附录 A 将 RL 后训练、搜索增强 RL 和多 agent 协作串联起来。本文沿这些引用扩展文献，而不是仅按方法缩写搜集。来源：[C3 §7](https://arxiv.org/html/2603.06859)、[MARS² 附录 A](https://arxiv.org/html/2604.14564)。

## 7. 2026 新工作：单 agent 的细粒度信用、异步学习与自进化

| 方法 / 首发 | arXiv | 方法简述 | 应与谁对照 |
| --- | --- | --- | --- |
| **RLAnything · 2026-02** | [2602.02488](https://arxiv.org/abs/2602.02488) | 在闭环中共同改进环境、策略和奖励模型，融合步骤级及结果级信号 | 对照固定数据/奖励的 RL；也与 CoEvolve 的动态任务分布相关 |
| **OpenClaw-RL · 2026-03** | [2603.10165](https://arxiv.org/abs/2603.10165) | 从用户回复、工具输出等 next-state signals 提取评价与指令性反馈，异步在线更新策略 | 与离线 replay 不同；更关注部署交互产生的持续学习信号 |
| **CW-GRPO · 2026-04** | [2604.14267](https://arxiv.org/abs/2604.14267) | judge 评价每轮检索效用与推理正确性，以贡献分数缩放结果优势 | 对照 GiGPO：judge 过程评价 vs. 同状态采样比较 |
| **CoEvolve · 2026-04** | [2604.15840](https://arxiv.org/abs/2604.15840) | 用遗忘、不确定性等交互信号引导任务生成，经环境验证后更新训练分布 | 不只是重放旧经验，而是让任务数据随 agent 能力变化 |
| **GraphGPO · 2026-05** | [2605.26684](https://arxiv.org/abs/2605.26684) | 把 rollout 聚成状态转移图，利用到目标的图距离变化估计动作贡献 | 从 GiGPO 的局部同状态分组扩展到跨轨迹图结构，是优先阅读的后续方向 |
| **TRIAGE · 2026-06** | [2606.32017](https://arxiv.org/abs/2606.32017) | judge 将交互片段分为进展、有效探索、无进展基础动作、倒退，形成有界过程奖励 | 此处 role 是动作片段的语义作用，不是 Dr.MAS 的 agent 身份 |
| **SAO · 2026-07** | [2607.07508](https://arxiv.org/abs/2607.07508) | 每题单 rollout 的异步优化，结合 value model 训练与严格双侧 token clipping | 对照组采样的等待与 policy lag；不是无 critic 的 GRPO 小改动 |
| **GRAFT / Graph GAE · 2026-09** | [2609.28963](https://arxiv.org/abs/2609.28963) | 聚合轨迹图，用 Bellman 迭代恢复状态价值，并以 Graph GAE 改进步骤优势 | 与 GraphGPO 同用图，但价值构造不同，不能仅按“图分组”归为同一方法 |
| **ProVer · 2026-09** | [2609.36178](https://arxiv.org/abs/2609.36178) | judge 定位可能关键的片段，再用片段前后 continuation 成功率验证，增强局部优势 | 将“定位关键步骤”与“通过采样验证贡献”分开，需计入额外 continuation 成本 |
| **SHARPO · 2026-09/10** | [2610.00838](https://arxiv.org/abs/2610.00838) | 使用片段内 teacher–student log-prob 差异构造有界乘子，对 GRPO 优势做分段重加权 | 连接 on-policy self-distillation 与 segment-level credit；与多 agent 的 SHARP 不是同一方法 |

最后三篇属于截至本次检索刚公开的预印本，适合作为新方向跟踪；表中概括的是论文提出的机制，不代表已经获得充分独立复现。SHARPO 的 arXiv 编号为 2610，但页面首次提交日期为 2026-09-30，因此标成 09/10。

**GraphGPO 的相关工作与问题展开** 将 RLVR、agent RL、GiGPO 式 step grouping 连起来；**SAO §5** 则把组相对优化与异步训练系统的 policy lag 联系起来。这说明 2026 的发展并不只是不断出现新 GRPO 名称，而是在重新选择学习信号、采样组织和价值估计方式。来源：[GraphGPO §2–3](https://arxiv.org/html/2605.26684)、[SAO §5](https://arxiv.org/html/2607.07508)。

## 8. 训练框架与课程：近期 agent 论文不能只看优化器

| 工作 / 首发 | 论文入口 | 核心内容 |
| --- | --- | --- |
| **Agent Lightning · 2025-08** | [2508.03680](https://arxiv.org/abs/2508.03680) | 解耦 agent 执行与训练，以统一接口和 LightningRL 信用分配处理复杂工作流 |
| **VerlTool · 2025-09** | [2509.01055](https://arxiv.org/abs/2509.01055) | 在 veRL 上统一工具管理、环境交互与异步 rollout |
| **AgentGym-RL / ScalingInter-RL · 2025-09** | [2509.08755](https://arxiv.org/abs/2509.08755) | 多环境训练框架；先限制交互长度，再逐步扩大 horizon，平衡训练稳定性与探索 |
| **AgentRL · 2025-10** | [2510.04206](https://arxiv.org/abs/2510.04206) | 异步多轮多任务训练，cross-policy sampling 与 task advantage normalization |
| **Agent-R1 · 2025-11；2026 修订** | [2511.14460](https://arxiv.org/abs/2511.14460) | 以 step-level trajectory representation 和灵活上下文管理统一 agent RL 接口，兼容不同优化器 |
| **MARTI · 2025 公开框架 / 2026 会议论文** | [OpenReview](https://openreview.net/forum?id=E7jZqo0A50)；[代码](https://github.com/TsinghuaC3I/MARTI) | 集中组织多 agent 交互与奖励、分布式训练策略，支持动态工作流；当前项目还包括异构训练与树搜索扩展 |

AgentRL 的 **task normalization**、Dr.MAS 的 **agent normalization**、GiGPO 的 **state grouping** 针对的是不同条件变量。可以放在“训练数据异质性”这一共同视角下比较，但不能因为都做归一化就认为它们等价。

MARTI 原始框架此次未确认独立 arXiv 编号，保留官方 OpenReview 链接。其后续 [MARTI-MARS² 技术报告](https://arxiv.org/abs/2602.07848) 与 [MARS² 论文](https://arxiv.org/abs/2604.14564)各有条目，不能拿它们替换原框架论文的编号。

## 9. 实际阅读过的 related work 如何串成文献线

以下“线”表示主题与引用上的联系，不是声称每篇都直接继承前一篇的算法。

| 阅读起点及来源 | related work 提供的线索 | 本次据此扩展的方向 |
| --- | --- | --- |
| [Dr.MAS §2](./DrMAS.pdf#page=2) | 单模型组优化、agent RL、多 agent RL、训练系统 | 形成最初分类，并重点核对 SPIRAL、MARSHAL、AT-GRPO |
| [ExGRPO §2](https://arxiv.org/html/2510.02245) | RLVR 的 off-policy 扩展与 experience-based RL | RePO、ReMix、RLEP、TOPR、GUI-ARPO |
| [GiGPO §2](https://arxiv.org/html/2505.10978) | 长程交互训练、组相对方法与细粒度信用分配 | LOOP、RAGEN，以及后续 GraphGPO/GRAFT 的比较问题 |
| [SORL §2](https://arxiv.org/html/2511.20718) | 工具 agent、off-policy 更新、多轮信用分配 | 把经验复用和多轮更新稳定性连接起来 |
| [C3 §7](https://arxiv.org/html/2603.06859) | 团队奖励、角色贡献、消息反事实与状态恢复 | MARFT、SHARP、CCPO、COSAC，并对照 Dr.MAS |
| [MARS² 附录 A](https://arxiv.org/html/2604.14564) | RL 后训练、tree-search RL、多 agent 协作 | 将“多角色训练”扩展到“多策略共同探索搜索树” |
| [OpenClaw-RL §5](https://arxiv.org/html/2603.10165) | agentic RL、过程奖励、在线蒸馏及基础设施 | RLAnything，以及从离线轨迹学习到持续交互学习 |
| [SAO §5](https://arxiv.org/html/2607.07508) | 组优化与异步 RL 的同步屏障、staleness | 明确单 rollout/value model 路线与 group-based 路线的区别 |

补充两个综述入口，用于继续扩展目录：

- [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547)：按 agent 能力和应用分类，适合查环境、工具、任务及框架。
- [From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2604.09459)：按信用粒度和估计方法组织，适合追踪 token / step / turn / multi-agent credit assignment。

综述用来找线索，具体方法判断仍以各篇原文为准。例如，有论文自己的 related work 会把范围宽泛的前序方法概括成“静态”或“单场景”，不应未经核对就转写成领域共识。

## 10. 如果要写 related work，建议用这四段结构

### 10.1 组相对优化与经验复用

先用 GRPO/RLOO 交代无需额外价值网络的背景，再简述 Dr. GRPO、DAPO、GSPO 从不同角度改进更新。随后加入 **ExGRPO、RePO、ReMix、RLEP、TOPR**：近期研究不只关心一次更新怎样计算，也关心如何复用昂贵 rollout、选择有价值经验并控制 off-policy 偏差。明确这部分主要来自 reasoning RLVR，是 agent 训练的优化基础。[ExGRPO](https://arxiv.org/abs/2510.02245)、[RePO](https://arxiv.org/abs/2506.09340)、[ReMix](https://arxiv.org/abs/2507.06892)。

### 10.2 多轮 agent 的信用分配与训练稳定性

以 **LOOP、Search-R1、WebAgent-R1** 说明优化对象从完整回答转为环境中的连续行动。再用 **GiGPO、SPA-RL、ARPO、GraphGPO、CW-GRPO** 对比状态分组、学习进展模型、局部探索、轨迹图和 judge 评分等信用分配路线。稳定性部分连接 **RAGEN、SimpleTIR、SORL、SAO**，分别讨论多轮动力学、工具反馈、离策略权重及异步采样。[GiGPO](https://arxiv.org/abs/2505.10978)、[GraphGPO](https://arxiv.org/abs/2605.26684)、[SORL](https://arxiv.org/abs/2511.20718)、[SAO](https://arxiv.org/abs/2607.07508)。

### 10.3 多 agent 联合训练、角色尺度与贡献归因

先介绍 **MAPoRL、MAGRPO、MHGPO、ReMA、MATPO、M-GRPO** 如何组织协作和训练，再区分两条线：**SPIRAL、MARSHAL、Dr.MAS** 涉及角色条件优势及稳定性；**SHARP、C3、CCPO、COSAC** 更直接研究个体或消息的增量贡献。**AT-GRPO** 连接角色/轮次分组与训练系统，**MARS²** 则把联合训练扩展到共享搜索树。这样能明确 Dr.MAS 的位置，也避免将优势归一化误写成完整的因果信用分配。[Dr.MAS](https://arxiv.org/abs/2602.08847)、[SHARP](https://arxiv.org/abs/2602.08335)、[C3](https://arxiv.org/abs/2603.06859)、[CCPO](https://arxiv.org/abs/2603.21563)。

### 10.4 框架与持续学习

最后简述 **Agent Lightning、VerlTool、AgentGym-RL、AgentRL、Agent-R1、MARTI** 如何连接环境、工作流和训练，并补充 **CoEvolve、RLAnything、OpenClaw-RL** 对任务分布、奖励模型和在线反馈的动态优化。这里只讨论与论文实现或问题设定相关的部分，避免将框架名称全部堆入算法段落。[AgentRL](https://arxiv.org/abs/2510.04206)、[CoEvolve](https://arxiv.org/abs/2604.15840)、[RLAnything](https://arxiv.org/abs/2602.02488)、[OpenClaw-RL](https://arxiv.org/abs/2603.10165)。

## 11. 优先阅读清单

如果先读 12 篇，建议按这三组走：

1. **单模型与经验利用：Dr. GRPO → ExGRPO → RePO → SORL。** 比较损失偏差、经验筛选、回放策略与离策略稳定性。
2. **单 agent 步骤学习：GiGPO → ARPO（Reinforced）→ GraphGPO → SAO。** 比较局部信用、探索、图结构和异步训练；SAO 也能帮助理解 critic-free 路线的取舍。
3. **多 agent：M-GRPO → MARSHAL → Dr.MAS → C3。** 比较层级协作、角色归一化、梯度尺度和消息级反事实；随后补读 SHARP、CCPO、AT-GRPO。

对每篇论文记录以下字段，就能形成自己的 related work 表：**训练对象；是否共享参数；rollout/group 定义；奖励来源；信用粒度；是否增加 critic/judge/分支采样；off-policy 处理；额外计算预算。**

## 12. 版本与名称校对

- **ExGRPO** 的原文是 *ExGRPO: Learning to Reason from Experience*，2025-10 首发，2026-02 修订。不能因其 2026 会议年份而忽略 2025 首发时间。
- **C3** 初版标题为 *Contextual Counterfactual Credit Assignment for Multi-Agent Reinforcement Learning in LLM Collaboration*；当前 v3 为 *The Trace Is the State: Exact Credit Assignment for LLM Agent Teams*。本表概括当前版本；对照初版请使用 [v1](https://arxiv.org/abs/2603.06859v1)。
- **MHGPO** 当前标题扩展为 *End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning*；[v1](https://arxiv.org/abs/2506.02718v1) 对应 Dr.MAS 引用的旧标题。
- **AT-GRPO / PettingLLMs** 的 [v1](https://arxiv.org/abs/2510.11062v1) 标题是 *Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs*，当前标题为 *Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs*。
- **Agent-R1** 初版是 *Training Powerful LLM Agents with End-to-End Reinforcement Learning*，当前版本强调 *A Unified and Modular Framework for Agentic Reinforcement Learning*。
- **SHARP ≠ SHARPO；ARPO（Replay）≠ ARPO（Reinforced）；MAGRPO ≠ M-GRPO。** 搜索时优先用 arXiv 编号定位。

## 13. 完整题名索引（便于检索与引用）

下列按 arXiv 编号排序；各方法的机制与关系见上面的分类表。MARTI 原始论文使用 §8 的 OpenReview 入口。

- [Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/abs/2502.01600) — `2502.01600`。
- [MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](https://arxiv.org/abs/2502.18439) — `2502.18439`。
- [ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2503.09501) — `2503.09501`。
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) — `2503.09516`。
- [Tapered Off-Policy REINFORCE: Stable and efficient reinforcement learning for LLMs](https://arxiv.org/abs/2503.14286) — `2503.14286`。
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) — `2503.14476`。
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) — `2503.20783`。
- [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) — `2504.13958`。
- [MARFT: Multi-Agent Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.16129) — `2504.16129`。
- [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073) — `2504.20073`。
- [Group-in-Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2505.10978) — `2505.10978`。
- [Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment](https://arxiv.org/abs/2505.11821) — `2505.11821`。
- [ARPO:End-to-End Policy Optimization for GUI Agents with Experience Replay](https://arxiv.org/abs/2505.16282) — `2505.16282`。
- [WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2505.16421) — `2505.16421`。
- [SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](https://arxiv.org/abs/2505.20732) — `2505.20732`。
- [End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](https://arxiv.org/abs/2506.02718) — `2506.02718`。
- [RePO: Replay-Enhanced Policy Optimization](https://arxiv.org/abs/2506.09340) — `2506.09340`。
- [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2506.24119) — `2506.24119`。
- [Squeeze the Soaked Sponge: Efficient Off-policy Reinforcement Finetuning for Large Language Model](https://arxiv.org/abs/2507.06892) — `2507.06892`。
- [RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning](https://arxiv.org/abs/2507.07451) — `2507.07451`。
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) — `2507.18071`。
- [Agentic Reinforced Policy Optimization](https://arxiv.org/abs/2507.19849) — `2507.19849`。
- [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680) — `2508.03680`。
- [LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.04652) — `2508.04652`。
- [Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL](https://arxiv.org/abs/2508.13167) — `2508.13167`。
- [VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use](https://arxiv.org/abs/2509.01055) — `2509.01055`。
- [SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning](https://arxiv.org/abs/2509.02479) — `2509.02479`。
- [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547) — `2509.02547`。
- [AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2509.08755) — `2509.08755`。
- [ExGRPO: Learning to Reason from Experience](https://arxiv.org/abs/2510.02245) — `2510.02245`。
- [AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework](https://arxiv.org/abs/2510.04206) — `2510.04206`。
- [Multi-Agent Tool-Integrated Policy Optimization](https://arxiv.org/abs/2510.04678) — `2510.04678`。
- [In-the-Flow Agentic System Optimization for Effective Planning and Tool Use](https://arxiv.org/abs/2510.05592) — `2510.05592`。
- [CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards](https://arxiv.org/abs/2510.08529) — `2510.08529`。
- [Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs](https://arxiv.org/abs/2510.11062) — `2510.11062`。
- [MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs](https://arxiv.org/abs/2510.15414) — `2510.15414`。
- [Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO](https://arxiv.org/abs/2511.13288) — `2511.13288`。
- [Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2511.14460) — `2511.14460`。
- [Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization](https://arxiv.org/abs/2511.20718) — `2511.20718`。
- [RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System](https://arxiv.org/abs/2602.02488) — `2602.02488`。
- [MARTI-MARS²: Scaling Multi-Agent Self-Search via Reinforcement Learning for Code Generation](https://arxiv.org/abs/2602.07848) — `2602.07848`。
- [Who Deserves the Reward? SHARP: Shapley Credit-based Optimization for Multi-Agent System](https://arxiv.org/abs/2602.08335) — `2602.08335`。
- [Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems](https://arxiv.org/abs/2602.08847) — `2602.08847`。
- [The Trace Is the State: Exact Credit Assignment for LLM Agent Teams](https://arxiv.org/abs/2603.06859) — `2603.06859`。
- [OpenClaw-RL: Train Any Agent Simply by Talking](https://arxiv.org/abs/2603.10165) — `2603.10165`。
- [Counterfactual Credit Policy Optimization for Multi-Agent Collaboration](https://arxiv.org/abs/2603.21563) — `2603.21563`。
- [From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2604.09459) — `2604.09459`。
- [Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](https://arxiv.org/abs/2604.14267) — `2604.14267`。
- [MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](https://arxiv.org/abs/2604.14564) — `2604.14564`。
- [CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](https://arxiv.org/abs/2604.15840) — `2604.15840`。
- [COSAC: Counterfactual Credit Assignment in Sequential Cooperative Teams](https://arxiv.org/abs/2604.17693) — `2604.17693`。
- [Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2605.26684) — `2605.26684`。
- [TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.32017) — `2606.32017`。
- [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508) — `2607.07508`。
- [Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning](https://arxiv.org/abs/2609.28963) — `2609.28963`。
- [Targeting Pivotal Decisions for Credit Assignment in Agentic Reinforcement Learning](https://arxiv.org/abs/2609.36178) — `2609.36178`。
- [SHARPO: Segment-Level Credit Assignment for Agentic Reinforcement Learning](https://arxiv.org/abs/2610.00838) — `2610.00838`。
