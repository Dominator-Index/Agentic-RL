# 01｜Harness Engineering：研究地图与 Agentic RL 的关系

整理日期：2026-10-08。本篇是本次文献筛选的横向导读，具体来源、会议状态、PDF 和代码入口见 [专题索引](./README.md) 与 [录用核对](./02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)。

## 1. Harness Engineering 研究的到底是什么

同一个 LLM，在不同的上下文组织、工具接口、执行循环、验证与错误恢复机制下，会形成不同的 agent。Harness 是实现这些机制的外部执行系统；harness engineering 研究如何设计、评估和改进这一系统。

研究对象可以是 system prompt，也可以是 Python 程序、调用图、工具定义、middleware、记忆结构、验证关卡、预算分配与停止规则。它与 context engineering 有交集，但后者主要关注模型接收的信息；harness 还决定系统能执行哪些操作、如何维护状态、怎样获得反馈、何时继续或终止。

本类别不因为论文写了“使用 lm-evaluation-harness”就将它归入 harness engineering；那可能仅是使用现成评测工具。

## 2. 用与 Agentic RL 一致的符号理解

下面是为对比而建立的统一记号，**不是声称所有论文都使用同一个公式或 setup**。

设任务为 $x$，底层 LLM 为 $\pi_\theta$，harness 的程序与配置为 $H_\phi$。这里 $\phi$ 可以是源码和文本，不要求是可微向量。

在第 $t$ 个 agent turn，系统有历史 $h_t$、环境观察 $o_t$ 和外部状态／记忆 $m_t$。harness 构造实际输入：

$$
c_t=H_\phi^{\mathrm{context}}(x,h_t,o_t,m_t),
\qquad
a_t\sim\pi_\theta(\cdot\mid c_t).
$$

harness 执行工具、处理反馈并更新状态：

$$
(o_{t+1},m_{t+1},h_{t+1})
=\operatorname{Execute}_{\phi}(a_t,o_t,m_t,h_t).
$$

整个系统的任务表现可以记为：

$$
J(\theta,\phi)
=\mathbb{E}_{x,\tau\sim\operatorname{Agent}(\theta,\phi)}
\bigl[R(x,\tau)-\lambda C(\tau)\bigr],
$$

其中 $C$ 是预算成本的抽象表示，具体可以是 token、工具调用或时间。**这一目标只是对比视角；不是每篇论文都采用加权成本目标。**

| 优化方式 | 通常固定什么 | 实际更新什么 | 例子 |
|---|---|---|---|
| Agentic RL | 任务接口及大部分运行逻辑 | $\theta$：策略模型参数 | GiGPO、Dr.MAS |
| Prompt／context 演化 | 基础模型参数 | $\phi$ 中的提示、经验与记忆 | GEPA、ACE |
| Harness 程序搜索 | 基础模型参数 | $\phi$ 中的源码、组件、工作流 | DGM、Meta-Harness、AutoSaddler、CE-Graph |
| Harness 生成器训练 | 可以固定被辅助的执行模型 | 生成 $H_\phi$ 的 helper 参数 | JIT-Agent |
| Harness 辅助参数学习 | 给定一类 harness 机制 | 任务模型参数；有的方法还改系统组件 | OPHSD、Continual Harness 的扩展、EvoC2F |

所以“用了反馈”“不断变好”“称为 learning”都不足以判定一个方法在训练任务 LLM。需要看优化变量究竟是哪一个。

## 3. 已录用主会论文中，哪些最值得先读

### 3.1 VeRO：先定义怎样可靠地研究 harness 优化

[ICML2026-VeRO](./ICML2026-VeRO/README.md) 提供外层运行框架，让优化器能够修改目标 agent，同时保证版本、评测预算、执行轨迹和结果有统一记录；VeRO-Bench 则把“agent 优化 agent”本身变成实验任务。

它适合作为新研究的实验基础。需要区分 target agent 的解题能力和 optimizer 改进 target agent 的能力。仅报告最终成功率，无法知道优化器是否用了更多预算、过拟合开发样本或读取了不该用于选择候选的测试反馈。

### 3.2 AutoSaddler 与 CE-Graph：从失败中产生可复用的系统修改

[NeurIPS2026-AutoSaddler](./NeurIPS2026-AutoSaddler/README.md) 将失败轨迹诊断、组件级补丁和验证选择组织为改进循环。重点是从一次失败中找到能改善更多任务的修改，而不是只修好眼前这一条轨迹。

[ICML2026-CE-Graph](./ICML2026-CE-Graph/README.md) 将失败组织在 Failure Signature Space 中，针对反复出现的失败模式修改 workflow 图。它用过程信息决定“改哪种工作流结构”，不是用这些信息直接生成 token 级 policy-gradient advantage。

两者与你之前“全失败 rollout 也应有学习价值”的动机相关，但优化输出不同：**它们主要产出可复用的系统修改，你的原始设想产出训练样本内的更新掩码／信用信号。**

### 3.3 GEPA 与 DGM：提示演化和程序演化的两个基准

[ICLR2026-Oral-GEPA](./ICLR2026-Oral-GEPA/README.md) 在模块提示上做自然语言反思与进化搜索，保留对不同开发样本有优势的候选。它适合比较“从执行过程提炼规则”是否比只看标量分数更有效。

[ICLR2026-DGM](./ICLR2026-DGM/README.md) 则直接修改 agent 程序，维护多条演化路径，允许工具和上下文管理等机制改变。它拓宽了可优化对象，但实验评测与开放式探索的成本也需要计入。

两者都不应被描述为“换一种 GRPO 分组方法”。

### 3.4 OPHSD：harness 可以成为训练期的辅助流程

[NeurIPS2026-OPHSD](./NeurIPS2026-OPHSD/README.md) 将 harness 产生的帮助蒸馏到模型参数中，而不只是让推理时的外部系统更复杂。

论文 §3.2 区分了两个角色：当前学生参数 $\theta$ 驱动 harness 生成动态上下文；冻结基础模型参数 $\tilde\theta$ 在该上下文下给出 token 级监督。它用学生直接生成的轨迹参与训练，并把 harness 上下文作为停止梯度的目标。

因此，它也有 frozen LLM，但这个模型提供教师分布，**不是只输出“该更新哪个 turn／token”的 mask**。数学实例还把参考解答提供给 planner 作为特权信息；分类实例使用先前带标签的记忆。它的监督条件和“只有题目、最终对错奖励”的 RLVR setup 不相同。

若继续研究你的 assessor idea，应比较：是否需要参考解答、是否只在全失败时触发、增加多少模型调用、输出是概率分布还是掩码、最终改变了监督目标还是样本权重。

### 3.5 EvoC2F 与 MoMHa：执行可靠性和多目标取舍

[ICML2026-EvoC2F](./ICML2026-EvoC2F/README.md) 将工具计划编译成受约束的中间表示，并要求新技能通过验证后再进入技能库。它还明确训练 planner 和检索 router，不能整体归为固定模型的代码改写。

[NeurIPS2026-MoMHa](./NeurIPS2026-MoMHa/README.md) 同时考虑准确率、行为安全和 token 成本。它提醒我们：harness 变复杂带来的提高，可能伴随显著额外调用，评价不能只看最终成功率。

## 4. 其他论文补充了哪些维度

| 维度 | 论文 | 与上述主会工作的区别 |
|---|---|---|
| 完整历史作为搜索经验 | [Meta-Harness](./COLM2026-Meta-Harness/README.md) | 提议者可访问过往源码、分数和完整轨迹 |
| 可编辑组件和可验证修改预测 | [AHE](./arXiv2026-AHE/README.md) | 强调系统可观测性与组件级回退 |
| 同一条长交互中自适应 | [Continual Harness](./arXiv2026-Continual-Harness/README.md) | 不要求每次改进后重置环境 |
| 持续异质任务流上的选择 | [Adaptive Auto-Harness](./arXiv2026-Adaptive-Auto-Harness/README.md) | 既改 harness，也在解题时路由 |
| 多个互补搜索分支 | [Mixture of Self-Improving Branches](./arXiv2026-Mixture-Self-Improving-Branches/README.md) | 开发子集、提议策略和部署路由共同参与 |
| 按任务生成 harness | [JIT-Agent](./arXiv2026-JIT-Agent/README.md) | 把 harness 设计能力放进可训练 helper |
| 自然语言运行规则 | [NLAH](./arXiv2026-NLAH/README.md) | 由共享运行时解释文档，赋予执行语义 |
| 最小可递归编程原语 | [JAZ](./NeurIPS2026-TTCL-Workshop-JAZ/README.md) | 用统一 invoke 表达更复杂行为；属于 Workshop |
| 动作合法性与代码策略 | [AutoHarness](./arXiv2026-AutoHarness/README.md) | 搜索检查器乃至完整代码策略 |
| 分类法与工程实现全貌 | [七层综述](./preprint2026-Harness-Engineering-Survey/README.md)、[Code as Agent Harness](./arXiv2026-Code-as-Agent-Harness/README.md)、[源码研究](./arXiv2026-Harness-Anatomy/README.md) | 用于界定设计空间，不作为单一训练算法比较 |

## 5. 对 Agentic RL 研究的启发

以下是基于这些工作的研究判断，不是已证明的新方法或创新性结论。

**第一，先把参数优化和系统优化分开控制。** 当训练时同时换 context、tool schema、agent 数量和 loss，很难判断收益来自哪里。一个有解释力的起点是固定模型比较 harness，再固定 harness 比较训练方法，最后研究协同更新。

**第二，失败信息可以有两个出口。** 一条失败轨迹既能指导某次参数更新，也能指导跨任务复用的系统修补。值得研究何种失败属于模型能力不足，何种失败属于信息缺失、接口设计或运行规则问题；这需要实验诊断，不能仅凭 LLM 的解释断言原因。

**第三，与你的设想最重要的比较不只是成功率。** 额外 assessor 是否触发、是否读取参考解答、对失败样本的利用方式、总 token 与工具成本、对未见任务的迁移，都会影响结论。只有这些条件明确，才能判断一个简单方法是否真正比已有 harness 搜索或蒸馏更有价值。

[返回专题索引](./README.md)
