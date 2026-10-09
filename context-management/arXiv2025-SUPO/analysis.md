# SUPO：端到端摘要上下文管理与 GRPO 训练解读

> 论文：**Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management**，Miao Lu 等，arXiv:2510.06727v1，2025-10-08。本文依据本地 [SUPO.pdf](./SUPO.pdf) 正文、算法、附录 A–C 整理，核对日期 2026-10-07。
>
> SUPO = SUmmarization augmented Policy Optimization。一句话：**把摘要也视为同一个 policy 的动作，用最终任务奖励同时训练工具使用和历史压缩，从而让一次完整 rollout 跨越多个有限上下文窗口。**

## 1. 这篇文章在解决什么问题

多轮 agent 不断产生推理、工具调用和工具返回，完整历史很快超过训练能够处理的上下文长度。直接停止会做不完任务；简单截断会丢失必要信息；一直扩展原始上下文又增加训练成本。

SUPO 关心的是：能否每次只训练一段较短上下文，却让 agent 在同一个任务上持续进行更长时间的交互？答案是让模型自己生成摘要，以摘要连接前后两段，并让摘要生成动作参与 RL。

它主要属于**可训练的上下文管理**主线。虽然名称也是 PO，但其主要贡献不是像 GiGPO 那样把 trajectory advantage 细化成状态条件下的 step advantage。

## 2. Motivation：为什么摘要需要跟着 RL 一起学

一般的文字摘要可能保留“发生过什么”，却漏掉“下一步继续执行需要什么”。论文的 CodeGym 案例很直观：只记住已经计数多少次不够，还需要记住数组比较到了哪个索引，否则下一段可能重复或遗漏计算。

因此，摘要质量应由后续任务是否完成来约束。SUPO 不需要给每份摘要人工打分，而让最终奖励影响生成这份摘要的概率。摘要和工具使用共享模型参数，并非两个独立的 solver/verifier 模型。

触发摘要的**时机**主要由长度阈值控制；模型学习的是摘要**内容**。不要把它理解成已经学会了任意时刻主动决定何时压缩。

## 3. Setup：同题 G 个 rollout，但一个 rollout 有多个分段

沿用论文符号：

- $s_1$：原始任务 prompt。
- $G$：每道题独立采样的完整 rollout 数，实验为 8；对应我们之前说的 $N$ 条轨迹。
- $j\in\{1,\ldots,G\}$：完整 rollout 的编号。
- $I^j$：第 $j$ 个 rollout 实际生成摘要的次数。
- $i\in\{1,\ldots,I^j+1\}$：该 rollout 的分段编号。
- $R^j$：该完整 rollout 的最终任务奖励。

原文经常把一个分段也叫 trajectory。为避免误读，本文统一称它为“分段”。例如一个 rollout 经历两次摘要：

```text
同一题的一次完整 rollout j：
原始问题 → 交互分段 1 → 摘要 1
原始问题 + 摘要 1 → 交互分段 2 → 摘要 2
原始问题 + 摘要 2 → 交互分段 3 → 最终答案 → R^j
```

这三个分段前后依赖，只有一个最终奖励，并不是对同一道题重新独立作答三次。对于一个问题，保存的分段总数为：

$$
N_{\mathrm{seg}}=\sum_{j=1}^{G}(I^j+1).
$$

这里 $N_{\mathrm{seg}}$ 是本文为区分用途增加的记号；论文 §4.2 的 $N$ 是整个 prompt batch 上的分段总数。**独立采样数 $G$ 不随摘要分段数增加。**

## 4. 方法：摘要如何改变 state

普通多轮过程不断拼接历史：

$$
s_{t+1}=(s_t,a_t,o_t),\qquad a_t\sim\pi_\theta(\cdot\mid s_t).
$$

$s_t$ 是当前输入上下文，$a_t$ 是模型生成的推理和动作，$o_t$ 是工具观测。SUPO 设置长度阈值 $L$，达到阈值后追加摘要指令 $v_{\mathrm{sum}}$，让**同一个模型**输出摘要，然后把后续上下文重建为：

$$
s_{t+1}=(s_1,a_t)\quad\text{当 }a_t\text{ 是摘要时}.
$$

旧的长历史不再全部进入下一次调用；保留原始问题和这次摘要，再继续积累新的交互。

论文理论过程的工作上下文上界为：

$$
|s_t|+|a_t|\le L+2L_A+L_O+|v_{\mathrm{sum}}|.
$$

$L_A$、$L_O$ 分别限制单次模型输出和工具返回长度。这个上界说明压缩能限制每次调用的上下文，但阈值 $L$ 并不等于精确硬上限，工具返回可能造成超出。

**实现细节不能忽略：**Algorithm 2 检测到超长后，在摘要输入中丢弃刚刚导致超长的最后一对 action/observation，采用 $(s_t,v_{\mathrm{sum}})$。这与理论式 (1) 先保留该轮再追加指令略有区别，目的是避免巨大的工具返回挤掉摘要生成空间。论文没有在这里完整讨论有副作用工具的环境回滚；实现到可修改环境时需要另外核实环境状态与可见历史的一致性。

## 5. 为什么摘要也能用终局奖励训练

论文 Theorem 3.2 和附录 A.1 的核心是常规 score-function policy gradient。以下用更清晰的等价分段形式重写，避免正文展开式的边界索引歧义：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)],
\qquad
\nabla_\theta J(\theta)=
\mathbb E\left[R(\tau)\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
$$

动作序列既包括正常工具使用，也包括摘要生成。环境转移在给定动作后不显式依赖参数，所以对 rollout 概率求对数导数，只需对 policy 生成的动作求和。

按摘要边界把动作索引划分为集合 $\mathcal T_1,\ldots,\mathcal T_{I+1}$：

$$
\nabla_\theta J(\theta)=
\mathbb E\left[R(\tau)
\sum_{i=1}^{I+1}\sum_{t\in\mathcal T_i}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
$$

每段可用它实际见到的“问题＋上一段摘要＋本段历史”计算 log probability，然后累加梯度。无需把整个长 rollout 重新拼成一个巨大的训练序列。

**这也不是穿过离散摘要文本反向传播。**摘要在生成它的那一段作为 action 接收 policy-gradient 更新；复制到下一段时作为输入上下文，不应再作为新生成的 response 重复计入 loss。模型输出 token 可训练，工具返回和 prompt 不作为 policy 输出目标。

该定理说明的是分段计算方式；它不证明摘要无信息损失，也不保证能恢复完整历史策略的最优性能。

## 6. Advantage：SUPO 坚持按完整 rollout 分组

论文式 (3)：

$$
\widehat A^j=
\frac{R^j-\operatorname{mean}_{j'=1}^{G}R^{j'}}
{\operatorname{std}_{j'=1}^{G}R^{j'}}.
$$

一个 rollout 的所有分段、所有可训练输出 token，包括摘要 token，都共享这个 $\widehat A^j$。公式照原文；实现通常需对零标准差加稳定处理，不能在未核对代码时声称某个具体 epsilon 是作者设置。

它没有让不同摘要、不同工具动作得到不同的 state-conditioned advantage。**仍然是 rollout-level advantage。**

### 6.1 为什么不把所有分段当成 group 元素

如果每个分段复制一次它所属 rollout 的奖励，均值会变成：

$$
\mu_{\mathrm{seg}}=
\frac{\sum_{j=1}^{G}(I^j+1)R^j}
{\sum_{j=1}^{G}(I^j+1)},
\qquad
\mu_{\mathrm{rollout}}=\frac1G\sum_{j=1}^{G}R^j.
$$

区别在于：前者按分段数给完整 rollout 的奖励加权，摘要更多的 rollout 对 baseline 影响更大。论文式 (4) 将这种分段组归一化作为消融。

**举例，采用总体标准差便于手算：**同一道题两个 rollout，奖励分别为 1、0，分段数分别为 3、1。按完整 rollout 统计，奖励列表为 $[1,0]$，advantage 为 $[1,-1]$。按分段统计，列表为 $[1,1,1,0]$，成功 rollout 的 advantage 变为 $1/\sqrt3$，失败 rollout 变为 $-\sqrt3$。

这里没有新增信息，只是成功样本被重复计入统计量，就改变了更新权重。SUPO 用完整 rollout 作为统计单位，避免这类分段数影响。不过其 token 平均目标仍可能让长 rollout 提供更多 token 梯度，**不能说消除了所有长度相关加权。**

### 6.2 和 importance ratio 的关系

论文仍使用实际采样上下文中的 token ratio：

$$
\rho_{t,\ell}^{j}=\frac{
\pi_\theta(v_{t,\ell}^{j}\mid s_t^{j},v_{t,<\ell}^{j})}
{\pi_{\mathrm{old}}(v_{t,\ell}^{j}\mid s_t^{j},v_{t,<\ell}^{j})}.
$$

分子和分母条件于同一个实际上下文。摘要改变 rollout 后续使用的 $s_t$，但 ratio 仍然按各自样本计算。不存在要求不同分段 state 相同才能相加 loss 的限制；advantage 的分组统计和 ratio 是两个不同问题。

## 7. 训练目标与 overlong masking

用 $\mathcal U^j$ 表示 rollout $j$ 中全部可训练生成 token，$u$ 同时编码分段、轮次和 token 位置。论文式 (2) 可紧凑重写为：

$$
J_{\mathrm{SUPO}}(\theta)=
\mathbb E\left[
\frac{1}{\sum_{j=1}^{G}|\mathcal U^j|}
\sum_{j=1}^{G}m^j\sum_{u\in\mathcal U^j}
\min\left(
\rho_u^j\widehat A^j,
\operatorname{clip}(\rho_u^j,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})\widehat A^j
\right)\right].
$$

$m^j$ 根据终止原因定义：因达到最大步数或最大摘要次数仍未提交最终答案的 rollout 为 0，其余为 1。这里按正文对 overlong 的语义说明，避免只看式 (2) 的边界不等式误以为所有达到上限的样本都已经完成。

为什么要屏蔽？在学习早期，尝试长程搜索和摘要的样本可能因预算耗尽而失败；如果不断给这些行为负向更新，模型可能学成“尽快结束、不要摘要”，而不是学会更好的长程策略。论文消融观察到不加 mask 时摘要行为与摘要条件成功率退化。

注意三个区别：

1. 已经提交答案但答错的 rollout 不等于 overlong，仍可能得到负 advantage。
2. mask 是 rollout 级的截断样本过滤，不是 ECHO 的证据来源 token mask。
3. 式 (3) 仍以 $G$ 个 rollout 奖励统计优势；loss 屏蔽不等于已经把该样本从均值、方差中移除。

这是实践性训练选择，不是对原始期望奖励无偏性的额外保证。

## 8. 实验和 learning dynamics 证据

### 8.1 设置

| 项目 | CodeGym | BrowseComp-Plus |
|---|---|---|
| 基础模型 | Qwen2.5-32B-Instruct | Seed-OSS-36B-Instruct |
| 训练 / 测试 | 12,800 / 128 | 730 / 100 |
| GRPO 工作上下文 | 32K | 64K |
| SUPO 工作上下文 | 4K | 64K |
| SUPO 最多分段 | 8 | 3 |
| SUPO 名义有效长度 | 32K | 192K |

有效长度定义为 $L_{\mathrm{effect}}=L_{\mathrm{RL}}(S+1)$，是跨段预算口径，不意味着模型能够一次注意到全部原始历史。

共同设置：$G=8$，最多 100 步，摘要阈值为工作上下文的 95%，学习率 $10^{-6}$、常数调度，clip 下界参数 0.20、上界参数 0.28，不使用额外 entropy loss 或 KL loss。本文没有从已核对的原文确认具体 GPU 型号/张数或优化器名称，不默认补成 AdamW。

### 8.2 结果与消融

| 方法 | CodeGym | BrowseComp-Plus |
|---|---:|---:|
| GRPO | 44.5% | 39.0% |
| SUPO，无 overlong mask | 45.3% | 44.0% |
| SUPO，按分段计算 advantage | 42.1% | 49.0% |
| SUPO，完整设置 | 47.7% | 53.0% |

来源为原论文 Table 1。CodeGym 在相同名义有效长度下工作窗口更小；BrowseComp-Plus 则允许 SUPO 更多总交互预算。因此后者不是“完全相同计算量下只改变 loss”的对照。论文也明确说明，这是自定义 100 题 holdout，不能直接比较公开 BrowseComp 排行榜。

Appendix C.2 中，SUPO 在 64K 工作窗口下把测试时最多分段从训练的 3 增加到 24，准确率从 53% 到 60%。这里增加的是**同一个 rollout 可继续交互的分段预算**，不是独立采样 24 个答案做 best-of-N。

### 8.3 是否研究 learning dynamics

有针对性的动态观察：摘要触发比例、发生摘要条件下的成功率、工具调用次数，以及 overlong mask 消融引起的摘要行为退化。它解释了训练如何形成或抑制长程行为。

但它没有系统分析优化器谱性质、梯度噪声或曲率。它的动态分析更接近“奖励与截断处理如何改变探索和上下文管理行为”。案例显示训练后的摘要更能保留关键索引或证据；这些案例支持机制，但不构成摘要信息充分性的证明。

## 9. 与 GRPO、GiGPO、ECHO 的区别

| 方法 | 主要改动 | advantage / 信用粒度 | 历史处理 |
|---|---|---|---|
| GRPO | 组内相对终局奖励 | 一个 rollout 共享 advantage | 基础多轮版本累积历史 |
| GiGPO | 跨 rollout 聚合重复状态下的动作结果 | 轨迹信号加 anchor-state 条件的步骤信号 | 分组本身不解决长上下文压缩 |
| SUPO | 可训练摘要、分段训练、rollout 统计与截断 mask | 每个完整 rollout 共享 advantage，包括摘要动作 | 滚动摘要替换旧历史 |
| ECHO，已归档 v5 | 带来源的逐轮记忆、选择、信用路由 | 对正 advantage 用来源相关 token mask | 保留可寻址来源索引 |

GiGPO 问的是“同类 state 下，哪个动作更值得强化”；SUPO 问的是“如何在有限上下文下继续行动，并训练负责压缩信息的动作”；ECHO 进一步问“压缩以后如何追溯哪些历史输出值得获得信用”。

SUPO 摘要文本也可能含网页文献编号，不能说它完全没有引用。与 ECHO 的关键区别是：**SUPO 的训练目标没有把这些文本引用转换成针对原始轮次生成 token 的信用选择规则。**

另一个容易混淆的点：SUPO 原论文 Table 1 的 53% 与 ECHO 中 SUPO 基线的 36.1% 属于不同模型、预算和实验配置，不能直接拿来判断复现是否下降。

## 10. 真正贡献与边界

真正贡献是把上下文压缩纳入 policy 的动作空间，并用分段梯度计算让现有短窗口 RL 基础设施承载更长的任务交互；配套的 rollout 级归一化和 overlong mask 则避免一些由分段、预算截断带来的训练问题。

边界包括：摘要可能丢失或歪曲证据；同一个终局优势无法区分好的摘要与无效搜索；增加交互预算也增加工具成本；固定阈值不等于学会最优压缩时机；屏蔽未完成样本可能减弱对无效长行为的惩罚。摘要中的多个事实也不自动成为可验证的中间 reward。

对我们此前分组方向的启发是：**先分清真正独立的采样单位与为训练而产生的数据分段，再定义统计量。**如果按分段重组 group，应明确是在有意改变 baseline，还是无意让长 rollout 的奖励被多次统计。进一步结合 GiGPO 类方法时，摘要相同也不自动代表底层环境状态或完整历史相同。

## 11. 来源和代码状态

- [arXiv 论文](https://arxiv.org/abs/2510.06727)：本文详细公式与结果以归档 PDF v1 为准。
- [OpenReview 投稿记录](https://openreview.net/forum?id=778Yl5j1TE)：检索到 ICLR 2026 在审稿件，但当前访问遇到 challenge，未核实最终决定，因此暂用 arXiv2025-SUPO 命名。
- [Hugging Face 页面](https://huggingface.co/papers/2510.06727) 的自动 GitHub / project 链接实际指向 [ReSpect](https://github.com/lil-lab/respect)，标题和作者不匹配，不能当 SUPO 官方代码。
- [ECHO 仓库](https://github.com/xiezijun714-lang/Echo) 曾公开 SUPO-style rolling-summary 基线说明，可作为相关实现线索；这是另一篇论文的实现，不等于 SUPO 作者原始代码。目前没有将其改名为 SUPO 克隆。
- 本次没有确认 SUPO 官方实现，所以没有创建冒充官方仓库的代码软链接。未运行训练实验。
