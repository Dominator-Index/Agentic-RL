# 01. 四篇论文的核心 idea、算法步骤与归因方式对比

> 阅读顺序：先看本页，再看各论文 `analysis.md`。依据本地 PDF 与已检出代码，整理日期：2026-10-07。

## 1. 它们都在补信用信号，但补法不同

| 方法 | 核心 idea | 额外信息从哪里来 | 核心操作 |
|---|---|---|---|
| [ProGPO](./arXiv2026-ProGPO/analysis.md) | 全失败时，用探索覆盖代理补上零优势 | 原轨迹的观测变化历史 | 条件切换评分来源 |
| [CriticSearch](./ACL2026-Findings-CriticSearch/analysis.md) | 回看完整轨迹，判断哪些搜索有用 | 冻结 LLM、标准答案和未来轨迹 | 局部正向分数与全局优势混合 |
| [MulFeRL](./arXiv2026-MulFeRL/analysis.md) | 全失败时请求反馈，重试直到出现可验证进步 | 反馈后的新回答及 verifier | 按组状态在 GRPO/FCO 之间路由 |
| [C3](./arXiv2026-C3/analysis.md) | 同一协作历史下更换消息，看下游实际结果 | 候选消息的反事实续跑回报 | 同历史候选的价值差与 LOO 基线 |

因此不能都概括成“对原来的 N 条轨迹重新分组”。ProGPO 换分数，CriticSearch 加外部评分，MulFeRL 改采样上下文后重试，C3 主动组织共享前缀的分支采样。

## 2. 统一符号，避免把三种 turn 混淆

- $x$：一个问题。
- $N$：一组完整候选轨迹数；原文有的写 $G$ 或 $K$。
- $R_i$：第 $i$ 条完整轨迹的终局奖励。
- $t$：轨迹内部交互步骤；**MulFeRL 中的 $t$ 特指整组回答的重试轮次**。
- $u$：C3 中某个角色的一次消息决策点；$h_u$ 包含它所需的历史。

| 方法 | 单/多 agent | 被分配分数的单位 | 同一组比较什么 |
|---|---|---|---|
| ProGPO | 主实验为单 agent 环境交互 | 轨迹，或基础算法的步骤组 | 同题轨迹或状态/历史子组里的 progress |
| CriticSearch | 单个搜索 agent，外加冻结评估器 | 搜索 turn | 全局项跨轨迹；局部项只在一条轨迹内按 Good 总数归一化 |
| MulFeRL | 单策略，多轮反馈重试 | 完整回答/前后回答对 | 当前混合组；或反馈前失败与反馈后成功 |
| C3 | 多 agent 协作 | 角色在一个决策点的整条消息 | 精确相同前缀下的多个候选消息 |

使用两个 LLM 不自动构成多 agent RL；使用 token loss 也不自动构成 token 级独立归因。

## 3. ProGPO 的算法主线

```text
原 rollout → 统计首次访问比例 P
→ reward 有区分度：原 GRPO/GiGPO/HGPO
→ reward 全零且 P 有区分度：归一化 P 作为辅助优势
→ 原 policy 更新
```

轨迹版本的核心是：

$$
P_i=\frac1{T_i}\sum_t\mathbf 1\{o_{i,t+1}\notin\{o_{i,1},\ldots,o_{i,t}\}\},
\qquad A_i^{\mathrm{fallback}}=\lambda\frac{P_i-\bar P}{\sigma_P+\epsilon}.
$$

$P_i$ 只衡量该轨迹是否访问了新观测，不代表距离任务目标更近。组内 progress 全相同仍没有信号。详细阈值及基础算法接法见单篇分析。

**代码差异：** 当前 ProGPO C2 用相邻 anchor 是否变化，回到旧观测也可能计分；这与论文首次访问集合不同，不能混写。

## 4. CriticSearch 的算法主线

```text
原搜索轨迹 → 终局答案与格式分 → GRPO 全局优势
          → 冻结 LLM + 标准答案 → 每次搜索 Good/Bad
→ 轨迹内局部归一化 → 两种优势混合 → 更新 policy
```

$$
A^a_{i,t}=\frac{r^a_{i,t}}{\sum_u r^a_{i,u}+\epsilon},
\qquad \widetilde A_{i,t}=\alpha A^a_{i,t}+(1-\alpha)A_i^\tau.
$$

局部 $r^a$ 是 0/1 判断；不减均值，也不要求相同 state。全局项为零时，Good turn 仍能产生正向信号。Bad 的局部项为零，不是独立负惩罚。

**代码状态：** 当前公开 commit 的 `core_algos.py` 存在未闭合三引号，训练分支还引用未找到有效定义的局部优势函数。本文按论文解释公式，不声称现有仓库已可直接复现。

## 5. MulFeRL 的算法主线

```text
当前上下文采样 N 条完整回答
  混合成功/失败 → GRPO
  初始全成功 → 跳过
  全失败 → 语言反馈 → 注入上下文 → 再生成 N 条
      混合 → GRPO
      全成功 → 和前一失败组做 FCO
      全失败 → 再试，耗尽预算则跳过
```

FCO 用成功回答的参考 log-ratio 减去失败回答的参考 log-ratio，再做 logistic loss。正负回答的上下文可以不同，因而它不是同一个 state 下的动作比较。注入的反馈 token 不计算输出 loss，但仍影响后续条件概率。

**它不是给原始失败轨迹直接输出 mask/credit。** 必须有重试后的 verifier 成功，始终失败就不更新。

## 6. C3 的算法主线

```text
记录团队消息前缀 → 同一决策点生成不同候选消息
→ 每个候选交给下游 agent 续跑 → 终局奖励均值 Q
→ 减去其他候选的 LOO 基线 → 用消息优势更新
```

$$
\widehat Q_j=\frac1{c_j}\sum_mR_{j,m},\qquad
\widehat A_j=\widehat Q_j-
\frac{\sum_{k\ne j}c_k\widehat Q_k}{\sum_{k\ne j}c_k}.
$$

$c_j$ 是候选 $j$ 的续跑次数。历史可恢复、采样独立、预算非自适应等条件支持论文的局部无偏性结论。它没有训练一个预测这些 $Q$ 的 critic。

**预算细节：** 论文两角色训练用 2 条 Reasoner 消息，每条接 4 条 Actor 回答，得到 8 个终局结果，共 10 条消息；基线八条独立团队轨迹生成 16 条消息。不能笼统说 C3 一定要在原八条轨迹之外再增加一套 rollout。

## 7. 全失败时，谁真的还能学？

这里假定终局二元奖励且组内全为 0；格式分、其他辅助项和跨题 baseline 需单独检查。

| 方法 | 不产生任何新成功时能否有奖励驱动更新 | 需要满足什么 |
|---|---|---|
| 原始组内 GRPO | 不能 | 组内 advantage 为零 |
| ProGPO | 可以 | progress 在比较组中有差异 |
| CriticSearch | 可以 | 评估器至少认定某些搜索为 Good |
| MulFeRL | 不能保证；预算内始终失败就跳过 | 反馈重试产生 verifier 成功 |
| C3 | 不能保证；所有相关续跑回报全零仍为零 | 不同候选的续跑回报出现差异 |
| 我们的 frozen assessor 设想 | 可以，但必须另给非零信用 | 不能只把原零 advantage 乘 mask |

“不需要额外环境 rollout”和“不需要额外模型推理”是两件事。CriticSearch 有评估器推理；MulFeRL 有反馈和重新采样；C3 有分支续跑但能复用前缀；ProGPO 主要增加观测统计。

## 8. 对当前多 agent idea 的判断

我们的目标可以准确写成：**只对全失败的团队组启用冻结评估器，在原始协作消息中定位可保留的纠错和应抑制的信息采纳错误，并给出非零辅助更新信号。**

其中已有先例：ProGPO/MulFeRL 的全失败门控、CriticSearch 的冻结 turn 评分、C3 的多 agent 消息级归因。需要进一步论证的是跨 agent 协作依赖是否比独立 Good/Bad 判断更有用，以及在等成本下能否优于反馈重试和共享前缀续跑。

最小数学区别是：mask 决定位置，credit 决定方向；$M\times0=0$。若评估器只输出“这里值得学”，还需要定义究竟提高还是降低这些 token 的概率。

一个关键例子是“solver 犯错 → verifier 正确纠错 → solver 忽略意见”。最终全失败，却不应把三个 turn 当成同样责任。该例子可以同时作为评估器输入设计、归因标注和消融实验的出发点，不能仅凭直觉当成已验证结论。
