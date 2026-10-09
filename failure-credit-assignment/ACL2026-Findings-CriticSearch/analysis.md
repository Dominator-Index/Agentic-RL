# CriticSearch：用冻结 LLM 的事后判断补充搜索 turn 的信用

> 依据 [ACL 2026 Findings 论文](./CriticSearch.pdf) §2–5、附录 B–F，以及 [本地代码](./CriticSearch)。代码 commit：`2141a49a849ed81ee52c50df3643b246ffc16abc`。整理日期：2026-10-07。下文区分论文算法和当前公开实现；代码存在静态语法问题，见第 8 节。

## 1. 核心 idea 与 motivation

一次搜索问答可能包含三个搜索：第一个找到关键实体，第二个重复已有资料，第三个走错方向。终局正确并不说明三个搜索都好；终局错误也不说明三个搜索都无用。

GRPO 用一个终局相对分数训练整条回答，难以区分这些局部差异。CriticSearch 的做法是：**完整回答生成后，让一个冻结 LLM 同时看问题、完整轨迹和标准答案，回头判断各次搜索有没有价值，再把这个局部信号加到原来的全局优势上。**

它用推理开销换取更细的监督，无需为每个搜索再进行 Monte Carlo 续跑，也无需人工标注逐步奖励。

## 2. Setup：单个搜索 agent，不是多 agent 团队

同一道题 $q$ 采样 $G$ 条完整轨迹 $y_i$。每条轨迹重复：

1. policy 生成 reasoning 和搜索 query；
2. 搜索引擎返回检索内容；
3. policy 根据已有历史继续搜索或给出最终答案。

论文称这些交互为 turn。为避免原文 token 下标与 turn 下标混用，下文用 $t$ 表示交互 turn，用 $\ell$ 表示 token。

可训练的是 policy LLM；搜索引擎不是被 RL 更新的 agent，critique LLM 也被冻结。因此，存在两个 LLM 不等于训练了一个 MAGRPO 多智能体团队。

## 3. 算法关键步骤

1. **采样**：用行为策略为每题生成 $G$ 条搜索轨迹，记录工具调用与返回内容。
2. **终局评分**：检查答案正确性和格式，计算每条轨迹的全局奖励。
3. **组内比较**：在同题 $G$ 条轨迹之间计算全局 GRPO advantage。
4. **事后评价**：冻结 critique LLM 读取完整轨迹和标准答案，输出每次有效搜索的 Good/Bad 序列。
5. **局部归一化**：将 Good 映射为 1、Bad 为 0，在一条轨迹内部按分数总和归一化。
6. **融合**：把局部值和全局 advantage 加权组合，并对齐到相应 turn 的 policy token。
7. **更新 policy**：使用 clipped policy objective；检索内容作为上下文保留，但不作为 policy 输出计算 loss；冻结评估器不更新。

评估器在论文算法中对各条轨迹进行评价，没有“只有整组全失败才调用”的门控。

## 4. 关键公式：从终局奖励到混合 advantage

### 4.1 全局奖励与相对优势

论文 Eq. (4) 的奖励可以等价写成：

$$
r_i^\tau=(1-\lambda_f)\mathbf 1\{\text{答案正确}\}
+\lambda_f\mathbf 1\{\text{格式正确}\}.
$$

$r_i^\tau$ 是整条轨迹的分数，$\lambda_f$ 是格式权重，论文取 0.2。因此全都答错不必意味着奖励全为零：格式正确的错误答案仍可能得到 0.2。真正令组内 advantage 消失的是组内总分相同。

$$
A_i^\tau=\frac{r_i^\tau-\operatorname{mean}_j(r_j^\tau)}
{\operatorname{std}_j(r_j^\tau)+\epsilon}.
$$

比较对象是同题的完整轨迹。这个值不区分同一轨迹内部的搜索质量；$\epsilon$ 表示数值稳定项。

### 4.2 冻结模型的局部分数

$$
\{\ell_{i,t}\}_t=C_\phi(q,y_i,o_{\mathrm{gold}}),
\qquad r^a_{i,t}=\mathbf 1\{\ell_{i,t}=\mathrm{Good}\}.
$$

$C_\phi$ 是冻结评估器，$o_{\mathrm{gold}}$ 是标准答案。它拥有 policy 决策时不可见的未来轨迹和答案，所以是 hindsight / privileged supervision，而不是在线价值预测器。

$$
A^a_{i,t}=\frac{r^a_{i,t}}{\sum_{u=1}^{T_i}r^a_{i,u}+\epsilon}.
$$

这是论文 Eq. (7)。**它不减均值、不除标准差，也不跨轨迹对同状态动作分组。** 它在单条轨迹内部把约一单位正向信用分给 Good turn。名字叫 advantage，但它不是对真实 $Q-V$ 的无偏估计。

若分数为 $(1,0,1)$，局部值约为 $(0.5,0,0.5)$。若全为 Bad，局部值全为零。Bad 对应的是“没有局部正向奖励”，不是自动给负 advantage。

### 4.3 混合与更新

$$
\widetilde A_{i,t}=\alpha A^a_{i,t}+(1-\alpha)A_i^\tau.
$$

论文 Eq. (8) 中 $\alpha$ 平衡局部监督和终局目标；主实验取 0.25。如果全局优势为 $-1$，局部值为 $0.5$，混合后仍是 $-0.625$。因此 Good 也不保证最终受到正向强化。

令 $m_{i,\ell}$ 标识 policy 生成的 token，$t(\ell)$ 为 token 所属 turn，一个清晰的 token 展开写法是：

$$
J(\theta)=\mathbb E\!\left[\sum_{i,\ell}m_{i,\ell}
\min\!\left(\rho_{i,\ell}\widetilde A_{i,t(\ell)},
\operatorname{clip}(\rho_{i,\ell},1-\varepsilon,1+\varepsilon)
\widetilde A_{i,t(\ell)}\right)\right]-\beta D_{\mathrm{KL}}.
$$

这里省略 batch/token 平均常数；$\rho$ 仍是同一历史下新旧策略对该 token 的概率比，$m$ 排除工具输出。改变的是信用信号，并没有改写重要性比率。turn 内广播同一个分数，不等于每个 token 都有独立的贡献估计。

## 5. 为什么能利用失败轨迹

如果组内全局分数相同，则 $A_i^\tau=0$。只要评估器认为某些搜索有用，这些位置仍有：

$$
\widetilde A_{i,t}=\alpha A^a_{i,t}>0.
$$

正向信号来自额外的语义监督，不来自对零 advantage 乘 mask。其价值是保留失败回答中的有效检索行为，同时利用全局项避免完全偏离最终答对的目标。

与 GiGPO 不同，它不要求采到相同 state；与 Monte Carlo 归因不同，它没有通过干预动作、重新执行来验证因果贡献。它依赖评估器判断的可靠性。

## 6. 证据与真实贡献

论文在 HotpotQA 训练，并在 HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle 评估。以 3B 模型为例，Table 2 中平均 EM 为 34.3；StepSearch-instruct 为 29.6。不要把不同初始化、额外模型开销等差异压缩成单一归因结论。

§5 与附录检查混合权重、评估器大小、去掉标准答案、与人工标注的一致性，以及固定 rollout 下的优势方差。附录 C.2 还比较 $A^\tau$、$0.75A^\tau$、$0.75A^\tau+0.25A^a$，避免把单纯缩小全局项误当成全部方差收益。

真正贡献是将冻结模型的事后搜索判断接入 policy 优化，并验证局部反馈的作用。没有给出语义评分等价于真实因果信用或任务梯度无偏的定理。

## 7. 局限与我们方案的关系

- 标准答案和完整轨迹给了评估器额外信息；无标准答案时需单独验证。
- 冗余搜索、必要重复搜索、失败但合理的尝试，可能被评估器混淆。
- 无额外环境 rollout 不代表没有评估器推理和显存成本。
- 全 Bad 时局部信号仍为零；对每条轨迹都评价的成本随 rollout 数增长。
- 实验限于搜索任务，不能直接声称已经解决多智能体的信息采纳责任。

对我们的 idea：冻结 LLM 的 turn 评分不是新点；可研究的部分是全失败团队中“谁正确纠错、谁忽略纠错”的跨 agent 责任，以及条件触发能否节省成本。

## 8. 代码核对与可运行状态

| 位置 | 核对内容 |
|---|---|
| [评估器 prompt](./CriticSearch/search_r1/contribute_reward/reward_until.py) | 明确只评价有效 search，先输出分析，再输出分数；不让 LLM 评价 answer action |
| [评分对齐](./CriticSearch/verl/trainer/main_ppo_format_contribute.py) | `align_action_and_contribute_score` 对齐 turn，处理无效调用、分数缺失与 answer 分数回退 |
| [启动脚本](./CriticSearch/train.sh) | `trajectory_advantage_weight=0.75`、`action_adv_estimator=trajectory_average`、`advantage_mask=false` |
| [局部优势分发](./CriticSearch/verl/trainer/ppo/ray_trainer_contribute.py) | 存在多种实验分支，包括 `trajectory_average`；不能把所有分支视作论文主算法 |
| [核心算法文件](./CriticSearch/verl/trainer/ppo/core_algos.py) | 当前 commit 的 Python AST 解析失败：第 278 行开始的三引号字符串未闭合，文件止于 295 行 |

此外，trainer 引用的 `compute_grpo_trajectory_average_advantage` 在当前核心文件中没有找到有效定义。因此这里只能确认 prompt、配置及调用意图，不能声称已经从这份代码验证完整训练闭环，更不能称它直接可运行。未修改上游源码、未安装训练环境。

## 9. 一句话记忆

**生成后再请冻结 LLM 回看哪些搜索有用，把这些局部正向分数与 GRPO 的终局相对优势混合。**
