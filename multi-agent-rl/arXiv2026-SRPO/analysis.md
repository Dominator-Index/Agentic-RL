# SRPO：将一次团队决策作为一个联合更新单元

> 基于 [SRPO v2 本地论文](./SRPO.pdf) §3–5、附录 A–F 和流程示例。官方代码尚未确认，下文是论文与伪代码分析，不声称完成代码复核。注意 v1/v2 标题及元数据有变化，见 README。

## 1. 最值得抓住的贡献

之前讨论 GiGPO、HGPO、ProGPO 时，主要改变的是每个样本的 advantage；其概率比的定义保持不变。**SRPO 则改变什么算一个 policy action，以及这些 action 的概率比如何合并和裁剪。**

例如五个 search worker 从同一个共享上下文出发，并行查询，五份结果一起写入共享记忆，形成下一状态。如果每个回答独立 clip，一次团队决策会受到五个分开的限制。若把整条轨迹合成一个 action，又混入了读取新信息之后作出的其他决策。

SRPO 选择中间粒度：一次状态转移所共同消费的输出，组成一个 active set；集合共享优势、合并 log-ratio，再 clip 一次。

## 2. 三个不同的“集合”，不要混淆

| 概念 | 成员是什么 | 用途 |
|---|---|---|
| 同题 rollout group | 同一道题的多条轨迹 | 提供不同结果用于相对比较 |
| advantage comparison group | 来自不同轨迹的可比 decision events | 计算回报均值和标准差 |
| active set $S_e$ | 同一次决策中共同消费的多个新输出 | 构造一个联合 action 与一个 clipping 单元 |

因此 active set 不是跨 trajectory 搜索相同 state 得到的组。它首先由系统的执行语义决定，而不是为了最小化回报方差任意聚类。

本文用 $N$ 表示 rollout 数，$e$ 表示一次决策事件，$j$ 表示该事件中的成员，$K_e=|S_e|$ 表示 active-set 大小；这些数量不能互换。

## 3. 什么算一次 event：顺序和并行必须区分

论文将状态定义为外部环境状态加共享编排记忆。因此把候选方案提交到共享记忆，也可以是一次状态转移，不要求每次都调用外部工具。

- 多个 worker 从同一逻辑前置状态产生输出，一起被下一转移消费：可以组成一个多成员 event。
- Verifier 先读到 Solver 的新回答，再生成评审：两者属于先后两个 event，不能简单乘起来当同状态下的独立输出。
- Router 决定调用哪些 worker：router 的决定是一个先行的 singleton event。
- 最终 aggregator 在读取候选后生成答案：它自身通常是 singleton event。
- 若若干私有草稿既不写入共享记忆、也不被某次转移消费，就不能只因计算过它们而把它们都塞进 active set。论文特别指出，若只提交 aggregator 输出，对应提交事件是 aggregator 的 singleton。

在给定前置状态和选定集合后，成员采样条件独立时，联合策略因子化为：

$$
\pi_\Theta(a_e\mid s_e,S_e)=\prod_{j\in S_e}q_{e,j}(a_{e,j}\mid o_{e,j}).
$$

这里 $q$ 可以是 worker、router 或 aggregator 对应的策略。角色可以共享参数，也可以不共享；active set 不等于必须有 $K_e$ 份独立模型权重。是否共享参数与采样是否条件独立也不是同一个问题。

## 4. 先计算每个完整回答的 log-ratio

令 $u$ 为生成 token 的编号，$m_{e,j,u}$ 是排除 prompt、padding 和环境 token 的 mask：

$$
\ell_{e,j}=\sum_u m_{e,j,u}
\log\frac{q_\theta(a_{e,j,u}\mid o_{e,j,u})}
{q_{\rm old}(a_{e,j,u}\mid o_{e,j,u})}.
$$

这是完整成员回答的序列 log-ratio，包含每个 token 的生成前缀条件。它不是将 token ratio 做平均；回答长度会影响这个量，论文承认该限制。

若直接用真实联合 action 的 likelihood ratio，则为：

$$
\rho_e^{\rm joint}=\exp\left(\sum_{j\in S_e}\ell_{e,j}\right).
$$

这可以在成员数增加时迅速偏离 1，影响 clipping。

## 5. 核心公式：开平方缩放，再共同 clip

SRPO 使用：

$$
\boxed{
L_e^{\rm set}=\frac1{\sqrt{K_e}}\sum_{j\in S_e}\ell_{e,j},\qquad
\rho_e^{\rm SRPO}=\exp(L_e^{\rm set})
=(\rho_e^{\rm joint})^{1/\sqrt{K_e}}.
}
$$

当 $K_e=1$ 时，退化为普通 sequence-level ratio。当 $K_e>1$ 时，**它是经过幂次缩放的 surrogate，不是真实联合策略的原始 importance sampling ratio，也不是几何均值**；几何均值对应指数 $1/K_e$。

每个 event 得到一个优势：

$$
A_e=\frac{R_e-\mu_{g(e)}}{\sigma_{g(e)}+\varepsilon}.
$$

$R_e$ 是下游团队回报，$g(e)$ 是可比 prompt 和事件类型的比较组；论文没有把这固定为 GiGPO 式的精确同状态匹配。所有 event 成员共享 $A_e$。

最终对完整 events 求平均：

$$
\boxed{
J(\Theta)=\frac1{|\mathcal B|}\sum_{e\in\mathcal B}
\min\left(\rho_e^{\rm SRPO}A_e,
\operatorname{clip}(\rho_e^{\rm SRPO},1-\epsilon,1+\epsilon)A_e\right).
}
$$

这里每个 event 贡献一次，不按成员行数或 token 数给更大的 event 额外重复计权。但是事件梯度本身仍包含多个成员的贡献，不能据此声称每个 event 梯度范数相等。

### 一个数值例子

两名成员的序列概率比分别是 $1.4$ 和 $0.8$，则真实联合 ratio 是 $1.12$，SRPO ratio 是 $1.12^{1/\sqrt2}\approx1.083$。

若 $A>0$、clip 区间为 $[0.8,1.2]$，逐成员更新中第一个成员已经触及上界，但 SRPO 的整体分数仍在区间内。两个更新规则因此不同。

这也说明 joint clipping 并不单独约束每个成员：成员之间的 log-ratio 可以抵消。PPO clipping 本身也是 surrogate 中的裁剪机制，不是保证所有概率比都被硬限制在区间内。

## 6. 为什么选择平方根，而不是求和或平均

在成员 log-ratio 条件不相关，且平均方差有正的上下界时：

$$
\operatorname{Var}\left(K^{-\alpha}\sum_{j=1}^K\ell_j\right)
=K^{1-2\alpha}\bar v_K.
$$

因此求和的方差随 $K$ 增长，平均的方差随 $K$ 下降，$\alpha=1/2$ 在这些假设下保持非退化尺度。这是尺度动机，不是任务性能最优性定理。

若成员方差相同但相关：

$$
\operatorname{Var}(L_e^{\rm set})
=\sigma_e^2+\frac2{K_e}\sum_{i<j}\operatorname{Cov}(\ell_{e,i},\ell_{e,j}).
$$

协方差仍然存在；强相关时平方根缩放不保证尺度与集合大小无关。非零均值还会影响 clipping 触发率，不能只凭方差尺度就推出固定裁剪概率。

附录还给出局部 score/Fisher 的解释，但明确不证明准确率、收敛性或等算力效率。对于 singleton，其对应的是序列 ratio 版本，不应直接写成与所有 token-level GRPO 实现完全等价。

## 7. 它有没有解决每个 agent 的真实贡献分配

只解决了一部分问题：共同决策获得共同的 proximal 约束，避免在 loss 中完全拆开。但它仍将 $A_e$ 共享给成员，没有估计“去掉某个成员后的反事实收益”。

若一个成员错误、另一个成员纠正它，最终集合成功，SRPO 并不自动给前者负优势。不同 event 如果继承同一个终局回报，早期错误与后期修正也不必然被区分。文章的 motivation 不等于公式已经提供完整的因果信用分配。

所以准确定位是：**优化单元与团队决策单元对齐，加上集合大小相关的更新尺度控制。**

## 8. 与 Dr.MAS、GiGPO、HGPO、ProGPO 的不同

| 方法 | 主要修改位置 | 是否改变 ratio/clip 单元 |
|---|---|---|
| GiGPO | 同状态步骤的回报比较，叠加 episode 项 | 通常沿用基础 ratio |
| HGPO | history 条件下的多层优势比较 | 通常沿用基础 ratio |
| Dr.MAS | 实际参与输出的角色回报均值与尺度 | 主要是 advantage 归一化 |
| ProGPO | 全失败时切换到 progress 评分 | 基础 ratio 不变 |
| SRPO | 共同决策输出的联合 log-ratio、集合缩放、一次 clip | **明确改变** |

因此，前面“这些方法只改 advantage，和 ratio 定义无关”的说法不能扩展到 SRPO。理论上可以把某种 history-aware baseline 接到 SRPO 的 event 优势上，但需先定义可比 event 和完整集合采样，不能直接将任意单步混合公式套上去。

## 9. 实验与证据边界

数学训练使用 processed DAPO-Math、Qwen3-4B/8B；评估包括 AIME24/25、AMC23、MATH500、Minerva、OlympiadBench。Search 使用 processed HotpotQA、Qwen2.5-3B/7B，评估 NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue 和 Bamboogle。

Avg@16 是每题 16 次采样的平均正确率；Pass@16 是至少一次答对的题目比例。16 是评估采样数，不能直接当作训练 rollout group size。

数学宏平均：4B 为 61.3/77.9，8B 为 62.5/77.8，分别是 Avg@16/Pass@16。论文表中对最强 Dr.MAS 宏平均的增益仅约 0.2 个百分点，不能据此夸大为显著优势。

**论文明确说明基线行来自跨论文引用，不是配对重跑。** 所以主表是描述性比较；训练、编排与预算差异可能影响结果。部分单数据集也没有提升。

更直接的 Search 消融在同一个 $K=5$ 下比较：

| log-ratio 归约 | Avg@16 | Pass@16 |
|---|---:|---:|
| 求和 | 38.6 | 54.3 |
| 除以 $K$ | 34.7 | 48.6 |
| 除以 $\sqrt K$ | 40.1 | 56.1 |

这个结果支持所测设置下的平方根选择，但 $K=1$ 与 $K=5$ 的对照不是等算力对照。论文只给出 token/call/tool 等输出侧成本信息，未据此证明端到端 GPU 成本更优；本次未核实明确 GPU 型号和卡数。

## 10. 复现必须注意的系统约束

论文要求每个响应记录 event ID、前置状态 ID、成员编号、预期集合大小、策略版本及旧 log-prob。完整 event 的所有成员应保留在同一个全局 minibatch，缺失成员时应丢弃该 event，而不是悄悄缩小 $K_e$。

这一点与普通“先算优势，再随意打乱单步行”不同：SRPO 可以调整 batch 顺序，但不能拆散需要一起计算 ratio 的 active set。分布式实现需要跨成员正确聚合和梯度传播。

目前仅找到论文及伪代码，没有可确认的官方代码地址，因此未创建代码软链接，也不根据同名 SRPO 仓库猜测实现。还需核实具体 group key、奖励构造、角色共享参数配置与完整训练超参数，才能做可靠复现。

## 11. 对我们研究方案的启发

SRPO 补出了另一条主线：除了“比较谁、用什么回报、如何估计 baseline”，还有“哪些输出应该一起受一次 clip”。active set 应从执行依赖关系确定，而不是按奖励相似度组合。

如果目标仍是 simple yet effective，先选一个失败点：history baseline 的稀疏性、全失败的零信号、或多成员 clip 不一致。三者一起改会让贡献难以定位。这篇最可复用的思想是把优化单元定义清楚，同时如实保留联合缩放的假设与系统实现约束。

## 12. 本轮补充核对：SRPO 是否使用 progress？

**所读 SRPO v2 没有在方法中定义 ProGPO 式 progress 优势。** §4.2 Eq. (7) 使用下游团队回报 $R_e$，按可比事件组归一化；§4 开头明确说 rewards 和 advantage estimation 可以保持不变。核心变化在 ratio 与 clipping。

全文中的相关表述需要分开：

- §2 的 “progress-aware updates” 属于 Related Work，并引用了 Progress-conditioned Group Policy Optimization。
- Search prompt 的 “make progress toward the answer” 是行为指令，不是 reward 公式。
- 图 3 标注 “dense training score”，但所读方法与附录未提供可据此确认的 ProGPO 首次访问奖励定义。因此不能反过来宣称所有实验都只用二元终局奖励，也不能把曲线名称解释成明确的 progress 算法。
- Avg@16 / Pass@16 的 solution coverage 是评估指标，不是 ProGPO 的轨迹内首次访问覆盖。

这意味着：可以研究将 progress 优势接到 SRPO，但属于额外组合，不能描述为这份论文已经完整提出的方法。

## 13. 对当前分组 idea 的整合，而非模块堆叠

已将完整讨论写入 [研究笔记 §20](../../03-最优分组研究方案-轨迹梯度风险与组合优化.md)，并整理成 [简明整合方案](../../04-分组Idea整合-SRPO启发与可计算选择准则.md)。

SRPO 最重要的启发是区分“比较集合”和“更新单元”。前者像 HGPO 的历史组，用来估计 baseline；后者像 SRPO 的 active set，决定哪些输出形成一次联合 ratio。执行依赖先约束合法单位，统计选择再决定哪些单位可共享比较基准。

另一条研究扩展是用固定小幅候选策略与 old policy 的真实概率比，构造每条轨迹的局部收益估计，再比较不同历史分组造成的估计方差。它避免提取完整参数梯度，但只衡量特定策略变化的 surrogate 估计质量。这不是 SRPO 原方法，不能声称完整梯度 risk 最优。

特别需要保留三条边界：

1. SRPO 的平方根幂次 ratio 不是真实联合 importance ratio，不能直接套“ratio 减 1 的条件均值为零”。
2. 全失败且没有额外任务证据时，重分组和 ratio 都不能创造真实回报差异。
3. 将 ProGPO 加进去可以验证兼容性，但不是主要贡献；需要证明分组本身优于固定组、层级融合及简单收缩。

附录 F.3 还提供了一个对我们很有价值的提醒：所报 token-reduced 与 event-reduced 诊断中，较低梯度波动并未同时给出更高的可用验证准确率。这些只是诊断 checkpoint，不应直接视作完整方法排名，但足以提醒我们不要用“更稳定”替代“任务更好”。

本轮检查了全文与附录的事件定义、归一化假设、伪代码和构造示例；没有新增训练复现。有限环境的 ratio 恒等式校验属于我们扩展方案的数学验证，不属于 SRPO 的实验结果。
