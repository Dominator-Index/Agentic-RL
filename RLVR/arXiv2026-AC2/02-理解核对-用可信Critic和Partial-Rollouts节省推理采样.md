# 02｜理解核对：AC2 用可信 Critic 与 Partial Rollouts 节省推理采样

> 本文整理并核对我们的讨论。依据 [AC2 本地论文](./AC2-Trust-the-Critic-More.pdf) §3 与 Algorithm 1，以及已有[全文与代码解读](./analysis.md)。核对日期：2026-10-08。

## 1. 这个理解是否正确

**正确。AC2 的核心是：当 critic 对某个问题足够可信时，actor 只生成一段新 action chunk，就可以用 chunk 末端的价值预测构造优势，不必为了这次更新把所有后续推理生成到结束。**

也就是，用可信的价值估计替代一部分昂贵的实际后续采样，使训练可以使用 partial rollouts。

需要精确化两点：

1. Critic 预测的是“生成完 chunk 后，继续完成整题的预期终局得分”，不是 chunk 本身独立获得的局部正确性奖励。
2. 一般 actor-critic 本来就支持 bootstrap 和不完整轨迹更新。AC2 针对的是长文本推理中常见的完整 rollout 依赖，并非首次发现 actor-critic 可以提前截断。

## 2. 先生成多少，之后预测什么

设 $s$ 是从历史完整轨迹中截取的前缀，包含问题及此前全部推理文本。Actor 从这个前缀生成多个候选 chunk：

$$
c_1,\ldots,c_g\sim\pi_{\mathrm{old}}(\cdot\mid s).
$$

$g$ 是同一前缀的候选数。主实验 $g=16$，每个可信截断分支最多生成 10,000 个新 token，完整响应预算为 50,000 token，包含复用前缀。

生成 $c_i$ 后，当前状态为 $s\cdot c_i$。Critic 估计：

$$
V^{\pi_{\mathrm{old}}}(s\cdot c_i)
=\mathbb E_{z\sim\pi_{\mathrm{old}}(\cdot\mid s\cdot c_i)}
\left[R(x,y_{\mathrm{prefix}}\cdot c_i\cdot z)\right].
$$

其中 $z$ 是尚未生成的后续推理，$R$ 是整份证明的终局评分。公式是理想价值定义，实际由学习得到的 critic 近似；这里用 $\pi_{\mathrm{old}}$ 明确表示本轮采样策略。

**AC2 节省的是：不必为每条候选实际生成 $z$，也能获得对其最终得分的估计。** 它仍然必须先生成 $c_i$，并对末端上下文进行 critic 推理。

## 3. Advantage 怎样得到

对第 $i$ 条候选，末端分数为：

$$
v_i=
\begin{cases}
R_i,&\text{已经按终局规则评分},\\
\widehat V(s\cdot c_i),&\text{可信分支在中间位置截断}.
\end{cases}
$$

然后计算：

$$
\bar v=\frac1g\sum_{k=1}^{g}v_k,
\qquad
\widehat A_i=v_i-\bar v.
$$

这里 $\widehat V$ 表示 critic 预测；组均值充当共同前缀的价值 baseline。**AC2 主实验不再除以组内标准差。**

假设四条未完成续写的末端预测为 $[0.2,0.4,0.4,0.8]$，组均值为 $0.45$，优势就是 $[-0.25,-0.05,-0.05,0.35]$。这样已经可以比较续写方向，无需为这次更新把四条全部写完。这个例子是教学构造，不是实验测量。

同一 chunk 的所有新 token 共用该优势，所以仍然没有在 chunk 内逐 token 识别任务贡献。

## 4. 概率比与裁剪结构是否和 GRPO 一样

**在 token 级的重要性概率比与 clipped surrogate 这一层，形式相同。**

对 $c_i$ 中第 $j$ 个 token：

$$
\rho_{i,j}(\theta)
=\frac{\pi_\theta(c_{i,j}\mid s,c_{i,<j})}
{\pi_{\mathrm{old}}(c_{i,j}\mid s,c_{i,<j})}.
$$

分子与分母条件于同一个前缀，评估同一个 token；分子是正在更新的策略，分母是生成该 chunk 的采样策略。此处的分母不是 critic，也不是生成历史 replay 前缀的更早模型。

把 AC2 的 $\widehat A_i$ 代入常见裁剪目标，可示意为：

$$
\mathcal L_{\mathrm{actor}}
=-\operatorname{Avg}_{i,j}
\left[
\min\left(
\rho_{i,j}(\theta)\widehat A_i,
\operatorname{clip}(\rho_{i,j}(\theta),1-\epsilon,1+\epsilon)\widehat A_i
\right)
\right].
$$

上式突出共同结构，省略实现中的具体聚合与裁剪配置。相同的是概率比和 surrogate 的数学角色，不代表 AC2 与某个 GRPO 实现的所有超参数、归一化和 batch 聚合都完全一致。

旧前缀 $s$ 作为条件输入，不在这些 replay token 上直接计算本轮 actor loss；策略损失针对新采样的 $c_i$。

## 5. 本质上改变了哪些环节

| 环节 | 常见完整 rollout GRPO | AC2 的可信截断分支 |
|---|---|---|
| 共同起点 | 问题及任务上下文 | 问题加复用的完整推理前缀 |
| 新生成内容 | 一份完整回答 | 一段新的连续 token chunk |
| 用于比较的分数 | 实际终局回报 | chunk 末端的预测终局价值；若已终结则用真实评分 |
| 优势计算 | 常见为组内中心化并标准化 | 末端分数减组均值，主实验不除标准差 |
| 概率比 | 同上下文同 token 的新旧策略概率比 | 相同定义 |
| Actor 更新位置 | 新生成的完整回答 | 新 chunk |
| 可否省掉剩余推理生成 | 单靠组相对优势无法省掉 | critic 可用时可以省掉 |

因此，“改变 critic 的使用方法”抓住了关键，但如果理解成“先照常生成完整轨迹，再换一个 critic 算优势”，就不准确。

**它把 critic 真正用于替代未采样的未来回报，所以连采样流程和训练成本都发生了变化。** Prefix replay 决定从哪里开始，chunk 决定可以在哪里停止，critic 决定停止后怎样得到更新信号。

## 6. 与传统 actor-critic 的关系

传统 actor-critic 可以使用 TD bootstrap，例如：

$$
\widehat A_t=r_t+\gamma\widehat V(s_{t+1})-\widehat V(s_t).
$$

只要价值估计可用，就不必每次采样到 episode 结束。因此，“以前所有 actor-critic 都必须生成全部 token”是不成立的。

更准确的背景是：长文本 RLVR 中，GRPO 等方法依赖终局评分；某些采用 critic 的训练配方也仍依赖较完整的采样来构造回报／优势。AC2 通过较大的 token chunk、按题 readiness 判断、参考证明与终局抽查，让 learned value 在长证明任务中承担更多未来回报估计工作。

在本文无中间奖励、无折扣的文本拼接设置下，理想关系是：

$$
A^\pi(s,c)=V^\pi(s\cdot c)-V^\pi(s).
$$

AC2 用组内候选末端价值的均值估计起点 baseline。这个联系说明它与 actor-critic／bootstrap 的关系，但不是对其学习到的 critic 始终准确的保证。

## 7. 节省 token 的条件与代价

并非所有训练分支都只做 partial rollout：

- 尚未达到 readiness 门槛的问题，仍完整续写到终局。
- Ready 问题中，约四分之一的组会进行 audit，继续完整生成以获得外部评分。
- Fresh 完整 rollout 仍用于补充历史轨迹缓存。

所以这里的收益是**减少一部分完整 continuation 的 actor 解码**。Critic 推理、前缀 prefill、训练反向传播和终局 judge 仍有成本，不能简单把缩短的 token 数等同于全流程节省比例。

对应的风险是：critic 若错误地看好某个方向，partial rollout 更新也会跟着出错。Readiness 和 audit 的作用就是为这种提前停止提供经验上的可靠性控制；它们不构成无偏或精确的理论保证。

## 8. 可以直接保留的核心表述

**AC2 保留了 GRPO 类的概率比与裁剪策略更新结构，但改变了采样流程和优势的价值来源：从历史推理前缀出发采样 action chunks，在 critic 足够可信时，用 chunk 末端的预期终局价值替代必须等待的真实终局回报，使 actor 可以用 partial rollouts 更新，从而减少后续推理的生成成本。**

它不是重新分组既有终局奖励，也不是把每个 token 都赋予独立贡献；核心是用受可靠性条件控制的价值自举，减少实际必须生成的未来。

[01｜Action Chunking 的概念与例子](./01-Action-Chunking是什么-与GRPO和状态分组的区别.md) · [完整解读](./analysis.md) · [返回目录](./README.md)
