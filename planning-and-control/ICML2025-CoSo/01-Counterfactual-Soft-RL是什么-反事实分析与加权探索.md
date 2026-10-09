# 01｜Counterfactual Soft RL 是什么：反事实分析与加权探索

> 本文结合 [CoSo 论文](./CoSo.pdf) §3–§4 和本地代码解释概念。核对日期：2026-10-08。公式采用统一教学符号；论文原始影响值与代码实际权重分别说明。

**Counterfactual Soft RL 的核心是：先通过反事实扰动估计哪些 token 影响执行动作，再用这个估计给 token 熵加权，使探索更多发生在能改变动作的位置。**

## 1. 先明确 Agent 的执行过程

在一次环境交互中，VLM 接收状态／图像，生成文本，程序再把文本解析为动作：

$$
s_t\longrightarrow y_t=(y_{t,1},\ldots,y_{t,L_t})
\longrightarrow a_t=f_{\mathrm{parse}}(y_t)
\longrightarrow (s_{t+1},r_t).
$$

其中 $t$ 表示环境交互步，$i$ 表示一次输出内部的 token 位置，$L_t$ 是文本长度。状态可以包含任务指令、截图和历史；$r_t$ 是环境反馈。

例如，以下两种文字可能解析成同一个动作：

```text
我应该点击确认按钮。click(100, 200)
接下来选择确认。click(100, 200)
```

改变措辞不一定改变执行动作；改变点击坐标或动作类型则可能改变环境行为。这里是教学例子，实际动作格式以具体环境为准。

**CoSo 的 motivation：文本变得更多样，不保证环境动作也变得更多样。** 如果鼓励探索的预算主要用在无关措辞上，训练可能收集到大量行为相似的轨迹。

## 2. Soft RL：在任务奖励之外鼓励探索

普通 RL 的目标是最大化折扣回报：

$$
J(\theta)=\mathbb E_{\pi_\theta}
\left[\sum_t\gamma^t r_t\right].
$$

$\pi_\theta$ 是待训练策略，$\gamma$ 是折扣因子。这个目标直接关心任务完成得好不好。

Soft RL 通常指带熵正则的最大熵强化学习：

$$
J_{\mathrm{soft}}(\theta)
=\mathbb E_{\pi_\theta}
\left[\sum_t\gamma^t
\left(r_t+\alpha\mathcal H(\pi_\theta(\cdot\mid s_t))\right)\right].
$$

$\alpha\ge0$ 控制探索强度。对离散动作分布，熵为：

$$
\mathcal H(\pi(\cdot\mid s))
=-\sum_a\pi(a\mid s)\log\pi(a\mid s).
$$

它衡量分布的不确定性。以两个动作为例，概率为 $(0.99,0.01)$ 时熵约为 $0.056$；概率为 $(0.5,0.5)$ 时熵约为 $0.693$，这里使用自然对数。

加入正的熵项，鼓励策略不要过早固定在少数动作上；任务奖励仍然负责区分什么行为更好。**Soft 不是软标签，也不是稀疏奖励，更不意味着模型始终均匀随机。**

## 3. 语言 Agent 的 token 熵与执行动作熵要区分

令 $h_{t,i}=(s_t,y_{t,<i})$ 为生成第 $i$ 个 token 时的上下文。沿一条采样文本计算的普通 token 熵正则可写成：

$$
\widehat{\mathcal H}_{\mathrm{text},t}
=\sum_{i=1}^{L_t}
\mathcal H\left(\pi_\theta(\cdot\mid h_{t,i})\right).
$$

这里每个位置使用相同系数，并不代表每个位置的实际熵值相同。对策略生成的前缀取期望，才对应完整的序列熵链式分解；训练中通常使用采样估计。

对于确定性的动作解析器，因为多个文本可以映射到同一动作，有：

$$
\mathcal H(A\mid S)\le\mathcal H(Y\mid S).
$$

$Y$ 是生成文本，$A=f_{\mathrm{parse}}(Y)$ 是执行动作。这解释了为什么只提高文本熵，可能没有有效提高动作多样性。

这是阅读论文时应保留的数学区别：论文用 token 熵构造探索目标，但不能在一般的多对一解析下，直接把执行动作熵与文本熵画等号。

## 4. Counterfactual：如果这个 token 被改掉，会怎么样

CoSo 构建一个辅助动作预测模型 $P_\phi(a\mid y)$，用文本及其解析动作进行监督训练。它不是负责生成整个 rollout 的主策略。

给定原始文本 $y$ 及对应动作 $a$，保持其他文本不变，将第 $i$ 个 token 替换为空值／遮盖值。论文 Eq. (6) 的影响估计可简写为：

$$
B_i=P_\phi(a\mid y)
-P_\phi(a\mid y^{\mathrm{mask}(i)}).
$$

$y^{\mathrm{mask}(i)}$ 表示被干预后的文本；这里省略了论文结构因果模型中的外生噪声记号。

这个差值回答：“去掉这部分信息后，辅助模型对原动作的预测概率下降了多少？”例如，假设原动作的预测概率是 $0.90$：

| 教学假设 | 遮盖后的概率 | 概率差 |
|---|---:|---:|
| 遮盖一个无关措辞 token | 0.89 | 0.01 |
| 遮盖一个动作相关 token | 0.35 | 0.55 |

以上数值只是解释机制，不是论文测量结果。它们提示第二个位置对动作预测更敏感。

“反事实”在这里指人为构造“这个 token 不存在时”的对照输入。**不是给每个 token 重新运行一次环境，也不是评估所有备选 token 最终能否成功。**

## 5. 合起来：反事实影响决定探索权重

用 $w_{t,i}$ 表示最终用于训练的影响权重，CoSo 的加权熵可写成：

$$
\widehat{\mathcal H}_{\mathrm{CoSo},t}
=\sum_{i=1}^{L_t}w_{t,i}
\mathcal H\left(\pi_\theta(\cdot\mid h_{t,i})\right).
$$

论文 Eq. (7) 用 $B_i$ 作为权重；这里另设 $w_i$，是为了区分论文原始概率差与实际代码处理后的非负权重。训练目标的核心结构为：

$$
J_{\mathrm{CoSo}}(\theta)
=\mathbb E\left[\sum_t\gamma^t
\left(r_t+\alpha\widehat{\mathcal H}_{\mathrm{CoSo},t}\right)\right].
$$

权重越大，该位置受到的熵奖励越强，更鼓励尝试不同输出。这是通过训练目标影响之后的策略采样；不应把公式直接理解为推理时给每个位置手工设置不同温度。

**权重大说明值得加强探索，不说明当前 token 是正确的。** 比如坐标决定点击位置，值得探索；但哪个坐标有助于完成任务，仍要由任务奖励判断。

## 6. 它到底改了哪个 loss

为了对应实现，可以把最小化的策略损失概括为：

$$
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{policy}}
-\alpha\widehat{\mathcal H}_{\mathrm{CoSo}}.
$$

第一项使用基础 RL 算法的任务学习信号；第二项鼓励加权探索。负号来自“最小化损失”和“最大化熵”的方向转换。

因此，$w_i=0$ 通常表示该位置不获得这个额外熵项的鼓励，**不表示该 token 的所有梯度都被屏蔽**。CoSo 也没有因此把 trajectory reward 自动分解为每个 token 的正确性奖励。

## 7. 论文与代码如何对应

以下位置来自已归档的官方代码，commit 为 `eb4f205cb8af37f9dedbf124a0ac643050d6fc5b`，本次为静态核对，没有运行训练。

| 步骤 | 代码入口 | 实现说明 |
|---|---|---|
| 构造扰动、计算影响 | [executor.py](/usr/data/CoSo/digirl/coso/executor.py:38) | `get_token_effect` 默认窗口大小为 4，使用滑动窗口扰动，而非严格逐个 token 的孤立干预 |
| 概率差转换 | [executor.py](/usr/data/CoSo/digirl/coso/executor.py:105) | 使用概率差的绝对值，另有特殊符号屏蔽、序列内 min-max 和平方根变换，最后 detach |
| 对 token 熵加权 | [autoui_agent.py](/usr/data/CoSo/digirl/models/autoui_agent.py:151) | 将 `causal_effects` 与熵相乘，结合有效 token mask 聚合并按长度归一化；具体熵计算还区分代码分支 |
| 接入训练目标 | [trainer.py](/usr/data/CoSo/digirl/algorithms/causal_digirl/trainer.py:198) | 在线且开启相关选项时，使用 `pg_loss - entropy_coeff * causal_entropy.mean()` |

**Android 实现还有一个重要范围限制：** `executor.py` 的 `parse_action` 使用 `android_action.action_type.value` 作为辅助模型标签。因此该分支比较的是原始**动作类型**的预测概率，并非包含全部坐标、输入文字等参数的完整动作概率。本文前面的坐标例子用于解释文本与动作之间的关系，不能据此声称代码会精确衡量每个坐标对完整执行动作的影响。

这些处理意味着，代码最终权重不能直接解释为论文 Eq. (6) 的原始、有符号单 token 概率差。`detach` 则表示在此策略更新中，不通过权重计算过程反向传播。

## 8. 与 token 级信用归因的边界

CoSo 测量的是文本扰动对**动作代理预测**的影响。若要评估 token 对任务回报的贡献，概念上还需要比较不同干预所造成的后续行为与回报，例如：

$$
\Delta_i^{\mathrm{return}}
=\mathbb E[R\mid\mathrm{do}(y_i=v)]
-\mathbb E[R\mid\mathrm{do}(y_i=v')].
$$

这里 $v,v'$ 是两种干预，$R$ 是后续任务回报；该式只是对更强归因目标的示意，**不是 CoSo 实际估计的量**。要定义它，还必须说明后续 token 是否重新生成、环境如何继续执行，以及与什么替代输出比较。

尤其是，CoSo 保持其他文本不变做扰动，无法充分刻画早期推理 token 通过改变后续生成而影响最终动作的路径。一个解释 token 没有被解析器读取，不代表它对后续推理没有作用。

辅助模型预测不准、遮盖文本偏离训练分布、多个 token 相互配合等情况，也会使权重存在偏差。因此应把它看作指导探索的影响估计，而非精确的任务因果贡献标签。

## 9. 与 GiGPO、Dr.MAS 和我们之前的想法比较

| 方法 | 主要改变什么 | 权重／优势在回答什么 |
|---|---|---|
| GiGPO | 利用跨轨迹重复状态细化优势参照 | 相同状态下，哪个动作的回报相对更好？ |
| Dr.MAS | 按 agent 角色调整奖励统计与归一化 | 如何处理不同角色的训练信号分布？ |
| CoSo | token 熵正则的分配 | 哪些文本位置影响动作，值得加强探索？ |
| 我们讨论的 turn／token assessor | 待设计的贡献评估与更新分配 | 哪些输出促成成功或失败，应该怎样更新？ |

CoSo 对细粒度训练的启发是：不同 token 可以获得不同的训练侧重点。但从它继续研究任务信用归因，需要补上“动作变化如何影响后续回报”的证据，而不能把动作敏感性直接当成成功贡献。

## 10. 来源

- [CoSo 论文页面](https://arxiv.org/abs/2505.03792v2)与[本地 PDF](./CoSo.pdf)：§3 的熵正则动机，§4.2 的反事实估计，§4.3 的加权熵目标。
- [官方代码](https://github.com/langfengQ/CoSo)与[本地代码入口](./CoSo)。
- [已有方法解读](./analysis.md) · [返回本篇目录](./README.md)。
