# Dr. MAS 与 GRPO 等方法的数学对比

本文围绕《Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems》展开，重点比较 **GRPO、PPO、RLOO、Dr. GRPO、DAPO、GiGPO 和 GSPO**。依据为本目录的 [DrMAS.pdf](./DrMAS.pdf)、论文附录，以及已克隆的官方代码；代码版本为 c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9。整理日期：2026-10-05。

角色和采样流程请结合 [02 · 角色分工与协作流程](./02-DrMAS-角色分工与协作流程.md) 阅读：公式中的 $K$ 是协作角色数量，$N$ 是同题完整协作轨迹的采样数量；独立参数与独立完成任务是两个概念。

关于 rollout 预算是否对齐、团队奖励为何相同、Verifier 的完整评审输出，以及 Solver / Verifier 如何共同更新，参见 [03 · Rollout 预算、共享奖励与联合更新](./03-DrMAS-Rollout预算、共享奖励与联合更新.md)。该篇给出论文与源码定位、调用次数示例，以及同一份共享任务奖励如何得到各角色的优势。

**Dr. MAS 的算法核心：在同一道题的 rollout group 内，将所有 agent 共用的奖励均值、标准差，改为各 agent 活跃步骤上的奖励均值、标准差。**

$$
\boxed{
A_{\mathrm{GRPO}}^i=\frac{R^i-\mu}{\sigma}
\quad\longrightarrow\quad
A_{\mathrm{DrMAS}}^{i,k}=\frac{R^i-\mu_k}{\sigma_k}
}
$$

这里 $R^i$ 仍然是第 $i$ 条完整协作轨迹的最终奖励；$\mu_k,\sigma_k$ 根据 agent $k$ 实际执行过的步骤计算。变化发生在奖励的中心化和缩放方式，同时保留基于最终奖励的协作训练。实现中会在分母加数值稳定项。

## 1. 它解决什么问题，为什么需要这个改动

论文研究的是多个 LLM agent 共同完成任务的 RL 后训练。例如：

- 数学任务：Solver 提出或修改解答，Verifier 判断是否继续修改。
- 搜索任务：Verifier 决定是否检索，Search agent 发起搜索，Answer agent 组织答案。

一次完整协作只有一个最终任务奖励，但不同 agent 的调用次数和调用条件可能不同。需要继续检索的轨迹、直接回答的轨迹、反复修改的轨迹，会给各角色留下不同的训练样本分布。

因此，必须区分两个采样口径：

1. **GRPO 的统计口径**：同一道题的每条轨迹各贡献一次奖励。
2. **某个 agent 的优化口径**：每个实际执行的 agent 输出各贡献一次训练样本。

当调用次数与任务成败相关时，这两个口径下的奖励分布不同。论文将它们之间的均值、方差错配，与梯度二阶矩的放大联系起来。来源：[论文 §3–4](./DrMAS.pdf#page=3)。

**多个模型参数不同，并不自动意味着 $\mu_k$ 一定不同。** 在论文“所有步骤继承同一最终奖励”的设定下，奖励统计的差异必须通过 agent 的出场集合或出场次数体现出来。每个角色在所有轨迹中都出场相同次数时，下面的统计差异会消失。

## 2. 统一符号：轨迹、agent 步骤与 token

固定一道题 $x$，采样 $N$ 条协作轨迹：

$$
\tau^i=\bigl\{(s_t^i,a_t^i,k_t^i)\bigr\}_{t=1}^{T_i},
\qquad
R^i=R(\tau^i),
\qquad i=1,\ldots,N.
$$

$s_t^i$ 是当前状态，$a_t^i$ 是一次 agent 输出，$k_t^i$ 是执行该输出的角色。一个输出通常包含多个 token，所以环境步骤 $t$ 与 token 位置 $\ell$ 是不同的索引。

| 符号 | 含义 |
| --- | --- |
| $K$ | 逻辑 agent 数量 |
| $\theta_k$ | agent $k$ 对应的模型参数，允许多个角色共享参数 |
| $\mathcal{Y}_k=\{(i,t):k_t^i=k\}$ | agent $k$ 的活跃步骤集合；保留轨迹、步骤索引，重复文本也分别计数 |
| $n_{ik}=\sum_t\mathbf{1}\{k_t^i=k\}$ | agent $k$ 在轨迹 $i$ 中的出场次数 |
| $M_k=\sum_i n_{ik}$ | agent $k$ 的活跃输出总数 |
| $L_{it}$ | 输出 $a_t^i$ 的有效生成 token 数 |
| $\epsilon_{\mathrm{clip}}$ | 策略比率 clipping 的阈值 |
| $\varepsilon_{\mathrm{num}}$ | 避免除零的数值稳定项 |

论文用 $\mathcal{Y}_k$ 表示输出集合；这里显式写成索引集合，避免把相同字符串误当成同一个样本。表中的 $n_{ik},M_k$ 只是将论文的活跃步骤统计改写成易于计算的形式。

定义通用 clipped surrogate：

$$
\ell_{\mathrm{clip}}(\rho,A)
=\min\!\left(
\rho A,\,
\operatorname{clip}\!\left(
\rho,1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}}
\right)A
\right).
$$

$\rho$ 是新旧策略的概率比率，$A$ 是固定下来的优势估计。它限制部分过度更新的收益。下文用最大化目标的形式书写；实际训练通常最小化它的负值。为突出方法差异，主公式省略可选 KL 和熵项。

## 3. GRPO：同题轨迹组的均值与标准差

### 3.1 原始 outcome-supervision GRPO

原始 GRPO 对同一个输入采样多个完整回答 $y^1,\ldots,y^N$，计算：

$$
\mu=\frac{1}{N}\sum_{i=1}^{N}R^i,
\qquad
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(R^i-\mu)^2,
\qquad
A_{\mathrm{global}}^i=\frac{R^i-\mu}{\sigma}.
$$

这些统计量属于**同一道题的 rollout group**。Dr. MAS 文中的“global”表示该题内各 agent 共用统计量，并不表示对整个训练集或整个 minibatch 求均值。

原始 GRPO 的 token 比率和目标可写为：

$$
\rho_{i\ell}(\theta)
=\frac{\pi_\theta(y_\ell^i\mid x,y_{<\ell}^i)}
{\pi_{\theta_{\mathrm{old}}}(y_\ell^i\mid x,y_{<\ell}^i)},
$$

$$
J_{\mathrm{GRPO}}(\theta)
=\mathbb{E}\!\left[
\frac{1}{N}\sum_{i=1}^{N}
\frac{1}{L_i}\sum_{\ell=1}^{L_i}
\ell_{\mathrm{clip}}(\rho_{i\ell},A_{\mathrm{global}}^i)
\right].
$$

$L_i$ 是回答长度。outcome-supervision 版本把同一个 $A_{\mathrm{global}}^i$ 广播给回答中的所有 token；用组内奖励估计基线，省去 learned critic。此处只比较最终奖励版本，原始 GRPO 也讨论了 process supervision。来源：[DeepSeekMath §4.1](https://arxiv.org/pdf/2402.03300)。

### 3.2 Dr. MAS 论文采用的多 agent GRPO 基线

将一条回答换成完整多 agent 轨迹后，同一个 $A_{\mathrm{global}}^i$ 被传给该轨迹中所有角色的输出。论文 Eq. (3) 对 agent $k$ 写成：

$$
J_k^{\mathrm{global}}(\theta_k)
=\mathbb{E}\!\left[
\frac{1}{M_k}\sum_{(i,t)\in\mathcal{Y}_k}
\ell_{\mathrm{clip}}(\rho_{\theta_k}(a_t^i),A_{\mathrm{global}}^i)
\right],
$$

$$
\rho_{\theta_k}(a_t^i)
=\frac{\pi_{\theta_k}(a_t^i\mid s_t^i)}
{\pi_{\theta_{k,\mathrm{old}}}(a_t^i\mid s_t^i)}.
$$

这是论文按“完整 agent 输出”写出的抽象目标：每个 agent 输出使用其整条轨迹的优势。**论文这里的 action 比率与原始 GRPO 的 token 比率，需要按各自粒度理解。** 官方代码实际在 token 上计算比率和 loss；第 8 节单独说明这一点。

## 4. Dr. MAS：根据 agent 实际出场步骤重新计算优势

### 4.1 具体改动

Dr. MAS 在每个题目的 group 内，为 agent $k$ 计算：

$$
\mu_k
=\frac{1}{M_k}\sum_{(i,t)\in\mathcal{Y}_k}R^i
=\frac{\sum_i n_{ik}R^i}{\sum_i n_{ik}},
$$

$$
\sigma_k^2
=\frac{1}{M_k}\sum_{(i,t)\in\mathcal{Y}_k}(R^i-\mu_k)^2
=\frac{\sum_i n_{ik}(R^i-\mu_k)^2}{\sum_i n_{ik}},
$$

$$
A_{\mathrm{agent}}^{i,k}
=\frac{R^i-\mu_k}{\sigma_k+\varepsilon_{\mathrm{num}}}.
$$

这里每次出场都贡献一次奖励。如果 agent $k$ 在某条轨迹中出现三次，该轨迹奖励就在它的统计量中出现三次。因此这是 **agent 活跃输出上的加权统计**，不能简单替换成“所有包含该 agent 的轨迹各计一次”。

论文的最终目标保留相同的 surrogate，只替换优势：

$$
J_k^{\mathrm{DrMAS}}(\theta_k)
=\mathbb{E}\!\left[
\frac{1}{M_k}\sum_{(i,t)\in\mathcal{Y}_k}
\ell_{\mathrm{clip}}(\rho_{\theta_k}(a_t^i),A_{\mathrm{agent}}^{i,k})
\right].
$$

它仍然优化最终任务奖励，没有新增 critic，也不需要为每个角色再标注一个独立奖励。来源：[论文 §4.2、附录 E](./DrMAS.pdf#page=5)。

### 4.2 为什么它不等于单纯调整学习率

忽略数值稳定项，两个优势有如下精确关系：

$$
A_{\mathrm{agent}}^{i,k}
=\frac{\sigma}{\sigma_k}A_{\mathrm{global}}^i
+\frac{\mu-\mu_k}{\sigma_k}.
$$

第一项改变尺度，第二项改变中心。当 $\mu_k\ne\mu$ 时，这个变换可能改变某些样本优势的正负号，不能只用一个 agent-specific learning rate 完整复现。

按活跃步骤均匀采样，若 $\sigma_k>0$，则：

$$
\mathbb{E}_k[A_{\mathrm{agent}}]=0,
\qquad
\mathbb{E}_k[A_{\mathrm{agent}}^2]=1.
$$

这里 $\mathbb{E}_k$ 表示在固定 rollout group 的 $\mathcal{Y}_k$ 上均匀取样。中心和尺度因此与该 agent 实际拿到的输出样本相匹配。上述等式使用论文的总体方差口径，并忽略稳定项；代码的样本标准差口径见第 8 节。

### 4.3 什么时候两者没有算法差异

若对某个 agent，所有轨迹的出场次数都相同，即 $n_{ik}=c_k>0$，则：

$$
\mu_k=\mu,\qquad \sigma_k^2=\sigma^2.
$$

这是因为权重 $c_k$ 在分子、分母中抵消。此时按论文的方差定义，Dr. MAS 与 GRPO 的优势相同。

因此，角色名称不同或参数不共享，本身不足以保证这个改动有效。相反，即使只有一个 agent，只要多轮轨迹长度不同，活跃步骤加权统计也可能与轨迹等权统计不同；论文主要研究的是这种差异在协作多 agent 系统中的稳定性影响。

## 5. 为什么会影响梯度：重建论文的数学推导

### 5.1 分析的是单个活跃输出的未裁剪梯度贡献

论文定义：

$$
z_{it}^{(k)}
=\nabla_{\theta_k}\rho_{\theta_k}(a_t^i),
\qquad
\widetilde g_k^{\mathrm{global}}
=\frac{R^i-\mu}{\sigma}z_{it}^{(k)}.
$$

$z_{it}^{(k)}$ 表示策略比率对参数的导数；$\widetilde g_k$ 是这个输出对未裁剪 surrogate 的梯度贡献。虽然论文称 $z$ 为 score function，但它是 $\nabla\rho$。一般有 $\nabla\rho=\rho\nabla\log\pi$；在 $\theta_k=\theta_{k,\mathrm{old}}$ 时，$\rho=1$，才与通常的 log-policy score 一致。

这里把奖励、均值和标准差当作已固定的 advantage 估计，不对它们反向传播。该分析也没有直接覆盖 KL 项、clipping 的分支、共享参数混合梯度或优化器。

### 5.2 先看优势自身的二阶矩

对于 agent $k$：

$$
\mathbb{E}_k[A_{\mathrm{global}}]
=\frac{\mu_k-\mu}{\sigma},
$$

$$
\mathbb{E}_k[A_{\mathrm{global}}^2]
=\frac{\sigma_k^2+(\mu_k-\mu)^2}{\sigma^2}
\equiv c_k.
$$

第一式说明，共享基线下该 agent 的平均优势不一定是零。第二式将其二阶矩分成局部方差和均值错配两部分：

$$
c_k
=\frac{\sigma_k^2}{\sigma^2}
+\frac{(\mu_k-\mu)^2}{\sigma^2}.
$$

$c_k>1$ 表示该 agent 活跃样本上的优势二阶矩大于 1；$c_k<1$ 也可能发生。Dr. MAS 将这个奖励统计因子调整为 1，所以它既可能压低过大的尺度，也可能放大过小的尺度。

### 5.3 保留 score–reward 协方差后，得到精确恒等式

先展开梯度范数平方：

$$
\mathbb{E}_k\!\left[\|\widetilde g_k^{\mathrm{global}}\|^2\right]
=\mathbb{E}_k\!\left[
\|z_{it}^{(k)}\|^2\frac{(R^i-\mu)^2}{\sigma^2}
\right].
$$

两个随机量通常有关联，不能直接把乘积的期望拆成期望的乘积。利用协方差恒等式：

$$
\mathbb{E}_k\!\left[\|\widetilde g_k^{\mathrm{global}}\|^2\right]
=\mathbb{E}_k[\|z_{it}^{(k)}\|^2]\,
\mathbb{E}_k\!\left[\frac{(R^i-\mu)^2}{\sigma^2}\right]
+\Delta_k,
$$

$$
\Delta_k
=\operatorname{Cov}_k\!\left(
\|z_{it}^{(k)}\|^2,\frac{(R^i-\mu)^2}{\sigma^2}
\right).
$$

$\Delta_k$ 记录策略导数大小与奖励偏移平方的关联。它可以为正，也可以为负；保留它就不需要假设两者独立。

再把奖励围绕 agent 均值展开：

$$
\begin{aligned}
\mathbb{E}_k[(R^i-\mu)^2]
&=\mathbb{E}_k[(R^i-\mu_k+\mu_k-\mu)^2]\\
&=\sigma_k^2+(\mu_k-\mu)^2.
\end{aligned}
$$

交叉项为零，因为 $\mathbb{E}_k[R^i-\mu_k]=0$。于是得到论文 Lemma 4.2：

$$
\boxed{
\mathbb{E}_k[\|\widetilde g_k^{\mathrm{global}}\|^2]
=\mathbb{E}_k[\|z_{it}^{(k)}\|^2]\,
\frac{\sigma_k^2+(\mu_k-\mu)^2}{\sigma^2}
+\Delta_k
}
$$

它解释的是：当 score 的规模和协方差没有抵消变化时，错配因子会放大梯度二阶矩。**二阶矩不等于方差**；它也包括均值梯度的平方。来源：[论文附录 B.1](./DrMAS.pdf#page=15)。

### 5.4 Dr. MAS 去掉了哪一部分

对 agent-wise 优势做同样计算：

$$
\widetilde g_k^{\mathrm{agent}}
=\frac{R^i-\mu_k}{\sigma_k}z_{it}^{(k)},
$$

$$
\boxed{
\mathbb{E}_k[\|\widetilde g_k^{\mathrm{agent}}\|^2]
=\mathbb{E}_k[\|z_{it}^{(k)}\|^2]
+\Delta_k^{\mathrm{agent}}
}
$$

其中：

$$
\Delta_k^{\mathrm{agent}}
=\operatorname{Cov}_k\!\left(
\|z_{it}^{(k)}\|^2,\frac{(R^i-\mu_k)^2}{\sigma_k^2}
\right).
$$

归一化让优势平方的平均值成为 1，从而去掉 $c_k$ 这个错配因子。但协方差并没有被消除，也不保证比原来的协方差更小。

**对理论结论的解读：**若额外有 $\|z_{it}^{(k)}\|^2\le B_k$ 的逐样本界，则立即得到 $\mathbb{E}_k[\|\widetilde g_k^{\mathrm{agent}}\|^2]\le B_k$。若只有论文 Assumption 4.1 的 score 二阶矩上界，不能仅凭优势二阶矩为 1，就认为梯度二阶矩由同一个常数控制；还需要控制相关性，或利用有限组中优势的逐样本界等信息。

### 5.5 “梯度发散”结论的适用条件

论文正文 Proposition 4.3 将 $c_{k,m}\to\infty$ 与梯度二阶矩发散联系起来。附录 B.2 实际补充了条件：score 二阶矩或协方差不能抵消这种增长。上界假设本身没有给出 score 的正下界，因此不能只凭 $c_{k,m}$ 变大，就无条件推出发散。

还有一个由论文统计定义得到的有限组约束。固定 $N$ 条轨迹、同一 $R^i$ 广播到各步骤，令 $p_{ik}=n_{ik}/M_k$，则：

$$
\begin{aligned}
c_k
&=\sum_i p_{ik}\frac{(R^i-\mu)^2}{\sigma^2}\\
&\le \max_i\frac{(R^i-\mu)^2}{\sigma^2}\\
&\le N,
\qquad \sigma>0.
\end{aligned}
$$

最后一步使用 $\sum_i(R^i-\mu)^2=N\sigma^2$。这是本文根据公式作出的推导，不是论文新增的定理。因此，在固定 group size 为 5 或 8 的理想设定中，这个奖励错配因子本身不能趋于数学上的无穷大。若还有统一的 score 二阶矩界 $C_k$，则全局优势逐样本满足 $A^2\le N$，进而有 $\mathbb{E}_k[\|\widetilde g_k^{\mathrm{global}}\|^2]\le NC_k$。

这不否定实验中观察到的梯度尖峰，而是限定理论能解释到什么程度：有限的奖励尺度放大、策略导数变化、概率比率和数值问题可以共同导致尖峰；不能将固定组实验直接解释为该因子趋于无穷大的实例。若全组奖励相同，则 $\sigma=0$，所有角色继承的奖励也相同，局部方差同样为零，需要使用稳定项处理。

### 5.6 单样本结论与整批更新之间的区别

论文分析的 $\widetilde g$ 是一个输出的梯度贡献。若整批梯度为 $\bar g_k=M_k^{-1}\sum_u\widetilde g_{k,u}$，则：

$$
\mathbb{E}\|\bar g_k\|^2
=\frac{1}{M_k^2}
\sum_{u,v}\mathbb{E}
\left[\left\langle\widetilde g_{k,u},\widetilde g_{k,v}\right\rangle\right].
$$

$u,v$ 遍历活跃输出。整批梯度还取决于样本之间的方向和相关性；同一轨迹的多个输出也不是独立样本。因此，优势二阶矩校准并不直接等于完整优化器更新具有相同范数。

## 6. 一个可以手算的 GRPO / Dr. MAS 对比例子

考虑同一道题的五条轨迹。前四条失败，最后一条成功。agent $k$ 在失败轨迹中各出场一次，在成功轨迹中出场四次：

| 轨迹 | 最终奖励 $R^i$ | agent $k$ 的出场次数 $n_{ik}$ |
| --- | --- | --- |
| 1 | 0 | 1 |
| 2 | 0 | 1 |
| 3 | 0 | 1 |
| 4 | 0 | 1 |
| 5 | 1 | 4 |

GRPO 的轨迹等权统计为：

$$
\mu=\frac{1}{5}=0.2,\qquad
\sigma^2=0.2(1-0.2)=0.16,\qquad \sigma=0.4.
$$

agent $k$ 的八个活跃输出中，四个继承失败奖励，四个继承成功奖励。因此：

$$
\mu_k=\frac{4}{8}=0.5,\qquad
\sigma_k^2=0.25,\qquad \sigma_k=0.5.
$$

忽略稳定项、采用论文总体方差口径，两种优势如下：

| 活跃输出所属轨迹 | GRPO 优势 | Dr. MAS 优势 |
| --- | --- | --- |
| 失败轨迹 | $(0-0.2)/0.4=-0.5$ | $(0-0.5)/0.5=-1$ |
| 成功轨迹 | $(1-0.2)/0.4=2$ | $(1-0.5)/0.5=1$ |

在 agent 的活跃样本上计算：

$$
\mathbb{E}_k[A_{\mathrm{global}}]=0.75,\qquad
\mathbb{E}_k[A_{\mathrm{global}}^2]=2.125,
$$

$$
\mathbb{E}_k[A_{\mathrm{agent}}]=0,\qquad
\mathbb{E}_k[A_{\mathrm{agent}}^2]=1.
$$

这个例子中，共享统计放大了成功样本相对于失败样本的尺度；agent-wise 统计将其重新校准。若为演示而假设 score 范数平方恒定，使两项协方差均为零，则未裁剪梯度二阶矩之比也是 2.125。真实模型不必满足该演示假设。

这里没有给 agent 新增成功奖励，也没有判断它在成功轨迹中作出了多少因果贡献；改变的是它拿到的最终奖励如何被转换成优势。

## 7. 与其他方法逐项比较

这些方法修改的环节不同。下面比较数学机制，不将它们概括为“谁全面优于谁”；Dr. MAS 的主要实验也没有在同一协议下逐一比较这里所有算法。

### 7.1 与 PPO：learned state baseline 和 empirical group baseline

典型 actor–critic PPO 使用价值函数 $V_\phi(s)$ 和 GAE：

$$
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t),
\qquad
\widehat A_t^{\mathrm{GAE}}
=\sum_{h=0}^{T-t}(\gamma\lambda)^h\delta_{t+h}.
$$

$r_t$ 是步骤奖励，$\gamma$ 为折扣因子，$\lambda$ 控制 GAE 的偏差与方差权衡，$V_\phi$ 预测状态的后续价值。优势可以随状态和步骤变化。

$$
J_{\mathrm{PPO}}(\theta)
=\mathbb{E}_t[
\ell_{\mathrm{clip}}(\rho_t,\widehat A_t^{\mathrm{GAE}})
],
\qquad
L_V(\phi)
=\mathbb{E}_t[(V_\phi(s_t)-\widehat V_t)^2].
$$

PPO 的策略目标同样使用 clipped surrogate，但通常额外训练价值模型。Dr. MAS 使用 $R^i-\mu_k$ 这个同题、同角色的经验比较，再除以 $\sigma_k$，没有学习状态条件化的 $V_\phi(s)$。

**区别：**PPO/GAE 可提供随状态变化的回报估计，代价是价值估计的训练和误差；Dr. MAS 用 group 统计提供较简单的角色级尺度校准。PPO 本身是一种 surrogate 优化方式，不能将“必须用 critic”视为它的数学定义；这里对比的是常见 actor–critic 实例。来源：[PPO §3、§5](https://arxiv.org/pdf/1707.06347)。

### 7.2 与 RLOO：leave-one-out 基线和 agent-conditioned 归一化

RLOO 用同一道题其余 $N-1$ 个样本的奖励作为当前样本的基线：

$$
b_{-i}=\frac{1}{N-1}\sum_{j\ne i}R^j,
\qquad
A_{\mathrm{RLOO}}^i=R^i-b_{-i}.
$$

排除当前样本，是为了在独立采样条件下构造与当前回答动作独立的基线。原始 RLOO 使用 REINFORCE 式序列梯度：

$$
\widehat g_{\mathrm{RLOO}}
=\frac{1}{N}\sum_i
A_{\mathrm{RLOO}}^i\nabla_\theta\log\pi_\theta(y^i\mid x).
$$

$y^i$ 表示完整回答。这个原始形式没有要求使用 PPO clipping；现代框架把 RLOO 优势接入 clipped loss 时，属于相应的实现变体。这里 $R^i$ 可理解为已包含所选 reward shaping 的回报。

对于轨迹等权组，基线存在一个精确关系：

$$
A_{\mathrm{RLOO}}^i
=\frac{N}{N-1}(R^i-\mu).
$$

所以 RLOO 与去掉标准差的组内中心化优势，只相差一个由 group size 决定的常数。它既没有原始 GRPO 的除以 $\sigma$，也没有 Dr. MAS 的 $\mu_k,\sigma_k$。

**区别：**RLOO 主要改变当前样本基线的构造方式，Dr. MAS 主要改变基线统计所条件化的 agent 及其活跃样本权重。不能将 agent-wise whitening 自动等同于 RLOO 的独立基线与无偏性质。来源：[RLOO 原文 §2.2–2.3](https://arxiv.org/pdf/2402.14740)。

### 7.3 与 Dr. GRPO：移除归一化和改变归一化对象

**Dr. GRPO 与 Dr. MAS 是两篇不同的工作。** Dr. GRPO 研究的是回答长度和题目级标准差归一化带来的优化权重偏移：

$$
A_{\mathrm{DrGRPO}}^i=R^i-\mu,
$$

$$
J_{\mathrm{DrGRPO}}(\theta)
=\mathbb{E}\!\left[
\frac{1}{N L_{\max}}
\sum_i\sum_{\ell=1}^{L_i}
\ell_{\mathrm{clip}}(\rho_{i\ell},R^i-\mu)
\right].
$$

$L_{\max}$ 是固定的生成预算或其他固定归一化常数。这个写法体现两处修改：不除以题目组奖励标准差；不按每条回答实际长度 $L_i$ 分别归一化。固定常数影响总体尺度，但不按当前回答长度改变样本权重。

对比 Dr. MAS：

$$
A_{\mathrm{DrGRPO}}^i=R^i-\mu,
\qquad
A_{\mathrm{DrMAS}}^{i,k}=\frac{R^i-\mu_k}{\sigma_k}.
$$

**区别：**Dr. GRPO 移除两类样本相关的归一化；Dr. MAS 保留奖励标准化，并按角色活跃输出调整它的中心和尺度。前者消除了该标准差分母带来的放大途径，但没有自动建立 agent-specific 基线；后者也没有自动解决 Dr. GRPO 所讨论的全部长度和题目权重问题。

官方代码中的 norm_adv_by_std_in_grpo=False 只对应“不除标准差”这一部分；完整复现 Dr. GRPO 还要配置固定长度归一化。来源：[Dr. GRPO 原文 §3.1–3.2](https://arxiv.org/pdf/2503.20783)。

### 7.4 与 DAPO：loss 加权、采样和 clipping 的改动

DAPO 仍使用题目组的标准化优势：

$$
A_{\mathrm{DAPO}}^i=\frac{R^i-\mu}{\sigma}.
$$

它的主要目标可写为：

$$
J_{\mathrm{DAPO}}(\theta)
=\mathbb{E}\!\left[
\frac{1}{\sum_i L_i}
\sum_i\sum_{\ell=1}^{L_i}
\min\!\left(
\rho_{i\ell}A^i,\,
\operatorname{clip}(
\rho_{i\ell},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}}
)A^i
\right)
\right].
$$

分母为整个组的有效 token 总数；上下 clipping 阈值分开设置，Clip-Higher 提高上界，以允许更大的概率提升。此外，二值正确性任务的动态采样保留：

$$
0<\sum_i\mathbf{1}\{y^i\text{ is correct}\}<N.
$$

这个条件保留既有正确回答又有错误回答的组，减少全同奖励导致的无效训练信号。DAPO 还处理过长回答的奖励 shaping。

**区别：**DAPO 改变 token 加权、概率比率裁剪与采样；Dr. MAS 改变 $\mu,\sigma$ 的 agent 条件化口径。采用 token-mean loss，并不意味着已经采用了 agent-wise normalization；其余组件也不自动产生 $\mu_k,\sigma_k$。来源：[DAPO §3](https://arxiv.org/pdf/2503.14476)。

### 7.5 与 GiGPO：按共同状态进行步骤比较

GiGPO 同时使用 episode-level 优势和 step-level 优势。对步骤 $t$ 定义折扣回报：

$$
G_{it}=\sum_{u=t}^{T_i}\gamma^{u-t}r_{iu}.
$$

$r_{iu}$ 是步骤奖励，$G_{it}$ 衡量该动作之后的回报。对在同一 anchor state $\widetilde s$ 出现的动作建立组：

$$
\mathcal{G}_{\mathrm{step}}(\widetilde s)
=\{(i,t):s_t^i=\widetilde s\}.
$$

该组通过已采样轨迹中的重复状态构造，不要求额外从每个状态重新 rollout。GiGPO 的优势为：

$$
A_i^{E}
=\frac{R^i-\mu}{F_E},
\qquad
A_{it}^{S}
=\frac{G_{it}-\mu_{\widetilde s}}{F_{\widetilde s}},
\qquad
A_{it}^{\mathrm{GiGPO}}
=A_i^E+\omega A_{it}^S.
$$

$\mu_{\widetilde s}$ 是 anchor-state 组内回报均值，$\omega\ge0$ 控制步骤信号的权重。$F_E,F_{\widetilde s}$ 是各组的归一化因子，原文研究了标准差与常数 1 两种口径。

**区别：**GiGPO 比较“在相同状态下，哪些动作得到更好的后续回报”，提供更细的步骤级比较；Dr. MAS 比较“这个角色实际拿到的最终奖励，应该如何中心化和缩放”。同一 agent 的不同状态，在 Dr. MAS 中仍可共享同一个基线；同一轨迹中该角色各步骤的最终奖励优势也相同。

两者在机制上可以组合，但将 GiGPO 接入多角色系统时，要重新考虑状态、角色和动作空间的可比性。Dr. MAS 本文没有给出该组合的全面实验证明。来源：[GiGPO §4](https://arxiv.org/pdf/2505.10978)。

### 7.6 与 GSPO：序列比率和优势统计是两个维度

GSPO 用长度归一化的序列似然比率替换 token 比率：

$$
s_i(\theta)
=\left(
\frac{\pi_\theta(y^i\mid x)}
{\pi_{\theta_{\mathrm{old}}}(y^i\mid x)}
\right)^{1/L_i}
=\exp\!\left(
\frac{1}{L_i}\sum_{\ell=1}^{L_i}\log\rho_{i\ell}
\right).
$$

这里 $s_i$ 是整条回答内 token 比率的几何平均。它使用序列似然并对长度做了变换，不能直接称为未经变换的整序列重要性采样比率。

$$
J_{\mathrm{GSPO}}(\theta)
=\mathbb{E}\!\left[
\frac{1}{N}\sum_i
\ell_{\mathrm{clip}}\!\left(
s_i(\theta),\frac{R^i-\mu}{\sigma}
\right)
\right].
$$

GSPO 的主要变化在比率与裁剪粒度；原始优势仍是题目组的奖励标准化。Dr. MAS 的主要变化在优势统计。因此，机制上可以将 sequence-level surrogate 与 agent-wise 优势组合。

扩展到一次多 agent 输出 $a_t^i$ 时，可以对该输出定义 $s_{it}^{(k)}=(\pi_{\theta_k}(a_t^i\mid s_t^i)/\pi_{\theta_{k,\mathrm{old}}}(a_t^i\mid s_t^i))^{1/L_{it}}$，再将 $A_{\mathrm{global}}^i$ 换成 $A_{\mathrm{agent}}^{i,k}$。这是说明组合方式的公式改写，不能将一个 agent 输出与整条多人协作轨迹混为同一“序列”。

Dr. MAS 附录 F.5 实际报告了 GSPO 与 GSPO + Dr. MAS 的对比，支持二者在该实验配置下互补。来源：[GSPO §4.1](https://arxiv.org/pdf/2507.18071)、[Dr. MAS 附录 F.5](./DrMAS.pdf#page=22)。

### 7.7 对比表

表中公式省略数值稳定项和可选 KL 项。

| 方法 | 核心优势或信号 | 主要改动 | 与 Dr. MAS 的区别 |
| --- | --- | --- | --- |
| GRPO | $(R^i-\mu)/\sigma$ | 用同题组统计替代 critic | 各角色共用轨迹等权统计 |
| Dr. MAS | $(R^i-\mu_k)/\sigma_k$ | 按角色活跃输出校准优势 | 本文方法 |
| PPO/GAE | $\sum_h(\gamma\lambda)^h\delta_{t+h}$ | 通常学习状态价值、构造 GAE | 基线可依赖状态，需要价值估计 |
| RLOO | $R^i-b_{-i}$ | leave-one-out 基线 | 原始形式不除组标准差，也不按角色统计 |
| Dr. GRPO | $R^i-\mu$ | 去掉 std 与样本实际长度归一化 | 改为固定归一化，不使用 $\mu_k,\sigma_k$ |
| DAPO | $(R^i-\mu)/\sigma$ | token-mean、Clip-Higher、动态采样、奖励 shaping | 主要修改采样与 loss，而非角色统计 |
| GiGPO | $A_i^E+\omega A_{it}^S$ | anchor-state 步骤分组 | 按状态比较后续回报，提供步骤级信号 |
| GSPO | $(R^i-\mu)/\sigma$ | 序列比率与序列 clipping | 主要修改比率，可与 agent-wise 优势组合 |

## 8. 公式对应到官方代码：实际实现有哪些细节

本节依据本地冻结的代码版本，不代表重新运行了论文训练。

### 8.1 最直接的开关

官方 [数学训练脚本](./DrMAS/examples/drmas_trainer/run_math.sh) 和 [搜索训练脚本](./DrMAS/examples/drmas_trainer/run_search.sh) 仍选择 grpo，并通过 group_by_agent_id=True 启用 Dr. MAS。相关配置可概括为：

~~~yaml
algorithm:
  adv_estimator: grpo
  group_by_agent_id: true
  norm_adv_by_std_in_grpo: true
~~~

这说明 Dr. MAS 在算法实现上接入现有 GRPO advantage estimator，而不是必须新增一个完全不同的优化器。

| 代码位置 | 核验到的行为 |
| --- | --- |
| [ray_trainer.py：compute_advantage](./DrMAS/verl/trainer/ppo/ray_trainer.py) | 关闭开关时按 uid 分组；打开时按 uid 与 agent_id 的组合分组 |
| [core_algos.py：compute_grpo_outcome_advantage](./DrMAS/verl/trainer/ppo/core_algos.py) | 关闭时先在轨迹内平均 response score，再按轨迹计算统计；打开时保留每个活跃 response score |
| [ppo_trainer.yaml](./DrMAS/verl/trainer/config/ppo_trainer.yaml) | 标准差归一化默认打开，loss_agg_mode 为 token-mean |

两处核心逻辑必须一起看：仅给 group key 加 agent_id，却又按轨迹去重，就不等价于论文的活跃步骤加权统计。冻结代码可直接定位到 [分组逻辑](https://github.com/langfengQ/DrMAS/blob/c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9/verl/trainer/ppo/ray_trainer.py#L270) 与 [GRPO 优势实现](https://github.com/langfengQ/DrMAS/blob/c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9/verl/trainer/ppo/core_algos.py#L497)。

### 8.2 论文的总体方差与代码的样本标准差

代码使用未显式修改 correction 的 torch.std，其默认 correction=1 可在 [PyTorch 官方文档源码](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_torch_docs.py#L10444)中核对，样本方差口径对应：

$$
\widehat\sigma_k^2
=\frac{1}{M_k-1}\sum_{(i,t)\in\mathcal{Y}_k}
(R^i-\widehat\mu_k)^2,
\qquad M_k>1.
$$

论文使用分母 $M_k$。若忽略稳定项，在代码的口径下，经验优势二阶矩为 $(M_k-1)/M_k$，而不是严格等于 1。第 6 节的数值例子按论文公式计算，不能将那些数值直接当成该函数的精确输出。

相应地，第 4.3 节“相同出场次数时完全相同”的结论属于总体方差口径；代码在两边样本总数不同时，仍可能存在有限样本 correction 引起的尺度差异。

实现还包括两种边界处理：

- 多个活跃样本奖励相同：中心化分子为零，稳定项避免除零，组内没有相对奖励信号。
- 组内只有一个活跃样本：代码把均值设为 0、标准差设为 1，保留近似原始 reward 信号。这是实现回退，不是样本标准化公式的结果。

### 8.3 response score、token loss 与论文抽象目标

该函数先对每个 response 的 token-level rewards 求和，得到实际 response score。如果各 response 都继承同一个轨迹奖励 $R^i$，就回到前面的公式；若 reward shaping 使不同 response 的 score 不同，则实际参与统计的是这些 response score，不能继续假设它们都严格等于 $R^i$。

随后，一个 response 的优势广播到它的有效生成 token。对非共享模型的 agent $k$，省略 dual-clip 等细节后，默认 token-mean 的主体可写成：

$$
J_{k,\mathrm{token}}^{\mathrm{DrMAS}}
=\mathbb{E}\!\left[
\frac{
\displaystyle\sum_{(i,t)\in\mathcal{Y}_k}
\sum_{\ell=1}^{L_{it}}
\ell_{\mathrm{clip}}(
\rho_{it\ell}^{(k)},A_{\mathrm{agent}}^{i,k})
}{
\displaystyle\sum_{(i,t)\in\mathcal{Y}_k}L_{it}
}
\right],
$$

$$
\rho_{it\ell}^{(k)}
=\frac{
\pi_{\theta_k}(a_{t,\ell}^i\mid s_t^i,a_{t,<\ell}^i)
}{
\pi_{\theta_{k,\mathrm{old}}}(a_{t,\ell}^i\mid s_t^i,a_{t,<\ell}^i)
}.
$$

这是 token 比率和 token 平均；它与论文按输出写出的 $M_k^{-1}\sum\ell_{\mathrm{clip}}(\rho(a_t^i),A)$，在比率与聚合粒度上并非逐字相同。**奖励统计按活跃输出计数，loss 默认按有效 token 计数**，这两个口径也必须区分。

代码的 compute_policy_loss 还包含负优势的 dual-clip 保护；示例脚本关闭显式 KL loss。本文主公式省略这些细节，以单独说明归一化差异。对多 agent GRPO 与 Dr. MAS 做受控比较时，应固定这些其他设置。来源：[loss 实现](https://github.com/langfengQ/DrMAS/blob/c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9/verl/trainer/ppo/core_algos.py#L790)。

## 9. 方法步骤、共享参数与真正的系统贡献

### 9.1 完整训练过程

论文附录 E 的训练过程可以理解为：

1. 将逻辑角色映射到实际 LLM worker group，决定哪些角色共享模型。
2. 对每道题采样一组完整多 agent 轨迹，记录角色、输出、旧策略概率和奖励。
3. 按同题、同角色的活跃输出计算均值、标准差与优势。
4. 将优势广播到对应输出的生成 token，按 worker group 划分训练样本。
5. 对各模型执行策略更新，再进入下一轮采样。

Dr. MAS 没有增加单独的 agent 奖励标注，也不靠为每个状态追加 rollout 来实现核心归一化。

### 9.2 逻辑角色与物理模型可以分开

令 $w(k)$ 表示角色 $k$ 使用的模型 worker group，则：

$$
\theta_k=\theta_{w(k)}.
$$

多个角色可以使用同一个 $\theta_g$，但仍分别计算 $\mu_k,\sigma_k$。共享 worker group 的角色需要使用一致的训练配置。共享参数时，这些角色的训练贡献共同更新该模型。按角色拆开目标后，可写成：

$$
\nabla_{\theta_g}J_g
=\sum_{k:w(k)=g}
\alpha_{kg}\nabla_{\theta_g}J_k.
$$

$\alpha_{kg}$ 由样本组成和 loss 聚合方式决定；采用 token-mean 时，角色的有效 token 数会影响权重。因此，“每个角色分别标准化”并不保证每个角色在共享模型更新中拥有相同权重或相同梯度范数。

### 9.3 论文真正增加了什么

算法上的贡献，是把同题轨迹的组内统计与各角色实际参与优化的样本分布联系起来，并通过优势标准化调整这种错配。最终任务 reward、clipped surrogate、无需 critic 的基本结构均来自现有方法。

系统上的贡献，则包括可插拔的角色编排、共享或异构模型映射、模型对应的训练配置、共享 GPU 资源池，以及后端按需加载和 offload。它让多 LLM 的 rollout 和分别优化可以在统一训练循环中运行。

这些系统能力具有独立价值；不能仅靠优势公式推出吞吐量或 GPU 节省，也不能因两个方法公式相似，就认为其多模型训练能力相同。来源：[论文 §4.3、附录 E、F.6](./DrMAS.pdf#page=6)。

## 10. 实验与消融是否支持这个解释

### 10.1 最关键的消融：均值和标准差分别切换

论文 Table 3 在 Qwen2.5-7B、搜索任务、LLM non-sharing 下比较四种优势：

| 统计选择 | 优势公式 | avg@16 | pass@16 |
| --- | --- | --- | --- |
| 全局均值、全局标准差 | $(R^i-\mu)/\sigma$ | 28.0 | 40.5 |
| agent 均值、全局标准差 | $(R^i-\mu_k)/\sigma$ | 39.1 | 53.5 |
| 全局均值、agent 标准差 | $(R^i-\mu)/\sigma_k$ | 42.9 | 57.6 |
| agent 均值、agent 标准差 | $(R^i-\mu_k)/\sigma_k$ | 43.8 | 58.3 |

只调整均值和只调整标准差都能恢复大量性能，完整版本最好。这支持中心与尺度错配都有关；在这个配置中，标准差切换带来的改善更大，不能据此断言所有任务都由同一部分主导。来源：[论文 §5.4](./DrMAS.pdf#page=9)。

### 10.2 其他结果与梯度曲线

下表摘取论文中同配置的多 agent GRPO / Dr. MAS 对照。数值均按百分数报告，斜杠两侧依次为 avg@16 / pass@16。

| 任务与模型配置 | 多 agent GRPO | Dr. MAS |
| --- | --- | --- |
| 数学，Qwen3-4B，sharing | 56.8 / 70.7 | 59.0 / 73.2 |
| 数学，Qwen3-4B，non-sharing | 57.5 / 74.4 | 61.1 / 77.7 |
| 数学，Qwen3-8B，sharing | 57.8 / 72.1 | 62.3 / 77.6 |
| 数学，Qwen3-8B，non-sharing | 58.1 / 72.2 | 60.7 / 74.2 |
| 搜索，Qwen2.5-3B，non-sharing | 34.5 / 46.7 | 36.9 / 53.0 |
| 搜索，Qwen2.5-7B，non-sharing | 28.0 / 40.5 | 43.8 / 58.3 |

来源：[论文 Table 1–2](./DrMAS.pdf#page=7)。平均指标改善不等于每个 benchmark 单元格都改善，论文表格中也存在个别下降。

Figure 3、Figure 6 和 Figure 7 记录了训练梯度。作者报告，在 7B 搜索 non-sharing 配置中，GRPO 的 Search agent 梯度范数超过 80，随后出现 NaN；Dr. MAS 的曲线更平稳。它是实验现象，不能直接替代第 5 节所讨论的理论条件。

### 10.3 GSPO 组合结果证明了什么

附录 F.5 的 Qwen2.5-3B、搜索、non-sharing 配置中：

| 方法 | 平均 avg@16 |
| --- | --- |
| GSPO | 34.0 |
| GSPO + Dr. MAS | 37.1 |

这是 **3.1 个百分点**的提升。它说明，在该设置中，改变序列比率之后，agent-wise 优势仍然提供额外收益；没有证明所有 group-based 算法或所有配置下都可获得同等收益。

### 10.4 百分点、成功采样指标和统计显著性

对于每题采样 16 次、二值正确性评价的情形：

$$
\mathrm{avg@16}
=\frac{1}{Q}\sum_{j=1}^{Q}
\frac{1}{16}\sum_{\ell=1}^{16}
\mathbf{1}\{\text{sample}_{j\ell}\text{ is correct}\},
$$

$$
\mathrm{pass@16}
=\frac{1}{Q}\sum_{j=1}^{Q}
\mathbf{1}\!\left\{
\sum_{\ell=1}^{16}
\mathbf{1}\{\text{sample}_{j\ell}\text{ is correct}\}>0
\right\}.
$$

$Q$ 是测试题数。avg@16 衡量平均采样正确率，pass@16 衡量至少有一个正确样本的题目比例；它们不是置信区间，也不反映不同训练随机种子的波动。

例如 28.0 到 43.8，是增加 **15.8 个百分点**，相对增加约 56.4%。不能将某一个配置的百分点差，直接当成摘要跨配置汇总的相对百分比。

论文主表没有给出跨训练种子的误差条或置信区间。其 checklist 用报告 avg@16 和 pass@16 作为统计显著性问题的 justification，这两种指标本身并不能建立显著性检验。应将文中数字理解为报告的实验结果，而不是已经获得统计证明的普遍优势。

## 11. 它没有解决哪些问题，什么条件下收益可能有限

### 11.1 角色级标准化仍然不等于因果信用分配

Dr. MAS 对同一轨迹中角色 $k$ 的多个步骤，使用相同的 $R^i$ 和相同的归一化统计：

$$
A_{\mathrm{agent}}^{i,k}(t_1)
=A_{\mathrm{agent}}^{i,k}(t_2),
\qquad k_{t_1}^i=k_{t_2}^i=k.
$$

因此，它不能仅靠这个优势判断“第一轮检索是错误的，第二轮纠正才是关键”。GiGPO 的状态组或 PPO 的状态价值估计能引入额外的步骤区分，但也各有估计与可比性问题。Dr. MAS 没有新增 counterfactual reward。

### 11.2 其他边界

- **角色很少出场**：$\mu_k,\sigma_k$ 的估计可能很不稳定，甚至只能使用 singleton 回退。
- **组内 reward 没有差异**：标准化无法凭空产生学习信号；改变采样或监督是另一个问题。
- **出场次数与奖励无关或各轨迹出场次数相同**：两种统计接近或相同，核心算法改动的收益可能很小。
- **score 或比率本身异常**：agent-wise 优势没有控制所有策略导数、off-policy 偏移和数值误差。
- **共享参数的角色梯度冲突**：标准化不会保证不同角色的梯度方向一致。
- **有限采样与优化偏差**：统计量由同批 rollout 估计，不能自动套用动作独立基线的无偏结论。
- **任务与规模推广**：主要实验是两角色数学、三角色搜索和所选模型；附录的 Stronger-MAS 对照包含额外搜索机制，作者也说明它并非只控制 RL 算法的公平对照。

这些边界与论文结论一致：稳定优势尺度是合作 RL 的一个改进环节，跨 agent、跨轮次的 credit assignment 仍然存在。来源：[论文 §6、附录 F.4](./DrMAS.pdf#page=9)。

## 12. 用公式定位每种方法的主要差别

比较一个实际训练方案时，可以依次检查：

1. **奖励来源**：是最终 reward、过程 reward，还是带 KL / invalid-action shaping 的 response score？
2. **基线与尺度**：是 $V_\phi(s)$、$b_{-i}$、$(\mu,\sigma)$，还是 $(\mu_k,\sigma_k)$？
3. **样本分组**：是同题完整轨迹、同题同角色的活跃输出，还是相同 anchor state？
4. **比率与聚合**：采用 token ratio 还是序列 ratio？按回答平均、token 平均，还是固定常数归一化？

Dr. MAS 最值得记住的关系是：

$$
\boxed{
\text{trajectory-equal statistics}
\quad\longrightarrow\quad
\text{agent-active-output-weighted statistics}
}
$$

它使优势的统计口径更接近角色实际参与训练的输出分布。与 GRPO 的主要区别由 $\mu,\sigma$ 到 $\mu_k,\sigma_k$ 的替换体现；与其他方法的关系，要结合它们各自修改的基线、分组、比率和 loss 加权一起判断。

## 13. 原始资料

- [Dr. MAS 本地 PDF](./DrMAS.pdf)、[项目页](https://langfengq.github.io/DrMAS/)、[官方仓库](https://github.com/langfengQ/DrMAS)。
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)，重点 §4.1。
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，重点 §3、§5。
- [Back to Basics / RLOO](https://arxiv.org/abs/2402.14740)，重点 §2.2–2.3。
- [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/abs/2503.20783)，重点 §3.1–3.2。
- [DAPO](https://arxiv.org/abs/2503.14476)，重点 §3。
- [GiGPO](https://arxiv.org/abs/2505.10978)，重点 §4。
- [GSPO](https://arxiv.org/abs/2507.18071)，重点 §4.1。

文中的公式已统一符号；原文事实在对应段落标注来源。有限组上界、手算例子、等价条件和组合机制的说明属于本文推导或解读，不应当作作者额外声称的定理。
