# Trust the Critic More：AC2 方法、公式与代码解读

> 基于 [arXiv:2609.39247v2](https://arxiv.org/abs/2609.39247v2)／[本地 PDF](./AC2-Trust-the-Critic-More.pdf)，以及[官方源码](https://github.com/WhenWen/AC2) commit `5fbb11ba3fbd720b526632259ab047e7e3b70384`。核对日期：2026-10-08。下文将论文主张、实际实现和我的解释分开说明。

## 1. 这篇文章解决什么问题

**AC2 试图让 LLM RL 在没有生成完整答案、尚未取得本条轨迹的终局奖励时，也能进行有用的策略更新。**

GRPO 从同一问题采样多条完整回答，根据每条回答的最终得分计算 advantage，再把这个数分配给该回答的所有生成 token。这里有两个成本：信用粒度粗，以及必须支付完整生成的计算开销。即使一段中间推理已经显著改变成功前景，仍要等后续几万 token 生成完才能得到监督。

AC2 的选择是：保留 actor-critic 思路，用 critic 估计“从当前 prefix 继续生成，预期能得到多少终局分数”。在 critic 足够可靠的问题上，只生成一个 **action chunk**，用末端价值替代本次完整 rollout 的结果。

本篇是**单个证明生成策略的 RL**，不属于 Dr.MAS 那种 solver/verifier 多智能体协作。actor、critic、终局 judge 是训练系统中的不同功能；其中 actor 和 critic 实际共享同一套模型参数。

## 2. Motivation：为什么只改善 baseline 还不够

设一条证明中，前 40k token 的推导有价值，最后一个局部错误导致最终评分低。只用最终回报的训练，会把所有生成 token 放在同一相对优势下，难以局部解释这一段推理本身是否推进了问题。

给 GRPO 加一个更好的 baseline，可以减少方差，甚至让不同位置有不同 advantage；但如果优势仍然写成“实际终局分数减去 baseline”，依然需要完整生成。

因此作者关注的不是“有没有一个叫 critic 的模块”，而是它承担什么职责：

- **baseline 作用**：在拿到真实终局回报后，提供减去的参照项。
- **bootstrap 作用**：直接估计未生成的后半段预期回报，让当前更新不再必须等到终局。

后者更节省采样，但也更容易受 critic 偏差影响。AC2 的主要设计围绕“在什么问题上、什么粒度上、凭什么信息，可以开始信任这个预测”。

## 3. Setup：状态、动作和奖励

沿用论文符号，$x$ 是题目，$y$ 是生成的证明／推理文本，$s=x\cdot y_{<t}$ 是题目加完整 prefix，$\cdot$ 表示拼接。动作既可以是单 token，也可以是一个 token chunk $c$。文本生成的转移确定为 $s'=s\cdot c$。

$$
J(\theta)=\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot\mid x)}[R(x,y)].
$$

$R(x,y)$ 是完整回答的终局分数，目标是提高其期望。当前主实验用证明 judge 的分数归一化作为 $R$，不应理解为整数答案 exact match，也不是 Lean 等形式证明器的机械核验。

$$
V^\pi(s)=\mathbb E_{y_{\mathrm{future}}\sim\pi(\cdot\mid s)}[R(x,y)],
\qquad Q^\pi(s,c)=V^\pi(s\cdot c).
$$

这里的 $V^\pi(s)$ 预测的是**未来完成整题后的预期得分**。它不是“当前已经完成了多少步骤”，也不是给刚才某个 token 直接打正确／错误标签。$Q=V$ 的关系使用了本文确定的文本拼接转移，以及无中间奖励的定义；在一般有即时奖励的工具环境中不能无条件照搬。

$$
A^\pi(s,c)=V^\pi(s\cdot c)-V^\pi(s).
$$

chunk 的价值在于：执行它之后的预期终局得分，比从原 prefix 正常继续生成的预期高多少。这个关系解释了为什么只要价值估计足够好，就不必为每个 chunk 都补完后面的全文。

## 4. 方法总览：三个缓冲区与每轮训练

| 组件 | 保存什么 | 为什么需要 |
|---|---|---|
| actor replay buffer $\mathcal B$ | 历史完整轨迹，成功和失败都收 | 提供可以直接恢复的文本 prefix |
| critic buffer | prefix 与离散价值目标 | 复用价值学习样本，训练 generative critic |
| reference bank | 每题至多一份过去达到评分门槛的证明 | 给 critic 提供参考信息，并参与 readiness 判断 |

主实验三个容器都从空开始。每轮流程如下，细节见论文 §3、Algorithm 1、附录 C：

1. 采样 192 个 fresh problems，每题生成一条完整轨迹，用于补充 replay buffer；这一路的轨迹**不直接进入本轮 actor/critic loss**。
2. 从 replay 中选择 192 个题目对应的轨迹，并抽取 prefix。buffer 尚空时，训练组也从空 prefix 开始。
3. 每个 prefix 采样 $g=16$ 条新 continuation。题目 unready 时生成到终局；ready 时通常只生成最多 10k 新 token。
4. 对 ready 组抽查约 $1/4$，抽查组仍生成到终局，取得独立于 critic 预测的评分信号。
5. 根据每条 continuation 的终局评分或末端预测价值计算组内 advantage。
6. 用新生成 token 做 actor 更新；用组均值构造 critic 学习目标。
7. 更新缓存、reference bank 和 readiness 状态，继续下一轮。

**这是在同一个完整 prefix 上主动分叉采样**。它与 GiGPO 在既有不同轨迹中寻找重复状态再重分组的操作不同。prefix 包含 history，因此同一 observation 但历史不同，在这里并不会自动被合并为同一状态。

## 5. 数学公式与训练细节

### 5.1 从 GRPO 的终局分数换成 AC2 的末端价值

对同一个 prefix $s$，采样 $c_1,\ldots,c_g$，定义

$$
v_i=
\begin{cases}
R(x,s\cdot c_i),&\text{该 continuation 按终局评分规则处理},\\
V^\pi_\theta(s\cdot c_i),&\text{达到 chunk 截断点且仍需继续生成}.
\end{cases}
$$

$v_i$ 是这条 continuation 末端的价值。实现中，已经形成完整 proof、结束生成或耗尽总响应预算的输出交给 judge；不能简单把“生成到了长度上限”全部当作可 bootstrap 的中间状态。

$$
\bar v=\frac1g\sum_{j=1}^g v_j,
\qquad \hat A_i=v_i-\bar v.
$$

$\bar v$ 充当共同 prefix 的价值 baseline。**主实验没有除以组内标准差**。同一 chunk 的所有新 token 共用 $\hat A_i$，不同 chunk 可以获得不同的符号与大小。

在 unready／audit 的完整 continuation 组中，$v_i=r_i$，公式成为从 prefix 出发的 GRPO 型结果优势。ready 的截断组才真正使用 critic 值替代本次终局回报。

一个纯示意例子：四条 chunk 还没完成证明，critic 给出 $[0.2,0.4,0.4,0.8]$，组均值为 $0.45$，则 advantage 为 $[-0.25,-0.05,-0.05,0.35]$。这里已经能比较续写方向，而不必为了这次更新把四条都写到结束。预测如果有系统性错误，更新方向也会错，这正是 readiness 必须处理的风险。

### 5.2 与 GAE 的关系

论文取无折扣的终局奖励设置，并把终端状态的 value 定义为终局评分。于是

$$
\delta_t=V(s_{t+1})-V(s_t),
\qquad
\hat A_t^{\mathrm{GAE}(\lambda)}=\sum_{l=0}^{T-t}\lambda^l\delta_{t+l}.
$$

$\lambda$ 决定沿未来状态累加多少价值残差。在论文写出的完整轨迹形式下，两个端点是

$$
\hat A_t^{\mathrm{GAE}(0)}=V(s_{t+1})-V(s_t),
\qquad
\hat A_t^{\mathrm{GAE}(1)}=R(x,y)-V(s_t).
$$

前者可以只凭相邻 chunk 状态的预测值计算；后者通过望远镜求和变成终局回报减 baseline。AC2 的 readiness 可理解为在不同问题上选择更接近 $\lambda=0$ 的 bootstrap 更新，或完整采样的结果更新，组均值则替代直接的起点值 baseline。

**边界说明**：论文将其与完整轨迹 GAE 对照；一般 RL 也存在截断 rollout 后 bootstrap 的 GAE／n-step 方法，不能将文中的叙述扩展成“所有 $\lambda>0$ 的算法在数学上都必须跑到真实终局”。AC2 的创新主要是其 LLM 设置与可靠性配方，而不是重新发明 Bellman 关系。

### 5.3 actor loss：改变 advantage 和采样，仍使用 PPO 型更新

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(c_{i,t}\mid s,c_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(c_{i,t}\mid s,c_{i,<t})}.
$$

每个 importance ratio 都在该 token 自己对应的同一条件上下文下比较新旧策略概率。换用 critic advantage 不改变这一点。

$$
\mathcal L_{\mathrm{actor}}(\theta)=
-\frac1{\sum_i\lvert c_i\rvert}
\sum_i\sum_{t=1}^{\lvert c_i\rvert}
\min\!\left(
\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})\hat A_i
\right).
$$

这是说明主机制的 clipped surrogate，$\hat A_i$ 在更新中固定。实际实现另有 dual clip 和上界自适应，见附录 C。**replay prefix 只作 conditioning，不进入 actor loss**；被强化的是当前策略新生成的 continuation。复用的是旧状态，后续动作重新采样，不能描述成不加区别地反复拟合旧轨迹所有 token。

### 5.4 critic 是生成式价值模型，与 actor 共享权重

critic 输入包含题目、部分尝试，以及可用时的参考证明；输出是离散集合

$$
\mathcal Z=\{0,0.1,0.2,\ldots,1\}
$$

中的数值，例如 `Q value: 0.6`。代码命名含 Q，但论文建模解释为 prefix 的 $V^\pi(s)$。它最多贪心生成四个新 token；**不是输出详细评审意见，也不是一个固定不训练的外部大模型**。

目标由组均值量化而来：

$$
z(s)=\operatorname{round}_{\mathcal Z}\!\left(\frac1g\sum_i v_i\right).
$$

就近舍入，正好同距时向较小值舍入。完整组提供终局平均分目标，截断组提供 bootstrap 目标。它不是人工标注的逐 token 正误，也不是对所有 prefix 配了 gold step score。

$$
\mathcal L_V(\theta)
=-\frac14\sum_{u\in\operatorname{tokens}(z)\,+\,\mathrm{EOT}}
\log p_\theta(u\mid\operatorname{critic\_prompt}(s),u_{<}).
$$

这里监督的是价值答案 token 和结束 token，按代码最大输出长度 4 归一化。**不是标量头上的 MSE 回归**。若有参考证明，对带参考与不带参考两个 prompt 分别训练同一个目标，各权重 $1/2$；无参考时只训练不带参考的版本。

### 5.5 readiness：按题目决定何时允许 bootstrap

实现使用绝对误差：

$$
e(s)=\left|V^\pi_\theta(s)-z(s)\right|.
$$

论文正文一处用未加绝对值的残差记 $\varepsilon$，但诊断称 MAE，代码明确用 `abs`，因此这里按实现解释。门槛使用更新前的预测；不可解析的预测记误差 1。

新进入 ready 集合，主设计需要同时满足：

- 最近 5 个完整训练步中所采 prefix 的平均误差低于 $\tau_{\mathrm{global}}=0.20$。
- 该题最新一次 probe 的误差低于 $\tau_{\mathrm{local}}=0.18$。
- 当前或过去至少预测过非零价值。
- reference bank 中已有该题达到 judge 门槛的证明。

reference 的门槛是 **至少 6/7 分**，并非形式保证绝对正确。每题首次收录后不再替换。replay buffer 可以收失败轨迹，reference bank 的收录条件则更严格，两者不要混淆。

**一旦 ready，当前实现保持 ready**，并非 audit 一发现误差变大就自动撤销资格。audit 提供真实结果、监控和校准信号，但没有实现完整的动态撤销机制。

还应记录论文公开承认的实现偏差：主 run 直到 **step 42** 才启用“必须有 reference”这项门槛；此前已 ready 的题不会被撤回。当前公开 launch 脚本默认从开始启用该条件，因此“按当前默认脚本重跑”不完全等同于复现论文主 run 的前 41 步。

### 5.6 两套优化器：共享参数不等于共享 Adam 状态

每步按下面的顺序更新同一组 $\theta$：

```text
actor minibatch 1 → critic update → actor minibatch 2
```

两种目标分别保存优化器状态和学习率。critic 使用 AdamW，betas 为 $(0.9,0.999)$，weight decay 为 0；actor 使用 FSDP 配方的 AdamW，学习率 $2\times10^{-6}$，weight decay 0.01。代码入口见第 12 节。

令 $\Delta\theta_V$、$\Delta\theta_{P,1}$、$\Delta\theta_{P,2}$ 分别为三次更新引起的参数位移，critic 的调度监控

$$
q_{\mathrm{update}}=
\frac{\lVert\Delta\theta_V\rVert_2}
{\lVert\Delta\theta_{P,1}\rVert_2+\lVert\Delta\theta_{P,2}\rVert_2}.
$$

若该比值连续两步大于 $\sqrt2$，critic 学习率减半，初值 $2\sqrt2\times10^{-6}$，下限 $5\sqrt2\times10^{-7}$。这是控制 critic 更新对共享模型的影响，与我们之前讨论的“辅助目标如何影响优化器动态”直接相关；不能把它归为单纯 advantage 分组技巧。

## 6. 为什么它可能有效

**第一，缩短实际采样。** 价值模型能够估计剩余生成的期望收益，就能用一个 chunk 替代完整 continuation。节省来自提前停止，而不只是把同一批完整 rollout 的损失重新加权。

**第二，信用单位变成 continuation chunk。** 其共同起点是同一个 prefix，差别来自刚刚采取的不同续写；成功前景可以在终局前产生可学习差异。不过 chunk 内 token 仍共用一个 advantage，所以 AC2 不提供逐 token 因果归因。

**第三，先在局部证明 critic 有用，再提高对它的依赖。** global MAE 低不意味着所有题都估计得准；按题设置门槛可以避免容易题掩盖困难题。论文的消融支持这种作用，但不是对所有问题的误差保证。

**第四，reference 和较大的 chunk 让价值判断更有依据。** reference 提供一种已经接近完整正确的解法作为参照；10k token 通常比单 token 体现出更可判别的推理变化。这是作者的机制解释，不能推导为“chunk 越大越好”或“10k 对所有任务都最优”。

**第五，保留真实结果通路。** fresh refill、unready 的完整 continuation 与 ready 题的 audit 仍持续产生终局评分。AC2 是受条件控制的 bootstrap 系统，不是脱离外部反馈后让模型永久自我打分。

## 7. 与 GRPO、GiGPO、Dr.MAS 的区别

| 方法 | 比较／归一化对象 | 信用依赖什么 | 是否因此能省掉本次完整 continuation |
|---|---|---|---|
| GRPO | 同题多条完整回答 | 终局回报的组内相对差 | 通常不能 |
| GiGPO | 从采样轨迹中找到相同状态，对相关动作建立更细的组 | 已采样轨迹的回报／回报到达 | 其核心重分组本身不提供未生成后续的价值预测 |
| HGPO／GraphGPO | 层级组／轨迹合并图上的状态动作结构 | 采样结果与其结构化信用计算 | 核心贡献不是 AC2 这种受门控的 chunk bootstrap |
| Dr.MAS | 按参与协作的 agent 统计优势归一化量 | 系统 rollout 的结果奖励 | agent 级归一化本身不能替代未获得的终局分数 |
| AC2 | **从同一个 replay prefix 主动分叉出的 chunks** | ready 时用末端 critic 值，其他组用真实终局评分 | **可以，对 ready 且非 audit 的截断 continuation** |

这里的前几行是对本地已整理方法的比较，不是说 AC2 论文实验直接跑了 GiGPO、HGPO、GraphGPO 或 Dr.MAS。AC2 的主对照是 **GRPO 与 Prefix GRPO**。

Prefix GRPO 非常关键：它也复用历史 prefix，但每个 continuation 仍生成到终局。这个对照用于区分“replay 改变起点分布”的影响与“critic 使截断更新成为可能”的影响。

## 8. 数据集：有没有过程、到底使用什么监督

| 数据／资源 | 内容 | AC2 的实际使用 |
|---|---|---|
| [`lm-provers/FineProofs-RL`](https://huggingface.co/datasets/lm-provers/FineProofs-RL) | 5,227 道证明题；数据卡列 problem、source、**rubrics**、已有 rollout 的 scores/rewards 及统计。rubric 是模型生成的评分检查点，不等于一条完整标准 CoT | actor parquet 只由 problem 构造题目；**主实验训练 judge 不读取逐题 rubric 或标准参考解答**；历史 scores 也不是当前策略新 rollout 的 reward |
| [`lm-provers/IMOProofBench`](https://huggingface.co/datasets/lm-provers/IMOProofBench)／[官方 proofbench CSV](https://github.com/google-deepmind/superhuman/blob/main/imobench/proofbench_v2.csv) | 60 道评测题，官方资源提供 Solution 和 Grading guidelines | 评测 judge 读取参考解答与逐题评分指南；这些内容不输入证明生成策略 |
| 在线 reference bank | 当前训练中生成、judge 至少给 6/7 分的证明 | 提供给 critic 的 privileged context；从空开始积累，不是预先载入所有训练题的 gold solutions |

**因此不能说该数据生态完全没有过程资源，也不能说 AC2 直接拿 FineProofs 的标准步骤训练 critic。** 原始 rubric、评测参考解答、在线生成的 reference，是三个不同来源。

代码还澄清了奖励粒度：论文概括为 0–7 分，训练默认 prompt 的主要评分菜单是 $\{0,1,6,7\}$，reward 为 points/7；解析器可接受 0–7 整数。`points >= 6` 是收录成功 reference 的判定，和连续／分级的 actor reward 是不同变量。

预处理文件的开头仍有“binary 1/0”旧注释，但主配置接入的是 `ds4_finegrained_judge`，不能只读那个注释就把整篇实验写成二元奖励。论文和实际选中的 reward 模块一致支持分级评分。

## 9. 实验、消融与证据

| 项目 | 结果／设置 | 应如何理解 |
|---|---|---|
| backbone | Qwen3-4B-Thinking-2507 | 当前主要实证限于这个模型与长证明任务 |
| 平均评测得分峰值 | AC2 20.57%；GRPO 18.50% | 是归一化证明评分的均值，不是 20.57% 的题拿满分 |
| 超过 GRPO 峰值所需 decoding FLOPs | $0.79\times10^{20}$ 对 $1.99\times10^{20}$，约 2.5 倍效率 | 只针对论文的 actor decoding 计量口径 |
| 达到该水平的步数 | AC2 90；GRPO 120 | 少 25% steps；两者每步采样预算不同，不能直接当成等样本量比较 |
| main batch | AC2 192 prefix groups × 16，加 192 单条 refill；GRPO 256 fresh groups × 16 | 明确回答“同一个训练 step 的 rollout 数是否一样”：**不一样** |
| group size 1 变体 | ready 题用 $v-V(s)$，表现相近 | 每题改采 16 个不同 prefix，各续写一次，维持该题总 continuation 数；不是简单把 16 条缩成 1 条 |
| 2k chunk 消融 | 后期弱于 10k | **同时改了 prefix cut 间隔**，不能宣称单独证明 chunk 长度的因果效果 |
| 只保留成功 replay | 初期有益，后期变差 | 主方法保留失败轨迹；结果不支持必须做成功轨迹过滤 |
| 取消 local readiness | 学习停滞／下降 | 该消融在去 group 与 audit 的设置上，并从 step-20 checkpoint 分支，需带着实验条件解释 |
| 去 audit | 峰值 18.66%，略不稳定 | 不是完全失败，但主方法更好，audit 仍有监控价值 |
| critic 诊断 | step 80 混合末端组均值 MAE 0.065，起点预测 MAE 0.211 | 部分组全由真实终局值组成，误差按构造为 0；全 critic 组的 MAE 为 0.106，不能把 0.065 当纯 critic 的单样本预测误差 |

算力口径尤其要注意：附录 B 的 decoding FLOPs **不包含 prefill、critic 推理、judge、训练前反向传播、验证和通信**，但计入 actor refill 生成。作者展示它与主 run GPU-hours 相关，不能因此直接推出完整训练成本节省 2.5 倍。

代码和附录确认主实验使用 **32 张 GPU（4×8）**，judge 共置在训练节点上。当前核对到的论文与启动文件没有可靠的卡型号说明，不能自行补成 A100／H100／H200。没有在本地复跑训练。

## 10. 局限与适用边界

1. **需要多次遇到同一道题。** local readiness 要积累该题的真实结果和误差信息；作者明确把单 epoch／持续全新题目设置留作未来工作。
2. **bootstrap 存在偏差累积。** 当前 global error 也可能包含 critic 目标，不能用模型对自身预测一致性替代外部校准；audit 的纯终局诊断需单独看。
3. **reference 门槛是 judge 判定。** 6/7 分证明可能仍有缺陷，错误参考可能影响价值判断；并且 reference 不会被更新替换。
4. **readiness 没有自动撤销。** 策略变化后已 ready 的题可能不再可靠，目前实现依赖后续训练和 audit，而不是严格持续可信保证。
5. **10k token 仍是粗粒度。** 它比 50k 完整证明细，但不等于 turn-level 工具调用或 token-level 归因；chunk 内仍统一 advantage。
6. **文本 prefix 可以直接恢复，一般环境未必可以。** 把它扩展到 WebShop、ALFWorld 或 Dr.MAS，需要分别处理外部状态恢复、多角色上下文和团队价值定义。
7. **共享参数与两个优化器可能相互干扰。** critic LR 的位移控制是实际稳定性设计；两个目标并非彼此独立的网络训练。
8. **实证范围与实现偏差有限制。** 主任务、模型、评测规模有限；默认脚本与论文主 run 的 reference gate 启用时间也不同。

论文没有给出新的 AC2 收敛定理或全局最优保证。Bellman／policy-gradient 公式提供动机，readiness、chunk 长度、reference 条件主要由实验验证。

## 11. 真正的贡献与对我们讨论的启发

我认为核心贡献是：**给出一套能在长文本证明任务上实际运行的、按题选择何时依赖 learned value 的截断 actor-critic 配方**。分组优势仍在，但更重要的是将其输入从“必须等到的终局回报”变成“可在中间状态预测的预期收益”，由此改变了采样成本。

这与我们之前的研究主线有两处直接联系：

- 研究信用分配时，除了“如何对既有 $(s,a)$ 重新分组”，还可以研究“如何选择恢复点，并决定何时停止 rollout 去更新”。AC2 已经实现了其中一条路线，新方法需要进一步区别于其 readiness 和 chunk bootstrap。
- 你提出的“只在全失败组启用 frozen credit assessor”与它并不相同。AC2 的 critic **会训练并与 actor 共享参数**，主要在题目达到可靠门槛后使用；未达门槛、尤其尚无成功 reference 的题，默认保留完整采样。它不是专门为 all-fail group 输出 token mask 的方法。

后续若研究 multi-agent 扩展，关键不只是换上 solver/verifier 标签，而是明确 critic 预测谁的策略下的团队终局值、是否能恢复该 turn 的真实环境／共享上下文，以及可靠性门槛如何覆盖不同角色和不同历史。这些是本文未验证的扩展方向。

## 12. 代码对应位置

| 要核对的问题 | 文件 |
|---|---|
| 主实验配置、step-42 门槛问题、硬件与公开复现说明 | [experiments/ac2/README.md](./AC2/experiments/ac2/README.md) |
| rollout 数、chunk、readiness、学习率开关 | [run_attach.sh](./AC2/experiments/ac2/run_attach.sh) |
| 题目与 replay prefix 的采样路由 | [q_dataset.py](./AC2/experiments/ac2/q_dataset.py) |
| critic prompt、量化、误差、ready 状态与 reference bank | [sp_q_readiness.py](./AC2/src/verl/verl/trainer/ppo/sp_q_readiness.py) |
| actor／critic 交错更新 | [ray_trainer.py](./AC2/src/verl/verl/trainer/ppo/ray_trainer.py) |
| 第二套 critic AdamW | [engine_workers.py](./AC2/src/verl/verl/workers/engine_workers.py) |
| actor 的 FSDP optimizer 默认配置 | [optimizer.py](./AC2/src/verl/verl/workers/config/optimizer.py) |
| FineProofs 与评测题的 parquet 构造 | [prepare_fineproofs.py](./AC2/src/ac2/data/prepare_fineproofs.py) |
| 训练与验证 judge 的不同输入／评分分支 | [ds4_finegrained_judge.py](./AC2/src/ac2/rewards/ds4_finegrained_judge.py) |
| 评测 reference solution 和 grading guidelines 来源 | [build_rubric_map.py](./AC2/src/ac2/data/build_rubric_map.py) |

本次完成论文与源码静态核对、PDF 与文档格式检查；没有运行 GPU 训练，也没有将源代码仓库的测试结果冒充实验复现。
