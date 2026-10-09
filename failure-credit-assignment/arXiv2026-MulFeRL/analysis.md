# MulFeRL：全失败时请求反馈，用重新生成后的真实进步训练

> 依据 [论文 v2](./MulFeRL.pdf) §3–5、附录 A–B、F，以及 [公开代码](./MulFeRL)。commit：`6616a07eb89d32721e39faa187eae01e48685978`。整理日期：2026-10-07。公开仓库的来源边界与缺失文件见 [归档说明](./README.md)。

## 1. 核心 idea 与问题

同一道题采样 $K$ 条解答全部错误，GRPO 没有组内对比。但直接让语言模型把错误轨迹“评得有高有低”，又可能学习到不可靠的主观评分。

MulFeRL 的选择是：**语言反馈只负责帮助模型再做一次；训练信号必须来自 verifier 确认的结果改进。** 它把全失败组变成可学习组，而不是直接对原始失败轨迹的每个 turn 评分。

## 2. Setup 与 turn 的含义

一题 $x$、一个可训练策略 $\pi_\theta$、每轮 $K$ 条完整回答、结果 verifier $\mathcal V$，另有反馈来源 $F_\psi$。反馈可以由外部模型提供，论文还测试冻结基础模型自反馈。

这里的 $t$ 是第几轮“生成一组回答—得到反馈—重试”，不是团队里某个角色的一次发言。论文主设置初始一轮加至多两轮反馈重试，每轮 $K=8$；最坏会生成 24 条完整解答，还要支付反馈推理成本。

它属于单策略推理训练，不是 MAGRPO 团队训练。冻结行为策略 $\pi_{\mathrm{old}}$ 是当前外层迭代的快照；不应与整个训练保持固定的外部反馈模型混为一谈。

## 3. 算法关键步骤

1. 用原始问题上下文 $c^{(0)}=x$ 采样 $K$ 条回答。
2. verifier 对每条回答计算二元奖励，区分全成功、混合、全失败。
3. **混合**：直接用当前组的成功/失败差异进行 GRPO，结束该题本次处理。
4. **初始就全成功**：没有组内对比，不需要反馈，该题返回零目标。
5. **全失败**：请求反馈，汇总主要错误和修正步骤，把反馈注入固定槽位，再生成一组 $K$ 条回答。
6. 如果新组是混合组，使用新组的 GRPO；如果新组全成功，把当前成功组与紧邻的上一失败组组成 FCO 配对。
7. 如果仍全失败，继续反馈重试，直到达到预算；耗尽后仍无成功就不更新。
8. 计算训练 loss 时屏蔽外部注入的反馈 token；它们仍是后续生成可见的上下文。

关键在第 6 步：**全失败变成全成功也是进步，但新组内部全是 1，GRPO 本身仍学不到；FCO 专门补上这部分。**

## 4. 关键公式

### 4.1 状态判定与反馈干预

$$
\mathcal G^{(t)}=\{y_i^{(t)}\}_{i=1}^{K},\qquad
r_i^{(t)}=R_{\mathcal V}(x,y_i^{(t)})\in\{0,1\}.
$$

$\mathcal G^{(t)}$ 是一次完整候选组，$r$ 由结果验证器给出；论文还要求答案正确且输出格式合规才得 1。

$$
z(\mathcal G^{(t)})=
\begin{cases}
\mathrm{Failed},&\sum_i r_i^{(t)}=0,\\
\mathrm{Contrastive},&0<\sum_i r_i^{(t)}<K,\\
\mathrm{Solved},&\sum_i r_i^{(t)}=K.
\end{cases}
$$

这个判定决定去哪里找对比信号，不是给 agent 分角色。

$$
f^{(t)}\sim F_\psi(\cdot\mid x,\mathcal G^{(t)},\{r_i^{(t)}\}),
\qquad c^{(t+1)}=\operatorname{Inject}(x,f^{(t)}).
$$

反馈先改变下一轮的生成条件。实现采用 subgroup 反馈再汇总，以处理长回答；反馈包含 Issue 和 Fix steps。反馈文字本身不被直接当成 reward。

### 4.2 混合组：GRPO

$$
A_i^{(t)}=\frac{r_i^{(t)}-\bar r^{(t)}}{\sigma_r^{(t)}+\epsilon},
\qquad
\rho_i^{(t)}=\frac{\pi_\theta(y_i^{(t)}\mid c^{(t)})}
{\pi_{\mathrm{old}}(y_i^{(t)}\mid c^{(t)})}.
$$

这是论文主文使用的回答级记号；实际训练按 token 计算 masked loss。组内所有回答共享当前上下文 $c^{(t)}$，成功为正、失败为负。

$$
\mathcal L_{\mathrm{GRPO}}=-\frac1K\sum_i
\min\bigl(\rho_i^{(t)}A_i^{(t)},
\operatorname{clip}(\rho_i^{(t)},1-\varepsilon,1+\varepsilon)A_i^{(t)}\bigr)
+\beta_{\mathrm{KL}}D_{\mathrm{KL}}.
$$

混合组已有 verifier 提供的区分度，不需要再添加语义奖励。

### 4.3 全失败到全成功：FCO

从当前成功组取 $y^+$，从上一失败组取 $y^-$，随机打乱失败回答后配对。不是认为相同 rollout 编号之间具有因果对应。

为把反馈 mask 写清楚，用 $m_\ell$ 标记可训练输出 token，定义：

$$
s_\theta(y;c)=\frac{\sum_\ell m_\ell\left[\log\pi_\theta(y_\ell\mid c,y_{<\ell})-
\log\pi_{\mathrm{ref}}(y_\ell\mid c,y_{<\ell})\right]}{\sum_\ell m_\ell}.
$$

这是论文长度归一化 log-ratio 的 token 展开。分子只统计策略生成 token；反馈仍存在于条件历史中。$\pi_{\mathrm{ref}}$ 是参考策略，不是用于采样的行为快照。

$$
\mathcal L_{\mathrm{FCO}}^{(t)}=-\frac1{|\widehat{\mathcal Q}_t|}
\sum_{(y^+,y^-)\in\widehat{\mathcal Q}_t}
\log\sigma\!\left(\tau\left[s_\theta(y^+;c^{(t)})-
 s_\theta(y^-;c^{(t-1)})-\gamma\right]\right).
$$

$\tau$ 是温度尺度，$\gamma$ 是 margin。最小化该式提高成功回答相对参考模型的分数，并降低失败回答的对应分数。形式接近偏好优化，但两项条件上下文不同；不能把它解读为同一 state 下两个 action 的因果优劣比较。

### 4.4 完整分支

令 $t^*$ 为首次脱离 Failed 的轮次：

$$
\mathcal L(x)=
\begin{cases}
\mathcal L_{\mathrm{GRPO}},&z(\mathcal G^{(t^*)})=\mathrm{Contrastive},\\
\lambda_{\mathrm{FCO}}\mathcal L_{\mathrm{FCO}}^{(t^*)},&t^*>0,\ z(\mathcal G^{(t^*)})=\mathrm{Solved},\\
0,&\text{初始全成功，或预算内始终全失败}.
\end{cases}
$$

这就是训练路由：在当前组内有差异时比较当前组，在反馈前后有差异时比较前后组。它没有给原始失败回答里的每个 token 输出独立 credit。

## 5. 一个 N=8 的例子

- 第一轮：`00000000`，请求反馈。
- 第二轮若是 `00100100`：对第二轮八条回答做 GRPO，结束。
- 第二轮若是 `11111111`：第二轮的 GRPO 全零，用第一轮失败与第二轮成功组成 FCO 配对。
- 第二轮若仍是 `00000000`：继续反馈；最后一轮仍全失败则跳过。

所以它能学习“从失败变好”的样本，但不能保证让“怎么重试都失败”的样本获得非零信用。

## 6. 理论到底支持什么

在给定反馈上下文后，假设回答独立、单条成功概率为 $q$：

$$
P(\text{混合组})=1-q^K-(1-q)^K,\qquad
P(\text{全成功})=q^K.
$$

GRPO 只能利用前一类，而 FCO 在已有失败前驱时补充后一类，于是可用更新事件概率成为 $1-(1-q)^K$。这解释 FCO 恢复的事件概率质量为 $q^K$。

附录还推导局部梯度的成功/失败对比方向，以及随机配对对全配对平均目标的无偏性。**这些不证明任意反馈都能提高 $q$，也不证明 FCO 是原始无反馈策略目标的无偏梯度。**

## 7. 实验证据和限制

训练使用 OpenR1-Math 的 4k 样本，模型包括 Qwen2.5-7B-Base 和 Qwen3-4B-Instruct-2507，主要反馈来源为 GPT-4o。主评估包括数学和科学/通用推理任务，默认测试时单次生成且无外部反馈。论文比较去掉 regeneration、去掉 FCO、去掉结构化注入，以及不同反馈模型。

Table 2 的 7B 消融中，MATH500 从完整方法 78.72 降到无 regeneration 的 70.84、无 FCO 的 76.12。这支持不同模块各有作用，但不能证明对多智能体归因也有效。

主要代价是重试和反馈开销；需要可靠 verifier；反馈可能泄露解题信息，使监督强度与纯终局 RL 不同。比较训练成本应计入反馈模型的生成 token，不能只对齐 policy 更新次数。

## 8. 代码核对

- [启动配置](./MulFeRL/verl-main/examples/sglang_multiturn/MulFeRL/mulferl.sh)：`rollout.n=8`，配置反馈工具和 FCO 参数。
- [训练路由](./MulFeRL/verl-main/verl/trainer/ppo/ray_trainer.py)：`_prepare_grpo_and_fco_batches` 按 uid 区分 base/regen；regen mixed 进入 GRPO、all-positive 进入 FCO、all-negative 丢弃。实现以总分 `>0` 判正，因此依赖接入的 verifier 遵循预期奖励定义。
- [FCO loss](./MulFeRL/verl-main/verl/workers/actor/dp_actor.py)：`compute_fco_loss_from_micro` 计算 masked、长度归一化的参考 log-ratio 差及 logistic loss；同一对的 chosen/rejected 必须同时出现在 microbatch。
- 代码有缺省 reference log-prob 时使用零值的分支；论文目标需要参考项，复现应确认实际配置提供了 `ref_log_prob`。
- README 明确未发布 reward functions、benchmark evaluators 和 evaluation scripts。这里核对的是算法关键分支，未运行完整训练，也未验证每轮缓存始终对应论文的紧邻失败组。

## 9. 对我们 idea 的启发

“全失败才触发外部帮助”已有明确先例。可区别之处是：我们想不再生成新答案，而直接识别失败团队内部有价值的行为；MulFeRL 则要求实际重试产生成功。前者省 rollout 但更依赖 assessor 的语义判断，后者依赖结果验证但多花采样预算。
