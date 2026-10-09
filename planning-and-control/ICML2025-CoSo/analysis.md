# CoSo：让探索熵集中在真正改变环境动作的 token 上

## 1. 为什么普通 token entropy 不够

语言策略生成文本 $y$，环境执行的是解析后的动作 $a=f_{\rm parse}(y)$。很多不同文字会映射到同一个动作。提高无关措辞的熵不一定增加环境动作多样性，还可能破坏输出格式。

CoSo 的关键不是重新分组 rollout，而是估计 token 对实际解析动作的影响，把熵奖励集中到有作用的位置。来源：[论文](https://arxiv.org/abs/2505.03792)、[本地 PDF](./CoSo.pdf)、[代码](./CoSo)。

## 2. 代理模型与反事实影响

训练小型辅助模型 $P_\phi(a\mid y)$，用实际解析动作作为监督标签。对文本中的第 $i$ 个 token 做替换，保持其余文字不变，论文构造类似：

$$
B_i=P_\phi(a\mid y)-P_\phi(a\mid y_{-i}\cup\{\mathrm{null}\}).
$$

如果删改该 token 后预测原动作的概率明显变化，说明它对动作解析更重要。再用影响权重构造熵：

$$
\mathcal H_B(\pi_\theta;y,s)
=\sum_i B_i\,\mathcal H\bigl(\pi_\theta(\cdot\mid s,y_{<i})\bigr),
$$

并在基础 RL 目标中加入：

$$
J(\theta)=\mathbb E\left[\sum_t\gamma^t
\left(r_t+\alpha\mathcal H_B(\pi_\theta;y_t,s_t)\right)\right].
$$

策略接受任务奖励和定向熵正则；辅助模型接受动作标签监督。它可以与不同基础 RL 算法结合，不是另一个 group-relative advantage estimator。

一个直观例子：若环境只读取动作文本中的坐标，那么改变礼貌用语可能不改变 $a$，改变坐标则可能改变 $a$。把探索预算分配给坐标，比让整段说明文字同样随机更直接。

## 3. 论文公式与公开代码的差别

Android 实现的 `parse_action` 用 `android_action.action_type.value` 构造辅助模型标签，因此这里实际预测的是动作类型，不是带全部坐标和输入文本参数的完整动作。上面的坐标例子说明一般机制，不构成代码能精确归因坐标影响的证据。

`digirl/coso/executor.py` 中 `get_token_effect` 默认使用大小为 4 的替换窗口，计算概率变化的绝对值，屏蔽部分特殊符号，再进行序列内 min-max 与平方根变换并 detach。实际权重因此不能直接当成论文单 token、有符号的 $B_i$。

`digirl/models/autoui_agent.py` 的 `get_log_prob` 使用这些影响值计算加权熵；`digirl/algorithms/causal_digirl/trainer.py` 把它作为策略损失中的熵项，并另行训练辅助模型。实现解释需要明确“论文动机公式”和“工程中的非负平滑权重”。

## 4. 这里的 causal 到底指什么

这里做的是 **文本扰动对动作代理预测的影响**。没有为每个 token 真正重新运行环境，也没有重新采样所有后续 token。因此，它不是 token 对最终任务成功率的严格因果贡献估计。

辅助模型不准确、替换文本离开训练分布、多个 token 强交互时，局部影响会有偏差。对解析动作敏感也不必然意味着对长期任务回报重要。方法的合理性来自把探索对准执行接口，而非已经消除了所有因果识别问题。

## 5. 理论与实验怎样解读

论文的软 Bellman 性质依赖固定、受控的加权熵及相应常规假设。不能将该结论扩大为辅助模型与大策略联合训练的全局收敛保证。对确定性的多对一解析，$H(A\mid s)\le H(Y\mid s)$，也说明文本熵和动作熵不能直接画等号。

Android 实验中，DigiRL 平均指标约从 64.9 提升到 72.9，是 8.0 个百分点、约 12.3% 相对提升。论文还在其他环境中结合基础算法验证。实际效益应同时核算辅助模型和多次遮盖预测的成本。

## 6. 与分组优势主线的关系

GRPO/GiGPO/HGPO 决定一个输出“相对于谁表现好”；CoSo 决定探索正则“主要施加在哪些 token 上”。前者是比较与基线估计，后者是探索分配，两者可以互补，但不能把 CoSo 归类成 state 分组算法。
