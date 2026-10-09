# ProGPO：全失败组没有 reward 对比时，用观测覆盖补充优势

> 依据 [本地论文](./ProGPO.pdf) §3–5、附录 §7–12，以及 [本地代码](./ProGPO)。论文 arXiv:2607.22724v1；代码 commit `b14b06248a6021485df14a19a018505fa1f57e5f`。论文与当前代码存在实质差异，见第 6 节。本文沿用 $N$ 条轨迹的符号，论文使用 $K$。

## 1. 核心 motivation：分组再精细，回报全零还是没有比较信号

同一道题采样 $N$ 条轨迹。如果全部失败，且只有二元终局奖励，则：

$$
R_1=\cdots=R_N=0,\qquad G_{i,t}=\gamma^{T_i-t}R_i=0.
$$

GRPO 的轨迹优势、GiGPO 的步骤优势、HGPO 的历史组优势都缺少 reward 对比。这不是组划得不够精细：**把全零样本怎样重分组，仍然是零。** 此处说的是 reward 驱动的 surrogate 项，不是说 KL 等全部训练项必然为零。

但失败过程不一定相同：有的轨迹一直重复观察，有的探索了多个不同位置，只在最后一步失败。ProGPO 把这种行为差异变成替代信号。

它的主要贡献是“何时换一种评分”，而不是新建一种 history 分组。其 simple yet effective 的结构是：**原回报有效时用原优势；全失败时才用弱探索代理；代理也无差异时继续为零。**

## 2. progress 到底是什么

对轨迹 $i$ 的第 $t$ 步，记执行动作后的观测为 $o_{i,t+1}$，此前访问过的观测集合为：

$$
V_{i,t}=\{o_{i,1},\ldots,o_{i,t}\},\qquad
\delta_{i,t}=\mathbf1\{o_{i,t+1}\notin V_{i,t}\}.
$$

只有新观测在**该轨迹中首次出现**才计 1。不是不同 trajectory 之间的新颖性，也不是整个训练过程中的全局访问计数。

轨迹分数为：

$$
P_i=\frac1{T_i}\sum_{t=1}^{T_i}\delta_{i,t}.
$$

这是首次访问转移的比例，不只是新状态总数。除以长度之后，更长的轨迹并不自动得高分。

论文附录定义按环境输出的观测字符串精确匹配，初始观测先放入集合，终止动作后的观测也参与统计。它不是语义相似度，也不等于真实隐藏状态。

例如 $A\to B\to A\to C$，论文指标为 $(1,0,1)$，$P=2/3$。第二步虽然发生状态变化，但只是回到已访问的 $A$，不计首次访问。

## 3. 条件切换的公式

记同题轨迹组的奖励均值、标准差为 $\bar R,\sigma_R$，progress 对应为 $\bar P,\sigma_P$。论文 Eq. (7) 可写为：

$$
A_i^{\rm ProGPO}=\begin{cases}
(R_i-\bar R)/(\sigma_R+\varepsilon),&\sigma_R\ge\tau_R,\\
\lambda(P_i-\bar P)/(\sigma_P+\varepsilon),&\sigma_R<\tau_R,\ \bar R=0,\ \sigma_P\ge\tau_P,\\
0,&\text{其他情形}.
\end{cases}
$$

第一行保留正常 outcome 比较；第二行只修复全失败但探索程度不同的组；第三行包含 progress 也相同及全成功等情况。阈值用于数值判定，不能把任意低 reward 方差都等同于全失败。

**它不是总在 task reward 上叠加 novelty bonus。** 这样设计是因为新观测多不保证更接近目标，漫游失败可能比高效成功更“新颖”。有正常奖励信号时，不让弱代理与它竞争。

若全失败组 $P=(0.75,0.50,0.25,0)$，则 $\bar P=0.375$，总体标准差约 $0.2795$，未乘 $\lambda$ 的优势约为 $(1.342,0.447,-0.447,-1.342)$。若 $P$ 也全相同，仍然没有相对信号。

## 4. 怎么接到 GiGPO、HGPO 和 PPO 上

轨迹级 GRPO 版本可以使用 $P_i$，将一个轨迹优势赋给该轨迹的各步骤。步骤版本保留原有状态或历史分组，在组内对 `step_progress` 做相对比较。不能把这两种实现粒度混成一个公式。

设步骤组是 $\mathcal G$，基础分数为 $G_{i,t}$，步骤 progress 为 $p_{i,t}$，替代分支为：

$$
A_{i,t}^{\rm progress}=\lambda_{\rm eff}
\frac{p_{i,t}-\bar p_{\mathcal G}}{\sigma_{p,\mathcal G}+\varepsilon}.
$$

GiGPO 的代码保留原 episode 项，只在 step 组归一化中加入修复，再合成总优势。HGPO 在历史组计算中加入修复，然后继续层级融合。因此某个步骤组全零，不必意味着整个任务的所有轨迹都失败；论文轨迹级定义与代码局部组条件要分开看。

每步仍使用自己的重要性比率：

$$
\rho_{i,t}=\frac{\pi_\theta(a_{i,t}\mid h_{i,t})}{\pi_{\rm old}(a_{i,t}\mid h_{i,t})}.
$$

ProGPO 改的是 $A$，不是 $\rho$ 的定义、策略架构或 rollout 数量。

## 5. 论文理论能支持什么

### 5.1 覆盖是成功的弱必要条件，不是成功价值函数

令 $D(\tau)$ 是一条轨迹访问的不同观测数，则 $P(\tau)=(D(\tau)-1)/T$。若任务成功至少需要 $m_x\ge2$ 个不同观测，那么成功轨迹必满足 $P\ge(m_x-1)/T$。

这是附加任务条件下的集合包含关系。它**不能推出提高 coverage 必然提高成功率，也不能推出 coverage 梯度与成功梯度方向一致**。必要条件远弱于充分条件；“首次新观测必然是成功必要步骤”也不能对任意环境无条件成立。

### 5.2 非干扰仅指同一批样本上的分支保持

在有正常回报差异的组中，优势与基础方法完全相同。该结论不代表全失败组上的额外参数更新不会影响其他任务或未来 rollout，也不代表整个训练轨迹保持不变。

### 5.3 替代分支不是原任务奖励的无偏梯度

论文附录明确分析了条件失败分布。暂不考虑随机标准差缩放和 progress 阈值筛选，组均值形式的期望包含 $(N-1)/N$ 因子，对应失败条件下的平均 progress 梯度。加入数据依赖的标准差和筛选后，不能继续简单称其为该目标的固定倍数。

若独立轨迹的成功概率为 $p$，整组全失败的概率为 $(1-p)^N$，替代分支触发率不超过它。这个界针对同题轨迹级全失败事件，不可直接替代各历史子组的触发统计。

## 6. 代码核对：当前 commit 不能直接当成论文的完全复现

### 6.1 最重要的差别：C2 没有实现首次访问集合

`agent_system/multi_turn_rollout/rollout_loop.py:501` 的 vanilla 分支实际采用：

```python
obs_changed = str(cur_anchor_i) != str(next_anchor_i)
progress_val = float(obs_changed)
if reward_delta_positive:
    progress_val = max(progress_val, 1.0)
```

另一 rollout 分支约第 1068 行也这样处理。代码注释明确说 revisit 只要观测变化仍计分。这与论文的 $o_{t+1}\notin V_t$、以及仓库 README 对 C2 的描述不同。

对 $A\to B\to A\to C$，当前实现得到 $(1,1,1)$，而论文应为 $(1,0,1)$。反复 $A\leftrightarrow B$ 也会不断计分。实现比较的是环境的 `anchor` 字段，并含正奖励覆盖规则，不可直接描述为严格原始观测首次访问。

因此论文有关首次访问覆盖的恒等式及抑制循环的解释，不能未经修改直接套到这个 commit 的 C2。这里只记录差异，未修改官方源码。

### 6.2 触发条件与自适应尺度

实现一般以组内 $\max|G_{i,t}|<10^{-3}$ 判定全零，并要求 progress 总体标准差至少 $10^{-4}$。论文二元终局设定与折扣步骤分数、WebShop 连续分数的等价性不能直接假定。

GiGPO `gigpo/core_gigpo.py` 中，尺度按 step cluster 的全失败比例计算：

$$
\lambda_{\rm eff}=\lambda_{\rm aux}\,f_{\rm fail},\qquad \lambda_{\rm aux}=0.3\text{（论文主配置）}.
$$

但 HGPO `recipe/hgpo/core_hgpo.py:209` 先按 `index` 的任务/episode groups 估计比例，然后在历史子组上触发修复。因此其缩放统计单位与修复组并非完全相同，不能只用一个“全失败率”掩盖差别。

GRPO 的 `verl/trainer/ppo/core_algos.py` 同时有 episode progress 聚合与 step-level 分支，实际走哪条路径需核对参数；不能只读算法名推断执行粒度。

### 6.3 复现入口

- `recipe/progpo/main_progpo.py`：启动入口。
- `recipe/progpo/progpo_ray_trainer.py`：基础优势算法分发与 progress 传递。
- `recipe/progpo/config/progpo_trainer.yaml`：配置。
- `gigpo/core_gigpo.py:143`：episode 优势保留、step 优势修复与合成。
- `recipe/hgpo/core_hgpo.py:325`：历史组上的修复条件。

本次只做静态核对，未运行分布式训练，也未确认其他历史 commit 是否与论文一致。

## 7. 实验支持与报告边界

论文 Table 1 的部分同文对照如下，均为成功率百分数：

| Qwen2.5-1.5B 方法 | ALFWorld | WebShop |
|---|---:|---:|
| GiGPO | 86.7 | 65.0 |
| GiGPO + ProGPO | 91.4 | 72.4 |
| HGPO | 92.8 | 71.5 |
| HGPO + ProGPO | 95.7 | 75.7 |

并非每个子任务都提升，7B 强基线上的部分增益较小。论文的最终 checkpoint 使用多个推理随机种子重新评估，不能把所有误差条直接理解为多个完整训练种子的方差。

附录列出：rollout group size 8，ALFWorld 最多 50 步，WebShop 15 步，150 epochs；1.5B 用 4 GPU，7B 用 8 GPU。所读计算设置未明确 GPU 型号，不推断为 A100。

论文声称 progress 统计额外耗时不足每 epoch 的 0.5%，无需额外模型或 rollout。条件开关、去掉 novelty 门控以及随机/打乱 progress 的对照用于支持代理信号设计；但鉴于上述代码差异，复现这些结论前必须固定对应实现版本。

## 8. 与我们前面方案的关系

| 方法/方案 | 主要解决的问题 | 全部真实回报为零时 |
|---|---|---|
| GiGPO | 同状态下的局部回报比较 | 没有 reward 对比 |
| HGPO | 当前状态相同但 history 不同 | 仍然没有 reward 对比 |
| 两层历史 baseline 收缩原型 | 历史可比性与小组统计可靠性 | 若全部参考回报也为零，不会凭空产生信号 |
| GraphGPO | 利用经验图结构重评动作 | 没有成功路径时存在结构信号不足问题 |
| ProGPO | 全失败但行为过程不同 | 若 progress 有差异，主动引入探索代理 |

这篇论文对前面讨论最重要的补充是：**“奖励相同但行为不同”需要额外的评分信息，单靠重分组不够。** 历史收缩与 ProGPO 可互补，但组合需要分别验证，不能把 ProGPO 的已有门控创新当成我们的新点。

## 9. 局限及可以借鉴的研究方式

首次访问只是弱代理：无关页面、观测噪声、无效搜索也可能提高分数；必要的回访不一定得到信用。全成功组也不由该设计修复。若 reward 和 progress 均无差异，仍没有更新方向。

最值得借鉴的是贡献组织方式：先定位一个确切失败条件，再用廉价且可观测的差异补救，明确限定何时使用，并设计反例说明为什么不能一直使用。方法简单不等于理论要夸大：这里的覆盖必要性不能替代任务性能保证，代码与论文的一致性也必须单独审计。
