# 01｜AC2 的 Action Chunking 是什么：与 GRPO 和状态分组的区别

> 依据 [Trust the Critic More 本地论文](./AC2-Trust-the-Critic-More.pdf) §2–§3、Algorithm 1、附录，以及[已归档官方代码](./AC2)。核对日期：2026-10-08。这里解释论文主方法；数值例子为教学构造。完整技术细节见 [analysis.md](./analysis.md)。

**AC2 的 action chunk 是：从已有推理前缀出发，新生成的一段连续 token。它把这段 token 当作一个较大的动作，用续写后状态的价值来评价它。**

关键变化是：critic 可信时，不必把每条续写都生成到最终答案，就能计算优势并更新策略。

## 1. 这里的 action 为什么是一段文字

在文本生成的 MDP 中，可以把每个 token 看成一个动作：

$$
s_j=(x,y_{<j}),\qquad a_j=y_j,\qquad s_{j+1}=s_j\cdot y_j.
$$

$x$ 是问题，$y_{<j}$ 是此前生成的全部推理文本，$\cdot$ 表示拼接。模型生成一个 token，就把它加入上下文。

也可以把连续 $b$ 个 token 合起来，视为一个大动作：

$$
c=(y_j,y_{j+1},\ldots,y_{j+b-1}),\qquad s'=s\cdot c.
$$

这里的 $c$ 就是 action chunk。它可能包含若干推理句子、公式、探索或纠错内容，没有必要刚好对应一个完整语义步骤。

**合成大动作不改变语言模型逐 token 自回归生成的方式。** Chunk 内部依然按顺序生成：

$$
\pi_\theta(c\mid s)
=\prod_{j=1}^{|c|}\pi_\theta(c_j\mid s,c_{<j}).
$$

变化的是采样停止位置和信用分配粒度，不是让 Transformer 一次并行生成全部 chunk token。

## 2. 它不是 CoSo 中的 Thought／Plan／Action 字段

| 概念 | 含义 |
|---|---|
| CoSo 的文本输出 | 生成文字，再解析成实际环境动作 |
| Thought／Plan／Action | 某些 Agent 使用的输出组织形式 |
| AC2 的 action chunk | 连续生成的一段 token，本身就被视为文本 MDP 中的动作 |

AC2 的主任务是长篇数学证明。生成 chunk 后的状态变化，就是把 chunk 拼接到已有证明前缀中。这里不要求存在点击、搜索等外部工具，也不靠动作解析器从 chunk 中抽取一个执行指令。

主实验 chunk 上限为 **10,000 个新 token**，完整响应总预算为 **50,000 token**，后者包含复用的前缀。所以论文所说的短 chunk 是相对于很长的完整证明而言，不是只有几个词的一小步。

## 3. 具体怎样生成：同一个前缀，分叉出多个 chunk

假设过去生成过一份完整证明。训练时，从中截取前缀 $s$，例如已经写了 20,000 token，但尚未完成证明。

AC2 将这个**完全相同的前缀**作为起点，独立采样多条新续写：

$$
c_1,c_2,\ldots,c_g\sim\pi_\theta(\cdot\mid s).
$$

主实验 $g=16$。在允许截断的情况下，每条续写最多生成 10,000 个新 token，然后停止。

例如，为方便展示，只画出四条：

```text
原有前缀 s：问题 + 已生成的 20k token 推理

候选 1：s + 新续写 c1（最多 10k token）
候选 2：s + 新续写 c2（最多 10k token）
候选 3：s + 新续写 c3（最多 10k token）
候选 4：s + 新续写 c4（最多 10k token）
```

这些是四个备选方向，不是四个按顺序执行的 chunk，也不是四个不同角色的 agent。每条 chunk 可以包含复杂推理，但仍属于同一个证明策略的采样。

**它也不是先把所有完整回答生成出来，再事后切成小段打分。** 为这次更新采样时，可信分支本来就可以只生成到 chunk 末尾，因此能省掉剩余 continuation 的解码。

## 4. 没有最终答案，怎么知道这个 chunk 好不好

AC2 使用一个学习得到的 critic，预测从当前前缀继续生成，最终能获得的期望分数：

$$
V^\pi(s)
=\mathbb E_{\text{后续文本}\sim\pi(\cdot\mid s)}[R].
$$

这里 $R$ 是完整证明的终局评分。主实验由证明 judge 给分并归一化，不应直接解释为数学答案 exact match 或形式证明器的二值判定。

生成 chunk $c_i$ 后，状态变成 $s\cdot c_i$。其末端价值为：

$$
v_i=
\begin{cases}
R_i,&\text{该续写已经按终局规则评分},\\
V^\pi_\theta(s\cdot c_i),&\text{该续写在中间位置截断，尚未终结}.
\end{cases}
$$

critic 判断的不是“这段写了多少正确步骤”，而是“从这个完整前缀继续，最终得分预期如何”。

**价值预测会使用原问题、已有前缀和新 chunk，而不是孤立看 chunk。** 实际 generative critic 还可获得参考证明等辅助上下文，详见完整解读。Actor 和 critic 在论文实现中共享参数，通过不同任务 prompt 使用，不是一个额外 frozen LLM assessor。

## 5. Advantage 怎样计算

同一前缀产生一组末端价值后，AC2 主方法计算：

$$
\bar v=\frac1g\sum_{k=1}^{g}v_k,
\qquad
\widehat A_i=v_i-\bar v.
$$

组均值作为共同起点的价值基准，衡量这条续写相对于其他候选是否更有前景。主实验没有再除以组内标准差。

例如四条未完成的续写，critic 给出：

$$
(v_1,v_2,v_3,v_4)=(0.2,0.4,0.4,0.8).
$$

则均值为 $0.45$，优势为：

$$
(\widehat A_1,\widehat A_2,\widehat A_3,\widehat A_4)
=(-0.25,-0.05,-0.05,+0.35).
$$

模型因此可以鼓励第四条续写、抑制相对较差的方向，而不必为了得到这次优势，先把四份证明全部写完。

这个做法依赖 critic 的准确性。若它把错误方向预测得很好，就会提供错误的更新信号；不能把数值预测当成已验证的真实奖励。

## 6. 一个 chunk 内部的 token 都用同一个优势吗

**是，主方法中，同一 chunk 的新 token 共用该 chunk 的 $\widehat A_i$。** 策略的概率比仍然逐 token 计算：

$$
\rho_{i,j}(\theta)
=\frac{\pi_\theta(c_{i,j}\mid s,c_{i,<j})}
{\pi_{\mathrm{old}}(c_{i,j}\mid s,c_{i,<j})}.
$$

然后把 $\rho_{i,j}$ 和共同的 $\widehat A_i$ 放入 clipped policy-gradient 损失。

因此：

- 不同候选 chunk 可以有不同 advantage。
- 同一个 chunk 内，没有进一步给每个 token 估计独立的任务贡献。
- 旧前缀 $s$ 作为条件输入，**不作为本次 actor loss 的目标 token**。这不是说它不影响模型计算，而是没有在这些旧 token 上直接计入策略损失。

所以它是 chunk-level 的信用单位，不是精确的 token-level causal credit assignment。

## 7. 为什么有了 critic 就能提前停止

在本文无中间奖励、无折扣的文本拼接设置下：

$$
Q^\pi(s,c)=V^\pi(s\cdot c),
\qquad
A^\pi(s,c)=V^\pi(s\cdot c)-V^\pi(s).
$$

这表示，如果知道采取 chunk 后的预期终局价值，就可以评价这个 chunk，不必每次亲自采样到终局。AC2 用各候选的末端价值均值近似 $V^\pi(s)$。

使用中间状态的价值估计代替尚未观察到的未来回报，叫作 **bootstrapping（自举）**。代价是引入价值估计偏差，收益是缩短实际生成长度。

在一般有中间环境奖励、折扣或工具调用成本的任务中，$Q$ 还需要包含 chunk 内的奖励及后续价值折扣，不能无条件沿用这个简化等式。

## 8. 不是所有问题都直接信任 critic

AC2 按问题判断 critic 是否达到 readiness 条件，条件涉及近期全局误差、该题局部误差，以及已发现的高分证明等。

| 情况 | 采样方式 | 末端监督 |
|---|---|---|
| 题目尚未 ready | 从 prefix 继续生成到终局 | 终局 judge 分数 |
| ready 且未被抽查 | 通常最多生成一个 10k token chunk | 未终结时使用 critic 值 |
| ready 但被 audit 抽查 | 继续生成到终局 | 终局 judge 分数 |
| chunk 内已经结束或触发终局处理 | 按终局规则处理 | 终局评分，不能一概继续 bootstrap |

主实验对 ready 组的抽查比例约为 $1/4$。此外，系统还生成 fresh 完整轨迹补充 replay。这些通路持续提供外部结果监督，并用于训练 critic。

因此，AC2 没有完全取消完整 rollout，也不是只靠 critic 自我打分。Readiness 也只是经验门槛，不是对该题所有前缀都准确的保证。当前实现 ready 状态保持，不会因一次 audit 错误自动撤销；完整解读另记录了论文主 run 与默认脚本的门槛启用差别。

## 9. 与之前的方法究竟哪里不一样

| 方法 | 从哪里开始比较 | 比较的对象 | 主要评分来源 | 能否因此省掉末尾生成 |
|---|---|---|---|---|
| 普通 GRPO | 同一个问题 | 多份完整回答 | 终局回报 | 通常不能 |
| Prefix GRPO | 同一个历史前缀 | 多份完整续写 | 终局回报 | 省复用前缀的生成，但仍需写完后续 |
| AC2 | 同一个历史前缀 | 多段新续写 chunk | 条件允许时使用末端 critic 值 | 可以省掉该次截断分支的剩余续写 |
| GiGPO | 在已采样轨迹中找重复状态 | 对应动作 | 采样得到的回报信息 | 核心分组本身不能预测未生成的未来 |
| DyJR | 当前在线样本与近期历史成功样本 | 在线 GRPO，加历史 token 概率约束 | 在线终局奖励和历史概率 | 不靠 critic 提前结束当前回答 |
| CoSo | 一次输出内部 | token／扰动窗口 | 动作代理敏感性，用于加权熵 | 不靠终局价值预测截断回答 |

**AC2 的核心不只是“把完整轨迹切小了”。** 它同时改变了采样起点、采样终止时机，以及优势所依赖的评分来源。Prefix GRPO 是重要对照：仅复用前缀，还不足以获得 critic 截断带来的节省。

## 10. 数学任务没有环境返回相同 state，它怎么分组

AC2 无需等待不同推理自然到达相同状态。它主动复制同一段文本前缀，再从这个前缀分叉生成。

因此，共同状态是**同一个问题加同一份完整历史文本**，不是“不同推理已经被证明在数学语义上等价”。

这与 GiGPO 的操作不同：GiGPO 利用轨迹中出现的相同状态来建立比较组；AC2 直接通过固定 prefix 构造共同起点。两种方式都能形成可比较的候选，但建立比较组的方式不同。

## 11. 论文与代码入口

- 论文 §3.1：历史完整轨迹抽取 prefix、ready／unready 采样、audit。
- 论文 §3.2：末端价值、组均值 baseline 与 chunk advantage。
- 论文 §3.3：critic 用组均值目标训练。
- [run_attach.sh](./AC2/experiments/ac2/run_attach.sh)：主实验 10k chunk、50k 总预算及 readiness 配置。
- [ray_trainer.py](./AC2/src/verl/verl/trainer/ppo/ray_trainer.py)：训练与采样流程。
- [sp_q_readiness.py](./AC2/src/verl/verl/trainer/ppo/sp_q_readiness.py)：critic 提示、reference、误差与 ready 状态管理。

代码实体位于 `/usr/data/AC2`，版本为 `5fbb11ba3fbd720b526632259ab047e7e3b70384`。本文核对配置与论文，未运行训练。

## 12. 可直接记录的理解

AC2 将一段连续生成的推理 token 视为一个 action chunk。从历史轨迹截取同一个完整前缀后，模型采样多条候选续写；当 critic 对该题足够可信时，每条续写可以在证明完成前停止，用末端状态的预测终局价值计算组内优势。更新只针对新生成的 chunk，同一 chunk 内共享优势。它用可学习的未来价值预测换取更短的 rollout，而不只是重新分组已经获得的终局奖励。

[完整方法解读](./analysis.md) · [返回 AC2 目录](./README.md)
