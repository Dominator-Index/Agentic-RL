# Memento：不微调底层 LLM，通过案例记忆改善 Agent

> 依据 [论文 PDF](./Memento.pdf) arXiv:2508.16153v2（2025-08-25）、正文 §3–6 和附录 A/B，以及 [官方代码](https://github.com/Memento-Teams/Memento) 整理。代码 commit 见 [code_manifest.json](./code_manifest.json)，核对日期 2026-10-07。

## 1. 这篇文章解决什么问题

Memento 希望 agent 能从过去成功和失败的任务中持续学习，而不必更新负责规划与执行的 LLM 权重。它采用 planner–executor 架构，把历史任务、方案和结果保存为案例，在处理新任务时选择有用案例放入 planner 的上下文。

**核心变化是案例库及其检索规则，而不是 GRPO 的 advantage。** 它不要求对同一题采样 N 条 rollout、在组内做奖励归一化，也不是将终局奖励分成 agent/turn/token 级优势的方法。

标题中的 fine-tuning agents 指系统行为会适应；without fine-tuning LLMs 指底层规划和执行模型保持固定。参数化版本依然训练检索网络，不能概括成“整个系统没有梯度更新”。

## 2. Motivation：相似案例不等于有用案例

固定提示模板不会自动记住部署后积累的经验；微调大模型则需要权重访问和训练资源。外部案例库提供了另一条适应路径：把经验变成未来输入的一部分。

非参数版本按语义相似度找案例，已经能复用过去方案。但相似任务的方案未必适用于当前任务，因此参数化版本进一步学习：**在当前任务下，某个案例作为参考时，完成任务的概率有多高？**

成功案例提供可借鉴的方案，失败案例提供需要避免的模式。保存失败案例不等于鼓励照抄失败动作，案例结果标签也是提供给规划器的参考信息。

## 3. 角色与三种记忆

| 组件 | 分工 | 是否更新底层 LLM |
|---|---|---|
| Planner | 读取案例，分解任务，根据执行反馈重新规划 | 否 |
| Executor | 执行子任务，调用搜索、浏览、代码及多模态工具 | 否 |
| 案例检索器 | 决定哪些过去案例值得作为参考 | 非参数版本不训练；参数化版本训练检索模型 |

论文使用 GPT-4.1 作为 planner，GAIA 使用 o3 executor，其他主要实验使用 o4-mini；视觉、视频、音频还接入各自工具模型。这是具有角色分工的系统，但不是 Dr.MAS 那种用团队奖励联合更新各角色 LLM 的 multi-agent RL。

| 记忆 | 内容 | 主要用途 |
|---|---|---|
| Case Memory | 过去任务、规划方案、成功/失败结果 | 跨任务经验复用 |
| Subtask Memory | 当前任务的子任务及执行结果 | Planner–executor 协作 |
| Tool Memory | 某个子任务内部的工具交互记录 | Executor 继续执行 |

流程：检索历史案例 → planner 生成子任务 → executor 调用工具完成子任务 → planner 查看结果并决定继续或结束 → 完成后写入案例库。

不要把 Case Memory 等同于 ECHO 的“当前 rollout 的历史 turn”：Memento 的案例库主要用于跨任务复用经验，ECHO 主要重建当前交互的有界上下文并路由信用。

## 4. 一般建模：把检索当成可以学习的动作

论文把案例表示为 $c_i=(s_i,a_i,r_i)$，案例库为 $M$。当前状态 $s$ 下先选择案例，再让固定 LLM 生成动作：

$$
c\sim\mu(\cdot\mid s,M),\qquad a\sim p_{\mathrm{LLM}}(\cdot\mid s,c).
$$

$\mu$ 是案例选择策略；$p_{\mathrm{LLM}}$ 是固定的生成模型。整体 agent 策略为：

$$
\pi(a\mid s,M)=\sum_{c\in M}\mu(c\mid s,M)p_{\mathrm{LLM}}(a\mid s,c).
$$

即使生成模型权重不变，改变案例库或选择策略，也会改变最终动作分布。这是“不微调 LLM，也能改善 agent”的数学依据。

经验写入的一般表达为：

$$
M_{t+1}=M_t\cup\{(s_t,a_t,r_t)\}.
$$

需要区分一般模型与实验实现：§3/§4 的抽象描述允许逐步追加，§4.1 和 §5.3 则说明深度研究系统在任务完成后写入，并只保存轨迹最后一步的状态、动作和奖励以减少冗余；这些状态/规划可能已包含之前累积的信息。不能据一般式断言公开实验保存了每一步独立案例。

## 5. 理论：soft Q-learning 学习选哪个案例

案例选择的价值为：

$$
Q^\pi(s,M,c)=\mathbb E\left[r+\gamma V^\pi(s',M')\mid s,M,c\right],
$$

$$
V^\pi(s,M)=\sum_{c\in M}\mu(c\mid s,M)
\left[Q^\pi(s,M,c)-\alpha\log\mu(c\mid s,M)\right].
$$

$Q$ 衡量选择该案例后，由固定 LLM 和环境产生的后续效用；$\alpha$ 控制检索策略的熵。它不是某条历史消息对当前答案的因果贡献，也不是 planner 输出 token 的 advantage。

对固定状态及案例价值，熵正则化的最优检索分布是：

$$
\mu^*(c\mid s,M)=
\frac{\exp(Q^*(s,M,c)/\alpha)}
{\sum_{c'\in M}\exp(Q^*(s,M,c')/\alpha)}.
$$

附录 A 通过概率和为 1 的约束求解得到该形式。高价值案例概率更高，熵项避免检索过早集中于单一案例。该闭式形式以已知/正确的价值为条件，不表示实际学习已经找到全局最优 agent。

一致的 soft Bellman target 写为：

$$
y=r+\gamma\alpha\log\sum_{c'\in M'}
\exp\left(\frac{Q_{\bar\theta}(s',M',c')}{\alpha}\right),
\qquad
\mathcal L_{\mathrm{TD}}=\mathbb E[(Q_\theta(s,M,c)-y)^2].
$$

这里 $\bar\theta$ 是目标网络。**记号核对：**原文 Eq. (7) 的 softmax 使用 $Q/\alpha$，而若干 TD 展开式的指数里未显式写出这个除数；上式按标准熵系数约定保持一致，不能当作原文逐字符抄写。

论文还提出 episodic-control 风格的核估计：

$$
Q_{\mathrm{EC}}(s,M,c;\theta)
=\sum_{(s_i,c,Q_i)\in\mathcal D_c}
\frac{k_\theta(s,s_i)}{\sum_{(s_j,c,Q_j)\in\mathcal D_c}k_\theta(s,s_j)}Q_i.
$$

$\mathcal D_c$ 收集以前选择同一个案例的交互，核网络决定如何按状态相似度平均已存价值。优化核可以在复用历史估计的同时学习泛化；这属于一般方法设计，不能与下面实验采用的简化二分类器混为一谈。

## 6. 实际 Memento：两种检索实现

### 6.1 非参数 CBR

$$
\operatorname{Read}_{\mathrm{NP}}(s,M)
=\operatorname{TopK}_{(s_i,a_i,r_i)\in M}
\cos(\operatorname{enc}(s),\operatorname{enc}(s_i)).
$$

用文本编码器找相似案例，将案例放入 planner prompt。学习主要表现为案例库增长，没有必要执行一般 soft Q-learning 的 TD 更新。

### 6.2 参数化 CBR

§4.2 将实际案例选择简化为单步预测问题，去掉未来值 bootstrap：

$$
Q_\theta(s,c)\approx P(r=1\mid s,c),
$$

$$
\mathcal L_{\mathrm{CE}}(\theta)=
\mathbb E[-r\log Q_\theta(s,c)-(1-r)\log(1-Q_\theta(s,c))].
$$

利用“当前任务—参考案例—结果”的数据训练成功概率预测器；它学习参考案例与任务结果的统计关系。这里的“单步”是对检索学习问题的简化，**不是说 executor 只调用一次工具**。

推理使用：

$$
\operatorname{Read}_{\mathrm P}(s,M)=\operatorname{TopK}_{c\in M}Q_\theta(s,c).
$$

理论允许 softmax 随机采样，实际采用 TopK 降低选择随机性。因此不能把实际训练描述为对 planner 做 PPO/GRPO，也不能把每次检索说成在线执行一次完整多步 Bellman backup。

从选择一个案例推广到一次放入 K 个案例，还会出现案例间互补、冲突和冗余；逐案例打分 TopK 并不证明最优联合上下文选择。

## 7. 为什么有效，原有方法忽略了什么

机制有三层：planner 将困难任务拆成可执行子任务；工具提供模型参数之外的实时信息；案例记忆让相似任务不必每次从零尝试。参数化检索再从“语义相似”推进到“预测有用”。

这些机制分别有消融，不能把全部收益归给记忆。静态文档检索主要提供事实，案例检索还提供任务策略与结果；但“静态 RAG”只是一个对照类别，不能说所有 RAG 都不能更新或不能保存经验。

这篇真正的贡献是将固定 LLM 与可适应检索策略分开，让系统通过外部经验改变行为，并在 planner–executor 场景展示效果。softmax 推导和 Q-learning 本身是已有工具，新的重点是优化对象与 agent 系统的结合。

## 8. 实验与证据边界

论文研究 GAIA、DeepResearcher、SimpleQA、HLE。主文报告 GAIA validation 87.88% **Pass@3**、test 79.40%；这是论文当时的结果，不代表当前排行榜名次。§6 的 GPT-4.1/o3 70.91% 是另一个 **Pass@1** 口径，不应混用。

DeepResearcher 的 Table 5 消融为：

| 系统 | F1 | PM |
|---|---:|---:|
| 仅离线 executor | 48.8 | 62.8 |
| 接工具的 online executor | 30.8 | 60.7 |
| Planner–executor，无 CBR | 59.9 | 72.2 |
| 完整 Memento | 66.6 | 80.4 |

PM 是使用 evaluator 的部分语义匹配指标，不是严格 exact-match 成功率。这里完整系统相对无 CBR 增加 6.7 F1、8.2 PM；工具本身并不保证提升，规划和工具编排也是关键部分。

Table 3 中检索 K 从 0 增加到 4 时平均 F1 从 59.9 提升到 64.5，而 K=32 为 63.9。这支持“有用记忆有帮助、更多记忆不一定更好”，但不证明 K=4 对任何模型任务都最优。

Table 4 的五轮结果显示参数化 CBR 最后为 85.44，非参数为 84.85，无 CBR 为 84.47。三个版本都随迭代提升，因此不能把所有跨轮增长都归因于案例库。该表与 Table 5 不是同一个直接可替换的汇总口径，不混算收益。

§6 比较 planner 类型，发现在相应实验配置中简洁的规划器优于更长的思考模式。作者通过轨迹观察将部分差异归因为跳过规划、冗长计划和角色混淆。这是有限配置下的系统结论，不是“推理模型不适合规划”的一般定律。

## 9. 代码核对：已经公开什么

代码实体位于 `/usr/data/Memento`，本目录 [Memento](./Memento) 为软链接。

| 文件 | 核对内容 |
|---|---|
| [np_memory.py](./Memento/memory/np_memory.py) | 非参数案例检索工具 |
| [train_memory_retriever.py](./Memento/memory/train_memory_retriever.py) | 任务与案例成对编码、分类器训练 |
| [parametric_memory.py](./Memento/memory/parametric_memory.py) | 加载检索器 checkpoint，按预测分数排序选择 |
| [no_parametric_cbr.py](./Memento/client/no_parametric_cbr.py) | 非参数 CBR 客户端 |
| [parametric_memory_cbr.py](./Memento/client/parametric_memory_cbr.py) | 参数化 CBR 客户端 |

`MemoryRetrieverClassifier` 将两个文本的 SimCSE/RoBERTa 表示拼接，经 Linear–ReLU–Dropout–Linear 输出二分类 logits。训练使用 AdamW 和交叉熵；概率来自类别 1 的 softmax 分数。

**代码与简略方法描述的重要区别：**当前训练脚本没有冻结 `sentence_bert`，优化器接收模型全部参数，所以会更新文本编码器与分类头。不能说参数化方案仅更新两层 MLP，更不能说所有神经网络都不训练。这个编码器不同于负责规划、执行的底层生成 LLM，后者仍然通过调用接口使用。

README 部分目录示例与实际文件名不完全一致，例如实际存在 `client/parametric_memory_cbr.py`；应以当前文件为准。已确认有检索训练和推理实现，但本次未安装依赖、未调用付费模型、未训练或复现实验，因此不声称验证了完整在线持续学习流程的运行正确性。

## 10. 与现有主线的关系

| 方法 | 优化对象 | context 的作用 | 是否在做 LLM turn-level advantage |
|---|---|---|---|
| GiGPO/HGPO/GraphGPO | 生成动作的 LLM policy | 动作的实际条件输入 | 是，步骤优势广播给该步 token |
| Dr.MAS | 各角色生成 policy | 角色之间传递任务历史 | 主要为角色级归一化 |
| SUPO | 工具使用和摘要生成 policy | 滚动摘要承接长历史 | 同一 rollout 通常共享优势 |
| ECHO v5 | 行动、记忆和选择的生成 policy | 选择当前 rollout 的来源记忆 | 来源 token 掩码分配正向信用 |
| Memento | 案例库、可选的案例检索模型 | 给固定 planner 提供跨任务参考案例 | 否 |

对我们的 multi-agent context selection 想法，Memento 提供一个直接启发：**上下文选择器可以独立于生成 agent 学习。** 因此可以先保持 solver/verifier 固定或沿用 Dr.MAS 更新，再研究角色条件下的信息选择。

但跨任务案例选择不等于当前协作轨迹里的贡献归因。高预测成功率不等于该案例的因果增益；如果要论证某条 verifier 意见推动了 solver 修正，还需要比较采用/不采用后的行为或可靠的替代证据。本文没有解决这个 turn-level 因果信用问题。

## 11. 局限与完整故事线

依赖高质量结果标签；错误、过时案例可能污染记忆；语义相似与策略适用性未必一致；二分类器学习的是观察关联而非反事实贡献；多案例相互作用没有被单独建模；强闭源模型、工具和评估器使整体性能不能仅归因于检索算法。持续学习还应明确题目隔离、记忆写入时机以及是否允许使用评估标签，避免把带反馈适应与零反馈泛化混淆。

完整逻辑是：固定 LLM 的行为仍可通过输入改变 → 将成功/失败经验存为案例 → 检索案例指导规划和执行 → 通过更新案例库或训练检索器改善未来上下文。**它是一篇记忆与检索驱动的 agent 适应方法，不是新的 group-based advantage 或 token credit 算法。**
