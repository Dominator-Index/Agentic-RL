# 04｜有详细过程时，为什么不直接 SFT？

日期：2026-10-07。本文区分原论文结果、数学解释与实验建议；没有进行新的模型训练。

**用户的判断抓住了核心：如果训练集已经有高质量、适合学生学习的完整过程，直接 SFT 是强基线。把这些过程丢掉，只保留终局 0/1 奖励，确实会损失监督信息。不能先丢掉信息，再把恢复部分过程信号包装成当然优于 SFT 的贡献。**

但“有详细过程”也不能推出 SFT 必然胜过 RLVR：要看过程质量、学生能力、示范覆盖、分布变化、最终评价目标和训练预算。

## 1. 同一道题：两者到底使用了哪些信息？

设一条标注数据为

$$
(x,y^*,z^*),\qquad y^*=(y_1^*,\ldots,y_L^*),
$$

其中 $x$ 是问题，$y^*$ 是完整示范输出的 token 序列，含推理过程和最终作答，$z^*$ 是可单独提取的标准答案。

SFT 最小化

$$
L_{\mathrm{SFT}}
=-\mathbb E_{(x,y^*)}\sum_{j=1}^L
\log\pi_\theta(y_j^*\mid x,y_{<j}^*).
$$

每个示范 token 都有明确目标。即使学生目前独立生成 $N$ 次都答错，也能沿示范前缀获得非零训练信号。这里的“每 token 监督”是下一 token 标签，不等于每一步都有因果贡献或正确性评分。

纯 outcome RLVR 使用

$$
J_{\mathrm{RLVR}}
=\mathbb E_x\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[\mathbf 1\{\mathrm{Verify}(\mathrm{answer}(y),z^*)=1\}\right].
$$

它优化模型自己生成的轨迹的成功概率，通常不读取标准过程 $y^*$。GRPO 是实现这类优化的一种算法。对同题 $N$ 条输出，其常用优势为

$$
A_i=\frac{R_i-\overline R}{\operatorname{std}(R_1,\ldots,R_N)+\epsilon}.
$$

若这次所有 $R_i=0$，该组 outcome advantage 为零；KL、熵等其他项仍可能更新参数。这个现象属于上述组相对 outcome 信号，不能概括为所有 RL 算法在失败时都必然没有梯度。

所以在“学生暂时几乎不会、教师已经给出可学习步骤”的条件下，SFT 有直接的信号优势。是否转换成更高测试准确率，还需实测。

## 2. 为什么 SFT 并不自动解决所有问题？

**标准过程中的前缀，与学生自己生成的前缀不同。** SFT 的目标条件为 $(x,y_{<j}^*)$。部署时条件是 $(x,\widehat y_{<j})$。一旦偏离示范，数据未必告诉模型此时怎样纠错。

例如成功轨迹是“搜索→读取正确文档→回答”。学生可能拿到无关文档、工具报错或证据冲突，下一步应该改写查询、换工具还是停止？一条成功示范未必包含这些分支。RL 可以在实际访问到的上下文中采样、接受最终验证，从而学习有用行为；仍需足够探索，不能保证自动学会恢复。

**一个问题也可能有多条有效路径。** SFT 直接拟合给出的示范，RLVR 则按验证结果认可不同输出。SFT 也可以通过多样示范、纠错轨迹和更充分训练获得泛化，不能把它简单等同于死记硬背。

**监督密集与监督目标一致是两件事。** 完整过程可能冗长、有误、包含学生难以理解的跳步；逐 token 标签很多，并不意味着每个 token 都对最终成功同样重要。反过来，稀疏终局奖励若验证可靠，则与任务目标较直接，但学习效率和归因较弱。

这些是两种训练方式的结构性差异，不是一个普遍的优劣定理。

## 3. 已有论文怎么说？

### 3.1 DeepSeek-R1：学习完整教师过程，可以优于同规模模型直接 RL

[DeepSeek-R1，§4.1、Table 6](https://arxiv.org/html/2501.12948v1#S4.SS1) 比较 Qwen-32B 基础模型直接进行超过 10K steps 的 RL，与使用 DeepSeek-R1 生成数据的蒸馏模型。文中蒸馏阶段采用 SFT。

| 路线 | AIME 2024 pass@1 | MATH-500 pass@1 |
| --- | ---: | ---: |
| DeepSeek-R1-Zero-Qwen-32B，直接 RL | 47.0 | 91.6 |
| DeepSeek-R1-Distill-Qwen-32B，教师数据 SFT | 72.6 | 94.3 |

这个结果直接反驳“RL 一定比过程 SFT 强”。但它包含更强教师提供的额外监督，不能当作相同信息、相同总计算预算下的纯优化算法对照。

### 3.2 SFT Memorizes, RL Generalizes：特定分布变化下 RL 更有优势

[论文](https://arxiv.org/abs/2501.17161) 在 GeneralPoints 算术游戏和 V-IRL 导航等任务中比较规则/视觉变化后的泛化。其结果支持 outcome RL 在所测设置下比 SFT 更好地迁移，也讨论了 SFT 对后续 RL 的帮助。

应把结论限定在对应模型、数据和训练设置。论文标题不意味着所有有过程监督的任务都该放弃 SFT。

### 3.3 Rethinking Generalization in Reasoning SFT：SFT 也能学习可迁移的过程

[论文](https://arxiv.org/abs/2604.06628) 发现长 CoT 的 SFT 泛化取决于优化时长、数据质量和基础模型能力；部分跨域指标会先下降、随后恢复并超过起点。只比较一个短训练 checkpoint，可能低估 SFT。

这项工作支持认真训练和调参 SFT；它本身也不能证明 SFT 普遍优于 RLVR。

### 3.4 SRL：已经有人把完整过程转成逐步 RL 奖励

[Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](https://arxiv.org/abs/2510.25992)，[ICLR 2026 版本](https://openreview.net/pdf?id=Uro84w2xz5)，把专家过程拆成关键动作，利用学生动作与专家动作的相似性提供逐步奖励，允许每步生成自己的推理。

其数学实验中，长而困难的 s1k 示范对小模型直接 SFT 并不总有效；纯 outcome RLVR 又缺少成功样本。作者报告该逐步方案及后续 RLVR 的收益。它直接说明“有过程时如何超越直接 SFT”已有专门工作，不能只提出“把过程拿来打分”就算新方法。

### 3.5 一个相关理论结果：完整的成功路径不等于错误恢复示范

[Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack Efficiently](https://arxiv.org/abs/2606.22938) 在图寻路的形式化设置中，对比只学习最短成功路径的 SFT 与 outcome RLVR 的回溯能力；还说明可将 RL 获得的回溯轨迹蒸馏给其他模型。

它的启发是“示范包含什么行为”很关键。结论受该理论模型与示范假设限制，不是任意 Transformer 上 RLVR 优于 SFT 的证明。此处核对到原始摘要层面，不据此复述具体定理假设或证明。

## 4. 回到 GiGPO 与 Dr.MAS：它们训练时有没有用标准步骤？

| 场景 | RL 训练实际使用的信息 | 是否用完整标准步骤做 token 监督 |
| --- | --- | --- |
| Dr.MAS 数学 | 问题、用于验证的 ground-truth 答案、在线 solver/verifier 交互 | 所检查的 RL 路径没有 |
| Dr.MAS 搜索 | NQ 与 HotpotQA 混合训练，问题与标准答案，在线搜索与协作 | 所检查的 RL 路径没有 |
| GiGPO 的 ALFWorld/WebShop | 环境任务、在线动作和环境观察、环境奖励 | 所检查的 RL 路径没有 |

Dr.MAS 论文 §5.1 使用 DAPO 训练语料，§5.2 说明 NQ+HotpotQA。对应代码 [drmas_math.py](/usr/data/DrMAS/examples/data_preprocess/drmas_math.py) 加载 `aaabiao/dapo_filter`，构造 question 与 `reward_model["ground_truth"]`；[drmas_search.py](/usr/data/DrMAS/examples/data_preprocess/drmas_search.py) 读取 question 和 ground truth/golden answers。这里判断的是实际训练输入，不宣称所有上游数据集均不存在解题过程或支持事实标注。

GiGPO 代码的 [prepare.py](/usr/data/verl-agent/examples/data_preprocess/prepare.py) 明确说明占位数据只用于标记模态和数量，真正任务输入来自环境；[README 数据说明](/usr/data/verl-agent/README.md:468) 解释了这一点。其当前文本 ALFWorld 奖励来自任务成功，WebShop wrapper 也将终局成功转换为训练奖励；不是人工为每个操作标注标准答案。

**在线生成的 trajectory 很详细，不等于数据集给了标准 trajectory。GiGPO 的 step advantage 很细，也不等于它获得了人工标注的 step reward。**

## 5. 我们的新方法应该怎样比较才有说服力？

若训练时有完整过程，最少比较：SFT、纯 RLVR、SFT→RLVR、新方法。若新方法直接消费专家步骤，还应加入 SRL 等适用近邻，并确保使用相同示范资源。

同时报告两类预算：学生训练的 GPU 时间/生成与训练 token；教师示范或评价的生成成本。已有数据可视为沉没成本的部署问题，与要从零生成这些数据的方法比较，是两个不同问题，应分别报告。

若只有题目与终局验证器，缺乏完整示范，那么提升全失败组利用率、改善 credit assignment 才有直接动机。若使用强教师补充评分，也要给“同等教师调用预算生成示范并 SFT”一个公平比较机会。

因此对当前选题的要求是：**明确缺少哪一种监督，或者明确同样拥有详细监督时，我们的利用方式为什么比充分调参的 SFT/SFT→RLVR 更好。** 不能只以“稀疏奖励太粗”为理由，跳过已经可用的监督学习方案。
