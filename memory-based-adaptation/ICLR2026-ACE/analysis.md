# ACE：用增量策略手册实现上下文适应

> **Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models**，ICLR 2026。归档 arXiv:2510.04618v3（2026-03-29），核对日期 2026-10-07。依据 [本地 PDF](./ACE.pdf) 正文与附录，以及 [官方仓库](https://github.com/ace-agent/ace)；固定代码版本见 [code_manifest.json](./code_manifest.json)。

## 1. 这篇文章到底解决什么问题

ACE 保持 LLM 权重固定，通过维护输入中的策略手册（playbook）来改善后续行为。它希望保存可复用的工具用法、领域规则、失败模式和代码片段，而不是每次任务都从零开始。

它关注两种失效：**brevity bias**，为了简短而丢掉关键领域细节；**context collapse**，不断要求 LLM 全量重写越来越长的经验文档，最后把丰富内容压成少数泛泛而谈的句子。

主要贡献是改变记忆维护方式：**结构化条目＋局部增量更新＋反思和整理分工＋去重维护。** 它不是新的 GRPO loss，也没有训练一个 turn/token advantage 模型。

## 2. Motivation：为什么全量总结可能越总结越差

假设策略手册已经记录多个 API 的参数约束、分页细节和异常处理。如果每次都要求 LLM“根据新经验重写整份总结”，它可能把具体操作压成“注意验证输入、合理使用工具”，看起来更简洁，实际可执行信息却减少了。

论文 §2.2 的案例中，上下文从 18,282 tokens 突然缩为 122 tokens，准确率从 66.7 降到 57.1，低于无适应的 63.7。这是一个支持动机的案例，不是任何 summarization 都必然崩溃的证明。

ACE 让新反馈只形成局部补丁，由程序合并到已有条目中，避免每次都重新生成已验证的历史知识。

## 3. 三个角色分别做什么

| 角色 | 输入 | 输出 |
|---|---|---|
| Generator | 当前任务、playbook、必要环境上下文 | 答案或工具交互轨迹，以及使用的条目 ID |
| Reflector | 生成轨迹、执行反馈、用过的条目；有些设置包含答案标签 | 成败分析、可复用经验、helpful/harmful 等条目标记 |
| Curator | 反思结果、已有 playbook | 结构化增量条目或维护建议 |

主实验三个角色都使用 DeepSeek-V3.1 non-thinking，避免“强老师教弱学生”混淆。三个角色可以是同一模型的不同提示调用，不等于训练了三个不同 LLM。

流程是：Generator 尝试任务 → Reflector 分析 → Curator 形成小补丁 → 程序合并并维护手册 → 后续任务使用新手册。角色分工发生在上下文优化流程中，和 Dr.MAS 对 solver/verifier 做联合 RL 更新不同。

## 4. Playbook 存的不是原始 trajectory

一条 bullet 通常包含唯一 ID、正文、helpful/harmful 计数。例如下面只是说明格式的自造例子：

```text
[api-00012] helpful=3 harmful=0 :: 列表接口返回分页结果；继续请求下一页，直到没有 continuation token。
```

这里记录的是从经验提炼的规则。计数来自模型反馈及程序累积，不是已经验证的因果贡献，也不是 PPO advantage。

新知识可以添加为新条目；旧条目可以增加使用反馈；语义相近条目再去重或整合。论文允许周期性维护，也允许上下文到达预算阈值时再维护。

论文指出条目化方便精细检索，但不能因此推断每个实验都使用了独立学习的 TopK selector。公开核心路径会将 playbook 传给 Generator，具体上下文裁剪与选择要看任务实现。

## 5. 用数学形式理解方法

**以下是本文对算法的统一表达，不是论文提出的额外定理或梯度目标。** 令 $P_n$ 是处理第 $n$ 个样本前的策略手册，$x_n$ 是任务，$\theta$ 是固定 LLM 参数：

$$
\tau_n\sim\pi_\theta(\cdot\mid x_n,P_n),\qquad \theta_{n+1}=\theta_n.
$$

$\tau_n$ 是生成结果或完整任务轨迹；上下文变了，行为分布可以改变，但没有对 $\theta$ 做梯度更新。

设 $f_n$ 为可获得的反馈，$B_n$ 为被使用的条目集合：

$$
z_n=\operatorname{Reflect}(x_n,\tau_n,f_n,B_n),
\qquad
\Delta_n=\operatorname{Curate}(P_n,z_n).
$$

$z_n$ 是自然语言经验与条目标记，$\Delta_n$ 是少量结构化更新。它不代表 reward 数值差分，更不是 $A_t$。

更新上下文：

$$
P_{n+1}=\operatorname{Refine}\big(\operatorname{Apply}(P_n,\Delta_n)\big).
$$

Apply 根据 ID 和章节执行确定性合并；Refine 表示必要时进行去重与整理。确定性只描述合并逻辑，不表示 LLM 生成的补丁永远正确。

若第 $b$ 条经验被标记 helpful 或 harmful，其计数变化可写为：

$$
h_{b,n+1}=h_{b,n}+\mathbf1\{\operatorname{tag}_{b,n}=\mathrm{helpful}\},
\qquad
u_{b,n+1}=u_{b,n}+\mathbf1\{\operatorname{tag}_{b,n}=\mathrm{harmful}\}.
$$

这些统计帮助维护知识条目，未构成 Dr.MAS 那样的 reward normalization，更不意味着对该条目对应的生成 token 反向传播。

概念上，它在离散文本空间中寻找更有用的 $P$，而不是优化 LLM 权重。论文没有证明该更新单调提升任务奖励，也没有给出全局最优上下文求解器。

## 6. Offline 与 online 的 setup

**Offline adaptation：**用训练任务产生反馈、改进 playbook，再冻结得到的上下文到测试集上评估。多 epoch 表示多次处理训练问题以完善文本记忆，不是对 LLM 做多轮 SGD。

**Online adaptation：**顺序处理测试流，每题先用当前 playbook 预测并计分，再根据该题反馈更新 playbook供后续题目使用。不能把看过答案后修正的本题输出当成首次预测成绩。

论文同时讨论带 ground-truth 标签和不向 Reflector 提供标签的设置。AppWorld 的执行成功/错误提供额外反馈；金融任务缺乏可靠反馈时可能退化。因此，**不更新权重、不提供标准答案、没有任何外部反馈，是三个不同概念。**

这也不是固定“一个问题 N 条独立轨迹、组内归一化”的 setup。可能有多次反思和重试，但重试是上下文适应流程，不是由 GRPO group 定义。

## 7. 为什么有效，真正的新意是什么

局部更新降低对完整长历史进行再生成的依赖，使正确细节不必每次重新通过语言模型生成；Reflector 专注于提炼经验，Curator 专注于组织条目；去重减少无谓堆积。结合起来，既保存具体知识，又控制重复与维护成本。

与简短 prompt 优化相比，它更适合复杂 API、领域规则与可复用代码；与反复全量总结相比，它降低知识被误删的风险。但任务如果只需要一条高层策略，长 playbook 未必有价值，论文也明确讨论这个边界。

“多角色反思”与“保存经验”不是全新概念。ACE 的贡献重点是将这些组件组织为可持续增量维护的上下文流程，并用消融说明增量更新的重要性。

## 8. 实验与消融

主实验使用 AppWorld，以及金融 FiNER、Formula；附录增加 DDXPlus、BIRD-SQL 和其他模型。AppWorld 指标包括 test-normal/test-challenge 的 TGC 与 SGC，表中 Average 是四项平均，不能当作单一 task success rate。

| AppWorld 方法 | 四项平均 |
|---|---:|
| 基础 ReAct | 42.4 |
| Offline GEPA，有标签 | 46.4 |
| Offline ACE，有标签 | 59.4 |
| Offline ACE，无标准答案标签 | 57.2 |
| Online Dynamic Cheatsheet | 51.9 |
| Online ACE，含 offline warmup | 59.5 |

来源 Table 1 和 Table 3。Table 3 的未 warmup online ACE 为 56.1，因此不要把 59.5 描述成完全从空手册在线启动的结果。59.4 相对 42.4 是 +17.0 个百分点，论文摘要 +10.6 的口径是跨所选 baseline 的平均，不应互换。

最直接支撑机制的证据是附录 Table 18：仅在 test-normal 的 TGC/SGC 两项平均中，无增量更新为 56.9，有增量更新为 70.3；这个均值范围不同于主表的四项平均。它比“完整方法打败基础模型”更能说明局部更新的重要性。

反思和多 epoch 消融也支持各模块：Table 3 离线平均从同时去掉 Reflector/多 epoch 的 55.1，到保留反思但无多 epoch 的 56.8，再到完整 59.4。不过这些是顺序配置消融，不能解释成完全独立可相加的因果收益。

金融在线无标签时，FiNER 从基础 70.7 降到 ACE 67.3，而 Formula 提升。这说明反馈不可靠时上下文可能积累错误，不是“自反思越多就一定越强”。附录也发现 10 轮反思不必优于 5 轮。

成本方面，论文分别报告 adaptation latency、rollout 数与 token 成本。较低维护成本不代表最终推理上下文一定更短；附录 A.3 指出评估阶段可能消耗更多原始输入 tokens。不要将上下文适应成本改善直接表述为相同质量下所有端到端费用都下降。

## 9. 代码核对与复现注意点

实体仓库 `/usr/data/ACE`，本目录 [ACE](./ACE) 为软链接。对应论文首页与作者项目页的官方仓库，不是同名社区实现。

| 文件 | 作用 |
|---|---|
| [ace/ace.py](./ACE/ace/ace.py) | 调度生成、反思、整理，以及离线/在线模式 |
| [generator.py](./ACE/ace/core/generator.py) | 基于 playbook 生成回答 |
| [reflector.py](./ACE/ace/core/reflector.py) | 分析轨迹与标记条目 |
| [curator.py](./ACE/ace/core/curator.py) | 生成上下文增量 |
| [playbook_utils.py](./ACE/playbook_utils.py) | ID、条目解析、计数更新、增量应用 |
| [bulletpoint_analyzer.py](./ACE/ace/core/bulletpoint_analyzer.py) | embedding 相似度与去重/整合 |

两点需要区分设计与当前代码：

1. `apply_curator_operations` 当前有效操作主要是 `ADD`；`UPDATE/MERGE/DELETE/CREATE_META` 在这里标注为未来操作。计数更新由单独函数完成，语义去重另有 analyzer 路径。不能只看论文的维护描述，就断言每个编辑操作都已在该函数实现；也不能因此说整个仓库完全没有合并能力。
2. `ace/ace.py::_train_single_sample` 中，即使 `no_ground_truth=True` 不向 Reflector 显示标准答案，仍会通过 `answer_is_correct(final_answer,target)` 分支，并传入“匹配/不匹配答案”的反馈。**隐藏答案文本不等于移除答案派生监督。** 这只是已核对的通用适应路径，不能据此断言独立 `ace-appworld` 执行反馈路径也使用相同标签逻辑。

代码中的 `train` 指上下文适应循环，不应默认理解成反向传播训练模型。此次仅做源码检查，没有安装依赖、启动 AppWorld、调用 API 或复现实验成绩。

## 10. ACE、Memento、ECHO 与 Dr.MAS 的区别

| 方法 | 积累或选择什么 | 更新对象 | 信用信号 |
|---|---|---|---|
| Memento | 过去任务与规划结果的案例 | 案例库；参数化版本还训练检索器 | 案例有用性/成功预测 |
| ACE | 从轨迹提炼的策略、规则与失败经验 | 结构化文本 playbook | 自然语言反思与 helpful/harmful 标记 |
| ECHO v5 | 当前长 rollout 中带来源的历史轮次记忆 | LLM policy | 正 advantage 与来源 token 掩码 |
| Dr.MAS | 多角色输出样本 | 各角色 LLM policy | 按角色归一化的团队回报 |

Memento 更接近“这个新任务应参考哪个旧案例”；ACE 更接近“把旧经验变成可维护的规则手册”；ECHO 更接近“当前任务的哪些历史输出应保留并获得训练信用”。

对于我们的 multi-agent turn-level credit 方向，ACE 的 ID 和条目标记可以提供启发，但不能把“模型说这条经验 helpful”直接当作真实贡献。ACE 也没有用这些计数计算各角色 turn 的 PPO advantage。

## 11. 引用 Memento 的核对与归档依据

ACE v3 参考文献确实列出 Zhou 等、arXiv:2508.16153，但标题写成 **AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs**；该 arXiv 编号对应我们归档的 **Memento**。这是引用标题与来源标题的差异，不能误认为已确认是另一篇独立论文。

ACE 的正文明确强调受 Dynamic Cheatsheet 启发；引用 Memento 不等于以 Memento 为主要实验 baseline。主实验 baseline 是 ReAct/Base LLM、ICL、MIPROv2、GEPA、Dynamic Cheatsheet 等。

会议依据：[ICLR 2026 官方列表](https://iclr.cc/Downloads/2026)、[作者项目页](https://ace-agent.github.io/) 与归档 PDF 首页一致。故使用 `ICLR2026-ACE`。

## 12. 局限与总结

生成的规则可能错误或过拟合；去重未必消除矛盾；更新顺序、任务分布、反馈质量和长上下文成本会影响结果。条目保留减少信息丢失，却不保证所有新增经验都值得保留。论文没有给出上下文更新的收敛性或最优性证明。

ACE 的完整故事是：固定 LLM 仍可通过上下文改变行为 → 经验应保留具体可执行细节 → 全量重写容易丢失知识 → 用分工反思和增量条目维护策略手册 → 在可靠反馈与合适任务上持续改善。这使它适合放入“记忆与上下文驱动的适应”分类，而不是 turn-level policy-gradient 方法集合。
