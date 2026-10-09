# MoMHa：MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens

- 归档日期：2026-10-08。
- 会议／状态：NeurIPS2026；[录用依据](https://neurips.cc/virtual/2026/poster/153336)。
- 来源：[论文页面](https://arxiv.org/abs/2609.30967) · [本地 PDF](./MoMHa.pdf)；版本：`2609.30967v1`。
- 类别：准确率、行为安全和 token 成本的多目标优化。
- 更新对象：**由 agent 搜索 Python harness；主要不更新底层 LLM 权重**。
- 代码／资源：截至 2026-10-08，未从本次核对的论文／作者来源确认可归属本论文的公开实现。未用同名项目或基线仓库代替。

## 1. 主要 idea

将提示构造、调用路由和输出解析的代码视作优化对象，同时考虑准确率、行为安全和 token 消耗，避免只优化单一分数。

## 2. 关键步骤

1. 定义每个任务域的联合评测目标，并保存历史代码、轨迹及分数。
2. 提议 agent 在这些历史记录上生成新的 harness 候选。
3. 以联合目标评价和选择候选，对比只优化准确率或分阶段优化的方案。

## 3. 与本专题的关系

适合讨论 harness 的预算约束。标题对应的方法简称是 MoMHa；其摘要也使用 Meta-Harness 这一名称，但不是 Lee 等人的同一篇 Meta-Harness 论文，需按作者与论文编号区分。

任务／评测：17 个任务域，包括 HumanEval、MBPP、Spider、FEVER、MMLU-Pro、LawBench、NuminaMath 及安全任务。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
