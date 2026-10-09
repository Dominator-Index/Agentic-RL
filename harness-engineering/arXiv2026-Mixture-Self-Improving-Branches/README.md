# Mixture-Self-Improving-Branches：Mixture of Self-Improving Branches For Agent Harness Optimization

- 归档日期：2026-10-08。
- 会议／状态：arXiv2026；未确认正式会议录用。
- 来源：[论文页面](https://arxiv.org/abs/2609.37834v1) · [本地 PDF](./Mixture-Self-Improving-Branches.pdf)；版本：`2609.37834v1`。
- 类别：多分支自改进与推理时路由。
- 更新对象：**主要演化 harness、开发子集及提议策略；不据此声称训练基础 LLM**。
- 代码／资源：截至 2026-10-08，未从本次核对的论文／作者来源确认可归属本论文的公开实现。未用同名项目或基线仓库代替。

## 1. 主要 idea

让不同搜索分支逐渐形成互补能力：开发样本和改进指导也随历史变化，部署时为新问题选择一个分支的 harness。

## 2. 关键步骤

1. 从种子 harness 开出多个分支。
2. 各分支根据领先候选对任务的表现调整开发子集，并更新提议策略。
3. 只使用开发数据选择各分支代表，再由 router 为测试输入选一个 harness。

## 3. 与本专题的关系

不是同一道题让所有分支都答完后用真实答案选优。核心比较应计入搜索和路由成本，并避免测试结果参与分支挑选。

任务／评测：Olympiad-level 数学、Terminal-Bench 2.0、SWE-bench Lite。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
