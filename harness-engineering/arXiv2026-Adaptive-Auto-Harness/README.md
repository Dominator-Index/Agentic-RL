# Adaptive-Auto-Harness：Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams

- 归档日期：2026-10-08。
- 会议／状态：arXiv2026；未确认正式会议录用。
- 来源：[论文页面](https://arxiv.org/abs/2606.01770v2) · [本地 PDF](./Adaptive-Auto-Harness.pdf)；版本：`2606.01770v2`。
- 类别：异质任务流上的构建与路由。
- 更新对象：**主要演化和选择外部 harness；核心并非任务 LLM 的 GRPO 更新**。
- 代码／资源：[官方仓库](https://github.com/A-EVO-Lab/a-evolve) · [本地入口](./Adaptive-Auto-Harness) → `/usr/data/harness-engineering/Adaptive-Auto-Harness`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`17bc9ebb7d4d142af1b109b43ef160031967cc9a`。

## 1. 主要 idea

面对不断变化的任务流，一个反复修改的统一 harness 可能退化；需要既构建更好的 harness，也给每个任务选择适合的 harness。

## 2. 关键步骤

1. 有状态的多 agent evolver 汇总历史经验并演化系统。
2. 维护 harness tree，在解题时路由到合适的分支。
3. 历史信号不足时通过明确的人类引导入口补充目标或方向。

## 3. 与本专题的关系

区分 evolution loss 与 adaptation loss。与 Mixture of Self-Improving Branches 都涉及路由，但这里重点是开放、持续变化的任务流，不是固定离线开发集上的多分支搜索。

任务／评测：预测市场、安全竞赛、事件预测任务流。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
