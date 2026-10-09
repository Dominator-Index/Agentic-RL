# AHE：Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses

- 归档日期：2026-10-08。
- 会议／状态：arXiv2026；未确认正式会议录用。
- 来源：[论文页面](https://arxiv.org/abs/2604.25850) · [本地 PDF](./AHE.pdf)；版本：`2604.25850v4`。
- 类别：可观测性驱动的 harness 组件演化。
- 更新对象：**主要修改组件文件，不更新任务 LLM 权重**。
- 代码／资源：[官方仓库](https://github.com/china-qijizhifeng/agentic-harness-engineering) · [本地入口](./AHE) → `/usr/data/harness-engineering/AHE`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`8b2a55d97590363fe50c3cc6b5e833b020a4bb4c`。

## 1. 主要 idea

让演化 agent 看清楚能改哪些组件、任务为何失败、某次修改预期改善什么，才能从轨迹中稳定迭代运行系统。

## 2. 关键步骤

1. 将工具、middleware、记忆、提示等可编辑部分解耦为明确组件。
2. 把大量执行日志组织成可逐层查看的诊断证据。
3. 为修改记录预测和 change manifest，下一轮按任务结果验证，并对无效文件修改回退。

## 3. 与本专题的关系

强调修改的对象和证据，而非仅扩大 search budget。文中的多类组件与可回溯修改机制对设计研究 baseline 有用。

任务／评测：Terminal-Bench 2.0；迁移到 SWE-bench Verified 与不同基础模型。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
