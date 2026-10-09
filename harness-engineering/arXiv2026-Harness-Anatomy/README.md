# Harness-Anatomy：Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents -- A Source-Code Study of Eleven Systems

- 归档日期：2026-10-08。
- 会议／状态：arXiv2026；未确认正式会议录用。
- 来源：[论文页面](https://arxiv.org/abs/2609.00006v1) · [本地 PDF](./Harness-Anatomy.pdf)；版本：`2609.00006v1`。
- 类别：生产 coding harness 的源码研究。
- 更新对象：**源码审计，无单一训练算法**。
- 代码／资源：截至 2026-10-08，未从本次核对的论文／作者来源确认可归属本论文的公开实现。未用同名项目或基线仓库代替。

## 1. 主要 idea

沿执行循环、工具、上下文、控制与扩展等子系统检查多个实际 coding-agent 实现，提炼共性与演化规律。

## 2. 关键步骤

1. 固定被比较系统的源码版本。
2. 按统一子系统和设计模式分析实现。
3. 比较时间快照，归纳工程建议与最小 harness。

## 3. 与本专题的关系

适合明确实验中到底改变了系统哪一层。观察只覆盖论文选定版本；不应把某个样本中的“未使用某技术”当作所有 agent 的普遍事实。arXiv 编号月份与页面标注日期有差异，保留来源字段，不仅按编号推断时间。

任务／评测：11 个 coding 系统及额外 meta-harness 对照。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
