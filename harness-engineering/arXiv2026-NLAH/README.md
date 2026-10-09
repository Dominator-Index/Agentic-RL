# NLAH：Natural-Language Agent Harnesses

- 归档日期：2026-10-08。
- 会议／状态：arXiv2026；未确认正式会议录用。
- 来源：[论文页面](https://arxiv.org/abs/2603.25723v2) · [本地 PDF](./NLAH.pdf)；版本：`2603.25723v2`。
- 类别：自然语言形式的可执行 harness。
- 更新对象：**主要是表示与运行时设计，不训练任务模型**。
- 代码／资源：[官方仓库](https://github.com/curated-skills/LinguaClaw) · [本地入口](./NLAH) → `/usr/data/harness-engineering/NLAH`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`01232139ee8b2642dad240ce4e9488eab25b953c`。

## 1. 主要 idea

将原本藏在 controller 源码中的运行规则写成可编辑文档，由 Intelligent Harness Runtime 将规则落实成子 agent 调用、交接、状态更新与验证关卡。

## 2. 关键步骤

1. 把角色、阶段、产物契约和停止条件外显为 NLAH 文档。
2. 由共享 IHR 按规则执行，并保留中间状态与产物。
3. 比较原生代码 harness、普通 prompt 载入和 IHR 执行三种实现，并做模块消融。

## 3. 与本专题的关系

区别在于自然语言文档具有运行时执行语义，不能等同于给 agent 多塞一段 system prompt。论文附录 B 将 LinguaClaw 标为复现源码。

任务／评测：SWE-bench Verified、Terminal-Bench 2.0、OSWorld。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
