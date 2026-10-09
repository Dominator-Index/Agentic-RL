# AutoSaddler：AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces

- 归档日期：2026-10-08。
- 会议／状态：NeurIPS2026；[录用依据](https://neurips.cc/virtual/2026/poster/151256)。
- 来源：[论文页面](https://arxiv.org/abs/2608.23041) · [本地 PDF](./AutoSaddler.pdf)；版本：`2608.23041v1`。
- 类别：从执行轨迹自动修改完整 harness。
- 更新对象：**优化 harness 代码与配置；文中 offline learning 不应直接解读为微调任务 LLM**。
- 代码／资源：[官方仓库](https://github.com/microsoft/AutoSaddler) · [本地入口](./AutoSaddler) → `/usr/data/harness-engineering/AutoSaddler`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`e841543ce6cf6bdf690e65134b12a829ec9f66d1`。

## 1. 主要 idea

从小批量失败轨迹诊断根因，再对提示、工具和 middleware 等组件施加结构化补丁，并用额外验证选择能够泛化的修改。

## 2. 关键步骤

1. 收集一批任务的执行结果，结合源码深入定位失败。
2. 按组件和补丁类型产生针对性修改，而非任意重写系统。
3. 在验证任务上比较候选，保留稳定、可追踪的更新与演化记录。

## 3. 与本专题的关系

与你关注的“失败 case 不应白白丢掉”相关，但产物是跨任务复用的 harness 更新，不是失败轨迹的 token 梯度 mask。会议摘要与项目页的实验覆盖不完全一致，导读不混报单一结果表。

任务／评测：GAIA2、Terminal-Bench 2.0；项目页另报告 SWE-Bench Pro，版本口径需区分。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
