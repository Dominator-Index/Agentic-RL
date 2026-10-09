# JIT-Agent：JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution

- 归档日期：2026-10-08。
- 会议／状态：arXiv2026；未确认正式会议录用。
- 来源：[论文页面](https://arxiv.org/abs/2608.25593v2) · [本地 PDF](./JIT-Agent.pdf)；版本：`2608.25593v2`。
- 类别：训练按任务生成 harness 的模型。
- 更新对象：**训练 harness 生成／修复模型（含 SFT、DPO 等阶段）；被辅助的执行 LLM 可以保持固定**。
- 代码／资源：[官方仓库](https://github.com/bingreeky/JIT) · [本地入口](./JIT-Agent) → `/usr/data/harness-engineering/JIT-Agent`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`ababa06c2f54d799fd9fbc356e5368f61a452260`。

## 1. 主要 idea

把 harness 设计变成可学习的能力：收到任务后，生成 memory、planning、action、capability 四类模块，失败后修复，并从历史配置中学习。

## 2. 关键步骤

1. 按固定模块协议表示可组合的 harness。
2. 为新任务生成配置／代码并检查运行结果，必要时修复。
3. 从历史 harness 与表现记录中蒸馏经验，提升生成器的泛化能力。

## 3. 与本专题的关系

这篇最需要问清“哪个模型在训练”：生成 harness 的 helper 被训练，不代表所有执行 agent 的权重都跟着更新。

任务／评测：DeepSearchQA、OdysseyBench 等跨任务／跨基础模型评测。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
