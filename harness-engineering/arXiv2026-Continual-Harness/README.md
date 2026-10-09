# Continual-Harness：Continual Harness: Online Adaptation for Self-Improving Foundation Agents

- 归档日期：2026-10-08。
- 会议／状态：arXiv2026；未确认正式会议录用。
- 来源：[论文页面](https://arxiv.org/abs/2605.09998v1) · [本地 PDF](./Continual-Harness.pdf)；版本：`2605.09998v1`。
- 类别：不重置环境的在线 harness 适应。
- 更新对象：**基础版本改 harness；扩展实验还包含模型参数 co-learning**。
- 代码／资源：[官方仓库](https://github.com/sethkarten/continual-harness) · [本地入口](./Continual-Harness) → `/usr/data/harness-engineering/Continual-Harness`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`bbab97ad73e460b7cd7c08527d10ced30cc03fbe`。

## 1. 主要 idea

在一条持续进行的长交互中交替行动和修改系统提示、子 agent、技能与记忆，适应当前遇到的障碍；环境不需要每次更新后重置。

## 2. 关键步骤

1. 从最小环境接口开始行动并记录轨迹。
2. Refiner 从经历中修改 prompt、sub-agents、skills、memory。
3. 在同一条持续交互中继续行动；扩展设置再用 frontier teacher 重标轨迹来训练开放模型。

## 3. 与本专题的关系

setup 与每个问题独立采 N 条完整 rollout 不同。必须区分早期有人参与的 GPP 工程、自动 Continual Harness 和额外的参数 co-learning 三部分。

任务／评测：Pokémon Red、Emerald；GPP 的其他游戏实验在论文中另述。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
