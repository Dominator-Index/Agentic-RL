# DGM：Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents

- 归档日期：2026-10-08。
- 会议／状态：ICLR2026；[录用依据](https://iclr.cc/virtual/2026/poster/10007327)。
- 来源：[论文页面](https://arxiv.org/abs/2505.22954v3) · [本地 PDF](./DGM.pdf)；版本：`2505.22954v3`。
- 类别：Agent 程序与工具的开放式进化。
- 更新对象：**主要修改 agent 自身程序；底层 foundation model 保持固定**。
- 代码／资源：[官方仓库](https://github.com/jennyzzt/dgm) · [本地入口](./DGM) → `/usr/data/harness-engineering/DGM`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2`。

## 1. 主要 idea

让 coding agent 修改自己的实现，再用编码任务实测修改的效果；维护一个不断增长的 agent 档案，让过去的分支成为未来改进的起点。

## 2. 关键步骤

1. 从 agent 档案中选择父代。
2. 分析表现并修改父代源码，例如工具、上下文管理或审查流程。
3. 在编码基准上检验子代并加入档案，继续探索多个演化分支。

## 3. 与本专题的关系

非常直接的 harness 自我修改前驱。这里的“自我改进”主要发生在可执行 agent 程序，不是对 LLM 参数做梯度下降；经验验证也不等于形式化证明每次修改必然有益。

任务／评测：SWE-bench、Polyglot。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
