# VeRO：VeRO: A Harness for Agents to Optimize Agents

- 归档日期：2026-10-08。
- 会议／状态：ICML2026；[录用依据](https://icml.cc/virtual/2026/poster/60518)。
- 来源：[论文页面](https://proceedings.mlr.press/v306/ursekar26a.html) · [本地 PDF](./VeRO.pdf)；版本：官方论文集／作者发布 PDF，详见来源与 manifest。
- 类别：Harness 优化的外层基础设施与评测。
- 更新对象：**框架本身不要求任务 LLM 权重更新；优化器通过代码修改目标 agent**。
- 代码／资源：[官方仓库](https://github.com/scaleapi/vero) · [本地入口](./VeRO) → `/usr/data/harness-engineering/VeRO`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`4ae9c1df1e5d303b875629fa53b686a211897c4e`。

## 1. 主要 idea

研究 coding agent 能否成为 agent 的优化器，先要有可靠的实验环境：目标版本可追踪、评测预算可控、执行轨迹可查看、测试数据与开发反馈有边界。

## 2. 关键步骤

1. 用 Versioning、Rewards、Observations 包装待优化 agent。
2. 优化器编辑代码，并通过受控接口提交评测、检查轨迹和预算。
3. 用 VeRO-Bench 比较优化器与修改效果，区分有效改进与评测流程中的漏洞。

## 3. 与本专题的关系

最适合借鉴实验设计；贡献包括优化研究的运行环境和基准，而不是一个新的 GRPO advantage 公式。代码选择 paper-v1 标签，避免把之后扩展的库版本误当作论文原始实现。

任务／评测：VeRO-Bench：数学、科学、事实检索、研究、客户对话与终端编码等。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
