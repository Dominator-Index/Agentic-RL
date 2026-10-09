# EvoC2F：EvoC2F: Compiling Tool Orchestration for Efficient and Evolvable LLM Agents

- 归档日期：2026-10-08。
- 会议／状态：ICML2026；[录用依据](https://icml.cc/virtual/2026/poster/63166)。
- 来源：[论文页面](https://proceedings.mlr.press/v306/wei26q.html) · [本地 PDF](./EvoC2F.pdf)；版本：官方论文集／作者发布 PDF，详见来源与 manifest。
- 类别：工具编排编译与经验证的技能演化。
- 更新对象：**会训练：planner 使用离线 DPO，技能检索 router 使用排序目标**。
- 代码／资源：截至 2026-10-08，未从本次核对的论文／作者来源确认可归属本论文的公开实现。未用同名项目或基线仓库代替。

## 1. 主要 idea

把工具调用计划约束成带依赖与副作用信息的 Plan IR，使执行器能在合法条件下并行、重试和恢复；成功经验提炼成技能后，先验证再加入库。

## 2. 关键步骤

1. 生成受约束的 Plan IR，标注数据依赖和读写／副作用信息。
2. 对满足条件的子计划编译与并行执行，记录轨迹。
3. 从成功轨迹提炼候选技能，经过功能、契约和回归检查后分阶段准入；用轨迹偏好更新 planner。

## 3. 与本专题的关系

同时涉及执行系统设计与参数训练。论文 §3 的公式 (9)–(10) 明确写出轨迹奖励与 DPO；不能把整个方法标成 training-free，也不能把有前提的编译保证外推为所有 agent 行为都正确。

任务／评测：StableToolBench、ShortcutsBench、τ-bench、ToolComp、TRAJECT-Bench、ToolSeq-500（后者为从 StableToolBench 抽样的 500 个顺序任务）。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
