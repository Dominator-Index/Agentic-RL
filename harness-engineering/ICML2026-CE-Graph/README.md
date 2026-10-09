# CE-Graph：Failure-Driven Workflow Refinement

- 归档日期：2026-10-08。
- 会议／状态：ICML2026；[录用依据](https://icml.cc/virtual/2026/poster/65124)。
- 来源：[论文页面](https://proceedings.mlr.press/v306/zhang26f.html) · [本地 PDF](./CE-Graph.pdf)；版本：官方论文集／作者发布 PDF，详见来源与 manifest。
- 类别：失败驱动的工作流图优化。
- 更新对象：**主要修改 workflow 图；此处的分布式／类梯度表述不等于 LLM policy-gradient 更新**。
- 代码／资源：截至 2026-10-08，未从本次核对的论文／作者来源确认可归属本论文的公开实现。未用同名项目或基线仓库代替。

归档的是 [ICML 论文集正式版](https://proceedings.mlr.press/v306/zhang26f.html)，不将早期 arXiv:2510.10035 的作者列表与正式版混写。

## 1. 主要 idea

把失败轨迹编码到 Failure Signature Space，集中修复反复出现的失败模式，而不是仅将一条轨迹压缩成 0/1 后做盲目的整体图搜索。

## 2. 关键步骤

1. 执行 workflow，保存反例，并按失败特征近似失败分布。
2. 定位高密度失败模式，选择受约束的图修改操作。
3. 通过 Propose-and-Verify 验证修改，并用收敛相关规则决定是否继续。

## 3. 与本专题的关系

适合研究“失败轨迹如何指导系统修改”。它把细粒度失败信息用于 workflow 编辑，不是在同一批轨迹中重新计算 GRPO advantage。本目录归档 ICML 正式版本；较早 arXiv 版本作者列表不同，不混用版本元数据。

任务／评测：数学、代码、问答任务。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
