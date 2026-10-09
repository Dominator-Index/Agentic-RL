# OPHSD：Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning

- 归档日期：2026-10-08。
- 会议／状态：NeurIPS2026；[录用依据](https://neurips.cc/virtual/2026/poster/149380)。
- 来源：[论文页面](https://arxiv.org/abs/2605.08741v1) · [本地 PDF](./OPHSD.pdf)；版本：`2605.08741v1`。
- 类别：把 harness 能力蒸馏进模型。
- 更新对象：**会更新任务模型参数：on-policy harness self-distillation**。
- 代码／资源：[官方仓库](https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation) · [本地入口](./OPHSD) → `/usr/data/harness-engineering/OPHSD`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`20eaa0c8b0674fbf81aa780a8cf4e0d7c550d4ea`。

## 1. 主要 idea

当前学生驱动 draft–verify 或 plan–solve harness 生成增强上下文，再由冻结的基础模型在该上下文中提供 token 级监督，将外部流程的收益蒸馏给学生。

## 2. 关键步骤

1. 用当前学生生成 on-policy 轨迹／前缀。
2. 当前学生运行 harness 得到动态上下文；冻结基础模型在该上下文与学生前缀上计算教师分布，上下文停止梯度。
3. 用蒸馏目标更新学生，评价去掉 harness 后的独立推理能力。

## 3. 与本专题的关系

与“额外模块辅助训练”很相关。它也使用冻结基础模型，但输出的是教师分布，不是只选哪些 token 更新的 mask。动态的是学生驱动的 harness 上下文；论文 §3.2 将 logit teacher 参数固定。数学设置还使用参考解答作为特权信息，不能当作只有最终 0/1 奖励的同一 setup。关于推理时重新加 harness 无增益的结论限于论文所测任务和 harness。

任务／评测：文本分类、数学推理；包括 HMMT25 等。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
