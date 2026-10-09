# GEPA：GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

- 归档日期：2026-10-08。
- 会议／状态：ICLR2026-Oral；[录用依据](https://iclr.cc/virtual/2026/poster/10009493)。
- 来源：[论文页面](https://arxiv.org/abs/2507.19457) · [本地 PDF](./GEPA.pdf)；版本：`2507.19457v2`。
- 类别：提示与复合 LLM 程序优化。
- 更新对象：**不更新任务 LLM 权重；进化各模块的 prompt**。
- 代码／资源：[官方仓库](https://github.com/gepa-ai/gepa) · [本地入口](./GEPA) → `/usr/data/harness-engineering/GEPA`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`fb1ed589fd83372caef499cffc2c73173d3b096b`。

## 1. 主要 idea

将 rollout 中的推理、工具调用和错误反馈保留下来，用自然语言反思提出 prompt 修改；按不同验证样本上的优势保留多样候选，避免只追逐一个总体平均分。

## 2. 关键步骤

1. 执行当前复合 LLM 系统，收集分数和可解释的执行轨迹。
2. 从候选的逐样本 Pareto 前沿选择父代，用小批量反馈反思并修改模块提示。
3. 验证修改，保存有互补优势的候选；可合并不同候选积累的规则。

## 3. 与本专题的关系

属于 harness 中 prompt 这一层的自动优化；原论文并不等同于任意运行时源码的全面搜索。与 GRPO 比较的是不同优化对象和反馈利用方式。

任务／评测：HotpotQA、HoVer、AIME、LiveBench-Math、IFBench、PUPA 等。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
