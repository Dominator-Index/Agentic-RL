# 02｜ICLR 2026、ICML 2026、NeurIPS 2026：Harness Engineering 录用核对

核对日期：2026-10-08。结论：三个会议均找到直接或紧密相关的主会论文。本表是本次筛选结果，不声称穷尽全部录用论文。判定依据为官方会议展示页／正式论文集，不能将投稿模板、OpenReview 页面或 Workshop 展示视作主会录用。

## 1. 已核实主会

| 会议 | 论文 | 一手录用依据 | 本地阅读 |
|---|---|---|---|
| ICLR2026-Oral | GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning | [官方会议页面](https://iclr.cc/virtual/2026/poster/10009493) | [GEPA](./ICLR2026-Oral-GEPA/README.md) |
| ICLR2026 | Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents | [官方会议页面](https://iclr.cc/virtual/2026/poster/10007327) | [DGM](./ICLR2026-DGM/README.md) |
| ICML2026 | EvoC2F: Compiling Tool Orchestration for Efficient and Evolvable LLM Agents | [官方会议页面](https://icml.cc/virtual/2026/poster/63166) | [EvoC2F](./ICML2026-EvoC2F/README.md) |
| ICML2026 | Failure-Driven Workflow Refinement | [官方会议页面](https://icml.cc/virtual/2026/poster/65124) | [CE-Graph](./ICML2026-CE-Graph/README.md) |
| ICML2026 | VeRO: A Harness for Agents to Optimize Agents | [官方会议页面](https://icml.cc/virtual/2026/poster/60518) | [VeRO](./ICML2026-VeRO/README.md) |
| NeurIPS2026 | AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces | [官方会议页面](https://neurips.cc/virtual/2026/poster/151256) | [AutoSaddler](./NeurIPS2026-AutoSaddler/README.md) |
| NeurIPS2026 | MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens | [官方会议页面](https://neurips.cc/virtual/2026/poster/153336) | [MoMHa](./NeurIPS2026-MoMHa/README.md) |
| NeurIPS2026 | Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning | [官方会议页面](https://neurips.cc/virtual/2026/poster/149380) | [OPHSD](./NeurIPS2026-OPHSD/README.md) |
| ICLR2026（既有归档） | Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models | [作者录用公告](https://ace-agent.github.io/blogs/2026-01-30-iclr-acceptance/) | [ACE](../memory-based-adaptation/ICLR2026-ACE/README.md) |

ICML 的三篇另有正式论文集：[VeRO](https://proceedings.mlr.press/v306/ursekar26a.html)、[EvoC2F](https://proceedings.mlr.press/v306/wei26q.html)、[Failure-Driven Workflow Refinement](https://proceedings.mlr.press/v306/zhang26f.html)。GEPA 的 Oral 标注同时由归档 PDF 首页确认；其他论文未因具有 poster 页面而擅自推断 Oral 或 Spotlight 等等级。

## 2. 必须区分的 Workshop 与其他会议

- **JAZ / Harness as a Language**：[arXiv v2](https://arxiv.org/abs/2609.26891v2) 的 Comments 写明 NeurIPS 2026 **TTCL Workshop**。目录名明确保留 Workshop。
- **Meta-Harness**：[作者 Kangwook Lee 页面](https://kangwooklee.com/) 将 End-to-End Optimization 版本列为 **COLM 2026**；同时将 Post-Training Reliable Agent Systems via Harness Search 列为 **ICML 2026 AIWILD Workshop Spotlight**。两条都不是 ICML 2026 主会录用证据。
- 本次其余条目缺少可确认的正式会议记录，保持 `arXiv2026-...` 或 `preprint2026-...`，不根据排版模板和二手数据库自动补会议名。

## 3. 和你关心的研究线如何对应

| 研究问题 | 优先参考 |
|---|---|
| 固定基础模型，自动改提示或运行系统 | GEPA、DGM、AutoSaddler、MoMHa |
| 怎样公平评价“优化 agent 的 agent” | VeRO |
| 如何利用失败过程来改工作流 | CE-Graph、AutoSaddler |
| 工具调用的编译、验证与技能学习 | EvoC2F |
| harness 能否辅助模型训练 | OPHSD；进一步交叉读 JIT-Agent、Continual Harness |

与信用分配方向的关系是研究判断，并非这些论文都使用相同的 RL setup。各自更新的是提示、源码、工作流、技能库、生成器或任务模型，比较时应明确优化对象与成本。

[返回专题](./README.md)
