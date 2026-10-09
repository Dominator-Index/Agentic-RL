# 记忆与上下文驱动的 Agent 适应

这个专题收录保持底层规划、执行 LLM 权重固定，通过外部经验记忆、策略手册、案例检索或可学习的记忆选择来改善 agent 行为的方法。

## 分类边界

- 可以更新案例库、检索索引，以及独立的检索器或选择器参数。
- “不微调底层 LLM”不等于“整个系统无训练”。
- 与直接用 policy gradient 更新生成 LLM 的 GiGPO、Dr.MAS 等方法区分。
- SUPO、ECHO 虽然也使用记忆或上下文压缩，但会训练生成 policy，暂不移入本专题。

## 论文

- **arXiv2025: Memento:** [Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](https://arxiv.org/abs/2508.16153) — [PDF](./arXiv2025-Memento/Memento.pdf) · [解读](./arXiv2025-Memento/analysis.md) · [代码软链接](./arXiv2025-Memento/Memento)

- **ICLR2026: ACE:** [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) — [PDF](./ICLR2026-ACE/ACE.pdf) · [解读](./ICLR2026-ACE/analysis.md) · [代码软链接](./ICLR2026-ACE/ACE)

## 存储规则

专题下继续采用“会议年份或 arXiv年份-方法简称”的论文目录。代码实体放在 `/usr/data`，论文目录仅保留软链接。
