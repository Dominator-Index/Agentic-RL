# Learning Dynamics 与优化器论文索引

更新日期：2026-10-08。共 8 篇论文，均已下载 PDF。已确认会议使用“会议年份-名称”；其余以 arXiv 年份归档，会议未确认不代表未录用。本专题也收录 actor–policy 分布失配与训练稳定性研究。

4 个作者公开仓库已克隆并建立软链接；新增 Jackpot 采用官方固定 commit 源码快照，也已建立软链接，共 5 个代码／仓库入口。其中 Do We Need Adam 的仓库目前仅有空 README，没有可运行代码。AEM、Low-Rank Dynamics RLVR、GeoRA 尚未找到可确认的公开实现，不以相关框架冒充官方代码。

| 会议 / 状态 | 论文 | 主题 | 本地材料 |
| --- | --- | --- | --- |
| 会议未确认 | [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073v2) | 多轮 agent 训练动力学 | [PDF](./arXiv2025-RAGEN/RAGEN.pdf) · [说明](./arXiv2025-RAGEN/README.md) |
| NeurIPS2026 | [AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning](https://arxiv.org/abs/2605.00425v3) | 熵动力学与信用分配 | [PDF](./NeurIPS2026-AEM/AEM.pdf) · [说明](./NeurIPS2026-AEM/README.md) |
| 会议未确认 | [When Does Muon Help Agentic Reinforcement Learning?](https://arxiv.org/abs/2607.16169v4) | 优化器与优势估计的相互作用 | [PDF](./arXiv2026-Muon-Agentic-RL/Muon-Agentic-RL.pdf) · [说明](./arXiv2026-Muon-Agentic-RL/README.md) |
| ICML2026 | [Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs](https://arxiv.org/abs/2602.07729v2) | RLVR 优化器与更新稀疏性 | [PDF](./ICML2026-Do-We-Need-Adam/Do-We-Need-Adam.pdf) · [说明](./ICML2026-Do-We-Need-Adam/README.md) |
| ICLR2026 | [On Predictability of Reinforcement Learning Dynamics for Large Language Models](https://arxiv.org/abs/2510.00553v3) | 参数更新的低秩与时间演化 | [PDF](./ICLR2026-AlphaRL/AlphaRL.pdf) · [说明](./ICLR2026-AlphaRL/README.md) |
| 会议未确认 | [On the Implicit Reward Overfitting and the Low-rank Dynamics in RLVR](https://arxiv.org/abs/2605.06523v1) | 隐式奖励过拟合与低秩动力学 | [PDF](./arXiv2026-Low-Rank-Dynamics-RLVR/Low-Rank-Dynamics-RLVR.pdf) · [说明](./arXiv2026-Low-Rank-Dynamics-RLVR/README.md) |
| ACL2026 Outstanding Paper | [GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](https://arxiv.org/abs/2601.09361v4) | RLVR 的几何结构与低秩适配 | [PDF](./ACL2026-GeoRA/GeoRA.pdf) · [说明](./ACL2026-GeoRA/README.md) |
| ICLR2026 | [Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch Reinforcement Learning](https://arxiv.org/abs/2602.06107v1) | OBRS 分布对齐、异策略采样与训练稳定性 | [PDF](./ICLR2026-Jackpot/Jackpot.pdf) · [导读与代码](./ICLR2026-Jackpot/README.md) |

## 优化器核对

[GiGPO、HGPO、GraphGPO 与 Dr. MAS 的优化器代码核对](./01-分组方法的优化器核对.md)。

## 归档记录

- [PDF 清单与哈希](./manifest.jsonl)
- [下载校验报告](./download_summary.json)
- [代码路径与 commit](./code_manifest.json)
