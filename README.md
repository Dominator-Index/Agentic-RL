# Agentic-RL 论文分类导航

更新日期：2026-10-08。按论文主要研究问题归档，保留各论文原有的会议／年份／方法目录名。共 **52 篇独立论文**，其中 **51 篇已有本地 PDF**；NOPO 暂只有归档记录。ProGPO 的两份历史专题笔记都保留，因此实际论文子目录有 53 个。

| 类别 | 主要论文 | 归类依据 |
|---|---|---|
| [harness-engineering](./harness-engineering/README.md) | GEPA、DGM、VeRO、EvoC2F、CE-Graph、AutoSaddler、MoMHa、OPHSD 等 20 篇 | Agent 执行系统的设计、评估、自动优化与训练协同 |
| [credit-assignment](./credit-assignment/README.md) | GiGPO、HGPO、GraphGPO、ProGPO、CRBC、ProCredit、SCA | 分组优势、状态／动作／进度层面的信用分配 |
| [multi-agent-rl](./multi-agent-rl/README.md) | Dr.MAS、SRPO | 多智能体协作中的策略优化 |
| [context-management](./context-management/README.md) | AgentOCR、SUPO、ECHO | 交互历史、摘要、压缩与上下文选择 |
| [RLVR](./RLVR/README.md) | AC2、DyJR、NOPO | 推理任务中的可验证奖励、critic 与策略更新 |
| [learning-dynamic](./learning-dynamic/README.md) | RAGEN、AEM、Muon-Agentic-RL、Do We Need Adam?、AlphaRL、Low-Rank Dynamics、GeoRA、Jackpot | 学习动态、优化器、参数更新与采样分布失配 |
| [failure-credit-assignment](./failure-credit-assignment/README.md) | ProGPO、CriticSearch、MulFeRL、C3、FAULT | 失败反馈、自诊断与反事实信用 |
| [memory-based-adaptation](./memory-based-adaptation/README.md) | Memento、ACE | 固定底层生成模型的记忆／上下文适应 |
| [planning-and-control](./planning-and-control/README.md) | Agentic Monte Carlo、CoSo、ASOR | 采样规划、视觉策略与跨动力学控制 |

主题可能交叉。例如 C3 同时涉及 multi-agent，SCA 面向 GUI，AC2 也研究细粒度信用。本目录按主主题提供稳定位置，跨主题关系通过笔记和索引表达。

## 常用入口

- [Harness Engineering：方法地图与 2026 年会议录用核对](./harness-engineering/README.md)

- [完整论文索引：标题、会议、PDF 与解读](./论文索引.md)
- [逐篇论文的数据集与任务分类总表](./07-逐篇论文的数据集与任务分类总表.md)
- [数据集有无详细过程：原始字段与训练用法](./06-数据集监督类型核对-原始字段与训练用法.md)
- [GRPO 到 GraphGPO 与 Dr.MAS：时间线和分组优势](./01-GRPO到GraphGPO与DrMAS-时间线与分组优势对比.md)
- [研究 idea](./idea/README.md)
- [AC2：Trust the Critic More 解读](./RLVR/arXiv2026-AC2/analysis.md)

## 目录迁移

2026-10-08 将原来散放在本目录的 18 个论文目录实际移动到主题子目录，保留论文 PDF、已有笔记及代码入口，并更新内部链接。17 份随目录移动的 PDF 已核对 SHA-256，内容未变。[迁移路径与检查记录](./目录迁移记录-2026-10-08.json) · [分类与链接校验](./目录分类校验-2026-10-08.json)

根目录保留跨论文综述、索引、数据审计和 idea。已有 `/usr/data` 代码软链接继续可用；AC2 的官方源码快照和 Agentic Monte Carlo 的代码实体也放在 `/usr/data`。

## 本地工作目录与代码链接

2026-10-09 将本仓库工作目录迁至 `/usr/data/Agentic-RL`，原路径 `/usr/commondata/public/hf_hub/cc/awesome-exploration-rl/Agentic-RL` 通过软链接访问同一目录。

仓库保存论文、笔记及代码软链接，代码实体仍位于各自的 `/usr/data` 目录中。Git 只记录软链接本身，不包含其指向的代码文件；在其他机器上使用代码时，请按各论文 README 或代码清单中的官方地址下载并重建本地链接。笔记中指向 `/usr/data` 的源码行号链接用于本机阅读。
