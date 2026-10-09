# Harness Engineering：设计、评估与自动优化 Agent 执行系统

更新：2026-10-08。首批归档 **20 篇新论文、20 份作者／会议原生 PDF**，其中已核实 **ICLR 2026 主会 2 篇、ICML 2026 主会 3 篇、NeurIPS 2026 主会 3 篇**。另有 COLM 2026 论文、NeurIPS Workshop 论文和预印本。既有 ACE 提供跨专题入口，不重复下载计数。

这里的 harness engineering 指模型外部执行系统的设计与优化：上下文与状态、工具接口、执行循环、任务编排、反馈与验证、失败恢复等。研究对象包括可执行程序、配置和提示，也包括与模型参数协同学习的方法。纯粹使用 lm-evaluation-harness 做评测的论文不因名称相同而收入。

## 阅读入口

- [01｜研究地图、方法分组与 Agentic RL 的关系](./01-Harness-Engineering-研究地图与Agentic-RL关系.md)
- [02｜ICLR 2026、ICML 2026、NeurIPS 2026 录用核对](./02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
- 优先阅读：**VeRO → AutoSaddler → GEPA／DGM → OPHSD**。前两篇帮助定义实验对象与优化过程，中间两篇比较提示／源码搜索，OPHSD 连接到参数训练。

## 论文与方法对照

| 会议／状态 | 论文 | 主要优化对象 | 更新任务 LLM 参数？ |
|---|---|---|---|
| ICLR2026-Oral | [GEPA](./ICLR2026-Oral-GEPA/README.md) | 提示与复合 LLM 程序优化 | 不更新任务 LLM 权重；进化各模块的 prompt |
| ICLR2026 | [DGM](./ICLR2026-DGM/README.md) | Agent 程序与工具的开放式进化 | 主要修改 agent 自身程序；底层 foundation model 保持固定 |
| ICML2026 | [EvoC2F](./ICML2026-EvoC2F/README.md) | 工具编排编译与经验证的技能演化 | 会训练：planner 使用离线 DPO，技能检索 router 使用排序目标 |
| ICML2026 | [CE-Graph](./ICML2026-CE-Graph/README.md) | 失败驱动的工作流图优化 | 主要修改 workflow 图；此处的分布式／类梯度表述不等于 LLM policy-gradient 更新 |
| ICML2026 | [VeRO](./ICML2026-VeRO/README.md) | Harness 优化的外层基础设施与评测 | 框架本身不要求任务 LLM 权重更新；优化器通过代码修改目标 agent |
| NeurIPS2026 | [AutoSaddler](./NeurIPS2026-AutoSaddler/README.md) | 从执行轨迹自动修改完整 harness | 优化 harness 代码与配置；文中 offline learning 不应直接解读为微调任务 LLM |
| NeurIPS2026 | [MoMHa](./NeurIPS2026-MoMHa/README.md) | 准确率、行为安全和 token 成本的多目标优化 | 由 agent 搜索 Python harness；主要不更新底层 LLM 权重 |
| NeurIPS2026 | [OPHSD](./NeurIPS2026-OPHSD/README.md) | 把 harness 能力蒸馏进模型 | 会更新任务模型参数：on-policy harness self-distillation |
| arXiv2026 | [AutoHarness](./arXiv2026-AutoHarness/README.md) | 合法动作检查器与代码策略合成 | 搜索与修改代码；不微调任务 LLM |
| arXiv2026 | [NLAH](./arXiv2026-NLAH/README.md) | 自然语言形式的可执行 harness | 主要是表示与运行时设计，不训练任务模型 |
| COLM2026 | [Meta-Harness](./COLM2026-Meta-Harness/README.md) | 通过完整历史搜索 harness 源码 | 优化外部 harness，底层任务模型保持固定 |
| arXiv2026 | [AHE](./arXiv2026-AHE/README.md) | 可观测性驱动的 harness 组件演化 | 主要修改组件文件，不更新任务 LLM 权重 |
| arXiv2026 | [Continual-Harness](./arXiv2026-Continual-Harness/README.md) | 不重置环境的在线 harness 适应 | 基础版本改 harness；扩展实验还包含模型参数 co-learning |
| preprint2026 | [Harness-Engineering-Survey](./preprint2026-Harness-Engineering-Survey/README.md) | Harness 分层综述 | 综述，无单一训练算法 |
| arXiv2026 | [Code-as-Agent-Harness](./arXiv2026-Code-as-Agent-Harness/README.md) | 以代码为载体的 harness 综述 | 综述，无单一训练算法 |
| arXiv2026 | [Adaptive-Auto-Harness](./arXiv2026-Adaptive-Auto-Harness/README.md) | 异质任务流上的构建与路由 | 主要演化和选择外部 harness；核心并非任务 LLM 的 GRPO 更新 |
| arXiv2026 | [Harness-Anatomy](./arXiv2026-Harness-Anatomy/README.md) | 生产 coding harness 的源码研究 | 源码审计，无单一训练算法 |
| arXiv2026 | [JIT-Agent](./arXiv2026-JIT-Agent/README.md) | 训练按任务生成 harness 的模型 | 训练 harness 生成／修复模型（含 SFT、DPO 等阶段）；被辅助的执行 LLM 可以保持固定 |
| NeurIPS2026 TTCL Workshop | [JAZ](./NeurIPS2026-TTCL-Workshop-JAZ/README.md) | 最小递归 agent-loop 语言抽象 | 论文核心评测不微调任务模型 |
| arXiv2026 | [Mixture-Self-Improving-Branches](./arXiv2026-Mixture-Self-Improving-Branches/README.md) | 多分支自改进与推理时路由 | 主要演化 harness、开发子集及提议策略；不据此声称训练基础 LLM |

## 与现有分类交叉阅读

- [ICLR2026 ACE](../memory-based-adaptation/ICLR2026-ACE/README.md)：上下文 playbook 的演化；[作者录用公告](https://ace-agent.github.io/blogs/2026-01-30-iclr-acceptance/)。
- [Memento](../memory-based-adaptation/arXiv2025-Memento/README.md)：外部经验记忆与检索。
- [上下文管理专题](../context-management/README.md)：SUPO、ECHO 等涉及实际 policy 训练，与固定模型的 context 改写不能混同。
- [多 agent RL](../multi-agent-rl/README.md) 与 [信用分配](../credit-assignment/README.md)：Dr.MAS、GiGPO 等主要更新 policy 参数，和 harness 搜索是不同的优化维度。

## 归档说明

每篇的 README 包含 idea、关键步骤、更新对象、任务类型与来源；本批是检索导读，尚非逐篇完整精读。主会、Workshop、作者确认的其他会议和未确认预印本分别标注。检索优先使用官方会议页面、论文集、arXiv、作者项目页与论文中指定的代码链接。

代码或配套资源实体放在 `/usr/data/harness-engineering/`，论文目录通过同名软链接访问。Git 连接失败时使用官方 GitHub 固定 commit 源码快照，每篇 `code_manifest.json` 记录方式与版本；未找到官方代码的文章明确标为未定位，不克隆名称相似的第三方实现。

[PDF manifest](./manifest.jsonl) · [下载与校验报告](./download_summary.json) · [来源与版本](./sources.json) · [代码归档记录](./code_inventory.json) · [返回总目录](../README.md)
