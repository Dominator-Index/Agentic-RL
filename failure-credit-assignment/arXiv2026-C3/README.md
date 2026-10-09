# C3

The Trace Is the State: Exact Credit Assignment for LLM Agent Teams

- 归档分类：arXiv2026，会议未确认。
- [论文页面](https://arxiv.org/abs/2603.06859v3) · [本地 PDF](./C3.pdf)。
- [代码仓库](https://github.com/EIT-EAST-Lab/C3) · [本地软链接](./C3) → `/usr/data/C3`。
- 检出 commit：`6c6191ab892ca2b25180eb850454d9612ba3a5c2`。
- 代码来源核对：官方实现；仓库标题、arXiv 编号与论文匹配。
- 整理日期：2026-10-07。

## 方法与本次 idea 的关系

固定消息前缀，在同一决策点替换候选消息并续跑下游 agent，以终局回报与其他候选的基线构造消息级贡献。直接比较 MAGRPO；需要反事实续跑，不是冻结 LLM 语义评分。

## 代码与版本说明

本地 PDF 为 v3（2026-09-26）。早期标题为 Contextual Counterfactual Credit Assignment for Multi-Agent Reinforcement Learning in LLM Collaboration；它们属于同一 arXiv 条目。代码入口：[统一训练脚本](./C3/scripts/40_train/paper_train.sh)、[训练配置](./C3/c3/utils/paper_train_contract.py)。其无偏性结论依赖论文的轨迹状态充分性等条件。

本次完成下载、版本记录和静态检查，未安装依赖或运行训练。

[专题对比与研究动机](../README.md) · [下载记录](../manifest.jsonl)

[详细算法解读](./analysis.md) · [四篇方法对比](../01-四篇论文-Idea与算法关键步骤对比.md)
