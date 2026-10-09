# CriticSearch

CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic

- 归档分类：ACL 2026 Findings。
- [论文页面](https://aclanthology.org/2026.findings-acl.596/) · [本地 PDF](./CriticSearch.pdf)。
- [代码仓库](https://github.com/zhyaoch/CriticSearch) · [本地软链接](./CriticSearch) → `/usr/data/CriticSearch`。
- 检出 commit：`2141a49a849ed81ee52c50df3643b246ffc16abc`。
- 代码来源核对：仓库 README 明确声明为官方实现，并引用 arXiv:2511.12159。
- 整理日期：2026-10-07。

## 方法与本次 idea 的关系

冻结 LLM 利用完整轨迹和标准答案，对搜索调用给出 Good/Bad，再将局部信号与终局 advantage 混合。是单个搜索 agent 的多轮训练，不是多智能体 MAGRPO。

## 代码与版本说明

代码入口：[训练脚本](./CriticSearch/train.sh)、[冻结评估器服务](./CriticSearch/search_r1/contribute_reward/vllm_reward_server.py)。仓库默认模型配置不应直接视为论文所有实验的配置。

本次完成下载、版本记录和静态检查，未安装依赖或运行训练。

[专题对比与研究动机](../README.md) · [下载记录](../manifest.jsonl)

[详细算法解读](./analysis.md) · [四篇方法对比](../01-四篇论文-Idea与算法关键步骤对比.md)
