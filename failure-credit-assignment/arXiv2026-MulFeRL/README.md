# MulFeRL

MulFeRL: Enhancing Reinforcement Learning with Verbal Feedback in a Multi-turn Loop

- 归档分类：arXiv2026，会议未确认。
- [论文页面](https://arxiv.org/abs/2601.22900v2) · [本地 PDF](./MulFeRL.pdf)。
- [代码仓库](https://github.com/Cocobalt/MulFeRL) · [本地软链接](./MulFeRL) → `/usr/data/MulFeRL`。
- 检出 commit：`6616a07eb89d32721e39faa187eae01e48685978`。
- 代码来源核对：论文链接匿名仓库 https://anonymous.4open.science/r/MulFeRL；已克隆公开的 Cocobalt/MulFeRL，其 README 自述为该方法的匿名实现，方法与路径吻合。匿名站 API 返回 401，未独立核实它与 GitHub 仓库的映射，因此不把作者身份链标为已确认。
- 整理日期：2026-10-07。

## 方法与本次 idea 的关系

全失败组触发语言反馈和重新采样；出现成功/失败对比后使用 GRPO，反馈导致全失败到全成功时使用 FCO。最终始终失败则不更新。multi-turn 不等于 multi-agent。

## 代码与版本说明

代码入口：[启动脚本](./MulFeRL/verl-main/examples/sglang_multiturn/MulFeRL/mulferl.sh)、[训练流程](./MulFeRL/verl-main/verl/trainer/ppo/ray_trainer.py)、[actor 更新](./MulFeRL/verl-main/verl/workers/actor/dp_actor.py)。README 标注默认 K=8，并明确公开版未包含 reward functions、benchmark evaluators 和 evaluation scripts；运行前需补充 verifier。

本次完成下载、版本记录和静态检查，未安装依赖或运行训练。

[专题对比与研究动机](../README.md) · [下载记录](../manifest.jsonl)

[详细算法解读](./analysis.md) · [四篇方法对比](../01-四篇论文-Idea与算法关键步骤对比.md)
