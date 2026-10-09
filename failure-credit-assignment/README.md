# 失败样本学习与信用分配

更新日期：2026-10-08。共 5 篇：ProGPO 复用已有归档，CriticSearch、MulFeRL、C3，以及本次从根目录移入的 FAULT。代码实体统一放在 `/usr/data`，各论文目录通过软链接访问。

先读 [四篇论文的核心 idea 与算法步骤对比](./01-四篇论文-Idea与算法关键步骤对比.md)。每篇目录另有 `analysis.md`，覆盖公式、例子、理论边界和代码核对。

## 1. 论文与代码索引

| 会议/年份 | 方法及本地资料 | 主要机制 | 与全失败多智能体归因的关系 |
|---|---|---|---|
| arXiv2026 | [ProGPO](./arXiv2026-ProGPO/README.md) · [PDF](./arXiv2026-ProGPO/ProGPO.pdf) · [代码](./arXiv2026-ProGPO/ProGPO) | 全失败时，以首次访问 observation coverage 补充 advantage | 已覆盖“全失败才触发”的设计 |
| ACL2026 Findings | [CriticSearch](./ACL2026-Findings-CriticSearch/README.md) · [PDF](./ACL2026-Findings-CriticSearch/CriticSearch.pdf) · [代码](./ACL2026-Findings-CriticSearch/CriticSearch) | 冻结 LLM 回看轨迹，对搜索 turn 做 Good/Bad 判断 | 已覆盖 frozen LLM 的局部归因；单个搜索 agent |
| arXiv2026 | [MulFeRL](./arXiv2026-MulFeRL/README.md) · [PDF](./arXiv2026-MulFeRL/MulFeRL.pdf) · [代码](./arXiv2026-MulFeRL/MulFeRL) | 全失败触发反馈、重新生成，以 verifier 确认的进步训练 | 触发与动机接近，但需要重新采样；不是团队归因 |
| arXiv2026 | [C3](./arXiv2026-C3/README.md) · [PDF](./arXiv2026-C3/C3.pdf) · [代码](./arXiv2026-C3/C3) | 替换团队消息、续跑下游，估计反事实贡献 | 直接研究多智能体并对比 MAGRPO；额外成本来自续跑 |
| arXiv2026 | [FAULT](./arXiv2026-FAULT/README.md) · [PDF](./arXiv2026-FAULT/FAULT.pdf) · [解读](./arXiv2026-FAULT/analysis.md) | 自诊断误差机制、信用与辅助训练 | 研究失败轨迹中错误机制的可学习归因 |

MulFeRL 的 GitHub 公开实现自述为匿名版本；论文匿名链接与 GitHub 的身份映射未独立确认。公开版缺少 reward functions 和评估脚本，详见该目录 README。其他仓库的官方声明、commit 和来源记录见 [代码清单](./code_manifest.json)。会议未确认的论文保持 arXiv 命名。

## 2. 本次 idea 的准确表述

针对一个问题，采样 N=8 条**完整团队协作轨迹**，每条可以包含多个 agent 的多次发言，并非八个独立角色。以终局二元奖励为例：

- 至少一条成功：保留基础算法的训练方式，不调用额外评估器。
- 八条全部失败：调用冻结 LLM，读取各条完整协作历史，定位有价值的纠错、错误引入、正确意见被忽略、错误意见被采纳等事件。
- 评估器输出 agent / turn / token span 位置、正负信用信号和置信度；只在足够确定的位置产生辅助更新。

这是一项待验证的方案。本次检索未确认完整相同的组合，但触发逻辑和冻结 turn 评估器分别已有先例，不能据此宣称首次。

## 3. Mask 不足以让零 advantage 恢复更新

如果原始组内奖励全部为零，则中心化 advantage 也为零：

$$
A_i=R_i-\frac{1}{N}\sum_jR_j=0.
$$

乘以任意位置掩码，仍然有 $M_{i,k,t,\ell}A_i=0$。这里指奖励驱动的策略梯度，独立 KL 项仍可能更新参数。

因此，一个最小的候选实现需要另外提供非零信号：

$$
\widetilde A_{i,k,t,\ell}=
\begin{cases}
A^{\mathrm{base}}_{i,k,t,\ell},&\sum_jR_j>0,\\
\lambda M_{i,k,t,\ell}C_{i,k,t},&\sum_jR_j=0.
\end{cases}
$$

其中 $i$ 是团队轨迹，$k$ 是 agent，$t$ 是该 agent 的 turn，$\ell$ 是 token；$M$ 定位，$C\in\{-1,0,+1\}$ 提供更新方向。该信号是评估器辅助监督，不自动等于真实因果 advantage，也没有自动的无偏保证。

原始 MAGRPO 允许每轮团队奖励，不是所有任务都只有终局 0/1。上述零信号讨论针对终局稀疏二元奖励及按该问题组中心化的设定；不能直接推广到任意跨问题归一化实现。

## 4. 最值得检验的研究点

终局失败不代表团队所有发言都无价值。例如 solver 犯错、verifier 正确纠错、solver 忽略纠错，归因应识别三个不同事件。这个跨 agent 的信息采纳关系，比对每段文字独立评 Good/Bad 更贴合多智能体协作。

建议在匹配总生成 token 和评估器成本后，对比基础 MAGRPO/Dr.MAS、额外 rollout、反馈重试、普通 turn 评分以及反事实续跑。另做全量评估与全失败触发、turn 与 token span、随机 mask 与语义 mask 的消融。至少一条成功并不保证有组内对比：八条全成功同样可能产生零 advantage，本方案优先处理的是困难全失败组。

## 5. 来源与校验

- [ProGPO](https://arxiv.org/abs/2607.22724v1)
- [CriticSearch 正式出版页](https://aclanthology.org/2026.findings-acl.596/)
- [MulFeRL v2](https://arxiv.org/abs/2601.22900v2)
- [C3 v3](https://arxiv.org/abs/2603.06859v3)
- [MAGRPO 原文](https://arxiv.org/abs/2508.04652)
- [PDF 下载清单](./manifest.jsonl) · [校验摘要](./download_summary.json) · [代码版本](./code_manifest.json)

仅进行归档和静态核对；未运行训练或复现实验。
