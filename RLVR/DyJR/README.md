# DyJR

**DyJR: Preserving Diversity in Reinforcement Learning with Verifiable Rewards via Dynamic Jensen-Shannon Replay**

- 分类：RLVR；交叉主题为经验回放、探索多样性、learning dynamics。
- 来源：[arXiv:2603.16157v1](https://arxiv.org/abs/2603.16157v1)，2026-03-17；本地 PDF 14 页。
- [论文 PDF](./DyJR.pdf)
- [完整中文解读：方法、公式推导、实验与局限](./analysis.md)
- [01｜如何理解历史数据应优先维持多样性而非仅强化准确率](./01-如何理解历史数据应优先维持多样性而非仅强化准确率.md)
- 截至 2026-10-08，本次核对的[作者主页](https://bimu.wang/)标注 Under Review；没有据此追加会议录用标签。
- 本次未找到可确认的官方代码，目录中没有代码软链接。解读基于论文，未做训练复现。

核心：保留 GRPO 的组相对优势，用近期成功轨迹及其旧 token 概率构造额外正则，缓解生成分布过早集中。它不是重新划分 state/action group，也不是为失败轨迹提供过程奖励。

[返回 RLVR 分类](../README.md)
