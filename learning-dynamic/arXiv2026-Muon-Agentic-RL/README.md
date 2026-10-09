# When Does Muon Help Agentic Reinforcement Learning?

- 整理日期：2026-10-06
- 会议 / 状态：会议未确认
- 会议或元信息来源：[来源](https://arxiv.org/abs/2607.16169)
- arXiv：[论文页面](https://arxiv.org/abs/2607.16169v4)
- 本地 PDF：[Muon-Agentic-RL.pdf](./Muon-Agentic-RL.pdf)
- 下载来源：[原始 PDF](https://arxiv.org/pdf/2607.16169v4)

## 研究主题

优化器与优势估计的相互作用。在 ALFWorld 上比较 Muon 与 AdamW，并结合 GRPO、GiGPO、GraphGPO 研究学习率、有效更新尺度和最终成功率。

## 阅读与版本说明

采用 v4：更新 RMS 匹配后晚期成功率收益消失，调优 AdamW 后部分差距收窄；不能把早期版本的提升数字概括为 Muon 普遍更优。

## 代码

- 官方仓库：[https://github.com/x66ccff/verl-muon](https://github.com/x66ccff/verl-muon)
- 实体目录：`/usr/data/verl-muon`
- 本地入口：[verl-muon](./verl-muon)
- commit：`6692817d594d45af8558f719261edde14fcb63b2`
- 归档仅克隆仓库；未安装依赖、下载模型或运行训练。

## 校验

文件大小、SHA-256、PDF 有效性及标题匹配记录见 [manifest](../manifest.jsonl) 与 [校验报告](../download_summary.json)。

## 深入解读

[论文贡献、关键控制实验与局限](./analysis.md)。
