# Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs

- 整理日期：2026-10-06
- 会议 / 状态：ICML2026
- 会议或元信息来源：[来源](https://proceedings.mlr.press/v306/mukherjee26c.html)
- arXiv：[论文页面](https://arxiv.org/abs/2602.07729v2)
- 本地 PDF：[Do-We-Need-Adam.pdf](./Do-We-Need-Adam.pdf)
- 下载来源：[原始 PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/mukherjee26c/mukherjee26c.pdf)

## 研究主题

RLVR 优化器与更新稀疏性。比较 SGD 与 AdamW，检验动量和逐参数自适应学习率在 RLVR 中的作用，研究参数更新稀疏性。

## 阅读与版本说明

下载的是 PMLR 正式会议版。官方仓库已克隆，但当前 commit 只有空 README.md，尚无可用实现。主要研究 RLVR，不应直接等同于长程环境交互 agent。

## 代码

- 官方仓库：[https://github.com/SagnikMukherjee/sgd_adam_rlvr](https://github.com/SagnikMukherjee/sgd_adam_rlvr)
- 实体目录：`/usr/data/sgd_adam_rlvr`
- 本地入口：[sgd_adam_rlvr](./sgd_adam_rlvr)
- commit：`9e932356da2f7d2263ed6af1097e70782fb4b5f4`
- 归档仅克隆仓库；未安装依赖、下载模型或运行训练。

## 校验

文件大小、SHA-256、PDF 有效性及标题匹配记录见 [manifest](../manifest.jsonl) 与 [校验报告](../download_summary.json)。
