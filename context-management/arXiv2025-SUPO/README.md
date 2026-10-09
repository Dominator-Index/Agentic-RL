# SUPO

**Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management**

- 作者：Miao Lu、Weiwei Sun、Weihua Du、Zhan Ling、Xuesong Yao、Kang Liu、Jiecao Chen。
- arXiv：2510.06727v1，2025-10-08；核对日期：2026-10-07。
- [论文页面](https://arxiv.org/abs/2510.06727) · [本地 PDF](./SUPO.pdf) · [完整方法解读](./analysis.md)
- [下载校验清单](./manifest.jsonl) · [校验汇总](./download_summary.json) · [代码来源记录](./code_manifest.json)

SUPO 将摘要作为同一 policy 的动作，用终局奖励联合训练工具调用和摘要生成。一次 rollout 可拆成多个上下文分段，所有分段共享按完整 rollout 组计算的 advantage；超预算未完成的 rollout 使用 overlong mask。

## 会议与代码核对

检索到 [ICLR 2026 投稿](https://openreview.net/forum?id=778Yl5j1TE)，但当前未核实最终录用决定，暂按 arXiv2025-SUPO 归档。

没有确认原作者公开代码。Hugging Face 自动关联的 `lil-lab/respect` 是另一篇论文，已排除。ECHO 仓库包含相关 SUPO-style 基线说明，但不应标作 SUPO 官方代码。因此本次无新增代码 clone 或软链接。

PDF 为 arXiv 原生 PDF，经临时文件下载后导入；清单中 `manual_pdf` 表示导入方式，并非网页打印。已检查 PDF 格式、标题、页数、大小和 SHA-256。
