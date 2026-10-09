# GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR

- 整理日期：2026-10-06
- 会议 / 状态：ACL2026 Outstanding Paper
- 会议或元信息来源：[来源](https://aclanthology.org/2026.acl-long.1110/)
- arXiv：[论文页面](https://arxiv.org/abs/2601.09361v4)
- 本地 PDF：[GeoRA.pdf](./GeoRA.pdf)
- 下载来源：[原始 PDF](https://arxiv.org/pdf/2601.09361v4)

## 研究主题

RLVR 的几何结构与低秩适配。利用 RL 更新子空间的几何先验，以 SVD 提取方向初始化低秩适配器，并冻结残余部分作为结构锚点；研究训练稳定性、效率和遗忘。

## 阅读与版本说明

ACL2026 Main，Outstanding Paper，由 ACL Anthology 和作者主页确认。采用 arXiv v4。它改变参数化与可训练子空间，不是独立的 AdamW 替代优化器。

## 代码

截至整理日期，已检查论文页面、正文中的代码链接及相关检索结果，尚未找到可确认的作者公开实现，未创建代码软链接。

## 校验

文件大小、SHA-256、PDF 有效性及标题匹配记录见 [manifest](../manifest.jsonl) 与 [校验报告](../download_summary.json)。

## 深入解读

[方法、数学公式与证据边界](./analysis.md)。
