# SRPO

SRPO: Setwise Relative Policy Optimization for Multi-Agent Systems

- arXiv：2609.08452；首次公开 2026-09-08，本地版本 v2，更新于 2026-09-29。
- [论文页面](https://arxiv.org/abs/2609.08452) · [本地 PDF](./SRPO.pdf)（23 页）。
- [方法分析与数学对比](./analysis.md)。
- 会议尚未确认，暂按 `arXiv2026-SRPO` 归档。
- 尚未找到可确认的官方代码仓库，因此未 clone，也未创建空的代码软链接。
- v1 标题末尾为 “Multi-Agent LLMs”，v2 改为 “Multi-Agent Systems”。
- 元数据差异：PDF 正文首页作者包含 Bo An，arXiv 页面和 PDF Author 字段未包含；引用时需核对版本。
- SHA-256：`89225ec2bb27553432509cb1d91f1da443425d6108944c782b2f018de690fde4`。
- [归档记录](./manifest.jsonl) · [校验摘要](./download_summary.json)。

关键区别：SRPO 的 active set 是一次团队决策的联合输出，不是 GiGPO 那种跨轨迹比较组；它修改 ratio 的组合、缩放和 clipping 单位。
