# ICLR 2026 · ACE

**Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models**

- [论文页面](https://arxiv.org/abs/2510.04618) · [本地 PDF](./ACE.pdf) · [中文方法解读](./analysis.md)
- 归档版本：arXiv:2510.04618v3，2026-03-29；首次提交 2025-10-06。
- 会议：**ICLR 2026**，由 [会议官网](https://iclr.cc/Downloads/2026)、[作者项目页](https://ace-agent.github.io/) 和 PDF 首页确认。
- [官方代码](https://github.com/ace-agent/ace) → `/usr/data/ACE` → [本目录软链接](./ACE)。
- [代码版本](./code_manifest.json) · [PDF 清单](./manifest.jsonl) · [校验汇总](./download_summary.json)

固定 LLM 权重，以 Generator、Reflector、Curator 三个角色从执行经验中提炼策略，用增量条目维护 playbook，避免全量上下文重写造成知识丢失。属于记忆/上下文驱动的适应，不是 group-based advantage 算法。

解读包含与 Memento、ECHO、Dr.MAS 的对比，以及代码操作支持范围、无标准答案与无监督的区别。未执行训练或付费 API 调用。PDF 为 arXiv 原生文件，`manual_pdf` 仅表示临时文件导入方式。
