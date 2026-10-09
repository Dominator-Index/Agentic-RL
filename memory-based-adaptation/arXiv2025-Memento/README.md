# Memento

**Memento: Fine-tuning LLM Agents without Fine-tuning LLMs**

- [arXiv](https://arxiv.org/abs/2508.16153) · [PDF](./Memento.pdf) · [中文方法解读](./analysis.md)
- 归档版本：v2，2025-08-25；首次提交：2025-08-22。
- 会议：本次未确认正式录用信息，暂用 arXiv2025-Memento。
- [官方代码](https://github.com/Memento-Teams/Memento) → `/usr/data/Memento` → 本目录软链接 [Memento](./Memento)。
- [代码版本记录](./code_manifest.json) · [PDF 清单](./manifest.jsonl) · [下载校验](./download_summary.json)

核心方法：保持 planner/executor 的底层 LLM 固定，积累历史任务案例，通过相似度或学习到的案例效用检索，改善新任务的规划。参数化检索器有梯度训练，不应理解成整个系统没有参数更新。

已检查论文和检索训练/推理代码，未执行依赖安装、API 调用或训练。PDF 为 arXiv 原生文件，`manual_pdf` 仅表示通过临时下载文件导入归档。
