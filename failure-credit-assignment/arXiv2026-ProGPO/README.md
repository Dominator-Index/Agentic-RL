# ProGPO

Progress-conditioned Group Policy Optimization for Long-Horizon Agentic Tasks

- 归档分类：arXiv2026，会议未确认。
- [论文页面](https://arxiv.org/abs/2607.22724v1) · [本地 PDF](./ProGPO.pdf)。
- [代码仓库](https://github.com/KeibingYang/Progress-conditioned-Group-Policy-Optimization-for-Long-Horizon-Agentic-Task) · [本地软链接](./ProGPO) → `/usr/data/ProGPO`。
- 检出 commit：`b14b06248a6021485df14a19a018505fa1f57e5f`。
- 代码来源核对：官方代码；复用已有检出。
- 整理日期：2026-10-07。

## 方法与本次 idea 的关系

仅在全失败组启用首次访问 observation coverage，产生进度相关的相对优势。它不使用冻结 LLM 评估器。

## 代码与版本说明

复用原目录的有效 PDF 和 analysis.md，原路径保持可用。已有分析指出：代码 C2 的相邻观测变化计分与论文首次访问定义存在区别，见 [原有分析](./analysis.md)。

本次完成下载、版本记录和静态检查，未安装依赖或运行训练。

[专题对比与研究动机](../README.md) · [下载记录](../manifest.jsonl)

[详细算法解读](./analysis.md) · [四篇方法对比](../01-四篇论文-Idea与算法关键步骤对比.md)
