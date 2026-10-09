# Meta-Harness：Meta-Harness: End-to-End Optimization of Model Harnesses

- 归档日期：2026-10-08。
- 会议／状态：COLM2026；[录用依据](https://kangwooklee.com/)。
- 来源：[论文页面](https://arxiv.org/abs/2603.28052v1) · [本地 PDF](./Meta-Harness.pdf)；版本：`2603.28052v1`。
- 类别：通过完整历史搜索 harness 源码。
- 更新对象：**优化外部 harness，底层任务模型保持固定**。
- 代码／资源：[官方仓库](https://github.com/stanford-iris-lab/meta-harness) · [本地入口](./Meta-Harness) → `/usr/data/harness-engineering/Meta-Harness`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`8123ccabe2b19fa1123090b2e5f1bacc20963ce8`。

另见作者页面所列 **ICML 2026 AIWILD Workshop Spotlight** 版本；本目录按其列出的 COLM 2026 归档。

## 1. 主要 idea

让提议 agent 直接访问历次候选的完整源码、分数和执行轨迹，避免将历史过早压缩成几条评语，再迭代提出新的 harness 程序。

## 2. 关键步骤

1. 把每次候选及其运行记录保存为文件。
2. 提议 agent 自主检索历史并生成修改后的 harness。
3. 执行评测，将完整反馈加入档案，继续外层搜索。

## 3. 与本专题的关系

是本类别的重要主线。作者页面列 COLM 2026，并单独列 ICML 2026 AIWILD Workshop Spotlight 版本；后者不应写成 ICML 主会。

任务／评测：在线文本分类、检索增强数学推理、Terminal-Bench 2.0。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
