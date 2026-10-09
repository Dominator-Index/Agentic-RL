# JAZ：Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity

- 归档日期：2026-10-08。
- 会议／状态：NeurIPS2026 TTCL Workshop；[录用依据](https://arxiv.org/abs/2609.26891v2)。
- 来源：[论文页面](https://arxiv.org/abs/2609.26891v2) · [本地 PDF](./JAZ.pdf)；版本：`2609.26891v2`。
- 类别：最小递归 agent-loop 语言抽象。
- 更新对象：**论文核心评测不微调任务模型**。
- 代码／资源：[官方仓库](https://github.com/jaz-lang/jaz) · [本地入口](./JAZ) → `/usr/data/harness-engineering/JAZ`。方式：官方 GitHub commit 源码快照（Git 连接失败后回退，不含 Git 历史）；commit：`0803d4971be785e95b80054b02259664d70fa3da`。

## 1. 主要 idea

以一个可递归的 invoke 原语组织执行：模型可写可执行代码，也能把输入与历史当作程序变量操作，从统一循环表达记忆和自我改进。

## 2. 关键步骤

1. 通过 invoke 让 LLM 在代码环境中实现当前调用。
2. 允许生成代码递归调用 invoke，并显式访问输入及交互历史。
3. 通过 hooks 施加约束／监控，对长记忆与持续改进任务做验证。

## 3. 与本专题的关系

是 harness 表达能力与抽象设计论文。已核实的录用标注是 NeurIPS 2026 TTCL Workshop，不能写成 NeurIPS 主会。

任务／评测：StuLife、AppWorld。

以上为基于原始论文和官方来源的检索导读；未运行论文实验。不同文章的实验预算、版本和指标不同，不直接用各自的提升数字横向排名。

[专题索引](../README.md) · [研究地图](../01-Harness-Engineering-研究地图与Agentic-RL关系.md) · [会议录用核对](../02-ICLR2026-ICML2026-NeurIPS2026-录用核对.md)
