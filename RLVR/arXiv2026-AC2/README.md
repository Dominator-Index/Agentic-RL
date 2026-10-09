# AC2：Trust the Critic More

- **标题**：Trust the Critic More
- **方法**：Actor-Critic with Action Chunking（AC2）
- **作者**：Kaiyue Wen、Luke Bailey、Arvind Mahankali、Tengyu Ma；Stanford University。
- **版本**：arXiv:2609.39247v2，2026-10-01；首次提交 2026-09-30。归档日期：2026-10-08。
- **会议**：截至本次核对，未确认录用会议，使用 `arXiv2026-AC2` 命名。
- **论文**：[arXiv 页面](https://arxiv.org/abs/2609.39247) · [本地 PDF](./AC2-Trust-the-Critic-More.pdf)，29 页。
- **解读**：[analysis.md](./analysis.md)，包括公式、算法流程、数据监督、代码核对及与 GRPO／GiGPO／Dr.MAS 的区别。
- **概念专题**：[01｜Action Chunking 是什么：与 GRPO 和状态分组的区别](./01-Action-Chunking是什么-与GRPO和状态分组的区别.md)。
- **理解核对**：[02｜用可信 Critic 与 Partial Rollouts 节省推理采样](./02-理解核对-用可信Critic和Partial-Rollouts节省推理采样.md)。
- **飞书精简版**：[03｜AC2 核心理解](./03-AC2核心理解-飞书精简版.md)。
- **官方代码**：[WhenWen/AC2](https://github.com/WhenWen/AC2) · [本地代码入口](./AC2) → `/usr/data/AC2`。
- **代码版本**：`5fbb11ba3fbd720b526632259ab047e7e3b70384`。两次 Git 直连克隆失败，已改用 GitHub 官方 codeload 下载该 commit 的源码快照；目录包含完整源码文件，不含 Git 历史。来源见 [code_manifest.json](./code_manifest.json)。

核心区别：从历史完整轨迹抽取 prefix，在同一个 prefix 上采样若干新 action chunk。对 critic 已达到可信门槛的问题，用 chunk 末端的预测价值构造 advantage；对尚未达到门槛的问题与抽查组，仍完整 rollout 到终局评分。**它改变了优势估计和采样长度，并非只重分组既有终局奖励。**

| 项目 | 论文设置 |
|---|---|
| 策略模型 | Qwen3-4B-Thinking-2507 |
| 训练集 | FineProofs-RL，约 5,200 题；公开数据卡列 5,227 题 |
| 评测集 | IMO-ProofBench，60 题，每题采样 16 次 |
| actor 与 critic | 共享参数，critic 通过不同 prompt 输出离散价值；两套优化器状态 |
| 终局 judge | DeepSeek-V4-Flash；证明评分归一化到 0–1 |
| chunk / 总响应预算 | 10,000 个新 token / 50,000 token，后者包含 replay prefix |
| 主实验硬件 | 官方配置为 4 节点 × 8 GPU，共 32 GPU；已核对文件未明确给出卡型号 |
| 主结果 | 平均归一化评分峰值 20.57%，GRPO 为 18.50%；达到 GRPO 峰值用约 2.5 倍更少的 **actor decoding FLOPs**，不是全流程算力的等比例结论 |

没有安装训练依赖、下载模型权重或启动训练。源码中的启动脚本需要按本地集群配置后使用。
