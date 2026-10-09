# ICLR2026｜Jackpot

**Jackpot: Optimal Budgeted Rejection Sampling for Extreme Actor-Policy Mismatch Reinforcement Learning**

- 作者：Zhuoming Chen、Hongyi Liu、Yang Zhou、Haizhong Zheng、Beidi Chen；Carnegie Mellon University。
- 会议：**ICLR 2026**；依据 [arXiv 的会议标注](https://arxiv.org/abs/2602.06107)和[第一作者的录用公告](https://dreaming-panda.github.io/)。
- 归档：2026-10-08；版本 `2602.06107v1`，2026-02-05；21 页。
- [本地 PDF](./Jackpot.pdf) · [arXiv](https://arxiv.org/abs/2602.06107v1) · [官方项目页](https://infini-ai-lab.github.io/jpt_website/)。
- [官方代码](https://github.com/Infini-AI-Lab/jackpot) · [本地源码入口](./jackpot) → `/usr/data/jackpot`。
- 代码 commit：`c3f936e50a929bb436c77ba1512c5ef22fb79e12`。Git clone 超时后下载官方固定 commit 的源码快照，**不含 Git 历史**。版本记录见 [code_manifest.json](./code_manifest.json)。

## 1. 为什么归入 learning-dynamic

主分类：**训练稳定性／异策略采样／actor–policy 分布失配**。

论文问的是：能否让一个更小、更便宜的模型生成 rollout，再用这些 rollout 训练更大的模型？两者输出分布差异很大时，直接训练或仅做截断重要性采样可能不稳定；Jackpot 对进入更新的 token 分布进行对齐。

因此归入现有的 `learning-dynamic`，与 RAGEN 等训练稳定性研究交叉阅读。它主要改进采样与更新之间的分布关系，**不要求像 GiGPO／HGPO 那样跨 trajectory 寻找相同中间 state，也不是通过换优化器达到效果**。实验主要是数学 RL，另可从 RLVR 角度阅读。

这里的 actor 指 rollout 生成模型，policy 指被训练的目标模型；这不是 Dr.MAS 中 solver／verifier 的多角色协作设置。

## 2. 核心 idea：把极端不匹配的 token 排除出直接训练信号

传统重要性采样主要调整已有样本的损失权重。Jackpot 在此基础上引入 **Optimal Budgeted Rejection Sampling（OBRS）**：根据采样模型与目标模型对同一个 token 的概率，决定是否保留该 token 的损失，再对保留下来的样本做概率修正。

固定一段上下文 $h$，令：

- $q(v\mid h)$：rollout 模型对下一个 token $v$ 的分布。
- $p(v\mid h)$：希望对齐的目标分布，可选旧策略快照或当前策略。
- $\lambda>0$：控制接受严格程度的参数。

接受概率与随机掩码为：

$$
\alpha(v\mid h)=\min\left(1,\frac{p(v\mid h)}{\lambda q(v\mid h)}\right),\qquad
m(v\mid h)\sim\operatorname{Bernoulli}(\alpha(v\mid h)).
$$

如果采样模型对某个 token 的概率很高，而目标模型认为它很不可能，保留概率就会降低。掩码衡量的是**分布匹配程度**，不是这个 token 对最终答案的因果贡献或数学正确性。

接受后的条件分布是：

$$
\widetilde q(v\mid h)=\frac{q(v\mid h)\alpha(v\mid h)}{Z(h)},\qquad
Z(h)=\sum_{u\in\mathcal V}\min\left(q(u\mid h),\frac{p(u\mid h)}\lambda\right).
$$

$\mathcal V$ 是词表，$Z(h)$ 同时是该上下文下的期望接受率。后续重要性修正应考虑 $\widetilde q$，而不能在筛选后仍假设样本分布完全等于原始 $q$。方法中的 stop-gradient 避免通过这些修正权重反向优化接受规则本身。

**拒绝发生在训练损失上。** 论文 §4.4.3 明确不因某个 token 被拒绝而重新生成整段后缀；原始 rollout 和其上下文仍然存在。屏蔽该 token 的直接损失，也不等于删除它对后续上下文的影响。

## 3. 算法的关键步骤

1. 较便宜的 rollout 模型生成轨迹，保存采样 token 的 log-probability 和 top-k 概率信息。
2. 策略模型在这些已生成上下文上计算相应概率，确定 OBRS 接受掩码。
3. 用两侧 top-k token 的并集近似 $Z(h)$，再利用 batch 接受率校正整体尺度，降低保存全词表概率的开销。
4. 为保留的 token 计算基于接受后分布的截断重要性修正，施加到 PPO／GRPO 类策略更新。
5. 双模型训练方案还更新 rollout 模型，并通过从策略模型到 rollout 模型的蒸馏来限制两者继续偏离。

论文采用的 OBRS 来自已有拒绝采样研究；Jackpot 的贡献在于将它用于极端 actor–policy 失配的 RL，结合双模型训练、损失修正和 top-k 实现。固定分布与给定接受预算下的分布对齐结论，不等于“整个训练过程必然收敛”或“所有上下文都已经来自目标策略”。

## 4. 公式与实际代码对应

核心实现：[verl/utils/jackpot.py](/usr/data/jackpot/verl/utils/jackpot.py:26)，函数 `compute_jackpot_weights`。

| 方法步骤 | 代码位置 | 说明 |
|---|---|---|
| 根据概率比采样接受掩码 | [jackpot.py:53](/usr/data/jackpot/verl/utils/jackpot.py:53) | 调用 `bernoulli_mask_min_ratio_from_logp`，再与原始 `response_mask` 取交集 |
| 近似 $Z$ 并校正尺度 | [jackpot.py:60](/usr/data/jackpot/verl/utils/jackpot.py:60) | top-k overlap，加上 `acceptance_rate / overlap_mean` 校准 |
| 接受后分布的 log-probability | [jackpot.py:74](/usr/data/jackpot/verl/utils/jackpot.py:74) | `log_qacc = min(log_q, log_p - log_lambda)`，再减去归一化常数的 log |
| 构造重要性修正 | [jackpot.py:78](/usr/data/jackpot/verl/utils/jackpot.py:78) | 两个概率比各自截断，再相乘并 detach |
| 将掩码和权重接入 policy loss | [dp_actor.py:622](/usr/data/jackpot/verl/workers/actor/dp_actor.py:622) | 替换 `rollout_is_weights` 与 `response_mask`；原有 `advantages` 仍传给损失函数 |

下面是原始关键语句，省略了调用参数和日志：

```python
mask_kept = response_mask.bool() & mask
log_qacc = torch.minimum(rollout_log_probs, prob_to_align_detached - log_lambda)
log_p_star = log_qacc - torch.log(overlap_mass.clamp_min(1e-12))
weights = (jackpot_ratio * mask_kept.to(jackpot_ratio.dtype)).detach()
```

这与 GiGPO 的区别很直观：**GiGPO 主要重构 `advantages`；Jackpot 主要构造分布修正权重和 token 接受掩码。** 二者概念上可作用于不同环节，但组合后的有效性需要实际验证。

运行入口可阅读 [官方 GSM8K 示例目录](./jackpot/examples/jackpot_examples_gsm8k/README.md)。该目录提供单模型和双模型例子；它是发布版演示配置，不能据此将论文主训练集全部改写成 GSM8K。本文只做静态阅读，没有安装依赖或启动训练。

## 5. 实际使用的数据集与模型

依据 PDF §5、Table 1、§6 和附录：

| 实验设置 | 训练数据 | 说明 |
|---|---|---|
| Qwen2.5-1.5B-Base rollout → Qwen2.5-3B-Base policy | **MATH**，Table 1 标注 MATH-8K | 不同尺寸模型之间的分布失配 |
| Qwen3-1.7B-Base rollout → Qwen3-4B-Base policy | **DeepScaleR** | 双模型训练 |
| Qwen3-1.7B-Base rollout → Qwen3-8B-Base policy | **DeepScaleR** | 更大的 actor–policy 能力／分布差距 |
| rollout staleness、FP8 KV 等消融 | **DeepScaleR-Preview** 等文中配置 | 研究不同失配来源与已有修正的作用 |

评测包括 **GSM8K、MATH-500、AMC22/23、AMC12 2024、AIME 2024、AIME 2025**。论文主要实验不是 ALFWorld／WebShop 的交互环境。

MATH 原始数据可有详细解答，但本文使用 rollout 上的 RL 训练信号，并加入模型之间的蒸馏；不能因此将其描述为对数据集人工标准步骤做 SFT。蒸馏教师分布与数据集自带解题过程是两种不同监督来源。

## 6. 结论与适用边界

论文报告：在严重失配的双模型训练中，Jackpot 比直接 off-policy 更新和 TIS 基线更稳定；Qwen3-1.7B rollout 支持训练 Qwen3-8B，在文中设置下可稳定到约 300 个更新步，并接近 on-policy 表现。

但论文也明确指出：

- 分布差异本来较小、已有 clipping／TIS 足以控制时，额外收益有限。
- 双模型训练继续进行仍可能崩溃，不能把“延缓失稳”写成“彻底解决失配”。
- 未验证 32B 等更大模型设置，端到端成本与稳定性结论都受实验规模限制。

与你之前的 token mask 想法相关的启发是：**token 选择可以服务于分布对齐，而不一定服务于信用归因。** 若想用 frozen LLM 判断“哪些 turn／token 对任务贡献更大”，需要与 Jackpot 的概率比接受规则明确区分。

## 7. 文件与来源校验

- PDF 已校验 `%PDF-` 文件头、`pdfinfo` 页数和首页标题；大小为 2,030,337 字节。
- PDF SHA-256、来源版本见 [本篇归档记录](./manifest.jsonl)；校验结果见 [下载校验报告](./download_summary.json)。
- 代码实体在 `/usr/data/jackpot`，论文目录仅保留软链接；commit 和下载方式见 [代码记录](./code_manifest.json)。
- 本次为分类归档和方法导读，尚未完成逐项训练复现。

[返回 Learning Dynamics](../README.md) · [Agentic-RL 总目录](../../README.md) · [逐篇数据集总表](../../07-逐篇论文的数据集与任务分类总表.md)
