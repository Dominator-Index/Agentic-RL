# 02｜CoSo 如何估计 token 对 action 的影响：反事实公式与代码对照

> 核对日期：2026-10-08。依据 [CoSo 论文](./CoSo.pdf) §4.2–§4.3 和归档的官方 Android 实现。代码 commit：`eb4f205cb8af37f9dedbf124a0ac643050d6fc5b`。本文为静态阅读，未运行训练；数值例子为教学构造。

**核心操作：遮盖一小段 token，保持其他文本不变，比较辅助模型在扰动前后对同一个原始动作标签的预测概率。**

论文以动作 $a$ 表述这个过程；Android 代码实际使用的是**动作类型标签**，不是包含全部坐标、输入文字等参数的完整动作。这个区别会影响对结果的解释。

## 1. 三个组件各自负责什么

| 组件 | 输入与输出 | 作用 |
|---|---|---|
| 主 VLM 策略 $\pi_\theta$ | 任务、截图、历史 → 文本 $y$ | 决定 Agent 如何行动 |
| 动作解析器 $f_{\mathrm{parse}}$ | 文本 $y$ → 结构化动作 $a$ | 把模型输出转换为可执行操作 |
| 辅助模型 $P_\phi$ | 文本 $y$ → 动作标签概率 | 为反事实扰动提供可连续比较的分数 |

辅助模型通过文本及解析动作标签进行监督学习。它不负责生成整个 rollout，也不直接预测任务成功率。

为什么不只检查扰动后解析结果是否改变？离散解析结果可能保持不变，也可能因格式损坏直接报错，难以提供细致的影响分数。论文 §4.2 因此使用辅助模型预测原动作的概率变化，获得更细的敏感性信号。但这个概率是代理模型的判断，不是环境执行后的真实成功概率。

## 2. 第一步：确定原始标签与原始概率

设主模型已经生成一段文本：

$$
y=(y_1,y_2,\ldots,y_L).
$$

$L$ 是 token 数量。解析器先确定原始动作标签，再由辅助模型计算这个标签的概率。

Android 代码中，标签构造是：

```python
android_action = autoui_translate_action(raw_action)
labels.append(android_action.action_type.value)
```

位置：[executor.py 的 parse_action](/usr/data/CoSo/digirl/coso/executor.py:12)。

为避免与完整动作混淆，下面用 $c$ 表示这个动作类型。原始概率为：

$$
p_0=P_\phi(c\mid y).
$$

**$c$ 来自解析器，而非辅助模型自行选择的最大概率类别。** 即使辅助模型把另一类别预测得更高，代码仍然提取原解析标签 $c$ 对应的概率。

## 3. 第二步：构造反事实文本

论文的基本描述是：把第 $i$ 个 token 替换为空值，其他 token 保持原样：

$$
y^{\mathrm{mask}(i)}
=(y_1,\ldots,y_{i-1},y_{\mathrm{null}},y_{i+1},\ldots,y_L).
$$

它提出的反事实问题是：“如果这一位置的信息被移除，辅助模型还会认为文本对应原来的动作吗？”

**实际 Android 代码默认使用长度为 4 的窗口。** 从位置 $i$ 开始，遮盖 $i,i+1,i+2,i+3$；序列末尾按代码的边界截断处理。替换值是 tokenizer 的 `unk_token_id`，原来的 attention mask 保留。

```python
def get_token_effect(..., window_size=4):
    ...
    # 将窗口中的 token ID 替换为 unk_token_id
```

位置：[executor.py:38](/usr/data/CoSo/digirl/coso/executor.py:38) 和[窗口替换部分](/usr/data/CoSo/digirl/coso/executor.py:85)。

因此，代码中存到位置 $i$ 的值，更准确地说是**从该位置开始的一段窗口的扰动影响**。相邻窗口会重叠，不能认为已经独立识别出每个 token 自身的贡献。

此外，这个过程不重新生成被遮盖位置之后的文本，也不重新运行环境。

## 4. 第三步：始终比较同一个标签的概率

对每份扰动后的文本，再计算原标签 $c$ 的概率：

$$
p_i=P_\phi(c\mid y^{\mathrm{mask}(i:i+4)}).
$$

这里 $i:i+4$ 表示最多四个位置的窗口。原始输入与扰动输入比较的是**同一个类别 $c$**，不是各自预测中概率最大的类别。

假设 $p_0=0.90$，可以有如下结果：

| 扰动窗口 | 遮盖后原标签概率 $p_i$ | 有符号差 $p_0-p_i$ | 绝对差 |
|---|---:|---:|---:|
| 一段无关措辞 | 0.88 | 0.02 | 0.02 |
| 一段动作相关文本 | 0.30 | 0.60 | 0.60 |
| 一段干扰动作识别的文本 | 0.95 | -0.05 | 0.05 |

第三行说明：遮盖文本也可能让辅助模型更确信原标签。因此，原始概率差并不天然非负。

论文 Eq. (6) 的简化写法为：

$$
B_i=P_\phi(a\mid y)-P_\phi(a\mid y^{\mathrm{mask}(i)}).
$$

这里沿用论文动作记号 $a$，省略外生噪声。**代码取绝对值**：

$$
e_i=\left|p_0-p_i\right|.
$$

对应语句为：

```python
effect_batch = torch.abs(true_prob_expanded - counterfactual_true_prob)
```

位置：[executor.py:105](/usr/data/CoSo/digirl/coso/executor.py:105)。绝对值保留变化幅度，但丢掉“遮盖使原动作概率上升还是下降”的方向。因此它更适合解释为敏感性权重，不能解释成该 token 对原动作具有多大的正向支持。

## 5. 第四步：将影响值转成训练权重

代码先将一些特殊 token 和指定标点位置的影响设为零，然后在每条序列内部计算最小值和最大值：

$$
w_i=
\sqrt{\frac{e_i-e_{\min}}{e_{\max}-e_{\min}+10^{-10}}}.
$$

这个步骤把影响值转换到约 $[0,1]$ 的范围，平方根则提高较小非零归一化值的相对权重。若整条序列的影响完全相同，分子为零，输出权重为零。

代码最后返回：

```python
return act_tokens, causal_effects.detach()
```

位置：[executor.py:114](/usr/data/CoSo/digirl/coso/executor.py:114)。`detach` 表示在使用这些权重更新主策略时，不通过权重计算过程反向传播；不表示辅助模型在整个训练过程中都被冻结。

由于每条序列单独归一化，某条序列里的 $w_i=1$ 只意味着该位置具有该序列内最大的处理后影响。不能把不同序列的最大权重直接解释为同样大的绝对因果效应。

## 6. 第五步：用权重分配探索，而不是分配正确性奖励

主策略在位置 $i$ 的 token 分布熵为：

$$
H_i=-\sum_{v\in\mathcal V}
\pi_\theta(v\mid s,y_{<i})
\log\pi_\theta(v\mid s,y_{<i}).
$$

$\mathcal V$ 是词表。这个熵越高，说明当前上下文下下一 token 的分布越不集中。它与辅助模型预测动作类型的熵是不同的量。

以 $m_i\in\{0,1\}$ 表示有效 token mask，代码所表达的逐样本加权熵意图可写为：

$$
\widehat H_{\mathrm{CoSo}}
=\frac{\sum_i m_iw_iH_i}{\sum_i m_i}.
$$

这里分母是有效 token 数量，不是权重之和。代码还有 tensor 维度与广播细节，本文公式表达聚合意图，不代表已对所有 batch 形状运行验证。

位置：[autoui_agent.py:151](/usr/data/CoSo/digirl/models/autoui_agent.py:151)。其核心语句是：

```python
causal_entropy = causal_effects * entropy
```

在线训练且开启相关选项时，策略损失为：

```python
total_loss = pg_loss - self.entropy_coeff * causal_entropy.mean()
```

位置：[trainer.py:198](/usr/data/CoSo/digirl/algorithms/causal_digirl/trainer.py:198)。最小化损失时，负号使模型倾向于增大加权熵。

所以，**较大的权重鼓励该位置尝试更多不同输出，不是在奖励当前位置已经选中的 token**。权重为零也不等于该位置的任务策略梯度全部归零。

## 7. 把算法串起来

1. 主 VLM 生成文本 $y$，解析器给出原动作类型 $c$。
2. 辅助模型计算 $p_0=P_\phi(c\mid y)$。
3. 对各位置构造四 token 窗口扰动，保持其余文本不变。
4. 辅助模型计算每份扰动文本上原标签 $c$ 的概率 $p_i$。
5. 计算 $|p_0-p_i|$，屏蔽特定符号并归一化，得到权重 $w_i$。
6. 用 $w_i$ 加权主策略的 token 熵，将熵项加入基础策略损失。

辅助模型前向计算会带来额外开销，但不需要为每个扰动额外调用一次真实环境。代码按位置批量计算扰动输入，以处理多个窗口。

## 8. 它能说明什么，不能说明什么

| 可以据此解释 | 不能直接据此推出 |
|---|---|
| 某段文本对辅助模型识别动作类型很敏感 | 这段文本对任务成功贡献很大 |
| 某个位置适合获得更多探索鼓励 | 当前位置生成的 token 是正确答案 |
| 遮盖某窗口使代理预测明显变化 | 真正执行修改后的动作一定改变回报 |
| 代码比较动作类型标签 | 代码精确归因了所有坐标、输入文本等动作参数 |

保持后续文字不变，会漏掉一种重要路径：早期推理 token 改变后续生成，后续生成再改变动作。辅助模型误差、遮盖输入偏离训练分布、多个 token 的交互作用，也会影响估计。

因此，**CoSo 的反事实分析是动作代理模型上的局部扰动分析，服务于探索分配；它不是对最终任务回报的完整 token 级因果归因。**

## 9. 来源与关联笔记

- [论文页面](https://arxiv.org/abs/2505.03792v2) · [本地 PDF](./CoSo.pdf)。
- [官方代码](https://github.com/langfengQ/CoSo) · [本地代码](./CoSo)。
- [01｜Counterfactual Soft RL 是什么](./01-Counterfactual-Soft-RL是什么-反事实分析与加权探索.md)。
- [CoSo 方法解读](./analysis.md) · [返回目录](./README.md)。
