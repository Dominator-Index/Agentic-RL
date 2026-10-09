# HGPO：当前状态相同还不够，要按历史上下文分层比较

> 论文：Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks，ICLR 2026。依据 `HGPO.pdf`（arXiv v1），重点核对 §4、§5、附录 A–D；代码：`/usr/data/verl-agent/recipe/hgpo`，commit `20bd331bdbc9026a5668e11362178e10ab7400c8`。

## 1. 一句话抓住方法

**GiGPO 问“这些动作是否从同一个当前状态发出”；HGPO 再问“生成这些动作时，模型看见的历史是否相近”，建立由宽到严的多个嵌套组，再合并各层优势。**

它改变的核心是 baseline/比较集合的条件精度。它仍使用每个步骤所在轨迹的实际后续折扣回报；不建状态转移图、不搜索最短路径，也不从其他轨迹拼出替代 continuation。

| 关键问题 | HGPO 的答案 |
|---|---|
| 采样单位 | 同任务、同初始条件下的 $N$ 条完整轨迹 |
| 分组单位 | 步骤实例；按当前状态及不同长度历史状态后缀匹配 |
| 一个步骤属于几组 | 多组；第 0 层只匹配当前状态，更高层增加历史 |
| 动作评分 | 该实例自身的 Monte Carlo return-to-go |
| 更新粒度 | 每步骤一个合成优势，广播给该步 response token |
| 额外成本 | 多层 key/hash 分组；没有额外 rollout 或 critic |
| 默认保留 episode advantage 吗 | 论文主方法及代码 `base_group=False` 不保留；可选项/消融可加入 |

## 2. 动机：同状态不代表同一个条件策略

假设两条轨迹都来到同一商品页面，但一条历史中已经确认尺寸，另一条历史中刚发现颜色不符。即便当前页面相同，模型输入仍不同。

$$
\pi_\theta(a_t\mid x,s_t,h_t)
\quad\text{中，}\quad
s_t=s'_u\ \not\Rightarrow\ h_t=h'_u.
$$

$h_t$ 表示历史上下文。GiGPO 按当前状态匹配，却可能把不同 prompt 下的动作与 continuation 混起来。这样计算出的相对优势，不一定是同一个决策条件下的公平比较。

这里要区别两个层面：在真正 Markov 且策略不使用历史的设置中，同状态比较很自然；问题出在实际 agent 往往使用有限历史、动作记录或部分观测。HGPO 不意味着所有同状态分组都错误。

## 3. 为什么不只保留完全相同的 prompt

最严格的比较要求“当前状态和全部可见历史都一样”，论文称其为 Oracle steps。但是历史越长，重复的后缀越少，很多组只剩一个样本，优势无法比较或估计噪声很大。

因此面临取舍：粗组样本多，却混合不同历史；细组条件更一致，却样本少。HGPO 同时利用多个尺度。

## 4. 分层分组的数学定义

对步骤 $(i,t)$，令 $K$ 是最多考虑的历史步数，定义当前状态加最近 $k$ 个历史状态的后缀：

$$
C_k(i,t)=(s_{i,\max(1,t-k)},\ldots,s_{i,t}),\qquad 0\leq k\leq K.
$$

这是对论文 Eq. (3) 的统一索引写法；序列开头只能使用已有历史。当前状态是 $C_0(i,t)$，更长的 $C_k$ 增加历史约束。

$$
\mathcal G_k(i,t)=\{(j,u):x_j=x_i,\ C_k(j,u)=C_k(i,t)\}.
$$

这里比较的是相同任务组内的步骤。由于要求逐渐严格，有：

$$
\mathcal G_0(i,t)\supseteq\mathcal G_1(i,t)
\supseteq\cdots\supseteq\mathcal G_K(i,t).
$$

同一动作实例可同时参与多个组，不能把它理解为给每一步分配唯一的“历史类别”。论文公式写的是状态序列；本地代码也按 `anchor_obs` 序列匹配，不能自动等同于包含历史动作、推理文本、步数等全部字段的 prompt 精确相等。

## 5. 每层优势与最终目标

先使用与 GiGPO 相同的实际后续回报：

$$
G_{i,t}=\sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u}.
$$

这里没有借用另一条轨迹的成功后缀。改变的是给该回报选什么 baseline。

每层独立做标准化：

$$
A_{i,t}^{(k)}=
\frac{G_{i,t}-\mu_k(i,t)}{\sigma_k(i,t)+\varepsilon},
\qquad
\mu_k(i,t)=\frac{1}{|\mathcal G_k(i,t)|}
\sum_{(j,u)\in\mathcal G_k(i,t)}G_{j,u}.
$$

$\sigma_k$ 为该层组内回报标准差。同一 $G_{i,t}$ 对不同层的均值比较，可能出现不同符号。实现也支持不除标准差的 `mean_norm`。

论文 Eq. (7) 的聚合为：

$$
w_k=\frac{(k+1)^\alpha}{\sum_{j=0}^K(j+1)^\alpha},
\qquad
A_{i,t}^{H}=\sum_{k=0}^K w_k A_{i,t}^{(k)},\qquad \alpha\geq0.
$$

$\alpha=0$ 表示均匀融合，增大 $\alpha$ 更偏向长历史层。**这里的权重不是学习出来的 critic，也不是直接根据每组方差得到的最优逆方差权重**；主公式是基于层级长度的幂次权重。论文称其 adaptive，应结合“各步骤可用组不同”的实现来理解。

优化仍使用 clipped policy gradient：

$$
J_H(\theta)=\mathbb E\left[
\operatorname{Avg}_{i,t,\ell}
\min\left(\rho_{i,t,\ell}A_{i,t}^{H},
\operatorname{clip}(\rho_{i,t,\ell},1-\epsilon,1+\epsilon)A_{i,t}^{H}\right)
\right]-\beta\mathcal K.
$$

$\rho$ 是生成 token 相对 old policy 的概率比；只在有效 response token 上聚合。不同步骤优势不同，同一步各 response token 共享该优势。主方法没有再固定加一个 $A_i^E$，这点与 GiGPO、GraphGPO 的默认组合不同。

## 6. 一个能看出“为什么要看历史”的例子

下面是解释性构造，四次到达同一当前状态 $U$，但此前观测不同：

| 出现实例 | 最近历史与当前状态 | 实际后续回报 |
|---|---|---:|
| 1 | A → U | 0.9 |
| 2 | A → U | 0.8 |
| 3 | B → U | 0.2 |
| 4 | B → U | 0.1 |

为突出 baseline 的作用，先看不除标准差的版本。第 0 层把四行放一起，均值 $0.5$；第 1 层按 `A→U` 和 `B→U` 分成两组，均值分别为 $0.85$ 和 $0.15$。

$$
A^{(0)}=(0.4,0.3,-0.3,-0.4),\qquad
A^{(1)}=(0.05,-0.05,0.05,-0.05).
$$

例如实例 3：在所有 $U$ 出现实例中较差，但在带 `B` 历史的可比样本中较好。HGPO 同时考虑两种信息。它不会因局部优势转正就保证最终合成优势也转正，结果取决于各层权重。

这也只是展示条件 baseline 的差异，不证明 `A` 与 `B` 必然对应不同难度，更不把历史差异自动当作因果解释。

## 7. 代码核对：主公式之外的细节

核心文件：[core_hgpo.py](./HGPO/recipe/hgpo/core_hgpo.py)，函数 `hgpo_advantage_estimate`。

实际流程是：按任务 `index` 分组 → 按 `traj_index` 恢复各轨迹顺序 → 枚举 `anchor_obs` 后缀 → 构造多层 `clusters` → 每组计算回报优势 → 每步骤聚合 → 广播到 response mask。

必须区分以下实现细节：

1. **单例组跳过。** `nL <= 1` 时不计算该组优势。
2. **聚合时丢弃等于零的优势。** `valid_mask = (advs != 0)` 后对剩余层重新归一化权重。因此可用层及有效权重随步骤变化；真正处在组均值上的样本也会被该判断过滤。
3. **存在长度索引偏移。** 论文第 $k$ 层匹配 $k+1$ 个状态，权重为 $(k+1)^\alpha$；当前代码中的 `k` 已表示后缀含几个状态，聚合却再用 `L=k+1`。不含 base group 时，按论文索引 $k$ 对应的代码权重实际正比于 $(k+2)^\alpha$，而不是 $(k+1)^\alpha$。
4. **并非每步都会创建所有层。** 轨迹开头只枚举足够长的已有后缀，不人为补足缺失历史。
5. **episode 分量可选。** `base_group=False` 是函数及训练脚本的主配置；开启后把轨迹优势作为额外 item 参与加权，而不是简单固定相加。
6. **标准差版本也不同。** 此处使用 `std(unbiased=False)`；GiGPO 的对应代码使用 `torch.std` 默认样本标准差。比较精确数值时要统一约定。

代码头部注释把 HGPO 展开为 “History-based”，但论文正式名称是 **Hierarchy-of-Groups Policy Optimization**；本文按论文命名。

这些是对归档 commit 的静态核对，未运行完整训练，不代表已测得这些实现差异对指标的实际影响。

## 8. 为什么有效，理论又证明到哪一步

HGPO 用较细条件组减轻历史混杂，同时用较粗组维持比较样本量。论文 Proposition 4.1 用偏差—方差取舍解释该设计。

若第 $k$ 层相对目标优势的偏差为 $b_k$、方差为 $v_k$，固定权重时有：

$$
\operatorname{Bias}\left[\sum_kw_kA^{(k)}\right]=\sum_kw_kb_k,
$$

$$
\operatorname{Var}\left[\sum_kw_kA^{(k)}\right]
=\sum_kw_k^2v_k+2\sum_{k<j}w_kw_j\operatorname{Cov}(A^{(k)},A^{(j)}).
$$

附录 B **额外假设层间协方差为零**，才得到论文展示的 $\sum_kw_k^2v_k$。实际层级嵌套、共享样本，因此不能无条件忽略协方差。偏差随历史层数下降、方差升高也属于分析条件，并非所有环境自动满足。

此外，本地实现会根据样本优势是否非零调整有效权重；固定权重下的简洁公式不能直接作为该数据相关实现的完整保证。正确读法是：理论提出设计动机，消融提供经验支持。

## 9. 实验与消融的关键证据

论文在 ALFWorld、WebShop 上评估 Qwen2.5-1.5B/7B，比较不同可见历史长度 $K=2,4$。同一论文 Table 1 的 1.5B、$K=4$ 设置中，WebShop 成功率 GiGPO 为 73.24，HGPO 为 78.12。

Table 2 更直接解释了机制：ALFWorld、$K=4$ 时，0-context 的步骤利用率约 0.98，4-context 约 0.19；WebShop 对应约 0.90 和 0.09。仅靠最长历史组会丢失大量比较信号。

消融发现：去掉历史分层通常退化；只使用 Oracle steps 的训练失败；加入 episode 分量并非稳定收益；$\alpha$ 过大也可能退化。因此方法的核心是取舍，而不是“历史越长越好”。

论文耗时表显示分组增加的是亚秒级成本，相对于每轮约数百秒的共有计算较小。正文的极小百分比表述与表中秒数不完全匹配，本文不照抄该百分比。

## 10. 局限、贡献与方法位置

HGPO 对原始可切分历史最直接；若记忆是模型自由生成的摘要，严格匹配后缀会很困难。状态后缀匹配也不必然确保完整 prompt 一致。它仍受 MC continuation 噪声影响，没有解决失败轨迹中的好动作被后续失败拖累这一根本问题。

它的真正贡献是：**相对优势应在合适的条件分布内比较；比较条件过宽与过窄之间，可以用多层分组折中。**

对照记忆：GiGPO 改成同状态比较；HGPO 细化历史条件；GraphGPO 则利用跨轨迹转移图重新定义动作评分。三者不是简单按发表时间逐级包含。

继续阅读：[GiGPO](../NeurIPS2025-GiGPO/analysis.md) · [GraphGPO](../ICML2026-GraphGPO/analysis.md) · [横向关键点总览](../../00-方法关键点总览与数学对照.md)。
