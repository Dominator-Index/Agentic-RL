# CRBC：在跨轨迹经验图上求行为策略的 Bellman 不动点

> 根据 [论文 v1](./CRBC.pdf) §3–5、附录 B 的证明和 C 的实验设置整理。官方代码未确认，以下是论文公式分析，不能当作实现核对。

## 1. Motivation：既要跨轨迹传播，也要保留路径发生频率

GiGPO 从相同 state 的样本中比较各自实际后续回报，但某条轨迹在下游共享 state 发现的新成功路径，不会递归传播回其他轨迹的上游。GraphGPO 用图最短路径传播成功可达性，却可能让罕见的短成功路径主导评分。

CRBC 同样合并状态，但求的是**按观察到的动作和转移频率执行时，折扣成功概率是多少**。所以其核心不是更细的分组，而是用经验 Bellman 方程计算 $V$、$Q$ 和 $Q-V$。

## 2. 同题 N 条轨迹如何组成经验过程

保留每步 anchor、解析后的语义动作与后继：$(s_{i,t},a_{i,t},x_{i,t})$。最后一步后继替换成成功吸收边界 $z^+$ 或失败边界 $z^-$；超时也作为失败。

统计重复转移次数 $n(s,a,x)$，定义：

$$
\widehat P(x\mid s,a)=\frac{n(s,a,x)}{n(s,a)},\qquad
\widehat\mu(a\mid s)=\frac{n(s,a)}{n(s)}.
$$

$\widehat P$ 是经验转移，$\widehat\mu$ 是观察到的语义动作频率。未观察转移概率为零，没有借助模型预测新边。重复状态—动作出现保留为计数，不能先去重再假装保留了频率。

它们定义的是**本批轨迹诱导的经验过程**，不是未知真实环境的精确动力学，也不等于 LLM 在所有历史下的真实动作概率。

## 3. Bellman 方程与线性求解

固定 $V(z^+)=1,V(z^-)=0$，折扣 $0<\beta<1$。定义：

$$
Q(s,a)=\beta\sum_x\widehat P(x\mid s,a)V(x),\qquad
V(s)=\sum_a\widehat\mu(a\mid s)Q(s,a).
$$

这不是 max backup：下一步动作也按经验行为频率平均。成功转移折扣一次，故论文的实际 suffix return 写为 $\beta^{T_i-t+1}R_i$，与有些论文的指数差一位属于边界约定差异。

对非终止状态组成矩阵：

$$
M(s,s')=\sum_a\widehat\mu(a\mid s)\widehat P(s'\mid s,a),\qquad
b(s)=\beta\sum_a\widehat\mu(a\mid s)\widehat P(z^+\mid s,a).
$$

于是：

$$
\boxed{(I-\beta M)V=b,\qquad Q(s,a)=\beta\sum_x\widehat P(x\mid s,a)V(x).}
$$

实际直接解线性方程，不显式求逆，也不训练神经 critic。由于 $M$ 行和至多为 1，Bellman 算子是 $\beta$ 收缩映射，固定点唯一；循环状态也可处理。

矩阵求解仍有计算和内存成本。稠密一般求解可达状态数的立方时间量级，不能因为“一次 solve”就称为常数成本。

## 4. 为什么与最短路径不同：一个例子

状态 $U$ 下有 10 次动作记录，其中一种动作出现一次并直接成功，另一种出现九次并失败，$\beta=0.9$。

$$
Q(U,a_{\rm good})=0.9,\quad Q(U,a_{\rm bad})=0,\quad
V(U)=0.1\times0.9+0.9\times0=0.09.
$$

若上游 $S$ 的一个动作总能到 $U$，其值为 $Q(S,a)=0.9\times0.09=0.081$。它考虑到“到达 $U$ 后，经验行为往往仍会失败”；最短成功路指标主要看存在一条短路，不表达这种行为频率。

即使到达 $U$ 的成功 continuation 出现在别的 rollout，它也会参与 $V(U)$ 并传播给 $S$。这正是 cross-rollout closure。

## 5. 从 Q−V 到 PPO 优势

定义经验动作频率加权的局部尺度：

$$
\sigma^2(s)=\sum_a\widehat\mu(a\mid s)[Q(s,a)-V(s)]^2.
$$

至少观察到两个不同语义动作时，才启用局部优势：

$$
A^S(s,a)=\frac{\mathbf1\{|\mathcal A(s)|\ge2\}}
{\max(\sigma(s),\sigma_{\min})}[Q(s,a)-V(s)].
$$

同一个 $(s,a)$ 的所有出现共享这个分数；这不同于 GiGPO 每次出现都使用自己 suffix return。最终：

$$
\boxed{A_{i,t}^{\rm final}=w_G A_i^{\rm trajectory}+w_S A^S(s_{i,t},a_{i,t}).}
$$

保留 trajectory 项，给没有局部动作对照的步骤提供信号。主设置 $\beta=0.98,(w_G,w_S)=(1,5)$。ratio、PPO clip 与基础优化框架保持不变，经验图与优势不反向传播。

若全组完全没有成功转移，则 $b=0$，唯一解是 $V=Q=0$，局部信用也为零。CRBC 不像 ProGPO/FAULT 那样额外引入全失败的替代信号。

## 6. 有限深度 family 是什么意思

初始 $Q^{(0)}(s,a)$ 是该 pair 所有出现的实际 suffix return 均值，$V^{(0)}(s)$ 按经验动作频率加权。反复做 Bellman backup：

$$
V^{(k)}=T^kV^{(0)},\qquad Q^{(k)}=BV^{(k-1)},\qquad A^{(k)}=Q^{(k)}-V^{(k)}.
$$

$k$ 计的是跨 rollout 重组传播次数，不是从零开始只看未来 $k$ 步；初始化已经使用完整实际 suffix。极限收敛到 closure。

$k=0$ 恢复 visit-local 的回报平均思想，但先按相同语义动作求均值，因此不应不加条件称其与 GiGPO 的逐出现 advantage 完全一样。最短路径使用极值而不是期望，不属于这个深度 family。

## 7. 理论保证在哪个世界里成立

论文证明：固定这张经验图，取局部中心化信用 $A=Q-V$，并用非负步长进行可行的表格策略重加权：

$$
\nu(a\mid s)=\widehat\mu(a\mid s)
[1+\alpha w(s)A(s,a)],\qquad w(s)\ge0,
$$

只要 $\nu$ 非负且归一化，就不会降低**固定经验过程**上的折扣成功值。中心化保证总概率不变，改进项与局部 $A$ 的加权平方有关。

这不是实际共享参数 LLM 做一次 PPO 后真实环境性能必增的保证。新策略可能改变未观测动作、历史分布、转移样本及多个状态的概率，均超出固定经验图的表格重加权前提。

尤其 anchor 若忽略历史，会将不同隐藏条件的 continuation 拼接。附录的 WebShop key 是当前页面、Sokoban 是图像，prompt 中最近交互记录不参与 key；ALFWorld key 加了地点、持有物和状态记录等。经验模型自洽不等于这种聚合在真实环境中满足 Markov 性。

## 8. 实验、卡数与比较范围

ALFWorld/WebShop 使用 Qwen2.5-1.5B/7B-Instruct，Sokoban 使用 Qwen2.5-VL-3B。每题 8 条 rollout，文本环境每更新 16 组，Sokoban 32 组；最多步数 50/15/15；训练 150 updates。

论文明确本地实验使用**单机 8 张 A100 80GB**。1.5B ALFWorld 中 CRBC 96.09、表中最强基线 HGPO 90.50，相差 5.59 个百分点；WebShop 成功率 CRBC 81.51、GraphGPO 79.69。

需区分本地重跑与文献引用基线；同文附录描述重跑方法共用协议，但各方法仍有自己的 discount 和组合权重。实际训练的成功奖励还采用 10、失败 0，并加入无效动作惩罚，不能把二元边界理论不经解释套成完整工程奖励。

## 9. 与主线及新 idea 的关系

| 方法 | 从共享状态得到什么 |
|---|---|
| GiGPO | 当前出现的 suffix return 与同状态均值比较 |
| HGPO | 增加 history，改善比较条件一致性 |
| GraphGPO | 经验图中到成功的最短路径评分 |
| CRBC | 经验频率下递归平均的 $Q-V$ |

CRBC 的创新重点是**跨轨迹传播深度与频率一致的聚合**，不是更换 baseline 均值这一处。我们的 history 方案可研究如何让 anchor 更可靠，但细分会减弱跨轨迹连接，应同时测量状态混淆和图覆盖，不能默认 history 越长就越好。

## 10. 对当前分组 idea 的修订

本轮核对 §4 与附录 B 的经验过程保证。CRBC 不只是 baseline 分组：状态合并决定跨 rollout 的图连接，Bellman closure 决定证据传播。

可研究的相关问题是“哪些历史允许共享 continuation”，而不是重新提出 closure。错误合并的经验图也能被精确解到 Bellman residual 为零，因此候选状态抽象必须用留出数据或真实环境结果比较，不能仅以各自图内的自洽为准。

综合讨论见 [新增论文与分组边界](../../05-新论文对分组Idea的修正-ProCredit-SCA-CRBC-FAULT.md)。这条扩展会涉及状态抽象和模型选择，不能自动继承固定 baseline 成本下的树 DP 最优性。未新增训练或代码复现。
