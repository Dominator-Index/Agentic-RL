# GiGPO：跨轨迹重组同状态动作，构造两级优势

> 论文：Group-in-Group Policy Optimization for LLM Agent Training，NeurIPS 2025。依据本目录 `GiGPO.pdf`（arXiv v3），重点核对 §4、§5、附录 C–E；代码核对 `/usr/data/verl-agent`，commit `20bd331bdbc9026a5668e11362178e10ab7400c8`。本文的数值例子是解释性构造，不是论文实验。

## 1. 最值得记住的关键点

**GiGPO 把“同一个任务下的整条轨迹比较”，细化为“同一个任务下、从同一个状态出发的动作出现实例比较”；动作的评分仍然是其自身轨迹的实际后续折扣回报。最终同时使用整轨迹优势与步骤优势。**

你说的“跨 trajectory 找同样 state 作为锚点，然后把 next action 组成 group”基本正确，但应补全四点：

1. 分组通常在 rollout 完成后进行，不是到达每个锚点时再分叉采样。
2. 分组对象是从当前状态 $s_t$ 发出的 $a_t$，不是从后继状态 $s_{t+1}$ 发出的 $a_{t+1}$。
3. 既可以跨 trajectory，也可以包含同一 trajectory 在不同时间再次访问相同状态的实例；不会先按动作文本去重。
4. 主要产生的是 **step-level advantage**。环境即时奖励没有被重新标注；先从已有奖励计算 return-to-go，再在新的组里中心化。

**GiGPO 没有显式建图、反向搜索或最短路径操作。** 可以把同状态分组画成图来帮助理解，但真正把图结构用于重新评分的是 [GraphGPO](../ICML2026-GraphGPO/analysis.md)。

## 2. 它解决的具体问题

在多轮任务中，成功轨迹也可能包含绕路、重复搜索、错误点击；失败轨迹也可能有正确的前半段。GRPO 为整条轨迹给一个标量优势，不能直接区分其中各步的相对效果。

一种直接办法是在每个状态额外生成多个候选动作，并分别继续 rollout 来比较。但这需要额外环境分支和模型生成，成本大。GiGPO 发现：同任务、同初始条件下采出的轨迹本来就有大量重复状态，可以回收这些已经发生的比较机会。

## 3. 统一记号：reward、return 和 advantage

对任务 $x$ 采样 $N$ 条轨迹，记为：

$$
\tau_i=\{(s_{i,t},a_{i,t},r_{i,t})\}_{t=1}^{T_i},\qquad
R_i=\sum_{t=1}^{T_i}r_{i,t},\qquad
G_{i,t}=\sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u}.
$$

$r_{i,t}$ 是环境在该步提供的即时 reward，$R_i$ 是论文所用的整 episode 总回报，$G_{i,t}$ 是从该步往后的折扣 return。$G_{i,t}$ 取决于该轨迹实际走出的后续行为，不能视为已经精确估计的 $Q^\pi(s,a)$。

给任意出现实例集合 $\mathcal B$ 和评分 $z_j$，定义统一中心化操作：

$$
\mathcal N_{\mathcal B}(z_j)=
\frac{z_j-\mu_{\mathcal B}}{F_{\mathcal B}},\qquad
\mu_{\mathcal B}=\frac1{|\mathcal B|}\sum_{k\in\mathcal B}z_k.
$$

$F_{\mathcal B}$ 可以取组内标准差加小常数，也可以取 $1$。GiGPO 论文研究了这两种模式，不能把“除以标准差”写成不可关闭的组成部分。

## 4. 从 GRPO 到 GiGPO，具体改了哪里

### 4.1 整轨迹层：保留宏观信号

$$
A_i^E=\frac{R_i-\operatorname{mean}_{j=1}^N R_j}
{F_E(R_1,\ldots,R_N)}.
$$

它判断轨迹 $i$ 在同任务的 $N$ 条轨迹中整体表现如何。只使用这项时，同一轨迹内所有步骤得到相同的宏观信号；这是与 trajectory-level GRPO 对照的部分。

### 4.2 锚点层：改变比较集合

$$
\mathcal G_x(s)=\{(i,t):s_{i,t}=s,\ \tau_i\text{ 属于任务组 }x\}.
$$

$\mathcal G_x(s)$ 收集同任务组内所有“当前状态等于 $s$”的步骤。时间下标不必相同，轨迹编号也不必不同。一个轨迹多次绕回 $s$ 时，会贡献多个实例，所以这个组的大小可以大于 $N$。

$$
A_{i,t}^S=
\frac{G_{i,t}-\operatorname{mean}_{(j,u)\in\mathcal G_x(s_{i,t})}G_{j,u}}
{F_S(\{G_{j,u}:(j,u)\in\mathcal G_x(s_{i,t})\})}.
$$

这是论文 Eq. (7) 的清晰索引写法：比较的是**同状态出发后各次实际 continuation 的回报**。不是直接把原来的 $A_i^E$ 分割，也不是按相同 next action 或相同后继状态聚类。

### 4.3 联合更新

$$
A_{i,t}^{\mathrm{GiGPO}}=A_i^E+\omega A_{i,t}^S,\qquad \omega\geq0.
$$

$\omega$ 控制局部信号权重；论文主要设置为 $1$。因此，“GiGPO 用 step advantage 替换掉原有 trajectory advantage”不准确：标准方法是两者相加。

每一步生成一段 token 序列 $a_{i,t}=(y_{i,t,1},\ldots,y_{i,t,L})$。实际 token-level PPO 实现可写成：

$$
\rho_{i,t,\ell}(\theta)=
\frac{\pi_\theta(y_{i,t,\ell}\mid h_{i,t},y_{i,t,<\ell})}
{\pi_{\rm old}(y_{i,t,\ell}\mid h_{i,t},y_{i,t,<\ell})},
$$

$$
J(\theta)=\mathbb E\!\left[
\operatorname{Avg}_{i,t,\ell:m_{i,t,\ell}=1}
\min\!\left(\rho_{i,t,\ell}A_{i,t},
\operatorname{clip}(\rho_{i,t,\ell},1-\epsilon,1+\epsilon)A_{i,t}\right)
\right]-\beta\mathcal K.
$$

$h_{i,t}$ 是该步真实输入，$m$ 是 response mask，$\mathcal K$ 是可选参考策略 KL 项；不同训练配置可以关闭 KL。**同一步的 response token 共享该步优势**，但每个 token 的概率比及梯度不同；prompt 和环境 observation token 不作为 response 动作直接计入该损失。GiGPO 还没有在一个动作内部给各 token 分别做因果信用分配。

## 5. 具体例子：不建图时到底算了什么

设只有最后成功时奖励为 $1$，失败为 $0$，$\gamma=0.9$：

```text
轨迹 1：S → B → S → U → G（成功）
轨迹 2：S → U → F        （失败）
```

GiGPO 遍历数据表，找到三次从 $S$ 出发的动作实例：

| 实例 | 当前动作造成的转移 | 自己这条轨迹的后续 return |
|---|---|---:|
| 轨迹 1，第 1 步 | S → B | $0.9^3=0.729$ |
| 轨迹 1，第 3 步 | S → U | $0.9=0.9$ |
| 轨迹 2，第 1 步 | S → U | $0$ |

这三行组成同一个 anchor group。均值为 $0.543$；为了演示，使用总体标准差约 $0.3903$，步骤优势约为 $0.477,0.915,-1.391$。实际 GiGPO 代码使用 `torch.std` 默认的样本标准差，数值尺度会不同，但排序和符号相同。

这个例子同时说明两件事：

- 成功轨迹中较晚的直接有效动作可高于早期绕路动作；折扣因子在这里提供了时间区分。
- **同样的 S → U，因为所在轨迹后续不同，仍会得到不同评分。** 轨迹 2 的正确前半步可能因后面失败而被惩罚，这是 GraphGPO 要继续处理的问题。

如果 $\gamma=1$，且只有终局二值奖励，那么一条成功轨迹内部所有 $G_{i,t}$ 都相同。不同状态的 baseline 仍可能不同，但“同一状态下绕路与较晚动作”的上述时间区分会消失。

## 6. 为什么有效，以及没有做到什么

它把 baseline 从“整个任务”变成“当前状态”，因此在已到达相同决策位置的样本中比较动作更有针对性。它同时保留整体成功导向，又回收重复访问带来的局部比较机会，且无需额外 critic 或每状态新 rollout。

但 $G_{i,t}$ 混合了当前动作与所有后续动作的影响。这里不是对 $a_t$ 单独进行环境干预的因果效果估计。它也不把轨迹 2 的前缀接到轨迹 1 的成功后缀上重新评分：那是图方法才引入的思路。

附录 C 给出不除标准差时的关系：

$$
R_i-\frac1N\sum_jR_j
=\frac{N-1}{N}\left(R_i-\frac1{N-1}\sum_{j\ne i}R_j\right).
$$

右侧是 RLOO advantage 的常数倍。这是整 episode 组上的代数关系；将它推广为“含相关重复访问、变化组大小、任意状态抽象时整个 GiGPO 更新都严格无偏”，超出了该证明。

## 7. 状态究竟是什么：不能只凭相似语义合并

论文把它写为环境状态 $s$；代码实际使用环境构造的 `anchor_obs`。通常是用于匹配的观测/状态表示，不等于包含全部历史的完整 LLM prompt。

这很重要：同一网页文本可能对应不同购物车；同一房间描述可能对应不同持有物品。如果抽象遗漏了决定后续转移的字段，组内比较就会混入不同决策条件。即便物理状态真的相同，带记忆策略看到的历史不同，也可能有不同 continuation 分布，这正是 [HGPO](../ICLR2026-HGPO/analysis.md) 的切入点。

## 8. 论文与本地代码对应

代码入口：[core_gigpo.py](./GiGPO/gigpo/core_gigpo.py)。

| 函数 | 实际作用 |
|---|---|
| `compute_step_discounted_returns` | 按 `traj_uid` 倒序计算 $G_t=r_t+\gamma G_{t+1}$ |
| `build_step_group` | 先按任务 `index` 隔离，再按 `anchor_obs` 建 hashmap 组 |
| `step_norm_reward` | 组内中心化，可选除标准差；单例组步骤优势为 0 |
| `compute_gigpo_outcome_advantage` | 返回 episode advantage 加加权 step advantage |

还需注意三个实现细节：

- 可开启文本相似度分组，但默认是精确 key 匹配；不能默认论文中所有环境都做了语义聚类。
- `episode_norm_reward(..., compute_mean_std_cross_steps=True)` 默认会对展开后的 step 行统计 episode reward，长轨迹会在均值中出现更多次。选择 `False` 才按轨迹 ID 去重。论文的每轨迹等权公式与该默认实现应分开理解。
- `step_rewards` 这个变量在主流程中存的是 discounted return，不意味着环境额外给了逐步过程奖励。

## 9. 实验与证据

论文在 ALFWorld、WebShop 和 search QA 上评估。ALFWorld/WebShop 的主要 group 大小为 $N=8$，GiGPO 与 GRPO 采用相同的常规 rollout 预算；增加的是已有数据上的分组计算。

同一篇论文 Table 1 中，Qwen2.5-1.5B 的 ALFWorld 总成功率从 GRPO 的 72.8 提高到 GiGPO（含 std）的 86.7；WebShop 成功率从 56.8 到 65.0。不含 std 的版本在 WebShop 为 67.4，说明归一化选择是任务相关的。不能把这些数值与 GraphGPO 论文中重新跑出的 GiGPO 基线直接当成同一实验横比。

论文的两级优势消融、训练过程中 anchor group 大小分布，以及计算成本分析，共同支持“回收重复状态比较机会”这个解释。它们并不证明在没有可重复状态的开放式环境中也会同样有效。

## 10. 适用边界与真正贡献

主要边界包括：重复状态少时局部信号稀疏；同组回报相同则该分量为零；重复访问并非独立样本；状态抽象和历史混杂会影响比较；实际失败 continuation 仍可污染好动作评分。

真正贡献是：**在固定 rollout 数据上重新定义比较单位，得到 critic-free 的局部优势，而不是付出树搜索成本来重新采样。** 与 GRPO 的关系是两级 group 与 advantage；与 GraphGPO 的关系是后者进一步更换局部评分来源。

继续阅读：[HGPO](../ICLR2026-HGPO/analysis.md) · [GraphGPO 的完整图操作与同例计算](../ICML2026-GraphGPO/analysis.md) · [横向关键点总览](../../00-方法关键点总览与数学对照.md)。
