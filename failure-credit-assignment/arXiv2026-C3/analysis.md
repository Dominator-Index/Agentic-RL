# C3：在相同协作历史下换一条消息，再运行下游来估计贡献

> 依据 [本地 v3 论文](./C3.pdf) §2–6、附录 A–B 与 [本地代码](./C3)。当前标题为 *The Trace Is the State: Exact Credit Assignment for LLM Agent Teams*；旧标题为 *Contextual Counterfactual Credit Assignment for Multi-Agent Reinforcement Learning in LLM Collaboration*。代码 commit：`6c6191ab892ca2b25180eb850454d9612ba3a5c2`。整理日期：2026-10-07。

## 1. 核心 idea 与 motivation

多智能体团队拿到一个终局奖励后，很难知道哪个角色的哪条消息真正起作用。某个 planner 给出合理方案，但下游执行失败；如果所有消息共享终局负信号，就可能连好方案一起惩罚。

C3 认为，在消息构成可恢复上下文的系统中，可以直接进行局部实验：**固定发言前的历史，只替换当前 agent 的消息，让后面的角色真的接着执行，然后比较终局结果。**

这里的价值来源是实际续跑，不是让 LLM 给候选消息直接打一个主观分数，也不是训练一个神经网络 critic 来猜价值。

## 2. Setup：它确实研究多智能体，并与 MAGRPO 比较

论文主要训练流程是 Reasoner → Actor：前者生成计划，后者生成最终解答。另有加入 Verifier 的三角色流程，以及 2–10 个决策点的归因诊断。

记 $u$ 为一次具体发言的决策点，$h_u$ 为发言者看到的上下文，$\tau_{<u}$ 为系统记录的此前完整轨迹，$a$ 为该次发言的整条消息。一个角色可以多次出场，因此“决策点数”不必总等于“独立模型数”。

论文比较中的 MAGRPO 将轨迹分数提供给各决策点；C3 则给不同消息单独估计优势。该比较采用论文自己的共同训练流程，不应把它当作对原始 MAGRPO 所有奖励设置的定义。

## 3. 算法关键步骤

1. **冻结本轮采样快照**：$\pi_b\leftarrow\pi_\theta$，本轮候选与下游续跑都用它。
2. **记录决策点**：执行基础轨迹，记录角色、上下文、此前消息及恢复所需信息。
3. **同历史分支**：在同一个 $h_u$ 下采样 $n\ge2$ 条候选消息 $a_1,\ldots,a_n$。
4. **下游续跑**：分别把候选放到同一个轨迹前缀之后，每条候选接着运行 $c_j$ 次至终局。
5. **估计候选价值**：用该候选下的终局奖励平均值估计 $Q(h_u,a_j)$。
6. **同组 leave-one-out 比较**：减去其他候选的回报基线，得到这条消息的 advantage。
7. **策略更新**：将消息及其局部优势送入 clipped policy objective。下一轮再更新采样快照。

只有当前决策点之前的历史固定；换了消息之后，下游 agent 的上下文和行为当然可以变化。这个变化正是要衡量的协作影响。

## 4. 关键公式

### 4.1 要估计的量

$$
Q^{D_b}(h_u,a)=\mathbb E[R\mid h_u,a;\text{下游按 }\pi_b\text{ 执行}],
\qquad
V^{D_b}(h_u)=\mathbb E_{a\sim\pi_b(\cdot\mid h_u)}Q^{D_b}(h_u,a).
$$

$D_b$ 表示固定行为策略所诱导的下游执行分布。$Q$ 包含这条消息对后续角色的影响，不是只评价这段文字本身是否正确。

$$
A^{*,D_b}(h_u,a)=Q^{D_b}(h_u,a)-V^{D_b}(h_u).
$$

它回答“这条消息比该历史下通常会产生的消息好多少”，而不是“把整个 agent 删除后会怎样”。

### 4.2 Monte Carlo 候选价值

$$
\widehat Q_j=\frac1{c_j}\sum_{m=1}^{c_j}R_{j,m}.
$$

$R_{j,m}$ 是注入候选 $a_j$ 后第 $m$ 次续跑的终局奖励。平均多个续跑可减少下游随机性；$c_j=1$ 时只有一次样本，噪声通常更大。

### 4.3 排除自身的比较基线

$$
C=\sum_{k=1}^{n}c_k,\qquad
b_{-j}=\frac{\sum_{k\ne j}c_k\widehat Q_k}{C-c_j},
\qquad \widehat A_j=\widehat Q_j-b_{-j}.
$$

这是论文 Eq. (4)。候选不能把自己的回报放进评价自己的 baseline；其他候选按续跑次数加权。如果次数相等：

$$
\widehat A_j=\widehat Q_j-\frac1{n-1}\sum_{k\ne j}\widehat Q_k.
$$

它没有要求再除组内标准差。数学上的 group 是**同一个精确历史下的候选消息**，不是任意具有相同角色名的输出。

### 4.4 策略目标

$$
J(\theta)=\mathbb E\left[\min\left(
\rho_j(\theta)\widehat A_j,
\operatorname{clip}(\rho_j(\theta),1-\varepsilon,1+\varepsilon)\widehat A_j
\right)\right],\qquad
\rho_j(\theta)=\frac{\pi_\theta(a_j\mid h_u)}{\pi_b(a_j\mid h_u)}.
$$

这里沿用主文消息级示意，实际 LLM 更新按 token 展开，并按配置处理 KL。信用粒度是 message/turn：同一消息里的 token 可以共享优势，不代表已经推断哪个词具有独立因果贡献。

## 5. 关键例子：同样八个终局结果，采样结构不同

论文 §5 使用 $B=8$ 个终局结果预算。MAGRPO 基线采样八条独立 Reasoner → Actor 轨迹；C3 使用 $2\times4$ 的共享前缀树：

```text
同一道题
  ├─ Reasoner 消息 a1
  │    ├─ Actor 回答 b11 → R11
  │    ├─ Actor 回答 b12 → R12
  │    ├─ Actor 回答 b13 → R13
  │    └─ Actor 回答 b14 → R14
  └─ Reasoner 消息 a2
       ├─ Actor 回答 b21 → R21
       ├─ Actor 回答 b22 → R22
       ├─ Actor 回答 b23 → R23
       └─ Actor 回答 b24 → R24
```

对于 Reasoner，两个候选的 $Q$ 分别是各自四个终局结果的均值。对于 Actor，每个 Reasoner 前缀下的四条回答组成自己的局部 group，末端消息的 $Q$ 就是终局奖励。

假设结果如下：

| Reasoner 消息 | 四次 Actor 奖励 | 消息价值估计 | Reasoner 优势 |
|---|---|---:|---:|
| $a_1$ | 1, 1, 1, 0 | 0.75 | $0.75-0.25=0.5$ |
| $a_2$ | 1, 0, 0, 0 | 0.25 | $0.25-0.75=-0.5$ |

在 $a_1$ 分支下，前三个 Actor 回答各得 $1-2/3=1/3$，失败回答得 $0-1=-1$。同一条成功路径上的 Reasoner 与 Actor，因此可以得到不同优势。

**八个叶子结果不等于生成八条独立完整前缀。** C3 生成 2 条 Reasoner 消息加 8 条 Actor 消息，共 10 条；八条独立团队轨迹要生成 16 条消息。共享前缀节省生成，不代表在任意任务上都固定节省同样 token 比例，因为消息长度会变化。

这也修正一个容易产生的误解：C3 需要反事实续跑，但不一定是“先花完八条独立 rollout，再额外加一套预算”。论文训练可把同样的终局预算重排成树。

## 6. 为什么必须强调完整 history

论文的 trace-is-state 条件是：给定记录的前缀并干预消息，应该可以恢复真实系统从该状态继续执行的分布。

$$
\mathcal L(Y_{>u},R\mid\operatorname{do}(a_u=a),\tau_{<u})
=
\mathcal L(Y_{>u},R\mid\operatorname{do}(a_u=a),S_u).
$$

$S_u$ 是真实完整状态，$Y_{>u}$ 是后续消息。该式要求工具输出、持久记忆、路由决策等关键因素要么可恢复，要么按正确条件分布重采样。只保存一张相同网页或一个 state 字符串不必满足它。

因此，与跨轨迹“碰巧找到同一 observation”不同，C3 主动从同一个可恢复前缀分支，使候选可比。它与我们之前讨论的 history 可比性直接相关。

## 7. 理论保证与边界

在前缀充分、候选独立采样、不同候选的续跑独立、续跑次数不依赖采样动作或结果等条件下，论文证明：

$$
\mathbb E[\widehat A_j\mid h_u,a_j]
=Q^{D_b}(h_u,a_j)-V^{D_b}(h_u).
$$

理由是候选价值均值无偏，且排除自身的其他候选基线期望为 $V$。这是局部优势估计的性质，不自动意味着 clipping、标准化、有限步优化之后的完整训练梯度仍无偏。

当每条候选有相同回报噪声方差 $\sigma^2$，给定已采样候选：

$$
\operatorname{Var}(\widehat A_j\mid a_1,\ldots,a_n)
=\frac{\sigma^2}{c_j}+\frac{\sigma^2}{C-c_j}.
$$

等分局部预算 $B_u=nc$ 时，上式为 $\sigma^2n^2/[B_u(n-1)]$。它没有显式 agent 数项，但当总预算分给更多决策点时，每点预算变少，估计会更不可靠；不能推断“增加 agent 不影响成本或质量”。

如果所有续跑奖励都为 0，所有 $\widehat Q_j$ 和 $\widehat A_j$ 仍为 0。C3 不能从没有差异的结果中凭空发现“虽然失败但这条消息是好的”。

## 8. 实验、比较和贡献

论文五训练种子、两角色、512-token 设置中，MATH500 的 MAGRPO 为 $74.5\pm0.4$，C3 为 $82.8\pm0.6$。2–10 决策点的实验主要用于估计器可靠性诊断，不应表述为全部完成了同等规模的端到端多智能体训练。

主文还比较 learned critic 与高续跑数参考的排序一致性、不同预算下估计噪声，以及删除 agent 消息导致的分布偏移。删除消息衡量“角色存在的价值”，不等于同历史下特定消息相对其他正常消息的质量。

真正贡献是把 LLM 团队可恢复的上下文用作可执行反事实实验，并把局部价值差接到策略更新；并非单独发明 leave-one-out 基线。

## 9. 代码核对

- [统一训练入口](./C3/scripts/40_train/paper_train.sh) 与 [论文配置](./C3/c3/utils/paper_train_contract.py)：论文训练选择 `reward_only`、`loo`；仓库还有其他变体，不都属于主实验。
- [树 rollout](./C3/c3/protocol/rollout_generator.py)：每个消息节点使用其子树叶子回报均值作为 `node.reward`，与上面的候选价值对应。
- [优势计算](./C3/c3/credit/provider.py)：`_baseline_loo` 为其他节点值的均值；`reward_only` 执行 `adv = r - baseline_fn(r)`，单元素组为零。
- 论文一般公式支持不等续跑次数的加权 baseline；所核对的 provider 为不加权 LOO，直接对应每组候选等续跑预算的情形。若自行引入不等分配，不能原样沿用并声称实现了论文加权式。

本次只检查关键路径，未安装依赖或执行分布式训练。

## 10. 与我们 idea 的区别

我们的设想是：在既有八条团队轨迹全部失败时，用冻结 LLM 诊断原消息；C3 是从相同前缀主动分支，并用真实终局结果估计贡献。两者的区别是**语义诊断与实际干预**，也包括如何组织预算。

可以把 C3 的高预算反事实结果作为小规模归因质量参照，检验 assessor 的判断是否与真实下游收益一致。但这需要额外实验；冻结 LLM 打分不能直接称为 C3 的无偏低成本替代。
