# 02. Dr. MAS：角色分工、协作流程与 agent 数量

阅读顺序：先看 [01 · 方法与数学公式对比](./analysis.md)，再结合本篇理解角色和采样流程。本篇单独说明 $K$ 的含义，避免把算法公式理解成 $K$ 个模型分别独立完成同一道题。

rollout 预算、共享奖励、Verifier 的完整评审与两个角色的联合更新，详见 [03 · Rollout 预算、共享奖励与联合更新](./03-DrMAS-Rollout预算、共享奖励与联合更新.md)，包含论文与源码核对以及数值例子。

**Dr. MAS 中，$K$ 表示一次协作任务中配置的逻辑 agent 数量。这些 agent 承担不同职责，通过历史上下文和编排流程共同完成一条轨迹。**

论文的数学实验配置两个角色：Solver、Verifier；搜索实验配置三个角色：Verifier、Search、Answer。角色可以共享同一个模型，也可以拥有独立的模型参数。模型参数是否独立，与任务执行是否相互依赖，是两个问题。

本文依据 [论文 §3–4、附录 C.1](./DrMAS.pdf#page=3) 和本目录官方代码编写。核对代码版本：c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9。

## 1. 先区分四个数量：角色、轨迹、调用次数、模型

| 符号 | 表示什么 | 不能混为一谈的数量 |
| --- | --- | --- |
| $K$ | 配置的逻辑 agent / 角色数量 | 一道题采样的答案或轨迹数量 |
| $N$ | 同一道题重复采样的完整协作轨迹数量 | 一次协作中的角色数量 |
| $T_i$ | 第 $i$ 条轨迹中实际发生的 agent 输出次数 | 角色数量；同一角色可以多次出场 |
| $P$ | 独立训练的模型参数组数量 | 角色数量、GPU 数量、进程数量 |

本篇用 $P$ 表示模型参数组数量；它是为解释角色与模型的关系而引入的记号。

以数学实验为例，$K=2$、训练 group size 为 $N=8$，表示：

1. 每次协作由 Solver 和 Verifier 两种角色参与。
2. 对同一道题，重复采样八次完整协作过程。
3. 每次过程可能在审核后结束，也可能继续修改。
4. 八条轨迹通常使用同一套当前模型参数进行采样，不意味着额外创建八套独立训练的团队模型。

**同一道题采样多条轨迹可以并行执行；每条轨迹内部仍遵守角色间的依赖和调用顺序。**

评估指标 avg@16 / pass@16 中的 16 对应每题的测试采样次数，也不表示配置了 16 个 agent。$K$ 描述可用角色，实际调用由流程决定，并不要求每条轨迹都调用所有角色。

## 2. 数学任务：两个角色，负责解答与审核

### 2.1 角色分工

| 角色 | 主要输入 | 职责 | 输出 | 何时执行 |
| --- | --- | --- | --- | --- |
| Solver Agent | 原题、历史解答与团队上下文 | 生成推理和答案；后续轮次结合历史审核继续修改 | 分步解答与最终答案 | 首轮执行；尚未通过审核的样本可再次调用 |
| Verifier Agent | 原题、Solver 的解答与团队上下文 | 检查解答的正确性、完整性和逻辑；决定是否接受 | 完整审核响应，可包含自然语言评审；末尾附 approve / reject 判定 | Solver 输出后执行；当前实现跳过最后一次 Solver 轮次的审核 |

Solver 的职责是构造解答。Verifier 的职责是检查已有解答。它们获得的上下文和产出的内容不同，最终提交的解答来自 Solver。

Verifier 的输出末尾使用以下两种判定标签之一：

~~~text
<verify>approve</verify>
<verify>reject</verify>
~~~

以上只展示末尾控制标签，分别示意两种判定，不表示一次输出需要同时包含两者，也不表示完整响应只有标签。数学 Verifier 的 prompt 要求检查推理步骤、公式和结论；代码设置 return_whole_response=True，保留完整评审响应，并传入后续 Solver 的团队上下文。评审是否详尽取决于实际生成。

来源：[Verifier prompt 与完整响应解析](./DrMAS/agent_system/agent/agents/math/verifier_agent.py)、[数学编排中的上下文更新](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py)。

### 2.2 一条协作轨迹如何产生

概念流程如下：

1. Solver 读取题目，生成初次解答。
2. Solver 的输出写入团队上下文。
3. Verifier 读取题目及这份解答，给出审核。
4. 若通过，该样本不再进入后续修改轮次。
5. 若拒绝且还有轮次预算，Solver 读取包含审核的上下文，再次生成解答。
6. 达到上限时，使用当前 Solver 输出结束这次过程。

官方示例设 max_loop_num=2。按当前数学编排代码，一次编排执行可能出现：

| 情况 | 实际角色调用顺序 | agent 输出次数 |
| --- | --- | --- |
| 首次解答通过 | Solver 初次解答；Verifier 接受 | 2 |
| 首次解答被拒绝 | Solver 初次解答；Verifier 拒绝；Solver 再次解答 | 3 |

**实现细节：最后一次 Solver 输出之后，代码不再调用 Verifier。** 因此，不能机械地认为每条轨迹里两个角色的调用次数一定相同，也不能把“有两个角色”理解成“只有两次模型调用”。

来源：[数学流程说明](./DrMAS/agent_system/agent/orchestra/math/README.md)、[数学编排代码](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py)、[数学训练配置](./DrMAS/examples/drmas_trainer/run_math.sh)。

### 2.3 LLM Verifier 与最终奖励判定的区别

这里的 Verifier 是参与协作、可以被 RL 更新的 LLM 角色；最终任务 reward 则由训练环境的评分规则决定。

例如，Verifier 认为某份解答应当接受，并不保证它真的正确。若最终答案不满足任务评分规则，轨迹仍可能得到失败奖励。

因此，不能将 Verifier 的 approve 判定直接等同于 $R^i=1$。前者控制协作流程，后者用于最终任务评价和训练。

## 3. 搜索任务：三个角色，负责路由、检索和回答

### 3.1 角色分工

| 角色 | 主要输入 | 职责 | 输出 | 调用条件 |
| --- | --- | --- | --- | --- |
| Verifier Agent | 原问题、已有搜索历史、已获取的信息 | 判断现有信息是否足够回答，并选择下一步角色 | yes / no | 每个仍活跃的编排轮次先执行 |
| Search Agent | 原问题、已有信息与团队上下文 | 生成新的检索 query，获取外部证据 | 搜索 query | Verifier 判断信息不足，且仍允许继续搜索 |
| Answer Agent | 原问题、已获得的证据与团队上下文 | 综合证据，生成最终答案 | 最终回答 | Verifier 判断信息足够 |

这里的 Verifier 主要充当 **信息充分性判断器和路由器**，与数学任务中“检查 Solver 解答”的职责有所不同。

规范输出示例：

~~~text
Verifier: <verify>no</verify>
Search:   <search>query</search>

Verifier: <verify>yes</verify>
Answer:   <answer>final answer</answer>
~~~

前两行说明一次需要检索的分支，后两行说明一次可以回答的分支。

### 3.2 一条搜索协作轨迹如何产生

1. Verifier 读取问题与当前信息。
2. 如果信息不足，调用 Search 生成搜索 query。
3. 检索环境执行 query，返回证据，更新历史。
4. 下一个轮次，Verifier 根据新增证据重新判断。
5. 如果信息足够，调用 Answer 组织最终答案。
6. 轮次预算、环境终止条件和无效输出处理，也会影响轨迹是否提前结束。

例如，一条正常完成的轨迹可以包括：

| 次序 | 当前参与者 | 本次工作 |
| --- | --- | --- |
| 1 | Verifier | 判断现有信息不足 |
| 2 | Search | 生成检索 query |
| 3 | 检索环境 / 工具 | 执行检索并返回信息；这一项不是 LLM agent |
| 4 | Verifier | 读取证据，判断信息足够 |
| 5 | Answer | 基于证据生成最终答案 |

这条示例有三个角色，但有四次 LLM 输出：Verifier 出场两次，Search、Answer 各一次。检索工具的调用不计入 $K=3$。

### 3.3 Search 与 Answer 按分支激活

令 $m$ 表示样本当前是否活跃，$v$ 表示 Verifier 是否认为信息充分。代码对每个样本构造：

$$
m_{\mathrm{Search}}=m(1-v),
\qquad
m_{\mathrm{Answer}}=mv,
\qquad v\in\{0,1\}.
$$

因此，在同一个样本的同一个路由轮次，两者的激活条件互斥：

$$
m_{\mathrm{Search}}m_{\mathrm{Answer}}=0.
$$

也就是说，在一次判断之后，要么去搜索，要么去回答。批量执行时，不同样本可以进入不同分支，但这不表示同一道题在该轮由两个角色各自独立完成答案。

官方搜索训练配置为 group_size=5、max_turn=4。这两个值分别控制轨迹采样数与环境轮次；环境轮次也不能直接等同于 LLM 输出次数。

来源：[搜索流程说明](./DrMAS/agent_system/agent/orchestra/search/README.md)、[搜索编排代码](./DrMAS/agent_system/agent/orchestra/search/search_orchestra.py)、[搜索训练配置](./DrMAS/examples/drmas_trainer/run_search.sh)。

## 4. 用数学式说明“角色协作”和“独立答题”的区别

以数学任务的一轮 Solver / Verifier 交互为例，令 $u$ 为 Solver 解答，$v$ 为 Verifier 审核：

$$
p(u,v\mid x)
=\pi_{\theta_{\mathrm{Solver}}}(u\mid x)\,
\pi_{\theta_{\mathrm{Verifier}}}(v\mid x,u).
$$

Verifier 的输出显式依赖 Solver 已生成的解答 $u$。后续 Solver 修改时，还会依赖已有审核及历史。这就是协作过程中的信息依赖。

如果另外设计“两个模型互不读取对方输出，各自提交完整答案”的流程，联合分布才会具有相应的独立采样形式：

$$
p_{\mathrm{independent}}(y_1,y_2\mid x)
=\pi_{\theta_1}(y_1\mid x)\,
\pi_{\theta_2}(y_2\mid x).
$$

后一种式子是用于说明区别的假设方案，不是论文数学实验采用的 Solver–Verifier 编排。参数集合可以独立，但上下文和输出之间仍然可以相互依赖。

完整协作轨迹写成：

$$
\tau^i=\{(s_t^i,a_t^i,k_t^i)\}_{t=1}^{T_i},
\qquad
k_t^i\in\{1,\ldots,K\},
\qquad R^i=R(\tau^i).
$$

这里 $k_t^i$ 指明当前由哪个角色执行，$s_t^i$ 包含题目和已有协作历史，$R^i$ 是这条完整轨迹的任务奖励。

对同一道题重复采样：

$$
\{\tau^1,\ldots,\tau^N\}
\quad\longrightarrow\quad
\{R^1,\ldots,R^N\}.
$$

**GRPO 的同题 group 比较这 $N$ 次完整协作的结果；$K$ 则描述每次协作内部配置了哪些角色。**

## 5. 角色数量与模型参数数量分别配置

令 $w(k)$ 表示角色 $k$ 映射到哪个模型参数组：

$$
\theta_k=\theta_{w(k)}.
$$

### 5.1 模型共享

多个角色可以使用同一个 LLM 参数集合，通过不同角色 prompt 承担不同工作。数学任务中可以配置：

$$
K=2,\qquad P=1,\qquad
\theta_{\mathrm{Solver}}=\theta_{\mathrm{Verifier}}=\theta.
$$

这仍然是两个逻辑角色。它们被分别调用，读取对应上下文，并执行各自的职责。

### 5.2 模型不共享

两个角色也可以分别拥有模型参数集合，此时数学配置可以是 $K=2,P=2$。两个实例即使从相同 checkpoint 初始化，仍可以分别更新。

因此，配置中的两个 model_ids 都写 Qwen3-4B，并不自动表示参数共享；还需要看 model_sharing 以及角色到 worker group 的映射。

| 示例配置 | $K$：角色数 | $P$：参数组数 | 表示什么 |
| --- | --- | --- | --- |
| 数学，全部共享一个模型 | 2 | 1 | 两个角色共享权重，按职责调用 |
| 数学，两个角色不共享 | 2 | 2 | 两套参数参与同一协作流程 |
| 搜索，全部共享一个模型 | 3 | 1 | 三种角色使用一套权重 |
| 搜索，三个角色不共享 | 3 | 3 | 三套参数分别承担路由、检索、回答 |

还可以出现部分共享或异构模型配置。上述 $P$ 统计模型参数组，不统计 GPU、模型分片或推理副本。

## 6. 能否修改角色数量：框架支持，内置流程有明确分工

**可以自定义角色数量与职责，但需要一起定义协作流程。** 论文算法采用一般的 $K$，主要实验则验证了数学的两个角色与搜索的三个角色。

当前内置实现也体现了这一点：

- 数学编排显式调用 Solver Agent 和 Verifier Agent。
- 搜索编排要求 Verifier Agent、Search Agent、Answer Agent 都存在，并根据审核结果选择执行角色。
- 在 agent_ids 列表中追加一个名字，不会自动让现有编排调用该新角色。

扩展到自己的角色组合时，需要：

1. 定义各角色的职责、prompt、输入与输出。
2. 注册对应 agent，并实现输出解析。
3. 定义或修改 Orchestra：调用顺序、分支、共享上下文与结束条件。
4. 配置角色和模型的映射，以及是否共享模型。
5. 保留采样记录中的角色标识，使优势统计与模型更新对应到正确角色。

例如，可为数学任务设计“Planner、Solver、Verifier”三角色流程。这个例子是扩展设想，不是论文数学实验的原始配置；需要实现 Planner 的调用与输出如何进入后续上下文，并单独验证效果。

$K$ 通常在一次训练任务开始前由配置确定；每条轨迹动态变化的是当前激活角色与其出场次数。论文方法没有自动搜索最优角色数量，也没有在每道题中自动新增角色。

来源：[官方多 agent 开发指南](./DrMAS/docs/MULTI_AGENT_DEVELOPMENT_GUIDE.md)。

## 7. 这些分工如何对应到 Dr. MAS 的优势公式

固定一道题 $x$ 的 $N$ 条轨迹，为角色 $k$ 收集其实际出场的输出：

$$
\mathcal{Y}_k(x)=\{(i,t):k_t^i=k\},
\qquad
M_k=\lvert\mathcal{Y}_k(x)\rvert.
$$

如果某条轨迹反复调用 Solver 或 Search，该轨迹奖励就会在对应角色的活跃步骤统计中重复出现。

$$
\mu_k(x)=\frac{1}{M_k}
\sum_{(i,t)\in\mathcal{Y}_k(x)}R^i,
$$

$$
\sigma_k^2(x)=\frac{1}{M_k}
\sum_{(i,t)\in\mathcal{Y}_k(x)}(R^i-\mu_k(x))^2,
$$

$$
A^{i,k}_{\mathrm{DrMAS}}
=\frac{R^i-\mu_k(x)}
{\sigma_k(x)+\varepsilon_{\mathrm{num}}}.
$$

这组公式使用论文的总体方差口径；代码的样本标准差与边界处理见 [方法对比笔记第 8 节](./analysis.md#8-公式对应到官方代码实际实现有哪些细节)。

上式针对 $M_k>0$ 的角色；没有实际出场的角色，不产生该题的活跃输出样本。

需要这样理解 $A^{i,k}$：

- $R^i$ 属于第 $i$ 次完整协作。
- $k$ 指这次协作中的一个角色。
- 只有该角色实际执行的输出，才使用这一角色的优势参与更新。
- 角色参数共享时，多个角色的训练贡献汇入同一个模型；参数不共享时，更新各自对应的模型。

**角色级归一化调整的是各角色接收到的最终奖励尺度，不是把一次任务拆成 $K$ 次各自独立作答。**

## 8. 阅读或介绍论文时，可使用的准确表述

> Dr. MAS 对同一道题采样多条完整的多 agent 协作轨迹。数学流程配置 Solver 与 Verifier 两个角色；搜索流程配置 Verifier、Search 与 Answer 三个角色。角色按编排和历史上下文协作，可以共享模型，也可以使用独立或异构模型。算法按同题、同角色的活跃输出计算奖励统计；角色数量、轨迹采样数量和模型参数组数量分别配置。

这段话中的三个重点是：**角色有分工、轨迹是完整协作、参数独立不等于任务独立。**

## 9. 对照资料

- [01 · 方法与数学公式对比](./analysis.md)。
- [03 · Rollout 预算、共享奖励与联合更新](./03-DrMAS-Rollout预算、共享奖励与联合更新.md)。
- [Dr. MAS 论文](./DrMAS.pdf)，重点 §3.1、§4.3、附录 C.1 与 E。
- [数学角色与流程说明](./DrMAS/agent_system/agent/orchestra/math/README.md)。
- [搜索角色与流程说明](./DrMAS/agent_system/agent/orchestra/search/README.md)。
- [官方自定义 agent 与 Orchestra 开发指南](./DrMAS/docs/MULTI_AGENT_DEVELOPMENT_GUIDE.md)。
