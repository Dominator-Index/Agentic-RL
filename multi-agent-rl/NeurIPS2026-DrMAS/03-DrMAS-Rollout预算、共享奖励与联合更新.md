# 03. Dr. MAS：Rollout 预算、共享奖励与 Solver–Verifier 联合更新

阅读顺序：[01 · 方法与数学公式对比](./analysis.md) → [02 · 角色分工与协作流程](./02-DrMAS-角色分工与协作流程.md) → 本篇。本文逐项回答：采样数量是否相同、奖励是否共享、Verifier 输出什么、两个角色是否使用不同模型、谁会被更新，以及 Dr. MAS 的归一化究竟改变了什么。

依据：本地 [DrMAS.pdf](./DrMAS.pdf) 的 §3.1–3.2、§4.2、§5、附录 C 与 E；官方仓库代码版本 c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9。核对日期：2026-10-05。本文是论文与源码阅读核对，没有启动模型训练或复现论文指标。

**按照同一道题、同一组 rollout 的数学比较，你的理解正确：GRPO 对所有角色共用奖励均值与方差；Dr. MAS 根据每个角色实际出场的输出，分别计算奖励统计。两者使用同一份最终任务奖励 $R^i$，改变的是把奖励转换成优势的方式。**

对应你最后概括的理解，可以准确表述为：

> 普通多 agent GRPO 对同一道题采样的完整协作轨迹计算一套公共奖励均值与方差，所有 subagent 共用。Dr. MAS 为每个角色 $k$ 收集实际出场的输出集合 $\mathcal{Y}_k$，以 $M_k=\lvert\mathcal{Y}_k\rvert$ 个输出对应的任务奖励计算角色自己的均值与方差；同一条轨迹的任务奖励会按该角色的出场次数重复计入。因为各角色的实际出场情况和统计集合不同，同一个任务 reward 可以被转换为不同角色的优势。两者每题的 rollout 数量在论文设置中对齐；在同一组 rollout 的数学比较中，$R^i$ 是同一个共享任务奖励，并没有改成各角色各自的贡献奖励。

这里的“均值与方差”指**奖励归一化用的统计量**，不是模型参数本身的均值与方差；公式分母使用的是标准差，即方差的平方根。“按角色不同”指分别收集样本、分别计算统计。

## 1. 对你理解的逐项判断与问题速查

### 1.1 对你刚才概括的逐句评判

**按照数学公式和同一组 rollout 来理解，你的概括是正确的。** 逐项判断如下：

| 你的概括 | 判断 | 澄清 |
| --- | --- | --- |
| “Dr. MAS 改变的是对于每个 subagent 更新的均值和方差” | 正确 | 更精确地说，是每个角色计算优势时使用的奖励均值 $\mu_k$ 和方差 $\sigma_k^2$；分母实际使用标准差 $\sigma_k$。这些统计随后影响策略更新。 |
| “GRPO 对于所有 subagent 的均值和方差是完全一样的” | 正确，限于同一道题的采样 group | 论文中的普通多 agent GRPO 对所有角色共用该题的 $\mu,\sigma$；不同题目仍有不同 group 统计。 |
| “但是 Dr. MAS 是不一样的” | 正确，理解为按角色分别统计 | 每个角色各自计算 $\mu_k,\sigma_k$，不再统一使用该题的公共 $\mu,\sigma$。 |
| “根据角色 $k$ 收集其实际出场的输出，$\mathcal{Y}_k$、$M_k$ 可能不同” | 正确 | 一个角色出场一次与另一个出场多次，会产生不同输出记录和样本数。一次出场是一份完整响应，不是一个 token。 |
| “agent i 出场少、agent j 出场多” | 正确，是统计集合不同的来源之一 | 少出场的角色收集较少响应，多出场的角色收集较多响应；奖励按实际出场次数进入各自的统计集合。 |
| “每题 rollout 次数和 multi-agent GRPO 一样” | 在论文的对照设置与默认脚本中正确 | 数学训练两者为 8，搜索为 5；指完整协作轨迹条数。 |
| “所以 $R^i$ 一样” | 在你指的数学比较中正确 | 同一组 rollout 的最终任务奖励作为共同输入；Dr. MAS 不修改 $R^i$ 的定义，只改变计算优势的统计方式。 |

**你的主线可以保留：共享 $R^i$ → 按角色收集 $\mathcal{Y}_k$ → 计算 $\mu_k,\sigma_k$ → 得到各角色的优势 → 更新对应模型。** GRPO 与 Dr. MAS 的区别就在这条链路中统计量的计算范围。

第 5.2 节给出对应公式；第 5.3 节给出实际调用路径中的数值例子；第 5.4 节明确两种算法的共同数学输入。

### 1.2 前面六个问题的核对结果

| 问题 | 核对结果 |
| --- | --- |
| single-agent 和 multi-agent 每题 rollout 数一样吗？ | 按论文报告的统一设置，数学训练 group size 为 8，搜索为 5，正式评估采样 16 次。源码直接确认多 agent GRPO 与 Dr. MAS 的默认采样数一致；每次采样是完整协作轨迹。相同轨迹条数不能推出相同调用次数或 token 预算。 |
| 同一轨迹里 Solver 和 Verifier 最终 reward 一样吗？ | 是，共享最终任务 reward。官方实现另外可扣除各自输出的无效格式惩罚，因此送进优势函数的实际 score 不一定完全一致。 |
| Solver 和 Verifier 是两个不同模型吗？ | 两个角色。参数可共享，也可独立。默认数学脚本使用两份从同一个 Qwen3-4B checkpoint 初始化、分别训练的参数。 |
| Verifier 只输出 reject / accept 吗？ | 源码标签是 approve / reject。它被要求检查推理，可生成完整评审文本，末尾附标签；代码保留完整响应。标签用于决定是否继续修改。 |
| Solver 和 Verifier 都会被更新吗？ | 会。两个角色实际生成的响应都被保存，计算优势后交给各自模型更新；共享参数时，它们的数据共同更新同一模型。 |
| Dr. MAS 的主要算法区别是什么？ | 普通 GRPO 对同题的完整轨迹奖励计算统一均值、标准差；Dr. MAS 对同题、同角色的实际输出计算均值、标准差。同一轨迹中某角色出现多次，其奖励在该角色统计中出现多次。 |

下面把这些答案对应到实际流程和公式。

## 2. rollout 数量相同，到底对齐了什么？

### 2.1 论文报告的采样设置

论文附录 C.2 说明数学任务各方法采用统一超参数，训练 group size 为 8；C.3 对搜索任务作同样说明，训练 group size 为 5。§5 与结果表使用 avg@16、pass@16。

| 设置 | 数学任务 | 搜索任务 | 统计单位 |
| --- | --- | --- | --- |
| 每题训练 group size | 8 | 5 | 完整 rollout / episode |
| 正式 avg@16、pass@16 评估 | 16 | 16 | 每题测试 rollout |

**single-agent 与 multi-agent 的“每题采样数相同”，应当理解为按论文报告对齐完整 rollout 条数。** 不应解释成它们生成同样数量的文本、调用同样次数的模型，或使用同样数量的训练参数。

证据：[论文附录 C.2–C.3](./DrMAS.pdf#page=17)、[数学结果表](./DrMAS.pdf#page=7)、[搜索结果表](./DrMAS.pdf#page=8)。

### 2.2 源码直接确认了哪些设置？

当前仓库的两个数学脚本均配置：

~~~bash
algorithm=grpo
group_size=8
agent_ids='["Solver Agent","Verifier Agent"]'
model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B"]'
model_sharing=False
max_loop_num=2
~~~

Dr. MAS 脚本另设：

~~~bash
group_by_agent_id=True
~~~

并将其传入 algorithm.group_by_agent_id。GRPO 脚本未覆盖该项，默认配置为 False。

**examples/grpo_trainer/run_math.sh 的默认配置本身也是多 agent GRPO baseline，不能把它当成 single-agent baseline 脚本。** 因此，本次源码直接核对的是“多 agent GRPO vs. Dr. MAS”的设置；single-agent 每题采样数的依据是论文报告的统一配置，不是这两份脚本对 single-agent 的独立复现。

两份脚本都通过 env.rollout.n 传入 group_size。rollout_loop.py 在运行协作流程前，将每道题重复 effective_rollout_n 次；这些副本共享同题 uid，每个副本又有自己的 traj_uid。所以这里的 8 是八次完整协作，而非每个角色独立提交八个最终答案。

来源：[GRPO 数学脚本](./DrMAS/examples/grpo_trainer/run_math.sh)、[Dr. MAS 数学脚本](./DrMAS/examples/drmas_trainer/run_math.sh)、[默认算法配置](./DrMAS/verl/trainer/config/ppo_trainer.yaml)、[采样循环](./DrMAS/agent_system/multi_turn_rollout/rollout_loop.py)，对应 L319–327、L426–437。

### 2.3 八条协作轨迹需要多少次模型调用？

以当前数学脚本 max_loop_num=2 为例，设八条轨迹中有 $d$ 条被第一次审核拒绝。正常调用路径是：

~~~text
通过：Solver 初次解答 → Verifier approve → 提交 Solver 答案
拒绝：Solver 初次解答 → Verifier reject → Solver 修改 → 提交修改答案
~~~

于是：

$$
C_{\mathrm{Solver}}=8+d,\qquad
C_{\mathrm{Verifier}}=8,\qquad
C_{\mathrm{total}}=16+d,\qquad 0\le d\le 8.
$$

这是按每个样本计数的 16–24 次 LLM 生成；批量 API 调用的次数还取决于如何合批。若另一个 single-agent 数学流程每条 rollout 只生成一次解答，它的八条 rollout 就是八次样本级生成。这个比较说明为什么“相同 $N$”不能作为“相同计算量”的证据。

当前实现最后一次 Solver 输出后直接结束，没有再次调用 Verifier；所以两轮预算也不能机械地记为每条轨迹四次调用。

正式 eval 模式两份脚本设置 val_group_size=16；训练过程中的快速验证默认为 1。不能混淆正式 avg@16 评估与训练中的快速验证。

来源：[数学编排 MathMultiAgentOrchestra.run](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py)，L76–122；[数学训练脚本](./DrMAS/examples/drmas_trainer/run_math.sh)，L3–19。

## 3. Verifier 实际输出什么？评审意见会传给 Solver 吗？

### 3.1 prompt 要求检查推理，判定标签放在末尾

数学 Verifier 的 prompt 要求它审查最新的 Solver 解答，检查每个推理步骤、公式和结论的正确性、完整性与逻辑一致性，并在输出末尾写：

~~~text
<verify>approve</verify>
~~~

或者：

~~~text
<verify>reject</verify>
~~~

**这两个标签只是末尾的流程判定。prompt 没有要求整个响应只能包含一个标签，也没有保证每次都会生成同样详尽的评审。**

例如，假设题目是求方程 $x^2=9$ 的全部实数解，Solver 只写了 $x=3$。Verifier 可以输出：

~~~text
目前的解答遗漏了负根。由 x²=9 得到的实数解包括 3 和 -3，
不能只保留正根。请补充另一个解并检查题目要求的解集。
<verify>reject</verify>
~~~

以上是说明格式的人工示例，并非从真实 rollout 日志摘录。

### 3.2 代码保留完整响应，再写入团队上下文

verifier_agent.py 调用输出解析函数时明确使用：

~~~python
general_projection(
    text_repsonses,
    start_tag=self.start_tag,
    end_tag=self.end_tag,
    check_think_tag=False,
    return_whole_response=True,
)
~~~

return_whole_response=True 表示保留完整响应。数学编排随后把这份响应追加到 team_context；若需要继续修改，Solver 的 prompt 会读到它。approve/reject 则被单独用于更新 approved_vector。

因此信息流是：

~~~text
Solver 完整解答
    → Verifier 检查并生成评审与判定
    → 完整评审写入团队上下文
    → 判定决定是否再次调用 Solver
    → Solver 根据已有解答与评审继续生成
~~~

来源：[Verifier prompt 与 call](./DrMAS/agent_system/agent/agents/math/verifier_agent.py)，L26–65；[写入上下文与审核分支](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py)，L28–36、L102–120；[Solver prompt](./DrMAS/agent_system/agent/agents/math/solver_agent.py)，L25–37。

前一篇笔记只展示了这两个标签，没有展示标签之前的完整评审文本，容易造成“Verifier 只输出二分类”的误解；现在已同步修正。

## 4. 最终奖励是不是共享？源码如何把它传给两个角色？

### 4.1 最终评分对象是 Solver 提交的答案

数学编排中的 text_actions 只在 Solver 输出后更新。Verifier 的评审写入团队上下文，不替换最终提交答案。

MathEnv.step 随后将提交的答案与 ground_truth 交给数学奖励函数评分，并结束 episode。论文实验的任务奖励是成功 1、失败 0：

$$
R^i=R(\tau^i)\in\{0,1\}.
$$

这里 $i$ 是完整协作轨迹编号。**Verifier 的 approve 不等于奖励为 1；它不是训练环境中的标准答案裁判。** Verifier 可能错误地接受解答，也可能拒绝后让 Solver 修改成功。

来源：[最终答案更新](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py)、[MathEnv.step](./DrMAS/agent_system/environments/env_package/math/envs.py)，L24–34；[论文 §3.1–3.2](./DrMAS.pdf#page=3)。

### 4.2 同一 episode_rewards 写到轨迹中的每个响应

采样收集函数 gather_rollout_data 对同一条轨迹里的所有有效输出执行：

~~~python
data['episode_rewards'] = episode_rewards[bs]
~~~

EpisodeRewardManager 再将这份奖励写到每个响应的最后一个有效 token 位置；main_ppo.py 初始化该 manager 时设置 normalize_by_length=False。

这意味着：如果一条轨迹执行了“Solver 初稿 → Verifier 拒绝 → Solver 修改”，最终任务奖励为 1，那么三份输出都先获得任务 reward 1：

| 响应 | 共享的最终任务 reward |
| --- | --- |
| Solver 初稿 | 1 |
| Verifier 评审与 reject | 1 |
| Solver 修改稿 | 1 |

**最终 reward 不会单独判定初稿是否正确，也不会单独判定评审是否正确。** 即便初稿有错、修改后成功，初稿仍得到这条成功轨迹的最终任务奖励。这正是共享团队奖励带来的信用分配局限。

来源：[gather_rollout_data](./DrMAS/agent_system/multi_turn_rollout/rollout_loop.py)，L72–90；[EpisodeRewardManager](./DrMAS/agent_system/reward_manager/episode.py)，L77–84；[manager 初始化](./DrMAS/verl/trainer/main_ppo.py)，L146–156。

### 4.3 实际训练 score 还可能包含逐输出的格式惩罚

默认数学脚本启用了无效动作惩罚，系数为 0.1。apply_invalid_action_penalty 按每个响应自己的 is_action_valid 扣分。忽略 KL 奖励时，可以把实际用于优势的标量写为：

$$
r_{i,t}=R^i-\lambda_{\mathrm{invalid}}(1-v_{i,t}),
\qquad v_{i,t}\in\{0,1\}.
$$

其中 $v_{i,t}=1$ 表示该输出格式有效；有效时 $r_{i,t}=R^i$，无效时扣除 $\lambda_{\mathrm{invalid}}$。这里的 $r_{i,t}$ 是本篇为区分实际 score 与共享任务 reward 而引入的记号。

所以“两个角色奖励一样”有两个层次：

- **最终任务 reward $R^i$：相同。**
- **进入优势计算的响应 score $r_{i,t}$：可能因各自的格式有效性而不同。**

默认配置 use_kl_in_reward=False，数学脚本也关闭 use_kl_loss。下面解释论文的算法核心时，先使用没有额外惩罚的共享 $R^i$，避免把格式奖励误当成 Dr. MAS 的核心贡献。

来源：[惩罚函数](./DrMAS/verl/trainer/ppo/ray_trainer.py)，L203–220；[奖励与优势计算顺序](./DrMAS/verl/trainer/ppo/ray_trainer.py)，L1338–1365；[数学惩罚配置](./DrMAS/examples/drmas_trainer/run_math.sh)，L78–80。

## 5. 同一个 reward，为什么还可能有不同的角色统计？

### 5.1 普通多 agent GRPO：每条完整轨迹计一次

固定一道题，采样 $N$ 条协作轨迹，得到最终奖励 $R^1,\ldots,R^N$。按论文公式：

$$
\mu=\frac{1}{N}\sum_{i=1}^{N}R^i,
\qquad
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(R^i-\mu)^2.
$$

每条轨迹的共享优势为：

$$
A_{\mathrm{GRPO}}^i
=\frac{R^i-\mu}{\sigma+\varepsilon_{\mathrm{num}}}.
$$

同一条轨迹里的 Solver、Verifier，以及重复出现的 Solver 输出，都使用同一个 $A_{\mathrm{GRPO}}^i$。

### 5.2 Dr. MAS：某角色出现几次，就把该奖励计几次

角色 $k$ 的实际输出集合为：

$$
\mathcal{Y}_k(x)=\{(i,t,a_t^i):k_t^i=k\},
\qquad M_k=\lvert\mathcal{Y}_k(x)\rvert.
$$

这个集合按每次出场的记录计数；相同文本的两次生成仍是两个样本，不会因字符串相同而去重。为避免普通集合按值去重的歧义，本篇显式保留索引 $(i,t)$；论文写作 $\mathcal{Y}_k=\{a_t^i:k_t^i=k\}$。

令 $n_{i,k}$ 表示角色 $k$ 在第 $i$ 条轨迹中实际生成响应的次数：

$$
n_{i,k}=\sum_{t=1}^{T_i}\mathbf{1}\{k_t^i=k\},
\qquad M_k=\sum_{i=1}^{N}n_{i,k}.
$$

则论文的活跃步骤统计可以等价写成：

$$
\mu_k=\frac{\sum_{i=1}^{N}n_{i,k}R^i}{M_k},
\qquad
\sigma_k^2=
\frac{\sum_{i=1}^{N}n_{i,k}(R^i-\mu_k)^2}{M_k}.
$$

角色 $k$ 在轨迹 $i$ 中的每次输出，使用：

$$
A_{\mathrm{DrMAS}}^{i,k}
=\frac{R^i-\mu_k}{\sigma_k+\varepsilon_{\mathrm{num}}}.
$$

**分子里的任务奖励仍是同一个 $R^i$。角色差别来自整个 group 的统计权重 $n_{i,k}$。** 某角色没有出场，对应权重为 0；出现两次，该奖励在它的统计中出现两次。

也可以直接比较两种统计的轨迹权重：

$$
p_{\mathrm{GRPO}}(i\mid x)=\frac1N,
\qquad
p_{\mathrm{DrMAS},k}(i\mid x)=\frac{n_{i,k}}{M_k}.
$$

普通 GRPO 对每条完整轨迹等权；Dr. MAS 对某角色的每次实际输出等权，因而在轨迹层面按该角色的出场次数加权。**“同一个 $R^i$、不同的统计权重”是两者区别的直接表达。**

论文 §4.2 明确说明统计建立在 active steps 上，而不是 unique trajectories 上。源码同样采用按响应收集的方式；“active step”在这里是一整次 agent 响应，不是一个 token。

来源：[论文式 (2)、(5)](./DrMAS.pdf#page=5)、[compute_grpo_outcome_advantage](./DrMAS/verl/trainer/ppo/core_algos.py)，L541–559。

### 5.3 一个符合当前数学调用路径的数值例子

为了便于计算，示例只取 $N=4$；实际数学训练默认 $N=8$。假设本次采样恰好产生：

| 轨迹 | 实际调用路径 | 最终 $R^i$ | $n_{i,\mathrm{Solver}}$ | $n_{i,\mathrm{Verifier}}$ |
| --- | --- | --- | --- | --- |
| 1 | Solver → Verifier approve | 1 | 1 | 1 |
| 2 | Solver → Verifier approve | 1 | 1 | 1 |
| 3 | Solver → Verifier reject → Solver 修改 | 0 | 2 | 1 |
| 4 | Solver → Verifier reject → Solver 修改 | 0 | 2 | 1 |

本例的两条拒绝轨迹修改后仍失败。这只是示例结果，并不意味着 reject 必然对应失败。

三个统计集合分别为：

~~~text
普通 GRPO：  [1, 1, 0, 0]        每条完整轨迹一次
Dr. MAS Verifier：[1, 1, 0, 0]    每条轨迹只审核一次
Dr. MAS Solver：  [1, 1, 0, 0, 0, 0]  两条失败轨迹各生成两次
~~~

按论文总体方差定义，忽略数值稳定项：

$$
\mu=\mu_{\mathrm{Verifier}}=\frac12,\qquad
\sigma=\sigma_{\mathrm{Verifier}}=\frac12;
\qquad
\mu_{\mathrm{Solver}}=\frac13,\qquad
\sigma_{\mathrm{Solver}}=\frac{\sqrt{2}}{3}.
$$

于是：

| 响应所属的最终结果 | 普通多 agent GRPO，两个角色共用 | Dr. MAS Solver | Dr. MAS Verifier |
| --- | --- | --- | --- |
| 成功轨迹，$R^i=1$ | $+1$ | $+\sqrt{2}\approx+1.414$ | $+1$ |
| 失败轨迹，$R^i=0$ | $-1$ | $-1/\sqrt{2}\approx-0.707$ | $-1$ |

例如，轨迹 3 的两个 Solver 输出与一个 Verifier 输出，原始任务 reward 都为 0。普通 GRPO 对三份输出都给优势 −1；Dr. MAS 对两个 Solver 输出各给约 −0.707，对 Verifier 给 −1。

**区别发生在相同 reward 的中心化、缩放方式，而非奖励来源。**

### 5.4 两种算法的共同数学输入：同一组 rollout 与任务奖励

按照你所指的数学比较，固定同一道题 $x$ 的采样组：

$$
\mathcal{D}_x=\{(\tau^i,R^i)\}_{i=1}^{N},
\qquad R^i=R(\tau^i).
$$

GRPO 与 Dr. MAS 使用共同的 $\mathcal{D}_x$，同样的 $N$，同样的 $R^i$。同一条轨迹内部，各参与角色也共享这份任务奖励。

因此可以直接写：

$$
A_{\mathrm{GRPO}}^i
=\frac{R^i-\mu}{\sigma+\varepsilon_{\mathrm{num}}}
\quad\longrightarrow\quad
A_{\mathrm{DrMAS}}^{i,k}
=\frac{R^i-\mu_k}{\sigma_k+\varepsilon_{\mathrm{num}}}.
$$

**两边的 $R^i$ 是同一个数学量；变化只有 $\mu,\sigma$ 的统计方式。** 对于 Dr. MAS，同一个 $R^i$ 可以在 Solver 的统计中出现两次、在 Verifier 的统计中出现一次，再使用各自的统计量标准化。

这就是“共享最终任务奖励，但各角色具有各自的优势归一化”的含义。这里的核心比较忽略逐输出格式惩罚，额外实现细节单列在第 4.3、8 节。

## 6. 两个角色到底是不是两个模型？

数学训练脚本的默认设置是：

~~~bash
model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B"]'
model_sharing=False
~~~

build_wg_ids 在不共享模式下，为每个角色构建自己的 worker group 标识。因此这里表示：

~~~text
同一个预训练 checkpoint
    ├─ Solver 的模型参数 θS，使用解答角色 prompt
    └─ Verifier 的模型参数 θV，使用审核角色 prompt
~~~

初始化来源一样，但参数组独立；训练后可能分化。**不是默认配置一个 Qwen Solver，再配置另一个模型家族作为 Verifier。**

| 配置 | 参数关系 | 更新方式 |
| --- | --- | --- |
| 相同 model_id，model_sharing=False | 两份独立参数 $\theta_{\mathrm{Solver}}$、$\theta_{\mathrm{Verifier}}$ | 分别由对应角色的生成数据更新 |
| 相同 model_id，model_sharing=True | 同一份参数 $\theta_{\mathrm{Solver}}=\theta_{\mathrm{Verifier}}=\theta$ | 两个角色的数据共同更新 |
| 不同 model_id | 可使用不同模型组成异构团队 | 各自更新对应模型 |

共享模式按 model_id 分组；相同 model_id 的角色进入同一个 worker group。角色不同仍由不同 prompt 和上下文体现。

来源：[默认数学配置](./DrMAS/examples/drmas_trainer/run_math.sh)，L28–33；[build_wg_ids](./DrMAS/agent_system/agent/utils.py)，L95–156；[论文 §3.1](./DrMAS.pdf#page=3)。

## 7. Solver 和 Verifier 都更新，共同 reward 怎么训练两个不同任务？

### 7.1 两者的生成数据都被保留并参与更新

数学编排对两个角色分别调用 save_to_buffer。采样器保留 agent_active_mask 为真的实际输出。训练器完成奖励与优势计算后，按 wg_id 分组，逐组执行：

~~~python
for wg_id in multiagent_batch.keys():
    sub_batch = multiagent_batch[wg_id]
    actor_output = self.actor_rollout_wg[wg_id].update_actor(sub_batch)
~~~

所以 Verifier 是协作中被训练的 actor，不是固定不动的 reward model。它自己生成的评审文本和判定标签属于响应，会参与策略更新；被放在上下文里的队友文字不是它自己本次生成的动作。

来源：[两角色保存响应](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py)，L96、L114；[保留实际激活输出](./DrMAS/agent_system/multi_turn_rollout/rollout_loop.py)，L471–483；[逐 worker group 更新](./DrMAS/verl/trainer/ppo/ray_trainer.py)，L1383–1396。

### 7.2 相同 reward 不等于相同梯度

先看一次 Solver、Verifier 交互。令 $u$ 为 Solver 解答，$v$ 为 Verifier 评审：

$$
p(u,v\mid x)
=\pi_{\theta_{\mathrm{Solver}}}(u\mid x)
\pi_{\theta_{\mathrm{Verifier}}}(v\mid x,u).
$$

两个角色生成的动作和条件上下文不同。为说明训练方向，忽略 clipping、KL 与聚合权重，在旧策略处可写出相应策略梯度贡献：

$$
g_{\mathrm{Solver}}
\propto
A_{\mathrm{Solver}}\,
\nabla_{\theta_{\mathrm{Solver}}}
\log\pi_{\theta_{\mathrm{Solver}}}(u\mid x),
$$

$$
g_{\mathrm{Verifier}}
\propto
A_{\mathrm{Verifier}}\,
\nabla_{\theta_{\mathrm{Verifier}}}
\log\pi_{\theta_{\mathrm{Verifier}}}(v\mid x,u).
$$

即使 $A_{\mathrm{Solver}}=A_{\mathrm{Verifier}}$，对数概率的梯度项也不同。实际代码计算的是各自生成响应中 token 的策略损失；上式只是帮助理解的响应级简化。

直观上：

- Solver 学习生成更容易使团队最终成功的解答和修改。
- Verifier 学习生成更容易使后续流程成功的评审与路由判定。
- 独立参数时分别更新；共享参数时，两类训练贡献汇入同一个模型。

这里使用采样到的文本作为上下文，没有把梯度通过离散评审文本直接反传到队友。共同 reward 为不同角色自己的生成动作提供策略梯度权重。

### 7.3 这没有解决“谁真正有功、谁真正犯错”

训练目标是团队最终成功，源码没有给每份评审额外提供一个“评审内容正确性”的监督标签。因而可能出现：

- Verifier 给了不正确的评审，但 Solver 最后仍答对，评审也获得成功任务 reward。
- Solver 初稿有错，经评审修正后最终成功，初稿也获得成功任务 reward。
- 最终失败时，所有参与响应共同获得失败任务 reward，无法直接定位哪个角色造成失败。

**Dr. MAS 的核心改动是奖励统计和优势尺度校准；它没有估计各角色的因果贡献。** 它也不能保证单份评审在逻辑上正确。跨大量采样，能改善团队结果的行为才有机会形成有效的统计学习信号；这是团队奖励训练的机制与局限。

此外，正 reward 不等于正优势：若同题同角色的其他样本得分更高，当前响应仍可能得到负优势；若 group 得分完全一致，常规中心化优势可能为零。

## 8. 对照代码时，需要保留哪些实现细节？

### 8.1 Dr. MAS 的分组键是“题目 + 角色”

ray_trainer.py 的 compute_advantage：

~~~python
if group_by_agent_id:
    group_index = np.array([
        f"{uid}_{agent_id}"
        for uid, agent_id in zip(
            data.non_tensor_batch["uid"],
            data.non_tensor_batch["agent_id"],
        )
    ], dtype=object)
else:
    group_index = data.non_tensor_batch["uid"]
~~~

uid 区分同题采样 group，agent_id 区分逻辑角色，traj_uid 区分完整轨迹。**按角色分组没有跨不同题目把所有 Solver 输出混在一起，也没有为每个角色重新独立采样最终答案。**

core_algos.py 接着处理每个“group 键、轨迹”对应的响应 score 列表：

~~~python
if group_by_agent_id:
    id2score[idx].extend(reward_list)
else:
    avg_score = torch.stack(reward_list).mean()
    traj2avg[(idx, t_idx)] = avg_score
    id2score[idx].append(avg_score)
~~~

True 分支保留同角色的每个响应 score；False 分支先把一条轨迹的响应 score 平均，统计时每条轨迹计一次，随后将这个轨迹平均 score 用于该轨迹所有响应。

纯共享任务奖励下，同轨迹 score 都是 $R^i$，这个平均仍是 $R^i$，于是得到第 5 节的公式。若存在不同响应的格式惩罚，两条代码分支还会在保留逐响应差异与先平均之间产生区别。

来源：[分组键](./DrMAS/verl/trainer/ppo/ray_trainer.py)，L269–275；[两条统计分支](./DrMAS/verl/trainer/ppo/core_algos.py)，L541–559。

### 8.2 源码默认使用样本标准差

论文公式使用分母为样本数的总体方差；源码直接调用 torch.std(scores_tensor)，默认使用样本标准差，即方差分母为样本数减一。

因此，第 5.3 节例子若直接按源码统计，忽略 $\varepsilon_{\mathrm{num}}$、格式惩罚和补齐复制，数值为：

| 设置 | 成功响应优势 | 失败响应优势 |
| --- | --- | --- |
| 普通 GRPO，统计四条轨迹 | 约 +0.8660 | 约 −0.8660 |
| Dr. MAS Verifier，四个响应 | 约 +0.8660 | 约 −0.8660 |
| Dr. MAS Solver，六个响应 | 约 +1.2910 | 约 −0.6455 |

这些值与论文总体方差示例不完全一样，但“Solver 的权重因重复调用发生变化，Verifier 在此例中不变”的机制一致。

只有一个统计样本时，源码采用 mean=0、std=1 的特殊回退；这也不同于直接套用论文总体方差公式。

来源：[标准差与单样本分支](./DrMAS/verl/trainer/ppo/core_algos.py)，L552–567；数学细节另见 [01 · 方法对比第 8 节](./analysis.md#8-公式对应到官方代码实际实现有哪些细节)。

### 8.3 批次补齐可能再次改变统计权重

ray_trainer.py 在优势计算之前按 worker group 调用 adjust_batch。该函数默认 mode="copy"，若响应条数无法满足并行训练批次的整除要求，会复制全部或随机选择的部分响应进行补齐。

Dr. MAS 分支的统计会计入这些响应副本；普通 GRPO 分支先按 traj_uid 合并再计每条轨迹。因此源码运行中的有效统计权重还可能受补齐影响。

第 5 节给出的公式和例子描述论文的原始活跃输出统计；不能把它们当成包含批次补齐、惩罚等全部实现细节的逐位复现。实际审计训练 batch 时，必须同时查看角色、traj_uid、有效响应和补齐副本。

来源：[优势计算前的 adjust_batch](./DrMAS/verl/trainer/ppo/ray_trainer.py)，L1221–1250；[补齐实现](./DrMAS/agent_system/multi_turn_rollout/utils.py)，L284–347。

## 9. 应当怎样准确理解 Dr. MAS？

MAS 指 Multi-Agent Systems。Dr. MAS 是本文方法和训练框架的名称，可以从两层理解：

1. **算法层：**在协作 GRPO 中，以同题、同角色活跃输出的奖励统计替换所有角色共用的轨迹奖励统计，校准各角色的优势尺度。
2. **系统层：**支持角色编排、共享或独立的模型参数、异构模型和多个 actor 的联合训练，并管理 GPU 资源与模型后端调度。

对 Solver–Verifier 数学例子，一个完整训练批次是：

~~~text
同一道题重复采样 N 次完整协作
    → 每条轨迹由 Solver 解答、Verifier 评审、必要时 Solver 修改
    → 环境只评分最终提交答案
    → 该轨迹所有参与响应获得相同任务 reward
    → 按各响应需要扣除格式惩罚
    → 普通 GRPO：按完整轨迹计算公共统计
      Dr. MAS：按同题同角色的实际响应计算统计
    → 将对应优势用于各角色自己的生成 token
    → 更新各角色模型；参数共享时共同更新同一个模型
~~~

最重要的区分是：**团队 reward 可以相同；各角色用于归一化的样本组成可以不同；各角色的策略梯度也可以不同。** 这三件事并不矛盾。

## 10. 论文与源码定位表

以下行号对应本文核对的固定 commit；源码升级后应以函数名重新定位。

| 要核对的结论 | 论文依据 | 代码入口 |
| --- | --- | --- |
| 每条协作轨迹一个最终任务 reward | §3.1–3.2，PDF 第 3–4 页 | [gather_rollout_data，L72](./DrMAS/agent_system/multi_turn_rollout/rollout_loop.py)；[EpisodeRewardManager，L77](./DrMAS/agent_system/reward_manager/episode.py) |
| 默认数学 group size=8，两角色，max_loop_num=2 | §5.1，附录 C.1–C.2 | [GRPO 脚本，L23](./DrMAS/examples/grpo_trainer/run_math.sh)；[Dr. MAS 脚本，L23](./DrMAS/examples/drmas_trainer/run_math.sh) |
| 复制题目后才运行完整协作流程 | 式 (1)，附录 E | [generate_sequences 采样分支，L319](./DrMAS/agent_system/multi_turn_rollout/rollout_loop.py) |
| Verifier 审查解答并附最终判定 | 附录 C.1、图 5 | [VERIFIER_PROMPT，L26](./DrMAS/agent_system/agent/agents/math/verifier_agent.py) |
| 保留完整评审，传入下一次 Solver 上下文 | 附录 C.1 | [return_whole_response，L64](./DrMAS/agent_system/agent/agents/math/verifier_agent.py)；[team_context 更新，L113](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py) |
| 最后一个 Solver 轮次后不再审核 | 附录 C.2 给出轮次预算；具体路径以源码为准 | [最后轮次 break，L99](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py) |
| 格式惩罚按响应扣分 | 附录 C.2–C.3 | [apply_invalid_action_penalty，L203](./DrMAS/verl/trainer/ppo/ray_trainer.py) |
| Dr. MAS 按活跃输出加权统计 | §4.2，式 (5) | [角色分组，L269](./DrMAS/verl/trainer/ppo/ray_trainer.py)；[extend 与 append 分支，L542](./DrMAS/verl/trainer/ppo/core_algos.py) |
| 角色可以共享或独立参数 | §3.1、§4.3 | [build_wg_ids，L95](./DrMAS/agent_system/agent/utils.py) |
| 两个角色生成的数据都训练 | 附录 E，Algorithm 1 | [save_to_buffer，L96 / L114](./DrMAS/agent_system/agent/orchestra/math/math_orchestra.py)；[update_actor，L1391](./DrMAS/verl/trainer/ppo/ray_trainer.py) |
