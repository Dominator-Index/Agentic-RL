# My FAULT：自诊断定位错误，用终局结果学习错误价格，再分配步骤惩罚

> 根据 [论文 v1](./FAULT.pdf) §2–3、附录 B 的完整算法与理论、C 的配置以及结果分析整理。论文给出[代码地址](https://github.com/SKYLENAGE-AI/FAULT_Agentic_RL)，但本次 HTTP 404、匿名 clone 失败，所以未完成代码核对。

## 1. Motivation：诊断回答“错在哪”，结果回答“这类错多重要”

终局 reward 同时有两个不足：同结果组缺少差异，一条失败轨迹中也不容易定位真正应改的步骤。自由文本反思虽然能指出错误，却可能引用错误证据，也没有可靠的严重程度数值。

FAULT 将其拆开：先把诊断写成“错误类型、步骤、原文证据”，再从真实终局结果统计估计各类错误的相对成本，最后把有限惩罚分配给被诊断的步骤。

它引入了额外诊断和统计学习，不是仅对现有回报重分组，也不属于无需额外生成的轻量算法。

## 2. 冷启动：actor 与 diagnoser 如何建立

先用初始模型采样，teacher（论文记作 GPT-5.6 Sol）读取完整轨迹和终局结果，提出错误类别、步骤编号和原文引用。经过校验形成固定 taxonomy，并用诊断文本做 SFT。

这个 checkpoint 同时初始化 actor 和 diagnoser，两者共享权重。RL 期间不再使用 teacher，但每条轨迹生成三份 self-diagnosis，按解析、证据和一致性规则筛选。

**引用能在轨迹对应步骤中找到，不等于该错误的语义判断或因果归属已被完全证明。** 证据校验减少错引，但诊断偏差仍是限制。

令 $n_{i,k}$ 是轨迹 $i$ 中第 $k$ 类已接受错误的统计量，使用按轨迹长度归一化的率：

$$
x_{i,k}=\frac{n_{i,k}}{T_i}.
$$

实际多诊断汇总、L1/L2 特征与缺失诊断的插补规则见附录 B/C；不能只把“没有有效诊断”当成“没有错误”。

## 3. 错误价格不是 LLM 自报的 severity

对同一道题的保留轨迹，已知其中哪些成功，记集合 $Y_q$。在所有同样大小的候选成功集合 $\Gamma$ 中，条件似然为：

$$
\Pr_q(\beta)=
\frac{\exp(\sum_{i\in Y_q}\beta^\top x_{q,i})}
{\sum_{\Gamma:\,|\Gamma|=|Y_q|}\exp(\sum_{i\in\Gamma}\beta^\top x_{q,i})}.
$$

拟合负对数似然，比较的是同题内哪些错误率与失败相关，避免直接把任务本身难度当错误成本。只有有成功也有失败的组能够提供这类识别信息。二元拟合还使用小样本修正、收缩和在线接受规则；WebShop 的 graded reward 采用组内 ridge 变体，不能全部写成同一个二元 logistic 公式。

负系数转成归一化非负价格：

$$
w_k=\begin{cases}
|\beta_k|/\max_{j:\beta_j<0}|\beta_j|,&\beta_k<0,\\
0,&\beta_k\ge0.
\end{cases}
$$

这些是 outcome association，不是控制干预之后测得的因果错误损失。severity_hint、引用长度不是这个价格的直接来源。

## 4. 每条轨迹只有一个有上限的惩罚预算

主文的简化得分为：

$$
S_i=\operatorname{clip}\left(\sum_kw_kx_{i,k},0,1\right),\qquad
P_i=\min(\lambda_u S_i,\eta\Delta R),\quad 0\le\eta<1.
$$

$\Delta R$ 是成功失败奖励差，$\lambda_u$ 控制当前训练阶段的惩罚强度。实验取 $\eta=0.9$；附录还有首个错误加权，不能省略后声称完整复现。

惩罚强度 warm-up 后按 cosine 调整，最后一次更新降为 0。因此是训练中的过程引导，不是永久改写终局 reward。

## 5. 定位只决定“预算分到哪里”

设第 $t$ 步的已接受错误条目集合为 $V_{i,t}$，局部权重为：

$$
h_{i,t}=\sum_{e\in V_{i,t}}w_{k(e)},\qquad
\delta_{i,t}=P_i\frac{h_{i,t}}{\sum_uh_{i,u}}.
$$

相同错误由多个诊断重复指出会贡献多份支持。附录对最早的正权重步骤再乘 1.5；预算得分中的 first-error boost 也是 1.5，两者作用不同：一个改变预算，一个只改变位置分配。

若正预算来自插补但没有有效位置，则惩罚放在最后一个 active step。无论哪种情况：

$$
\delta_{i,t}\ge0,\qquad\sum_t\delta_{i,t}=P_i.
$$

例如 $P_i=0.6$、两处定位权重为 2 和 1，得到惩罚 0.4 和 0.2。注意这是惩罚预算守恒，**不是原始终局回报被无损地重新分配**：最终分数中会减掉总惩罚。

## 6. 双通道优势怎么计算

对同题全部 active steps $\Omega_q$，episode 通道使用：

$$
E_{i,t}=R_i-\delta_{i,t},\qquad
A_{i,t}^{\rm ep}=E_{i,t}-\operatorname{mean}_{(j,u)\in\Omega_q}E_{j,u}.
$$

虽然叫 episode 通道，均值的统计单位是步骤行，不是先对 $N$ 条轨迹各取一次均值。不同长度因此影响出现权重。

step 通道在相同/相似动作前 anchor 组 $H(s_{i,t})$ 中中心化：

$$
\widetilde Z_{i,t}=\gamma^{T_i-t}R_i-\delta_{i,t},\qquad
A_{i,t}^{\rm step}=\widetilde Z_{i,t}-\operatorname{mean}_{(j,u)\in H(s_{i,t})}\widetilde Z_{j,u}.
$$

最终 $A_{i,t}=A_{i,t}^{\rm ep}+\omega A_{i,t}^{\rm step}$，分配给该步 response tokens，用常规 token ratio 与 PPO clip 更新。

这里减的是当前步骤被分配的惩罚，不是先把所有未来诊断惩罚当 reward 再重新累计 return-to-go。ALFWorld/WebShop 主要均值中心化，Search 配置还做标准差缩放；不能说所有实验完全不除 std。

全失败时 $R_i=0$，不同 $\delta_{i,t}$ 就能制造相对差异；没有错误或少受罚步骤可以获得正相对优势。这来自已学到的诊断价格，不是单纯从同一批全零终局标签推导出了额外信息。

## 7. 共享 diagnoser 怎样更新，有什么风险

Actor 与 diagnoser 使用同一模型权重，诊断 tokens 不接受 policy gradient；actor 参数变化后，下一批诊断也随之变化。价格从近期滑动窗口定期重估，论文主设置从第 30 次更新起每 10 次 refit。

不对诊断 token 直接做 RL，减少了直接学习“少报错误就少受罚”的路径，但不保证共享权重下诊断能力不退化。错误 taxonomy 固定，分布变化、证据遗漏和类别关联都可能造成偏差，需要监测而不是假定自演化必然改善。

## 8. 理论保证：仅 episode 通道保留二元成功排序

因为 $0\le\delta_{i,t}\le P_i\le\eta\Delta R$，任意成功轨迹步骤与失败轨迹步骤在 episode 通道的差至少为：

$$
A_{i,t}^{\rm ep}-A_{j,u}^{\rm ep}\ge(1-\eta)\Delta R>0.
$$

共同均值抵消后得到这个界。它不自动覆盖叠加 step 通道后的总优势、无效动作惩罚等其他工程项，也不覆盖任意连续奖励的排序。graded reward 的两个样本原始差值若小于 cap，仍可能换序。

因此该定理保护的是有限范围内的结果优先级，不是诊断准确率、因果信用或最终策略性能保证。

## 9. 实验与成本必须一起看

模型为 Qwen3-1.7B 和 Qwen3-4B-Instruct-2507，RL 160 updates、每题 8 条 rollout、每轨迹 3 份诊断，主要评估 ALFWorld、WebShop、Search-based QA。

4B ALFWorld：FAULT 91.0、SEED 84.3；1.7B 为 68.7/65.7。4B WebShop task score 为 FAULT 88.5、SEED 86.2，但 strict success 为 71.5/74.2，FAULT 并非全部指标更好。Search 4B FAULT 43.8，低于 SEED 44.4。

原始 GRPO/GiGPO 不含 diagnosis-SFT，直接比较混入初始化差异；论文在 4B 提供相同冷启动的对照，但也说明不是完美单因素控制。WebShop 使用 graded score 训练，而部分基线使用不同奖励口径，应明确区分任务分数与严格成功率。

95% signal coverage 是论文定义的优势对比度覆盖诊断量，不是 95% 成功率、95% 正确诊断或 95% 正确梯度。更多非零信号不必然更准确，还要看错误定位与真实任务指标。

成本包括 teacher 冷启动、SFT、每条轨迹多次诊断和在线价格拟合，不应与 ProGPO 的无额外生成方案等成本比较。没有完成硬件核验，不推断 GPU 型号。

## 10. 与前面方法的关系

| 方法 | 额外信用从哪里来 |
|---|---|
| GiGPO/HGPO | 终局回报与 state/history 比较 |
| CRBC | 固定经验图上递归频率平均的 $Q-V$ |
| ProGPO | 全失败分支的观测覆盖 |
| ProCredit | 可执行验收检查的进度差分 |
| FAULT | 自诊断错误位置 + outcome 拟合的价格 |

FAULT 的关键拆分是“定位”和“定价”，这比直接让 LLM 口头给每步打分更可审计，但价格仍是关联、诊断仍可能错。对 simple yet effective 的新方案，应先判断是否真的需要整个 diagnosis pipeline，而不是把复杂框架全部搬入历史分组。

## 11. 代码与复现状态

论文列出的 `https://github.com/SKYLENAGE-AI/FAULT_Agentic_RL` 本次返回 HTTP 404，`git clone` 无法匿名读取仓库。未克隆成功，未建立软链接，未对上述公式与实际训练源码作一致性声明。恢复公开访问后，应优先检查诊断插补、first-error boost、mean/std 模式、价格 refit 和附加奖励项。

## 12. 对当前分组 idea 的修订

本轮核对 §2 与附录 B 的定价、惩罚边界和同结果组分析。FAULT 使用 anchor 比较，但主要新增的是诊断、outcome 定价及步骤惩罚分配，不是新的 history partition。

其条件似然在全失败组只有一个候选成功集合，概率为 1，因此该组本身无法识别错误价格。应用于全失败组的信用依赖已有诊断能力和其他混合结果数据，不能将它解释为仅凭当前全零标签恢复了额外任务信息。

综合讨论见 [新增论文与分组边界](../../05-新论文对分组Idea的修正-ProCredit-SCA-CRBC-FAULT.md)。事后诊断可以作为额外监督，但不能作为动作前 baseline 路由后继续沿用无偏证明。未重新核验官方仓库在线状态，也未新增训练复现。
