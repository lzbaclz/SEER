# 06 · Method 章排法

## 总体规则

THUML method 章的特点：

1. **3 个或 4 个并列子组件**（component-style decomposition）
2. **每个 component 半页到一页**
3. **每个 component 末段一句话点明 *作用* + *对应实验编号***
4. **§4 末段一段 architecture overview** 把组件拼起来 + 给 loss 公式
5. **(可选) §4 末加一节 theoretical analysis**

整个 §4 不超过 3 页。

## 子节模板（每个 component 共用）

```
### 4.x 〈Component NAME〉

(opening paragraph, 3-5 句)
- 1 句：本 component 解决的问题（指回 §3 Observation X）
- 1 句：传统做法是什么、为什么不够
- 1 句：我们怎么做的（一句 high-level）
- (可选) 1 句：和 component A 的关系

(formal description paragraph, 1-2 段)
- 给 1-3 个编号公式 (Eq. (n))
- 用 hyphenated capitalised 名字称呼子操作
- 配 mini diagram 或 architecture sub-figure（可选）

(closing paragraph, 1-2 句)
- 这个 component 的 *作用 / 性质*
- 对应实验在哪验证：例 "We validate this in §5.4 (Ablation, Table 4)."
```

## 实例：Autoformer 的 Method 章

§3 子节是这样组织的（高置信度，多篇互证）：

```
3.1 Series Decomposition Block
3.2 Auto-Correlation Mechanism
3.3 Encoder-Decoder Architecture
```

每节大概 0.5-0.8 页：

- §3.1 给出 trend / seasonal 分解的公式（moving average）+ 一段说明这是 *inner block* 不是 pre-processing。
- §3.2 给出 auto-correlation 公式（基于 FFT 的 lag 选取）+ 一段对比 self-attention。
- §3.3 给出整体 stacked layer 的图 + training objective。

## 实例：Non-stationary Transformers 的 Method 章

```
4.1 Analysis of Over-stationarization
4.2 Series Stationarization
4.3 De-stationary Attention
4.4 Putting Things Together
```

特点：

- §4.1 不是直接给方法，而是**专门花半页分析 prior 方法的失败**（用 attention map 对比图）。这是 THUML 经常采用的"先分析再设计"格式。
- §4.2 / §4.3 是两个互补 component：一个去归一化（让训练稳定），一个补回归一化抹掉的信息（让 attention 多样）。
- §4.4 一段 wraps things up，给 architecture diagram + final loss。

## 实例：MDD 的 Theory + Algorithm 双章

```
3 Margin Disparity Discrepancy (Theory)
   3.1 Definition: Disparity Discrepancy
   3.2 Margin Disparity Discrepancy
   3.3 Generalization Bounds (Theorem 1, Theorem 2)
4 Algorithm
   4.1 Empirical Margin Disparity Discrepancy
   4.2 Adversarial Optimization
   4.3 Algorithm Summary
```

老线偏理论的论文常常这样：**§3 给定义 + theorem，§4 把 §3 落到 empirical optimization**。这种结构 reviewer 一看就觉得严谨。

## 写作微习惯（method 章）

### 1. 公式编号要"省着用，但用得显眼"

THUML 的论文 method 章一般 **5-15 个编号公式**。每个编号公式都"值钱"——是一个有命名的操作。例：

```
Auto-Correlation:
   R_xx(τ) = ∑_t x_t · x_{t-τ}                                 (1)
   
Adaptive Period Discovery:
   {τ_k} = TopK_argmax R_xx(τ)                                 (2)
   
Time-Delay Aggregation:
   y_t = ∑_k softmax(R_xx(τ_k)) · roll(x, τ_k)                 (3)
```

**不写 "Eq. (1)"** —— 写 "Equation 1" 或直接引用 "the auto-correlation in (1)"。

### 2. 子操作命名

每个子操作都给一个 hyphenated capitalised 名字，全文专名：

- *Auto-Correlation*（不写 auto-correlation）
- *Time-Delay Aggregation*
- *Adaptive Period Discovery*

不命名的子操作只是"实现细节"，方法章不写。

### 3. 架构图 (Fig 2 或 Fig 3)

THUML 的架构图特点：

- **彩色块表示 component**，每个 component 颜色对应 §4 子节
- **箭头标注的是数据流类型**（feature / loss / attention）
- **不画到 implementation 细节**：只到 component 级
- 图下 caption 一段长 caption（150 词左右）解释整个流程

避免 "黑箱方框堆叠图"——审稿人讨厌看不懂的架构图。

### 4. 训练目标 (Loss) 一节末尾给

```
(end of §4)

We train Seer end-to-end with:
   L_total = L_pred + λ_1 · L_pareto + λ_2 · L_reg            (Eq. 8)

where L_pred is the reuse-distance regression loss, L_pareto is the policy
imitation loss, and L_reg is a sparsity regularizer. We set λ_1 = 0.5 and
λ_2 = 0.01 throughout (sensitivity in §B.3).
```

特点：

- loss 公式编号
- 每一项命名 + 一句解释
- 超参数默认值在这里给（不是放 appendix）
- 敏感性实验 pointer 给到 appendix

### 5. Algorithm Box

某些论文用 algorithm 框（algorithm 1）写训练或推理流程，但 **THUML 用得不多**——更倾向于用文字 + 公式表达。如果用 algorithm box，会非常简洁（< 10 行）。

### 6. *(Insight 1.)* / *(Observation 1.)* 显式标注

method 章里会零散嵌入：

```
Insight 1. (Why decomposition as inner block?)
Decomposing the series at every layer allows the trend to be progressively
refined, which is fundamentally different from one-shot pre-processing...
```

这种 italic block 让 reviewer 在 skim 时也能 picked up 关键论断。

## §4 末尾常见的"theoretical analysis"小节（中等置信度）

旧线必有，新线 nice-to-have。Pattern：

```
4.5 Theoretical Analysis

Proposition 1. (Asymptotic equivalence to optimal stationary process)
Under mild assumptions, our De-stationary Attention recovers the residual
non-stationary information up to a constant scale.

Proof Sketch. (1 段). Full proof in Appendix C.
```

不强证明，但**说一个 reviewer 喜欢听的命题**。Seer 可以放：

```
Proposition 1. (Pareto Improvement)
For any reuse-distance prediction with non-zero accuracy, the joint
policy (PAE + LPS) Pareto-dominates LRU + heuristic prefetch under
fixed capacity, in expectation over the workload distribution.
```

简洁 + 显得严谨。完整证明可以放 appendix。

## Seer 的 §4 草稿骨架

```
4 Seer

4.1 Insight: Eviction is Prefetch in Reverse
    1 段 motivation, 引用 Observation 1, Insight 1.
    
4.2 SeerNet: A Reuse-Distance Predictor
    1 段：输入特征（block id pattern, query, recency, position）
    1 段 + 公式：MLP 架构
    1 段 + Eq.: training loss (regression on reuse distance)
    last sentence: validated in §5.5 (component ablation)
    
4.3 Pareto-Aware Eviction (PAE)
    1 段：基于 SeerNet 输出做 candidate set
    1 段 + Eq.: PAE 决策规则（greedy / DP）
    last sentence: §5.3 main results
    
4.4 Look-ahead Prefetch Scheduler (LPS)
    1 段：用 SeerNet 输出预测下一步 miss 候选
    1 段 + Eq.: 调度器规则
    last sentence: §5.4 generalization tests
    
4.5 Architecture & Training
    1 段 architecture diagram (Fig 3)
    1 段 + Eq. (8): L_total
    
4.6 Theoretical Analysis (optional, 0.3-0.5 页)
    Proposition 1: Pareto improvement under predictability hypothesis
    Proof sketch + pointer to Appendix C
```

按这个骨架写，§4 大约 2.5 页，留足空间给 §5 实验。
