# 05 · Problem Formulation —— "命名隐藏瓶颈 / 概念反转" 的招牌写法

THUML 论文之所以好看，最关键的就是这一招：**把问题"重新形容一遍"，使得 reviewer 觉得"啊原来这个问题这么深"**。这是其他中国组最难复制的一环。下面把这一招的工艺细节拆开。

## 招式一：命名一个隐藏瓶颈

### Pattern

```
步骤 1：观察一个 prior 方法的失败模式（不显眼但可重复出现）。
步骤 2：给这个失败模式起一个 *专有名字*（hyphenated capitalised）。
步骤 3：一段话把名字 + 现象 + 后果讲清楚，配 Fig 1 或 Fig 2。
步骤 4：把"解决这个名字的问题"作为论文的目标陈述。
```

### 案例

#### Autoformer：**Information Utilization Bottleneck**

> "Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, **resulting in the information utilization bottleneck**."

- 名字：information utilization bottleneck
- 现象：sparse attention 略过了大量 token
- 后果：长序列预测精度下降
- Fig 1：用一个长序列示意 sparse attention 跳过的位置

#### Non-stationary Transformers：**Over-Stationarization**

> "...This problem, **termed over-stationarization**, leads Transformers to generate indistinguishable temporal attentions for different series and impedes the predictive capability of deep models."

- 名字：over-stationarization
- 现象：归一化抹掉了序列差异
- 后果：所有序列的 attention map 长得一样
- Fig 1：归一化前后 attention map 对比

#### iTransformer：**Meaningless Attention Maps**

> "...the embedding for each temporal token fuses multiple variates ... which may fail in learning variate-centric representations and result in **meaningless attention maps**."

- 名字：meaningless attention maps
- 现象：把同一时刻多变量挤成一个 token，attention 学不出有用模式
- 后果：长 lookback 反而变差

#### SimMTM：**Random Masking Ruins Temporal Variations**

- 名字：(隐式) ruptured temporal manifold
- 现象：random mask 把序列结构打散
- 后果：reconstruction 任务变得过于困难，pretrain 信号变差

#### CDAN：**Under-matching by Marginal Adversarial Methods**

> "Existing adversarial domain adaptation methods may struggle to align different domains of multimodal distributions native in classification problems."

- 名字：under-matching
- 现象：纯 marginal alignment 忽略类别条件分布
- 后果：分类边界没对齐，分类精度低

### 起名字的诀窍

1. **hyphenated**："over-stationarization", "under-matching", "negative transfer", "information utilization bottleneck"
2. **动词性 / 形容词性都行**，但避免名词堆砌（"under-matching" 比 "marginal-distribution-alignment-only-method" 好得多）
3. **不要太通用**："suboptimal" 不是名字，"reuse-distance blindness" 才是名字
4. **能在 Fig 1 里画出来**：起名字的同时设计一张反例图
5. **3-5 个词以内**：太长就不会被引用

### Seer 的"隐藏瓶颈"候选

我建议 Seer 起 1-2 个名字，从下面四个里选：

- **Reuse-Distance Blindness**：现有 policy 不预测下一次访问，只看过去。
- **Eviction-Prefetch Decoupling**：现有系统把 eviction 和 prefetch 当独立模块，浪费 capacity。
- **Static Capacity Misallocation**：现有系统给所有 head / layer / batch 平均分 KV 预算。
- **Coarse-Grained Importance**：用 attention score 做 importance 太粗，忽视长距 reuse。

我倾向 **"Reuse-Distance Blindness"** —— 简短、可视化、能直接连到 SeerNet（"我们的方法学了 reuse distance"）。

## 招式二：概念反转 (Conceptual Inversion)

很多 THUML 论文的核心 insight 不是"加一块"，而是"**把某件事翻过来做**"。

### 案例

#### Autoformer：decomposition 从 *pre-processing* → *inner block*

> "We **break with the pre-processing convention** of series decomposition and **renovate it as a basic inner block** of deep models."

把 prior 方法里"先 detrend 再喂网络"这个外置流程，改成"网络内部反复 detrend"。Fig 2 直接画 stacked decomposition layers。

#### iTransformer：tokens 从 *temporal* → *variate*

> "We propose iTransformer that **simply applies the attention and feed-forward network on the inverted dimensions**."

不改 attention 也不改 FFN，只是把 token 维度从"时间步"翻到"变量"。一个 Fig 1 就讲清楚整篇贡献。

#### Non-stationary Transformers：从"先归一化"到"归一化但显式去归一化"

把 prior 的"stationarize → forecast" 做成"stationarize → forecast → de-stationarize attention"，让模型既享受归一化的稳定性又保留 raw series 的多样性。

#### SimMTM：从"随机 mask → 直接重建"到"随机 mask → 在 manifold 上 aggregate 邻居重建"

#### MDD：从"基于 0-1 loss 的 H∆H-divergence"到"基于 margin loss 的 disparity discrepancy"

把 Ben-David 的经典 bound 从 0-1 loss 推广到 margin loss，bound 更紧、algorithm 更稳定。

### 起"概念反转"的诀窍

1. 找一个 prior 方法里的**默认假设**或**约定动作**（比如 "decomposition 是 pre-processing"，"token 是按时间切"）。
2. 问"假如颠倒过来呢？"
3. 把颠倒后的版本用一句话讲清楚，再用 Fig 1 画清楚。
4. **关键是 reviewer 一看就 get**——颠倒得太奇怪 reviewer 反而困惑。

### Seer 的概念反转候选

- **"Eviction is prefetch in reverse"**：把 eviction 和 prefetch 视为同一个 reuse-distance 排序问题的两端 —— 这正好对应 Seer 的 joint-policy 思路。
- **"Predict the future, retain accordingly"**：从"基于过去访问决定保留谁"翻到"基于未来访问决定保留谁"。
- **"Block-level importance"**：从 token-level / head-level attention score 翻到 block-level reuse distance。

我推荐 §1 / §3 用 **"Eviction is prefetch in reverse"** 当 framing；这是 Seer 在 NeurIPS 评审里最容易被记住的句子。

## 招式三：把 problem formulation 写得像 *观察陈述* 而不是 *任务定义*

THUML 的 §3 Preliminary 通常很短，**不会在 problem statement 上花太多笔墨**。形式化用一段简洁的语言交代：

```
Given a multivariate time series x_{1:L} ∈ R^{L×N}, the goal is to predict the
future series x_{L+1:L+H} ∈ R^{H×N}, where L is the lookback window and H is the
forecasting horizon.
```

然后立刻进入 *观察陈述*：

```
Observation 1. (Multi-Periodicity) Real-world time series exhibit multiple periods
that overlap in time but operate at different frequencies. Existing methods process
the 1D sequence directly, which is suboptimal because... <见 Fig 2>.
```

这个 *Observation 1.* 的格式（italic 或 boxed）是 THUML 标志性写法之一。让 reviewer 不会在 §3 里迷路。

### 给 Seer 的 §3 模板

```
3 Preliminaries

3.1 KV-Cache Inference Workflow
  Notation table (Table 1):
    L     prompt length
    M     decode length
    B     KV block size (in tokens)
    K = ⌈(L+M)/B⌉   total blocks per request
    C     HBM capacity (in blocks)
  Each generation step accesses a sequence of KV blocks; we record the access trace
  as π = (π_1, π_2, ...).

3.2 The Eviction-Prefetch Problem
  Given a workload π and a capacity budget C, the goal of Seer is to choose, at every
  step, which blocks to RETAIN (eviction policy) and which to LOAD (prefetch policy)
  such that the *miss rate* and *fetch latency* are jointly minimized.

  Observation 1. (Reuse-Distance Predictability)
  Despite the apparent randomness of decoding, block-level reuse distances are highly
  predictable from a small set of features (block_id pattern, query embeddings, recency).
  Figure 2 shows that on the ShareGPT trace, a 2-layer MLP achieves R² = 0.74 on
  reuse-distance prediction, leaving 18-30% of LRU's mistakes recoverable.
```

§3 不超过半页，但**已经把 problem + insight + 实证证据全都给出来了**。这种密度是 THUML 标准。

## 招式四：用 *(Observation 1.)* / *(Insight 1.)* / *(Proposition 1.)* 显式标注关键论断

整篇论文里凡是关键论断，都用 italic / bold 显式标记：

```
Observation 1. ...                   ← 经验发现
Insight 1. ...                       ← 设计 rationale
Proposition 1. ...                   ← 形式化陈述（弱于 theorem）
Theorem 1. ...                       ← 正式定理（旧线 / 理论章必备）
Lemma 1. ... / Corollary 1. ...      ← 证明辅助
```

这样 reviewer 在 skim 论文时**一眼能看到关键论断**，对接受率正面。

### Seer 落地

至少埋 3 个：

```
Observation 1. (Reuse-distance predictability)            — §3.2
Insight 1.    (Joint formulation reduces capacity waste)   — §4.1
Proposition 1. (Pareto-optimality of SeerNet's policy)    — §4.5（可选）
```

数量克制，**每篇 3-5 条**就够。不要满文档全是 italic block。

## 总结：Seer §3 / §4 应当回答的 5 个问题

1. 你的 problem 是什么？（一段 + Table 1）
2. 你 *命名* 的隐藏瓶颈叫什么？（一段 + Fig 1）
3. 你的 *概念反转* 是什么？（一段 + Insight 1）
4. 你的 *方法* 由什么组成？（§4.2 / 4.3 / 4.4）
5. 这个方法在哪个意义上 *最优* 或 *理论上有保证* 的？（Proposition 1，可选）

把这 5 项答完，§3+§4 就成型了。
