# 07 · 理论 / 分析章节怎么写

THUML 在理论上的"门面"是迁移学习老线（DAN / JAN / CDAN / MDD），新时序系列只是"轻 grounded"——但是即便不证 theorem，他们也擅长**让论文显得有理论温度**。这一节把两种工艺都讲一遍。

## A. 重理论：MDD 范本

*Bridging Theory and Algorithm for Domain Adaptation* (ICML 19) 是 THUML 老线最理论的一篇。它的章节结构是 ICML 偏理论论文的范本。

### 章节布局

```
1  Introduction
2  Related Work
3  Background and Notations
   3.1 Domain Adaptation
   3.2 Hypothesis Space and Margin Loss
   3.3 Existing Discrepancies (H∆H, …) ← 把 prior bounds 摆出来
4  Margin Disparity Discrepancy
   4.1 Definition (Definition 1, 2)
   4.2 Properties (Lemma 1, 2)
   4.3 Generalization Bounds (Theorem 1, 2)
5  Algorithm
   5.1 Empirical MDD
   5.2 Adversarial Optimization
   5.3 Implementation
6  Experiments
7  Discussion / Conclusion
   Appendix A: Full Proofs
```

### Theorem 的 *写法* （这是关键工艺）

THUML 的 theorem 永远做到三件事：

1. **简短**：一个 theorem 不超过半页。
2. **可读**：变量在前文都已定义；不会临时引入 6 个新符号。
3. **直接对接 algorithm**：theorem 的 conclusion 对应 §5.1 的 empirical loss。

Pattern：

```
Theorem 1. (Generalization Bound via Margin Disparity Discrepancy)
For any hypothesis h ∈ H and ρ > 0, the target risk is bounded by:
   R_T(h) ≤ R_S^ρ(h) + d_{H,ρ}(D_S, D_T) + λ + O(complexity terms)
where R_S^ρ is the source margin loss, d_{H,ρ} is the proposed margin
disparity discrepancy, and λ is the ideal joint hypothesis error.

Remark 1. The bound interpolates between [Ben-David et al. 2010] (ρ=0)
and our proposal, achieving tighter complexity by adjusting margin ρ.
```

Pattern 拆解：

- **theorem 后立刻 follow 一个 Remark**，把 theorem 的实际意义讲清楚（reviewer 不需要自己 parse 公式）。
- **Remark 把这个新 bound 和 prior bound 比**："interpolates between" / "tighter than"。
- 如果有 corollary，常用来从 theorem 推 algorithm（"Corollary 2 gives the empirical objective"）。

### Proof sketch 的写法

```
Proof Sketch. The result follows from three steps.
(1) Bound the target margin loss by source margin loss plus discrepancy (Lemma 1).
(2) Apply Rademacher complexity bound for margin loss class (standard).
(3) Combine via union bound. Full proof in Appendix A.
```

主文 1 段 + appendix full proof。**绝不在主文堆 8 行 ε-δ 推导**——那是 appendix 的事。

### Rademacher complexity 的引用

MDD 用 *Rademacher complexity* 做 hypothesis space measure。这是 ICML / COLT 评委的"舒适区"术语。Seer 如果要 prove a bound on hit ratio，可以引用同样的工具：

```
The hit-ratio risk is bounded as:
   R(π) ≤ R_emp(π) + 2 R_n(F) + O(√(log(1/δ)/n))
where R_n(F) is the Rademacher complexity of the policy class F.
```

不需要严证，只要"形式正确 + Remark 把意义讲清"。

## B. 轻理论 (Theory-Informed Justification) ：时序系列范本

新时序系列（Autoformer / TimesNet / Non-stationary / SimMTM）几乎不证 theorem，但用以下三种"轻理论"工具撑起严谨感：

### 1. 引用经典 *理论灵感来源*

- **Autoformer**: "Inspired by stochastic process theory, we design the Auto-Correlation mechanism based on series periodicity."
- **TimesNet**: 引 multi-periodicity 的 signal-processing 经典 (Fast Fourier Transform)
- **FEDformer**: "We exploit the fact that most time series have a sparse representation in a well-known basis such as Fourier transform" — 这是 signal processing 100 年的常识，但写出来很有分量。
- **SimMTM**: 引 manifold learning（Roweis-Saul 2000）的几何直觉
- **Non-stationary**: 引 econometrics 的 stationarity 概念

**Pattern**：每个核心 component 都对应一个**经典理论根**。reviewer 一读"哦这个不是凭空设计的"。

### 2. *Proposition 1.* 取代 *Theorem 1.*

新时序系列倾向用 *Proposition* 而不是 *Theorem*，要求弱一些但仍显得正式：

```
Proposition 1. (FFT-based Period Discovery is Optimal under Periodicity)
If x_{1:L} contains a dominant period P, then arg max_τ R_{xx}(τ) = P
almost surely as L → ∞.
```

不需要严证，1-2 段说明即可。

### 3. *Empirical 验证* 充当部分理论

很多 THUML 论文用一张 *empirical 图* 充当"理论灵感的实证"。例：

- Autoformer 的 §3.2 末尾用一张 attention map vs auto-correlation 对比图证明"period-based attention 比 sparse attention 抓得到更多结构"。
- iTransformer 的 §3 末尾用 t-SNE 或 attention heatmap 证明"variate token 学到了变量间相关性，temporal token 学不到"。

**reviewer 接受这种"半理论 + 半实证"的混合，前提是论文整体严谨**。

## C. 给 Seer 的 theory 章策略

Seer 是 ML systems 论文，发 NeurIPS 比发 SOSP 风格更倾向 ML——所以 theory 章**值得写但不要太重**。建议如下分层：

### 层 1（最低成本）：Observation + Empirical Justification

```
§4.1 末尾：
Observation 1. (Reuse-Distance Predictability)
On four production traces (ShareGPT, Mooncake, ...), block reuse distances
are highly predictable from a small feature set: a 2-layer MLP achieves
R² = 0.74, while LRU's reuse-distance estimate has R² = 0.18.
```

一段 + 一张图 + 一组数字。**这是必备的，没有的话 §4 就空了**。

### 层 2（中等成本）：Proposition 1 with Sketch

```
§4.5 末尾：
Proposition 1. (Pareto Improvement Under Predictability)
Let π* be the offline-optimal policy and π_LRU the LRU policy. For any
reuse-distance predictor f with prediction error ε, the policy induced
by Seer satisfies:
    miss_rate(π_Seer) ≤ miss_rate(π_LRU) - Δ + O(ε)
where Δ > 0 is the predictability gap (Definition 1).

Proof Sketch. (1 段). Full proof in Appendix C.
```

一段 proposition + 半段 proof sketch + appendix 完整证明。**可以做但不强求**——如果 ablation 数据说明力强，proposition 不一定加分。

### 层 3（最高成本）：Theorem 1 + Rademacher complexity

如果 Seer 要走"learning-augmented online algorithm"的路线，可以走 Theorem 形式：

```
Theorem 1. (Hit-Ratio Regret Bound)
For policy class F with Rademacher complexity R_n(F), the expected
hit-ratio regret satisfies:
    Regret(π_Seer) ≤ Regret(π*) + 2 R_n(F) + O(√(log(1/δ)/n))
```

这种 theorem 是 NeurIPS reviewer 在"theory track"上的最爱。但要证全 NeurIPS supplementary 至少 2-3 页，**值不值看你想冲什么 score**。

## D. 工艺要点（无论哪一层）

### 1. 一个 theorem 配一个 remark

theorem 的 conclusion 必须紧跟 remark 解释意义。reviewer skim 时只读 remark；详读时才看公式。

### 2. 用 *Definition* 给关键概念正名

```
Definition 1. (Predictability Gap)
The predictability gap of a workload π is
    Δ(π) := miss_rate(π_LRU) - miss_rate(π*)
where π* is the offline-optimal policy.
```

Definition 让后面 theorem 的陈述变干净。

### 3. proof sketch 要"3 步以内"

```
Proof sketch. (1) ... (2) ... (3) ...
```

每步 1 句话。完整 proof 进 appendix。

### 4. 避免"理论装饰"

不要为了显得严谨而塞一个和 method 没关系的 theorem。**reviewer 一眼能看出来 theorem 是不是 load-bearing**。如果 theorem 删掉论文也照样成立，那它就是装饰，删掉。

### 5. 把 theory 的 *insight* 翻译成 algorithm 的 *训练目标*

MDD 范本就是这样：theorem 给 bound，algorithm 在 §5 把 bound 的 right-hand side 当作 empirical loss 优化。这种"理论 → 算法"对应让论文变成一个有机整体。

Seer 例：

```
Theorem 1 implies that minimizing miss rate ≈ minimizing the predictability
gap weighted by predictor accuracy. We therefore train SeerNet by:

    L_pred = E[ |\hat{d} - d|^2 ]    (Eq. 7)

minimizing the prediction error directly, which is the dominant term in
the bound of Theorem 1.
```

这一段把 theorem → loss 对应起来，让 reviewer 觉得"这个 loss 不是拍脑袋而是 derived"。

## E. Seer Theory 章建议（综合）

我推荐 Seer 走 **层 1 + 层 2 的混合**：

- **§4.1** 末段 / **§4.5**：放 **Observation 1**（predictability empirical 图）+ **Proposition 1**（Pareto improvement under predictability）
- **Appendix C**：1-2 页给 Proposition 1 完整证明 + 给一个简单的 toy regret bound（Theorem A.1）作为 supplementary 加分项

这样**主文不重 theory，但 reviewer 翻 appendix 能看到 5 页严谨内容**。NeurIPS 评分上"Soundness" 和 "Significance" 都会受益。
