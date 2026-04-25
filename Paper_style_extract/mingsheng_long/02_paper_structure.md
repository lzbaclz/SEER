# 02 · 章节骨架

## A. NeurIPS / ICML / ICLR 模板（9 页正文 + appendix）—— 时序系列主战场

```
1  Introduction                       (1.0–1.3 页)
2  Related Work                       (0.4–0.6 页)         ← NeurIPS 偏好放前面
3  Preliminary / Problem Setup        (0.3–0.5 页)         ← 短，列符号
4  〈Method NAME〉                     (2.0–3.0 页)
   4.1 〈Insight / motivation 子节〉
   4.2 〈Component A〉                                       ← 比如 Auto-Correlation
   4.3 〈Component B〉                                       ← 比如 Series Decomp
   4.4 Architecture / Overall pipeline
5  Experiments                        (3.0–3.5 页)
   5.1 Setup                                                ← 数据集 / baselines / metrics / protocols
   5.2 Main Results
   5.3 Model Analysis / Ablation
   5.4 Showcase / Visualization                             ← 几乎每篇都有，配 attention map / forecast plot
6  Conclusion                         (0.15–0.25 页)        ← 极短
   References
   Appendix (任意长度)：proofs, hyperparameters, more datasets, more visualizations
```

特点：

- **Related Work 放在 §2** 是 THUML 的常态（不像 USENIX 那样放最后）。
- §3 Preliminary 通常**只有半页**：一段 setup + 一个 notation 表。
- §4 Method 是**整篇核心**，子节按 component 切；每个 component 半页到一页。
- §5 Experiments 几乎一定包含 **Showcase 子节**（attention map / forecasting curve / period detection），用 1-2 张视觉图收尾。
- §6 Conclusion 永远短：3-5 句，**不写 future work**。Future work 放 broader impact / limitations 章节里。

## B. CVPR 模板（8 页正文 + 1-2 页 ref）—— 迁移学习视觉系列

```
1  Introduction
2  Related Work
3  Approach / Method
   3.1 Problem Formulation
   3.2 〈Method NAME〉
   3.3 Optimization / Training
4  Experiments
   4.1 Setup
   4.2 Comparison with State-of-the-Art
   4.3 Ablation
   4.4 Analysis / Visualization (e.g. t-SNE)
5  Conclusion
```

特点：

- §3 通常先有 *Problem Formulation* 子节，然后才进入方法（CVPR 偏好这种"先形式化再讲做法"）。
- 实验里几乎一定有 **t-SNE 可视化** + **混淆矩阵**（迁移学习 community 的 convention）。
- DA 论文标配 benchmark：Office-31 / Office-Home / VisDA-2017 / DomainNet。

## C. ICML 长版（理论重）—— MDD 是范本

ICML 给到 8 页正文，但 *Bridging Theory and Algorithm for Domain Adaptation* 这种偏理论的论文会专门腾出一章给 theory：

```
1  Introduction
2  Background
   2.1 Notations
   2.2 Domain Adaptation Theory
3  Theory: Margin Disparity Discrepancy
   3.1 Definition
   3.2 Generalization Bounds (Theorem 1)
   3.3 Rademacher Complexity Analysis (Theorem 2)
4  Algorithm: From Theory to Practice
   4.1 Empirical MDD
   4.2 Adversarial Optimization
   4.3 Implementation
5  Experiments
6  Discussion / Conclusion
   Appendix A: Proofs
```

特点：

- **§3 给定义 + theorem，§4 从 theorem 退到算法**。"Bridging" 这个动作就是一节理论 + 一节算法的并列结构。
- Proof sketches 在主文，full proofs 在 appendix。
- §4.1 的 empirical MDD 通常**正好是 §3 的目标在样本上的近似**。这个对应关系明确写出来。

## D. arXiv 长版 / Journal extension —— Timer / Sundial 这种 foundation model

新一代 foundation model 论文（Timer, Timer-XL, Sundial）页数变大，结构更接近 LLM paper：

```
1  Introduction
2  Related Work / Time Series Foundation Models
3  Method
   3.1 Data: TimeBench / scale claim
   3.2 Format: 〈S3 / TimeFlow〉 unified representation
   3.3 Architecture: GPT-style decoder / Encoder-Decoder
   3.4 Pre-training Objective
4  Experiments
   4.1 Pre-training Setup (datasets, scale, FLOPs)
   4.2 Zero-shot Forecasting
   4.3 Few-shot / Fine-tuning
   4.4 Scaling Laws
   4.5 Probabilistic Evaluation (for Sundial)
5  Conclusion
```

特点：

- §3 一定有 *数据 / 格式 / 架构 / 目标* 四个子节并列，类似 LLM paper 的 "Data → Tokenizer → Architecture → Objective"。
- §4 一定有 *zero-shot* + *few-shot* + *scaling* 三件套；**zero-shot 主图通常是论文最重要的图**，用大柱状图 / heatmap 表达"我没在这数据上 train 也很强"。
- Scaling laws 子节常常画 log-log 曲线（参数 / 数据规模 vs 误差），借鉴 GPT 风格。

## E. NeurIPS 9 页里 Method 子节怎么编（最常用模板）

NeurIPS 主战场的 method 章需要既详尽又紧凑。THUML 的常用排法：

```
4  〈Method NAME〉

4.1 Motivation / Insight
    1 段：把 §1 的痛点收紧成 1-2 句结论性陈述。
    1 段：给 *(Observation 1)* 之类的 italic 标注，作为 insight statement。

4.2 〈Component A〉                                ← 反转 / 替换 prior 的核心模块
    1-2 段 + 1-2 个 numbered equations + 1 张子图
    末尾 1 句：把 component A 的"作用"显式写出来。

4.3 〈Component B〉                                ← 第二个 design primitive
    同上

4.4 Architecture / Overall Pipeline
    一段把 A、B 怎么组装画出来；架构图（Fig 2 或 Fig 3）。
    一段写 training objective（loss 公式）。

(可选) 4.5 Theoretical Analysis / Justification
    1-2 段：观察 / 引理 / 定理（旧线必备，新线 nice-to-have）。
```

每个 component 半页到一页是节奏。**整个 §4 不超过 3 页**。

## F. 常见的"小章节"——容易被忽视但 reviewer 喜欢的位置

- *Discussion / Limitations* 子节：一段 honest 地说"我们方法在 X 情况下不行，Y 情况下需要更多算力"。NeurIPS 现在强制要求，THUML 的写法往往最干净——**不超过 5 句，但每句具体**。
- *Broader Impact*：通常很短，1 段。强调"应用价值大、风险低"（时序 / DA 论文容易）。
- *Reproducibility*：在 main paper 末尾一段或 appendix 开头一段，列 GitHub URL、checkpoint URL、训练 script、随机种子设置。

## G. Appendix 怎么用

THUML 的 appendix 不是垃圾桶，是一个**完整的 supplementary paper**：

```
A  Implementation Details
   A.1 Hyperparameters (table)
   A.2 Hardware (GPUs / time)
   A.3 Data Preprocessing
B  Additional Experiments
   B.1 More datasets
   B.2 More baselines
   B.3 Additional ablations
C  Theoretical Proofs (旧线)
   C.1 Proof of Theorem 1
   C.2 Proof of Lemma 2
D  Visualization Showcase
E  Broader Impact / Limitations Detailed
```

主文里看不到的细节都在这；reviewer 翻 appendix 看到 30 页扎实内容会信任度上升。**Seer 的 appendix 至少要做到 D 这一块**。

## Seer 的 NeurIPS 9 页骨架建议

```
1  Introduction
2  Related Work
   - LLM serving systems & KV management
   - Cache replacement & learning-augmented data structures
   - Workload prediction for caching
   - Prefetching in deep learning systems
3  Preliminaries
   3.1 KV-cache and inference workflow
   3.2 Notation table (block, hit ratio, eviction, prefetch, …)
4  Seer
   4.1 Motivation: Predictability of Block Reuse  (Observation 1)
   4.2 Reuse-Distance Predictor (SeerNet)         ← Component A
   4.3 Pareto-Aware Eviction Policy               ← Component B
   4.4 Look-ahead Prefetch Scheduler              ← Component C
   4.5 Architecture & Training Objective
5  Experiments
   5.1 Setup
   5.2 Main Results: Hit Ratio & TTFT
   5.3 Generalization across Models / Workloads
   5.4 Ablation: Effect of each component
   5.5 Showcase: Eviction visualization on a long-context trace
6  Conclusion
```

Appendix：实现细节 + 更多 ablation + 更多 trace + 训练曲线 + limitations + broader impact。
