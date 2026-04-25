# 10 · 9 篇代表论文的逐段拆解

下面对 9 篇 THUML 论文按 abstract → §1 → method → eval 拆解。
落笔时直接对照同类论文找句式与节奏。

> **注**：abstract 直接引用的句子来自论文公开页面（NeurIPS / ICML / ICLR / CVPR / arXiv）；段落级别推断用 *(inferred)* 标记，建议落笔时翻 PDF 二次确认。

---

## 10.1 Autoformer (NeurIPS 2021)

**作者**：Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long
**意义**：开启 THUML 时序系列的旗舰作；后续 TimesNet / iTransformer 大量引用为 baseline。

### 摘要五段映射

> [1] "Extending the forecasting time is a critical demand for real applications, such as **extreme weather early warning** and **long-term energy consumption planning**."
> [2] "Prior Transformer-based models adopt various self-attention mechanisms ... However, intricate temporal patterns ... prohibit ... **resulting in the information utilization bottleneck**."
> [3] "We design **Autoformer** as a novel decomposition architecture with an **Auto-Correlation mechanism**. We **break with the pre-processing convention** of series decomposition and **renovate it as a basic inner block** of deep models."
> [4] "Autoformer is task-general for various practical applications."
> [5] "Autoformer yields state-of-the-art accuracy, with a **38% relative improvement** on six benchmarks, covering five practical applications: **energy, traffic, economics, weather and disease**."

### §1 走向 *(inferred from abstract pattern)*

1. 应用驱动开题：极端天气、能源规划等长期预测的实际需求。
2. Transformer-based 时序方法发展史：sparse attention 一脉。
3. **命名瓶颈**：information utilization bottleneck。
4. 我们的反转：series decomposition 从 pre-processing → inner block。
5. Method 概述：Auto-Correlation Mechanism + Series Decomposition Block。
6. Contributions list（4 条左右）。

### Method 章

```
3.1 Series Decomposition Block
3.2 Auto-Correlation Mechanism
3.3 Encoder-Decoder Architecture
```

§3.2 是核心，给出基于 FFT 的 lag 选取 + Time-Delay Aggregation 公式。

### Evaluation

- 6 数据集 × 14-15 baselines
- 主表：Multivariate forecasting on ETTh1, ETTm1, ECL, Weather, Traffic, ILI, Exchange
- Ablation：decomposition block on/off, auto-correlation vs self-attention
- Showcase：forecast curve true vs predicted

### 给 Seer 的复用要点

- "把 prior 的 pre-processing 步骤变成 inner block" 这个反转 → Seer 可以"把 eviction 从 *访问后* 决策变成 *访问前* 决策"。
- 5 个应用领域命名 → Seer 可以列 "dialog / RAG / code / agent / longform"。

---

## 10.2 TimesNet (ICLR 2023)

**作者**：Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long
**意义**：把"时序" benchmarking 一举拓展到 5 个任务的"task-general" 范式。

### 摘要五段

> [1] "Time series analysis is of immense importance in extensive applications, such as **weather forecasting, anomaly detection, and action recognition**."
> [2] "Previous methods attempt to accomplish this directly from the **1D time series**, which is extremely challenging due to the intricate temporal patterns. ... we **ravel out** the complex temporal variations into the multiple **intraperiod- and interperiod-variations**."
> [3] "We extend the analysis of temporal variations into the 2D space ... we propose the **TimesNet** with **TimesBlock** as a task-general backbone for time series analysis."
> [4] "Achieves consistent state-of-the-art in **five mainstream time series analysis tasks**, including short- and long-term forecasting, imputation, classification, and anomaly detection."

### Method 走向

```
3.1 Multi-Periodicity Analysis (FFT 找 top-K 周期)
3.2 1D-to-2D Transformation (按周期折叠)
3.3 TimesBlock: Inception-style 2D conv processing
3.4 Task-General Architecture
```

### Evaluation 特点

- **5 个任务各一张主表**：forecasting / imputation / classification / anomaly detection / short-term forecasting
- 每张表 12-15 baselines

### 给 Seer 的复用要点

- "task-general backbone" → Seer 可以包装成 "general KV management" 而不是 "eviction policy"，让 reviewer 看到 scope。
- TimesBlock 的 block 后缀命名 → SeerBlock？或者 SeerNet。

---

## 10.3 iTransformer (ICLR 2024 Spotlight)

**作者**：Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long
**意义**：极致简洁的"概念反转"范本，整篇贡献能用 Fig 1 一图说尽。

### 摘要五段

> [1] "Recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters."
> [2] "Transformers are challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. ... the embedding for each temporal token fuses multiple variates ... result in **meaningless attention maps**."
> [3] "We propose **iTransformer** that **simply applies the attention and feed-forward network on the inverted dimensions**. The time points of individual series are embedded into **variate tokens** ..."
> [4] (后文)"...with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows."

### Method 走向

```
3.1 Limitations of Temporal Tokenization
3.2 Inverted Tokenization (variate-as-token)
3.3 iTransformer Architecture
3.4 Generalization to Arbitrary Variates / Lookbacks
```

### 给 Seer 的复用要点

- "**i 前缀** 即点出核心 insight" → Seer 可以考虑给关键 component 起一个 "j-Eviction"（joint）或类似带前缀的名字。
- "我们没改 attention，只是 inverted token"（minimum-change framing）—— Seer 可以强调 "we don't modify the model, only manage the cache"。
- 一图杀人：Fig 1 把整篇 idea 讲完——Seer 务必做出这种图。

---

## 10.4 Non-stationary Transformers (NeurIPS 2022)

**作者**：Yong Liu, Haixu Wu, Jianmin Wang, Mingsheng Long
**意义**：发明问题名 ("over-stationarization") 的范本。

### 摘要五段

> [1] "Transformers have shown great power in time series forecasting due to their global-range modeling ability."
> [2] "Their performance can degenerate terribly on non-stationary real-world data ... Previous studies primarily adopt stationarization to attenuate non-stationarity ... But the stationarized series deprived of inherent non-stationary information can be less instructive ..."
> [3] (problem naming) "...termed **over-stationarization** ..."
> [4] "We propose **Non-stationary Transformers** with two interdependent modules: **Series Stationarization** and **De-stationary Attention** ..."
> [5] "Reduces MSE by **49.43%** on Transformer, **47.34%** on Informer, and **46.89%** on Reformer."

### Method 走向

```
4.1 Analysis of Over-stationarization
4.2 Series Stationarization
4.3 De-stationary Attention
4.4 Putting Things Together
```

§4.1 是 THUML 标志写法："**method 章先用半页分析 prior 的失败模式**"。

### 给 Seer 的复用要点

- **创造一个问题名字**：建议 Seer 用 "Reuse-Distance Blindness" 或 "Eviction-Prefetch Decoupling"。
- **Plug-and-Play framing**：Non-stationary 模块嫁接到 Vanilla / Informer / Reformer 三种 Transformer，每个都涨 47-49%。Seer 可以嫁接到 vLLM / SGLang / TGI 三个 serving 系统，强化"plug-and-play" 信号。
- **配对模块命名**："Series Stationarization" + "De-stationary Attention" 是一对 paired concepts。Seer 可以做 "Pareto-Aware Eviction" + "Look-ahead Prefetch Scheduler" 这种对称命名。

---

## 10.5 SimMTM (NeurIPS 2023 Spotlight)

**作者**：Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, Mingsheng Long
**意义**：把 manifold learning 引入 SSL 时序的范本。

### 摘要五段

> [1] "Self-supervised pre-training has attracted immense interest..."
> [2] "...random masking will seriously ruin vital temporal variations, making the reconstruction task too difficult."
> [3] "By **relating masked modeling to manifold learning**, **SimMTM** proposes to recover masked time points by weighted aggregation of multiple neighbors **outside the manifold**..."
> [4] "...both in- and cross-domain settings."

### 给 Seer 的复用要点

- **Relating X to Y framing**："masked modeling" → "manifold learning"。Seer 可以 "Relating KV-cache management to online learning under reuse-distance bandits"。
- **In-domain + cross-domain**：Seer 在 in-distribution + OOD trace / model 上同时验证。

---

## 10.6 Timer (ICML 2024)

**作者**：Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, Mingsheng Long
**意义**：THUML 进入 foundation model 时代的标志，借 GPT 范式包装时序。

### 摘要五段

> [1] (analogy) "Large language models have demonstrated remarkable capabilities via generative pre-training..."
> [2] "However, the time series modeling has largely relied on task-specific supervised learning..."
> [3] "We propose **Timer**, a generative pre-trained Transformer for general time series analysis. We curate large-scale datasets with up to **1 billion time points**, unify heterogeneous time series into **single-series sequence (S3) format**, and develop GPT-style architecture..."
> [4] "Convert forecasting, imputation, and anomaly detection of time series into a unified generative task."
> [5] "...zero-shot and few-shot learning abilities."

### 给 Seer 的复用要点

- **借 LLM 类比包装**：Seer 可以"borrow LLM serving framing"（"like GPT, but for KV management"）。
- **统一格式**：S3 把异构时序统一成单序列。Seer 可以发明 "Block Trajectory Trace (BTT)" 把异构 workload 统一成 access trace。
- **Zero-shot generalization** 是 NeurIPS 评审 buzzword，论文的 Q1 实验最好是 zero-shot 跨模型 / 跨 trace。

---

## 10.7 DAN (ICML 2015)

**作者**：Mingsheng Long, Yue Cao, Jianmin Wang, Michael I. Jordan
**意义**：THUML 的开山作，把深度学习和 RKHS / MMD 结合。

### 摘要骨架

> [1] "Recent studies reveal that a deep neural network can learn transferable features..."
> [2] "However, ... the feature transferability drops significantly in higher layers with increasing domain discrepancy."
> [3] "We propose a new **Deep Adaptation Network (DAN)** architecture, which generalizes deep convolutional neural network to the domain adaptation scenario. ... hidden representations are embedded in a **reproducing kernel Hilbert space** where the **mean embeddings** of different domain distributions can be explicitly matched. ... The domain discrepancy is further reduced using an **optimal multi-kernel selection method**."

### 给 Seer 的复用要点（对比新线）

- 老线偏 "我们把经典统计工具（kernel mean embedding）搬进深度学习"；新线偏 "我们发明一个反直觉概念"。两种 framing 都可以做 Seer。
- DAN 的 "explicit guarantee + multi-kernel selection" 是工业级的稳重——如果 Seer 走 theory 路线，可以借鉴 DAN 的 framing：把 reuse-distance prediction 视为"online distribution matching"。

---

## 10.8 CDAN (NeurIPS 2018)

**作者**：Mingsheng Long, Zhangjie Cao, Jianmin Wang, Michael I. Jordan
**意义**：从 DANN 升级到"条件对抗"的重要一跃。

### 摘要骨架

> [1] "Adversarial learning has been embedded into deep networks to learn disentangled and transferable representations..."
> [2] "However, existing adversarial domain adaptation methods may struggle to align different domains of multimodal distributions..."
> [3] "...**conditional adversarial domain adaptation**, a **principled framework** that conditions the adversarial adaptation models on **discriminative information** ... with two novel conditioning strategies: **multilinear conditioning** ... and **entropy conditioning**."
> [4] "Theoretical guarantee on the generalization error bound based on domain adaptation theory."

### 给 Seer 的复用要点

- **"Principled framework" framing**：CDAN 不说 "we propose a new method"，说 "we propose a principled framework"。这个词 reviewer 喜欢。
- **两个 conditioning strategies**："multilinear" + "entropy"——配对模块命名再次出现。
- **A theoretical guarantee** 即便是简单的 generalization bound，仍提一句 — Seer 可以照做。

---

## 10.9 MDD (ICML 2019)

**作者**：Yuchen Zhang, Tianle Liu, Mingsheng Long, Michael I. Jordan
**意义**：THUML 老线最理论的论文，是发理论 ML 论文的 reference。

### 摘要骨架

> [1] "Domain adaptation is critical for success in new, unseen environments. ..."
> [2] (gap) "However, ... several disconnections still exist between domain adaptation theory and algorithm."
> [3] "We propose a novel divergence, **Margin Disparity Discrepancy** ..."
> [4] "We provide **margin-aware generalization bounds based on Rademacher complexity**, revealing the trade-off between generalization error and margin."
> [5] "...state-of-the-art accuracies on several challenging real tasks."

### 给 Seer 的复用要点（如果走 theory 路线）

- **"Bridging Theory and Algorithm"** 这个标题模式 — Seer 如果有 theorem，可以叫 "Bridging Online Learning and KV-Cache Management"。
- Theorem + Remark 紧贴算法（empirical loss 是 theorem 的 right-hand side）—— Seer Theorem 1 → SeerNet loss。
- Rademacher complexity 是 ML reviewer 的 comfort term。

---

## 横向对比表

| 维度 | Autoformer | TimesNet | iTransformer | Non-stationary | SimMTM | Timer | DAN | CDAN | MDD |
|---|---|---|---|---|---|---|---|---|---|
| 命名瓶颈 | info utilization bottleneck | 1D inadequacy | meaningless attn | over-stationarization | ruptured manifold | task-specific TS | feature transferability drop | under-matching | theory-algorithm gap |
| 概念反转 | decomp → inner block | 1D → 2D | temporal → variate | stationarize→ destationarize | random → manifold | supervised → generative pretrain | feature → kernel | marginal → conditional | 0-1 → margin |
| Theorem | none | none | none | none | none | none | RKHS bound | gen error bound | full Rademacher |
| 任务数 | 1 (forecast) | 5 | 1 | 1 | 2 | 4 | 1 (DA cls) | 1 | 1 |
| 数据集数 | 6 | 8-10 | 10+ | 6 | 9+ | 5+ | 3-4 | 5 | 5 |
| Baselines 数 | 14-15 | 12-15 | 15+ | 7-8 | 15+ | many | 8-10 | 8-10 | 10+ |
| GitHub | thuml/Autoformer | thuml/TimesNet | thuml/iTransformer | thuml/Nonstationary | thuml/SimMTM | thuml/Large-Time-Series-Model | thuml/DAN | thuml/CDAN | thuml/MDD |

把这一行 *Seer* 填进去，就能看到自己缺哪格。
