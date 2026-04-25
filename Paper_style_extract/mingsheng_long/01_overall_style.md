# 01 · 总体风格画像

## 一句话定义

**"理论扎实 + 工程精致 + 命名鲜明 + 一图杀人 + 大规模验证"** 的标杆型 ML 论文。把它和 Patrick Lee 组对比的话：Patrick 偏 systems engineering（"我嫁接到 HDFS 干掉 X% 流量"）；THUML 偏 ML craftsmanship（"我命名一个反直觉问题、给一个干净的方法、6 任务 10 数据集 12 baseline 全打过去"）。

## 核心 DNA（高置信度，多篇互证）

### 1. 永远从 *应用* 切入，不从 *算法* 切入

Abstract 第一句几乎不会是 "Time series forecasting is an important problem"。而是：

- *Autoformer*: "Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning."
- *TimesNet*: "Time series analysis is of immense importance in extensive applications, such as weather forecasting, anomaly detection, and action recognition."
- *Non-stationary Transformers*: 从 Transformer "global-range modeling ability" 切入。

引子里**点名 3-5 个真实应用**，是 THUML 的 trademark；让 reviewer 立刻觉得"这是为现实服务的工作"。

### 2. **命名一个反直觉的隐藏问题或瓶颈**

不写 "existing methods are suboptimal"。而是发明一个名字让问题变得显眼：

- "**information utilization bottleneck**" (Autoformer，对 sparse attention 的批评)
- "**over-stationarization**" (Non-stationary Transformers，对归一化的批评)
- "**meaningless attention maps**" (iTransformer，对 temporal token fusion 的批评)
- "**ruptured temporal variations**" (SimMTM，对随机 mask 的批评)
- "**under-matching**" (CDAN，对纯 feature alignment 的批评)
- "**negative transfer / partial label space mismatch**" (PADA / UDA)

> **这是 THUML 最容易偷的招**：与其说"现有方法不好"，不如**发明一个名字**给问题，让 reviewer 觉得这个问题之前没人意识到。

### 3. 核心贡献往往是一次"概念反转"

Method 不是"加一个新模块"，而是**让某个东西颠倒过来**：

- Autoformer：series decomposition 从 *pre-processing* → *inner block*
- TimesNet：从 *1D 时序* → *2D tensor (intra-period × inter-period)*
- iTransformer：tokens 从 *temporal* → *variate*（"i" = inverted）
- Non-stationary：从 *stationarize-then-forecast* → *stationarize + 学习去 destationarize attention*
- SimMTM：random masking → *manifold-aware neighbor aggregation*
- MDD：从 *0-1 loss bounds* → *margin loss bounds*（更紧、可微）

读者读到 §3 的时候有一种"啊原来要这样想"的顿悟感。**这是 THUML 让 reviewer 给 strong accept 的关键**。

### 4. 一图杀人 (One-Figure Motivation)

每篇都有一张 Fig. 1 把 method 的核心 insight 画清楚：

| 论文 | Fig 1 是什么 |
|---|---|
| Autoformer | point-wise attention vs period-based auto-correlation |
| TimesNet | 1D 时序 → 2D tensor（行=interperiod，列=intraperiod）的折叠 |
| iTransformer | temporal token vs variate token 的对比 |
| Non-stationary | 归一化前后 attention map 完全相同 vs 多样 |
| SimMTM | manifold + off-manifold neighbor aggregation 的几何示意 |
| Timer | 多种异构序列被映射到 S3 (Single-Series Sequence) 格式 |
| MDD | margin disparity discrepancy 在 hypothesis space 里的几何含义 |

**没有一张能让 reviewer 看完 Fig 1 就懂大意的图，THUML 一般不投**。

### 5. Hyphenated Capitalised 命名学

子组件 / 概念都被命名为"形容词-名词"（hyphenated）+ Capitalized 形式，作为论文里的"专名"全篇大写：

- *Auto-Correlation Mechanism* / *Series Decomposition Block* (Autoformer)
- *Temporal 2D-Variation* / *intraperiod-variations* / *interperiod-variations* / *TimesBlock* (TimesNet)
- *De-stationary Attention* / *Series Stationarization* (Non-stationary)
- *Multilinear Conditioning* / *Entropy Conditioning* (CDAN)
- *Margin Disparity Discrepancy* (MDD)
- *Universal Adaptation Network (UAN)* / *sample-level transferability* (UDA)

整篇论文这些专名**绝不变体**：不会前面 "auto-correlation"、后面 "Auto-Correlation"。

### 6. **量化的胜利**：百分比 / 倍数 / 排名总要写出来

- Autoformer: "**38%** relative improvement on six benchmarks"
- Non-stationary: "**49.43%** MSE reduction on Transformer, **47.34%** on Informer, **46.89%** on Reformer"
- FEDformer: "**14.8%** and **22.6%** for multivariate and univariate"
- Timer: pre-trained on **1B time points**

不写 "significant"。要么给 % 要么给绝对数字。**range** ("47-50%") 也常见，暗示"我们跑了一族 setting"。

### 7. **任务通用化** (task-general framing)

很多 THUML 时序论文不再"只解 forecasting"，而是同一个 backbone 横扫多个任务：

- TimesNet：同时解 forecasting / imputation / classification / anomaly detection 4 任务
- Timer：把 forecasting / imputation / anomaly detection 统一成 generative next-token prediction
- Non-stationary Transformers：同一个 module 嫁接到 Vanilla Transformer / Informer / Reformer
- MDD：理论同时支持二分类 / 多分类 / 回归

> **这给 reviewer 的信号是**："不是某个 narrow 问题里的小招，而是基础工具"。Seer 借鉴：把 Seer 写成不仅能 prefill / decode 都用，也能在不同模型 / 不同硬件 / 不同 workload 上 work。

### 8. 大规模、对齐过的实验：**6-10 数据集 / 12-15 baseline**

Patrick 组爱"3 baselines + 6 traces"；THUML 是"15 baselines + 8 数据集"。每篇主表通常长这样：

```
                     ETTh1   ETTm1   ECL   Weather  Traffic   ILI    Exchange   …
Method A             …       …       …     …        …         …      …
Method B             …       …       …     …        …         …      …
…
Method N (ours)      bold    bold    bold  …        bold      …      bold
```

数据集是"Time-Series-Library 标准 8 件套"，baseline 是 "近 2 年同领域 12-15 篇"。这是 NeurIPS reviewer 一眼就喜欢的格式。

### 9. 显式的 Theory-Algorithm Bridge（旧线特色）

老的迁移学习线 **每篇至少给一个 theorem**：

- DAN：MMD 的统计保证 + 多核选择
- JAN：JMMD（joint MMD）的 closed-form 公式
- CDAN：generalization error bound
- MDD：完整的 Rademacher complexity 推导，theorem 配 lemma 配 proof sketch
- UAN：transferability 的 quantification

新的时序线**普遍不证 theorem**，但用 *signal-processing theory*（FFT / 周期性 / 自相关）和 *stochastic process theory* 做"理论灵感来源"，不严格证明，但显得有出处。

### 10. 开源、可复现，**且统一**到一个 library

- Time-Series-Library: https://github.com/thuml/Time-Series-Library — Autoformer/TimesNet/iTransformer/Informer/PatchTST 都在里头
- Transfer-Learning-Library: https://github.com/thuml/Transfer-Learning-Library — DAN/JAN/CDAN/MDD/UDA/PADA 都在里头
- OpenLTM (Large Time-series Models)：Timer / Sundial 体系
- 每篇论文的代码 GitHub link **在 abstract 或 §1 末尾就出现**（不是放 conclusion）

> 把"我们组的所有 baseline 都在同一个 framework 下跑"作为 trust signal。reviewer 不再怀疑"是不是你们家比别人家强是因为别人家代码烂"。

## 共同的"不做"清单

- 不用 "novel"、"first time"、"unprecedented" 这种自夸词
- 不写 "we believe"、"we conjecture"——要写就 "we observe"、"we show"、"we demonstrate"
- 不让 method 章超过 5 个子节（保持脉络清晰）
- 不让主表跨页（**用更小的字号 / 缩写也要塞进一页**，便于扫读）
- 不在 abstract 里写未来工作

## 共同的"必做"清单（Abstract+§1 收尾时所有问题都答完）

1. 我们解决什么真实场景下的问题？
2. 这个场景下有什么 *隐藏的* 瓶颈？给它命个名。
3. 我们的核心做法是什么 *概念反转* / *insight*？
4. 我们怎么具体实现？子模块叫什么？
5. 在多少数据集 / 多少 baseline 上验证？
6. 提升多少（百分比 / 倍数）？
7. 是否覆盖多任务 / 多场景以体现通用性？
8. 是否有理论保证 / 灵感（旧线必有，新线 nice-to-have）？
9. 代码 / checkpoint 在哪儿？
10. 这是哪个系列论文里的位置？（很多 THUML 论文会暗暗 link 到组里前作，比如 Autoformer→FEDformer→Non-stationary 是个 saga）

把这 10 项当 Seer §1 的 outline checklist。
