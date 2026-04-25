# 03 · 摘要 + 引言模板

## 摘要五段式（时序 / foundation-model 系列）

```
[1] 应用陈述 (1-2 句)：在 〈具体应用 X / Y / Z〉 中，〈某个任务〉 越来越关键。
[2] 现有路径 + 隐藏限制 (1-2 句)：〈prior 方法〉 已被广泛使用，但是 〈隐藏的 / 反直觉的瓶颈〉 限制了它们。
[3] 我们的提议 (1-2 句)：In this paper, we propose 〈NAME〉, a 〈类别〉 that 〈核心反转 / insight〉。〈一句话讲 component〉。
[4] 任务通用化 (0-1 句，可选)：〈NAME〉 同时支持 〈4-5 个任务〉 / 嫁接到 〈n 个 backbone〉。
[5] 量化结论 (1 句)：On 〈6-10 数据集〉, 〈NAME〉 achieves state-of-the-art with 〈X%〉 improvement over 〈12-15 baselines〉. Code is available at 〈GitHub URL〉。
```

### 范例对照

**Autoformer (NeurIPS 21)** — 从公开 abstract 提取：

> [1] "Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning."
> [2] "Prior Transformer-based models adopt various self-attention mechanisms ... However, intricate temporal patterns ... prohibit the model from finding reliable dependencies. ... resulting in the **information utilization bottleneck**."
> [3] "We design Autoformer as a novel decomposition architecture with an **Auto-Correlation mechanism**. We break with the pre-processing convention of series decomposition and **renovate it as a basic inner block** of deep models."
> [4] (后文)"Autoformer is task-general for various practical applications."
> [5] "Autoformer yields state-of-the-art accuracy, with a **38% relative improvement** on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease."

**TimesNet (ICLR 23)**：

> [1] "Time series analysis is of immense importance in extensive applications..."
> [2] "Previous methods attempt to accomplish this directly from the 1D time series, which is extremely challenging due to the intricate temporal patterns. ... we **ravel out** the complex temporal variations into the multiple intraperiod- and interperiod-variations."
> [3] "We extend the analysis of temporal variations into the 2D space ... we propose the **TimesNet** with **TimesBlock** as a task-general backbone..."
> [4] "...consistent state-of-the-art in **five** mainstream time series analysis tasks..."
> [5] (具体数字 + GitHub)

**Non-stationary Transformers (NeurIPS 22)** — 五段映射特别清晰：

> [1] "Transformers have shown great power in time series forecasting ..."
> [2] "Their performance can degenerate terribly on non-stationary real-world data ... we name this problem **over-stationarization**, which leads Transformers to generate **indistinguishable temporal attentions** for different series."
> [3] "We propose **Non-stationary Transformers** with two interdependent modules: **Series Stationarization** and **De-stationary Attention**."
> [4] "...generic framework, plug-and-play to mainstream Transformer-based forecasters."
> [5] "Reduces MSE by **49.43%** on Transformer, **47.34%** on Informer, and **46.89%** on Reformer."

## 摘要五段式（迁移学习 / 理论系列）

老线的摘要稍微换味道，**多一段 theory bridge**：

```
[1] 领域陈述 + 应用 (1 句)
[2] 现有方法的局限 (1-2 句)，命名一个限制 (e.g., "negative transfer", "under-matching")
[3] 我们提议 (1-2 句)：In this paper, we propose 〈NAME〉, a 〈类别〉 grounded on 〈theory〉。
[4] Theory bridge (1 句，可选但加分)：The proposed method comes with 〈generalization bound / theoretical guarantee〉。
[5] 实验 (1 句)：On 〈benchmark suite〉, 〈NAME〉 achieves state-of-the-art with 〈X%〉。
```

### 范例

**MDD (ICML 19)**：

> [1] "Domain adaptation is critical for success in new, unseen environments. Adversarial adaptation models applied in feature spaces discover domain invariant representations..."
> [2] "However, by an **adversarial feature alignment**, several disconnections still exist between domain adaptation theory and algorithm."
> [3] "We propose a novel divergence, **Margin Disparity Discrepancy**, that is tailored to the distribution comparison with the asymmetric margin loss, and to the minimax optimization for easier training."
> [4] "We provide **margin-aware generalization bounds based on Rademacher complexity**, revealing the trade-off between generalization error and margin."
> [5] "...achieves state-of-the-art accuracies on several challenging real-world tasks."

**CDAN (NeurIPS 18)**：

> [1] "Adversarial learning has been embedded into deep networks to learn disentangled and transferable representations for domain adaptation."
> [2] "However, existing adversarial domain adaptation methods may struggle to align different domains of multimodal distributions native in classification problems."
> [3] "We present **conditional domain adversarial networks (CDANs)**, a principled framework that conditions adversarial adaptation models on discriminative information ... with two novel conditioning strategies: **multilinear conditioning** ... and **entropy conditioning**."
> [4] "...exceed state-of-the-art results on five datasets."

**Universal Domain Adaptation (CVPR 19)**：

> [1] "Domain adaptation aims to transfer knowledge in the presence of the domain gap."
> [2] "Existing methods rely on rich prior knowledge about the relationship between the label sets of source and target domains, which greatly limits their application in the wild."
> [3] "This paper introduces **Universal Domain Adaptation (UDA)** ... which requires no prior knowledge on the label sets. ... We propose **Universal Adaptation Network (UAN)** that quantifies sample-level transferability ..."
> [4] (任务通用化的暗示：universal)

## Introduction 七段式（推断 + 多篇互证）

```
§1.1 (隐式) 应用驱动开题
       1 段：3-5 个真实应用名 + 引文 + 工业部署事实

§1.2 现状梳理
       1 段：把 prior approaches 分成 2-3 类，给每类一个名字

§1.3 *命名隐藏瓶颈或冲突*  ← 这是 THUML 的招牌
       1 段：用一个具体的"反直觉"现象暴露 prior 的失效。常常配 §2 里的 motivation 子图。
       
§1.4 我们的核心 insight
       1 段：一句"翻转":"What if we ...?"。给出 method 的 high-level 概念名。

§1.5 我们的 method 概述
       1-2 段：给系统名 / 子组件名，每个 component 用 1-2 句解释。

§1.6 显式 contributions list
       通常是 bullet list 或 numbered list，3-5 条。每条动词开头。

§1.7 (可选) 路线图
       "The rest of the paper is organized as follows..." — NeurIPS 越来越省略，但 ICML/CVPR 仍然常见。
```

### Contributions 段的实例（推断）

THUML 喜欢这种 4-条结构（参考 Non-stationary、TimesNet 等）：

```
Our contributions are summarized as follows:
 • We identify the problem of 〈hidden bottleneck name〉 in 〈existing methods〉 ...
 • We propose 〈SYSTEM NAME〉, a 〈category〉 that addresses this by 〈insight〉.
 • We design 〈Component A〉 and 〈Component B〉 ...; 〈系列论文也会强调 plug-and-play〉.
 • Extensive experiments on 〈n datasets / m tasks〉 demonstrate 〈X% improvement〉, 
   establishing new state-of-the-art on 〈benchmarks〉.
```

### 高频金句 / 句式

下面这些短语在 THUML 论文里反复出现（中等置信度，落笔时建议抽样翻 PDF 二次确认）：

- "**Extending / improving X is a critical demand** for real applications such as ..."
- "**However, the intricate / complex / non-stationary nature** of [domain] **prohibits / impedes / degrades** [prior method]..."
- "We **break with** [convention] and **renovate** it as ..."
- "Inspired by [classical theory], we propose ..."
- "We **bridge** the gap between theory and algorithm by ..."
- "Notably, [METHOD] is a **task-general** backbone / **plug-and-play** framework / **generic** module."
- "Code is available at https://github.com/thuml/[REPO]"
- "We provide **margin-aware generalization bounds based on Rademacher complexity**" (MDD-style)
- "Our findings reveal that ..."
- "...with **N% relative improvement** on **M benchmarks** covering **K applications**."

## Seer 摘要 + §1 落地模板

填空（在 [`11_apply_to_seer.md`](11_apply_to_seer.md) 还会再细化一次）：

```
Abstract:
[1] Long-context LLM serving relies on KV-cache to amortize prefill cost across decoding,
    which is critical for applications such as multi-turn dialog, RAG, and code agents.
[2] Existing eviction policies (LRU, FIFO, attention-score-based) operate on a single
    aggregated signal and treat eviction as independent of prefetch. We identify that
    this leads to <NAMED HIDDEN BOTTLENECK, e.g. "reuse-distance blindness">,
    where blocks with imminent re-use are evicted while seldom-used blocks are retained.
[3] In this paper, we propose Seer, a learned KV-cache management framework that
    JOINTLY performs eviction and prefetch from a unified reuse-distance prediction.
    Seer consists of three components: <SeerNet, the per-block predictor>, 
    <Pareto-Aware Eviction>, and <Look-ahead Prefetch Scheduler>.
[4] Seer is plug-and-play: it requires no model changes and integrates as a runtime
    middleware into vLLM and SGLang.
[5] On <N traces × M models>, Seer reduces p99 TTFT by <X%>, improves throughput by
    <Y×>, and lifts hit ratio by <Z%> over <12 baselines>. Code at github.com/.../seer.

§1 Outline:
  para 1: applications (RAG / dialog / code agents) and KV-cache importance
  para 2: prior approaches taxonomy (eviction-only / prefetch-only / static budgeting)
  para 3: NAME the hidden problem (reuse-distance blindness) — show motivation chart
  para 4: insight — predictability of reuse + joint optimization
  para 5: Seer overview, three components named
  para 6: contributions list (4-5 bullets)
  para 7: (optional) paper roadmap
```

把这套 outline 在你脑子里跑一遍，五分钟就能写出 §1 框架的第一稿。
