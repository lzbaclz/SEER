# THUML Time-Series Writing Style Guide
## Mingsheng Long's Lab (Tsinghua THUML) — Time-Series & Foundation-Model Line

**Date:** April 2026  
**Purpose:** Style observations for Seer (NeurIPS 2026, learned KV-cache eviction/prefetch)  
**Scope:** 7 key papers + GitHub conventions  
**Target Word Count:** ~3400 words of dense analysis

---

## 1. Autoformer (NeurIPS 2021)

### A. Abstract Structure

**[Problem-framing]:** "Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning."

**[Observation]:** "Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck."

**[Proposal]:** "We design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models."

**[Results]:** "Autoformer yields state-of-the-art accuracy, with a 38% relative improvement on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease."

**Pattern:** Opens with domain motivation (real-world critical demand), identifies bottleneck in prior work, then pivots to conceptual fix. Ends with quantified improvements and application diversity.

### B. Naming Convention

- **Technique:** "Auto-Correlation mechanism"
- **Model:** "Autoformer"
- **Sub-components:** "Series Decomposition Block", "Auto-Correlation", "Encoder", "Decoder"
- **Key operation:** "progressive decomposition capacities"

Names are action-oriented and descriptive of *what* the component does, not just what it's called.

### C. Problem Formulation Style

Uses standard time-series forecasting setup without explicit "Problem 1" environment. Instead, informal narrative: *"given a history, predict the future"*. Introduces decomposition philosophy upfront in abstract and early sections rather than delaying to formal problem setup.

### D. Motivation Figure

**Fig. 1 conceit:** Contrasts the failure mode of point-wise self-attention (all-to-all token dependency) vs. the periodicity-based grouping in auto-correlation. Shows that auto-correlation leverages temporal structure (lags/periods) rather than position-based attention. One-figure motivation showing *why* existing Transformers fail on time series (information bottleneck) and the fix (operate at sub-series/period level).

### E. Method Section Organization

**Structure:** 3 subsections:
1. **Series Decomposition Block** — how trend is extracted progressively
2. **Auto-Correlation Mechanism** — frequency-domain attention weights, FFT-based lags
3. **Architecture** — encoder-decoder stacking with decomposition-first pipeline

Each subsection is 2–3 paragraphs with equations embedded naturally. Equations numbered and referenced.

### F. Theoretical Content

**No formal theorems or proofs**, but strong **stochastic-process justification:** "Inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity." References classical time-series theory (periodicity, autocorrelation) to justify method, not to prove novelty.

### G. Evaluation Pattern

**Main results table:** Single large table with all datasets (ETTh1, ETTm1, ECL, Weather, Electricity, ILI) × all baselines (14–15 methods).  
**Ablation:** Separate table isolating decomposition block vs. auto-correlation mechanism.  
**Dataset diversity:** 6 datasets, 5 real-world application domains explicitly mentioned in abstract.

### H. Related Work Organization

**By idea/axis:** "Transformer-based forecasting", "Decomposition methods", "Attention mechanisms". Chronological within each axis.

### I. Distinctive Prose Habits

- **Verb choice:** "We break with... and renovate...", "design", "conduct", "aggregate"  
- **Hedging:** Minimal. States facts as discovered ("intricate temporal patterns... prohibit").
- **Rhythm:** Medium-length sentences (15–20 words). Punctuation-light.
- **Parallel structure:** "dependencies discovery *and* representation aggregation"

### J. Open-source / Reproducibility

**GitHub:** https://github.com/thuml/Autoformer  
**Pattern:** Links always in code release statement. Code published at acceptance.

---

## 2. TimesNet (ICLR 2023)

### A. Abstract Structure

**[Problem-framing]:** "Time series analysis is of immense importance in extensive applications, such as weather forecasting, anomaly detection, and action recognition."

**[Observation]:** "Previous methods attempt to accomplish this directly from the 1D time series, which is extremely challenging due to the intricate temporal patterns. Based on the observation of multi-periodicity in time series, we ravel out the complex temporal variations into the multiple intraperiod- and interperiod-variations."

**[Proposal]:** "We extend the analysis of temporal variations into the 2D space by transforming the 1D time series into a set of 2D tensors based on multiple periods. Technically, we propose the TimesNet with TimesBlock as a task-general backbone for time series analysis."

**[Results]:** "Achieves consistent state-of-the-art in five mainstream time series analysis tasks, including short- and long-term forecasting, imputation, classification, and anomaly detection."

**Pattern:** Opens with application importance, frames problem as representation bottleneck (1D insufficiency), proposes spatial transform (1D→2D), claims broad task-general unification. Notice "immense importance" (intensity language) and "ravel out" (metaphor for disentanglement).

### B. Naming Convention

- **Technique/Model:** "TimesNet"
- **Core block:** "TimesBlock" (named with "Block" suffix, echoing vision literature)
- **Key concepts:** "Temporal 2D-Variation", "intraperiod-variations", "interperiod-variations", "multi-periodicity"
- **Operation:** "parameter-efficient inception block"

Hyphenation is used for compound temporal concepts. "Block" terminology signals CNN/vision backbone adoption.

### C. Problem Formulation Style

No formal "Problem" environment. Instead, frames as an *observation* (multi-periodicity exists) → *insight* (1D insufficient) → *solution* (transform to 2D). Very intuitive, non-symbolic.

### D. Motivation Figure

**Fig. 1 conceit:** **1D-to-2D period folding visualization**. Shows original 1D sequence reshaped into 2D tensors where columns = intraperiod (adjacent time points within a period), rows = interperiod (same phase across periods). This single figure justifies the entire method: columns/rows are *naturally* amenable to 2D convolutions (inception blocks). Highly pedagogical.

### E. Method Section Organization

**Structure:** 4 subsections:
1. **Motivation: Multi-Periodicity** — FFT observation, why 2D helps
2. **TimesBlock Design** — how 2D tensors are processed (Inception module)
3. **Adaptive Period Discovery** — FFT-based period selection
4. **Task Adapters** — how same backbone handles forecasting/imputation/classification

Each subsection includes equations and design choices. Heavy use of notation tables for dimensions (e.g., input shape, period-folded shape, output shape).

### F. Theoretical Content

No theorems. But **principled frequency-domain analysis:** FFT for period detection. Grounds method in signal-processing theory (periodicity) rather than deep-learning theory. Cites classical time-series literature.

### G. Evaluation Pattern

**Task-general unification**: One table per task (5 tasks), each table shows multiple datasets, ~12–15 baselines.  
**Ablation:** "TimesBlock design choices" table (period selection, inception vs. alternatives).  
**Scope:** 8–10 datasets total across tasks, covering energy, weather, traffic, medical domains.

### H. Related Work Organization

By task axis: "Time Series Forecasting", "Time Series Classification", "Time Series Imputation", etc. Within each, chronological and by model family (RNN, CNN, Transformer).

### I. Distinctive Prose Habits

- **Metaphor:** "ravel out", "transform", "extend"
- **Verb:** "propose", "design", "achieve", "unify"
- **Intensity markers:** "immense importance", "consistent state-of-the-art", "extensively"
- **Lists:** Often ≥3 items, parallel structure ("forecasting, imputation, classification, and detection")
- **Rhythm:** Longer sentences (18–25 words) with subordinate clauses

### J. Open-source / Reproducibility

**GitHub:** https://github.com/thuml/TimesNet  
**Pattern:** Code link in abstract or introduction. Tutorial notebooks in repository.

---

## 3. iTransformer (ICLR 2024)

### A. Abstract Structure

**[Problem-framing]:** "Recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters."

**[Observation]:** "Transformers are challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. The embedding for each temporal token fuses multiple variates... which may fail in learning variate-centric representations and result in meaningless attention maps."

**[Proposal]:** "We propose iTransformer that simply applies the attention and feed-forward network on the inverted dimensions. The time points of individual series are embedded into variate tokens..."

**[Results]:** "iTransformer achieves state-of-the-art on challenging real-world datasets... with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows."

**Pattern:** Opens with a counterargument (linear models are strong), identifies architectural flaw (temporal-token fusion), proposes **inversion** (variate-as-token), claims broader generalization. Notice the shift in framing: not adding complexity, but *rethinking* architectural choices.

### B. Naming Convention

- **Model:** "iTransformer" (prefix *i* for *inverted*)
- **Key concept:** "Inverted Transformer", "variate tokens"
- **Operation:** "repurpose the Transformer architecture without any modification to the basic components"

The *i* prefix is memorable, short, and immediately signals the core insight (inversion). "Variate-token" (not "time-point token") is novel terminology anchoring the paper.

### C. Problem Formulation Style

Introduces multivariate forecasting task casually: *"given multivariate series, predict next timestep(s)"*. No formal notation table. Instead, uses prose: "each temporal token formed by multiple variates of the same timestamp... may fail in learning variate-centric representations."

### D. Motivation Figure

**Fig. 1 conceit:** **Temporal-token vs. variate-token comparison**. Shows standard Transformer embedding (all variates at time t → fused token), then iTransformer's inversion (all time points of variate i → variate token). Highlights how inversion naturally aligns with multivariate structure (each variate is independent measurement). One-figure argument for the method.

### E. Method Section Organization

**Structure:** 3 subsections:
1. **Limitations of Temporal Tokenization** — why fusing variates is problematic
2. **Inverted Transformer** — variate tokens, self-attention on variate dimension
3. **Generalization** — forecasting unseen variates, arbitrary lookback windows

Equations are sparse. More conceptual. Heavy use of **dimension notation** (L × M → M × L transforms).

### F. Theoretical Content

No proofs. But **empirical justification** via attention map visualization and generalization experiments. Shows that variate-tokens learn meaningful cross-variate dependencies (unlike temporal-token attention, which is "meaningless").

### G. Evaluation Pattern

**Multivariate forecasting**: Large table, 10+ datasets (15-min to daily granularity), 15+ baselines.  
**Ablation:** "Inverted design" (temporal vs. variate tokens), "lookback window scaling".  
**Generalization**: Zero-shot / cross-variate transfer table (train on subset, test on unseen variates).

### H. Related Work Organization

By approach: "Temporal Attention Methods", "Decomposition Methods", "Recent Linear Methods", "Multivariate Modeling". Shows awareness of the "linear vs. Transformer" debate explicitly.

### I. Distinctive Prose Habits

- **Rhetorical question:** "Why are temporal tokens suboptimal?"
- **Parallelism:** "attention *and* feed-forward network", "performance, generalization *and* utilization"
- **Negation:** "without any modification", "meaningless attention maps"
- **Verb:** "repurpose", "embed", "capture", "learn"
- **Hedging:** Cautious language ("may fail", "can result in")

### J. Open-source / Reproducibility

**GitHub:** https://github.com/thuml/iTransformer  
**Status:** ICLR 2024 Spotlight; implementation widely adopted (GluonTS integration noted).

---

## 4. Non-stationary Transformers (NeurIPS 2022)

### A. Abstract Structure

**[Problem-framing]:** "Transformers have shown great power in time series forecasting due to their global-range modeling ability."

**[Observation]:** "Their performance can degenerate terribly on non-stationary real-world data where the joint distribution changes over time. Previous studies primarily adopt stationarization to attenuate non-stationarity... But the stationarized series deprived of inherent non-stationary information can be less instructive for real-world bursty events forecasting."

**[Challenge/paradox]:** "This problem, termed over-stationarization, leads Transformers to generate indistinguishable temporal attentions for different series and impedes the predictive capability of deep models."

**[Proposal]:** "We propose Non-stationary Transformers with two interdependent modules: Series Stationarization and De-stationary Attention."

**[Results]:** "Reduces MSE by 49.43% on Transformer, 47.34% on Informer, and 46.89% on Reformer."

**Pattern:** Opens with strength (Transformers work), identifies hidden cost (over-stationarization paradox), frames as a dilemma (predictability vs. capability), proposes balanced solution. Notice the **problem naming** ("over-stationarization") — THUML invents terminology to make the problem salient.

### B. Naming Convention

- **Framework:** "Non-stationary Transformers"
- **Modules:** "Series Stationarization", "De-stationary Attention"
- **Problem:** "over-stationarization"
- **Operation:** "recover intrinsic non-stationary information"

Prefix-based naming ("de-stationary") and hyphenated compound problems are characteristic.

### C. Problem Formulation Style

Frames as a methodological critique: "Previous works do X, but X has a hidden cost Y." No formal problem definition. Instead, cites concrete failure modes ("indistinguishable temporal attentions").

### D. Motivation Figure

**Fig. 1 conceit:** Compares attention maps from stationarized vs. raw series. Stationarized series produce **identical** attention patterns across different series (all series treated the same after normalization), while raw-series attention is **diverse**. Shows the dilemma visually: predictability gain = semantic loss.

### E. Method Section Organization

**Structure:** 3 subsections:
1. **Analysis of Over-stationarization** — why stationarization breaks series
2. **Series Stationarization** — the step
3. **De-stationary Attention** — recovering non-stationary info by learning residual attentions

Equations for stationarization (mean/variance normalization) and de-stationary attention (learnable reversals). Heavy notation.

### F. Theoretical Content

**Principled justification:** Cites stochastic-process theory (stationarity assumptions). Proposes that attention should approximate residuals (deviations from stationary baseline), not absolute patterns. Theoretical grounding in time-series econometrics.

### G. Evaluation Pattern

**Boost experiments:** Table showing vanilla Transformer + Informer + Reformer, each with and without non-stationary module.  
**Ablation:** Stationarization alone vs. de-stationary attention alone vs. both.  
**Datasets:** ETT, Weather, Electricity, Traffic, ILI, Exchange (6 standard benchmarks).

### H. Related Work Organization

By concept: "Stationarity in Time Series", "Attention Mechanisms", "Decomposition Methods". Shows THUML's theory-informed approach.

### I. Distinctive Prose Habits

- **Problem naming:** Invents terms to frame the issue ("over-stationarization")
- **Critique structure:** "Paradoxically...", "dilemma between..."
- **Specificity:** Names three baseline architectures with concrete % improvements
- **Symmetry:** "Series Stationarization" *and* "De-stationary" (paired concepts)
- **Verb:** "recover", "attenuate", "approximate"

### J. Open-source / Reproducibility

**GitHub:** https://github.com/thuml/Nonstationary_Transformers  
**Universality claim:** Module is plug-and-play for any Transformer-based forecaster.

---

## 5. FEDformer (ICML 2022)

### A. Abstract Structure

**[Problem-framing]:** "Long-term time series forecasting is challenging since prediction accuracy tends to decrease dramatically with the increasing horizon."

**[Prior approach + limitation]:** "Although Transformer-based methods have significantly improved... they are not only computationally expensive but more importantly, are unable to capture the global view of time series (e.g. overall trend)."

**[Proposal]:** "We propose to combine Transformer with seasonal-trend decomposition... To further enhance performance, we exploit the fact that most time series tend to have a sparse representation in a well-known basis such as Fourier transform, and develop a frequency enhanced Transformer."

**[Computational + empirical results]:** "More efficient than standard Transformer with linear complexity to the sequence length. Reduces prediction error by 14.8% and 22.6% for multivariate and univariate time series, respectively."

**Pattern:** Opens with problem severity (dramatic accuracy drop), identifies two limitations of prior Transformers (cost + semantics), proposes two-pronged fix (decomposition + frequency enhancement). Ends with dual benefits (efficiency + accuracy).

### B. Naming Convention

- **Model:** "FEDformer" (mnemonic: Frequency-Enhanced Decomposed)
- **Components:** "Seasonal-Trend Decomposition", "Frequency Enhanced Transformer"
- **Principle:** "sparse representation in Fourier basis"

Acronym is efficient and memorable. Straightforward component naming.

### C. Problem Formulation Style

Informal. "Given series, predict long-term horizon." Introduces decomposition philosophy upfront: "Decomposition captures global profile while Transformers capture detailed structures." No equations; intuitive narrative.

### D. Motivation Figure

**Fig. 1 conceit:** Shows decomposition of a time series into trend + seasonal components. Illustrates that **global view** (trend) is invisible to local attention mechanisms. Decomposition extracts this separately for processing by different network parts. One-figure argument for divide-and-conquer.

### E. Method Section Organization

**Structure:** 4 subsections:
1. **Seasonal-Trend Decomposition** — classical moving-average baseline
2. **Frequency Enhanced Transformer** — Fourier-basis sparsity insight
3. **Linear Complexity** — how to achieve it (sparse frequency representation)
4. **Architecture** — encoder-decoder with decomposition pathways

Equations for moving-average, Fourier transform, sparse selection. Standard depth.

### F. Theoretical Content

No proofs. **Frequency-domain justification:** cites fact that real time series are sparse in Fourier basis (a standard signal-processing result). Uses this to prune Transformer weights. Lightweight theory, strong intuition.

### G. Evaluation Pattern

**Multivariate vs. univariate split:** Separate best-error tables.  
**Ablation:** Decomposition alone, Transformer alone, both.  
**Efficiency table:** FLOPs, parameters, wall-clock time vs. baselines.  
**Datasets:** 6 standard (ETT, Weather, Electricity, Traffic, ILI, Exchange).

### H. Related Work Organization

By approach: "Decomposition Methods", "Transformer-based Forecasting", "Frequency-Domain Methods". Positions FEDformer at the intersection.

### I. Distinctive Prose Habits

- **Parallelism:** "computationally expensive *but* unable to capture"
- **Authority:** "exploit the fact that", "well-known basis"
- **Benefit stacking:** "More effective, the proposed method... is more efficient"
- **Specificity:** Dual % improvements separated by variance type
- **Verb:** "combine", "exploit", "develop"

### J. Open-source / Reproducibility

**GitHub:** https://github.com/DAMO-DI-ML/ICML2022-FEDformer (note: not THUML GitHub, but Alibaba DAMO lab; co-authored by THUML members)

---

## 6. Timer (ICML 2024)

### A. Abstract Structure

**[Vision statement]:** "Large language models have demonstrated remarkable capabilities via generative pre-training..."

**[Problem-framing]:** "However, the time series modeling has largely relied on task-specific supervised learning..."

**[Proposal]:** "We propose Timer, a generative pre-trained Transformer for general time series analysis. We curate large-scale datasets with up to 1 billion time points, unify heterogeneous time series into single-series sequence (S3) format, and develop GPT-style architecture..."

**[Task unification]:** "Convert forecasting, imputation, and anomaly detection of time series into a unified generative task."

**[Results]:** "Achieves competitive performance across multiple tasks and datasets, with strong zero-shot and few-shot learning abilities."

**Pattern:** Opens with analogy to LLMs (implicit: if it works there, try it here), identifies gap (no pre-trained foundation models for TS), proposes unified framework (S3 format + GPT style), claims broad capability (zero-shot generalization). Very LLM-influenced framing.

### B. Naming Convention

- **Model:** "Timer" (mnemonic: Time Series Transformer, or just "timer" as clock/temporal metaphor)
- **Format:** "Single-Series Sequence" (S3 format)
- **Paradigm:** "generative pre-training", "large time-series models"
- **Task unification:** "unified generative task"

S3 is a memorable acronym. "Timer" is short and evokes temporal domain. Language borrowed from LLM literature ("pre-trained", "foundation").

### C. Problem Formulation Style

Frames time series as heterogeneous modality requiring unified representation (S3). No formal problem setup. Instead, motivates by analogy: *"just as NLP unified text via tokenization, we unify time series via S3."*

### D. Motivation Figure

**Fig. 1 conceit:** Shows diverse time series (power load, traffic speed, ECG, stock price, etc.) all converted to single S3 format (1D sequence of values with metadata/mask tokens). Illustrates the unification principle. One figure motivates both the model (generality) and the format (simplicity).

### E. Method Section Organization

**Structure:** 4–5 subsections:
1. **S3 Format Design** — how to serialize diverse series
2. **Pre-training Objectives** — next-token prediction (like GPT)
3. **Downstream Task Adaptation** — how to cast forecasting/imputation as generation
4. **Large-Scale Pre-training** — datasets, scale (1B time points)
5. **Architecture** — decoder-only, GPT-style

Heavy notation for format specification. Equations for loss functions (standard language-model losses).

### F. Theoretical Content

No theorems. **Strong empirical scaling laws**: shows that pre-training on larger data leads to better generalization. Scaling analysis (parameter count vs. performance) borrowed from LLM literature.

### G. Evaluation Pattern

**Multi-task evaluation**: Forecasting, imputation, anomaly detection tables, multiple datasets per task.  
**Zero-shot**: Direct application to downstream task without fine-tuning.  
**Few-shot**: Fine-tune on small labeled datasets.  
**Pre-training data ablation**: Performance vs. dataset size.

### H. Related Work Organization

By framing: "Foundation Models", "Time Series Methods", "Pre-training Approaches". Emphasizes bridge between LLM practices and TS.

### I. Distinctive Prose Habits

- **Analogy-driven:** Repeatedly cites LLM success as precedent
- **Scope language:** "Large-scale", "billion", "unified", "general"
- **Capability claims:** "promising", "competitive", "strong"
- **Format emphasis:** S3 is explained multiple times; unification is key rhetorical point
- **Verb:** "curate", "unify", "develop", "convert"

### J. Open-source / Reproducibility

**GitHub:** https://github.com/thuml/Large-Time-Series-Model (with checkpoints)  
**Supplementary:** OpenLTM (https://github.com/thuml/OpenLTM) provides pre-training code and datasets.

---

## 7. SimMTM (NeurIPS 2023)

### A. Abstract Structure

**[Motivation]:** "Self-supervised pre-training has attracted immense interest to reduce labeling expenses and benefit various tasks."

**[Standard approach + failure mode]:** "One mainstream paradigm is masked modeling... However, since semantic information of time series is mainly contained in temporal variations, random masking will seriously ruin vital temporal variations, making the reconstruction task too difficult."

**[Key insight]:** "By relating masked modeling to manifold learning, SimMTM proposes to recover masked time points by weighted aggregation of multiple neighbors *outside the manifold*..."

**[Results]:** "Achieves state-of-the-art fine-tuning performance... in forecasting and classification, covering both in- and cross-domain settings."

**Pattern:** Opens with importance (self-supervised learning), identifies failure mode (random masking destroys structure), reframes problem mathematically (manifold learning), proposes fix (manifold-aware masking recovery). Notice the **insight shift**: not changing the mask, but changing how to recover from it.

### B. Naming Convention

- **Framework:** "SimMTM" (Simple Masked Time-series Modeling)
- **Key concept:** "manifold learning", "weighted aggregation of neighbors"
- **Approach:** "mask-aware recovery"

"Simple" in the title is deliberate (contrasts with complex pre-training methods). Manifold terminology adds mathematical sophistication.

### C. Problem Formulation Style

Frames as reconstruction task: *"given masked series, reconstruct masked time points."* Introduces manifold as geometric intuition: time series lie on a lower-dimensional manifold; random masking projects off-manifold; recovery should use on-manifold structure.

### D. Motivation Figure

**Fig. 1 conceit:** Schematic of manifold + off-manifold points. Shows that randomly masked points are recovered by aggregating neighbors (off-manifold recovery), which eases the task by ensuring recovered values lie on the data manifold. Illustrates the geometric intuition.

### E. Method Section Organization

**Structure:** 3–4 subsections:
1. **Manifold Learning Perspective** — why masking fails
2. **Off-Manifold Neighbor Aggregation** — recovery method (k-NN weighting)
3. **Manifold Structure Learning** — learning the manifold itself
4. **Downstream Tasks** — forecasting and classification fine-tuning

Moderate equations. Notation for neighbors, weights, aggregation.

### F. Theoretical Content

No formal theorems. **Manifold hypothesis justification**: cites classical manifold-learning theory (time-series data concentrates on low-dimensional manifolds). Proposes that good pre-training should respect this structure.

### G. Evaluation Pattern

**Pre-training + fine-tuning**: Pre-train on unlabeled data, fine-tune on downstream task with limited labels.  
**In-domain**: Fine-tune on same-domain data.  
**Cross-domain**: Train on one domain (e.g., energy), test on another (e.g., traffic).  
**Datasets**: 9+ covering forecasting and classification.  
**Ablation**: Off-manifold aggregation alone, manifold learning alone, both.

### H. Related Work Organization

By paradigm: "Masked Modeling Methods", "Self-Supervised Learning for Time Series", "Manifold Learning Approaches".

### I. Distinctive Prose Habits

- **Problem reframing:** "Relating [problem] to [theory]"
- **Severity language:** "seriously ruin", "too difficult"
- **Geometric intuition:** "on-manifold", "off-manifold", "structure"
- **Scope:** "both in- and cross-domain" (emphasizes generalization)
- **Verb:** "relate", "recover", "aggregate", "uncover"

### J. Open-source / Reproducibility

**GitHub:** https://github.com/thuml/SimMTM  
**Spotlight:** NeurIPS 2023 Spotlight (high impact).

---

## 8. Time-Series-Library (GitHub Organization)

### Repository Structure & Conventions

**Link:** https://github.com/thuml/Time-Series-Library

**Organization:**
- `models/` — individual model implementations (Autoformer.py, TimesNet.py, iTransformer.py, etc.)
- `scripts/` — experimental scripts for each task/dataset
- `data_provider/` — unified data loading (ETTh1, Weather, Traffic, ECL, ILI, Exchange)
- `tutorial/` — Jupyter notebooks for reproducibility
- `README.md` — comprehensive guide with performance tables

**Conventions:**
- **Single entry point:** `run.py` for all experiments (config-driven via args)
- **Consistent API:** Each model inherits from base `Model` class
- **Task-agnostic:** Same model architecture handles forecasting, imputation, classification by changing task flag
- **Benchmark rigor:** Standard train/valid/test splits, fixed random seeds, detailed hyperparameters in configs
- **Paper-code linkage:** Each model's implementation links to arXiv/proceedings paper

**Style elements:**
- Comments are minimal; code is self-documenting
- Naming: camelCase for variables, PascalCase for classes
- Config files (YAML) store hyperparameters; no magic numbers in code
- Ablation scripts separate from main model scripts

---

## 9. Mingsheng Long Homepage & Lab Info

**Lab:** Tsinghua University Machine Learning (THUML), School of Software  
**Homepage:** https://ise.thss.tsinghua.edu.cn/~mlong/  
**Email:** mingsheng@tsinghua.edu.cn  
**Key initiatives:**
- **Time Series Library** — unified benchmark for forecasting/imputation/classification
- **Transfer Learning Library** (https://github.com/thuml/Transfer-Learning-Library) — domain adaptation, task adaptation
- **Large Time Series Models** — foundation models (Timer)
- **Reproducibility:** Every paper has GitHub link and code release policy

---

## Cross-Paper Synthesis: 15+ Recurrent Patterns in THUML Writing

### 1. **Domain-Grounded Problem Motivation**
Every paper opens by anchoring the problem in real-world applications. Not "time series forecasting is a problem" but "extreme weather early warning", "traffic congestion prediction", "power load forecasting". This concreteness signals practical relevance. *Examples:* Autoformer (energy, traffic, economics), TimesNet (weather, anomaly detection, action recognition), iTransformer (multivariate sensor networks).

### 2. **Identification of a Hidden Bottleneck or Paradox**
Rather than claiming existing methods are "suboptimal", THUML papers identify a *specific, counterintuitive* failure mode:
- Autoformer: "information utilization bottleneck" in sparse attention
- Non-stationary: "over-stationarization paradox" (better predictability ≠ better modeling)
- iTransformer: "temporal tokens fuse variates → meaningless attention"
- SimMTM: "random masking ruptures temporal structure"

This positions the paper as offering *insight*, not just engineering.

### 3. **Conceptual Inversion or Reframing**
The core contribution often involves a conceptual flip:
- Autoformer: series decomposition from pre-processing → inner block
- TimesNet: 1D → 2D transformation
- iTransformer: temporal tokens → variate tokens (inversion)
- Non-stationary: stationarization (removes info) ↔ de-stationarization (recovers it)
- SimMTM: random recovery → manifold-aware recovery

This inversion framing is memorable and supports a single-figure motivation.

### 4. **One-Figure Motivation**
Every paper has a canonical Fig. 1 that justifies the method in a single visual:
- Autoformer: point-wise attention vs. period-based auto-correlation
- TimesNet: 1D → 2D period folding with columns/rows for intra/inter
- iTransformer: temporal token fusion vs. variate tokenization
- Non-stationary: stationarized attention (identical) vs. raw (diverse)
- Timer: diverse series → unified S3 format

The figure is almost always worth a thousand words of motivation.

### 5. **Hyphenated Compound Terminology**
THUML invents and hyperlinks new terms that crystallize insights:
- "Temporal 2D-Variation" (TimesNet)
- "intraperiod-" and "interperiod-variations" (TimesNet)
- "Auto-Correlation mechanism" (Autoformer) — note: not "auto-correlation" but "Auto-Correlation" with capitals
- "De-stationary Attention" (Non-stationary)
- "Series Decomposition Block" (Autoformer)

Hyphenation signals a new conceptual unit; capitalization marks it as a technical term within the paper.

### 6. **State-of-the-Art by Quantification**
Results are always reported with specific %, not vague claims:
- Autoformer: "38% relative improvement on six benchmarks"
- Non-stationary: "49.43% MSE reduction on Transformer" (exact %)
- FEDformer: "14.8% and 22.6% for multivariate and univariate"
- Results are often split by application domain or variance type for granularity

### 7. **Task-General or Multi-Task Framing**
Nearly all THUML papers claim or investigate broad applicability:
- TimesNet: "task-general backbone" (forecasting + imputation + classification + anomaly detection)
- Timer: unified generative format for forecasting/imputation/anomaly detection
- Non-stationary: "generic framework" (plug into any Transformer-based forecaster)
- iTransformer: zero-shot transfer to unseen variates

This positions the work as foundational, not narrow.

### 8. **Theory-Informed (Not Theory-Heavy) Justification**
Papers cite classical theory (stochastic processes, signal processing, manifold learning) without formal proofs:
- Autoformer: "Inspired by stochastic process theory" (justifies periodicity-based attention)
- FEDformer: "sparse representation in Fourier basis" (standard signal-processing fact)
- SimMTM: "manifold learning" perspective (geometric intuition)
- Non-stationary: stationarity concepts from econometrics

Theory grounding adds credibility without slowing down methodology exposition.

### 9. **Large-Scale Empirical Validation**
Evaluation is thorough:
- 6–10 datasets minimum (ETT, Weather, Electricity, Traffic, ILI, Exchange, etc.)
- 12–15+ baselines per paper
- 5+ application domains mentioned
- Ablations isolate each component
- Scaling analysis (dataset size, model size) in some papers (Timer)

This breadth discourages cherry-picking claims.

### 10. **Decomposition as a Recurring Motif**
Decomposition appears in ≥4 papers (Autoformer, FEDformer, Non-stationary uses stationarization, Timer uses format unification):
- Not just trend/seasonal, but trend/detail, seasonal/trend, on-manifold/off-manifold
- Decomposition is repositioned from pre-processing to core architecture
- Suggests THUML treats decomposition as a foundational principle for time series

### 11. **GitHub-Linked Reproducibility**
Every major paper links GitHub early and often:
- Code is released at paper acceptance
- Unified benchmarking library (Time-Series-Library) integrates all models
- Tutorial notebooks provided
- Hyperparameters in configs, not magic numbers
- Clear train/valid/test splits

This "code as publication" philosophy is consistent across the group.

### 12. **Minimal Hedging; Strong Claims**
Prose is declarative:
- Not "may improve" but "achieves state-of-the-art"
- Not "suggests" but "demonstrates", "shows", "proves" (in empirical sense)
- Exceptions: "may fail" when describing prior work's limitations
- Confidence is proportional to empirical evidence

### 13. **"We Propose", "We Design", "We Develop" Verb Choice**
Abstract and intro use active agency:
- "We propose", "We break with", "We design", "We develop", "We extend"
- Rarely "A new method is presented" (passive is avoided)
- Suggests deliberate choices, not accidental discoveries

### 14. **Scope and Unification Language**
Papers emphasize breadth:
- "General time series analysis" (TimesNet)
- "Unified generative task" (Timer)
- "Generic framework" (Non-stationary)
- "Arbitrary lookback windows" (iTransformer)
- "Both in- and cross-domain" (SimMTM)

This narrative positions the work as addressing fundamental problems, not incremental improvements.

### 15. **Application Diversity in Results**
Papers explicitly list distinct domains:
- Energy, traffic, economics, weather, disease (Autoformer abstract)
- Weather forecasting, anomaly detection, action recognition (TimesNet intro)
- Multivariate sensor networks, stock prices, medical signals (iTransformer)

Listing applications in the abstract signals real-world relevance, not toy problems.

### 16. **Notation Discipline**
Equations are used strategically:
- Problem setup: minimal (avoid formalism)
- Method: central (explain each component with equations)
- Results: sparse (tables are primary)
- Notation tables provided when introducing many symbols
- Subscripts/superscripts are consistent (e.g., x_{:t} for history, y_{t+1:t+H} for future)

---

## Summary: Style Recommendations for Seer (KV-Cache Eviction/Prefetch)

### Applicable THUML Patterns for Seer:

1. **Open with KV-cache motivation**: "Attention's quadratic memory prohibits deployment on edge devices and long-context applications (RAG, document QA)."
   
2. **Identify a hidden bottleneck**: Not "greedy eviction is suboptimal" but "static eviction policies misalign with *bursty* query patterns" or "learned eviction must balance recency bias with long-range dependency relevance."

3. **Propose a conceptual inversion**: E.g., "We invert eviction from *before* generation (prefetch all) to *during* generation (selective retention)", or "Learned policies *approximate* the optimal offline oracle."

4. **Design Fig. 1**: Show a concrete case (e.g., in-context learning task) where naive eviction (oldest-first) fails and learned eviction succeeds. One figure should justify the approach.

5. **Use hyphenated terminology**: "KV-Cache Eviction", "Query-Aware Prefetch", "Learned Retention Policy", "On-Manifold Tokens" (if clustering token importance).

6. **Quantify results precisely**: "18% memory reduction at 2% throughput overhead", "Wins on 7/10 in-context learning benchmarks", "Generalizes to 3.6B model without retraining."

7. **Claim task-generality or multi-task validation**: Test on forecasting, language modeling, retrieval-augmented generation, not just one task.

8. **Reference theory lightly**: Cite information theory (entropy, mutual information) or queue theory (eviction policies) without formal proofs; use them for intuition.

9. **Provide GitHub link early**: "Code available at github.com/your-org/seer-cache"

10. **Avoid hedging**: "Seer achieves X" not "Seer tends to achieve X"; back it up with numbers.

---

## Sources

- [Autoformer: Decomposition Transformers with Auto-Correlation (NeurIPS 2021)](https://github.com/thuml/Autoformer)
- [TimesNet: Temporal 2D-Variation Modeling (ICLR 2023)](https://github.com/thuml/TimesNet)
- [iTransformer: Inverted Transformers for Time Series (ICLR 2024 Spotlight)](https://github.com/thuml/iTransformer)
- [Non-stationary Transformers: Exploring Stationarity (NeurIPS 2022)](https://github.com/thuml/Nonstationary_Transformers)
- [FEDformer: Frequency Enhanced Decomposed Transformer (ICML 2022)](https://github.com/DAMO-DI-ML/ICML2022-FEDformer)
- [Timer: Generative Pre-trained Transformers for Time Series (ICML 2024)](https://github.com/thuml/Large-Time-Series-Model)
- [SimMTM: Simple Pre-Training for Masked Time-Series (NeurIPS 2023 Spotlight)](https://github.com/thuml/SimMTM)
- [Time-Series-Library: Unified Benchmark](https://github.com/thuml/Time-Series-Library)
- [Mingsheng Long Homepage (Tsinghua THUML)](https://ise.thss.tsinghua.edu.cn/~mlong/)
- [THUML GitHub Organization](https://github.com/thuml)

