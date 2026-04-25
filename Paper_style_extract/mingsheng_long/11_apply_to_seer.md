# 11 · 把 THUML 风格落地到 Seer (NeurIPS 2026)

这一章把前 10 章浓缩成一份 Seer 直接可用的 *作战清单*。

## 一句话定位

把 Seer 写成 "**iTransformer of KV-cache management**"——
有命名瓶颈 (Reuse-Distance Blindness)、有概念反转 (Eviction is Prefetch in Reverse)、有一图杀人的 Fig 1、有 3 个并列 design components、有 12+ baselines 的大表、有 Time-Series-Library 级别的开源 codebase。

## 必做的 12 件事

### 1. 给 Seer 起一个 *双层命名*

```
System name           : Seer
Predictive model name : SeerNet
Eviction policy name  : Pareto-Aware Eviction (PAE)
Prefetch policy name  : Look-ahead Prefetch Scheduler (LPS)
Loss name             : Reuse-Distance Loss (L_rd)
Trace format name     : Block Trajectory Trace (BTT)
Theory term name      : Predictability Gap Δ(π)  (Definition 1)
```

整篇论文这些专名**绝不变体**（参考 [`04_naming_and_terminology.md`](04_naming_and_terminology.md)）。

### 2. 给隐藏瓶颈起一个名字

候选：

- **Reuse-Distance Blindness** ← 推荐
- **Eviction-Prefetch Decoupling**
- **Static Capacity Misallocation**

文章一开头（abstract [2] / §1.3）就引出这个名字，整篇反复用 4-5 次。

### 3. 设计一个 *概念反转* slogan

候选：

- **"Eviction is Prefetch in Reverse"** ← 推荐
- **"Predict the future, retain accordingly"**
- **"Block-level reuse distance is the right unit of importance"**

reviewer 读完这一行，已经把整篇 Seer 的 idea 装进脑子里了。

### 4. 一图杀人的 Fig 1

要做出这种图：

```
(Top)    LRU policy on a long-context conversation:
          ▓▓▓▓▓░░░░░░▓▓▓▓░░░░░░▓▓▓▓
         "evicted block IS reused 200 steps later" (red dot)

(Bottom) Seer: predicts reuse distance per block;
          ░░▓▓▓▓░░░▓▓▓▓░░▓▓▓▓░░▓▓▓▓
         "blocks predicted to be reused soon retained" (green dots)
```

或：

```
(Left)  Eviction module: scores blocks by recency only.
        Prefetch module: separately decides what to load.
        |  Two modules, no shared signal → Decoupling waste.

(Right) Seer: a single reuse-distance signal drives BOTH eviction
        AND prefetch decisions.
        |  One signal, joint decisions → Pareto improvement.
```

无论选哪种构图，**目标是 reviewer 看完 Fig 1 不读正文也能 grasp 论文**。

### 5. 摘要五段式硬填

```
[1] Long-context LLM serving relies on KV-cache to amortize prefill cost
    across decoding, which is critical for applications such as multi-turn 
    dialog, retrieval-augmented generation (RAG), and code agents.

[2] Existing eviction policies (LRU, FIFO, attention-score-based) and
    prefetch heuristics treat eviction and prefetch as independent modules
    operating on coarse signals. We identify that this leads to *Reuse-
    Distance Blindness*, where blocks with imminent re-use are evicted
    while seldom-used blocks are retained.

[3] In this paper, we propose Seer, a learned KV-cache management framework
    that JOINTLY performs eviction and prefetch from a unified reuse-distance
    prediction. Seer consists of three components: SeerNet (a per-block
    reuse-distance predictor), Pareto-Aware Eviction (PAE), and a Look-ahead
    Prefetch Scheduler (LPS). The unifying insight: eviction is prefetch
    in reverse.

[4] Seer is plug-and-play: it requires no model changes and integrates as a
    runtime middleware into vLLM, SGLang, and TensorRT-LLM.

[5] On 4 production traces across 4 model scales (Llama-7B/13B/70B, Mistral-
    7B), Seer reduces p99 TTFT by 31-46%, improves throughput by 1.4-1.8x,
    and lifts hit ratio by 8-15 absolute points over 12 strong baselines.
    Code: github.com/.../seer.
```

把数字 (31-46%, 1.4-1.8×, 8-15) 当占位符，等实验出来填实际数据。

### 6. §1 走七步

| 步 | 内容 |
|---|---|
| §1.1 | LLM serving 应用 + KV-cache 部署事实（vLLM/SGLang 论文 + 工业部署引用）|
| §1.2 | 现有 KV 管理三类：eviction-only / prefetch-only / static budgeting |
| §1.3 | 命名隐藏瓶颈：**Reuse-Distance Blindness**，配 Fig 1 |
| §1.4 | 我们的反转 insight：**Eviction is Prefetch in Reverse** |
| §1.5 | Seer 概述（SeerNet + PAE + LPS） |
| §1.6 | 4-5 条 contributions（动词开头） |
| §1.7 | (可选) paper roadmap |

### 7. §3 Preliminary 给 notation table + Observation 1

参考 [`05_problem_formulation.md`](05_problem_formulation.md) 末尾的草稿。

### 8. §4 Method 三个并列 component

```
4.1 Insight: Eviction is Prefetch in Reverse
4.2 SeerNet: A Reuse-Distance Predictor
4.3 Pareto-Aware Eviction (PAE)
4.4 Look-ahead Prefetch Scheduler (LPS)
4.5 Architecture & Training Objective
4.6 Theoretical Analysis (Proposition 1, optional)
```

每个 component 末尾一行 "**validated in §5.X (Table Y)**"。

### 9. §5 实验 6 个子节

```
5.1 Setup            (12 baselines, 4 traces, 4 models, vLLM v0.X)
5.2 Main Results
5.3 Generalization   (cross-model, cross-workload)
5.4 Ablation         (component on/off + feature ablation)
5.5 Showcase         (eviction trace heatmap, predictor scatter)
5.6 Efficiency       (SeerNet params/FLOPs/latency)
```

数据集和 baselines 至少凑到这个量级：

```
Traces:   ShareGPT, Mooncake, LongBench-Chat, Production (内部)
Models:   Llama-7B, Llama-13B, Llama-70B, Mistral-7B
Baselines (12+):
  - LRU, FIFO, Belady-oracle (offline lower bound)
  - vLLM-default, AttentionStore, BlockSwap, Quest
  - 3-4 个 prior learned policies (LRB-style, LCache-style)
  - Random eviction (sanity)
```

### 10. 把每个 component 末段的 take-away 量化

```
4.2 末尾：SeerNet alone (no PAE/LPS) achieves 27% reuse-distance regression
          R², a 4.1× lift over LRU's reuse-distance estimate (validated in §5.4).

4.3 末尾：PAE without LPS already reduces miss rate by 18% over LRU; combined
          with LPS the gain reaches 31% (Table 5).

4.4 末尾：LPS reduces fetch latency by 24% over the heuristic prefetch
          baseline (Table 5).
```

### 11. Limitations / Broader Impact 子节

放在主文末或 appendix 开头：

```
Limitations.
  1. Seer requires offline trace collection for SeerNet training; bootstrapping
     on cold start workloads remains future work.
  2. Reuse-distance predictability degrades for highly adversarial workloads
     (Appendix E.3).
  3. SeerNet inference adds X μs per step; in extreme low-latency regimes the
     overhead may dominate (mitigated by quantization, §5.6).

Broader Impact.
  Seer enables longer-context LLM deployment under fixed HBM budgets,
  reducing serving cost and energy. We see no significant negative impact.
```

NeurIPS 现在硬要求这两节。

### 12. Reproducibility 一字不落

```
Code:        github.com/.../seer
Checkpoints: huggingface.co/.../SeerNet
Data:        traces in repo (de-identified) + scripts to reproduce synthetic
Hardware:    "训练用 4× A100 共 X 小时；推理用 1× A100"
Seeds:       3 random seeds, mean ± std reported throughout
```

abstract 末尾给 GitHub URL。Appendix B 给完整超参表 + 训练协议 + 数据预处理。

---

## 与 Patrick 风格对照（什么时候用谁）

| 写法元素 | THUML | Patrick (CUHK ADSLab) | Seer 的选择 |
|---|---|---|---|
| 摘要 | 五段式（含任务通用化） | 五段式（含 prototype + EC2） | **THUML** for NeurIPS |
| 引言 | 7 步（命名瓶颈 + 反转） | 7 步（systems framing + LoC） | **THUML** 主，Patrick 的 LoC 数字加分 |
| 命名 | hyphenated capitalised + 双层 | acronym 优先 | **THUML** + 借 Patrick 的双层（system + technique）|
| 子组件数 | 3-4 并列 | 3 个 design primitives | 一致：**3 components** |
| Implementation | 嵌入 §4 末段 + appendix | 单独 §4 / §5 一节 | **Patrick** —— Seer 是系统型工作，单独留 §5 给 implementation 加分 |
| Theory | Proposition / Theorem (可选) | 无 | **THUML** —— NeurIPS 评委吃 theory |
| Evaluation | 6-10 数据集 / 12-15 baseline | 3-5 baseline / 真实 trace | 混合：**多 baseline** + Patrick 的 trace 风味 |
| Open source | 统一 library + GitHub | GitHub + middleware claim | **混合**：Time-Series-Library 风格 + middleware claim |
| Showcase | attention map / forecast curve | latency p99 plot | 两者皆做：eviction trace + p99 plot |

> **结论**：Seer 主体走 THUML 风格，但**保留 Patrick 风格的 implementation 章 + 真实 trace + middleware framing**。这是 Seer 在 NeurIPS 评审里独有的 differentiator——既有 ML 论文的 craftsmanship，又有 systems 论文的 credibility。

---

## Self-Review 卡（打印贴桌前）

```
□ Seer 命名了一个隐藏瓶颈？(Reuse-Distance Blindness)
□ 有一句概念反转 slogan？(Eviction is Prefetch in Reverse)
□ Fig 1 一图杀人？
□ Abstract 五段都齐全且有命名 / 数字？
□ §1 七步都覆盖？
□ §3 有 notation table + Observation 1？
□ §4 三个并列 component (SeerNet / PAE / LPS)？
□ 每个 component 末尾有 quantified take-away + §5.X pointer？
□ §5 ≥ 12 baselines + ≥ 4 traces + ≥ 4 模型？
□ 主表 bold-best / underline-second / metric ↑↓ 标注？
□ Ablation 表覆盖每个 component + 关键 features？
□ Showcase 至少 2 张视觉图？
□ Efficiency 子节给 SeerNet 自己的 overhead？
□ Limitations 子节诚恳 3-4 条？
□ Broader Impact 一段？
□ GitHub URL 在 abstract / §1 末尾？
□ 命名 / 术语全文一致？
□ 数字至少 3 处出现 (abstract / §1 / §5)？
□ 引用 Related Work 按 idea axis 分 4-5 段？
□ "Closest work to ours" 一句存在？
□ 所有数字带 ± std？
□ 引用自家前作（OrchKvCache 等）做 series continuity？
```

每个 NO 都是潜在的 reviewer 扣分点。

---

## 时间线建议

| 周 | 工作 | 对照本文档 |
|---|---|---|
| 1 | 把 abstract + §1 框架填出来 | [`03`](03_abstract_and_intro.md) [`05`](05_problem_formulation.md) |
| 2 | §3 Preliminary + Fig 1 motivation | [`05`](05_problem_formulation.md) |
| 3 | §4 三个 component 草稿 | [`06`](06_method_section.md) |
| 4 | §4.6 Proposition 1 + appendix proof | [`07`](07_theory_and_analysis.md) |
| 5-6 | 实验 + 主表 + ablation | [`08`](08_evaluation.md) |
| 7 | Showcase / efficiency / limitations | [`08`](08_evaluation.md) |
| 8 | Related Work + Conclusion | [`09`](09_writing_microconventions.md) |
| 9 | Self-review + reproducibility | [`09`](09_writing_microconventions.md) |
| 10 | 全文打磨 + 命名一致性扫一遍 | [`04`](04_naming_and_terminology.md) |

NeurIPS 2026 5 月截稿的话，**3 月底 §4 必须完工**——后面 2 个月是实验和打磨。

---

## 最后一句

> 不要试图同时模仿 Autoformer 的优雅 + MDD 的严谨 + Timer 的气派。**选一个 model paper 当北极星**（我推荐 **iTransformer**——简洁、概念反转、一图杀人，最适合 Seer），其他论文当辅助。
