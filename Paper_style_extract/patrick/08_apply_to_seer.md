# 08 · 把 Patrick 风格应用到 Seer (NeurIPS 2026)

NeurIPS 论文格式（9 页正文 + appendix）和 USENIX/IEEE 系统论文不同，但**Patrick 组的写作哲学**几乎全部可以迁移。下面给出可以直接执行的清单。

## 一句话定位

把 Seer 写成"**学习增强的 KV-cache 管理 ≈ 我们这一代的 ECPipe**"——一个有名字、有 prototype、有真实部署、量化结论清晰、设计要素正交可消融的系统论文，而不是仅仅一个 ML model。

## 必须迁移的 8 件事

### 1. 摘要套五段公式（见 [`03_abstract_and_intro.md`](03_abstract_and_intro.md)）

填空模板：

```
[1] LLM serving systems rely on KV-cache to amortize prompt prefill;
    HBM 上的 cache 容量直接限制了 throughput / TTFT。
[2] However, prevailing eviction policies (LRU/LFU/BlockSwap …) 
    在长上下文 + 强复用的工作负载下 hit ratio 远低于 oracle，
    浪费 X% 的 HBM 带宽 / Y% 的 prefill 计算。
[3] We propose Seer, a learned KV-cache management policy that 
    jointly performs eviction and prefetch by predicting per-block 
    re-use distance from a lightweight model trained on production 
    traces.
[4] We implement Seer in vLLM (~Y k LoC) as a drop-in cache controller, 
    requiring no model changes.
[5] On Mooncake / ShareGPT / LongBench traces and a 4×A100 testbed, 
    Seer reduces p99 TTFT by Z% and improves throughput by W× over 
    LRU and AttentionStore.
```

### 2. 三个或四个 design primitives（必做）

强烈建议把 Seer 的设计分解成 3 个正交可消融的 primitive，让 §4 长这样：

- **§4.1 Reuse-distance predictor (SeerNet)**
- **§4.2 Pareto-aware eviction policy**
- **§4.3 Look-ahead prefetch scheduler**
- *(可选)* **§4.4 Online retraining loop**

末段一行总结："The three primitives are designed to be orthogonal: §5.6 ablation shows each contributes ~X-Y%."

### 3. Implementation 单独成节

NeurIPS 论文常常省掉这块，但**对 Seer 这种系统型工作，单独的 §5 Implementation 是 reviewer 信任度的关键**：

- vLLM 集成点（哪个文件 / 接口）
- LoC 数量
- 是否需要修改模型 / 仅 runtime
- 离线训练流程
- 在线推断成本（µs / step）
- Open source 计划（如有）

NeurIPS 主文如果写不下，可以单独放一段 + appendix 详细描述。

### 4. Evaluation 用 Q-style 拆分（见 [`05_evaluation.md`](05_evaluation.md)）

```
§6.1 Setup
§6.2 Q1: How predictable is KV-block reuse?           （motivation 实验，证明 SeerNet 能学）
§6.3 Q2: How does Seer affect hit ratio?             （主图 1）
§6.4 Q3: How does Seer affect TTFT and throughput?    （主图 2，含 p99）
§6.5 Q4: Does Seer generalize across models / contexts?
§6.6 Q5: Effect of each design primitive (ablation table)
§6.7 Q6: Overhead breakdown (model size, infer latency, memory)
§6.8 Q7: Sensitivity (HBM budget, block size, retrain freq)
```

每个子节固定五句话节奏：目的句 → 方法句 → 图/表 → take-away → 机制解释。

### 5. Trade-off framing 显式化

NeurIPS reviewer 看到 "we improve cache" 会问"代价是什么"。Patrick 组招牌的 **trade-off 显式化**：

> *Design space.* Seer trades off prediction overhead against eviction quality. Aggressive prediction yields higher hit ratio at the cost of model inference latency; conservative prediction reduces overhead but degrades hit ratio. §6.7 quantifies this trade-off across operating points.

这一段值得在 §3 design overview 末尾就出现，让审稿人后面读 ablation 时心里有谱。

### 6. 命名层级化、贯穿全篇

按 [`04_naming_and_terminology.md`](04_naming_and_terminology.md)：

- **Seer** = 整体系统 / runtime
- **SeerNet** = 预测模型
- 给方法（policy）也起一个名字（e.g. **PEP** = Predictive Eviction & Prefetch；或者直接复用 Seer 双名）

整篇论文一旦定下，**绝不混用**。Reviewer 翻一眼就知道 SeerNet 指的是什么。

### 7. Related Work 主题分组

按 [`06_writing_microconventions.md`](06_writing_microconventions.md)：

```
Related Work

LLM serving systems and KV management.    [vLLM, SGLang, Mooncake, AttentionStore, ...]
Cache replacement and learning-augmented data structures.    [LRU/LFU survey, LRB, Lecar, ...]
Workload prediction for caching.    [Hawkeye, Glider, ...]
Prefetching in deep learning systems.    [...]

The closest work to ours is X et al. \[N], which … differs from Seer in that …
```

最后那句"closest work"是 Patrick 组招牌之一，**强烈建议照抄到 Seer**。

### 8. 实验数字 → 全篇出现 ≥ 3 次

Patrick 组让数字反复曝光：

- 摘要：粗略数字（"reduces TTFT by 30-45%"）
- §1 末段贡献清单：同一组数字
- §6 主图：精确数字
- §7 conclusion：再呼应一次

NeurIPS 论文为了"克制"经常只在 §6 出现一次结果数字 —— 这是读者印象深度上的损失。**至少出现 3 次** 是低成本 high-leverage 的修改。

## 可选迁移（看篇幅）

- **Motivation observation 一段配一张小图** —— Patrick 组的 Fig. 1 经常就是一张 motivation chart。Seer 完全可以在 §1 / §2 末尾放一张 "oracle vs LRU hit ratio gap" 图，胜过千言万语。
- **"Observation X" 显式标注** —— 在 motivation / design 段落里用 italic *Observation 1* 标注关键论断；NeurIPS 不限制，效果不错。
- **Notation table** —— Seer 的符号不算多，但单独一个 Table 1 列 KV block size / capacity / hit ratio / eviction rate 等会让审稿人轻松。

## 不迁移的部分

- **罗马数字大写章节标题 (I. INTRODUCTION)** — NeurIPS 不用。
- **§II BACKGROUND AND RELATED WORK 合并** — NeurIPS 喜欢 background 单独 §2，related work 单独 §7 或 §8。
- **专门的 Extensions 节** — NeurIPS 篇幅紧，extensions 放 appendix。
- **被 trace replay 跑 Yahoo/Microsoft** — Seer 用 LLM 工作负载，对应改成 ShareGPT / Mooncake / LongBench / production trace。

## 执行节奏建议

1. **第 1 周**：把 README + 01-03 内化，写完 abstract + §1 引言（套模板）。
2. **第 2 周**：填 §2 背景 + §3 design overview（含 trade-off framing）。
3. **第 3 周**：写 §4 design primitives（每个 primitive 一天）。
4. **第 4 周**：写 §5 implementation + §6 evaluation 框架（Q1-Q7 列出 + setup 段）。
5. **第 5-6 周**：跑实验 + 填图 + 写 take-away。
6. **第 7 周**：related work + conclusion + abstract 最终修订。

每完成一节，对照本文件夹的 checklist 自查："命名一致？数字三处出现？trade-off 显式？baseline ≥ 3？"

## 一份 self-review 卡（打印在桌前）

```
□ Abstract 五段都齐全？
□ 引言七步都出现？
□ Contributions 用动词开头、可独立证伪？
□ Design 拆成 N 个并列的 primitive？
□ 每个 primitive 末段提到对应实验编号？
□ Implementation 单独成节，给出 LoC + 嫁接点？
□ Evaluation 至少 3 个 baseline + 真实 trace？
□ Ablation 表覆盖每个 primitive？
□ Overhead / Cost 子节存在？
□ Related Work 按主题分组 + "closest work" 显式句？
□ 命名 / 术语全篇一致？
□ 数字至少出现 3 处（abstract / §1 / §6）？
□ Trade-off 至少显式出现 1 次？
```
