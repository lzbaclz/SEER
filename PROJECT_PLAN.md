# NeurIPS 2026 投稿计划：Learned KV-Cache Management for Long-Context LLM Inference

> **项目名**：**SEER** (Selective Eviction via Expected Recall)
> **基础项目**：OrchKvCache (`~/codes/papers/OrchKvCache`)
> **创建时间**：2026-04-25
> **状态**：规划中
> **目标会场**：NeurIPS 2026（备投：MLSys 2027 / EuroSys 2027 / OSDI 2027）

---

## 0. 时间线警告（最重要）

NeurIPS 2026 的 abstract / full-paper deadline 历年都在 **5 月中旬**（NeurIPS 2025 是 5 月 11 / 15 日）。今天是 **2026-04-25**，距离合理估计的 deadline 仅剩 **3 周左右**。**第一件事是去 https://neurips.cc/Conferences/2026/ 核对准确日期**，再决定接下来的执行节奏。

按 3 周倒推，本计划必须有清晰的"最小可行版本（MVP）"，避免被 ambitious feature 拖死。MVP 的标准：**单模型 + 长上下文 benchmark 的两个子集 + 一种学习预测器 + 三类基线**，在这个范围内做扎实，剩下的进入 supplementary 或 rebuttal。

---

## 1. TL;DR（一段话能讲清的论文）

长上下文 LLM 推理中，KV-Cache 的 eviction / prefetch 决策决定了延迟和质量，但现有方法都依赖**启发式**：StreamingLLM 用滑动窗口，H2O 用历史累积注意力，SnapKV 只在 prefill 一次性压缩。我们的实证测量（来自 OrchKvCache Exp1 M2）显示：**注意力分布是幂律的（Top-10% token 贡献 90%+ attention），跨 decode step 的 Jaccard 稳定性 0.47–0.70**——这意味着未来注意力是**可预测的**，启发式只利用了其中很小一部分信号。

我们提出 **LAP (Learned Attention Predictor)**：一个超轻量（< 0.1% 模型 FLOPs）神经预测器，输入历史 attention trace、recency、position 等特征，输出未来 H 步内每个 KV-block 进入 top-k attention 的概率。基于 LAP 的预测，我们设计了一个**联合 eviction-prefetch 策略**，能在固定 HBM 预算下显式平衡"保留高重要性 block"与"换入换出 IO 代价"。

我们将 LAP 部署在 OrchKvCache 的 4 级存储层次（HBM/DRAM/NVM/SSD）上，证明：(1) 在相同 HBM 预算下，LongBench 任务质量比最强 baseline 高 X%；(2) 在相同任务质量下，端到端 throughput 提升 Y×；(3) LAP 在跨模型（Llama / Qwen / Mistral）和跨任务上具有 zero-shot 泛化能力。

---

## 2. 核心研究问题与假设

### 2.1 Research Questions

- **RQ1（可预测性）**：未来 H 步 attention 的"top-k 集合"在多大程度上可被一个 sub-1% FLOPs 的预测器从历史 trace 中估计？
- **RQ2（学习的边际价值）**：相对于最强启发式（H2O / StreamingLLM / Recency），学习预测器在 quality-budget Pareto 上能再推进多少？
- **RQ3（联合策略）**：当 prefetch 和 eviction 由同一个预测信号驱动、并感知存储层级 IO 代价时，相比"只优化 eviction"的方法收益如何？
- **RQ4（泛化）**：在一个模型上训练的 LAP 能否 zero-shot 迁移到同 family 的其他规模、以及不同 family 的模型？

### 2.2 假设（在动手前先写下来，便于事后 falsifiability）

- **H1**：未来 attention top-k 与过去 N 步的 attention pattern 之间存在 **AUC > 0.85** 的可预测性（M2 的 Jaccard 0.47–0.70 是粗粒度上界，细粒度建模应能更高）。
- **H2**：LAP 的边际收益主要来自 **中间层（layer 4–20）**，因为 Exp1 M2 已显示这些层的注意力集中度最高（Top-10% > 95%）。
- **H3**：在 chunk-prefill 场景下，prefetch 命中率提升 1× 等价于 throughput 提升 0.3–0.5×（受 IO/计算重叠程度限制）。
- **H4**：LAP 的泛化性来自"注意力 dynamics 的模型无关结构"（如局部连续性、attention sink），而非具体语义。

---

## 3. 相关工作与差异化定位

> 这一节的要求：每个 baseline 必须列出它与我们方法的**精确 delta**，避免审稿人说"这就是 H2O + 一个 MLP"。

### 3.1 KV-Cache 重要性估计 / Eviction

| 工作 | 信号来源 | 决策时机 | Delta vs. 我们 |
|------|---------|---------|---------------|
| **StreamingLLM** (ICLR'24) | 位置启发式（sink + sliding） | 每步固定 | 完全不利用 attention，我们用学习的预测 |
| **H2O** (NeurIPS'23) | 累积历史 attention | 每步贪心 | **使用过去**作为未来代理；我们**预测未来**，并量化二者差距 |
| **SnapKV** (NeurIPS'24) | Prompt 末尾 attention | 仅 prefill 一次 | Prefill-only，对长 decode 失效；我们贯穿 decode 全程 |
| **Scissorhands** (NeurIPS'23) | Persistence of importance | 每步 | 假设 "important now ≈ important later"，我们直接学习 transition |
| **LESS** (ICML'24) | Low-rank residual cache | 每步 | 是 cache 表示压缩，正交于策略 |
| **Quest** (ICML'24) | Page-level attention 估计 | 每步 | 启发式估计；我们用学习模型 |
| **PyramidKV / SnapKV-pyramid** | 层间预算分配启发式 | 每步 | 层间分配是固定模板，我们让预测器隐式学习层差异 |

### 3.2 KV-Cache Offloading / 多级存储

| 工作 | 存储层级 | 调度策略 | Delta vs. 我们 |
|------|---------|---------|---------------|
| **FlexGen** (ICML'23) | GPU/CPU/Disk | 静态 schedule | 无在线学习 |
| **InfiniGen** (OSDI'24) | GPU/CPU | Speculative attention 启发式 prefetch | 启发式 + 单层 lookahead；我们 multi-step learned prediction + IO-aware |
| **CacheGen** (SIGCOMM'24) | KV-cache 流式压缩传输 | 压缩为主 | 正交工作，可叠加 |
| **Mooncake / Splitwise / DistServe** | Prefill-decode 分离 | 集群调度 | 系统级；我们是 per-request policy |

### 3.3 Attention Prediction / Sketching

| 工作 | 预测对象 | Delta vs. 我们 |
|------|---------|---------------|
| **Sparse Transformer / Reformer** | 训练时 sparsify | 改动模型结构；我们不改模型 |
| **Routing Transformer** | 学习 attention pattern | 训练时；我们 inference-time |
| **Magic-PIG / NEST** | 用 LSH 检索 top-k KV | 检索 ≠ 调度，且需精确 KV；我们决定**驻留位置** |

**我们的"独占地带"**：在 inference time、policy-only、不改模型权重、跨多级存储 IO-aware、显式预测未来 H 步注意力分布——目前没有同时勾选这五项的工作。

---

## 4. 方法设计

### 4.1 Learned Attention Predictor (LAP)

#### 输入特征（per layer × per head × per KV-block）

KV-block 粒度沿用 OrchKvCache 的 32-token block（与 OrchFS 32KB SSD block 对齐）。

- **Attention history**：过去 N 步该 block 的 attention 分数（aggregated over heads in the head group），N = 32 或 64
- **Recency**：自上次进入 top-k 已过去多少 step（log-scaled）
- **Position**：block 在序列中的相对位置（normalized）+ 距 query 的距离
- **Persistence**：滑动窗口内进入 top-k 的频率
- **Layer / head id embeddings**：使预测器能学习层间/头间差异
- **(可选) Token-block content embedding**：从 KV 中下采样得到的 32-d 向量（增加成本，先消融）

#### 模型架构（候选，按复杂度排序）

1. **Tiny-MLP**：3 层 MLP，每 block 独立预测，参数量 < 100K — 主推方案
2. **Block-RNN**：跨 step 的 GRU（捕捉 attention 时序），参数量 < 500K
3. **Block-Transformer**：跨 block 的 1 层 self-attention（捕捉 block 间依赖），参数量 < 2M

候选 (1) 是 NeurIPS 友好的 baseline-of-baselines，(2)(3) 用作 ablation 展示"复杂度-收益曲线"。

#### 输出

每个 block 在未来 H 步内**至少进入一次 top-k**的概率 $\hat{p}_{i,h}$。多个 horizon $h \in \{1, 4, 16, 64\}$ 同时预测（multi-task），便于不同决策（短期 prefetch / 长期 eviction）使用不同 horizon。

#### 训练

- **数据采集**：用 OrchKvCache 已有的 attention hook，离线在 LongBench / RULER / The Pile 子集上跑 inference，记录每层每头每 block 的 attention 分数序列。预算：每模型 ~1000 个 trajectory × 平均 8K tokens decode = ~10M block-step samples。
- **标签**：未来 H 步的 ground-truth top-k 集合（以全 cache attention 为 oracle）。
- **损失**：multi-horizon BCE + focal weighting（class imbalance：top-k=128 / total ≈ 几千 blocks → 正例 < 5%）。
- **Auxiliary loss**：直接回归 attention rank（pairwise hinge），稳定训练。
- **训练时间**：单 GPU 上 < 2 小时（模型小）。

### 4.2 Joint Eviction-Prefetch Policy

#### 目标函数

$$
\max_{S \subseteq \text{HBM blocks}} \; \mathbb{E}\Big[\text{Quality}(S) \mid \hat{p}\Big] \quad \text{s.t.} \quad |S| \le B_{\text{HBM}},\; \text{IO}(S \to S') \le B_{\text{IO}}
$$

其中 Quality 用 attention coverage（在 S 中的 token 分到的总 attention 占比）作为代理。

#### 决策算法

每 K 个 decode step（K=8 或 16，动态调整）触发一次：

1. **打分**：用 LAP 给所有 block（HBM + DRAM + NVM + SSD）打 $\hat{p}_{i,h}$。
2. **联合排序**：定义 utility 
   $$U_i = \hat{p}_{i,h} - \lambda_t \cdot \text{IO\_cost}(\text{from\_tier}_i \to \text{HBM})$$
   其中 $\lambda_t$ 是 IO 预算的 Lagrange 乘子（通过反向 line search 调整以满足 IO budget）。
3. **Top-B 选择**：选 top-$B_{\text{HBM}}$ 的 block 进入 HBM。
4. **执行**：被选中但当前不在 HBM 的 → prefetch；当前在 HBM 但未被选中 → evict 到下一层（DRAM 优先，HWM 触发再下沉）。
5. **覆盖保证**：强制保留 attention sink（前 4 token）+ sliding window（最近 W 个 step 的 KV）+ pinning set（用户指定）。

#### IO-计算重叠

- Prefetch 在 attention 计算之前 K=2 个 layer 触发（layer-pipelining），用 OrchKvCache 已有的 prefetch_scheduler 机制。
- Eviction 异步执行，不阻塞前向。

### 4.3 与 OrchKvCache 的耦合

OrchKvCache 提供：
- 4 级存储层次的 placement primitive
- Block-level 元数据（`kv_block_meta_t.hotness_score`、`current_tier`、`location`）
- Attention hook（`python/orchkv/vllm_integration/attention_hook.py`）

LAP / 新策略接入点：
- 替换 `src/classifier/hotcold_classifier.c` 的启发式打分，改成调用 LAP 推理
- 修改 `src/scheduler/eviction_policy.c` 和 `prefetch_scheduler.c`，使用联合策略
- 保持 C API 不变，仅在 Python binding 中暴露 LAP 加载/切换

**关键 invariance**：系统层（4-tier、OrchFS backend、IO 流水线）作为论文的 substrate，不是论文的卖点。论文 Method 一章只用一段话提系统层；System 描述放 Appendix 或 supplementary。

---

## 5. 理论分析（薄但要有）

NeurIPS 喜欢有理论 framing。预算只够做"中等深度"的分析，不追求 STOC-level。

### 5.1 框架：Online Set Selection with Predictions

形式化：在每步 $t$，需要选择一个 size-$B$ 子集 $S_t$ 进入 HBM；future top-k set $T_t^*$ 由对手决定；regret = $\sum_t |T_t^* \setminus S_t| \cdot w_t$，其中 $w_t$ 是该 token 的 attention weight。

### 5.2 主要 lemma（目标，非承诺）

- **Lemma 1**（预测精度 → regret 上界）：若 LAP 预测的 top-k AUC 为 $\rho$，且 attention 权重满足幂律 $w_i \propto i^{-\alpha}$，则期望 regret $\le f(\rho, \alpha, B)$，其中 $f$ 是显式函数。
- **Lemma 2**（Cost-aware 改进）：将 IO cost 纳入决策后，total wall-clock objective 比纯 quality-only 策略改善 $\Theta(\Delta_{\text{tier}} / B_{\text{IO}})$。
- **Corollary**：与 H2O 等用过去代替未来的方法对比，差距 = $|H_{\text{past}} \cap T^*| - |\hat{T}_{\text{LAP}} \cap T^*|$，可被显式量化。

### 5.3 实证验证

实证测：把 LAP 的预测当作"完美 oracle 的近似"，量化与真实 oracle 的 gap，以及与启发式的 gap。这部分作为 §6 的 first experiment（"Predictability Study"）。

---

## 6. 实验设计

### 6.1 主实验矩阵

| 编号 | 实验 | 变量 | 主要 takeaway |
|------|------|------|--------------|
| **E1: Predictability Study** | Attention 可预测性的实证刻画 | 模型 × 任务 × horizon | LAP 的 AUC，启发式的上界，可预测性的层间/任务间差异（验证 H1, H2） |
| **E2: Quality-Budget Pareto** | 不同 HBM 预算下任务质量 | 预算 ∈ {10%, 20%, 40%, 80%} × 5 baselines × 3 模型 | LongBench / RULER 上 SEER 帕累托占优 |
| **E3: Throughput-Quality** | 固定质量下 throughput | 5 baselines × 3 序列长度 (8K/32K/128K) | 同等质量下 SEER 的 tokens/s 最高 |
| **E4: 消融 LAP 架构** | MLP / RNN / Transformer | 复杂度-收益曲线 | Tiny-MLP 已经吃掉大部分收益 |
| **E5: 消融特征** | History / Recency / Position / Embedding 逐个去掉 | 哪个特征最重要 | 通常 history > recency > position |
| **E6: 消融策略** | LAP-only eviction / LAP-only prefetch / Joint | 联合策略的边际价值 | Joint 比单独至少 +Z% |
| **E7: 跨模型泛化** | Llama-3-8B 训练 → Qwen / Mistral / Llama-3-70B 测试 | Zero-shot 迁移 | 验证 H4 |
| **E8: 系统 IO-tier 价值** | 关闭 NVM / 关闭 SSD | 各级存储贡献 | 留给 supplementary，因为太"系统" |
| **E9: Predictor overhead** | LAP 推理时延 / FLOPs | < 0.1% 端到端开销 | 显示方法是 lightweight 的 |

E1 + E2 + E3 + E4 是 main paper。E5–E7 视空间放主文或 appendix。E8 进 appendix。E9 是 method section 的 inline 表格。

### 6.2 模型

- **主体（必须）**：Llama-3-8B (32K context), Qwen2.5-7B (128K), Mistral-7B-v0.3 (32K)
- **scaling**（若硬件允许）：Llama-3-70B (Q4 quantized 以单卡跑)
- **Sanity check**：Llama-2-7B（与 H2O / SnapKV 原文对齐）

### 6.3 数据集 & 任务

- **LongBench v2**：15+ 任务，涵盖 single-doc QA、multi-doc QA、summarization、code completion
- **RULER**：长度可控的 needle-in-a-haystack 类合成 benchmark（4K → 128K），最严格的 long-context 测试
- **∞Bench**：超长上下文（200K+）
- **Multi-turn**：MT-Bench long-form 子集，模拟多轮对话累积上下文

### 6.4 Baselines（必须复现 / 直接用作者代码）

1. **Full Cache**（quality 上界，throughput 下界）
2. **StreamingLLM**：sink=4 + sliding=2K
3. **H2O**：预算的 50% heavy-hitter + 50% recent
4. **SnapKV**：prefill 末尾 32 token 注意力 + cluster pooling
5. **Quest**：page-level top-k 估计
6. **InfiniGen**（若代码可复现）
7. **Random eviction**（sanity baseline）
8. **Recency-only**（sanity baseline）

### 6.5 度量

- **Quality**：F1 / EM / ROUGE-L / pass@1（任务相关）
- **Speed**：throughput (tokens/s)、TTFT、TPOT
- **资源**：peak HBM、IO bandwidth used
- **Predictor-specific**：AUC、NDCG、prediction latency

### 6.6 实现栈

- 推理引擎：vLLM 0.7+（fork，复用 OrchKvCache 已有 vllm_integration）
- 预测器训练：PyTorch 2.x
- 部署：LAP 推理用 ONNX Runtime 或 TensorRT，避免 PyTorch overhead
- 硬件：作者已有 2× A100-80GB + 376GB DRAM + Gen5 NVMe

---

## 7. 从 OrchKvCache 复用 vs. 新写

### 7.1 直接复用（don't touch）

- 4-tier 存储抽象（`src/tiered_store/`）
- OrchFS 后端（`src/tiered_store/orchfs_tier.c`）
- Block 元数据（`src/core/kv_block.{h,c}`）
- Attention hook（`python/orchkv/vllm_integration/attention_hook.py`）
- IO 流水线 / 异步迁移（`src/scheduler/pipeline.c`）
- 现有 benchmark scripts（`benchmarks/`）和 motivation 实验

### 7.2 需要重写 / 新增

- `src/classifier/lap_predictor.{h,c}`：新模块，加载 ONNX 模型，每 K step 推理
- `src/scheduler/joint_policy.c`：联合 eviction-prefetch 策略
- `python/orchkv/lap/`：训练 pipeline、数据采集、模型导出
- `experiments/exp_nips26/`：本论文专属实验脚本和结果
- 现有 `hotcold_classifier.c` 退化为 baseline（不删除，作为对比）

### 7.3 论文-代码对应表

| 论文章节 | 对应代码位置 |
|---------|------------|
| §3 Predictability Study | `experiments/exp_nips26/predictability/` |
| §4 LAP 模型 | `python/orchkv/lap/model.py` |
| §5 Joint Policy | `src/scheduler/joint_policy.c` |
| §6 Experiments | `experiments/exp_nips26/{e2..e9}/` |

---

## 8. 风险与对应

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|---------|
| **预测器精度不够，LAP < H2O** | 中 | 高 | 先做 E1（Predictability Study）作为 go/no-go gate；若 AUC < 0.75，转方向 3（Hierarchical Approximation）或缩为 workshop paper |
| **Predictor 推理 overhead 抵消收益** | 中 | 中 | 严格预算 < 0.1% FLOPs，必要时移到 CPU 异步执行；写在 §4 末尾的 "Cost Analysis" 表里 |
| **vLLM 集成踩坑** | 高 | 中 | 已有 vllm_integration 基础；必要时退而用 HuggingFace transformers 的 generate hook，virtual decode loop 跑 benchmark |
| **跨模型泛化不成立** | 中 | 中 | 作为 limitation 诚实报告；提出 per-model fine-tuning 方案（30 min retrain）作为 fallback |
| **InfiniGen / Quest 复现困难** | 高 | 低 | 用论文报告数字 + 自己实现简化版本 |
| **3 周来不及** | **高** | **高** | **MVP 优先**：单模型（Llama-3-8B）+ LongBench + RULER 8K/32K + 4 个 baselines + Tiny-MLP；其他进 appendix 或 rebuttal |
| **被 reviewer 质疑"只是工程"** | 中 | 高 | 强化 §3 理论 framing 和 §6.1 Predictability Study，把 "学习" 而非 "实现" 当成卖点 |
| **匿名性破坏（OrchKvCache 公开）** | 低 | 高 | 论文中代号化系统名（如 "a 4-tier hierarchical storage backend"），不出现 OrchKvCache 字样；GitHub anonymized fork |

---

## 9. 时间表（按 deadline = 5 月 15 日 倒推，请核对实际 deadline）

| 日期 | 里程碑 | 产出 |
|------|--------|------|
| 4/25 – 4/27 | **核对 deadline + 决定投或不投** | 拍板 / 写好 abstract 框架 |
| 4/28 – 5/1 | **E1 Predictability Study** | AUC 数字 → go/no-go gate |
| 5/2 – 5/5 | LAP 训练 pipeline + Tiny-MLP 训练 + ONNX 导出 | 一个能 inference 的模型 |
| 5/6 – 5/8 | Joint policy 集成 + LongBench / RULER 主实验（E2/E3） | 主表数据 |
| 5/9 – 5/11 | 消融 + 跨模型 + 写作 (intro / method) | 论文初稿 70% |
| 5/12 – 5/13 | 写作（experiments / related / abstract）+ 图表 | 完整 draft |
| 5/14 | 内部 review + 修订 | submission-ready |
| 5/15 | **提交** | 投出 |

如果 deadline 是 5 月 22 日，则每个阶段 +1 周缓冲，把 E5 / E7 / E9 都加进 main paper。

如果你判断 3 周内拿不出强结果，**可选路径**：
- **A. 投 NeurIPS Workshop**（如 ML for Systems / Efficient ML），deadline 通常在 8–9 月，要求更松
- **B. 投 ICLR 2027**（10 月 deadline），有 4–5 个月做扎实
- **C. 投 MLSys 2027**（10 月 deadline），系统 + ML 双向友好，OrchKvCache 的工程深度反而是优势
- **D. NeurIPS + MLSys 双投不同侧重**：NeurIPS 走方法（LAP）、MLSys 走系统（OrchKvCache + LAP 完整栈）

---

## 10. 立即行动 TODO（未来 7 天）

- [ ] 核对 NeurIPS 2026 准确 deadline（abstract / full paper）
- [ ] 在 Latex 模板里建空 paper 骨架（intro / method / exp / related 五节）
- [ ] 跑通 OrchKvCache 现有 attention hook，确保能落 attention-trace 到磁盘
- [ ] 设计 LAP 数据格式 schema：`{layer, head, block_id, step, attn_score, label_top_k_in_next_H}`
- [ ] 用 Llama-3-8B + RULER 4K/8K 跑一遍小规模 trace 采集（< 100 prompts，看流程通不通）
- [ ] 训练第一个 Tiny-MLP，目标 AUC > 0.8（即 E1 的 go/no-go）
- [ ] **GO/NO-GO 决策点**（5/1 前）：基于 E1 结果决定继续投 NeurIPS 还是转备投会场

---

## 11. Open Questions（需要 Chet 决定）

1. **作者排序与合作者**：是否需要拉本系统方向的合作者（处理理论部分）？
2. **匿名化策略**：投稿期间 OrchKvCache GitHub repo 是否暂时设为 private？
3. **预算**：3 周冲刺需不需要追加 GPU 资源（如果要跑 70B / 128K context）？
4. **次要会议**：若 NeurIPS 不投或被拒，备投会议优先级（MLSys / EuroSys / OSDI / ICLR）？
5. **是否 carve out**：把 E1 Predictability Study 单独抽出来作为 short paper / workshop submission，作为正文 paper 的"学术广告"？

---

## 12. 后续文档计划

`nips26/` 下后续会拆分：
- `notes/related_work_detailed.md` — 每个 baseline 的详细笔记（论文链接、复现要点、官方代码）
- `notes/lap_design.md` — LAP 模型迭代日志
- `experiments/e1_predictability/` — E1 实验脚本和结果
- `paper/` — Latex 源码（建议 git submodule，便于匿名化）
- `meetings/` — 关键决策记录

---

> **下一步**：先回答 §11 中的 5 个问题，特别是 deadline 核对和 GO/NO-GO 决策点。然后我可以帮你具体写出 §10 的第一个 TODO（attention trace schema 和采集脚本）。
