# NeurIPS 2026 TODO List — SEER

> **SEER** = **S**elective **E**viction via **E**xpected **R**ecall
> Paper title (working): *SEER: Learning to Evict KV-Cache by Predicting Future Attention*

> 同步配套文档：`PROJECT_PLAN.md`
> 状态符号：`[ ]` 待办 / `[~]` 进行中 / `[x]` 完成 / `[-]` 放弃 / `[?]` 阻塞
> 优先级：⭐⭐⭐ critical path / ⭐⭐ important / ⭐ nice-to-have
> 估时单位：h = 小时，d = 天（按 8h 工作日计）

---

## 0. 决策门 (Phase 0) — 4/25 ~ 4/27 ⭐⭐⭐

> **目的**：在投入巨大工作量前，确定是否真的要投 NeurIPS 2026。

- [ ] **P0.1 核对 NeurIPS 2026 准确 deadline** ⭐⭐⭐ (0.5h)
  - 访问 https://neurips.cc/Conferences/2026/CallForPapers
  - 记录：abstract deadline、full paper deadline、supplementary deadline、rebuttal 时间
  - DONE WHEN：在 `nips26/notes/deadlines.md` 写下三个时间点 + 时区
  - 阻塞所有后续任务

- [ ] **P0.2 计算可用工作日 & 评估可行性** ⭐⭐⭐ (0.5h)
  - 用 deadline 倒推，扣除休息日、出差、其他承诺
  - 若可用工作日 < 15，标记"高风险投递"，准备 fallback 会场（MLSys / ICLR / Workshop）
  - DONE WHEN：在 `notes/timeline.md` 写下"实际可用工作日 X 天"+ 风险评级

- [ ] **P0.3 答复 PROJECT_PLAN.md §11 的 5 个 open questions** ⭐⭐⭐ (1h)
  - 作者排序 / 合作者
  - 匿名化策略
  - GPU 预算
  - 备投会议优先级
  - 是否 carve out E1 单独发 workshop
  - DONE WHEN：每个问题在 `notes/decisions.md` 有一段决定 + 理由

- [ ] **P0.4 GO / NO-GO 决策（基于 P0.1–P0.3）** ⭐⭐⭐ (0.5h)
  - 三种结果：(A) GO NeurIPS / (B) PIVOT 到 MLSys 2027 / (C) 暂缓全部
  - DONE WHEN：决定写入 `notes/decisions.md` 顶部 + 通知合作者

---

## 1. 项目基础设施 (Phase 1a) — 4/28 ~ 4/29 ⭐⭐⭐

### 1.1 仓库 & 工作流

- [ ] **P1.1 在 nips26/ 下建立目录结构** (0.5h)
  ```
  nips26/
  ├── PROJECT_PLAN.md  ✓
  ├── TODO.md          ✓
  ├── notes/           # 决策、阅读笔记
  ├── code/            # LAP 训练 / 数据处理脚本（Python）
  │   ├── lap/         # 模型定义 + 训练
  │   ├── trace/       # Attention trace 采集与解析
  │   └── eval/        # benchmark 编排脚本
  ├── experiments/     # 每个实验一个子目录（e1_predictability, e2_pareto, ...）
  ├── data/            # （gitignore）trace 文件、模型权重
  └── paper/           # Latex 源码（git submodule 或独立 repo）
  ```
  - DONE WHEN：所有目录创建，`.gitignore` 写好（忽略 data/、*.pt、*.onnx）

- [ ] **P1.2 拉 NeurIPS 2026 LaTeX 模板，建空 paper 骨架** ⭐⭐⭐ (1h)
  - 从 NeurIPS 官方下载 style files
  - 建好 9 节空 sections：abstract / intro / related / preliminaries / method / experiments / discussion / limitations / conclusion
  - 设置 bib 文件骨架
  - DONE WHEN：能 `pdflatex` 编译出空 PDF

- [ ] **P1.3 创建 Python 环境** (1h)
  - `conda create -n nips26 python=3.11`
  - 装：torch, transformers, vllm, datasets, pandas, sklearn, onnx, onnxruntime-gpu, matplotlib
  - 导出 `environment.yml` 提交到 repo
  - DONE WHEN：能 `import vllm` + load Llama-3-8B 推理一条样本

- [ ] **P1.4 OrchKvCache 仓库匿名化预案** ⭐⭐ (1h)
  - 准备一个 `anonymized` 分支，把所有"OrchKvCache"、作者名、commit 中的署名替换为占位符
  - 暂不 push public，等 deadline 前一周再做
  - DONE WHEN：`scripts/anonymize.sh` 写好并测试通过

### 1.2 实验编排骨架

- [ ] **P1.5 写通用 benchmark runner** ⭐⭐ (3h)
  - `code/eval/runner.py`：参数化（model、dataset、policy、budget、seed），输出统一 JSON
  - 统一 metric：F1 / EM / ROUGE / throughput / TTFT / TPOT / peak_HBM
  - DONE WHEN：能跑 Llama-3-8B + LongBench narrative_qa 子集 50 条 + Full Cache，输出有效 JSON

- [ ] **P1.6 写 baseline policy 抽象层** ⭐⭐ (2h)
  - `code/eval/policies.py`：抽象类 `KVPolicy`，方法 `select_to_keep(blocks, budget) -> set[block_id]`
  - 实现 `FullCachePolicy`（不淘汰）
  - DONE WHEN：抽象 + Full Cache 能在 P1.5 runner 中替换运行

---

## 2. E1 Predictability Study (Phase 1b) — 4/29 ~ 5/1 ⭐⭐⭐

> **目的**：验证 H1（注意力可预测性）。这是 GO/NO-GO 决策的根本依据。

### 2.1 Attention Trace 采集

- [ ] **P2.1 设计 trace schema** ⭐⭐⭐ (1h)
  - 字段：`request_id, layer, head_group, block_id, step, attn_score_aggr, is_in_top_k_now, future_top_k_horizons={1,4,16,64}`
  - block_id 按 32-token 分块（与 OrchKvCache 一致）
  - 决定 head 聚合：sum / mean / max？建议 mean（与 GQA 兼容）
  - DONE WHEN：写在 `notes/trace_schema.md`，有示例 row

- [ ] **P2.2 修改 OrchKvCache attention_hook 落 trace** ⭐⭐⭐ (3h)
  - 复用 `python/orchkv/vllm_integration/attention_hook.py`
  - 增加 trace dump 到 parquet（小文件，列存便于后续训练）
  - 加开关：`ORCHKV_DUMP_TRACE=1`，避免影响正常 inference
  - DONE WHEN：跑 1 条 RULER prompt，得到一个 parquet 文件且字段齐全

- [ ] **P2.3 大规模 trace 采集（Llama-3-8B）** ⭐⭐⭐ (4h, 含 GPU 时间)
  - 数据集：RULER 4K/8K/16K + LongBench 子集（5 个任务，每个 50 条）
  - 单卡 A100 估算：~6h GPU 时间，并行两卡 ~3h
  - 输出：`data/traces/llama3-8b/*.parquet`，总量 ~10GB
  - DONE WHEN：trace 总样本数 > 5M block-step

### 2.2 可预测性分析

- [ ] **P2.4 构建训练/验证集划分** (1h)
  - 按 request_id 划分，避免泄漏
  - 80% train / 10% val / 10% test
  - DONE WHEN：`code/lap/data.py` 有 `make_splits()` 函数

- [ ] **P2.5 训练第一个 Tiny-MLP（最小 LAP）** ⭐⭐⭐ (4h)
  - 输入：过去 32 步 attn score + recency + position（共 ~36 维）
  - 模型：3 层 MLP，hidden=128，参数 ~50K
  - Multi-horizon BCE loss（H ∈ {1, 4, 16, 64}）
  - 训练时间 < 2h
  - DONE WHEN：在 val 上 horizon=4 的 AUC 落盘

- [ ] **P2.6 计算所有 baselines 的 predictability 上界** ⭐⭐⭐ (2h)
  - 用同样的 train/val/test，计算：
    - Recency-only（按 last access step 排序）
    - Cumulative attention（H2O 的逻辑）
    - Random
  - 所有方法在同一 val set 上算 AUC、NDCG、Recall@k
  - DONE WHEN：`experiments/e1_predictability/results.csv` 有完整对比表

- [ ] **P2.7 GO/NO-GO 评估 ⭐⭐⭐ (1h)**
  - 准则：horizon=4 时 LAP 的 AUC > H2O baseline +0.05 且绝对值 > 0.80 → **GO**
  - 准则不达：评估是否(a) 增大 LAP 容量 +1 天 / (b) 转方向 / (c) 转 workshop
  - DONE WHEN：在 `notes/decisions.md` 记录 GO/NO-GO 二次决策

---

## 3. LAP 模型与训练 (Phase 2) — 5/2 ~ 5/5 ⭐⭐⭐

### 3.1 模型架构

- [ ] **P3.1 实现 Block-RNN LAP（中等复杂度）** ⭐⭐ (4h)
  - 1 层 GRU，hidden=64
  - 输入：(history_window=32, feat_dim≈36)
  - DONE WHEN：训练完 + AUC 与 Tiny-MLP 对比记录

- [ ] **P3.2 实现 Block-Transformer LAP（最复杂版本）** ⭐ (4h)
  - 1 层 self-attention（query 一个 block，key/value 是其他 blocks 的当前状态）
  - 跨 block 共享参数
  - DONE WHEN：训练完 + AUC 记录；如果不显著优于 RNN 就放弃，仅作 ablation

- [ ] **P3.3 选定 production model** ⭐⭐⭐ (1h)
  - 综合 AUC、推理 latency、参数量，选一个作为主推
  - 默认假设是 Tiny-MLP 或 Block-RNN
  - DONE WHEN：`code/lap/model.py` 有 `ProductionLAP` 类指向选定模型

### 3.2 训练 Pipeline

- [ ] **P3.4 完善训练脚本** ⭐⭐⭐ (3h)
  - `code/lap/train.py`：argparse 支持模型 / 超参 / 数据路径
  - 加 wandb 或 tensorboard logging
  - 加 checkpoint resume
  - DONE WHEN：`python train.py --model tiny_mlp` 一行能复现 P2.5

- [ ] **P3.5 训练数据增强（如有时间）** ⭐ (3h)
  - 加 dropout on attention features
  - 加 attention-rank shuffling 作为 noise
  - DONE WHEN：增强后 val AUC 不降反升或持平

- [ ] **P3.6 ONNX 导出 + 推理 latency profile** ⭐⭐⭐ (2h)
  - 导出 ONNX，在 CPU 和 GPU 上各 profile 推理时延
  - 目标：单次预测 (per-block) < 5µs；batched 1024 blocks < 1ms
  - DONE WHEN：`experiments/e1_predictability/lap_latency.json` 有数据

### 3.3 跨模型预训练（推后但要规划）

- [ ] **P3.7 Qwen2.5-7B trace 采集** ⭐⭐ (4h GPU)
  - 同 P2.3，换模型
  - DONE WHEN：data/traces/qwen2.5-7b/ 有数据

- [ ] **P3.8 Mistral-7B trace 采集** ⭐⭐ (4h GPU)
  - 同上
  - DONE WHEN：data/traces/mistral-7b/ 有数据

- [ ] **P3.9 跨模型训练 LAP（zero-shot 泛化实验的训练步）** ⭐⭐ (3h)
  - 在 Llama-3-8B trace 上训练，在 Qwen / Mistral 上测试
  - 记录跨模型迁移的 AUC drop
  - DONE WHEN：`experiments/e7_generalization/results.csv` 有数字

---

## 4. Joint Policy 集成 (Phase 3a) — 5/6 ~ 5/7 ⭐⭐⭐

### 4.1 在 OrchKvCache 中落地

- [ ] **P4.1 写 LAP C 推理 wrapper** ⭐⭐⭐ (3h)
  - `OrchKvCache/src/classifier/lap_predictor.{h,c}`
  - 用 ONNX Runtime C API 加载模型
  - 接口：`lap_predict(features, n_blocks) -> scores[]`
  - DONE WHEN：单测能跑通

- [ ] **P4.2 实现 Joint Eviction-Prefetch 策略** ⭐⭐⭐ (4h)
  - `OrchKvCache/src/scheduler/joint_policy.{h,c}`
  - utility = $\hat{p}_i - \lambda \cdot \text{IO\_cost}_i$
  - Greedy top-B 选择
  - 强制保留 attention sink + sliding window
  - DONE WHEN：单元测试覆盖：(a) 全 HBM 时不动 (b) 极小预算时只保留 sink

- [ ] **P4.3 把 joint policy 接入 vllm_integration** ⭐⭐⭐ (3h)
  - 修改 `python/orchkv/vllm_integration/connector.py`
  - 通过 env var 切换：`ORCHKV_POLICY=heuristic|h2o|streaming|lap`
  - DONE WHEN：4 种策略均可运行 RULER 8K 不崩溃

### 4.2 端到端 Sanity Check

- [ ] **P4.4 端到端 sanity 测试** ⭐⭐⭐ (2h)
  - Llama-3-8B + RULER 8K + 50% budget + 4 种 policy
  - 验证 LAP 至少打平最差 baseline（Random）
  - DONE WHEN：4 个 policy 的 F1 都有数字记录

---

## 5. 主实验 (Phase 3b) — 5/7 ~ 5/9 ⭐⭐⭐

### 5.1 Baselines 实现

- [ ] **P5.1 StreamingLLM baseline** ⭐⭐⭐ (2h)
  - `code/eval/policies/streaming.py`
  - sink=4, sliding=2K
  - DONE WHEN：跑通 LongBench 1 个任务

- [ ] **P5.2 H2O baseline** ⭐⭐⭐ (3h)
  - 用作者代码或自己实现：50% heavy hitter + 50% recent
  - DONE WHEN：在 RULER 8K 复现作者论文报告数字 ±5%

- [ ] **P5.3 SnapKV baseline** ⭐⭐⭐ (3h)
  - Prefill 末尾 32 token attention + cluster pooling
  - DONE WHEN：跑通 + 数字与论文一致

- [ ] **P5.4 Quest baseline** ⭐⭐ (3h)
  - Page-level attention 估计
  - DONE WHEN：跑通；若复现困难，标记为 "follow paper numbers" 并在 §6 注明

- [ ] **P5.5 Recency / Random sanity baselines** (1h)
  - DONE WHEN：实现 + 跑通

### 5.2 E2 Quality-Budget Pareto

- [ ] **P5.6 E2 实验脚本** ⭐⭐⭐ (2h)
  - `experiments/e2_pareto/run.sh`
  - matrix：3 模型 × 4 预算 (10/20/40/80%) × 6 baseline + SEER × 2 dataset (LongBench, RULER)
  - 总跑次：3×4×7×2 = 168 runs
  - DONE WHEN：脚本能从 csv 配置文件展开所有 run

- [ ] **P5.7 E2 主实验跑数** ⭐⭐⭐ (16h GPU)
  - 用 P5.6 脚本，单卡 ~40h，双卡 ~20h
  - 中途 checkpoint，意外中断可续跑
  - DONE WHEN：所有 168 runs 落盘 + 错误数 < 5%

- [ ] **P5.8 E2 Pareto 图绘制** ⭐⭐⭐ (2h)
  - X=budget，Y=task quality，每个 baseline 一条线
  - 3 个 subplot（每个模型一个）
  - DONE WHEN：`paper/figures/e2_pareto.pdf` 生成

### 5.3 E3 Throughput-Quality

- [ ] **P5.9 端到端 throughput 测量脚本** ⭐⭐⭐ (3h)
  - 固定 quality 目标（例如 LongBench 平均 F1 = 0.7），调整 budget 找各 baseline 的最小 budget，然后测 throughput
  - 测 8K / 32K / 128K context
  - DONE WHEN：`experiments/e3_throughput/results.csv`

- [ ] **P5.10 E3 跑数** ⭐⭐⭐ (12h GPU)
  - DONE WHEN：所有数据落盘

- [ ] **P5.11 E3 表格 / 图** (2h)
  - DONE WHEN：`paper/figures/e3_throughput.pdf`

---

## 6. Ablations & Generalization (Phase 4) — 5/9 ~ 5/11 ⭐⭐

> 时间紧时，只做 E4 + E6，其他放 appendix 或 rebuttal。

- [ ] **P6.1 E4 LAP 架构消融** ⭐⭐ (4h)
  - MLP / RNN / Transformer 三种在 RULER 8K 上的 quality + latency
  - DONE WHEN：`experiments/e4_arch/results.csv`

- [ ] **P6.2 E5 特征消融** ⭐ (4h)
  - 逐一去掉 history / recency / position / persistence
  - DONE WHEN：5 行表格

- [ ] **P6.3 E6 策略消融（最重要）** ⭐⭐⭐ (4h)
  - LAP-only-eviction vs LAP-only-prefetch vs Joint
  - DONE WHEN：3 行对比表 + 解读段落

- [ ] **P6.4 E7 跨模型泛化** ⭐⭐ (8h GPU)
  - 训 Llama-3-8B，零样本测 Qwen2.5-7B / Mistral-7B-v0.3
  - 对比：(a) 直接迁移 (b) 30min finetune
  - DONE WHEN：`experiments/e7_generalization/`

- [ ] **P6.5 E8 系统 IO-tier 价值（appendix）** ⭐ (4h)
  - 关闭 NVM / 关闭 SSD，看 throughput 下降
  - DONE WHEN：appendix 表

- [ ] **P6.6 E9 Predictor overhead（method section 内联）** ⭐⭐⭐ (1h)
  - 主流硬件下 LAP 推理占总时延 < 0.1%
  - DONE WHEN：1 个数字 + 1 段说明

---

## 7. 写作 (Phase 5) — 5/11 ~ 5/13 ⭐⭐⭐

> 写作建议：先写 method（最清楚的部分） → experiments（数据驱动） → related → intro → abstract → discussion → limitations。

### 7.1 主体章节

- [ ] **P7.1 Method §4 草稿** ⭐⭐⭐ (4h)
  - LAP 模型 + 联合策略 + 与 backend 的接口
  - 1 张架构图（method overview）
  - DONE WHEN：~3 页 + 1 张图

- [ ] **P7.2 Experiments §6 草稿** ⭐⭐⭐ (6h)
  - 6.0 setup（模型/数据/baseline/metric）
  - 6.1 Predictability Study (E1)
  - 6.2 Quality-Budget Pareto (E2)
  - 6.3 Throughput-Quality (E3)
  - 6.4 Ablations (E4/E6)
  - 6.5 Generalization (E7)
  - DONE WHEN：~3.5 页 + 4 张图

- [ ] **P7.3 Related Work §3 草稿** ⭐⭐⭐ (3h)
  - 严格按 PROJECT_PLAN §3 的对比表展开
  - 每个 baseline 必须有"why our method differs"
  - DONE WHEN：~1 页

- [ ] **P7.4 Introduction §1 草稿** ⭐⭐⭐ (4h)
  - 第一段：长上下文 LLM 推理瓶颈
  - 第二段：现有方法的 fundamental limitation（启发式 vs. 学习）
  - 第三段：本文 contributions（3 条 bullet）
  - DONE WHEN：~1.5 页 + 1 张 teaser 图

- [ ] **P7.5 Preliminaries §2 草稿** ⭐⭐ (2h)
  - KV-Cache 基础 + attention top-k 抽象 + 存储层 IO 模型
  - DONE WHEN：~0.5 页

- [ ] **P7.6 Discussion / Limitations §7-8** ⭐⭐ (2h)
  - 诚实写明：训练数据依赖、单 GPU 实验、未在 70B 上验证、等等
  - DONE WHEN：~0.5 页

- [ ] **P7.7 Abstract** ⭐⭐⭐ (1h)
  - 4 句：(背景)(问题)(方法)(结果)
  - DONE WHEN：< 200 词

- [ ] **P7.8 Conclusion §9** ⭐ (0.5h)
  - DONE WHEN：1 段

### 7.2 图表

- [ ] **P7.9 Teaser figure（Figure 1）** ⭐⭐⭐ (3h)
  - 直观展示 SEER 的 idea：past attention + LAP → future top-k → eviction & prefetch
  - DONE WHEN：`paper/figures/teaser.pdf`

- [ ] **P7.10 Architecture figure** ⭐⭐ (2h)
  - LAP + Joint Policy + 4-tier hierarchy
  - DONE WHEN：`paper/figures/architecture.pdf`

- [ ] **P7.11 检查所有图表清晰度（300+ DPI）** ⭐⭐ (1h)
  - DONE WHEN：所有图都是 vector PDF 或 ≥300 DPI

### 7.3 Bibliography

- [ ] **P7.12 收集 ~50 篇引用** ⭐⭐ (3h)
  - 主要类别：long-context inference、KV-cache、attention sparsity、online learning
  - DONE WHEN：`paper/refs.bib` 有 50+ 条目

---

## 8. 提交 (Phase 6) — 5/14 ~ 5/15 ⭐⭐⭐

### 8.1 内部 Review

- [ ] **P8.1 自审 checklist** ⭐⭐⭐ (3h)
  - claim 是否被实验支持
  - baseline 是否合理
  - 数字是否一致（abstract / intro / experiments）
  - 限制是否诚实
  - DONE WHEN：checklist 全过

- [ ] **P8.2 找 1-2 位合作者预审** ⭐⭐⭐ (1d 等待 + 4h 修订)
  - 至少留 24h 给 reviewer 看
  - DONE WHEN：返回 comments 已 address

### 8.2 匿名化

- [ ] **P8.3 论文匿名化** ⭐⭐⭐ (2h)
  - 检查 author 字段、致谢、引用自己时用 third person、figure 中无作者标识
  - 系统名 OrchKvCache → 占位符（如 "HierKV"）
  - DONE WHEN：检查清单全过

- [ ] **P8.4 代码匿名化** ⭐⭐⭐ (3h)
  - 把 anonymized 分支推到 GitHub anonymous GitHub
  - 移除所有作者信息
  - DONE WHEN：anonymous repo URL 可访问

- [ ] **P8.5 Supplementary material 打包** ⭐⭐ (2h)
  - 训练好的模型 / 代码 / 额外实验
  - DONE WHEN：单个 zip < 100MB

### 8.3 提交

- [ ] **P8.6 OpenReview 账号检查** ⭐⭐⭐ (0.5h)
  - 提前注册，避免 deadline 当天注册延迟
  - DONE WHEN：能登录 OpenReview NeurIPS 2026 track

- [ ] **P8.7 Abstract 提交（早于 full paper）** ⭐⭐⭐ (0.5h)
  - DONE WHEN：abstract submitted（注意 NeurIPS 通常要求 abstract 提前 4-7 天）

- [ ] **P8.8 Full paper 提交** ⭐⭐⭐ (1h)
  - 包括：main pdf、supplementary、code、checklist
  - **目标：deadline 前 6 小时提交，避开服务器拥堵**
  - DONE WHEN：收到 OpenReview 确认邮件

- [ ] **P8.9 NeurIPS Reproducibility / Ethics Checklist** ⭐⭐⭐ (1h)
  - 仔细回答每条
  - DONE WHEN：已附在论文末尾

---

## 9. Rebuttal 准备 (Phase 7) — Aug 2026 ⭐⭐

> 通常 NeurIPS 8 月发 review，作者有 1 周 rebuttal 期。

- [ ] **P9.1 监控 review 发布** (0.1h/d)

- [ ] **P9.2 rebuttal 实验预案** ⭐⭐ (备份)
  - 提前准备：70B 实验、更多 baseline、长 context（256K）实验脚本
  - DONE WHEN：脚本准备好，rebuttal 期 1-2 天就能跑

- [ ] **P9.3 撰写 rebuttal** ⭐⭐⭐ (2d)
  - 每个 reviewer 一段，逐点回应
  - DONE WHEN：在 rebuttal 截止前 24h 内提交

---

## 10. 横切关注点（贯穿全程）

### 10.1 文献阅读

- [ ] **P10.1 精读 H2O 论文 + 代码** ⭐⭐⭐ (4h)
- [ ] **P10.2 精读 SnapKV 论文 + 代码** ⭐⭐⭐ (3h)
- [ ] **P10.3 精读 Quest 论文 + 代码** ⭐⭐ (3h)
- [ ] **P10.4 精读 InfiniGen 论文** ⭐⭐ (2h)
- [ ] **P10.5 精读 StreamingLLM** ⭐⭐ (2h)
- [ ] **P10.6 精读 Scissorhands** ⭐ (2h)
- [ ] **P10.7 精读 PyramidKV / LESS / Magic-PIG（择一）** ⭐ (3h)
- [ ] **P10.8 维护阅读笔记 `notes/related_work_detailed.md`** ⭐⭐ (持续)

### 10.2 实验复现监控

- [ ] **P10.9 每天检查 GPU 任务队列** ⭐⭐⭐ (持续, 0.2h/d)
- [ ] **P10.10 实验失败时立即 root-cause（避免静默失败积累）** ⭐⭐⭐ (持续)

### 10.3 写作素材积累

- [ ] **P10.11 每完成一个实验，立即在 `notes/findings.md` 记一段中文 takeaway** ⭐⭐ (持续, 0.3h/exp)
- [ ] **P10.12 每周整理一次 figure 雏形（用真实数据，哪怕粗糙）** ⭐⭐ (持续, 1h/w)

### 10.4 备份与版本控制

- [ ] **P10.13 每天 git commit + push 一次** ⭐⭐⭐ (持续)
- [ ] **P10.14 实验结果（trace / model）weekly 备份到外部存储** ⭐⭐ (持续)

---

## 11. 备投策略（如 NeurIPS 失败 / pivot）

> 如果 P0.4 决定 PIVOT，或 P2.7 GO/NO-GO 失败，激活以下分支。

- [ ] **P11.1 MLSys 2027 准备**（deadline 通常 10 月）⭐⭐
  - 重写 framing：系统贡献 + 学习扩展并重
  - 增加 §：完整的 4-tier 系统设计、IO 流水线 profiling、scheduling 微基准
  - DONE WHEN：MLSys 版 outline 写完

- [ ] **P11.2 ICLR 2027 准备**（deadline 通常 9 月）⭐⭐
  - 与 NeurIPS 形态最接近，可直接复用 90%
  - 强化理论 §（lemma 1-2 的证明细节）
  - DONE WHEN：ICLR outline + 改动列表

- [ ] **P11.3 NeurIPS Workshop（如 ML for Systems）** ⭐
  - deadline 通常 8 月
  - 4-page short paper，重点 E1 + E2
  - DONE WHEN：workshop short paper draft

---

## 12. 进度追踪汇总（每周更新）

| 周 | 计划完成 | 实际完成 | 关键风险 |
|---|---------|---------|---------|
| 4/25–5/1 (Week 1) | P0.1–P0.4, P1.1–P1.6, P2.1–P2.7 | | |
| 5/2–5/8 (Week 2) | P3.1–P3.9, P4.1–P4.4, P5.1–P5.5 | | |
| 5/9–5/15 (Week 3) | P5.6–P5.11, P6.*, P7.*, P8.* | | |

---

## 13. 决策日志

> 每个重大决策（model 选型、baseline 取舍、实验范围调整）记一行：日期 + 决策 + 理由。

```
2026-04-25  创建 TODO list  | NeurIPS 2026 投递准备启动
```

---

> **使用方式**：每天开工前看一眼"当周计划"，用 `[~]` 标记当前 in-progress 任务（同时只允许 1-2 个），完成后改 `[x]`。每完成 P-级任务就 git commit。
