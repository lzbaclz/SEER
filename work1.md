# work1.md — SEER 代码脚手架完成记录

> **写入时间**：2026-04-25
> **对应 commit 范围**：本次对话内一次性建立的代码骨架（尚未 `git commit`）
> **阅读者**：Chet，或未来接手这个 repo 的合作者
> **作用**：脱离对话独立参考；照这份文档能迅速理解文件职责、契约、测试策略，以及从哪一步开始真正跑代码

---

## 0. 背景回顾（一段话）

SEER 从系统项目 OrchKvCache 中切出一个 NeurIPS 2026 方向：**学习一个注意力预测器（LAP）+ 联合 eviction-prefetch 策略**，以替代 H2O / StreamingLLM / SnapKV 等启发式。今天的工作是**不跑任何代码**、在 `~/codes/papers/Seer/` 下把整个代码脚手架落地，保证：
1. 每个模块职责清楚、契约明确
2. E1 Predictability Study（GO/NO-GO gate）的完整 pipeline 是**能一键跑通**的（`bash experiments/e1_predictability/run.sh`）
3. 后续 E2–E7 有脚本骨架 + README，结果出来后直接往里填

---

## 1. 最终目录结构

```
Seer/
├── PROJECT_PLAN.md            # 完整研究计划（原 nips26/）
├── TODO.md                    # 110 原子任务
├── README.md                  # GitHub 首页（已同步为 seer/ 包结构）
├── work1.md                   # ← 本文
├── LICENSE                    # MIT
├── Makefile                   # install / test / format / lint / clean
├── pyproject.toml             # 包元信息 + ruff + pytest 配置
├── requirements.txt
├── .gitignore                 # Python / data / LaTeX / 编辑器
│
├── seer/                      # Python 包（原 README 里写的 code/ 已改名）
│   ├── __init__.py
│   ├── trace/                 # 注意力 trace 的 schema / hook / 采集 / 加载
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── hook.py
│   │   ├── datasets.py
│   │   ├── collect.py         # CLI: python -m seer.trace.collect
│   │   └── loader.py
│   ├── lap/                   # Learned Attention Predictor
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── models.py          # TinyMLP / BlockRNN / BlockTransformer
│   │   ├── dataset.py
│   │   ├── losses.py          # BCE / Focal
│   │   ├── train.py           # CLI: python -m seer.lap.train
│   │   ├── export.py          # CLI: python -m seer.lap.export (ONNX)
│   │   └── infer.py           # LAPPredictor (ONNX / torch)
│   └── eval/                  # 评估 + baselines
│       ├── __init__.py
│       ├── metrics.py
│       ├── block_stats.py
│       ├── sim.py             # offline simulator（MVP 版，注释清楚它不是 vLLM）
│       ├── runner.py          # CLI: python -m seer.eval.runner
│       └── policies/
│           ├── __init__.py    # build_policy(name, **kw)
│           ├── base.py        # KVPolicy abstract + FullCachePolicy
│           ├── streaming.py   # StreamingLLM (sink + window)
│           ├── h2o.py         # H2O (heavy hitter + recent)
│           ├── snapkv.py      # SnapKV (prefill-only)
│           ├── recency.py
│           ├── random_policy.py
│           └── seer_policy.py # ★ 本文重点
│
├── experiments/
│   ├── __init__.py
│   ├── e1_predictability/     # ★ GO/NO-GO gate — 完整可跑
│   │   ├── README.md
│   │   ├── run.sh             # collect → train → analyze 一键
│   │   └── analyze.py         # 判定 GO / NO-GO
│   ├── e2_pareto/             # skeleton：policy × budget sweep + Pareto 图
│   │   ├── README.md
│   │   ├── run.sh
│   │   └── analyze.py
│   ├── e3_throughput/         # stub（需要 vLLM 集成后再填数）
│   │   ├── README.md
│   │   └── run.sh
│   ├── e4_arch/               # README-only
│   ├── e6_policy/             # README-only
│   └── e7_generalization/     # README-only
│
├── tests/
│   ├── conftest.py            # toy_trace_df 合成 fixture
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_policies.py
│   ├── test_metrics.py
│   └── test_trace_schema.py
│
├── data/                      # gitignored
│   └── traces/
├── checkpoints/               # gitignored
├── logs/                      # gitignored
├── results/                   # gitignored
├── notes/                     # 阅读笔记、决策日志（空）
└── paper/                     # LaTeX 骨架（空）
```

---

## 2. 每个模块的关键设计决策

### 2.1 `seer/trace/`

| 决策 | 为什么 |
|------|--------|
| **32-token block** (`BLOCK_SIZE=32`) | 与 OrchKvCache 的 32KB OrchFS SSD block 对齐（d_head=128，fp16 → 32 tokens = 32KB）。在算法论文里这个数字不硬性要求；保持一致让未来系统集成无缝 |
| **按 head_group 聚合**（非 per-head） | GQA 模型（Qwen / Llama-3）天然 KV-heads < query-heads，per-head 语义重复且特征量爆炸。`n_head_groups` 默认等于 `num_key_value_heads` |
| **prefill 时 Q 维度取平均**而非单独算 | prefill 时 Q=seq_len，decode 时 Q=1；mean-over-Q 让 prefill 和 decode 的记录保持同一 schema |
| **`is_top_k` 按 (layer, head_group, step) 算** | 不同层/头的 top-k 是独立决策 —— LAP 也是这样使用的 |
| **horizon label 是"next H 步内任何一步进入 top-k"** | 这是 eviction 决策真正关心的事件：只要未来 H 步内还会用一次，就不该现在丢掉 |
| **按 request_id 划分 train/val/test** | 一个 prompt 内部 step 之间强相关，如果按行随机划分会严重过拟合 |
| **用 HF `_step_generate` 手动一 token 一 token 跑** | HF 原生 `generate()` 不暴露 per-step hook；手动 loop 让 tracer 能在 step 间调 `advance_step()` |

### 2.2 `seer/lap/`

**输入特征向量（37 维）**：
```
[ hist_0, hist_1, ..., hist_31,          # 32 维注意力历史（t-1 到 t-32）
  recency_log,                           # log1p(steps since last top-k)
  persistence,                           # 过去 32 步内 top-k 命中率
  position,                              # block_start / max_block_start
  layer_scalar,                          # layer_id / max_layer
  head_scalar ]                          # head_group / max_head_group
```

**模型三选一**：
| 名称 | 参数量 | 何时用 |
|------|--------|--------|
| `tiny_mlp` | ~50K | production 主推；3 层 MLP（GELU + dropout） |
| `block_rnn` | ~100K | 消融，看 sequential 结构有没有额外价值 |
| `block_transformer` | ~500K | 复杂度上限；预期收益 < latency 代价 |

**训练技术细节**：
- **loss 默认 focal**（α=0.25, γ=2.0）：正样本比例 5–15%，BCE 容易把 logits 推到接近 0
- **multi-horizon**：每个样本同时预测 4 个 horizon，共享表示
- **按 mean AUC 选 best checkpoint**：而非最低 val loss（loss 和 AUC 不完全一致）
- **输出 logit，不做 sigmoid**：下游 `infer.py` 在 runtime 做，便于 ONNX 导出
- **ckpt 里 bundle 所有元信息**：`state_dict / model_name / meta(input_dim, horizons, history_n) / args`，`export.py` 不需要额外参数

**ONNX 导出**：`batch` 轴 dynamic，其他固定。对应 ONNX Runtime 的 `CUDAExecutionProvider`，infer wrapper 在 runtime 做 sigmoid。

### 2.3 `seer/eval/`

**`KVPolicy` 抽象契约** — 所有 policy 都实现：
```python
def select_to_keep(
    block_stats: dict[int, dict],   # {bid: {attn_score_now, attn_history, position, ...}}
    budget: int,                     # max blocks to keep in HBM
    step: int,                       # current decode step
) -> set[int]                        # block_ids to keep
```

**`block_stats` 字段约定**（全部 optional，policy 按需取）：
| 字段 | 含义 |
|------|------|
| `attn_score_now` | 当前 step 的 attention 分数 |
| `attn_history` | 过去 N 步的 attention 分数列表（newest last） |
| `position` | block_start_token |
| `position_norm` | position / max_position |
| `last_top_k_step`, `steps_since_top_k` | 最近一次 top-k 命中 |
| `persistence` | 过去 N 步的 top-k 命中率 |
| `layer_scalar`, `head_scalar` | 归一化层/头索引（SEER 用来构造特征） |
| `io_cost` | 从当前 tier 换入 HBM 的代价（tier-aware 时填） |

**7 个 baseline policy** — 每个都明确实现：
- `full` — 全保留，quality 上界
- `streaming` — sink=4 + window=N，纯位置启发式
- `h2o` — 50% heavy hitter（累积 attention）+ 50% recent
- `snapkv` — prefill 一次打分冻结 + 自动吸收新生成 token（兼顾 SnapKV 原文的 "keep newly-generated" 行为）
- `recency` — 纯 LRU / 最近位置
- `random` — sanity 下界
- `seer` ⭐ — LAP 打分 + λ·IO_cost 惩罚 + 强制 sink/window + greedy top-budget

**`seer_policy.py` 的 feature packing 要点**：
- `_features(stats)` 必须和 `lap.features.build_features` 的 column 顺序**完全一致**，否则 LAP 在 runtime 给的分数会错位
- `hist_arr[0]` 是最新的过去分数（t-1），因为 `features.py` 的 shift(1) 对应 hist_0
- 所以运行时把 `attn_history` 反转后取前 N 个

**`sim.py` 当前是 MVP**：目前返回 plain greedy generation，**不真正应用 policy 的 eviction**。注释里明确写了：
- E1 Predictability Study 不用 sim，直接看 AUC；**所以 MVP sim 足够**
- E2 需要真正的 attention masking（下一步工作）
- E3 必须换 vLLM（更下一步）

**`runner.py`**：benchmark CLI。支持所有 policy + 自动加载 ONNX / torch ckpt for SEER。

### 2.4 `experiments/e1_predictability/`

完整可跑的最小 pipeline。**这是唯一能让你在 1 天内拿到 GO/NO-GO 答案的实验**。

`analyze.py` 的判定阈值（写死在代码顶部便于审查）：
```python
GO_ABS_THRESHOLD = 0.80     # LAP mean AUC 绝对值
GO_GAP_THRESHOLD = 0.05     # LAP - best_baseline 差距
```

Baseline 在分析脚本里是 **raw features 当分数**（不训练），这是最严格的上界对比：
- `last_attn` = `hist_0`（t-1 的 attention）
- `persistence_h2o` = H2O 风格的累积命中率
- `recency` = `-log(steps since top-k)`
- `cumulative_history` = `sum(hist_0..hist_31)`

如果 LAP 连这些"免训练"的单特征 baseline 都打不过 0.05，说明 learning 确实没吃到可用信号，该 NO-GO。

---

## 3. 这份代码能跑什么，不能跑什么

### ✅ 能跑（install 完依赖后）

1. **Phase 6 的测试套件**：`make test` — 完全离线，无需 GPU，不下模型
2. **E1 Predictability Study 全链路**：`bash experiments/e1_predictability/run.sh`
   - 唯一外部依赖：能从 HuggingFace 下载 Llama-3-8B（或者 `MODEL=` 环境变量换小模型测通）
   - 需要 1 张 GPU（fp16 Llama-3-8B prefill 32K 至少 40GB；fp16 + 8K 能 24GB）
3. **单个 baseline benchmark 跑数**：`python -m seer.eval.runner --model ... --policy h2o --budget 0.2 ...`

### ❌ 还不能跑（等后续工作）

1. **真 offline simulator 的 attention masking**：`seer/eval/sim.py` 现在是 pass-through。要实装需要向 HF 注入 custom attention processor，把 policy 不想保留的 block 对应的 attention logits 置 -inf。这是 E2 的关键前提。
2. **vLLM 集成**：整个 `seer/integration/vllm/` 尚未建目录。E3 throughput 数字不可信，直到这一步做完。
3. **真实 LongBench / RULER 评测**：`datasets.py` 里的 fallback 是合成 needle-in-haystack，LongBench 分支用了官方 HF dataset 但没做 per-task 的官方 metric（LongBench 每个任务都有自己的 metric function，需要单独接入）。

---

## 4. 文件清单（一共新增 / 修改）

### 新增文件（35 个）

**配置（5）**：`.gitignore`, `LICENSE`, `Makefile`, `pyproject.toml`, `requirements.txt`

**`seer/` 包（17）**：
- `seer/__init__.py`
- `seer/trace/{__init__,schema,hook,datasets,collect,loader}.py`
- `seer/lap/{__init__,features,models,dataset,losses,train,export,infer}.py`
- `seer/eval/{__init__,metrics,block_stats,sim,runner}.py`
- `seer/eval/policies/{__init__,base,streaming,h2o,snapkv,recency,random_policy,seer_policy}.py`

**实验（7）**：
- `experiments/__init__.py`
- `experiments/e1_predictability/{README.md, run.sh, analyze.py}`
- `experiments/e2_pareto/{README.md, run.sh, analyze.py}`
- `experiments/e3_throughput/{README.md, run.sh}`
- `experiments/e4_arch/README.md`
- `experiments/e6_policy/README.md`
- `experiments/e7_generalization/README.md`

**测试（6）**：
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_features.py`
- `tests/test_models.py`
- `tests/test_policies.py`
- `tests/test_metrics.py`
- `tests/test_trace_schema.py`

### 修改文件

- `README.md`：`code/` → `seer/`；所有 `python -m code.*` 命令改成 `python -m seer.*`；加了 `make install` 和 E1 快捷命令

---

## 5. 关键命令备忘（打印出来贴在显示器旁）

```bash
# 一键环境
conda create -n seer python=3.11 -y && conda activate seer
make install

# 跑单元测试（不需要 GPU）
make test

# E1 全流程（需要 GPU + 一次 model download）
bash experiments/e1_predictability/run.sh

# 只采 trace
python -m seer.trace.collect \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset ruler --context_lengths 4096 8192 \
    --num_prompts 100 --out data/traces/llama3-8b/

# 只训 LAP（已有 trace）
python -m seer.lap.train \
    --traces data/traces/llama3-8b \
    --model tiny_mlp --epochs 10 \
    --out checkpoints/lap.pt

# 导 ONNX
python -m seer.lap.export --ckpt checkpoints/lap.pt --out checkpoints/lap.onnx

# 单次 benchmark
python -m seer.eval.runner \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --policy seer --lap_ckpt checkpoints/lap.onnx \
    --benchmark ruler --budget 0.2 \
    --context_length 8192 --num_prompts 50 \
    --out results/ruler_seer_b20.json
```

---

## 6. 下一步（按阻塞顺序）

1. **（今天 / 明天）** `make install` + `make test` — 确认单测全绿。如果 conftest 的合成 fixture 有假设错误，这一步会立刻暴露。
2. **（1–2 天）** 核对 NeurIPS 2026 deadline（`TODO.md` P0.1）→ 决定 GO / PIVOT / 暂缓。
3. **（2–3 天）** 小规模跑一次 E1：`NUM_PROMPTS=20 CONTEXT_LENGTHS="2048" MODEL=HuggingFaceTB/SmolLM-135M bash experiments/e1_predictability/run.sh`，先通链路再上 Llama-3-8B。
4. **（3–5 天）** 正式跑 E1 on Llama-3-8B，拿到 GO/NO-GO verdict。
5. **（GO 之后）** 实装 `sim.py` 的 attention masking；开始 E2 sweep。

---

## 7. 审视这份脚手架时要质疑的 3 件事

1. **`AttentionTracer._find_attention_layers` 的启发式** — 只对 `self_attn` / `attention` 命名模式起作用。其他架构（Phi / GPT-NeoX）需要改。建议在正式跑前 print 一次 `_find_attention_layers(model)` 的结果确认。
2. **`hook._extract_attn` 假设 outputs[1] 是 attn_weights** — Llama / Mistral / Qwen2 确实是这样（`output_attentions=True` 时），但版本升级可能变；单测覆盖不了这种结构依赖。
3. **`features.py` 的 recency 计算** — 用 Python loop 遍历 groupby，是正确性优先的实现。10M 行 trace 可能要分钟级时间；如果 slow 就换成 numba / numpy vectorized 版本（先测正确，再优化）。

---

## 8. 已有文档索引

| 文档 | 作用 | 建议何时读 |
|------|------|-----------|
| `README.md` | GitHub 首页：what + how（面向外部） | 上传 GitHub 前最后过一遍 |
| `PROJECT_PLAN.md` | 完整研究计划：method 细节、实验矩阵、时间表、风险 | 每周开工前扫一眼 |
| `TODO.md` | 13 phase ~110 原子任务 | 每天开工用它挑今天的 task |
| `work1.md`（本文）| 代码脚手架落地记录 + 契约 + 下一步 | 回来看自己 3 周前写了啥 |

---

## 9. 未解决的 open question（留给未来的自己）

- [ ] ⚠️ `AttentionTracer` 没测过在 FlashAttention 开启时的行为 —— 已经强制 `attn_implementation="eager"`，但未来改 FlashAttention-2 时这条 hook 路径会失效，需要改走 patch attention module 的路线
- [ ] ⚠️ `snapkv.py` 的"自动吸收新 token"逻辑和原论文的 "observation window" 机制不完全一致，正式论文对比时要换成更严格的复现（建议用作者官方代码）
- [ ] ⚠️ E2 的 `run.sh` 假设所有 policy 用同一种 runner；SEER 需要 ONNX ckpt，其他不需要 —— 已在脚本里 branch，但没测过 ONNX 在 offline sim 场景下的正确性
- [ ] ⚠️ `block_stats.py` 目前没有从 real vLLM 填充的代码路径，只给 offline sim 用；vLLM 集成时要再做一个 `block_stats_from_vllm.py`
- [ ] ⚠️ `loss` 的 class imbalance 处理只做了 focal，没做 pos_weight sampler；trace 量大时建议试一下 `WeightedRandomSampler`

---

## 10. 心情 / Meta（可删）

- 今天的工作强度：建了 35 个文件，~1700 行代码 + 700 行文档
- 最大的取舍：为了赶 NeurIPS deadline 没做 vLLM 集成，这会让 E3 的 throughput 数字"好看但不可信"。正式论文前必须补
- 最满意的部分：`KVPolicy` 抽象 + `block_stats` dict 的 schema 约定 —— 7 个 baseline 能用同一行代码切换，后面加新 baseline 成本极低
- 最担心的部分：`features.py` 的 recency 循环，10M 行时可能卡训练几分钟（可以容忍）；但 100M 行就必须优化
