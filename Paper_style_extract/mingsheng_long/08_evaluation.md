# 08 · 实验方法学（THUML 的另一项金牌）

## 实验三件套

THUML 的实验跟 Patrick 组完全不同的 flavor：

| 维度 | Patrick 组 (systems) | THUML (ML) |
|---|---|---|
| 数据 | 几个真实 trace（YCSB, Hadoop） | 6-10 个 benchmark 数据集 |
| Baseline | 3-5 个 prior systems | **12-15** 个 prior models |
| 主表 | 单 testbed 单任务 | 多任务 × 多数据集大表 |
| 实现验证 | 嫁接 HDFS / QFS prototype | Time-Series-Library 统一实现 |
| Showcase | latency p99 plot | attention map / forecast curve / t-SNE |
| Ablation | "+primitive 1, +2, +3" 累加表 | "remove component / replace with X" 表 |

THUML 的实验美在 **breadth + 标准化**：在 Time-Series-Library 这个统一 codebase 里把 12-15 个方法都跑一遍，是为人称道的"为人民服务"行为。

## §5 子节标准结构

```
5  Experiments

5.1 Setup
    数据集（dataset table）+ baselines（method list）+ metrics + protocols + hyperparams

5.2 Main Results
    一张 / 两张主表，覆盖所有数据集 × 所有 baselines

5.3 Model Analysis (任务通用化)
    跨 4-5 个任务的"task-general" 验证（TimesNet 风格）

5.4 Ablation Study
    每个 component 单独开关 / 替换 / 移除

5.5 Showcase / Visualization
    几个 case 的可视化（attention map, forecasting curves, t-SNE）

5.6 Efficiency / Computational Cost
    参数量、FLOPs、训练时间、推理 latency
```

5-6 个子节，每个 0.4-0.6 页，加起来 §5 大约 3 页。

## 5.1 Setup 子节填空模板

```
Datasets.
We evaluate on N datasets covering K applications:
    - 〈数据集 1〉 (〈application〉)
    - 〈数据集 2〉
    - ...
Statistics are summarized in Table 1.

Baselines.
We compare with M state-of-the-art methods, classified into K categories:
    - Category A: Method1 [12], Method2 [34], ...
    - Category B: ...

Implementation.
All methods are implemented in 〈Time-Series-Library / Transfer-Learning-Library〉
under the same training protocol (Adam optimizer, ...). For fairness, we use
〈input window L = ...〉, 〈horizon H = ...〉, batch size = ..., 〈epochs〉.
Hyperparameters are tuned on the validation set; details in Appendix B.

Metrics.
We report 〈MSE / MAE / Accuracy / F1〉 averaged over N runs with different seeds.
```

特点：

- **数据集分类**到 application（让 reviewer 觉得 evaluation 覆盖广）
- **baselines 分类**到 method category（让 reviewer 觉得"对手都有代表"）
- **统一 framework**（Time-Series-Library / Transfer-Learning-Library）—— trust signal
- **多 seed + 平均** ——尤其 NeurIPS / ICML 现在很在意

## 5.2 主表样式

THUML 的主表是论文里**最多笔墨打磨**的元素。pattern：

```
                          ETTh1                ETTm1                ECL
                       MSE     MAE          MSE     MAE          MSE     MAE
Method 1               0.123   0.234        ...                  ...
Method 2               ...
...
Method 14              ...
〈Ours〉                **0.098** **0.187** ... ... ... ...

          Highlight (rule below)
```

- 每个数据集分两列（MSE / MAE）或三列（forecasting horizons）
- **bold 表示最佳，underline 表示次佳**（universally）
- 行末加 "Avg." 列，列出 row-wise 平均
- 极大表会跨整页（CVPR/ICML 接受，NeurIPS 9 页紧）—— 用小字号 + 缩写硬塞
- caption 极长（150-200 词），把"如何读这张表 + 主要发现"全写出来

### Caption 范例（推断 + 多篇互证）

```
Table 2. Multivariate forecasting results on N datasets and 4 forecast horizons
(96, 192, 336, 720). Lower MSE/MAE is better. Best in **bold**, second-best
underlined. We average across all horizons and datasets. Our 〈Method〉 achieves
state-of-the-art on 28/32 settings, with 38% relative MSE reduction over the
strongest baseline (〈Method X〉) and consistent gains on every dataset.
```

caption 把"看哪一行"、"我们赢了多少"、"对手是谁"全部讲清楚——reviewer 不必读正文也能看懂。

## 5.3 任务通用化（TimesNet 范本）

新时序系列爱在主表外加一个 **multi-task table**，把 forecasting / imputation / classification / anomaly detection 都跑一遍：

```
                    Forecasting         Imputation          Classification     Anomaly
                    MSE↓    MAE↓        MSE↓    MAE↓        Acc↑              F1↑
Method 1            ...     ...         ...                  ...
Method 2            ...
〈Ours〉              **bold** ...         **bold** ...         **bold**           **bold**
```

这种 "1 model, 5 tasks" 的对比是 THUML 招牌。

### Seer 类比

Seer 是单任务（KV 管理）但可以跨 dimensional axes：

```
                Hit Ratio↑      p99 TTFT↓      Throughput↑     HBM Util↑
Llama-7B                 ...            ...            ...             ...
Llama-13B
Llama-70B
Mistral-7B
〈Ours〉
```

或者跨 **workload type**（dialog / RAG / code / summarize），让 evaluation 看起来"task-general"。

## 5.4 Ablation Study

THUML 的 ablation 通常以下面三种形式呈现：

### 形式 A：Component on/off

```
                    Hit Ratio↑   p99 TTFT↓
LRU baseline           60.1        220ms
+ SeerNet              68.4        180ms
+ Pareto Eviction      71.2        170ms
+ Look-ahead Prefetch  73.5        155ms (= 〈Seer〉)
```

累加贡献。

### 形式 B：Replace with simpler version

```
SeerNet variant       MSE
Full (3 features)     0.082
- query embedding     0.097  (+18%)
- recency feature     0.105  (+28%)
- positional feature  0.094  (+15%)
```

把 "若无某 feature 会变差多少" 量化。

### 形式 C：Sensitivity to hyperparameters

```
λ_1            Hit Ratio
0.0            68.4
0.1            71.6
0.5 (default)  73.5
1.0            72.8
2.0            70.1
```

证明默认设置 robust。

## 5.5 Showcase / Visualization

每篇都至少有一张"showcase"图，让 reviewer 看完印象深刻：

- **Autoformer**: forecast curve（true vs predicted vs baseline，3-4 datasets）
- **TimesNet**: 1D → 2D folding 的实际样本可视化
- **Non-stationary**: attention map heatmap 对比
- **iTransformer**: attention map（variate token 学到的相关性）
- **CDAN / MDD**: t-SNE on source vs target features
- **VideoMAE-style** (非 THUML): masked patch reconstruction

### Seer 的 showcase 候选

- **eviction trace**：x 轴是 generation step，y 轴是 KV block id，颜色表示 retention status。LRU vs Seer 对比。
- **reuse-distance prediction scatter**：predicted 反 actual reuse distance。
- **HBM occupancy curve**：在一个长 trace 上 LRU vs Seer 的占用率走势。

## 5.6 Efficiency / Computational Cost

NeurIPS / ICML 现在很重视 efficiency：

```
Method            Params(M)   FLOPs(G)   Train(min)   Infer(ms/req)
Method 1            45         12         60           8.2
Method 2            ...
〈Ours〉              ...
```

特别是 Seer 这种 "we add a learned model on top" 的论文，**审稿人一定会问"你的 SeerNet 本身要花多少"**，所以这一节绝对不能省。

## 大表 vs 多个小表的取舍

NeurIPS 9 页正文紧——如果**一张主表能塞下所有 baselines × 所有 datasets** 就用大表（更显气派）。如果塞不下：

- 用 *分行* 的方式：多变量预测一张表，单变量一张表。
- 用 *合并 metric*：4 horizons 的平均放主文，详细每个 horizon 放 appendix。
- 用 *缩写 + 小字号*：6-7pt 字号在 NeurIPS 接受，但要保证可读。

## 关于 random seeds 和 error bars

NeurIPS 2024 之后明显强制 error bars。THUML 论文：

- 至少 3 个 random seeds
- 主表里给 mean ± std（"0.098 ± 0.003" 形式）
- caption 里说明 "averaged over N runs with seeds {1,2,3}"

Seer 的实验记得**每个数字都做 3 个 seed**，数字写 mean ± std。

## §5 写作微习惯

- 每个子节开头一句话点明"这一节回答什么问题"。例 "**Q1: How does Seer compare against existing eviction policies on standard inference workloads?**"
- 每张表 / 图后**正文只写 1-2 段 take-away**，不复述数字。
- 用 "We see that ..." / "This confirms ..." / "We hypothesize that ..." 作为段落开头。
- 不在 §5 自夸："our method achieves SOTA" 这种话写一次就够，剩下让数字说话。

## Seer §5 草稿骨架（NeurIPS 9 页版）

```
5.1 Setup (0.5 页)
    数据集 (Table 1): ShareGPT / Mooncake / LongBench / synthetic Zipf
    Models: Llama-7B/13B/70B, Mistral-7B
    Baselines (12 个):
      - LRU, FIFO, Belady-oracle (lower bound)
      - AttentionStore, BlockSwap, Quest, vLLM-default
      - 3-4 个 prior learned policies (LCache, LRB-style)
    Metrics: hit ratio, p99 TTFT, throughput, HBM util, prefetch overhead
    Hardware: 4× A100 / H100, vLLM v0.X
    
5.2 Main Results (0.7 页)
    Table 2: Hit Ratio + p99 TTFT 主表（traces × baselines）
    1-2 段 take-aways
    
5.3 Generalization (0.5 页)
    Table 3: cross-model generalization (train on 7B, test on 13B/70B)
    Table 4: cross-workload generalization (train on dialog, test on RAG/code)
    
5.4 Ablation Study (0.5 页)
    Table 5: component on/off
    Table 6: feature ablation
    
5.5 Showcase (0.4 页)
    Fig 4: eviction trace heatmap (LRU vs Seer)
    Fig 5: reuse-distance prediction scatter
    
5.6 Efficiency (0.3 页)
    Table 7: parameter / FLOPs / latency overhead
    1 段 take-away
```

整个 §5 大约 2.9 页，留 0.5 页给 §6 conclusion + references buffer。
