# 05 · 实验方法学（这是 Patrick 组的金牌之一）

## 实验四件套

几乎每篇 USENIX/IEEE 论文都符合下面的四件套：

1. **本地小集群 testbed**（6–14 个节点是常见规模）
2. **Amazon EC2 部署**（云上跑相同实验做泛化）
3. **真实 trace** （Yahoo, Microsoft, Alibaba, Tencent 的 IO trace；近年也用 Hadoop / HDFS workload generator）
4. **多基线对比**（至少 3 个 prior approaches，且是同一个 testbed 上重新实现 / 重新跑的）

每篇至少占 4 件中的 3 件，几乎每篇都集齐。

## Setup 子节的固定要素

USENIX 风格的 §X.1 Experimental Setup 通常按下列 checklist 一段一段写：

```
- Hardware:        节点数、CPU/GPU、内存、网卡 (10 Gb/s 居多)、磁盘 (HDD vs SSD)
- Software:        操作系统、HDFS / QFS 版本、Hadoop 版本、自家 prototype 版本
- Cluster topology:rack 数、跨 rack 带宽（如有）
- Workloads:       生成方式（trace replay / synthetic Zipf / YCSB）、key/value 大小、读写比
- Parameters:      (n, k)、sub-packetization、stripe 大小、block 大小
- Baselines:       Vanilla HDFS, RS code, MSR code A, MSR code B, Prior work X
- Metrics:         repair time, repair bandwidth, normalized I/O, throughput, latency tail (p99)
- Default config:  「除非另行说明，默认 (n, k) = (9, 6) 等」
```

**默认参数显式声明** 这一招很重要：所有图都默认基于这套，单独某图换了某参数会直接在子节首句说"In this experiment we vary k while keeping …"。

## Q-style 子节

Evaluation 通常按"研究问题"组织，而不是按"图序号"组织：

```
§X.2 Single-block repair time (Q1: how does NAME compare for single-block degraded reads?)
§X.3 Full-node recovery   (Q2: how does NAME scale for full-node recovery?)
§X.4 Heterogeneous setting (Q3: does NAME's benefit hold under heterogeneous bandwidth?)
§X.5 Sensitivity to (n, k) (Q4: …)
§X.6 Overhead             (Q5: what is the implementation overhead of NAME?)
```

每个子节固定结构：

1. **目的句**：what we want to know.
2. **方法句**：how we measure it (parameters, baselines, runs).
3. **图 / 表**：1-2 张主图 + 偶尔附 1 张 sensitivity 表。
4. **结论句 (1 sentence)**：一句话总结这张图给的 take-away (常用 "We see that …", "Figure X shows that … by Y%")。
5. **机制解释句 (1-2 sentences)**：为什么是这个结果，关联回 design primitives。

## 数字一定具体

绝不写"significantly improves"而不给数字。范例：

- "ParaRC reduces single-block repair time by **44.6 %** under (14, 10) MSR codes."
- "CAU saves **25.6 %–74.5 %** of cross-rack update traffic across the four traces."
- "ECPipe brings the single-block repair time to within **5 %** of the normal-read time."

百分比、倍数、绝对值至少出现一种。**range 形式**（"25–74%"）很常用 —— 它隐含"我们跑了一族 setting"。

## 图表风格（推断 + 高频观察）

- **柱状图**最多，line chart 用来表达"随 k 变化"的 sensitivity。
- 每张图都标 baseline 名字（不是 "ours"，是 *ECPipe*；这样跨论文也能比）。
- y 轴标签明确 unit，例如 "Repair time (s)"；不会写 "Time"。
- **Normalized 形式很常见**：相对 vanilla baseline 标 1.0，让加速倍数一目了然。
- 极少用堆叠柱状图（容易误导）；偏好分组柱状图。

## Ablation / Component Breakdown

USENIX 论文里几乎都有一节"Effect of each design component"，对应每个 design primitive 的开关：

```
Setting      | Repair time | Improvement
-------------|-------------|------------
Baseline     |  100.0      |   —
+ Primitive 1|   71.4      |  -28.6%
+ Primitive 2|   55.2      |  -22.7%
+ Primitive 3|   42.1      |  -23.7%
```

把"我们三个 primitive 各贡献多少"用一个简单的累加表交代清楚 —— 是审稿人最爱的格式。

## Overhead / Cost 子节

最后一个子节通常专门量化"我们的方法本身有什么代价"：

- 训练 / 预处理时间
- 内存占用
- 计算开销
- 在线决策延迟（µs 级）

诚实给出"我们的开销是多少"是这个组让 reviewer 信任的关键。

## Seer 落地建议

把 Patrick 组的 evaluation 模板搬到 Seer：

```
§5.1 Setup
     Hardware: 1× / 4× / 8× A100 / H100 节点；vLLM/SGLang 版本；CUDA 版本。
     Workloads: ShareGPT, Mooncake, LongBench, conversational replay；prompt/response 长度分布。
     Parameters: HBM budget, KV block size, prediction window.
     Baselines: LRU, FIFO, BlockSwap, AttentionStore, no-eviction, oracle (offline-optimal).
     Metrics: hit ratio, mean / p99 TTFT, throughput (tokens/s), HBM utilization, prediction overhead.
     Default: 「除非说明，HBM = 8 GB, block = 16 tokens」.

§5.2 Q1: How accurate is SeerNet's eviction prediction?    （per-block top-k accuracy / AUC）
§5.3 Q2: How does Seer compare on hit ratio across traces? （主图 1）
§5.4 Q3: How does Seer affect TTFT and throughput?         （主图 2，含 p99）
§5.5 Q4: Does Seer generalize across model sizes / context lengths?
§5.6 Q5: Effect of each design component (ablation table)
§5.7 Q6: Overhead — model size, inference latency, memory.
§5.8 Q7: Sensitivity — HBM budget, prefetch aggressiveness, prediction window.
```

把 Q1-Q7 列出来后，每个 Q 对应**一张主图 + 一段 5 句话**，论文实验部分就基本成形。
