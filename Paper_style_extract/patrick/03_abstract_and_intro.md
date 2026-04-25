# 03 · 摘要 & 引言的固定模板

## 摘要 (Abstract) 五段式公式

样本：Repair Pipelining (ATC 17), ParaRC (FAST 23), CAU (TPDS 20), RepairBoost (ATC 21), LESS (FAST 26), OpenEC (FAST 19)。

```
[1] 领域陈述 (1-2 句)：这个子领域是什么，已知的一个 *优良性质或目标*。
[2] 现有方法的限制 (1-2 句)：但是 (However / Yet / Nevertheless) 现有方法在某个维度上不够好。
[3] 我们提议什么 (1-2 句)：In this paper, we propose 〈NAME〉, a 〈类别〉 that 〈核心机制 / insight〉。
[4] 实现与集成 (1 句)：We implement 〈NAME〉 (in C++/Java) and integrate it into 〈HDFS / QFS / Memcached / …〉。
[5] 实验与数字 (1 句)：Experiments on Amazon EC2 / a local cluster / using realistic traces show that 〈NAME〉 reduces 〈metric〉 by 〈X%〉 / improves 〈metric〉 by 〈factor〉 over 〈baselines〉。
```

### 范例对照（来源：USENIX / IEEE 公开摘要）

**Repair Pipelining (ATC 17)** —— 段落映射：

> [1] Erasure-coded storage systems incur high repair penalties.
> [2] Existing repair approaches still have substantial room for improvement.
> [3] We propose *repair pipelining*, a technique that speeds up the repair performance in general erasure-coded storage. By pipelining the repair of failed data in small-size units across storage nodes, repair pipelining reduces the single-block repair time to approximately the same as the normal read time for a single block in homogeneous environments.
> [4] We implement a repair pipelining prototype called *ECPipe* and integrate it as a middleware system into two open-source distributed storage systems, HDFS and QFS.
> [5] Experiments on a local testbed and Amazon EC2 show that repair pipelining significantly improves the performance of both degraded reads and full-node recovery over existing repair techniques.

**ParaRC (FAST 23)** —— 段落映射：

> [1] Minimum-storage regenerating (MSR) codes are provably optimal erasure codes that minimize the repair bandwidth with minimum storage redundancy in distributed storage systems.
> [2] However, the practical repair performance of MSR codes still has significant room to improve, as the mathematical structure of MSR codes makes their repair operations difficult to parallelize.
> [3] We propose *ParaRC*, a parallel repair framework for MSR codes that exploits the sub-packetization nature of MSR codes to parallelize the repair of sub-blocks and balance the repair load.
> [4–5] (Implementation + experiment 句紧随其后)

**CAU (TPDS 20)** —— 同公式：领域 → "but cross-rack bandwidth is constrained" → "We propose CAU, …" → "trace-driven analysis on Yahoo / Microsoft traces shows 25.6%-74.5% reduction in cross-rack update traffic."

> 套用建议：先把这个公式当模板套，**别先写**漂亮句子；填空填完再润色。Seer 的 abstract 完全可以直接套：
> 
> "[1] LLM serving systems use KV-cache to … However, [2] existing eviction/prefetch policies (LRU, BlockSwap, …) under skewed conversational workloads waste up to X% of HBM bandwidth. [3] We propose **Seer**, a learned KV-cache management policy that …  [4] We implement Seer in vLLM (~Y k LoC) and …  [5] On Mooncake / ShareGPT / LongBench traces, Seer reduces TTFT tail latency by Z% and improves throughput by W× over LRU / SGLang baseline."

## Introduction 七段式（推断 + 多篇互证）

引言基本上严格按以下 7 步推进，每步通常一段：

```
§1.1 (隐式) 领域 / 应用价值
       → 一段：什么场景，部署了什么技术，为什么大家关心。
       → 通常带一两个工业部署的 reference (Microsoft, Meta, Tencent, Alibaba…)。

§1.2 (隐式) 现有技术家谱
       → 一段：把 prior approaches 分成 2-3 类，给每类一个名字 + 代表论文。

§1.3 (隐式) 现有技术的限制
       → 一段：列出 2-3 个具体的限制，*带可量化的代价*。
       → 经常用 "However, …" 起头。

§1.4 (隐式) Motivating Observation
       → 一段（有时配一张小图 Fig 1）：用一个测得到的现象 / 一个简单的例子说明问题。
       → 这是组里"一招毙命"的开场技巧。

§1.5 (隐式) 我们的提议
       → 一段：In this paper, we present 〈NAME〉, a 〈type〉 that 〈how〉.
       → 紧接着用 1-2 句话总结 design insight：核心 trick 是什么、依赖什么观察。

§1.6 (显式或半显式) Contributions
       → 一段或一个 bullet list：
         - 我们提出了 〈design primitive 1〉；
         - 我们设计了 〈primitive 2〉；
         - 我们实现了 〈prototype name〉，集成到 〈real system〉；
         - 我们在 〈testbed + EC2 + traces〉 上做了广泛实验，结果 〈X%〉。

§1.7 (可选) Paper organization
       → 一段："The rest of the paper is organized as follows. Section 2 …"
       → IEEE 期刊几乎每篇都有；USENIX 论文有时省略。
```

### Contributions 段的写法（多篇观察）

虽然不是每篇都用 bullet，**但只要有 bullet，就基本是动词开头**：

- "*We propose* 〈名字〉 …"
- "*We design* 〈算法 / 抽象〉 …"
- "*We implement* a prototype called 〈名字〉 …"
- "*We conduct* experiments on …"
- "*We show that* 〈结论〉 …"

关键：**每条 contribution 都是可被审稿人独立检查的可证伪命题**，不写成方法论 (methodology) 而写成事件 (action + evidence)。

## 一些组里反复出现的"金句句式"

下面这些短语出现在多篇论文摘要 / 引言中（中等置信度，落笔时翻 PDF 二次确认）：

- "Erasure coding has been **widely deployed** in modern data centers for fault tolerance with low storage redundancy."
- "**However**, 〈现有方案〉 still have(s) significant room for improvement in terms of 〈维度〉."
- "We **propose** 〈NAME〉, a 〈type〉 that 〈mechanism〉."
- "We integrate 〈NAME〉 as a **middleware** system into 〈系统〉."
- "Experiments on a local cluster and Amazon EC2 show that 〈NAME〉 〈数量化结论〉."
- "To our knowledge, this is the first work that …"
- "The rest of the paper is organized as follows."

## Seer 落地

为方便 Seer §1 直接对照，把上面的 7 步改写一下：

| 步骤 | Seer 版 |
|---|---|
| §1.1 | LLM serving 的 KV-cache 重要性，举 vLLM/SGLang/Mooncake 的部署事实 |
| §1.2 | 现有 KV 管理三类：LRU/eviction、BlockSwap/offload、prefix cache & radix tree |
| §1.3 | 这三类在 conversational workload 下的具体浪费（命中率、HBM 占用、tail TTFT） |
| §1.4 | 用一张图：在某个真实 trace 上，最优"先知" oracle 比 LRU 多 X% 命中、节省 Y% 流量 |
| §1.5 | We propose **Seer**, a learned KV-cache management policy that … |
| §1.6 | 4-5 条 contributions：predictability finding / Pareto framework / SeerNet 模型 / vLLM 集成 / 实验数字 |
| §1.7 | Paper organization（NeurIPS 风格不强求，但加了不会扣分） |
