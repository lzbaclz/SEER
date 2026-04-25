# 07 · 几篇代表论文的逐段拆解

下面对四篇代表性论文按 "abstract → §1 → design → eval" 拆解，便于落笔时直接对照。

> **注**：以下"摘要原句"来自 USENIX / IEEE 公开页面与 ACM 数字图书馆条目；其它段落属于结构推断 + 多篇互证。引用时请翻 PDF 二次校对。

---

## 7.1 ECPipe / Repair Pipelining (USENIX ATC 2017)

作者：Runhui Li, Xiaolu Li, Patrick P. C. Lee, Qun Huang (CUHK)。
意义：组里被引最多的几篇之一，**奠定 prototype + middleware 风格的范式**。

### 摘要骨架

> [1] Erasure-coded storage systems incur high repair penalties.
> [2] Existing repair approaches still have substantial room for improvement.
> [3] We propose *repair pipelining*, a technique that speeds up the repair performance in general erasure-coded storage. By pipelining the repair of failed data in small-size units across storage nodes, repair pipelining reduces the single-block repair time to approximately the same as the normal read time for a single block in homogeneous environments.
> [4] We implement a repair pipelining prototype called ECPipe and integrate it as a middleware system into two open-source distributed storage systems, HDFS and QFS.
> [5] Experiments on a local testbed and Amazon EC2 show that repair pipelining significantly improves the performance of both degraded reads and full-node recovery over existing repair techniques.

### 引言走向

1. 现状：erasure code 在工业部署广泛（点名 Microsoft / Facebook / Google）。
2. 问题：修一个 block 要 k 个 helper 节点同时往 requestor 推数据，瓶颈在最慢的链路。
3. 痛点量化：一个 block 的修复时间 ≫ 一个 block 的正常读时间。
4. Motivation observation：**slot-based pipelining** 可以把 k×L 单链路传输变成 ~L 的总时间。
5. 提议：pipelining + 给出 ECPipe 名字。
6. Contributions：算法 / 异构扩展 / 多块修复扩展 / 集成 HDFS+QFS / 实验。

### Design

- §3 Repair Pipelining for Single-Block Repair（基础算法）
- §4 Heterogeneous Bandwidth（异构带宽下的扩展）
- §5 Multi-Block Repair（多块修复）
- §6 Implementation：ECPipe middleware，~6000 LoC C++，HDFS-RAID 接口、QFS coordinator 接口。

### Evaluation

- 本地 testbed 14 nodes + Amazon EC2.
- baselines: vanilla HDFS-RAID, conventional repair (CR), partial parallel repair (PPR).
- 主要指标：single-block repair time, full-node recovery time，breakdown by stage。
- 结论：repair pipelining brings repair time within ~5% of normal-read time。

### 给 Seer 的复用要点

- "把一个昂贵操作的时间打到与正常操作同档"是非常吸引人的 framing。Seer 可以用："让 KV miss / eviction 的代价回到与命中差距 X% 之内"。

---

## 7.2 OpenEC (USENIX FAST 2019)

作者：Xiaolu Li, Runhui Li, Patrick P. C. Lee, Yuchong Hu。
意义：**开源框架文章 + 抽象引入**。Seer 可以参考"框架/抽象"章节如何写。

### 核心抽象

ECDAG —— 用 *directed acyclic graph* 把任意 erasure-coding 操作的数据流形式化。

> "OpenEC … builds on an abstraction called ECDAG (a directed acyclic graph that defines the workflows of erasure coding operations) and shows how a general erasure coding solution can be feasibly realized through the ECDAG abstraction."

### Design 章节固定走法

1. 给抽象的形式定义 (DAG, node = data unit, edge = computation/transfer)。
2. 列出 abstraction 上能表达的几类典型操作。
3. 提出一组 *DAG 优化变换* (例如 placement-aware, locality-aware reordering)。
4. 给一个 mini example：用一张图把变换前后的 DAG 画出来。

### 给 Seer 的复用要点

- 如果 Seer 要把"learned policy"上升为一个**框架**而不仅仅一个 model，可以照抄这种"先给一个抽象 → 再给优化变换 → 再实例化几个 policy"的写法。
- 抽象命名：Patrick 组爱给抽象起单独的名字（ECDAG），便于 reviewer 引用。Seer 的对应抽象可以叫"KV-attribution graph"或"prefetch-eviction trajectory"之类。

---

## 7.3 ParaRC (USENIX FAST 2023)

作者：Xiaolu Li (HUST), Keyun Cheng, Kaicheng Tang, Patrick P. C. Lee (CUHK), Yuchong Hu, Dan Feng (HUST), Jie Li, Ting-Yi Wu (Huawei)。
意义：典型的**多机构合作 + 重新审视已知 trade-off** 范本。

### 摘要骨架

> [1] Minimum-storage regenerating (MSR) codes are provably optimal erasure codes that minimize the repair bandwidth with minimum storage redundancy in distributed storage systems.
> [2] However, the practical repair performance of MSR codes still has significant room to improve, as the mathematical structure of MSR codes makes their repair operations difficult to parallelize.
> [3] We propose ParaRC, a parallel repair framework for MSR codes that exploits the sub-packetization nature of MSR codes to parallelize the repair of sub-blocks and balance the repair load.
> [4–5] (Implementation + experiment 句紧随)

### 引言核心 framing

"Existing MSR codes 让 sub-packetization 看起来是负担；我们把它当资源用 ⇒ 并行修复机会。"

这是 Patrick 组特别喜欢的招式：**把别人当成本的东西，重新解读成资源**。

### Design

- 把 MSR 修复的子块依赖关系建成 DAG（注意 — 与 OpenEC 的 ECDAG 复用了同一种思维框架；多论文一脉相承）。
- 提出三个 design components：
  1. parallel sub-stripe identification
  2. load-balanced sub-block scheduling
  3. heterogeneous extension

### 给 Seer 的复用要点

- "把别人的成本翻译成你的资源" 用在 Seer 上极易写：例如"以前 KV-cache 大被认为浪费，我们用部分块的可预测 attention 模式作为信号 ⇒ 主动 evict / prefetch"。
- 三个 components 的写法直接套（design primitive 模板）。

---

## 7.4 ET / Elastic Transformation (IEEE INFOCOM 2023) — Kaicheng Tang 一作

作者：Kaicheng Tang, Keyun Cheng, Helen H. W. Chan, Xiaolu Li, Patrick P. C. Lee, Yuchong Hu, Jie Li, Ting-Yi Wu。
意义：用户**显式点名要看的论文**。其结构非常代表 IEEE conference style。

### 文章核心

把 *任意 base code* 通过 elastic transformation 变成新代码：可以减少 repair bandwidth，并允许配置不同 sub-packetization。

### 章节走向（IEEE INFOCOM 9-10 页）

```
I.   INTRODUCTION
II.  BACKGROUND AND RELATED WORK
     A. Erasure Codes 基础
     B. Repair Bandwidth & Sub-packetization Trade-off
III. ELASTIC TRANSFORMATION
     A. Notation
     B. Transformation Construction
     C. Properties (lemma / theorem)
IV.  ANALYSIS
     A. Repair Bandwidth Bounds
     B. Storage Overhead
V.   EXTENSIONS / GENERALIZATION
     A. Multiple Failures
     B. Heterogeneous Settings
VI.  IMPLEMENTATION
     在 OpenEC framework 上集成（点名 OpenEC，复用早期工作）
VII. EVALUATION
     A. Setup
     B. Repair Bandwidth Comparison
     C. Sub-packetization Impact
     D. Runtime Performance
VIII. CONCLUSIONS
```

### Style 亮点

- §III–§IV 把 *构造 + 证明性质* 紧紧耦合，**先给变换的形式，再给 lemma**。
- §V Extensions 是组里"展示完整性"的固定节，用一两个变体覆盖审稿人会问的"那如果情况是 X 呢"。
- §VI Implementation 短而硬：直接落到 OpenEC 之上，不重复造轮子。

### 给 Seer 的复用要点

- IEEE 期刊版的 Seer (e.g. TPDS) 投递时，Extensions 章节是好用的"防审稿"段。即便 NeurIPS 也可以用 appendix 写"扩展到 multi-tenant / multi-model 场景"。

---

## 7.5 横向对比（看完四篇你会注意到的不变量）

| 维度 | ECPipe (ATC 17) | OpenEC (FAST 19) | ParaRC (FAST 23) | ET (INFOCOM 23) |
|---|---|---|---|---|
| 命名 | ECPipe + repair pipelining | OpenEC + ECDAG | ParaRC | ET / Elastic Transformation |
| Design primitives 数 | 3 | 2-3（抽象 + 优化）| 3 | 1 + 扩展 |
| 嫁接系统 | HDFS, QFS | HDFS, HDFS-3, QFS | OpenEC | OpenEC |
| Testbed | local + EC2 | local + EC2 | local + EC2 | local |
| Trace | YCSB / Hadoop | YCSB | Hadoop | (合成 + 真实) |
| 多 baseline | ≥3 | ≥3 | ≥3 | ≥3 |
| Implementation 节 | 单独 | 单独 | 单独 | 单独 (短) |
| Related Work 位置 | §9 (USENIX 末尾) | §9 | §9 | §II 合并 |

把这张表当成**"我每篇 Patrick 组论文都要中的格子"**：Seer 落笔之前可以先填一行 *Seer*，看看哪格还空着。
