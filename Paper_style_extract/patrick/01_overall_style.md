# 01 · 总体风格画像

## 一句话定义

**"工程驱动的系统论文 (engineering-driven systems paper)"**：先讲清楚一个真实部署里测得到的瓶颈，再用一两个干净的 design primitive 端到端解决，并落地成一个有名字的 prototype，最后用多基线、多负载、量化的实验把每个声明都钉死。

## 核心 DNA（高置信度，多篇互证）

### 1. 永远从 *实际部署的痛点* 出发

不写 "we study X for X is interesting"，开篇一定是数据中心 / 存储系统里测得出来的瓶颈。例子：

- **CAU (TPDS 20)**：开篇直接给数据 — *Yahoo 数据中心负载里 update 比例接近 50%，且持续增长；Microsoft 的 erasure-coded 数据中心 delete 也很常见* —— 论证小写更新和 cross-rack 流量是真问题。
- **LESS (FAST 26)**：从 LRC（Azure 实际用的）vs. 其他 erasure code 的代价对立切入，主张要在小 sub-packetization 下做 I/O-高效修复。
- **Repair Pipelining (ATC 17)**：直接对标 "在 erasure-coded storage 里修一个失败块要花的时间"，把它和"读一块的时间"对齐成目标。

### 2. 每篇论文都有一个 *叫得出名字* 的产物

几乎找不到一篇没有命名系统/算法的：**ECPipe, OpenEC, ECDAG, ParaRC, CAU, RepairBoost, ET, ERS, LESS, MOTrack, ECWide**…

命名规律见 [`04_naming_and_terminology.md`](04_naming_and_terminology.md)。命名是**摘要里出现频率最高的词**，从 §1 一直贯穿到 conclusion。

### 3. "三个 / 四个 design primitives" 这个套路被用滥了（褒义）

设计章节经常用并列的 primitive 拆解，每个 primitive 半页到一页。例子：

- *"CAU builds on three design elements. First, … Second, … Furthermore, …"* — selective parity update / data grouping / interim replication.
- *"RepairBoost builds on three design primitives: (i) repair abstraction, … (ii) repair traffic balancing, … (iii) transmission scheduling."*

效果：reviewer 立刻能数出贡献是几个、互相是不是正交、消融实验该怎么布。

### 4. **永远**有真原型，**永远**嫁接到现成系统

不是只在 Python notebook 上跑算法。每篇都报告：
- 用什么语言写的（C++ 居多）；
- 多少行代码（"around 6,000 LoC"）；
- 嫁接到 HDFS / QFS / HDFS-RAID / Memcached / Ceph 的哪个版本；
- 是 middleware / drop-in / 改 NameNode 还是改 DataNode。

### 5. 实验四件套：**Amazon EC2 + 本地 testbed + 真实 trace + 多基线**

详见 [`05_evaluation.md`](05_evaluation.md)。这套组合拳已经成为他们的"招牌"。

### 6. 把 *trade-off* 摆到台面上

"我们牺牲了什么、换来了什么"是 Patrick 组论文的标志性 framing：

- storage redundancy vs. repair bandwidth
- repair bandwidth vs. sub-packetization (ParaRC, ET 整篇都在 balance 这两个)
- locality vs. bandwidth (Wide-stripe, LESS)
- update locality vs. cross-rack traffic (CAU)

读者看完不会有"为什么这玩意儿之前没人做"的疑问，因为论文已经把 design space 显式画出来了。

## 写作语气（中等置信度，需采样确认）

- **谦逊但定量**：避免 "novel"、"groundbreaking" 这种词；偏好 "We propose …, which …"、"Our analysis shows that …"。
- **事实优先**：很少出现 "intuition is" 这种软语言；要么给一个 motivating example，要么给一个 lemma / observation。
- **节制使用第一人称复数**："We propose / We design / We implement / We evaluate" — 几乎是固定四件。
- **缩写引入即定义，之后从不展开**：摘要里写一次 "minimum-storage regenerating (MSR) codes"，后文全篇 MSR codes。

## 共同的"不做"列表

- 不写过度的 selling 话术（论文不是 PR）。
- 不滥用 metaphor / analogy（"like a highway" 这种基本看不到）。
- 不在 §1 漫长地铺陈 ML / 优化背景。技术 background 集中放 §2。
- 不让 evaluation 跑 *单一* 数据集 / 单一基线。

## 共同的"必须做"清单（在引言前完成）

读 Patrick 组论文时，几乎每一篇 §1 结束之前都已经回答了下面这些问题：

1. 这是什么子领域、有什么部署事实？（带数字）
2. 现有方法的限制是什么？（按维度列）
3. 我们提议的系统叫什么？干什么？基于什么核心 insight？
4. 我们做了几个 design primitive？分别是什么？
5. 我们在哪个真实系统上落地？多少 LoC？
6. 实验跑在什么上？相比谁、提升了多少？
7. （可选）本文结构 ("The rest of the paper is organized as follows…")。

把 (1)-(7) 整理成清单，可以直接当 Seer §1 的 outline。
