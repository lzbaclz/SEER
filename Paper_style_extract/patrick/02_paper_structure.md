# 02 · 论文骨架（按 venue 拆开）

Patrick 组的版面会随 venue 变。下面把同一组研究在不同 venue 下的章节模板分别整理。

## A. USENIX 模板（FAST / ATC）— ECPipe, OpenEC, ParaRC, RepairBoost, LESS, Wide-Stripe

USENIX 是 single-column 12-page，对版面最宽容。Patrick 组在这里的常用骨架：

```
1  Introduction               (1.0–1.3 页)
2  Background and Motivation  (1.0–1.5 页)
   2.1 〈基础知识：erasure code / MSR / sub-packetization 等〉
   2.2 〈现有方法回顾〉
   2.3 Motivating Observation / Limitations of Prior Work
3  〈系统名称〉 Overview         (0.5–1.0 页)
   3.1 Design Goals
   3.2 Architecture / High-Level Idea
4  〈Design Primitive 1〉      (1.0–1.5 页)
5  〈Design Primitive 2〉      (1.0–1.5 页)
6  〈Design Primitive 3〉      (1.0–1.5 页)   ← 通常 3 个，偶尔 2 或 4 个
7  Implementation             (0.5–1.0 页)
8  Evaluation                 (2.5–3.5 页)
   8.1 Experimental Setup
   8.2 Q1: 〈研究问题 1〉
   8.3 Q2: 〈研究问题 2〉
   ...
9  Related Work               (0.5 页)
10 Conclusion                 (0.25 页)
   References                 (剩余空间)
```

特征：
- 章节用阿拉伯数字、"Section X" / "§X" 都会出现。
- "Background and Motivation" 经常 *合并* 而不是拆成两节，**Motivation 子节**是组里非常标志性的写法 —— 在 §2 末尾用一段 case study 或一张图论证"现有做法在我们关心的部署条件下有 X% 的浪费"。
- Implementation 单独成节，**1/2 页起步**，给出 LoC、嫁接点、关键代码改动。
- Related Work 放在 Evaluation **之后**，是 USENIX 的常规惯例，组里跟随。

## B. IEEE Conference 模板（INFOCOM / ICDCS / MSST）— ET, ECWide…

IEEE conf 是 2-column 9-10 页的 IEEEtran。这组论文章节用 *罗马数字大写*：

```
I.    INTRODUCTION
II.   BACKGROUND AND RELATED WORK
III.  〈System / Construction Name〉 OVERVIEW
IV.   DESIGN / ANALYSIS
V.    EXTENSIONS                       ← 异构、多失效、etc.
VI.   IMPLEMENTATION
VII.  EVALUATION
VIII. CONCLUSIONS
```

特征：
- Related Work 常**合并到 §II Background**（INFOCOM 的常规挤压）。
- §V EXTENSIONS 是这组很爱用的一节：把核心算法做"异构带宽 / 多失效 / 不同子打包度"等扩展，体现完备性。
- INFOCOM 23 ET 这篇就明确从 transformation 框架出发，再讨论 base code 的扩展。

## C. IEEE Journal 模板（TPDS / TON / TOS）— CAU, ERS, Repair Pipelining (extended)

期刊版面多 4-5 页，基本是把 conference 版扩成更完整的文档：

```
1  Introduction
2  Background
3  Related Work          ← 期刊版常常单独成节，不再挤进 §2
4  System Model & Problem Formulation
5  Design (一组 primitives，每个 primitive 单独成节也常见)
6  Extensions / Generalization
7  Analysis              ← 期刊版常常多一节正式分析（lower bound / proof / cost model）
8  Implementation
9  Evaluation
10 Conclusion
```

特征：
- §4 通常显式给出 *system model* + *notation table* + *problem statement*。这是期刊扩展时一定加的"形式化包裹层"。
- §7 Analysis 在期刊版里会加上诸如 *cross-rack traffic lower bound*、*expected repair time analytical model* 之类的可推导内容。

## D. ACM Journal 模板（TOS）— Erasure Coding Survey, ECPipe-extended

ACM TOS 是 single-column。文章会更长（25-50 页），但章节风格和 IEEE journal 接近。Survey 论文（TOS 25）有自己的范式（taxonomy + 时间轴），这里不展开。

## 跨 venue 的不变量

无论 venue 怎么变，下面这些**几乎都成立**：

1. **Introduction 一定包含一段"Contributions of this paper"**（不一定是 bullet 形式，更多是把贡献编织进末段，但若是 IEEE 期刊则常用 bullet/numbered）。
2. **Implementation 一定单独成节**（最少半页，绝不放进 Evaluation）。
3. **Evaluation 一定有 setup 子节 + 多个 Q-style 子节**。
4. **Related Work 一定提及 *最近 3-5 年的紧邻工作*** 并显式标差异（"Unlike X, we …"）。
5. **Conclusion 短**：往往就是 4-6 句，回扣 contributions + 提一句 future work。

## Seer 借鉴建议（NeurIPS 风格 vs 上述）

NeurIPS 模板是 9 页正文 + 无限附录，不分 §I 罗马数字。可借鉴：

- 把"Background and Motivation"作为独立 §2，做 **motivating measurement** 子节（这是 Patrick 组最强的写法之一）。
- "Implementation 单独成节"在 NeurIPS 里不常见，但我们 Seer 涉及到 vLLM 集成，可以单设一个 **§ System / Integration**（短，1 页内），让审稿人意识到这不是 paper-only 工作。
- 把 Evaluation 拆成几个 "Q1: …" "Q2: …" 形式的子问题，会让 NeurIPS 审稿人感觉清爽（NeurIPS 这边普遍是 "5.1 Setup / 5.2 Main results / 5.3 Ablations" 套路；用 Q-style 是个差异化）。
