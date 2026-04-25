# Patrick P. C. Lee 大组论文风格提取（CUHK ADSLab + HUST 协作）

本文件夹用来梳理 Patrick P. C. Lee（CUHK 应用分布式系统实验室 ADSLab）及其协作群体（HUST 胡燏翀、李晓露 等）在系统/存储论文上的写作风格，便于我们在 Seer (NeurIPS 2026) 中借鉴他们打磨多年的"系统论文写法"。注意：Seer 是机器学习偏向的会议论文，而 Patrick 大组主要发 USENIX FAST/ATC、INFOCOM、TPDS、TON、SC、TOS，所以这里抽出的是"可迁移的系统论文工程感"，不是直接套版式。

## 涉及到的人和代表论文

- **Patrick P. C. Lee** — CUHK CSE，AdsLab PI。以 erasure coding、分布式存储、可靠性闻名。 [主页](https://www.cse.cuhk.edu.hk/~pclee/www/index.html), [按主题分类的发表](https://www.cse.cuhk.edu.hk/~pclee/www/pubs_topics.html)。
- **Yuchong Hu (胡燏翀)** — HUST CSE 教授，Patrick 的早期博士生，长期合作者。 [主页](http://yuchonghu.com/)。
- **Xiaolu Li (李晓露)** — HUST CSE 副教授，Patrick 的博士毕业生，OpenEC / ParaRC 一作。 [主页](https://ukulililixl.github.io/)。
- **Kaicheng Tang** — CUHK ADSLab 学生，INFOCOM 23 ET 一作。

代表性论文（本风格提取参考的样本）：
- ECPipe / Repair Pipelining — USENIX ATC 2017 ([PDF](https://www.cse.cuhk.edu.hk/~pclee/www/pubs/atc17.pdf))
- OpenEC — USENIX FAST 2019
- CAU (Cross-Rack-Aware Updates) — IEEE TPDS 2020 ([PDF](https://www.cse.cuhk.edu.hk/~pclee/www/pubs/tpds20cau.pdf))
- RepairBoost — USENIX ATC 2021
- Combined Locality / Wide-Stripe — USENIX FAST 2021
- ParaRC — USENIX FAST 2023
- ET (Elastic Transformation) — IEEE INFOCOM 2023 ([PDF](https://www.cse.cuhk.edu.hk/~pclee/www/pubs/infocom23et.pdf))
- ERS — IEEE/ACM TON 2023 ([PDF](https://www.cse.cuhk.edu.hk/~pclee/www/pubs/ton23ers.pdf))
- TOS Survey of Erasure Coding — ACM TOS 2025 ([PDF](https://www.cse.cuhk.edu.hk/~pclee/www/pubs/tos24ecsurvey.pdf))
- LESS — USENIX FAST 2026 ([PDF](https://www.cse.cuhk.edu.hk/~pclee/www/pubs/fast26less.pdf))

## 怎么读这个文件夹

按顺序读，一篇 5 分钟即可：

1. [`01_overall_style.md`](01_overall_style.md) — 一页纸抓住"这是什么风格"
2. [`02_paper_structure.md`](02_paper_structure.md) — 章节布局（按 USENIX vs IEEE 拆开）
3. [`03_abstract_and_intro.md`](03_abstract_and_intro.md) — 摘要和引言的固定模板
4. [`04_naming_and_terminology.md`](04_naming_and_terminology.md) — 系统命名 / 术语 / 缩写
5. [`05_evaluation.md`](05_evaluation.md) — 实验方法学（这是他们最强的部分之一）
6. [`06_writing_microconventions.md`](06_writing_microconventions.md) — 语态、引用、图表小习惯
7. [`07_paper_dissections.md`](07_paper_dissections.md) — 几篇论文的逐段拆解
8. [`08_apply_to_seer.md`](08_apply_to_seer.md) — 这些写法怎么搬到 Seer 上

## 一句话总结

Patrick 大组的论文 = **明确的痛点 → 命名鲜明的系统 (有缩写) → 三 / 四 个设计要素 (primitives) 并列 → 嫁接到真实存储系统的 prototype (HDFS/QFS/Memcached) → Amazon EC2 + 本地 testbed + 真实 trace 的多基线对比 → 量化结论 (X% 提升 / Y× 加速)**。每篇都长得很像，但每篇都让人立刻知道这是 ADSLab。

## 信息可信度声明

本提取的素材来源是 WebSearch 返回的论文摘要 / 介绍片段（USENIX、IEEE、ACM 公开页面）以及对该组多年公开 PDF 的整体观察。**没有**直接访问 PDF 全文（egress 受限）。因此：

- 章节级、摘要级、命名级的观察是高置信度的（多篇互证）。
- 段落级别的具体措辞（如"In this paper we make the following contributions"）属于推断，标 *(inferred)* 处需要在落笔时再翻 PDF 二次确认。
