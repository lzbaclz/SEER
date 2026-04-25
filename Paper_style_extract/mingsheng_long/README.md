# 龙明盛 (Mingsheng Long) / 清华 THUML 论文风格提取

本目录把 **Mingsheng Long 教授（清华软件学院 / THUML）** 一组的论文风格蒸馏成可直接套用的写作指南，目标是把 Seer (NeurIPS 2026) 的 §1-§6 写得像 NeurIPS / ICML / ICLR 评委心目中"标杆中国组"的样子。

## 为什么是 THUML

> "中国组里写得最像 well-trained NeurIPS 评委心目中标杆"——problem formulation 简洁、motivation figure 一图杀人、ablation 表精致、related work 永远按 *idea axis* 而不是按时间堆。

THUML 的产出横跨两条主线：

1. **迁移学习 / 域适应 (2015-2020)**：DAN (ICML 15) → JAN (ICML 17) → CDAN (NeurIPS 18) → PADA (ECCV 18) → MDD (ICML 19) → Universal DA (CVPR 19)。这条线很多有正式 theorem + Rademacher complexity 推导，是中国 ML 组里少有的"理论与算法都打满"。
2. **时序 / 基础模型 (2021–现在)**：Autoformer (NeurIPS 21) → FEDformer (ICML 22) → Non-stationary Transformers (NeurIPS 22) → TimesNet (ICLR 23) → SimMTM (NeurIPS 23 Spotlight) → iTransformer (ICLR 24 Spotlight) → Timer (ICML 24) → Timer-XL (ICLR 25) → Sundial (ICML 25 Oral)。

两条线写法不完全一样（旧线偏理论 + 域适应 benchmark，新线偏架构创新 + 多任务统一 + foundation model framing），但有一组**贯穿始终的 DNA**——这是我们最想偷的东西。

## 文件清单

按顺序读，每篇 5-10 分钟。每一节都尽量自带"Seer 落地"段。

1. [`01_overall_style.md`](01_overall_style.md) — 一页纸抓住"这是什么风格"
2. [`02_paper_structure.md`](02_paper_structure.md) — 章节布局（NeurIPS/ICML/ICLR 9 页 vs CVPR 8 页 vs ICML/journal 长版）
3. [`03_abstract_and_intro.md`](03_abstract_and_intro.md) — 摘要 / 引言固定模板
4. [`04_naming_and_terminology.md`](04_naming_and_terminology.md) — 命名学（i-prefix、Auto-Correlation 大写、TimesBlock 后缀、hyphenated 行话…）
5. [`05_problem_formulation.md`](05_problem_formulation.md) — "找一个隐藏瓶颈 / 概念反转"是 THUML 的招牌写法
6. [`06_method_section.md`](06_method_section.md) — Method 章怎么排，每个 component 一个子节
7. [`07_theory_and_analysis.md`](07_theory_and_analysis.md) — 理论章怎么写（MDD 是范本，时序系列 lightly grounded 是另一种范本）
8. [`08_evaluation.md`](08_evaluation.md) — 实验方法论：6-10 数据集 / 12-15 baseline / 任务通用化
9. [`09_writing_microconventions.md`](09_writing_microconventions.md) — 语态、公式编号、图表、related work 组织、reproducibility
10. [`10_paper_dissections.md`](10_paper_dissections.md) — 9 篇代表论文的逐段拆解（Autoformer / TimesNet / iTransformer / Non-stationary / SimMTM / Timer / DAN / CDAN / MDD）
11. [`11_apply_to_seer.md`](11_apply_to_seer.md) — Seer 落地清单（self-review 卡）

另有 [`_raw_research/`](_raw_research/) 目录，保存了底稿研究素材（time-series + transfer-learning），便于二次查证。

## 一句话总结

THUML 论文 = **应用驱动开题 (energy / weather / RAG / ...) → 命名出一个反直觉的"隐藏瓶颈"或"概念反转" (over-stationarization / inverted token / temporal 2D-variation / margin disparity discrepancy) → 一图说明的 Fig 1 → 几个 hyphenated capitalised 命名的子模块 (Auto-Correlation, De-stationary Attention, TimesBlock, Multilinear Conditioning) → 6-10 数据集 / 12-15 baseline 大表 → 量化结论 (38%, 49.43%, 14.8%) → Time-Series-Library / Transfer-Learning-Library 里能跑通的 GitHub repo**。

老线 (DA) 多一个 "**bridge theory and algorithm**" 的层级——会有 theorem + Rademacher complexity 推导，对 reviewer 杀伤力极大。

## 信息可信度声明 & 重要勘误

素材来源：(a) 论文 abstract / intro 公开片段（NeurIPS / ICML / ICLR / CVPR / arXiv 公开页面）；(b) 龙老师主页 https://ise.thss.tsinghua.edu.cn/~mlong/ 的论文清单与 PDF；(c) THUML GitHub https://github.com/thuml 各 repo 的 README 与 paper-code 链接；(d) 调研期间的 sub-agent 报告（已交叉核对）。**没有**直接全文抓 PDF（沙箱 egress 受限）。

**高置信度**（多篇互证，可直接落笔）：abstract 五段式、命名规律、motivation figure 套路、related work 组织、Time-Series-Library 复用、ablation 表样式、6-10 数据集 / 12-15 baseline 数量级、quantification 习惯。

**中置信度**（基于已知模式 + 个别 abstract 句子推断）：method 章子节数、theory 章 lemma 数量、引言段数、microconvention（具体措辞）。落笔时翻 PDF 二次确认。

**勘误**：早期 sub-agent 报告里把 **VideoMAE / VideoMAE V2** 放进 THUML——**这是错的**。VideoMAE 是 MCG-NJU + 腾讯（Tong et al. NeurIPS 22），VideoMAE V2 是南大 + 上海 AI Lab（Wang et al. CVPR 23）。本提取里**已剔除**这两篇；THUML 自己的视觉/SSL 线主要是迁移学习领域的 vision benchmarks（Office-31 / Office-Home / VisDA / DomainNet）和近年的 SimMTM。

**关于 FEDformer 归属**：FEDformer 由阿里达摩 + THUML 联合，一作 Tian Zhou 当时在 Alibaba DAMO；Mingsheng Long 是合作者。归属上算"THUML 协作"而非"THUML 一作"，写法非常接近，下面文档里仍按 THUML 风格分析。
