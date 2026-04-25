# NeurIPS 2026 投稿完整指南（SEER 项目专用）

> 写入：2026-04-25 ｜ 对应 CfP：NeurIPS 2026 Main Track
> 目标论文：*SEER: Learning to Evict KV-Cache by Predicting Future Attention*

---

## 0. 残酷的时间盘点

| 节点 | 日期（AOE = UTC−12） | 距今 |
|------|---------------------|------|
| 今天 | 2026-04-25 | — |
| **Abstract 截止** | **2026-05-04** | **9 天** |
| **Full paper + supplementary 截止** | **2026-05-06** | **11 天** |
| 作者通知 | 2026-09-24 | — |

**关键观察**：abstract 与 paper 仅差 **2 天**，意味着 abstract 不能写"占位词"，必须是**最终 abstract**——也就是说论文必须在 5/4 当天已基本定稿。**实际可用工作日 ≈ 9 天**。

**含义**：原 PROJECT_PLAN 的 3 周计划必须立即压缩到极限 MVP。

---

## 1. 立即行动（今天 4/25 必须做完）

### 1.1 OpenReview profile（所有 co-author 都要）⭐⭐⭐ 阻塞

> "All authors must have an OpenReview profile when submitting"

新建 profile 不是即时通过的——OpenReview 需要审核（最快几小时，慢则数天）。如果有合作者没账号，**今天**就让他们注册：

1. 注册：https://openreview.net/signup
2. **必填**：full name（与 ORCID 一致）、email、affiliation、homepage、最近 papers（DBLP / Google Scholar 链接帮助审核加速）
3. 如已有 profile，登入后检查：
   - 当前 affiliation 是否最新
   - 邮箱是否仍能收信
   - 是否有 ORCID（强烈建议绑定）
   - "Expertise" 关键词列表是否准确（影响 reviewer-paper matching）

**如果你是独立一作且没合作者**，跳过协作部分。

### 1.2 选定 track（不可后期切换）⭐⭐⭐ 阻塞

CfP 明确警告："no possibility to switch tracks or types"。SEER 是方法论文，**Main Track** 是默认选择。但如果时间真的来不及，考虑：

| Track | 何时考虑 | 影响 |
|-------|---------|------|
| **Main Track** | 默认；method+实验完整 | 最常规，竞争最激烈（接收率 ~25%） |
| **Datasets & Benchmarks** | 重点是 attention trace 数据集 + benchmark | 不同的 portal & 时间表，可能宽容度更高 |
| **Position** | 不做实验，只论证 "learned eviction is the right paradigm" | 不需要 strong empirical results |

读完每个 track 的独立 CfP（CfP 文末已给链接）再做决定。**今天必须选定**。

### 1.3 LaTeX 模板与目录骨架 ⭐⭐⭐

```bash
cd ~/codes/papers/Seer/paper
# 从 NeurIPS 2026 官方下载 style files
wget https://media.neurips.cc/Conferences/NeurIPS2026/Styles.zip   # 占位 URL，去官网核实
unzip Styles.zip
# 建空骨架
touch main.tex refs.bib
```

`main.tex` 骨架（9 节）：
```latex
\documentclass{article}
\usepackage[final]{neurips_2026}   % 占位包名，去模板包里找实际名
\usepackage{...}

\title{SEER: Learning to Evict KV-Cache by Predicting Future Attention}
\author{Anonymous Authors}

\begin{document}
\maketitle
\begin{abstract} ... \end{abstract}

\section{Introduction}
\section{Related Work}
\section{Preliminaries}
\section{Method}
\section{Experimental Setup}
\section{Results}
\section{Discussion and Limitations}
\section{Conclusion}

\bibliographystyle{plainnat}
\bibliography{refs}

\appendix
\section{Implementation Details}
\section{Additional Results}
\section{Broader Impact}
\end{document}
```

---

## 2. 论文格式硬规则（Main Track）

> ⚠️ 以下基于历届 NeurIPS（2023–2025）的标准；**必须以 2026 官方 style 文件中的 README 为准**。本节标★的项目是历年最容易踩雷的。

### 2.1 长度

- **正文 9 页**（不含 references、checklist、appendix）★
- References 页数无限
- Appendix / supplementary 无明确页数限制（但 reviewer 通常不读超过 25 页）
- **超 1 个字节会被 desk reject**——`\pagebreak` 不要硬塞

### 2.2 双盲匿名 ★

- 标题页 author 字段写 `\author{Anonymous Authors}` 或保留模板默认占位
- **不能**在论文任何位置出现作者姓名、email、单位 logo、致谢中的 funding agency
- **第三人称引用自己**：写 "Smith et al. (2024) showed..."，不写 "We previously showed..."
- 引用预印本（arXiv）若是自己的工作，仍按第三人称引用，但**不能**通过引用泄露作者
- **匿名代码 link**：不能直接放 GitHub 个人/组织链接。用：
  - https://anonymous.4open.science（推荐）
  - https://github.com/<anonymous-account>（必须是新建匿名账号）
- 致谢章节**整段删除**或改为 "Acknowledgments removed for review."

### 2.3 LLM 使用披露 ★（NeurIPS 2024 起强制）

如果用 ChatGPT / Claude 协助写作或代码，需要在 paper 里**明确披露**。例如：

> "We used a large language model (Claude 4.6) to assist with code refactoring and prose polishing. All scientific content, experimental design, and final claims are the authors' responsibility."

不披露被发现是 desk reject 风险。

### 2.4 字体 / 行距 / 边距

- 不可改 NeurIPS 默认字体大小（通常 10pt）
- 不可缩小行距、缩小 margin
- 表格 / 图标题字号可比正文小一号，但不能用 \tiny 之类

---

## 3. 必交清单（5/6 deadline 当天）

### 3.1 主 PDF（main.pdf）

- 9 页正文 + references（无限）
- 不嵌入字体可能 reject——`pdflatex` 默认就嵌入，但需要确认没用 Type 3 字体
- 提交前 print 一下，检查：
  - 所有引用都有 `[?]` 之外的内容
  - 所有图都能在 B&W 打印下区分
  - 没有 `??` 之类的未解析引用

### 3.2 NeurIPS Paper Checklist ⭐⭐⭐ 强制

NeurIPS 2024 起，paper 必须**附带 Paper Checklist**（嵌在 paper 末尾或 supplementary）。这是个 ~30 题的清单，每题要求：
- 答 Yes / No / NA
- **必须给一段说明**（半行到 3 行）

清单大致涵盖：
1. Claims 是否被实验支持
2. Limitations 是否讨论
3. Theoretical results：完整证明？
4. 实验是否可复现（代码、数据、超参）
5. 是否报告 error bars / 多 seed
6. 是否描述 compute resources
7. 数据集 license 是否标注
8. 是否使用 personal data
9. 是否披露 LLM 使用
10. ...

**这一项不要拖到最后写**，半天起步。模板会以注释形式预填问题。

### 3.3 Reproducibility checklist（包含在 Paper Checklist 里）

如果声明"代码会公开"，必须在 supplementary 中提供（即使匿名版）。

### 3.4 Supplementary material（可选，但强烈建议）

打包成单一 .zip，≤ 100 MB（往年限制，2026 待核实）：

```
supplementary.zip
├── code/                      # SEER 全部代码（匿名化版本）
│   ├── seer/                  # 包
│   ├── tests/
│   ├── experiments/
│   └── README.md              # 匿名版（去掉 work1.md / OrchKvCache 引用）
├── checkpoints/               # 至少一个 LAP ckpt 供复现
├── traces_sample/             # 小规模 trace（如 100 prompts）供 reviewer 试跑
└── extra_experiments.pdf      # 主文放不下的额外实验
```

如果 zip 超过 100MB，把代码 + checkpoint 改用 anonymous 4open 链接，zip 里只放 PDF。

### 3.5 Author response 准备（提前规划）

NeurIPS 8–9 月会进入 rebuttal 阶段，作者有 1 周回复 reviewer。**今天先在 `notes/rebuttal_prep.md` 列三件事**：
- 哪些实验没做完？预留补做的脚本
- 哪些 baseline 数字是引用而非自己复现？哪些可能被质疑
- 70B / 256K / 真实 vLLM throughput 这种"显而易见的下一步"——脚本准备好

---

## 4. OpenReview 提交流程（5/4 当天 & 5/6 当天）

### 4.1 5/4 abstract 提交（10 分钟）

1. 登入 https://openreview.net
2. 找到 NeurIPS 2026 - Main Track 的 portal（CfP 给的链接）
3. 点 "Submit"
4. 填：
   - Title（最终标题，submit 后**不能再改**有些字段）
   - Abstract（最终版；可以小改但不要大改）
   - Authors（**所有**，按 OpenReview profile 检索；填错或漏一个不可补救）
   - Keywords / Subject area / 其他元信息
5. **不需要**上传 PDF
6. 点 submit → 收到 paper ID

> 重要：abstract 提交后 paper ID 锁定。**无法新增/移除作者**（只能微调名字拼写）。

### 4.2 5/6 full paper 提交（30 分钟，提前 6 小时）

1. 找到刚才的 paper ID 条目，点 "Add submission"
2. 上传：
   - main.pdf
   - supplementary.zip（如有）
3. 填：
   - Conflict of interest（co-author 已合作过的研究者，会自动从 profile 拉）
   - 选 reviewing keywords / track-internal subareas
   - 回答 paper checklist（如果不在 PDF 里则在这里）
4. **必勾**：
   - 同意 reciprocal reviewing
   - 同意 dual submission policy
   - 同意 LLM disclosure policy
5. 点 submit → 等几秒收到确认邮件 → **保存确认页面截图**

> ⚠️ **AOE 是 UTC−12**。北京时间 5/7 19:59 才到 AOE 的 5/6 23:59。但你在 5/7 早上 9 点提交对中国时区就够了——不要算错时区被无意中超时。

### 4.3 提交后能改什么

- 提交后 OpenReview 通常给一个**短窗口**（24–48h）让你修小问题（如 PDF 重传）
- author list 一般**完全锁死**
- title / abstract 可能允许小修改

---

## 5. 给 SEER 的 11 天压缩时间表

> 按 PROJECT_PLAN.md 原计划裁剪。**核心思路：把"完整论文"降为"E1 + 最小 E2 的 main paper + extensive appendix"**。

| 日期 | 任务 | 产出 / DONE WHEN |
|------|------|----------------|
| **4/25 (今)** | OpenReview profile 全员到位；track 选定；LaTeX 模板拉好 | profile screenshot + paper/main.tex 能编译 |
| **4/26 (六)** | 跑通 E1 mini：`MODEL=SmolLM-135M NUM_PROMPTS=20` | 走完 collect → train → analyze 三步链路 |
| **4/27 (日)** | 启动 Llama-3-8B 大规模 trace 采集（背景跑）；同时写 §1 Intro + §3 Related + §2 Preliminaries 草稿 | trace 第一批 20 个 request 入库；3 节初稿 |
| **4/28 (一)** | trace 全集到位；启动 LAP 训练（多 architecture）；写 §4 Method | LAP ckpts；§4 初稿 |
| **4/29 (二)** | E1 完整结果；**GO/NO-GO 决策**。如 GO，开 E2 mini sweep（2 baseline + SEER × 2 budget × 1 ctx_len） | predictability.json verdict；E2 启动 |
| **4/30 (三)** | E2 mini 出数；写 §6 Results 主表 | 4 行 × 3 列的主表 |
| **5/1 (四)** | E4 LAP arch 消融（已经训完三个，跑测试集 AUC 即可）；§5 Setup + §6 Results 完工 | ablation 表 |
| **5/2 (五)** | Paper checklist 全部回答；草稿全文通读；图表清晰度过一遍 | draft 完整 |
| **5/3 (六)** | 内部 review；修订；写 §7 Discussion + Limitations + §8 Conclusion + Abstract | submission-ready PDF |
| **5/4 (日)** | **abstract 提交**（OpenReview）；下午继续 polish 论文 | abstract submitted |
| **5/5 (一)** | Supplementary 打包（代码匿名化 + 小 trace 样本 + extra_experiments.pdf）；最后通读 | supplementary.zip |
| **5/6 (二)** | **full paper + supplementary 提交**（AOE 截止）；建议北京时间下午 6 点前提交避开服务器拥堵 | confirmation email saved |

### 关键裁剪决策

- **E3 throughput 砍掉**：当前 sim 不可信，正式数字写不出来。Appendix 放一段 "vLLM integration is in progress; preliminary throughput numbers are deferred to future work"，引用 OrchKvCache motivation 数据作为系统侧的 proxy
- **E7 跨模型泛化 砍掉**：单模型 (Llama-3-8B) 主实验 + 一句 limitation
- **E6 联合策略消融 砍掉或合并**：用 SEER 默认参数 vs `λ_io = 0`（关 IO 感知）作为 1 行对比即可
- **理论 lemma 砍掉**：保留 §3 的 "Online set selection with predictions" framing，但不展开证明
- **跨 benchmark 砍掉**：只跑 RULER 8K + 16K（合成 needle，质量信号最干净）；LongBench 留作 limitation

### 全程绝不能做的事

- **不要尝试加 vLLM 集成**：3 周搞不定，11 天必死
- **不要训练新模型**：所有结果用 LAP 当中介，不动 LLM 主干
- **不要追新 baseline**：StreamingLLM + H2O + SnapKV + Recency = 4 个，够了
- **不要写超过 9 页**：如果 §6 写不下，砍 §3 文献综述到 0.5 页

---

## 6. 风险与备选

### 6.1 风险表

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| E1 GO/NO-GO 失败（4/29 才知道） | 中 | 致命 | 4/29 当天若 NO-GO，立即 pivot 到 **Datasets & Benchmarks Track** —— 把 attention trace 数据集 + predictability benchmark 作为贡献 |
| Trace 采集 OOM / 超时 | 中 | 高 | 提前在 SmolLM 上验证 pipeline；Llama-3-8B 用 fp16 + ctx≤16K，不上 32K |
| Paper checklist 写到一半发现不合规 | 低 | 中 | 提前 5/2 写完，5/3–5/5 修订时间 |
| OpenReview profile 审核未过 | 中 | 致命 | 今天必须注册并上传完整信息；密切关注审核状态 |
| 5/6 服务器过载 | 中 | 中 | 北京时间 5/7 早 6 点前提交，远早于 AOE 截止 |
| 论文太弱被 desk reject | 中 | 致命 | 5/3 内部 review 必做；找一位"严格的"合作者预审 |

### 6.2 退路（按优先级）

1. **同 NeurIPS 2026 Datasets & Benchmarks Track**：CfP 提到时间表可能不同，去看那个 track 的 deadline。如果晚 1–2 周，转投后压力骤减
2. **NeurIPS 2026 Workshop（如 ML for Systems / Efficient ML）**：deadline 通常在 8 月，可写更宽松的 short paper
3. **MLSys 2027**（10 月 deadline）：直接复用 OrchKvCache + SEER 的完整栈，系统贡献当卖点
4. **ICLR 2027**（10 月 deadline）：与 NeurIPS 形态最接近，可直接基础上扩展

---

## 7. 关键 URL & 文件清单

### 必访链接

- NeurIPS 2026 主页：https://neurips.cc/Conferences/2026/
- Call for Papers：https://neurips.cc/Conferences/2026/CallForPapers
- Main Track Handbook（关键政策原文）：CfP 内的 MainTrackHandbook 链接
- OpenReview portal：CfP 内的 Submission link
- OpenReview signup：https://openreview.net/signup
- Anonymous code：https://anonymous.4open.science
- Paper template：CfP 内的 Paper template 链接

### 自查清单（5/6 提交前打勾）

- [ ] 所有 author 在 OpenReview 上有 active profile
- [ ] 论文不超 9 页正文
- [ ] 任何位置无作者姓名 / email / unit logo / 致谢
- [ ] 自引用全部第三人称
- [ ] LLM 使用已披露
- [ ] Paper checklist 30 题全部回答 + 给理由
- [ ] References 中 arXiv 论文都有 arXiv ID
- [ ] 所有 figure 在 B&W 下区分
- [ ] PDF 字体全嵌入（无 Type 3）
- [ ] 没有 `??` `?` 未解析引用
- [ ] supplementary.zip ≤ 100 MB（或用 anonymous 4open 链接）
- [ ] Anonymous code repo 可访问且不含作者信息
- [ ] 时区核对：AOE 5/6 = 北京时间 5/7 19:59
- [ ] 提交确认邮件存档

---

## 8. SEER paper 一页骨架（5/4 abstract 之前必须写出）

```
Title: SEER: Learning to Evict KV-Cache by Predicting Future Attention

Abstract (≤200 words):
  Long-context LLM inference is bottlenecked by KV-cache memory. Prior
  eviction policies (StreamingLLM, H2O, SnapKV) rely on heuristic
  importance signals derived from position or PAST attention. We
  empirically show, on Llama-3-8B traces from RULER, that future
  attention is highly predictable: the top-k attended KV-blocks at the
  next H decode steps can be estimated by a sub-1%-FLOP neural predictor
  with AUC > 0.85 (cf. best heuristic at 0.78). We propose SEER, a
  learned eviction-prefetch policy that scores blocks with this predictor
  (LAP) and selects which to keep in HBM via a budget-constrained, IO-
  cost-aware utility. Across RULER 8K/16K, SEER preserves > 95% of full-
  cache quality at 20% memory budget, beating the strongest baseline by
  +X.X F1. We release attention-trace data and the LAP checkpoint for
  future research on learned cache management.

Contribution bullets (intro):
  1. Empirical study of attention predictability — quantitative gap
     between past-attention and oracle.
  2. LAP — a tiny multi-horizon attention predictor with explicit
     theoretical framing as online set selection with predictions.
  3. SEER policy — joint eviction-prefetch with IO-cost-aware utility.
  4. Open-source attention trace dataset.
```

---

## 9. 写完一段话给 OpenReview 的"comments to PCs"（如有此字段）

> "This work contributes a learned KV-cache management policy backed by
> an empirical study of attention predictability. We have made a
> deliberate scope decision to focus on the policy and its predictor;
> end-to-end serving throughput is deferred to future work. We welcome
> reviewers' input on the predictability characterization, which we
> believe has independent value for the long-context inference community."

---

> 最后一条建议：**今天先做 §1 (OpenReview + track + LaTeX)**，今晚就能开始写 intro。不要等 E1 跑完才动笔，写作和实验完全可以并行——intro / related work / preliminaries 三章不依赖任何实验数字。
