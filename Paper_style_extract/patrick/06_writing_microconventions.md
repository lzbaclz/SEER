# 06 · 写作微习惯（语态 / 引用 / 图表 / 排版）

这是落笔时容易忽略但贯穿全篇的"小毛细血管"。

## 语态与人称

- **"We"** 表"作者"，**几乎每段都至少一次**。例：*"We propose / We design / We implement / We evaluate / We show that"*。
- **"This paper"** / **"In this paper"** 用法保留 — 主要在 §1 末段引出贡献，以及 conclusion 起头。
- **被动语态**留给"客观结果 / 通用事实"：*"The repair bandwidth is bounded by ..."*；不要让被动语态主导描述自己工作的句子。
- **避免强词**：novel / innovative / groundbreaking / state-of-the-art —— 几乎不用。读 Patrick 组论文你会发现他们用的是 "we propose / we design / our experiments show"，把"是不是 novel"留给 reviewer 自己判断。

## 句长与节奏

- 偏好**中等长度的明确句子**（15-25 词）。
- 引言里偶尔出现一句长句把多个限制串起来，但**永远不超过两个分号**。
- "However" / "Yet" / "Nevertheless" / "On the other hand" 是高频转折词。
- "We make the following observation" 这类**显式提示语**频繁出现，让 reviewer 不会漏掉关键论断。

## 引用习惯（IEEE/USENIX 通用）

- 引用编号风格随 venue：USENIX 是 [12]，IEEE 也是 [12]。组里几乎不用 author-year (Smith 2019) 风格。
- **每个 prior 工作至少出现一次时**带一句解释："Recent work \[12] proposes 〈technique〉, but \[12] focuses on 〈X〉, which differs from our 〈Y〉."
- **Related Work 节**几乎按"主题 + 时间"双索引组织：
  ```
  Code constructions for repair bandwidth.   [a, b, c, d]
  I/O-efficient code constructions.          [e, f, g]
  Repair scheduling.                         [h, i, j]
  Wide-stripe codes.                         [k, l]
  ```
  每段开头小标题 + 句号，2-4 句话总结整条线。
- "The closest work to ours is \[X], …" 这一句几乎每篇都出现一次，明确"我们和谁最像、差异在哪"。
- 引用自己的早期工作非常正常 —— Patrick 组论文会大量引 ECPipe / OpenEC / RepairBoost 来解释 design space 演化。

## 图表小习惯

### Figure caption

- 一两句话：**不只是"Figure X: Y"**，而是把图的主结论也写进去。
- 例（推断）："Figure 4. Single-block repair time of ECPipe vs. baselines under (14, 10) MSR codes. ECPipe is within 5% of the normal-read time."

### Table caption

- 在 caption 里写"加粗的是最好结果"或"省略号表示超出阈值不再测量"。

### 文中引用图

- 用 "Figure X shows that …" / "Table X reports …"，**每张图至少在文中被讨论一次**。
- 不写"As can be seen from Figure 4"这种冗长开头。

### 图序与行序对齐

- IEEE 双栏排版下，把 fig/table 排到对应段落上方或同页是 IEEEtran 默认；组里不会硬塞跨页大图。

## 数学 / 符号

- **每节使用前**先有一段或一张 notation table 定义所有符号。
- 字母选择克制：**小写斜体 n, k, r 表示参数**，**大写 N, K 表示集合**，**斜体 N(·)表示函数 / set**。少用 𝒜 𝓑 这种花体。
- 定理 / 引理 / 观察 / 结论用专门 environment（`\theorem`, `\lemma`, `\observation`），数字独立编号或全篇统一编号都可。
- 证明放在 appendix 还是正文里，看 venue：USENIX 通常 appendix，IEEE journal 通常正文。

## 一些小套路（高频但不必每篇出现）

- **"Observation X."** 在 motivation 和 design 段落里频繁出现，用 italic 提示："*Observation 1.* Repair traffic is bottlenecked by the slowest link …"
- **"Intuition."** 提示性小标题（可能是 inline italic）：用一段话讲核心 design 为什么 work。
- **Proof sketch** 替代 full proof 在主文，full proof 放 appendix。
- "We have the following theorem …" — 提示型导入。
- 章节末段常有一句**总结**："In summary, 〈primitive name〉 …"
- 每个 design 节末段常有一句"To this end, we propose …"承上启下。

## 章节小结

USENIX 论文很多章节有半段 summary 收尾："We summarize the design as follows …"。这是给 reviewer 的"喘气位"，强烈推荐照做。

## 不做的写法

- 不用 emoji。
- 不用 footnote 写关键论断（footnote 只放 URL / acknowledgment / 不重要旁注）。
- 不在正文用代码块（代码片段或者放 figure，或者放 appendix）。
- 不用过度形容词（"very", "extremely", "dramatically" 偶尔可，**"absolutely"、"undoubtedly"** 不用）。
- 不写"Note that …"超过两次/页。

## Seer 写作的迁移建议

- NeurIPS reviewer 一般不在乎"我们这是 novel 的"，但很在意"你的 claim 有没有量化证据"。Patrick 组的"少 selling、多数据"完全适用。
- Seer 论文每个 §4 design primitive 末尾都加一句"In summary, this primitive contributes 〈X% / Y× improvement〉 (Section 5.X)"，让 reader 立刻知道实验在哪验证。
- Related Work 用 Patrick 组的"主题分组" 格式，能避免 NeurIPS 论文常见的"按时间堆"问题：
  - LLM serving systems & KV management
  - Cache replacement policies (classical)
  - Learning-augmented data structures
  - Workload prediction for caching
- "The closest work to ours is …" 这一句一定要在 Related Work 写一次。
