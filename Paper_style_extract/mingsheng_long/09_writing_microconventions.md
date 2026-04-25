# 09 · 写作微习惯（语态 / 引用 / 公式 / 图表 / Related Work / 复现）

这一节把"小毛细血管"集中起来。落笔时一条一条对照。

## A. 语态与人称

- **第一人称复数 "We"** 主导：we propose / we design / we develop / we observe / we show / we demonstrate / we conduct / we ablate
- **避免**：we claim / we believe / we conjecture / we feel
- **避免**：novel / first / unprecedented / state-of-the-art-by-a-large-margin（极少在主文使用，留给 abstract 里**一次**）
- **被动语态**仅用于"客观事实陈述"：例 "These results are reported as MSE averaged over 3 seeds"

## B. 句长与节奏

- 中等句长：15-25 词
- 摘要里偶尔出现一句 30+ 词把多个观察连起来，但**不超过两个分号**
- 节首句往往是**结论性 / 总结性**陈述（topic sentence），后面再展开
- "However" / "Yet" / "Nevertheless" / "On the other hand" 是高频转折

## C. 引用格式与 Related Work 组织

### 引用风格

- NeurIPS / ICML / ICLR：`\citep{...}` 编号 [12, 34]
- CVPR：[12, 34]
- 句中提到"Method X by Author et al."：`Authors \citep{...}` → "Wang et al. [12]"

### Related Work 的"按 idea / axis 组织"

THUML 的 §2 几乎不按时间堆——而是**按 idea axis 切成 3-5 段**：

```
2 Related Work

Time Series Forecasting with Transformers.
   1 段：把 Transformer-based forecasters 总结成"sparse attention" / "decomposition" 
   两条线，列代表方法。

Decomposition Methods.
   1 段：moving average / STL / wavelet 等经典方法 + 现代继承者。

Frequency-Domain Methods.
   1 段：FFT-based / Fourier / wavelet 类工作。

Foundation Models for Time Series.
   1 段：Timer / Sundial / TimeGPT / Lag-Llama 等。
```

每段开头小标题 + 句号（粗体或斜体），2-4 句概括。最后总能写一句：

> "**The closest work to ours is X et al. [12]**, which … Differently, we propose 〈Method〉 that 〈key difference〉."

这一句 reviewer 一定会读，是 Related Work 的"压舱石"。

### 引用密度

- 摘要：0-1 个引用
- Introduction：8-15 个引用（每段 2-3 个）
- Related Work：30-50 个引用（密集）
- Method：5-10 个引用（重要技术 baseline / 灵感来源）
- Experiments：每个 baseline 第一次出现引用一次
- 总引用量：50-80 篇（NeurIPS / ICML 标准），CVPR 较多

### 引用自家组的工作

THUML 每篇都引用自家以前的论文，用来体现 *系列连续性*：

- Autoformer 引 stochastic process classics（不是组里前作）
- TimesNet 引 Autoformer / FEDformer 作为 Transformer-based 代表
- iTransformer 引 Autoformer / TimesNet / Non-stationary 作为 baseline
- Timer 引 TimesNet / iTransformer 作为 supervised baseline

**Seer 可以引用 OrchKvCache（如果同领域）作为前置工作**——让 reviewer 看到"这是一个连续的研究 program"，加分。

## D. 公式排版

### 编号公式

- **method 章核心公式必编号**，例 (1)–(8)
- **不重要的中间步骤不编号**（用 inline）
- 公式末用句号 / 逗号（句子的一部分）

### Notation 表

§3 Preliminary 必备一个 small notation table（4-8 行）：

```
Symbol            Meaning
L                 lookback / history length
H                 forecast horizon
N                 number of variates
B                 batch size
\mathbf{x}_{1:L}  input series
\mathbf{y}_{L+1:L+H}  target future series
```

**reviewer 看一眼就能明白整篇符号**——不用临时翻段落。

### 行内公式 vs displayed 公式

- 单变量小公式 inline：`R^2 = 0.74`
- 多步骤 / 关键公式 displayed
- 不要用 displayed 公式塞 1-2 个变量（浪费空间）

### LaTeX 习惯

- 算子用 `\operatorname`：`\operatorname{TopK}`、`\operatorname{softmax}`
- 集合用 `\mathcal{}` 或 `\mathbb{}`
- 期望用 `\mathbb{E}` （不要用 `E`）
- 范数用 `\|\cdot\|_2`（不要用 `||·||`）

## E. 图 / 表的工艺

### 图 (Figure)

THUML 的图特点：

1. **Fig 1 = motivation**（一图杀人）
2. **Fig 2 / Fig 3 = architecture**
3. **Fig 4+ = showcase / visualization**
4. **每张图都是矢量图（pdf / svg）**——绝不放 jpg / 模糊图
5. **配色克制**：3-5 种颜色，每种颜色含义贯穿整篇
6. **caption 长且自包含**：150-200 词，能在不读正文情况下看懂

### Figure caption 范例（推断）

```
Figure 1. (Top) Standard Transformer fuses all variates at each timestamp into
a single token, producing meaningless attention maps. (Bottom) iTransformer
inverts the tokenization: each variate becomes a token, enabling attention to
capture cross-variate dependencies. We highlight three variates for clarity.
This inversion is the core insight of this paper.
```

caption 解释：(a) 上下两图的对比；(b) 论文核心 insight。**不只是 "Architecture of method X"**。

### 表 (Table)

- bold 最佳，underline 次佳，统一约定
- 表头用粗体 + 横线
- caption 长，**包含 take-away**
- 跨页大表 → 拆 + 把详细放 appendix
- error bars: "0.098 ± 0.003" 格式

### 表中的 metric 方向标注

NeurIPS 现在推荐：

```
Method          Hit Ratio (↑)   p99 TTFT (↓)   FLOPs (↓)
```

让 reader 不用想"越高越好还是越低越好"。

## F. *Insight / Observation / Proposition / Theorem* 的标注规范

```
*Observation 1.*  (informal empirical finding)
*Insight 1.*     (design rationale)
*Definition 1.*  (formal naming of a concept)
*Proposition 1.* (formal statement, easier than theorem)
*Theorem 1.*     (full theorem, requires proof)
*Lemma 1.*       (auxiliary)
*Corollary 1.*   (downstream of theorem)
*Remark 1.*      (informal note after theorem)
```

italic 或 bold 都可以，但**全文一致**。每个 environment 用 `\newtheorem{...}`。

## G. Reproducibility（NeurIPS 现在硬要求）

THUML 在 NeurIPS / ICML 评审里得到 reproducibility 高分的原因：

1. **GitHub URL 在 abstract 末尾或 §1 末尾就给**
2. **代码包含训练 + 评估 + 数据下载脚本**
3. **统一 framework**：Time-Series-Library / Transfer-Learning-Library
4. **超参数表**：appendix 有完整的超参数 + 训练 protocol 表
5. **Checkpoint** 公开：模型权重在 repo 或 HF 上
6. **统一的评估协议**：所有 baselines 在同一 dataloader / metric / seed 下运行
7. **计算成本透明**："training takes 4 hours on a single A100" — 让 reviewer 信任你跑过

### Seer 的 reproducibility 落地

- 主文 abstract 末尾：`Code: https://github.com/.../seer`
- §5.1 末尾：`We use vLLM v0.X with all baselines reimplemented under the same KV-cache interface (Appendix B).`
- Appendix B：超参数表（SeerNet 训练 / inference 配置）
- Appendix D：训练硬件 / 时间 / 成本明细
- HuggingFace Hub：SeerNet checkpoint

## H. 写作时的"自检清单"（5 分钟自查）

每写完一节，对照这一组问题：

1. 这一节的 *第一句* 是不是 topic sentence？
2. 有没有用 "we" 主动表达？
3. 有没有用 hyphenated capitalised 给关键概念命名？
4. 关键论断是不是用 *(Observation)* / *(Insight)* / *(Proposition)* 显式标注？
5. 有没有 5+ 编号公式且每个都有命名意义？
6. 有没有至少一张能"一图杀人"的视觉？
7. 表格有没有 bold-best / underline-second / metric 方向标注？
8. 引用有没有按 idea-axis 组织（Related Work）？
9. 数字有没有给 ± std？
10. 有没有 GitHub URL？

每个 NO 都是潜在的 reviewer 扣分点。

## I. 一些小忌讳（NeurIPS 评委不喜欢）

- ❌ Future work 写在 conclusion 里
- ❌ "We are the first to ..."
- ❌ "Our method significantly outperforms" 没数字支撑
- ❌ Architecture 图全是黑灰方框
- ❌ 一张图横跨两页
- ❌ 缩写没在 §1 里 expand 一次
- ❌ Related Work 完全按时间堆
- ❌ Implementation 细节藏在 footnote
- ❌ 实验只用单 seed
- ❌ 没有 limitations 子节
- ❌ 摘要超过 250 词

## J. 一些受欢迎的小习惯

- ✅ 在 §1 末尾给 paper outline（NeurIPS 现在不强求，但有比没好）
- ✅ Method 章每个子节末尾一句"validated in §5.X"
- ✅ Ablation 的每行都给一个**自然语言注释**（"removing X breaks Y"）
- ✅ 每张主图配 caption ≥ 100 词
- ✅ Appendix 有 "Glossary of Notation" 表（如果符号多）
- ✅ "Limitations" 子节诚恳列 3-4 条
