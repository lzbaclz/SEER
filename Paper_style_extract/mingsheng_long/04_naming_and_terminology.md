# 04 · 命名 + 术语规范

THUML 在命名学上的功夫极深，**论文是不是 well-crafted，看一眼系统名 + 子组件名就知道**。

## 系统名 (Top-level)

| 论文 | 系统名 | 命名思路 | 备注 |
|---|---|---|---|
| ICML 15 | DAN | 直白缩写：Deep Adaptation Networks | 朴素，2015 当时审美 |
| ICML 17 | JAN | 同上：Joint Adaptation Networks | "Joint" 强调 marginal→joint 升级 |
| NeurIPS 18 | CDAN | Conditional + Domain Adversarial Networks | "C" 前缀承前作 (DANN) |
| ECCV 18 | PADA | Partial Adversarial Domain Adaptation | "Partial" 直接表达问题 |
| CVPR 19 | UAN (在 UDA 任务里) | Universal Adaptation Network | 系统名 ≠ 任务名（很重要的区分）|
| ICML 19 | MDD | 概念名：Margin Disparity Discrepancy | 一个 *divergence* 的名字，不像系统 |
| NeurIPS 21 | Autoformer | Auto + former（former = transformer） | 后缀型，借 "Transformer" 词根 |
| ICML 22 | FEDformer | Frequency Enhanced Decomposed + former | 三段缩写 + former |
| NeurIPS 22 | Non-stationary Transformers | 直白形容词 + Transformer | 不缩写也行 |
| ICLR 23 | TimesNet | Times + Net | "Times" 表"时间"，避开 Time Series 烂大街 |
| NeurIPS 23 | SimMTM | Simple Masked Time-series Modeling | 谦逊命名（"Simple"）有反差感 |
| ICLR 24 | iTransformer | i 前缀 = inverted | **当前最有 brand 感的命名** |
| ICML 24 | Timer | 双关：Time + 器 / 计时器 | 简短易记 |
| ICLR 25 | Timer-XL | Timer 的扩展版 | 借 GPT-XL 命名习惯 |
| ICML 25 Oral | Sundial | 日晷——具象时间隐喻 | 完全脱离 transformer 形态命名 |

### 命名学规律

1. **缩写用大写**：DAN, JAN, CDAN, MDD, UAN, S3。
2. **借后缀**："-Net"（TimesNet）、"-former"（Autoformer / FEDformer）—— 让读者第一眼归类（"哦这是个网络/Transformer 变种"）。
3. **小写前缀**带含义："i" = inverted（iTransformer）。这种命名**最贵**，能自带 "key insight is in the prefix" 的 marketing 效果。
4. **现实物**当名字（Timer、Sundial）：脱离技术术语，更接近 LLM 系统命名风格（GPT、Gemini、Claude）。
5. **系统名 ≠ 任务名**：UAN 是系统，UDA 是任务（universal domain adaptation）。**不要混用**。

### 给 Seer 的命名建议

我们已经叫 **Seer**。可以做一个**两层命名**让审稿人喜欢：

- **方法 / framework name**：Seer
- **预测模型 name**：SeerNet（预测 reuse-distance 的小模型）—— 借 TimesNet/Sundial 的 "-Net" 后缀
- **policy name**：考虑给 eviction/prefetch policy 起一个有 brand 的名字。可选项：
  - *Pareto-Aware Eviction* (PAE)
  - *Look-ahead Prefetch Scheduler* (LAPS)
  - *Reuse-Distance-Driven Retention* (RDR)

整篇论文一旦定下名字，**全篇大小写、连字符都不变**：永远写 "Auto-Correlation"，不会一会儿 "auto-correlation" 一会儿 "auto correlation"。

## 子组件命名 (Hyphenated Capitalised)

THUML 的招牌：**hyphenated capitalised**。重要的子模块全文都按专名处理：

- *Auto-Correlation Mechanism* (Autoformer)
- *Series Decomposition Block* (Autoformer)
- *Temporal 2D-Variation* (TimesNet)
- *intraperiod-variation* / *interperiod-variation* (TimesNet) ← 注意这两个仍小写但带连字符
- *TimesBlock* (TimesNet) ← 单字（无连字符）但首字母大写
- *Series Stationarization* / *De-stationary Attention* (Non-stationary Transformers)
- *S3 (Single-Series Sequence)* (Timer)
- *TimeFlow Loss* (Sundial)
- *Multilinear Conditioning* / *Entropy Conditioning* (CDAN)
- *Sample-level Transferability* (UAN)
- *Margin Disparity Discrepancy* (MDD)
- *Joint Maximum Mean Discrepancy (JMMD)* (JAN)

### 命名规则总结

1. 子组件 = "形容词-形容词 名词" 或 "动词形容词 + 名词"，**每个词首字母大写**，**用连字符**（如果是双词）。
2. 缩写在第一次出现时用括号给出，之后用缩写。
3. 子组件命名应当**既描述功能又有形状**——例如 "Auto-Correlation" 同时表达 "operation = correlation" 和 "domain = auto/temporal"。
4. **不命名实现细节**——具体函数 / 类名留给代码，论文里只有概念名。

### 给 Seer 的子组件命名

```
SeerNet                            # 预测模型主干
Reuse-Distance Predictor (RDP)     # SeerNet 的子任务名
Pareto-Aware Eviction (PAE)        # eviction policy 的命名
Look-ahead Prefetch Scheduler (LPS) # prefetch policy
Hit-Distance Loss                  # 训练 SeerNet 用的 loss
Block Trajectory Trace (BTT)       # 训练数据格式名
```

> **小贴士**：如果某个组件你写不清楚名字，那它在 method 里就还**不够清晰**——返工 design，别返工命名。

## 术语规范

### THUML 圈子的"行话" — 时序

在时序系列论文里，下面术语是固定的，圈子里读者一看就知道你在哪条线上：

| 概念 | 推荐固定术语 | 不要混用 |
|---|---|---|
| 输入历史 | **lookback window** / **history horizon** | observation window |
| 预测时长 | **forecasting horizon** / **future horizon** | prediction length |
| 一个 token / patch | **time point** / **temporal token** / **patch** | step / unit |
| 多变量 | **variates** / **channels** | features / dimensions |
| 多尺度 | **multi-periodicity** / **multi-scale** | multi-resolution |
| 平稳化 | **stationarization** / **detrending** | normalization |
| 测试集 | **forecast horizon evaluation** | test set 不算行话错 |
| 预训练 | **pre-training on TimeBench / S3** | training |

### 圈子行话 — 迁移学习

| 概念 | 推荐固定术语 | 不要混用 |
|---|---|---|
| 源 / 目标域 | **source / target domain** | training/testing domain |
| 共享标签空间 | **shared label space** | overlapping classes |
| 私有标签空间 | **private label space** | unknown classes |
| 域分类器 | **domain discriminator** | domain classifier |
| 特征对齐 | **feature alignment** / **distribution matching** | embedding align |
| 转移性 | **transferability** | transfer ability |
| 负迁移 | **negative transfer** | bad transfer |
| 通用性 | **universality** (UDA-style) / **task-general** | generic |

### Seer 域的对应行话清单

| 概念 | 推荐固定术语 | 不要混用 |
|---|---|---|
| KV cache 单元 | **KV block** | KV chunk / page / tile |
| 命中 | **hit** | use / access |
| 失效 | **miss** | fault |
| 替换 | **eviction** | replacement / drop |
| 预取 | **prefetch** | preload / fetch-ahead |
| 命中率 | **hit ratio** | hit rate |
| 占用率 | **HBM occupancy** | memory usage |
| 重用距离 | **reuse distance** | reuse interval |
| 工作负载 | **inference workload** / **trace** | data |
| 模型 | **SeerNet** | predictor / our model |
| 系统 | **Seer** | our framework / our system |
| Policy | **Pareto-Aware Eviction (PAE)** / **Look-ahead Prefetch Scheduler (LPS)** | (任何模糊代称) |

把这一表打印贴桌前。每次审稿人问 "your X is the same as our Y?" 时，只翻这一表。

## 公式符号约定

THUML 的公式符号选择克制，以下是反复出现的写法（mid-confidence）：

```
n   = batch size
N   = number of variates / channels (multivariate dimension)
L   = lookback length
H   = forecasting horizon
d_model  = model hidden dim
T   = (在 DA 论文中) target domain
S   = source domain
\mathcal{D}_s, \mathcal{D}_t   = source / target distribution
K   = number of classes
H_\rho   = margin-loss-induced hypothesis space (MDD)
```

使用习惯：

- 标量小写斜体：`n, k, t, l, h`
- 集合 / 分布大写花体：`\mathcal{D}, \mathcal{H}`
- 向量加粗：`\mathbf{x}`（可选）
- 矩阵大写斜体：`X, W`
- "ground truth" 用 `\hat{y}` 或 `y^\star`

第一次引入符号时**永远在 §3 给一个 notation table**（4-8 行），让 reviewer 不必满文档找定义。

## 标点 / 排版的小规则

- 公式末尾用句号 / 逗号（句子的一部分）
- "Equation (3)" 而不是 "Eq.(3)"，"Section 4.2" 而不是 "Sec. 4.2"
- 中文论文不要混排——**全英文**
- 缩写第一次：完整写法（首字母大写）+ 括号缩写。后文只用缩写。
- "We" 用首字母大写不只在句首，**专有方法名也是**：例如 "We propose **Seer**"，永远 Seer 大写。
- 引用风格随 venue：NeurIPS/ICML/ICLR 用 `\citep` 数字风格 [12]；CVPR 同样。
