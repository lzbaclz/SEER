# 04 · 命名与术语

## 系统命名规律

Patrick 大组**几乎每篇论文都有一个命名鲜明的产物**。统计一下样本：

| 论文 | 系统/算法名 | 命名思路 |
|---|---|---|
| ATC 17 | **ECPipe** / repair pipelining | EC + Pipe（功能描述拼接）|
| FAST 19 | **OpenEC** / ECDAG | Open + EC（用途明示）|
| TPDS 20 | **CAU** | Cross-rack Aware Update（首字母）|
| FAST 21 | **ECWide** | EC + Wide-stripe |
| ATC 21 | **RepairBoost** | Repair + Boost（动名词拼接）|
| FAST 23 | **ParaRC** | Parallel + RC（Regenerating Codes）|
| INFOCOM 23 | **ET** / Elastic Transformation | 首字母 + 动名词 |
| TON 23 | **ERS** | Elastic Reed-Solomon |
| FAST 26 | **LESS** | 双关：又是首字母又是 "less is more" 的 less |
| TOS 25 | (survey) | survey 论文不命名系统 |

### 规律

1. **2-8 字母的醒目缩写最常见**：CAU, ERS, ET, LESS, ParaRC.
2. **如果不用首字母，则用"功能动词 + 域"拼接**：RepairBoost, ECPipe, ECWide.
3. **缩写要好读 / 好听 / 好记**：LESS、ParaRC 都是双关；不堆砌生僻字母。
4. **同一篇论文里命名层级**：通常有 *方法名*（technique）+ *系统名*（prototype）两层。例子：
   - ATC 17：方法 = repair pipelining，系统 = ECPipe。
   - FAST 23：方法 = parallel repair via sub-packetization，系统 = ParaRC。
5. **命名先于动笔**：命名一旦定下，整篇论文从摘要到 conclusion 用相同的拼写、相同的大小写、相同的小标题前缀。

## Seer 命名建议

我们已经叫 **Seer**。可以再补一个 *方法层* 的命名（Patrick 风格：方法 / 系统两层）。例子：

- *Method:* "Learned cache attribution policy" 或 "Look-ahead-aware KV management (LAKE)"，*System:* Seer.
- *Method:* "Predictive eviction-prefetch (PEP)"，*System:* Seer.
- 或者就让 Seer 同时担任方法名和系统名（很多 USENIX 论文也这么做，OK）。

无论选哪个，落地后要**全篇不变**：不要前面叫 Seer、后面又叫 SeerNet / SeerCache 混用；如果需要分层，一开始就用 SeerNet（model）/ Seer-Runtime（系统）这种**显式分层**的命名。

## 术语规范

### 引入即定义，全文从一

- 第一次出现：完整全名 + 缩写括号，例 *"minimum-storage regenerating (MSR) codes"*。
- 之后：只用缩写。**不要**两段后又写一次完整全名。
- 摘要里如果用了缩写，§1 里第一次出现还是要再展开一次。

### 维度词的固定搭配（组里的"行话"）

下面是这组论文里反复出现的术语；研究人员看一眼就知道你在哪条线上：

- **storage redundancy** （而不是 "storage overhead" / "storage cost"）
- **repair bandwidth** （单位通常是 "blocks read"，不是 bytes）
- **sub-packetization** / sub-packetization level（一个 stripe 切多少 sub-block）
- **degraded read** vs. **full-node recovery** vs. **multi-block repair**（三类失效场景）
- **cross-rack traffic** / cross-rack repair bandwidth
- **stripe** / **block** / **chunk** （stripe 是逻辑单位，block 是 chunk 的同义词）
- **(n, k)** code, **rate** = k/n
- **MDS** (Maximum Distance Separable) code
- **MSR** / **MBR** (regenerating codes)
- **LRC** (locally repairable codes)

### Seer 域里的对应"行话"清单（需要同样固定下来）

| 概念 | 推荐固定术语 | 不要混用的同义词 |
|---|---|---|
| KV cache 单位 | **KV block** (size = page size, e.g. 16 tokens) | KV chunk / page / slot / cell |
| 命中率 | **hit ratio** | hit rate / cache rate |
| 替换决策 | **eviction policy** | replacement policy / cache policy |
| 预取决策 | **prefetch policy** | preload / fetch-ahead |
| 错过 | **miss** / **stall** | cache fault / cache miss event |
| 容量上限 | **HBM budget** | memory budget / cache size |
| 失败回退 | **fallback recompute** | recomputation / re-prefill |
| 模型 | **SeerNet** | Seer model / our predictor |
| 系统 | **Seer runtime** | our framework / our system |
| 方法 | **〈给 method 起一个名字〉** | (避免不命名) |

把这个表贴在 paper draft 的最前面，每次审稿人提问"你的 X 跟 Y 是不是一回事"时只翻这一表。

## 复用 Patrick 组的"trade-off 术语"

他们爱用 X **vs.** Y、X **at the cost of** Y、X **trades off against** Y。Seer 可以套用：

- "*hit ratio* **vs.** *prediction overhead*"
- "*HBM occupancy* **vs.** *prefetch aggressiveness*"
- "*recompute fallback latency* **trades off against** *eviction risk*"

每个 trade-off 都对应一组实验图，在 design space 里画出来 —— 这是 Patrick 组让 reviewer 服气的常用技巧。
