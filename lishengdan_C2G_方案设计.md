# lishengdan_C2G_方案设计

**挑战：** C2G 参数高尔夫
**作者：** lishengdan
**目标等级：** 主目标 **Level 2（Silver）**；条件目标 **Level 3（Gold）**；同时按 **Level 4** 的形式要求交付英文技术报告（见 §7 的诚实说明）。

---

## 1. 约束的再表述：这个赛道到底在优化什么

先把这个挑战的目标函数写清楚，因为它不是"训一个好模型"：

```
minimize     BPB( θ_quantized )
subject to   |artifact(θ_quantized)| ≤ 16,000,000 bytes
             train_time(θ) ≤ 600 s on 8×H100 SXM
             eval_time ≤ 600 s on 8×H100 SXM
             θ 只能看到 FineWeb 训练分片
```

按官方 README 的表述，这是一个 **L(N) 优化**：参数量 N 被 16MB 硬钉死，数据、算力、步数、架构都不设限。所以真正的自由度只有三个：

1. **把 16MB 花在哪。** 参数量、精度、tokenizer、代码长度——四者在争同一个预算。
2. **在 10 分钟里让参数学到多少。** 这是系统和优化器的战场。
3. **部署时损失多少。** 训练是在 fp32/bf16 里做的，交付是在 int6 里做的，两者的差距就是"量化税"（quantization tax）。

**本方案的结构假设：** 三者里 **第 1 条和第 3 条是 16MB 这个约束独有的**，也是官方 baseline（1.2244）表现最保守的地方——它用 percentile clip 的全 int8，用 1024 词表。所以本方案把火力集中在「预算怎么分配」而不是「架构怎么改」。

---

## 2. 目标等级与达标路径

| 等级 | 官方门槛 | 本方案的达标路径 |
|---|---|---|
| Level 1（Bronze） | 复现 1.2244 ± 0.005 | **本提交的脚本可直接跑出 baseline 语义**；已在代理语料上完成等价流水线验证（见 §5） |
| Level 2（Silver） | BPB < 1.18 | 单点：SP8192 换掉 SP1024。榜单证据显示 SP 谱系能把 SP1024 的 1.1586 推到 1.0810 量级 |
| Level 3（Gold） | BPB < 1.12 | 组合：SP8192 + 并行残差 + 深度递归 + QK-Gain 5.25 + QAT + GPTQ-SDClip + QA-EMA |
| Level 4（Platinum） | BPB < 1.085 | 需要原创架构改动。**本方案不声称达标**，详见 §7 |

**我对目标的诚实界定：** 由于执行环境无 GPU（详见 §5.1），本提交在**本地**拿到的是组件排序与量化税测量，不是榜单分数。因此上表的「达标路径」是**设计文档意义上的路线**，而不是已完成的成绩。任何把两者混为一谈的写法我都会在交付物里拆开标注。

---

## 3. 技术选型

### 3.1 Tokenizer：SP-8192 BPE

| 项 | SP-1024（baseline） | SP-8192（本方案） |
|---|---|---|
| 训练算法 | SentencePiece BPE | 同（`model_type=bpe`, `character_coverage=0.999`, `byte_fallback=True`, `split_digits=True`, `normalization_rule_name=nmt_nfkc`, `add_dummy_prefix=False`, pad/bos/eos/unk = 0/1/2/3, `hard_vocab_limit=False`） |
| tied embedding 参数量 @ d=512 | 1024×512 = 0.52M | 8192×512 = 4.19M |
| 每 token 平均字节 | ≈ 2.7 | ≈ 4.6 |
| 相对基线收益（榜单） | — | -0.03 ~ -0.05 BPB |

**关键取舍：** 词表从 1024 到 8192 会让 tied embedding 膨胀 8 倍。在 16MB 预算里这是**真实的挤占**，所以它不是一个"免费的改动"。本方案的应对是把它纳入 §3.5 的混合精度量化：embedding 给 int8，block 矩阵给 int6。

### 3.2 架构：三个互补的改动

**(a) Parallel Residuals（GPT-J 风格并行残差）**

```
串行（baseline）:  x ← x + attn(norm(x));  x ← x + mlp(norm(x))
并行（本方案）  :  x ← x + attn(norm(x)) + mlp(norm(x))
```

两条支路从**同一个 pre-residual 激活**读取，输出相加。理由：串行结构里 MLP 读到的是"已经被 attention 更新过"的表示，两条支路实际上在串联深度；并行结构让它们在同一"深度层"上叠加，**在物理层数固定的前提下提高了有效宽度**——这在参数量被 16MB 钉死时是有意义的。榜单上 `2026-03-31_ParallelResiduals_MiniDepthRecurrence` = 1.1063 是这条路径的第一个公开胜果。

**(b) Depth Recurrence（深度递归）**

把"层列表"抽象成"**执行调度列表**"。例如 6 个物理层配调度 `0,1,2,3,4,5,2,3,4,5`，就是让第 2–4 层各执行两次，共 10 个执行位置：

- 参数量按**物理层**算（6 层）
- 计算深度按**执行位置**算（10 层）

这是在 16MB 约束下唯一能"免费加深度"的办法。榜单 SOTA 用 11 个物理层展开出 17 个虚拟层。

**(c) QK-Gain**

`q ← q · g`，`g` 是可学习的 per-head 正标量（初始化 5.25）。作用是在 QK-RMSNorm 之后重新给 attention logits 一个合适的尺度，避免 softmax 过早饱和。榜单上从 `QK_GAIN_INIT=4.0` 到 `5.25` 是单调改善的。

### 3.3 优化器：Muon（矩阵）+ AdamW（embedding/标量）

保留官方分组：
- Transformer block 里的 2D 矩阵 → **Muon**（Newton-Schulz 正交化，5 步，momentum 0.85→0.95 warmup）
- tied embedding → Adam，`lr = 0.05`；标量/控制张量 → Adam，`lr = 0.04`

**为什么不动它：** Muon 是 Keller Jordan 的成果、已被榜单反复验证，重写只会引入 bug。本方案的新增只有一处：给 Muon 加了**可选 weight decay**（`WEIGHT_DECAY`，默认 0），因为榜单上 `WD=0.09~0.095` 的配方需要它。默认关闭以保证 baseline 可复现。

### 3.4 量化：SDClip + 混合精度 + GPTQ，以及原创的 QA-EMA

**(a) SDClip 取代 percentile clip**

```
percentile（baseline）: clip(row) = quantile(|row|, 0.9999984)
SDClip（本方案）      : clip(row) = k · std(row),   k = 12.85
scale(row)            = clip(row) / (2^(b-1) − 1)
```

percentile 的阈值由极值驱动；`k·std` 由分布的尺度驱动。率失真视角下，后者的有效位宽分配更稳定。

**(b) 混合精度分配**

| 张量类型 | 位数 | 理由 |
|---|---|---|
| tied embedding（输入查表 + 输出投影，一物两用） | int8 | 量化误差同时污染两条路径，多给 2 bit |
| block 内 2D 矩阵（attn / mlp 投影） | int6 | 占体积最大，且对量化相对不敏感 |
| 控制张量（`attn_scale`/`mlp_scale`/`resid_mix`/`q_gain`/`skip_weights`） | fp32 | 数值小、数量少，量化它们的收益是负的 |
| 其它小张量 | fp16 | 省字节 |

**(c) 训练末期 QAT**：`QAT_START_FRAC`，`<0` 表示"从 warmdown 起点开始"。让"适应量化格点"和"收敛"发生在同一段时间里；warmdown 之前 LR 太大，格点会被冲掉。

**(d) QA-EMA（本提交原创）**

朴素 EMA 在连续空间平均，部署在离散格点；QA-EMA 把平均做在格点上：

```
src    = dequant(quant(θ_t, b, k))
shadow ← decay · shadow + (1 − decay) · src
```

假设：让被平均的量始终落在格点上，平均结果就更接近"格点意义上的好点"，从而压低 quantization tax。**这是可证伪的**，已在方案草案里预注册为 P7。

**(e) GPTQ 导出路径**：通过 `register_forward_pre_hook` 累积每个 `CastedLinear` 的 `XᵀX`，做逐层列序 GPTQ（阻尼 1%、Cholesky 逆、128 列分块、误差回灌）。**校准数据只取训练分片**，绝不碰验证集。

### 3.5 打包：压缩

`zlib-9`（baseline）或 `brotli-11`（本方案）。模型张量经过量化后字节分布高度集中，brotli 的上下文建模通常能多压 3–8%。实测（本地小模型）brotli 比 zlib 少 11% 字节。

---

## 4. 实验矩阵

### 4.1 设计原则（三条，按重要性排序）

1. **唯一变量。** 所有臂共用同一份 `lishengdan_C2G_train_gpt.py`、同一套 BASE 超参、同一条数据流，**只有目标组件对应的那一个环境变量不同**。臂之间代码不同 → 实验不成立。
2. **配对比较。** 同一个 seed 下算 `Δ = bpb(arm, seed) − bpb(baseline, seed)`，再对 Δ 求均值。两条臂共享数据顺序与初始化分布，seed 方差在相减时大幅抵消。
3. **看部署指标，不看训练指标。** 唯一的主指标是 **post-quantization `val_bpb`**（量化 + 压缩 + roundtrip 之后）。同时记录 `quantization tax = pre_quant_bpb − final_bpb`。

### 4.2 臂清单

下表是**实际运行**的臂（与 `experiments/run_matrix.py` 中的 `ARMS` 一一对应）。

| 臂 | 变量 | 对应预测 | seed 数 |
|---|---|---|---|
| `A0_baseline` | 无（SP1024 + 训练脚本默认导出） | 参照 | 3 |
| `A1_sp4096` | 词表 4096 | P1 | 2 |
| `A2_sp8192` | 词表 8192 | P1 | 3 |
| `A3_qkgain` | `QK_GAIN_INIT=5.25` | P2 | 2 |
| `A4_parresid` | `PARALLEL_RESID_FROM=4` | P3 | 2 |
| `A5_recurrence` | `LAYER_SCHEDULE=0,1,2,3,4,5,2,3,4,5` | P4 | 2 |
| `A6_ema` | `EMA_DECAY=0.996`,`EMA_START_FRAC=0.65` | P5 | 2 |
| `Q1_int6_pct` | int6 + percentile clip（与 `A0` 同配置 → 确定性对照） | — | 2 |
| `Q2_int6_sd` | int6 + SDClip | 裁剪规则一格（P6/P8 的分母） | 2 |
| `Q3_int8_sd` | int8 + SDClip | 位宽一格 | 2 |
| `Q4_int6_sd_gptq` | int6 + SDClip + GPTQ | P8 | 2 |
| `Q5_int6_sd_qat` | int6 + SDClip + QAT（warmdown 对齐） | P6 | 2 |
| `B1_combo` | 全栈，无 QA-EMA | — | 3 |
| `B2_combo_qaema` | 全栈 + QA-EMA（本提交原创项） | P7 | 3 |
| `B3_combo_gptqoff` | 全栈 + QA-EMA，导出退回 RTN+SDClip | P8 的交叉验证 | 2 |

共 **15 臂、34 次运行**。

> **量化块的事后更正（记录在案，不静默修改）。** 上表初版把 `A0` 写成
> "官方默认导出 int8/percentile"，这是**错的**。训练脚本 `lishengdan_C2G_train_gpt.py` 的
> 实际默认是 `QUANT_BITS=6`、`QUANT_CLIP=percentile`（见 `_env("QUANT_BITS", 6)`），
> 而 `A0` 走的就是默认值。实测配置佐证：三个 `A0` 种子的 `quant_bits` 字段都是 `6`。
>
> 后果有两条：
> 1. **(8, percentile) 这一格缺失** —— 原计划的 2×2 只跑了 (6, percentile)、(6, SDClip)、
>    (8, SDClip) 三格。裁剪规则的对比因此只在 int6 上得到一次检验。
> 2. **`Q1` 与 `A0` 配置完全相同**，不再是一个独立的消融臂。但它成了本包里最直接的
>    **确定性证据**：同配置同种子下，`final_bpb`/`final_loss`/`pre_quant_bpb`/`artifact_bytes`/
>    `n_params`/`steps_completed`/`train_tokens_seen` **七个字段逐位相同**。所以保留它、并改称"确定性对照"。
>
> **为什么不重跑：** 所有组件对比都是同 seed 配对，量化配置在每条臂内部是常量，
> `A1`–`A6` 相对 `A0` 的 Δ 不受影响。重跑基线会作废 16 次已完成的参照运行，代价大于收益。
> 数据保留、标签改正 —— `run_matrix.py` 的 `ARMS` 注释与自动生成的
> `lishengdan_C2G_ablation.md` 已同步更正，`A0` 的量化参数现在也在 `ARMS` 里**显式写出**，
> 不再依赖读者去翻训练脚本的默认值。
>
> **另一处偏离（记录在案）。** 规划时量化块只写了 3 个臂
> （`A7_rtn_int6` / `A8_gptq_int6` / `A9_qat_int6`），实现时扩成 5 个（Q1–Q5）。
> 原因是原设计的 `A7_rtn_int6` 把**位宽**和**裁剪规则**两个变量混在同一个臂里——
> 这恰恰是本方案自己在 §4.1 里反对的做法（"一次只动一个变量"）。
> 拆开之后，"位宽"与"裁剪规则"才各自可归因。同时把 B3 从 3 seeds 降到 2 seeds，
> 把省下的算力让给 B1/B2（P7 是唯一原创项，需要最强证据）。

### 4.3 阶段划分与算力分配

| 阶段 | 硬件 | 规模 | 用途 | 成本 |
|---|---|---|---|---|
| **P0 本地筛选** | CPU 22 线程 | 6 层 × 128 维，约 1.6M token/次 | 组件排序、量化税测量、回归测试 | $0 |
| **P1 单卡验证** | 1×H100 | 完整 10 分钟配置 | 确认排序在大模型上保持；确认 16MB 上限被触发 | ≈ $4 |
| **P2 正式提交** | 8×H100 | 最终 3 组 × 3 seeds | 产出 `submission.json` | ≈ $20–24 |

**$25 分配：** P1 约 $4，P2 约 $20。不留余量是刻意的 —— 超支意味着 P0/P1 的筛选不合格。

---

## 5. 执行环境与它带来的限制

### 5.1 事实陈述

本提交的执行环境：

| 项 | 实际值 |
|---|---|
| GPU | **无**（`nvidia-smi: command not found`） |
| PyTorch | 2.14.0+**cpu**（从 `download.pytorch.org/whl/cpu` 装，装在 `D:\c2g_env` 隔离 venv） |
| CPU | Intel 22 逻辑核心 |
| huggingface.co / hf-mirror.com | **HTTP 000，不可达** |
| dumps.wikimedia.org | 可达 |
| PyPI（阿里/清华镜像） | 可达 |

### 5.2 我做了什么来换取复现性

因为拿不到 GPU 和 FineWeb，我把能做的都做了：

1. **设备无关化训练脚本**：同一份 `train_gpt.py` 在 CPU 上跑筛选、在 CUDA 上跑正式配置。
2. **自建代理语料**：Simple English Wikipedia（CC BY-SA 4.0），用 multistream BZ2 的**前缀抽取 + 完整块解压**得到一个**确定性**的文本集合（同一字节区间永远得到同一份文本）。按官方 `download_hf_docs_and_tokenize.py` 的**逐字配方**训练 SP-1024 / SP-4096 / SP-8192 三个 tokenizer，按官方 `.bin` 分片格式打包。
3. **完整流水线的真实运行**：训练 → 验证 → 量化 → 压缩 → roundtrip 重评估，全部跑通，产出真实数字。

### 5.3 这个替代方案**不能**证明什么（必须写明）

| 能证明 | 不能证明 |
|---|---|
| 各组件在**同一模型、同一数据**上的**相对排序** | 组件的收益幅度在官方设定下是多少 |
| quantization tax 的**相对**大小与 QAT/QA-EMA 对它的影响 | 绝对 BPB 是否达到 1.18 / 1.12 |
| 代码路径正确、可复现、可回归 | 10 分钟 / 16MB 两个硬约束是否满足 |
| 代理语料上 tokenizer 升级的收益方向 | 该收益在 FineWeb 域名分布下是否同样成立 |

**结论：本提交提供的是"排序证据 + 可运行产物"，不是"成绩"。** 任何引用排行榜数字（1.2244 / 1.1586 / 1.0810）的地方都明确标注为官方数字。

---

## 6. 风险登记表

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | 代理语料域名分布与 FineWeb 差异过大，tokenizer 结论不可迁移 | 高 | 高 | 结论只声明为"方向性"；在报告里给出域差异的量化对照（每 token 字节数、字符类别分布） |
| R2 | 本地模型太小，SP8192 的 embedding 挤占效应被放大 | 高 | 中 | 这是**真实存在的效应**，正是 P1 预测要检验的对象；在 AAR 中单独讨论容量补偿 |
| R3 | 训练 token 数远低于 Chinchilla 最优点，组件收益被欠训练掩盖 | 高 | 高 | 配对比较（同一 seed 下相减）；报告 3 个检查点的 BPB 曲线而不只报终值 |
| R4 | QA-EMA 的收益与 QAT 的收益混淆，无法归因 | 中 | 中 | 严格配对臂 `B1_combo` vs `B2_combo_qaema`，唯一变量是 `EMA_QUANT_AWARE` |
| R5 | GPTQ 校准用了验证集 → 违规 | 低 | **致命** | 代码层面强制：校准 batch 只从 `fineweb_train_*.bin` 取 |
| R6 | artifact 超 16MB 未被发现 | 中 | 高 | 每次运行都在导出后打印 `Total submission size`，并与 16,000,000 比较 |
| R7 | 深拷贝式地保留官方 1126 行导致代码本身超过 16MB 预算 | 低 | 中 | 官方限制代码应集中在 `train_gpt.py`；本提交脚本约 62 KB，占比 0.4% |

---

## 7. 关于 Level 4 的诚实说明

Level 4 的硬门槛是 **BPB < 1.085**，且明确要求「原创的架构改动、原创的训练策略、原创的量化方案、完整的 ablation + 理论解释、可复现脚本 + 3 seeds 日志、英文技术报告」。

**本提交的准备情况：**

| Level 4 要求 | 状态 |
|---|---|
| 原创量化方案 | ✅ 提出 QA-EMA 并给出可证伪假设；本地实验给出检验结果 |
| 完整 ablation | ✅ 15 臂 / 34 次运行，配对比较 |
| 可复现脚本 | ✅ 设备无关、环境变量驱动、附独立复现评估器 |
| 3 seeds 日志 | ✅ 关键臂 3 seeds（日志在 `lishengdan_C2G_logs/`） |
| 英文技术报告 | ✅ 已交付（`lishengdan_C2G_tech_report.pdf`，NeurIPS workshop 双栏格式） |
| **BPB < 1.085** | ❌ **未达成，且本提交不声称达成** |

**我的判断：** 完成 Level 4 的**形式要求**不等于达成 Level 4。官方门槛是一个客观数字，而我没有 8×H100 去把它跑出来。所以我按 Level 4 的格式交付，但在所有地方标注目标是 Level 2 / 条件 Level 3。**把"我写了一份 6 页报告"说成"我达到了 Level 4"，是这个挑战里最容易做、也最没有价值的一种造假。**
