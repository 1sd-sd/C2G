# lishengdan_C2G_拿来说明

> 挑战规则要求的三句话：**拿了什么、改了什么、为什么改。**
> 本文档逐项列出本提交的代码来源、改造点与改造理由。凡是"站在巨人肩膀上"的部分，我都不掩饰来源；凡是这个提交自己贡献的部分，我也标清楚它是否被本地实验验证过。

---

## 1. 起点：我拿了什么

### 1.1 OpenAI Parameter Golf 官方脚手架（`train_gpt.py`，1126 行）

这是本提交的主干来源。逐块列出我**原样保留**的部分：

| 模块 | 我拿了什么 | 为什么直接拿 |
|---|---|---|
| `Hyperparameters` | 全环境变量配置的风格 | 与官方复现脚本对齐，评审可以直接套用官方的运行方式 |
| `Muon` + `zeropower_via_newtonschulz5` | 完整实现（含 momentum warmup、`max(1, rows/cols)^0.5` 尺度修正、DDP all-reduce 分片） | 这是 Keller Jordan 的成果，重写只会引入 bug。参数分组（矩阵走 Muon / embedding 与标量走 Adam）也照搬 |
| `build_sentencepiece_luts` | 逐字保留 | **BPB 是 tokenizer 无关指标，这段 LUT 逻辑就是指标的定义本身。** 改一个字节都可能让分数不可信，规则里明确写了"改 tokenizer 的提交会被更严格审查" |
| `eval_val` | 逐字保留（仅把 `inference_mode` 换成 `no_grad`，见 §2.1） | 同上，这是成绩的计算口径 |
| `load_data_shard` / `TokenStream` / `DistributedTokenLoader` | 逐字保留 | 复用了官方 `.bin` 分片格式（256×int32 头 + uint16 token），使本地语料与官方语料可以互换 |
| `quantize_state_dict_int8` / `dequantize_state_dict_int8` | 拿走了**结构**，换了量化数值方案 | 见 §2.3 |
| `CausalSelfAttention` / `Rotary` / `apply_rotary_emb` | 逐字保留 | GQA + QK-RMSNorm + RoPE 是 baseline 的定义 |
| `MLP`（relu² 结构）、`CastedLinear`、`Block` 的 `resid_mix`/`attn_scale`/`mlp_scale` | 逐字保留 | 同上 |
| U-Net skip 连接 + `skip_weights` | 保留机制，扩展到"按调度位置"索引 | 见 §2.2 |
| 训练主循环骨架（warmdown 的 `lr_mul`、wallclock cap、peak memory 统计、`final_int8_zlib_roundtrip` 双重评估） | 保留 | 这套 I/O 契约是官方评审要看的格式 |

### 1.2 modded-nanogpt（Keller Jordan）

官方脚手架本身已经在 `THIRD_PARTY_NOTICES.md` 里声明改编自 `modded-nanogpt`。我顺着这条线确认了以下组件的真正源头，并在报告里按原始出处引用而不是"从 parameter-golf 拿的"：

- **Muon 优化器** —— Keller Jordan, [Muon blog](https://kellerjordan.github.io/posts/muon/)，2024-12。Newton-Schulz 正交化的 `(3.4445, -4.7750, 2.0315)` 系数出自此处。
- **relu² MLP、zero-init 输出投影、U-Net skip、logit softcap、RMSNorm-on-QK** —— modded-nanogpt 的历次 speedrun 提交。
- **QK-Gain**（可学习 per-head query 缩放）—— 官方脚手架里已有该参数（默认 `QK_GAIN_INIT=1.5`），但其单调收益是在榜单上被 `2026-04-06_SP8192_QK5_LegalTTT` 等提交确立的。

### 1.3 官方榜单谱系（`records/track_10min_16mb/`，33 个记录目录）

我没有下载任何外部权重或代码，只读了仓库里公开的记录 README 与 `submission.json`，用于确定**哪些方向已经被公开验证过**。被我在本提交中重新实现的组件，逐条标源：

| 组件 | 来源 | 我读到的证据 |
|---|---|---|
| SP8192 tokenizer | clarkkev (PR #1394)，经 dexhunter 发扬 | 榜单从 SP1024 的 1.1586 分界到 SP8192 谱系的 1.0810 |
| Parallel Residuals（GPT-J 风格并行残差） | Robby955 (PR #1412)、msisovic (PR #1204) | `2026-03-31_ParallelResiduals_MiniDepthRecurrence` 1.1063 |
| Depth Recurrence（循环中间层） | dexhunter (PR #1331, #1437) | `2026-04-09` SOTA 用 11 物理层展开出 17 虚拟层 |
| GPTQ + SDClip 量化 | clarkkev (PR #1394) | `clip = k · std(row)`，int6 矩阵 / int8 embedding |
| 超参（WD、EMA 0.9965、warmdown 72%） | X-Abhishek-X (PR #1445) | `2026-04-09` 记录的 Tuned Hyperparameters 一节 |
| 自由评测（滑窗 / TTT） | abaybektursun (PR #549)、dexhunter (PR #1413) | 我**没有**实现 TTT（见 §3.2 的取舍说明） |

---

## 2. 我改了什么

### 2.1 设备无关化（为了能在没有 GPU 的环境里做筛选）

官方脚本第一行 CUDA 检查就是 `raise RuntimeError("CUDA is required")`。这个提交把它换成一个统一的设备抽象：

- `DEVICE=auto|cuda|cpu`；CPU 路径下 `grad_accum_steps=1`、关闭 `torch.compile`、关闭 `autocast`、Adam 的 `fused=True` 关闭、Muon 的 Newton-Schulz 迭代改成 fp32（CPU 上 bf16 matmul 反而更慢）。
- `nvidia-smi` 调用被包进 `if use_cuda`。
- **`eval_val` 里的 `torch.inference_mode()` → `torch.no_grad()`。** 这条改动是调试中被迫做出的：`Rotary` 模块缓存 cos/sin 表，在 `inference_mode` 里创建的张量是"inference tensor"，一旦被后续 autograd 图引用就会抛 `Inference tensors cannot be saved for backward`。官方脚本不会踩到，因为它的 `WARMUP_STEPS=20` 会先在正常模式下预热并建好缓存；我把 `WARMUP_STEPS` 默认设为 0（CPU 上预热太贵），于是第一次验证就崩了。**这是一个"改动引发的连锁 bug"，我在 AAR 里如实记录。**

**为什么改：** 我拿不到 8×H100。如果不做设备无关化，我就只能"提交一份从未运行过的脚本"——那等于零复现性。设备无关化让我能在真实数据上跑完整条流水线（训练 → 量化 → 压缩 → roundtrip 评估），代价是本地只能跑小模型。

### 2.2 组件化：所有改进都由环境变量开关

`PARALLEL_RESID_FROM`、`LAYER_SCHEDULE`、`QK_GAIN_INIT`、`EMA_DECAY`/`EMA_QUANT_AWARE`、`QAT_START_FRAC`、`QUANT_METHOD`/`QUANT_BITS`/`SDCLIP_K`、`COMPRESS` 全部是环境变量。

**为什么改：** 这是本提交方法论的核心。如果消融实验的每个臂用不同的代码副本，那么臂与臂之间的差异就同时包含了"代码差异"和"组件差异"，实验就不成立。**同一份脚本 + 不同环境变量** 保证了唯一变量原则。

具体实现上：

- **Parallel Residuals** —— `Block` 增加 `parallel` 标志。`False` 时是官方的串行写法（attn 输出后接 MLP）；`True` 时（GPT-J 风格）attention 和 MLP 都从同一个 pre-residual 激活读取，两条输出相加回主干。
- **Depth Recurrence** —— 把"物理层列表"换成"执行调度列表" `LAYER_SCHEDULE`。例如 6 个物理层配 `0,1,2,3,4,5,2,3,4,5` 就是让第 2–4 层各执行两次，共 10 个执行位置。U-Net skip 的切分点从"物理层的一半"改成"**调度位置的一半**"，因为 skip 是张量层面的配对，必须按执行顺序而不是物理编号来配。
- **QAT** —— `CastedLinear.forward` 在 `self.training and self.qat_bits > 0` 时对权重做 fake-quant（STE），量化格点与最终导出用的格点一致。`QAT_START_FRAC<0` 时表示"从 warmdown 起点开始"。
- **GPTQ** —— 通过 `register_forward_pre_hook` 在 calibration 前向里累积每个 `CastedLinear` 的 `XᵀX`，再逐层做列序 GPTQ（含阻尼、Cholesky 逆、128 列分块、误差回灌）。**校准数据只从训练分片取，不碰验证集** —— 因为规则明确禁止在训练/压缩阶段接触验证数据。
- **压缩** —— `zlib` 或 `brotli-11` 可选。

### 2.3 量化数值方案：从 percentile-clip 换成 SDClip

官方 baseline 的 int8 导出让 int8 的裁剪阈值取**每行绝对值分布的 99.99984 百分位**。本提交换成 SDClip：

```
clip(row) = k · std(row)          k = SDCLIP_K（默认 12.85）
scale(row) = clip(row) / (2^(bits-1) - 1)
```

**为什么改：** 百分位裁剪的阈值由**极值**驱动，而极值在权重里恰恰是最病态、最不重要的部分；`k·std` 由**分布的尺度**驱动，对离群点更稳健。这条来自 clarkkev (PR #1394) 的公开记录，我保留了它的率失真解释。同时把 scheme 从"全是 int8"改成 **int6 矩阵 + int8 embedding** 的混合精度 —— 因为 embedding 是 tied 的，同时充当输入查表和输出投影，量化误差会同时污染两条路径，值得多给 2 bit。

**为什么保留 percentile 的实现路径：** 见 §3.2。`QUANT_METHOD` 与 `QUANT_CLIP` 是**两个独立的旋钮**：前者选**算法**（`rtn` 就近取整 / `gptq` 顺序量化带误差反馈），后者选**裁剪规则**（`percentile` / `sdclip`）。我把两条算法路径都实现了，因为**"GPTQ 比 RTN 好多少"本身是一个需要测量的消融项**，而不是一个可以假设的结论。

### 2.4 本提交的原创部分

**(a) QA-EMA（量化感知权重平均）。** 朴素 EMA 在连续空间里做 `shadow ← decay·shadow + (1−decay)·θ`，然后部署到离散格点上。QA-EMA 改成先把当前权重按**部署用的格点**量化再反量化，然后才做平均：

```
src = dequant(quant(θ, bits, k))
shadow ← decay · shadow + (1 − decay) · src
```

直觉：朴素 EMA 的平均结果是一个"浮点意义上的好点"，它不一定是"格点意义上的好点"；QA-EMA 让被平均的量始终落在格点上，因此平均结果也处在格点邻域内。

**实现代价：** 每步多一次逐元素量化，约 2M 参数在 CPU 上是毫秒级，可忽略。

**诚实标注：** 这个想法**不是**我从榜单上抄的，但它也**没有**已被公开验证。我用配对 seed 实验（`B1_combo` vs `B2_combo_qaema`）去检验它，并且预先在 `lishengdan_C2G_方案草案.md` 里注册了预测 P7。**如果实验证伪它，我撤回这条声明。**

**(b) warmdown 对齐的 QAT 起点。** 官方方案里 QAT 起点是一个固定步数比例。本提交支持 `QAT_START_FRAC=-1`，语义是"**从 warmdown 的起点开始 QAT**"。理由是：QAT 的作用是让模型适应量化格点，而适应需要"模型还在移动"才有意义；warmdown 之前的 LR 太大，量化格点会被立刻冲掉。把 QAT 起点和 warmdown 起点绑定，让"适应量化"和"收敛"发生在同一段时间里。

**(c) 一份可审计的实验台。** 代理语料构造（multistream bz2 前缀抽取）、tokenizer 训练、分片打包、实验矩阵驱动、独立复现评估，全部脚本化。这不是算法贡献，但它让整个提交的每个数字都能被重跑出来。

---

## 3. 我**没有**拿、也**没有**改的部分（以及为什么）

### 3.1 数据集

官方用 Hugging Face 上的 `fineweb10B_sp1024` / `sp8192` 导出。**在本执行环境里 huggingface.co 与 hf-mirror.com 均不可达**（HTTP 000），所以本地筛选用的是自建代理语料（Simple English Wikipedia，CC BY-SA 4.0，经 `dumps.wikimedia.org` 的 multistream BZ2 前缀抽取得到）。

**这带来的限制我必须说清楚：**

1. 本地测得的绝对 BPB **不等于**官方 BPB，两者不能直接比。
2. 域名分布不同（百科 vs 网页），tokenizer 的相对收益可能会被高估或低估。
3. 因此本提交把本地数字定位为**排序证据**（哪一组件有效、相对大小），而不是**成绩证据**。凡是引用 1.22 / 1.08 这类数字的地方，都明确标注为"官方榜单数字，非本地测量"。

### 3.2 我明确跳过的东西

| 跳过的组件 | 为什么跳过 |
|---|---|
| **Test-Time Training (TTT)** | 它需要完整 10 分钟训练 + 10 分钟评测预算才谈得上收益；在小模型本地筛选阶段，TTT 的开销会淹没训练本身。而且 TTT 的合法性条件（score-first、single-pass）极易写错，**写错的 TTT 会给出一个看起来很好但无效的分数**。宁可不用。 |
| **FlashAttention-3 / 自定义 CUDA kernel** | 本环境无 NVIDIA GPU。写一个无法测试的 kernel 是纯粹的负债。 |
| **EnCodec/语义 tokenizer、H-Net 等未上榜方向** | 属于探索性方向，在只有排序证据的条件下无法判断。 |
| **1-bit / ternary 量化** | 榜单上 ternary 的最好成绩 1.1570 还不如 int6 路线，且它偏"非记录提交"赛道。 |
| **从零写 Transformer** | 规则明令禁止，且没有意义。 |

### 3.3 关于"官方榜单数字"的使用纪律

本提交中出现的 `1.2244`（Naive Baseline）、`1.1586`（SP1024 谱系最好成绩）、`1.0810`（当前 SOTA）等数字，**全部来自 `README.md` 榜单表格与 `records/` 下的 `submission.json`**，我一个字都没有改写成"我自己跑出来的"。本地实验产生的数字一律带 `local proxy` 前缀或出现在 `experiments/results/` 里。这条纪律写在这里，是为了让评审能在 30 秒内判断任意一个数字的出处。

---

## 4. 一页纸总结

| 问题 | 答案 |
|---|---|
| **拿了什么** | OpenAI `train_gpt.py` 的全部脚手架契约（Muon、BPB LUT、分片格式、导出格式、训练循环 I/O）；modded-nanogpt 的 relu²/zero-init/U-Net skip/QK-Gain；榜单公开的 SP8192、并行残差、深度递归、GPTQ-SDClip、超参配方 |
| **改了什么** | 设备无关化；全组件环境变量化（消融可同代码复现）；**新增** SDClip 裁剪路径（实测后**出货仍用 percentile**，见 §3.2 与 `lishengdan_C2G_AAR.md` §4.9）；全 int8 → int6 矩阵 + int8 embedding；新增 QA-EMA 与 warmdown 对齐 QAT；新增 GPTQ 导出路径与独立复现评估器；自建代理语料流水线 |
| **为什么改** | 因为我要**测量**每一个组件的贡献，而不是**引用**它们的贡献。同代码多变量开关 + 配对 seed 是唯一能让"组件贡献"可归因的做法；SDClip/混合精度是因为量化是这个赛道真正的约束（16MB），而不是附属步骤；设备无关化是因为没有它这个提交就一次都没运行过 |
