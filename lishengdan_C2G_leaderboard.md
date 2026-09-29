# lishengdan_C2G_leaderboard

## ⚠️ 两条数字不能混着读

| 类别 | 来源 | 可否与另一类直接比较 |
|---|---|---|
| **官方榜单数字** | 官方 `README.md` 榜单表与 `records/` 内的 `submission.json`，8×H100、10 分钟、16MB、FineWeb val | ❌ |
| **本提交本地数字** | CPU 22 线程、代理语料（Simple English Wikipedia）、约 0.9–1.8M 参数 | ❌ |

**它们唯一的共同点是 BPB 这个指标的定义。** 数值区间完全不重叠：官方在 1.08–1.23，本提交在 2.0 以上，因为本地模型欠训练约 3 个数量级。把两者放进同一个表里比大小是没有意义的，本文件把它们分表列出，并说明每一行的出处。

## 1. 官方榜单（公开记录，8×H100 / 600s / 16MB / FineWeb val）

| 方案 | val_bpb | 出处 |
|---|---|---|
| Naive Baseline（官方） | 1.2244 | 官方 README 榜单表 + records/.../2026-03-17_NaiveBaseline/submission.json |
| 11L MLP3x + Int6 QAT（官方） | 1.1502 | 官方榜单表 |
| 11L XSA4 + EMA + Int6 MLP3x（官方） | 1.1271 | 官方榜单表 |
| 11L EMA + GPTQ-lite + warmdown3500（官方） | 1.1228 | 官方榜单表 |
| 11L AR Self-Gen GPTQ + XSA（官方） | 1.1147 | 官方榜单表 |
| Parallel Residuals + Mini Depth Recurrence（官方） | 1.1063 | 官方榜单表 |
| MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6（官方） | 1.0912 | 官方榜单表 |
| SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R（官方） | 1.0897 | 官方榜单表 |
| SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip（官方） | 1.0856 | 官方榜单表 |
| SP8192 + Parallel Residuals + Hessian SDClip（官方） | 1.0835 | 官方榜单表 |
| **SOTA**：SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT（官方） | 1.0810 | 官方榜单表 + records/.../2026-04-09_.../submission.json |

**本提交在官方赛道上没有成绩。** 我没有 8×H100，也没有 FineWeb。任何把本地数字填进上表的做法都是造假，因此上表只列官方记录。

## 2. 本提交本地代理语料结果

| 臂 | final val_bpb | deployed pre-quant | artifact |
|---|---|---|---|
| A0_baseline | 2.0817 ± 0.0055 | 2.0801 | 640,141 B |
| A1_sp4096 | 1.9293 ± 0.0003 | 1.9286 | 1,015,479 B |
| A2_sp8192 | 1.8569 ± 0.0041 | 1.8561 | 1,521,621 B |
| A3_qkgain | 2.0884 ± 0.0006 | 2.0862 | 639,799 B |
| A4_parresid | 2.0835 ± 0.0060 | 2.0815 | 640,817 B |
| A5_recurrence | 2.0977 ± 0.0014 | 2.0961 | 640,036 B |
| A6_ema | 2.1561 ± 0.0094 | 2.1548 | 640,698 B |
| Q1_int6_pct | 2.0829 ± 0.0065 | 2.0812 | 640,141 B |
| Q2_int6_sd | 2.1072 ± 0.0052 | 2.0812 | 454,999 B |
| Q3_int8_sd | 2.0931 ± 0.0058 | 2.0812 | 603,561 B |
| Q4_int6_sd_gptq | 2.1024 ± 0.0048 | 2.0812 | 455,662 B |
| Q5_int6_sd_qat | 2.1162 ± 0.0023 | 2.1019 | 452,865 B |
| B1_combo | 1.9538 ± 0.0060 | 1.9433 | 1,025,623 B |
| B2_combo_qaema | 1.9634 ± 0.0061 | 1.9485 | 1,024,222 B |
| B3_combo_gptqoff | 1.9676 ± 0.0074 | 1.9473 | 1,022,757 B |

评估语料：Simple English Wikipedia（CC BY-SA 4.0），由 `dumps.wikimedia.org` 的 multistream BZ2 dump 前缀抽取得到，**确定性可复现**。val 集固定，所有臂看到完全相同的验证 token。

## 3. 本提交相对官方的差距（诚实的算术）

| 维度 | 官方 | 本提交 | 差距 |
|---|---|---|---|
| 硬件 | 8×H100 SXM | 22 线程 CPU | ≈ 无 GPU |
| 参数量 | 30–100M | 0.9–1.8M | ≈ 30–50× |
| 训练 token | ≈ 7×10⁹ | 1.64×10⁶ | ≈ 4000× |
| 数据 | FineWeb（网页） | Simple English Wikipedia（百科） | 域不同 |
| 16MB 约束 | 紧约束 | 完全不触发 | 无法观察 |

**结论：** 这份提交的价值在方法论与可复现产物，不在分数。我在 `lishengdan_C2G_方案设计.md` §7 与 `lishengdan_C2G_AAR.md` 里都写明了这一点。

