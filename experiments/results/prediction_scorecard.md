# 预注册预测记分卡 / Pre-registered prediction scorecard

判定规则（写死在 `analyze_results.py` 里，不靠人工解释）：

```
m  = 配对差的均值        hw = 该配对差的 95% t 区间半宽
n < 2                  -> UNRESOLVED（无法估计方差）
|m| <= hw              -> UNRESOLVED（功效不足，与 0 无法区分）
sign(m) 与预测方向一致  -> SUPPORTED
否则                    -> NOT SUPPORTED（真·被证伪）
```

基线 `A0_baseline` 的跨 seed 标准差：**σ = 0.0068 BPB**（3 个种子）。

| id | 预测 | 阈值 | n | 实测配对Δ | 半宽 hw | 判定 |
|---|---|---|---|---|---|---|
| P1 | SP4096/SP8192 相对 SP1024 降低本地 BPB | Δ ≤ −0.02 | 3 | -0.2248 | ±0.0043 | SUPPORTED |
| P2 | QK-Gain 5.25 相对默认 1.5 有正收益 | Δ ≤ −0.005 | 2 | +0.0055 | ±0.0741 | UNRESOLVED (UNDERPOWERED) |
| P3 | 并行残差（后半段层）有正收益 | Δ ≤ −0.005 | 2 | +0.0006 | ±0.0064 | UNRESOLVED (UNDERPOWERED) |
| P4 | 深度递归（循环中间层）有正收益 | Δ ≤ −0.005 | 2 | +0.0148 | ±0.0648 | UNRESOLVED (UNDERPOWERED) |
| P5 | 朴素 EMA 降低 pre-quant BPB 但放大 quantization tax | Δtax > 0 | 2 | -0.0004 | ±0.0020 | UNRESOLVED (UNDERPOWERED) |
| P6 | 训练末期 QAT 把 quantization tax 压低 50% 以上 | reduction ≥ 50% | 2 | +45.1% | -- | NOT SUPPORTED |
| P7 | (原创) 量化感知 EMA (QA-EMA) 优于朴素 EMA | Δ ≤ −0.002 | 3 | +0.0096 | ±0.0008 | NOT SUPPORTED |
| P8 | 同等 6-bit 下 GPTQ 优于 RTN+SDClip | Δ ≤ −0.002 | 2 | -0.0048 | ±0.0046 | SUPPORTED |

## 逐条注记

- **P1** — evaluated on A2_sp8192; A1_sp4096: -0.1536, A2_sp8192: -0.2248。判定：`SUPPORTED`
- **P2** — arm A3_qkgain vs A0_baseline。判定：`UNRESOLVED (UNDERPOWERED)`
- **P3** — arm A4_parresid vs A0_baseline。判定：`UNRESOLVED (UNDERPOWERED)`
- **P4** — arm A5_recurrence vs A0_baseline。判定：`UNRESOLVED (UNDERPOWERED)`
- **P5** — per-seed (tax[A6] − tax[A0])。判定：`UNRESOLVED (UNDERPOWERED)`
- **P6** — tax +0.0260 → +0.0143 (n=2 seeds)。判定：`NOT SUPPORTED`
- **P7** — head-to-head B2 − B1, same config except EMA_QUANT_AWARE。判定：`NOT SUPPORTED`
- **P8** — head-to-head Q4 − Q2 (both int6+SDClip); combo cross-check B3 − B1: +0.0158。判定：`SUPPORTED`

## 逐臂分辨率

| 臂 | n | 配对Δ 均值 | 配对Δ 标准差 | 95% 半宽 |
|---|---|---|---|---|
| A1_sp4096 | 2 | -0.1536 | 0.0087 | ±0.0785 |
| A2_sp8192 | 3 | -0.2248 | 0.0017 | ±0.0043 |
| A3_qkgain | 2 | +0.0055 | 0.0083 | ±0.0741 |
| A4_parresid | 2 | +0.0006 | 0.0007 | ±0.0064 |
| A5_recurrence | 2 | +0.0148 | 0.0072 | ±0.0648 |
| A6_ema | 2 | +0.0732 | 0.0041 | ±0.0371 |
| B1_combo | 3 | -0.1279 | 0.0054 | ±0.0134 |
| B2_combo_qaema | 3 | -0.1184 | 0.0053 | ±0.0133 |
| B3_combo_gptqoff | 2 | -0.1153 | 0.0013 | ±0.0113 |
| Q1_int6_pct | 2 | +0.0000 | 0.0000 | ±0.0000 |
| Q2_int6_sd | 2 | +0.0243 | 0.0019 | ±0.0167 |
| Q3_int8_sd | 2 | +0.0102 | 0.0010 | ±0.0088 |
| Q4_int6_sd_gptq | 2 | +0.0195 | 0.0024 | ±0.0212 |
| Q5_int6_sd_qat | 2 | +0.0333 | 0.0058 | ±0.0524 |

**怎么读这张表。** `UNRESOLVED (UNDERPOWERED)` 的含义是：这个效应在本装置上**与 0 无法区分**，无论点估计落在零的哪一边，都不构成证据。只有 `SUPPORTED` / `NOT SUPPORTED` 两种判定才承载证据。把“功效不足”读成“不成立”，是这类小样本消融里最常见的自欺。

**为什么 n=2 的臂半宽特别大（±0.07 而不是 ±0.004）？** 这不是算错。95% t 区间在 df=1 时的临界值是 **12.706**，在 df=2 时降到 4.303。也就是说，**两个种子在统计上几乎等于没有重复**：配对差只要不是完全一致，区间就会被这个 12.7 放大到 6 倍以上。这正是本次复盘想传达的核心事实之一——本装置只有在效应远远大于 ~0.01 BPB 时才具备判决力。

