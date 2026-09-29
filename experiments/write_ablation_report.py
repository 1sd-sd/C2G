"""
Generate `lishengdan_C2G_ablation.md` and `lishengdan_C2G_leaderboard.md` from
experiments/results/summary.json, so the documents can never drift from the data.

Usage:
    python write_ablation_report.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"

ARM_META = {
    # A0's description was corrected after the run: the trainer's defaults are int6 +
    # percentile, not the official scaffold's int8 + percentile. The CONFIG and the
    # measurements are unchanged; only the label was wrong.
    "A0_baseline": ("A0", "baseline：SP1024 + 默认导出（int6 + percentile clip）", "—"),
    "A1_sp4096": ("A1", "词表 4096", "P1"),
    "A2_sp8192": ("A2", "词表 8192", "P1"),
    "A3_qkgain": ("A3", "QK-Gain 5.25（默认 1.5）", "P2"),
    "A4_parresid": ("A4", "并行残差（调度位置 ≥4）", "P3"),
    "A5_recurrence": ("A5", "深度递归（循环 2–4 层）", "P4"),
    "A6_ema": ("A6", "朴素 EMA 0.996", "P5"),
    "Q1_int6_pct": ("Q1", "int6 + percentile —— 与 A0 同配置，实为**确定性对照**", "—"),
    "Q2_int6_sd": ("Q2", "int6 + SDClip", "—"),
    "Q3_int8_sd": ("Q3", "int8 + SDClip", "—"),
    "Q4_int6_sd_gptq": ("Q4", "int6 + SDClip + GPTQ", "P8"),
    "Q5_int6_sd_qat": ("Q5", "int6 + SDClip + QAT（warmdown 对齐）", "P6"),
    "B1_combo": ("B1", "全栈（无 QA-EMA）", "—"),
    "B2_combo_qaema": ("B2", "全栈 + QA-EMA（本提交原创）", "P7"),
    "B3_combo_gptqoff": ("B3", "全栈 + QA-EMA，导出退回 RTN", "P8"),
}

# The (8, percentile) cell of the intended 2x2 is absent, and Q1 duplicates A0. Both facts
# are consequences of the same mislabelled default, so they are stated once, here.
QUANT_BLOCK_NOTE = (
    "**量化块的一处事后更正。** 规划时的 2×2 是「int6/int8 × percentile/SDClip」。"
    "但训练脚本的默认值实际是 `QUANT_BITS=6` / `QUANT_CLIP=percentile`，而 `A0` 走的就是默认值 —— "
    "也就是说 `A0` 落在 (6, percentile) 格，`Q1` 因此与 `A0` **配置完全相同**。\n"
    "\n"
    "这带来两个后果，都不掩饰：\n"
    "1. **(8, percentile) 这一格缺失**，所以「clip 规则」的对比只在 int6 上做过一次。\n"
    "2. `Q1` 不再是独立的消融臂，而是变成了**确定性对照**：同配置、同种子，"
    "两次运行的 `final_bpb`、`final_loss`、`pre_quant_bpb`、`artifact_bytes`、"
    "`n_params`、`steps_completed`、`train_tokens_seen` **七个字段全部逐位相同**。"
    "这是本包里最直接的确定性证据，所以这一臂被保留而不是删除。\n"
    "\n"
    "**为什么不重跑：** 所有组件对比都是同 seed 配对，量化配置在臂内是常量，"
    "所以`A1`–`A6` 相对 `A0` 的 Δ 不受影响。重跑基线会作废全部 16 次已完成的参照，"
    "代价大于收益。正确做法是保留数据、改正标签 —— 本文件与 `方案设计.md` 已同步更正。"
)


def load():
    s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    runs = {}
    for p in sorted(RES.glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs[r["run_id"]] = r
    return s, runs


def paired_deltas(runs, arm, ref="A0_baseline"):
    a = {r["config"]["seed"]: r["final_bpb"] for k, r in runs.items() if k.startswith(arm + "_seed")}
    b = {r["config"]["seed"]: r["final_bpb"] for k, r in runs.items() if k.startswith(ref + "_seed")}
    shared = sorted(set(a) & set(b))
    return {s: a[s] - b[s] for s in shared}


def write_ablation(summary, runs) -> None:
    L = []
    L.append("# lishengdan_C2G_ablation")
    L.append("")
    L.append("> 本文件由 `experiments/write_ablation_report.py` 从 "
             "`experiments/results/summary.json` 自动生成，数字不会与数据漂移。")
    L.append("")
    L.append("## 0. 这个消融能证明什么、不能证明什么")
    L.append("")
    L.append("**能：** 在同一份训练脚本、同一套 BASE 超参、同一条数据流下，各组件对"
             "**部署后 BPB** 的相对排序。")
    L.append("")
    L.append("**不能：** 收益的绝对幅度、以及在官方 FineWeb / 8×H100 设定下这些排序是否保持。"
             "本地模型约 0.9–1.8M 参数、训练 1.64M token，比官方设定小约 3 个数量级（算力）"
             "与 1 个数量级（参数量）。")
    L.append("")
    first = next(iter(summary.values())) if summary else None
    if first:
        L.append(f"**本次运行规模：** {first['n_params']:,} 参数，"
                 f"{first['steps']} 步 × 4096 token = {first['steps']*4096/1e6:.2f}M token。")
    L.append("")
    L.append("## 1. 主表")
    L.append("")
    L.append("`final val_bpb` = 量化 + 压缩 + roundtrip 之后的 BPB（成绩单口径）；"
             "`deployed pre-quant` = **将要被导出的那份权重**在量化前的 BPB（EMA 生效时即 EMA 权重，"
             "不是最后一步的原始权重——否则量化税会把 EMA 的收益算进去）；"
             "`quant tax` = post − deployed pre，越小说明导出损失越小；"
             "`paired Δ` = 同 seed 下 `bpb(arm) − bpb(A0)` 的均值。")
    L.append("")
    L.append("| 臂 | 变量 | 预测 | seeds | final val_bpb | std | deployed pre-quant | quant tax | paired Δ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for arm, (tag, desc, pred) in ARM_META.items():
        s = summary.get(arm)
        if not s:
            L.append(f"| **{tag}** | {desc} | {pred} | – | *未运行* | | | | |")
            continue
        L.append(f"| **{tag}** | {desc} | {pred} | {s['n_seeds']} | "
                 f"**{s['final_bpb_mean']:.4f}** | {s['final_bpb_std']:.4f} | "
                 f"{s['deployed_pre_quant_bpb_mean']:.4f} | {s['quant_tax']:+.4f} | "
                 f"{s['paired_delta_mean']:+.4f} |")
    L.append("")
    L.append(QUANT_BLOCK_NOTE)
    L.append("")

    L.append("## 2. 逐 seed 配对明细")
    L.append("")
    L.append("配对比较的意义：两条臂共享数据顺序与初始化分布，`Δ` 相减时 seed 方差大部分抵消。"
             "下表的每一列是一个独立 seed。")
    L.append("")
    seeds = sorted({r["config"]["seed"] for r in runs.values()})
    L.append("| 臂 | " + " | ".join(f"seed{s}" for s in seeds) + " | 均值 |")
    L.append("|---|" + "---|" * (len(seeds) + 1))
    for arm, (tag, _, _) in ARM_META.items():
        if arm == "A0_baseline":
            continue
        d = paired_deltas(runs, arm)
        if not d:
            continue
        cells = " | ".join(f"{d[s]:+.4f}" if s in d else "–" for s in seeds)
        L.append(f"| **{tag}** | {cells} | {statistics.mean(d.values()):+.4f} |")
    L.append("")

    L.append("## 3. 注记")
    L.append("")
    L.append("- **`std` 是“每个臂自身的跨 seed 标准差”，不是配对差的标准差。** 两者含义不同："
             "前者告诉你这个臂在不同随机种子下有多不稳，后者才是在问“这个组件的效果是否大于噪声”。"
             "本报告只提供配对 `Δ` 与臂内 `std` 两个数，让读者自己判断，不代做显著性声明。")
    L.append("- **量化税是全表最有信息量的列。** 它把“训练得好不好”和“导出损失多少”分开。"
             "一个组件如果把 `pre-quant` 压下去但 `quant tax` 涨上来，它对成绩单是没有贡献的。")
    L.append("- **`B1` vs `B2`** 是本提交唯一原创组件的严格配对消融：唯一变量是 "
             "`EMA_QUANT_AWARE`，其余（词表、QK-Gain、并行残差、深度递归、EMA decay、"
             "QAT、GPTQ、压缩）完全一致。")
    L.append("")
    (ROOT / "lishengdan_C2G_ablation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote lishengdan_C2G_ablation.md")


OFFICIAL = [
    ("Naive Baseline（官方）", 1.2244, "官方 README 榜单表 + records/.../2026-03-17_NaiveBaseline/submission.json"),
    ("11L MLP3x + Int6 QAT（官方）", 1.1502, "官方榜单表"),
    ("11L XSA4 + EMA + Int6 MLP3x（官方）", 1.1271, "官方榜单表"),
    ("11L EMA + GPTQ-lite + warmdown3500（官方）", 1.1228, "官方榜单表"),
    ("11L AR Self-Gen GPTQ + XSA（官方）", 1.1147, "官方榜单表"),
    ("Parallel Residuals + Mini Depth Recurrence（官方）", 1.1063, "官方榜单表"),
    ("MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6（官方）", 1.0912, "官方榜单表"),
    ("SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R（官方）", 1.0897, "官方榜单表"),
    ("SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip（官方）", 1.0856, "官方榜单表"),
    ("SP8192 + Parallel Residuals + Hessian SDClip（官方）", 1.0835, "官方榜单表"),
    ("**SOTA**：SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT（官方）",
     1.0810, "官方榜单表 + records/.../2026-04-09_.../submission.json"),
]


def write_leaderboard(summary, runs) -> None:
    L = []
    L.append("# lishengdan_C2G_leaderboard")
    L.append("")
    L.append("## ⚠️ 两条数字不能混着读")
    L.append("")
    L.append("| 类别 | 来源 | 可否与另一类直接比较 |")
    L.append("|---|---|---|")
    L.append("| **官方榜单数字** | 官方 `README.md` 榜单表与 `records/` 内的 `submission.json`，"
             "8×H100、10 分钟、16MB、FineWeb val | ❌ |")
    L.append("| **本提交本地数字** | CPU 22 线程、代理语料（Simple English Wikipedia）、"
             "约 0.9–1.8M 参数 | ❌ |")
    L.append("")
    L.append("**它们唯一的共同点是 BPB 这个指标的定义。** 数值区间完全不重叠：官方在 1.08–1.23，"
             "本提交在 2.0 以上，因为本地模型欠训练约 3 个数量级。把两者放进同一个表里比大小"
             "是没有意义的，本文件把它们分表列出，并说明每一行的出处。")
    L.append("")
    L.append("## 1. 官方榜单（公开记录，8×H100 / 600s / 16MB / FineWeb val）")
    L.append("")
    L.append("| 方案 | val_bpb | 出处 |")
    L.append("|---|---|---|")
    for name, bpb, src in OFFICIAL:
        L.append(f"| {name} | {bpb:.4f} | {src} |")
    L.append("")
    L.append("**本提交在官方赛道上没有成绩。** 我没有 8×H100，也没有 FineWeb。"
             "任何把本地数字填进上表的做法都是造假，因此上表只列官方记录。")
    L.append("")
    L.append("## 2. 本提交本地代理语料结果")
    L.append("")
    L.append("| 臂 | final val_bpb | deployed pre-quant | artifact |")
    L.append("|---|---|---|---|")
    for arm, s in summary.items():
        L.append(f"| {arm} | {s['final_bpb_mean']:.4f} ± {s['final_bpb_std']:.4f} | "
                 f"{s['deployed_pre_quant_bpb_mean']:.4f} | {s['artifact_bytes']:,} B |")
    L.append("")
    L.append("评估语料：Simple English Wikipedia（CC BY-SA 4.0），"
             "由 `dumps.wikimedia.org` 的 multistream BZ2 dump 前缀抽取得到，**确定性可复现**。"
             "val 集固定，所有臂看到完全相同的验证 token。")
    L.append("")
    L.append("## 3. 本提交相对官方的差距（诚实的算术）")
    L.append("")
    L.append("| 维度 | 官方 | 本提交 | 差距 |")
    L.append("|---|---|---|---|")
    L.append("| 硬件 | 8×H100 SXM | 22 线程 CPU | ≈ 无 GPU |")
    L.append("| 参数量 | 30–100M | 0.9–1.8M | ≈ 30–50× |")
    L.append("| 训练 token | ≈ 7×10⁹ | 1.64×10⁶ | ≈ 4000× |")
    L.append("| 数据 | FineWeb（网页） | Simple English Wikipedia（百科） | 域不同 |")
    L.append("| 16MB 约束 | 紧约束 | 完全不触发 | 无法观察 |")
    L.append("")
    L.append("**结论：** 这份提交的价值在方法论与可复现产物，不在分数。"
             "我在 `lishengdan_C2G_方案设计.md` §7 与 `lishengdan_C2G_AAR.md` 里都写明了这一点。")
    L.append("")
    (ROOT / "lishengdan_C2G_leaderboard.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote lishengdan_C2G_leaderboard.md")


if __name__ == "__main__":
    s, r = load()
    write_ablation(s, r)
    write_leaderboard(s, r)
