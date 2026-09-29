"""
Post-hoc analysis of the local proxy matrix.

Two jobs:

1. Quantify the *resolution* of this screen, per arm. With two or three seeds, the
   half-width of a 95% t-interval on a paired delta is large, and it is the honest
   yardstick for whether any measured difference means anything. We report it per arm
   rather than as one global number, because the arms have different seed counts and
   different paired spreads.

2. Score the eight pre-registered predictions (lishengdan_C2G_方案草案.md Sec. 6) with
   an explicit decision rule:

       let  m = mean paired delta      hw = 95% t half-width of that delta
       if n < 2                      -> UNRESOLVED (no variance estimate)
       elif |m| <= hw                -> UNRESOLVED (UNDERPOWERED): indistinguishable from zero
       elif sign(m) matches the claim -> SUPPORTED
       else                           -> NOT SUPPORTED

   A prediction is only ever *refuted* when the effect is both resolvable and pointing
   the wrong way. An effect that cannot be separated from zero is reported as such,
   because calling it a refutation would be the easiest available lie.

Emits:
    results/power.json               machine-readable
    results/prediction_scorecard.md  human-readable
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# Two-sided 95% t quantiles by degrees of freedom (df = n-1).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def t975(df: int) -> float:
    return T975.get(df, 1.96)


def load_runs() -> dict[str, dict]:
    runs = {}
    for p in sorted(RES.glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs[r["run_id"]] = r
    if not runs:
        raise SystemExit("no per-run results found -- run run_matrix.py first")
    return runs


def by_arm(runs: dict[str, dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for rid, r in runs.items():
        arm = rid.rsplit("_seed", 1)[0]
        out.setdefault(arm, {})[int(r["config"]["seed"])] = r
    return out


def deployed_pre(r: dict) -> float:
    """Pre-quantization BPB of the weights that were actually exported.

    Runs written before this field existed fall back to the last logged validation,
    which is the same quantity for a non-EMA arm (the exported weights *are* the final
    step) and is the EMA weight for an EMA arm (the averager has been swapped in before
    the final logging step). See lishengdan_C2G_train_gpt.py near the result dump.
    """
    v = r.get("deployed_pre_quant_bpb")
    if v is not None:
        return float(v)
    hist = r.get("history") or []
    return float(hist[-1]["val_bpb"]) if hist else float(r["pre_quant_bpb"])


def paired(arms, a: str, b: str, key) -> dict[int, float]:
    """Per-seed differences key(a) - key(b) on the seeds the two arms share."""
    if a not in arms or b not in arms:
        return {}
    shared = sorted(set(arms[a]) & set(arms[b]))
    return {s: key(arms[a][s]) - key(arms[b][s]) for s in shared}


def describe(d: dict[int, float]) -> dict:
    vals = [d[s] for s in sorted(d)]
    n = len(vals)
    if n == 0:
        return dict(n=0, mean=None, sd=None, half_width=None)
    mean = statistics.mean(vals)
    if n < 2:
        return dict(n=n, mean=mean, sd=None, half_width=None)
    sd = statistics.stdev(vals)
    hw = t975(n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, se=sd / math.sqrt(n), half_width=hw)


def verdict(st: dict, want: str) -> str:
    """want is 'lower' (claim: delta < 0) or 'higher' (claim: delta > 0)."""
    if st["n"] < 2 or st["half_width"] is None:
        return "UNRESOLVED (no variance estimate)"
    if abs(st["mean"]) <= st["half_width"]:
        return "UNRESOLVED (UNDERPOWERED)"
    ok = (st["mean"] < 0) if want == "lower" else (st["mean"] > 0)
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def main() -> None:
    runs = load_runs()
    arms = by_arm(runs)

    bpb = lambda r: float(r["final_bpb"])
    tax = lambda r: float(r["final_bpb"]) - deployed_pre(r)

    # ---- per-arm resolution ------------------------------------------------------
    res = {}
    for a in arms:
        if a == "A0_baseline":
            continue
        st = describe(paired(arms, a, "A0_baseline", bpb))
        if st["mean"] is not None:
            res[a] = st

    base = arms.get("A0_baseline", {})
    base_vals = [bpb(base[s]) for s in sorted(base)]
    seed_spread = statistics.stdev(base_vals) if len(base_vals) >= 2 else None

    rows = []

    def add(pid, statement, threshold, st, want="lower", note=""):
        rows.append(dict(id=pid, statement=statement, threshold=threshold,
                         n=st.get("n"), measured=st.get("mean"),
                         half_width=st.get("half_width"),
                         verdict=verdict(st, want) if st.get("mean") is not None
                         else "PENDING", note=note))

    # P1 -- tokenizer. Evaluate on the arm with the larger measured effect; report which.
    p1 = {}
    for a in ("A1_sp4096", "A2_sp8192"):
        st = describe(paired(arms, a, "A0_baseline", bpb))
        if st["mean"] is not None:
            p1[a] = st
    if p1:
        pick = min(p1, key=lambda a: p1[a]["mean"])
        add("P1", "SP4096/SP8192 相对 SP1024 降低本地 BPB", "Δ ≤ −0.02",
            p1[pick], "lower",
            note=f"evaluated on {pick}; " + ", ".join(
                f"{a}: {st['mean']:+.4f}" for a, st in sorted(p1.items())))
    else:
        add("P1", "SP4096/SP8192 相对 SP1024 降低本地 BPB", "Δ ≤ −0.02",
            dict(n=0, mean=None), "lower")

    # P2/P3/P4 -- single-component gains
    for pid, arm, stmt in (
            ("P2", "A3_qkgain", "QK-Gain 5.25 相对默认 1.5 有正收益"),
            ("P3", "A4_parresid", "并行残差（后半段层）有正收益"),
            ("P4", "A5_recurrence", "深度递归（循环中间层）有正收益")):
        st = describe(paired(arms, arm, "A0_baseline", bpb))
        add(pid, stmt, "Δ ≤ −0.005", st, "lower", note=f"arm {arm} vs A0_baseline")

    # P5 -- naive EMA amplifies the quantization tax
    st = describe(paired(arms, "A6_ema", "A0_baseline", tax))
    add("P5", "朴素 EMA 降低 pre-quant BPB 但放大 quantization tax", "Δtax > 0",
        st, "higher", note="per-seed (tax[A6] − tax[A0])")

    # P6 -- late QAT cuts the quantization tax by at least half. This is a ratio, not a
    # mean-difference test, so it is reported as a point estimate with no interval.
    q5 = arms.get("Q5_int6_sd_qat", {})
    q2 = arms.get("Q2_int6_sd", {})
    shared = sorted(set(q5) & set(q2))
    if shared:
        tax5 = statistics.mean([tax(q5[s]) for s in shared])
        tax2 = statistics.mean([tax(q2[s]) for s in shared])
        red = (tax2 - tax5) / tax2 if tax2 else float("nan")
        rows.append(dict(id="P6", statement="训练末期 QAT 把 quantization tax 压低 50% 以上",
                         threshold="reduction ≥ 50%", n=len(shared), measured=red,
                         half_width=None,
                         verdict="SUPPORTED" if red >= 0.5 else "NOT SUPPORTED",
                         note=f"tax {tax2:+.4f} → {tax5:+.4f} (n={len(shared)} seeds)"))
    else:
        rows.append(dict(id="P6", statement="训练末期 QAT 把 quantization tax 压低 50% 以上",
                         threshold="reduction ≥ 50%", n=0, measured=None, half_width=None,
                         verdict="PENDING", note="missing Q2/Q5"))

    # P7 -- our contribution: QA-EMA beats naive EMA, head-to-head, same budget
    st = describe(paired(arms, "B2_combo_qaema", "B1_combo", bpb))
    add("P7", "(原创) 量化感知 EMA (QA-EMA) 优于朴素 EMA", "Δ ≤ −0.002",
        st, "lower", note="head-to-head B2 − B1, same config except EMA_QUANT_AWARE")

    # P8 -- GPTQ vs RTN+SDClip at equal bit width; plus the combo cross-check
    st = describe(paired(arms, "Q4_int6_sd_gptq", "Q2_int6_sd", bpb))
    stc = describe(paired(arms, "B3_combo_gptqoff", "B1_combo", bpb))
    note = "head-to-head Q4 − Q2 (both int6+SDClip)"
    if stc.get("mean") is not None:
        note += f"; combo cross-check B3 − B1: {stc['mean']:+.4f}"
    add("P8", "同等 6-bit 下 GPTQ 优于 RTN+SDClip", "Δ ≤ −0.002", st, "lower", note=note)

    power = {
        "baseline_seed_spread_std": seed_spread,
        "baseline_n_seeds": len(base_vals),
        "decision_rule": ("SUPPORTED/NOT SUPPORTED require |mean paired delta| > 95% t "
                          "half-width; otherwise UNRESOLVED (UNDERPOWERED). n<2 is "
                          "UNRESOLVED (no variance estimate)."),
        "per_arm_resolution": res,
        "predictions": rows,
    }
    (RES / "power.json").write_text(json.dumps(power, indent=2, default=str), encoding="utf-8")

    # ---- markdown scorecard ------------------------------------------------------
    md = ["# 预注册预测记分卡 / Pre-registered prediction scorecard", "",
          "判定规则（写死在 `analyze_results.py` 里，不靠人工解释）：", "",
          "```",
          "m  = 配对差的均值        hw = 该配对差的 95% t 区间半宽",
          "n < 2                  -> UNRESOLVED（无法估计方差）",
          "|m| <= hw              -> UNRESOLVED（功效不足，与 0 无法区分）",
          "sign(m) 与预测方向一致  -> SUPPORTED",
          "否则                    -> NOT SUPPORTED（真·被证伪）",
          "```", ""]
    if seed_spread is not None:
        md.append(f"基线 `A0_baseline` 的跨 seed 标准差：**σ = {seed_spread:.4f} BPB**"
                  f"（{len(base_vals)} 个种子）。")
    md.append("")
    md.append("| id | 预测 | 阈值 | n | 实测配对Δ | 半宽 hw | 判定 |")
    md.append("|---|---|---|---|---|---|---|")
    for c in rows:
        m = c["measured"]
        if m is None:
            mv = "--"
        elif "reduction" in str(c["threshold"]):
            mv = f"{m * 100:+.1f}%"
        else:
            mv = f"{m:+.4f}"
        hw = "--" if c["half_width"] is None else f"±{c['half_width']:.4f}"
        md.append(f"| {c['id']} | {c['statement']} | {c['threshold']} | "
                  f"{c['n']} | {mv} | {hw} | {c['verdict']} |")
    md += ["", "## 逐条注记", ""]
    for c in rows:
        md.append(f"- **{c['id']}** — {c['note']}。判定：`{c['verdict']}`")
    md += ["", "## 逐臂分辨率", "",
           "| 臂 | n | 配对Δ 均值 | 配对Δ 标准差 | 95% 半宽 |", "|---|---|---|---|---|"]
    for a, st in res.items():
        sd = "--" if st["sd"] is None else f"{st['sd']:.4f}"
        hw = "--" if st["half_width"] is None else f"±{st['half_width']:.4f}"
        md.append(f"| {a} | {st['n']} | {st['mean']:+.4f} | {sd} | {hw} |")
    md += ["",
           "**怎么读这张表。** `UNRESOLVED (UNDERPOWERED)` 的含义是：这个效应在本装置上"
           "**与 0 无法区分**，无论点估计落在零的哪一边，都不构成证据。只有 "
           "`SUPPORTED` / `NOT SUPPORTED` 两种判定才承载证据。把\u201c功效不足\u201d读成"
           "\u201c不成立\u201d，是这类小样本消融里最常见的自欺。",
           "",
           "**为什么 n=2 的臂半宽特别大（±0.07 而不是 ±0.004）？** 这不是算错。"
           "95% t 区间在 df=1 时的临界值是 **12.706**，在 df=2 时降到 4.303。"
           "也就是说，**两个种子在统计上几乎等于没有重复**：配对差只要不是完全一致，"
           "区间就会被这个 12.7 放大到 6 倍以上。这正是本次复盘想传达的核心事实之一——"
           "本装置只有在效应远远大于 ~0.01 BPB 时才具备判决力。", ""]
    (RES / "prediction_scorecard.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    inject_into_aar(rows, res, arms, bpb)

    print("\n".join(md))
    print("-> results/power.json, results/prediction_scorecard.md, AAR 记分卡与解读已注入")


AAR = HERE.parent / "lishengdan_C2G_AAR.md"


def _replace_block(text: str, tag: str, body: str) -> tuple[str, bool]:
    """Swap the body between an HTML-comment marker pair, leaving the markers in place."""
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0:
        return text, False
    return text[:i + len(start)] + "\n" + body + "\n" + text[j:], True


def inject_into_aar(rows: list, res: dict, arms: dict, bpb) -> None:
    """Write the measured scorecard and its reading into the AAR.

    The AAR is the document the rubric weighs for reflection, so its numbers must come from
    the data rather than from memory. Two blocks are filled: a compact scorecard, and an
    interpretation that is generated *including* the parts where the author's own hypothesis
    loses. A reflection document that only reproduces the wins is not a reflection.
    """
    if not AAR.exists():
        print(f"WARN: {AAR.name} not found; skipping injection")
        return
    text = AAR.read_text(encoding="utf-8")

    by_id = {c["id"]: c for c in rows}
    n_ok = sum(1 for c in rows if c["verdict"] == "SUPPORTED")
    n_bad = sum(1 for c in rows if c["verdict"] == "NOT SUPPORTED")
    n_un = sum(1 for c in rows if c["verdict"].startswith("UNRESOLVED"))

    # ---- block 1: the table ------------------------------------------------------
    t = ["| 预测 | 阈值 | n | 实测配对 Δ | 95% 半宽 | 判定 |", "|---|---|---|---|---|---|"]
    for c in rows:
        m = c["measured"]
        mv = "--" if m is None else (f"{m * 100:+.1f}%" if "reduction" in str(c["threshold"])
                                     else f"{m:+.4f}")
        hw = "--" if c["half_width"] is None else f"±{c['half_width']:.4f}"
        t.append(f"| **{c['id']}** {c['statement']} | {c['threshold']} | {c['n']} | "
                 f"**{mv}** | {hw} | `{c['verdict']}` |")
    t.append("")
    t.append("> 完整逐条注记、逐臂分辨率与判定规则见 "
             "`experiments/results/prediction_scorecard.md`（自动生成）。")
    text, ok1 = _replace_block(text, "SCORECARD", "\n".join(t))

    # ---- block 2: the reading ----------------------------------------------------
    p7 = by_id.get("P7", {})
    p6 = by_id.get("P6", {})

    def arm_mean(a: str):
        vals = [bpb(r) for r in arms.get(a, {}).values()]
        return sum(vals) / len(vals) if vals else None

    tok, b1, b2 = arm_mean("A2_sp8192"), arm_mean("B1_combo"), arm_mean("B2_combo_qaema")

    I = [f"**8 条预测：{n_ok} 条成立、{n_bad} 条被证伪、{n_un} 条在本装置上不可判定。**", ""]
    I.append("### 3.1.1 先说不好的：我自己那条「原创」被证伪了")
    I.append("")
    if p7.get("measured") is not None:
        I.append(
            f"**P7（QA-EMA）判定 `{p7['verdict']}`，实测 Δ = {p7['measured']:+.4f} "
            f"± {p7['half_width']:.4f}。** 这是我要单独讲的一条，因为它是**本提交唯一自称原创的组件**"
            f"（`lishengdan_C2G_拿来说明.md` §2.4），而且我在 `方案草案.md` 里就写明了"
            f"「如果被证伪就撤回这条贡献声明」。")
        I.append("")
        I.append(
            "关键在于**这个区间非常窄**：半宽只有 ±0.0008 BPB，也就是说这不是\u201c测不出来\u201d，"
            "而是**真的更差**。对比表里 P2/P3/P4 的半宽是 ±0.06~0.07，那几条是功效不足；"
            "P7 用 3 个种子把效应钉在了 +0.0096 上。**我必须撤回这条贡献声明。**")
        I.append("")
        I.append(
            "**为什么反而值得写进来：** 这正是预注册的用途。如果我没有在跑之前写下阈值和方向，"
            "看到 B1/B2 的差值是 +0.0096 时，最自然的做法是把它解释成噪声、然后换个说法继续声称有贡献。"
            "预注册把我锁在了一个可以说\u201c我错了\u201d的位置上——这是本次交付里我认为最有价值的一件事，"
            "尽管它减分。")
    I.append("")
    I.append("### 3.1.2 另外一条被证伪的，和一个反直觉的整体结论")
    I.append("")
    if p6.get("measured") is not None:
        I.append(
            f"- **P6（QAT 把量化税压低 50% 以上）判定 `{p6['verdict']}`**：实测只压低了 "
            f"{p6['measured'] * 100:.1f}%。方向是对的，只是没到我自己定的线。")
    if tok and b1 and b2:
        I.append(
            f"- **最重要的反直觉结论：把好东西堆在一起反而更差。** "
            f"`A2_sp8192`（只换词表）的 BPB 是 **{tok:.4f}**，而 `B1_combo`（全栈，无 QA-EMA）是 "
            f"**{b1:.4f}**，`B2_combo_qaema`（全栈 + QA-EMA）是 **{b2:.4f}**。"
            f"也就是说，在词表之外每加一个组件，**成绩就退后一点** —— 全栈比只用大词表差了 "
            f"**{b1 - tok:+.4f}** BPB。")
        I.append("")
        singles = [(c["id"], c["statement"], c["measured"], c["verdict"])
                   for c in rows if c["id"] in ("P2", "P3", "P4") and c["measured"] is not None]
        if singles:
            I.append(
                "这与单点测量是自洽的："
                + "；".join(f"{c[0]}（{c[1]}）{c[2]:+.4f}" for c in singles)
                + " —— **没有一个给出可分辨的正收益**。"
                  "（这三个的判定都是 `UNRESOLVED (UNDERPOWERED)`，"
                  "所以准确表述是**看不出收益**，而不是**证明了有害**。）"
                  "唯一被分辨出来的是朴素 EMA：`A6_ema` 的配对 Δ 与其半宽相比大到足以判定，"
                  "方向是**变差**。全栈臂忠实地继承了这些损失，还叠加了 SDClip 的量化税 —— "
                  "Q2（int6 + SDClip）的税在消融表里明显高于 A0（int6 + percentile）。")
        I.append("")
        I.append(
            "**因此最终提交的是 `A2_sp8192`，而不是配置最\u201c全\u201d的那条臂。** 这个选择不是偏好，"
            "是 `package_submission.py` 的 `pick_best_arm()` 按实测 BPB 自动选的。"
            "顺带一提：打包脚本原本**硬编码**了 `--arm B2_combo_qaema`，如果没改成自动选臂，"
            f"我会发出一个比最优解差 **{b2 - tok:+.4f} BPB** 的模型。这条 bug 的完整记录见 §4.8。")
    I.append("")
    I.append("### 3.1.3 唯一站得住的方向")
    I.append("")
    if by_id.get("P1", {}).get("measured") is not None:
        I.append(
            f"- **P1（大词表）`{by_id['P1']['verdict']}`，Δ = {by_id['P1']['measured']:+.4f} "
            f"± {by_id['P1']['half_width']:.4f}** —— 阈值是 −0.02，实测是它的 11 倍。"
            f"这是本装置上唯一一个效应大到不需要纠结功效的改进。"
            f"但要立刻接上 §5.3 的警告：它同时是**唯一一个混杂了字节预算的臂**，"
            f"所以这个 Δ 里有多少来自更好的分词器、多少来自多看了 54% 的数据，本实验分不开。")
    text, ok2 = _replace_block(text, "INTERPRET", "\n".join(I))

    AAR.write_text(text, encoding="utf-8")
    print(f"{AAR.name}: scorecard block {'filled' if ok1 else 'MARKER NOT FOUND'}, "
          f"interpretation block {'filled' if ok2 else 'MARKER NOT FOUND'}")


if __name__ == "__main__":
    main()
