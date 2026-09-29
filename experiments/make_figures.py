"""
C2G / Parameter Golf - figure generation for the technical report.

Reads experiments/results/summary.json (plus the per-run JSONs for curves) and
writes the figures used in `lishengdan_C2G_tech_report.pdf`.

Usage:
    python make_figures.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4, "axes.spines.top": False,
    "axes.spines.right": False,
})

POS = "#C0392B"   # chinese convention: improvement / lower BPB is "good" -> red
NEG = "#1E8449"
GREY = "#555555"


def load():
    s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    runs = {}
    for p in sorted(RES.glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs[r["run_id"]] = r
    return s, runs


def paired_err(s: dict) -> float:
    """Uncertainty that belongs on a PAIRED bar: the spread of the per-seed deltas.

    Using the arm's own cross-seed std here would be the wrong quantity -- it measures
    how unstable the arm is across seeds, not how noisy the *difference* from the
    baseline is. The paired spread is the latter, and it is much smaller, because the
    shared seed effect cancels in the subtraction. Falls back to the arm std only when
    fewer than two shared seeds exist and no paired spread can be formed.
    """
    per = s.get("paired_delta_per_seed") or {}
    vals = [float(v) for v in per.values()]
    if len(vals) >= 2:
        return statistics.pstdev(vals)
    return s.get("final_bpb_std", 0.0)


def fig_ablation(summary: dict) -> None:
    arms = [a for a in summary if a != "A0_baseline"]
    if not arms:
        return
    arms_sorted = sorted(arms, key=lambda a: summary[a]["paired_delta_mean"])
    vals = [summary[a]["paired_delta_mean"] for a in arms_sorted]
    errs = [paired_err(summary[a]) for a in arms_sorted]
    cols = [POS if v < 0 else NEG for v in vals]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = range(len(arms_sorted))
    ax.barh(list(y), vals, xerr=errs, color=cols, alpha=0.85, height=0.62,
            error_kw=dict(lw=0.7, capsize=1.6, ecolor=GREY))
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels([a.split("_", 1)[1].replace("_", " ") for a in arms_sorted])
    ax.invert_yaxis()
    ax.set_xlabel(r"paired $\Delta$ val_bpb vs. baseline (lower = better)")
    ax.set_title("Component effect, paired by seed\n(error bars: spread of the paired delta)")
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)


def fig_quant_tax(summary: dict) -> None:
    arms = [a for a in summary if "quant_tax" in summary[a]]
    if not arms:
        return
    order = sorted(arms, key=lambda a: summary[a]["quant_tax"])
    pre = [summary[a]["deployed_pre_quant_bpb_mean"] for a in order]
    post = [summary[a]["final_bpb_mean"] for a in order]
    x = range(len(order))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar([i - 0.19 for i in x], pre, width=0.36, label="pre-quant", color="#5DADE2")
    ax.bar([i + 0.19 for i in x], post, width=0.36, label="post-quant", color="#E67E22")
    for i, (p, q) in enumerate(zip(pre, post)):
        ax.annotate(f"+{q - p:.3f}", (i, q), ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([a.split("_", 1)[1].replace("_", " ") for a in order],
                       rotation=45, ha="right")
    ax.set_ylabel("val_bpb (local proxy)")
    ax.set_title("Quantization tax = post − pre")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig_quant_tax.png")
    plt.close(fig)


def fig_curves(runs: dict, arms=("A0_baseline", "B1_combo", "B2_combo_qaema")) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    colors = {arms[0]: GREY, "B1_combo": "#2E86C1", "B2_combo_qaema": POS}
    plotted = False
    for arm in arms:
        series = {}
        for rid, r in runs.items():
            if not rid.startswith(arm):
                continue
            for h in r.get("history", []):
                series.setdefault(h["step"], []).append(h["val_bpb"])
        if not series:
            continue
        xs = sorted(series)
        ys = [sum(series[s]) / len(series[s]) for s in xs]
        ax.plot(xs, ys, marker="o", ms=2.5, lw=1.2, label=arm.split("_", 1)[1],
                color=colors.get(arm, "#7D3C98"))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("step")
    ax.set_ylabel("val_bpb (local proxy, pre-quant)")
    ax.set_title("Validation trajectory (seed-averaged)")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig_curves.png")
    plt.close(fig)


# Entropy-coding factor calibrated against the one artifact size that is known
# independently: the official naive baseline reports 15,815,847 model bytes for a
# 9x512xMLP2 tied SP1024 model packed as per-row int8 + zlib. The analytic
# uncompressed figure for that geometry is 17.20 MB, so the codec is worth 0.92x.
COMPRESS_RATIO = 0.92


def budget_mb(n_layers: int, d: int, mlp_mult: int, vocab: int,
              matrix_bits: int = 6, embed_bits: int = 8,
              compress_ratio: float = COMPRESS_RATIO,
              tokenizer_mb: float = 0.371, code_mb: float = 0.062) -> float:
    """Analytic artifact size in MB for a tied-embedding transformer.

    Prices the embedding, every block matrix and the control vectors, then applies the
    entropy-coding factor, then adds the tokenizer file and the trainer source --
    all of which count against the same 16,000,000-byte budget.
    """
    kv_dim = 4 * (d // 8)
    per_layer = d * d + 2 * (d * kv_dim) + d * d + 2 * (d * mlp_mult * d)
    embed = vocab * d
    control = n_layers * (5 * d + 2 * d) * 4  # fp32 attn/mlp scales + resid_mix
    payload = embed * embed_bits / 8 + per_layer * n_layers * matrix_bits / 8 + control
    payload *= compress_ratio
    return (payload + tokenizer_mb * 1e6 + code_mb * 1e6) / 1e6


def fig_budget() -> None:
    """Artifact size vs. width, showing how tight the 16 MB cap really is."""
    widths = list(range(256, 769, 32))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    series = [("MLP 2x, int6/int8", 2, 6, 8, "#2E86C1"),
              ("MLP 4x, int6/int8", 4, 6, 8, "#E67E22"),
              ("MLP 2x, int8/int8", 2, 8, 8, "#7D3C98")]
    for label, mlp, mbits, ebits, col in series:
        ys = [budget_mb(11, d, mlp, 8192, mbits, ebits) for d in widths]
        ax.plot(widths, ys, lw=1.3, color=col, label=label)
    ax.axhline(16.0, color="k", ls="--", lw=0.9)
    ax.text(262, 16.5, "16 MB cap", fontsize=6)
    ax.scatter([512], [budget_mb(9, 512, 2, 1024, 8, 8)], s=18, color="#C0392B",
               zorder=5, label="naive baseline (9x512, SP1024)")
    ax.set_xlabel("model width d  (11 layers, SP8192 tied embedding)")
    ax.set_ylabel("artifact size (MB)")
    ax.set_title("The byte cap is the binding constraint")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "fig_budget.png")
    plt.close(fig)

    print("budget sanity checks (analytic, with 0.92x entropy coding):")
    checks = [
        ("naive baseline 9x512 SP1024 MLP2 int8", (9, 512, 2, 1024, 8, 8)),
        ("11x512 SP8192 MLP2 int6", (11, 512, 2, 8192, 6, 8)),
        ("11x512 SP8192 MLP4 int6", (11, 512, 4, 8192, 6, 8)),
        ("8x512 SP8192 MLP2 int6", (8, 512, 2, 8192, 6, 8)),
        ("11x448 SP8192 MLP3 int6", (11, 448, 3, 8192, 6, 8)),
        ("11x384 SP8192 MLP2 int6", (11, 384, 2, 8192, 6, 8)),
    ]
    for label, a in checks:
        mb = budget_mb(*a)
        print(f"  {label:38s} {mb:6.2f} MB  {'FITS' if mb <= 16 else 'OVER'}")


def fig_tokenizer() -> None:
    """Tokens-per-byte for the three vocabularies, measured from the proxy corpus."""
    p = RES / "tokenizer_stats.json"
    if not p.exists():
        return
    st = json.loads(p.read_text(encoding="utf-8"))
    vs = [str(k) for k in sorted(st, key=lambda x: int(x))]
    tpb = [st[k]["tokens_per_byte"] for k in vs]
    size = [st[k]["embedding_mb_at_d512"] for k in vs]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(vs, tpb, color="#2E86C1", width=0.5)
    for i, (t, s) in enumerate(zip(tpb, size)):
        ax.text(i, t, f"{t:.3f}\n{s:.1f} MB emb", ha="center", va="bottom", fontsize=5.5)
    ax.set_xlabel("SentencePiece vocab size")
    ax.set_ylabel("tokens per byte (lower = denser)")
    ax.set_title("Tokenizer density vs. embedding budget")
    fig.savefig(FIG / "fig_tokenizer.png")
    plt.close(fig)


def main() -> None:
    summary, runs = load()
    fig_ablation(summary)
    fig_quant_tax(summary)
    fig_curves(runs)
    fig_budget()
    fig_tokenizer()
    print("figures ->", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
