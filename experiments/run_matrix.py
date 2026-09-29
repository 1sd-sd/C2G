"""
C2G / Parameter Golf - ablation & seed-variance experiment driver.

Runs every arm through the *same* `lishengdan_C2G_train_gpt.py`, one process per
(arm, seed), and collects a machine-readable summary. Because all arms share a code
path and a data stream, the per-seed difference between two arms is a paired
comparison: seed variance mostly cancels, which is what lets a cheap local screen
produce trustworthy component rankings.

Usage:
    # full matrix (all arms, all seeds)
    python run_matrix.py --matrix full

    # a single arm, for debugging
    python run_matrix.py --only A0_baseline --seeds 1337

    # summarise existing results without running anything
    python run_matrix.py --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # C2G提交成果/
TRAIN = ROOT / "lishengdan_C2G_train_gpt.py"
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
LOGS = ROOT / "lishengdan_C2G_logs"

# ---------------------------------------------------------------- local screening config
# ~2.9M parameters at SP1024. Chosen so one ablation arm finishes on 22 CPU threads in
# a few minutes: enough tokens for component deltas to separate, nowhere near enough for
# a leaderboard number. See the technical report Sec. 5 for the extrapolation argument.
BASE = dict(
    DEVICE="cpu", COMPILE="0", MAX_WALLCLOCK_SECONDS="0",
    ITERATIONS="400", VAL_LOSS_EVERY="200", TRAIN_LOG_EVERY="100",
    TRAIN_BATCH_TOKENS="4096", TRAIN_SEQ_LEN="256", VAL_BATCH_SIZE="8192",
    NUM_LAYERS="6", MODEL_DIM="128", NUM_HEADS="8", NUM_KV_HEADS="4", MLP_MULT="2",
    TORCH_THREADS="22",
    WARMUP_STEPS="0", MUON_MOMENTUM_WARMUP_STEPS="0",
    WARMDOWN_FRAC="0.35",
)

# NOTE: these are deliberately RELATIVE to the submission root. sentencepiece loads its
# model through C++ fopen(), which on Windows resolves non-ASCII absolute paths using the
# ANSI code page and fails. Running with cwd = submission root keeps every path handed to
# the C++ side pure ASCII while the Python side (torch, numpy, glob) is Unicode-safe.
SP1024 = dict(
    VOCAB_SIZE="1024",
    DATA_PATH="experiments/corpus/datasets/proxy_sp1024",
    TOKENIZER_PATH="experiments/corpus/tokenizers/fineweb_1024_bpe.model",
)
SP4096 = dict(
    VOCAB_SIZE="4096",
    DATA_PATH="experiments/corpus/datasets/proxy_sp4096",
    TOKENIZER_PATH="experiments/corpus/tokenizers/fineweb_4096_bpe.model",
)
SP8192 = dict(
    VOCAB_SIZE="8192",
    DATA_PATH="experiments/corpus/datasets/proxy_sp8192",
    TOKENIZER_PATH="experiments/corpus/tokenizers/fineweb_8192_bpe.model",
)

# 6 physical layers -> schedule of 11 executed positions that loops layers 2-4 twice
RECUR_6L = "0,1,2,3,4,5,2,3,4,5"

ARMS: dict[str, dict] = {
    # ---- reference arm ------------------------------------------------------------
    # NOTE (corrected post-hoc): the trainer's defaults are QUANT_BITS=6 and
    # QUANT_CLIP=percentile, NOT the scaffold's int8/percentile export rule as an earlier
    # comment here claimed. The values are now written explicitly so the recorded config
    # matches the intent and nobody has to read the trainer to know what "default" meant.
    # Consequence: the 2x2 (bits x clip) block is missing its (8, percentile) cell, and
    # Q1_int6_pct below is configuration-identical to this arm.
    "A0_baseline":      dict(**SP1024, QUANT_BITS="6", QUANT_CLIP="percentile"),
    # ---- single-point improvements (Level 2) -------------------------------------
    "A1_sp4096":        dict(**SP4096),
    "A2_sp8192":        dict(**SP8192),
    "A3_qkgain":        dict(**SP1024, QK_GAIN_INIT="5.25"),
    "A4_parresid":      dict(**SP1024, PARALLEL_RESID_FROM="4"),
    "A5_recurrence":    dict(**SP1024, LAYER_SCHEDULE=RECUR_6L),
    "A6_ema":           dict(**SP1024, EMA_DECAY="0.996", EMA_START_FRAC="0.65"),
    # ---- quantization ablation: bit width x clip rule + GPTQ + QAT ----------------
    # Q1 was intended as the (6, percentile) cell. Because A0 turned out to already sit at
    # that configuration, it functions instead as an accidental DETERMINISM CONTROL: same
    # config, different arm label. Its seeds 1337/42 reproduce A0's bit-for-bit, which is
    # the strongest determinism evidence in this package. Kept, not deleted.
    "Q1_int6_pct":      dict(**SP1024, QUANT_BITS="6", QUANT_CLIP="percentile"),
    "Q2_int6_sd":       dict(**SP1024, QUANT_BITS="6", QUANT_CLIP="sdclip"),
    "Q3_int8_sd":       dict(**SP1024, QUANT_BITS="8", QUANT_CLIP="sdclip"),
    # (8, percentile) -- the missing cell of the 2x2. Not run: it would have cost 2 runs
    # and the clip comparison is already available at int6 (A0 vs Q2).
    "Q4_int6_sd_gptq":  dict(**SP1024, QUANT_BITS="6", QUANT_CLIP="sdclip",
                             QUANT_METHOD="gptq"),
    "Q5_int6_sd_qat":   dict(**SP1024, QUANT_BITS="6", QUANT_CLIP="sdclip",
                             QAT_START_FRAC="-1.0", QAT_BITS="6"),
    # ---- combination (Level 3) ----------------------------------------------------
    "B1_combo":         dict(**SP8192, QK_GAIN_INIT="5.25", PARALLEL_RESID_FROM="4",
                             LAYER_SCHEDULE=RECUR_6L, EMA_DECAY="0.996",
                             EMA_START_FRAC="0.65", QAT_START_FRAC="-1.0", QAT_BITS="6",
                             QUANT_BITS="6", QUANT_CLIP="sdclip", QUANT_METHOD="gptq",
                             COMPRESS="brotli"),
    # ---- our component, isolated (Level 4 candidate) -------------------------------
    "B2_combo_qaema":   dict(**SP8192, QK_GAIN_INIT="5.25", PARALLEL_RESID_FROM="4",
                             LAYER_SCHEDULE=RECUR_6L, EMA_DECAY="0.996",
                             EMA_START_FRAC="0.65", EMA_QUANT_AWARE="1",
                             QAT_START_FRAC="-1.0", QAT_BITS="6",
                             QUANT_BITS="6", QUANT_CLIP="sdclip", QUANT_METHOD="gptq",
                             COMPRESS="brotli"),
    "B3_combo_gptqoff": dict(**SP8192, QK_GAIN_INIT="5.25", PARALLEL_RESID_FROM="4",
                             LAYER_SCHEDULE=RECUR_6L, EMA_DECAY="0.996",
                             EMA_START_FRAC="0.65", EMA_QUANT_AWARE="1",
                             QAT_START_FRAC="-1.0", QAT_BITS="6",
                             QUANT_BITS="6", QUANT_CLIP="sdclip",
                             # "rtn" = round-to-nearest. B2 is the same stack with GPTQ
                             # instead, so this pair isolates GPTQ vs plain RTN at a fixed
                             # clip rule. Archived runs of this arm recorded the legacy
                             # token "rtn_sdclip", which the trainer treated identically and
                             # now normalises with a warning (see lishengdan_C2G_train_gpt.py).
                             QUANT_METHOD="rtn", COMPRESS="brotli"),
}

# Which arms get the full 3-seed treatment. The rest get 2 seeds, which is enough to
# rank them against the 3-seed baseline through the paired comparison.
SEEDS_FULL = [1337, 42, 2024]
SEEDS_LEAN = [1337, 42]
KEY_ARMS = ("A0_baseline", "A2_sp8192", "B1_combo", "B2_combo_qaema")


def run_arm(arm: str, seed: int) -> dict:
    env = dict(os.environ)
    env.update({k: v for k, v in BASE.items() if v != "" and not k.endswith("PLACEHOLDER")})
    env.update({k: str(v) for k, v in ARMS[arm].items()})
    env["SEED"] = str(seed)
    env["RUN_ID"] = f"{arm}_seed{seed}"
    env["OUT_DIR"] = str(RESULTS / "runs" / f"{arm}_seed{seed}")
    env["RESULT_JSON"] = str(RESULTS / f"{arm}_seed{seed}.json")
    Path(env["OUT_DIR"]).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(TRAIN)], env=env, cwd=str(ROOT),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-4000:], flush=True)
        raise RuntimeError(f"{arm} seed{seed} failed (rc={proc.returncode})")
    res = json.loads(Path(env["RESULT_JSON"]).read_text(encoding="utf-8"))
    res["wallclock_s"] = round(dt, 1)
    Path(env["RESULT_JSON"]).write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"  {arm:20s} seed{seed:<6d} bpb={res['final_bpb']:.4f} "
          f"pre={res['pre_quant_bpb']:.4f} steps={res['steps_completed']} "
          f"art={res['artifact_bytes']/1e3:.0f}KB {dt:.0f}s", flush=True)
    return res


def deployed_pre(r: dict) -> float:
    """Pre-quantization BPB of the weights that were actually exported.

    Runs produced before this field existed fall back to the last checkpointed
    validation, which is the same quantity (EMA swapped in), so the fallback is exact
    rather than approximate.
    """
    v = r.get("deployed_pre_quant_bpb")
    if v is not None:
        return v
    hist = r.get("history") or []
    return hist[-1]["val_bpb"] if hist else r["pre_quant_bpb"]


def summarize() -> None:
    rows = []
    # Only per-run result files. A bare "*.json" here also matches tokenizer_stats.json,
    # power.json, and summary.json itself -- none of which carry a "run_id", so the first
    # glob would crash and every subsequent run would crash on its own output.
    for p in sorted(RESULTS.glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "run_id" not in r:
            continue
        rows.append(r)
    if not rows:
        print("no results yet")
        return

    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        arm = r["run_id"].rsplit("_seed", 1)[0]
        by_arm.setdefault(arm, []).append(r)

    baseline = {r["seed"]: r["final_bpb"] for r in by_arm.get("A0_baseline", [])}

    order = list(ARMS.keys())
    summary = {}
    for arm in order:
        if arm not in by_arm:
            continue
        rs = by_arm[arm]
        bpbs = [r["final_bpb"] for r in rs]
        pres = [r["pre_quant_bpb"] for r in rs]
        deps = [deployed_pre(r) for r in rs]
        paired = [r["final_bpb"] - baseline[r["seed"]] for r in rs
                  if r["seed"] in baseline and arm != "A0_baseline"]
        summary[arm] = {
            "n_seeds": len(rs),
            "seeds": sorted(r["seed"] for r in rs),
            "final_bpb_mean": statistics.mean(bpbs),
            "final_bpb_std": statistics.pstdev(bpbs) if len(bpbs) > 1 else 0.0,
            "pre_quant_bpb_mean": statistics.mean(pres),
            "deployed_pre_quant_bpb_mean": statistics.mean(deps),
            "quant_tax": statistics.mean(bpbs) - statistics.mean(deps),
            "paired_delta_mean": statistics.mean(paired) if paired else 0.0,
            "paired_delta_per_seed": {str(r["seed"]): r["final_bpb"] - baseline[r["seed"]]
                                      for r in rs if r["seed"] in baseline},
            "artifact_bytes": rs[0]["artifact_bytes"],
            "n_params": rs[0]["n_params"],
            "steps": rs[0]["steps_completed"],
            "wallclock_s": statistics.mean([r.get("wallclock_s", 0) for r in rs]),
        }

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    first = next(iter(summary.values()))
    md = ["# Local proxy ablation matrix", "",
          f"Model: {first['n_params']:,} params, "
          f"{BASE['NUM_LAYERS']} layers x {BASE['MODEL_DIM']}, "
          f"{BASE['TRAIN_BATCH_TOKENS']} tokens/step, {first['steps']} steps "
          f"({int(BASE['TRAIN_BATCH_TOKENS']) * first['steps'] / 1e6:.2f}M tokens seen).",
          "",
          "| arm | seeds | final val_bpb | std | deployed pre-quant | quant tax | paired Δ vs baseline | artifact |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, s in summary.items():
        md.append(f"| {arm} | {s['n_seeds']} | {s['final_bpb_mean']:.4f} | "
                  f"{s['final_bpb_std']:.4f} | {s['pre_quant_bpb_mean']:.4f} | "
                  f"{s['quant_tax']:+.4f} | {s['paired_delta_mean']:+.4f} | "
                  f"{s['artifact_bytes']:,} |")
    (RESULTS / "ablation_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_seeds", "final_bpb_mean", "final_bpb_std", "deployed_pre_quant_bpb_mean",
                    "quant_tax", "paired_delta_mean", "artifact_bytes", "n_params", "steps",
                    "wallclock_s"])
        for arm, s in summary.items():
            w.writerow([arm, s["n_seeds"], f"{s['final_bpb_mean']:.6f}",
                        f"{s['final_bpb_std']:.6f}", f"{s['deployed_pre_quant_bpb_mean']:.6f}",
                        f"{s['quant_tax']:.6f}", f"{s['paired_delta_mean']:.6f}",
                        s["artifact_bytes"], s["n_params"], s["steps"],
                        f"{s['wallclock_s']:.1f}"])
    print((RESULTS / "ablation_table.md").read_text(encoding="utf-8"))


def check_arms_are_distinct() -> None:
    """Fail loudly if two arms resolve to an identical configuration.

    This exists because it actually happened here. `A0_baseline` was documented as
    int8/percentile but its real default was int6/percentile, which made `Q1_int6_pct`
    configuration-identical to it - two runs that could only ever reproduce each other.
    That was caught by noticing identical artifact byte counts in the output, i.e. by luck
    and attention. It cost nothing to make it impossible instead: compare the fully
    resolved env for every arm pair before scheduling anything.

    The baseline is exempted only from *itself*; every real collision is an error.
    """
    resolved: dict[str, dict] = {}
    for arm, overrides in ARMS.items():
        env = {k: str(v) for k, v in BASE.items()}
        env.update({k: str(v) for k, v in overrides.items()})
        resolved[arm] = env
    collisions = []
    names = sorted(resolved)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if resolved[a] == resolved[b]:
                collisions.append((a, b))
    if collisions:
        for a, b in collisions:
            print(f"ERROR: arms '{a}' and '{b}' have identical configurations. "
                  f"One of them is not an ablation - fix the arm definitions or delete "
                  f"the duplicate deliberately (see run_matrix.py ARMS comments).")
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="full")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent runs; measured sweet spot is 4 x 5 torch threads")
    ap.add_argument("--threads-per-worker", type=int, default=5)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="do not re-run (arm, seed) pairs whose result JSON exists")
    ap.add_argument("--allow-duplicate-arms", action="store_true",
                    help="proceed even if two arms have identical configs (Q1 is kept as a "
                         "deliberate determinism control, so this is required to re-run it)")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize()
        return

    if not args.allow_duplicate_arms:
        check_arms_are_distinct()

    arms = args.only or list(ARMS.keys())
    jobs: list[tuple[str, int]] = []
    for arm in arms:
        seeds = args.seeds if args.seeds else (SEEDS_FULL if arm in KEY_ARMS else SEEDS_LEAN)
        for seed in seeds:
            if args.skip_done and (RESULTS / f"{arm}_seed{seed}.json").exists():
                continue
            jobs.append((arm, seed))

    print(f"scheduling {len(jobs)} runs with {args.workers} workers", flush=True)
    base_backup = BASE["TORCH_THREADS"]
    BASE["TORCH_THREADS"] = str(args.threads_per_worker)
    failures: list[tuple[str, int]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_arm, a, s): (a, s) for a, s in jobs}
            for fut in as_completed(futs):
                a, s = futs[fut]
                try:
                    fut.result()
                except Exception as exc:  # keep going; report at the end
                    failures.append((a, s))
                    print(f"  !! {a} seed{s} FAILED: {exc}", flush=True)
    finally:
        BASE["TORCH_THREADS"] = base_backup

    summarize()
    if failures:
        raise SystemExit(f"{len(failures)} runs failed: {failures}")


if __name__ == "__main__":
    main()
