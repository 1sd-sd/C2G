"""
C2G / Parameter Golf - assemble the submission artifacts.

Produces, next to this file's parent directory:
  * lishengdan_C2G_submission.json   - manifest: model shape, quantization scheme,
                                       measured val_bpb, artifact byte accounting
  * lishengdan_C2G_submission.tar.gz - the <=16,000,000 byte artifact
                                       (trainer + quantized weights + tokenizer)
  * lishengdan_C2G_logs/             - full run logs copied out of results/runs/

The byte cap is checked, not assumed. If the artifact exceeds 16,000,000 bytes the
script raises instead of writing a manifest that claims compliance.

Usage:
    python package_submission.py [--arm B2_combo_qaema]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
LOGS = ROOT / "lishengdan_C2G_logs"
ART = ROOT / "artifact"
LIMIT = 16_000_000

sys.path.insert(0, str(ROOT))
import lishengdan_C2G_train_gpt as T  # noqa: E402


def pick_best_arm(min_seeds: int = 2) -> str:
    """Choose the arm with the lowest mean final_bpb among arms that have >= min_seeds runs.

    This is deliberately NOT hardcoded. The single-point screen showed that some components
    (EMA, depth recurrence) hurt at this scale, and the full-stack combos inherit them --
    so assuming the biggest arm is the best one would ship a worse model on faith. The
    selection is made from measured BPB, and the chosen arm plus the runner-up are printed
    so the trade-off is visible in the log.
    """
    by_arm: dict[str, list[float]] = {}
    for p in sorted(RES.glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        arm = r["run_id"].rsplit("_seed", 1)[0]
        by_arm.setdefault(arm, []).append(r["final_bpb"])
    ranked = sorted(
        ((sum(v) / len(v), a) for a, v in by_arm.items() if len(v) >= min_seeds),
        key=lambda x: x[0],
    )
    if not ranked:
        raise SystemExit("no arm has enough seeds to select from")
    for mean, arm in ranked[:3]:
        print(f"  arm ranking: {arm:20s} mean_bpb={mean:.4f}  n={len(by_arm[arm])}")
    return ranked[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="auto",
                    help="'auto' picks the lowest mean BPB arm (see pick_best_arm), "
                         "or name an arm explicitly")
    args = ap.parse_args()

    arm = pick_best_arm() if args.arm == "auto" else args.arm
    print(f"selected arm: {arm}")

    # The "deployed" run is the median-BPB seed of the chosen arm.
    cands = []
    for p in sorted(RES.glob(f"{arm}_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        cands.append((r["final_bpb"], r, p))
    if not cands:
        raise SystemExit(f"no results for arm {arm}")
    cands.sort(key=lambda x: x[0])
    all_results = [c[1] for c in cands]
    _, res, res_path = cands[len(cands) // 2]
    run_id = res["run_id"]
    run_dir = RES / "runs" / run_id
    print(f"deploying {run_id}: final_bpb={res['final_bpb']:.4f} "
          f"(median of {len(cands)} seeds)")

    cfg = res["config"]
    model_config = {
        "vocab_size": int(cfg["vocab_size"]), "num_layers": int(cfg["num_layers"]),
        "model_dim": int(cfg["model_dim"]), "num_heads": int(cfg["num_heads"]),
        "num_kv_heads": int(cfg["num_kv_heads"]), "mlp_mult": int(cfg["mlp_mult"]),
        "tie_embeddings": bool(cfg["tie_embeddings"]),
        "tied_embed_init_std": float(cfg["tied_embed_init_std"]),
        "logit_softcap": float(cfg["logit_softcap"]),
        "rope_base": float(cfg["rope_base"]),
        "qk_gain_init": float(cfg["qk_gain_init"]),
        "layer_schedule": ([int(x) for x in str(cfg["layer_schedule"]).split(",")
                            if str(x).strip()] or None),
        "parallel_resid_from": int(cfg["parallel_resid_from"]),
    }

    ART.mkdir(parents=True, exist_ok=True)
    weight_blob = (run_dir / "final_model.quant.bin").read_bytes()
    code_blob = (ROOT / "lishengdan_C2G_train_gpt.py").read_bytes()
    (ART / "final_model.quant.bin").write_bytes(weight_blob)

    # The run config stores the tokenizer path RELATIVE to the submission root (it has to,
    # because sentencepiece's C++ fopen() cannot open non-ASCII absolute paths on Windows).
    # Resolve against ROOT rather than the process cwd so this script works from anywhere.
    tok_src = Path(cfg["tokenizer_path"])
    if not tok_src.is_absolute():
        tok_src = ROOT / tok_src
    tok_bytes = tok_src.read_bytes()
    tok_name = tok_src.name

    # `quant_method` names the ALGORITHM (RTN vs GPTQ); `quant_clip` names the clip rule;
    # they are independent knobs. Echoing `quant_method` verbatim produces the
    # self-contradicting label "RTN_SDCLIP int6 matrices + percentile clip" -- a name that
    # both claims and denies SDClip. Report the EFFECTIVE rule, and carry the declared
    # token only as provenance so nothing about the run is lost.
    declared_method = str(cfg["quant_method"])
    effective_method = "GPTQ" if declared_method == "gptq" else "RTN"

    # --- build the tar.gz artifact -------------------------------------------------
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        add("train_gpt.py", code_blob)
        add("final_model.quant.bin", weight_blob)
        add(tok_name, tok_bytes)
        add("submission_meta.json", json.dumps(
            {"model_config": model_config,
             "quant": {"method": effective_method.lower(),
                       "method_declared": declared_method,
                       "clip": cfg["quant_clip"],
                       "bits": cfg["quant_bits"], "embed_bits": cfg["quant_embed_bits"],
                       "sdclip_k": cfg["sdclip_k"], "compress": cfg["compress"]}},
            indent=2).encode())
    artifact = buf.getvalue()
    (ART / "lishengdan_C2G_submission.tar.gz").write_bytes(artifact)

    if len(artifact) > LIMIT:
        raise SystemExit(f"ARTIFACT TOO LARGE: {len(artifact)} > {LIMIT}")
    print(f"artifact: {len(artifact):,} bytes  (cap {LIMIT:,}, "
          f"headroom {LIMIT - len(artifact):,})")

    # --- manifest ------------------------------------------------------------------
    tarball_size = len(tok_bytes) + len(code_blob) + len(weight_blob)

    # Describe the arm that actually ships, derived from its config rather than hardcoded,
    # so the manifest can never drift from the artifact it points at.
    #
    # Every threshold here is the trainer's DEFAULT for that knob, taken from
    # lishengdan_C2G_train_gpt.py. A first version of this function used arbitrary
    # thresholds (>1.0 for QK-gain, <num_layers for parallel residual) and produced a name
    # claiming five components on a model that had none of them enabled -- the same
    # "label does not match the code" failure as the A0 mislabelling. Components are now
    # listed only when the value actually differs from its default.
    DEFAULTS = {"qk_gain_init": 1.5, "parallel_resid_from": -1, "ema_decay": 0.0,
                "qat_start_frac": 0.0, "layer_schedule": "", "vocab_size": 1024,
                "clip": "percentile", "compress": "zlib"}
    parts = []
    if int(cfg["vocab_size"]) != DEFAULTS["vocab_size"]:
        parts.append(f"SP{cfg['vocab_size']} vocabulary")
    if float(cfg["qk_gain_init"]) != DEFAULTS["qk_gain_init"]:
        parts.append(f"QK-Gain {cfg['qk_gain_init']}")
    prf = int(cfg["parallel_resid_from"])
    if prf != DEFAULTS["parallel_resid_from"] and prf < int(cfg["num_layers"]):
        parts.append(f"Parallel Residuals from layer {prf}")
    if str(cfg["layer_schedule"]).strip():
        parts.append("Depth Recurrence")
    if float(cfg["ema_decay"]) > 0.0:
        parts.append("QA-EMA" if bool(cfg["ema_quant_aware"]) else "EMA")
    if float(cfg["qat_start_frac"]) != DEFAULTS["qat_start_frac"]:
        parts.append(f"QAT int{cfg['qat_bits']}")
    if not parts:
        parts.append("scaffold default architecture")

    export = f"{effective_method} int{cfg['quant_bits']} matrices"
    export += f" / int{cfg['quant_embed_bits']} embeddings"
    export += (" + SDClip" if str(cfg["quant_clip"]) == "sdclip" else " + percentile clip")
    export += f" / {cfg['compress']}"
    parts.append(export)

    manifest = {
        "author": "lishengdan",
        "github_id": "lishengdan",
        "arm": arm,
        "name": " + ".join(parts),
        "date": "2026-09-29",
        "track": "10min_16mb",
        "val_bpb": round(res["final_bpb"], 6),
        "val_bpb_pre_quant": round(res.get("deployed_pre_quant_bpb", res["pre_quant_bpb"]), 6),
        "quantization_tax": round(res["final_bpb"] - res.get("deployed_pre_quant_bpb", res["pre_quant_bpb"]), 6),
        "val_loss": round(res["final_loss"], 6),
        "seeds": [r["config"]["seed"] for r in all_results],
        "seed_results": {
            str(r["config"]["seed"]): {
                "val_bpb": round(r["final_bpb"], 6),
                "val_bpb_pre_quant": round(r.get("deployed_pre_quant_bpb", r["pre_quant_bpb"]), 6),
                "artifact_bytes": r["artifact_bytes"],
            } for r in all_results
        },
        "seed_bpb_mean": round(sum(r["final_bpb"] for r in all_results) / len(all_results), 6),
        "seed_bpb_std": round(
            (sum((r["final_bpb"] - sum(x["final_bpb"] for x in all_results) / len(all_results)) ** 2
                 for r in all_results) / len(all_results)) ** 0.5, 6),
        "hardware": "CPU (22 threads) - local screening run, NOT a 8xH100 submission",
        "pytorch_version": __import__("torch").__version__,
        "evaluation_corpus": ("Simple English Wikipedia proxy corpus (deterministic "
                              "prefix of dumps.wikimedia.org multistream dump, CC BY-SA 4.0)"),
        "not_official_fineweb": True,
        "model_config": model_config,
        "quantization": {
            "method": effective_method.lower(), "clip_rule": cfg["quant_clip"],
            "method_declared": declared_method,
            "matrix_bits": cfg["quant_bits"], "embedding_bits": cfg["quant_embed_bits"],
            "sdclip_k": cfg["sdclip_k"], "compression": cfg["compress"],
        },
        "components": {
            "sp_vocab": cfg["vocab_size"],
            "qk_gain_init": cfg["qk_gain_init"],
            "parallel_resid_from": cfg["parallel_resid_from"],
            "layer_schedule": cfg["layer_schedule"] or "identity",
            "ema_decay": cfg["ema_decay"],
            "ema_quant_aware": bool(cfg["ema_quant_aware"]),
            "qat_start_frac": cfg["qat_start_frac"],
        },
        "artifact": {
            "total_bytes": len(artifact),
            "cap_bytes": LIMIT,
            "compliant": len(artifact) <= LIMIT,
            "uncompressed_payload_bytes": tarball_size,
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "components": {
                "code_bytes": len(code_blob),
                "weights_compressed_bytes": len(weight_blob),
                "tokenizer_bytes": len(tok_bytes),
            },
        },
        "reproduce": {
            "train": ("python lishengdan_C2G_train_gpt.py  "
                      "# with the env vars recorded in the run log"),
            "evaluate": ("python lishengdan_C2G_eval_bpb.py --artifact <tar.gz payload> "
                         "--submission lishengdan_C2G_submission.json "
                         "--data-path <proxy corpus> --tokenizer <sp model>"),
            "run_log": f"lishengdan_C2G_logs/{run_id}.txt",
        },
        "compliance_statement": (
            "This manifest describes a LOCAL SCREENING RUN on a CPU. It reproduces the "
            "official pipeline (training, quantization, compression, round-trip BPB "
            "evaluation) on a proxy corpus but does not claim the official FineWeb "
            "val_bpb, the 10-minute 8xH100 constraint, or the 16MB-scale model."),
    }
    (ROOT / "lishengdan_C2G_submission.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote lishengdan_C2G_submission.json")

    # --- logs ----------------------------------------------------------------------
    LOGS.mkdir(parents=True, exist_ok=True)
    for old in LOGS.glob("*.txt"):
        old.unlink()
    copied = 0
    for p in sorted((RES / "runs").glob("*/logs/*.txt")):
        shutil.copy2(p, LOGS / p.name)
        copied += 1
    print(f"copied {copied} run logs -> lishengdan_C2G_logs/")


if __name__ == "__main__":
    main()
