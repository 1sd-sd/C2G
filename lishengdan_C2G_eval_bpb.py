"""
lishengdan_C2G_eval_bpb.py
==========================

Standalone, tokenizer-agnostic BPB evaluator for a compressed C2G artifact.

Given
  * the artifact written by `lishengdan_C2G_train_gpt.py` (`final_model.quant.bin`)
  * the `submission.json` produced next to it (model shape + quantization metadata)
  * a SentencePiece tokenizer and the validation shards

it reconstructs the model, dequantizes the weights, and recomputes `val_bpb`
exactly the way the official evaluator does: mean token cross-entropy in nats
divided by ln 2, multiplied by (tokens / byte) where token byte counts come from
the tokenizer LUT. Re-running this file is the reproducibility check.

Usage:
    python lishengdan_C2G_eval_bpb.py \
        --artifact artifact/final_model.quant.bin \
        --submission artifact/submission.json \
        --data-path experiments/corpus/datasets/proxy_sp1024 \
        --tokenizer experiments/corpus/tokenizers/fineweb_1024_bpe.model
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lishengdan_C2G_train_gpt as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-seqs", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sub = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    cfg = sub["model_config"]
    device = torch.device(args.device)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = int(sp.vocab_size())
    if cfg["vocab_size"] != vocab_size:
        raise SystemExit(f"submission vocab {cfg['vocab_size']} != tokenizer {vocab_size}")

    model = T.GPT(
        vocab_size=cfg["vocab_size"], num_layers=cfg["num_layers"], model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"], num_kv_heads=cfg["num_kv_heads"],
        mlp_mult=cfg["mlp_mult"], tie_embeddings=cfg["tie_embeddings"],
        tied_embed_init_std=cfg["tied_embed_init_std"], logit_softcap=cfg["logit_softcap"],
        rope_base=cfg["rope_base"], qk_gain_init=cfg["qk_gain_init"],
        layer_schedule=cfg.get("layer_schedule"), parallel_resid_from=cfg.get("parallel_resid_from", -1),
    ).to(device)
    for m in model.modules():
        if isinstance(m, T.CastedLinear):
            m.float()

    blob = Path(args.artifact).read_bytes()
    # The compression scheme lives under `quantization.compression` in the manifest
    # (see package_submission.py). Fall back to a top-level `compress` key and then to
    # zlib, so the evaluator still works on older manifests -- but never silently
    # default a brotli artifact to zlib, which would corrupt the round-trip.
    compress = (sub.get("compress")
                or sub.get("quantization", {}).get("compression")
                or "zlib")
    obj = torch.load(io.BytesIO(T.decompress_blob(blob, compress)),
                     map_location="cpu", weights_only=False)
    state = T.dequantize_state_dict(obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch missing={missing} unexpected={unexpected}")

    val_tokens = T.load_validation_tokens(
        str(Path(args.data_path) / "fineweb_val_*.bin"), args.seq_len)
    luts = T.build_sentencepiece_luts(sp, vocab_size, device)

    class A:
        val_batch_size = args.batch_seqs * args.seq_len
        train_seq_len = args.seq_len

    loss, bpb = T.eval_val(A(), model, 0, 1, device, 1, val_tokens, *luts)
    print(f"artifact_bytes      : {len(blob)}")
    print(f"val_tokens          : {val_tokens.numel() - 1}")
    print(f"val_loss (nats)     : {loss:.6f}")
    print(f"val_bpb             : {bpb:.6f}")
    print(f"reported val_bpb    : {sub['val_bpb']:.6f}")
    print(f"delta               : {abs(bpb - sub['val_bpb']):.3e}")
    if abs(bpb - sub["val_bpb"]) > 1e-6:
        raise SystemExit("REPRODUCIBILITY FAILURE: standalone eval disagrees with submission.json")
    print("OK - artifact reproduces the reported val_bpb")


if __name__ == "__main__":
    main()
