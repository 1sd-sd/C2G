"""
C2G / Parameter Golf - proxy corpus preparation.

Turns the raw proxy text into the *exact* artifact layout the official
`train_gpt.py` expects:

    data/datasets/<name>_sp1024/fineweb_train_000000.bin
    data/datasets/<name>_sp1024/fineweb_val_000000.bin
    data/tokenizers/fineweb_1024_bpe.model

Shard format is byte-identical to `data/download_hf_docs_and_tokenize.py`:
256 x int32 header (MAGIC=20240520, VERSION=1, num_tokens), then uint16 tokens.

Tokenizer training reproduces the official recipe verbatim:
    model_type=bpe, character_coverage=0.999, byte_fallback=True,
    split_digits=True, normalization_rule_name=nmt_nfkc, add_dummy_prefix=False,
    pad/bos/eos/unk = 0/1/2/3, hard_vocab_limit=False

Usage:
    python prepare_corpus.py --text raw/simplewiki.txt --vocab-sizes 1024 4096 8192
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
SHARD_TOKENS = 50_000_000  # smaller than the official 1e8 so local shards stay light


def write_datafile(path: Path, toks: np.ndarray) -> None:
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    toks = np.asarray(toks, dtype="<u2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def load_paragraphs(text_path: Path) -> list[str]:
    lines = [ln.rstrip("\n") for ln in text_path.open(encoding="utf-8")]
    return [ln for ln in lines if len(ln) >= 200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="raw/simplewiki.txt")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[1024, 4096, 8192])
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--val-max-chars", type=int, default=400_000,
                    help="cap the held-out split so per-run evaluation stays cheap")
    args = ap.parse_args()

    import sentencepiece as spm

    out_root = Path(args.out_root)
    tok_dir = out_root / "tokenizers"
    ds_root = out_root / "datasets"
    tok_dir.mkdir(parents=True, exist_ok=True)

    paras = load_paragraphs(Path(args.text))
    total_chars = sum(len(p) for p in paras)
    print(f"paragraphs: {len(paras)}  text: {total_chars/1e6:.1f} MB", file=sys.stderr)

    # Deterministic shuffle before splitting. The raw dump is ordered by page id, which
    # correlates with creation date, so a naive tail-split would give train and val
    # different topic mixes and inflate val BPB for every arm equally-but-oddly.
    # A seeded permutation makes the held-out split representative of the whole corpus.
    rng = random.Random(1234)
    rng.shuffle(paras)

    n_val = max(1, int(len(paras) * args.val_frac))
    val_paras, chars = [], 0
    for p in paras[:n_val]:
        if chars >= args.val_max_chars:
            break
        val_paras.append(p)
        chars += len(p)
    train_paras = paras[len(val_paras):]
    print(f"split -> train {len(train_paras)} paras / val {len(val_paras)} paras "
          f"({sum(len(p) for p in val_paras)/1e6:.2f} MB val)", file=sys.stderr)

    train_txt = out_root / "raw" / "train.txt"
    val_txt = out_root / "raw" / "val.txt"
    train_txt.write_text("\n".join(train_paras), encoding="utf-8")
    val_txt.write_text("\n".join(val_paras), encoding="utf-8")

    manifest = {"text_file": str(Path(args.text)), "total_chars": total_chars,
                "train_paras": len(train_paras), "val_paras": len(val_paras),
                "tokenizers": []}
    tok_stats: dict[str, dict] = {}

    for vs in args.vocab_sizes:
        prefix = tok_dir / f"fineweb_{vs}_bpe"
        model_path = prefix.with_suffix(".model")
        t0 = time.perf_counter()
        if not model_path.exists():
            print(f"[spm] training vocab_size={vs} ...", file=sys.stderr)
            kwargs = {
                "input": str(train_txt),
                "model_prefix": str(prefix),
                "model_type": "bpe",
                "vocab_size": vs,
                "character_coverage": 0.999,
                "byte_fallback": True,
                "split_digits": True,
                "normalization_rule_name": "nmt_nfkc",
                "add_dummy_prefix": False,
                "pad_id": 0, "bos_id": 1, "eos_id": 2, "unk_id": 3,
                "hard_vocab_limit": False,
                "num_threads": 8,
            }
            spm.SentencePieceTrainer.train(**kwargs)
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        print(f"[spm] vocab={vs} actual_size={sp.vocab_size()} "
              f"({time.perf_counter()-t0:.1f}s)", file=sys.stderr)

        ds_dir = ds_root / f"proxy_sp{vs}"
        enc = lambda s: sp.encode(s, out_type=int)
        t0 = time.perf_counter()
        train_ids = np.fromiter((i for p in train_paras for i in enc(p)),
                                dtype=np.int64, count=-1)
        val_ids = np.fromiter((i for p in val_paras for i in enc(p)),
                              dtype=np.int64, count=-1)
        print(f"[shard] vocab={vs} train_tokens={train_ids.size/1e6:.2f}M "
              f"val_tokens={val_ids.size/1e6:.2f}M ({time.perf_counter()-t0:.1f}s)", file=sys.stderr)

        for si, start in enumerate(range(0, train_ids.size, SHARD_TOKENS)):
            write_datafile(ds_dir / f"fineweb_train_{si:06d}.bin",
                           train_ids[start:start + SHARD_TOKENS].astype(np.uint16))
        for si, start in enumerate(range(0, val_ids.size, SHARD_TOKENS)):
            write_datafile(ds_dir / f"fineweb_val_{si:06d}.bin",
                           val_ids[start:start + SHARD_TOKENS].astype(np.uint16))

        manifest["tokenizers"].append({
            "vocab_size": vs,
            "model_path": str(model_path),
            "dataset_dir": str(ds_dir),
            "train_tokens": int(train_ids.size),
            "val_tokens": int(val_ids.size),
            "bytes_per_token": float(total_chars / max(train_ids.size, 1)),
        })
        tok_stats[str(vs)] = {
            "vocab_size": vs,
            "train_tokens": int(train_ids.size),
            "val_tokens": int(val_ids.size),
            "tokens_per_byte": float(train_ids.size / max(total_chars, 1)),
            "bytes_per_token": float(total_chars / max(train_ids.size, 1)),
            "embedding_mb_at_d512": vs * 512 * 4 / 1e6,
            "embedding_mb_at_d512_int8": vs * 512 / 1e6,
        }

    (out_root / "tokenizer_stats.json").write_text(json.dumps(tok_stats, indent=2),
                                                   encoding="utf-8")
    results_dir = out_root.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tokenizer_stats.json").write_text(json.dumps(tok_stats, indent=2),
                                                      encoding="utf-8")

    man_path = out_root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {man_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
