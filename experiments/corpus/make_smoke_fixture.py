"""
Smoke-test fixture: build a tiny tokenizer + shards so every code path in
`lishengdan_C2G_train_gpt.py` (baseline / parallel residuals / depth recurrence /
EMA / QA-EMA / QAT / RTN / GPTQ / export / roundtrip) can be exercised in seconds
before touching the real proxy corpus.

Source text is the challenge description itself, so the fixture is self-contained
and requires no network.

Usage:
    python make_smoke_fixture.py --text ../../CHALLENGE.md --out smoke --vocab 1024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def write_datafile(path: Path, toks: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(toks)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(toks, dtype="<u2").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="smoke")
    ap.add_argument("--vocab", type=int, default=1024)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "tokenizers").mkdir(parents=True, exist_ok=True)
    text = Path(args.text).read_text(encoding="utf-8")

    # Repeat the source so sentencepiece has enough material for the target vocab.
    corpus = (text + "\n") * 8
    src = out / "smoke_corpus.txt"
    src.write_text(corpus, encoding="utf-8")

    tok_dir = out / "tokenizers"
    prefix = tok_dir / f"fineweb_{args.vocab}_bpe"
    spm.SentencePieceTrainer.train(
        input=str(src), model_prefix=str(prefix), model_type="bpe",
        vocab_size=args.vocab, character_coverage=0.999, byte_fallback=True,
        split_digits=True, normalization_rule_name="nmt_nfkc", add_dummy_prefix=False,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3, hard_vocab_limit=False, num_threads=8,
    )
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    vs = int(sp.vocab_size())

    ids = np.array(sp.encode(text, out_type=int), dtype=np.uint16)
    # train streams wrap around forever, so a small buffer is enough
    write_datafile(out / f"datasets/proxy_sp{vs}/fineweb_train_000000.bin", ids[: len(ids) // 2])
    write_datafile(out / f"datasets/proxy_sp{vs}/fineweb_val_000000.bin", ids[len(ids) // 2:])
    (out / "fixture.json").write_text(json.dumps(
        {"vocab_size": vs, "tokenizer": str(prefix) + ".model",
         "data_path": str(out / f"datasets/proxy_sp{vs}")}, indent=2), encoding="utf-8")
    print(json.dumps({"vocab_size": vs, "tokenizer": str(prefix) + ".model",
                      "data_path": str(out / f"datasets/proxy_sp{vs}")}, indent=2))


if __name__ == "__main__":
    main()
