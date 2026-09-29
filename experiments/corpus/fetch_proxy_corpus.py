"""
C2G / Parameter Golf - proxy corpus builder.

The official challenge evaluates on the held-out FineWeb validation split, which is
served from Hugging Face. In this environment HF is unreachable, so for *local
ablation screening* we build a reproducible English proxy corpus from a stable,
publicly mirrored source:

    Simple English Wikipedia  (dumps.wikimedia.org, CC BY-SA 4.0)

The dump is a multistream BZ2 archive: many independent bz2 blocks concatenated.
That means we can download an arbitrary *prefix* of the file, decompress every
complete block, and simply drop the trailing (truncated) block. This gives us a
few tens of MB of real encyclopedic prose with a single small HTTP range request,
instead of the full 386 MB dump.

Usage:
    python fetch_proxy_corpus.py --target-bytes 26000000 --out raw/simplewiki.txt
"""
from __future__ import annotations

import argparse
import bz2
import html
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://dumps.wikimedia.org/simplewiki/latest/"
MULTISTREAM = "simplewiki-latest-pages-articles-multistream.xml.bz2"

PAGE_RE = re.compile(rb"<page>(.*?)</page>", re.DOTALL)
TEXT_RE = re.compile(rb"<text[^>]*>(.*?)</text>", re.DOTALL)
TITLE_RE = re.compile(rb"<title>(.*?)</title>", re.DOTALL)
REDIRECT_RE = re.compile(rb"<redirect\b")

DROP_PATTERNS = [
    re.compile(r"\{\{[^{}]*\}\}", re.DOTALL),
    re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL),
    re.compile(r"<ref[^>]*/>", re.DOTALL),
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"\[\[(?:File|Image|Category|Template|Wikipedia|Help|Portal):[^\]]*\]\]", re.IGNORECASE),
    re.compile(r"\{\|.*?\|\}", re.DOTALL),
    re.compile(r"<[^>]+>"),
]


def fetch_prefix(url: str, nbytes: int, offset: int = 0, chunk: int = 1 << 20) -> bytes:
    end = offset + nbytes - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}",
                                               "User-Agent": "c2g-parameter-golf/local-screening"})
    buf = bytearray()
    with urllib.request.urlopen(req, timeout=180) as resp:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            buf.extend(block)
            if len(buf) >= nbytes:
                break
            print(f"\r  downloaded {len(buf)/1e6:6.1f} / {nbytes/1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    return bytes(buf)


def decompress_streams(blob: bytes) -> bytes:
    """Concatenated-bz2 aware decompression; the final truncated block is dropped.

    A byte offset into a multistream dump can land *inside* a bz2 block, so we
    re-align to the next `BZh` header instead of assuming the buffer starts on a
    block boundary. This is what makes `--offset-bytes` usable.
    """
    out = bytearray()
    pos = 0
    nblocks = 0
    while pos < len(blob):
        idx = blob.find(b"BZh", pos)
        if idx < 0:
            break
        d = bz2.BZ2Decompressor()
        try:
            out.extend(d.decompress(blob[idx:]))
        except (OSError, EOFError):
            pos = idx + 3          # false header or mid-block start; keep scanning
            continue
        if not d.eof:
            break                  # truncated trailing block
        pos = idx + (len(blob[idx:]) - len(d.unused_data))
        nblocks += 1
    print(f"  decompressed {nblocks} complete bz2 blocks -> {len(out)/1e6:.1f} MB xml", file=sys.stderr)
    return bytes(out)


def strip_wiki_markup(text: str) -> str:
    for _ in range(6):
        for pat in DROP_PATTERNS:
            text = pat.sub(" " if pat.pattern.startswith(r"\{\{") else "\n", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", " ", text)
    text = re.sub(r"^[=]{2,}.*?[=]{2,}\s*$", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-bytes", type=int, default=26_000_000,
                    help="compressed bytes to pull from the multistream dump")
    ap.add_argument("--offset-bytes", type=int, default=0,
                    help="compressed byte offset to start at (multi-stream seeks are safe)")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of overwriting")
    ap.add_argument("--out", default="raw/simplewiki.txt")
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] fetching {args.target_bytes/1e6:.1f} MB at offset "
          f"{args.offset_bytes/1e6:.1f} MB of {MULTISTREAM}", file=sys.stderr)
    blob = fetch_prefix(BASE + MULTISTREAM, args.target_bytes, args.offset_bytes)

    print("[2/4] decompressing concatenated bz2 blocks", file=sys.stderr)
    xml = decompress_streams(blob)

    print("[3/4] extracting + cleaning article text", file=sys.stderr)
    paras: list[str] = []
    n_articles = 0
    for page in PAGE_RE.findall(xml):
        if REDIRECT_RE.search(page):
            continue
        m = TEXT_RE.search(page)
        if not m:
            continue
        t = TITLE_RE.search(page)
        title = html.unescape(t.group(1).decode("utf-8", "replace")).strip() if t else ""
        if ":" in title:
            continue
        raw = html.unescape(m.group(1).decode("utf-8", "replace"))
        clean = strip_wiki_markup(raw)
        n_articles += 1
        for para in clean.split("\n"):
            para = para.strip()
            if len(para) >= args.min_chars:
                paras.append(para)

    print(f"  kept {n_articles} articles -> {len(paras)} paragraphs "
          f"({sum(len(p) for p in paras)/1e6:.1f} MB text)", file=sys.stderr)

    print(f"[4/4] writing {out_path}", file=sys.stderr)
    with out_path.open("a" if args.append else "w", encoding="utf-8", newline="\n") as f:
        for p in paras:
            f.write(p.replace("\n", " "))
            f.write("\n")
    print(f"done: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
