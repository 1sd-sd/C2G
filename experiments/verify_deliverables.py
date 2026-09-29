"""
Final deliverable audit for the C2G submission package.

Run this AFTER wait_and_finalize.sh finishes. It is the last line of defence: it checks
that every file the challenge asks for actually exists, that no placeholder/dummy content
survived into the shipped documents, that the shipped numbers agree with the result JSONs
they came from, and that the artifact respects the 16,000,000 byte cap.

Exit code 0 = clean. Non-zero = at least one FAIL; warnings alone do not fail.

Usage:
    python verify_deliverables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
LIMIT = 16_000_000

fails: list[str] = []
warns: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    fails.append(msg)


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")
    warns.append(msg)


def chk(cond: bool, msg: str, soft: bool = False) -> bool:
    if cond:
        ok(msg)
    elif soft:
        warn(msg)
    else:
        fail(msg)
    return bool(cond)


print("=" * 78)
print("1. REQUIRED DELIVERABLES PRESENT")
print("=" * 78)
REQUIRED = [
    "README.md",
    "lishengdan_C2G_方案草案.md",
    "lishengdan_C2G_方案设计.md",
    "lishengdan_C2G_train_gpt.py",
    "lishengdan_C2G_eval_bpb.py",
    "lishengdan_C2G_拿来说明.md",
    "lishengdan_C2G_AI日志.md",
    "lishengdan_C2G_AAR.md",
    "lishengdan_C2G_ablation.md",
    "lishengdan_C2G_leaderboard.md",
    "lishengdan_C2G_tech_report.pdf",
    "lishengdan_C2G_submission.json",
]
for rel in REQUIRED:
    p = ROOT / rel
    chk(p.exists() and p.stat().st_size > 0,
        f"{rel}  ({p.stat().st_size:,} B)" if p.exists() else f"MISSING: {rel}")

# The tarball may live at the root, under artifact/, or both. At least one must exist.
tars = [ROOT / "lishengdan_C2G_submission.tar.gz",
        ROOT / "artifact" / "lishengdan_C2G_submission.tar.gz"]
exist_tars = [p for p in tars if p.exists()]
chk(bool(exist_tars), f"submission tarball ({[str(p.relative_to(ROOT)) for p in exist_tars]})")

logs = sorted((ROOT / "lishengdan_C2G_logs").glob("*.txt"))
chk(len(logs) >= 30, f"run logs copied: {len(logs)} files", soft=(len(logs) > 0))

print()
print("=" * 78)
print("2. NO PLACEHOLDER CONTENT IN SHIPPED DOCUMENTS")
print("=" * 78)
# "dummy" is deliberately NOT in this list: the SentencePiece recipe legitimately contains
# `add_dummy_prefix=False`, so including it produced a permanent false positive. A scanner
# that cries wolf on every real run gets ignored, which is worse than not having one.
# Only markers that cannot appear in correct prose are listed here.
BAD = ["PLACEHOLDER", "placeholder", "TODO", "FIXME", "XXX", "lorem ipsum", "待填", "待补",
       "TBD", "占位"]
# A document that legitimately *narrates* placeholder history (e.g. the AI log explaining a
# placeholder-derived artifact it found and deleted) can declare that with this marker. An
# explicit opt-out keeps the check meaningful: a stray placeholder in a shipping document
# still warns, while a deliberate discussion of one no longer produces permanent noise.
ALLOW_MARKER = "allow-placeholder-mentions"
for ext in ("*.md",):
    for p in sorted(ROOT.glob(ext)):
        txt = p.read_text(encoding="utf-8", errors="replace")
        if ALLOW_MARKER in txt:
            ok(f"{p.name}: placeholder scan skipped (declares {ALLOW_MARKER})")
            continue
        hits = [b for b in BAD if b in txt]
        if hits:
            warn(f"{p.name}: contains {hits}")
ok("placeholder scan complete")

print()
print("=" * 78)
print("3. ARTIFACT BYTE CAP")
print("=" * 78)
if exist_tars:
    for p in exist_tars:
        n = p.stat().st_size
        chk(n <= LIMIT, f"{p.relative_to(ROOT)} = {n:,} B  (cap {LIMIT:,}, "
                        f"headroom {LIMIT - n:,})")
sub_path = ROOT / "lishengdan_C2G_submission.json"
sub = json.loads(sub_path.read_text(encoding="utf-8")) if sub_path.exists() else {}

print()
print("=" * 78)
print("4. SHIPPED NUMBERS AGREE WITH THE RESULT JSONs THEY CAME FROM")
print("=" * 78)
arm = sub.get("arm", "?")
seed_results = sub.get("seed_results", {})
if arm != "?":
    print(f"  shipped arm: {arm}")
for seed_str, s in seed_results.items():
    rid = f"{arm}_seed{seed_str}"
    rp = RES / f"{rid}.json"
    if not rp.exists():
        warn(f"{rid}.json not found - cannot cross-check")
        continue
    r = json.loads(rp.read_text(encoding="utf-8"))
    chk(abs(r["final_bpb"] - s["val_bpb"]) < 1e-6,
        f"{rid}: final_bpb {r['final_bpb']:.6f} == manifest {s['val_bpb']:.6f}")

# The manifest's headline number must be the median-seed run it claims to deploy.
if seed_results and sub.get("val_bpb") is not None:
    vals = sorted(v["val_bpb"] for v in seed_results.values())
    median = vals[len(vals) // 2]
    chk(abs(median - sub["val_bpb"]) < 1e-6,
        f"manifest val_bpb {sub['val_bpb']:.6f} is the median seed ({median:.6f})")

# quantisation tax must be non-negative and small; a negative tax means we mis-defined it.
if "quantization_tax" in sub:
    t = sub["quantization_tax"]
    chk(t >= -1e-6, f"quantization_tax = {t:+.6f} (>=0 means quantisation costs BPB)")
    chk(t < 0.25, f"quantization_tax = {t:+.6f} (sanity: a benign int6/int8 tax)")

# The manifest must not describe its own export rule self-contradictingly. `quant_method`
# names the algorithm and `quant_clip` names the clip rule; a name that claims SDClip while
# the clip rule is percentile is exactly the "label does not match the fact" defect this
# audit exists to catch -- and it shipped once, in `name`, before being fixed.
q = sub.get("quantization", {})
if q:
    _m, _c = str(q.get("method", "")), str(q.get("clip_rule", ""))
    chk(not (("sdclip" in _m) and _c != "sdclip"),
        f"manifest method='{_m}' is consistent with clip_rule='{_c}'")
    lbl = str(sub.get("name", ""))
    chk(not (("percentile" in lbl) and ("SDCLIP" in lbl.upper())),
        "manifest name does not both claim and deny SDClip")
    if "method_declared" in q:
        chk(str(q["method_declared"]).lower() in ("rtn", "rtn_sdclip", "gptq"),
            f"method_declared='{q['method_declared']}' is a recognised export token")

# Root-level tarball must be the one the manifest describes, byte for byte.
_root_tar = ROOT / "lishengdan_C2G_submission.tar.gz"
_art_tar = ROOT / "artifact" / "lishengdan_C2G_submission.tar.gz"
if _root_tar.exists() and _art_tar.exists():
    chk(_root_tar.read_bytes() == _art_tar.read_bytes(),
        "root tarball is byte-identical to artifact/ tarball")
    chk(_root_tar.stat().st_size == sub.get("artifact", {}).get("total_bytes"),
        f"root tarball {_root_tar.stat().st_size:,} B == manifest artifact.total_bytes")

# Any artifact byte count hardcoded in prose must equal the real one: this figure has
# gone stale before, whenever the trainer source changed.
_rd = (ROOT / "README.md")
if _rd.exists():
    _txt = _rd.read_text(encoding="utf-8")
    _declared = sub.get("artifact", {}).get("total_bytes")
    _hits = re.findall(r"artifact\s*\*\*([\d,]+)\s*B\*\*", _txt)
    for _h in _hits:
        chk(int(_h.replace(",", "")) == _declared,
            f"README claims artifact {_h} B == manifest {_declared:,} B")
    if not _hits:
        warn("README quotes no explicit artifact byte count (nothing to cross-check)")

print()
print("=" * 78)
print("5. EXPERIMENT COUNT CONSISTENCY ACROSS DOCUMENTS")
print("=" * 78)
n_runs = len(list(RES.glob("*_seed*.json")))
arms_present = sorted({p.stem.rsplit("_seed", 1)[0] for p in RES.glob("*_seed*.json")})
ok(f"result JSONs on disk: {n_runs} runs / {len(arms_present)} arms")
if n_runs != 34:
    warn(f"expected 34 runs in the preregistered matrix, found {n_runs}")

# Any document claiming a different arm/run count than reality is a real inconsistency.
#
# Two sources of legitimate noise are excluded explicitly:
#   * sub-block counts ("the quantisation block only had 3 arms") - only totals are checked,
#     so we require the claimed number to be plausibly a total (>=5 arms / >=10 runs);
#   * the pre-registration's original text, which is deliberately preserved verbatim and
#     marked as superseded by a post-hoc note. Rewriting history there would be worse than
#     the mismatch, so lines carrying those markers are skipped.
HISTORY_MARKERS = ("初版", "预注册", "规划时", "约 32", "约32", "原计划", "偏离",
                   "初稿", "AI 的输出", "曾经", "原本", "共约")
for p in sorted(ROOT.glob("*.md")):
    txt = p.read_text(encoding="utf-8", errors="replace")
    for pat, actual, label, floor in (
        (r"(\d+)\s*(?:个)?\s*(?:实验)?臂", len(arms_present), "arms", 5),
        (r"(\d+)\s*次运行", n_runs, "runs", 10),
    ):
        for m in re.finditer(pat, txt):
            claimed = int(m.group(1))
            if claimed < floor or claimed == actual:
                continue
            line = txt[txt.rfind("\n", 0, m.start()) + 1: txt.find("\n", m.end())]
            if any(h in line for h in HISTORY_MARKERS):
                continue
            warn(f"{p.name}: claims {claimed} {label}, disk has {actual} "
                 f"-> \"{line.strip()[:110]}\"")
ok("arm/run-count scan complete")

print()
print("=" * 78)
print("6. SUMMARY / POWER ARTIFACTS EXIST AND PARSE")
print("=" * 78)
for rel in ("results/summary.json", "results/power.json", "results/prediction_scorecard.md"):
    p = HERE / rel
    if not p.exists():
        fail(f"missing {rel}")
        continue
    if p.suffix == ".json":
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            chk(len(obj) > 0, f"{rel} parses, {len(obj)} entries")
        except Exception as e:  # noqa: BLE001
            fail(f"{rel} does not parse: {e}")
    else:
        chk(p.stat().st_size > 200, f"{rel} ({p.stat().st_size:,} B)")

sm = HERE / "results" / "summary.json"
if sm.exists():
    s = json.loads(sm.read_text(encoding="utf-8"))
    if not isinstance(s, dict) or not s:
        fail("summary.json is empty or not an object")
    else:
        # Presence of the right KEY is not enough: the placeholder file written during
        # development also had final_bpb_mean. The only real test is that the summary's
        # arm set equals the arms on disk AND its numbers reproduce from the result JSONs.
        if set(s) != set(arms_present):
            fail(f"summary.json covers {sorted(s)}, disk has {sorted(arms_present)} "
                 "- summary is stale or partial")
        else:
            ok(f"summary.json covers exactly the {len(s)} arms on disk")
        recomputed: dict[str, float] = {}
        for p in RES.glob("*_seed*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            recomputed.setdefault(r["run_id"].rsplit("_seed", 1)[0], []).append(r["final_bpb"])
        bad = []
        for arm_name, v in s.items():
            if arm_name not in recomputed or "final_bpb_mean" not in v:
                bad.append(arm_name)
                continue
            truth = sum(recomputed[arm_name]) / len(recomputed[arm_name])
            if abs(v["final_bpb_mean"] - truth) > 1e-6:
                bad.append(f"{arm_name}: summary {v['final_bpb_mean']:.6f} vs "
                           f"recomputed {truth:.6f}")
        chk(not bad, f"summary.json means reproduce from result JSONs"
            if not bad else f"summary.json disagrees with result JSONs: {bad[:3]}")

print()
print("=" * 78)
print("7. FIGURES + PDF SIZE SANITY")
print("=" * 78)
for f in ("fig_ablation.png", "fig_quant_tax.png", "fig_curves.png",
          "fig_tokenizer.png", "fig_budget.png"):
    p = HERE / "figures" / f
    if not p.exists():
        p = HERE / f
    chk(p.exists() and p.stat().st_size > 3000, f"{f}")

pdf = ROOT / "lishengdan_C2G_tech_report.pdf"
if pdf.exists():
    chk(pdf.stat().st_size > 20_000, f"tech_report.pdf = {pdf.stat().st_size:,} B")
    # CHALLENGE.md requires the Level-4 deliverable to be a "6+ page" English report. An
    # earlier build shipped 4 pages, so the requirement is asserted rather than assumed.
    # Pages are counted from the PDF's own /Count without needing a PDF library.
    import re as _re
    _counts = [int(x) for x in _re.findall(rb"/Count\s+(\d+)", pdf.read_bytes())]
    _pages = max(_counts) if _counts else 0
    chk(_pages >= 6, f"tech_report.pdf is {_pages} pages (challenge requires 6+)")

print()
print("=" * 78)
print("7b. CHALLENGE WORD-COUNT AND NAMING REQUIREMENTS")
print("=" * 78)
import re as _re2


def _cjk(p: Path) -> int:
    return len(_re2.findall(r"[\u4e00-\u9fff]", p.read_text(encoding="utf-8", errors="replace")))


for rel, floor, label in (("lishengdan_C2G_方案草案.md", 500, "draft (>=500 chars)"),
                          ("lishengdan_C2G_方案设计.md", 1000, "RFC (>=1000 chars)")):
    p = ROOT / rel
    if p.exists():
        n = _cjk(p)
        chk(n >= floor, f"{rel}: {n} CJK chars ({label})")

# Naming convention from CHALLENGE.md: 姓名拼音_C2G_内容描述.扩展名
# Directories are exempt: lishengdan_C2G_logs/ follows the same convention without an
# extension, and flagging it would be a false positive that trains the reader to ignore
# this check.
_bad_names = [p.name for p in ROOT.glob("lishengdan_C2G_*")
              if p.is_file()
              and not _re2.match(r"^lishengdan_C2G_.*\.(md|py|json|pdf|tar\.gz)$", p.name)]
chk(not _bad_names, "all deliverable files follow 姓名拼音_C2G_内容描述.扩展名"
    if not _bad_names else f"naming violations: {_bad_names}")

print()
print("=" * 78)
print("8. DERIVED-ARTIFACT FRESHNESS  (catches stale files built from placeholder data)")
print("=" * 78)
# Every derived artifact must be at least as new as its input. During development a
# placeholder summary.json produced a power.json full of invented numbers, which would
# have flowed straight into the PDF. Age ordering is the cheapest way to catch that class
# of bug automatically.
def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


newest_run = max((mtime(p) for p in RES.glob("*_seed*.json")), default=0.0)
sat = mtime(HERE / "results" / "summary.json")
fats = max((mtime(f) for f in (HERE / "figures").glob("*.png")), default=0.0)
pat = mtime(HERE / "results" / "power.json")
sub_at = mtime(ROOT / "lishengdan_C2G_submission.json")
sc_at = mtime(HERE / "results" / "prediction_scorecard.md")

chk(sat >= newest_run, "summary.json is at least as new as the newest run result")
chk(pat >= sat, "power.json is at least as new as summary.json")
chk(sc_at >= sat, "prediction_scorecard.md is at least as new as summary.json")
chk(fats >= sat, "figures are at least as new as summary.json")
chk(mtime(pdf) >= max(fats, pat), "tech_report.pdf is at least as new as figures and power.json")
chk(sub_at >= newest_run, "submission.json is at least as new as the newest run result")

print()
print("=" * 78)
print(f"RESULT: {len(fails)} failures, {len(warns)} warnings")
print("=" * 78)
for f in fails:
    print(f"  FAIL: {f}")
for w in warns:
    print(f"  warn: {w}")
raise SystemExit(1 if fails else 0)
