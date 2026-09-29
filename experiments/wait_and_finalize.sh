#!/usr/bin/env bash
# Wait for the ablation matrix to finish, then run the whole finalisation pipeline.
# Every step's exit code is logged; a failure does not abort the later steps, so one
# bad step still leaves the rest of the artifacts refreshed and visible in the log.
cd "$(dirname "$0")" || exit 1

# Locate a Python interpreter without hardcoding one machine's venv path. The interpreter
# must be able to import torch, sentencepiece, brotli, reportlab and matplotlib, so the
# dedicated venv is tried first; anything else is accepted only if it already has torch.
PY="${C2G_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in /d/c2g_env/Scripts/python.exe /d/c2g_env/bin/python python3 python; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c "import torch" >/dev/null 2>&1; then PY="$cand"; break; fi
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "ERROR: no Python interpreter with torch found." >&2
  echo "Set C2G_PYTHON to one, e.g. C2G_PYTHON=/d/c2g_env/Scripts/python.exe" >&2
  exit 1
fi
echo "using interpreter: $PY"

LOG=finalize.log
: > "$LOG"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for matrix (target 34 result JSONs)"
while true; do
  n=$(ls results/*_seed*.json 2>/dev/null | wc -l)
  alive=$(tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -c "python.exe")
  if [ "$n" -ge 34 ]; then log "all 34 results present"; break; fi
  if [ "$alive" -eq 0 ]; then log "no python.exe alive and n=$n -- matrix stopped early"; break; fi
  sleep 60
done

log "results: $(ls results/*_seed*.json 2>/dev/null | wc -l)/34"

run() {
  log "=== $* ==="
  "$PY" "$@" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "!!! STEP FAILED rc=$rc : $*"; else log "ok: $1"; fi
}

run run_matrix.py --summarize-only
run analyze_results.py
run write_ablation_report.py
run make_figures.py
run make_report.py
run package_submission.py

# The challenge's deliverable table names the artifact 姓名_C2G_submission.tar.gz and puts
# it at the submission root; the packager writes it under artifact/. Mirror it so both
# locations agree byte-for-byte.
log "=== mirror tarball to submission root ==="
if cp -f ../artifact/lishengdan_C2G_submission.tar.gz ../lishengdan_C2G_submission.tar.gz 2>>"$LOG"; then
  log "ok: tarball mirrored to root"
else
  log "!!! could not mirror tarball"
fi

# Final audit: file presence, byte cap, shipped-vs-measured numbers, doc/count consistency,
# and freshness of every derived artifact. Non-zero exit means something is actually wrong.
run verify_deliverables.py

log "PIPELINE COMPLETE (audit exit code recorded above)"
ls -la ../artifact ../lishengdan_C2G_submission.json ../lishengdan_C2G_logs 2>&1 | tail -20 | tee -a "$LOG"
