#!/usr/bin/env bash
# Phase 4 prep — extract HEP problems from GRACE, build merged problem sets,
# print counts. Stops before sbatch so you can eyeball counts before burning SUs.
#
# Safe to run on a Delta login node — no GPU work, no SLURM submission.
# Idempotent: re-running just overwrites the JSONL outputs.
#
# Usage:
#   bash scripts/run_phase4_prep.sh
#
# After this prints the summary, submit trace generation manually:
#   sbatch scripts/run_trace_generation.sh        # physics, port 8192
#   sbatch scripts/run_code_trace_generation.sh   # code,    port 8193

set -euo pipefail

# ─── Config ───
JARVIS_DIR="$HOME/JARVIS"
GRACE_REPO="$HOME/grace"
DATA_DIR="/work/hdd/bgde/jhill5/data"
VENV="/work/hdd/bgde/jhill5/jarvis-venv"
TEACHER_MODEL="/projects/bgde/jhill5/models/qwen3.5-27b"

HEP_PHYSICS_JSONL="$DATA_DIR/hep_physics_problems.jsonl"
HEP_CODE_JSONL="$DATA_DIR/hep_code_problems.jsonl"
PHYSICS_JSONL="$DATA_DIR/physics_problems.jsonl"
CODE_JSONL="$DATA_DIR/code_problems.jsonl"

# ─── Pretty-print helpers ───
hr() { printf '%s\n' "────────────────────────────────────────────────────────────"; }
say() { printf '\n[phase4-prep] %s\n' "$*"; }

# ─── 0. Sync repo ───
say "syncing JARVIS repo"
cd "$JARVIS_DIR"
git fetch --quiet
LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse @{u})
if [ "$LOCAL_HEAD" != "$REMOTE_HEAD" ]; then
    echo "  pulling: $LOCAL_HEAD -> $REMOTE_HEAD"
    git pull --ff-only
else
    echo "  already up to date at $LOCAL_HEAD"
fi
git log --oneline -3

# ─── 1. Sanity-check paths ───
say "checking paths"
for p in "$GRACE_REPO" "$VENV/bin/activate" "$TEACHER_MODEL"; do
    if [ -e "$p" ]; then
        echo "  OK   $p"
    else
        echo "  MISS $p"
        echo "ERROR: required path missing — fix before continuing"
        exit 1
    fi
done
mkdir -p "$DATA_DIR"
echo "  OK   $DATA_DIR (created if missing)"

# ─── 2. Activate venv ───
say "activating venv"
module load python/3.13.5-gcc13.3.1
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HOME=/work/hdd/bgde/jhill5/hf_cache
echo "  python:  $(which python)"
echo "  version: $(python --version)"

# ─── 3. Extract HEP problems from GRACE ───
say "extracting HEP physics problems from GRACE"
python -m training.data.extract_hep_physics \
    --grace-repo "$GRACE_REPO" \
    --output "$HEP_PHYSICS_JSONL"

say "extracting HEP code problems from GRACE"
python -m training.data.extract_hep_code \
    --grace-repo "$GRACE_REPO" \
    --output "$HEP_CODE_JSONL"

# ─── 4. Build merged problem sets ───
say "building merged physics problem set"
python -m training.data.build_physics_problems \
    --grace-hep-jsonl "$HEP_PHYSICS_JSONL" \
    --output "$PHYSICS_JSONL"

say "building merged code problem set"
python -m training.data.build_code_problems \
    --grace-hep-jsonl "$HEP_CODE_JSONL" \
    --output "$CODE_JSONL"

# ─── 5. Summary ───
say "summary"
hr
printf '%-50s %s\n' "FILE" "LINES"
hr
for f in "$HEP_PHYSICS_JSONL" "$HEP_CODE_JSONL" "$PHYSICS_JSONL" "$CODE_JSONL"; do
    if [ -f "$f" ]; then
        printf '%-50s %s\n' "$f" "$(wc -l < "$f")"
    else
        printf '%-50s %s\n' "$f" "MISSING"
    fi
done
hr

# Per-source breakdown for the merged sets (helpful to verify HEP signal isn't
# drowned by ballast).
say "physics_problems.jsonl by source"
python -c "
import json, collections
c = collections.Counter()
with open('$PHYSICS_JSONL') as f:
    for line in f:
        c[json.loads(line).get('source', '?')] += 1
for k, v in c.most_common():
    print(f'  {v:5d}  {k}')
"

say "code_problems.jsonl by source"
python -c "
import json, collections
c = collections.Counter()
with open('$CODE_JSONL') as f:
    for line in f:
        c[json.loads(line).get('source', '?')] += 1
for k, v in c.most_common():
    print(f'  {v:5d}  {k}')
"

# ─── 6. Next steps ───
hr
cat <<'EOF'

PREP COMPLETE. To launch Phase 4 trace generation (~50 SU each, 24h max):

  sbatch scripts/run_trace_generation.sh        # physics, vLLM port 8192
  sbatch scripts/run_code_trace_generation.sh   # code,    vLLM port 8193
  squeue -u jhill5

If you'd rather de-risk with the physics job first, submit it alone, watch
/work/hdd/bgde/jhill5/logs/vllm-hep-traces-*.log for vLLM startup (20-30 min
on A100x4), confirm traces start writing to /work/hdd/bgde/jhill5/data/hep_traces,
then submit the code job.

EOF
