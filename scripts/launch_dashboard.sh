#!/usr/bin/env bash
# Launch Aim dashboard for monitoring JARVIS training on Delta.
#
# Usage (on Delta login node):
#   bash scripts/launch_dashboard.sh
#
# Then on your LOCAL machine, open the SSH tunnel:
#   ssh -L 43800:localhost:43800 jhill5@login.delta.ncsa.illinois.edu
#
# Then open http://localhost:43800 in your browser.

set -euo pipefail

AIM_REPO="${AIM_REPO:-/scratch/bgde-delta-gpu/aim}"
PORT="${AIM_PORT:-43800}"

echo "=== JARVIS Training Dashboard (Aim) ==="
echo ""
echo "Aim repo: ${AIM_REPO}"
echo "Port:     ${PORT}"
echo ""

# Initialize repo if it doesn't exist
if [ ! -d "${AIM_REPO}/.aim" ]; then
    echo "Initializing Aim repo at ${AIM_REPO} ..."
    mkdir -p "${AIM_REPO}"
    aim init --repo "${AIM_REPO}"
fi

echo "Starting Aim dashboard..."
echo ""
echo "============================================"
echo "  On your LOCAL machine, run:"
echo ""
echo "  ssh -L ${PORT}:localhost:${PORT} jhill5@login.delta.ncsa.illinois.edu"
echo ""
echo "  Then open: http://localhost:${PORT}"
echo "============================================"
echo ""

aim up --repo "${AIM_REPO}" --host 0.0.0.0 --port "${PORT}"
