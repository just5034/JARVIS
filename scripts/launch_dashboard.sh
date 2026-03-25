#!/usr/bin/env bash
# Launch TensorBoard dashboard for monitoring JARVIS training on Delta.
#
# Usage (on Delta login node):
#   bash scripts/launch_dashboard.sh
#
# Then on your LOCAL machine, open the SSH tunnel:
#   ssh -L 6006:localhost:6006 jhill5@login.delta.ncsa.illinois.edu
#
# Then open http://localhost:6006 in your browser.

set -euo pipefail

LOG_DIR="${TB_LOG_DIR:-/scratch/bgde-delta-gpu/tb_logs}"
PORT="${TB_PORT:-6006}"

echo "=== JARVIS Training Dashboard (TensorBoard) ==="
echo ""
echo "Log dir: ${LOG_DIR}"
echo "Port:    ${PORT}"
echo ""

# Create log dir if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "Starting TensorBoard..."
echo ""
echo "============================================"
echo "  On your LOCAL machine, run:"
echo ""
echo "  ssh -L ${PORT}:localhost:${PORT} jhill5@login.delta.ncsa.illinois.edu"
echo ""
echo "  Then open: http://localhost:${PORT}"
echo "============================================"
echo ""

tensorboard --logdir "${LOG_DIR}" --host 0.0.0.0 --port "${PORT}"
