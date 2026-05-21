#!/usr/bin/env bash
# Sequential runner for phase diagram experiments.
# Waits for an optional PID to finish, then runs remaining ε points in order.
# Usage: bash run_queue.sh [WAIT_PID]

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WAIT_PID=${1:-}
if [[ -n "${WAIT_PID}" ]]; then
  echo "[queue] Waiting for PID ${WAIT_PID} (eps=0.05) to finish..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    true
  done
  echo "[queue] PID ${WAIT_PID} done."
fi

for EPS in 0.10 0.20 0.40 0.70; do
  SCRIPT="${SCRIPT_DIR}/phase_eps_${EPS}.sh"
  echo ""
  echo "[queue] ============================================================"
  echo "[queue] Starting eps=${EPS} at $(date)"
  echo "[queue] ============================================================"
  ENABLE_WANDB=1 bash "${SCRIPT}"
  echo "[queue] eps=${EPS} finished at $(date)"
done

echo "[queue] All phase diagram experiments complete."
