#!/usr/bin/env bash
set -uo pipefail

REPO=/path/to/azr-grpo
# Resume the gate-relaxation schedules from a stable strict-gate (eps=0) checkpoint.
# Point CKPT at one of your own II+exec runs that has completed >=150 steps.
# Example layout: ${REPO}/artifacts/runs/<run_name>/checkpoints/.../global_step_150
CKPT="${REPO}/path/to/your/strict_gate_run/checkpoints/global_step_150"
LOG_DIR="${REPO}/artifacts/runs/selfplay_grpo/_p0ext_runner_logs"
mkdir -p "${LOG_DIR}"

export ENABLE_WANDB=1
export RESUME_FROM_PATH="${CKPT}"

cd "${REPO}"

for EPS in 0.05 0.10 0.20; do
  TS=$(date +%Y%m%d_%H%M%S)
  TAG="schedule_eps0to${EPS}"
  RUNNER_LOG="${LOG_DIR}/${TAG}_${TS}.log"
  echo "===========================================" | tee -a "${RUNNER_LOG}"
  echo "[$(date)] START ${TAG}" | tee -a "${RUNNER_LOG}"
  echo "  RESUME_FROM_PATH=${RESUME_FROM_PATH}" | tee -a "${RUNNER_LOG}"
  echo "===========================================" | tee -a "${RUNNER_LOG}"

  bash "${REPO}/scripts/phase_diagram/schedule_eps_0to${EPS}.sh" 2>&1 | tee -a "${RUNNER_LOG}"
  RC=${PIPESTATUS[0]}

  echo "[$(date)] END ${TAG} (rc=${RC})" | tee -a "${RUNNER_LOG}"
  if [[ ${RC} -ne 0 ]]; then
    echo "[$(date)] ABORT: ${TAG} failed with rc=${RC}, stopping serial chain" | tee -a "${RUNNER_LOG}"
    exit ${RC}
  fi

  sleep 15
done

echo "[$(date)] ALL P0-ext schedules completed."
