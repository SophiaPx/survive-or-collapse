#!/usr/bin/env bash
# Resume Schedule A (II + ε=0.05, started from baseline ε=0 step_150).
# The run crashed at step ~500 during validation; resume from its own latest
# checkpoint (auto mode) and continue training. Wandb run ID is loaded from
# the existing wandb_run_id.txt so the same run is appended to.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/base.sh"

REPO=/path/to/azr-grpo
EXISTING_RUN_ROOT="${REPO}/artifacts/runs/selfplay_grpo/pred_code_o__solver-intrinsic_self_consistency__proposer-intrinsic-hard__program-schedule_eps0to0.05__qwen3-4b-base/selfplay_grpo_qwen3_4b_pred_code_o_II_schedule_eps0to0.05_20260502_235839"

if [[ ! -d "${EXISTING_RUN_ROOT}/checkpoints" ]]; then
  echo "ERROR: ${EXISTING_RUN_ROOT}/checkpoints not found"
  exit 1
fi

export PROGRAM_VALIDITY_LEAK_RATE=0.05
export RESUME_MODE=auto
export RESUME_RUN_ROOT="${EXISTING_RUN_ROOT}"

exec bash "${SCRIPT_DIR}/../run_grpo.sh" \
  eval.intrinsic_val_freq=2 \
  "$@"
