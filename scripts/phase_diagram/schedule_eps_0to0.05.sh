#!/usr/bin/env bash
# Adaptive ε schedule: resume from II+exec (ε=0) checkpoint at step 160,
# switch to ε=0.05 for the remaining training.
#
# Usage:
#   CHECKPOINT_DIR=/path/to/step160/checkpoint bash scripts/phase_diagram/schedule_eps_0to0.05.sh
#
# The checkpoint should be from the II+exec run (intrinsic solver + intrinsic proposer + execution gate).
# This script resumes from that checkpoint but changes the gate to execution_noisy with leak_rate=0.05.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/base.sh"

if [[ -z "${RESUME_FROM_PATH:-}" ]]; then
  echo "ERROR: RESUME_FROM_PATH must be set to the global_step_NNN checkpoint of II+exec (ε=0)."
  echo "Usage: RESUME_FROM_PATH=/path/to/global_step_150 bash $0"
  exit 1
fi

export PROGRAM_VALIDITY_LEAK_RATE=0.05
export RESUME_MODE=resume_path
export RUN_GROUP="pred_code_o__solver-intrinsic_self_consistency__proposer-intrinsic-hard__program-schedule_eps0to0.05__qwen3-4b-base"
export EXP_NAME="selfplay_grpo_qwen3_4b_pred_code_o_II_schedule_eps0to0.05_${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/../run_grpo.sh" \
  eval.intrinsic_val_freq=2 \
  trainer.resume_from_path="${RESUME_FROM_PATH}" \
  "$@"
