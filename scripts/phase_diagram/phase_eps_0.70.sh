#!/usr/bin/env bash
# Phase diagram: II+exec_noisy(ε=0.70)
# Gate leaks 70% of execution-invalid programs as valid.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/base.sh"

export PROGRAM_VALIDITY_LEAK_RATE=0.70
export RUN_GROUP="pred_code_o__solver-intrinsic_self_consistency__proposer-intrinsic-hard__program-execution_noisy_eps0.70__qwen3-4b-base"
export EXP_NAME="selfplay_grpo_qwen3_4b_pred_code_o_II_exec_noisy_eps0.70_${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/../run_grpo_autoresume.sh" \
  eval.intrinsic_val_freq=2 \
  "$@"
