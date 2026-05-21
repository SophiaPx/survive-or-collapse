#!/usr/bin/env bash
# DSL-C: proposer=intrinsic, solver=intrinsic, gate=execution
# Gate rescues intrinsic/intrinsic. Mirrors code_o Run C.
# Key contrast with Run B: same reward modes, different gate.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/base.sh"

export RUN_GROUP="pred_dsl_o__solver-intrinsic_self_consistency__proposer-intrinsic-hard__program-execution__qwen3-4b-base"
export EXP_NAME="selfplay_grpo_qwen3_4b_pred_dsl_o_solver_intrinsic_self_consistency_proposer_intrinsic_hard_program_execution_${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export SOLVER_REWARD_MODE=intrinsic_self_consistency
export PROPOSER_REWARD_MODE=intrinsic

export PROGRAM_VALIDITY_MODE=execution
export PROGRAM_VALIDITY_CHECK_DETERMINISM=False
export PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT=True
export PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT=True

exec bash "${SCRIPT_DIR}/../run_grpo_autoresume.sh" \
  eval.intrinsic_val_freq=2 \
  "selfplay.seed_dataset=${DSL_SEED_DATASET}" \
  "$@"
