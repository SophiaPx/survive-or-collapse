#!/usr/bin/env bash
# DSL-D: proposer=intrinsic, solver=grounded, gate=off
# Grounded solver without gate. Mirrors code_o Run D.
# Key contrast with Run E: swap solver/proposer reward modes.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/base.sh"

export RUN_GROUP="pred_dsl_o__solver-grounded__proposer-intrinsic-hard__program-off__qwen3-4b-base"
export EXP_NAME="selfplay_grpo_qwen3_4b_pred_dsl_o_solver_grounded_proposer_intrinsic_hard_program_off_${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export SOLVER_REWARD_MODE=grounded
export PROPOSER_REWARD_MODE=intrinsic

export PROGRAM_VALIDITY_MODE=off
export PROGRAM_VALIDITY_CHECK_DETERMINISM=False
export PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT=True
export PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT=False

exec bash "${SCRIPT_DIR}/../run_grpo_autoresume.sh" \
  "selfplay.seed_dataset=${DSL_SEED_DATASET}" \
  "$@"
