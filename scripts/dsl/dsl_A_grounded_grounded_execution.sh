#!/usr/bin/env bash
# DSL-A: proposer=grounded, solver=grounded, gate=execution
# Fully grounded stable baseline. Mirrors code_o Run A.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/base.sh"

export RUN_GROUP="pred_dsl_o__solver-grounded__proposer-grounded-hard__program-execution__qwen3-4b-base"
export EXP_NAME="selfplay_grpo_qwen3_4b_pred_dsl_o_solver_grounded_proposer_grounded_hard_program_execution_${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export SOLVER_REWARD_MODE=grounded
export PROPOSER_REWARD_MODE=grounded

export PROGRAM_VALIDITY_MODE=execution
export PROGRAM_VALIDITY_CHECK_DETERMINISM=False
export PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT=True
export PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT=True

exec bash "${SCRIPT_DIR}/../run_grpo_autoresume.sh" \
  "selfplay.seed_dataset=${DSL_SEED_DATASET}" \
  "$@"
