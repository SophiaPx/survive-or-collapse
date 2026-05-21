#!/usr/bin/env bash
# Base config for Gate Leakage Rate Phase Diagram experiments (P0-new).
# Fixed: II configuration (intrinsic solver + intrinsic proposer + hard difficulty)
# Variable: PROGRAM_VALIDITY_MODE=execution_noisy, PROGRAM_VALIDITY_LEAK_RATE=ε
#
# ε=0.0  → execution gate (reuse existing II+exec run as endpoint)
# ε=1.0  → off gate (reuse existing II+off run as endpoint)
# ε∈{0.05, 0.10, 0.20, 0.40, 0.70} → new runs (created in this directory)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
export TRAIN_FILES=${TRAIN_FILES:-data/code_reason/test_answer.parquet}
export VAL_FILES=${VAL_FILES:-data/code_reason/test_answer.parquet}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1312}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export CONTENT_MAX_LENGTH=${CONTENT_MAX_LENGTH:-2800}

export LR=${LR:-1e-6}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}
export ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
export ROLLOUT_N=${ROLLOUT_N:-16}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}

export PROJECT_NAME=${PROJECT_NAME:-selfplay_grpo}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-50}
export ENABLE_WANDB=${ENABLE_WANDB:-1}
export RESUME_MODE=${RESUME_MODE:-auto}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}

export EXTRACTION_TYPE=${EXTRACTION_TYPE:-none}
export MATH_METRIC=${MATH_METRIC:-math_verify}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}

export UPDATE_ITERATION=${UPDATE_ITERATION:-1}
export EXECUTOR=${EXECUTOR:-qwq}
export N_SAMPLES=${N_SAMPLES:-8}
export TASK=${TASK:-pred_code_o}

# II configuration (intrinsic solver + intrinsic proposer + hard difficulty)
export SOLVER_REWARD_MODE=${SOLVER_REWARD_MODE:-intrinsic_self_consistency}
export REWARD_MODE=${REWARD_MODE:-intrinsic_self_consistency}
export PROPOSER_REWARD_MODE=${PROPOSER_REWARD_MODE:-intrinsic}
export PROPOSER_INTRINSIC_MODE=${PROPOSER_INTRINSIC_MODE:-hard}
export TRAIN_PROPOSE=${TRAIN_PROPOSE:-True}

# execution_noisy gate with configurable leak rate
# - mode=execution_noisy: valid programs always pass; invalid programs pass with prob leak_rate
# - require_execution_output=False: leaked programs (execution-failed) have no output but must
#   still enter the dataset; otherwise the leak is silently discarded downstream
export PROGRAM_VALIDITY_MODE=${PROGRAM_VALIDITY_MODE:-execution_noisy}
export PROGRAM_VALIDITY_LEAK_RATE=${PROGRAM_VALIDITY_LEAK_RATE:-0.0}
export PROGRAM_VALIDITY_SEED=${PROGRAM_VALIDITY_SEED:-1}
export PROGRAM_VALIDITY_CHECK_DETERMINISM=${PROGRAM_VALIDITY_CHECK_DETERMINISM:-True}
export PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT=${PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT:-True}
export PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT=${PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT:-False}

export PRED_DATA_MIX_STRATEGY=${PRED_DATA_MIX_STRATEGY:-max_new}
export SEED_BATCH_FACTOR=${SEED_BATCH_FACTOR:-4}
export VALID_PROGRAM_FILTER=${VALID_PROGRAM_FILTER:-all}
export MAX_PROGRAMS=${MAX_PROGRAMS:-16384}
export GEN_DATA_PROBABILITIES_STRATEGY=${GEN_DATA_PROBABILITIES_STRATEGY:-uniform}
export CODE_F_REWARD_TYPE=${CODE_F_REWARD_TYPE:-binary}

export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
