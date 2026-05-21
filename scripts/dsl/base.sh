#!/usr/bin/env bash
# Base config for all DSL (pred_dsl_o) experiments.
# Source this file, then set SOLVER_REWARD_MODE / PROPOSER_REWARD_MODE /
# PROGRAM_VALIDITY_MODE as needed, then exec run_grpo_autoresume.sh.
#
# Mirrors the Qwen3-4B-Base / 8-GPU setup used for the existing code_o runs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
export TRAIN_FILES=${TRAIN_FILES:-data/code_reason/test_answer.parquet}
export VAL_FILES=${VAL_FILES:-data/dsl_val.parquet}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1312}
# DSL prompts are shorter than code_o: proposer ~800 tokens, solver ~450 tokens
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
# DSL solver only needs integer + short chain-of-thought
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export CONTENT_MAX_LENGTH=${CONTENT_MAX_LENGTH:-1400}

export LR=${LR:-1e-6}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}
export ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
export TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
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
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}

export EXTRACTION_TYPE=${EXTRACTION_TYPE:-none}
export MATH_METRIC=${MATH_METRIC:-math_verify}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}

export UPDATE_ITERATION=${UPDATE_ITERATION:-1}
export N_SAMPLES=${N_SAMPLES:-8}

# DSL-specific
export TASK=pred_dsl_o
export EXECUTOR=dsl

# Proposer training: hard = reward(1 - solver_accuracy), same as code_o runs
export PROPOSER_INTRINSIC_MODE=${PROPOSER_INTRINSIC_MODE:-hard}
export TRAIN_PROPOSE=${TRAIN_PROPOSE:-True}

export PRED_DATA_MIX_STRATEGY=${PRED_DATA_MIX_STRATEGY:-max_new}
export SEED_BATCH_FACTOR=${SEED_BATCH_FACTOR:-4}
export VALID_PROGRAM_FILTER=${VALID_PROGRAM_FILTER:-all}
export MAX_PROGRAMS=${MAX_PROGRAMS:-16384}
export GEN_DATA_PROBABILITIES_STRATEGY=${GEN_DATA_PROBABILITIES_STRATEGY:-uniform}
export CODE_F_REWARD_TYPE=${CODE_F_REWARD_TYPE:-binary}

export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}

# DSL seed dataset (required — no _init_seed_dataset support for dsl_o)
DSL_SEED_DATASET="${ROOT_DIR}/data/dsl_seed_hard.jsonl"
