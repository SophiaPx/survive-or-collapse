#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SELFPLAY_PYTHON=${SELFPLAY_PYTHON:-python}
if [[ ! -x "${SELFPLAY_PYTHON}" ]]; then
  echo "[selfplay-grpo] Python executable not found at ${SELFPLAY_PYTHON}" >&2
  exit 1
fi
SELFPLAY_BIN_DIR=$(dirname "${SELFPLAY_PYTHON}")
export PATH="${SELFPLAY_BIN_DIR}:${PATH}"

CONDA_GCC=${SELFPLAY_BIN_DIR}/x86_64-conda-linux-gnu-gcc
CONDA_GXX=${SELFPLAY_BIN_DIR}/x86_64-conda-linux-gnu-g++
if [[ -z "${CC:-}" && -x "${CONDA_GCC}" ]]; then
  export CC="${CONDA_GCC}"
fi
if [[ -z "${CXX:-}" && -x "${CONDA_GXX}" ]]; then
  export CXX="${CONDA_GXX}"
fi
if [[ -z "${CUDAHOSTCXX:-}" && -n "${CXX:-}" ]]; then
  export CUDAHOSTCXX="${CXX}"
fi

VERL_ROOT=${VERL_ROOT:-/path/to/verl}
VERL_SENTINEL=verl/trainer/ppo/ray_trainer.py

verl_root_has_dataproto() {
  local candidate_root="$1"
  if [[ ! -f "${candidate_root}/${VERL_SENTINEL}" ]]; then
    return 1
  fi
  PYTHONPATH="${candidate_root}" "${SELFPLAY_PYTHON}" - <<'PY2' >/dev/null 2>&1
try:
    import verl
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if hasattr(verl, "DataProto") else 1)
PY2
}

if ! verl_root_has_dataproto "${VERL_ROOT}"; then
  if [[ -f "${VERL_ROOT}/${VERL_SENTINEL}" ]]; then
    echo "[selfplay-grpo] Ignoring incompatible local verl at ${VERL_ROOT} because it does not export DataProto."
  fi
  INSTALLED_VERL_ROOT=$("${SELFPLAY_PYTHON}" - <<'PY2'
from pathlib import Path

try:
    import verl
except Exception:
    raise SystemExit(0)

if hasattr(verl, "DataProto"):
    print(Path(verl.__file__).resolve().parent.parent)
PY2
)
  if [[ -n "${INSTALLED_VERL_ROOT}" ]]; then
    if verl_root_has_dataproto "${INSTALLED_VERL_ROOT}"; then
      echo "[selfplay-grpo] Using installed verl from ${INSTALLED_VERL_ROOT} instead of ${VERL_ROOT}"
      VERL_ROOT=${INSTALLED_VERL_ROOT}
    fi
  fi
fi

if ! verl_root_has_dataproto "${VERL_ROOT}"; then
  echo "[selfplay-grpo] Compatible VERL root with DataProto not found at ${VERL_ROOT}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-0}
export RAY_LOGGING_LEVEL=${RAY_LOGGING_LEVEL:-DEBUG}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONPATH="${ROOT_DIR}:${VERL_ROOT}:${PYTHONPATH:-}"

COMPILER_VERSION_CMD=${CXX:-${CC:-gcc}}
if [[ -z "${VLLM_USE_FLASHINFER_SAMPLER:-}" ]]; then
  GCC_VERSION=$("${COMPILER_VERSION_CMD}" -dumpfullversion -dumpversion 2>/dev/null || true)
  GCC_MAJOR=${GCC_VERSION%%.*}
  if [[ -z "${GCC_VERSION}" ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "[selfplay-grpo] Disabling flashinfer sampler because compiler ${COMPILER_VERSION_CMD} is unavailable."
  elif [[ "${GCC_MAJOR}" =~ ^[0-9]+$ ]] && (( GCC_MAJOR < 9 )); then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "[selfplay-grpo] Disabling flashinfer sampler because compiler ${COMPILER_VERSION_CMD} reports ${GCC_VERSION}, older than 9."
  fi
fi

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  if [[ "${SELFPLAY_KEEP_PYTORCH_ALLOC_CONF:-0}" == "1" ]]; then
    echo "[selfplay-grpo] Keeping PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} (SELFPLAY_KEEP_PYTORCH_ALLOC_CONF=1)"
  else
    echo "[selfplay-grpo] Unsetting PYTORCH_CUDA_ALLOC_CONF due to vLLM incompatibility: ${PYTORCH_CUDA_ALLOC_CONF}"
    unset PYTORCH_CUDA_ALLOC_CONF
  fi
fi

MODEL_PATH=${MODEL_PATH:-~/models/deepseek-llm-7b-chat}
TRAIN_FILES=${TRAIN_FILES:-data/code_reason/test_answer.parquet}
VAL_FILES=${VAL_FILES:-data/code_reason/test_answer.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1312}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-6144}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8096}
CONTENT_MAX_LENGTH=${CONTENT_MAX_LENGTH:-5600}
LR=${LR:-1e-6}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-False}
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
ROLLOUT_N=${ROLLOUT_N:-16}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
PROJECT_NAME=${PROJECT_NAME:-selfplay_grpo}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
NNODES=${NNODES:-1}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
ENABLE_WANDB=${ENABLE_WANDB:-1}
TRAINER_LOGGER=${TRAINER_LOGGER:-}
if [[ -z "${TRAINER_LOGGER}" ]]; then
  case "${ENABLE_WANDB,,}" in
    0|false|no|off)
      TRAINER_LOGGER="['console']"
      ;;
    *)
      TRAINER_LOGGER="['console','wandb']"
      ;;
  esac
fi
RESUME_MODE=${RESUME_MODE:-auto}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-}
EXTRACTION_TYPE=${EXTRACTION_TYPE:-none}
MATH_METRIC=${MATH_METRIC:-math_verify}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
UPDATE_ITERATION=${UPDATE_ITERATION:-1}
EXECUTOR=${EXECUTOR:-qwq}
N_SAMPLES=${N_SAMPLES:-8}
TASK=${TASK:-pred_code_o}
REWARD_MODE=${REWARD_MODE:-grounded}
SOLVER_REWARD_MODE=${SOLVER_REWARD_MODE:-${REWARD_MODE}}
PROPOSER_REWARD_MODE=${PROPOSER_REWARD_MODE:-grounded}
PROPOSER_INTRINSIC_MODE=${PROPOSER_INTRINSIC_MODE:-none}
PROGRAM_VALIDITY_MODE=${PROGRAM_VALIDITY_MODE:-execution}
PROGRAM_VALIDITY_LEAK_RATE=${PROGRAM_VALIDITY_LEAK_RATE:-0.0}
PROGRAM_VALIDITY_SEED=${PROGRAM_VALIDITY_SEED:-42}
PROGRAM_VALIDITY_CHECK_DETERMINISM=${PROGRAM_VALIDITY_CHECK_DETERMINISM:-True}
PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT=${PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT:-True}
PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT=${PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT:-True}
SEED_BATCH_FACTOR=${SEED_BATCH_FACTOR:-4}
VALID_PROGRAM_FILTER=${VALID_PROGRAM_FILTER:-all}
MAX_PROGRAMS=${MAX_PROGRAMS:-16384}
GEN_DATA_PROBABILITIES_STRATEGY=${GEN_DATA_PROBABILITIES_STRATEGY:-uniform}
PRED_DATA_MIX_STRATEGY=${PRED_DATA_MIX_STRATEGY:-max_new}
CODE_F_REWARD_TYPE=${CODE_F_REWARD_TYPE:-binary}
TRAIN_PROPOSE=${TRAIN_PROPOSE:-True}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}

sanitize_slug() {
  local value="$1"
  value=${value//\~/$HOME}
  value=$(basename "${value}")
  value=${value// /_}
  value=$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')
  value=$(printf '%s' "${value}" | sed 's/[^a-z0-9._-]/_/g; s/__*/_/g; s/^_//; s/_$//')
  printf '%s' "${value}"
}

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
MODEL_SLUG=${MODEL_SLUG:-$(sanitize_slug "${MODEL_PATH}")}
TASK_SLUG=${TASK_SLUG:-$(sanitize_slug "${TASK}")}
REWARD_SLUG=${REWARD_SLUG:-$(sanitize_slug "${REWARD_MODE}")}
RUN_GROUP=${RUN_GROUP:-"${TASK_SLUG}__${REWARD_SLUG}__${MODEL_SLUG}"}
EXP_NAME_INPUT=${EXP_NAME:-}
EXP_NAME=${EXP_NAME_INPUT:-"${RUN_GROUP}__${RUN_TIMESTAMP}"}
ARTIFACTS_ROOT=${ARTIFACTS_ROOT:-"${ROOT_DIR}/artifacts"}
RUN_ROOT=${RUN_ROOT:-"${ARTIFACTS_ROOT}/runs/${PROJECT_NAME}/${RUN_GROUP}/${EXP_NAME}"}
if [[ -n "${RESUME_RUN_ROOT}" ]]; then
  RUN_ROOT=${RESUME_RUN_ROOT}
  if [[ -z "${EXP_NAME_INPUT}" ]]; then
    EXP_NAME=$(basename "${RUN_ROOT}")
  fi
fi
LOG_DIR=${LOG_DIR:-"${RUN_ROOT}/logs"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${RUN_ROOT}/checkpoints"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${RUN_ROOT}/rollout_data"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${RUN_ROOT}/validation_data"}
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/train.log"}
LATEST_DIR=${LATEST_DIR:-"${ARTIFACTS_ROOT}/latest/${PROJECT_NAME}"}
LATEST_LINK=${LATEST_LINK:-"${LATEST_DIR}/${RUN_GROUP}"}
WANDB_RUN_ID_FILE=${WANDB_RUN_ID_FILE:-"${RUN_ROOT}/wandb_run_id.txt"}

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}" "${LATEST_DIR}"
ln -sfn "${RUN_ROOT}" "${LATEST_LINK}"

exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -z "${WANDB_RUN_ID}" && -f "${WANDB_RUN_ID_FILE}" ]]; then
  WANDB_RUN_ID=$(<"${WANDB_RUN_ID_FILE}")
  WANDB_RUN_ID=${WANDB_RUN_ID//$'\r'/}
  WANDB_RUN_ID=${WANDB_RUN_ID//$'\n'/}
fi
export AZR_WANDB_RUN_ID_FILE="${WANDB_RUN_ID_FILE}"

echo "[selfplay-grpo] Python       : ${SELFPLAY_PYTHON}"
echo "[selfplay-grpo] VERL root    : ${VERL_ROOT}"
echo "[selfplay-grpo] CC           : ${CC:-<unset>}"
echo "[selfplay-grpo] CXX          : ${CXX:-<unset>}"
echo "[selfplay-grpo] CUDAHOSTCXX  : ${CUDAHOSTCXX:-<unset>}"
echo "[selfplay-grpo] Project      : ${PROJECT_NAME}"
echo "[selfplay-grpo] Experiment   : ${EXP_NAME}"
echo "[selfplay-grpo] Run group    : ${RUN_GROUP}"
echo "[selfplay-grpo] Run root     : ${RUN_ROOT}"
echo "[selfplay-grpo] Logger       : ${TRAINER_LOGGER}"
echo "[selfplay-grpo] W&B run id   : ${WANDB_RUN_ID:-<unset>}"
echo "[selfplay-grpo] W&B id file  : ${WANDB_RUN_ID_FILE}"
echo "[selfplay-grpo] Log file     : ${LOG_FILE}"
echo "[selfplay-grpo] Checkpoints  : ${CHECKPOINT_DIR}"
echo "[selfplay-grpo] Validation   : ${VALIDATION_DATA_DIR}"
echo "[selfplay-grpo] Latest link  : ${LATEST_LINK}"

EXTRA_ARGS=("$@")
if [[ -n "${WANDB_RUN_ID}" && "${TRAINER_LOGGER}" == *wandb* ]]; then
  EXTRA_ARGS+=(trainer.wandb_run_id="${WANDB_RUN_ID}")
fi

set -x
"${SELFPLAY_PYTHON}" -m selfplay_grpo.main_grpo \
  data.shuffle=True \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  selfplay.data_selection_strategy.content_max_length="${CONTENT_MAX_LENGTH}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING:-True} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${ACTOR_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.pretrained_tokenizer=True \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  +trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
  reward_fn.extraction_type="${EXTRACTION_TYPE}" \
  reward_fn.math_metric="${MATH_METRIC}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
  selfplay.data_selection_strategy.update_iteration="${UPDATE_ITERATION}" \
  selfplay.executor="${EXECUTOR}" \
  selfplay.ast_check=True \
  selfplay.reward.n_samples="${N_SAMPLES}" \
  selfplay.task="${TASK}" \
  selfplay.reward_mode="${SOLVER_REWARD_MODE}" \
  selfplay.solver_reward_mode="${SOLVER_REWARD_MODE}" \
  selfplay.proposer_reward_mode="${PROPOSER_REWARD_MODE}" \
  selfplay.reward.generation_reward_config.difficulty_reward.mode="${PROPOSER_INTRINSIC_MODE}" \
  selfplay.program_validity.mode="${PROGRAM_VALIDITY_MODE}" \
  selfplay.program_validity.leak_rate="${PROGRAM_VALIDITY_LEAK_RATE}" \
  selfplay.program_validity.seed="${PROGRAM_VALIDITY_SEED}" \
  selfplay.program_validity.check_determinism="${PROGRAM_VALIDITY_CHECK_DETERMINISM}" \
  selfplay.program_validity.store_execution_output="${PROGRAM_VALIDITY_STORE_EXECUTION_OUTPUT}" \
  selfplay.program_validity.require_execution_output_for_dataset="${PROGRAM_VALIDITY_REQUIRE_EXECUTION_OUTPUT}" \
  selfplay.pred_data_mix_strategy="${PRED_DATA_MIX_STRATEGY}" \
  selfplay.data_selection_strategy.seed_batch_factor="${SEED_BATCH_FACTOR}" \
  selfplay.data_selection_strategy.valid_program_filter="${VALID_PROGRAM_FILTER}" \
  selfplay.data_selection_strategy.max_programs="${MAX_PROGRAMS}" \
  selfplay.gen_data_probabilities_strategy="${GEN_DATA_PROBABILITIES_STRATEGY}" \
  selfplay.reward.code_f_reward_type="${CODE_F_REWARD_TYPE}" \
  selfplay.train_propose="${TRAIN_PROPOSE}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${EXTRA_ARGS[@]}"
