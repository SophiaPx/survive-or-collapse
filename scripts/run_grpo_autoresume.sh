#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

AUTO_RESUME_MAX_RESTARTS=${AUTO_RESUME_MAX_RESTARTS:-20}
AUTO_RESUME_SLEEP_SECONDS=${AUTO_RESUME_SLEEP_SECONDS:-30}
RESUME_MODE=${RESUME_MODE:-auto}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

export RESUME_MODE
export RUN_TIMESTAMP

attempt=1
while true; do
  echo "[selfplay-grpo] Auto-resume attempt ${attempt}"
  set +e
  bash "${SCRIPT_DIR}/run_grpo.sh" "$@"
  status=$?
  set -e

  if [[ ${status} -eq 0 ]]; then
    echo "[selfplay-grpo] Training finished successfully on attempt ${attempt}."
    exit 0
  fi

  if [[ ${AUTO_RESUME_MAX_RESTARTS} -ge 0 && ${attempt} -ge ${AUTO_RESUME_MAX_RESTARTS} ]]; then
    echo "[selfplay-grpo] Training failed with status ${status} and reached AUTO_RESUME_MAX_RESTARTS=${AUTO_RESUME_MAX_RESTARTS}."
    exit ${status}
  fi

  echo "[selfplay-grpo] Training exited with status ${status}. Retrying in ${AUTO_RESUME_SLEEP_SECONDS}s using the same run root."
  sleep "${AUTO_RESUME_SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done
