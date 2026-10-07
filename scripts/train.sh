#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
METHOD=${METHOD:-va_opd}
DATASET=${DATASET:-geometry3k}
[[ "$DATASET" == geo3k ]] && DATASET=geometry3k
if [[ "$METHOD" != va_opd && "$METHOD" != opd ]]; then
  echo "METHOD must be va_opd or opd" >&2
  exit 2
fi
if [[ "$DATASET" != geometry3k && "$DATASET" != virl39k ]]; then
  echo "DATASET must be geometry3k or virl39k" >&2
  exit 2
fi
ARGS=(--config configs/va_opd.yaml --method "$METHOD")
ARGS+=(--set "data.train_files=${DATA_ROOT:-data}/$DATASET/train.parquet")
ARGS+=(--set "data.val_files=${DATA_ROOT:-data}/geometry3k/validation200.parquet")
ARGS+=(--set "worker.actor.model.model_path=${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}")
ARGS+=(--set "worker.teacher.model.model_path=${TEACHER_PATH:-Qwen/Qwen3-VL-8B-Instruct}")
ARGS+=(--set "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}")
ARGS+=(--set "trainer.experiment_name=${EXPERIMENT_NAME:-qwen3_vl_2b_${METHOD}_${DATASET}}")
exec "${PYTHON:-python}" -m va_opd.train "${ARGS[@]}" "$@"

