#!/usr/bin/env bash
set -euo pipefail

# This script trains and evaluates gpt2s.py across four runs:
# - gpt2, LoRA r=4/alpha=8 -> output_dir gpt2s4 -> eval
# - gpt2, LoRA r=8/alpha=16 -> output_dir gpt2s8 -> eval
# - gpt2-medium, LoRA r=4/alpha=8 -> output_dir gpt2m4 -> eval
# - gpt2-medium, LoRA r=8/alpha=16 -> output_dir gpt2m8 -> eval
#
# All trainings: 3 epochs, batch size 8, lr 2e-4, fp16 on.
# Evaluations: uses your specified E2E settings (rerank + second pass).

PY=python
SCRIPT=gpt2s.py

# Common training hyperparameters
EPOCHS=3
BATCH_SIZE=8
LR=2e-4
FP16_FLAG=--fp16

# Common evaluation parameters (exactly as you provided, with alt nbest <= beams)
EVAL_FLAGS=(
  --e2e_eval
  --e2e_eval_split test
  --e2e_rerank
  --num_beams 12
  --e2e_nbest 20
  --e2e_beam_groups 4
  --e2e_diversity_penalty 0.15
  --no_repeat_ngram_size 3
  --length_penalty 1.10
  --min_new_tokens 8
  --repetition_penalty 1.03
  --rerank_cov_w 0.55
  --rerank_len_w 0.25
  --rerank_ngram_w 0.15
  --rerank_comp_w 0.05
  --e2e_second_pass
  --e2e_alt_num_beams 16
  --e2e_alt_nbest 16
  --e2e_alt_beam_groups 4
  --e2e_alt_diversity_penalty 0.20
  --e2e_alt_no_repeat_ngram_size 4
  --e2e_alt_length_penalty 1.20
  --e2e_alt_repetition_penalty 1.05
)

train_and_eval() {
  local MODEL_NAME="$1"   # gpt2 or gpt2-medium
  local OUT_DIR="$2"      # e.g., gpt2s4
  local LORA_R="$3"       # 4 or 8
  local LORA_ALPHA="$4"   # 8 or 16

  echo "============================================================"
  echo "Training: model=${MODEL_NAME}, out=${OUT_DIR}, r=${LORA_R}, alpha=${LORA_ALPHA}"
  echo "============================================================"

  mkdir -p "${OUT_DIR}"

  ${PY} "${SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUT_DIR}" \
    --num_epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    ${FP16_FLAG}

  # Read the latest checkpoint path written by training
  if [[ ! -f "${OUT_DIR}/latest_checkpoint.txt" ]]; then
    echo "ERROR: ${OUT_DIR}/latest_checkpoint.txt not found. Training may have failed."
    exit 1
  fi
  CKPT_PATH="$(head -n 1 "${OUT_DIR}/latest_checkpoint.txt" | tr -d '\r\n')"

  if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "ERROR: Checkpoint directory not found: ${CKPT_PATH}"
    exit 1
  fi

  echo "============================================================"
  echo "Evaluating: resume_from=${CKPT_PATH} (results will be saved in ${OUT_DIR})"
  echo "============================================================"

  ${PY} "${SCRIPT}" \
    --output_dir "${OUT_DIR}" \
    --resume_from "${CKPT_PATH}" \
    "${EVAL_FLAGS[@]}"

  echo "Done: ${MODEL_NAME} r=${LORA_R} alpha=${LORA_ALPHA}"
  echo
}

# 1) gpt2, r=4, alpha=8 -> gpt2s4
train_and_eval "gpt2" "gpt2s4" 4 8

# 2) gpt2, r=8, alpha=16 -> gpt2s8
train_and_eval "gpt2" "gpt2s8" 8 16

# 3) gpt2-medium, r=4, alpha=8 -> gpt2m4
train_and_eval "gpt2-medium" "gpt2m4" 4 8

# 4) gpt2-medium, r=8, alpha=16 -> gpt2m8
train_and_eval "gpt2-medium" "gpt2m8" 8 16

echo "All training and evaluations completed."