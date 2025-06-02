#!/bin/bash

# Launch evaluation-only mode using load.py
export CUDA_VISIBLE_DEVICES=0,1

echo "Starting evaluation-only mode using saved model..."

# Run load.py in evaluation-only mode (no training)
torchrun --standalone --nproc_per_node=2 load.py \
    --model_path ./server_model \
    --eval_only \
    --batch_size 64 \
    --server_url http://127.0.0.1:8000

echo "Evaluation completed!"
