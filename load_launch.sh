#!/bin/bash

# Load saved model and continue training with distributed server
export CUDA_VISIBLE_DEVICES=0,1

echo "Loading saved model and continuing training with distributed server..."

# Use torchrun for proper distributed launch
torchrun --standalone --nproc_per_node=2 load.py \
    --model_path ./server_model \
    --continue_training \
    --epochs 1 \
    --batch_size 64 \
    --server_url http://127.0.0.1:8000

echo "Incremental training completed!"
