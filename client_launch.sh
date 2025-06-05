#!/bin/bash

# Multi-GPU distributed training launch script
export CUDA_VISIBLE_DEVICES=0,1

# Wait for servers to be ready
echo "Waiting for distributed servers to be ready..."
sleep 15

echo "Starting distributed training on 2 H100 GPUs..."

# Use torchrun for proper distributed launch
torchrun --standalone --nproc_per_node=2 client.py \
    --batch_size 64 \
    --epochs 3 \
    --server_url http://127.0.0.1:8000

echo "Training completed!"
