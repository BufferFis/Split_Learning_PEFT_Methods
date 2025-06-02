#!/bin/bash

# Launch evaluation-only mode using load.py
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

echo "Starting evaluation-only mode using saved model..."

# FIXED: Use torchrun with the module approach
torchrun --standalone --nproc_per_node=2 load.py \
    --model_path ./server_model \
    --eval_only \
    --batch_size 64 \
    --server_url http://127.0.0.1:8000

echo "Evaluation completed!"

echo "Checking for evaluation results..."
if [ -f "./server_model/evaluation_results.json" ]; then
    echo "=== EVALUATION RESULTS FOUND ==="
    cat ./server_model/evaluation_results.json
    echo ""
    echo "================================"
else
    echo "❌ No evaluation results file found"
fi

if [ -f "./server_model/eval_only_results.json" ]; then
    echo "=== EVAL ONLY RESULTS FOUND ==="
    cat ./server_model/eval_only_results.json
    echo ""
    echo "=============================="
fi

