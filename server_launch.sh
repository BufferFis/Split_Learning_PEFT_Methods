#!/bin/bash

# Launch distributed server on both GPUs using torchrun
export CUDA_VISIBLE_DEVICES=0,1

echo "Starting distributed server on 2 H100 GPUs..."

# Use torchrun to launch distributed server processes
torchrun --standalone --nproc_per_node=2 server.py &
SERVER_PID=$!

# Wait for server to start
sleep 10

echo "Distributed server started with PID: $SERVER_PID"
echo "Server is running on http://localhost:8000"
echo "To stop the server, run: kill $SERVER_PID"

# Keep script running
wait $SERVER_PID
