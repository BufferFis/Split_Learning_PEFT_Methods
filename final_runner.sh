# bash / SLURM / PBS
CUDA_VISIBLE_DEVICES=1  python final.py \
    --epochs 1  --batch_size 4  --learning_rate 2e-4 \
    --load_checkpoint ./splitlora_checkpoint
