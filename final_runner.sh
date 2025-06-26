# one-liner in the shell
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=AF:00.0 \
python final.py --epochs 1 --batch_size 64 --learning_rate 2e-4 
