#!/bin/bash

# Multi-GPU training script for CLIP-to-Seg Distillation
# Usage: bash train_multi_gpu.sh [config_file]

CONFIG=${1:-"/home/zjq/document/plant_seg/configs/plant_fewshot_optimized.yaml"}
NUM_GPUS=2
PORT=29500

echo "Starting multi-GPU training with $NUM_GPUS GPUs"
echo "Config file: $CONFIG"
echo "Port: $PORT"

# Launch distributed training with conda environment python
$CONDA_PREFIX/bin/python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train.py \
    --config $CONFIG --val_split val
    

echo "Training completed!"