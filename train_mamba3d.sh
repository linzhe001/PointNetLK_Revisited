#!/bin/bash

DATASET_PATH="/mnt/f/Datasets/ModelNet40"

# 训练参数
OUTPUT_FILE="./logs/mamba3dv2_modelnet40_new"  # 修改输出文件名
CATEGORY_FILE="./dataset/modelnet40_half1.txt"
NUM_POINTS=1000
NUM_RANDOM_POINTS=100
MAG=0.8
BATCH_SIZE=32
MAX_EPOCHS=100
LEARNING_RATE=1e-5
DECAY_RATE=1e-4
WARMUP_EPOCHS=5
MIN_LR=1e-8
EMBEDDING="ssm_v2"
DIM_K=1024  # 确保这个维度与特征提取器匹配
MAX_ITER=10

# 运行训练命令
python train.py \
    --outfile "${OUTPUT_FILE}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_type "modelnet" \
    --data_type "synthetic" \
    --categoryfile "${CATEGORY_FILE}" \
    --num_points ${NUM_POINTS} \
    --num_random_points ${NUM_RANDOM_POINTS} \
    --mag ${MAG} \
    --batch_size ${BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --decay_rate ${DECAY_RATE} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --min_lr ${MIN_LR} \
    --embedding ${EMBEDDING} \
    --dim_k ${DIM_K} \
    --max_iter ${MAX_ITER} \
    --device "cuda:0" \
    --resume "" \
    --pretrained "" \
    --start_epoch 0

echo "训练完成！Mamba3D模型保存在 ${OUTPUT_FILE}_*.pth"