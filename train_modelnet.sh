#!/bin/bash


DATASET_PATH="/mnt/f/Datasets/ModelNet40"

# 训练参数
OUTPUT_FILE="./logs/pointnetlk_modelnet40"
CATEGORY_FILE="./dataset/modelnet40_half1.txt"
NUM_POINTS=1000
NUM_RANDOM_POINTS=100
MAG=0.8
BATCH_SIZE=32
MAX_EPOCHS=100
LEARNING_RATE=1e-3
DECAY_RATE=1e-4
EMBEDDING="pointnet"
DIM_K=1024
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
    --embedding ${EMBEDDING} \
    --dim_k ${DIM_K} \
    --max_iter ${MAX_ITER} \
    --device "cuda:0"

echo "训练完成！模型保存在 ${OUTPUT_FILE}_*.pth"