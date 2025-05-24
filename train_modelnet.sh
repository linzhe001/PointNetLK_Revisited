#!/bin/bash

DATASET_PATH="/mnt/f/Datasets/ModelNet40"

# 训练参数
OUTPUT_FILE="./logs/pointnetlk_modelnet40_apx"
CATEGORY_FILE="./dataset/modelnet40_half1.txt"
NUM_POINTS=1000
NUM_RANDOM_POINTS=50  # 减少随机点数量以加速训练
MAG=0.8
BATCH_SIZE=32
MAX_EPOCHS=80  # 稍微减少epoch数
LEARNING_RATE=1e-3
DECAY_RATE=1e-4
EMBEDDING="pointnet"
DIM_K=1024
MAX_ITER=10

# 数值近似法参数
USE_APPROX=true
DELTA=1e-3  # 数值微分的扰动参数

# 创建日志目录
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${EMBEDDING}_apx_${TIMESTAMP}.log"
OUTPUT_FILE_WITH_TIMESTAMP="${OUTPUT_FILE}_${TIMESTAMP}"

echo "开始使用数值近似法训练PointNetLK..."
echo "数值微分扰动参数: ${DELTA}"
echo "训练日志将保存到: ${LOG_FILE}"

# 运行训练命令
python train.py \
    --outfile "${OUTPUT_FILE_WITH_TIMESTAMP}" \
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
    --device "cuda:0" \
    --logfile "${LOG_FILE}" \
    --use-approx \
    --delta ${DELTA} \
    --workers 4

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "数值近似法训练完成！"
    echo "模型保存位置: ${OUTPUT_FILE_WITH_TIMESTAMP}.pth"
    echo "训练日志: ${LOG_FILE}"
    echo ""
    echo "与解析方法的主要区别："
    echo "- 使用数值微分计算雅可比矩阵"
    echo "- 扰动参数 delta = ${DELTA}"
    echo "- 可能略微降低训练速度但更通用"
else
    echo "训练失败，请检查日志: ${LOG_FILE}"
    exit 1
fi