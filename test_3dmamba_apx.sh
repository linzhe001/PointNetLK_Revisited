#!/bin/bash

# 数据集路径
DATASET_PATH="/mnt/f/Datasets/ModelNet40"

# 优化后的训练参数
OUTPUT_PREFIX="./logs/numerical_mamba3d_train"
CATEGORY_FILE="./dataset/modelnet40_half1.txt"
DATASET_TYPE="modelnet"
DATA_TYPE="synthetic"
NUM_RANDOM_POINTS=50  # 减少随机点数量，从100降到50
EMBEDDING="3dmamba_v1"
DIM_K=512  # 减少特征维度，从1024降到512
MAX_ITER=10  # 减少迭代次数，从20降到10

# 数值近似法参数 - 增大delta可以加速但可能降低精度
USE_APPROX=true
DELTA=1e-3  # 从1e-4增大到1e-3

# 创建日志目录
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_numerical_${EMBEDDING}_${TIMESTAMP}.log"
OUTPUT_FILE="${OUTPUT_PREFIX}_${TIMESTAMP}"

# 运行训练命令 - 增加批次大小和减少epoch
python train.py \
    --outfile "${OUTPUT_FILE}" \
    --dataset_path "${DATASET_PATH}" \
    --categoryfile "${CATEGORY_FILE}" \
    --dataset_type "${DATASET_TYPE}" \
    --data_type "${DATA_TYPE}" \
    --num_random_points ${NUM_RANDOM_POINTS} \
    --embedding ${EMBEDDING} \
    --dim_k ${DIM_K} \
    --max_iter ${MAX_ITER} \
    --device "cuda:0" \
    --logfile "${LOG_FILE}" \
    --batch_size 16 \
    --max_epochs 50 \
    --use-approx \
    --delta ${DELTA} \
    --workers 4

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "数值近似法训练完成！"
    echo "模型保存位置: ${OUTPUT_FILE}.pth"
    echo "训练日志: ${LOG_FILE}"
else
    echo "训练失败，请检查日志: ${LOG_FILE}"
fi