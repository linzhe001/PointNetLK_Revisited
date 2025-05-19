#!/bin/bash

# C3VD数据集路径 - 修改为WSL路径
DATASET_PATH="/mnt/f/Datasets/C3VD_sever_datasets"

# 训练参数
OUTPUT_FILE="./logs/c3vd_ssm_v2"  # 输出文件名
NUM_POINTS=1024  # 点云中的点数量
NUM_RANDOM_POINTS=100  # 计算雅可比矩阵的随机点数
MAG=0.8  # 训练时扰动的最大幅度
BATCH_SIZE=24  # 批处理大小
MAX_EPOCHS=100  # 总训练回合数
LEARNING_RATE=1e-5  # 学习率
DECAY_RATE=1e-4  # 学习率衰减率
WARMUP_EPOCHS=5  # 预热训练回合数
MIN_LR=1e-8  # 最小学习率
EMBEDDING="ssm_v2"  # 特征提取器类型
DIM_K=1024  # 特征维度
MAX_ITER=10  # LK最大迭代次数

# C3VD特定参数
PAIR_MODE="one_to_one"  # 点云配对模式
TEST_SCENES="sigmoid,cecum"  # 测试场景名称，逗号分隔

# 运行训练命令
python train.py \
    --outfile "${OUTPUT_FILE}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset_type "c3vd" \
    --data_type "synthetic" \
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
    --start_epoch 0 \
    --pair-mode "${PAIR_MODE}" \
    --scene-split \
    --test-scenes "${TEST_SCENES}"

echo "训练完成！C3VD模型保存在 ${OUTPUT_FILE}_*.pth" 