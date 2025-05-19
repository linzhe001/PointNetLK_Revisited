#!/bin/bash

# C3VD数据集路径 - 修改为WSL路径
DATASET_PATH="/mnt/f/Datasets/C3VD_sever_datasets"

# 模型路径
MODEL_PATH="./logs/c3vd_ssm_v2_model_best.pth"

# 测试参数
NUM_POINTS=1024  # 点云中的点数量
NUM_RANDOM_POINTS=100  # 计算雅可比矩阵的随机点数
MAG=0.8  # 训练时扰动的最大幅度
BATCH_SIZE=16  # 批处理大小
EMBEDDING="ssm_v2"  # 特征提取器类型
DIM_K=1024  # 特征维度
MAX_ITER=10  # LK最大迭代次数

# C3VD特定参数
PAIR_MODE="one_to_one"  # 点云配对模式
TEST_SCENES="duodenum,appendix"  # 测试场景名称，与训练时不同

# 运行测试命令
python test.py \
    --dataset_path "${DATASET_PATH}" \
    --dataset_type "c3vd" \
    --data_type "synthetic" \
    --num_points ${NUM_POINTS} \
    --num_random_points ${NUM_RANDOM_POINTS} \
    --mag ${MAG} \
    --batch_size ${BATCH_SIZE} \
    --embedding ${EMBEDDING} \
    --dim_k ${DIM_K} \
    --max_iter ${MAX_ITER} \
    --device "cuda:0" \
    --model_path "${MODEL_PATH}" \
    --pair-mode "${PAIR_MODE}" \
    --scene-split \
    --test-scenes "${TEST_SCENES}"

echo "测试完成！" 