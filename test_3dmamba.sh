#!/bin/bash

# 数据集路径
DATASET_PATH="/mnt/f/Datasets/ModelNet40"

# 测试参数
OUTPUT_FILE_PREFIX="./test_logs/mamba3d_test_run" # 使用前缀，日志文件会添加时间戳
CATEGORY_FILE="./dataset/modelnet40_half2.txt"  # 为ModelNet数据集使用正确的类别文件
POSE_FILE="./dataset/gt_poses.csv"
DATASET_TYPE="modelnet"  # 可选: "modelnet" 或 "3dmatch"
DATA_TYPE="synthetic"  # 可选: "synthetic" 或 "real"
NUM_RANDOM_POINTS=200  # 修改：使用 num_random_points, 并根据 test.py 默认值设置为100
EMBEDDING="3dmamba_v1"  # 使用Mamba3D特征提取器
DIM_K=1024
MAX_ITER=20
PRETRAINED="./logs/mamba3d_modelnet40_model_best.pth"  # 预训练模型路径

# 结果保存设置
SAVE_RESULTS=true  # 是否保存测试结果
RESULTS_FORMAT="csv"  # 结果保存格式：csv 或 json

# 创建日志目录 (test_coy.py 会根据 logfile 参数自动创建)
LOG_DIR="./test_logs"
mkdir -p ${LOG_DIR}

# 生成当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/test_${EMBEDDING}_${DATASET_TYPE}_${TIMESTAMP}.log"
OUTPUT_FILE="${OUTPUT_FILE_PREFIX}_${TIMESTAMP}" # 给 outfile 也加上时间戳避免覆盖

# 运行测试命令
# 注意：使用修改后的 test_coy.py 才能支持结果保存功能
python test_coy.py \
    --outfile "${OUTPUT_FILE}" \
    --dataset_path "${DATASET_PATH}" \
    --categoryfile "${CATEGORY_FILE}" \
    --pose_file "${POSE_FILE}" \
    --dataset_type "${DATASET_TYPE}" \
    --data_type "${DATA_TYPE}" \
    --num_random_points ${NUM_RANDOM_POINTS} \
    --embedding ${EMBEDDING} \
    --dim_k ${DIM_K} \
    --max_iter ${MAX_ITER} \
    --pretrained "${PRETRAINED}" \
    --device "cuda:0" \
    --logfile "${LOG_FILE}" \
    --batch_size 1 \
    --test_size 2000 \
    --save_results \
    --results_format ${RESULTS_FORMAT}

# 检查测试是否成功完成
if [ $? -eq 0 ]; then
    echo "测试完成！请检查以下文件："
    echo "- 详细日志和汇总统计指标: ${LOG_FILE}"
    echo "- 测试结果文件: ${OUTPUT_FILE}_results.${RESULTS_FORMAT}"
else
    echo "测试过程中出现错误，请检查日志文件: ${LOG_FILE}"
fi