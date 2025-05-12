#!/bin/bash

# 数据集路径
DATASET_PATH="/mnt/f/Datasets/ModelNet40"

# 测试参数
OUTPUT_FILE_PREFIX="./test_logs/mamba3d_test_run" # 使用前缀，日志文件会添加时间戳
CATEGORY_FILE="./dataset/modelnet40_half2.txt"  # 为ModelNet数据集使用正确的类别文件
POSE_FILE="./dataset/gt_poses.csv"
DATASET_TYPE="modelnet"  # 可选: "modelnet" 或 "3dmatch"
DATA_TYPE="synthetic"  # 可选: "synthetic" 或 "real"
NUM_RANDOM_POINTS=100  # 修改：使用 num_random_points, 并根据 test.py 默认值设置为100
EMBEDDING="3dmamba_v1"  # 使用Mamba3D特征提取器
DIM_K=1024
MAX_ITER=10
PRETRAINED="./logs/mamba3d_modelnet40_model_best.pth"  # 预训练模型路径

# 移除临时修复测试中的梯度问题，这个问题最好在Python代码中解决
# echo "修改测试脚本以解决梯度计算问题..."
# cp test.py test_backup.py
# sed -i 's/with torch.no_grad():/# with torch.no_grad():/' test.py

# 创建日志目录 (test.py 会根据 logfile 参数自动创建)
LOG_DIR="./test_logs"
mkdir -p ${LOG_DIR}
# 注意：不再需要单独的 results 目录，因为未实现单样本结果保存

# 生成当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/test_${EMBEDDING}_${DATASET_TYPE}_${TIMESTAMP}.log"
OUTPUT_FILE="${OUTPUT_FILE_PREFIX}_${TIMESTAMP}" # 给 outfile 也加上时间戳避免覆盖

# 运行测试命令
# 注意：移除了 --results_dir 参数，因为它未在 test.py 中实现
python test.py \
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
    --test_size 1600 \
    --logfile "${LOG_FILE}" \
    --batch_size 1
    # --results_dir "${RESULTS_DIR}" # 已移除

# 检查测试是否成功完成
if [ $? -eq 0 ]; then
    echo "测试完成！请检查以下文件："
    # echo "- 结果目录: ${RESULTS_DIR}" # 已移除
    # echo "- 单个样本结果: ${RESULTS_DIR}/cloud_*.npz" # 已移除
    # echo "- 合并结果: ${RESULTS_DIR}/all_results.npz" # 已移除
    echo "- 详细日志和汇总统计指标: ${LOG_FILE}"
    echo "- （注意：当前未配置保存每个样本的详细位姿结果）"
else
    echo "测试过程中出现错误，请检查日志文件: ${LOG_FILE}"
fi

# 如果之前创建了备份，可以在这里恢复（但我们移除了备份逻辑）
# echo "恢复原始 test.py 文件..."
# mv test_backup.py test.py