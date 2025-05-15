""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import argparse
import os
import logging
import torch
import torch.utils.data
import torchvision
import numpy as np

import data_utils
import trainer
import utils

LOGGER = logging.getLogger(__name__)
# 移除这里的 NullHandler，让根日志记录器处理
# LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./test_logs/test_run', # 修改默认值以避免覆盖
                        metavar='BASENAME', help='output filename (prefix) for logs and potentially results') # 澄清用途
    parser.add_argument('--dataset_path', type=str, default='./dataset/ThreeDMatch',
                        metavar='PATH', help='path to the input dataset')
    parser.add_argument('--categoryfile', type=str, default='./dataset/test_3dmatch.txt',
                        metavar='PATH', choices=['./dataset/test_3dmatch.txt', './dataset/modelnet40_half2.txt'],
                        help='path to the categories to be tested')
    parser.add_argument('--pose_file', type=str, default='./dataset/gt_poses.csv',
                        metavar='PATH', help='path to the testing pose files')

    # settings for input data
    parser.add_argument('--dataset_type', default='3dmatch', type=str,
                        metavar='DATASET', choices=['modelnet', '3dmatch'], help='dataset type')
    parser.add_argument('--data_type', default='real', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    # 移除 --num_points，因为我们将用 num_random_points 控制 modelnet/shapenet 的点数
    # parser.add_argument('--num_points', default=1000, type=int,
    #                     metavar='N', help='points in point-cloud')
    parser.add_argument('--num_random_points', default=100, type=int, # 添加此参数以控制测试点数
                        metavar='N', help='测试时用于 ModelNet/ShapeNet 的点数 (默认为 100)') # 测试时用于 ModelNet/ShapeNet 的点数 (默认为 100)
    parser.add_argument('--sigma', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--clip', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--workers', default=0, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for voxelization
    parser.add_argument('--overlap_ratio', default=0.7, type=float,
                        metavar='D', help='overlapping ratio for 3DMatch dataset.')
    parser.add_argument('--voxel_ratio', default=0.05, type=float,
                        metavar='D', help='voxel ratio')
    parser.add_argument('--voxel', default=2, type=float,
                        metavar='D', help='how many voxels you want to divide in each axis')
    parser.add_argument('--max_voxel_points', default=1000, type=int,
                        metavar='N', help='maximum points allowed in a voxel')
    parser.add_argument('--num_voxels', default=8, type=int,
                        metavar='N', help='number of voxels')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='whether to visualize or not')
    parser.add_argument('--voxel_after_transf', action='store_true', default=False,
                        help='given voxelization before or after transformation')

    # settings for Embedding
    parser.add_argument('--embedding', default='pointnet',
                        type=str, help='特征提取器类型: pointnet, 3dmamba_v1') # 特征提取器类型: pointnet, 3dmamba_v1
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector')

    # settings for LK
    parser.add_argument('-mi', '--max_iter', default=20, type=int,
                        metavar='N', help='max-iter on LK.')

    # settings for training.
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    # settings for log
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (e.g., ./test_logs/test.log)') # 添加示例
    parser.add_argument('--pretrained', default='./logs/model_trained_on_ModelNet40_model_best.pth', type=str,
                        metavar='PATH', help='path to pretrained model file ')

    # 添加这一行来限制测试样本数量
    parser.add_argument('--test_size', default=0, type=int,
                        metavar='N', help='限制测试样本数量，0表示不限制') # 限制测试样本数量，0表示不限制

    # 添加结果保存相关参数
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='是否保存测试结果到文件') # 是否保存测试结果到文件
    parser.add_argument('--results_format', type=str, default='csv', choices=['csv', 'json'],
                        help='测试结果保存格式') # 测试结果保存格式

    args = parser.parse_args(argv)
    return args


def test(args, testset, dptnetlk):
    """执行测试过程"""
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    model = dptnetlk.create_model()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            LOGGER.info(f"Loading pretrained model from: {args.pretrained}")
            # 显式指定 map_location=args.device，确保模型加载到正确的设备
            model.load_state_dict(torch.load(args.pretrained, map_location=args.device))
        else:
            LOGGER.error(f"Pretrained model file not found: {args.pretrained}")
            # 可以选择抛出错误或继续使用未初始化的模型
            # raise FileNotFoundError(f"Pretrained model file not found: {args.pretrained}")
            LOGGER.warning("Proceeding with an untrained model.")
    else:
        LOGGER.warning("No pretrained model provided. Starting with an untrained model.") # 添加警告

    model.to(args.device)
    model.eval() # 确保模型在评估模式

    # 如果指定了测试大小限制，则只用部分数据
    if args.test_size > 0:
        # 确保 test_size 不超过数据集大小
        actual_test_size = min(args.test_size, len(testset))
        if actual_test_size < args.test_size:
             LOGGER.warning(f"Requested test_size {args.test_size} is larger than dataset size {len(testset)}. Using {actual_test_size}.")
        indices = list(range(actual_test_size))
        subset = torch.utils.data.Subset(testset, indices)
        testloader = torch.utils.data.DataLoader(
            subset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
        LOGGER.info(f'Testing with limited size: {len(subset)} samples') # 使用 info 级别
    else:
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
        LOGGER.info(f'Testing with full dataset: {len(testset)} samples') # 使用 info 级别

    # 创建容器来保存测试结果
    results_dict = {
        'rotations_gt': [],
        'translation_gt': [],
        'rotations_ab': [],
        'translation_ab': []
    }
    
    # 修改 utils.test_metrics 函数以返回而不仅仅是打印结果
    original_test_metrics = utils.test_metrics
    
    metrics_results = {}
    
    def modified_test_metrics(rotations_gt, translation_gt, rotations_ab, translation_ab, filename):
        # 调用原始函数
        original_test_metrics(rotations_gt, translation_gt, rotations_ab, translation_ab, filename)
        
        # 计算相同的指标用于保存
        rotations_gt = np.concatenate(rotations_gt, axis=0).reshape(-1, 3)
        translation_gt = np.concatenate(translation_gt, axis=0).reshape(-1, 3)
        rotations_ab = np.concatenate(rotations_ab, axis=0).reshape(-1, 3)
        translation_ab = np.concatenate(translation_ab, axis=0).reshape(-1,3)

        # 计算角度误差和平移误差
        rot_err = np.sqrt(np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2, axis=1))
        trans_err = np.sqrt(np.mean((translation_ab - translation_gt) ** 2, axis=1))

        # 成功案例统计
        suc_tab = np.zeros(11)
        rot_err_tab = np.arange(11) * 0.5
        trans_err_tab = np.arange(11) * 0.05
        err_count_tab = np.triu(np.ones((11, 11)))
        
        for i in range(rot_err.shape[0]):
            for j in range(11):
                if rot_err[i] <= rot_err_tab[j] and trans_err[i] <= trans_err_tab[j]:
                    suc_tab = suc_tab + err_count_tab[j]
                    break

        # 均值统计
        rot_mse_ab = np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
        rot_rmse_ab = np.sqrt(rot_mse_ab)
        rot_mae_ab = np.mean(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))

        trans_mse_ab = np.mean((translation_ab - translation_gt) ** 2)
        trans_rmse_ab = np.sqrt(trans_mse_ab)
        trans_mae_ab = np.mean(np.abs(translation_ab - translation_gt))
        
        # 中位数统计
        rot_mse_ab_02 = np.median((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
        rot_rmse_ab_02 = np.sqrt(rot_mse_ab_02)
        rot_mae_ab_02 = np.median(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))
        
        trans_mse_ab_02 = np.median((translation_ab - translation_gt) ** 2)
        trans_rmse_ab_02 = np.sqrt(trans_mse_ab_02)
        trans_mae_ab_02 = np.median(np.abs(translation_ab - translation_gt))
        
        # 保存计算结果
        metrics_results['success_cases'] = suc_tab.tolist()
        metrics_results['rot_thresholds'] = rot_err_tab.tolist()
        metrics_results['trans_thresholds'] = trans_err_tab.tolist()
        
        # 均值统计
        metrics_results['mean'] = {
            'rot_MSE': float(rot_mse_ab),
            'rot_RMSE': float(rot_rmse_ab),
            'rot_MAE': float(rot_mae_ab),
            'trans_MSE': float(trans_mse_ab),
            'trans_RMSE': float(trans_rmse_ab),
            'trans_MAE': float(trans_mae_ab)
        }
        
        # 中位数统计
        metrics_results['median'] = {
            'rot_MSE': float(rot_mse_ab_02),
            'rot_RMSE': float(rot_rmse_ab_02),
            'rot_MAE': float(rot_mae_ab_02),
            'trans_MSE': float(trans_mse_ab_02),
            'trans_RMSE': float(trans_rmse_ab_02),
            'trans_MAE': float(trans_mae_ab_02)
        }
        
        # 样本级别的错误
        metrics_results['sample_errors'] = {
            'rot_err': rot_err.tolist(),
            'trans_err': trans_err.tolist()
        }
        
    # 暂时替换 utils.test_metrics 函数
    utils.test_metrics = modified_test_metrics

    # testing
    LOGGER.info('Begin Testing!') # 使用 info 级别
    try:
        # 修改 trainer.py 的 test_one_epoch 方法以访问内部收集的数据
        original_test_one_epoch = dptnetlk.test_one_epoch
        
        def modified_test_one_epoch(model, testloader, device, mode, data_type='synthetic', vis=False):
            # 设置一个捕获方法来获取trainer.py中收集的旋转和平移数据
            dptnetlk.rotations_gt = []
            dptnetlk.translation_gt = []
            dptnetlk.rotations_ab = []
            dptnetlk.translation_ab = []
            
            # 调用原始方法
            original_test_one_epoch(model, testloader, device, mode, data_type, vis)
            
            # 复制收集的结果
            results_dict['rotations_gt'] = dptnetlk.rotations_gt
            results_dict['translation_gt'] = dptnetlk.translation_gt
            results_dict['rotations_ab'] = dptnetlk.rotations_ab
            results_dict['translation_ab'] = dptnetlk.translation_ab
            
        # 替换测试方法
        dptnetlk.test_one_epoch = modified_test_one_epoch
        
        # 运行测试
        dptnetlk.test_one_epoch(model, testloader, args.device, 'test', args.data_type, args.vis)
        LOGGER.info('Testing finished.') # 使用 info 级别
        
        # 保存测试结果
        if args.save_results:
            if args.results_format == 'csv':
                results_filename = f"{args.outfile}_detailed_results.csv"
                LOGGER.info(f"Saving detailed test results to {results_filename}")
                
                with open(results_filename, 'w') as f:
                    # 写入均值统计
                    f.write("********************mean********************\n")
                    f.write(f"rot_MSE,{metrics_results['mean']['rot_MSE']}\n")
                    f.write(f"rot_RMSE,{metrics_results['mean']['rot_RMSE']}\n")
                    f.write(f"rot_MAE,{metrics_results['mean']['rot_MAE']}\n")
                    f.write(f"trans_MSE,{metrics_results['mean']['trans_MSE']}\n")
                    f.write(f"trans_RMSE,{metrics_results['mean']['trans_RMSE']}\n")
                    f.write(f"trans_MAE,{metrics_results['mean']['trans_MAE']}\n")
                    
                    # 写入中位数统计
                    f.write("\n********************median********************\n")
                    f.write(f"rot_MSE,{metrics_results['median']['rot_MSE']}\n")
                    f.write(f"rot_RMSE,{metrics_results['median']['rot_RMSE']}\n")
                    f.write(f"rot_MAE,{metrics_results['median']['rot_MAE']}\n")
                    f.write(f"trans_MSE,{metrics_results['median']['trans_MSE']}\n")
                    f.write(f"trans_RMSE,{metrics_results['median']['trans_RMSE']}\n")
                    f.write(f"trans_MAE,{metrics_results['median']['trans_MAE']}\n")
                    
                    # 写入成功案例统计
                    f.write("\n********************success cases********************\n")
                    for i, success_count in enumerate(metrics_results['success_cases']):
                        rot_threshold = metrics_results['rot_thresholds'][i]
                        trans_threshold = metrics_results['trans_thresholds'][i]
                        f.write(f"rot<={rot_threshold:.1f},trans<={trans_threshold:.2f},{success_count}\n")
                    
                LOGGER.info(f"Test detailed results saved to {results_filename}")
                
            elif args.results_format == 'json':
                import json
                results_filename = f"{args.outfile}_detailed_results.json"
                LOGGER.info(f"Saving detailed test results to {results_filename}")
                
                with open(results_filename, 'w') as f:
                    json.dump(metrics_results, f, indent=2)
                
                LOGGER.info(f"Test detailed results saved to {results_filename}")
                
        
    except Exception as e:
        LOGGER.exception("An error occurred during test_one_epoch execution.") # 记录测试执行中的错误
        raise # 重新抛出异常
    finally:
        # 恢复原始方法
        utils.test_metrics = original_test_metrics
        dptnetlk.test_one_epoch = original_test_one_epoch

    # 在 test_one_epoch 调用结束后尝试清理 GPU 缓存。
    if torch.cuda.is_available():
        LOGGER.info("Attempting to clear CUDA cache after testing.") # 使用 info 级别
        torch.cuda.empty_cache()

def main(args):
    """主函数，设置并运行测试""" # 添加函数文档字符串
    # 检查并创建输出目录 (如果 outfile 参数指定了目录)
    outfile_dir = os.path.dirname(args.outfile)
    if outfile_dir and not os.path.exists(outfile_dir):
        try:
            os.makedirs(outfile_dir)
            LOGGER.info(f"Created output directory: {outfile_dir}")
        except OSError as e:
            LOGGER.error(f"Failed to create output directory {outfile_dir}: {e}")
            # 根据需要决定是否继续
            # return

    # 检查并创建日志文件目录 (如果 logfile 参数指定了目录)
    if args.logfile:
        logfile_dir = os.path.dirname(args.logfile)
        if logfile_dir and not os.path.exists(logfile_dir):
             try:
                 os.makedirs(logfile_dir)
                 LOGGER.info(f"Created log directory: {logfile_dir}")
             except OSError as e:
                LOGGER.error(f"Failed to create log directory {logfile_dir}: {e}")
                # 根据需要决定是否继续
                # args.logfile = '' # 例如，禁用文件日志记录

    # 配置日志记录 (移到 main 函数外部的 __main__ 块更合适)

    LOGGER.info("Loading dataset...") # 添加日志
    testset = get_datasets(args)
    LOGGER.info("Dataset loaded.") # 添加日志

    # 确认 Trainer 初始化时使用了正确的 embedding 参数
    LOGGER.info("Initializing trainer...") # 添加日志
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(args)
    LOGGER.info("Trainer initialized.") # 添加日志

    test(args, testset, dptnetlk)


def get_datasets(args):
    """根据参数加载测试数据集""" # 添加函数文档字符串
    cinfo = None
    # 修正: 只有 modelnet/shapenet2 需要 categoryfile 来过滤类别
    if args.categoryfile and args.dataset_type in ['modelnet', 'shapenet2']:
        LOGGER.info(f"Loading categories from: {args.categoryfile}")
        try:
            with open(args.categoryfile, 'r') as f:
                categories = [line.rstrip('\n') for line in f]
            categories.sort()
            c_to_idx = {categories[i]: i for i in range(len(categories))}
            cinfo = (categories, c_to_idx)
            LOGGER.info(f"Loaded {len(categories)} categories.")
        except FileNotFoundError:
            LOGGER.error(f"Category file not found: {args.categoryfile}")
            raise # 应该中止执行
    elif args.dataset_type == '3dmatch' and not args.categoryfile:
        # 3dmatch 也需要 categoryfile 来指定场景
        LOGGER.error("Category file is required for 3dmatch dataset.")
        raise ValueError("Category file is required for 3dmatch dataset.")
    elif args.dataset_type == '3dmatch':
         LOGGER.info(f"Using scene list from: {args.categoryfile} for 3DMatch.")


    LOGGER.info(f"Preparing dataset: {args.dataset_type}")
    if args.dataset_type == 'modelnet':
        # 在 ModelNet 的 transform 中加入 Resampler
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube(),\
                    data_utils.Resampler(args.num_random_points)]) # 添加重采样步骤

        testdata = data_utils.ModelNet(args.dataset_path, train=-1, transform=transform, classinfo=cinfo)
        testset = data_utils.PointRegistration_fixed_perturbation(testdata, args.pose_file, sigma=args.sigma, clip=args.clip)

    elif args.dataset_type == 'shapenet2': # 原始代码没有 Shapenet2 测试逻辑，但按要求修改
        # 在 ShapeNet2 的 transform 中加入 Resampler
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube(),\
                    data_utils.Resampler(args.num_random_points)]) # 添加重采样步骤

        testdata = data_utils.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        testset = data_utils.PointRegistration_fixed_perturbation(testdata, args.pose_file, sigma=args.sigma, clip=args.clip)

    elif args.dataset_type == '3dmatch':
        # 对于 3DMatch，由于 voxelization 逻辑，num_random_points 参数不直接适用
        LOGGER.warning("`--num_random_points` argument is not directly used for the '3dmatch' dataset due to its voxelization pipeline.")
        testset = data_utils.ThreeDMatch_Testing(args.dataset_path, args.categoryfile, args.overlap_ratio,
                                                 args.voxel_ratio, args.voxel, args.max_voxel_points,
                                                 args.num_voxels, args.pose_file, args.vis, args.voxel_after_transf)
    else:
        LOGGER.error(f"Unsupported dataset type: {args.dataset_type}")
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    LOGGER.info(f"Dataset size: {len(testset)}")
    # 如果是 ModelNet 或 ShapeNet，记录实际使用的点数
    if args.dataset_type in ['modelnet', 'shapenet2']:
        LOGGER.info(f"Number of points per sample (for ModelNet/ShapeNet test): {args.num_random_points}")
    return testset


if __name__ == '__main__':
    ARGS = options()

    # 配置日志记录
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s') # 改进格式
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # 设置根日志级别为 INFO (更常用)
    # 清除可能存在的旧处理器，以防脚本被多次调用
    root_logger.handlers.clear()

    # 文件日志处理器
    if ARGS.logfile:
        # 确保目录存在 (在 main 中已处理，这里可以省略检查)
        try:
            file_handler = logging.FileHandler(ARGS.logfile, mode='a') # 使用追加模式 'a'
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(logging.DEBUG) # 文件记录更详细的 DEBUG 信息
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file logger at {ARGS.logfile}: {e}") # 如果日志设置失败，打印到控制台

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO) # 控制台输出 INFO 及以上级别
    root_logger.addHandler(console_handler)

    LOGGER.info('Starting Test Script (PID=%d)', os.getpid()) # 使用 info
    LOGGER.debug('Command line arguments: %s', ARGS) # DEBUG 级别记录参数详情

    try: # 添加 try...finally 块确保日志记录完整
        main(ARGS)
        LOGGER.info('Testing completed successfully! (PID=%d)', os.getpid()) # 使用 info
    except Exception as e:
        # 记录异常信息到日志
        LOGGER.exception("An critical error occurred during testing: %s", e)
        LOGGER.error('Testing failed due to an error! (PID=%d)', os.getpid()) # 使用 error
    finally:
        LOGGER.info('Shutting down logging.') # 添加结束日志
        logging.shutdown() # 关闭日志处理器
