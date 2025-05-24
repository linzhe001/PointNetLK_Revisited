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

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./logs/2021_04_17_train_modelnet',
                        metavar='BASENAME', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='./dataset/ModelNet',
                        metavar='PATH', help='path to the input dataset')
    
    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', type=str,
                        metavar='DATASET', help='dataset type')
    parser.add_argument('--data_type', default='synthetic', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    parser.add_argument('--categoryfile', type=str, default='./dataset/modelnet40_half1.txt',
                        metavar='PATH', help='path to the categories to be trained')
    parser.add_argument('--num_points', default=1000, type=int,
                        metavar='N', help='points in point-cloud.')
    parser.add_argument('--num_random_points', default=100, type=int,
                        metavar='N', help='number of random points to compute Jacobian.')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='D', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    parser.add_argument('--sigma', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--clip', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--workers', default=12, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for Embedding
    parser.add_argument('--embedding', default='pointnet',
                        type=str, help='特征提取器类型: pointnet, 3dmamba_v1, 3dmamba_v2, pointnet_attention_v1, fastpointtransformer_v1, fastpointtransformer_v2, swinattention_v1, swinattention_v2, ssm_v1, ssm_v2')
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector')
    
    # settings for LK
    parser.add_argument('--max_iter', default=10, type=int,
                        metavar='N', help='max-iter on LK.')

    # settings for training.
    parser.add_argument('--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        metavar='METHOD', help='name of an optimizer')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--lr', type=float, default=1e-3,
                        metavar='D', help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, 
                        metavar='D', help='decay rate of learning rate')

    # settings for log
    parser.add_argument('--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file')

    # settings for learning rate warmup
    parser.add_argument('--warmup_epochs', default=5, type=int, 
                        metavar='N', help='预热训练的回合数')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        metavar='D', help='余弦退火的最小学习率')
                        
    # 增加C3VD相关参数
    parser.add_argument('--pair-mode', default='one_to_one', type=str,
                      metavar='MODE', help='点云配对模式: one_to_one, scene_reference, source_to_source, target_to_target, all')
    parser.add_argument('--reference-name', default='', type=str,
                      metavar='NAME', help='scene_reference模式下的参考点云名称')
    parser.add_argument('--scene-split', action='store_true',
                      help='是否使用场景划分')
    parser.add_argument('--test-scenes', type=str, default='',
                       help='指定测试场景，用逗号分隔场景名称，如"sigmoid,cecum"')

    # 数值近似法参数
    parser.add_argument('--use-approx', action='store_true',
                       help='使用数值近似方法计算雅可比矩阵')
    parser.add_argument('--delta', type=float, default=1.0e-3,
                       metavar='D', help='数值近似方法的扰动参数')

    args = parser.parse_args(argv)
    return args


def train(args, trainset, evalset, dptnetlk):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    model = dptnetlk.create_model()

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if args.start_epoch >= args.warmup_epochs and 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
            cosine_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    print('resume epoch from {}'.format(args.start_epoch+1))

    evalloader = torch.utils.data.DataLoader(evalset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    min_loss = float('inf')
    min_info = float('inf')

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=args.lr, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        min_info = checkpoint['min_info']
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        # 如果有保存的学习率调度器状态，则加载
        if args.start_epoch >= args.warmup_epochs and 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
            cosine_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # 创建余弦退火调度器
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.max_epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )

    # 定义学习率更新函数，包含预热和余弦退火
    def update_lr(epoch):
        if epoch < args.warmup_epochs:
            # 线性预热学习率
            lr_scale = min(1., float(epoch + 1) / float(args.warmup_epochs))
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * lr_scale
        else:
            # 使用余弦退火调度器
            cosine_scheduler.step()

    # training
    LOGGER.debug('Begin Training!')
    for epoch in range(args.start_epoch, args.max_epochs):
        # 更新学习率
        update_lr(epoch)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        running_loss, running_info = dptnetlk.train_one_epoch(
            model, trainloader, optimizer, args.device, 'train', args.data_type, num_random_points=args.num_random_points)
        val_loss, val_info = dptnetlk.eval_one_epoch(
            model, evalloader, args.device, 'eval', args.data_type, num_random_points=args.num_random_points)
        
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        LOGGER.info('epoch, %04d, %f, %f, %f, %f, %.10f', epoch + 1,
                    running_loss, val_loss, running_info, val_info, current_lr)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'min_info': min_info,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': cosine_scheduler.state_dict() if epoch >= args.warmup_epochs else None}
        if is_best:
            torch.save(model.state_dict(), '{}_{}.pth'.format(args.outfile, 'model_best'))
        torch.save(snap, '{}_{}.pth'.format(args.outfile, 'snap_last'))


def main(args):
    trainset, evalset = get_datasets(args)
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(args)
    train(args, trainset, evalset, dptnetlk)


def get_datasets(args):
    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube(),\
                    data_utils.Resampler(args.num_points)])

        traindata = data_utils.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        evaldata = data_utils.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        trainset = data_utils.PointRegistration(traindata, data_utils.RandomTransformSE3(args.mag))
        evalset = data_utils.PointRegistration(evaldata, data_utils.RandomTransformSE3(args.mag))
    elif args.dataset_type == 'c3vd':
        # 添加C3VD数据集支持
        transform = torchvision.transforms.Compose([
                    data_utils.Resampler(args.num_points)])
        
        # 确定数据路径
        source_path = os.path.join(args.dataset_path, 'C3VD_ply_source')
        target_path = os.path.join(args.dataset_path, 'visible_point_cloud_ply_depth')
        
        print(f"加载C3VD数据集，配对模式: {args.pair_mode}")
        print(f"源点云路径: {source_path}")
        print(f"目标点云路径: {target_path}")
        
        # 创建C3VD数据集
        c3vd_dataset = data_utils.C3VDDataset(
            source_path, target_path, transform,
            pair_mode=args.pair_mode,
            reference_name=args.reference_name
        )
        
        # 实现场景划分或随机划分
        if args.scene_split:
            print("使用场景划分...")
            # 获取所有场景名称
            all_scenes = c3vd_dataset.scenes
            
            # 检查是否有指定测试场景
            if args.test_scenes:
                # 按逗号分隔获取指定的测试场景列表
                test_scenes = args.test_scenes.split(',')
                # 筛选有效的测试场景
                valid_test_scenes = [scene for scene in test_scenes if scene in all_scenes]
                
                if not valid_test_scenes:
                    print("警告: 指定的测试场景不在数据集中，将使用随机划分")
                    np.random.shuffle(all_scenes)
                    split_idx = int(len(all_scenes) * 0.8)
                    train_scenes = all_scenes[:split_idx]
                    test_scenes = all_scenes[split_idx:]
                else:
                    # 使用指定的测试场景，其余作为训练场景
                    test_scenes = valid_test_scenes
                    train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
                    print(f"使用指定测试场景: {test_scenes}")
            else:
                # 没有指定测试场景，使用随机划分
                np.random.shuffle(all_scenes)
                split_idx = int(len(all_scenes) * 0.8)
                train_scenes = all_scenes[:split_idx]
                test_scenes = all_scenes[split_idx:]
            
            print(f"训练场景: {len(train_scenes)}个, 测试场景: {len(test_scenes)}个")
            print(f"训练场景列表: {train_scenes}")
            print(f"测试场景列表: {test_scenes}")
            
            # 获取训练和测试样本的索引
            train_indices = c3vd_dataset.get_scene_indices(train_scenes)
            test_indices = c3vd_dataset.get_scene_indices(test_scenes)
            
            print(f"训练样本数量: {len(train_indices)}")
            print(f"测试样本数量: {len(test_indices)}")
            
            # 创建子集
            train_dataset = torch.utils.data.Subset(c3vd_dataset, train_indices)
            eval_dataset = torch.utils.data.Subset(c3vd_dataset, test_indices)
        else:
            print("使用随机划分...")
            # 随机划分训练和测试集
            dataset_size = len(c3vd_dataset)
            indices = list(range(dataset_size))
            np.random.shuffle(indices)
            split_idx = int(dataset_size * 0.8)
            
            train_dataset = torch.utils.data.Subset(c3vd_dataset, indices[:split_idx])
            eval_dataset = torch.utils.data.Subset(c3vd_dataset, indices[split_idx:])
        
        # 传递mag参数给C3VDset4tracking
        trainset = data_utils.C3VDset4tracking(train_dataset, args.num_points, mag=args.mag)
        evalset = data_utils.C3VDset4tracking(eval_dataset, args.num_points, mag=args.mag)
    else:
        print('wrong dataset type!')

    return trainset, evalset


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)

    LOGGER.debug('Training completed! Yay~~ (PID=%d)', os.getpid())
