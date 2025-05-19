""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import torch
import numpy as np
from scipy.spatial.transform import Rotation
import tqdm
import logging
import os
import time
from pathlib import Path

import model
import utils
from PointNet_files import pointnetlk_3dmamba_improved
from PointNet_files import pointnetlk_attention_improved as pointnetlk_attention
# 导入近似雅可比矩阵模型
from PointNet_files import pointnetlk_attention_approx
# 导入FastPointTransformer模型文件
from PointNet_files import fastpointtransformer_analytical_jacobian as fpt_analytical
from PointNet_files import fastpointtransformer_approx_jacobian as fpt_approx
# 导入CFormer模型文件
from CFormer_files import cformer_attention_improved as cformer_analytical
from CFormer_files import cformer_attention_approx as cformer_approx

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class TrainerAnalyticalPointNetLK:
    def __init__(self, args):
        # PointNet
        self.dim_k = args.dim_k
        # LK
        self.device = args.device
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True
        # network
        self.embedding = args.embedding
        self.filename = args.outfile
        # 存储参数
        self.args = args
        self.mamba_type = args.mamba_type if hasattr(args, 'mamba_type') else 'standard'
        
        # 添加近似雅可比矩阵方法的参数
        self.use_approx = hasattr(args, 'use_approx') and args.use_approx
        self.delta = args.delta if hasattr(args, 'delta') else 1.0e-2
        self.learn_delta = args.learn_delta if hasattr(args, 'learn_delta') else False
        
    def create_features(self):
        if self.embedding == 'pointnet':
            ptnet = model.Pointnet_Features(dim_k=self.dim_k)
        elif self.embedding == '3dmamba':
            # 获取Mamba特定参数
            mamba_params = {}
            if hasattr(self.args, 'mamba_state_dim'):
                mamba_params['state_dim'] = self.args.mamba_state_dim
            if hasattr(self.args, 'mamba_expand'):
                mamba_params['expand'] = self.args.mamba_expand
            if hasattr(self.args, 'mamba_conv_size'):
                mamba_params['conv_size'] = self.args.mamba_conv_size
            
            # 根据mamba_type选择特征提取器
            if self.mamba_type == 'standard':
                ptnet = pointnetlk_3dmamba_improved.Pointnet_Features_Mamba_Improved(
                    dim_k=self.dim_k, use_mamba=True, **mamba_params)
            elif self.mamba_type == 'bissm':
                ptnet = pointnetlk_3dmamba_improved.Pointnet_Features_BiSSM(
                    dim_k=self.dim_k, use_mamba=True, **mamba_params)
            else:  # 'none'
                ptnet = pointnetlk_3dmamba_improved.Pointnet_Features_Mamba_Improved(
                    dim_k=self.dim_k, use_mamba=False)
        elif self.embedding == 'attention':
            # 获取Attention特定参数
            attention_params = {}
            if hasattr(self.args, 'attention_heads'):
                attention_params['num_heads'] = self.args.attention_heads
            if hasattr(self.args, 'attention_dropout'):
                attention_params['dropout'] = self.args.attention_dropout
            
            # 根据是否使用近似雅可比选择不同实现
            if self.use_approx:
                ptnet = pointnetlk_attention_approx.Pointnet_Features_Attention(
                    dim_k=self.dim_k, **attention_params)
            else:
                ptnet = pointnetlk_attention.Pointnet_Features_Attention(
                    dim_k=self.dim_k, **attention_params)
        elif self.embedding == 'fastpointtransformer':
            # FastPointTransformer也可能使用类似的头部和dropout参数
            # 您可以根据需要为FastPointTransformer添加特定的命令行参数
            fpt_params = {}
            if hasattr(self.args, 'attention_heads'): # 复用attention的参数名
                fpt_params['num_heads'] = self.args.attention_heads
            if hasattr(self.args, 'attention_dropout'): # 复用attention的参数名
                fpt_params['dropout'] = self.args.attention_dropout
            
            if self.use_approx:
                ptnet = fpt_approx.FastPointTransformer_Features_Attention(
                    dim_k=self.dim_k, **fpt_params)
            else:
                ptnet = fpt_analytical.FastPointTransformer_Features_Attention(
                    dim_k=self.dim_k, **fpt_params)
        elif self.embedding == 'cformer':
            # 获取Attention特定参数 (CFormer复用这些参数)
            cformer_params = {}
            if hasattr(self.args, 'attention_heads'):
                cformer_params['num_heads'] = self.args.attention_heads
            if hasattr(self.args, 'attention_dropout'):
                cformer_params['dropout'] = self.args.attention_dropout
            
            if self.use_approx:
                ptnet = cformer_approx.Cformer_Features_Attention(
                    dim_k=self.dim_k, **cformer_params)
            else:
                ptnet = cformer_analytical.Cformer_Features_Attention(
                    dim_k=self.dim_k, **cformer_params)
        return ptnet.float()

    def create_from_pointnet_features(self, ptnet):
        if self.embedding == '3dmamba':
            return pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba(ptnet, self.device)
        elif self.embedding == 'attention':
            if self.use_approx:
                return pointnetlk_attention_approx.AnalyticalPointNetLK_Attention(
                    ptnet, self.device, delta=self.delta, learn_delta=self.learn_delta)
            else:
                return pointnetlk_attention.AnalyticalPointNetLK_Attention(ptnet, self.device)
        elif self.embedding == 'fastpointtransformer':
            if self.use_approx:
                return fpt_approx.AnalyticalFastPointTransformerLK_Attention(
                    ptnet, self.device, delta=self.delta, learn_delta=self.learn_delta)
            else:
                return fpt_analytical.AnalyticalFastPointTransformerLK_Attention(ptnet, self.device)
        elif self.embedding == 'cformer':
            if self.use_approx:
                return cformer_approx.AnalyticalCFormerLK_Attention(
                    ptnet, self.device, delta=self.delta, learn_delta=self.learn_delta)
            else:
                return cformer_analytical.AnalyticalCFormerLK_Attention(ptnet, self.device)
        else:
            return model.AnalyticalPointNetLK(ptnet, self.device)

    def create_model(self):
        device = torch.device(self.args.device)

        if self.args.model_type == 'pointnet_attention':
            # 从正确的路径导入
            from PointNet_files.pointnetlk_attention_approx import Pointnet_Features_Attention, AnalyticalPointNetLK_Attention
            
            # 实例化特征提取器
            feature_extractor = Pointnet_Features_Attention(
                dim_k=self.args.dim_k,
                num_heads=self.args.num_attention_heads,
                dropout=self.args.attention_dropout
            )
            # 实例化LK模型
            model = AnalyticalPointNetLK_Attention(
                ptnet=feature_extractor,
                device=device,
                delta=self.args.lk_delta,
                learn_delta=self.args.lk_learn_delta
            )
            return model
            
        elif self.args.model_type == 'fpt_attention':
            # 从正确的路径导入
            from PointNet_files.fastpointtransformer_approx_jacobian import FastPointTransformer_Features_Attention, AnalyticalFastPointTransformerLK_Attention
            
            # 实例化特征提取器
            feature_extractor = FastPointTransformer_Features_Attention(
                dim_k=self.args.dim_k,
                num_heads=self.args.num_attention_heads,
                dropout=self.args.attention_dropout
            )
            # 实例化的LK模型
            model = AnalyticalFastPointTransformerLK_Attention(
                fpt_model=feature_extractor, # 注意构造函数参数名
                device=device,
                delta=self.args.lk_delta,
                learn_delta=self.args.lk_learn_delta
            )
            return model
            
        elif self.args.model_type == 'original_pointnetlk':
            # 这里是您原始的PointNetLK模型创建逻辑
            # 可能依赖 self.args.embedding 等参数
            # 例如:
            # import PointNet_files.pointnet_pytorch as pt_pytorch # 假设原始模型在这里
            # ptnet = pt_pytorch.PointNet_features(dim_k=self.args.dim_k, use_tnet=True)
            # model = AnalyticalPointNetLK(ptnet=ptnet, device=device) 
            # return model
            pass # 请替换为您的原始模型加载逻辑
            
        else:
            raise ValueError(f"未知的模型类型: {self.args.model_type}")

        # ... 可能还有其他模型加载后的处理 ...

    def train_one_epoch(self, ptnetlk, trainloader, optimizer, device, mode, data_type='synthetic', num_random_points=100):
        ptnetlk.float()
        ptnetlk.train()
        vloss = 0.0
        gloss = 0.0
        batches = 0

        for i, data in enumerate(trainloader):
            loss, loss_pose = self.compute_loss(ptnetlk, data, device, mode, data_type, num_random_points)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            vloss += (loss.item())
            gloss += (loss_pose.item())
            batches += 1

        ave_vloss = float(vloss) / batches
        ave_loss_pose = float(gloss) / batches
        
        return ave_vloss, ave_loss_pose

    def eval_one_epoch(self, ptnetlk, evalloader, device, mode, data_type='synthetic', num_random_points=100):
        ptnetlk.eval()
        vloss = 0.0
        gloss = 0.0
        batches = 0

        for _, data in enumerate(evalloader):
            loss, loss_pose = self.compute_loss(ptnetlk, data, device, mode, data_type, num_random_points)

            vloss += (loss.item())
            gloss += (loss_pose.item())
            batches += 1

        ave_vloss = float(vloss)/batches
        ave_loss_pose = float(gloss)/batches
        
        return ave_vloss, ave_loss_pose

    def test_one_epoch(self, ptnetlk, testloader, device, mode, data_type='synthetic', vis=False, toyexample=False):
        ptnetlk.eval()
        rotations_gt = []
        translation_gt = []
        rotations_ab = []
        translation_ab = []
        
        # 强制禁用可视化
        vis = False
        
        # 按场景统计结果
        scene_results = {}
        
        for i, data in tqdm.tqdm(enumerate(testloader), total=len(testloader), ncols=73, leave=False):
            # 获取样本场景信息（如果有）
            scene_name = None
            if hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'pair_scenes'):
                if isinstance(testloader.dataset, torch.utils.data.Subset):
                    idx = testloader.dataset.indices[i]
                    scene_name = testloader.dataset.dataset.pair_scenes[idx]
            
            # if voxelization: VxNx3, Vx3, 1x4x4
            if data_type == 'real':
                if vis:
                    voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose, p0, p1 = data
                    p0 = p0.float().to(device)
                    p1 = p1.float().to(device)
                else:
                    voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose = data
                voxel_features_p0 = voxel_features_p0.reshape(-1, voxel_features_p0.shape[2], voxel_features_p0.shape[3]).to(device)
                voxel_features_p1 = voxel_features_p1.reshape(-1, voxel_features_p1.shape[2], voxel_features_p1.shape[3]).to(device)
                voxel_coords_p0 = voxel_coords_p0.reshape(-1, voxel_coords_p0.shape[2]).to(device)
                voxel_coords_p1 = voxel_coords_p1.reshape(-1, voxel_coords_p1.shape[2]).to(device)
                gt_pose = gt_pose.float().to(device)
                if voxel_features_p0.shape[0] == 0 or voxel_features_p1.shape[0] == 0:
                    data_flag = -1
                else:
                    data_flag = 1
            else:
                p0, p1, gt_pose = data
                p0 = p0.float().to(device)
                p1 = p1.float().to(device)
                if p0.shape[0] == 0 or p1.shape[0] == 0:
                    data_flag = -1
                else:
                    data_flag = 1
                    
            if data_flag == -1:
                print('empty data, continue!')
                continue
            
            if vis:
                start_idx = 0
                end_idx = self.max_iter + 1
            else:
                start_idx = self.max_iter
                end_idx = self.max_iter + 1 
                
            for j in range(start_idx, end_idx):
                # 根据模型类型选择正确的do_forward方法
                if isinstance(ptnetlk, pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba):
                    do_forward_func = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.do_forward
                elif isinstance(ptnetlk, pointnetlk_attention_approx.AnalyticalPointNetLK_Attention):
                    do_forward_func = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.do_forward
                elif isinstance(ptnetlk, pointnetlk_attention.AnalyticalPointNetLK_Attention): # 添加对普通Attention模型的检查
                    do_forward_func = pointnetlk_attention.AnalyticalPointNetLK_Attention.do_forward
                elif isinstance(ptnetlk, fpt_analytical.AnalyticalFastPointTransformerLK_Attention):
                    do_forward_func = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.do_forward
                elif isinstance(ptnetlk, fpt_approx.AnalyticalFastPointTransformerLK_Attention):
                    do_forward_func = fpt_approx.AnalyticalFastPointTransformerLK_Attention.do_forward
                elif isinstance(ptnetlk, cformer_approx.AnalyticalCFormerLK_Attention):
                    do_forward_func = cformer_approx.AnalyticalCFormerLK_Attention.do_forward
                elif isinstance(ptnetlk, cformer_analytical.AnalyticalCFormerLK_Attention):
                    do_forward_func = cformer_analytical.AnalyticalCFormerLK_Attention.do_forward
                else:
                    do_forward_func = model.AnalyticalPointNetLK.do_forward
                
                if data_type == 'real':
                    r = do_forward_func(ptnetlk, voxel_features_p0, voxel_coords_p0,
                                voxel_features_p1, voxel_coords_p1, j, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)
                else:
                    r = do_forward_func(ptnetlk, p0, None,
                                p1, None, j, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)

                estimated_pose = ptnetlk.g

                ig_gt = gt_pose.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
                g_hat = estimated_pose.cpu().contiguous().view(-1, 4, 4).detach() # --> [1, 4, 4], p1->p0 (S->T)

                # 在传递给comp函数前检查并调整维度
                if gt_pose.dim() > 3:
                    gt_pose = gt_pose.squeeze(1)
                    
                # 根据模型类型选择正确的comp方法
                if isinstance(ptnetlk, pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba):
                    loss_pose = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.comp(estimated_pose, gt_pose)
                elif isinstance(ptnetlk, pointnetlk_attention_approx.AnalyticalPointNetLK_Attention):
                    loss_pose = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.comp(estimated_pose, gt_pose)
                elif isinstance(ptnetlk, pointnetlk_attention.AnalyticalPointNetLK_Attention): # 添加对普通Attention模型的检查
                    loss_pose = pointnetlk_attention.AnalyticalPointNetLK_Attention.comp(estimated_pose, gt_pose)
                elif isinstance(ptnetlk, fpt_analytical.AnalyticalFastPointTransformerLK_Attention):
                    loss_pose = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.comp(estimated_pose, gt_pose)
                elif isinstance(ptnetlk, fpt_approx.AnalyticalFastPointTransformerLK_Attention):
                    loss_pose = fpt_approx.AnalyticalFastPointTransformerLK_Attention.comp(estimated_pose, gt_pose)
                elif isinstance(ptnetlk, cformer_approx.AnalyticalCFormerLK_Attention):
                    loss_pose = cformer_approx.AnalyticalCFormerLK_Attention.comp(estimated_pose, gt_pose)
                elif isinstance(ptnetlk, cformer_analytical.AnalyticalCFormerLK_Attention):
                    loss_pose = cformer_analytical.AnalyticalCFormerLK_Attention.comp(estimated_pose, gt_pose)
                else:
                    loss_pose = model.AnalyticalPointNetLK.comp(estimated_pose, gt_pose)
                
                pr = ptnetlk.prev_r
                if pr is not None:
                    # 根据模型类型选择正确的rsq方法
                    if isinstance(ptnetlk, pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba):
                        loss_r = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.rsq(r - pr)
                    elif isinstance(ptnetlk, pointnetlk_attention_approx.AnalyticalPointNetLK_Attention):
                        loss_r = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.rsq(r - pr)
                    elif isinstance(ptnetlk, pointnetlk_attention.AnalyticalPointNetLK_Attention): # 添加对普通Attention模型的检查
                        loss_r = pointnetlk_attention.AnalyticalPointNetLK_Attention.rsq(r - pr)
                    elif isinstance(ptnetlk, fpt_analytical.AnalyticalFastPointTransformerLK_Attention):
                        loss_r = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.rsq(r - pr)
                    elif isinstance(ptnetlk, fpt_approx.AnalyticalFastPointTransformerLK_Attention):
                        loss_r = fpt_approx.AnalyticalFastPointTransformerLK_Attention.rsq(r - pr)
                    elif isinstance(ptnetlk, cformer_approx.AnalyticalCFormerLK_Attention):
                        loss_r = cformer_approx.AnalyticalCFormerLK_Attention.rsq(r - pr)
                    elif isinstance(ptnetlk, cformer_analytical.AnalyticalCFormerLK_Attention):
                        loss_r = cformer_analytical.AnalyticalCFormerLK_Attention.rsq(r - pr)
                    else:
                        loss_r = model.AnalyticalPointNetLK.rsq(r - pr)
                else:
                    if isinstance(ptnetlk, pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba):
                        loss_r = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.rsq(r)
                    elif isinstance(ptnetlk, pointnetlk_attention_approx.AnalyticalPointNetLK_Attention):
                        loss_r = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.rsq(r)
                    elif isinstance(ptnetlk, pointnetlk_attention.AnalyticalPointNetLK_Attention): # 添加对普通Attention模型的检查
                        loss_r = pointnetlk_attention.AnalyticalPointNetLK_Attention.rsq(r)
                    elif isinstance(ptnetlk, fpt_analytical.AnalyticalFastPointTransformerLK_Attention):
                        loss_r = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.rsq(r)
                    elif isinstance(ptnetlk, fpt_approx.AnalyticalFastPointTransformerLK_Attention):
                        loss_r = fpt_approx.AnalyticalFastPointTransformerLK_Attention.rsq(r)
                    elif isinstance(ptnetlk, cformer_approx.AnalyticalCFormerLK_Attention):
                        loss_r = cformer_approx.AnalyticalCFormerLK_Attention.rsq(r)
                    elif isinstance(ptnetlk, cformer_analytical.AnalyticalCFormerLK_Attention):
                        loss_r = cformer_analytical.AnalyticalCFormerLK_Attention.rsq(r)
                    else:
                        loss_r = model.AnalyticalPointNetLK.rsq(r)
                
                loss = loss_r + loss_pose

                dg = g_hat.bmm(ig_gt)   # if correct, dg == identity matrix.
                dx = utils.log(dg)   # --> [1, 6] (if corerct, dx == zero vector)
                dn = dx.norm(p=2, dim=1)   # --> [1]
                dm = dn.mean()

                
                LOGGER.info('test, %d/%d, %d iterations, %f', i, len(testloader), j, dm)
                    
            # euler representation for ground truth
            tform_gt = ig_gt.squeeze().numpy().transpose()
            R_gt = tform_gt[:3, :3]
            euler_angle = Rotation.from_matrix(R_gt)
            anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler('zyx')
            angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
            rotations_gt.append(angle_gt)
            trans_gt_t = -R_gt.dot(tform_gt[3, :3])
            translation_gt.append(trans_gt_t)
            # euler representation for predicted transformation
            tform_ab = g_hat.squeeze().numpy()
            R_ab = tform_ab[:3, :3]
            euler_angle = Rotation.from_matrix(R_ab)
            anglez_ab, angley_ab, anglex_ab = euler_angle.as_euler('zyx')
            angle_ab = np.array([anglex_ab, angley_ab, anglez_ab])
            rotations_ab.append(angle_ab)
            trans_ab = tform_ab[:3, 3]
            translation_ab.append(trans_ab)

            # 记录按场景分类的结果
            if scene_name:
                if scene_name not in scene_results:
                    scene_results[scene_name] = {
                        'rotations_gt': [],
                        'translation_gt': [],
                        'rotations_ab': [],
                        'translation_ab': []
                    }
                scene_results[scene_name]['rotations_gt'].append(angle_gt)
                scene_results[scene_name]['translation_gt'].append(trans_gt_t)
                scene_results[scene_name]['rotations_ab'].append(angle_ab)
                scene_results[scene_name]['translation_ab'].append(trans_ab)

        # 计算并保存总体结果
        rot_err, trans_err = utils.test_metrics(rotations_gt, translation_gt, rotations_ab, translation_ab, self.filename)
        
        # 保存按场景分类的结果
        try:
            if scene_results:
                scene_summary_file = f'{self.filename}_scene_results.txt'
                with open(scene_summary_file, 'w') as f:
                    f.write(f'总体旋转误差: {rot_err:.4f} 度\n')
                    f.write(f'总体平移误差: {trans_err:.4f} 厘米\n\n')
                    f.write('按场景统计结果:\n')
                    f.write('=' * 50 + '\n')
                    
                    for scene in sorted(scene_results.keys()):
                        scene_rot_gt = np.concatenate(scene_results[scene]['rotations_gt'], axis=0).reshape(-1, 3)
                        scene_trans_gt = np.concatenate(scene_results[scene]['translation_gt'], axis=0).reshape(-1, 3)
                        scene_rot_ab = np.concatenate(scene_results[scene]['rotations_ab'], axis=0).reshape(-1, 3)
                        scene_trans_ab = np.concatenate(scene_results[scene]['translation_ab'], axis=0).reshape(-1, 3)
                        
                        scene_rot_err = np.sqrt(np.mean((np.degrees(scene_rot_ab) - np.degrees(scene_rot_gt)) ** 2, axis=1)).mean()
                        scene_trans_err = np.sqrt(np.mean((scene_trans_ab - scene_trans_gt) ** 2, axis=1)).mean()
                        
                        f.write(f'场景 {scene}:\n')
                        f.write(f'  样本数量: {len(scene_results[scene]["rotations_gt"])}\n')
                        f.write(f'  旋转误差: {scene_rot_err:.4f} 度\n')
                        f.write(f'  平移误差: {scene_trans_err:.4f} 厘米\n')
                        f.write('-' * 50 + '\n')
            
                print(f"按场景统计结果已保存至: {scene_summary_file}")
        except Exception as e:
            print(f"按场景保存结果时出错: {e}")
        
        return rot_err, trans_err

    def compute_loss(self, ptnetlk, data, device, mode, data_type='synthetic', num_random_points=100):
        # 根据模型类型选择正确的静态方法
        if isinstance(ptnetlk, pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba):
            do_forward_func = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.do_forward
            comp_func = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.comp
            rsq_func = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.rsq
        elif isinstance(ptnetlk, pointnetlk_attention_approx.AnalyticalPointNetLK_Attention):
            do_forward_func = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.do_forward
            comp_func = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.comp
            rsq_func = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.rsq
        elif isinstance(ptnetlk, pointnetlk_attention.AnalyticalPointNetLK_Attention): # 添加对普通Attention模型的检查
            do_forward_func = pointnetlk_attention.AnalyticalPointNetLK_Attention.do_forward
            comp_func = pointnetlk_attention.AnalyticalPointNetLK_Attention.comp
            rsq_func = pointnetlk_attention.AnalyticalPointNetLK_Attention.rsq
        elif isinstance(ptnetlk, fpt_analytical.AnalyticalFastPointTransformerLK_Attention):
            do_forward_func = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.do_forward
            comp_func = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.comp
            rsq_func = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.rsq
        elif isinstance(ptnetlk, fpt_approx.AnalyticalFastPointTransformerLK_Attention):
            do_forward_func = fpt_approx.AnalyticalFastPointTransformerLK_Attention.do_forward
            comp_func = fpt_approx.AnalyticalFastPointTransformerLK_Attention.comp
            rsq_func = fpt_approx.AnalyticalFastPointTransformerLK_Attention.rsq
        elif isinstance(ptnetlk, cformer_approx.AnalyticalCFormerLK_Attention):
            do_forward_func = cformer_approx.AnalyticalCFormerLK_Attention.do_forward
            comp_func = cformer_approx.AnalyticalCFormerLK_Attention.comp
            rsq_func = cformer_approx.AnalyticalCFormerLK_Attention.rsq
        elif isinstance(ptnetlk, cformer_analytical.AnalyticalCFormerLK_Attention):
            do_forward_func = cformer_analytical.AnalyticalCFormerLK_Attention.do_forward
            comp_func = cformer_analytical.AnalyticalCFormerLK_Attention.comp
            rsq_func = cformer_analytical.AnalyticalCFormerLK_Attention.rsq
        else:
            do_forward_func = model.AnalyticalPointNetLK.do_forward
            comp_func = model.AnalyticalPointNetLK.comp
            rsq_func = model.AnalyticalPointNetLK.rsq
        
        # 1. non-voxelization
        if data_type == 'synthetic':
            p0, p1, gt_pose = data
            p0 = p0.to(self.device)
            p1 = p1.to(self.device)
            gt_pose = gt_pose.to(device)
            r = do_forward_func(ptnetlk, p0, None,
                                p1, None, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type, num_random_points)
        else:
            # 2. voxelization
            voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose = data
            voxel_features_p0 = voxel_features_p0.reshape(-1, voxel_features_p0.shape[2], voxel_features_p0.shape[3]).to(device)
            voxel_features_p1 = voxel_features_p1.reshape(-1, voxel_features_p1.shape[2], voxel_features_p1.shape[3]).to(device)
            voxel_coords_p0 = voxel_coords_p0.reshape(-1, voxel_coords_p0.shape[2]).to(device)
            voxel_coords_p1 = voxel_coords_p1.reshape(-1, voxel_coords_p1.shape[2]).to(device)
            gt_pose = gt_pose.reshape(-1, gt_pose.shape[2], gt_pose.shape[3]).to(device)
            
            r = do_forward_func(ptnetlk, voxel_features_p0, voxel_coords_p0,
                    voxel_features_p1, voxel_coords_p1, self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type, num_random_points)

        estimated_pose = ptnetlk.g

        # 在传递给comp函数前检查并调整维度
        if gt_pose.dim() > 3:
            gt_pose = gt_pose.squeeze(1)
        loss_pose = comp_func(estimated_pose, gt_pose)
        pr = ptnetlk.prev_r
        if pr is not None:
            loss_r = rsq_func(r - pr)
        else:
            loss_r = rsq_func(r)
        loss = loss_r + loss_pose

        return loss, loss_pose

    def test_one_epoch_with_vis(self, ptnetlk, testloader, device, mode, data_type='synthetic', 
                              visualize_pert=None, visualize_samples=1, outfile=None, pose_file=None):
        """测试并保存点云数据"""
        ptnetlk.eval()
        rotations_gt = []
        translation_gt = []
        rotations_ab = []
        translation_ab = []
        
        # 记录总误差
        total_rot_error = 0.0
        total_trans_error = 0.0
        valid_samples = 0
        
        # 获取当前扰动文件名称
        current_pert_file = None
        if outfile:
            # 获取父目录名称（表示角度），例如"060"
            results_path = Path(outfile)
            angle_dir = results_path.parent.name
            if angle_dir.isdigit():
                # 构造pert_XXX.csv格式
                current_pert_file = f"pert_{angle_dir}.csv"
                print(f"当前扰动文件: {current_pert_file}")
        
        # 直接从姿态文件加载扰动矩阵
        real_gt_matrix = np.eye(4)  # 默认为单位矩阵
        if pose_file and os.path.exists(pose_file):
            print(f"直接从姿态文件加载GT矩阵: {pose_file}")
            try:
                # 从CSV读取3x4矩阵 (前3行)
                pose_data = np.loadtxt(pose_file, delimiter=',')
                
                # 确保数据正确加载
                print(f"姿态文件数据形状: {pose_data.shape}")
                print(f"姿态文件内容预览:\n{pose_data[:3, :]}")
                
                # 构建4x4变换矩阵
                real_gt_matrix = np.eye(4)
                real_gt_matrix[:3, :4] = pose_data[:3, :]
                
                print(f"构建的GT矩阵:\n{real_gt_matrix}")
            except Exception as e:
                print(f"加载姿态文件时出错: {e}")
                import traceback
                print(traceback.format_exc())
        else:
            print(f"警告: 未提供姿态文件或文件不存在: {pose_file}")
        
        # 创建可视化目录
        vis_dir = None
        if outfile and visualize_pert and current_pert_file in visualize_pert:
            base_dir = Path(outfile).parent.parent
            vis_dir = base_dir / "visualize" / Path(outfile).parent.name
            os.makedirs(vis_dir, exist_ok=True)
            print(f"已创建可视化目录: {vis_dir}")
        
        # 决定需要可视化的样本索引
        vis_indices = []
        if vis_dir and visualize_samples > 0:
            import random
            total_samples = len(testloader)
            vis_indices = random.sample(range(min(total_samples, 100)), min(total_samples, visualize_samples))
            print(f"将对以下样本索引进行可视化: {vis_indices}")
        
        # 创建可视化日志文件
        vis_log_file = None
        if vis_dir:
            vis_log_file = vis_dir / "visualization_log.txt"
            with open(vis_log_file, 'w') as f:
                f.write("# PointNetLK 配准可视化日志\n")
                f.write(f"# 扰动文件: {current_pert_file}\n")
                f.write(f"# 创建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("样本ID,场景名称,旋转误差(度),平移误差(厘米),源文件,目标文件\n")
                f.write("-" * 80 + "\n")
        
        for i, data in enumerate(testloader):
            # 获取样本场景信息（如果有）
            scene_name = "unknown"
            source_file = None
            target_file = None
            
            # 尝试获取场景名称和文件路径
            try:
                if hasattr(testloader.dataset, 'get_cloud_info'):
                    info = testloader.dataset.get_cloud_info(i)
                    if info:
                        scene_name = info.get('scene_name', scene_name)
                        source_file = info.get('source_file', None)
                        target_file = info.get('target_file', None)
            
                # 如果没有获取到，尝试从数据集的其他属性获取
                if scene_name == "unknown" and hasattr(testloader.dataset, 'dataset'):
                    if hasattr(testloader.dataset.dataset, 'pair_scenes'):
                        if isinstance(testloader.dataset, torch.utils.data.Subset):
                            idx = testloader.dataset.indices[i]
                            scene_name = testloader.dataset.dataset.pair_scenes[idx]
                            
                    # 尝试获取源文件和目标文件
                    if source_file is None and hasattr(testloader.dataset.dataset, 'pairs'):
                        if isinstance(testloader.dataset, torch.utils.data.Subset):
                            idx = testloader.dataset.indices[i]
                            if idx < len(testloader.dataset.dataset.pairs):
                                source_file, target_file = testloader.dataset.dataset.pairs[idx]
            except Exception as e:
                print(f"获取样本信息时出错: {e}")
            
            # if voxelization: VxNx3, Vx3, 1x4x4
            if data_type == 'real':
                voxel_features_p0, voxel_coords_p0, voxel_features_p1, voxel_coords_p1, gt_pose = data
                voxel_features_p0 = voxel_features_p0.reshape(-1, voxel_features_p0.shape[2], voxel_features_p0.shape[3]).to(device)
                voxel_features_p1 = voxel_features_p1.reshape(-1, voxel_features_p1.shape[2], voxel_features_p1.shape[3]).to(device)
                voxel_coords_p0 = voxel_coords_p0.reshape(-1, voxel_coords_p0.shape[2]).to(device)
                voxel_coords_p1 = voxel_coords_p1.reshape(-1, voxel_coords_p1.shape[2]).to(device)
                gt_pose = gt_pose.float().to(device)
                if voxel_features_p0.shape[0] == 0 or voxel_features_p1.shape[0] == 0:
                    data_flag = -1
                else:
                    data_flag = 1
            else:
                p0, p1, gt_pose = data
                p0 = p0.float().to(device)
                p1 = p1.float().to(device)
                gt_pose = gt_pose.float().to(device)
                if p0.shape[0] == 0 or p1.shape[0] == 0:
                    data_flag = -1
                else:
                    data_flag = 1
                    
            if data_flag == -1:
                print('empty data, continue!')
                continue
            
            # 根据最大迭代次数获取最终结果
            j = self.max_iter
            
            # 为test_one_epoch_with_vis选择正确的do_forward和comp函数
            current_do_forward_func = model.AnalyticalPointNetLK.do_forward # 默认
            current_comp_func = model.AnalyticalPointNetLK.comp # 默认

            if isinstance(ptnetlk, pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba):
                current_do_forward_func = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.do_forward
                current_comp_func = pointnetlk_3dmamba_improved.AnalyticalPointNetLK_Mamba.comp
            elif isinstance(ptnetlk, pointnetlk_attention_approx.AnalyticalPointNetLK_Attention):
                current_do_forward_func = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.do_forward
                current_comp_func = pointnetlk_attention_approx.AnalyticalPointNetLK_Attention.comp
            elif isinstance(ptnetlk, pointnetlk_attention.AnalyticalPointNetLK_Attention):
                 current_do_forward_func = pointnetlk_attention.AnalyticalPointNetLK_Attention.do_forward
                 current_comp_func = pointnetlk_attention.AnalyticalPointNetLK_Attention.comp
            elif isinstance(ptnetlk, fpt_analytical.AnalyticalFastPointTransformerLK_Attention):
                current_do_forward_func = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.do_forward
                current_comp_func = fpt_analytical.AnalyticalFastPointTransformerLK_Attention.comp
            elif isinstance(ptnetlk, fpt_approx.AnalyticalFastPointTransformerLK_Attention):
                current_do_forward_func = fpt_approx.AnalyticalFastPointTransformerLK_Attention.do_forward
                current_comp_func = fpt_approx.AnalyticalFastPointTransformerLK_Attention.comp
            elif isinstance(ptnetlk, cformer_approx.AnalyticalCFormerLK_Attention):
                current_do_forward_func = cformer_approx.AnalyticalCFormerLK_Attention.do_forward
                current_comp_func = cformer_approx.AnalyticalCFormerLK_Attention.comp
            elif isinstance(ptnetlk, cformer_analytical.AnalyticalCFormerLK_Attention):
                current_do_forward_func = cformer_analytical.AnalyticalCFormerLK_Attention.do_forward
                current_comp_func = cformer_analytical.AnalyticalCFormerLK_Attention.comp
            
            if data_type == 'real':
                r = current_do_forward_func(ptnetlk, voxel_features_p0, voxel_coords_p0,
                            voxel_features_p1, voxel_coords_p1, j, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)
                loss_pose = current_comp_func(ptnetlk.g, gt_pose)
            else:
                r = current_do_forward_func(ptnetlk, p0, None,
                            p1, None, j, self.xtol, self.p0_zero_mean, self.p1_zero_mean, mode, data_type)
                loss_pose = current_comp_func(ptnetlk.g, gt_pose)

            estimated_pose = ptnetlk.g

            ig_gt = gt_pose.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
            g_hat = estimated_pose.cpu().contiguous().view(-1, 4, 4).detach() # --> [1, 4, 4], p1->p0 (S->T)

            # 计算误差
            dg = g_hat.bmm(ig_gt)   # if correct, dg == identity matrix.
            dx = utils.log(dg)   # --> [1, 6] (if corerct, dx == zero vector)
            
            # 分别计算旋转和平移误差
            rot_error = dx[:, :3]  # 旋转误差
            trans_error = dx[:, 3:]  # 平移误差
            
            rot_norm = rot_error.norm(p=2, dim=1)  # 旋转误差范数
            trans_norm = trans_error.norm(p=2, dim=1)  # 平移误差范数
            
            # 累加误差
            total_rot_error += rot_norm.item()
            total_trans_error += trans_norm.item()
            valid_samples += 1
            
            dn = dx.norm(p=2, dim=1)   # --> [1]
            dm = dn.mean()
            
            print(f'test, {i}/{len(testloader)}, {j} iterations, rot_error: {rot_norm.item():.4f}, trans_error: {trans_norm.item():.4f}')
                
            # 判断是否是需要可视化的样本
            if i in vis_indices and vis_dir:
                try:
                    print(f"正在为样本 {i} 保存点云和矩阵...")
                    
                    # 创建当前样本的唯一标识符
                    sample_id = f"{scene_name}_{i:04d}"
                    
                    # 创建输出文件路径
                    source_out = vis_dir / f"{sample_id}_source.ply"
                    target_out = vis_dir / f"{sample_id}_target.ply"
                    est_out = vis_dir / f"{sample_id}_estimated.ply"
                    
                    # 获取点云数据
                    p0_np = p0[0].cpu().numpy()  # 目标点云
                    p1_np = p1[0].cpu().numpy()  # 源点云
                    
                    # 计算变换后的点云数据 - 使用估计变换将源点云变换到目标空间
                    est_tensor = utils.transform(estimated_pose, p1.to(device))
                    p0_hat_np = est_tensor.cpu().numpy()[0]
                    
                    # 创建点云文件
                    self.save_ply(p0_np, target_out)
                    self.save_ply(p1_np, source_out)
                    self.save_ply(p0_hat_np, est_out)
                    
                    # 记录到日志
                    with open(vis_log_file, 'a') as f:
                        f.write(f"{sample_id},{scene_name},{rot_norm.item():.4f},{trans_norm.item():.4f}")
                        if source_file:
                            f.write(f",{source_file}")
                        if target_file:
                            f.write(f",{target_file}")
                        f.write("\n")
                    
                    # 保存变换矩阵 - 使用从姿态文件加载的真实GT矩阵
                    gt_matrix_file = vis_dir / f"{sample_id}_gt_matrix.txt"
                    est_matrix_file = vis_dir / f"{sample_id}_est_matrix.txt"
                    
                    print(f"保存GT矩阵到: {gt_matrix_file}")
                    print(f"保存估计矩阵到: {est_matrix_file}")
                    
                    # 使用从姿态文件加载的真实GT矩阵，不使用ig_gt
                    print(f"使用从姿态文件加载的真实GT矩阵")
                    np.savetxt(str(gt_matrix_file), real_gt_matrix, fmt='%.6f')
                    
                    # 保存估计变换矩阵
                    g_hat_np = g_hat.squeeze().cpu().numpy()
                    np.savetxt(str(est_matrix_file), g_hat_np, fmt='%.6f')
                    
                    print(f"矩阵保存完成")
                    
                except Exception as e:
                    print(f"保存可视化结果时出错: {e}")
                    import traceback
                    print(traceback.format_exc())
                
            # euler representation for ground truth
            tform_gt = ig_gt.squeeze().numpy().transpose()
            R_gt = tform_gt[:3, :3]
            euler_angle = Rotation.from_matrix(R_gt)
            anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler('zyx')
            angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
            rotations_gt.append(angle_gt)
            trans_gt_t = -R_gt.dot(tform_gt[3, :3])
            translation_gt.append(trans_gt_t)
            # euler representation for predicted transformation
            tform_ab = g_hat.squeeze().numpy()
            R_ab = tform_ab[:3, :3]
            euler_angle = Rotation.from_matrix(R_ab)
            anglez_ab, angley_ab, anglex_ab = euler_angle.as_euler('zyx')
            angle_ab = np.array([anglex_ab, angley_ab, anglez_ab])
            rotations_ab.append(angle_ab)
            trans_ab = tform_ab[:3, 3]
            translation_ab.append(trans_ab)
        
        # 计算平均误差
        avg_rot_error = total_rot_error / valid_samples if valid_samples > 0 else 0
        avg_trans_error = total_trans_error / valid_samples if valid_samples > 0 else 0
        
        print(f"测试完成: 平均旋转误差={avg_rot_error:.4f}度, 平均平移误差={avg_trans_error:.4f}厘米")
        
        return avg_rot_error, avg_trans_error

    def save_ply(self, points, filename):
        """使用纯文本格式保存点云，不使用Open3D"""
        try:
            # 替换文件扩展名为xyz格式
            output_file = str(filename).replace('.ply', '.xyz')
            
            # 使用简单格式保存点云：X Y Z
            np.savetxt(output_file, points, fmt='%.6f')
            print(f"已保存点云到: {output_file}")
            return True
        except Exception as e:
            print(f"保存点云失败: {e}")
            return False

