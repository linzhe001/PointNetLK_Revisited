import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extraction import FeatureExtractor
import utils

class FastPointTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FastPointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: [B, N, C]
        x = self.linear1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        return x

class FastPointTransformer_Features(FeatureExtractor):
    """
    基于FastPointTransformer的特征提取网络
    """
    def __init__(self, dim_k=1024):
        super(FastPointTransformer_Features, self).__init__(dim_k)
        self.extractor_type = "fastpointtransformer_v1"
        
        self.block1 = FastPointTransformerBlock(3, 64)
        self.block2 = FastPointTransformerBlock(64, 128)
        self.block3 = FastPointTransformerBlock(128, dim_k)

    def forward(self, points, iter):
        """
        前向传播，提取点云特征
        
        参数:
            points: 输入点云 [B, N, 3]
            iter: 迭代标志
            
        返回:
            当iter=-1时: 返回特征和中间结果用于雅可比矩阵计算
            其他情况: 返回特征向量 [B,dim_k]
        """
        # points: [B, N, 3]
        x = points  # [B, N, 3]
        if iter == -1:
            x1 = self.block1(x)  # [B, N, 64]
            x2 = self.block2(x1)  # [B, N, 128]
            x3 = self.block3(x2)  # [B, N, dim_k]
            x_global = torch.max(x3, dim=1)[0]  # [B, dim_k]
            
            # 激活掩码
            M1 = (x1 > 0).float()
            M2 = (x2 > 0).float()
            M3 = (x3 > 0).float()
            
            # 提取权重
            A1 = self.block1.linear1.weight
            A2 = self.block2.linear1.weight
            A3 = self.block3.linear1.weight
            
            # 线性层输出
            A1_x = x1
            A2_x = x2
            A3_x = x3
            
            # 批归一化输出
            bn1_x = x1
            bn2_x = x2
            bn3_x = x3
            
            # 最大池化索引
            max_idx = torch.argmax(x3, dim=1)
            
            return x_global, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = torch.max(x, dim=1)[0]
            return x
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, mode="train", 
                     voxel_coords_diff=None, data_type='synthetic', num_points=None):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn, a_fn, ax_fn, bn_fn: 中间层输出
            max_idx: 最大池化索引
            mode: 训练或测试模式
            voxel_coords_diff: 体素坐标差异（用于真实数据）
            data_type: 数据类型（'synthetic'或'real'）
            num_points: 点云中的点数量
            
        返回:
            雅可比矩阵 [B,K,6]
        """
        if num_points is None:
            num_points = p0.shape[1]
        batch_size = p0.shape[0]
        dim_k = self.dim_k
        device = p0.device
        
        # 1. 计算变形雅可比矩阵
        g_ = torch.zeros(batch_size, 6).to(device)
        warp_jac = utils.compute_warp_jac(g_, p0, num_points)   # B x N x 3 x 6
        
        # 2. 计算特征雅可比矩阵
        feature_j = feature_jac_fastpointtransformer_v1(mask_fn, a_fn, ax_fn, bn_fn, device).to(device)
        feature_j = feature_j.permute(0, 3, 1, 2)   # B x N x 6 x K
        
        # 3. 组合得到最终雅可比矩阵
        J_ = torch.einsum('ijkl,ijkm->ijlm', feature_j, warp_jac)   # B x N x K x 6
        
        # 4. 根据最大池化索引进行处理
        jac_max = J_.permute(0, 2, 1, 3)   # B x K x N x 6
        jac_max_ = []
        
        for i in range(batch_size):
            jac_max_t = jac_max[i, torch.arange(dim_k), max_idx[i]]
            jac_max_.append(jac_max_t)
        jac_max_ = torch.cat(jac_max_)
        J_ = jac_max_.reshape(batch_size, dim_k, 6)   # B x K x 6
        
        if len(J_.size()) < 3:
            J = J_.unsqueeze(0)
        else:
            J = J_
        
        # 处理真实数据的特殊情况
        if mode == 'test' and data_type == 'real':
            J_ = J_.permute(1, 0, 2).reshape(dim_k, -1)   # K x (V6)
            warp_condition = utils.cal_conditioned_warp_jacobian(voxel_coords_diff)   # V x 6 x 6
            warp_condition = warp_condition.permute(0,2,1).reshape(-1, 6)   # (V6) x 6
            J = torch.einsum('ij,jk->ik', J_, warp_condition).unsqueeze(0)   # 1 X K X 6
            
        return J


# 特征雅可比矩阵计算 for FastPointTransformer
def feature_jac_fastpointtransformer_v1(M, A, Ax, BN, device):
    """
    计算FastPointTransformer特征雅可比矩阵
    
    参数:
        M: 激活掩码列表 [M1, M2, M3]
        A: 权重列表 [A1, A2, A3]
        Ax: 线性层输出列表 [A1_x, A2_x, A3_x]
        BN: 批归一化输出列表 [bn1_x, bn2_x, bn3_x]
        device: 计算设备
        
    返回:
        特征雅可比矩阵
    """
    # 解包输入列表
    A1, A2, A3 = A
    M1, M2, M3 = M
    Ax1, Ax2, Ax3 = Ax
    BN1, BN2, BN3 = BN

    # 1 x c_in x c_out x 1
    A1 = (A1.T).detach().unsqueeze(-1)
    A2 = (A2.T).detach().unsqueeze(-1)
    A3 = (A3.T).detach().unsqueeze(-1)

    # 使用自动微分计算批量归一化的梯度
    # B x 1 x c_out x N
    dBN1 = torch.autograd.grad(outputs=BN1, inputs=Ax1, grad_outputs=torch.ones(BN1.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN2 = torch.autograd.grad(outputs=BN2, inputs=Ax2, grad_outputs=torch.ones(BN2.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN3 = torch.autograd.grad(outputs=BN3, inputs=Ax3, grad_outputs=torch.ones(BN3.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()

    # B x 1 x c_out x N
    M1 = M1.detach().unsqueeze(1)
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)

    # 使用广播计算 --> B x c_in x c_out x N
    A1BN1M1 = A1 * dBN1 * M1
    A2BN2M2 = A2 * dBN2 * M2
    A3BN3M3 = M3 * dBN3 * A3

    # 使用einsum组合
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)   # B x 3 x 64 x N
    A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)   # B x 3 x K x N
    
    feat_jac = A2BN2M2_A3BN3M3

    return feat_jac   # B x 3 x K x N
