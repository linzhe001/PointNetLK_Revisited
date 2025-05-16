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
            # Block 1 详细信息
            # 线性层1
            x = self.block1.linear1(x)
            A1_x = x
            # 批归一化1
            x = self.block1.bn1(x.transpose(1, 2)).transpose(1, 2)
            bn1_x = x
            # ReLU激活1
            x = self.block1.relu(x)
            M1 = (x > 0).float()
            
            # 线性层2
            x = self.block1.linear2(x)
            A1_2_x = x
            # 批归一化2
            x = self.block1.bn2(x.transpose(1, 2)).transpose(1, 2)
            bn1_2_x = x
            # ReLU激活2
            x = self.block1.relu(x)
            M1_2 = (x > 0).float()
            
            # Block 2 详细信息
            # 线性层1
            x = self.block2.linear1(x)
            A2_x = x
            # 批归一化1
            x = self.block2.bn1(x.transpose(1, 2)).transpose(1, 2)
            bn2_x = x
            # ReLU激活1
            x = self.block2.relu(x)
            M2 = (x > 0).float()
            
            # 线性层2
            x = self.block2.linear2(x)
            A2_2_x = x
            # 批归一化2
            x = self.block2.bn2(x.transpose(1, 2)).transpose(1, 2)
            bn2_2_x = x
            # ReLU激活2
            x = self.block2.relu(x)
            M2_2 = (x > 0).float()
            
            # Block 3 详细信息
            # 线性层1
            x = self.block3.linear1(x)
            A3_x = x
            # 批归一化1
            x = self.block3.bn1(x.transpose(1, 2)).transpose(1, 2)
            bn3_x = x
            # ReLU激活1
            x = self.block3.relu(x)
            M3 = (x > 0).float()
            
            # 线性层2
            x = self.block3.linear2(x)
            A3_2_x = x
            # 批归一化2
            x = self.block3.bn2(x.transpose(1, 2)).transpose(1, 2)
            bn3_2_x = x
            # ReLU激活2
            x = self.block3.relu(x)
            M3_2 = (x > 0).float()
            
            # 最大池化
            max_idx = torch.argmax(x, dim=1)
            x = torch.max(x, dim=1)[0]  # [B, dim_k]
            
            # 提取权重
            W1_1 = self.block1.linear1.weight
            W1_2 = self.block1.linear2.weight
            W2_1 = self.block2.linear1.weight
            W2_2 = self.block2.linear2.weight
            W3_1 = self.block3.linear1.weight
            W3_2 = self.block3.linear2.weight
            
            # 收集并返回中间结果
            # 修复多行返回语句
            return x, [M1, M2, M3, M1_2, M2_2, M3_2], [W1_1, W2_1, W3_1, W1_2, W2_2, W3_2], \
                   [A1_x, A2_x, A3_x, A1_2_x, A2_2_x, A3_2_x], \
                   [bn1_x, bn2_x, bn3_x, bn1_2_x, bn2_2_x, bn3_2_x], max_idx
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
            mask_fn: 激活掩码列表
            a_fn: 权重列表
            ax_fn: 线性层输出列表
            bn_fn: 批归一化输出列表
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
        M: 激活掩码列表 [M1, M2, M3, M1_2, M2_2, M3_2]
        A: 权重列表 [W1_1, W2_1, W3_1, W1_2, W2_2, W3_2]
        Ax: 线性层输出列表 [A1_x, A2_x, A3_x, A1_2_x, A2_2_x, A3_2_x]
        BN: 批归一化输出列表 [bn1_x, bn2_x, bn3_x, bn1_2_x, bn2_2_x, bn3_2_x]
        device: 计算设备
        
    返回:
        特征雅可比矩阵
    """
    # 解包输入列表 - 线性层1的权重和激活
    W1_1, W2_1, W3_1 = A[0], A[1], A[2]
    M1, M2, M3 = M[0], M[1], M[2]
    A1_x, A2_x, A3_x = Ax[0], Ax[1], Ax[2]
    bn1_x, bn2_x, bn3_x = BN[0], BN[1], BN[2]
    
    # 解包输入列表 - 线性层2的权重和激活
    W1_2, W2_2, W3_2 = A[3], A[4], A[5]
    M1_2, M2_2, M3_2 = M[3], M[4], M[5]
    A1_2_x, A2_2_x, A3_2_x = Ax[3], Ax[4], Ax[5]
    bn1_2_x, bn2_2_x, bn3_2_x = BN[3], BN[4], BN[5]

    # 权重转置并调整维度
    W1_1 = (W1_1.T).detach().unsqueeze(-1)  # 1 x c_in x c_out x 1
    W2_1 = (W2_1.T).detach().unsqueeze(-1)
    W3_1 = (W3_1.T).detach().unsqueeze(-1)
    
    W1_2 = (W1_2.T).detach().unsqueeze(-1)
    W2_2 = (W2_2.T).detach().unsqueeze(-1)
    W3_2 = (W3_2.T).detach().unsqueeze(-1)

    # 计算批归一化梯度
    dBN1 = torch.autograd.grad(outputs=bn1_x, inputs=A1_x, 
                             grad_outputs=torch.ones(bn1_x.size()).to(device), 
                             retain_graph=True)[0].unsqueeze(1).detach()
    
    dBN2 = torch.autograd.grad(outputs=bn2_x, inputs=A2_x, 
                             grad_outputs=torch.ones(bn2_x.size()).to(device), 
                             retain_graph=True)[0].unsqueeze(1).detach()
    
    dBN3 = torch.autograd.grad(outputs=bn3_x, inputs=A3_x, 
                             grad_outputs=torch.ones(bn3_x.size()).to(device), 
                             retain_graph=True)[0].unsqueeze(1).detach()
    
    dBN1_2 = torch.autograd.grad(outputs=bn1_2_x, inputs=A1_2_x, 
                               grad_outputs=torch.ones(bn1_2_x.size()).to(device), 
                               retain_graph=True)[0].unsqueeze(1).detach()
    
    dBN2_2 = torch.autograd.grad(outputs=bn2_2_x, inputs=A2_2_x, 
                               grad_outputs=torch.ones(bn2_2_x.size()).to(device), 
                               retain_graph=True)[0].unsqueeze(1).detach()
    
    dBN3_2 = torch.autograd.grad(outputs=bn3_2_x, inputs=A3_2_x, 
                               grad_outputs=torch.ones(bn3_2_x.size()).to(device), 
                               retain_graph=True)[0].unsqueeze(1).detach()

    # 调整激活掩码维度
    M1 = M1.detach().unsqueeze(1)  # B x 1 x c_out x N
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)
    
    M1_2 = M1_2.detach().unsqueeze(1)
    M2_2 = M2_2.detach().unsqueeze(1)
    M3_2 = M3_2.detach().unsqueeze(1)

    # 组合每个块内的梯度
    # 块1的梯度计算
    # 线性层1 -> BN -> ReLU -> 线性层2 -> BN -> ReLU
    grad_block1_layer1 = W1_1 * dBN1 * M1  # 线性层1 + BN + ReLU
    grad_block1_layer2 = W1_2 * dBN1_2 * M1_2  # 线性层2 + BN + ReLU
    grad_block1 = torch.einsum('ijkl,ikml->ijml', grad_block1_layer1, grad_block1_layer2)
    
    # 块2的梯度计算
    grad_block2_layer1 = W2_1 * dBN2 * M2
    grad_block2_layer2 = W2_2 * dBN2_2 * M2_2
    grad_block2 = torch.einsum('ijkl,ikml->ijml', grad_block2_layer1, grad_block2_layer2)
    
    # 块3的梯度计算
    grad_block3_layer1 = W3_1 * dBN3 * M3
    grad_block3_layer2 = W3_2 * dBN3_2 * M3_2
    grad_block3 = torch.einsum('ijkl,ikml->ijml', grad_block3_layer1, grad_block3_layer2)
    
    # 块间链式法则组合
    # 块1 -> 块2
    grad_block1_to_block2 = torch.einsum('ijkl,ikml->ijml', grad_block1, grad_block2)
    
    # 块1 -> 块2 -> 块3
    grad_final = torch.einsum('ijkl,ikml->ijml', grad_block1_to_block2, grad_block3)
    
    return grad_final  # B x 3 x K x N
