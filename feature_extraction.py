import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops import rearrange

import utils



#### base class for feature extractor ####
class FeatureExtractor(nn.Module):
    """
    特征提取器的基类，定义所有特征提取器必须实现的接口
    """
    def __init__(self, dim_k=1024):
        """
        初始化特征提取器
        
        参数:
            dim_k: 输出特征维度
        """
        super(FeatureExtractor, self).__init__()
        self.dim_k = dim_k
        self.extractor_type = "base"  # 基类标识
        
    def forward(self, points, iter):
        """
        前向传播函数
        
        参数:
            points: 输入点云 [B,N,3]
            iter: 迭代标志，用于控制输出内容
                当iter=-1时返回用于计算雅可比矩阵的中间结果
            
        返回:
            当iter=-1时: [x, [M1,M2,M3], [A1,A2,A3], [A1_x,A2_x,A3_x], [bn1_x,bn2_x,bn3_x], max_idx]
            其他情况: 特征向量 [B,dim_k]
        """
        raise NotImplementedError("子类必须实现forward方法")
    
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
        raise NotImplementedError("子类必须实现get_jacobian方法")


#### original model using PointNet ####
def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


class Pointnet_Features(FeatureExtractor):
    """
    PointNet特征提取网络，与原始模型完全一致
    """
    def __init__(self, dim_k=1024):
        super(Pointnet_Features, self).__init__(dim_k)
        self.extractor_type = "pointnet"
        
        # PointNet特征提取层
        self.mlp1 = MLPNet(3, [64], b_shared=True).layers
        self.mlp2 = MLPNet(64, [128], b_shared=True).layers
        self.mlp3 = MLPNet(128, [dim_k], b_shared=True).layers
        
    def forward(self, points, iter):
        """
        前向传播，提取点云特征
        
        参数:
            points: 输入点云 [B,N,3]
            iter: 迭代标志
            
        返回:
            与原始Pointnet_Features一致的输出
        """
        x = points.transpose(1, 2) # [B, 3, N]
        if iter == -1:
            x = self.mlp1[0](x)
            A1_x = x
            x = self.mlp1[1](x)
            bn1_x = x
            x = self.mlp1[2](x)
            M1 = (x > 0).type(torch.float)
            
            x = self.mlp2[0](x)
            A2_x = x
            x = self.mlp2[1](x)
            bn2_x = x
            x = self.mlp2[2](x)
            M2 = (x > 0).type(torch.float)
            
            x = self.mlp3[0](x)
            A3_x = x
            x = self.mlp3[1](x)
            bn3_x = x
            x = self.mlp3[2](x)
            M3 = (x > 0).type(torch.float)
            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # 提取权重
            A1 = self.mlp1[0].weight
            A2 = self.mlp2[0].weight
            A3 = self.mlp3[0].weight

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx
        else:
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

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
        feature_j = feature_jac(mask_fn, a_fn, ax_fn, bn_fn, device).to(device)
        #feature_j = feature_jac_auto_diff(p0, self, device).to(device)
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
            
            # 计算条件变形雅可比矩阵
            warp_condition = utils.cal_conditioned_warp_jacobian(voxel_coords_diff)   # V x 6 x 6
            warp_condition = warp_condition.permute(0,2,1).reshape(-1, 6)   # (V6) x 6

            J = torch.einsum('ij,jk->ik', J_, warp_condition).unsqueeze(0)   # 1 X K X 6
            
        return J


# 从utils.py移植的函数：特征雅可比矩阵计算
def feature_jac(M, A, Ax, BN, device):
    """
    计算特征雅可比矩阵
    
    参数:
        M, A, Ax, BN: 中间层输出列表
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


### auto diff
def feature_jac_auto_diff(points, model, device=None):
    """
    使用PyTorch自动微分计算特征雅可比矩阵
    
    参数:
        points: 输入点云 [B,N,3]
        model: 特征提取模型
        device: 计算设备
        
    返回:
        特征雅可比矩阵 [B,3,K,N]
    """
    batch_size, num_points, _ = points.shape
    
    # 确保输入可导
    points_clone = points.clone().detach().requires_grad_(True)
    
    # 为前向计算定义一个包装器函数
    def feature_extractor(p):
        # 使用iter=0表示普通前向计算模式
        return model(p, iter=0)
    
    # 计算特征对点云坐标的雅可比矩阵
    jacobian = torch.zeros(batch_size, model.dim_k, num_points, 3).to(device)
    
    for b in range(batch_size):
        # 计算单个批次的雅可比矩阵
        jac = torch.func.jacrev(feature_extractor)(points_clone[b:b+1])  # [K,1,N,3]
        
        # 直接通过view方法重塑张量，跳过squeeze操作
        # 已知张量形状为[1024, 1, 100, 3]
        jacobian[b] = jac.view(model.dim_k, num_points, 3)
    
    # 调整维度顺序以匹配原函数输出格式
    return jacobian.permute(0, 3, 1, 2)  # [B,3,K,N]


##### new model for mamba3d ####

class Mamba3DBlock(nn.Module):
    """
    Mamba3D块，用于点云处理的并行化实现
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * dim)
        
        # 投影层
        self.in_proj = nn.Conv1d(dim, self.d_inner * 2, 1)
        self.out_proj = nn.Conv1d(self.d_inner, dim, 1)
        
        # 深度卷积
        self.conv = nn.Conv1d(
            self.d_inner, 
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM参数
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        self.B = nn.Conv1d(self.d_inner, self.d_state, 1, bias=False)
        self.C = nn.Conv1d(self.d_inner, self.d_state, 1, bias=False)
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        输入: [B, C, N]
        输出: [B, C, N]
        """
        B, C, N = x.shape
        residual = x.transpose(1, 2)  # [B, N, C]
        
        # 投影
        x_proj = self.in_proj(x)  # [B, 2*d_inner, N]
        x_conv, z = x_proj.chunk(2, dim=1)  # [B, d_inner, N] each
        
        # 获取实际的d_inner值
        actual_d_inner = x_conv.shape[1]
        
        # 卷积激活
        x_conv = self.conv(x_conv)[..., :N]  # 因果填充
        x_conv = F.silu(x_conv)
        
        # 状态空间模型参数 - 确保维度匹配
        A = -torch.exp(self.A_log.float()[:actual_d_inner])  # 截取匹配的维度
        D = self.D[:actual_d_inner].unsqueeze(-1)
        
        delta = torch.sigmoid(z)  # [B, d_inner, N]
        
        # 使用转置卷积代替einsum
        y = x_conv * delta
        
        # 输出投影
        out = self.out_proj(y)
        out = self.norm(out.transpose(1, 2) + residual)
        
        return out.transpose(1, 2)  # [B, C, N]


class Mamba3D_Features(FeatureExtractor):
    """
    基于Mamba3D的特征提取网络
    """
    def __init__(self, dim_k=1024):
        super(Mamba3D_Features, self).__init__(dim_k)
        self.extractor_type = "3dmamba_v1"
        self.use_mamba = True
        
        # MLP层与可选的Mamba3D块
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            Mamba3DBlock(64)
            
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            Mamba3DBlock(128)
        )
        
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, dim_k, 1),
            nn.BatchNorm1d(dim_k),
            nn.ReLU(),
            Mamba3DBlock(dim_k)
        )
        
        # 存储层引用，用于雅可比矩阵计算
        self.mlp1_layers = [self.mlp1[0], self.mlp1[1], self.mlp1[2], self.mlp1[3]]
        self.mlp2_layers = [self.mlp2[0], self.mlp2[1], self.mlp2[2], self.mlp2[3]]
        self.mlp3_layers = [self.mlp3[0], self.mlp3[1], self.mlp3[2], self.mlp3[3]]
        
    def forward(self, points, iter):
        """
        前向传播，提取点云特征
        
        参数:
            points: 输入点云 [B,N,3]
            iter: 迭代标志
            
        返回:
            当iter=-1时: 返回特征和中间结果用于雅可比矩阵计算
            其他情况: 返回特征向量 [B,dim_k]
        """
        x = points.transpose(1, 2)  # [B, 3, N]
        
        if iter == -1:  # 特征提取模式（用于雅可比矩阵）
            # MLP1
            x = self.mlp1_layers[0](x)
            A1_x = x
            x = self.mlp1_layers[1](x)
            bn1_x = x
            x = self.mlp1_layers[2](x)
            mamba1_in = x  # 保存Mamba层输入
            x = self.mlp1_layers[3](x)
            mamba1_out = x  # 保存Mamba层输出
            M1 = (x > 0).float()
            
            # MLP2
            x = self.mlp2_layers[0](x)
            A2_x = x
            x = self.mlp2_layers[1](x)
            bn2_x = x
            x = self.mlp2_layers[2](x)
            mamba2_in = x  # 保存Mamba层输入
            x = self.mlp2_layers[3](x)
            mamba2_out = x # 保存Mamba层输出
            M2 = (x > 0).float()
            
            # MLP3
            x = self.mlp3_layers[0](x)
            A3_x = x
            x = self.mlp3_layers[1](x)
            bn3_x = x
            x = self.mlp3_layers[2](x)
            mamba3_in = x # 保存Mamba层输入
            x = self.mlp3_layers[3](x)
            mamba3_out = x  # 保存Mamba层输出
            M3 = (x > 0).float()
            
            # 全局最大池化
            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            # 提取权重
            A1 = self.mlp1_layers[0].weight
            A2 = self.mlp2_layers[0].weight
            A3 = self.mlp3_layers[0].weight
            
            # 返回增加了Mamba层的输入输出
            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx, [mamba1_in, mamba2_in, mamba3_in], [mamba1_out, mamba2_out, mamba3_out]
        else:  # 标准前向传播
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, mamba_in=None, mamba_out=None, mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn, a_fn, ax_fn, bn_fn: 中间层输出
            max_idx: 最大池化索引
            mamba_in: Mamba层的输入列表 [mamba1_in, mamba2_in, mamba3_in]
            mamba_out: Mamba层的输出列表 [mamba1_out, mamba2_out, mamba3_out]
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
        feature_j = feature_jac_mamba3dv1(mask_fn, a_fn, ax_fn, bn_fn, mamba_in, mamba_out, device).to(device)
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


# 特征雅可比矩阵计算 for mamba3d
def feature_jac_mamba3dv1(M, A, Ax, BN, mamba_in, mamba_out, device=None):
    """
    计算特征雅可比矩阵，包含Mamba层的处理
    
    参数:
        M, A, Ax, BN: 中间层输出列表
        mamba_in: Mamba层的输入列表 [mamba1_in, mamba2_in, mamba3_in]
        mamba_out: Mamba层的输出列表 [mamba1_out, mamba2_out, mamba3_out]
        device: 计算设备
        
    返回:
        特征雅可比矩阵
    """
    import time
    total_start = time.time()
    
    # 解包输入列表
    A1, A2, A3 = A
    M1, M2, M3 = M
    Ax1, Ax2, Ax3 = Ax
    BN1, BN2, BN3 = BN
    
    # 解包Mamba层输入输出
    mamba1_in, mamba2_in, mamba3_in = mamba_in
    mamba1_out, mamba2_out, mamba3_out = mamba_out

    # 1 x c_in x c_out x 1
    prep_start = time.time()
    A1 = (A1.T).detach().unsqueeze(-1)
    A2 = (A2.T).detach().unsqueeze(-1)
    A3 = (A3.T).detach().unsqueeze(-1)
    prep_end = time.time()
    #print(f"准备张量耗时: {prep_end - prep_start:.4f} 秒")

    # 使用自动微分计算批量归一化的梯度
    # B x 1 x c_out x N
    bn_grad_start = time.time()
    dBN1 = torch.autograd.grad(outputs=BN1, inputs=Ax1, grad_outputs=torch.ones(BN1.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN2 = torch.autograd.grad(outputs=BN2, inputs=Ax2, grad_outputs=torch.ones(BN2.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN3 = torch.autograd.grad(outputs=BN3, inputs=Ax3, grad_outputs=torch.ones(BN3.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    bn_grad_end = time.time()
    #print(f"BN梯度计算耗时: {bn_grad_end - bn_grad_start:.4f} 秒")

    # B x 1 x c_out x N
    M1 = M1.detach().unsqueeze(1)
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)

    # 使用自动微分计算Mamba层的梯度
    # B x 1 x c_out x N
    mamba_grad_start = time.time()
    dMamba1 = torch.autograd.grad(outputs=mamba1_out, inputs=mamba1_in, grad_outputs=torch.ones(mamba1_out.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dMamba2 = torch.autograd.grad(outputs=mamba2_out, inputs=mamba2_in, grad_outputs=torch.ones(mamba2_out.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dMamba3 = torch.autograd.grad(outputs=mamba3_out, inputs=mamba3_in, grad_outputs=torch.ones(mamba3_out.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    mamba_grad_end = time.time()
    #print(f"Mamba梯度计算耗时: {mamba_grad_end - mamba_grad_start:.4f} 秒")

    # 使用广播计算，包含Mamba层 --> B x c_in x c_out x N
    comp_start = time.time()
    A1BN1M1 = A1 * (dBN1 * M1 * dMamba1)
    A2BN2M2 = A2 * (dBN2 * M2 * dMamba2)
    A3BN3M3 = (dMamba3 * M3 * dBN3) * A3
    comp_mid = time.time()
    #print(f"中间计算耗时: {comp_mid - comp_start:.4f} 秒")

    # 使用einsum组合
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)   # B x 3 x 64 x N
    A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)   # B x 3 x K x N
    
    feat_jac = A2BN2M2_A3BN3M3
    comp_end = time.time()
    #print(f"einsum计算耗时: {comp_end - comp_mid:.4f} 秒")
    
    total_end = time.time()
    #print(f"雅可比矩阵总计算耗时: {total_end - total_start:.4f} 秒")
    
    return feat_jac # B x 3 x K x N


##### 3DMamba V2 特征提取器 ####

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(3, d_model)
        self.weight = self.linear.weight  # 存储权重引用用于雅可比矩阵计算

    def forward(self, coords):
        # coords: [B, N, 3]
        return self.linear(coords)

class BiSSM(nn.Module):
    def __init__(self, d_model):
        super(BiSSM, self).__init__()
        self.forward_ssm = nn.GRU(d_model, d_model, batch_first=True)
        self.backward_ssm = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        # x: [B, N, C]
        x_forward, _ = self.forward_ssm(x)
        x_backward, _ = self.backward_ssm(torch.flip(x, dims=[1]))
        x_backward = torch.flip(x_backward, dims=[1])
        return x_forward + x_backward

class PointCloudMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointCloudMambaBlock, self).__init__()
        # 使用Sequential组织层
        self.seq = nn.Sequential(
            PositionalEncoding(in_channels),
            BiSSM(in_channels),
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # 保存各层引用以便访问
        self.pos_enc = self.seq[0]
        self.bi_ssm = self.seq[1]
        self.linear = self.seq[2]
        self.bn = self.seq[3]
        self.relu = self.seq[4]

    def forward(self, x):
        return self.seq(x)

class Mamba3D_V2_Features(FeatureExtractor):
    """
    基于Mamba3D V2的特征提取网络
    """
    def __init__(self, dim_k=1024):
        super(Mamba3D_V2_Features, self).__init__(dim_k)
        self.extractor_type = "3dmamba_v2"
        
        self.block1 = PointCloudMambaBlock(3, 64)
        self.block2 = PointCloudMambaBlock(64, 128)
        self.block3 = PointCloudMambaBlock(128, dim_k)

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
            # 第一个块 - 收集中间状态
            x = self.block1.pos_enc(x)
            bissm_in1 = x
            x = self.block1.bi_ssm(x)
            bissm_out1 = x
            x = self.block1.linear(x)
            bn_in1 = x
            x = self.block1.bn(x.transpose(1, 2)).transpose(1, 2)
            bn_out1 = x
            x = self.block1.relu(x)
            M1 = (x > 0).float()
            
            # 第二个块
            x = self.block2.pos_enc(x)
            bissm_in2 = x
            x = self.block2.bi_ssm(x)
            bissm_out2 = x
            x = self.block2.linear(x)
            bn_in2 = x
            x = self.block2.bn(x.transpose(1, 2)).transpose(1, 2)
            bn_out2 = x
            x = self.block2.relu(x)
            M2 = (x > 0).float()
            
            # 第三个块
            x = self.block3.pos_enc(x)
            bissm_in3 = x
            x = self.block3.bi_ssm(x)
            bissm_out3 = x
            x = self.block3.linear(x)
            bn_in3 = x
            x = self.block3.bn(x.transpose(1, 2)).transpose(1, 2)
            bn_out3 = x
            x = self.block3.relu(x)
            M3 = (x > 0).float()

            # 全局最大池化
            max_idx = torch.max(x, 1)[1]
            x = torch.max(x, dim=1)[0]
            
            # 提取权重
            A11 = self.block1.pos_enc.weight
            A12 = self.block2.pos_enc.weight
            A13 = self.block3.pos_enc.weight

            A21 = self.block1.linear.weight
            A22 = self.block2.linear.weight
            A23 = self.block3.linear.weight
            
            return x, [M1, M2, M3], [A21, A22, A23], [bn_in1, bn_in2, bn_in3], [bn_out1, bn_out2, bn_out3], max_idx, [A11, A12, A13], [bissm_in1, bissm_in2, bissm_in3], [bissm_out1, bissm_out2, bissm_out3]
        else:
            # 标准前向传播
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = torch.max(x, dim=1)[0]
            return x
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                    pos_enc_weights=None, bissm_in=None, bissm_out=None, mode="train", 
                    voxel_coords_diff=None, data_type='synthetic', num_points=None):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn, a_fn, ax_fn, bn_fn: 中间层输出
            max_idx: 最大池化索引
            pos_enc_weights: 位置编码权重
            bissm_in: BiSSM层的输入
            bissm_out: BiSSM层的输出
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
        feature_j = feature_jac_3dmamba_v2(mask_fn, a_fn, ax_fn, bn_fn, pos_enc_weights, bissm_in, bissm_out, device).to(device)
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

# 特征雅可比矩阵计算 for 3DMamba V2
def feature_jac_3dmamba_v2(M, linear_weights, bn_in, bn_out, pos_enc_weights, bissm_in, bissm_out, device=None):
    """
    计算3DMamba V2特征雅可比矩阵
    
    参数:
        M: 激活掩码列表 [M1, M2, M3]
        A: 线性层权重列表 [A21, A22, A23]
        Ax: 线性层输入列表 [bn_in1, bn_in2, bn_in3]
        BN: 批归一化输出列表 [bn_out1, bn_out2, bn_out3]
        pos_enc_weights: 位置编码权重列表 [A11, A12, A13]
        bissm_in: BiSSM层输入列表 [bissm_in1, bissm_in2, bissm_in3]
        bissm_out: BiSSM层输出列表 [bissm_out1, bissm_out2, bissm_out3]
        device: 计算设备
        
    返回:
        特征雅可比矩阵 [B, 3, K, N]
    """
    # 解包输入列表
    A21, A22, A23 = linear_weights
    M1, M2, M3 = M
    bn_in1, bn_in2, bn_in3 = bn_in
    bn_out1, bn_out2, bn_out3 = bn_out
    A11, A12, A13 = pos_enc_weights
    
    # 解包BiSSM层输入输出
    bissm_in1, bissm_in2, bissm_in3 = bissm_in
    bissm_out1, bissm_out2, bissm_out3 = bissm_out

    # 准备权重张量


    # 解包线性层权重
    A21 = (A21.T).detach().unsqueeze(-1)
    A22 = (A22.T).detach().unsqueeze(-1)
    A23 = (A23.T).detach().unsqueeze(-1)

    # 使用自动微分计算批量归一化的梯度
    dBN1 = torch.autograd.grad(outputs=bn_out1, inputs=bn_in1, grad_outputs=torch.ones(bn_out1.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN2 = torch.autograd.grad(outputs=bn_out2, inputs=bn_in2, grad_outputs=torch.ones(bn_out2.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN3 = torch.autograd.grad(outputs=bn_out3, inputs=bn_in3, grad_outputs=torch.ones(bn_out3.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    

    # 准备掩码张量
    M1 = M1.detach().unsqueeze(1)
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)

    # 计算BiSSM层的梯度
    dBiSSM1 = torch.autograd.grad(outputs=bissm_out1, inputs=bissm_in1, grad_outputs=torch.ones(bissm_out1.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBiSSM2 = torch.autograd.grad(outputs=bissm_out2, inputs=bissm_in2, grad_outputs=torch.ones(bissm_out2.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBiSSM3 = torch.autograd.grad(outputs=bissm_out3, inputs=bissm_in3, grad_outputs=torch.ones(bissm_out3.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    
    # 过滤极小梯度值
    dBiSSM1[torch.abs(dBiSSM1) < 1e-10] = 0.0
    dBiSSM2[torch.abs(dBiSSM2) < 1e-10] = 0.0
    dBiSSM3[torch.abs(dBiSSM3) < 1e-10] = 0.0

    # 位置编码梯度
    PE1 = A11.detach().unsqueeze(-1).to(device)  # [C_out, 3, 1]
    PE2 = A12.detach().unsqueeze(-1).to(device)
    PE3 = A13.detach().unsqueeze(-1).to(device)

    # 组合所有梯度计算特征雅可比矩阵
    L1 = dBiSSM1 * PE1  # 位置编码和BiSSM层的组合梯度
    L2 = dBiSSM2 * PE2
    L3 = dBiSSM3 * PE3
    
    # 完整的梯度链计算
    A1BN1M1 = A11 * dBN1 * M1 * L1
    A2BN2M2 = A22 * dBN2 * M2 * L2
    A3BN3M3 = A23 * dBN3 * M3 * L3
    
    # 使用einsum组合
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)   # B x 3 x 64 x N
    A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)   # B x 3 x K x N
    
    feat_jac = A2BN2M2_A3BN3M3

    return feat_jac   # B x 3 x K x N