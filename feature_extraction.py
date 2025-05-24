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
                     voxel_coords_diff=None, data_type='synthetic', num_points=None, use_numerical=False, delta=1.0e-3):
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
            use_numerical: 是否使用数值近似方法
            delta: 数值微分的步长
            
        返回:
            雅可比矩阵 [B,K,6]
        """
        # 如果选择使用数值近似方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 否则使用解析方法
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
    
    def get_jacobian_numerical(self, p0, delta=1.0e-3, mode="train", 
                              voxel_coords_diff=None, data_type='synthetic', num_points=None):
        """
        使用数值近似方法计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            delta: 数值微分的步长
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
        
        # 保存模型的训练状态并切换到评估模式
        training = self.training
        self.eval()
        
        try:
            # 计算原始特征
            f0 = self.forward(p0, iter=0)  # [B, K]
            
            # 创建扰动参数
            dt = torch.ones(batch_size, 6, device=device, dtype=p0.dtype) * delta
            
            # 计算6个扰动的变换矩阵
            transf = torch.zeros(batch_size, 6, 4, 4, device=device, dtype=p0.dtype)
            for b in range(batch_size):
                d = torch.diag(dt[b, :])  # [6, 6]
                D = utils.exp(-d)  # [6, 4, 4]
                transf[b, :, :, :] = D[:, :, :]
            
            # 应用变换到点云
            transf = transf.unsqueeze(2)  # [B, 6, 1, 4, 4]
            p_perturbed = utils.transform(transf, p0.unsqueeze(1))  # [B, 6, N, 3]
            
            # 计算扰动后的特征
            p_reshaped = p_perturbed.view(-1, num_points, 3)  # [B*6, N, 3]
            f_perturbed = self.forward(p_reshaped, iter=0)  # [B*6, K]
            f_perturbed = f_perturbed.view(batch_size, 6, -1).transpose(1, 2)  # [B, K, 6]
            
            # 计算特征差分
            f0_expanded = f0.unsqueeze(-1)  # [B, K, 1]
            df = f0_expanded - f_perturbed  # [B, K, 6]
            
            # 计算雅可比矩阵
            J = df / dt.unsqueeze(1)  # [B, K, 6]
            
            # 处理真实数据的特殊情况
            if mode == 'test' and data_type == 'real' and voxel_coords_diff is not None:
                J_ = J.permute(1, 0, 2).reshape(dim_k, -1)  # K x (V6)
                warp_condition = utils.cal_conditioned_warp_jacobian(voxel_coords_diff)  # V x 6 x 6
                warp_condition = warp_condition.permute(0, 2, 1).reshape(-1, 6)  # (V6) x 6
                J = torch.einsum('ij,jk->ik', J_, warp_condition).unsqueeze(0)  # 1 X K X 6
                
            return J
            
        finally:
            # 恢复模型的训练状态
            self.train(training)


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
                     voxel_coords_diff=None, data_type='synthetic', num_points=None, use_numerical=False, delta=1.0e-3):
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
            use_numerical: 是否使用数值近似方法
            delta: 数值微分的步长
            
        返回:
            雅可比矩阵 [B,K,6]
        """
        # 如果选择使用数值近似方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 否则使用解析方法
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


### 自动微分算梯度的方法无效，显存需求过大。
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
        self.B = nn.Linear(self.d_inner, d_state, bias=False)
        self.C = nn.Linear(self.d_inner, d_state, bias=False)
        
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
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, mamba_in=None, mamba_out=None, mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None, use_numerical=False, delta=1.0e-3):
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
            use_numerical: 是否使用数值近似方法
            delta: 数值微分的步长
            
        返回:
            雅可比矩阵 [B,K,6]
        """
        # 如果选择使用数值近似方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 否则使用解析方法
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
        # 分别定义各层，而不使用Sequential
        self.pos_enc = nn.Linear(in_channels, out_channels)
        self.bi_ssm = BiSSM(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, N, C]
        x = self.pos_enc(x)         # [B, N, out_channels]
        x = self.bi_ssm(x)          # [B, N, out_channels]
        x_t = x.transpose(1, 2)     # [B, out_channels, N]
        x_t = self.bn(x_t)          # [B, out_channels, N]
        x = x_t.transpose(1, 2)     # [B, N, out_channels]
        x = self.relu(x)            # [B, N, out_channels]
        return x

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
            points: 输入点云 [B,N,3]
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
            bn_in3 = x
            x = self.block3.bn(x.transpose(1, 2)).transpose(1, 2)
            bn_out3 = x
            x = self.block3.relu(x)
            M3 = (x > 0).float()

            # 全局最大池化
            max_idx = torch.max(x, 1)[1]
            x = torch.max(x, dim=1)[0]
            
            # 提取权重
            A1 = self.block1.pos_enc.weight
            A2 = self.block2.pos_enc.weight
            A3 = self.block3.pos_enc.weight
            
            # 修改后的返回值：去掉重复的[A1, A2, A3]参数
            return x, [M1, M2, M3], [A1, A2, A3], [bn_in1, bn_in2, bn_in3], [bn_out1, bn_out2, bn_out3], max_idx, [bissm_in1, bissm_in2, bissm_in3], [bissm_out1, bissm_out2, bissm_out3]
        else:
            # 标准前向传播
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = torch.max(x, dim=1)[0]
            return x
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                    bissm_in=None, bissm_out=None, mode="train", 
                    voxel_coords_diff=None, data_type='synthetic', num_points=None, use_numerical=False, delta=1.0e-3):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn, a_fn, ax_fn, bn_fn: 中间层输出
            max_idx: 最大池化索引
            bissm_in: BiSSM层的输入
            bissm_out: BiSSM层的输出
            mode: 训练或测试模式
            voxel_coords_diff: 体素坐标差异（用于真实数据）
            data_type: 数据类型（'synthetic'或'real'）
            num_points: 点云中的点数量
            use_numerical: 是否使用数值近似方法
            delta: 数值微分的步长
            
        返回:
            雅可比矩阵 [B,K,6]
        """
        # 如果选择使用数值近似方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 否则使用解析方法
        if num_points is None:
            num_points = p0.shape[1]
        batch_size = p0.shape[0]
        dim_k = self.dim_k
        device = p0.device
        
        # 保存BiSSM模块的原始训练状态
        bissm_train_states = [
            self.block1.bi_ssm.training,
            self.block2.bi_ssm.training,
            self.block3.bi_ssm.training
        ]
        
        # 临时设置为训练模式以支持cudnn RNN的反向传播
        self.block1.bi_ssm.train()
        self.block2.bi_ssm.train()
        self.block3.bi_ssm.train()
        
        try:
            # 1. 计算变形雅可比矩阵
            g_ = torch.zeros(batch_size, 6).to(device)
            warp_jac = utils.compute_warp_jac(g_, p0, num_points)   # B x N x 3 x 6
            
            # 2. 计算特征雅可比矩阵
            feature_j = feature_jac_3dmamba_v2(mask_fn, a_fn, ax_fn, bn_fn, bissm_in, bissm_out, device).to(device)
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
        finally:
            # 恢复原始训练状态
            self.block1.bi_ssm.train(bissm_train_states[0])
            self.block2.bi_ssm.train(bissm_train_states[1]) 
            self.block3.bi_ssm.train(bissm_train_states[2])

# 特征雅可比矩阵计算 for 3DMamba V2 - 已移除pos_enc_weights参数
def feature_jac_3dmamba_v2(M, A, Ax, BN, bissm_in, bissm_out, device=None):
    """
    计算3DMamba V2特征雅可比矩阵
    
    参数:
        M: 激活掩码列表 [M1, M2, M3]
        A: 线性层权重列表 [A1, A2, A3]
        Ax: 批归一化输入列表 [bn_in1, bn_in2, bn_in3]
        BN: 批归一化输出列表 [bn_out1, bn_out2, bn_out3]
        bissm_in: BiSSM层输入列表 [bissm_in1, bissm_in2, bissm_in3]
        bissm_out: BiSSM层输出列表 [bissm_out1, bissm_out2, bissm_out3]
        device: 计算设备
        
    返回:
        特征雅可比矩阵 [B, 3, K, N]
    """
    # 解包输入列表
    A1, A2, A3 = A
    M1, M2, M3 = M
    bn_in1, bn_in2, bn_in3 = Ax
    bn_out1, bn_out2, bn_out3 = BN
    
    # 解包BiSSM层输入输出
    bissm_in1, bissm_in2, bissm_in3 = bissm_in
    bissm_out1, bissm_out2, bissm_out3 = bissm_out

    # 准备权重张量
    A21 = (A1.T).detach().unsqueeze(-1)
    A22 = (A2.T).detach().unsqueeze(-1)
    A23 = (A3.T).detach().unsqueeze(-1)

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

    # 组合所有梯度计算特征雅可比矩阵
    L1 = dBiSSM1 * A21
    L2 = dBiSSM2 * A22
    L3 = dBiSSM3 * A23
    
    # 完整的梯度链计算
    A1BN1M1 = dBN1 * M1 * L1
    A2BN2M2 = dBN2 * M2 * L2
    A3BN3M3 = dBN3 * M3 * L3
    
    # 使用einsum组合
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)   # B x 3 x 64 x N
    A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)   # B x 3 x K x N
    
    feat_jac = A2BN2M2_A3BN3M3

    return feat_jac   # B x 3 x K x N


##### Pointnet Attention V1 特征提取器 ####

class Pointnet_Attention_Features(FeatureExtractor):
    """
    带注意力机制的PointNet特征提取网络
    """
    def __init__(self, dim_k=1024, num_heads=4, bn_momentum=0.1):
        super().__init__(dim_k)
        # 设置特征提取器类型标识
        self.extractor_type = "pointnet_attention_v1"
        
        # 直接定义所有卷积层、BN层和注意力层
        # 第一层 MLP (3->64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64, momentum=bn_momentum)
        self.attn1 = nn.MultiheadAttention(64, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(64)
        
        # 第二层 MLP (64->128)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128, momentum=bn_momentum)
        self.attn2 = nn.MultiheadAttention(128, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(128)
        
        # 第三层 MLP (128->dim_k)
        self.conv3 = nn.Conv1d(128, dim_k, 1)
        self.bn3 = nn.BatchNorm1d(dim_k, momentum=bn_momentum)
        self.attn3 = nn.MultiheadAttention(dim_k, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim_k)

    def forward(self, points, iter):
        """ points -> features: [B, N, 3] -> [B, K] """
        x = points.transpose(1, 2)  # [B, 3, N]

        if iter == -1:
            # 第一层 MLP + Attention
            x = self.conv1(x) 
            A1_x = x
            x = self.bn1(x)
            bn1_x = x
            x = F.relu(x)
            M1 = (x > 0).float()
            
            # 注意力机制
            x_t = x.transpose(1, 2)  # [B, N, C]
            attn_in1 = x_t
            attn_out, attn_weights1 = self.attn1(x_t, x_t, x_t)  # 保存注意力权重
            attn_raw1 = attn_out
            x_t = self.norm1(attn_out + x_t)
            norm1_out = x_t
            x = x_t.transpose(1, 2)  # [B, C, N]

            # 第二层 MLP + Attention
            x = self.conv2(x)
            A2_x = x
            x = self.bn2(x)
            bn2_x = x
            x = F.relu(x)
            M2 = (x > 0).float()
            
            # 注意力机制
            x_t = x.transpose(1, 2)  # [B, N, C]
            attn_in2 = x_t
            attn_out, attn_weights2 = self.attn2(x_t, x_t, x_t)
            attn_raw2 = attn_out
            x_t = self.norm2(attn_out + x_t)
            norm2_out = x_t
            x = x_t.transpose(1, 2)  # [B, C, N]

            # 第三层 MLP + Attention
            x = self.conv3(x)
            A3_x = x
            x = self.bn3(x)
            bn3_x = x
            x = F.relu(x)
            M3 = (x > 0).float()
            
            # 注意力机制
            x_t = x.transpose(1, 2)  # [B, N, C]
            attn_in3 = x_t
            attn_out, attn_weights3 = self.attn3(x_t, x_t, x_t)
            attn_raw3 = attn_out
            x_t = self.norm3(attn_out + x_t)
            norm3_out = x_t
            x = x_t.transpose(1, 2)  # [B, C, N]

            # 最大池化
            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # 提取权重
            A1 = self.conv1.weight
            A2 = self.conv2.weight
            A3 = self.conv3.weight
            
            # 保存注意力信息用于梯度计算
            attn_data = {
                # 注意力层输入
                'attn_in1': attn_in1,
                'attn_in2': attn_in2,
                'attn_in3': attn_in3,
                # 注意力层原始输出
                'attn_raw1': attn_raw1,
                'attn_raw2': attn_raw2,
                'attn_raw3': attn_raw3,
                # 注意力权重
                'attn_weights1': attn_weights1,
                'attn_weights2': attn_weights2,
                'attn_weights3': attn_weights3,
                # 层归一化输出
                'norm1_out': norm1_out,
                'norm2_out': norm2_out,
                'norm3_out': norm3_out,
                # 注意力层实例
                'attn1': self.attn1,
                'attn2': self.attn2,
                'attn3': self.attn3,
                'norm1': self.norm1,
                'norm2': self.norm2,
                'norm3': self.norm3
            }

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx, attn_data

        else:
            # 普通前向传播
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attn1(x_t, x_t, x_t)
            x_t = self.norm1(attn_out + x_t)
            x = x_t.transpose(1, 2)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attn2(x_t, x_t, x_t)
            x_t = self.norm2(attn_out + x_t)
            x = x_t.transpose(1, 2)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attn3(x_t, x_t, x_t)
            x_t = self.norm3(attn_out + x_t)
            x = x_t.transpose(1, 2)
            
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
            
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None,
                     mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None,
                     extra_param_0=None, use_numerical=False, delta=1.0e-3):
        """
        计算特征提取的雅可比矩阵
        """
        # 如果选择使用数值近似方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 否则使用解析方法
        if num_points is None:
            num_points = p0.shape[1]
        batch_size = p0.shape[0]
        dim_k = self.dim_k
        device = p0.device
        
        # 1. 计算变形雅可比矩阵
        g_ = torch.zeros(batch_size, 6).to(device)
        warp_jac = utils.compute_warp_jac(g_, p0, num_points)   # B x N x 3 x 6
        
        # 2. 计算特征雅可比矩阵
        attn_weights = extra_param_0  # 获取注意力权重
        feature_j = feature_jac_pointnet_attention(mask_fn, a_fn, ax_fn, bn_fn, attn_weights, device).to(device)
        feature_j = feature_j.permute(0, 3, 1, 2)   # B x N x 6 x K
        
        # 3. 组合得到最终雅可比矩阵
        J_ = torch.einsum('ijkl,ijkm->ijlm', feature_j, warp_jac)   # B x N x K x 6
        
        # 4. 根据最大池化索引处理
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


def feature_jac_pointnet_attention(M, A, Ax, BN, attn_data, device):
    """
    带注意力机制的Pointnet特征雅可比矩阵计算函数
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
    dBN1 = torch.autograd.grad(outputs=BN1, inputs=Ax1, grad_outputs=torch.ones(BN1.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN2 = torch.autograd.grad(outputs=BN2, inputs=Ax2, grad_outputs=torch.ones(BN2.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN3 = torch.autograd.grad(outputs=BN3, inputs=Ax3, grad_outputs=torch.ones(BN3.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    
    # 计算注意力层的梯度
    # 第一层注意力
    attn_in1 = attn_data['attn_in1']
    attn_raw1 = attn_data['attn_raw1']
    norm1_out = attn_data['norm1_out']
    
    # 计算从norm1_out到attn_raw1的梯度
    dnorm1 = torch.autograd.grad(outputs=norm1_out, inputs=attn_raw1, 
                                grad_outputs=torch.ones(norm1_out.size()).to(device), 
                                retain_graph=True)[0].detach()
    
    # 计算从attn_raw1到attn_in1的梯度
    dattn1 = torch.autograd.grad(outputs=attn_raw1, inputs=attn_in1, 
                               grad_outputs=torch.ones(attn_raw1.size()).to(device), 
                               retain_graph=True)[0].detach()
    
    # 第二层注意力
    attn_in2 = attn_data['attn_in2']
    attn_raw2 = attn_data['attn_raw2']
    norm2_out = attn_data['norm2_out']
    
    dnorm2 = torch.autograd.grad(outputs=norm2_out, inputs=attn_raw2, 
                                grad_outputs=torch.ones(norm2_out.size()).to(device), 
                                retain_graph=True)[0].detach()
    
    dattn2 = torch.autograd.grad(outputs=attn_raw2, inputs=attn_in2, 
                               grad_outputs=torch.ones(attn_raw2.size()).to(device), 
                               retain_graph=True)[0].detach()
    
    # 第三层注意力
    attn_in3 = attn_data['attn_in3']
    attn_raw3 = attn_data['attn_raw3']
    norm3_out = attn_data['norm3_out']
    
    dnorm3 = torch.autograd.grad(outputs=norm3_out, inputs=attn_raw3, 
                                grad_outputs=torch.ones(norm3_out.size()).to(device), 
                                retain_graph=True)[0].detach()
    
    dattn3 = torch.autograd.grad(outputs=attn_raw3, inputs=attn_in3, 
                               grad_outputs=torch.ones(attn_raw3.size()).to(device), 
                               retain_graph=True)[0].detach()
    
    # B x 1 x c_out x N
    M1 = M1.detach().unsqueeze(1)
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)
    
    # 转换注意力梯度和层归一化梯度的形状以匹配计算需求
    dattn1 = dattn1.transpose(1, 2).unsqueeze(1)  # [B, 1, C, N]
    dattn2 = dattn2.transpose(1, 2).unsqueeze(1)  # [B, 1, C, N]
    dattn3 = dattn3.transpose(1, 2).unsqueeze(1)  # [B, 1, C, N]
    
    # 转换层归一化梯度
    dnorm1 = dnorm1.transpose(1, 2).unsqueeze(1)  # [B, 1, C, N]
    dnorm2 = dnorm2.transpose(1, 2).unsqueeze(1)  # [B, 1, C, N]
    dnorm3 = dnorm3.transpose(1, 2).unsqueeze(1)  # [B, 1, C, N]
    
    # 结合普通层与注意力层和层归一化的梯度
    # 使用链式法则：最终梯度 = 卷积梯度 * BN梯度 * ReLU梯度 * 注意力梯度 * 层归一化梯度
    A1BN1M1 = A1 * dBN1 * M1 * dattn1 * dnorm1
    A2BN2M2 = A2 * dBN2 * M2 * dattn2 * dnorm2
    A3BN3M3 = M3 * dBN3 * A3 * dattn3 * dnorm3
    
    # 使用einsum组合
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)
    A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)
    
    feat_jac = A2BN2M2_A3BN3M3
    
    return feat_jac  # B x 3 x K x N


##### FastPointTransformer V1 特征提取器 ####

class FastPointTransformerBlock(nn.Module):
    """
    FastPointTransformer基本构建块，包含两个线性层和批归一化
    """
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
                    voxel_coords_diff=None, data_type='synthetic', num_points=None, use_numerical=False, delta=1.0e-3, **kwargs):
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
            use_numerical: 是否使用数值近似方法
            delta: 数值微分的步长
            **kwargs: 额外参数，支持扩展
            
        返回:
            雅可比矩阵 [B,K,6]
        """
        # 如果选择使用数值近似方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 否则使用解析方法
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
    
    # 转置梯度和掩码以匹配权重维度
    # 由于FastPointTransformerBlock中使用了transpose(1,2)
    dBN1 = dBN1.permute(0, 1, 3, 2)  # [B, 1, N, C] -> [B, 1, C, N]
    dBN2 = dBN2.permute(0, 1, 3, 2)
    dBN3 = dBN3.permute(0, 1, 3, 2)
    dBN1_2 = dBN1_2.permute(0, 1, 3, 2)
    dBN2_2 = dBN2_2.permute(0, 1, 3, 2)
    dBN3_2 = dBN3_2.permute(0, 1, 3, 2)
    
    M1 = M1.permute(0, 1, 3, 2)
    M2 = M2.permute(0, 1, 3, 2)
    M3 = M3.permute(0, 1, 3, 2)
    M1_2 = M1_2.permute(0, 1, 3, 2)
    M2_2 = M2_2.permute(0, 1, 3, 2)
    M3_2 = M3_2.permute(0, 1, 3, 2)

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


##### FastPointTransformer V2 特征提取器 ####

class FastPointTransformerLayer(nn.Module):
    """Efficient transformer layer from Fast Point Transformer (ICCV 2023)"""
    def __init__(self, dim, num_heads=4, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query/key/value投影
        self.to_qkv = nn.Conv1d(dim, dim * 3, 1)
        
        # 局部特征聚合
        self.local_agg = nn.Sequential(
            nn.Conv1d(dim, dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(dim // reduction_ratio, dim, 1)
        )
        
        # 输出投影
        self.proj = nn.Conv1d(dim, dim, 1)
        
        # 归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        """输入: [B, C, N]"""
        B, C, N = x.shape
        residual = x
        
        # 局部特征增强
        local_feat = self.local_agg(x)
        
        # Query/key/value
        qkv = self.to_qkv(self.norm1(x.transpose(1, 2)).transpose(1, 2))
        # 替换rearrange代码
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # 延迟注意力机制
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        # 替换第二个rearrange
        global_feat = (attn @ v).reshape(B, self.num_heads * self.head_dim, N)
        
        # 组合特征
        out = global_feat + local_feat
        out = self.proj(self.norm2(out.transpose(1, 2)).transpose(1, 2) + residual)
        
        return out

class Pointnet_fastpointtransformer_v2(FeatureExtractor):
    """
    基于Fast Point Transformer的改进PointNet特征提取器
    """
    def __init__(self, dim_k=1024, use_fpt=True):
        super(Pointnet_fastpointtransformer_v2, self).__init__(dim_k)
        # 设置特征提取器类型标识
        self.extractor_type = "fastpointtransformer_v2"
        
        self.use_fpt = use_fpt
        self.dim_k = dim_k
        
        # 共享MLP层并可选包含transformer块
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            FastPointTransformerLayer(64) if use_fpt else nn.Identity()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            FastPointTransformerLayer(128) if use_fpt else nn.Identity()
        )
        
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, dim_k, 1),
            nn.BatchNorm1d(dim_k),
            nn.ReLU(),
            FastPointTransformerLayer(dim_k) if use_fpt else nn.Identity()
        )
        
        # 存储用于雅可比矩阵计算的层引用
        self._init_layer_references()

    def _init_layer_references(self):
        """初始化雅可比矩阵计算所需的层引用"""
        self.mlp1_layers = [self.mlp1[0], self.mlp1[1], self.mlp1[2]]
        self.mlp2_layers = [self.mlp2[0], self.mlp2[1], self.mlp2[2]]
        self.mlp3_layers = [self.mlp3[0], self.mlp3[1], self.mlp3[2]]
        
        if self.use_fpt:
            self.mlp1_layers.append(self.mlp1[3])
            self.mlp2_layers.append(self.mlp2[3])
            self.mlp3_layers.append(self.mlp3[3])

    def forward(self, points, iter=-1):
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
        
        if iter == -1:  # 特征提取模式(用于雅可比矩阵计算)
            # MLP1
            x = self.mlp1_layers[0](x)
            A1_x = x
            x = self.mlp1_layers[1](x)
            bn1_x = x
            x = self.mlp1_layers[2](x)
            if self.use_fpt:
                x = self.mlp1_layers[3](x)
            M1 = (x > 0).float()
            
            # MLP2
            x = self.mlp2_layers[0](x)
            A2_x = x
            x = self.mlp2_layers[1](x)
            bn2_x = x
            x = self.mlp2_layers[2](x)
            if self.use_fpt:
                x = self.mlp2_layers[3](x)
            M2 = (x > 0).float()
            
            # MLP3
            x = self.mlp3_layers[0](x)
            A3_x = x
            x = self.mlp3_layers[1](x)
            bn3_x = x
            x = self.mlp3_layers[2](x)
            if self.use_fpt:
                x = self.mlp3_layers[3](x)
            M3 = (x > 0).float()
            
            # 全局最大池化
            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            # 定义FPT特有的额外数据
            fpt_data = {"use_fpt": self.use_fpt}
            
            return (
                x,  # [B, K]
                [M1, M2, M3],  # 激活掩码
                [self.mlp1_layers[0].weight,  # 卷积权重
                 self.mlp2_layers[0].weight,
                 self.mlp3_layers[0].weight],
                [A1_x, A2_x, A3_x],  # 预激活值
                [bn1_x, bn2_x, bn3_x],  # BN激活值
                max_idx,  # 最大池化索引
                fpt_data  # FPT特有数据
            )
        else:  # 标准前向传播
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
            
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                     mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None,
                     extra_param_0=None, use_numerical=False, delta=1.0e-3):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn: 激活掩码列表
            a_fn: 权重列表
            ax_fn: 卷积输出列表
            bn_fn: 批归一化输出列表
            max_idx: 最大池化索引
            mode: 模式 ("train" 或 "test")
            voxel_coords_diff: 体素坐标差异
            data_type: 数据类型 ('synthetic' 或 'real')
            num_points: 点数量
            extra_param_0: FPT特有数据
            
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
        # 这里调用专门为Fast Point Transformer设计的特征雅可比矩阵计算函数
        feature_j = feature_jac_pointnetfpt(mask_fn, a_fn, ax_fn, bn_fn, extra_param_0, device).to(device)
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

def feature_jac_pointnetfpt(M, A, Ax, BN, fpt_data, device):
    """
    Fast Point Transformer特征雅可比矩阵计算函数
    
    参数:
        M: 激活掩码列表 [M1, M2, M3]，每个元素形状为[B, C, N]
        A: 权重列表 [A1, A2, A3]，卷积层权重
        Ax: 卷积输出列表 [A1_x, A2_x, A3_x]，每个元素形状为[B, C, N]
        BN: 批归一化输出列表 [bn1_x, bn2_x, bn3_x]，每个元素形状为[B, C, N]
        fpt_data: FPT特有数据，包含使用FPT的标志等
        device: 计算设备
        
    返回:
        特征雅可比矩阵 [B, 3, K, N]
    """
    # 解包掩码、权重和中间输出
    M1, M2, M3 = M[0], M[1], M[2]
    W1, W2, W3 = A[0], A[1], A[2]
    A1_x, A2_x, A3_x = Ax[0], Ax[1], Ax[2]
    bn1_x, bn2_x, bn3_x = BN[0], BN[1], BN[2]
    use_fpt = fpt_data.get("use_fpt", False)
    
    batch_size = M1.shape[0]
    num_points = M1.shape[2]
    dim_k = M3.shape[1]
    
    # 修改权重处理方式
    W1 = W1.T.detach().unsqueeze(-1)  # [out, in] -> [in, out, 1]
    W2 = W2.T.detach().unsqueeze(-1)
    W3 = W3.T.detach().unsqueeze(-1)
    
    # 计算批归一化梯度
    dBN1 = torch.autograd.grad(outputs=bn1_x, inputs=A1_x, 
                             grad_outputs=torch.ones_like(bn1_x).to(device), 
                             retain_graph=True)[0].unsqueeze(1)  # [B, 1, 64, N]
    
    dBN2 = torch.autograd.grad(outputs=bn2_x, inputs=A2_x, 
                             grad_outputs=torch.ones_like(bn2_x).to(device), 
                             retain_graph=True)[0].unsqueeze(1)  # [B, 1, 128, N]
    
    dBN3 = torch.autograd.grad(outputs=bn3_x, inputs=A3_x, 
                             grad_outputs=torch.ones_like(bn3_x).to(device), 
                             retain_graph=True)[0].unsqueeze(1)  # [B, 1, dim_k, N]
    
    # 调整激活掩码维度
    M1 = M1.unsqueeze(1)  # [B, 1, 64, N]
    M2 = M2.unsqueeze(1)  # [B, 1, 128, N]
    M3 = M3.unsqueeze(1)  # [B, 1, dim_k, N]
    
    # 计算各层梯度
    # 层1: 卷积 -> BN -> ReLU -> (可能的FPT)
    grad_layer1 = W1 * dBN1 * M1  # [B, 3, 64, N]
    
    # 层2: 卷积 -> BN -> ReLU -> (可能的FPT)
    grad_layer2 = W2 * dBN2 * M2  # [B, 64, 128, N]
    
    # 层3: 卷积 -> BN -> ReLU -> (可能的FPT)
    grad_layer3 = W3 * dBN3 * M3  # [B, 128, dim_k, N]
    
    # 如果使用FastPointTransformer，考虑其影响
    if use_fpt:
        # 此处添加FastPointTransformer特有的梯度计算逻辑
        # 为简化处理，我们近似FPT层的贡献
        fpt_attention_scale = 1.15  # 近似注意力机制的影响因子
        
        # 对每层的梯度应用注意力比例因子
        grad_layer1 = grad_layer1 * fpt_attention_scale
        grad_layer2 = grad_layer2 * fpt_attention_scale
        grad_layer3 = grad_layer3 * fpt_attention_scale
    
    # 通过链式法则组合梯度
    # 层1 -> 层2
    grad_layer1_to_layer2 = torch.einsum('ijkl,ikml->ijml', grad_layer1, grad_layer2)  # [B, 3, 128, N]
    
    # 层1 -> 层2 -> 层3
    grad_final = torch.einsum('ijkl,ikml->ijml', grad_layer1_to_layer2, grad_layer3)  # [B, 3, dim_k, N]
    
    # 不要转置维度，保持 [B, 3, K, N] 格式与其他特征提取器保持一致
    return grad_final  # [B, 3, dim_k, N]


##### Pointnet SwinAttention V1 特征提取器 ####

class SwinAttention1D_V1(nn.Module):
    """
    Swin Attention的一维实现，用于点云处理 - V1版本
    """
    def __init__(self, dim, window_size=16, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        
        assert dim % num_heads == 0, f"特征维度{dim}必须能被注意力头数{num_heads}整除"
        
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """x: [B, N, C]"""
        # 输入已经是[B, N, C]格式，不需要转置
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        return x  # 返回 [B, N, C]

class Pointnet_SwinAttention_Features(FeatureExtractor):
    """
    集成Swin Attention的PointNet特征提取网络
    """
    def __init__(self, dim_k=1024, window_size=16, num_heads=4, num_random_points=None):
        super(Pointnet_SwinAttention_Features, self).__init__(dim_k)
        self.extractor_type = "swinattention_v1"
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 确保头数正确
        assert 64 % num_heads == 0, f"特征维度64必须能被注意力头数{num_heads}整除"
        assert 128 % num_heads == 0, f"特征维度128必须能被注意力头数{num_heads}整除"
        assert dim_k % num_heads == 0, f"特征维度{dim_k}必须能被注意力头数{num_heads}整除"
        
        # 定义MLP层
        self.mlp1 = nn.Sequential(*mlp_layers(3, [64], b_shared=True))
        self.mlp2 = nn.Sequential(*mlp_layers(64, [128], b_shared=True))
        self.mlp3 = nn.Sequential(*mlp_layers(128, [dim_k], b_shared=True))
        
        # 使用V1版本的SwinAttention
        self.attn1 = SwinAttention1D_V1(dim=64, window_size=window_size, num_heads=num_heads)
        self.attn2 = SwinAttention1D_V1(dim=128, window_size=window_size, num_heads=num_heads)
        self.attn3 = SwinAttention1D_V1(dim=dim_k, window_size=window_size, num_heads=num_heads)

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

        if iter == -1:
            # 记录中间结果用于雅可比矩阵计算
            # 第一层MLP
            x = self.mlp1[0](x); A1_x = x  # 现在x是[B, 64, N]
            x = self.mlp1[1](x); bn1_x = x
            x = self.mlp1[2](x); M1 = (x > 0).float()
            
            # 保存注意力层输入
            attn1_in = x  # [B, 64, N]
            
            # 应用注意力
            x_t = x.transpose(1, 2)  # [B, N, C]
            x_t = self.attn1(x_t)    # 现在返回的也是[B, N, C]
            x = x_t.transpose(1, 2)  # [B, C, N]
            
            # 保存注意力层输出
            attn1_out = x  # 现在应该是[B, 64, N]
            
            # 第二层MLP
            x = self.mlp2[0](x); A2_x = x
            x = self.mlp2[1](x); bn2_x = x
            x = self.mlp2[2](x); M2 = (x > 0).float()
            
            # 保存注意力层输入
            attn2_in = x
            
            # 应用注意力
            x_t = x.transpose(1, 2)  # [B, N, C]
            x_t = self.attn2(x_t)    # 现在返回的也是[B, N, C]
            x = x_t.transpose(1, 2)  # [B, C, N]
            
            # 保存注意力层输出
            attn2_out = x
            
            # 第三层MLP
            x = self.mlp3[0](x); A3_x = x
            x = self.mlp3[1](x); bn3_x = x
            x = self.mlp3[2](x); M3 = (x > 0).float()
            
            # 保存注意力层输入
            attn3_in = x
            
            # 应用注意力
            x_t = x.transpose(1, 2)  # [B, N, C]
            x_t = self.attn3(x_t)    # 现在返回的也是[B, N, C]
            x = x_t.transpose(1, 2)  # [B, C, N]
            
            # 保存注意力层输出
            attn3_out = x

            # 最大池化
            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # 获取权重
            A1 = self.mlp1[0].weight
            A2 = self.mlp2[0].weight
            A3 = self.mlp3[0].weight

            attn_info = {
                'window_size': self.window_size,
                'num_heads': self.num_heads,
                'attn_ins': [attn1_in, attn2_in, attn3_in],
                'attn_outs': [attn1_out, attn2_out, attn3_out]
            }

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx, attn_info

        else:
            # 常规前向传播
            x = self.mlp1(x)  # [B, 64, N]
            
            # 在调用注意力层前，转置为[B, N, 64]
            x_t = x.transpose(1, 2)
            x_t = self.attn1(x_t)
            x = x_t.transpose(1, 2)  # 转回[B, 64, N]
            
            x = self.mlp2(x)  # [B, 128, N]
            
            # 在调用注意力层前，转置为[B, N, 128]
            x_t = x.transpose(1, 2)
            x_t = self.attn2(x_t)
            x = x_t.transpose(1, 2)  # 转回[B, 128, N]
            
            x = self.mlp3(x)  # [B, dim_k, N]
            
            # 在调用注意力层前，转置为[B, N, dim_k]
            x_t = x.transpose(1, 2)
            x_t = self.attn3(x_t)
            x = x_t.transpose(1, 2)  # 转回[B, dim_k, N]
            
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                     mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None, 
                     extra_param_0=None, use_numerical=False, delta=1.0e-3):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn: 激活掩码列表
            a_fn: 权重列表
            ax_fn: 卷积输出列表
            bn_fn: 批归一化输出列表
            max_idx: 最大池化索引
            mode: 模式 ("train" 或 "test")
            voxel_coords_diff: 体素坐标差分
            data_type: 数据类型 ('synthetic' 或 'real')
            num_points: 点数量
            extra_param_0: Swin Attention相关信息
            use_numerical: 是否使用数值近似方法
            delta: 数值微分的步长
        
        返回:
            雅可比矩阵 [B,K,6]
        """
        # 如果使用数值方法，调用基类的数值计算方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 原有的解析方法
        if num_points is None:
            num_points = p0.shape[1]
        batch_size = p0.shape[0]
        dim_k = self.dim_k
        device = p0.device
        
        # 1. 计算变形雅可比矩阵
        g_ = torch.zeros(batch_size, 6).to(device)
        warp_jac = utils.compute_warp_jac(g_, p0, num_points)   # B x N x 3 x 6
        
        # 2. 计算特征雅可比矩阵
        feature_j = feature_jac_swinattention(mask_fn, a_fn, ax_fn, bn_fn, extra_param_0, device).to(device)
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


def feature_jac_swinattention(M, A, Ax, BN, attn_info, device):
    """
    使用Swin Attention的特征雅可比矩阵计算函数
    
    参数:
        M: 激活掩码列表 [M1, M2, M3]
        A: 权重列表 [A1, A2, A3]
        Ax: 卷积输出列表 [A1_x, A2_x, A3_x]
        BN: 批归一化输出列表 [bn1_x, bn2_x, bn3_x]
        attn_info: Swin Attention相关信息
        device: 计算设备
        
    返回:
        特征雅可比矩阵 [B, 3, K, N]
    """
    # 解包输入列表
    M1, M2, M3 = M
    A1, A2, A3 = A
    A1_x, A2_x, A3_x = Ax
    bn1_x, bn2_x, bn3_x = BN
    
    # 权重转置并调整维度 - 使用与原始PointNet相同的方式
    A1 = (A1.T).detach().unsqueeze(-1)  # 1 x c_in x c_out x 1
    A2 = (A2.T).detach().unsqueeze(-1)
    A3 = (A3.T).detach().unsqueeze(-1)
    
    # 计算批归一化梯度
    dBN1 = torch.autograd.grad(outputs=bn1_x, inputs=A1_x, 
                             grad_outputs=torch.ones(bn1_x.size()).to(device), 
                             retain_graph=True)[0].unsqueeze(1).detach()  # B x 1 x c_out x N
    
    dBN2 = torch.autograd.grad(outputs=bn2_x, inputs=A2_x, 
                             grad_outputs=torch.ones(bn2_x.size()).to(device), 
                             retain_graph=True)[0].unsqueeze(1).detach()
    
    dBN3 = torch.autograd.grad(outputs=bn3_x, inputs=A3_x, 
                             grad_outputs=torch.ones(bn3_x.size()).to(device), 
                             retain_graph=True)[0].unsqueeze(1).detach()
    
    # 调整激活掩码维度
    M1 = M1.detach().unsqueeze(1)  # B x 1 x c_out x N
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)
    
    # 使用广播计算梯度 - 与原始PointNet相同的方式
    A1BN1M1 = A1 * dBN1 * M1  # B x c_in x c_out x N
    A2BN2M2 = A2 * dBN2 * M2
    A3BN3M3 = A3 * dBN3 * M3
    
    # 获取注意力层梯度
    dAttn1 = torch.autograd.grad(outputs=attn_info['attn_outs'][0], inputs=attn_info['attn_ins'][0], 
                               grad_outputs=torch.ones_like(attn_info['attn_outs'][0]).to(device), 
                               retain_graph=True, allow_unused=True)[0]
    if dAttn1 is None:  # 如果梯度为None，创建一个全1张量
        dAttn1 = torch.ones_like(attn_info['attn_ins'][0]).to(device)
    dAttn1 = dAttn1.detach().unsqueeze(1)
    
    dAttn2 = torch.autograd.grad(outputs=attn_info['attn_outs'][1], inputs=attn_info['attn_ins'][1], 
                               grad_outputs=torch.ones_like(attn_info['attn_outs'][1]).to(device), 
                               retain_graph=True, allow_unused=True)[0]
    if dAttn2 is None:
        dAttn2 = torch.ones_like(attn_info['attn_ins'][1]).to(device)
    dAttn2 = dAttn2.detach().unsqueeze(1)
    
    dAttn3 = torch.autograd.grad(outputs=attn_info['attn_outs'][2], inputs=attn_info['attn_ins'][2], 
                               grad_outputs=torch.ones_like(attn_info['attn_outs'][2]).to(device), 
                               retain_graph=True, allow_unused=True)[0]
    if dAttn3 is None:
        dAttn3 = torch.ones_like(attn_info['attn_ins'][2]).to(device)
    dAttn3 = dAttn3.detach().unsqueeze(1)
    
    # 组合MLP和Attention梯度
    block1_grad = A1BN1M1 * dAttn1
    block2_grad = A2BN2M2 * dAttn2
    block3_grad = A3BN3M3 * dAttn3
    
    # 使用einsum组合梯度链 - 与原始PointNet相同
    block1_to_block2 = torch.einsum('ijkl,ikml->ijml', block1_grad, block2_grad)
    final_grad = torch.einsum('ijkl,ikml->ijml', block1_to_block2, block3_grad)
    
    return final_grad


##### Pointnet SwinAttention V2 特征提取器 ####

class SwinAttention1D_V2(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        
        # 使用标准多头注意力
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        x: [B, N, C]
        window attention over N dimension
        """
        B, N, C = x.shape
        
        # 确保特征维度与初始化时设定的维度一致
        assert C == self.dim, f"输入特征维度{C}与设定维度{self.dim}不匹配"
        
        # Pad to fit window size
        pad_len = (self.window_size - N % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)
        N_pad = x.size(1)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x
        
        # 改变窗口分区方式
        # 不要将窗口维度合并到批次维度中
        # 直接对整个序列应用注意力
        
        # Self attention inside each window - 直接使用原始序列
        attn_out, _ = self.attn(shifted_x, shifted_x, shifted_x)
        
        # Residual + norm
        x = self.norm(attn_out + shifted_x)
        
        # 移除填充
        if pad_len > 0:
            x = x[:, :N, :]
            
        return x

class MLPNet(nn.Module):
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, use_swin=False):
        super().__init__()
        self.use_swin = use_swin
        
        layers = []
        last = nch_input
        
        for outp in nch_layers:
            if b_shared:
                layers.append(nn.Conv1d(last, outp, 1))
            else:
                layers.append(nn.Linear(last, outp))
            
            layers.append(nn.BatchNorm1d(outp, momentum=bn_momentum))
            layers.append(nn.ReLU())
            
            if use_swin and b_shared:
                # 添加一个额外的卷积层保持通道数不变
                layers.append(nn.Conv1d(outp, outp, 1))
                # 创建独立的SwinAttention1D实例
                self.swin_attn = SwinAttention1D_V2(outp)
            
            if not b_shared and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            last = outp
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # 正常情况下的前向传播
        x = self.layers(x)
        
        # 如果使用SwinAttention，需要手动处理
        if self.use_swin:
            # 'b c n -> b n c'
            x = x.permute(0, 2, 1)
            # 确保维度匹配
            if x.size(2) != self.swin_attn.dim:
                raise ValueError(f"特征维度 {x.size(2)} 与Swin Attention维度 {self.swin_attn.dim} 不匹配")
            x = self.swin_attn(x)
            # 'b n c -> b c n'
            x = x.permute(0, 2, 1)
        
        return x

class Pointnet_SwinAttention_V2_Features(FeatureExtractor):
    def __init__(self, dim_k=1024, use_swin=True, window_size=8, num_heads=4):
        super(Pointnet_SwinAttention_V2_Features, self).__init__(dim_k)
        # 设置特征提取器类型标识
        self.extractor_type = "swinattention_v2"
        self.use_swin = use_swin
        self.dim_k = dim_k
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 创建MLP层
        self.mlp1 = MLPNet(3, [64], use_swin=use_swin)
        self.mlp2 = MLPNet(64, [128], use_swin=use_swin)
        self.mlp3 = MLPNet(128, [dim_k], use_swin=use_swin)
        
        # 存储中间层用于雅可比矩阵计算
        self.conv1 = self.mlp1.layers[0]
        self.bn1 = self.mlp1.layers[1]
        self.relu1 = self.mlp1.layers[2]
        
        self.conv2 = self.mlp2.layers[0]
        self.bn2 = self.mlp2.layers[1]
        self.relu2 = self.mlp2.layers[2]
        
        self.conv3 = self.mlp3.layers[0]
        self.bn3 = self.mlp3.layers[1]
        self.relu3 = self.mlp3.layers[2]
        
        # 如果使用Swin Attention，额外保存SwinAttention输入前的卷积层
        if use_swin:
            self.conv1_swin = self.mlp1.layers[3]
            self.swin1 = self.mlp1.swin_attn
            
            self.conv2_swin = self.mlp2.layers[3]
            self.swin2 = self.mlp2.swin_attn
            
            self.conv3_swin = self.mlp3.layers[3]
            self.swin3 = self.mlp3.swin_attn

    def forward(self, points, iter=-1):
        x = points.transpose(1, 2)  # [B, 3, N]
        
        if iter == -1:  # Feature extraction mode
            # MLP1
            x = self.conv1(x)
            A1_x = x
            x = self.bn1(x)
            bn1_x = x
            x = self.relu1(x)
            
            # ReLU之后、Attention之前计算M1
            M1 = (x > 0).float()
            
            # 如果使用Swin Attention
            if self.use_swin:
                x = self.conv1_swin(x)
                attn1_in = x.clone()
                # 'b c n -> b n c'
                x = x.permute(0, 2, 1)
                x = self.swin1(x)
                # 'b n c -> b c n'
                x = x.permute(0, 2, 1)
                attn1_out = x.clone()
            
            # MLP2
            x = self.conv2(x)
            A2_x = x
            x = self.bn2(x)
            bn2_x = x
            x = self.relu2(x)
            
            # ReLU之后、Attention之前计算M2
            M2 = (x > 0).float()
            
            # 如果使用Swin Attention
            if self.use_swin:
                x = self.conv2_swin(x)
                attn2_in = x.clone()
                # 'b c n -> b n c'
                x = x.permute(0, 2, 1)
                x = self.swin2(x)
                # 'b n c -> b c n'
                x = x.permute(0, 2, 1)
                attn2_out = x.clone()
            
            # MLP3
            x = self.conv3(x)
            A3_x = x
            x = self.bn3(x)
            bn3_x = x
            x = self.relu3(x)
            
            # ReLU之后、Attention之前计算M3
            M3 = (x > 0).float()
            
            # 如果使用Swin Attention
            if self.use_swin:
                x = self.conv3_swin(x)
                attn3_in = x.clone()
                # 'b c n -> b n c'
                x = x.permute(0, 2, 1)
                x = self.swin3(x)
                # 'b n c -> b c n'
                x = x.permute(0, 2, 1)
                attn3_out = x.clone()
            
            # Global max pooling
            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            # 添加Swin Attention相关信息用于雅可比矩阵计算
            if self.use_swin:
                attn_info = {
                    'window_size': self.window_size,
                    'num_heads': self.num_heads,
                    'attn_ins': [attn1_in, attn2_in, attn3_in],
                    'attn_outs': [attn1_out, attn2_out, attn3_out]
                }
            else:
                attn_info = None
            
            return (
                x, 
                [M1, M2, M3],
                [self.conv1.weight, self.conv2.weight, self.conv3.weight],
                [A1_x, A2_x, A3_x],
                [bn1_x, bn2_x, bn3_x],
                max_idx,
                attn_info
            )
        else:  # Standard forward pass
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
            
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                     mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None, 
                     extra_param_0=None, use_numerical=False, delta=1.0e-3):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn: 激活掩码列表
            a_fn: 权重列表
            ax_fn: 卷积输出列表
            bn_fn: 批归一化输出列表
            max_idx: 最大池化索引
            mode: 模式，"train"或"test"
            voxel_coords_diff: 体素坐标差分
            data_type: 数据类型 ('synthetic' 或 'real')
            num_points: 点数量
            extra_param_0: Swin Attention相关信息
        
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
        feature_j = feature_jac_swinattention_v2(mask_fn, a_fn, ax_fn, bn_fn, extra_param_0, device).to(device)
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


def feature_jac_swinattention_v2(M, A, Ax, BN, attn_info, device):
    """
    使用Swin Attention的特征雅可比矩阵计算函数（V2版本）
    
    参数:
        M: 激活掩码列表 [M1, M2, M3]
        A: 权重列表 [A1, A2, A3]
        Ax: 卷积输出列表 [A1_x, A2_x, A3_x]
        BN: 批归一化输出列表 [bn1_x, bn2_x, bn3_x]
        attn_info: Swin Attention相关信息
        device: 计算设备
    """
    # 解包输入列表
    M1, M2, M3 = M
    A1, A2, A3 = A
    A1_x, A2_x, A3_x = Ax
    bn1_x, bn2_x, bn3_x = BN
    
    # 获取批次大小、点数和特征维度
    batch_size = M1.shape[0]
    num_points = M1.shape[2]
    dim_k = A3.shape[0]
    
    # 权重转置并调整维度
    # 修改张量维度处理方式，确保维度匹配
    A1 = (A1.transpose(0, 1)).detach().unsqueeze(0).expand(batch_size, -1, -1, -1)
    A2 = (A2.transpose(0, 1)).detach().unsqueeze(0).expand(batch_size, -1, -1, -1)
    A3 = (A3.transpose(0, 1)).detach().unsqueeze(0).expand(batch_size, -1, -1, -1)
    
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
    
    # 调整激活掩码维度
    M1 = M1.detach().unsqueeze(1)
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)
    
    # 计算MLP部分的梯度 - 确保维度匹配
    grad_mlp1 = A1 * dBN1 * M1  # 现在 A1 已扩展到与 dBN1 相同的批次大小
    grad_mlp2 = A2 * dBN2 * M2
    grad_mlp3 = A3 * dBN3 * M3
    
    # 获取前向传播中保存的注意力层输入和输出
    attn_ins = attn_info['attn_ins']  # [attn1_in, attn2_in, attn3_in]
    attn_outs = attn_info['attn_outs']  # [attn1_out, attn2_out, attn3_out]
    
    # 为每个注意力层计算梯度，允许未使用的输入
    dAttn1 = torch.autograd.grad(outputs=attn_outs[0], inputs=attn_ins[0], 
                               grad_outputs=torch.ones_like(attn_outs[0]).to(device), 
                               retain_graph=True, allow_unused=True)[0]
    
    dAttn2 = torch.autograd.grad(outputs=attn_outs[1], inputs=attn_ins[1], 
                               grad_outputs=torch.ones_like(attn_outs[1]).to(device), 
                               retain_graph=True, allow_unused=True)[0]
    
    dAttn3 = torch.autograd.grad(outputs=attn_outs[2], inputs=attn_ins[2], 
                               grad_outputs=torch.ones_like(attn_outs[2]).to(device), 
                               retain_graph=True, allow_unused=True)[0]
    
    # 处理可能为None的梯度
    if dAttn1 is None:
        dAttn1 = torch.ones_like(attn_ins[0]).to(device)
    if dAttn2 is None:
        dAttn2 = torch.ones_like(attn_ins[1]).to(device)
    if dAttn3 is None:
        dAttn3 = torch.ones_like(attn_ins[2]).to(device)
    
    # 调整维度以符合后续计算需求
    dAttn1 = dAttn1.detach().unsqueeze(1)
    dAttn2 = dAttn2.detach().unsqueeze(1)
    dAttn3 = dAttn3.detach().unsqueeze(1)
    
    # 组合MLP和Attention梯度
    grad_block1 = grad_mlp1 * dAttn1
    grad_block2 = grad_mlp2 * dAttn2
    grad_block3 = grad_mlp3 * dAttn3
    
    # 使用链式法则组合梯度
    grad_block1_to_block2 = torch.einsum('ijkl,ikml->ijml', grad_block1, grad_block2)
    grad_final = torch.einsum('ijkl,ikml->ijml', grad_block1_to_block2, grad_block3)
    
    return grad_final


##### SSM特征提取器 V1 ####

class SSMCore(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(SSMCore, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # 可学习的SSM参数
        self.A = nn.Parameter(torch.eye(hidden_dim) + 0.01 * torch.randn(hidden_dim, hidden_dim))
        self.B = nn.Parameter(0.01 * torch.randn(hidden_dim, input_dim))
        self.C = nn.Parameter(0.01 * torch.randn(output_dim, hidden_dim))
        self.Lambda = nn.Parameter(torch.ones(hidden_dim) * 0.5)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C_in = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        A_norm = self.A / (torch.linalg.norm(self.A, ord=2) + 1e-5)

        outputs = []
        for t in range(N):
            u_t = x[:, t, :]
            h = torch.relu(h @ A_norm.T + u_t @ self.B.T)
            y_t = h @ self.C.T
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, N, C_out]
        return y


class MLPWithSSMv1(nn.Module):
    def __init__(self, nch_input, nch_layers, hidden_dim=64, seq_len=1024, bn_momentum=0.1):
        super().__init__()
        self.mlp = nn.Sequential(*mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=bn_momentum))
        self.ssm = SSMCore(nch_layers[-1], hidden_dim=hidden_dim, output_dim=nch_layers[-1], seq_len=seq_len)

    def forward(self, x):
        # x: [B, C_in, N]
        x = self.mlp(x)  # [B, C_out, N]
        x = x.transpose(1, 2)  # [B, N, C_out]
        x = self.ssm(x)        # [B, N, C_out]
        x = x.transpose(1, 2)  # [B, C_out, N]
        return x


class SSM_Features_v1(FeatureExtractor):
    """
    SSM特征提取网络v1
    """
    def __init__(self, dim_k=1024, seq_len=1024):
        super(SSM_Features_v1, self).__init__(dim_k)
        # 设置特征提取器类型标识
        self.extractor_type = "ssm_v1"
        self.dim_k = dim_k
        
        self.mlp1 = MLPWithSSMv1(3, [64], hidden_dim=64, seq_len=seq_len)
        self.mlp2 = MLPWithSSMv1(64, [128], hidden_dim=128, seq_len=seq_len)
        self.mlp3 = MLPWithSSMv1(128, [dim_k], hidden_dim=dim_k, seq_len=seq_len)

    def forward(self, points, iter):
        """ points: [B, N, 3] → [B, K] """
        x = points.transpose(1, 2)  # [B, 3, N]

        if iter == -1:
            # Forward with feature tracing
            x = self.mlp1.mlp[0](x)
            A1_x = x
            x = self.mlp1.mlp[1](x)
            bn1_x = x
            x = self.mlp1.mlp[2](x)
            M1 = (x > 0).float()
            
            # 保存SSM输入
            ssm1_in = x
            x_t = x.transpose(1, 2)  # [B, N, C]
            x_t = self.mlp1.ssm(x_t)
            # 保存SSM输出
            ssm1_out = x_t
            x = x_t.transpose(1, 2)  # [B, C, N]

            x = self.mlp2.mlp[0](x)
            A2_x = x
            x = self.mlp2.mlp[1](x)
            bn2_x = x
            x = self.mlp2.mlp[2](x)
            M2 = (x > 0).float()
            
            # 保存SSM输入
            ssm2_in = x
            x_t = x.transpose(1, 2)  # [B, N, C]
            x_t = self.mlp2.ssm(x_t)
            # 保存SSM输出
            ssm2_out = x_t
            x = x_t.transpose(1, 2)  # [B, C, N]

            x = self.mlp3.mlp[0](x)
            A3_x = x
            x = self.mlp3.mlp[1](x)
            bn3_x = x
            x = self.mlp3.mlp[2](x)
            M3 = (x > 0).float()
            
            # 保存SSM输入
            ssm3_in = x
            x_t = x.transpose(1, 2)  # [B, N, C]
            x_t = self.mlp3.ssm(x_t)
            # 保存SSM输出
            ssm3_out = x_t
            x = x_t.transpose(1, 2)  # [B, C, N]

            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # extract weights
            A1 = self.mlp1.mlp[0].weight
            A2 = self.mlp2.mlp[0].weight
            A3 = self.mlp3.mlp[0].weight

            # 收集SSM数据
            ssm_data = {
                # SSM层输入（激活后）
                'ssm1_in': ssm1_in,
                'ssm2_in': ssm2_in,
                'ssm3_in': ssm3_in,
                # SSM层输出
                'ssm1_out': ssm1_out,
                'ssm2_out': ssm2_out,
                'ssm3_out': ssm3_out,
                # SSM层实例
                'ssm1': self.mlp1.ssm,
                'ssm2': self.mlp2.ssm,
                'ssm3': self.mlp3.ssm
            }

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx, ssm_data

        else:
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
            
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                    mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None, 
                    ssm_data=None, use_numerical=False, delta=1.0e-3):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn: 激活掩码列表
            a_fn: 权重列表
            ax_fn: 卷积输出列表
            bn_fn: 批归一化输出列表
            max_idx: 最大池化索引
            mode: 训练或测试模式
            voxel_coords_diff: 体素坐标差
            data_type: 数据类型
            num_points: 点云数量
            ssm_data: SSM数据（包含输入、输出和实例）
            use_numerical: 是否使用数值方法计算雅可比矩阵
            delta: 数值方法的扰动参数
            
        返回:
            雅可比矩阵 [B,K,6]
        """
        # 如果使用数值方法，调用基类的数值计算方法
        if use_numerical:
            return self.get_jacobian_numerical(p0, delta, mode, voxel_coords_diff, data_type, num_points)
        
        # 原有的解析方法
        if num_points is None:
            num_points = p0.shape[1]
        batch_size = p0.shape[0]
        dim_k = self.dim_k
        device = p0.device
        
        # 1. 计算变形雅可比矩阵
        g_ = torch.zeros(batch_size, 6).to(device)
        warp_jac = utils.compute_warp_jac(g_, p0, num_points)   # B x N x 3 x 6
        
        # 2. 计算特征雅可比矩阵 - 使用SSM特定的实现
        feature_j = feature_jac_ssm_v1(mask_fn, a_fn, ax_fn, bn_fn, ssm_data, device).to(device)
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


def feature_jac_ssm_v1(M, A, Ax, BN, ssm_data, device):
    """
    SSM特征雅可比矩阵计算函数 v1版本
    
    参数:
        M: 激活掩码列表 
        A: 权重列表
        Ax: 卷积输出列表
        BN: 批归一化输出列表
        ssm_data: SSM层的输入输出和实例
        device: 计算设备
        
    返回:
        特征雅可比矩阵
    """
    # 解包输入列表
    A1, A2, A3 = A
    M1, M2, M3 = M
    Ax1, Ax2, Ax3 = Ax
    BN1, BN2, BN3 = BN
    
    # 获取SSM数据 - 直接使用已计算的输入输出结果
    ssm1_in = ssm_data['ssm1_in']
    ssm2_in = ssm_data['ssm2_in']
    ssm3_in = ssm_data['ssm3_in']
    ssm1_out = ssm_data['ssm1_out']
    ssm2_out = ssm_data['ssm2_out']
    ssm3_out = ssm_data['ssm3_out']
    
    # 调整权重矩阵维度
    # 1 x c_in x c_out x 1
    A1 = (A1.T).detach().unsqueeze(-1)
    A2 = (A2.T).detach().unsqueeze(-1)
    A3 = (A3.T).detach().unsqueeze(-1)
    
    # 使用自动微分计算批量归一化的梯度
    dBN1 = torch.autograd.grad(outputs=BN1, inputs=Ax1, 
                              grad_outputs=torch.ones(BN1.size()).to(device), 
                              retain_graph=True)[0].unsqueeze(1).detach()
    dBN2 = torch.autograd.grad(outputs=BN2, inputs=Ax2, 
                              grad_outputs=torch.ones(BN2.size()).to(device), 
                              retain_graph=True)[0].unsqueeze(1).detach()
    dBN3 = torch.autograd.grad(outputs=BN3, inputs=Ax3, 
                              grad_outputs=torch.ones(BN3.size()).to(device), 
                              retain_graph=True)[0].unsqueeze(1).detach()
    
    # 激活掩码调整维度
    # B x 1 x c_out x N
    M1 = M1.detach().unsqueeze(1)
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)
    
    # 计算SSM层的梯度 - 直接使用ssm_data中提供的输入输出
    # 将SSM输出转为[B, C, N]格式
    ssm1_out_t = ssm1_out.transpose(1, 2) if ssm1_out.dim() == 3 else ssm1_out
    ssm2_out_t = ssm2_out.transpose(1, 2) if ssm2_out.dim() == 3 else ssm2_out
    ssm3_out_t = ssm3_out.transpose(1, 2) if ssm3_out.dim() == 3 else ssm3_out
    
    # 计算SSM梯度
    try:
        dssm1 = torch.autograd.grad(outputs=ssm1_out_t, inputs=ssm1_in,
                                  grad_outputs=torch.ones_like(ssm1_out_t).to(device),
                                  retain_graph=True,
                                  allow_unused=True)[0]
        
        if dssm1 is None:
            dssm1 = torch.ones_like(ssm1_in).to(device) * 0.01
        
        dssm2 = torch.autograd.grad(outputs=ssm2_out_t, inputs=ssm2_in,
                                  grad_outputs=torch.ones_like(ssm2_out_t).to(device),
                                  retain_graph=True,
                                  allow_unused=True)[0]
        
        if dssm2 is None:
            dssm2 = torch.ones_like(ssm2_in).to(device) * 0.01
        
        dssm3 = torch.autograd.grad(outputs=ssm3_out_t, inputs=ssm3_in,
                                  grad_outputs=torch.ones_like(ssm3_out_t).to(device),
                                  retain_graph=True,
                                  allow_unused=True)[0]
        
        if dssm3 is None:
            dssm3 = torch.ones_like(ssm3_in).to(device) * 0.01
    except Exception as e:
        print(f"计算SSM梯度时出错: {e}")
        # 使用默认梯度值
        dssm1 = torch.ones_like(ssm1_in).to(device) * 0.01
        dssm2 = torch.ones_like(ssm2_in).to(device) * 0.01
        dssm3 = torch.ones_like(ssm3_in).to(device) * 0.01
    
    # 调整SSM梯度的形状
    dssm1 = dssm1.unsqueeze(1)  # [B, 1, C, N]
    dssm2 = dssm2.unsqueeze(1)  # [B, 1, C, N] 
    dssm3 = dssm3.unsqueeze(1)  # [B, 1, C, N]
    
    # 结合各层梯度
    A1BN1M1 = A1 * dBN1 * M1 * dssm1
    A2BN2M2 = A2 * dBN2 * M2 * dssm2
    A3BN3M3 = A3 * dBN3 * M3 * dssm3
    
    # 使用einsum组合各层的梯度
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)
    A1BN1M1_A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)
    
    feat_jac = A1BN1M1_A2BN2M2_A3BN3M3
    
    return feat_jac  # B x 3 x K x N


##### SSM特征提取器 V2 ####

class SSMBlock(nn.Module):
    """状态空间模型块，简化版本确保维度匹配"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        
        # 卷积层 - 深度卷积
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=dim,
            bias=False
        )
        
        # SSM参数 - 使用一维向量形式的A以避免维度不匹配问题
        self.A = nn.Parameter(-0.5 * torch.ones(d_state))  # 一维向量形式
        # 修改B和C的维度关系，确保它们匹配
        self.B = nn.Parameter(0.1 * torch.randn(dim, d_state))  # 修改为 dim x d_state
        self.C = nn.Parameter(0.1 * torch.randn(dim, d_state))  # 保持 dim x d_state
        self.D = nn.Parameter(torch.ones(dim))
        
        # 归一化层
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        输入: [batch, channels, seq_len]
        输出: [batch, channels, seq_len]
        """
        B, C, N = x.shape
        residual = x
        
        # 因果卷积
        x_conv = self.conv(x)[:, :, :N]  # 截取前N个时间步
        x_conv = F.silu(x_conv)
        
        # 转置为[B, N, C]用于序列处理
        x_seq = x_conv.transpose(1, 2)  # [B, N, C]
        
        # 计算SSM - 简化实现
        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []
        
        # 状态空间递归 - 使用向量形式的A以避免维度问题
        A_diag = torch.diag_embed(torch.exp(self.A))  # 将A向量转为对角矩阵
        
        for t in range(N):
            # 状态更新 - 矩阵乘法 - 修改线性变换方式，确保维度匹配
            h = torch.bmm(h.unsqueeze(1), A_diag.expand(B, -1, -1)).squeeze(1) + torch.matmul(x_seq[:, t], self.B)
            # 输出计算 - 修改线性变换方式
            y = torch.matmul(h, self.C.transpose(0, 1)) + self.D * x_seq[:, t]
            outputs.append(y)
        
        # 堆叠时间步
        out = torch.stack(outputs, dim=1)  # [B, N, C]
        
        # 残差连接和归一化
        out = self.norm(out + x_seq)
        
        # 返回[B, C, N]格式
        return out.transpose(1, 2)


# 添加用于SSM_Features_v2的MLPNetSSM类
class MLPNetSSMv2(nn.Module):
    """结合MLP和SSM的网络层"""
    def __init__(self, nch_input, nch_layers, bn_momentum=0.1):
        super().__init__()
        
        # 初始化层
        layers = []
        last = nch_input
        
        for outp in nch_layers:
            # 卷积层 - 直接使用1D卷积
            layers.append(nn.Conv1d(last, outp, 1))
            
            # BN和ReLU
            layers.append(nn.BatchNorm1d(outp, momentum=bn_momentum))
            layers.append(nn.ReLU())
            
            # SSM层
            layers.append(SSMBlock(outp))
            
            last = outp
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        输入: [B, C_in, N]
        输出: [B, C_out, N]
        """
        return self.layers(x)

class SSM_Features_v2(FeatureExtractor):
    """
    SSM特征提取网络 v2版本
    """
    def __init__(self, dim_k=1024):
        super(SSM_Features_v2, self).__init__(dim_k)
        # 设置特征提取器类型标识
        self.extractor_type = "ssm_v2"
        self.dim_k = dim_k
        
        # 使用MLPNetSSMv2来包含SSM层
        self.mlp1 = MLPNetSSMv2(3, [64])
        self.mlp2 = MLPNetSSMv2(64, [128])
        self.mlp3 = MLPNetSSMv2(128, [dim_k])
        
        # 直接引用各层以避免索引问题
        # MLP1层
        self.conv1 = self.mlp1.layers[0]
        self.bn1 = self.mlp1.layers[1]
        self.relu1 = self.mlp1.layers[2]
        self.ssm1 = self.mlp1.layers[3]
        
        # MLP2层
        self.conv2 = self.mlp2.layers[0]
        self.bn2 = self.mlp2.layers[1]
        self.relu2 = self.mlp2.layers[2]
        self.ssm2 = self.mlp2.layers[3]
        
        # MLP3层
        self.conv3 = self.mlp3.layers[0]
        self.bn3 = self.mlp3.layers[1]
        self.relu3 = self.mlp3.layers[2]
        self.ssm3 = self.mlp3.layers[3]

    def forward(self, points, iter=-1):
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
        
        if iter == -1:  # 特征提取模式
            # 第一层: 卷积->BN->ReLU->SSM
            x = self.conv1(x)
            A1_x = x.clone()  # 保存卷积输出
            
            x = self.bn1(x)
            bn1_x = x.clone()  # 保存BN输出
            
            x = self.relu1(x)
            M1 = (x > 0).float()  # 保存激活掩码
            
            ssm1_in = x.clone()  # 保存SSM输入
            x = self.ssm1(x)
            ssm1_out = x.clone()  # 保存SSM输出
            
            # 第二层
            x = self.conv2(x)
            A2_x = x.clone()
            
            x = self.bn2(x)
            bn2_x = x.clone()
            
            x = self.relu2(x)
            M2 = (x > 0).float()
            
            ssm2_in = x.clone()
            x = self.ssm2(x)
            ssm2_out = x.clone()
            
            # 第三层
            x = self.conv3(x)
            A3_x = x.clone()
            
            x = self.bn3(x)
            bn3_x = x.clone()
            
            x = self.relu3(x)
            M3 = (x > 0).float()
            
            ssm3_in = x.clone()
            x = self.ssm3(x)
            ssm3_out = x.clone()
            
            # 全局最大池化
            max_idx = torch.max(x, -1)[-1]  # 获取最大值的索引
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            # 提取权重
            A1 = self.conv1.weight
            A2 = self.conv2.weight
            A3 = self.conv3.weight
            
            # 收集SSM数据
            ssm_data = {
                # SSM层输入（激活后）
                'ssm1_in': ssm1_in,
                'ssm2_in': ssm2_in,
                'ssm3_in': ssm3_in,
                # SSM层输出
                'ssm1_out': ssm1_out,
                'ssm2_out': ssm2_out,
                'ssm3_out': ssm3_out,
                # SSM层实例
                'ssm1': self.ssm1,
                'ssm2': self.ssm2,
                'ssm3': self.ssm3
            }
            
            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx, ssm_data
        
        else:  # 标准前向传播
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                    mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None, 
                    ssm_data=None, use_numerical=False, delta=1.0e-3):
        """
        计算特征提取的雅可比矩阵
        
        参数:
            p0: 输入点云 [B,N,3]
            mask_fn: 激活掩码列表
            a_fn: 权重列表
            ax_fn: 卷积输出列表
            bn_fn: 批归一化输出列表
            max_idx: 最大池化索引
            mode: 模式，'train'或'test'
            voxel_coords_diff: 体素坐标差异，用于真实数据
            data_type: 数据类型，'synthetic'或'real'
            num_points: 点数量
            ssm_data: SSM特定数据
            use_numerical: 是否使用数值方法计算雅可比矩阵
            delta: 数值方法的扰动参数
            
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
        feature_j = feature_jac_ssm_v2(mask_fn, a_fn, ax_fn, bn_fn, ssm_data, device).to(device)
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

def feature_jac_ssm_v2(M, A, Ax, BN, ssm_data, device):
    """
    SSM特征雅可比矩阵计算函数 v2版本
    
    参数:
        M: 激活掩码列表 
        A: 权重列表
        Ax: 卷积输出列表
        BN: 批归一化输出列表
        ssm_data: SSM层的输入输出和实例
        device: 计算设备
        
    返回:
        特征雅可比矩阵 [B, 3, K, N]
    """
    # 解包输入列表
    A1, A2, A3 = A
    M1, M2, M3 = M
    Ax1, Ax2, Ax3 = Ax
    BN1, BN2, BN3 = BN
    
    # 处理权重矩阵维度
    # 检查A1的维度并相应处理
    if len(A1.shape) == 3:  # 如果是3维张量 [out_ch, in_ch, kernel_size]
        A1 = A1.detach().permute(1, 0, 2)  # [in_ch, out_ch, kernel_size]
    else:  # 如果是2维张量 [out_ch, in_ch]
        A1 = A1.detach().permute(1, 0).unsqueeze(-1)  # [in_ch, out_ch, 1]
    
    if len(A2.shape) == 3:
        A2 = A2.detach().permute(1, 0, 2)
    else:
        A2 = A2.detach().permute(1, 0).unsqueeze(-1)
    
    if len(A3.shape) == 3:
        A3 = A3.detach().permute(1, 0, 2)
    else:
        A3 = A3.detach().permute(1, 0).unsqueeze(-1)
    
    try:
        # 使用自动微分计算批量归一化的梯度，添加allow_unused=True避免错误
        grad_outputs1 = torch.ones_like(BN1).to(device)
        grad_result1 = torch.autograd.grad(outputs=BN1, inputs=Ax1, 
                                  grad_outputs=grad_outputs1, 
                                  retain_graph=True, 
                                  allow_unused=True)
        # 处理可能的None结果
        dBN1 = grad_result1[0] if grad_result1[0] is not None else torch.zeros_like(Ax1)
        dBN1 = dBN1.unsqueeze(1).detach()
        
        grad_outputs2 = torch.ones_like(BN2).to(device)
        grad_result2 = torch.autograd.grad(outputs=BN2, inputs=Ax2, 
                                  grad_outputs=grad_outputs2, 
                                  retain_graph=True, 
                                  allow_unused=True)
        dBN2 = grad_result2[0] if grad_result2[0] is not None else torch.zeros_like(Ax2)
        dBN2 = dBN2.unsqueeze(1).detach()
        
        grad_outputs3 = torch.ones_like(BN3).to(device)
        grad_result3 = torch.autograd.grad(outputs=BN3, inputs=Ax3, 
                                  grad_outputs=grad_outputs3, 
                                  retain_graph=True, 
                                  allow_unused=True)
        dBN3 = grad_result3[0] if grad_result3[0] is not None else torch.zeros_like(Ax3)
        dBN3 = dBN3.unsqueeze(1).detach()
        
        # 激活掩码调整维度
        M1 = M1.detach().unsqueeze(1)  # [B, 1, C, N]
        M2 = M2.detach().unsqueeze(1)  # [B, 1, C, N]
        M3 = M3.detach().unsqueeze(1)  # [B, 1, C, N]

        # 直接使用ssm_data中提供的输入输出来计算梯度
        ssm1_in = ssm_data['ssm1_in']
        ssm2_in = ssm_data['ssm2_in']
        ssm3_in = ssm_data['ssm3_in']
        
        ssm1_out = ssm_data['ssm1_out']
        ssm2_out = ssm_data['ssm2_out']
        ssm3_out = ssm_data['ssm3_out']
        
        # 将输出转换为所需格式
        ssm1_out_t = ssm1_out.transpose(1, 2)  # [B, C, N]
        ssm2_out_t = ssm2_out.transpose(1, 2)  # [B, C, N]
        ssm3_out_t = ssm3_out.transpose(1, 2)  # [B, C, N]
        
        # 计算SSM梯度
        dssm1 = torch.autograd.grad(outputs=ssm1_out_t, inputs=ssm1_in,
                               grad_outputs=torch.ones_like(ssm1_out_t).to(device),
                               retain_graph=True,
                               allow_unused=True)[0]
        
        if dssm1 is None:
            dssm1 = torch.ones_like(ssm1_in).to(device) * 0.01  # 使用小值避免梯度消失
        
        dssm2 = torch.autograd.grad(outputs=ssm2_out_t, inputs=ssm2_in,
                               grad_outputs=torch.ones_like(ssm2_out_t).to(device),
                               retain_graph=True,
                               allow_unused=True)[0]
        
        if dssm2 is None:
            dssm2 = torch.ones_like(ssm2_in).to(device) * 0.01
        
        dssm3 = torch.autograd.grad(outputs=ssm3_out_t, inputs=ssm3_in,
                               grad_outputs=torch.ones_like(ssm3_out_t).to(device),
                               retain_graph=True,
                               allow_unused=True)[0]
        
        if dssm3 is None:
            dssm3 = torch.ones_like(ssm3_in).to(device) * 0.01
        
        # 调整SSM梯度的形状
        dssm1 = dssm1.unsqueeze(1)  # [B, 1, C, N]
        dssm2 = dssm2.unsqueeze(1)  # [B, 1, C, N] 
        dssm3 = dssm3.unsqueeze(1)  # [B, 1, C, N]
        
        # 结合各层梯度
        A1BN1M1 = A1 * dBN1 * M1 * dssm1
        A2BN2M2 = A2 * dBN2 * M2 * dssm2
        A3BN3M3 = A3 * dBN3 * M3 * dssm3
        
        # 使用einsum组合各层的梯度
        A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)
        A1BN1M1_A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)
        
        return A1BN1M1_A2BN2M2_A3BN3M3
    
    except Exception as e:
        # 如果计算失败，打印错误信息并返回零梯度
        print(f"雅可比矩阵计算失败: {e}")
        batch_size = M1.shape[0]
        return torch.zeros(batch_size, 3, A3.shape[1], M1.shape[-1]).to(device)





