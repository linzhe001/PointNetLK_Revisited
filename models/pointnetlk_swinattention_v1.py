import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extraction import FeatureExtractor, compute_warp_jac, cal_conditioned_warp_jacobian


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or [B, Cin] -> [B, Cout] """
    layers = []
    last = nch_input
    for outp in nch_layers:
        if b_shared:
            weights = nn.Conv1d(last, outp, 1)
        else:
            weights = nn.Linear(last, outp)
        layers.append(weights)
        layers.append(nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(nn.ReLU())
        if not b_shared and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        last = outp
    return layers


class SwinAttention1D(nn.Module):
    def __init__(self, dim, window_size=16, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, C, N]
        window attention over N dimension
        """
        B, C, N = x.shape
        x = x.transpose(1, 2)  # [B, N, C]

        # Pad to fit window size
        pad_len = (self.window_size - N % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), mode='constant', value=0)
        N_pad = x.size(1)

        # Partition windows
        x = x.view(B, N_pad // self.window_size, self.window_size, C)

        # Merge batch and window dims
        x = x.reshape(-1, self.window_size, C)  # [B * num_windows, window_size, C]

        # Self attention inside each window
        attn_out, _ = self.attn(x, x, x)

        # Residual + norm
        x = self.norm(attn_out + x)

        # Restore shape
        x = x.view(B, N_pad // self.window_size, self.window_size, C)
        x = x.reshape(B, N_pad, C)

        if pad_len > 0:
            x = x[:, :-pad_len, :]  # remove padding

        return x.transpose(1, 2)  # [B, C, N]


class MLPWithSwinAttention(nn.Module):
    def __init__(self, nch_input, nch_layers, window_size=16, num_heads=4, bn_momentum=0.1):
        super().__init__()
        self.mlp = nn.Sequential(*mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=bn_momentum))
        self.attn = SwinAttention1D(nch_layers[-1], window_size=window_size, num_heads=num_heads)

    def forward(self, x):
        x = self.mlp(x)
        x = self.attn(x)
        return x


class Pointnet_Features(FeatureExtractor):
    def __init__(self, dim_k=1024, window_size=16, num_heads=4):
        super(Pointnet_Features, self).__init__(dim_k)
        # 设置特征提取器类型标识
        self.extractor_type = "swinattention_v1"
        self.window_size = window_size
        self.num_heads = num_heads
        
        self.mlp1 = MLPWithSwinAttention(3, [64], window_size, num_heads)
        self.mlp2 = MLPWithSwinAttention(64, [128], window_size, num_heads)
        self.mlp3 = MLPWithSwinAttention(128, [dim_k], window_size, num_heads)

    def forward(self, points, iter):
        """ points -> features: [B, N, 3] -> [B, K] """
        x = points.transpose(1, 2)  # [B, 3, N]

        if iter == -1:
            # Forward with feature tracing
            x = self.mlp1.mlp[0](x)
            A1_x = x
            x = self.mlp1.mlp[1](x)
            bn1_x = x
            x = self.mlp1.mlp[2](x)
            M1 = (x > 0).float()
            
            # 保存注意力层的输入
            attn1_in = x.clone()
            
            x = self.mlp1.attn(x)
            
            # 保存注意力层的输出
            attn1_out = x.clone()

            x = self.mlp2.mlp[0](x)
            A2_x = x
            x = self.mlp2.mlp[1](x)
            bn2_x = x
            x = self.mlp2.mlp[2](x)
            M2 = (x > 0).float()
            
            # 保存注意力层的输入
            attn2_in = x.clone()
            
            x = self.mlp2.attn(x)
            
            # 保存注意力层的输出
            attn2_out = x.clone()

            x = self.mlp3.mlp[0](x)
            A3_x = x
            x = self.mlp3.mlp[1](x)
            bn3_x = x
            x = self.mlp3.mlp[2](x)
            M3 = (x > 0).float()
            
            # 保存注意力层的输入
            attn3_in = x.clone()
            
            x = self.mlp3.attn(x)
            
            # 保存注意力层的输出
            attn3_out = x.clone()

            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # extract weights
            A1 = self.mlp1.mlp[0].weight
            A2 = self.mlp2.mlp[0].weight
            A3 = self.mlp3.mlp[0].weight

            # 添加Swin Attention相关信息用于雅可比矩阵计算
            attn_info = {
                'window_size': self.window_size,
                'num_heads': self.num_heads,
                'attn_ins': [attn1_in, attn2_in, attn3_in],
                'attn_outs': [attn1_out, attn2_out, attn3_out]
            }

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx, attn_info

        else:
            # Simple forward
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
    
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None, 
                     mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None, 
                     extra_param_0=None):
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
            data_type: 数据类型，"synthetic"或"real"
            num_points: 点云中的点数
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
        warp_jac = compute_warp_jac(g_, p0, num_points)   # B x N x 3 x 6
        
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
            warp_condition = cal_conditioned_warp_jacobian(voxel_coords_diff)   # V x 6 x 6
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
    A1 = (A1.transpose(0, 1)).detach().unsqueeze(-1)
    A2 = (A2.transpose(0, 1)).detach().unsqueeze(-1)
    A3 = (A3.transpose(0, 1)).detach().unsqueeze(-1)
    
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
    
    # 计算MLP部分的梯度
    grad_mlp1 = A1 * dBN1 * M1
    grad_mlp2 = A2 * dBN2 * M2
    grad_mlp3 = A3 * dBN3 * M3
    
    # 获取前向传播中保存的注意力层输入和输出
    attn_ins = attn_info['attn_ins']  # [attn1_in, attn2_in, attn3_in]
    attn_outs = attn_info['attn_outs']  # [attn1_out, attn2_out, attn3_out]
    
    # 为每个注意力层计算梯度
    dAttn1 = torch.autograd.grad(outputs=attn_outs[0], inputs=attn_ins[0], 
                               grad_outputs=torch.ones_like(attn_outs[0]).to(device), 
                               retain_graph=True)[0].detach()
    
    dAttn2 = torch.autograd.grad(outputs=attn_outs[1], inputs=attn_ins[1], 
                               grad_outputs=torch.ones_like(attn_outs[1]).to(device), 
                               retain_graph=True)[0].detach()
    
    dAttn3 = torch.autograd.grad(outputs=attn_outs[2], inputs=attn_ins[2], 
                               grad_outputs=torch.ones_like(attn_outs[2]).to(device), 
                               retain_graph=True)[0].detach()
    
    # 调整维度以符合后续计算需求
    dAttn1 = dAttn1.unsqueeze(1)
    dAttn2 = dAttn2.unsqueeze(1)
    dAttn3 = dAttn3.unsqueeze(1)
    
    # 组合MLP和Attention梯度
    grad_block1 = grad_mlp1 * dAttn1
    grad_block2 = grad_mlp2 * dAttn2
    grad_block3 = grad_mlp3 * dAttn3
    
    # 使用链式法则组合梯度
    grad_block1_to_block2 = torch.einsum('ijkl,ikml->ijml', grad_block1, grad_block2)
    grad_final = torch.einsum('ijkl,ikml->ijml', grad_block1_to_block2, grad_block3)
    
    return grad_final
