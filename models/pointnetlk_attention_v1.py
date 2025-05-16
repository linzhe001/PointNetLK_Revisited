import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import compute_warp_jac, cal_conditioned_warp_jacobian


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
            # 第一层 MLP + Attention
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attn1(x_t, x_t, x_t)
            x_t = self.norm1(attn_out + x_t)
            x = x_t.transpose(1, 2)
            
            # 第二层 MLP + Attention
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attn2(x_t, x_t, x_t)
            x_t = self.norm2(attn_out + x_t)
            x = x_t.transpose(1, 2)
            
            # 第三层 MLP + Attention
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attn3(x_t, x_t, x_t)
            x_t = self.norm3(attn_out + x_t)
            x = x_t.transpose(1, 2)
            
            # 最大池化
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            return x
            
    def get_jacobian(self, p0, mask_fn=None, a_fn=None, ax_fn=None, bn_fn=None, max_idx=None,
                     mode="train", voxel_coords_diff=None, data_type='synthetic', num_points=None,
                     extra_param_0=None):
        """
        计算特征提取的雅可比矩阵
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
            warp_condition = cal_conditioned_warp_jacobian(voxel_coords_diff)   # V x 6 x 6
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
