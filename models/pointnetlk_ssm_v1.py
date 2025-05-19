import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import compute_warp_jac, cal_conditioned_warp_jacobian
from models.extractor import FeatureExtractor


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


class SSMCore(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(SSMCore, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # Learnable SSM parameters
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


class MLPWithSSM(nn.Module):
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
        
        self.mlp1 = MLPWithSSM(3, [64], hidden_dim=64, seq_len=seq_len)
        self.mlp2 = MLPWithSSM(64, [128], hidden_dim=128, seq_len=seq_len)
        self.mlp3 = MLPWithSSM(128, [dim_k], hidden_dim=dim_k, seq_len=seq_len)

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
                    ssm_data=None):
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
    
    # 获取SSM数据
    ssm1_in = ssm_data['ssm1_in']
    ssm2_in = ssm_data['ssm2_in']
    ssm3_in = ssm_data['ssm3_in']
    ssm1_out = ssm_data['ssm1_out']
    ssm2_out = ssm_data['ssm2_out']
    ssm3_out = ssm_data['ssm3_out']
    ssm1 = ssm_data['ssm1']
    ssm2 = ssm_data['ssm2']
    ssm3 = ssm_data['ssm3']
    
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
    
    # 计算SSM层的梯度 - 使用自动微分
    # 为计算梯度准备输入
    ssm1_in_grad = ssm1_in.detach().requires_grad_(True)
    ssm2_in_grad = ssm2_in.detach().requires_grad_(True)
    ssm3_in_grad = ssm3_in.detach().requires_grad_(True)
    
    # 手动进行SSM前向传播（简化版）
    with torch.enable_grad():
        # 转置为[B, N, C]格式以适应SSM
        ssm1_t = ssm1_in_grad.transpose(1, 2)
        B1, N1, C1 = ssm1_t.shape
        h1 = torch.zeros(B1, ssm1.hidden_dim, device=device)
        A_norm1 = ssm1.A / (torch.linalg.norm(ssm1.A, ord=2) + 1e-5)
        
        # 模拟SSM的前向传播过程
        ssm1_outputs = []
        for t in range(N1):
            u_t = ssm1_t[:, t, :]
            h1 = torch.relu(h1 @ A_norm1.T + u_t @ ssm1.B.T)
            y_t = h1 @ ssm1.C.T
            ssm1_outputs.append(y_t)
        
        ssm1_out_grad = torch.stack(ssm1_outputs, dim=1)
        
        # 类似地处理第二层和第三层SSM
        # 第二层SSM
        ssm2_t = ssm2_in_grad.transpose(1, 2)
        B2, N2, C2 = ssm2_t.shape
        h2 = torch.zeros(B2, ssm2.hidden_dim, device=device)
        A_norm2 = ssm2.A / (torch.linalg.norm(ssm2.A, ord=2) + 1e-5)
        
        ssm2_outputs = []
        for t in range(N2):
            u_t = ssm2_t[:, t, :]
            h2 = torch.relu(h2 @ A_norm2.T + u_t @ ssm2.B.T)
            y_t = h2 @ ssm2.C.T
            ssm2_outputs.append(y_t)
        
        ssm2_out_grad = torch.stack(ssm2_outputs, dim=1)
        
        # 第三层SSM
        ssm3_t = ssm3_in_grad.transpose(1, 2)
        B3, N3, C3 = ssm3_t.shape
        h3 = torch.zeros(B3, ssm3.hidden_dim, device=device)
        A_norm3 = ssm3.A / (torch.linalg.norm(ssm3.A, ord=2) + 1e-5)
        
        ssm3_outputs = []
        for t in range(N3):
            u_t = ssm3_t[:, t, :]
            h3 = torch.relu(h3 @ A_norm3.T + u_t @ ssm3.B.T)
            y_t = h3 @ ssm3.C.T
            ssm3_outputs.append(y_t)
        
        ssm3_out_grad = torch.stack(ssm3_outputs, dim=1)
    
    # 计算SSM层的雅可比矩阵 - 从输出到输入
    dssm1 = torch.autograd.grad(outputs=ssm1_out_grad, inputs=ssm1_in_grad,
                               grad_outputs=torch.ones_like(ssm1_out_grad).to(device),
                               create_graph=False, retain_graph=True)[0].detach()
    
    dssm2 = torch.autograd.grad(outputs=ssm2_out_grad, inputs=ssm2_in_grad,
                               grad_outputs=torch.ones_like(ssm2_out_grad).to(device),
                               create_graph=False, retain_graph=True)[0].detach()
    
    dssm3 = torch.autograd.grad(outputs=ssm3_out_grad, inputs=ssm3_in_grad,
                               grad_outputs=torch.ones_like(ssm3_out_grad).to(device),
                               create_graph=False, retain_graph=True)[0].detach()
    
    # 调整SSM梯度的形状
    dssm1 = dssm1.unsqueeze(1)  # [B, 1, C, N]
    dssm2 = dssm2.unsqueeze(1)  # [B, 1, C, N] 
    dssm3 = dssm3.unsqueeze(1)  # [B, 1, C, N]
    
    # 结合各层梯度
    # 使用链式法则：最终梯度 = 卷积梯度 * BN梯度 * ReLU梯度 * SSM梯度
    A1BN1M1 = A1 * dBN1 * M1 * dssm1
    A2BN2M2 = A2 * dBN2 * M2 * dssm2
    A3BN3M3 = A3 * dBN3 * M3 * dssm3
    
    # 使用einsum组合各层的梯度
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)
    A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)
    
    feat_jac = A2BN2M2_A3BN3M3
    
    return feat_jac  # B x 3 x K x N
