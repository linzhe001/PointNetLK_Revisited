import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(3, d_model)

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

class Pointnet_Features(nn.Module):
    def __init__(self, dim_k=1024):
        super(Pointnet_Features, self).__init__()
        self.block1 = PointCloudMambaBlock(3, 64)
        self.block2 = PointCloudMambaBlock(64, 128)
        self.block3 = PointCloudMambaBlock(128, dim_k)

    def forward(self, points, iter):
        # points: [B, N, 3]
        x = points  # [B, N, 3]
        
        if iter == -1:
            # 第一个块 - 简化为单个赋值
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
            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
           # 提取权重
            A11 = self.block1.pos_enc.weight
            A12 = self.block2.pos_enc.weight
            A13 = self.block3.pos_enc.weight

            A21 = self.block1.linear.weight
            A22 = self.block2.linear.weight
            A23 = self.block3.linear.weight
            # 全局特征

            
            # 只返回全局特征，用户将自行添加中间状态收集
            return x, [M1, M2, M3], [A21, A22, A23], [bn_in1, bn_in2, bn_in3], [bn_out1, bn_out2, bn_out3], max_idx, [A11, A12, A13], [bissm_in1, bissm_in2, bissm_in3], [bissm_out1, bissm_out2, bissm_out3]
        else:
            # 标准前向传播
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = torch.max(x, dim=1)[0]
            return x
