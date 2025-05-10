import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Pointnet_Features(nn.Module):
    def __init__(self, dim_k=1024):
        super(Pointnet_Features, self).__init__()
        self.block1 = FastPointTransformerBlock(3, 64)
        self.block2 = FastPointTransformerBlock(64, 128)
        self.block3 = FastPointTransformerBlock(128, dim_k)

    def forward(self, points, iter):
        # points: [B, N, 3]
        x = points  # [B, N, 3]
        if iter == -1:
            x1 = self.block1(x)  # [B, N, 64]
            x2 = self.block2(x1)  # [B, N, 128]
            x3 = self.block3(x2)  # [B, N, dim_k]
            x_global = torch.max(x3, dim=1)[0]  # [B, dim_k]
            M1 = (x1 > 0).float()
            M2 = (x2 > 0).float()
            M3 = (x3 > 0).float()
            A1 = self.block1.linear1.weight
            A2 = self.block2.linear1.weight
            A3 = self.block3.linear1.weight
            A1_x = x1
            A2_x = x2
            A3_x = x3
            bn1_x = x1
            bn2_x = x2
            bn3_x = x3
            max_idx = torch.argmax(x3, dim=1)
            return x_global, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx
        else:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = torch.max(x, dim=1)[0]
            return x
