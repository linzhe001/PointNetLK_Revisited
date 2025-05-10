import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, N] → [B, N, C]
        x = x.transpose(1, 2)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        return x.transpose(1, 2)  # [B, C, N]


class MLPWithAttention(nn.Module):
    def __init__(self, nch_input, nch_layers, num_heads=4, bn_momentum=0.1):
        super().__init__()
        self.mlp = nn.Sequential(*mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=bn_momentum))
        self.attn = SelfAttention1D(nch_layers[-1], num_heads)

    def forward(self, x):
        x = self.mlp(x)
        x = self.attn(x)
        return x


class Pointnet_Features(nn.Module):
    def __init__(self, dim_k=1024, num_heads=4):
        super().__init__()
        self.mlp1 = MLPWithAttention(3, [64], num_heads)
        self.mlp2 = MLPWithAttention(64, [128], num_heads)
        self.mlp3 = MLPWithAttention(128, [dim_k], num_heads)

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
            x = self.mlp1.attn(x)

            x = self.mlp2.mlp[0](x)
            A2_x = x
            x = self.mlp2.mlp[1](x)
            bn2_x = x
            x = self.mlp2.mlp[2](x)
            M2 = (x > 0).float()
            x = self.mlp2.attn(x)

            x = self.mlp3.mlp[0](x)
            A3_x = x
            x = self.mlp3.mlp[1](x)
            bn3_x = x
            x = self.mlp3.mlp[2](x)
            M3 = (x > 0).float()
            x = self.mlp3.attn(x)

            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # extract weights
            A1 = self.mlp1.mlp[0].weight
            A2 = self.mlp2.mlp[0].weight
            A3 = self.mlp3.mlp[0].weight

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx

        else:
            # Simple forward
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
