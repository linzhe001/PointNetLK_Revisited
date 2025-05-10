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


class Pointnet_Features(nn.Module):
    def __init__(self, dim_k=1024, window_size=16, num_heads=4):
        super().__init__()
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
