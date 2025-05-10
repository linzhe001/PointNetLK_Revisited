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


class Pointnet_Features(nn.Module):
    def __init__(self, dim_k=1024, seq_len=1024):
        super().__init__()
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
            x = self.mlp1.ssm(x.transpose(1, 2)).transpose(1, 2)

            x = self.mlp2.mlp[0](x)
            A2_x = x
            x = self.mlp2.mlp[1](x)
            bn2_x = x
            x = self.mlp2.mlp[2](x)
            M2 = (x > 0).float()
            x = self.mlp2.ssm(x.transpose(1, 2)).transpose(1, 2)

            x = self.mlp3.mlp[0](x)
            A3_x = x
            x = self.mlp3.mlp[1](x)
            bn3_x = x
            x = self.mlp3.mlp[2](x)
            M3 = (x > 0).float()
            x = self.mlp3.ssm(x.transpose(1, 2)).transpose(1, 2)

            max_idx = torch.max(x, -1)[-1]
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # extract weights
            A1 = self.mlp1.mlp[0].weight
            A2 = self.mlp2.mlp[0].weight
            A3 = self.mlp3.mlp[0].weight

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx

        else:
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x
