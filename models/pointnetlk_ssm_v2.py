import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SSMBlock(nn.Module):
    """State Space Model block (similar to Mamba/S4 architecture)"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        
        # Convolutional preprocessing
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=dim,
            bias=False
        )
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.randn(dim))
        self.B = nn.Linear(dim, d_state, bias=False)
        self.C = nn.Linear(dim, d_state, bias=False)
        
        # Projection layers
        self.proj_in = nn.Linear(dim, self.d_inner)
        self.proj_out = nn.Linear(self.d_inner, dim)
        
        # Normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """Input shape: [batch, channels, num_points]"""
        residual = x
        x = x.transpose(1, 2)  # [B, N, C]
        
        # Convolutional processing
        x_conv = self.conv(x.transpose(1, 2))
        x_conv = x_conv[..., :x.size(1)]  # causal padding
        x_conv = F.silu(x_conv.transpose(1, 2))
        
        # Projection
        x_proj = self.proj_in(x_conv)
        
        # State space model
        A = -torch.exp(self.A)  # Ensure stability
        B = self.B(x_proj)  # [B, N, d_state]
        C = self.C(x_proj)  # [B, N, d_state]
        D = self.D
        
        # Discretization (simplified)
        delta = torch.sigmoid(x_proj[..., :self.d_state])
        A_bar = torch.exp(A[None, None] * delta.unsqueeze(-1))  # [B, N, d_state, d_state]
        B_bar = B.unsqueeze(-1) * delta.unsqueeze(-1)  # [B, N, d_state, 1]
        
        # Sequential scan (simplified)
        h = torch.zeros(x.size(0), self.dim, self.d_state, device=x.device)
        outputs = []
        for i in range(x.size(1)):
            h = (A_bar[:, i] @ h.unsqueeze(-1)).squeeze(-1) + B_bar[:, i].squeeze(-1)
            y_i = (h @ C[:, i].unsqueeze(-1)).squeeze(-1) + D * x_proj[:, i]
            outputs.append(y_i)
        
        x_ssm = torch.stack(outputs, dim=1)
        
        # Projection and residual
        x_out = self.proj_out(x_ssm)
        x_out = self.norm(x_out + residual)
        
        return x_out.transpose(1, 2)  # [B, C, N]

class MLPNet(nn.Module):
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, use_ssm=False):
        super().__init__()
        layers = []
        last = nch_input
        
        for outp in nch_layers:
            if b_shared:
                layers.append(nn.Conv1d(last, outp, 1))
            else:
                layers.append(nn.Linear(last, outp))
            
            layers.append(nn.BatchNorm1d(outp, momentum=bn_momentum))
            layers.append(nn.ReLU())
            
            if use_ssm and b_shared:
                layers.append(SSMBlock(outp))
            
            if not b_shared and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            last = outp
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class PointnetFeatures(nn.Module):
    def __init__(self, dim_k=1024, use_ssm=True):
        super().__init__()
        self.use_ssm = use_ssm
        
        self.mlp1 = MLPNet(3, [64], use_ssm=use_ssm)
        self.mlp2 = MLPNet(64, [128], use_ssm=use_ssm)
        self.mlp3 = MLPNet(128, [dim_k], use_ssm=use_ssm)
        
        # Store intermediate layers for Jacobian computation
        self.mlp1_layers = [self.mlp1.layers[0], self.mlp1.layers[1], self.mlp1.layers[2]]
        self.mlp2_layers = [self.mlp2.layers[0], self.mlp2.layers[1], self.mlp2.layers[2]]
        self.mlp3_layers = [self.mlp3.layers[0], self.mlp3.layers[1], self.mlp3.layers[2]]
        
        if use_ssm:
            self.mlp1_layers.append(self.mlp1.layers[3])
            self.mlp2_layers.append(self.mlp2.layers[3])
            self.mlp3_layers.append(self.mlp3.layers[3])

    def forward(self, points, iter=-1):
        x = points.transpose(1, 2)  # [B, 3, N]
        
        if iter == -1:  # Feature extraction mode
            # MLP1
            x = self.mlp1_layers[0](x)
            A1_x = x
            x = self.mlp1_layers[1](x)
            bn1_x = x
            x = self.mlp1_layers[2](x)
            if self.use_ssm:
                x = self.mlp1_layers[3](x)
            M1 = (x > 0).float()
            
            # MLP2
            x = self.mlp2_layers[0](x)
            A2_x = x
            x = self.mlp2_layers[1](x)
            bn2_x = x
            x = self.mlp2_layers[2](x)
            if self.use_ssm:
                x = self.mlp2_layers[3](x)
            M2 = (x > 0).float()
            
            # MLP3
            x = self.mlp3_layers[0](x)
            A3_x = x
            x = self.mlp3_layers[1](x)
            bn3_x = x
            x = self.mlp3_layers[2](x)
            if self.use_ssm:
                x = self.mlp3_layers[3](x)
            M3 = (x > 0).float()
            
            # Global max pooling
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            return (
                x, 
                [M1, M2, M3],
                [self.mlp1_layers[0].weight, self.mlp2_layers[0].weight, self.mlp3_layers[0].weight],
                [A1_x, A2_x, A3_x],
                [bn1_x, bn2_x, bn3_x],
                None
            )
        else:  # Standard forward pass
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x