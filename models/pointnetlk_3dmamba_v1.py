import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Mamba3DBlock(nn.Module):
    """Mamba3D block adapted for point clouds (simplified version)"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * dim)
        
        # Point-wise projections
        self.in_proj = nn.Conv1d(dim, self.d_inner * 2, 1)
        self.out_proj = nn.Conv1d(self.d_inner, dim, 1)
        
        # Depthwise convolution
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        self.B = nn.Linear(dim, d_state, bias=False)
        self.C = nn.Linear(dim, d_state, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """Input: [B, C, N]"""
        residual = x.transpose(1, 2)  # [B, N, C]
        x = x.transpose(1, 2)
        
        # Projection
        x_proj = self.in_proj(x.transpose(1, 2))  # [B, 2*d_inner, N]
        x, z = x_proj.chunk(2, dim=1)  # [B, d_inner, N] each
        
        # Conv activation
        x = self.conv(x)[..., :x.size(-1)]  # Causal padding
        x = F.silu(x)
        
        # State space model (simplified)
        A = -torch.exp(self.A)  # [d_inner, d_state]
        B = self.B(x.transpose(1, 2))  # [B, N, d_state]
        C = self.C(x.transpose(1, 2))  # [B, N, d_state]
        D = self.D
        
        # Discretization
        delta = torch.sigmoid(z.transpose(1, 2))  # [B, N, d_inner]
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, N, d_inner, d_state]
        
        # Sequential scan (simplified)
        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device)
        outputs = []
        for i in range(x.size(-1)):
            h = torch.einsum('bij,bj->bi', A, h) + delta_B[:, i]
            y_i = torch.einsum('bij,bj->bi', h, C[:, i]) + D * x[:, :, i]
            outputs.append(y_i)
        
        x_ssm = torch.stack(outputs, dim=-1)  # [B, d_inner, N]
        
        # Output projection
        out = self.out_proj(x_ssm).transpose(1, 2)
        out = self.norm(out + residual)
        
        return out.transpose(1, 2)  # [B, C, N]

class Pointnet_Features(nn.Module):
    def __init__(self, dim_k=1024, use_mamba=True):
        super().__init__()
        self.use_mamba = use_mamba
        
        # MLP layers with optional Mamba3D blocks
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            Mamba3DBlock(64) if use_mamba else nn.Identity()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            Mamba3DBlock(128) if use_mamba else nn.Identity()
        )
        
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, dim_k, 1),
            nn.BatchNorm1d(dim_k),
            nn.ReLU(),
            Mamba3DBlock(dim_k) if use_mamba else nn.Identity()
        )
        
        # Store layer references for Jacobian computation
        self.mlp1_layers = [self.mlp1[0], self.mlp1[1], self.mlp1[2]]
        self.mlp2_layers = [self.mlp2[0], self.mlp2[1], self.mlp2[2]]
        self.mlp3_layers = [self.mlp3[0], self.mlp3[1], self.mlp3[2]]
        
        if use_mamba:
            self.mlp1_layers.append(self.mlp1[3])
            self.mlp2_layers.append(self.mlp2[3])
            self.mlp3_layers.append(self.mlp3[3])

    def forward(self, points, iter=-1):
        """Maintains original PointNetLK output structure for Jacobian computation"""
        x = points.transpose(1, 2)  # [B, 3, N]
        
        if iter == -1:  # Feature extraction mode (for Jacobian)
            # MLP1
            x = self.mlp1_layers[0](x)
            A1_x = x
            x = self.mlp1_layers[1](x)
            bn1_x = x
            x = self.mlp1_layers[2](x)
            if self.use_mamba:
                x = self.mlp1_layers[3](x)
            M1 = (x > 0).float()
            
            # MLP2
            x = self.mlp2_layers[0](x)
            A2_x = x
            x = self.mlp2_layers[1](x)
            bn2_x = x
            x = self.mlp2_layers[2](x)
            if self.use_mamba:
                x = self.mlp2_layers[3](x)
            M2 = (x > 0).float()
            
            # MLP3
            x = self.mlp3_layers[0](x)
            A3_x = x
            x = self.mlp3_layers[1](x)
            bn3_x = x
            x = self.mlp3_layers[2](x)
            if self.use_mamba:
                x = self.mlp3_layers[3](x)
            M3 = (x > 0).float()
            
            # Global max pooling
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            return (
                x,  # Global features [B, K]
                [M1, M2, M3],  # Masks
                [self.mlp1_layers[0].weight,  # Weights
                 self.mlp2_layers[0].weight,
                 self.mlp3_layers[0].weight],
                [A1_x, A2_x, A3_x],  # Pre-activations
                [bn1_x, bn2_x, bn3_x],  # BN activations
                None  # max_idx (optional)
            )
        else:  # Standard forward pass
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x