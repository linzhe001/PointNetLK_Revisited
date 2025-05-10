import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class FastPointTransformerLayer(nn.Module):
    """Efficient transformer layer from Fast Point Transformer (ICCV 2023)"""
    def __init__(self, dim, num_heads=4, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query/key/value projections
        self.to_qkv = nn.Conv1d(dim, dim * 3, 1)
        
        # Local feature aggregation
        self.local_agg = nn.Sequential(
            nn.Conv1d(dim, dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(dim // reduction_ratio, dim, 1)
        )
        
        # Output projection
        self.proj = nn.Conv1d(dim, dim, 1)
        
        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        """Input: [B, C, N]"""
        B, C, N = x.shape
        residual = x
        
        # Local feature enhancement
        local_feat = self.local_agg(x)
        
        # Query/key/value
        qkv = self.to_qkv(self.norm1(x.transpose(1, 2)).transpose(1, 2)
        q, k, v = rearrange(qkv, 'b (three h d) n -> three b h n d', 
                           three=3, h=self.num_heads).unbind(0)
        
        # Deferred attention (from Fast Point Transformer)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        global_feat = rearrange(attn @ v, 'b h n d -> b (h d) n')
        
        # Combine features
        out = global_feat + local_feat
        out = self.proj(self.norm2(out.transpose(1, 2)).transpose(1, 2) + residual
        
        return out

class Pointnet_Features(nn.Module):
    def __init__(self, dim_k=1024, use_fpt=True):
        super().__init__()
        self.use_fpt = use_fpt
        
        # Shared MLP layers with optional transformer blocks
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            FastPointTransformerLayer(64) if use_fpt else nn.Identity()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            FastPointTransformerLayer(128) if use_fpt else nn.Identity()
        )
        
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, dim_k, 1),
            nn.BatchNorm1d(dim_k),
            nn.ReLU(),
            FastPointTransformerLayer(dim_k) if use_fpt else nn.Identity()
        )
        
        # Store layer references for Jacobian computation
        self._init_layer_references()

    def _init_layer_references(self):
        """Initialize references to layers needed for Jacobian computation"""
        self.mlp1_layers = [self.mlp1[0], self.mlp1[1], self.mlp1[2]]
        self.mlp2_layers = [self.mlp2[0], self.mlp2[1], self.mlp2[2]]
        self.mlp3_layers = [self.mlp3[0], self.mlp3[1], self.mlp3[2]]
        
        if self.use_fpt:
            self.mlp1_layers.append(self.mlp1[3])
            self.mlp2_layers.append(self.mlp2[3])
            self.mlp3_layers.append(self.mlp3[3])

    def forward(self, points, iter=-1):
        """
        Args:
            points: Input point cloud [B, N, 3]
            iter: If -1, returns intermediate features for Jacobian computation
        Returns:
            Same outputs as original PointNetLK:
            - features: Global features [B, K]
            - masks: List of activation masks [M1, M2, M3]
            - weights: List of weights [W1, W2, W3]
            - pre_activations: List of pre-activation values
            - bn_activations: List of batch norm activations
            - max_idx: None (for compatibility)
        """
        x = points.transpose(1, 2)  # [B, 3, N]
        
        if iter == -1:  # Feature extraction mode (for Jacobian)
            # MLP1
            x = self.mlp1_layers[0](x)
            A1_x = x
            x = self.mlp1_layers[1](x)
            bn1_x = x
            x = self.mlp1_layers[2](x)
            if self.use_fpt:
                x = self.mlp1_layers[3](x)
            M1 = (x > 0).float()
            
            # MLP2
            x = self.mlp2_layers[0](x)
            A2_x = x
            x = self.mlp2_layers[1](x)
            bn2_x = x
            x = self.mlp2_layers[2](x)
            if self.use_fpt:
                x = self.mlp2_layers[3](x)
            M2 = (x > 0).float()
            
            # MLP3
            x = self.mlp3_layers[0](x)
            A3_x = x
            x = self.mlp3_layers[1](x)
            bn3_x = x
            x = self.mlp3_layers[2](x)
            if self.use_fpt:
                x = self.mlp3_layers[3](x)
            M3 = (x > 0).float()
            
            # Global max pooling
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            
            return (
                x,  # [B, K]
                [M1, M2, M3],  # Activation masks
                [self.mlp1_layers[0].weight,  # Conv weights
                 self.mlp2_layers[0].weight,
                 self.mlp3_layers[0].weight],
                [A1_x, A2_x, A3_x],  # Pre-activations
                [bn1_x, bn2_x, bn3_x],  # BN activations
                None  # max_idx (unused)
            )
        else:  # Standard forward pass
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = F.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x