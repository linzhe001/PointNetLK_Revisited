import numpy as np
import torch
import torch.nn as nn
import utils

class MultiHeadedAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.d_k = feature_dim // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x):
        # x shape: [B, C, N] -> need to permute to [B, N, C] for attention
        x = x.permute(0, 2, 1)
        
        batch_size = x.size(0)
        
        # Linear projections
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k)
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose to get dimensions [B, num_heads, N, d_k]
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # Final linear layer and permute back to [B, C, N]
        output = self.out(output).permute(0, 2, 1)
        
        return output

def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, use_attention=False):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        
        # Add attention after the ReLU
        if use_attention and b_shared:
            layers.append(MultiHeadedAttention(outp))
            
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers

class MLPNet(torch.nn.Module):
    """ Multi-layer perception with optional attention.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, use_attention=False):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout, use_attention)
        self.layers = torch.nn.Sequential(*list_layers)
        self.use_attention = use_attention

    def forward(self, inp):
        out = self.layers(inp)
        return out

def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a

class Pointnet_Features(torch.nn.Module):
    def __init__(self, dim_k=1024, use_attention=True):
        super().__init__()
        self.mlp1 = MLPNet(3, [64], b_shared=True, use_attention=use_attention).layers
        self.mlp2 = MLPNet(64, [128], b_shared=True, use_attention=use_attention).layers
        self.mlp3 = MLPNet(128, [dim_k], b_shared=True, use_attention=use_attention).layers
        self.use_attention = use_attention

    def forward(self, points, iter):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        x = points.transpose(1, 2) # [B, 3, N]
        
        if iter == -1:
            # Process through mlp1
            x = self.mlp1[0](x)
            A1_x = x
            x = self.mlp1[1](x)
            bn1_x = x
            x = self.mlp1[2](x)
            if self.use_attention:
                x = self.mlp1[3](x)  # Attention layer
            M1 = (x > 0).type(torch.float)
            
            # Process through mlp2
            x = self.mlp2[0](x)
            A2_x = x
            x = self.mlp2[1](x)
            bn2_x = x
            x = self.mlp2[2](x)
            if self.use_attention:
                x = self.mlp2[3](x)  # Attention layer
            M2 = (x > 0).type(torch.float)
            
            # Process through mlp3
            x = self.mlp3[0](x)
            A3_x = x
            x = self.mlp3[1](x)
            bn3_x = x
            x = self.mlp3[2](x)
            if self.use_attention:
                x = self.mlp3[3](x)  # Attention layer
            M3 = (x > 0).type(torch.float)
            
            max_idx = torch.max(x, -1)[-1]
            x = torch.nn.functional.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)

            # extract weights....
            A1 = self.mlp1[0].weight
            A2 = self.mlp2[0].weight
            A3 = self.mlp3[0].weight

            return x, [M1, M2, M3], [A1, A2, A3], [A1_x, A2_x, A3_x], [bn1_x, bn2_x, bn3_x], max_idx
        else:
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = torch.nn.functional.max_pool1d(x, x.size(-1))
            x = x.view(x.size(0), -1)
            return x