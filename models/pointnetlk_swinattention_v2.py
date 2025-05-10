import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SwinAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.shift_size = shift_size

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative positional bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Generate relative position index
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B, N, C = x.shape
        
        # Pad feature maps to multiples of window size
        pad_len = (self.window_size - N % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = x

        # Partition windows
        x_windows = rearrange(
            shifted_x,
            'b (nw w) c -> (b nw) w c',
            w=self.window_size
        )

        # Compute QKV
        qkv = self.to_qkv(x_windows).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b w (h d) -> b h w d', h=self.num_heads), qkv)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h w d -> b w (h d)')
        out = self.proj(out)

        # Merge windows
        out = rearrange(
            out,
            '(b nw) w c -> b (nw w) c',
            b=B
        )

        # Reverse cyclic shift
        if self.shift_size > 0:
            out = torch.roll(out, shifts=self.shift_size, dims=1)

        # Remove padding
        out = out[:, :N, :]
        return out

class MLPNet(nn.Module):
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, use_swin=False):
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
            
            if use_swin and b_shared:
                layers.append(nn.Sequential(
                    nn.Conv1d(outp, outp, 1),
                    Rearrange('b c n -> b n c'),
                    SwinAttention(outp),
                    Rearrange('b n c -> b c n'),
                ))
            
            if not b_shared and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            last = outp
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class PointnetFeatures(nn.Module):
    def __init__(self, dim_k=1024, use_swin=True):
        super().__init__()
        self.use_swin = use_swin
        
        self.mlp1 = MLPNet(3, [64], use_swin=use_swin)
        self.mlp2 = MLPNet(64, [128], use_swin=use_swin)
        self.mlp3 = MLPNet(128, [dim_k], use_swin=use_swin)
        
        # Store intermediate layers for Jacobian computation
        self.mlp1_layers = [self.mlp1.layers[0], self.mlp1.layers[1], self.mlp1.layers[2]]
        self.mlp2_layers = [self.mlp2.layers[0], self.mlp2.layers[1], self.mlp2.layers[2]]
        self.mlp3_layers = [self.mlp3.layers[0], self.mlp3.layers[1], self.mlp3.layers[2]]
        
        if use_swin:
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
            if self.use_swin:
                x = self.mlp1_layers[3](x)
            M1 = (x > 0).float()
            
            # MLP2
            x = self.mlp2_layers[0](x)
            A2_x = x
            x = self.mlp2_layers[1](x)
            bn2_x = x
            x = self.mlp2_layers[2](x)
            if self.use_swin:
                x = self.mlp2_layers[3](x)
            M2 = (x > 0).float()
            
            # MLP3
            x = self.mlp3_layers[0](x)
            A3_x = x
            x = self.mlp3_layers[1](x)
            bn3_x = x
            x = self.mlp3_layers[2](x)
            if self.use_swin:
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