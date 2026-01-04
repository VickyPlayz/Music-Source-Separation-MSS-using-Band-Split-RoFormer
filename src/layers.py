import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BandSplitModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        dim: int = 256,
        n_fft: int = 4096,
        sr: int = 44100
    ):
        """
        Splits spectrogram into K bands and projects them to latent dim D.
        Uses a predefined band configuration or learns it? 
        Paper uses ~60 bands usually for 44kHz.
        Here we define a set of band widths (in frequency bins) covering the 2049 bins.
        """
        super().__init__()
        
        # Simplified band configuration for 2049 bins
        # We'll create ~41 bands of varying width (wider at high freq)
        # 2049 bins. 
        # Example: 
        # 0-100: 10 bands of 10
        # 100-500: ...
        
        # Let's generate some bands covering 0 to 2049
        splits = []
        curr = 0
        
        # Fine resolution at low freq
        for _ in range(10): 
            width = 5 
            if curr + width > 2049: break
            splits.append((curr, curr+width))
            curr += width
            
        for _ in range(10):
            width = 20
            if curr + width > 2049: break
            splits.append((curr, curr+width))
            curr += width
            
        for _ in range(10):
            width = 50
            if curr + width > 2049: break
            splits.append((curr, curr+width))
            curr += width
            
        # Coarse resolution at high freq
        while curr < 2049:
            width = 100
            end = min(curr + width, 2049)
            splits.append((curr, end))
            curr = end
            
        self.band_ranges = splits
        self.num_bands = len(splits)
        # print(f"BandSplitModule: Generated {self.num_bands} bands.")
        
        self.norm = nn.LayerNorm(in_channels * 2) 
        
        # Projections for each band
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=2), # Flatten F_band * 2ch
                nn.Linear(in_channels * 2 * (e - s), dim),
                nn.LayerNorm(dim)
            )
            for s, e in self.band_ranges
        ])
        
    def forward(self, x):
        """
        Input: [B, C, F, T, 2] 
        Output: [B, T, K, D]
        """
        B, C, F, T, _ = x.shape
        # Input features: [B, T, F, C*2]
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, F, C*2)
        
        # Apply Input Normalization (Crucial!)
        x = self.norm(x)
        
        outs = []
        for i, (start, end) in enumerate(self.band_ranges):
            # x[:, :, start:end, :] -> [B, T, width, C*2]
            band = x[:, :, start:end, :]
            proj = self.projections[i](band) # [B, T, D]
            outs.append(proj)
            
        return torch.stack(outs, dim=2) # [B, T, K, D]

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_dim=1):
        # x: [B, Seq, Heads, HeadDim] or similar
        seq_len = x.shape[seq_dim]
        
        if self.cached_cos is None or self.cached_cos.shape[0] < seq_len:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]
            self.cached_cos = emb.cos()[None, :, None, :] # [1, seq, 1, dim]
            self.cached_sin = emb.sin()[None, :, None, :]
            
        return self.cached_cos[:, :seq_len, :, :], self.cached_sin[:, :seq_len, :, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, Seq, Heads, HeadDim]
    # cos, sin: [1, Seq, 1, HeadDim]
    # Rotate
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class BandRoFormerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, rope_cos=None, rope_sin=None):
        # x: [B, Seq, D]
        # Self Attention with RoPE
        B, Seq, D = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, Seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2) # [B, Seq, H, Dh]
        
        # Apply RoPE
        if rope_cos is not None:
             q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
             
        # Attention
        q = q.transpose(1, 2) # [B, H, Seq, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, Seq, D)
        x = self.proj(x)
        x = self.dropout(x)
        
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class BandRoFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, depth=1):
        super().__init__()
        self.time_layers = nn.ModuleList([
            BandRoFormerEncoderLayer(dim, num_heads) for _ in range(depth)
        ])
        self.freq_layers = nn.ModuleList([
            BandRoFormerEncoderLayer(dim, num_heads) for _ in range(depth)
        ])
        
        self.rope_time = RotaryEmbedding(dim // num_heads)
        self.rope_freq = RotaryEmbedding(dim // num_heads)
        
    def forward(self, x):
        # x: [B, T, K, D]
        B, T, K, D = x.shape
        
        # 1. Intra-Band RNN/Transformer (Time axis) -> Independent for each band?
        # Processing time sequence for each band
        # Reshape to [B*K, T, D]
        x_time = x.permute(0, 2, 1, 3).reshape(B*K, T, D)
        cos, sin = self.rope_time(x_time, seq_dim=1)
        for layer in self.time_layers:
            x_time = layer(x_time, cos, sin)
        x = x_time.reshape(B, K, T, D).permute(0, 2, 1, 3) # Back to [B, T, K, D]
        
        # 2. Inter-Band RNN/Transformer (Freq axis) -> Independent for each time step
        # Reshape to [B*T, K, D]
        x_freq = x.reshape(B*T, K, D)
        cos, sin = self.rope_freq(x_freq, seq_dim=1)
        for layer in self.freq_layers:
            x_freq = layer(x_freq, cos, sin)
        x = x_freq.reshape(B, T, K, D)
        
        return x
