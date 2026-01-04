import torch
import torch.nn as nn
from src.layers import BandSplitModule, BandRoFormerBlock

class BandMergeModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        dim: int = 256,
        n_fft: int = 4096,
        bands_config: list = None
    ):
        super().__init__()
        # Use simple bands for now, must match Split
        self.band_ranges = bands_config # Passed from Split or shared config
        
        self.projections = nn.ModuleList()
        # We need to project latent D back to Frequency Bins
        for s, e in self.band_ranges:
             width = e - s
             self.projections.append(
                 nn.Sequential(
                     nn.LayerNorm(dim),
                     nn.Linear(dim, in_channels * 2 * width),
                     # Output: Flat [B, T, C*2*width]
                 )
             )
             

    def forward(self, x):
        # x: [B, T, K, D]
        # Reconstruct [B, T, F, C*2*Stems]
        
        B, T, K, D = x.shape
        outs = []
        for i, (s, e) in enumerate(self.band_ranges):
            band_out = self.projections[i](x[:, :, i, :]) # [B, T, C*2*Stems*width]
            
            # Reshape back to [B, T, Width, C*2*Stems]
            width = e - s
            band_out = band_out.reshape(B, T, width, -1)
            outs.append(band_out)
        
        # Concat along frequency (dim 2)
        full = torch.cat(outs, dim=2) # [B, T, F, C*2*Stems]
        return full

class BandSplitRoFormer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        dim=256,
        depth=6,
        num_heads=8,
        num_stems=4
    ):
        super().__init__()
        self.num_stems = num_stems
        self.in_channels = in_channels
        
        # Instantiate Splitter
        self.band_split = BandSplitModule(in_channels=in_channels, dim=dim)
        
        # Main Transformer Backbone (The "RoFormer")
        self.layers = nn.ModuleList([
            BandRoFormerBlock(dim, num_heads) for _ in range(depth)
        ])
        
        # Instantiate Merger
        # Projects to [Stems * Channels * 2] (Complex Mask)
        self.band_merge = BandMergeModule(
            in_channels=in_channels * num_stems, 
            dim=dim,
            bands_config=self.band_split.band_ranges
        )
        
        # Output non-linearity for mask?
        # Usually Tanh (bounded) or Unbounded Complex?
        # BS-RoFormer paper uses complex masking with Tanh on mag/phase separate or just linear.
        # We will use Linear -> View as Complex -> Apply.
        
    def forward(self, x):
        """
        x: [B, C, F, T, 2] Input Spectrogram (Complex).
        """
        # Save input for masking
        # Input: [B, C, F, T, 2]
        input_spec = x.clone()
        
        # 1. Band Split
        # [B, T, K, D]
        x = self.band_split(x)
        
        # 2. Transformer Blocks
        for layer in self.layers:
            x = layer(x) # [B, T, K, D]
            
        # 3. Band Merge -> Mask Prediction
        # [B, T, F, Stems*C*2]
        out_mask = self.band_merge(x)
        
        # Reshape to [B, Stems, C, F, T, 2]
        B, T, F, _ = out_mask.shape
        
        # [B, T, F, Stems, C, 2]
        out_mask = out_mask.reshape(B, T, F, self.num_stems, self.in_channels, 2)
        
        # Permute to [B, Stems, C, F, T, 2]
        out_mask = out_mask.permute(0, 3, 4, 2, 1, 5)
        
        # 4. Apply Mask (Complex Multiplication)
        # Input: [B, 1, C, F, T, 2] (Broadcast over Stems)
        input_spec = input_spec.unsqueeze(1) 
        
        # Mask: [B, S, C, F, T, 2]
        # Treat last dim as Real/Imag
        # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        
        # Tanh activation on mask (optional, but stabilizes training)
        # out_mask = torch.tanh(out_mask) 
        
        m_r = out_mask[..., 0]
        m_i = out_mask[..., 1]
        
        i_r = input_spec[..., 0]
        i_i = input_spec[..., 1]
        
        out_r = m_r * i_r - m_i * i_i
        out_i = m_r * i_i + m_i * i_r
        
        out_spec = torch.stack([out_r, out_i], dim=-1) # [B, Stems, C, F, T, 2]
        
        return out_spec

