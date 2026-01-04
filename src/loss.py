import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import compute_istft

class SISDRLoss(nn.Module):
    def __init__(self, zero_mean=True, eps=1e-8):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        
    def forward(self, est, target):
        # est, target: [B, C, T] or [B, T]
        # We process per sample or batch? Batch.
        if self.zero_mean:
            est = est - est.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)
            
        # Alpha: (est . target) / ||target||^2
        # Use simple dot product along Time
        dot = torch.sum(est * target, dim=-1, keepdim=True)
        norm_target = torch.sum(target ** 2, dim=-1, keepdim=True)
        
        alpha = dot / (norm_target + self.eps)
        
        target_proj = alpha * target
        noise = est - target_proj
        
        s_target = torch.sum(target_proj ** 2, dim=-1)
        s_noise = torch.sum(noise ** 2, dim=-1)
        
        sisdr = 10 * torch.log10(s_target / (s_noise + self.eps) + self.eps)
        
        # Return negative mean SISDR
        return -sisdr.mean()

class MultiDomainLoss(nn.Module):
    def __init__(self, lambda_mag=0.5, lambda_phase=0.3, lambda_sisdr=0.2):
        super().__init__()
        self.lambda_mag = lambda_mag
        self.lambda_phase = lambda_phase
        self.lambda_sisdr = lambda_sisdr
        self.sisdr = SISDRLoss()
        
    def forward(self, est_spec, target_spec, target_audio=None):
        """
        est_spec: [B, Stems, C, F, T, 2]
        target_spec: [B, Stems, C, F, T, 2]
        target_audio: [B, Stems, C, Time] (Optional, if we want exact SISDR)
                      If None, we inverse transform target_spec, but better to pass GT audio.
        """
        
        # 1. Magnitude Loss (L1)
        est_mag = torch.norm(est_spec, dim=-1)
        target_mag = torch.norm(target_spec, dim=-1)
        loss_mag = F.l1_loss(est_mag, target_mag)
        
        # 2. Phase Loss (MSE of angle? or complex MSE?)
        loss_complex = F.l1_loss(est_spec, target_spec)
        
        # 3. SISDR
        # Need Time Domain
        # Flatten Stems/Channels into [Batch*Stems*Channels, Time]
        B, n_stems, C, freq_bins, T, _ = est_spec.shape
        
        # iSTFT
        est_spec_flat = est_spec.reshape(B*n_stems, C, freq_bins, T, 2)
        est_audio = compute_istft(est_spec_flat) # [B*n_stems, C, Time]
        
        if target_audio is not None:
             target_audio_flat = target_audio.reshape(B*n_stems, C, -1)
             # Match lengths (iSTFT might differ slightly due to padding)
             min_len = min(est_audio.shape[-1], target_audio_flat.shape[-1])
             est_audio = est_audio[..., :min_len]
             target_audio_flat = target_audio_flat[..., :min_len]
             
             loss_sisdr = self.sisdr(est_audio, target_audio_flat)
        else:
            loss_sisdr = torch.tensor(0.0, device=est_spec.device)
            
        total = self.lambda_mag * loss_mag + self.lambda_phase * loss_complex + self.lambda_sisdr * loss_sisdr
        return total, {'mag': loss_mag, 'complex': loss_complex, 'sisdr': loss_sisdr}
