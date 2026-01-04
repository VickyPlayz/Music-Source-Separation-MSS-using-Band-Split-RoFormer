import torch
import torchaudio
import numpy as np
import librosa
from typing import Optional, Union, Tuple

def load_audio(
    file_path: str,
    sr: int = 44100,
    channels: str = 'stereo',
    duration: Optional[float] = None,
    offset: float = 0.0
) -> torch.Tensor:
    """
    Load audio file and convert to torch tensor.
    Returns: [Channels, Time]
    """
    # Load using torchaudio or librosa
    # torchaudio is faster but librosa handles formats better sometimes
    # Here using torchaudio as primary
    
    try:
        if duration is not None:
             num_frames = int(duration * sr)
             frame_offset = int(offset * sr)
             info = torchaudio.info(file_path)
             # Check if we have enough frames
             if info.num_frames < frame_offset + num_frames:
                 # Adjust or padding could be added, for now just load valid region
                 pass
        
        wav, original_sr = torchaudio.load(file_path)
        
        # Resample if needed
        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(original_sr, sr)
            wav = resampler(wav)
            
        # Handle channels
        if channels == 'stereo' and wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif channels == 'mono' and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        # Crop if needed manually if load arguments didn't work (simple approach)
        if duration is not None:
            target_len = int(duration * sr)
            start = int(offset * sr)
            if wav.shape[-1] > start + target_len:
                wav = wav[:, start:start+target_len]
            elif wav.shape[-1] > start:
                wav = wav[:, start:]
            else:
                # Fallback empty or noise?
                pass
                
        return wav
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return torch.zeros(2 if channels=='stereo' else 1, int(duration * sr) if duration else sr)

def compute_stft(
    wav: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    center: bool = True
) -> torch.Tensor:
    """
    Compute STFT.
    Input: [Batch, Channels, Time] or [Channels, Time]
    Output: [Batch, Channels, Freq, Frames, Complex] (as real, imag last dim)
    """
    if wav.dim() == 2:
        wav = wav.unsqueeze(0) # Add batch dim
        
    # wav: [B, C, T]
    B, C, T = wav.shape
    wav = wav.view(-1, T) # Merge [B*C, T]
    
    window = torch.hann_window(n_fft).to(wav.device)
    stft = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
        return_complex=True,
        normalized=False
    )
    
    # stft: [B*C, F, T_frames] complex
    _, F, T_frames = stft.shape
    stft = stft.view(B, C, F, T_frames)
    
    # Convert to [B, F, T_frames, 2] usually expected by current models?
    # Or keep as complex. 
    # The plan says: Returns [B, C, F, T, 2] implicitly by splitting complex
    stft_real = torch.view_as_real(stft) # [B, C, F, T, 2]
    
    # Often models like [B, F, T, 2] but we have channels.
    # Band-split RoFormer usually treats stereo as input feature dim or handles it.
    # We'll return [B, C, F, T, 2]
    return stft_real

def compute_istft(
    stft: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    center: bool = True,
    length: Optional[int] = None
) -> torch.Tensor:
    """
    Inverse STFT.
    Input: [B, C, F, T, 2]
    Output: [B, C, Time]
    """
    B, C, F, T_frames, _ = stft.shape
    stft_complex = torch.view_as_complex(stft) # [B, C, F, T]
    
    stft_complex = stft_complex.view(-1, F, T_frames)
    
    window = torch.hann_window(n_fft).to(stft.device)
    wav = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
        length=length,
        normalized=False
    )
    
    wav = wav.view(B, C, -1)
    return wav
