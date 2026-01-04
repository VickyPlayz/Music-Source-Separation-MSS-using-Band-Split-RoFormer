import torch
import librosa
import soundfile as sf
import argparse
import os
import numpy as np
from src.model import BandSplitRoFormer
from src.utils import compute_stft, compute_istft, load_audio

def separate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = BandSplitRoFormer(
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load Audio
    mix = load_audio(args.input, sr=44100).to(device).unsqueeze(0) # [1, 2, T]
    
    print(f"Processing {args.input}...")
    
    # Chunking (Simple overlap-add or just sequential)
    chunk_size = 10 * 44100
    overlap = 0 # Simple concatenation for demo
    length = mix.shape[-1]
    
    out_stems = torch.zeros(1, 4, 2, length, device=device)
    
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunk = mix[..., start:end]
        curr_len = chunk.shape[-1]
        
        with torch.no_grad():
             spec = compute_stft(chunk)
             est_spec = model(spec)
             
             B, S, C, F, T, _ = est_spec.shape
             est_spec = est_spec.view(B*S, C, F, T, 2)
             wav = compute_istft(est_spec, length=curr_len)
             wav = wav.view(B, S, C, -1)
             
             out_stems[..., start:end] = wav
             
    # Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    targets = ['vocals', 'drums', 'bass', 'other']
    
    basename = os.path.splitext(os.path.basename(args.input))[0]
    
    for i, name in enumerate(targets):
        stem = out_stems[0, i].cpu().numpy().T # [T, 2]
        out_path = os.path.join(args.output_dir, f"{basename}_{name}.wav")
        sf.write(out_path, stem, 44100)
        print(f"Saved {name} to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input audio file')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/separated')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    
    args = parser.parse_args()
    separate(args)
