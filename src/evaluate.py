import torch
import museval
import musdb
import os
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
from src.model import BandSplitRoFormer
from src.utils import compute_stft, compute_istft, load_audio

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = BandSplitRoFormer(
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load MUSDB18 Test Set
    mus = musdb.DB(root=args.root, subsets=['test'], download=args.download)
    
    results = museval.EvalStore()
    
    for track in tqdm(mus.tracks):
        # Process track
        # Sliding window or full?
        # For simplicity, let's process full if memory allows, or chunk.
        # Track duration ~3-4 mins. 44100Hz.
        # STFT on full might OOM on 3050.
        # Use simple chunking.
        
        mixture = torch.tensor(track.audio.T, dtype=torch.float32).unsqueeze(0).to(device) # [1, 2, T]
        
        # Chunk size ~10s
        chunk_size = 10 * 44100
        hop = chunk_size # Non-overlapping for speed/simplicity or overlap-add
        
        estimates = {
            'vocals': [], 'drums': [], 'bass': [], 'other': []
        }
        
        length = mixture.shape[-1]
        
        # Create zero output buffer
        out_stems = torch.zeros(1, 4, 2, length, device=device)
        
        # Overlap-add window
        # We'll just do simple cutting for now (artifacts at boundaries possible)
        # To do proper OLA:
        
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            chunk = mixture[..., start:end]
            pad_len = 0
            if chunk.shape[-1] < 1024: 
                # too small?
                pass
            
            with torch.no_grad():
                spec = compute_stft(chunk)
                current_len = chunk.shape[-1]
                
                # Forward
                # Check for padding require?
                out_spec = model(spec) # [1, 4, 2, F, T, 2]
                
                # Inverse
                # shape: [B, 4, C, F, T, 2] -> [B, 4, C, Time]
                B, S, C, F, T, _ = out_spec.shape
                out_spec = out_spec.view(B*S, C, F, T, 2)
                out_wav = compute_istft(out_spec, length=current_len)
                out_wav = out_wav.view(B, S, C, -1)
                
                out_stems[..., start:end] = out_wav
                
        # Convert to numpy [Time, Channels] for museval
        # out_stems: [1, 4, 2, T]
        
        targets = ['vocals', 'drums', 'bass', 'other']
        est_dict = {}
        for i, target in enumerate(targets):
            stem = out_stems[0, i].cpu().numpy().T # [T, 2]
            est_dict[target] = stem
            
        # Evaluate
        scores = museval.eval_mus_track(
            track, est_dict, output_dir=args.eval_dir
        )
        results.add_track(scores)
        print(scores)

    print(results)
    
    # Save results
    os.makedirs(args.eval_dir, exist_ok=True)
    results.save(os.path.join(args.eval_dir, 'results.pandas'))
    print(f"Saved results to {args.eval_dir}")
    try:
        results.df.to_csv(os.path.join(args.eval_dir, 'results.csv'))
        print(f"Saved CSV to {os.path.join(args.eval_dir, 'results.csv')}")
    except Exception as e:
        print(f"Could not save CSV: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--eval_dir', type=str, default='outputs/eval')
    
    args = parser.parse_args()
    evaluate(args)
