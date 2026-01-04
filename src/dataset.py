import torch
from torch.utils.data import Dataset, DataLoader
import musdb
import random
import tqdm
from pathlib import Path
from typing import Optional, List, Tuple

# Import our utils
from src.utils import load_audio, compute_stft

class MUSDBDataset(Dataset):
    def __init__(
        self,
        root: str = None,
        subsets: str = 'train',
        split: str = 'train',
        seq_duration: float = 6.0,
        samples_per_track: int = 64,
        sr: int = 44100,
        download: bool = False
    ):
        """
        Args:
            root: Path to MUSDB18 dataset. If None, uses default.
            subsets: 'train' or 'test'
            split: 'train' or 'valid' (only for train subset)
            seq_duration: Duration of crops in seconds
            samples_per_track: Number of random crops per track per epoch
            sr: Sampling rate
            download: Whether to download the dataset if not found (7s version Usually)
        """
        self.mus = musdb.DB(root=root, subsets=[subsets], split=split, download=download)
        self.split = split # Store split explicitly
        self.seq_duration = seq_duration
        self.sr = sr
        self.samples_per_track = samples_per_track
        
        # Pre-calculate track lengths or valid indices if needed
        # For simplicity, we just iterate tracks
        self.tracks = list(self.mus.tracks)
        print(f"Loaded {len(self.tracks)} tracks for {subsets}/{split}")

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def __getitem__(self, index):
        # Determine track index
        track_idx = index // self.samples_per_track
        track = self.tracks[track_idx]
        
        # Decide if we do "Original" or "RandomMix"
        # For training, RandomMix is very powerful.
        # Let's say 50% chance of RandomMix if in train split
        # Or always use RandomMix for robust training if dataset is small
        
        use_random_mix = (self.split == 'train') and (random.random() < 0.8) # 80% Random Mix
        
        targets = ['vocals', 'drums', 'bass', 'other']
        stems = []
        
        if use_random_mix:
            # Pick 4 random tracks (can include self)
            # Need to ensure we pick snippets of same duration
            
            for name in targets:
                # Pick random track
                rnd_track = random.choice(self.tracks)
                
                # Pick random start
                if rnd_track.duration > self.seq_duration:
                    start_t = random.uniform(0, rnd_track.duration - self.seq_duration)
                else:
                    start_t = 0
                
                # Check for silence? (Optional)
                
                # Load stem
                rnd_track.chunk_start = start_t
                rnd_track.chunk_duration = self.seq_duration
                
                # Needs try-catch if stem loading fails?
                stem = torch.tensor(rnd_track.targets[name].audio.T, dtype=torch.float32)
                
                # Random Gain (0.25 to 1.25)
                gain = random.uniform(0.25, 1.25)
                stem = stem * gain
                
                # Reset
                rnd_track.chunk_start = 0
                rnd_track.chunk_duration = None
                
                stems.append(stem)
                
        else:
            # Original Logic
            track_duration = track.duration
            if track_duration > self.seq_duration:
                start_time = random.uniform(0, track_duration - self.seq_duration)
            else:
                start_time = 0
                
            track.chunk_start = start_time
            track.chunk_duration = self.seq_duration
            
            for name in targets:
                stem_audio = torch.tensor(track.targets[name].audio.T, dtype=torch.float32)
                
                # Apply random gain even for original mix? 
                # Yes, but "Sum" must match. If we scale stems, we must re-sum.
                gain = random.uniform(0.5, 1.0) if self.split == 'train' else 1.0
                stem_audio = stem_audio * gain
                
                stems.append(stem_audio)
            
            # Reset
            track.chunk_start = 0
            track.chunk_duration = None

        # Stack stems: [4, Channels, Time]
        stems_tensor = torch.stack(stems)
        
        # Padding if short? (Shouldn't happen with seq_duration check, but safety)
        if stems_tensor.shape[-1] < int(self.seq_duration * self.sr):
            pad = int(self.seq_duration * self.sr) - stems_tensor.shape[-1]
            stems_tensor = torch.nn.functional.pad(stems_tensor, (0, pad))
            
        # Create Mixture from Stems (Ensures consistency)
        # Mix = Sum(Stems)
        audio_mix = stems_tensor.sum(dim=0)
        
        return audio_mix, stems_tensor

def get_dataloaders(
    root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    sr: int = 44100,
    samples_per_track: int = 64,
    seq_duration: float = 6.0,
    download: bool = False
):
    train_dataset = MUSDBDataset(
        root=root, 
        subsets='train', 
        split='train', 
        seq_duration=seq_duration, 
        samples_per_track=samples_per_track,
        sr=sr,
        download=download
    )
    
    valid_dataset = MUSDBDataset(
        root=root, 
        subsets='train', 
        split='valid', 
        seq_duration=seq_duration, 
        samples_per_track=max(1, samples_per_track // 8), # Less crops for validation
        sr=sr,
        download=download
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, valid_loader

if __name__ == "__main__":
    # Test script
    print("Testing dataset implementation...")
    # Using download=True to ensure it works with 7s dataset for testing if full not present
    ds = MUSDBDataset(download=True, samples_per_track=2)
    print(f"Dataset length: {len(ds)}")
    
    mix, stems = ds[0]
    print(f"Mix shape: {mix.shape}")
    print(f"Stems shape: {stems.shape}")
    
    # Compute STFT test
    mix_stft = compute_stft(mix)
    print(f"STFT shape: {mix_stft.shape}")
