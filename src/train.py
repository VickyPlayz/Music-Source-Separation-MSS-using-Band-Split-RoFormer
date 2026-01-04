import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from datetime import datetime

from src.dataset import MUSDBDataset, get_dataloaders
from src.model import BandSplitRoFormer
from src.loss import MultiDomainLoss
from src.utils import compute_stft

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    train_loader, valid_loader = get_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sr=44100,
        samples_per_track=args.samples_per_track
    )
     
    
    # 2. Model
    model = BandSplitRoFormer(
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads
    ).to(device)
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = MultiDomainLoss().to(device)
    start_epoch = 0
    # Load checkpoint if exists
    if args.resume_checkpoint:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        # Check if checkpoint is a state_dict or a full checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             model.load_state_dict(checkpoint['state_dict'])
        else:
             model.load_state_dict(checkpoint) # Assuming it's just the state dict as per save code
        
        # Try to parse epoch from filename
        try:
            # Expected format: checkpoint_ep{epoch}.pt
            filename = os.path.basename(args.resume_checkpoint)
            # Remove extension
            name = os.path.splitext(filename)[0]
            # Split by '_' and get the last part which should be 'ep{N}'
            ep_str = name.split('_')[-1]
            if ep_str.startswith('ep'):
                start_epoch = int(ep_str[2:])
                print(f"Resuming from epoch {start_epoch}")
        except ValueError:
            print("Could not parse epoch from filename, starting from epoch 0 but with loaded weights.")

        # Manually set initial_lr for scheduler resumption since we aren't loading optimizer state
        for group in optimizer.param_groups:
            group['initial_lr'] = args.lr

    # Re-init scheduler with Warmup
    # Custom LambdaLR for Warmup + Cosine
    import math
    def get_lr_lambda(current_step):
        warmup_steps = 1000 # Example warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0 # Or combine with Cosine if needed, but Torch efficient implementation:
        
    # Using SequentialLR or simple Warmup Wrapper is better
    # Let's use a simpler approach: HuggingFace style or just Cosine with Warmup if available?
    # PyTorch doesn't have "OneCycle" compatible easily with resume outside of exact steps.
    # We will just stick to Cosine for now but start with a smaller LR?
    # Or strict linear warmup logic inside loop?
    # Let's use torch.optim.lr_scheduler.LambdaLR implies we manage cosine manually.
    
    # Simpler: Use OneCycleLR? It handles warmup and cosine.
    # steps_per_epoch = len(train_loader)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    
    # NOTE: OneCycleLR is great but resuming from mid-epoch or arbitrary epoch is tricky.
    # Sticking to CosineAnnealing but let's trust AdamW to handle early dynamics or the Gradient Clipping.
    # User asked for "Fruitful". Adding Warmup is safe.
    
    # Let's manually implement warmup in the loop for the first few steps?
    # Or use transformers.get_cosine_schedule_with_warmup if available? No external dep.
    
    # We will modify the loop to adjust LR for first N steps.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=start_epoch - 1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting training from epoch {start_epoch+1}...")
    accumulation_steps = args.accumulation_steps
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        optimizer.zero_grad() # Initialize gradients
        
        for i, (mix_audio, stems_audio) in enumerate(pbar):
            # mix_audio: [B, C, Time]
            # stems_audio: [B, 4, C, Time]
            mix_audio = mix_audio.to(device)
            stems_audio = stems_audio.to(device)
            
            # Compute STFT Features
            # Model expects: [B, C, F, T, 2]
            mix_spec = compute_stft(mix_audio)
            
            # Target Specs: [B, 4, C, F, T, 2]
            B, S, C, Time = stems_audio.shape
            stems_flat = stems_audio.view(-1, C, Time)
            stems_spec = compute_stft(stems_flat)
            stems_spec = stems_spec.view(B, S, C, *stems_spec.shape[-3:])
            
            # Forward
            # Now returns predicted spectrogram (via masking)
            est_spec = model(mix_spec) # [B, 4, C, F, T, 2]
            
            # Loss
            loss, components = criterion(est_spec, stems_spec, stems_audio)
            
            if torch.isnan(loss).any():
                print("Loss is NaN!")
                continue
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Step Optimizer every accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Manual Warmup
                global_step = (epoch * len(train_loader) + i) // accumulation_steps
                warmup_steps = 1000
                if global_step < warmup_steps:
                    lr_scale = min(1.0, float(global_step + 1) / float(warmup_steps))
                    for pg in optimizer.param_groups:
                        pg['lr'] = args.lr * lr_scale
            
            train_loss += loss.item() * accumulation_steps # Back to full scale for logging
            pbar.set_postfix({'loss': loss.item() * accumulation_steps, 'sisdr': components['sisdr'].item()})
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Validation (Fast check)
        # For brevity, usually check one batch or full val
        # Let's save checkpoint
        if (epoch + 1) % args.save_interval == 0:
             path = os.path.join(args.output_dir, f"checkpoint_ep{epoch+1}.pt")
             torch.save(model.state_dict(), path)
             print(f"Saved checkpoint: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=None, help='Path to MUSDB18')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--accumulation_steps', type=int, default=16, help='Gradient accumulation steps (Simulate larger batch)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dim', type=int, default=128, help='Model dimension (reduced for 3050)')
    parser.add_argument('--depth', type=int, default=4, help='Model depth')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0) # 0 for windows safety sometimes
    parser.add_argument('--samples_per_track', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='models/checkpoints')
    parser.add_argument('--save_interval', type=int, default=5)
    
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    train(args)
