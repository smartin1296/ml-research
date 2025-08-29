"""
Clean, modular training script for SOTA Transformer
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import time
import logging

from sota_components.config import SOTAConfig
from sota_components.full_model import SOTATransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class TokenDataset(Dataset):
    """Simple dataset for pre-tokenized data"""
    
    def __init__(self, tokens, seq_len=128):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # input, target


def load_data():
    """Load cached tokens"""
    cache_path = Path("data/cached_tokens.npy")
    if cache_path.exists():
        tokens = np.load(cache_path)
        logging.info(f"Loaded {len(tokens):,} cached tokens")
        return torch.tensor(tokens, dtype=torch.long)
    else:
        raise FileNotFoundError(f"No cached tokens found at {cache_path}")


def train():
    """Main training function"""
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on: {device}")
    
    # Load data
    all_tokens = load_data()
    split_idx = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    # Create datasets and loaders
    seq_len = 128
    batch_size = 32
    
    train_dataset = TokenDataset(train_tokens, seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model configuration
    config = SOTAConfig(
        vocab_size=2048,  # Standard tokenizer size
        d_model=512,
        num_heads=8,
        num_layers=6,
        max_seq_len=seq_len,
        dropout=0.1,
        use_flash_attn=True  # Enable if available
    )
    
    # Create model
    model = SOTATransformer(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # Training loop
    num_epochs = 3
    logging.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        start_time = time.time()
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                targets.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Log progress
            if i % 100 == 0:
                logging.info(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    targets.reshape(-1)
                )
                
                val_loss += loss.item()
                val_steps += 1
                
                if val_steps >= 50:  # Quick validation
                    break
        
        # Report epoch metrics
        elapsed = time.time() - start_time
        logging.info(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss={train_loss/train_steps:.4f}, "
            f"Val Loss={val_loss/val_steps:.4f}, "
            f"Time={elapsed:.1f}s"
        )
    
    # Save model
    torch.save(model.state_dict(), "sota_transformer.pt")
    logging.info("Model saved to sota_transformer.pt")
    
    return model


if __name__ == "__main__":
    model = train()
    logging.info("✅ Training complete!")