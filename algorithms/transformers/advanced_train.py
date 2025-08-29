#!/usr/bin/env python3
"""
Advanced Training Script for SOTA Transformer
Implements all modern optimizations for maximum performance on M1 Max
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
from tqdm import tqdm
import multiprocessing as mp

# Import our components
from optimized_config import OptimizedSOTAConfig, get_config_125m, get_config_debug
from optimized_transformer import OptimizedSOTATransformer
from data.gpt2_tokenizer import GPT2CompatibleTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenWebTextCachedDataset(Dataset):
    """
    Dataset that loads cached OpenWebText data efficiently.
    Uses memory mapping for large datasets.
    """
    
    def __init__(
        self, 
        cache_dir: str,
        split: str = "train",
        seq_len: int = 512,
        subset_size: Optional[int] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.seq_len = seq_len
        
        logger.info(f"🗂️  Loading cached {split} data from {cache_dir}")
        
        # Try to load memory-mapped data first (most efficient)
        memmap_path = self.cache_dir / f"memmap_{split}_None_{seq_len}.npy"
        if memmap_path.exists():
            logger.info(f"   📋 Using memory-mapped data: {memmap_path}")
            self.data = np.memmap(str(memmap_path), dtype=np.int32, mode='r')
            # Reshape to sequences
            self.data = self.data.reshape(-1, seq_len + 1)  # +1 for target
        else:
            # Fallback to processed JSON data
            json_path = self.cache_dir / f"processed_{split}_512_500000.json"
            if json_path.exists():
                logger.info(f"   📋 Loading JSON data: {json_path}")
                with open(json_path, 'r') as f:
                    sequences = json.load(f)
                
                # Convert to numpy and reshape
                valid_sequences = []
                for seq in sequences:
                    if len(seq) >= seq_len + 1:
                        valid_sequences.append(seq[:seq_len + 1])
                
                self.data = np.array(valid_sequences, dtype=np.int32)
            else:
                raise FileNotFoundError(f"No cached data found in {cache_dir}")
        
        # Apply subset if requested
        if subset_size and len(self.data) > subset_size:
            self.data = self.data[:subset_size]
        
        logger.info(f"   ✅ Loaded {len(self.data):,} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids


class LearningRateScheduler:
    """
    Advanced learning rate scheduler with warmup and cosine annealing.
    """
    
    def __init__(self, config: OptimizedSOTAConfig):
        self.config = config
        self.peak_lr = config.learning_rate
        self.min_lr = config.learning_rate * config.min_lr_ratio
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps
        
        logger.info(f"📈 LR Schedule: {self.min_lr:.2e} → {self.peak_lr:.2e} → {self.min_lr:.2e}")
        logger.info(f"   Warmup: {self.warmup_steps:,} steps, Total: {self.max_steps:,} steps")
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for given step"""
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * step / self.warmup_steps
        elif step < self.max_steps:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return self.min_lr


class DropoutScheduler:
    """
    Scheduler that gradually reduces dropout during training.
    """
    
    def __init__(self, initial_dropout: float = 0.1, final_dropout: float = 0.0, total_steps: int = 100000):
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.total_steps = total_steps
    
    def get_dropout(self, step: int) -> float:
        """Get dropout rate for given step"""
        progress = min(step / self.total_steps, 1.0)
        return self.initial_dropout + (self.final_dropout - self.initial_dropout) * progress


class AdvancedTrainer:
    """
    Advanced trainer with all optimizations enabled.
    """
    
    def __init__(self, config: OptimizedSOTAConfig, model: OptimizedSOTATransformer, tokenizer: GPT2CompatibleTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Schedulers
        self.lr_scheduler = LearningRateScheduler(config)
        self.dropout_scheduler = DropoutScheduler(config.dropout, 0.0, config.max_steps)
        
        # Optimizer with proper AdamW settings
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,  # Will be overridden by scheduler
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            ignore_index=-100  # Ignore padding tokens
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logger.info(f"🎯 Advanced Trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Optimizer: AdamW(lr={config.learning_rate}, wd={config.weight_decay})")
        logger.info(f"   Label smoothing: {config.label_smoothing}")
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Execute one training step"""
        self.model.train()
        
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        
        return loss.item()
    
    def validation_step(self, val_loader: DataLoader) -> float:
        """Execute validation"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit validation batches for speed
                if num_batches >= 100:
                    break
        
        return total_loss / num_batches
    
    def generate_sample(self, prompt: str = "The future of artificial intelligence") -> str:
        """Generate a text sample for monitoring progress"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate_optimized(
                input_ids, 
                max_new_tokens=100, 
                temperature=0.8,
                top_p=0.9
            )
        
        # Decode
        return self.tokenizer.decode(generated[0].tolist(), skip_special=True)
    
    def update_model_config(self):
        """Update model configuration based on schedules"""
        # Update dropout in all modules
        current_dropout = self.dropout_scheduler.get_dropout(self.step)
        
        def set_dropout(module):
            if hasattr(module, 'dropout') and hasattr(module.dropout, 'p'):
                module.dropout.p = current_dropout
        
        self.model.apply(set_dropout)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        logger.info(f"🚀 Starting advanced training for {self.config.max_steps:,} steps")
        
        start_time = time.time()
        accumulated_loss = 0.0
        
        # Training loop
        for epoch in range(1000):  # Large number, will stop based on steps
            self.epoch = epoch
            
            for batch_idx, batch in enumerate(train_loader):
                # Update learning rate
                current_lr = self.lr_scheduler.get_lr(self.step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # Update model configuration
                self.update_model_config()
                
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                # Gradient accumulation
                if (self.step + 1) % self.config.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                self.step += 1
                
                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_loss = accumulated_loss / self.config.log_interval
                    elapsed = time.time() - start_time
                    tokens_per_sec = (self.step * self.config.effective_batch_size * self.config.seq_len_start) / elapsed
                    
                    logger.info(
                        f"Step {self.step:,}/{self.config.max_steps:,} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Tokens/sec: {tokens_per_sec:.0f}"
                    )
                    
                    self.train_losses.append(avg_loss)
                    self.learning_rates.append(current_lr)
                    accumulated_loss = 0.0
                
                # Validation
                if self.step % self.config.eval_interval == 0:
                    val_loss = self.validation_step(val_loader)
                    self.val_losses.append(val_loss)
                    
                    perplexity = math.exp(val_loss)
                    logger.info(f"📊 Validation | Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")
                        logger.info(f"💾 New best model saved (loss: {val_loss:.4f})")
                
                # Generation sample
                if self.step % self.config.generate_interval == 0:
                    sample = self.generate_sample()
                    logger.info(f"📝 Sample: '{sample[:100]}...'")
                
                # Checkpointing
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.step}.pt")
                
                # Early stopping
                if self.step >= self.config.max_steps:
                    logger.info(f"🏁 Training complete after {self.step:,} steps")
                    return
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"💾 Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"📂 Checkpoint loaded: {filename} (step {self.step:,})")


def create_data_loaders(config: OptimizedSOTAConfig) -> Tuple[DataLoader, DataLoader]:
    """Create optimized data loaders using parallel-processed data"""
    logger.info("📁 Creating data loaders from parallel-processed data...")
    
    # Use parallel-processed cache
    cache_dir = str(Path.home() / ".cache" / "openwebtext_gpt2_parallel")
    
    # Import the parallel dataset class
    from parallel_prepare_openwebtext import FastGPT2Dataset
    
    # Create datasets
    train_dataset = FastGPT2Dataset(
        data_dir=cache_dir,
        split="train",
        seq_len=config.seq_len_start
    )
    
    val_dataset = FastGPT2Dataset(
        data_dir=cache_dir,
        split="validation",
        seq_len=config.seq_len_start
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=min(4, mp.cpu_count()),
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    logger.info(f"✅ Fast data loaders created:")
    logger.info(f"   Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    logger.info(f"   Val: {len(val_dataset):,} samples, {len(val_loader):,} batches")
    
    return train_loader, val_loader


def main():
    """Main training function"""
    logger.info("🚀 ADVANCED SOTA TRANSFORMER TRAINING")
    logger.info("=" * 60)
    
    # Configuration
    config = get_config_125m()
    config.print_config()
    
    # Create model
    logger.info(f"\n🏗️  Creating model...")
    model = OptimizedSOTATransformer(config)
    
    # Load tokenizer
    logger.info(f"\n🔤 Loading tokenizer...")
    tokenizer = GPT2CompatibleTokenizer()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Create trainer
    logger.info(f"\n🎯 Creating trainer...")
    trainer = AdvancedTrainer(config, model, tokenizer)
    
    # Start training
    logger.info(f"\n🚀 Starting training...")
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info(f"\n⏹️  Training interrupted by user")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        trainer.save_checkpoint("error_checkpoint.pt")
        raise
    finally:
        # Save final checkpoint
        trainer.save_checkpoint("final_checkpoint.pt")
        logger.info(f"✅ Training session complete")


if __name__ == "__main__":
    main()