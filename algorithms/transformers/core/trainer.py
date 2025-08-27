#!/usr/bin/env python3
"""
Transformer Training Framework
Adapted from RNN/CNN training systems with M1 Max optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import os
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import numpy as np

try:
    from .device_utils import get_best_device, should_use_mixed_precision
except ImportError:
    # Fallback if device_utils symlink doesn't work
    def get_best_device():
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def should_use_mixed_precision(device):
        return device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] >= 7


class TransformerTrainer:
    """
    Modern training framework for Transformers with intelligent stopping
    Adapted from successful RNN/CNN training patterns
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        tokenizer: Optional[object] = None,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        save_dir: str = "results/transformer",
        # Phase 2 parameters - backward compatible
        label_smoothing: float = 0.0,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        # Device setup following RNN/CNN patterns
        self.device = device or get_best_device()
        self.mixed_precision = mixed_precision and should_use_mixed_precision(self.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Training parameters
        self.gradient_clip = gradient_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase 2 training enhancements
        self.label_smoothing = label_smoothing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Intelligent stopping criteria
        self.patience_counter = 0
        self.val_history = deque(maxlen=20)
        self.improvement_threshold = 0.001  # 0.1% minimum improvement
        
        print(f"Transformer Trainer initialized on {self.device}, {self.count_parameters():,} parameters")
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def create_optimizer(self, learning_rate: float = 1e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer following modern Transformer practices
        Original paper used Adam, but AdamW is now standard
        """
        return optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),  # Original paper values
            eps=1e-9,           # Original paper value
            weight_decay=weight_decay
        )
    
    def create_scheduler(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_steps: int = 4000,
        d_model: int = 512,
        schedule_type: str = 'transformer_original'
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create learning rate scheduler - Phase 2 enhanced with multiple options
        """
        if schedule_type == 'transformer_original':
            # Original Transformer paper schedule
            def lr_lambda(step):
                if step == 0:
                    step = 1
                return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        elif schedule_type == 'cosine_annealing':
            # Phase 2: Cosine annealing with warmup
            def cosine_lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (10000 - warmup_steps)  # Assume 10k total steps
                    return 0.5 * (1 + math.cos(math.pi * progress))
            return optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)
        
        elif schedule_type == 'polynomial_decay':
            # Phase 2: Polynomial decay with warmup
            def poly_lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = min(1.0, (step - warmup_steps) / (10000 - warmup_steps))
                    return (1 - progress) ** 0.9
            return optim.lr_scheduler.LambdaLR(optimizer, poly_lr_lambda)
        
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
    
    def should_stop_training(self, val_loss: float) -> bool:
        """
        Intelligent stopping criteria adapted from CNN module
        """
        self.val_history.append(val_loss)
        
        # Need at least 10 epochs of history
        if len(self.val_history) < 10:
            return False
        
        # Check if we have improvement
        if len(self.val_history) >= 5:
            recent_best = min(list(self.val_history)[-5:])  # Best in last 5 epochs
        else:
            recent_best = min(list(self.val_history))
            
        if len(self.val_history) >= 15:
            older_best = min(list(self.val_history)[-15:-5])  # Best in epochs 5-15 ago
        else:
            older_best = recent_best + 0.1  # Assume improvement needed
        
        improvement = (older_best - recent_best) / older_best
        
        if improvement < self.improvement_threshold:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
        
        # Adaptive patience based on training progress
        if len(self.val_history) < 20:
            patience = 10
        else:
            patience = 15 if self.patience_counter < 5 else 8
        
        return self.patience_counter >= patience
    
    def train_epoch(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
            else:
                # For language modeling where input and target are the same sequence
                src = batch.to(self.device)
                tgt = src.clone()
            
            # Prepare target input (shift right for decoder)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Zero grad moved for gradient accumulation
            if batch_idx % self.gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    # For now, use simple encoder-only setup for language modeling
                    if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
                        # Full seq2seq transformer
                        output = self.model(src, tgt_input)
                    else:
                        # Encoder-only for language modeling
                        output = self.model(tgt_input)
                    
                    # Phase 2: Label smoothing support
                    if self.label_smoothing > 0.0:
                        loss = self._label_smoothing_loss(
                            output.contiguous().view(-1, output.size(-1)),
                            tgt_output.contiguous().view(-1)
                        )
                    else:
                        loss = F.cross_entropy(
                            output.contiguous().view(-1, output.size(-1)),
                            tgt_output.contiguous().view(-1),
                            ignore_index=0  # Ignore padding tokens
                        )
                
                # Phase 2: Gradient accumulation for mixed precision
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                
                # Only step after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                # Standard forward pass
                if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
                    output = self.model(src, tgt_input)
                else:
                    output = self.model(tgt_input)
                
                # Phase 2: Label smoothing support
                if self.label_smoothing > 0.0:
                    loss = self._label_smoothing_loss(
                        output.contiguous().view(-1, output.size(-1)),
                        tgt_output.contiguous().view(-1)
                    )
                else:
                    loss = F.cross_entropy(
                        output.contiguous().view(-1, output.size(-1)),
                        tgt_output.contiguous().view(-1),
                        ignore_index=0
                    )
                
                # Phase 2: Gradient accumulation support
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    
                    optimizer.step()
            
            total_loss += loss.item()
            total_tokens += tgt_output.numel()
            num_batches += 1
            
            # Progress reporting
            if batch_idx % 100 == 0:
                current_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                
                # Batch progress every 20 batches (less verbose)
                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}: Loss={current_loss:.4f}", end='\r', flush=True)
        
        avg_loss = total_loss / num_batches
        elapsed_time = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'loss': avg_loss,
            'tokens_per_sec': tokens_per_sec,
            'time': elapsed_time
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    src, tgt = batch
                    src, tgt = src.to(self.device), tgt.to(self.device)
                else:
                    src = batch.to(self.device)
                    tgt = src.clone()
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Forward pass
                if hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder'):
                    output = self.model(src, tgt_input)
                else:
                    output = self.model(tgt_input)
                
                loss = F.cross_entropy(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_output.contiguous().view(-1),
                    ignore_index=0
                )
                
                # Calculate accuracy
                predictions = output.argmax(dim=-1)
                mask = (tgt_output != 0)  # Ignore padding tokens
                correct = (predictions == tgt_output) & mask
                
                total_loss += loss.item()
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity
        }
    
    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
    
    def _label_smoothing_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Phase 2: Label smoothing loss function
        Prevents overconfident predictions by distributing some probability mass
        """
        vocab_size = pred.size(-1)
        confidence = 1.0 - self.label_smoothing
        smooth_target = torch.full_like(pred, self.label_smoothing / vocab_size)
        
        # Ignore padding tokens
        mask = (target != 0)
        target = target[mask]
        pred = pred[mask]
        smooth_target = smooth_target[mask]
        
        if len(target) == 0:  # All padding
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Set confidence for true labels
        smooth_target.scatter_(-1, target.unsqueeze(-1), confidence)
        
        # KL divergence loss
        loss = F.kl_div(F.log_softmax(pred, dim=-1), smooth_target, reduction='batchmean')
        return loss
    
    def train(
        self, 
        learning_rate: float = 1e-4,
        max_epochs: int = 100,
        warmup_steps: int = 4000,
        d_model: int = 512,
        # Phase 2 parameters - backward compatible with Phase 1
        lr_schedule: str = 'transformer_original',
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.98
    ) -> Dict[str, List[float]]:
        """
        Main training loop with intelligent stopping
        """
        # Phase 2 training summary
        phase2_info = []
        if self.label_smoothing > 0:
            phase2_info.append(f"label_smoothing={self.label_smoothing}")
        if self.gradient_accumulation_steps > 1:
            phase2_info.append(f"grad_accum={self.gradient_accumulation_steps}")
        if lr_schedule != 'transformer_original':
            phase2_info.append(f"schedule={lr_schedule}")
        
        phase2_str = f" [{', '.join(phase2_info)}]" if phase2_info else " [Phase 1 compatible]"
        print(f"Starting training: LR={learning_rate}, warmup={warmup_steps}{phase2_str}")
        
        # Setup optimizer and scheduler with Phase 2 enhancements
        optimizer = self.create_optimizer(learning_rate, weight_decay)
        scheduler = self.create_scheduler(optimizer, warmup_steps, d_model, lr_schedule)
        
        training_start = time.time()
        
        epoch = 0
        while max_epochs is None or epoch < max_epochs:
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(optimizer)
            scheduler.step()
            
            # Validation
            val_metrics = self.validate()
            
            # Update history
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.learning_rates.append(scheduler.get_last_lr()[0])
            
            # Check for improvement
            is_best = False
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
                is_best = True
            
            # Save checkpoint
            self.save_checkpoint(epoch, optimizer, is_best)
            
            # Single line epoch summary like RNN/CNN
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}: Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.3f}, {train_metrics['tokens_per_sec']:.0f} tok/s, {elapsed:.1f}s")
            
            # Intelligent stopping
            if self.should_stop_training(val_metrics['loss']):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            epoch += 1
        
        total_time = time.time() - training_start
        print(f"Training completed in {total_time:.1f}s, Best Val Loss: {self.best_val_loss:.4f}, Best Val Accuracy: {self.best_val_accuracy:.3f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }