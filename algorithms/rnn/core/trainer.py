import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import time
from pathlib import Path
import json
import warnings

# Import amp modules conditionally to avoid warnings on MPS
try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    # Fallback for systems without CUDA
    GradScaler = None
    autocast = None

class RNNTrainer:
    """
    SOTA RNN trainer with modern PyTorch practices:
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling  
    - Early stopping
    - Comprehensive logging
    - Checkpointing
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 grad_clip_norm: float = 1.0,
                 mixed_precision: bool = True,
                 patience: int = 10,
                 min_delta: float = 1e-7,
                 checkpoint_dir: Optional[str] = None):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.mixed_precision = mixed_precision
        self.patience = patience
        self.min_delta = min_delta
        
        # Initialize mixed precision scaler (only on CUDA)
        self.scaler = GradScaler() if (mixed_precision and GradScaler and device.type == 'cuda') else None
        if mixed_precision and device.type != 'cuda':
            warnings.warn(f"Mixed precision not supported on {device.type}, disabling", UserWarning)
            mixed_precision = False
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = 0, verbose: bool = False) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        running_loss = 0.0
        
        import time
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start = time.time()
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            if self.mixed_precision and self.scaler:
                with autocast(device_type=self.device.type):
                    output, _ = self.model(data)
                    # Reshape for sequence-to-token prediction: use last timestep
                    if output.dim() == 3:  # (batch, seq, vocab)
                        output = output[:, -1, :]  # Take last timestep
                    loss = self.criterion(output, target)
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.grad_clip_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard training
                output, _ = self.model(data)
                # Reshape for sequence-to-token prediction: use last timestep
                if output.dim() == 3:  # (batch, seq, vocab)
                    output = output[:, -1, :]  # Take last timestep
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.grad_clip_norm)
                
                self.optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            running_loss += loss_val
            
            # Progress reporting
            if verbose and (batch_idx + 1) % max(1, num_batches // 10) == 0:
                batch_time = time.time() - batch_start
                avg_loss = running_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / num_batches * 100
                
                print(f"  Epoch {epoch+1} [{batch_idx+1:>6}/{num_batches}] "
                      f"({progress:>5.1f}%) | Loss: {loss_val:>7.4f} | "
                      f"Avg: {avg_loss:>7.4f} | {batch_time:>5.3f}s/batch", flush=True)
        
        avg_epoch_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        if verbose:
            print(f"  Epoch {epoch+1} completed in {epoch_time:.2f}s | Avg Loss: {avg_epoch_loss:.6f}")
        
        return avg_epoch_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Additional metrics
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision and self.scaler:
                    with autocast(device_type=self.device.type):
                        output, _ = self.model(data)
                        # Reshape for sequence-to-token prediction: use last timestep
                        if output.dim() == 3:  # (batch, seq, vocab)
                            output = output[:, -1, :]  # Take last timestep
                        loss = self.criterion(output, target)
                else:
                    output, _ = self.model(data)
                    # Reshape for sequence-to-token prediction: use last timestep
                    if output.dim() == 3:  # (batch, seq, vocab)
                        output = output[:, -1, :]  # Take last timestep
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy (for classification tasks)
                pred = output.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total_samples += target.size(0)
        
        avg_loss = total_loss / num_batches
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        return avg_loss, metrics
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """Check if training should stop early"""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stopped = True
                return True
        return False
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return epoch"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.learning_rates = checkpoint['learning_rates']
        
        return checkpoint['epoch']
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Returns:
            Dictionary containing training history
        """
        
        if verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Mixed precision: {self.mixed_precision}")
            print(f"Gradient clipping: {self.grad_clip_norm}")
            print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
            train_loss = self.train_epoch(train_loader, epoch, verbose)
            self.train_losses.append(train_loss)
            
            # Validation
            if verbose:
                print(f"  Validating...")
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Timing
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            
            # Save checkpoint
            if self.checkpoint_dir and (epoch % 10 == 0 or is_best):
                self.save_checkpoint(epoch, is_best)
            
            # Save metrics
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Logging
            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if self.check_early_stopping(val_loss):
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else 0.0,
            'early_stopped': self.early_stopped
        }

def create_optimizer(model: nn.Module, 
                    optimizer_type: str = 'adam',
                    lr: float = 1e-3,
                    weight_decay: float = 1e-4,
                    **kwargs) -> optim.Optimizer:
    """Create optimizer with proper hyperparameters for RNNs"""
    
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                         betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                          betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay,
                            momentum=0.9, alpha=0.99)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                        momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_type: str = 'cosine',
                    num_epochs: int = 100,
                    **kwargs) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    
    if scheduler_type.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                   eta_min=1e-6)
    elif scheduler_type.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.5, patience=5,
                                                   min_lr=1e-6)
    elif scheduler_type.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type.lower() == 'exponential':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")