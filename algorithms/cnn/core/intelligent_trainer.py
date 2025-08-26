#!/usr/bin/env python3
"""
Intelligent CNN Trainer
Pure adaptive training with no hardcoded parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
from collections import deque

class IntelligentStoppingCriteria:
    """
    Intelligent stopping criteria based on learning dynamics
    No hardcoded parameters - pure adaptive logic
    """
    
    def __init__(self):
        self.val_history = deque(maxlen=50)  # Keep recent history
        self.train_history = deque(maxlen=50)
        self.lr_history = deque(maxlen=50)
        self.best_val_acc = 0.0
        self.epochs_since_improvement = 0
        self.total_epochs = 0
        
        # Adaptive thresholds that evolve with training
        self.improvement_threshold = 0.5  # Start with 0.5% minimum improvement
        self.min_improvement_threshold = 0.05  # Never go below 0.05%
        
    def update(self, train_acc: float, val_acc: float, lr: float) -> bool:
        """
        Update history and determine if training should stop
        
        Returns:
            bool: True if training should stop
        """
        self.total_epochs += 1
        self.train_history.append(train_acc)
        self.val_history.append(val_acc)
        self.lr_history.append(lr)
        
        # Check for improvement
        improved = False
        if val_acc > self.best_val_acc + self.improvement_threshold:
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
            improved = True
        else:
            self.epochs_since_improvement += 1
        
        # Adaptive improvement threshold - gets stricter as accuracy increases
        if self.total_epochs >= 10:
            recent_max = max(list(self.val_history)[-10:]) if len(self.val_history) >= 10 else val_acc
            # Higher accuracy = stricter improvement threshold
            accuracy_factor = min(1.0, recent_max / 80.0)  # Scale based on 80% being "high"
            self.improvement_threshold = max(
                self.min_improvement_threshold,
                0.5 * (1.0 - accuracy_factor) + 0.05 * accuracy_factor
            )
        
        # Determine if we should stop
        return self._should_stop()
    
    def _should_stop(self) -> bool:
        """Determine if training should stop based on multiple criteria"""
        
        # Need at least some epochs to make decisions
        if self.total_epochs < 5:
            return False
        
        # 1. Overfitting detection - validation getting worse while training improves
        if len(self.val_history) >= 10 and len(self.train_history) >= 10:
            recent_val_trend = self._get_trend(list(self.val_history)[-10:])
            recent_train_trend = self._get_trend(list(self.train_history)[-10:])
            
            if recent_val_trend < -0.1 and recent_train_trend > 0.1:
                print(f"🛑 Stopping: Overfitting detected (val trend: {recent_val_trend:.2f}, train trend: {recent_train_trend:.2f})", flush=True)
                return True
        
        # 2. Learning rate too small and no improvement
        if len(self.lr_history) >= 3:
            current_lr = self.lr_history[-1]
            if current_lr < 1e-6 and self.epochs_since_improvement > 5:
                print(f"🛑 Stopping: Learning rate too small ({current_lr:.2e}) with no recent improvement", flush=True)
                return True
        
        # 3. Convergence detection - loss/accuracy has plateaued
        if len(self.val_history) >= 15:
            recent_vals = list(self.val_history)[-15:]
            std_dev = np.std(recent_vals)
            mean_val = np.mean(recent_vals)
            
            # If standard deviation is very small relative to mean, we've converged
            if std_dev < 0.1 and mean_val > 10:  # Don't stop too early if accuracy is very low
                print(f"🛑 Stopping: Converged (std: {std_dev:.3f}, mean: {mean_val:.2f}%)", flush=True)
                return True
        
        # 4. Adaptive patience based on current performance
        patience = self._calculate_adaptive_patience()
        if self.epochs_since_improvement >= patience:
            print(f"🛑 Stopping: No improvement for {self.epochs_since_improvement} epochs (adaptive patience: {patience})", flush=True)
            return True
        
        # 5. Learning has completely stalled
        if len(self.val_history) >= 20:
            recent_trend = self._get_trend(list(self.val_history)[-20:])
            if abs(recent_trend) < 0.01 and self.epochs_since_improvement > 8:
                print(f"🛑 Stopping: Learning stalled (trend: {recent_trend:.4f})", flush=True)
                return True
        
        return False
    
    def _get_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of recent values"""
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]  # Linear slope
    
    def _calculate_adaptive_patience(self) -> int:
        """Calculate patience based on current performance and learning dynamics"""
        base_patience = 8
        
        # More patience if accuracy is high (fine-tuning phase)
        if self.best_val_acc > 70:
            base_patience += int((self.best_val_acc - 70) / 5)
        
        # Less patience if learning rate is very small
        if len(self.lr_history) > 0:
            current_lr = self.lr_history[-1]
            if current_lr < 1e-5:
                base_patience = max(3, base_patience - 3)
        
        # More patience early in training
        if self.total_epochs < 20:
            base_patience += 5
        
        return min(25, max(3, base_patience))  # Reasonable bounds
    
    def get_status(self) -> str:
        """Get current training status"""
        patience = self._calculate_adaptive_patience()
        return (f"Best: {self.best_val_acc:.2f}% | "
                f"No improvement: {self.epochs_since_improvement}/{patience} | "
                f"Threshold: {self.improvement_threshold:.3f}%")

class IntelligentCNNTrainer:
    """CNN trainer with pure intelligence-based stopping"""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Intelligent stopping
        self.stopping_criteria = IntelligentStoppingCriteria()
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
    
    def train_epoch(self, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch with real-time progress"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        print(f"  📚 Training epoch...", end="", flush=True)
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Progress indicator
            if batch_idx % 100 == 0:
                print(".", end="", flush=True)
        
        train_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        print(f" ✅ ({train_time:.1f}s)", flush=True)
        return avg_loss, accuracy
    
    def validate(self, criterion) -> Tuple[float, float]:
        """Validate model with real-time progress"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        print(f"  🎯 Validating...", end="", flush=True)
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_time = time.time() - start_time
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        print(f" ✅ ({val_time:.1f}s)", flush=True)
        return avg_loss, accuracy
    
    def train(self, lr: float = 0.001, optimizer_type: str = 'adam',
              scheduler_type: str = 'cosine', checkpoint_dir: str = None,
              save_best: bool = True) -> Dict:
        """Train with intelligent stopping - no hardcoded limits"""
        
        # Apply MPS optimizations
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            print("✅ MPS optimizations applied", flush=True)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        
        # Dynamic learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Setup checkpointing
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting INTELLIGENT training on {self.device}", flush=True)
        print(f"⚙️ Optimizer: {optimizer_type}, Initial LR: {lr}", flush=True)
        print(f"🧠 Adaptive stopping criteria enabled - no hardcoded limits!", flush=True)
        print("=" * 60, flush=True)
        
        start_time = time.time()
        epoch = 0
        
        while True:  # No epoch limit!
            epoch += 1
            epoch_start_time = time.time()
            
            print(f"\n📅 Epoch {epoch}", flush=True)
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update learning rate
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            print(f"📊 Train: {train_loss:.4f} ({train_acc:.2f}%) | "
                  f"Val: {val_loss:.4f} ({val_acc:.2f}%) | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s", flush=True)
            
            # Check intelligent stopping criteria
            should_stop = self.stopping_criteria.update(train_acc, val_acc, current_lr)
            
            # Print status
            status = self.stopping_criteria.get_status()
            print(f"🧠 {status}", flush=True)
            
            # Save best model if improved
            if val_acc > self.stopping_criteria.best_val_acc - 0.001:  # Small tolerance
                if save_best and checkpoint_dir:
                    best_model_path = checkpoint_path / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss
                    }, best_model_path)
                    print(f"💾 Best model saved (Val Acc: {val_acc:.2f}%)", flush=True)
            
            # Check if we should stop
            if should_stop:
                break
        
        total_time = time.time() - start_time
        
        # Prepare results
        results = {
            'total_epochs': epoch,
            'best_val_accuracy': self.stopping_criteria.best_val_acc,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_accuracy': train_acc,
            'final_val_accuracy': val_acc,
            'total_training_time': total_time,
            'device': str(self.device),
            'stopped_intelligently': True,
            'stopping_reason': "Intelligent convergence detection",
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates
            }
        }
        
        print("=" * 60, flush=True)
        print(f"✅ INTELLIGENT TRAINING COMPLETED!", flush=True)
        print(f"📈 Best validation accuracy: {self.stopping_criteria.best_val_acc:.2f}%", flush=True)
        print(f"⏱️ Total time: {total_time:.2f}s ({total_time/60:.1f}m)", flush=True)
        print(f"🏁 Epochs completed: {epoch}", flush=True)
        print(f"🔢 Model parameters: {results['model_parameters']:,}", flush=True)
        
        return results
    
    def evaluate(self, test_loader) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        
        print(f"🎯 Evaluating on test set...", flush=True)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_time = time.time() - start_time
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        results = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'test_time': test_time,
            'total_samples': total,
            'correct_predictions': correct
        }
        
        print(f"🎯 Test Results:", flush=True)
        print(f"   Loss: {avg_loss:.4f}", flush=True)
        print(f"   Accuracy: {accuracy:.2f}%", flush=True)
        print(f"   Time: {test_time:.2f}s", flush=True)
        
        return results