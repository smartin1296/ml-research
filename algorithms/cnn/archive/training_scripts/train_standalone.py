#!/usr/bin/env python3
"""
Standalone CNN Training
No external dependencies beyond PyTorch - optimized for M1 Max
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from algorithms.cnn.core.models import SimpleCNN, resnet18
from algorithms.cnn.core.dataset import create_image_datasets

def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class StandaloneCNNTrainer:
    """Simplified CNN trainer without external dependencies"""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
    
    def train_epoch(self, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
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
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, criterion) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int = 100, lr: float = 0.001, optimizer_type: str = 'adam',
              early_stopping_patience: int = 15, checkpoint_dir: str = None,
              save_best: bool = True) -> Dict:
        """Train the model"""
        
        # Apply MPS optimizations
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            print("✅ MPS optimizations applied")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Setup checkpointing
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting training on {self.device}")
        print(f"⚙️ Optimizer: {optimizer_type}, LR: {lr}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Record metrics
            current_lr = optimizer.param_groups[0]['lr']
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
                  f"Val: {val_loss:.4f} ({val_acc:.2f}%) | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                
                if save_best and checkpoint_dir:
                    best_model_path = checkpoint_path / 'best_model.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss
                    }, best_model_path)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"🛑 Early stopping after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                break
        
        total_time = time.time() - start_time
        
        # Prepare results
        results = {
            'total_epochs': epoch + 1,
            'best_val_accuracy': self.best_val_acc,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_accuracy': train_acc,
            'final_val_accuracy': val_acc,
            'total_training_time': total_time,
            'device': str(self.device),
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
        
        print("=" * 60)
        print(f"✅ Training completed!")
        print(f"📈 Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"🔢 Model parameters: {results['model_parameters']:,}")
        
        return results
    
    def evaluate(self, test_loader) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        
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
        
        print(f"🎯 Test Results:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Time: {test_time:.2f}s")
        
        return results

def train_model_complete(model_name: str, model: nn.Module) -> Dict:
    """Complete training pipeline"""
    
    device = get_best_device()
    
    # Optimal settings based on our tests
    if model_name.lower() == 'simplecnn':
        batch_size = 128
        lr = 0.001
        optimizer_type = 'adam'
        epochs = 100
    elif 'resnet' in model_name.lower():
        batch_size = 128 
        lr = 0.1
        optimizer_type = 'sgd'
        epochs = 100
    else:
        batch_size = 128
        lr = 0.001
        optimizer_type = 'adam'
        epochs = 100
    
    print(f"🚀 Training {model_name}")
    print(f"📊 Settings: batch_size={batch_size}, lr={lr}, optimizer={optimizer_type}")
    print()
    
    # Create dataset
    dataset = create_image_datasets('cifar10', augment_train=True)
    
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=batch_size,
        num_workers=8 if device.type == 'mps' else 4,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Create test loader for final evaluation
    test_dataset = create_image_datasets('cifar10', augment_train=False)
    _, test_loader = test_dataset.get_dataloaders(
        batch_size=batch_size,
        num_workers=8 if device.type == 'mps' else 4,
        pin_memory=True
    )
    
    print(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('algorithms/cnn/results/standalone')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    run_dir = results_dir / f"{model_name.lower()}_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = run_dir / 'checkpoints'
    
    # Train model
    trainer = StandaloneCNNTrainer(model, train_loader, val_loader, device)
    
    results = trainer.train(
        epochs=epochs,
        lr=lr,
        optimizer_type=optimizer_type,
        early_stopping_patience=15,
        checkpoint_dir=str(checkpoint_dir),
        save_best=True
    )
    
    # Load best model and evaluate on test set
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        print(f"\n📂 Loading best checkpoint...")
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
        print(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        # Test evaluation
        test_trainer = StandaloneCNNTrainer(model, train_loader, val_loader, device)
        test_results = test_trainer.evaluate(test_loader)
    else:
        print("⚠️ No best checkpoint found")
        test_results = {'test_accuracy': 0.0, 'test_loss': 0.0}
    
    # Combine results
    final_results = {
        'model_name': model_name,
        'dataset': 'CIFAR-10',
        'device': str(device),
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'training_results': results,
        'test_results': test_results,
        'summary': {
            'best_val_accuracy': results['best_val_accuracy'],
            'test_accuracy': test_results['test_accuracy'],
            'training_time': results['total_training_time'],
            'epochs_trained': results['total_epochs'],
            'parameters': results['model_parameters']
        }
    }
    
    # Save results
    results_file = run_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save summary
    summary_file = run_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"CNN Training Results - {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: CIFAR-10\n")
        f.write(f"Device: {device}\n")
        f.write(f"Training Time: {results['total_training_time']:.1f}s ({results['total_training_time']/60:.1f}m)\n")
        f.write(f"Epochs: {results['total_epochs']}\n")
        f.write(f"Parameters: {results['model_parameters']:,}\n")
        f.write(f"Best Val Accuracy: {results['best_val_accuracy']:.2f}%\n")
        f.write(f"Test Accuracy: {test_results['test_accuracy']:.2f}%\n")
    
    print(f"\n💾 Results saved to: {run_dir}")
    
    return final_results

def main():
    """Run complete training"""
    
    print("🏋️ CNN Standalone Training")
    print("=" * 50)
    
    device = get_best_device()
    print(f"🔧 Device: {device} (PyTorch {torch.__version__})")
    print()
    
    try:
        # Train SimpleCNN
        model = SimpleCNN(num_classes=10, input_channels=3, base_channels=64)
        results = train_model_complete('SimpleCNN', model)
        
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETE!")
        print("="*50)
        print(f"✅ {results['model_name']}:")
        print(f"   📈 Best Val Accuracy: {results['summary']['best_val_accuracy']:.2f}%")
        print(f"   🎯 Test Accuracy: {results['summary']['test_accuracy']:.2f}%")
        print(f"   ⏱️ Training Time: {results['summary']['training_time']:.1f}s")
        print(f"   🏁 Epochs: {results['summary']['epochs_trained']}")
        print(f"   🔢 Parameters: {results['summary']['parameters']:,}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()