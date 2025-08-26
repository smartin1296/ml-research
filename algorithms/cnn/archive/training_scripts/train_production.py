#!/usr/bin/env python3
"""
Production CNN Training
Optimized for M1 Max with all lessons learned applied
"""

import torch
import torch.nn as nn
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from algorithms.cnn.core.models import SimpleCNN, resnet18
from algorithms.cnn.core.dataset import create_image_datasets
from algorithms.cnn.core.trainer import CNNTrainer

# Simple device utils
def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_optimal_settings(model_name: str, device: torch.device) -> dict:
    """Get optimal training settings based on our speed tests"""
    
    base_settings = {
        'device': device,
        'epochs': 100,
        'early_stopping_patience': 15,
        'gradient_accumulation_steps': 1,
        'apply_optimizations': True,
        'compile_model': False,  # Disabled due to MPS issues
    }
    
    if model_name.lower() == 'simplecnn':
        # Optimal settings for SimpleCNN based on tests
        base_settings.update({
            'batch_size': 128,  # Good balance of speed and memory
            'lr': 0.001,
            'optimizer_type': 'adam',
            'scheduler_type': 'cosine',
            'num_workers': 8 if device.type == 'mps' else 4,
        })
    
    elif 'resnet' in model_name.lower():
        # Optimal settings for ResNet
        base_settings.update({
            'batch_size': 128,  # Start conservative for ResNet
            'lr': 0.1,
            'optimizer_type': 'sgd', 
            'scheduler_type': 'cosine',
            'num_workers': 8 if device.type == 'mps' else 4,
        })
    
    return base_settings

def create_results_dir() -> Path:
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('algorithms/cnn/results/production')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    run_dir = results_dir / f"training_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    return run_dir

def train_model(model_name: str, model: nn.Module, run_dir: Path) -> dict:
    """Train a model with optimal settings"""
    
    device = get_best_device()
    settings = get_optimal_settings(model_name, device)
    
    print(f"🚀 Training {model_name} with optimal settings")
    print("=" * 60)
    
    # Print settings
    print("📋 Training Configuration:")
    for key, value in settings.items():
        if key != 'device':
            print(f"   {key}: {value}")
    print(f"   device: {device}")
    print()
    
    # Create dataset with optimal settings
    print("📊 Loading CIFAR-10 dataset...")
    dataset = create_image_datasets('cifar10', augment_train=True)
    
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=settings['batch_size'],
        num_workers=settings['num_workers'],
        persistent_workers=True,
        pin_memory=True
    )
    
    print(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print(f"   Batch size: {settings['batch_size']}")
    print(f"   Workers: {train_loader.num_workers}")
    print()
    
    # Create trainer
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        gradient_accumulation_steps=settings['gradient_accumulation_steps']
    )
    
    # Train model
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    results = trainer.train(
        epochs=settings['epochs'],
        lr=settings['lr'],
        optimizer_type=settings['optimizer_type'],
        scheduler_type=settings['scheduler_type'],
        early_stopping_patience=settings['early_stopping_patience'],
        checkpoint_dir=str(checkpoint_dir),
        save_best=True,
        apply_optimizations=settings['apply_optimizations'],
        compile_model=settings['compile_model']
    )
    
    total_time = time.time() - start_time
    
    # Add metadata to results
    results.update({
        'model_name': model_name,
        'dataset': 'CIFAR-10',
        'total_training_time': total_time,
        'settings': settings,
        'timestamp': datetime.now().isoformat(),
        'device_type': str(device),
        'pytorch_version': torch.__version__
    })
    
    return results

def validate_model(model: nn.Module, checkpoint_path: str, run_dir: Path) -> dict:
    """Load best model and run validation"""
    
    print("\n🎯 Running Final Validation")
    print("=" * 40)
    
    device = get_best_device()
    
    # Load best checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Validation accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Create test dataset
    dataset = create_image_datasets('cifar10', augment_train=False)  # No augmentation for test
    _, test_loader = dataset.get_dataloaders(batch_size=128, num_workers=8)
    
    # Create trainer for evaluation
    dummy_train_loader, val_loader = dataset.get_dataloaders(batch_size=128, num_workers=0)
    trainer = CNNTrainer(model, dummy_train_loader, val_loader, device=device)
    
    # Run evaluation
    test_results = trainer.evaluate(test_loader)
    
    return test_results

def save_results(results: dict, test_results: dict, run_dir: Path):
    """Save all results to files"""
    
    print("\n💾 Saving Results")
    print("=" * 30)
    
    # Combine results
    final_results = {
        'training_results': results,
        'test_results': test_results,
        'summary': {
            'model_name': results['model_name'],
            'final_val_accuracy': results['best_val_accuracy'],
            'test_accuracy': test_results['test_accuracy'],
            'training_time': results['total_training_time'],
            'epochs_trained': results['total_epochs'],
            'parameters': results['model_parameters']
        }
    }
    
    # Save detailed JSON results
    results_file = run_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save summary text file
    summary_file = run_dir / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"CNN Training Results - {results['model_name']}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"Model: {results['model_name']}\n")
        f.write(f"Dataset: CIFAR-10\n")
        f.write(f"Device: {results['device_type']}\n")
        f.write(f"PyTorch: {results['pytorch_version']}\n")
        f.write(f"Training Time: {results['total_training_time']:.1f}s ({results['total_training_time']/60:.1f}m)\n")
        f.write(f"Epochs Trained: {results['total_epochs']}\n")
        f.write(f"Model Parameters: {results['model_parameters']:,}\n\n")
        
        f.write("PERFORMANCE:\n")
        f.write(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_results['test_accuracy']:.2f}%\n")
        f.write(f"Final Training Loss: {results['final_train_loss']:.4f}\n")
        f.write(f"Final Validation Loss: {results['final_val_loss']:.4f}\n\n")
        
        f.write("TRAINING SETTINGS:\n")
        for key, value in results['settings'].items():
            if key != 'device':
                f.write(f"{key}: {value}\n")
    
    print(f"✅ Results saved to: {run_dir}")
    print(f"   - {results_file.name}")
    print(f"   - {summary_file.name}")
    print(f"   - checkpoints/ (best model)")

def main():
    """Run complete training pipeline"""
    
    print("🏋️ CNN Production Training Pipeline")
    print("=" * 60)
    
    # Print system info
    device = get_best_device()
    print(f"🔧 Device: {device} (PyTorch {torch.__version__})")
    print()
    
    # Create results directory
    run_dir = create_results_dir()
    print(f"📁 Results directory: {run_dir}")
    print()
    
    # Choose model for training - start with SimpleCNN
    model_configs = [
        ('SimpleCNN', SimpleCNN(num_classes=10, input_channels=3, base_channels=64)),
        # Add ResNet if you want to train it too
        # ('ResNet-18', resnet18(num_classes=10)),
    ]
    
    all_results = {}
    
    for model_name, model in model_configs:
        try:
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            # Train model
            results = train_model(model_name, model, run_dir)
            
            # Validate model
            best_checkpoint = run_dir / 'checkpoints' / 'best_model.pt'
            if best_checkpoint.exists():
                test_results = validate_model(model, str(best_checkpoint), run_dir)
            else:
                print("⚠️ No checkpoint found, skipping validation")
                test_results = {'test_accuracy': 0.0, 'test_loss': 0.0}
            
            # Save results
            save_results(results, test_results, run_dir)
            
            # Store in summary
            all_results[model_name] = {
                'val_accuracy': results['best_val_accuracy'],
                'test_accuracy': test_results['test_accuracy'],
                'training_time': results['total_training_time'],
                'epochs': results['total_epochs']
            }
            
        except Exception as e:
            print(f"❌ Training failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETE")
    print("="*60)
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"❌ {model_name}: Failed - {results['error']}")
        else:
            print(f"✅ {model_name}:")
            print(f"   📈 Best Val Accuracy: {results['val_accuracy']:.2f}%")
            print(f"   🎯 Test Accuracy: {results['test_accuracy']:.2f}%")
            print(f"   ⏱️ Training Time: {results['training_time']:.1f}s")
            print(f"   🏁 Epochs: {results['epochs']}")
    
    print(f"\n💾 All results saved in: {run_dir}")

if __name__ == '__main__':
    main()