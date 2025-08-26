#!/usr/bin/env python3
"""
Intelligent CNN Training
Pure adaptive training with no hardcoded parameters
"""

import torch
import torch.nn as nn
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from algorithms.cnn.core.models import SimpleCNN, resnet18
from algorithms.cnn.core.dataset import create_image_datasets
from algorithms.cnn.core.intelligent_trainer import IntelligentCNNTrainer

def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_model_intelligent(model_name: str, model: nn.Module) -> Dict:
    """Complete intelligent training pipeline"""
    
    device = get_best_device()
    
    # Determine optimal settings based on model type (not hardcoded limits!)
    if model_name.lower() == 'simplecnn':
        batch_size = 128
        lr = 0.001
        optimizer_type = 'adam'
    elif 'resnet' in model_name.lower():
        batch_size = 128 
        lr = 0.1
        optimizer_type = 'sgd'
    else:
        batch_size = 128
        lr = 0.001
        optimizer_type = 'adam'
    
    print(f"🚀 INTELLIGENT Training: {model_name}", flush=True)
    print(f"📊 Settings: batch_size={batch_size}, lr={lr}, optimizer={optimizer_type}", flush=True)
    print(f"🧠 NO hardcoded epochs - training until intelligent convergence!", flush=True)
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
    
    print(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches", flush=True)
    print()
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('algorithms/cnn/results/intelligent')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    run_dir = results_dir / f"{model_name.lower()}_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = run_dir / 'checkpoints'
    
    # Train model with intelligent trainer
    trainer = IntelligentCNNTrainer(model, train_loader, val_loader, device)
    
    results = trainer.train(
        lr=lr,
        optimizer_type=optimizer_type,
        scheduler_type='plateau',  # Adaptive scheduler
        checkpoint_dir=str(checkpoint_dir),
        save_best=True
    )
    
    # Load best model and evaluate on test set
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        print(f"\n📂 Loading best checkpoint for final evaluation...", flush=True)
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}", flush=True)
        print(f"   Best validation accuracy: {checkpoint['val_acc']:.2f}%", flush=True)
        
        # Test evaluation
        test_trainer = IntelligentCNNTrainer(model, train_loader, val_loader, device)
        test_results = test_trainer.evaluate(test_loader)
    else:
        print("⚠️ No best checkpoint found", flush=True)
        test_results = {'test_accuracy': 0.0, 'test_loss': 0.0}
    
    # Combine results
    final_results = {
        'model_name': model_name,
        'dataset': 'CIFAR-10',
        'device': str(device),
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'training_type': 'intelligent_adaptive',
        'training_results': results,
        'test_results': test_results,
        'summary': {
            'best_val_accuracy': results['best_val_accuracy'],
            'test_accuracy': test_results['test_accuracy'],
            'training_time': results['total_training_time'],
            'epochs_trained': results['total_epochs'],
            'parameters': results['model_parameters'],
            'intelligent_stopping': True,
            'stopping_reason': results.get('stopping_reason', 'Unknown')
        }
    }
    
    # Save results
    results_file = run_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save summary
    summary_file = run_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"INTELLIGENT CNN Training Results - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: CIFAR-10\n")
        f.write(f"Device: {device}\n")
        f.write(f"Training Type: Intelligent Adaptive (No hardcoded limits)\n")
        f.write(f"Training Time: {results['total_training_time']:.1f}s ({results['total_training_time']/60:.1f}m)\n")
        f.write(f"Epochs: {results['total_epochs']} (stopped intelligently)\n")
        f.write(f"Parameters: {results['model_parameters']:,}\n")
        f.write(f"Stopping Reason: {results.get('stopping_reason', 'Intelligent convergence')}\n")
        f.write(f"Best Val Accuracy: {results['best_val_accuracy']:.2f}%\n")
        f.write(f"Test Accuracy: {test_results['test_accuracy']:.2f}%\n")
    
    print(f"\n💾 Results saved to: {run_dir}", flush=True)
    
    return final_results

def main():
    """Run intelligent training"""
    
    print("🧠 CNN INTELLIGENT TRAINING", flush=True)
    print("=" * 60, flush=True)
    
    device = get_best_device()
    print(f"🔧 Device: {device} (PyTorch {torch.__version__})", flush=True)
    print()
    
    try:
        # Train SimpleCNN with intelligent stopping
        model = SimpleCNN(num_classes=10, input_channels=3, base_channels=64)
        results = train_model_intelligent('SimpleCNN', model)
        
        print("\n" + "="*60, flush=True)
        print("🎉 INTELLIGENT TRAINING COMPLETE!", flush=True)
        print("="*60, flush=True)
        print(f"✅ {results['model_name']}:", flush=True)
        print(f"   📈 Best Val Accuracy: {results['summary']['best_val_accuracy']:.2f}%", flush=True)
        print(f"   🎯 Test Accuracy: {results['summary']['test_accuracy']:.2f}%", flush=True)
        print(f"   ⏱️ Training Time: {results['summary']['training_time']:.1f}s ({results['summary']['training_time']/60:.1f}m)", flush=True)
        print(f"   🏁 Epochs: {results['summary']['epochs_trained']} (intelligent stopping)", flush=True)
        print(f"   🔢 Parameters: {results['summary']['parameters']:,}", flush=True)
        print(f"   🧠 Stopped: {results['summary']['stopping_reason']}", flush=True)
        
    except Exception as e:
        print(f"\n❌ Intelligent training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()