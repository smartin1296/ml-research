#!/usr/bin/env python3
"""
Resume Intelligent CNN Training
Load checkpoint from previous run and continue with intelligent stopping
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

def load_checkpoint_info(checkpoint_path: str) -> Dict:
    """Load and inspect checkpoint"""
    device = get_best_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"📂 Checkpoint Information:", flush=True)
    print(f"   Epoch: {checkpoint['epoch']}", flush=True)
    print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%", flush=True)
    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}", flush=True)
    print(f"   Checkpoint size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB", flush=True)
    
    return checkpoint

def resume_training(checkpoint_path: str, model: nn.Module, model_name: str) -> Dict:
    """Resume training from checkpoint with intelligent stopping"""
    
    device = get_best_device()
    
    # Load checkpoint
    print(f"🔄 Resuming training from checkpoint...", flush=True)
    checkpoint = load_checkpoint_info(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}", flush=True)
    print(f"✅ Previous best validation accuracy: {checkpoint['val_acc']:.2f}%", flush=True)
    print()
    
    # Setup data loaders (same as before)
    batch_size = 128
    lr = 0.001  # Could also load from checkpoint optimizer state
    optimizer_type = 'adam'
    
    dataset = create_image_datasets('cifar10', augment_train=True)
    
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=batch_size,
        num_workers=8 if device.type == 'mps' else 4,
        persistent_workers=True,
        pin_memory=True
    )
    
    # Create test loader
    test_dataset = create_image_datasets('cifar10', augment_train=False)
    _, test_loader = test_dataset.get_dataloaders(
        batch_size=batch_size,
        num_workers=8 if device.type == 'mps' else 4,
        pin_memory=True
    )
    
    print(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches", flush=True)
    print()
    
    # Create results directory for resumed training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('algorithms/cnn/results/resumed')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    run_dir = results_dir / f"{model_name.lower()}_resumed_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = run_dir / 'checkpoints'
    
    # Create intelligent trainer
    trainer = IntelligentCNNTrainer(model, train_loader, val_loader, device)
    
    # Initialize stopping criteria with previous performance
    trainer.stopping_criteria.best_val_acc = checkpoint['val_acc']
    trainer.stopping_criteria.total_epochs = checkpoint['epoch']
    
    print(f"🧠 Intelligent stopping criteria initialized with previous best: {checkpoint['val_acc']:.2f}%", flush=True)
    print(f"🚀 Resuming intelligent training (starting from epoch {checkpoint['epoch'] + 1})...", flush=True)
    print()
    
    # Resume training with intelligent stopping
    results = trainer.train(
        lr=lr,
        optimizer_type=optimizer_type,
        scheduler_type='plateau',
        checkpoint_dir=str(checkpoint_dir),
        save_best=True
    )
    
    # Update results with resume info
    results.update({
        'resumed_from_epoch': checkpoint['epoch'],
        'resumed_from_accuracy': checkpoint['val_acc'],
        'total_epochs_including_previous': results['total_epochs'] + checkpoint['epoch'],
        'resume_checkpoint': checkpoint_path
    })
    
    # Evaluate on test set
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        print(f"\n📂 Loading best model from resumed training...", flush=True)
        final_checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(final_checkpoint['model_state_dict'])
        
        print(f"✅ Final best validation accuracy: {final_checkpoint['val_acc']:.2f}%", flush=True)
        
        test_trainer = IntelligentCNNTrainer(model, train_loader, val_loader, device)
        test_results = test_trainer.evaluate(test_loader)
    else:
        print("⚠️ No new checkpoint saved - using original model for test", flush=True)
        test_trainer = IntelligentCNNTrainer(model, train_loader, val_loader, device)
        test_results = test_trainer.evaluate(test_loader)
    
    # Combine results
    final_results = {
        'model_name': model_name,
        'dataset': 'CIFAR-10',
        'device': str(device),
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'training_type': 'resumed_intelligent_adaptive',
        'original_checkpoint': checkpoint_path,
        'training_results': results,
        'test_results': test_results,
        'summary': {
            'resumed_from_epoch': checkpoint['epoch'],
            'resumed_from_val_acc': checkpoint['val_acc'],
            'final_best_val_accuracy': results['best_val_accuracy'],
            'test_accuracy': test_results['test_accuracy'],
            'additional_training_time': results['total_training_time'],
            'additional_epochs': results['total_epochs'],
            'total_epochs': results['total_epochs'] + checkpoint['epoch'],
            'parameters': results['model_parameters'],
            'intelligent_stopping': True,
            'stopping_reason': results.get('stopping_reason', 'Intelligent convergence')
        }
    }
    
    # Save results
    results_file = run_dir / 'resumed_results.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save summary
    summary_file = run_dir / 'resumed_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"RESUMED Intelligent CNN Training - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: CIFAR-10\n")
        f.write(f"Device: {device}\n")
        f.write(f"Training Type: Resumed Intelligent Adaptive\n")
        f.write(f"Original Checkpoint: {checkpoint_path}\n")
        f.write(f"Resumed from Epoch: {checkpoint['epoch']}\n")
        f.write(f"Resumed from Val Acc: {checkpoint['val_acc']:.2f}%\n")
        f.write(f"Additional Training Time: {results['total_training_time']:.1f}s ({results['total_training_time']/60:.1f}m)\n")
        f.write(f"Additional Epochs: {results['total_epochs']}\n")
        f.write(f"Total Epochs: {results['total_epochs'] + checkpoint['epoch']}\n")
        f.write(f"Final Best Val Accuracy: {results['best_val_accuracy']:.2f}%\n")
        f.write(f"Test Accuracy: {test_results['test_accuracy']:.2f}%\n")
        f.write(f"Improvement: {results['best_val_accuracy'] - checkpoint['val_acc']:.2f}% validation gain\n")
        f.write(f"Stopping Reason: {results.get('stopping_reason', 'Intelligent convergence')}\n")
    
    print(f"\n💾 Resumed training results saved to: {run_dir}", flush=True)
    
    return final_results

def main():
    """Resume intelligent training from previous checkpoint"""
    
    print("🔄 CNN RESUMED INTELLIGENT TRAINING", flush=True)
    print("=" * 60, flush=True)
    
    device = get_best_device()
    print(f"🔧 Device: {device} (PyTorch {torch.__version__})", flush=True)
    print()
    
    # Path to previous checkpoint
    checkpoint_path = "algorithms/cnn/results/standalone/simplecnn_20250826_140322/checkpoints/best_model.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}", flush=True)
        return
    
    try:
        # Create fresh model (same architecture as before)
        model = SimpleCNN(num_classes=10, input_channels=3, base_channels=64)
        
        # Resume training
        results = resume_training(checkpoint_path, model, 'SimpleCNN')
        
        print("\n" + "="*60, flush=True)
        print("🎉 RESUMED INTELLIGENT TRAINING COMPLETE!", flush=True)
        print("="*60, flush=True)
        print(f"✅ {results['model_name']}:", flush=True)
        print(f"   🔄 Resumed from: Epoch {results['summary']['resumed_from_epoch']} ({results['summary']['resumed_from_val_acc']:.2f}%)", flush=True)
        print(f"   📈 Final Val Accuracy: {results['summary']['final_best_val_accuracy']:.2f}%", flush=True)
        print(f"   📊 Improvement: +{results['summary']['final_best_val_accuracy'] - results['summary']['resumed_from_val_acc']:.2f}%", flush=True)
        print(f"   🎯 Test Accuracy: {results['summary']['test_accuracy']:.2f}%", flush=True)
        print(f"   ⏱️ Additional Time: {results['summary']['additional_training_time']:.1f}s ({results['summary']['additional_training_time']/60:.1f}m)", flush=True)
        print(f"   🏁 Additional Epochs: {results['summary']['additional_epochs']}", flush=True)
        print(f"   🏆 Total Epochs: {results['summary']['total_epochs']} (intelligent stopping)", flush=True)
        print(f"   🧠 Stopped: {results['summary']['stopping_reason']}", flush=True)
        
    except Exception as e:
        print(f"\n❌ Resumed training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()