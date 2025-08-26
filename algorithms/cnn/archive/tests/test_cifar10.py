#!/usr/bin/env python3
"""
CIFAR-10 CNN Benchmark
Comprehensive CIFAR-10 training and benchmarking for CNN models
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

from algorithms.cnn.core.models import SimpleCNN, resnet18, resnet34
from algorithms.cnn.core.dataset import create_image_datasets, ImageDatasetFactory
from algorithms.cnn.core.trainer import CNNTrainer
from algorithms.cnn.core.device_utils import get_best_device, print_device_info
from utils.benchmarking import ModelBenchmark

def benchmark_model(model_name: str, model: nn.Module, train_loader, val_loader, 
                   device, epochs: int = 20) -> dict:
    """
    Benchmark a CNN model on CIFAR-10
    
    Args:
        model_name: Name of the model for logging
        model: PyTorch model to benchmark
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of epochs to train
        
    Returns:
        Dictionary of benchmark results
    """
    
    print(f"\n{'='*60}")
    print(f"🏋️ Benchmarking {model_name}")
    print(f"{'='*60}")
    
    # Create trainer
    trainer = CNNTrainer(model, train_loader, val_loader, device=device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Determine optimal learning rate based on model
    if 'resnet' in model_name.lower():
        lr = 0.1
        optimizer_type = 'sgd'
        scheduler_type = 'cosine'
    else:
        lr = 0.001
        optimizer_type = 'adam'
        scheduler_type = 'cosine'
    
    print(f"⚙️ Training config: {optimizer_type.upper()} optimizer, {scheduler_type} scheduler, LR={lr}")
    
    # Train model
    start_time = time.time()
    results = trainer.train(
        epochs=epochs,
        lr=lr,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        early_stopping_patience=8,
        checkpoint_dir=f'algorithms/cnn/checkpoints/{model_name.lower()}',
        save_best=True
    )
    total_time = time.time() - start_time
    
    # Add model-specific info to results
    results.update({
        'model_name': model_name,
        'model_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_benchmark_time': total_time,
        'optimizer': optimizer_type,
        'scheduler': scheduler_type,
        'learning_rate': lr
    })
    
    print(f"\n✅ {model_name} benchmark completed!")
    print(f"📈 Best accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    return results

def run_cifar10_benchmark():
    """Run comprehensive CIFAR-10 benchmark"""
    
    print("🎯 CIFAR-10 CNN Benchmark Suite")
    print("=" * 70)
    
    # Print device info
    print_device_info()
    print()
    
    # Setup
    device = get_best_device()
    batch_size = 128  # Good balance for most devices
    epochs = 50  # Reasonable for quick benchmarking
    
    # Create dataset
    print("📊 Loading CIFAR-10 dataset...")
    cifar10_dataset = create_image_datasets('cifar10', augment_train=True)
    train_loader, val_loader = cifar10_dataset.get_dataloaders(
        batch_size=batch_size, 
        num_workers=2,  # Reduced for stability
        pin_memory=True
    )
    
    print(f"✅ Dataset loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Models to benchmark
    models_to_test = [
        ('SimpleCNN', SimpleCNN(num_classes=10, input_channels=3, base_channels=64)),
        ('ResNet-18', resnet18(num_classes=10)),
        # Uncomment for more comprehensive benchmarking
        # ('ResNet-34', resnet34(num_classes=10)),
    ]
    
    # Run benchmarks
    all_results = {}
    
    for model_name, model in models_to_test:
        try:
            results = benchmark_model(
                model_name, model, train_loader, val_loader, 
                device, epochs=epochs
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ Failed to benchmark {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('algorithms/cnn/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f'cifar10_benchmark_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 CIFAR-10 Benchmark Results Summary")
    print(f"{'='*70}")
    
    # Print summary
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"❌ {model_name}: Failed ({results['error']})")
        else:
            print(f"✅ {model_name}:")
            print(f"   📈 Best Accuracy: {results['best_val_accuracy']:.2f}%")
            print(f"   ⏱️ Training Time: {results.get('total_training_time', 0):.1f}s")
            print(f"   🔢 Parameters: {results['model_parameters']:,}")
            print(f"   🏁 Epochs: {results['total_epochs']}")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return all_results

def quick_test():
    """Quick test with minimal epochs for verification"""
    print("⚡ Quick CNN Test (5 epochs)")
    print("=" * 40)
    
    device = get_best_device()
    
    # Simple setup
    model = SimpleCNN(num_classes=10, input_channels=3, base_channels=32)  # Smaller model
    cifar10_dataset = create_image_datasets('cifar10', augment_train=False)  # No augmentation
    train_loader, val_loader = cifar10_dataset.get_dataloaders(
        batch_size=64,  # Smaller batch
        num_workers=0   # No multiprocessing for stability
    )
    
    # Quick benchmark
    results = benchmark_model(
        'SimpleCNN-Quick', model, train_loader, val_loader, 
        device, epochs=5
    )
    
    print(f"\n⚡ Quick test completed!")
    print(f"📈 Final accuracy: {results.get('final_val_accuracy', 0):.2f}%")
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick test (5 epochs)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for full benchmark')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            quick_test()
        else:
            run_cifar10_benchmark()
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        raise

if __name__ == '__main__':
    main()