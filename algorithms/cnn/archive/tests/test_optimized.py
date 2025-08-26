#!/usr/bin/env python3
"""
Optimized CNN Speed Test
Test the speed improvements from all optimizations
"""

import torch
import torch.nn as nn
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from algorithms.cnn.core.models import SimpleCNN, resnet18
from algorithms.cnn.core.dataset import create_image_datasets
from algorithms.cnn.core.optimizations import apply_all_optimizations, benchmark_throughput

# Simple device utils for testing
def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def print_device_info():
    print("🔧 Device Information")
    print("=" * 30)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {'✅' if torch.cuda.is_available() else '❌'}")
    print(f"MPS Available: {'✅' if torch.backends.mps.is_available() else '❌'}")
    print(f"Best Device: {get_best_device()}")
    print()

def test_baseline_performance():
    """Test baseline (unoptimized) performance"""
    print("📊 Testing Baseline Performance")
    print("=" * 40)
    
    device = get_best_device()
    
    # Test SimpleCNN
    model = SimpleCNN(num_classes=10, input_channels=3).to(device)
    
    batch_sizes = [32, 64, 128, 256]
    baseline_results = {}
    
    for batch_size in batch_sizes:
        try:
            throughput = benchmark_throughput(
                model, device, batch_size, 
                input_shape=(3, 32, 32), num_iterations=50
            )
            baseline_results[batch_size] = throughput
            print(f"✅ Batch size {batch_size}: {throughput:,.0f} samples/sec")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Batch size {batch_size}: OOM")
                break
            else:
                print(f"❌ Batch size {batch_size}: Error - {e}")
        finally:
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    return baseline_results

def test_optimized_performance():
    """Test optimized performance"""
    print("\n🚀 Testing Optimized Performance")
    print("=" * 40)
    
    device = get_best_device()
    
    # Test SimpleCNN with all optimizations
    model = SimpleCNN(num_classes=10, input_channels=3)
    
    print("Applying all optimizations...")
    optimized_model, optimal_batch_size = apply_all_optimizations(
        model, device, input_shape=(3, 32, 32), 
        compile_mode='default', find_batch_size=True
    )
    
    # Test with optimal batch size
    optimized_throughput = benchmark_throughput(
        optimized_model, device, optimal_batch_size,
        input_shape=(3, 32, 32), num_iterations=100
    )
    
    return optimal_batch_size, optimized_throughput

def test_data_loading_speed():
    """Test data loading optimizations"""
    print("\n📦 Testing Data Loading Speed")
    print("=" * 40)
    
    # Test old vs new data loading
    print("Creating datasets with optimizations...")
    
    # Optimized dataset
    cifar10_dataset = create_image_datasets('cifar10', augment_train=False)
    
    # Get data loaders with different configurations
    print("Testing data loader speeds...")
    
    # Default optimized loader
    train_loader, _ = cifar10_dataset.get_dataloaders(
        batch_size=128, 
        num_workers=None,  # Auto-detect
        persistent_workers=True
    )
    
    print(f"✅ Optimized data loader created")
    print(f"   - num_workers: {train_loader.num_workers}")
    print(f"   - persistent_workers: {train_loader.persistent_workers}")
    print(f"   - prefetch_factor: {train_loader.prefetch_factor}")
    
    # Time a few batches
    start_time = time.time()
    for i, (data, target) in enumerate(train_loader):
        if i >= 10:  # Test 10 batches
            break
    load_time = time.time() - start_time
    
    samples_loaded = 10 * train_loader.batch_size
    loading_speed = samples_loaded / load_time
    
    print(f"✅ Data loading speed: {loading_speed:,.0f} samples/sec")
    
    return loading_speed

def test_gradient_accumulation():
    """Test gradient accumulation"""
    print("\n🔄 Testing Gradient Accumulation")
    print("=" * 40)
    
    device = get_best_device()
    model = SimpleCNN(num_classes=10, input_channels=3).to(device)
    
    # Create small dataset for testing
    batch_size = 32
    data = torch.randn(batch_size, 3, 32, 32).to(device)
    target = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Test different accumulation steps
    accumulation_steps = [1, 2, 4]
    
    for steps in accumulation_steps:
        model_copy = SimpleCNN(num_classes=10, input_channels=3).to(device)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        # Simulate gradient accumulation
        optimizer.zero_grad()
        for i in range(steps):
            output = model_copy(data)
            loss = criterion(output, target) / steps
            loss.backward()
        optimizer.step()
        
        step_time = time.time() - start_time
        
        print(f"✅ Accumulation steps {steps}: {step_time:.4f}s per step")

def compare_results(baseline_results, optimal_batch_size, optimized_throughput):
    """Compare baseline vs optimized results"""
    print("\n📈 Performance Comparison")
    print("=" * 50)
    
    # Find best baseline result
    best_baseline_batch = max(baseline_results.keys(), key=lambda x: baseline_results[x])
    best_baseline_throughput = baseline_results[best_baseline_batch]
    
    print(f"🔹 Baseline Performance:")
    print(f"   Best batch size: {best_baseline_batch}")
    print(f"   Best throughput: {best_baseline_throughput:,.0f} samples/sec")
    
    print(f"\n🚀 Optimized Performance:")
    print(f"   Optimal batch size: {optimal_batch_size}")
    print(f"   Optimized throughput: {optimized_throughput:,.0f} samples/sec")
    
    if optimized_throughput > best_baseline_throughput:
        speedup = optimized_throughput / best_baseline_throughput
        print(f"\n🎉 SPEEDUP ACHIEVED: {speedup:.1f}x faster!")
    else:
        print(f"\n⚠️ No significant speedup detected")
    
    print("\n📊 All baseline results:")
    for batch_size, throughput in baseline_results.items():
        print(f"   Batch {batch_size}: {throughput:,.0f} samples/sec")

def main():
    """Run all speed tests"""
    print("⚡ CNN Speed Optimization Tests")
    print("=" * 50)
    
    # Print device info
    print_device_info()
    
    try:
        # Test baseline performance
        baseline_results = test_baseline_performance()
        
        # Test optimized performance
        optimal_batch_size, optimized_throughput = test_optimized_performance()
        
        # Test data loading speed
        loading_speed = test_data_loading_speed()
        
        # Test gradient accumulation
        test_gradient_accumulation()
        
        # Compare results
        compare_results(baseline_results, optimal_batch_size, optimized_throughput)
        
        print("\n" + "=" * 50)
        print("✅ All speed tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()