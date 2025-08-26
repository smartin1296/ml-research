#!/usr/bin/env python3
"""
Simple CNN Speed Test
Test basic optimizations without torch.compile (which has MPS issues)
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

def benchmark_model(model, device, batch_size, num_iterations=100, warmup=10):
    """Simple benchmark function"""
    model.eval()
    input_shape = (batch_size, 3, 32, 32)
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize and time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    total_samples = num_iterations * batch_size
    throughput = total_samples / total_time
    
    return throughput

def test_baseline_vs_optimized():
    """Compare baseline vs optimized performance"""
    print("⚡ Baseline vs Optimized Performance Test")
    print("=" * 50)
    
    device = get_best_device()
    print_device_info()
    
    # Create models
    baseline_model = SimpleCNN(num_classes=10, input_channels=3).to(device)
    optimized_model = SimpleCNN(num_classes=10, input_channels=3).to(device)
    
    # Apply basic optimizations to optimized model (no compilation)
    print("🚀 Applying basic optimizations...")
    
    # 1. Channels-last memory format
    try:
        optimized_model = optimized_model.to(memory_format=torch.channels_last)
        print("✅ Channels-last memory format applied")
        channels_last_success = True
    except Exception as e:
        print(f"⚠️ Channels-last failed: {e}")
        channels_last_success = False
    
    # 2. MPS optimizations  
    if device.type == 'mps':
        torch.mps.empty_cache()
        import os
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        print("✅ MPS optimizations applied")
    
    print()
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    
    print("Testing baseline model:")
    baseline_results = {}
    
    for batch_size in batch_sizes:
        try:
            throughput = benchmark_model(baseline_model, device, batch_size)
            baseline_results[batch_size] = throughput
            print(f"  Batch {batch_size:3d}: {throughput:8,.0f} samples/sec")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch {batch_size:3d}: OOM")
                break
            else:
                print(f"  Batch {batch_size:3d}: Error")
        finally:
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    print("\nTesting optimized model:")
    optimized_results = {}
    
    for batch_size in batch_sizes:
        try:
            # Create input with channels_last if supported
            input_shape = (batch_size, 3, 32, 32)
            dummy_input = torch.randn(*input_shape).to(device)
            
            if channels_last_success:
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            
            # Custom benchmark for channels_last
            optimized_model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = optimized_model(dummy_input)
            
            # Time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = optimized_model(dummy_input)
            end_time = time.time()
            
            total_time = end_time - start_time
            total_samples = 100 * batch_size
            throughput = total_samples / total_time
            
            optimized_results[batch_size] = throughput
            print(f"  Batch {batch_size:3d}: {throughput:8,.0f} samples/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch {batch_size:3d}: OOM")
                break
            else:
                print(f"  Batch {batch_size:3d}: Error")
        finally:
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    # Compare results
    print("\n📊 Performance Comparison:")
    print("=" * 50)
    
    improvements = {}
    for batch_size in baseline_results:
        if batch_size in optimized_results:
            baseline = baseline_results[batch_size]
            optimized = optimized_results[batch_size]
            improvement = optimized / baseline
            improvements[batch_size] = improvement
            
            print(f"Batch {batch_size:3d}: {baseline:8,.0f} → {optimized:8,.0f} samples/sec ({improvement:.2f}x)")
    
    if improvements:
        avg_improvement = sum(improvements.values()) / len(improvements)
        best_improvement = max(improvements.values())
        
        print(f"\n🎉 Results:")
        print(f"   Average speedup: {avg_improvement:.2f}x")
        print(f"   Best speedup: {best_improvement:.2f}x")
    
    return baseline_results, optimized_results

def test_data_loading():
    """Test data loading optimizations"""
    print("\n📦 Data Loading Performance Test")
    print("=" * 40)
    
    # Test with different configurations
    configs = [
        ("Basic", {"num_workers": 0, "persistent_workers": False}),
        ("Optimized", {"num_workers": None, "persistent_workers": True})
    ]
    
    for name, kwargs in configs:
        print(f"\nTesting {name} configuration...")
        
        try:
            dataset = create_image_datasets('cifar10', augment_train=False)
            train_loader, _ = dataset.get_dataloaders(batch_size=128, **kwargs)
            
            print(f"  - num_workers: {train_loader.num_workers}")
            print(f"  - persistent_workers: {getattr(train_loader, 'persistent_workers', 'N/A')}")
            
            # Time loading a few batches
            start_time = time.time()
            for i, (data, target) in enumerate(train_loader):
                if i >= 5:  # Test 5 batches
                    break
            load_time = time.time() - start_time
            
            samples_loaded = 5 * train_loader.batch_size
            loading_speed = samples_loaded / load_time
            
            print(f"  ✅ Loading speed: {loading_speed:,.0f} samples/sec")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def main():
    """Run all tests"""
    try:
        # Test model performance 
        test_baseline_vs_optimized()
        
        # Test data loading
        test_data_loading()
        
        print("\n" + "=" * 50)
        print("✅ All speed tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()