#!/usr/bin/env python3
"""
Basic CNN Test
Simple test to verify CNN implementation works correctly
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
from algorithms.cnn.core.dataset import create_image_datasets, ImageDatasetFactory
from algorithms.cnn.core.trainer import CNNTrainer
from algorithms.cnn.core.device_utils import get_best_device, print_device_info

def test_model_creation():
    """Test that CNN models can be created"""
    print("🔧 Testing model creation...")
    
    # Test SimpleCNN
    simple_cnn = SimpleCNN(num_classes=10, input_channels=3)
    print(f"✅ SimpleCNN created: {sum(p.numel() for p in simple_cnn.parameters()):,} parameters")
    
    # Test ResNet-18
    resnet = resnet18(num_classes=10)
    print(f"✅ ResNet-18 created: {sum(p.numel() for p in resnet.parameters()):,} parameters")
    
    return simple_cnn, resnet

def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n🚀 Testing forward pass...")
    
    device = get_best_device()
    
    # Create models
    simple_cnn = SimpleCNN(num_classes=10, input_channels=3).to(device)
    resnet = resnet18(num_classes=10).to(device)
    
    # Create dummy input (CIFAR-10 size)
    batch_size = 32
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Test SimpleCNN
    start_time = time.time()
    with torch.no_grad():
        output = simple_cnn(dummy_input)
    simple_time = time.time() - start_time
    
    print(f"✅ SimpleCNN forward pass: {output.shape} in {simple_time:.4f}s")
    assert output.shape == (batch_size, 10), f"Expected (32, 10), got {output.shape}"
    
    # Test ResNet-18
    start_time = time.time()
    with torch.no_grad():
        output = resnet(dummy_input)
    resnet_time = time.time() - start_time
    
    print(f"✅ ResNet-18 forward pass: {output.shape} in {resnet_time:.4f}s")
    assert output.shape == (batch_size, 10), f"Expected (32, 10), got {output.shape}"

def test_dataset_creation():
    """Test dataset creation"""
    print("\n📊 Testing dataset creation...")
    
    # Test CIFAR-10 dataset
    cifar10_dataset = create_image_datasets('cifar10', augment_train=False)
    train_loader, test_loader = cifar10_dataset.get_dataloaders(batch_size=32, num_workers=0)
    
    # Get a batch
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    
    print(f"✅ CIFAR-10 train batch: {train_batch[0].shape}, labels: {train_batch[1].shape}")
    print(f"✅ CIFAR-10 test batch: {test_batch[0].shape}, labels: {test_batch[1].shape}")
    
    # Test dataset info
    info = ImageDatasetFactory.get_dataset_info('cifar10')
    print(f"✅ CIFAR-10 info: {info['num_classes']} classes, {info['input_size']} input size")
    
    return train_loader, test_loader

def test_training_setup():
    """Test training setup (without full training)"""
    print("\n🏋️ Testing training setup...")
    
    device = get_best_device()
    
    # Create model and data
    model = SimpleCNN(num_classes=10, input_channels=3)
    cifar10_dataset = create_image_datasets('cifar10', augment_train=False)
    train_loader, val_loader = cifar10_dataset.get_dataloaders(batch_size=32, num_workers=0)
    
    # Create trainer
    trainer = CNNTrainer(model, train_loader, val_loader, device=device)
    
    print(f"✅ Trainer created on device: {trainer.device}")
    print(f"✅ Mixed precision enabled: {trainer.use_mixed_precision}")
    
    # Test single forward/backward pass
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Get a batch
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step completed: loss = {loss.item():.4f}")

def main():
    """Run all tests"""
    print("🧪 Starting Basic CNN Tests")
    print("=" * 50)
    
    # Print device info
    print_device_info()
    print()
    
    try:
        # Run tests
        test_model_creation()
        test_forward_pass()
        test_dataset_creation()
        test_training_setup()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! CNN implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == '__main__':
    main()