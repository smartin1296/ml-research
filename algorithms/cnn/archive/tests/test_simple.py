#!/usr/bin/env python3
"""
Simple CNN Test - No external dependencies beyond PyTorch
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

def test_training_step():
    """Test a single training step"""
    print("\n🏋️ Testing training step...")
    
    device = get_best_device()
    
    # Create model
    model = SimpleCNN(num_classes=10, input_channels=3).to(device)
    
    # Create dummy data
    batch_size = 16
    data = torch.randn(batch_size, 3, 32, 32).to(device)
    target = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Setup training components
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    pred = output.argmax(dim=1)
    accuracy = (pred == target).float().mean()
    
    print(f"✅ Training step completed:")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Accuracy: {accuracy.item():.2%}")
    
    return loss.item(), accuracy.item()

def test_model_size():
    """Test different model sizes"""
    print("\n📏 Testing model sizes...")
    
    models = [
        ("SimpleCNN (32ch)", SimpleCNN(num_classes=10, input_channels=3, base_channels=32)),
        ("SimpleCNN (64ch)", SimpleCNN(num_classes=10, input_channels=3, base_channels=64)),
        ("ResNet-18", resnet18(num_classes=10)),
    ]
    
    for name, model in models:
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ {name}: {params:,} total, {trainable:,} trainable")

def main():
    """Run all tests"""
    print("🧪 Simple CNN Tests")
    print("=" * 40)
    
    # Print device info
    print_device_info()
    print()
    
    try:
        # Run tests
        test_model_creation()
        test_forward_pass()
        test_training_step()
        test_model_size()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! CNN implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()