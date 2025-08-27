# Troubleshooting Guide

Common issues and their solutions for the ML Environment.

## Installation Issues

### PyTorch Installation Problems

**Problem**: `pip install torch` fails or takes forever
```bash
# Solution: Use direct PyTorch index
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For NVIDIA CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Problem**: Apple Silicon users getting x86 PyTorch
```bash
# Solution: Ensure you're using ARM64 Python
python -c "import platform; print(platform.machine())"
# Should show: arm64

# If shows x86_64, reinstall Python with Homebrew:
brew install python
```

### Missing Dependencies
**Problem**: `ModuleNotFoundError` for various packages
```bash
# Solution: Install missing scientific computing stack
pip install numpy scipy matplotlib seaborn pandas scikit-learn psutil
pip install tiktoken  # For transformer tokenization
```

## Runtime Issues

### Memory Problems

**Problem**: "CUDA out of memory" or "MPS out of memory"
```bash
# Solution 1: Reduce batch size
python run.py cnn --train --batch-size 64    # Instead of default 128
python run.py rnn --mode token --batch-size 1024  # Instead of default 4096

# Solution 2: Use smaller models
python run.py transformers --phase 1  # Smaller than maximal configs
```

**Problem**: System runs out of RAM
```bash
# Solution: Use CPU with smaller batch sizes
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
python run.py rnn --mode character  # Start with simplest algorithm
```

### Training Issues

**Problem**: Training loss becomes NaN
```bash
# Cause: Learning rate too high
# Solution: Reduce learning rate in the algorithm's config

# For manual debugging:
python -c "
import torch
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available()) 
print('MPS available:', torch.backends.mps.is_available())
"
```

**Problem**: "No gradients flowing" error
```bash
# This usually indicates LayerNorm issues
# Our implementations handle this correctly, but if you modify:

# ✅ Correct LayerNorm usage
nn.LayerNorm(hidden_size)  # elementwise_affine=True by default

# ❌ Wrong - breaks gradients  
nn.LayerNorm(hidden_size, elementwise_affine=False)
```

**Problem**: Training extremely slow
```bash
# Check device usage
python -c "
from algorithms.rnn.core.device_utils import get_best_device
print('Using device:', get_best_device())
"

# Expected outputs:
# Apple Silicon: mps
# NVIDIA GPU: cuda:0  
# CPU fallback: cpu

# If showing 'cpu' unexpectedly, check GPU drivers
```

## Data Issues

### Missing Datasets

**Problem**: "TinyStories not found" or similar dataset errors
```bash
# Solution: Create data directories
mkdir -p data/raw/text/tinystories
mkdir -p data/raw/images

# TinyStories will auto-download on first run
# Or manually download to: data/raw/text/tinystories/TinyStories-small.txt
```

**Problem**: CIFAR-10 download fails
```bash
# Solution: Manual download
python -c "
import torchvision.datasets as datasets
datasets.CIFAR10('data/raw/images', download=True)
"
```

## GPU Acceleration Issues

### Apple Silicon (M1/M2/M3) Problems

**Problem**: MPS not working
```bash
# Check MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# If False, update macOS and Xcode Command Line Tools
sudo xcode-select --install
```

**Problem**: MPS performance slower than expected
```bash
# Ensure you're using optimal batch sizes:
# M1 Max: batch_size=2048 for RNN, 128 for CNN
# M1/M2: batch_size=512 for RNN, 64 for CNN
```

### NVIDIA GPU Problems

**Problem**: CUDA out of memory on large GPU
```bash
# Check memory usage
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Use gradient accumulation instead of large batches
# (This is automatically handled in our implementations)
```

**Problem**: CUDA version mismatch
```bash
# Check CUDA version
nvcc --version
nvidia-smi  # Look at CUDA version in top right

# Install matching PyTorch version
# See: https://pytorch.org/get-started/locally/
```

## Algorithm-Specific Issues

### RNN Problems

**Problem**: Poor text generation quality
```bash
# Solution 1: Train longer
python run.py rnn --mode token --maximal  # Uses advanced training

# Solution 2: Check your text data quality
# Ensure data/raw/text/ contains good quality text files
```

**Problem**: RNN training very slow
```bash
# Check optimal batch sizes for your hardware:
# M1 Max: 2048 (character) or 4096 (token)
# NVIDIA: 32-128 typically optimal  
# CPU: 16-64 maximum

python run.py rnn --mode character  # Start simple
```

### CNN Problems

**Problem**: CNN accuracy much lower than expected
```bash
# Check data augmentation is enabled (it is by default)
# Verify using correct dataset (CIFAR-10 for benchmarking)
# Let intelligent training run to completion - don't stop early
```

**Problem**: CNN training stops too early
```bash
# Our intelligent trainer stops at true convergence
# This is correct behavior! It found the optimal stopping point
# Check results/experiments/latest/ for detailed analysis
```

### Transformer Problems

**Problem**: Transformer generates repetitive text
```bash
# This was debugged and fixed - use current implementations
python run.py transformers --phase 1  # Should work correctly

# If you see repetition, the model may need more training or different sampling
```

**Problem**: Phase 2 performs worse than Phase 1
```bash
# This is expected behavior on small datasets!
# Phase 2 optimizations (label smoothing, etc.) are designed for large-scale training
# Our findings confirm this is theoretically correct
```

## Performance Debugging

### Check Hardware Utilization

```bash
# Monitor GPU usage (NVIDIA)
watch -n 1 nvidia-smi

# Monitor system resources (all platforms)
python -c "
import psutil
print('CPU count:', psutil.cpu_count())
print('Memory:', round(psutil.virtual_memory().total / (1024**3)), 'GB')
"

# Check actual device being used
python -c "
import torch
if torch.backends.mps.is_available():
    print('Using MPS (Apple Silicon)')
elif torch.cuda.is_available():
    print(f'Using CUDA GPU: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU')
"
```

### Benchmark Your System

```bash
# Quick performance test
python run.py rnn --mode character  # Should complete in 2-3 minutes

# Expected performance ranges:
# M1 Max: 5,000+ samples/sec
# NVIDIA RTX 3080+: 3,000-8,000 samples/sec  
# NVIDIA GTX series: 1,000-3,000 samples/sec
# CPU: 20-50 samples/sec
```

## Getting More Help

If none of these solutions work:

1. **Check the algorithm-specific README** in `algorithms/*/README.md`
2. **Look at recent results** in `results/experiments/` to see what worked before  
3. **Try simpler configurations** first, then scale up
4. **Verify your Python environment** - virtual environments can help isolate issues

### Debug Commands
```bash
# Full system check
python run.py --status

# Test imports
python -c "
import torch
import numpy as np  
import pandas as pd
print('All core imports successful!')
"

# Check file structure
ls -la algorithms/*/
ls -la data/raw/
```

Remember: Start simple, then scale up! The character-level RNN is the fastest to debug and verify your setup works.