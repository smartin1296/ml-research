# Getting Started

This guide will get you running your first ML experiments in under 5 minutes.

## Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **8GB+ RAM** (16GB+ recommended for larger models)
- **GPU** (optional but recommended): Apple Silicon M1/M2/M3 or NVIDIA GPU

## Installation

### 1. Clone and Setup
```bash
cd your-workspace
git clone <repository-url>
cd lumina-research
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python run.py --status
```

You should see a status overview of all available algorithms.

## Your First Experiment

### Option A: Quick RNN Test (2 minutes)
```bash
# Character-level RNN on Shakespeare text
python run.py rnn --mode character

# Expected output: ~60% accuracy, very fast training
```

### Option B: CNN Image Classification (5 minutes)
```bash  
# Train CNN on CIFAR-10
python run.py cnn --train

# Expected output: 85%+ accuracy, intelligent stopping
```

### Option C: Transformer Language Model (3 minutes)
```bash
# Baseline transformer on TinyStories
python run.py transformers --phase 1

# Expected output: 99%+ validation accuracy
```

## Understanding the Output

Each experiment creates results in the following structure:
```
results/
├── experiments/        # Timestamped experiment runs
├── benchmarks/        # Performance comparisons  
└── comparisons/       # Algorithm comparisons
```

### Typical Output Format
```
🚀 Starting RNN Character Training...
📊 Dataset: 1.1M characters, vocab_size=67
🧠 Model: 2.1M parameters (2L-384H LSTM)
📈 Training Progress:
Epoch  1: Train Loss=2.456, Val Loss=2.234, Val Acc=0.421
Epoch  2: Train Loss=1.987, Val Loss=1.876, Val Acc=0.523
...
✅ Training complete! Best accuracy: 59.67%
💾 Results saved to: results/experiments/rnn_20250826_123456/
```

## Next Steps

### Explore Different Configurations
```bash
# Try token-level RNN for better language modeling
python run.py rnn --mode token --maximal

# Compare transformer phases
python run.py transformers --comparison

# Resume interrupted CNN training
python run.py cnn --resume
```

### Understanding Your Hardware

Check what acceleration is available:
```bash
# See device capabilities
python -c "from algorithms.rnn.core.device_utils import *; print(f'Device: {get_best_device()}')"

# Apple Silicon users: You have MPS acceleration! 🚀
# NVIDIA users: You have CUDA acceleration! ⚡
# Others: CPU training (slower but works everywhere) 💻
```

### Performance Expectations

| Hardware | RNN Character | CNN CIFAR-10 | Transformer |
|----------|---------------|--------------|-------------|
| **M1 Max** | 5.7K samples/sec | 9.5K samples/sec | Very fast |
| **NVIDIA RTX** | 3-8K samples/sec | 8-15K samples/sec | Very fast |
| **CPU** | 32 samples/sec | 200 samples/sec | Slow |

## Common First-Time Issues

### Installation Issues
```bash
# If PyTorch installation fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# If you have CUDA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```bash
# If you get CUDA out of memory
python run.py cnn --train --batch-size 64   # Reduce batch size

# If general memory issues  
python run.py rnn --mode character  # Start with simpler model
```

### Data Issues
```bash
# If missing TinyStories dataset
# It will be automatically downloaded on first run
# Or manually download to: data/raw/text/tinystories/
```

## What to Try Next

1. **Experiment with hyperparameters** - Edit the run scripts to try different learning rates, batch sizes
2. **Compare algorithms** - Use the comparison modes to see which works best for your use case
3. **Read algorithm docs** - Each algorithm has detailed documentation in `algorithms/*/README.md`
4. **Check out advanced guides** - See `docs/guides/` for optimization tips

## Getting Help

- **Stuck?** Check [Troubleshooting Guide](guides/troubleshooting.md)
- **Want to optimize?** See [Hardware Guide](guides/hardware.md)
- **Research usage?** Read [Research Guide](guides/research.md)

---

**Ready to dive deeper?** Check out the [full documentation](README.md) or jump into algorithm-specific guides!