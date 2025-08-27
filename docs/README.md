# ML Environment Documentation

Welcome to the Machine Learning Research Environment - a comprehensive platform for neural network experimentation and benchmarking.

## Quick Start

```bash
# Check what's available
python run.py --status

# Run algorithms
python run.py rnn --mode character      # Character-level RNN
python run.py cnn --train              # CNN training
python run.py transformers --phase 1   # Transformer baseline
```

## Documentation Structure

### 📚 **Core Documentation**
- [Getting Started Guide](getting-started.md) - Setup and first experiments
- [API Reference](api/) - Complete API documentation
- [Training Guides](guides/training.md) - Best practices and optimization tips
- [Hardware Guide](guides/hardware.md) - GPU acceleration setup
- [Troubleshooting](guides/troubleshooting.md) - Common issues and solutions

### 🧠 **Algorithm Documentation**
- [RNN Module](../algorithms/rnn/README.md) - Character and token-level RNNs
- [CNN Module](../algorithms/cnn/README.md) - Convolutional networks for images
- [Transformers](../algorithms/transformers/README.md) - Attention-based architectures

### 💡 **Examples and Tutorials**
- [Code Examples](examples/) - Working code snippets
- [Research Usage](guides/research.md) - Using for academic research

## Current Implementation Status

| Algorithm | Status | Best Performance | Usage |
|-----------|--------|------------------|--------|
| **RNN** | ✅ Complete | 39%+ token accuracy, 5.7K samples/sec | `python run.py rnn --mode token` |
| **CNN** | ✅ Complete | 86.15% CIFAR-10 accuracy | `python run.py cnn --train` |
| **Transformers** | ✅ Complete | 99.9% validation accuracy | `python run.py transformers --phase 1` |
| **Reasoning NNs** | 🔄 In Progress | - | Coming soon |
| **RCL Algorithm** | 📋 Planned | - | Future release |

## Key Features

- **🎯 Easy Interface**: Single `python run.py` command for everything
- **🚀 Optimized Performance**: M1 Max and CUDA acceleration
- **📊 Comprehensive Benchmarking**: Statistical analysis and comparison
- **🔬 Research Ready**: Clean code, proper documentation, reproducible results
- **⚡ Intelligent Training**: Adaptive stopping, no hardcoded parameters

## Getting Help

- **First time?** Start with [Getting Started Guide](getting-started.md)
- **Having issues?** Check [Troubleshooting](guides/troubleshooting.md)
- **Want to contribute?** See [Contribution Guidelines](guides/contributing.md)
- **Research usage?** Read [Research Guide](guides/research.md)

---

**Environment**: PyTorch 2.0+, Python 3.8+, Cross-platform (macOS/Linux/Windows)  
**Hardware**: Optimized for Apple Silicon (M1/M2/M3) and NVIDIA GPUs, CPU fallback available