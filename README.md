# ML Environment

Modern machine learning research platform with state-of-the-art neural network implementations, optimized for Apple Silicon and NVIDIA GPUs.

## Quick Start

```bash
# Install and check status
pip install -r requirements.txt
python run.py --status

# Run your first experiment (2 minutes)
python run.py rnn --mode character
```

## 🎯 What's Included

| Algorithm | Status | Best Performance | Quick Start |
|-----------|--------|------------------|-------------|
| **RNN** | ✅ Production | 39%+ token accuracy, 5.7K samples/sec | `python run.py rnn --mode token` |
| **CNN** | ✅ Production | 86.15% CIFAR-10 accuracy | `python run.py cnn --train` |
| **Transformers** | ✅ Production | 99.9% validation accuracy | `python run.py transformers --phase 1` |
| **Reasoning NNs** | 🔄 In Progress | Test-time compute models | Coming soon |

## 🚀 Key Features

- **Single Command Interface**: `python run.py algorithm --options`
- **Intelligent Training**: Adaptive stopping, no hardcoded parameters  
- **Hardware Optimized**: M1 Max (5.7K samples/sec), NVIDIA CUDA, CPU fallback
- **Research Ready**: Proper benchmarking, statistical analysis, reproducible results

## 📁 Project Structure

```
├── algorithms/          # Neural network implementations
│   ├── rnn/            # Character & token-level RNNs
│   ├── cnn/            # Image classification CNNs  
│   ├── transformers/   # Language transformers
│   └── run.py          # Unified entry point per algorithm
├── data/               # Datasets (auto-downloaded)
├── docs/               # Complete documentation
├── results/            # Experiment results
├── utils/              # Benchmarking & analysis tools
└── run.py              # Main entry point
```

## 🛠️ Hardware Support

- **🍎 Apple Silicon** (M1/M2/M3): MPS acceleration, unified memory optimization
- **🟢 NVIDIA GPUs**: CUDA with mixed precision training  
- **💻 CPU**: Universal fallback with optimized batch sizes

## 📚 Documentation

- **[Getting Started](docs/getting-started.md)** - 5-minute setup guide
- **[Full Documentation](docs/)** - Complete reference  
- **[Troubleshooting](docs/guides/troubleshooting.md)** - Common issues
- **[Hardware Guide](docs/guides/hardware.md)** - GPU optimization

## 💡 Usage Examples

```bash
# Character-level language modeling (fastest)
python run.py rnn --mode character

# Token-level with maximal accuracy training
python run.py rnn --mode token --maximal

# Intelligent CNN training on CIFAR-10
python run.py cnn --train

# Transformer baseline vs optimized comparison  
python run.py transformers --comparison

# Resume interrupted training
python run.py cnn --resume
```

## 🧪 Research Features

- **Comprehensive Benchmarking**: Statistical analysis with confidence intervals
- **Modular Architecture**: Clean, extensible implementations
- **Industry Standards**: PyTorch, NumPy, scikit-learn only
- **Cross-Platform**: Works on macOS, Linux, Windows
- **Reproducible**: Fixed seeds, deterministic algorithms

## ⚡ Performance Expectations

| Hardware | RNN (Character) | CNN (CIFAR-10) | Transformer |
|----------|-----------------|----------------|-------------|
| **M1 Max** | 5,720 samples/sec | 9,500 samples/sec | Very fast |
| **RTX 4090** | ~8,000 samples/sec | ~15,000 samples/sec | Very fast |
| **CPU** | 32 samples/sec | 200 samples/sec | Slow |

## 📋 Requirements

- **Python 3.8+** (3.9+ recommended)
- **PyTorch 2.0+** 
- **8GB+ RAM** (16GB+ recommended)
- **GPU** optional but recommended

---

**New here?** Start with the [Getting Started Guide](docs/getting-started.md) for a 5-minute walkthrough.

**Having issues?** Check the [Troubleshooting Guide](docs/guides/troubleshooting.md).

**Research usage?** See [full documentation](docs/) for detailed API reference and guides.