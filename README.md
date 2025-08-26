# Machine Learning Research Environment

Machine learning research environment for testing different neural network architectures on various datasets. All implementations use industry-standard libraries (PyTorch, NumPy, scikit-learn) with comprehensive benchmarking.

## Project Structure

```
├── algorithms/              # Neural network implementations
│   ├── rnn/                # Recurrent networks (LSTM, GRU)
│   ├── cnn/                # Convolutional networks  
│   ├── transformers/       # Transformer architectures
│   └── reasoning_nns/      # Test-time compute models
├── data/                   # Dataset storage and preprocessing
├── utils/                  # Benchmarking and analysis utilities
└── requirements.txt        # Dependencies
```

## Environment Setup

### Hardware
- Primary testing platform: M1 Max MacBook Pro, 64GB RAM
- Cross-platform support: Apple Silicon (MPS), NVIDIA CUDA, CPU

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
cd algorithms/rnn
python tokens/train.py  # Token-level RNN training
```

## Implementation Status

### RNN Module (Complete)
- **Character-level**: LSTM with layer normalization, gradient clipping
- **Token-level**: BPE tokenization with MPS optimization
- **Features**: Automatic device detection, plateau-based early stopping
- **Training**: OneCycleLR scheduling, mixed precision support

### CNN Module (Complete)
- **Architectures**: SimpleCNN, ResNet variants with proper initialization
- **Features**: Intelligent training with adaptive stopping criteria
- **Training**: No hardcoded epoch limits, automatic convergence detection
- **Performance**: 86.15% validation accuracy on CIFAR-10

## Test Results

### Character-level RNN (Shakespeare dataset)  
- **Architecture**: 2L-384H-192E (2.1M parameters)
- **Validation accuracy**: 59.67%
- **Training time**: 8.9 minutes (M1 Max)
- **Throughput**: 5,018 samples/second
- **Batch processing**: batch_size=2048 optimal for M1 Max

### Token-level RNN (Shakespeare dataset)
- **Dataset**: 1.1M character corpus, BPE tokenization (vocab_size=500)
- **Architecture**: 3L-512H-384E (6.5M parameters)
- **Training accuracy**: 57.18% (30 epochs, plateau detected)
- **Training time**: 24.6 minutes (M1 Max)
- **Throughput**: 6,149 samples/second
- **Batch processing**: ~0.59s/batch (batch_size=4096)
- **Optimizations**: MPS compilation, DataLoader optimization, power-of-2 sequence lengths

### SimpleCNN (CIFAR-10 dataset)
- **Architecture**: Base CNN with 64 channels (1.6M parameters)
- **Validation accuracy**: 86.15%
- **Test accuracy**: 80.80%
- **Training time**: 21.7 minutes total (55 epochs, intelligent stopping)
- **Throughput**: ~9,500 samples/second
- **Batch processing**: batch_size=128 optimal for M1 Max

## Usage Examples

```bash
# Best RNN (token-level with BPE)
cd algorithms/rnn
python tokens/train.py

# Best CNN (intelligent training)
cd algorithms/cnn  
python train_intelligent.py
```

## Core Principles

- Industry-standard libraries only (PyTorch, NumPy, scikit-learn)
- Cross-platform compatibility
- Comprehensive benchmarking with statistical analysis
- Reproducible results with standardized output formats
- No exotic dependencies for maximum portability