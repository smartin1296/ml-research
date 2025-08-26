# ML Environment - Research Grade Neural Networks

A comprehensive machine learning research environment implementing state-of-the-art neural network architectures with modern PyTorch practices and comprehensive benchmarking capabilities.

## 🚀 Key Features

- **Research-Grade Implementations**: SOTA RNN (LSTM, GRU) with proper initialization and modern training techniques
- **Cross-Platform GPU Support**: Optimized for Apple Silicon (M1/M2/M3), NVIDIA CUDA, and CPU
- **Peak Performance**: 5,720+ samples/sec on M1 Max with 59.67% validation accuracy
- **Modern Training**: Mixed precision, gradient clipping, learning rate scheduling, early stopping
- **Comprehensive Benchmarking**: Statistical testing, performance metrics, text generation
- **Standardized Results**: Consistent human-readable and machine-readable output formats

## 🏗️ Project Structure

```
ml_environment/
├── algorithms/              # Algorithm implementations
│   ├── rnn/                # ✅ RNN (LSTM, GRU) - COMPLETED
│   ├── cnn/                # 🔄 CNN (ResNet, etc.) - TODO  
│   ├── transformers/       # 🔄 Transformers (BERT, GPT, ViT) - TODO
│   ├── reasoning_nns/      # 🔄 Test-time compute models - TODO
│   └── rcl/                # 🔄 Proprietary RCL algorithm - TODO
├── utils/                  # Shared utilities
│   ├── benchmarking.py     # Performance benchmarking tools
│   ├── stats.py            # Statistical analysis & experiment logging
│   └── data_utils.py       # Dataset handling utilities
├── data/                   # Data organization
└── CLAUDE.md              # Development reference guide
```

## ⚡ Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run SOTA RNN Training
```bash
cd algorithms/rnn
python test.py  # 3 epochs, optimal M1 Max settings (batch_size=2048)
```

**Expected Results:**
- **Training Time**: ~9 minutes (M1 Max) / ~30 minutes (CPU)
- **Validation Accuracy**: ~59-60% (character-level next-token prediction)
- **Throughput**: 5,000-6,000 samples/sec (M1 Max) / ~32 samples/sec (CPU)
- **Model Size**: 2.1M parameters

## 🔥 Performance Benchmarks

**Apple Silicon M1 Max**: 5,720 samples/sec peak, 59.67% validation accuracy, 178x CPU speedup

**Optimal Configuration**: batch_size=2048, 2.1M parameters, ~9 minutes training

*Detailed benchmarks available in `algorithms/rnn/README.md`*

## 🧠 RNN Implementation

**Features**: SOTA LSTM with proper initialization, layer normalization, bidirectional support
**Training**: Mixed precision, gradient clipping, learning rate scheduling, early stopping  
**Generation**: Character-level text with controllable sampling

*Complete implementation details in `algorithms/rnn/README.md`*

## 📊 Results and Analysis

All experiments generate comprehensive results in both human-readable and JSON formats:

```
Primary RNN Test Results
========================
Timestamp: 2025-08-25 13:22:03
Device: mps
Model Parameters: 2,123,011
Validation Accuracy: 59.67%
Final Loss: 1.347237
Training Time: 533.81 seconds
Throughput: 5,018 samples/second

Text Generations:
Seed: 'To be'
Generated: To behall in sun: away desire, still we no more...
```

## 🔧 Development

### Core Principles
- **Industry Standards Only**: PyTorch, scikit-learn, NumPy - no exotic dependencies
- **Pure Python**: Maximum compatibility and reproducibility  
- **Benchmark Against Standards**: All algorithms compared to established baselines
- **Statistical Rigor**: Significance testing for model comparisons

### Key Lessons Learned
- LayerNorm requires default settings for proper gradient flow
- M1 Max optimal: batch_size=2048, unified memory scaling
- Cross-platform device detection essential for portability

## 🎯 Usage

```bash
# Quick start
cd algorithms/rnn && python test.py
```

```python
# Basic usage
from rnn import LSTM, RNNTrainer, create_sample_dataset
from device_utils import get_best_device

# Auto device detection and training
device = get_best_device()  # MPS > CUDA > CPU
train_loader, val_loader, tokenizer = create_sample_dataset()
model = LSTM(192, 384, 2, layer_norm=True)
trainer = RNNTrainer(model, optimizer, criterion, device)
history = trainer.train(train_loader, val_loader, num_epochs=3)
```

*Detailed examples in `algorithms/rnn/README.md`*

## 📈 Future Roadmap

- **CNN Module**: ResNet, EfficientNet implementations
- **Transformer Module**: BERT, GPT, Vision Transformer
- **Reasoning NNs**: Test-time compute and reasoning capabilities
- **Advanced Benchmarking**: Automated hyperparameter optimization
- **Multi-GPU Support**: Distributed training capabilities

## 🤝 Contributing

This environment follows strict engineering practices:
1. All new algorithms must be benchmarked against standards
2. Comprehensive testing with statistical significance analysis  
3. Cross-platform compatibility (Apple Silicon, NVIDIA, CPU)
4. Standardized results format for reproducibility

## 📄 License

Open source research environment. See individual algorithm implementations for specific licensing.

---

**Powered by**: PyTorch 2.0+, optimized for Apple Silicon M1/M2/M3 and NVIDIA GPUs

**Performance**: 5,720+ samples/sec peak throughput, 59.67% validation accuracy, research-grade implementations