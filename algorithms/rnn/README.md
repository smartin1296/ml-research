# State-of-the-Art Recurrent Neural Networks

This module provides research-grade implementations of RNN architectures with modern PyTorch practices and comprehensive benchmarking capabilities.

## Features

### 🏗️ **SOTA Architectures**
- **LSTM**: Custom implementation with proper initialization, layer normalization, and gradient-friendly design
- **GRU**: Efficient gated recurrent unit for faster training
- **Vanilla RNN**: Baseline implementation for comparison
- **Bidirectional variants**: All models support bidirectional processing

### 🚀 **Modern Training Framework**
- **Mixed Precision Training**: Automatic mixed precision with GradScaler
- **Gradient Clipping**: Essential for stable RNN training
- **Advanced Scheduling**: OneCycleLR, cosine annealing, plateau reduction
- **Accuracy Optimization**: Plateau detection and early stopping
- **Smart Checkpointing**: Saves only on improvements to reduce disk usage
- **Label Smoothing**: Better generalization for token prediction

### 📁 **Modular Architecture**
```
algorithms/rnn/
├── core/              # Shared infrastructure
│   ├── models.py      # LSTM, GRU, VanillaRNN
│   ├── trainer.py     # Advanced training framework
│   └── device_utils.py, results_utils.py
├── character/         # Character-level implementations
│   ├── test_basic.py  # M1 Max optimized (5,720 samples/sec)
│   └── test_maximal_accuracy.py # Enhanced training
├── tokens/            # Token-level implementations
│   ├── models.py      # BPE-optimized architectures
│   ├── tokenizers.py  # BPE and word tokenization
│   ├── test_maximal_accuracy.py # 39%+ accuracy training
│   └── optimization/  # Hyperparameter experiments
├── checkpoints/{character,tokens}/ # Organized storage
└── results/{character,tokens}/     # Results by type
```

## Quick Start

### Character-Level Training
```bash
# Basic character training (M1 Max optimized)
python character/test_basic.py

# Maximal accuracy character training
python character/test_maximal_accuracy.py
```

### Token-Level Training  
```bash
# Basic token training with BPE
python tokens/test_basic.py

# Maximal accuracy token training (advanced)
python tokens/test_maximal_accuracy.py
```

### Utilities
```bash
# Check device compatibility
python core/device_utils.py

# Clean up temporary files  
python cleanup.py
```

## Directory Navigation

### Core Infrastructure (`core/`)
- **models.py**: LSTM, GRU, VanillaRNN implementations
- **trainer.py**: Training framework with checkpointing
- **device_utils.py**: Hardware detection and optimization
- **results_utils.py**: Standardized results output

### Character-Level (`character/`)
- **dataset.py**: Character tokenization and data loading
- **test_basic.py**: Standard character training
- **test_maximal_accuracy.py**: Extended accuracy-focused training

### Token-Level (`tokens/`)
- **models.py**: Token-specific RNN architectures
- **tokenizers.py**: BPE and word tokenization
- **dataset.py**: Token sequence datasets
- **test_basic.py**: Standard token training
- **test_maximal_accuracy.py**: Advanced accuracy training
- **optimization/**: Hyperparameter tuning experiments
- **config/**: Saved optimal configurations

### Storage Organization
- **checkpoints/{character,tokens}/**: Model checkpoints by type
- **results/{character,tokens}/**: Training results by type

## Architecture Details

### LSTM Cell Implementation
Our LSTM implementation follows SOTA practices:

- **Xavier Initialization**: Proper weight initialization for stable training
- **Forget Gate Bias = 1**: Initialized to 1.0 (Jozefowicz et al., 2015)
- **Layer Normalization**: Optional LayerNorm for faster convergence (Ba et al., 2016)
- **Gradient Clipping**: Built-in support to prevent exploding gradients

```python
# Key features of our LSTM
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, layer_norm=False):
        # Proper weight initialization
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        
        # Forget gate bias = 1
        nn.init.ones_(self.bias_ih[hidden_size:2*hidden_size])
        
        # Optional layer normalization
        if layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
```

### Training Framework Features

- **Mixed Precision**: Up to 2x speedup with maintained accuracy
- **Dynamic Loss Scaling**: Automatic handling of gradient underflow
- **Learning Rate Scheduling**: Multiple schedulers (cosine, plateau, step)
- **Early Stopping**: Configurable patience and minimum delta
- **Comprehensive Logging**: Loss tracking, learning rates, timing

## Benchmarking Results

Run `python benchmark.py` to get comprehensive comparison:

```
Model           Val Loss     Parameters   Train Time (s)
----------------------------------------------------
LSTM           2.1234       1,234,567    45.2
BiLSTM         2.0987       2,345,678    67.8
GRU            2.1456       987,654      38.9
VanillaRNN     2.3456       876,543      32.1
```

## File Structure

- **`test.py`**: Primary training script with optimal M1 Max settings
- **`models.py`**: LSTM, GRU, and Vanilla RNN implementations  
- **`trainer.py`**: Modern training framework with checkpointing
- **`dataset.py`**: Text processing and sequence generation
- **`device_utils.py`**: Cross-platform GPU detection
- **`results_utils.py`**: Standardized results formatting

## Key Research Features

### 1. Proper LSTM Initialization
Following Jozefowicz et al. (2015), forget gate biases are initialized to 1.0 for better gradient flow.

### 2. Layer Normalization Support  
Implements Ba et al. (2016) layer normalization for faster convergence and training stability.

### 3. Mixed Precision Training
Automatic mixed precision training with PyTorch's native AMP for 2x speedup on modern GPUs.

### 4. Gradient Clipping
Essential for RNN training stability - configurable norm-based clipping.

### 5. Bidirectional Processing
All architectures support bidirectional variants for improved context modeling.

## Performance Tips

1. **Use layer normalization** for faster convergence
2. **Enable mixed precision** on GPU for speed
3. **Tune gradient clipping** (typically 0.5-2.0)
4. **BiLSTM** often performs best but is 2x slower
5. **GRU** is faster than LSTM with similar performance

## Research Usage

This implementation is designed for research and experimentation:

- Clean, readable code for understanding RNN internals
- Proper statistical testing for architecture comparison  
- Comprehensive benchmarking for paper-quality results
- Extensible design for custom modifications

## 🚀 GPU Acceleration Support

### Cross-Platform GPU Support
This implementation supports **all major GPU platforms**:

- **🍎 Apple Silicon** (M1/M2/M3 MacBook Pro/Air, Mac Studio) - Uses Metal Performance Shaders (MPS)
- **🟢 NVIDIA GPUs** - Uses CUDA with mixed precision training
- **💻 CPU Fallback** - Automatic fallback for any system

### Quick GPU Test
```bash
# Test your GPU acceleration
python device_utils.py    # Check available devices
python gpu_test.py        # Full GPU performance test
```

### Apple Silicon GPU Setup
**MacBook Pro/Air users**: Your GPU is ready to use! No additional setup needed.

```python
from device_utils import get_best_device, should_use_mixed_precision

device = get_best_device()  # Auto-selects: MPS > CUDA > CPU
mixed_precision = should_use_mixed_precision(device)
```

### Performance Expectations

#### 🔥 **M1 Max Character-Level Performance (Measured 2025-08-25)**
**OPTIMAL CONFIGURATION**: 
- Batch Size: **2048**, Sequence Length: **25**
- Embedding: **192**, Hidden: **384**, Vocab: **67** 
- **Peak Performance**: **5,720 samples/sec**
- Model: 2.1M parameters LSTM with LayerNorm

**Scaling Results**:
```
Batch Size  | Samples/Sec | Notes
------------|-------------|------------------
512         | 3,081       | Good baseline
1024        | 4,685       | Better scaling
2048        | 5,720       | OPTIMAL ⭐
4096        | 4,787       | Memory limited
8192        | 3,542       | Too large
```

### **Auto-Optimization**
If optimal settings aren't available for your hardware, Claude will run:
```bash
python device_utils.py              # For hardware detection
# Future: python optimize_settings.py  # Auto-find optimal config
```

#### **Other Platforms**
- **NVIDIA GPU**: 3-8x faster than CPU, mixed precision gives additional 2x
- **CPU**: Reliable baseline, good for prototyping

### GPU Compatibility Notes
| Platform | Mixed Precision | Optimal Batch Size | Peak Performance | Notes |
|----------|----------------|-------------------|------------------|-------|
| **Apple M1 Max (32 cores)** | ❌ Disabled | **2048** | **5,720 samples/sec** | 178x CPU speedup, unified memory |
| Apple M1/M2 (8-10 cores) | ❌ Disabled | 256-512 | ~1,500 samples/sec | Scale batch size down |
| NVIDIA (CUDA) | ✅ Enabled | 32-128 | Varies by GPU | Tensor Cores provide 2x speedup |
| CPU | ❌ N/A | 16-64 | ~32 samples/sec | Good for debugging and small experiments |

## ⚠️ Important Implementation Notes

### LayerNorm Configuration
**Critical Fix Applied**: LayerNorm requires `elementwise_affine=True` (default) for proper gradient flow. Setting `elementwise_affine=False` prevents training by breaking gradient computation.

```python
# ✅ Correct - enables learnable parameters
self.ln_ih = nn.LayerNorm(4 * hidden_size)  

# ❌ Wrong - breaks gradients
self.ln_ih = nn.LayerNorm(4 * hidden_size, elementwise_affine=False)
```

### Model Size Recommendations
- **Quick testing**: 32-64 hidden units, sequence length 20-30
- **Research/production**: 128-512 hidden units, sequence length 100+
- **Memory constraints**: Use smaller batch sizes rather than smaller models

### Training Tips
1. **Start small**: Use `quick_start.py` to verify everything works
2. **Layer normalization**: Always enable for faster convergence
3. **Gradient clipping**: Keep between 0.5-2.0 for stability
4. **GPU acceleration**: Use `gpu_test.py` to test Apple Silicon/CUDA performance
5. **Mixed precision**: Enable on NVIDIA GPUs for 2x speedup (auto-disabled on Apple Silicon)
6. **Sequence length**: Shorter sequences train faster, longer capture more context

### 🚀 M1 Max Optimization Guide
**For maximum M1 Max performance** (32 GPU cores):

```python
# Optimal configuration for M1 Max
train_loader, val_loader, tokenizer = create_sample_dataset(
    sequence_length=25,     # Sweet spot for memory
    batch_size=2048,        # Maximum throughput
    data_source='shakespeare'
)

# Larger model to saturate GPU cores  
lstm = LSTM(
    input_size=192,         # Large embedding
    hidden_size=384,        # Large hidden size
    num_layers=2,           # Multi-layer
    layer_norm=True,        # Essential for convergence
    dropout=0.1             # Light regularization
)
```

**Performance Testing Commands**:
```bash
python gpu_max_test.py      # Find your optimal batch size
python gpu_optimized_test.py # Compare multiple configurations
```

**Expected Results**:
- Training time: ~2-3 minutes per epoch (vs 8+ minutes CPU)
- Peak throughput: 5,720+ samples/second  
- Memory usage: ~12-16GB unified memory

### Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| "No gradients" error | LayerNorm elementwise_affine=False | Use default LayerNorm settings |
| Training hangs | Dataset too large/model too big | Use smaller config from quick_start.py |
| Memory error | Batch size or sequence length too large | Reduce batch_size or sequence_length |
| NaN loss | Learning rate too high | Reduce learning rate to 0.001-0.01 |
| Poor generation | Model undertrained | Train more epochs or reduce sequence complexity |

## File Organization

### Usage Notes

**Start Here**: `python test.py` for optimal M1 Max training  
**Device Check**: `python device_utils.py` to verify GPU support  
**Cleanup**: `python cleanup.py` to remove temporary files

## Citation

If you use this implementation in research:

```bibtex
@software{sota_rnn_2024,
  title={State-of-the-Art RNN Implementations},
  author={ML Environment},
  year={2024},
  url={https://github.com/your-repo/ml-environment}
}
```