# Token-Level RNN Training

## 🚀 Current Production Version

### **`train.py`** - MPS-Optimized Training ⭐
**This is the current best version to use.**

**Features:**
- **MPS-optimized compilation** (reduce-overhead mode)
- **Advanced DataLoader optimization** (2 workers, pin_memory, prefetch)
- **Intelligent training**: Trains until 80% accuracy target or plateau detection
- **Speed optimized**: ~2x faster than baseline (~0.5s/batch vs 1.0s/batch)
- **Smart checkpointing**: Only saves on improvements (no disk bloat)

**Usage:**
```bash
# Run optimized training (recommended)
PYTHONPATH=. python tokens/train.py
```

**Expected Results:**
- **Target**: 80% training accuracy or automatic plateau detection
- **Speed**: ~40-50s per epoch (MPS optimized)
- **Architecture**: 6.5M parameters (3L×512H×384E)
- **Checkpoints**: `latest_mps_optimized.pt`, `best_mps_optimized.pt`

---

## 📁 Core Infrastructure

### **`models.py`** - Token RNN Architectures
- `OptimizedTokenRNNModel` - Production model with weight tying
- BPE-optimized LSTM layers with LayerNorm

### **`dataset.py`** - Token Datasets  
- `TokenSequenceDataset` - Efficient sequence generation
- `TokenTextGenerator` - High-quality text generation
- `create_token_datasets` - Optimized data loading

### **`tokenizers.py`** - Tokenization
- `BPETokenizer` - Byte Pair Encoding (recommended)
- `WordTokenizer` - Word-level tokenization
- Vocab size 500 optimal for Shakespeare

### **`config/`** - Saved Configurations
- `token_optimal_final.json` - Benchmarked optimal settings
- `token_optimal_config.json` - Alternative configurations

---

## 📚 Archive (Previous Versions)

### **`archive/v1_development/`** - Early Development
- `test_basic.py` - Basic token training
- `test_token_optimal.py` - Initial optimization
- `demo_token.py` - Simple demo script

### **`archive/v2_optimization/`** - Optimization Experiments  
- `optimize_token*.py` - Hyperparameter tuning scripts
- Various batch size and architecture experiments

### **`archive/v3_maximal_accuracy/`** - Accuracy-Focused Training
- `test_maximal_accuracy.py` - Extended accuracy training
- `finalize_maximal_accuracy.py` - Results finalization
- `test_speed_optimized.py` - Initial speed optimization attempts

---

## 🎯 Quick Start

**For most users (recommended):**
```bash
PYTHONPATH=. python tokens/train.py
```

**For development/research:**
1. Check `archive/` for historical approaches
2. Modify `train.py` for custom experiments
3. Use `config/` for proven optimal settings

---

## 📊 Performance Benchmarks

| Version | Speed (s/batch) | Max Accuracy | Features |
|---------|----------------|--------------|----------|
| **train.py** ⭐ | ~0.5s | 80%+ target | MPS optimized, intelligent training |
| v3_maximal_accuracy | ~0.8s | 53.5% | Extended training, plateau detection |
| v2_optimization | ~0.8s | Variable | Hyperparameter experiments |
| v1_development | ~1.0s | 39% | Basic functionality |

---

## 🏆 Key Achievements

- **Token RNN Breakthrough**: From 2% failure to 53%+ accuracy
- **Speed Optimization**: 2x faster training with MPS optimizations  
- **Intelligent Training**: Automatic plateau detection and target-based stopping
- **Production Ready**: Clean, optimized, industry-standard organization