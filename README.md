# ML Research Environment - SOTA Transformer

## Overview
Streamlined machine learning research environment focused on state-of-the-art transformer implementation with modern architectural innovations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run architecture demo
cd algorithms/transformers
python demo.py

# Train model
python train.py
```

## Current Implementation

### SOTA Transformer ✅

**Architecture Features:**
- **RMSNorm**: Root Mean Square normalization (more efficient than LayerNorm)
- **RoPE**: Rotary Position Embeddings (superior position encoding)
- **SwiGLU**: Gated activation function (outperforms ReLU/GELU)
- **Pre-normalization**: Modern best practice for training stability
- **Flash Attention ready**: Support for efficient attention mechanisms

**Performance:**
- 22.5M parameters (configurable)
- Successfully trained on TinyStories dataset
- Loss reduction: 7.7 → 2.0 over 3 epochs
- Runs on Apple Silicon MPS in FP32

## Project Structure

```
research/
├── algorithms/
│   └── transformers/        # SOTA transformer implementation
│       ├── sota_components/ # Modular architecture components
│       │   ├── config.py       # Hyperparameter configuration
│       │   ├── rmsnorm.py      # RMS normalization
│       │   ├── rope.py         # Rotary position embeddings
│       │   ├── swiglu.py       # SwiGLU activation
│       │   ├── attention.py    # Multi-head attention
│       │   ├── transformer_block.py  # Complete block
│       │   ├── embeddings.py   # Token embeddings
│       │   └── full_model.py   # Main orchestrator
│       ├── data/           # Tokenizer and cached data
│       ├── train.py        # Training script
│       └── demo.py         # Architecture demonstration
├── data/                   # Raw datasets
├── utils/                  # Utility functions
└── requirements.txt        # Dependencies
```

## Architecture Flow

```
Input Token IDs
     ↓
Token Embeddings
     ↓
Transformer Block × N
  ├─ RMSNorm → Attention + RoPE → Residual
  └─ RMSNorm → SwiGLU FFN → Residual
     ↓
Final RMSNorm
     ↓
Output Projection
     ↓
Logits
```

## Key Technical Insights

### Mixed Precision on MPS
- Apple Silicon MPS has fundamental FP16 limitations
- Mixed dtype operations cause backend errors
- Solution: Use FP32 for stable training

### Modular Design
Each component is self-contained and can be understood independently:
- Easy to modify individual components
- Clear architectural flow
- Reusable building blocks

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- 8GB+ RAM (16GB recommended)
- GPU optional (MPS/CUDA supported)

## Hardware Support

- **Apple Silicon** (M1/M2/M3): MPS acceleration
- **NVIDIA GPUs**: CUDA support
- **CPU**: Universal fallback

## Usage Example

```python
from algorithms.transformers.sota_components.config import SOTAConfig
from algorithms.transformers.sota_components.full_model import SOTATransformer

# Configure model
config = SOTAConfig(
    vocab_size=50257,
    d_model=768,
    num_heads=12,
    num_layers=12
)

# Create and use model
model = SOTATransformer(config)
logits = model(input_ids)
generated = model.generate(prompt, max_new_tokens=100)
```

## License
MIT