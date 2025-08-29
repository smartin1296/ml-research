# SOTA Transformer - Clean Modular Implementation

## Overview
State-of-the-art transformer architecture with all modern innovations, organized into clear, readable components.

## Architecture Components

The implementation is modularized into distinct files, each representing a key architectural component:

1. **`config.py`** - Configuration dataclass with all hyperparameters
2. **`rmsnorm.py`** - Root Mean Square Normalization (more efficient than LayerNorm)
3. **`rope.py`** - Rotary Position Embeddings (superior to absolute positions)
4. **`swiglu.py`** - SwiGLU activation (gated feed-forward network)
5. **`attention.py`** - Multi-Head Attention with RoPE integration
6. **`transformer_block.py`** - Complete transformer block with residuals
7. **`embeddings.py`** - Token embeddings and output projection
8. **`full_model.py`** - Main orchestrator that assembles all components

## Key Features

- **RMSNorm**: More efficient normalization used in LLaMA, Gemma
- **RoPE**: Better position encoding with improved extrapolation
- **SwiGLU**: Superior feed-forward activation function
- **Pre-normalization**: Apply norm before sub-layers (modern best practice)
- **Flash Attention ready**: Optional efficient attention support
- **Clean architecture**: Each component in its own file for clarity

## Usage

### Quick Demo
```bash
python demo.py
```

### Training
```bash
python train.py
```

### Component Example
```python
from sota_components.config import SOTAConfig
from sota_components.full_model import SOTATransformer

# Configure model
config = SOTAConfig(
    vocab_size=50257,
    d_model=768,
    num_heads=12,
    num_layers=12
)

# Create model
model = SOTATransformer(config)

# Forward pass
logits = model(input_ids)

# Generate text
generated = model.generate(prompt, max_new_tokens=100)
```

## Architecture Flow

```
Input Token IDs
    ↓
Token Embeddings
    ↓
Transformer Block 1
  ├─ RMSNorm → Attention + RoPE → Residual
  └─ RMSNorm → SwiGLU FFN → Residual
    ↓
... (more blocks) ...
    ↓
Final RMSNorm
    ↓
Output Projection
    ↓
Logits
```

## Training Results

- Successfully trained on TinyStories dataset
- 22.5M parameters
- Loss: 7.7 → 2.0 over 3 epochs
- Pure FP32 on Apple Silicon MPS (FP16 has MPS backend limitations)

## Files

- `sota_components/` - Modular architecture components
- `data/` - Tokenizer and cached data
- `train.py` - Clean training script
- `demo.py` - Architecture walkthrough
- `README.md` - This file

## Technical Notes

- MPS (Apple Silicon) has FP16 limitations, use FP32 for stable training
- Each component is self-contained and can be understood independently
- The `full_model.py` serves as a high-level overview of the architecture