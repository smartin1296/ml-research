"""
Step 1: Configuration
Defines all hyperparameters for the SOTA transformer
"""

from dataclasses import dataclass


@dataclass
class SOTAConfig:
    """Configuration for State-of-the-Art Transformer"""
    # Model dimensions
    vocab_size: int = 50257
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = None  # Auto-computed if None (8/3 * d_model)
    
    # Sequence parameters
    max_seq_len: int = 2048
    
    # Training parameters
    dropout: float = 0.0  # Modern models use minimal dropout
    
    # Advanced features
    rope_base: float = 10000.0  # RoPE frequency base
    eps: float = 1e-5  # Layer norm epsilon
    tie_embeddings: bool = False  # Tie input/output embeddings
    use_flash_attn: bool = False  # Use Flash Attention if available
    gradient_checkpointing: bool = False  # Memory-efficient training
    
    def __post_init__(self):
        """Auto-compute feed-forward dimension if not specified"""
        if self.d_ff is None:
            # LLaMA formula: 8/3 * d_model, rounded to nearest 256
            self.d_ff = int(8 * self.d_model / 3)
            self.d_ff = ((self.d_ff + 255) // 256) * 256