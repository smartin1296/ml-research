#!/usr/bin/env python3
"""
Optimized Configuration for SOTA Transformer Training
Fixes critical training parameters for maximum performance
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizedSOTAConfig:
    """
    Optimized configuration with critical fixes:
    - Correct weight decay (0.01 not 0.1)
    - Proper AdamW betas (0.9, 0.98)
    - Optimal learning rate scheduling
    - Scaled architecture for M1 Max
    """
    
    # Model Architecture (125M parameters - optimal for M1 Max)
    vocab_size: int = 50258  # GPT-2 + PAD token
    d_model: int = 768       # Standard GPT-2 Small
    num_heads: int = 12      # 64 dimensions per head
    num_layers: int = 12     # Deep enough for good performance
    d_ff: int = None         # Auto-computed: 8/3 * d_model
    
    # Sequence Parameters
    max_seq_len: int = 1024  # Reduced from 2048 for memory efficiency
    
    # Training Parameters (CRITICAL FIXES)
    dropout: float = 0.1     # Start with dropout, will decay to 0
    
    # Optimizer Parameters (TRANSFORMER STANDARD)
    learning_rate: float = 1e-3      # Fixed: was too low at 3e-4
    weight_decay: float = 0.01       # Fixed: was too high at 0.1
    adam_beta1: float = 0.9          # Standard
    adam_beta2: float = 0.98         # Transformer standard (not 0.999)
    adam_eps: float = 1e-8           # Standard
    grad_clip_norm: float = 1.0      # Gradient clipping
    
    # Learning Rate Schedule
    warmup_steps: int = 4000         # Warmup period
    lr_schedule: str = "cosine"      # "cosine", "linear", or "onecycle"
    min_lr_ratio: float = 0.1        # Minimum LR as ratio of peak LR
    
    # Training Strategy
    batch_size: int = 16             # Base batch size
    gradient_accumulation: int = 4    # Effective batch size = 64
    max_steps: int = 100000          # Total training steps
    eval_interval: int = 2000        # Evaluate every N steps
    save_interval: int = 10000       # Save checkpoint every N steps
    
    # Data Parameters
    seq_len_start: int = 512         # Start with shorter sequences
    seq_len_end: int = 1024          # Scale up to longer sequences
    curriculum_steps: int = 20000    # Steps to reach max sequence length
    
    # Regularization
    label_smoothing: float = 0.1     # Label smoothing for better generalization
    dropout_schedule: bool = True    # Decay dropout from 0.1 to 0
    
    # Advanced Features
    rope_base: float = 10000.0       # RoPE frequency base
    eps: float = 1e-5               # RMSNorm epsilon
    tie_embeddings: bool = True      # Saves ~25M parameters
    use_flash_attn: bool = True      # Enable Flash Attention
    gradient_checkpointing: bool = False  # Start False, enable if OOM
    
    # Logging and Monitoring
    log_interval: int = 100          # Log every N steps
    generate_interval: int = 2000    # Generate samples every N steps
    generate_length: int = 200       # Length of generated samples
    
    def __post_init__(self):
        """Post-initialization computations and validations"""
        
        # Auto-compute feed-forward dimension
        if self.d_ff is None:
            # LLaMA formula: 8/3 * d_model, rounded to nearest 256
            self.d_ff = int(8 * self.d_model / 3)
            self.d_ff = ((self.d_ff + 255) // 256) * 256
        
        # Validate configuration
        assert self.d_model % self.num_heads == 0, f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.max_seq_len >= self.seq_len_end, f"max_seq_len must be >= seq_len_end"
        assert self.warmup_steps < self.max_steps, f"warmup_steps must be < max_steps"
        
        # Compute head dimension
        self.head_dim = self.d_model // self.num_heads
        
        # Compute effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation
        
        # Compute approximate parameter count
        self._compute_param_count()
    
    def _compute_param_count(self):
        """Approximate parameter count calculation"""
        
        # Token embeddings
        embed_params = self.vocab_size * self.d_model
        
        # Positional embeddings (none for RoPE)
        pos_params = 0
        
        # Transformer layers
        layer_params = (
            # Attention: Q, K, V, O projections
            4 * self.d_model * self.d_model +
            # Feed-forward: up, gate, down projections (SwiGLU)
            3 * self.d_model * self.d_ff +
            # RMSNorm: 2 per layer (attention + FFN)
            2 * self.d_model
        ) * self.num_layers
        
        # Final layer norm
        final_norm_params = self.d_model
        
        # Output projection (shared with embeddings if tie_embeddings=True)
        if self.tie_embeddings:
            output_params = 0
        else:
            output_params = self.vocab_size * self.d_model
        
        # Total
        self.total_params = (
            embed_params + pos_params + layer_params + 
            final_norm_params + output_params
        )
        
        # Convert to millions
        self.param_count_m = self.total_params / 1_000_000
    
    def get_lr_schedule_info(self) -> dict:
        """Get learning rate schedule information"""
        return {
            "schedule_type": self.lr_schedule,
            "peak_lr": self.learning_rate,
            "min_lr": self.learning_rate * self.min_lr_ratio,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.max_steps
        }
    
    def print_config(self):
        """Print configuration summary"""
        print("🚀 OPTIMIZED SOTA TRANSFORMER CONFIG")
        print("=" * 50)
        print(f"📊 Model Architecture:")
        print(f"   • Parameters: {self.param_count_m:.1f}M")
        print(f"   • Vocabulary: {self.vocab_size:,}")
        print(f"   • Model dim: {self.d_model}")
        print(f"   • Layers: {self.num_layers}")
        print(f"   • Heads: {self.num_heads} (dim {self.head_dim} each)")
        print(f"   • Feed-forward: {self.d_ff:,}")
        print(f"   • Sequence length: {self.seq_len_start}→{self.seq_len_end}")
        
        print(f"\n🎯 Training Setup:")
        print(f"   • Learning rate: {self.learning_rate}")
        print(f"   • Weight decay: {self.weight_decay}")
        print(f"   • AdamW betas: ({self.adam_beta1}, {self.adam_beta2})")
        print(f"   • Batch size: {self.batch_size} × {self.gradient_accumulation} = {self.effective_batch_size}")
        print(f"   • Max steps: {self.max_steps:,}")
        print(f"   • Warmup: {self.warmup_steps:,} steps")
        
        print(f"\n⚡ Optimizations:")
        print(f"   • Tied embeddings: {self.tie_embeddings}")
        print(f"   • Flash attention: {self.use_flash_attn}")
        print(f"   • Gradient checkpointing: {self.gradient_checkpointing}")
        print(f"   • Label smoothing: {self.label_smoothing}")
        print(f"   • Dropout schedule: {self.dropout_schedule}")


# Predefined configurations
def get_config_125m() -> OptimizedSOTAConfig:
    """Get 125M parameter configuration (recommended for M1 Max)"""
    return OptimizedSOTAConfig(
        d_model=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024,
        batch_size=16,
        gradient_accumulation=4,
        tie_embeddings=True
    )

def get_config_250m() -> OptimizedSOTAConfig:
    """Get 250M parameter configuration (if memory allows)"""
    return OptimizedSOTAConfig(
        d_model=1024,
        num_layers=16,
        num_heads=16,
        max_seq_len=1024,
        batch_size=8,
        gradient_accumulation=8,
        tie_embeddings=True,
        gradient_checkpointing=True
    )

def get_config_debug() -> OptimizedSOTAConfig:
    """Get small configuration for debugging"""
    return OptimizedSOTAConfig(
        d_model=256,
        num_layers=4,
        num_heads=8,
        max_seq_len=512,
        seq_len_start=256,
        seq_len_end=512,
        batch_size=8,
        gradient_accumulation=2,
        max_steps=1000,
        warmup_steps=100,
        eval_interval=100,
        log_interval=10
    )


if __name__ == "__main__":
    print("🧪 TESTING OPTIMIZED CONFIGURATIONS\n")
    
    # Test 125M config
    config_125m = get_config_125m()
    config_125m.print_config()
    
    print("\n" + "="*60 + "\n")
    
    # Test 250M config
    config_250m = get_config_250m()
    config_250m.print_config()
    
    print("\n" + "="*60 + "\n")
    
    # Test debug config
    config_debug = get_config_debug()
    config_debug.print_config()
    
    print(f"\n✅ All configurations validated!")