#!/usr/bin/env python3
"""
Full SOTA Configuration for 8M Document Training
Optimized for maximum performance on complete OpenWebText dataset
"""

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class FullSOTAConfig:
    """
    Configuration optimized for full 8M document OpenWebText training.
    
    Key optimizations:
    - Larger model (150-200M params) for full dataset scale
    - Extended training schedule for massive data
    - Advanced regularization and optimization
    - Multi-stage curriculum learning
    """
    
    # Model Architecture (Scaled for full dataset)
    vocab_size: int = 50258  # GPT-2 + PAD token
    d_model: int = 896       # Larger than 125M config for more capacity
    num_heads: int = 14      # 64 dimensions per head
    num_layers: int = 16     # Deeper for better performance
    d_ff: int = None         # Auto-computed: 8/3 * d_model
    
    # Sequence Parameters (Progressive)
    max_seq_len: int = 1024  # Full context capability
    seq_len_stage1: int = 256   # Start small for efficiency
    seq_len_stage2: int = 512   # Scale up
    seq_len_stage3: int = 1024  # Full context
    
    # Training Schedule (Extended for massive dataset)
    dropout: float = 0.1     # Start with dropout, will decay
    
    # Optimizer Parameters
    learning_rate: float = 8e-4      # Slightly lower for larger model
    weight_decay: float = 0.01       # Transformer standard
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    
    # Learning Rate Schedule (Multi-stage)
    warmup_steps: int = 8000         # Longer warmup for large dataset
    stage1_steps: int = 50000        # Short seq training
    stage2_steps: int = 200000       # Medium seq training  
    stage3_steps: int = 500000       # Full seq training
    total_steps: int = 750000        # Total training steps
    lr_schedule: str = "cosine"      # Cosine annealing
    min_lr_ratio: float = 0.1
    
    # Training Strategy (Optimized for M1 Max)
    batch_size: int = 12             # Optimized for memory
    gradient_accumulation: int = 6   # Effective batch size = 72
    max_eval_batches: int = 200      # Limit validation for speed
    
    # Progressive Training
    curriculum_learning: bool = True
    stage1_batch_size: int = 16      # Smaller seqs = larger batches
    stage2_batch_size: int = 14
    stage3_batch_size: int = 12
    
    # Evaluation and Checkpointing (For long training)
    eval_interval: int = 5000        # Less frequent for speed
    save_interval: int = 25000       # More frequent saves
    generate_interval: int = 10000   # Generation samples
    log_interval: int = 250          # Detailed logging
    
    # Data Parameters
    train_subset_size: Optional[int] = None  # Use full dataset
    curriculum_steps: int = 50000    # Steps to reach next stage
    
    # Advanced Regularization
    label_smoothing: float = 0.1
    dropout_schedule: bool = True
    layerwise_lr_decay: float = 0.8  # Different LR per layer
    
    # Memory and Performance Optimization
    rope_base: float = 10000.0
    eps: float = 1e-5
    tie_embeddings: bool = True      # Save parameters
    use_flash_attn: bool = True      # Faster attention
    gradient_checkpointing: bool = True  # Enable for larger model
    compile_model: bool = True       # PyTorch 2.0 compilation
    
    # Monitoring
    log_interval_detailed: int = 1000
    wandb_logging: bool = False      # Can enable for experiment tracking
    save_best_k: int = 3            # Keep best 3 checkpoints
    early_stopping_patience: int = 50000  # Steps without improvement
    
    def __post_init__(self):
        """Post-initialization computations and validation"""
        
        # Auto-compute feed-forward dimension
        if self.d_ff is None:
            self.d_ff = int(8 * self.d_model / 3)
            self.d_ff = ((self.d_ff + 255) // 256) * 256
        
        # Validate configuration
        assert self.d_model % self.num_heads == 0, f"d_model must be divisible by num_heads"
        assert self.warmup_steps < self.total_steps, f"warmup_steps must be < total_steps"
        assert self.stage1_steps + self.stage2_steps <= self.total_steps
        
        # Compute derived values
        self.head_dim = self.d_model // self.num_heads
        self.effective_batch_size = self.batch_size * self.gradient_accumulation
        
        # Compute parameter count
        self._compute_param_count()
        
        # Validate stages
        assert self.seq_len_stage1 <= self.seq_len_stage2 <= self.seq_len_stage3
        assert self.seq_len_stage3 <= self.max_seq_len
    
    def _compute_param_count(self):
        """Compute parameter count for the full model"""
        # Token embeddings
        embed_params = self.vocab_size * self.d_model
        
        # Transformer layers
        layer_params = (
            # Attention: Q, K, V, O projections
            4 * self.d_model * self.d_model +
            # Feed-forward: up, gate, down projections (SwiGLU)
            3 * self.d_model * self.d_ff +
            # RMSNorm: 2 per layer
            2 * self.d_model
        ) * self.num_layers
        
        # Final layer norm
        final_norm_params = self.d_model
        
        # Output projection (tied if enabled)
        output_params = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        
        # Total
        self.total_params = embed_params + layer_params + final_norm_params + output_params
        self.param_count_m = self.total_params / 1_000_000
    
    def get_stage_config(self, step: int) -> dict:
        """Get configuration for current training stage"""
        if step < self.stage1_steps:
            return {
                "stage": 1,
                "seq_len": self.seq_len_stage1,
                "batch_size": self.stage1_batch_size,
                "description": "Stage 1: Short sequences, efficient training"
            }
        elif step < self.stage1_steps + self.stage2_steps:
            return {
                "stage": 2, 
                "seq_len": self.seq_len_stage2,
                "batch_size": self.stage2_batch_size,
                "description": "Stage 2: Medium sequences, scaling up"
            }
        else:
            return {
                "stage": 3,
                "seq_len": self.seq_len_stage3,
                "batch_size": self.stage3_batch_size,
                "description": "Stage 3: Full context, maximum performance"
            }
    
    def print_full_config(self):
        """Print comprehensive configuration"""
        print("🌍 FULL SOTA TRANSFORMER CONFIG - 8M DOCUMENTS")
        print("=" * 70)
        print(f"📊 Model Architecture:")
        print(f"   • Parameters: {self.param_count_m:.1f}M")
        print(f"   • Vocabulary: {self.vocab_size:,}")
        print(f"   • Model dim: {self.d_model}")
        print(f"   • Layers: {self.num_layers}")
        print(f"   • Heads: {self.num_heads} (dim {self.head_dim} each)")
        print(f"   • Feed-forward: {self.d_ff:,}")
        
        print(f"\n🎯 Progressive Training:")
        print(f"   • Stage 1: {self.seq_len_stage1} tokens, {self.stage1_steps:,} steps")
        print(f"   • Stage 2: {self.seq_len_stage2} tokens, {self.stage2_steps:,} steps")
        print(f"   • Stage 3: {self.seq_len_stage3} tokens, {self.total_steps - self.stage1_steps - self.stage2_steps:,} steps")
        print(f"   • Total steps: {self.total_steps:,}")
        
        print(f"\n⚡ Training Setup:")
        print(f"   • Learning rate: {self.learning_rate}")
        print(f"   • Weight decay: {self.weight_decay}")
        print(f"   • Warmup steps: {self.warmup_steps:,}")
        print(f"   • Effective batch: {self.effective_batch_size}")
        
        print(f"\n🔧 Optimizations:")
        print(f"   • Tied embeddings: {self.tie_embeddings}")
        print(f"   • Flash attention: {self.use_flash_attn}")
        print(f"   • Gradient checkpointing: {self.gradient_checkpointing}")
        print(f"   • Model compilation: {self.compile_model}")
        print(f"   • Label smoothing: {self.label_smoothing}")


def get_full_sota_config() -> FullSOTAConfig:
    """Get the full SOTA configuration for 8M documents"""
    return FullSOTAConfig()


def get_full_sota_conservative() -> FullSOTAConfig:
    """Conservative version for system limitations"""
    config = FullSOTAConfig()
    
    # Reduce model size if needed
    config.d_model = 768
    config.num_heads = 12
    config.num_layers = 12
    config.batch_size = 16
    config.gradient_checkpointing = True
    
    return config


def estimate_training_time(config: FullSOTAConfig, sequences_count: int) -> dict:
    """Estimate training time and resource requirements"""
    
    # Estimates based on M1 Max performance
    tokens_per_sec_estimate = 250  # Conservative estimate for large model
    
    total_tokens = sequences_count * config.seq_len_stage2  # Average sequence length
    training_tokens = total_tokens * (config.total_steps / sequences_count)  # Multiple epochs
    
    training_time_seconds = training_tokens / tokens_per_sec_estimate
    training_time_hours = training_time_seconds / 3600
    training_time_days = training_time_hours / 24
    
    # Memory estimates
    param_memory_gb = config.total_params * 4 / (1024**3)  # FP32
    activation_memory_gb = config.effective_batch_size * config.max_seq_len * config.d_model * 4 / (1024**3)
    total_memory_gb = param_memory_gb * 2 + activation_memory_gb  # 2x for gradients
    
    return {
        "training_time_hours": training_time_hours,
        "training_time_days": training_time_days,
        "param_memory_gb": param_memory_gb,
        "total_memory_gb": total_memory_gb,
        "tokens_per_second": tokens_per_sec_estimate,
        "total_training_tokens": training_tokens
    }


if __name__ == "__main__":
    print("🧪 TESTING FULL SOTA CONFIGURATIONS\n")
    
    # Test full config
    config = get_full_sota_config()
    config.print_full_config()
    
    # Estimate resources
    print(f"\n⏰ Training Estimates (assuming 5M sequences):")
    estimates = estimate_training_time(config, 5_000_000)
    print(f"   Training time: {estimates['training_time_days']:.1f} days")
    print(f"   Memory needed: {estimates['total_memory_gb']:.1f} GB")
    print(f"   Training tokens: {estimates['total_training_tokens']:,}")
    
    print(f"\n✅ Full SOTA configuration ready!")