"""
Step 3: Rotary Position Embeddings (RoPE)
Superior to absolute/learned position embeddings
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    - Encodes relative positions through rotation
    - Better extrapolation to longer sequences
    - Used in LLaMA, PaLM, and many modern models
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos/sin for efficiency
        self._precompute(max_seq_len)
    
    def _precompute(self, seq_len: int):
        """Precompute rotation matrices for all positions"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
        
        Returns:
            Rotated (q, k) tensors
        """
        seq_len = q.shape[2]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._precompute(seq_len)
        
        # Get rotation matrices for current sequence
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation using complex number multiplication"""
        # Split features into pairs
        x1, x2 = x.chunk(2, dim=-1)
        
        # Expand cos/sin to match feature dimension
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        
        # Apply rotation: (x1 + ix2) * (cos + isin)
        rotated = torch.cat([x1, x2], dim=-1) * cos + torch.cat([-x2, x1], dim=-1) * sin
        return rotated