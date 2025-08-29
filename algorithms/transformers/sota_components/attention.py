"""
Step 5: Multi-Head Attention with RoPE
Core attention mechanism with rotary position embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .rope import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embeddings
    - Scaled dot-product attention
    - Rotary position embeddings instead of absolute positions
    - Optional Flash Attention support
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        use_flash_attn: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attn = use_flash_attn
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Rotary position embeddings
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [seq_len, seq_len]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        q, k = self.rope(q, k)
        
        # Compute attention
        if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0+ efficient attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=mask is None  # Use causal mask if no explicit mask
            )
        else:
            # Standard attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply mask
            if mask is not None:
                scores = scores + mask
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply to values
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(attn_output)
        
        return output