"""
Step 6: Transformer Block
Combines attention and feed-forward with residual connections
"""

import torch
import torch.nn as nn
from typing import Optional

from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    SOTA Transformer Block
    - Pre-normalization (apply norm before sub-layers)
    - RMSNorm instead of LayerNorm
    - SwiGLU feed-forward network
    - Residual connections
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        eps: float = 1e-5,
        use_flash_attn: bool = False
    ):
        super().__init__()
        
        # Layer normalization (RMSNorm)
        self.attn_norm = RMSNorm(d_model, eps)
        self.ffn_norm = RMSNorm(d_model, eps)
        
        # Multi-head attention with RoPE
        self.attention = MultiHeadAttention(
            d_model, num_heads, max_seq_len, 
            rope_base, dropout, use_flash_attn
        )
        
        # SwiGLU feed-forward network
        self.feed_forward = SwiGLU(d_model, d_ff)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply transformer block
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Attention block with pre-norm and residual
        residual = x
        x_norm = self.attn_norm(x)
        attn_out = self.attention(x_norm, mask)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out
        
        # Feed-forward block with pre-norm and residual
        residual = x
        x_norm = self.ffn_norm(x)
        ff_out = self.feed_forward(x_norm)
        ff_out = self.dropout(ff_out)
        x = residual + ff_out
        
        return x