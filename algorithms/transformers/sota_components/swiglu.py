"""
Step 4: SwiGLU Activation
Superior feed-forward network using gated linear units
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network
    - Gated activation function (Swish-Gated Linear Unit)
    - Outperforms standard FFN with ReLU/GELU
    - Used in LLaMA, PaLM, and other modern models
    
    Architecture:
        x -> gate_proj -> SiLU ⌉
                                 × -> down_proj -> output
        x -> up_proj ----------⌋
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Three projections instead of two (for gating)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Gate path: apply SiLU (Swish) activation
        gate = F.silu(self.gate_proj(x))
        
        # Up path: linear projection
        up = self.up_proj(x)
        
        # Combine with element-wise multiplication
        hidden = gate * up
        
        # Project back to model dimension
        output = self.down_proj(hidden)
        
        return output