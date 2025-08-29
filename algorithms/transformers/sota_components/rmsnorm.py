"""
Step 2: RMSNorm (Root Mean Square Normalization)
More efficient than LayerNorm, used in modern LLMs like LLaMA
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    - More efficient than standard LayerNorm
    - No mean centering, only variance normalization
    - Used in LLaMA, Gemma, and other modern models
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Normalized tensor with same shape
        """
        # Compute RMS
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(rms + self.eps)
        
        # Apply learned scaling
        return self.weight * x_normed