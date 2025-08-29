"""
Step 7: Token Embeddings
Input and output embeddings with optional weight tying
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenEmbeddings(nn.Module):
    """
    Token embedding layer
    - Converts token IDs to dense vectors
    - Optional weight initialization strategies
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize with proper scale
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with normal distribution"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings
        
        Args:
            input_ids: Token IDs [batch, seq_len]
        
        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        return self.embedding(input_ids)


class OutputProjection(nn.Module):
    """
    Output projection layer
    - Projects from model dimension to vocabulary
    - Can optionally tie weights with input embeddings
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: bool = False,
        embedding_layer: Optional[nn.Embedding] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        if tie_weights:
            assert embedding_layer is not None, "Must provide embedding layer for weight tying"
            self.embedding_layer = embedding_layer
            self.projection = None
        else:
            self.projection = nn.Linear(d_model, vocab_size, bias=False)
            self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        if self.projection is not None:
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to vocabulary dimension
        
        Args:
            x: Hidden states [batch, seq_len, d_model]
        
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        if self.tie_weights:
            # Use transposed embedding weights
            return F.linear(x, self.embedding_layer.weight)
        else:
            return self.projection(x)