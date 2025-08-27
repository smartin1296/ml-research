#!/usr/bin/env python3
"""
Multi-Head Attention Implementation
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from "Attention is All You Need"
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights following original paper
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights as in original paper"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def scaled_dot_product_attention(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention
        
        Args:
            query: [batch_size, num_heads, seq_len, d_k]
            key: [batch_size, num_heads, seq_len, d_k] 
            value: [batch_size, num_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len, seq_len] or broadcastable
        
        Returns:
            output: [batch_size, num_heads, seq_len, d_k]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or broadcastable
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Reshape mask for multi-head attention if provided
        if mask is not None and mask.dim() == 3:  # [batch_size, seq_len, seq_len]
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
        
        # Apply scaled dot-product attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(attn_output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need"
    
    Args:
        d_model: Model dimension
        max_seq_length: Maximum sequence length to precompute encodings for
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        # Compute div_term for sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            x + positional encoding: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


def create_padding_mask(sequences: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Create padding mask for attention
    
    Args:
        sequences: [batch_size, seq_len] - token sequences
        pad_token: Token ID used for padding
    
    Returns:
        mask: [batch_size, 1, 1, seq_len] - 0 for padded positions, 1 otherwise
    """
    mask = (sequences != pad_token).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
    return mask.float()


def create_look_ahead_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create look-ahead mask for decoder self-attention
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
    
    Returns:
        mask: [1, 1, seq_len, seq_len] - Upper triangular mask
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return (mask == 0).unsqueeze(0).unsqueeze(0).float()