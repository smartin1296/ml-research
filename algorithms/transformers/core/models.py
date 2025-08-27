#!/usr/bin/env python3
"""
Transformer Model Implementation
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .attention import MultiHeadAttention, PositionalEncoding


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with residual connections and layer normalization
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Attention mask
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with masked self-attention and encoder-decoder attention
        
        Args:
            x: [batch_size, seq_len, d_model] - Decoder input
            encoder_output: [batch_size, enc_seq_len, d_model] - Encoder output
            self_attn_mask: Look-ahead mask for decoder self-attention
            encoder_mask: Padding mask for encoder-decoder attention
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Masked self-attention with residual connection and layer norm
        self_attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-decoder attention with residual connection and layer norm
        enc_attn_output = self.encoder_attention(x, encoder_output, encoder_output, encoder_mask)
        x = self.norm2(x + self.dropout(enc_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Stack
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder stack
        
        Args:
            x: [batch_size, seq_len] - Token IDs
            mask: [batch_size, seq_len, seq_len] - Attention mask
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Embedding + positional encoding
        x = self.embedding(x) * (self.d_model ** 0.5)  # Scale embeddings
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder Stack
    
    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder stack
        
        Args:
            x: [batch_size, seq_len] - Target token IDs
            encoder_output: [batch_size, enc_seq_len, d_model] - Encoder output
            self_attn_mask: Look-ahead mask for decoder self-attention
            encoder_mask: Padding mask for encoder-decoder attention
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Embedding + positional encoding
        x = self.embedding(x) * (self.d_model ** 0.5)  # Scale embeddings
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, encoder_mask)
        
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model from "Attention is All You Need"
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size  
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 6)
        num_decoder_layers: Number of decoder layers (default: 6)
        d_ff: Feed-forward dimension (default: 2048)
        max_seq_length: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, 
            src_vocab_size, max_seq_length, dropout
        )
        
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff,
            tgt_vocab_size, max_seq_length, dropout
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through complete Transformer
        
        Args:
            src: [batch_size, src_seq_len] - Source token IDs
            tgt: [batch_size, tgt_seq_len] - Target token IDs
            src_mask: Padding mask for source sequence
            tgt_mask: Look-ahead + padding mask for target sequence  
            encoder_mask: Padding mask for encoder-decoder attention
        
        Returns:
            output: [batch_size, tgt_seq_len, tgt_vocab_size] - Logits
        """
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target sequence  
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, encoder_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)