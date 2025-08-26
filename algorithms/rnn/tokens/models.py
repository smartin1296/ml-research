"""Token-level RNN models optimized for larger vocabularies."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import LSTM, GRU, VanillaRNN


class TokenRNNModel(nn.Module):
    """RNN model for token-level next token prediction."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.2,
        tie_weights: bool = True,
        layer_norm: bool = True,
        bidirectional: bool = False,
    ):
        """Initialize token RNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of token embeddings
            hidden_size: Hidden size of RNN
            num_layers: Number of RNN layers
            rnn_type: Type of RNN (lstm, gru, vanilla)
            dropout: Dropout probability
            tie_weights: Whether to tie input/output embeddings
            layer_norm: Whether to use layer normalization
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.tie_weights = tie_weights
        self.bidirectional = bidirectional
        
        # Token embeddings with proper initialization
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Dropout layers
        self.embed_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # RNN layer
        rnn_dropout = 0.0 if num_layers == 1 else dropout
        
        if rnn_type == "lstm":
            self.rnn = LSTM(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                layer_norm=layer_norm,
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = GRU(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                layer_norm=layer_norm,
                bidirectional=bidirectional,
            )
        elif rnn_type == "vanilla":
            self.rnn = VanillaRNN(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=rnn_dropout,
                layer_norm=layer_norm,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output projection
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        
        if tie_weights and embed_dim == rnn_output_size:
            # Tie weights between input and output embeddings
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(rnn_output_size, vocab_size)
            nn.init.uniform_(self.output_projection.weight, -0.1, 0.1)
            nn.init.zeros_(self.output_projection.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            hidden: Hidden state from previous forward pass
            
        Returns:
            Tuple of (logits, hidden_state)
        """
        # Embed tokens
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = self.embed_dropout(x)
        
        # Pass through RNN
        rnn_output, hidden = self.rnn(x, hidden)  # [batch_size, seq_len, hidden_size]
        rnn_output = self.output_dropout(rnn_output)
        
        # Project to vocabulary
        if self.output_projection is not None:
            logits = self.output_projection(rnn_output)
        else:
            # Use tied weights
            logits = F.linear(rnn_output, self.embedding.weight)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Initial hidden state
        """
        num_directions = 2 if self.bidirectional else 1
        hidden_size = self.hidden_size
        
        if self.rnn_type == "lstm":
            h0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                hidden_size,
                device=device,
            )
            c0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                hidden_size,
                device=device,
            )
            return (h0, c0)
        else:
            h0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                hidden_size,
                device=device,
            )
            return h0


class OptimizedTokenRNNModel(nn.Module):
    """Optimized token RNN model with advanced features."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        adaptive_softmax_cutoff: Optional[list] = None,
        tie_weights: bool = True,
        layer_norm: bool = True,
    ):
        """Initialize optimized token RNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of token embeddings
            hidden_size: Hidden size of RNN
            num_layers: Number of RNN layers
            dropout: Dropout probability
            adaptive_softmax_cutoff: Cutoff points for adaptive softmax
            tie_weights: Whether to tie input/output embeddings
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        
        # Token embeddings with scaled initialization
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        scale = 1.0 / math.sqrt(embed_dim)
        nn.init.uniform_(self.embedding.weight, -scale, scale)
        
        # Input layer norm (helps with stability)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Dropout layers
        self.embed_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # LSTM with layer normalization (proven to work well)
        rnn_dropout = 0.0 if num_layers == 1 else dropout
        self.rnn = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            layer_norm=layer_norm,
            bidirectional=False,  # Keep unidirectional for autoregressive
        )
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        if adaptive_softmax_cutoff is not None:
            # Use adaptive softmax for large vocabularies
            self.output_projection = nn.AdaptiveLogSoftmaxWithLoss(
                hidden_size,
                vocab_size,
                cutoffs=adaptive_softmax_cutoff,
                div_value=4.0,
            )
        else:
            if tie_weights and embed_dim == hidden_size:
                self.output_projection = None
            else:
                self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
                scale = 1.0 / math.sqrt(hidden_size)
                nn.init.uniform_(self.output_projection.weight, -scale, scale)
        
        self.adaptive_softmax = adaptive_softmax_cutoff is not None
        
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            hidden: Hidden state from previous forward pass
            targets: Target token IDs for adaptive softmax [batch_size, seq_len]
            
        Returns:
            Tuple of (logits/loss, hidden_state)
        """
        # Embed tokens
        x = self.embedding(input_ids)
        x = self.input_norm(x)
        x = self.embed_dropout(x)
        
        # Pass through RNN
        rnn_output, hidden = self.rnn(x, hidden)
        rnn_output = self.output_norm(rnn_output)
        rnn_output = self.output_dropout(rnn_output)
        
        # Project to vocabulary
        if self.adaptive_softmax:
            if targets is not None:
                # Return loss directly for adaptive softmax
                output, loss = self.output_projection(
                    rnn_output.view(-1, self.hidden_size),
                    targets.view(-1),
                )
                return loss, hidden
            else:
                # Return log probabilities for generation
                output = self.output_projection.log_prob(
                    rnn_output.view(-1, self.hidden_size)
                )
                return output.view(rnn_output.shape[0], rnn_output.shape[1], -1), hidden
        else:
            if self.output_projection is not None:
                logits = self.output_projection(rnn_output)
            else:
                # Use tied weights
                logits = F.linear(rnn_output, self.embedding.weight)
            
            return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (h0, c0)


def get_token_model_config(vocab_size: int, model_size: str = "medium") -> dict:
    """Get recommended model configuration based on vocabulary size.
    
    Args:
        vocab_size: Size of vocabulary
        model_size: Model size (small, medium, large)
        
    Returns:
        Model configuration dictionary
    """
    configs = {
        "small": {
            "embed_dim": 128,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "medium": {
            "embed_dim": 256,
            "hidden_size": 512,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "large": {
            "embed_dim": 512,
            "hidden_size": 1024,
            "num_layers": 3,
            "dropout": 0.3,
        },
    }
    
    config = configs[model_size].copy()
    
    # Add adaptive softmax for very large vocabularies
    if vocab_size > 10000:
        # Set cutoff points for adaptive softmax
        if vocab_size > 50000:
            config["adaptive_softmax_cutoff"] = [4000, 20000, 50000]
        elif vocab_size > 20000:
            config["adaptive_softmax_cutoff"] = [2000, 10000]
        else:
            config["adaptive_softmax_cutoff"] = [2000, 5000]
    
    # Adjust embedding dimension for tied weights
    if model_size == "medium" and vocab_size < 5000:
        config["embed_dim"] = config["hidden_size"]  # Enable weight tying
    
    return config