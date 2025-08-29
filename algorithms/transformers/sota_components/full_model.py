"""
Step 8: Complete SOTA Transformer Model
High-level orchestrator that assembles all components
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import SOTAConfig
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock
from .embeddings import TokenEmbeddings, OutputProjection


class SOTATransformer(nn.Module):
    """
    State-of-the-Art Transformer Model
    
    This is the main orchestrator that brings together:
    1. Configuration
    2. Token embeddings
    3. Stack of transformer blocks
    4. Final normalization
    5. Output projection
    
    Architecture flow:
        Input IDs
            ↓
        Token Embeddings
            ↓
        Transformer Block 1 (Attention + SwiGLU)
            ↓
        Transformer Block 2
            ↓
            ...
            ↓
        Transformer Block N
            ↓
        Final RMSNorm
            ↓
        Output Projection
            ↓
        Logits
    """
    
    def __init__(self, config: SOTAConfig):
        super().__init__()
        self.config = config
        
        # Auto-compute feed-forward dimension if needed
        if config.d_ff is None:
            config.d_ff = int(8 * config.d_model / 3)
            config.d_ff = ((config.d_ff + 255) // 256) * 256
        
        # Step 1: Token embeddings
        self.token_embeddings = TokenEmbeddings(config.vocab_size, config.d_model)
        
        # Step 2: Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
                dropout=config.dropout,
                eps=config.eps,
                use_flash_attn=config.use_flash_attn
            )
            for _ in range(config.num_layers)
        ])
        
        # Step 3: Final normalization
        self.final_norm = RMSNorm(config.d_model, config.eps)
        
        # Step 4: Output projection
        self.output_projection = OutputProjection(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            tie_weights=config.tie_embeddings,
            embedding_layer=self.token_embeddings.embedding if config.tie_embeddings else None
        )
        
        # Initialize weights
        self._init_weights()
        
        # Report model size
        self._report_parameters()
    
    def _init_weights(self):
        """Initialize all weights with standard deviation 0.02"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _report_parameters(self):
        """Report total parameter count"""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"SOTA Transformer initialized: {total_params:,} parameters")
        print(f"Configuration: {self.config.num_layers} layers, "
              f"{self.config.d_model} dim, {self.config.num_heads} heads")
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal attention mask
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            Causal mask [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
        
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = self.create_causal_mask(L, device)
        
        # Step 1: Token embeddings
        x = self.token_embeddings(input_ids)
        
        # Step 2: Pass through transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                # Memory-efficient training
                x = torch.utils.checkpoint.checkpoint(block, x, attention_mask)
            else:
                x = block(x, attention_mask)
        
        # Step 3: Final normalization
        x = self.final_norm(x)
        
        # Step 4: Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Generate text using the model
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we exceed max length
            if input_ids.shape[1] >= self.config.max_seq_len:
                break
        
        return input_ids