#!/usr/bin/env python3
"""Simple demonstration of token-level RNN."""

import torch
from pathlib import Path

from tokenizers import BPETokenizer, WordTokenizer
from token_dataset import load_shakespeare_data, TokenTextGenerator
from token_models import TokenRNNModel, get_token_model_config
from device_utils import get_best_device


def demo_token_rnn():
    """Quick demonstration of token-level RNN."""
    
    device = get_best_device()
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Load sample text
    print("Loading Shakespeare text...")
    text = load_shakespeare_data()
    
    # Limit text size for quick demo
    text = text[:100000]  # Use first 100k characters
    print(f"Text length: {len(text):,} characters")
    
    # Initialize BPE tokenizer
    print("\n1. BPE Tokenizer Demo")
    print("-" * 40)
    bpe = BPETokenizer(vocab_size=200, min_freq=2)
    print("Fitting BPE tokenizer...")
    bpe.fit(text)
    print(f"Vocabulary size: {len(bpe)}")
    
    # Show sample tokenization
    sample = "To be or not to be, that is the question"
    tokens = bpe.encode(sample)
    decoded = bpe.decode(tokens)
    print(f"\nOriginal: {sample}")
    print(f"Tokens: {tokens[:20]}...")
    print(f"Decoded: {decoded}")
    
    # Show vocabulary samples
    print(f"\nSample vocabulary (first 20):")
    for i, token in list(bpe.id_to_token.items())[:20]:
        if i >= 4:  # Skip special tokens
            print(f"  {i}: '{token}'")
    
    # Initialize Word tokenizer for comparison
    print("\n2. Word Tokenizer Demo")
    print("-" * 40)
    word = WordTokenizer(vocab_size=500, min_freq=2)
    print("Fitting Word tokenizer...")
    word.fit(text)
    print(f"Vocabulary size: {len(word)}")
    
    tokens = word.encode(sample)
    decoded = word.decode(tokens)
    print(f"\nOriginal: {sample}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Create a small model
    print("\n3. Token RNN Model")
    print("-" * 40)
    
    config = get_token_model_config(len(bpe), "small")
    model = TokenRNNModel(
        vocab_size=len(bpe),
        rnn_type="lstm",
        **config
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model config: {config}")
    
    # Test forward pass
    print("\n4. Testing Forward Pass")
    print("-" * 40)
    
    # Create sample batch
    batch_size = 4
    seq_len = 10
    sample_input = torch.randint(0, len(bpe), (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        logits, hidden = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Hidden shape: {hidden[0].shape if isinstance(hidden, tuple) else hidden.shape}")
    
    # Compare tokenization efficiency
    print("\n5. Tokenization Efficiency Comparison")
    print("-" * 40)
    
    bpe_tokens = bpe.encode(text)
    word_tokens = word.encode(text)
    char_tokens = list(text)  # Character-level
    
    print(f"Original text: {len(text):,} characters")
    print(f"Character-level: {len(char_tokens):,} tokens")
    print(f"BPE tokenization: {len(bpe_tokens):,} tokens (compression: {len(char_tokens)/len(bpe_tokens):.2f}x)")
    print(f"Word tokenization: {len(word_tokens):,} tokens (compression: {len(char_tokens)/len(word_tokens):.2f}x)")
    
    print("\n" + "=" * 60)
    print("Demo complete! Token-level RNN is ready for training.")
    print("\nTo train a full model, run:")
    print("  python -m algorithms.rnn.train_token")
    print("\nTo run benchmarks, run:")
    print("  python -m algorithms.rnn.test_token --mode benchmark")


if __name__ == "__main__":
    demo_token_rnn()