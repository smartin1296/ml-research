"""
Demo: SOTA Transformer Architecture Walkthrough
Shows how all components work together
"""

import torch
from sota_components.config import SOTAConfig
from sota_components.full_model import SOTATransformer


def demo():
    """Demonstrate the SOTA transformer architecture"""
    
    print("=" * 60)
    print("SOTA TRANSFORMER ARCHITECTURE DEMO")
    print("=" * 60)
    
    # Step 1: Configuration
    print("\n1. CONFIGURATION")
    print("-" * 40)
    config = SOTAConfig(
        vocab_size=1000,    # Small vocab for demo
        d_model=256,        # Model dimension
        num_heads=8,        # Attention heads
        num_layers=4,       # Transformer blocks
        max_seq_len=128,    # Maximum sequence length
        dropout=0.0         # No dropout for demo
    )
    print(f"Model config: {config.num_layers} layers, {config.d_model} dim, {config.num_heads} heads")
    print(f"Feed-forward dim (auto-computed): {config.d_ff}")
    
    # Step 2: Model Creation
    print("\n2. MODEL ARCHITECTURE")
    print("-" * 40)
    model = SOTATransformer(config)
    
    # Step 3: Component Breakdown
    print("\n3. COMPONENT BREAKDOWN")
    print("-" * 40)
    print("The model consists of:")
    print("  → Token Embeddings (vocab_size × d_model)")
    print("  → Stack of Transformer Blocks:")
    for i in range(config.num_layers):
        print(f"     Block {i+1}:")
        print("       • RMSNorm (pre-attention)")
        print("       • Multi-Head Attention with RoPE")
        print("       • RMSNorm (pre-FFN)")
        print("       • SwiGLU Feed-Forward Network")
    print("  → Final RMSNorm")
    print("  → Output Projection (d_model × vocab_size)")
    
    # Step 4: Forward Pass Demo
    print("\n4. FORWARD PASS DEMO")
    print("-" * 40)
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"Input shape: {input_ids.shape} (batch_size={batch_size}, seq_len={seq_len})")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    print(f"Output shape: {logits.shape} (batch_size, seq_len, vocab_size)")
    
    # Step 5: Generation Demo
    print("\n5. GENERATION DEMO")
    print("-" * 40)
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    print(f"Prompt tokens: {prompt[0].tolist()}")
    
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    
    # Step 6: Architecture Flow
    print("\n6. ARCHITECTURE FLOW")
    print("-" * 40)
    print("""
    Input Token IDs
         ↓
    Token Embeddings (learned vectors)
         ↓
    ┌─────────────────────────┐
    │  Transformer Block 1    │
    │  ├─ RMSNorm            │
    │  ├─ Attention + RoPE   │
    │  ├─ Residual          │
    │  ├─ RMSNorm           │
    │  ├─ SwiGLU FFN        │
    │  └─ Residual          │
    └─────────────────────────┘
         ↓
    ... (more blocks) ...
         ↓
    Final RMSNorm
         ↓
    Output Projection
         ↓
    Logits (vocabulary probabilities)
    """)
    
    print("\n" + "=" * 60)
    print("✅ Demo complete! This is a state-of-the-art transformer")
    print("   combining all modern innovations in a clean architecture.")
    print("=" * 60)


if __name__ == "__main__":
    demo()