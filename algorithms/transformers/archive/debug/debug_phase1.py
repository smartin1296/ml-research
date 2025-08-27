#!/usr/bin/env python3
"""
Debug Phase 1 Training Failure
Minimal test to identify why accuracy is essentially 0%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer

def debug_tokenizer():
    """Test tokenizer on simple stories"""
    print("🔧 Testing Tokenizer...")
    
    # Simple test stories
    stories = [
        "Once upon a time, there was a little cat.",
        "The cat was very happy and played all day.",
        "One day, the cat found a big ball.",
        "The cat played with the ball in the garden."
    ]
    
    # Create tokenizer
    tokenizer = ScaledWordTokenizer(vocab_size=1000)
    tokenizer.build_vocab(stories)
    
    print(f"Vocab size: {len(tokenizer.word_to_idx)}")
    print(f"PAD token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    
    # Test encoding/decoding
    test_story = stories[0]
    tokens = tokenizer.encode(test_story, add_special=True)
    decoded = tokenizer.decode(tokens, skip_special=False)
    
    print(f"\nOriginal: {test_story}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    return tokenizer, stories

def debug_model_forward():
    """Test model forward pass"""
    print("\n🧠 Testing Model Forward Pass...")
    
    tokenizer, stories = debug_tokenizer()
    
    # Create small model
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=64,      # Very small for debugging
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    test_story = stories[0]
    tokens = tokenizer.encode(test_story, add_special=True)
    
    # Pad to length 16
    max_len = 16
    if len(tokens) < max_len:
        tokens.extend([tokenizer.pad_token] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]
    
    input_tokens = torch.tensor([tokens[:-1]], dtype=torch.long)  # Remove last for input
    target_tokens = torch.tensor([tokens[1:]], dtype=torch.long)  # Shift for target
    
    print(f"Input shape: {input_tokens.shape}")
    print(f"Target shape: {target_tokens.shape}")
    print(f"Input tokens: {input_tokens[0].tolist()}")
    print(f"Target tokens: {target_tokens[0].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tokens)
        print(f"Output shape: {output.shape}")
        
        # Check if any predictions are correct
        predictions = output.argmax(dim=-1)
        mask = (target_tokens != tokenizer.pad_token)
        correct = (predictions == target_tokens) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        print(f"Predictions: {predictions[0].tolist()}")
        print(f"Mask: {mask[0].tolist()}")
        print(f"Correct: {correct[0].tolist()}")
        print(f"Accuracy: {accuracy.item():.6f}")
    
    return model, tokenizer, input_tokens, target_tokens

def debug_loss_calculation():
    """Test loss calculation"""
    print("\n📉 Testing Loss Calculation...")
    
    model, tokenizer, input_tokens, target_tokens = debug_model_forward()
    
    # Forward pass
    output = model(input_tokens)
    
    # Calculate loss like trainer does
    loss = F.cross_entropy(
        output.contiguous().view(-1, output.size(-1)),
        target_tokens.contiguous().view(-1),
        ignore_index=tokenizer.pad_token
    )
    
    print(f"Loss: {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    
    # Check if gradients exist
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            if grad_norm > 0:
                print(f"{name}: grad_norm={grad_norm:.6f}")
    
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm < 1e-6:
        print("❌ PROBLEM: Gradients are too small or zero!")
    else:
        print("✅ Gradients look reasonable")

def debug_training_step():
    """Test one training step"""
    print("\n🏋️ Testing Training Step...")
    
    tokenizer, stories = debug_tokenizer()
    
    # Create model
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=64,
        num_heads=4, 
        num_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create batch from multiple stories
    batch_inputs = []
    batch_targets = []
    max_len = 20
    
    for story in stories:
        tokens = tokenizer.encode(story, add_special=True)
        if len(tokens) < max_len:
            tokens.extend([tokenizer.pad_token] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
            
        batch_inputs.append(tokens[:-1])
        batch_targets.append(tokens[1:])
    
    input_batch = torch.tensor(batch_inputs, dtype=torch.long)
    target_batch = torch.tensor(batch_targets, dtype=torch.long)
    
    print(f"Batch input shape: {input_batch.shape}")
    print(f"Batch target shape: {target_batch.shape}")
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_batch)
    loss = F.cross_entropy(
        output.contiguous().view(-1, output.size(-1)),
        target_batch.contiguous().view(-1),
        ignore_index=tokenizer.pad_token
    )
    
    print(f"Initial loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Check accuracy
    with torch.no_grad():
        predictions = output.argmax(dim=-1)
        mask = (target_batch != tokenizer.pad_token)
        correct = (predictions == target_batch) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        print(f"Initial accuracy: {accuracy.item():.6f} ({correct.sum().item()}/{mask.sum().item()})")
    
    # Try a few more steps
    for step in range(10):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = F.cross_entropy(
            output.contiguous().view(-1, output.size(-1)),
            target_batch.contiguous().view(-1),
            ignore_index=tokenizer.pad_token
        )
        loss.backward()
        optimizer.step()
        
        if step % 5 == 4:  # Every 5 steps
            with torch.no_grad():
                predictions = output.argmax(dim=-1)
                mask = (target_batch != tokenizer.pad_token)
                correct = (predictions == target_batch) & mask
                accuracy = correct.sum().float() / mask.sum().float()
                
                print(f"Step {step+1}: loss={loss.item():.4f}, accuracy={accuracy.item():.6f}")

if __name__ == "__main__":
    print("🐛 Debugging Phase 1 Training Failure on TinyStories")
    print("=" * 60)
    
    debug_training_step()