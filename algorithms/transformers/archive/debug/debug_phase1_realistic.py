#!/usr/bin/env python3
"""
Debug Phase 1 with Realistic Settings
Test with actual TinyStories data and Phase 1 hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer

def load_tinystories_sample():
    """Load a small sample of actual TinyStories"""
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    
    if not data_path.exists():
        print(f"❌ TinyStories not found at {data_path}")
        print("Using fallback stories...")
        return [
            "Once upon a time, there was a little girl named Anna. She loved to play in the garden with her toys.",
            "One day, Anna found a big red ball. She was very happy and decided to play with it all day long.",
            "The ball bounced high into the sky. Anna laughed and ran after it through the green grass.",
            "When evening came, Anna's mom called her for dinner. She put the ball away and went inside.",
            "Anna told her mom about the wonderful day she had with her new ball. They smiled together."
        ] * 100  # Repeat to get more training data
    
    stories = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Just use first 1000 for testing
                break
            if line.strip():
                try:
                    if line.strip().startswith('{'):
                        data = json.loads(line.strip())
                        story = data.get('story', data.get('text', ''))
                    else:
                        story = line.strip()
                    
                    if story and len(story) > 50:
                        stories.append(story)
                except:
                    continue
    
    print(f"📖 Loaded {len(stories)} TinyStories")
    return stories

def test_realistic_settings():
    """Test with Phase 1's actual settings"""
    print("🧪 Testing Phase 1 Realistic Settings")
    print("=" * 50)
    
    # Load stories
    stories = load_tinystories_sample()
    
    # Create tokenizer with Phase 1 settings
    tokenizer = ScaledWordTokenizer(vocab_size=1000)  # Start smaller to debug
    tokenizer.build_vocab(stories[:500])  # Build vocab from subset
    
    print(f"Tokenizer vocab size: {len(tokenizer.word_to_idx)}")
    
    # Create model with Phase 1 architecture
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=256,     # Phase 1 setting
        num_heads=8,     # Phase 1 setting
        num_layers=4,    # Phase 1 setting
        d_ff=1024,       # Phase 1 setting
        max_seq_len=64   # Phase 1 sequence length
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create training batch
    batch_size = 16  # Smaller for debugging
    max_len = 32     # Shorter sequences for debugging
    
    batch_inputs = []
    batch_targets = []
    
    for i, story in enumerate(stories[:batch_size]):
        tokens = tokenizer.encode(story, add_special=True)
        
        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([tokenizer.pad_token] * (max_len - len(tokens)))
        
        batch_inputs.append(tokens[:-1])
        batch_targets.append(tokens[1:])
    
    input_batch = torch.tensor(batch_inputs, dtype=torch.long)
    target_batch = torch.tensor(batch_targets, dtype=torch.long)
    
    print(f"Input batch shape: {input_batch.shape}")
    
    # Test different learning rates
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    
    for lr in learning_rates:
        print(f"\n📊 Testing learning rate: {lr}")
        
        # Fresh model copy
        test_model = SimpleTransformer(
            vocab_size=len(tokenizer.word_to_idx),
            d_model=256, num_heads=8, num_layers=4, d_ff=1024, max_seq_len=64
        )
        
        optimizer = torch.optim.AdamW(test_model.parameters(), lr=lr)
        
        # Initial accuracy
        test_model.eval()
        with torch.no_grad():
            output = test_model(input_batch)
            predictions = output.argmax(dim=-1)
            mask = (target_batch != tokenizer.pad_token)
            correct = (predictions == target_batch) & mask
            initial_acc = correct.sum().float() / mask.sum().float()
        
        print(f"  Initial accuracy: {initial_acc.item():.6f}")
        
        # Train for 20 steps
        test_model.train()
        for step in range(20):
            optimizer.zero_grad()
            
            output = test_model(input_batch)
            loss = F.cross_entropy(
                output.contiguous().view(-1, output.size(-1)),
                target_batch.contiguous().view(-1),
                ignore_index=tokenizer.pad_token
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(test_model.parameters(), 1.0)
            optimizer.step()
            
            if step % 10 == 9:
                test_model.eval()
                with torch.no_grad():
                    output = test_model(input_batch)
                    predictions = output.argmax(dim=-1)
                    mask = (target_batch != tokenizer.pad_token)
                    correct = (predictions == target_batch) & mask
                    accuracy = correct.sum().float() / mask.sum().float()
                
                print(f"  Step {step+1}: loss={loss.item():.4f}, accuracy={accuracy.item():.6f}")
                test_model.train()

def test_large_vocab():
    """Test what happens with Phase 1's large vocab"""
    print("\n🎯 Testing Large Vocabulary (8192)")
    print("=" * 50)
    
    stories = load_tinystories_sample()
    
    # Create large vocab tokenizer like Phase 1
    tokenizer = ScaledWordTokenizer(vocab_size=8192)
    tokenizer.build_vocab(stories)
    
    print(f"Large tokenizer vocab size: {len(tokenizer.word_to_idx)}")
    
    # Create model
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=256, num_heads=8, num_layers=4, d_ff=1024, max_seq_len=64
    )
    
    print(f"Large model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create batch
    batch_size = 8  # Smaller due to large model
    max_len = 32
    
    batch_inputs = []
    batch_targets = []
    
    for story in stories[:batch_size]:
        tokens = tokenizer.encode(story, add_special=True)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([tokenizer.pad_token] * (max_len - len(tokens)))
        
        batch_inputs.append(tokens[:-1])
        batch_targets.append(tokens[1:])
    
    input_batch = torch.tensor(batch_inputs, dtype=torch.long)
    target_batch = torch.tensor(batch_targets, dtype=torch.long)
    
    # Test higher learning rate for large vocab
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Higher LR
    
    print(f"Training large vocab model...")
    
    for step in range(50):
        optimizer.zero_grad()
        
        output = model(input_batch)
        loss = F.cross_entropy(
            output.contiguous().view(-1, output.size(-1)),
            target_batch.contiguous().view(-1),
            ignore_index=tokenizer.pad_token
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 9:
            model.eval()
            with torch.no_grad():
                output = model(input_batch)
                predictions = output.argmax(dim=-1)
                mask = (target_batch != tokenizer.pad_token)
                correct = (predictions == target_batch) & mask
                accuracy = correct.sum().float() / mask.sum().float()
            
            print(f"Step {step+1}: loss={loss.item():.4f}, accuracy={accuracy.item():.6f}")
            model.train()

if __name__ == "__main__":
    print("🔍 Debugging Phase 1 with Realistic TinyStories Settings")
    print("=" * 60)
    
    test_realistic_settings()
    test_large_vocab()