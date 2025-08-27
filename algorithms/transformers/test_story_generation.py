#!/usr/bin/env python3
"""
Test improved story generation to fix repetition issues
"""

import torch
import torch.nn.functional as F
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer

def generate_story_improved(model, tokenizer, device, prompt, max_length=50, temperature=0.8, top_k=40, repetition_penalty=1.2):
    """
    Improved story generation with multiple techniques to prevent repetition
    """
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special=False)
    generated_tokens = tokens.copy()
    
    # Track recent tokens to penalize repetition
    recent_tokens = set()
    
    with torch.no_grad():
        for step in range(max_length):
            if len(generated_tokens) >= 100:  # Hard limit
                break
            
            # Prepare input (limit context to avoid memory issues)
            context_tokens = generated_tokens[-50:] if len(generated_tokens) > 50 else generated_tokens
            input_tensor = torch.tensor([context_tokens], device=device)
            
            # Forward pass
            output = model(input_tensor)
            logits = output[0, -1, :]  # Last position logits
            
            # Apply repetition penalty for recently used tokens
            for token in recent_tokens:
                if token < len(logits):
                    logits[token] = logits[token] / repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all non-top-k to very low value
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(0, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop conditions
            if next_token == tokenizer.eos_token:
                break
            if next_token == tokenizer.pad_token:
                break
            
            generated_tokens.append(next_token)
            
            # Update recent tokens (sliding window)
            recent_tokens.add(next_token)
            if len(recent_tokens) > 10:  # Keep last 10 tokens for penalty
                # Remove oldest (this is approximate since sets are unordered)
                recent_tokens.pop()
            
            # Stop if we see too much repetition
            if len(generated_tokens) > 10:
                last_5 = generated_tokens[-5:]
                if len(set(last_5)) == 1:  # All same token
                    break
    
    # Decode story
    story = tokenizer.decode(generated_tokens, skip_special=True)
    return story

def test_generation_methods():
    """Test different generation methods"""
    print("🎨 Testing Story Generation Methods")
    print("=" * 50)
    
    # Load the fixed Phase 1 model
    results_dir = Path("algorithms/transformers/results/phases/phase_1_FIXED_20250826_185157")
    model_path = results_dir / "best_model.pt"
    tokenizer_path = results_dir / "tokenizer.json"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load tokenizer
    tokenizer = ScaledWordTokenizer()
    tokenizer.load(str(tokenizer_path))
    
    print(f"Loaded tokenizer with {len(tokenizer.word_to_idx)} words")
    
    # Create model
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=256,
        num_heads=8, 
        num_layers=4,
        d_ff=1024,
        max_seq_len=64
    )
    
    # Load trained weights
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model on {device}")
    
    # Test prompts
    test_prompts = [
        "Once upon a time",
        "There was a little girl",
        "In a magical forest",
        "The brave princess",
        "One sunny day"
    ]
    
    print("\n📖 Testing Different Generation Methods")
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Method 1: Greedy (what we had before)
        print("Greedy:", end=" ")
        tokens = tokenizer.encode(prompt, add_special=False)
        generated_tokens = tokens.copy()
        
        model.eval()
        with torch.no_grad():
            for _ in range(20):
                input_tensor = torch.tensor([generated_tokens], device=device)
                output = model(input_tensor)
                next_token = output[0, -1, :].argmax().item()
                
                if next_token in [tokenizer.eos_token, tokenizer.pad_token]:
                    break
                generated_tokens.append(next_token)
        
        greedy_story = tokenizer.decode(generated_tokens, skip_special=True)
        print(greedy_story)
        
        # Method 2: Improved generation
        print("Improved:", end=" ")
        improved_story = generate_story_improved(
            model, tokenizer, device, prompt, 
            max_length=30, temperature=0.9, top_k=20, repetition_penalty=1.3
        )
        print(improved_story)

if __name__ == "__main__":
    test_generation_methods()