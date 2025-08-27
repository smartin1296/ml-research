#!/usr/bin/env python3
"""
Quick Phase 1 vs Phase 2 Comparison
Faster training for immediate comparison results
"""

import sys
import torch
import torch.nn.functional as F
import json
import time
import math
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer
from torch.utils.data import Dataset, DataLoader

class QuickDataset(Dataset):
    """Quick dataset for comparison"""
    
    def __init__(self, data_path: str, tokenizer: ScaledWordTokenizer, max_length: int = 32, subset_size: int = 1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load smaller subset for quick comparison
        stories = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= subset_size:
                    break
                if line.strip():
                    try:
                        if line.strip().startswith('{'):
                            data = json.loads(line.strip())
                            story = data.get('story', data.get('text', ''))
                        else:
                            story = line.strip()
                        
                        if story and 50 <= len(story) <= 500:
                            stories.append(story.strip())
                    except:
                        continue
        
        # Pre-tokenize
        self.tokenized_stories = []
        for story in stories:
            tokens = tokenizer.encode(story, add_special=True)
            if 5 <= len(tokens) <= max_length:
                self.tokenized_stories.append(tokens)
        
        print(f"Quick dataset: {len(self.tokenized_stories)} stories")
    
    def __len__(self):
        return len(self.tokenized_stories)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_stories[idx].copy()
        
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

def label_smoothing_loss(pred, target, smoothing=0.1, ignore_index=0):
    """Label smoothing loss"""
    vocab_size = pred.size(-1)
    confidence = 1.0 - smoothing
    
    smooth_target = torch.full_like(pred, smoothing / vocab_size)
    mask = (target != ignore_index)
    target_masked = target[mask]
    pred_masked = pred[mask]
    smooth_target_masked = smooth_target[mask]
    
    if len(target_masked) == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    smooth_target_masked.scatter_(-1, target_masked.unsqueeze(-1), confidence)
    loss = F.kl_div(F.log_softmax(pred_masked, dim=-1), smooth_target_masked, reduction='batchmean')
    return loss

def train_phase(phase_name, use_optimizations=False, epochs=5):
    """Train a single phase"""
    print(f"\\n🚀 Training {phase_name}")
    print("=" * 40)
    
    # Load data and tokenizer
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    
    # Create or load tokenizer
    if phase_name == "Phase 1":
        # Create fresh tokenizer for Phase 1
        stories = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 2000:
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
        
        tokenizer = ScaledWordTokenizer(vocab_size=1000)
        tokenizer.build_vocab(stories)
    else:
        # Load Phase 1 tokenizer for fair comparison
        phase1_dir = Path("algorithms/transformers/results/phases/phase_1_FIXED_20250826_185157")
        tokenizer_path = phase1_dir / "tokenizer.json"
        
        if tokenizer_path.exists():
            tokenizer = ScaledWordTokenizer()
            tokenizer.load(str(tokenizer_path))
            print(f"Loaded Phase 1 tokenizer: {len(tokenizer.word_to_idx)} words")
        else:
            print("❌ Phase 1 tokenizer not found, creating new one")
            return None
    
    # Create dataset
    dataset = QuickDataset(str(data_path), tokenizer, max_length=32, subset_size=1000)
    
    # Split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # Create model
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=128,  # Smaller for speed
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_len=32
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    
    # Setup optimizer
    if use_optimizations:
        # Phase 2 optimizations
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98)
        )
        use_label_smoothing = True
        print("Using Phase 2 optimizations: label smoothing, enhanced AdamW")
    else:
        # Phase 1 baseline
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        use_label_smoothing = False
        print("Using Phase 1 baseline: standard training")
    
    # Training
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    training_start = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            output = model(input_batch)
            
            if use_label_smoothing:
                loss = label_smoothing_loss(
                    output.contiguous().view(-1, output.size(-1)),
                    target_batch.contiguous().view(-1),
                    smoothing=0.1,
                    ignore_index=tokenizer.pad_token
                )
            else:
                loss = F.cross_entropy(
                    output.contiguous().view(-1, output.size(-1)),
                    target_batch.contiguous().view(-1),
                    ignore_index=tokenizer.pad_token
                )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_tokens = 0
        val_batches = 0
        
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                
                output = model(input_batch)
                loss = F.cross_entropy(
                    output.contiguous().view(-1, output.size(-1)),
                    target_batch.contiguous().view(-1),
                    ignore_index=tokenizer.pad_token
                )
                
                val_loss += loss.item()
                val_batches += 1
                
                predictions = output.argmax(dim=-1)
                mask = (target_batch != tokenizer.pad_token)
                correct = (predictions == target_batch) & mask
                
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_val_loss = val_loss / val_batches
        val_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.3f}")
    
    training_time = time.time() - training_start
    
    # Test generation
    print("Testing story generation...")
    model.eval()
    
    test_prompts = ["Once upon a time", "There was a little"]
    generated_stories = []
    
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt, add_special=False)
            generated_tokens = tokens.copy()
            
            for _ in range(20):
                input_tensor = torch.tensor([generated_tokens], device=device)
                output = model(input_tensor)
                
                # Sample with temperature
                logits = output[0, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token in [tokenizer.eos_token, tokenizer.pad_token]:
                    break
                generated_tokens.append(next_token)
            
            story = tokenizer.decode(generated_tokens, skip_special=True)
            generated_stories.append(story)
            print(f"'{prompt}' -> {story}")
    
    return {
        "phase": phase_name,
        "training_time": training_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_val_accuracy": val_accuracies[-1],
        "parameters": param_count,
        "generated_stories": generated_stories,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

def compare_phases():
    """Run quick comparison between Phase 1 and Phase 2"""
    
    print("🔍 QUICK PHASE COMPARISON: Phase 1 vs Phase 2")
    print("=" * 60)
    print("Comparing baseline vs optimized training on TinyStories")
    
    # Check if data exists
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    if not data_path.exists():
        print(f"❌ TinyStories not found at {data_path}")
        return
    
    # Run Phase 1 (baseline)
    phase1_results = train_phase("Phase 1", use_optimizations=False, epochs=5)
    
    # Run Phase 2 (optimized)  
    phase2_results = train_phase("Phase 2", use_optimizations=True, epochs=5)
    
    # Comparison
    print("\\n📊 PHASE COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"Phase 1 (Baseline):")
    print(f"  Final Val Loss:     {phase1_results['final_val_loss']:.4f}")
    print(f"  Final Val Accuracy: {phase1_results['final_val_accuracy']:.3f}")
    print(f"  Training Time:      {phase1_results['training_time']:.1f}s")
    print(f"  Parameters:         {phase1_results['parameters']:,}")
    
    print(f"\\nPhase 2 (Optimized):")
    print(f"  Final Val Loss:     {phase2_results['final_val_loss']:.4f}")  
    print(f"  Final Val Accuracy: {phase2_results['final_val_accuracy']:.3f}")
    print(f"  Training Time:      {phase2_results['training_time']:.1f}s")
    print(f"  Parameters:         {phase2_results['parameters']:,}")
    
    # Improvements
    loss_improvement = phase1_results['final_val_loss'] - phase2_results['final_val_loss']
    acc_improvement = phase2_results['final_val_accuracy'] - phase1_results['final_val_accuracy']
    
    print(f"\\n🎯 IMPROVEMENTS (Phase 2 vs Phase 1):")
    print(f"  Validation Loss:    {loss_improvement:+.4f} ({'better' if loss_improvement > 0 else 'worse'})")
    print(f"  Validation Accuracy: {acc_improvement:+.3f} ({'better' if acc_improvement > 0 else 'worse'})")
    
    print(f"\\n📖 STORY GENERATION COMPARISON:")
    
    for i, (prompt, story1, story2) in enumerate(zip(
        ["Once upon a time", "There was a little"],
        phase1_results['generated_stories'],
        phase2_results['generated_stories']
    )):
        print(f"\\nPrompt: '{prompt}'")
        print(f"  Phase 1: {story1}")
        print(f"  Phase 2: {story2}")
    
    # Save comparison
    comparison = {
        "phase_1": phase1_results,
        "phase_2": phase2_results,
        "improvements": {
            "val_loss_improvement": loss_improvement,
            "val_accuracy_improvement": acc_improvement
        }
    }
    
    with open("quick_phase_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\\n💾 Comparison saved to quick_phase_comparison.json")
    
    return comparison

if __name__ == "__main__":
    compare_phases()