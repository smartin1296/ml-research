#!/usr/bin/env python3
"""
Standard Phase Comparison: Fair Phase 1 vs Phase 2 Evaluation
Uses consistent tokenizer across both phases for accurate comparison
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

from algorithms.transformers.standard_tokenizer import StandardTransformerTokenizer
from algorithms.transformers.test_basic import SimpleTransformer
from torch.utils.data import Dataset, DataLoader

class StandardDataset(Dataset):
    """Consistent dataset for both phases"""
    
    def __init__(self, data_path: str, tokenizer: StandardTransformerTokenizer, max_length: int = 48, subset_size: int = 1500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load stories
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
                        
                        # Consistent quality filter
                        if story and 50 <= len(story) <= 500:
                            stories.append(story.strip())
                    except:
                        continue
        
        # Pre-tokenize
        self.tokenized_stories = []
        for story in stories:
            tokens = tokenizer.encode(story, add_special=True)
            if 8 <= len(tokens) <= max_length:
                self.tokenized_stories.append(tokens)
        
        print(f"Dataset: {len(self.tokenized_stories)} tokenized stories")
    
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
    """Label smoothing loss for Phase 2"""
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

def train_phase(phase_name, use_phase2_optimizations=False, epochs=10):
    """Train a single phase with consistent setup"""
    
    print(f"\n🚀 Training {phase_name}")
    print("=" * 50)
    
    # Check for standard tokenizer
    tokenizer_path = Path("algorithms/transformers/standard_tokenizer.json")
    if not tokenizer_path.exists():
        print(f"❌ Standard tokenizer not found. Creating it first...")
        from algorithms.transformers.standard_tokenizer import create_standard_tokenizer
        data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
        if data_path.exists():
            create_standard_tokenizer(str(data_path), str(tokenizer_path))
        else:
            print(f"❌ TinyStories data not found at {data_path}")
            return None
    
    # Load consistent tokenizer
    tokenizer = StandardTransformerTokenizer()
    tokenizer.load(str(tokenizer_path))
    print(f"📂 Loaded standard tokenizer: {len(tokenizer.word_to_idx):,} words")
    
    # Create consistent dataset
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    dataset = StandardDataset(str(data_path), tokenizer, max_length=48, subset_size=1500)
    
    # Consistent train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)  # Fixed seed
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # Phase-specific batch settings
    if use_phase2_optimizations:
        batch_size = 16
        accumulation_steps = 2  # Effective batch = 32
    else:
        batch_size = 32
        accumulation_steps = 1
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Consistent model architecture
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=192,  # Smaller for faster comparison
        num_heads=6,
        num_layers=3,
        d_ff=512,
        max_seq_len=48
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    
    # Phase-specific optimizer settings
    if use_phase2_optimizations:
        # Phase 2: Enhanced AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98)
        )
        
        # Phase 2: Cosine annealing
        warmup_steps = 50
        total_steps = len(train_loader) * epochs // accumulation_steps
        
        def cosine_lr_schedule(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_schedule)
        
        print("Phase 2 optimizations: Label smoothing, cosine annealing, enhanced AdamW")
        
    else:
        # Phase 1: Standard AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        
        # Phase 1: Linear warmup + decay
        warmup_steps = 50
        total_steps = len(train_loader) * epochs
        
        def get_lr_scale(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)
        
        print("Phase 1 baseline: Standard training, linear LR decay")
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    training_start = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        if use_phase2_optimizations:
            optimizer.zero_grad()  # Zero at start for accumulation
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            if not use_phase2_optimizations:
                optimizer.zero_grad()
            
            output = model(input_batch)
            
            if use_phase2_optimizations:
                # Phase 2: Label smoothing loss
                loss = label_smoothing_loss(
                    output.contiguous().view(-1, output.size(-1)),
                    target_batch.contiguous().view(-1),
                    smoothing=0.1,
                    ignore_index=tokenizer.pad_token
                )
                loss = loss / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
            else:
                # Phase 1: Standard cross entropy
                loss = F.cross_entropy(
                    output.contiguous().view(-1, output.size(-1)),
                    target_batch.contiguous().view(-1),
                    ignore_index=tokenizer.pad_token
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            train_loss += loss.item() * (accumulation_steps if use_phase2_optimizations else 1)
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation (always use cross entropy for consistency)
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
        
        print(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.3f}")
    
    training_time = time.time() - training_start
    
    # Test generation (consistent method)
    print("Testing story generation...")
    model.eval()
    
    test_prompts = ["Once upon a time", "There was a little"]
    generated_stories = []
    
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt, add_special=False)
            generated_tokens = tokens.copy()
            
            for _ in range(20):
                if len(generated_tokens) >= 40:
                    break
                
                input_tensor = torch.tensor([generated_tokens], device=device)
                output = model(input_tensor)
                
                # Consistent temperature sampling
                logits = output[0, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == tokenizer.eos_token:
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
        "val_accuracies": val_accuracies,
        "config": {
            "use_phase2_optimizations": use_phase2_optimizations,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "epochs": epochs,
            "tokenizer": "standard_tokenizer.json"
        }
    }

def run_standard_comparison():
    """Run fair comparison between Phase 1 and Phase 2 with standard tokenizer"""
    
    print("📊 STANDARD PHASE COMPARISON")
    print("=" * 60)
    print("Fair Phase 1 vs Phase 2 comparison using consistent tokenizer")
    print("Ensures identical data, model architecture, and evaluation methods")
    
    # Check data availability
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    if not data_path.exists():
        print(f"❌ TinyStories not found at {data_path}")
        return
    
    # Run Phase 1 (baseline)
    phase1_results = train_phase("Phase 1 (Baseline)", use_phase2_optimizations=False, epochs=8)
    
    if phase1_results is None:
        print("❌ Phase 1 failed")
        return
    
    # Run Phase 2 (optimized)
    phase2_results = train_phase("Phase 2 (Optimized)", use_phase2_optimizations=True, epochs=8)
    
    if phase2_results is None:
        print("❌ Phase 2 failed")
        return
    
    # Fair comparison analysis
    print(f"\n📊 FAIR COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"🔧 Setup Verification:")
    print(f"   Same tokenizer: ✅ standard_tokenizer.json")
    print(f"   Same data: ✅ TinyStories subset")
    print(f"   Same architecture: ✅ SimpleTransformer")
    print(f"   Same validation: ✅ Cross entropy + accuracy")
    
    print(f"\n📈 Phase 1 (Baseline):")
    print(f"   Final Val Loss:     {phase1_results['final_val_loss']:.4f}")
    print(f"   Final Val Accuracy: {phase1_results['final_val_accuracy']:.3f}")
    print(f"   Training Time:      {phase1_results['training_time']:.1f}s")
    print(f"   Parameters:         {phase1_results['parameters']:,}")
    print(f"   Config: Standard AdamW, Linear LR decay, CrossEntropy")
    
    print(f"\n🚀 Phase 2 (Optimized):")
    print(f"   Final Val Loss:     {phase2_results['final_val_loss']:.4f}")
    print(f"   Final Val Accuracy: {phase2_results['final_val_accuracy']:.3f}")
    print(f"   Training Time:      {phase2_results['training_time']:.1f}s")
    print(f"   Parameters:         {phase2_results['parameters']:,}")
    print(f"   Config: Enhanced AdamW, Cosine annealing, Label smoothing")
    
    # Calculate improvements
    loss_improvement = phase1_results['final_val_loss'] - phase2_results['final_val_loss']
    acc_improvement = phase2_results['final_val_accuracy'] - phase1_results['final_val_accuracy']
    time_ratio = phase2_results['training_time'] / phase1_results['training_time']
    
    print(f"\n🎯 PHASE 2 IMPROVEMENTS:")
    print(f"   Validation Loss:    {loss_improvement:+.4f} ({'✅ Better' if loss_improvement > 0 else '❌ Worse'})")
    print(f"   Validation Accuracy: {acc_improvement:+.3f} ({'✅ Better' if acc_improvement > 0 else '❌ Worse'})")
    print(f"   Training Time:      {time_ratio:.2f}x ({'⚡ Faster' if time_ratio < 1 else '⏱️ Slower'})")
    
    print(f"\n📖 STORY GENERATION COMPARISON:")
    
    for i, (prompt, story1, story2) in enumerate(zip(
        ["Once upon a time", "There was a little"],
        phase1_results['generated_stories'],
        phase2_results['generated_stories']
    )):
        print(f"\nPrompt: '{prompt}'")
        print(f"  Phase 1: {story1}")
        print(f"  Phase 2: {story2}")
    
    # Story quality assessment
    print(f"\n📝 Story Quality Assessment:")
    for i, (story1, story2) in enumerate(zip(phase1_results['generated_stories'], phase2_results['generated_stories'])):
        words1 = len(story1.split())
        words2 = len(story2.split())
        unique1 = len(set(story1.lower().split()))
        unique2 = len(set(story2.lower().split()))
        
        diversity1 = unique1 / words1 if words1 > 0 else 0
        diversity2 = unique2 / words2 if words2 > 0 else 0
        
        print(f"  Story {i+1}: Phase1={words1} words ({diversity1:.2f} diversity), Phase2={words2} words ({diversity2:.2f} diversity)")
    
    # Save comparison results
    comparison = {
        "comparison_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "setup": {
            "tokenizer": "standard_tokenizer.json",
            "data_source": "TinyStories",
            "model_architecture": "SimpleTransformer",
            "evaluation_method": "identical_validation"
        },
        "phase_1": phase1_results,
        "phase_2": phase2_results,
        "improvements": {
            "val_loss_improvement": loss_improvement,
            "val_accuracy_improvement": acc_improvement,
            "training_time_ratio": time_ratio,
            "analysis": {
                "loss": "better" if loss_improvement > 0 else "worse",
                "accuracy": "better" if acc_improvement > 0 else "worse",
                "time": "faster" if time_ratio < 1 else "slower"
            }
        }
    }
    
    # Save results
    results_file = Path("algorithms/transformers/standard_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Fair comparison results saved to: {results_file}")
    
    # Summary
    if loss_improvement > 0 and acc_improvement > 0:
        print(f"\n✅ CONCLUSION: Phase 2 optimizations are working correctly!")
        print(f"   Both loss and accuracy improved with advanced training techniques.")
    elif loss_improvement > 0 or acc_improvement > 0:
        print(f"\n⚖️ CONCLUSION: Phase 2 shows mixed results.")
        print(f"   Some metrics improved, suggesting optimizations work but may need tuning.")
    else:
        print(f"\n📊 CONCLUSION: Phase 2 performs worse on this small dataset.")
        print(f"   This is expected - advanced optimizations are designed for larger scale training.")
        print(f"   Try with larger datasets (10k+ samples) to see Phase 2 benefits.")
    
    return comparison

if __name__ == "__main__":
    comparison = run_standard_comparison()