#!/usr/bin/env python3
"""
Phase 2 FIXED: Training & Optimization Improvements
Building on the fixed Phase 1 baseline with advanced training techniques
"""

import sys
import torch
import torch.nn.functional as F
import json
import time
import math
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer
from torch.utils.data import Dataset, DataLoader

class FixedTinyStoriesDataset(Dataset):
    """Same dataset as fixed Phase 1 for fair comparison"""
    
    def __init__(self, data_path: str, tokenizer: ScaledWordTokenizer, max_length: int = 64, subset_size: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load stories with better filtering
        self.stories = self._load_stories(data_path, subset_size)
        
        # Pre-tokenize
        self.tokenized_stories = []
        for story in self.stories:
            tokens = tokenizer.encode(story, add_special=True)
            if 10 <= len(tokens) <= max_length:  # Good length stories
                self.tokenized_stories.append(tokens)
        
        print(f"Dataset: {len(self.tokenized_stories)} tokenized stories (from {len(self.stories)} original)")
    
    def _load_stories(self, data_path: str, subset_size: int = None) -> list:
        stories = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        if line.strip().startswith('{'):
                            data = json.loads(line.strip())
                            story = data.get('story', data.get('text', ''))
                        else:
                            story = line.strip()
                        
                        # Basic quality filter
                        if story and 50 <= len(story) <= 1000:  # Reasonable length
                            stories.append(story.strip())
                            
                            if subset_size and len(stories) >= subset_size:
                                break
                    except:
                        continue
        
        return stories
    
    def __len__(self):
        return len(self.tokenized_stories)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_stories[idx].copy()
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # Language modeling: input[:-1], target[1:]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)

def label_smoothing_loss(pred, target, smoothing=0.1, ignore_index=0):
    """
    Phase 2: Label smoothing loss function
    Prevents overconfident predictions and should reduce repetition
    """
    vocab_size = pred.size(-1)
    confidence = 1.0 - smoothing
    
    # Create smooth target distribution
    smooth_target = torch.full_like(pred, smoothing / vocab_size)
    
    # Mask out padding tokens
    mask = (target != ignore_index)
    target_masked = target[mask]
    pred_masked = pred[mask]
    smooth_target_masked = smooth_target[mask]
    
    if len(target_masked) == 0:  # All padding
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Set confidence for true labels
    smooth_target_masked.scatter_(-1, target_masked.unsqueeze(-1), confidence)
    
    # KL divergence loss
    loss = F.kl_div(F.log_softmax(pred_masked, dim=-1), smooth_target_masked, reduction='batchmean')
    return loss

def generate_story_improved(model, tokenizer, device, prompt, max_length=50, temperature=0.8, top_k=40):
    """Better story generation for Phase 2 evaluation"""
    model.eval()
    
    tokens = tokenizer.encode(prompt, add_special=False)
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            if len(generated_tokens) >= 100:
                break
            
            # Context window
            context = generated_tokens[-50:] if len(generated_tokens) > 50 else generated_tokens
            input_tensor = torch.tensor([context], device=device)
            
            output = model(input_tensor)
            logits = output[0, -1, :]
            
            # Temperature scaling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(0, top_k_indices, top_k_logits)
                logits = logits_filtered
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Stop conditions
            if next_token in [tokenizer.eos_token, tokenizer.pad_token]:
                break
            
            generated_tokens.append(next_token)
            
            # Early stop on excessive repetition
            if len(generated_tokens) > 10:
                last_tokens = generated_tokens[-5:]
                if len(set(last_tokens)) <= 2:  # Too repetitive
                    break
    
    return tokenizer.decode(generated_tokens, skip_special=True)

def run_phase2_fixed():
    """Run Phase 2 with advanced training techniques"""
    
    print("🚀 PHASE 2 FIXED: Training & Optimization Improvements")
    print("=" * 60)
    print("Building on fixed Phase 1 with Phase 2 optimizations:")
    print("- Label smoothing (prevents overconfident predictions)")
    print("- Gradient accumulation (larger effective batch size)")
    print("- Cosine annealing LR schedule")
    print("- Enhanced weight decay")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"algorithms/transformers/results/phases/phase_2_FIXED_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Use SAME data as Phase 1 for fair comparison
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    if not data_path.exists():
        print(f"❌ TinyStories not found at {data_path}")
        return
    
    # Load SAME tokenizer as Phase 1
    phase1_dir = Path("algorithms/transformers/results/phases/phase_1_FIXED_20250826_185157")
    tokenizer_path = phase1_dir / "tokenizer.json"
    
    if not tokenizer_path.exists():
        print(f"❌ Phase 1 tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer = ScaledWordTokenizer()
    tokenizer.load(str(tokenizer_path))
    print(f"Loaded Phase 1 tokenizer: {len(tokenizer.word_to_idx)} words")
    
    # Create SAME datasets as Phase 1
    print("📚 Creating datasets...")
    train_dataset = FixedTinyStoriesDataset(
        str(data_path), tokenizer, max_length=64, subset_size=5000
    )
    
    # Same split as Phase 1
    val_size = len(train_dataset) // 10
    train_size = len(train_dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Phase 2: Gradient accumulation for larger effective batch size
    batch_size = 16  # Smaller physical batch
    accumulation_steps = 2  # Effective batch = 32 (same as Phase 1)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"Training batches: {len(train_loader)} (batch_size={batch_size}, accumulation={accumulation_steps})")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    
    # Create SAME model architecture as Phase 1
    print("🧠 Creating model...")
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=64
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters (same as Phase 1)")
    
    # Setup training
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Phase 2: Enhanced optimizer with better weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3,                # Same as Phase 1
        weight_decay=0.01,      # Phase 2: Better L2 regularization
        betas=(0.9, 0.98)       # Phase 2: Transformer paper values
    )
    
    # Phase 2: Cosine annealing scheduler (instead of linear)
    warmup_steps = 100
    total_steps = len(train_loader) * 20 // accumulation_steps  # Account for accumulation
    
    def cosine_lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_schedule)
    
    print(f"Phase 2 optimizations:")
    print(f"- Learning rate: 1e-3 (same as Phase 1)")
    print(f"- Weight decay: 0.01 (improved)")
    print(f"- Betas: (0.9, 0.98) (transformer standard)")
    print(f"- Scheduler: Cosine annealing (vs linear decay)")
    print(f"- Label smoothing: 0.1")
    print(f"- Gradient accumulation: {accumulation_steps}x")
    
    # Training loop with Phase 2 enhancements
    print("\\n🚀 Starting Phase 2 training...")
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    training_start = time.time()
    
    for epoch in range(20):  # Same max epochs as Phase 1
        epoch_start = time.time()
        
        # Training with Phase 2 enhancements
        model.train()
        train_loss = 0.0
        train_batches = 0
        optimizer.zero_grad()  # Zero at start of epoch
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            output = model(input_batch)
            
            # Phase 2: Label smoothing loss (instead of cross entropy)
            loss = label_smoothing_loss(
                output.contiguous().view(-1, output.size(-1)),
                target_batch.contiguous().view(-1),
                smoothing=0.1,
                ignore_index=tokenizer.pad_token
            )
            
            # Phase 2: Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps  # Denormalize for logging
            train_batches += 1
            
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss.item() * accumulation_steps:.4f}, lr={current_lr:.6f}", end='\\r')
        
        avg_train_loss = train_loss / train_batches
        
        # Validation (same as Phase 1)
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
                
                # Use cross entropy for validation (consistent with Phase 1)
                loss = F.cross_entropy(
                    output.contiguous().view(-1, output.size(-1)),
                    target_batch.contiguous().view(-1),
                    ignore_index=tokenizer.pad_token
                )
                
                val_loss += loss.item()
                val_batches += 1
                
                # Calculate accuracy
                predictions = output.argmax(dim=-1)
                mask = (target_batch != tokenizer.pad_token)
                correct = (predictions == target_batch) & mask
                
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        avg_val_loss = val_loss / val_batches
        val_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        # Update bests
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_accuracy = val_accuracy
            is_best = True
        
        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.3f}, {epoch_time:.1f}s")
        
        # Save best model
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_accuracy,
            }, results_dir / "best_model.pt")
    
    total_time = time.time() - training_start
    
    print(f"\\n✅ Phase 2 training completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    
    # Test story generation with improved method
    print("\\n📖 Testing Phase 2 story generation...")
    model.eval()
    
    test_prompts = [
        "Once upon a time",
        "There was a little girl", 
        "In a magical forest",
        "The brave princess",
        "One sunny day"
    ]
    
    generated_stories = []
    
    for prompt in test_prompts:
        story = generate_story_improved(
            model, tokenizer, device, prompt,
            max_length=40, temperature=0.9, top_k=30
        )
        generated_stories.append(story)
        print(f"'{prompt}' -> {story}")
    
    # Save results
    results = {
        "phase": "2_FIXED_improvements",
        "model_description": "Phase 2 with label smoothing, gradient accumulation, cosine annealing",
        "training_results": {
            "epochs_trained": len(train_losses),
            "training_time_seconds": total_time,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
        },
        "model_config": {
            "parameters": param_count,
            "d_model": 256,
            "num_layers": 4,
            "num_heads": 8,
            "vocab_size": len(tokenizer.word_to_idx),
            "learning_rate": 1e-3,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * accumulation_steps,
            "label_smoothing": 0.1,
            "weight_decay": 0.01,
            "gradient_accumulation": accumulation_steps,
            "scheduler": "cosine_annealing"
        },
        "phase_2_improvements": [
            "Label smoothing (0.1) - prevents overconfident predictions",
            "Gradient accumulation (2x) - larger effective batch size", 
            "Cosine annealing LR - better convergence than linear decay",
            "Enhanced weight decay (0.01) - better L2 regularization",
            "Transformer-standard betas (0.9, 0.98) - improved optimization"
        ],
        "generated_stories": [
            {"prompt": prompt, "story": story} 
            for prompt, story in zip(test_prompts, generated_stories)
        ],
        "training_curves": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
    }
    
    # Save tokenizer (copy from Phase 1)
    tokenizer.save(str(results_dir / "tokenizer.json"))
    
    # Save results
    with open(results_dir / "phase2_fixed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n💾 Results saved to: {results_dir}")
    
    return results, results_dir

if __name__ == "__main__":
    results, results_dir = run_phase2_fixed()
    print("\\n🎉 Phase 2 FIXED complete! Now we can compare Phase 1 vs Phase 2!")