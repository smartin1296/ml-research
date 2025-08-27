#!/usr/bin/env python3
"""
Phase 2 with Standard Tokenizer
Advanced optimizations with consistent tokenization for fair comparison
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

from algorithms.transformers.standard_tokenizer import StandardTransformerTokenizer
from algorithms.transformers.test_basic import SimpleTransformer
from torch.utils.data import Dataset, DataLoader

class StandardDataset(Dataset):
    """Dataset using standard tokenizer - same as Phase 1 for consistency"""
    
    def __init__(self, data_path: str, tokenizer: StandardTransformerTokenizer, max_length: int = 64, subset_size: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load stories with SAME filtering as Phase 1
        self.stories = self._load_stories(data_path, subset_size)
        
        # Pre-tokenize with standard tokenizer
        self.tokenized_stories = []
        for story in self.stories:
            tokens = tokenizer.encode(story, add_special=True)
            if 10 <= len(tokens) <= max_length:
                self.tokenized_stories.append(tokens)
        
        print(f"Standard Dataset: {len(self.tokenized_stories)} tokenized stories")
    
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
                        
                        # IDENTICAL quality filter as Phase 1
                        if story and 50 <= len(story) <= 800:
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

def run_phase2_standard():
    """Run Phase 2 with standard tokenizer and advanced optimizations"""
    
    print("🚀 PHASE 2 WITH STANDARD TOKENIZER")
    print("=" * 60)
    print("Advanced optimizations with consistent tokenization for fair comparison")
    print("Phase 2 improvements:")
    print("- Label smoothing (0.1)")
    print("- Gradient accumulation (2x)")
    print("- Cosine annealing LR schedule") 
    print("- Enhanced AdamW (transformer betas)")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"algorithms/transformers/results/phases/phase_2_STANDARD_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SAME standard tokenizer as Phase 1
    tokenizer_path = Path("algorithms/transformers/standard_tokenizer.json")
    if not tokenizer_path.exists():
        print(f"❌ Standard tokenizer not found at {tokenizer_path}")
        print("   Run: python standard_tokenizer.py --create")
        return
    
    tokenizer = StandardTransformerTokenizer()
    tokenizer.load(str(tokenizer_path))
    
    print(f"📂 Loaded SAME standard tokenizer: {len(tokenizer.word_to_idx):,} words")
    
    # Load TinyStories data
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    if not data_path.exists():
        print(f"❌ TinyStories not found at {data_path}")
        return
    
    # Create IDENTICAL datasets as Phase 1
    print("📚 Creating datasets with standard tokenizer...")
    train_dataset = StandardDataset(
        str(data_path), tokenizer, max_length=64, subset_size=5000  # SAME as Phase 1
    )
    
    # Create IDENTICAL validation split as Phase 1
    val_size = len(train_dataset) // 10  # 10% validation
    train_size = len(train_dataset) - val_size
    
    # Use SAME fixed seed for identical splits
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)
    
    # Phase 2: Gradient accumulation for larger effective batch size
    batch_size = 16  # Smaller physical batch
    accumulation_steps = 2  # Effective batch = 32 (same as Phase 1)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"Training batches: {len(train_loader)} (batch_size={batch_size}, accumulation={accumulation_steps})")
    print(f"Effective batch size: {batch_size * accumulation_steps} (same as Phase 1)")
    
    # Create IDENTICAL model architecture as Phase 1
    print("🧠 Creating Phase 2 model...")
    model = SimpleTransformer(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=256,      # SAME
        num_heads=8,      # SAME
        num_layers=4,     # SAME
        d_ff=1024,        # SAME
        max_seq_len=64    # SAME
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters (IDENTICAL to Phase 1)")
    
    # Setup training
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # Phase 2: Enhanced optimizer with transformer-standard betas
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3,                # SAME as Phase 1
        weight_decay=0.01,      # SAME as Phase 1
        betas=(0.9, 0.98)       # Phase 2: Transformer paper values
    )
    
    # Phase 2: Cosine annealing scheduler (instead of linear)
    warmup_steps = 100
    total_steps = len(train_loader) * 15 // accumulation_steps  # Account for accumulation
    
    def cosine_lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_schedule)
    
    print(f"Phase 2 Settings:")
    print(f"- Learning Rate: 1e-3 (same as Phase 1)")
    print(f"- Weight Decay: 0.01 (same as Phase 1)")
    print(f"- Betas: (0.9, 0.98) (transformer standard)")
    print(f"- Scheduler: Cosine annealing (vs linear decay)")
    print(f"- Loss: Label smoothing (0.1)")
    print(f"- Gradient accumulation: {accumulation_steps}x")
    print(f"- Device: {device}")
    
    # Training loop with Phase 2 enhancements
    print(f"\n🚀 Starting Phase 2 training...")
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    training_start = time.time()
    
    for epoch in range(15):  # SAME epochs as Phase 1
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
        
        avg_train_loss = train_loss / train_batches
        
        # Validation (use cross entropy for consistency with Phase 1)
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
    
    print(f"\n✅ Phase 2 training completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    
    # Test story generation with SAME method as Phase 1
    print(f"\n📖 Testing story generation...")
    model.eval()
    
    test_prompts = [
        "Once upon a time",
        "There was a little girl", 
        "In a magical forest"
    ]
    
    generated_stories = []
    
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt, add_special=False)
            generated_tokens = tokens.copy()
            
            for _ in range(25):  # Same as Phase 1
                if len(generated_tokens) >= 50:
                    break
                
                input_tensor = torch.tensor([generated_tokens], device=device)
                output = model(input_tensor)
                
                # Same temperature sampling as Phase 1
                logits = output[0, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == tokenizer.eos_token:
                    break
                
                generated_tokens.append(next_token)
            
            story = tokenizer.decode(generated_tokens, skip_special=True)
            generated_stories.append(story)
            print(f"'{prompt}' -> {story}")
    
    # Save results with standard format
    results = {
        "phase": "2_STANDARD_optimized",
        "model_description": "Phase 2 optimizations with standard tokenizer for fair comparison",
        "tokenizer": "standard_tokenizer.json", 
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
            "max_epochs": 15
        },
        "training_config": {
            "optimizer": "AdamW",
            "betas": [0.9, 0.98],
            "weight_decay": 0.01,
            "scheduler": "cosine_annealing",
            "loss_function": "label_smoothing",
            "label_smoothing": 0.1,
            "gradient_accumulation_steps": accumulation_steps,
            "gradient_clipping": 1.0
        },
        "phase_2_improvements": [
            "Label smoothing (0.1) - prevents overconfident predictions",
            "Gradient accumulation (2x) - larger effective batch size",
            "Cosine annealing LR - better convergence than linear decay", 
            "Enhanced AdamW betas (0.9, 0.98) - transformer standard",
            "Same architecture & data as Phase 1 for fair comparison"
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
    
    # Copy standard tokenizer to results
    import shutil
    shutil.copy(tokenizer_path, results_dir / "standard_tokenizer.json")
    
    # Save results
    with open(results_dir / "phase2_standard_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir}")
    print(f"📊 Standard tokenizer copied for consistency")
    
    return results, results_dir

if __name__ == "__main__":
    results, results_dir = run_phase2_standard()
    print("\n🎉 Phase 2 with Standard Tokenizer complete!")