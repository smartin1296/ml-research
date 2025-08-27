#!/usr/bin/env python3
"""
Phase 1 FIXED: Working "Attention is All You Need" Baseline
Fixed learning rate and training issues identified in debugging
"""

import sys
import torch
import torch.nn.functional as F
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer
from torch.utils.data import Dataset, DataLoader

class FixedTinyStoriesDataset(Dataset):
    """Fixed dataset implementation"""
    
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

def run_fixed_phase1():
    """Run fixed Phase 1 with proper learning rate"""
    
    print("🔧 PHASE 1 FIXED: Working Transformer Baseline")
    print("=" * 60)
    print("Fixed learning rate and training issues from debugging")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"algorithms/transformers/results/phases/phase_1_FIXED_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Load TinyStories data
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    if not data_path.exists():
        print(f"❌ TinyStories not found at {data_path}")
        return
    
    # Build tokenizer from data subset
    print("🔤 Building tokenizer...")
    vocab_stories = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000:  # First 10k for vocab
                break
            if line.strip():
                try:
                    if line.strip().startswith('{'):
                        data = json.loads(line.strip())
                        story = data.get('story', data.get('text', ''))
                    else:
                        story = line.strip()
                    
                    if story and len(story) > 50:
                        vocab_stories.append(story)
                except:
                    continue
    
    tokenizer = ScaledWordTokenizer(vocab_size=2000)  # Reasonable vocab size for debugging
    tokenizer.build_vocab(vocab_stories)
    
    print(f"Tokenizer: {len(tokenizer.word_to_idx)} words")
    
    # Create datasets
    print("📚 Creating datasets...")
    train_dataset = FixedTinyStoriesDataset(
        str(data_path), tokenizer, max_length=64, subset_size=5000
    )
    
    # Create validation split
    val_size = len(train_dataset) // 10
    train_size = len(train_dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
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
    print(f"Model: {param_count:,} parameters")
    
    # Setup training
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    # FIXED: Use higher learning rate that works (from debugging)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Simple linear warmup + decay (no complex scheduling)
    warmup_steps = 100
    total_steps = len(train_loader) * 20  # 20 epochs max
    
    def get_lr_scale(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)
    
    print(f"Training setup: lr=1e-3, device={device}")
    
    # Training loop
    print("\n🚀 Starting training...")
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    training_start = time.time()
    
    for epoch in range(20):  # Max 20 epochs
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
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
            scheduler.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}", end='\\r')
        
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
    
    print(f"\n✅ Training completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.3f}")
    
    # Test story generation
    print("\n📖 Testing story generation...")
    model.eval()
    
    test_prompts = [
        "Once upon a time",
        "There was a little",
        "The brave princess"
    ]
    
    generated_stories = []
    
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt, add_special=False)
            generated_tokens = tokens.copy()
            
            for _ in range(30):  # Generate 30 tokens
                if len(generated_tokens) >= 60:
                    break
                
                input_tensor = torch.tensor([generated_tokens], device=device)
                output = model(input_tensor)
                
                # Temperature sampling
                logits = output[0, -1, :] / 0.8
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == tokenizer.eos_token:
                    break
                
                generated_tokens.append(next_token)
            
            story = tokenizer.decode(generated_tokens, skip_special=True)
            generated_stories.append(story)
            print(f"'{prompt}' -> {story}")
    
    # Save results
    results = {
        "phase": "1_FIXED_baseline",
        "model_description": "Fixed Phase 1 with proper learning rate (1e-3)",
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
            "batch_size": 32
        },
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
    
    # Save tokenizer
    tokenizer.save(str(results_dir / "tokenizer.json"))
    
    # Save results
    with open(results_dir / "phase1_fixed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir}")
    
    return results, results_dir

if __name__ == "__main__":
    results, results_dir = run_fixed_phase1()
    print("\\n🎉 Phase 1 FIXED complete! Ready for Phase 2 comparison.")