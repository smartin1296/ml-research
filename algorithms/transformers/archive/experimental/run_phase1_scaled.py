#!/usr/bin/env python3
"""
Phase 1 Scaled: Proper "Attention is All You Need" Baseline
Full-scale implementation using M1 Max capabilities
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.core import TransformerTrainer
from algorithms.transformers.phase_benchmark import TransformerPhaseBenchmark
from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer

# Improved dataset loading
import json
from torch.utils.data import Dataset, DataLoader


class ScaledTinyStoriesDataset(Dataset):
    """Improved TinyStories dataset with word tokenization and better filtering"""
    
    def __init__(self, data_path: str, tokenizer: ScaledWordTokenizer, max_length: int = 256, subset_size: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.stories = self._load_and_filter_stories(data_path, subset_size)
        
        # Pre-tokenize for efficiency
        self.tokenized_stories = []
        
        for story in self.stories:
            tokens = tokenizer.encode(story, add_special=True)
            if 10 <= len(tokens) <= max_length:  # Filter reasonable length stories
                self.tokenized_stories.append(tokens)
    
    def _load_and_filter_stories(self, data_path: str, subset_size: int = None) -> list:
        """Load and filter high-quality stories"""
        stories = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        # Handle both JSON and plain text
                        if line.strip().startswith('{'):
                            data = json.loads(line.strip())
                            story = data.get('story', data.get('text', ''))
                        else:
                            story = line.strip()
                        
                        # Quality filters
                        if self._is_quality_story(story):
                            stories.append(story.strip())
                            
                            if subset_size and len(stories) >= subset_size:
                                break
                                
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
        
        return stories
    
    def _is_quality_story(self, story: str) -> bool:
        """Filter for high-quality stories"""
        # Length filters
        if len(story) < 100 or len(story) > 2000:  # 100-2000 characters
            return False
        
        # Word count filter
        word_count = len(story.split())
        if word_count < 20 or word_count > 400:  # 20-400 words
            return False
        
        # Content quality filters
        story_lower = story.lower()
        
        # Must contain story indicators
        story_indicators = ['once', 'there', 'was', 'little', 'big', 'happy', 'sad', 'day', 'time']
        if not any(indicator in story_lower for indicator in story_indicators):
            return False
        
        # Avoid repetitive or broken stories
        sentences = story.split('.')
        if len(sentences) < 3:  # At least 3 sentences
            return False
        
        # Check for reasonable sentence length variance
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sentence_lengths) == 0 or max(sentence_lengths) < 5:
            return False
        
        return True
    
    def __len__(self):
        return len(self.tokenized_stories)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_stories[idx].copy()
        
        # Pad to max_length if needed
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


def create_scaled_dataloaders(
    tokenizer: ScaledWordTokenizer,
    batch_size: int = 512,
    max_length: int = 256,
    train_subset: int = 100000,
    val_subset: int = 10000
) -> tuple:
    """Create scaled dataloaders with proper word tokenization"""
    
    data_path = Path("data/raw/text/tinystories")
    
    # Use the actual downloaded files
    train_file = data_path / "TinyStories-train.txt"
    if not train_file.exists():
        train_file = data_path / "TinyStories-small.txt"
    
    train_dataset = ScaledTinyStoriesDataset(
        str(train_file), tokenizer, max_length, train_subset
    )
    
    val_file = data_path / "TinyStories-valid.txt"
    if val_file.exists():
        val_dataset = ScaledTinyStoriesDataset(
            str(val_file), tokenizer, max_length, val_subset  
        )
    else:
        # Split from training data if no separate validation
        print("🔄 Creating train/val split...")
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        pin_memory=True, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=0
    )
    
    return train_loader, val_loader


def create_scaled_model(vocab_size: int) -> torch.nn.Module:
    """
    Create memory-efficient Phase 1 model: Scaled down for M1 Max
    """
    # Use M1 Max optimized settings from testing
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=256,        # From optimization: d_model=256 was optimal
        num_heads=8,        # Standard 8 heads (256/8 = 32 per head)
        num_layers=4,       # Reasonable depth 
        d_ff=1024,          # 4x d_model 
        max_seq_len=128     # Keep sequence length manageable
    )
    
    return model


def run_phase1_scaled():
    """Run scaled Phase 1 training"""
    
    print("PHASE 1 SCALED: Full 'Attention is All You Need' Baseline")
    print("Scaled 2017 Transformer Architecture with M1 Max optimized settings")
    
    # Initialize benchmark system
    benchmark = TransformerPhaseBenchmark(
        phase_name="1_scaled_baseline_2017",
        model_description="Full-scale 'Attention is All You Need' with word tokenization and quality data"
    )
    
    # Load stories for tokenizer training first
    print("Building word-level tokenizer...")
    data_path = Path("data/raw/text/tinystories")
    train_file = data_path / "TinyStories-train.txt"
    if not train_file.exists():
        train_file = data_path / "TinyStories-small.txt"
    
    # Load subset of stories for vocabulary building
    vocab_stories = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 50000:  # Use first 50K for vocab
                break
            if line.strip():
                try:
                    if line.strip().startswith('{'):
                        data = json.loads(line.strip())
                        story = data.get('story', data.get('text', ''))
                    else:
                        story = line.strip()
                    
                    if len(story) > 50:  # Basic filter
                        vocab_stories.append(story)
                except:
                    continue
    
    # Create word tokenizer
    from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
    tokenizer = ScaledWordTokenizer(vocab_size=8192)
    tokenizer.build_vocab(vocab_stories)
    
    # Create scaled datasets with M1 Max optimal batch size
    print("Creating datasets...")
    train_loader, val_loader = create_scaled_dataloaders(
        tokenizer=tokenizer,
        batch_size=512,      # M1 Max optimal from testing  
        max_length=64,       # From optimization: seq_len=64 was optimal
        train_subset=10000,  # Focused dataset size
        val_subset=1000      # Focused validation
    )
    
    # Create scaled model
    print("Creating Phase 1 Transformer...")
    model = create_scaled_model(tokenizer.vocab_size)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {param_count:,} parameters, 4-layer, 8-head, d_model=256, vocab={tokenizer.vocab_size:,}")
    
    # Override benchmark config for memory-efficient training
    benchmark.config.update({
        'batch_size': 512,          # M1 Max optimal (keep optimized setting)
        'max_length': 64,           # From optimization: seq_len=64 was optimal
        'train_subset': 10000,      # Focused dataset size
        'val_subset': 1000,         # Focused validation
        'max_epochs': None,         # Dynamic stopping on plateau
        'generation_samples': 10,   # More story samples
        'learning_rate': 3e-4       # Higher LR for better learning
    })
    
    # Setup trainer manually for scaled training
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        tokenizer=tokenizer,
        save_dir=str(benchmark.results_dir / "checkpoints")
    )
    
    # Run scaled training
    print("Starting Phase 1 training...")
    
    training_results = trainer.train(
        learning_rate=3e-4,      # Higher learning rate for better learning
        max_epochs=None,         # Dynamic stopping 
        warmup_steps=500,        # Much shorter warmup for focused dataset
        d_model=256              # Match actual model size
    )
    
    # Test story generation (manual implementation to avoid interface issues)
    print("Testing story generation...")
    model.eval()
    
    prompts = ["Once upon a time", "There was a little", "The brave princess"]
    stories = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            prompt_tokens = tokenizer.encode(prompt, add_special=False)
            if len(prompt_tokens) == 0:
                prompt_tokens = [tokenizer.bos_token]
                
            generated_tokens = prompt_tokens.copy()
            
            # Generate continuation
            for _ in range(50):  # Max 50 new tokens
                input_tensor = torch.tensor([generated_tokens], device=trainer.device)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = model(input_tensor)
                    logits = outputs[0, -1, :]  # Last token predictions
                    
                    # Apply temperature
                    logits = logits / 0.8
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Sample next token
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Stop if EOS token
                    if next_token == tokenizer.eos_token:
                        break
                        
                    generated_tokens.append(next_token)
            
            # Decode story
            story = tokenizer.decode(generated_tokens, skip_special=True)
            stories.append(story)
    
    print("Generated Stories:")
    for i, story in enumerate(stories, 1):
        print(f"{i}: {story}")
    
    # Save tokenizer for future phases
    tokenizer_path = benchmark.results_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    print("Phase 1 baseline complete")
    
    return training_results, benchmark.results_dir, tokenizer


if __name__ == "__main__":
    results, results_dir, tokenizer = run_phase1_scaled()
    print(f"Results saved to: {results_dir}")