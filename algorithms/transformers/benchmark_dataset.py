#!/usr/bin/env python3
"""
TinyStories Benchmark Dataset for Transformer Evolution
Tracks improvements across transformer phases with consistent evaluation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from tqdm import tqdm
# import tiktoken  # Will use fallback tokenizer
import numpy as np
import random

class TinyStoriesTokenizer:
    """
    GPT-4 compatible tokenizer for TinyStories
    Uses tiktoken for consistent tokenization across phases
    """
    
    def __init__(self):
        # Use simple character-based tokenizer for compatibility
        # This is for optimization testing - we'll use better tokenizer for actual training
        
        # Common characters in English text
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?':;-\"()")
        chars.extend(['\n', '\t'])  # Add whitespace
        
        # Create vocab mapping
        self.char_to_idx = {ch: i+3 for i, ch in enumerate(chars)}  # Start from 3
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = 1
        self.char_to_idx['<EOS>'] = 2
        
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Special tokens
        self.pad_token = 0
        self.eos_token = 2
        
        print(f"🔤 TinyStories Tokenizer initialized (Character-based for testing)")
        print(f"   Vocab size: {self.vocab_size:,}")
        print(f"   PAD token: {self.pad_token}")
        print(f"   EOS token: {self.eos_token}")
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = [self.char_to_idx.get(ch, 1) for ch in text]  # 1 is UNK
        if add_eos:
            tokens.append(self.eos_token)
        return tokens
    
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text"""
        if skip_special:
            # Remove pad and eos tokens
            tokens = [t for t in tokens if t not in [self.pad_token, self.eos_token]]
        
        chars = []
        for token in tokens:
            if token in self.idx_to_char:
                chars.append(self.idx_to_char[token])
            else:
                chars.append('<UNK>')
        return ''.join(chars)
    
    def batch_encode(self, texts: List[str], max_length: int = 256, add_eos: bool = True) -> torch.Tensor:
        """Batch encode with padding"""
        batch_tokens = []
        for text in texts:
            tokens = self.encode(text, add_eos=add_eos)
            # Truncate or pad
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens.extend([self.pad_token] * (max_length - len(tokens)))
            batch_tokens.append(tokens)
        return torch.tensor(batch_tokens, dtype=torch.long)


class TinyStoriesDataset(Dataset):
    """
    TinyStories dataset for language modeling
    Perfect for tracking transformer improvements across phases
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: TinyStoriesTokenizer,
        max_length: int = 256,
        subset_size: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load stories
        print(f"📖 Loading TinyStories from {data_path}")
        self.stories = self._load_stories(data_path, subset_size)
        
        print(f"📊 Dataset created with {len(self.stories):,} stories")
        print(f"   Max length: {max_length} tokens")
        
        # Pre-tokenize for efficiency
        print("🔄 Pre-tokenizing stories...")
        self.tokenized_stories = []
        for story in tqdm(self.stories, desc="Tokenizing"):
            tokens = self.tokenizer.encode(story, add_eos=True)
            if len(tokens) <= max_length:  # Only keep stories that fit
                self.tokenized_stories.append(tokens)
        
        print(f"✅ Pre-tokenized {len(self.tokenized_stories):,} stories")
    
    def _load_stories(self, data_path: str, subset_size: Optional[int] = None) -> List[str]:
        """Load stories from file"""
        stories = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        # Handle both JSON and plain text formats
                        if line.strip().startswith('{'):
                            data = json.loads(line.strip())
                            story = data.get('story', data.get('text', ''))
                        else:
                            story = line.strip()
                        
                        if story and len(story) > 50:  # Filter very short stories
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
        
        # Pad to max_length if needed
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


def download_tinystories(data_dir: Path, size: str = "small") -> Path:
    """
    Download TinyStories dataset
    
    Args:
        data_dir: Directory to save data
        size: 'small' (1M stories), 'medium' (5M), 'large' (15M)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for different sizes
    urls = {
        'small': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt',
        'valid': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt'
    }
    
    files = {}
    for split, url in urls.items():
        if split == 'small' and size != 'small':
            continue
            
        file_path = data_dir / f"TinyStories-{split}.txt"
        
        if file_path.exists():
            print(f"✅ {file_path} already exists")
            files[split] = file_path
            continue
        
        print(f"⬇️ Downloading {split} dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f, tqdm(
            desc=f"Downloading {split}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size_written = f.write(data)
                pbar.update(size_written)
        
        files[split] = file_path
        print(f"✅ Downloaded {file_path}")
    
    return files['small'] if 'small' in files else files[list(files.keys())[0]]


def create_benchmark_dataloaders(
    data_dir: str = "data/raw/text/tinystories",
    batch_size: int = 32,
    max_length: int = 256,
    train_subset: int = 50000,  # Smaller subset for fast iteration
    val_subset: int = 5000
) -> Tuple[DataLoader, DataLoader, TinyStoriesTokenizer]:
    """
    Create standardized dataloaders for transformer benchmarking
    
    Args:
        data_dir: Directory containing TinyStories data
        batch_size: Training batch size
        max_length: Maximum sequence length
        train_subset: Number of training examples (for fast iteration)
        val_subset: Number of validation examples
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    data_path = Path(data_dir)
    
    # Download if needed
    if not (data_path / "TinyStories-train.txt").exists():
        print("📥 TinyStories not found, downloading...")
        download_tinystories(data_path)
    
    # Create tokenizer
    tokenizer = TinyStoriesTokenizer()
    
    # Create datasets - use the actual downloaded file name
    train_file = data_path / "TinyStories-train.txt"
    if not train_file.exists():
        train_file = data_path / "TinyStories-small.txt"  # Fallback to downloaded name
    
    train_dataset = TinyStoriesDataset(
        train_file,
        tokenizer,
        max_length=max_length,
        subset_size=train_subset
    )
    
    val_path = data_path / "TinyStories-valid.txt"
    if val_path.exists():
        val_dataset = TinyStoriesDataset(
            val_path,
            tokenizer, 
            max_length=max_length,
            subset_size=val_subset
        )
    else:
        # Use part of train set for validation
        print("🔄 Creating train/val split...")
        val_size = min(val_subset, len(train_dataset) // 10)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Set to 0 for M1 Max compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    print(f"📊 Benchmark dataloaders created:")
    print(f"   Training: {len(train_loader):,} batches")
    print(f"   Validation: {len(val_loader):,} batches")
    print(f"   Batch size: {batch_size}")
    print(f"   Max length: {max_length} tokens")
    
    return train_loader, val_loader, tokenizer


class StoryQualityEvaluator:
    """
    Evaluate generated story quality with multiple metrics
    Tracks improvements across transformer phases
    """
    
    def __init__(self, tokenizer: TinyStoriesTokenizer):
        self.tokenizer = tokenizer
        
        # Common story words for vocabulary richness
        self.story_words = {
            'characters': ['boy', 'girl', 'man', 'woman', 'cat', 'dog', 'bird', 'princess', 'prince'],
            'actions': ['ran', 'walked', 'played', 'smiled', 'laughed', 'cried', 'jumped', 'climbed'],
            'emotions': ['happy', 'sad', 'excited', 'scared', 'surprised', 'angry', 'proud'],
            'objects': ['ball', 'toy', 'book', 'tree', 'house', 'car', 'flower', 'cake']
        }
    
    def evaluate_story(self, generated_text: str) -> Dict[str, float]:
        """Evaluate a single generated story"""
        words = generated_text.lower().split()
        
        metrics = {
            'length': len(words),
            'vocabulary_size': len(set(words)),
            'repetition_ratio': self._calculate_repetition(words),
            'story_word_coverage': self._calculate_story_coverage(words),
            'sentence_count': generated_text.count('.') + generated_text.count('!') + generated_text.count('?'),
            'coherence_score': self._estimate_coherence(generated_text)
        }
        
        return metrics
    
    def _calculate_repetition(self, words: List[str]) -> float:
        """Calculate word repetition ratio (lower is better)"""
        if len(words) == 0:
            return 1.0
        
        unique_words = len(set(words))
        total_words = len(words)
        return 1.0 - (unique_words / total_words)
    
    def _calculate_story_coverage(self, words: List[str]) -> float:
        """Calculate coverage of common story words"""
        word_set = set(words)
        all_story_words = []
        for category in self.story_words.values():
            all_story_words.extend(category)
        
        covered_words = len(word_set.intersection(set(all_story_words)))
        return covered_words / len(all_story_words)
    
    def _estimate_coherence(self, text: str) -> float:
        """Simple coherence estimation based on sentence structure"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.0
        
        # Check for basic narrative patterns
        coherence_indicators = [
            'once upon a time',
            'then', 'next', 'after', 'finally',
            'suddenly', 'meanwhile',
            'the end'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in coherence_indicators if indicator in text_lower)
        
        return min(1.0, indicator_count / 3.0)  # Normalize to 0-1
    
    def evaluate_batch(self, generated_texts: List[str]) -> Dict[str, float]:
        """Evaluate a batch of generated stories"""
        all_metrics = [self.evaluate_story(text) for text in generated_texts]
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in all_metrics])
            avg_metrics[f'std_{key}'] = np.std([m[key] for m in all_metrics])
        
        return avg_metrics


def generate_sample_stories(
    model: torch.nn.Module,
    tokenizer: TinyStoriesTokenizer,
    device: torch.device,
    num_stories: int = 5,
    max_length: int = 100,
    temperature: float = 0.8
) -> List[str]:
    """Generate sample stories for qualitative evaluation"""
    model.eval()
    
    # Story prompts to ensure consistent starting points
    prompts = [
        "Once upon a time, there was a little",
        "A brave princess decided to",
        "In a magical forest, a small",
        "Every morning, the friendly",
        "One sunny day, two children"
    ]
    
    stories = []
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts[:num_stories]):
            # Encode prompt
            prompt_tokens = tokenizer.encode(prompt, add_special=False)
            generated_tokens = prompt_tokens.copy()
            
            # Generate continuation
            for _ in range(max_length - len(prompt_tokens)):
                # Prepare input
                input_tensor = torch.tensor([generated_tokens], dtype=torch.long).to(device)
                
                # Forward pass (handle both simple and full transformer models)
                try:
                    if hasattr(model, 'encoder') and not hasattr(model, 'decoder'):
                        # Encoder-only model
                        output = model.encoder(input_tensor)
                        if hasattr(model, 'output_projection'):
                            logits = model.output_projection(output)
                        else:
                            logits = output
                    else:
                        # Full model forward
                        logits = model(input_tensor)
                    
                    # Apply temperature sampling
                    next_token_logits = logits[0, -1, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    # Stop on EOS or PAD
                    if next_token in [tokenizer.eos_token, tokenizer.pad_token]:
                        break
                    
                    generated_tokens.append(next_token)
                    
                except Exception as e:
                    print(f"Generation error: {e}")
                    break
            
            # Decode story
            story = tokenizer.decode(generated_tokens, skip_special=True)
            stories.append(story)
    
    return stories


if __name__ == "__main__":
    # Test the benchmark dataset
    print("🧪 Testing TinyStories Benchmark Dataset")
    
    # Create dataloaders
    train_loader, val_loader, tokenizer = create_benchmark_dataloaders(
        batch_size=16,
        max_length=128,
        train_subset=1000,  # Small test
        val_subset=100
    )
    
    # Test batch
    batch = next(iter(train_loader))
    input_tokens, target_tokens = batch
    print(f"\n📊 Sample batch:")
    print(f"   Input shape: {input_tokens.shape}")
    print(f"   Target shape: {target_tokens.shape}")
    
    # Decode sample
    sample_text = tokenizer.decode(input_tokens[0].tolist())
    print(f"\n📖 Sample story excerpt:")
    print(f"   {sample_text[:200]}...")
    
    print(f"\n✅ Benchmark dataset ready for transformer evolution tracking!")