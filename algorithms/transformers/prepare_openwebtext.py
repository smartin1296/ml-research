#!/usr/bin/env python3
"""
Prepare OpenWebText Dataset with GPT-2 Tokenizer
Re-tokenizes cached data with the new 50K vocabulary tokenizer
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
from datasets import load_dataset
from typing import List, Optional
import hashlib

from data.gpt2_tokenizer import GPT2CompatibleTokenizer


def retokenize_cached_openwebtext(
    input_cache_dir: str,
    output_cache_dir: str,
    tokenizer: GPT2CompatibleTokenizer,
    seq_len: int = 512,
    max_documents: Optional[int] = None
):
    """
    Re-tokenize existing OpenWebText cache with GPT-2 tokenizer.
    """
    input_path = Path(input_cache_dir)
    output_path = Path(output_cache_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Re-tokenizing OpenWebText data")
    print(f"   Input: {input_cache_dir}")
    print(f"   Output: {output_cache_dir}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Tokenizer vocab: {tokenizer.get_vocab_size():,}")
    
    # Check if we have the raw OpenWebText dataset cached
    try:
        print(f"📥 Loading raw OpenWebText dataset...")
        dataset = load_dataset("Skylion007/openwebtext", cache_dir=str(input_path))['train']
        print(f"   Found {len(dataset):,} documents")
    except Exception as e:
        print(f"❌ Failed to load raw dataset: {e}")
        return False
    
    # Process documents
    train_sequences = []
    val_sequences = []
    
    print(f"🔤 Tokenizing documents...")
    
    processed_docs = 0
    target_docs = min(max_documents or len(dataset), len(dataset))
    
    for i, example in enumerate(tqdm(dataset, desc="Processing", total=target_docs)):
        if processed_docs >= target_docs:
            break
        
        text = example.get('text', '').strip()
        
        # Quality filters
        if not text or len(text) < 200:
            continue
        
        # Skip very repetitive texts
        if _is_too_repetitive(text):
            continue
        
        # Tokenize with GPT-2 tokenizer
        try:
            tokens = tokenizer.encode(text, add_special=True)
        except Exception:
            continue  # Skip problematic texts
        
        # Split into sequences
        for start_idx in range(0, len(tokens) - seq_len, seq_len // 2):  # 50% overlap
            sequence = tokens[start_idx:start_idx + seq_len + 1]  # +1 for target
            
            if len(sequence) == seq_len + 1:
                # Content-based train/val split (consistent with original)
                text_hash = hashlib.md5(text[:100].encode()).hexdigest()
                hash_int = int(text_hash[:8], 16)
                
                if hash_int % 10 == 0:  # 10% validation
                    val_sequences.append(sequence)
                else:  # 90% training
                    train_sequences.append(sequence)
        
        processed_docs += 1
        
        # Progress update
        if processed_docs % 10000 == 0:
            print(f"   Processed: {processed_docs:,}/{target_docs:,} docs")
            print(f"   Train sequences: {len(train_sequences):,}")
            print(f"   Val sequences: {len(val_sequences):,}")
    
    print(f"✅ Tokenization complete:")
    print(f"   Train sequences: {len(train_sequences):,}")
    print(f"   Val sequences: {len(val_sequences):,}")
    
    # Save data efficiently
    _save_sequences(train_sequences, output_path / "train", seq_len)
    _save_sequences(val_sequences, output_path / "validation", seq_len)
    
    # Save tokenizer for consistency
    tokenizer.save(str(output_path / "tokenizer"))
    
    return True


def _is_too_repetitive(text: str) -> bool:
    """Check if text is too repetitive"""
    words = text.split()
    if len(words) < 10:
        return True
    
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # If any single word appears more than 30% of the time, it's repetitive
    max_word_pct = max(word_counts.values()) / len(words)
    if max_word_pct > 0.3:
        return True
    
    # Check diversity
    if len(set(words)) / len(words) < 0.3:
        return True
    
    return False


def _save_sequences(sequences: List[List[int]], output_dir: Path, seq_len: int):
    """Save sequences in multiple formats for flexibility"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_name = output_dir.name
    print(f"💾 Saving {split_name} data ({len(sequences):,} sequences)...")
    
    # Convert to numpy array
    sequences_array = np.array(sequences, dtype=np.int32)
    
    # Save as memory-mapped array (most efficient for training)
    memmap_path = output_dir / f"sequences_{seq_len}.npy"
    np.save(str(memmap_path), sequences_array)
    
    # Save metadata
    metadata = {
        "num_sequences": len(sequences),
        "seq_len": seq_len,
        "vocab_size": sequences_array.max() + 1,
        "split": split_name,
        "dtype": "int32",
        "shape": list(sequences_array.shape)
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Saved to {memmap_path}")
    print(f"   Shape: {sequences_array.shape}")
    print(f"   Size: {sequences_array.nbytes / (1024**2):.1f} MB")


class GPT2TokenizedDataset:
    """Dataset for loading GPT-2 tokenized OpenWebText data"""
    
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load data
        data_path = self.data_dir / split / f"sequences_512.npy"  # Assuming 512 seq_len
        metadata_path = self.data_dir / split / "metadata.json"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load data as memory map
        self.data = np.load(str(data_path), mmap_mode='r')
        
        print(f"📂 Loaded {split} dataset:")
        print(f"   Sequences: {len(self.data):,}")
        print(f"   Shape: {self.data.shape}")
        print(f"   Vocab size: {self.metadata['vocab_size']:,}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids


def test_gpt2_dataset(cache_dir: str):
    """Test the GPT-2 tokenized dataset"""
    print(f"\n🧪 Testing GPT-2 tokenized dataset")
    print("=" * 50)
    
    try:
        # Load datasets
        train_dataset = GPT2TokenizedDataset(cache_dir, "train")
        val_dataset = GPT2TokenizedDataset(cache_dir, "validation")
        
        # Test sample
        input_ids, target_ids = train_dataset[0]
        print(f"Sample input shape: {input_ids.shape}")
        print(f"Sample target shape: {target_ids.shape}")
        print(f"Sample tokens: {input_ids[:10].tolist()}")
        
        # Test with tokenizer
        tokenizer = GPT2CompatibleTokenizer.load(str(Path(cache_dir) / "tokenizer"))
        decoded = tokenizer.decode(input_ids[:50].tolist(), skip_special=True)
        print(f"Sample text: '{decoded[:100]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main data preparation function"""
    print("🌐 OPENWEBTEXT GPT-2 TOKENIZATION")
    print("=" * 50)
    
    # Paths
    input_cache = str(Path.home() / ".cache" / "openwebtext")
    output_cache = str(Path.home() / ".cache" / "openwebtext_gpt2")
    
    # Create GPT-2 tokenizer
    print("🤖 Creating GPT-2 tokenizer...")
    tokenizer = GPT2CompatibleTokenizer()
    
    # Re-tokenize data
    success = retokenize_cached_openwebtext(
        input_cache_dir=input_cache,
        output_cache_dir=output_cache,
        tokenizer=tokenizer,
        seq_len=512,
        max_documents=200000  # Subset for faster processing
    )
    
    if success:
        print(f"\n🎉 Data preparation complete!")
        print(f"Cache location: {output_cache}")
        
        # Test the dataset
        test_gpt2_dataset(output_cache)
    else:
        print(f"\n❌ Data preparation failed!")


if __name__ == "__main__":
    main()