#!/usr/bin/env python3
"""
Parallel OpenWebText Data Preparation
Uses multiprocessing to dramatically speed up tokenization
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
from datasets import load_dataset
from typing import List, Optional, Tuple
import hashlib
import multiprocessing as mp
from functools import partial
import math

from data.gpt2_tokenizer import GPT2CompatibleTokenizer


def process_document_batch(
    batch_data: Tuple[List[dict], GPT2CompatibleTokenizer, int],
    batch_idx: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Process a batch of documents in parallel.
    Returns (train_sequences, val_sequences)
    """
    documents, tokenizer_data, seq_len = batch_data
    
    # Reconstruct tokenizer (can't pickle HuggingFace tokenizer directly)
    tokenizer = GPT2CompatibleTokenizer()
    
    train_sequences = []
    val_sequences = []
    
    for doc in documents:
        text = doc.get('text', '').strip()
        
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
        
        # Split into sequences with overlap
        for start_idx in range(0, len(tokens) - seq_len, seq_len // 2):  # 50% overlap
            sequence = tokens[start_idx:start_idx + seq_len + 1]  # +1 for target
            
            if len(sequence) == seq_len + 1:
                # Content-based train/val split
                text_hash = hashlib.md5(text[:100].encode()).hexdigest()
                hash_int = int(text_hash[:8], 16)
                
                if hash_int % 10 == 0:  # 10% validation
                    val_sequences.append(sequence)
                else:  # 90% training
                    train_sequences.append(sequence)
    
    return train_sequences, val_sequences


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


def parallel_retokenize_openwebtext(
    input_cache_dir: str,
    output_cache_dir: str,
    seq_len: int = 512,
    max_documents: Optional[int] = None,
    batch_size: int = 1000,
    num_workers: Optional[int] = None
):
    """
    Re-tokenize OpenWebText using parallel processing.
    """
    input_path = Path(input_cache_dir)
    output_path = Path(output_cache_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    print(f"🚀 PARALLEL OPENWEBTEXT TOKENIZATION")
    print("=" * 50)
    print(f"📁 Input: {input_cache_dir}")
    print(f"📁 Output: {output_cache_dir}")
    print(f"🔤 Sequence length: {seq_len}")
    print(f"⚡ Workers: {num_workers}")
    print(f"📦 Batch size: {batch_size}")
    
    # Create tokenizer for vocab size info
    tokenizer = GPT2CompatibleTokenizer()
    print(f"🎯 Tokenizer vocab: {tokenizer.get_vocab_size():,}")
    
    # Load raw OpenWebText dataset
    try:
        print(f"📥 Loading raw OpenWebText dataset...")
        dataset = load_dataset("Skylion007/openwebtext", cache_dir=str(input_path))['train']
        print(f"   Found {len(dataset):,} documents")
    except Exception as e:
        print(f"❌ Failed to load raw dataset: {e}")
        return False
    
    # Determine processing scope
    target_docs = min(max_documents or len(dataset), len(dataset))
    num_batches = math.ceil(target_docs / batch_size)
    
    print(f"📊 Processing scope:")
    print(f"   Target documents: {target_docs:,}")
    print(f"   Batches: {num_batches:,}")
    print(f"   Docs per batch: {batch_size}")
    
    # Create document batches
    print(f"🔄 Creating document batches...")
    document_batches = []
    
    for i in range(0, target_docs, batch_size):
        end_idx = min(i + batch_size, target_docs)
        batch_docs = [dataset[j] for j in range(i, end_idx)]
        document_batches.append((batch_docs, None, seq_len))  # tokenizer will be recreated
    
    print(f"   Created {len(document_batches)} batches")
    
    # Process batches in parallel
    print(f"⚡ Processing batches in parallel...")
    start_time = time.time()
    
    all_train_sequences = []
    all_val_sequences = []
    
    with mp.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_document_batch, document_batches),
            total=len(document_batches),
            desc="Processing batches"
        ))
    
    # Combine results
    print(f"📊 Combining results...")
    for train_seqs, val_seqs in results:
        all_train_sequences.extend(train_seqs)
        all_val_sequences.extend(val_seqs)
    
    elapsed = time.time() - start_time
    docs_per_sec = target_docs / elapsed
    
    print(f"✅ Parallel processing complete:")
    print(f"   🕒 Time: {elapsed:.1f}s")
    print(f"   📈 Speed: {docs_per_sec:.0f} docs/sec")
    print(f"   🚂 Train sequences: {len(all_train_sequences):,}")
    print(f"   🧪 Val sequences: {len(all_val_sequences):,}")
    print(f"   📏 Total sequences: {len(all_train_sequences) + len(all_val_sequences):,}")
    
    # Save data efficiently
    print(f"💾 Saving processed data...")
    _save_sequences_parallel(all_train_sequences, output_path / "train", seq_len)
    _save_sequences_parallel(all_val_sequences, output_path / "validation", seq_len)
    
    # Save tokenizer for consistency
    tokenizer.save(str(output_path / "tokenizer"))
    
    print(f"🎉 Parallel tokenization complete!")
    return True


def _save_sequences_parallel(sequences: List[List[int]], output_dir: Path, seq_len: int):
    """Save sequences efficiently with parallel-optimized format"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_name = output_dir.name
    print(f"   💾 Saving {split_name} data ({len(sequences):,} sequences)...")
    
    if not sequences:
        print(f"   ⚠️  No sequences to save for {split_name}")
        return
    
    # Convert to numpy array
    sequences_array = np.array(sequences, dtype=np.int32)
    
    # Save as memory-mapped array (most efficient for training)
    memmap_path = output_dir / f"sequences_{seq_len}.npy"
    np.save(str(memmap_path), sequences_array)
    
    # Save metadata
    metadata = {
        "num_sequences": len(sequences),
        "seq_len": seq_len,
        "vocab_size": int(sequences_array.max()) + 1,
        "split": split_name,
        "dtype": "int32",
        "shape": list(sequences_array.shape),
        "size_mb": float(sequences_array.nbytes / (1024**2))
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"     ✅ Shape: {sequences_array.shape}")
    print(f"     📁 Size: {sequences_array.nbytes / (1024**2):.1f} MB")
    print(f"     🎯 Max token ID: {sequences_array.max()}")


class FastGPT2Dataset:
    """Fast dataset loader for parallel-processed data"""
    
    def __init__(self, data_dir: str, split: str = "train", seq_len: int = 512):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        
        # Load data
        data_path = self.data_dir / split / f"sequences_{seq_len}.npy"
        metadata_path = self.data_dir / split / "metadata.json"
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load data as memory map for efficient access
        self.data = np.load(str(data_path), mmap_mode='r')
        
        print(f"⚡ Fast {split} dataset loaded:")
        print(f"   📊 Sequences: {len(self.data):,}")
        print(f"   📏 Shape: {self.data.shape}")
        print(f"   📋 Size: {self.metadata['size_mb']:.1f} MB")
        print(f"   🎯 Vocab size: {self.metadata['vocab_size']:,}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, target_ids


def create_fast_data_loaders(cache_dir: str, seq_len: int = 512, batch_size: int = 16):
    """Create fast data loaders from parallel-processed data"""
    from torch.utils.data import DataLoader
    
    print(f"🚀 Creating fast data loaders...")
    
    train_dataset = FastGPT2Dataset(cache_dir, "train", seq_len)
    val_dataset = FastGPT2Dataset(cache_dir, "validation", seq_len)
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, mp.cpu_count()),  # Optimal for most systems
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"✅ Fast data loaders created:")
    print(f"   🚂 Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"   🧪 Val: {len(val_dataset):,} samples, {len(val_loader):,} batches")
    
    return train_loader, val_loader


def test_parallel_dataset(cache_dir: str, seq_len: int = 512):
    """Test the parallel-processed dataset"""
    print(f"\n🧪 TESTING PARALLEL DATASET")
    print("=" * 40)
    
    try:
        # Load datasets
        train_dataset = FastGPT2Dataset(cache_dir, "train", seq_len)
        val_dataset = FastGPT2Dataset(cache_dir, "validation", seq_len)
        
        # Test sample
        input_ids, target_ids = train_dataset[0]
        print(f"✅ Sample shapes: input {input_ids.shape}, target {target_ids.shape}")
        print(f"✅ Token range: {input_ids.min().item()} - {input_ids.max().item()}")
        
        # Test with tokenizer
        tokenizer = GPT2CompatibleTokenizer.load(str(Path(cache_dir) / "tokenizer"))
        decoded = tokenizer.decode(input_ids[:50].tolist(), skip_special=True)
        print(f"✅ Sample text: '{decoded[:80]}...'")
        
        # Test data loaders
        train_loader, val_loader = create_fast_data_loaders(cache_dir, seq_len, batch_size=4)
        
        # Time a batch
        start_time = time.time()
        batch = next(iter(train_loader))
        batch_time = time.time() - start_time
        
        print(f"✅ Batch loading: {batch_time:.4f}s for {batch[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main parallel data preparation function"""
    # Configuration
    input_cache = str(Path.home() / ".cache" / "openwebtext")
    output_cache = str(Path.home() / ".cache" / "openwebtext_gpt2_parallel")
    
    # Use CPU count for optimal parallelization
    num_workers = mp.cpu_count()
    
    print(f"💻 System info:")
    print(f"   CPU cores: {num_workers}")
    print(f"   Recommended workers: {num_workers - 1}")
    
    # Run parallel tokenization
    success = parallel_retokenize_openwebtext(
        input_cache_dir=input_cache,
        output_cache_dir=output_cache,
        seq_len=512,
        max_documents=50000,  # Smaller for testing, increase for full run
        batch_size=500,  # Smaller batches for better load balancing
        num_workers=num_workers - 1
    )
    
    if success:
        print(f"\n🎉 PARALLEL TOKENIZATION COMPLETE!")
        print(f"📁 Cache location: {output_cache}")
        
        # Test the dataset
        test_success = test_parallel_dataset(output_cache)
        
        if test_success:
            print(f"\n✅ Dataset ready for training!")
            print(f"🚀 Next step: Run advanced training script")
        else:
            print(f"\n❌ Dataset test failed!")
    else:
        print(f"\n❌ Parallel tokenization failed!")


if __name__ == "__main__":
    main()