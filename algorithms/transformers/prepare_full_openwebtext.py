#!/usr/bin/env python3
"""
Full OpenWebText Dataset Preparation - All 8M Documents
Optimized parallel processing for maximum SOTA performance
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
import gc
import psutil

from data.gpt2_tokenizer import GPT2CompatibleTokenizer


def process_document_batch_full(
    batch_data: Tuple[List[dict], int, int],
    batch_idx: int = 0
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    """
    Process a batch of documents for full dataset.
    Returns (train_sequences, val_sequences, processed_docs, total_tokens)
    """
    documents, seq_len, batch_size = batch_data
    
    # Create tokenizer for this process
    tokenizer = GPT2CompatibleTokenizer()
    
    train_sequences = []
    val_sequences = []
    processed_docs = 0
    total_tokens = 0
    
    for doc in documents:
        text = doc.get('text', '').strip()
        
        # Quality filters
        if not text or len(text) < 200:
            continue
        
        # Skip very repetitive texts
        if _is_too_repetitive_full(text):
            continue
        
        # Tokenize with GPT-2 tokenizer
        try:
            tokens = tokenizer.encode(text, add_special=True)
            total_tokens += len(tokens)
        except Exception:
            continue  # Skip problematic texts
        
        # Create more sequences with smaller overlap for full dataset
        overlap = seq_len // 4  # 25% overlap for more data
        for start_idx in range(0, len(tokens) - seq_len, overlap):
            sequence = tokens[start_idx:start_idx + seq_len + 1]  # +1 for target
            
            if len(sequence) == seq_len + 1:
                # Content-based train/val split
                text_hash = hashlib.md5(text[:100].encode()).hexdigest()
                hash_int = int(text_hash[:8], 16)
                
                if hash_int % 10 == 0:  # 10% validation
                    val_sequences.append(sequence)
                else:  # 90% training
                    train_sequences.append(sequence)
        
        processed_docs += 1
        
        # Memory management for long running processes
        if processed_docs % 1000 == 0:
            gc.collect()
    
    return train_sequences, val_sequences, processed_docs, total_tokens


def _is_too_repetitive_full(text: str) -> bool:
    """Optimized repetition check for full dataset"""
    words = text.split()
    if len(words) < 10:
        return True
    
    # Quick checks first
    if len(set(words)) / len(words) < 0.25:  # Very low diversity
        return True
    
    # Sample check for very long texts
    if len(words) > 1000:
        sample_words = words[:500] + words[-500:]
        words = sample_words
    
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # If any single word appears more than 25% of the time, it's repetitive
    max_word_pct = max(word_counts.values()) / len(words)
    if max_word_pct > 0.25:
        return True
    
    return False


def full_parallel_tokenization(
    output_cache_dir: str,
    seq_len: int = 512,
    batch_size: int = 2000,  # Larger batches for efficiency
    num_workers: Optional[int] = None
):
    """
    Process the complete 8M document OpenWebText dataset.
    """
    output_path = Path(output_cache_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine optimal number of workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)  # Leave 2 cores free
    
    # Get system memory info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"🌍 FULL OPENWEBTEXT TOKENIZATION - 8M DOCUMENTS")
    print("=" * 70)
    print(f"📁 Output: {output_cache_dir}")
    print(f"🔤 Sequence length: {seq_len}")
    print(f"⚡ Workers: {num_workers}")
    print(f"📦 Batch size: {batch_size:,}")
    print(f"💾 System memory: {memory_gb:.1f} GB")
    
    # Create tokenizer for vocab info
    tokenizer = GPT2CompatibleTokenizer()
    print(f"🎯 Tokenizer vocab: {tokenizer.get_vocab_size():,}")
    
    # Load the full OpenWebText dataset
    try:
        print(f"📥 Loading full OpenWebText dataset...")
        dataset = load_dataset("Skylion007/openwebtext")['train']
        print(f"   Found {len(dataset):,} documents")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Calculate processing scope
    total_docs = len(dataset)
    num_batches = math.ceil(total_docs / batch_size)
    estimated_sequences = total_docs * 3  # Rough estimate with overlap
    estimated_size_gb = estimated_sequences * seq_len * 4 / (1024**3)  # int32
    
    print(f"📊 Processing scope:")
    print(f"   Total documents: {total_docs:,}")
    print(f"   Batches: {num_batches:,}")
    print(f"   Estimated sequences: ~{estimated_sequences:,}")
    print(f"   Estimated output size: ~{estimated_size_gb:.1f} GB")
    
    # Memory check
    if estimated_size_gb * 1.5 > memory_gb * 0.8:  # 1.5x overhead, use 80% of RAM
        print(f"⚠️  Warning: Estimated size {estimated_size_gb:.1f}GB may exceed available memory")
        print(f"   Available: ~{memory_gb * 0.8:.1f}GB")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
    
    # Confirm processing
    hours_estimate = (num_batches * batch_size) / (num_workers * 500)  # Rough estimate
    print(f"⏰ Estimated processing time: ~{hours_estimate:.1f} hours")
    
    confirm = input(f"\nProcess all {total_docs:,} documents? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Processing cancelled.")
        return False
    
    # Create document batches with progress tracking
    print(f"🔄 Creating document batches...")
    
    # Process in chunks to manage memory
    chunk_size = batch_size * 10  # Process 10 batches at a time
    all_train_sequences = []
    all_val_sequences = []
    total_processed_docs = 0
    total_tokens_processed = 0
    
    start_time = time.time()
    
    for chunk_start in range(0, total_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_docs)
        chunk_batches = []
        
        print(f"\n📦 Processing chunk {chunk_start:,}-{chunk_end:,}")
        
        # Create batches for this chunk
        for i in range(chunk_start, chunk_end, batch_size):
            end_idx = min(i + batch_size, chunk_end)
            batch_docs = [dataset[j] for j in range(i, end_idx)]
            chunk_batches.append((batch_docs, seq_len, batch_size))
        
        # Process batches in parallel
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_document_batch_full, chunk_batches),
                total=len(chunk_batches),
                desc=f"Processing chunk batches"
            ))
        
        # Combine results
        chunk_train = []
        chunk_val = []
        chunk_docs = 0
        chunk_tokens = 0
        
        for train_seqs, val_seqs, proc_docs, tokens in results:
            chunk_train.extend(train_seqs)
            chunk_val.extend(val_seqs)
            chunk_docs += proc_docs
            chunk_tokens += tokens
        
        # Add to totals
        all_train_sequences.extend(chunk_train)
        all_val_sequences.extend(chunk_val)
        total_processed_docs += chunk_docs
        total_tokens_processed += chunk_tokens
        
        # Progress update
        elapsed = time.time() - start_time
        docs_per_sec = total_processed_docs / elapsed
        completion_pct = (chunk_end / total_docs) * 100
        
        print(f"📈 Progress: {completion_pct:.1f}% complete")
        print(f"   Documents processed: {total_processed_docs:,}")
        print(f"   Train sequences: {len(all_train_sequences):,}")
        print(f"   Val sequences: {len(all_val_sequences):,}")
        print(f"   Total tokens: {total_tokens_processed:,}")
        print(f"   Speed: {docs_per_sec:.0f} docs/sec")
        
        # Estimate remaining time
        if total_processed_docs > 0:
            remaining_docs = total_docs - total_processed_docs
            eta_seconds = remaining_docs / docs_per_sec
            eta_hours = eta_seconds / 3600
            print(f"   ETA: {eta_hours:.1f} hours remaining")
        
        # Force garbage collection
        gc.collect()
    
    # Final statistics
    elapsed = time.time() - start_time
    docs_per_sec = total_processed_docs / elapsed
    
    print(f"\n🎉 FULL DATASET PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"🕒 Total time: {elapsed/3600:.2f} hours")
    print(f"📈 Speed: {docs_per_sec:.0f} docs/sec")
    print(f"📊 Documents processed: {total_processed_docs:,}")
    print(f"🚂 Train sequences: {len(all_train_sequences):,}")
    print(f"🧪 Val sequences: {len(all_val_sequences):,}")
    print(f"📏 Total sequences: {len(all_train_sequences) + len(all_val_sequences):,}")
    print(f"🔤 Total tokens: {total_tokens_processed:,}")
    
    # Save data efficiently
    print(f"\n💾 Saving full dataset...")
    _save_sequences_full(all_train_sequences, output_path / "train", seq_len, "train")
    _save_sequences_full(all_val_sequences, output_path / "validation", seq_len, "validation")
    
    # Save tokenizer and metadata
    tokenizer.save(str(output_path / "tokenizer"))
    
    # Save processing metadata
    metadata = {
        "total_documents": total_docs,
        "processed_documents": total_processed_docs,
        "total_tokens": total_tokens_processed,
        "train_sequences": len(all_train_sequences),
        "val_sequences": len(all_val_sequences),
        "seq_len": seq_len,
        "processing_time_hours": elapsed / 3600,
        "docs_per_second": docs_per_sec,
        "vocab_size": tokenizer.get_vocab_size(),
        "timestamp": time.time()
    }
    
    with open(output_path / "processing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"🎉 Full OpenWebText dataset ready for SOTA training!")
    return True


def _save_sequences_full(sequences: List[List[int]], output_dir: Path, seq_len: int, split_name: str):
    """Save sequences optimized for very large datasets"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not sequences:
        print(f"   ⚠️  No {split_name} sequences to save")
        return
    
    print(f"   💾 Saving {split_name} data ({len(sequences):,} sequences)...")
    
    # Convert to numpy array in chunks to manage memory
    chunk_size = 100000  # Process 100K sequences at a time
    total_saved = 0
    
    # Save first chunk to get the file started
    first_chunk = sequences[:chunk_size]
    sequences_array = np.array(first_chunk, dtype=np.int32)
    
    memmap_path = output_dir / f"sequences_{seq_len}.npy"
    np.save(str(memmap_path), sequences_array)
    total_saved += len(first_chunk)
    
    # Append remaining chunks
    if len(sequences) > chunk_size:
        # Create memory-mapped array for appending
        total_sequences = len(sequences)
        full_array = np.lib.format.open_memmap(
            str(memmap_path), 
            mode='r+', 
            dtype=np.int32, 
            shape=(total_sequences, seq_len + 1)
        )
        
        # Copy first chunk data
        full_array[:len(first_chunk)] = sequences_array
        
        # Process remaining chunks
        for i in range(chunk_size, total_sequences, chunk_size):
            end_idx = min(i + chunk_size, total_sequences)
            chunk = sequences[i:end_idx]
            chunk_array = np.array(chunk, dtype=np.int32)
            full_array[i:end_idx] = chunk_array
            total_saved += len(chunk)
            
            if i % (chunk_size * 10) == 0:  # Progress every 1M sequences
                print(f"     Saved {total_saved:,}/{total_sequences:,} sequences...")
    
    # Save metadata
    size_mb = total_saved * (seq_len + 1) * 4 / (1024**2)
    max_token_id = max(max(seq) for seq in sequences[:1000])  # Sample for max token
    
    metadata = {
        "num_sequences": total_saved,
        "seq_len": seq_len,
        "vocab_size": max_token_id + 1,
        "split": split_name,
        "dtype": "int32",
        "shape": [total_saved, seq_len + 1],
        "size_mb": float(size_mb)
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"     ✅ {split_name}: {total_saved:,} sequences, {size_mb:.1f} MB")


def main():
    """Main function for full dataset processing"""
    print("🌍 FULL OPENWEBTEXT PROCESSING")
    print("Processing all 8+ million documents for maximum SOTA performance")
    print("=" * 70)
    
    output_cache = str(Path.home() / ".cache" / "openwebtext_gpt2_full")
    
    # System info
    num_workers = mp.cpu_count() - 2
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"💻 System resources:")
    print(f"   CPU cores: {mp.cpu_count()} (using {num_workers} workers)")
    print(f"   Memory: {memory_gb:.1f} GB")
    print(f"   Output: {output_cache}")
    
    # Start processing
    success = full_parallel_tokenization(
        output_cache_dir=output_cache,
        seq_len=512,
        batch_size=2000,  # Larger batches for full dataset
        num_workers=num_workers
    )
    
    if success:
        print(f"\n🎉 SUCCESS: Full OpenWebText dataset ready!")
        print(f"📁 Location: {output_cache}")
        print(f"🚀 Ready for full SOTA transformer training")
    else:
        print(f"\n❌ Processing failed or cancelled")


if __name__ == "__main__":
    main()