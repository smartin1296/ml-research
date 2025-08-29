#!/usr/bin/env python3
"""
Process Full OpenWebText Using Existing Cache
Uses the already downloaded 8M document dataset efficiently
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
import gc
import psutil

from data.gpt2_tokenizer import GPT2CompatibleTokenizer


def process_cached_document_batch(
    batch_data: Tuple[List[dict], int, int],
    batch_idx: int = 0
) -> Tuple[List[List[int]], List[List[int]], int, int]:
    """
    Process a batch of documents using cached OpenWebText.
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
        
        # Skip very repetitive texts (optimized check)
        if _is_too_repetitive_cached(text):
            continue
        
        # Tokenize with GPT-2 tokenizer
        try:
            tokens = tokenizer.encode(text, add_special=True)
            total_tokens += len(tokens)
        except Exception:
            continue
        
        # Create sequences with 25% overlap for maximum data utilization
        overlap = seq_len // 4
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
        
        # Periodic cleanup
        if processed_docs % 1000 == 0:
            gc.collect()
    
    return train_sequences, val_sequences, processed_docs, total_tokens


def _is_too_repetitive_cached(text: str) -> bool:
    """Fast repetition check for cached processing"""
    words = text.split()
    if len(words) < 10:
        return True
    
    # Quick diversity check
    unique_words = len(set(words[:500]))  # Check first 500 words
    total_words = min(500, len(words))
    
    if unique_words / total_words < 0.25:  # Less than 25% unique words
        return True
    
    return False


def process_full_cached_openwebtext(
    input_cache_dir: str,
    output_cache_dir: str,
    seq_len: int = 512,
    batch_size: int = 2000,
    num_workers: Optional[int] = None,
    max_documents: Optional[int] = None
):
    """
    Process the full cached OpenWebText dataset efficiently.
    """
    output_path = Path(output_cache_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine optimal workers
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)
    
    # System info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"🚀 FULL CACHED OPENWEBTEXT PROCESSING")
    print("=" * 60)
    print(f"📁 Input cache: {input_cache_dir}")
    print(f"📁 Output: {output_cache_dir}")
    print(f"⚡ Workers: {num_workers}")
    print(f"📦 Batch size: {batch_size:,}")
    print(f"💾 Memory: {memory_gb:.1f} GB")
    
    # Create tokenizer
    tokenizer = GPT2CompatibleTokenizer()
    print(f"🎯 Vocabulary: {tokenizer.get_vocab_size():,} tokens")
    
    # Load the cached dataset (no download needed!)
    print(f"📂 Loading cached OpenWebText dataset...")
    try:
        dataset = load_dataset("Skylion007/openwebtext", cache_dir=input_cache_dir)['train']
        print(f"   ✅ Found {len(dataset):,} documents in cache")
    except Exception as e:
        print(f"❌ Failed to load cached dataset: {e}")
        return False
    
    # Determine processing scope
    total_docs = len(dataset)
    if max_documents:
        total_docs = min(total_docs, max_documents)
    
    print(f"📊 Processing scope:")
    print(f"   Documents: {total_docs:,}")
    print(f"   Estimated sequences: ~{total_docs * 4:,}")  # ~4 sequences per doc
    print(f"   Estimated processing time: ~{total_docs / (num_workers * 600):.1f} hours")
    
    # Process in optimized chunks to manage memory
    chunk_size = batch_size * 20  # Larger chunks for efficiency
    all_train_sequences = []
    all_val_sequences = []
    total_processed_docs = 0
    total_tokens_processed = 0
    
    start_time = time.time()
    
    for chunk_start in range(0, total_docs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_docs)
        
        print(f"\n📦 Processing chunk {chunk_start:,} - {chunk_end:,}")
        
        # Create batches for this chunk
        chunk_batches = []
        for i in range(chunk_start, chunk_end, batch_size):
            end_idx = min(i + batch_size, chunk_end)
            # Load documents directly from cache
            batch_docs = [dataset[j] for j in range(i, end_idx)]
            chunk_batches.append((batch_docs, seq_len, batch_size))
        
        # Process batches in parallel
        print(f"   🔄 Processing {len(chunk_batches)} batches with {num_workers} workers...")
        
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_cached_document_batch, chunk_batches),
                total=len(chunk_batches),
                desc="   Batches"
            ))
        
        # Combine results
        chunk_train = []
        chunk_val = []
        chunk_docs = 0
        chunk_tokens = 0
        
        for train_seqs, val_seqs, docs, tokens in results:
            chunk_train.extend(train_seqs)
            chunk_val.extend(val_seqs)
            chunk_docs += docs
            chunk_tokens += tokens
        
        # Add to totals
        all_train_sequences.extend(chunk_train)
        all_val_sequences.extend(chunk_val)
        total_processed_docs += chunk_docs
        total_tokens_processed += chunk_tokens
        
        # Progress update
        elapsed = time.time() - start_time
        docs_per_sec = total_processed_docs / elapsed if elapsed > 0 else 0
        completion_pct = (chunk_end / total_docs) * 100
        
        print(f"   📈 Progress: {completion_pct:.1f}% | "
              f"Docs: {total_processed_docs:,} | "
              f"Speed: {docs_per_sec:.0f} docs/sec")
        print(f"   🔢 Train sequences: {len(all_train_sequences):,}")
        print(f"   🔢 Val sequences: {len(all_val_sequences):,}")
        
        # ETA calculation
        if docs_per_sec > 0:
            remaining_docs = total_docs - total_processed_docs
            eta_hours = remaining_docs / (docs_per_sec * 3600)
            print(f"   ⏰ ETA: {eta_hours:.1f} hours remaining")
        
        # Memory management
        gc.collect()
    
    # Final statistics
    elapsed = time.time() - start_time
    docs_per_sec = total_processed_docs / elapsed
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"🕒 Total time: {elapsed/3600:.2f} hours")
    print(f"📈 Average speed: {docs_per_sec:.0f} docs/sec")
    print(f"📊 Documents processed: {total_processed_docs:,}")
    print(f"🚂 Train sequences: {len(all_train_sequences):,}")
    print(f"🧪 Val sequences: {len(all_val_sequences):,}")
    print(f"📏 Total sequences: {len(all_train_sequences) + len(all_val_sequences):,}")
    print(f"🔤 Total tokens: {total_tokens_processed:,}")
    
    # Save data
    print(f"\n💾 Saving processed dataset...")
    _save_sequences_optimized(all_train_sequences, output_path / "train", seq_len, "train")
    _save_sequences_optimized(all_val_sequences, output_path / "validation", seq_len, "validation")
    
    # Save tokenizer and metadata
    tokenizer.save(str(output_path / "tokenizer"))
    
    # Save comprehensive metadata
    metadata = {
        "source_documents": total_docs,
        "processed_documents": total_processed_docs,
        "total_tokens": total_tokens_processed,
        "train_sequences": len(all_train_sequences),
        "val_sequences": len(all_val_sequences),
        "seq_len": seq_len,
        "vocab_size": tokenizer.get_vocab_size(),
        "processing_time_hours": elapsed / 3600,
        "docs_per_second": docs_per_sec,
        "overlap_ratio": 0.25,
        "timestamp": time.time(),
        "source_cache": input_cache_dir
    }
    
    with open(output_path / "full_processing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Full dataset ready for SOTA training!")
    print(f"📁 Location: {output_cache_dir}")
    
    return True


def _save_sequences_optimized(sequences: List[List[int]], output_dir: Path, seq_len: int, split_name: str):
    """Save sequences with memory optimization for large datasets"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not sequences:
        print(f"   ⚠️  No {split_name} sequences to save")
        return
    
    print(f"   💾 Saving {split_name}: {len(sequences):,} sequences")
    
    # Convert and save in one go for efficiency
    sequences_array = np.array(sequences, dtype=np.int32)
    
    # Save as memory-mapped array
    memmap_path = output_dir / f"sequences_{seq_len}.npy"
    np.save(str(memmap_path), sequences_array)
    
    # Calculate statistics
    size_mb = sequences_array.nbytes / (1024**2)
    max_token_id = int(sequences_array.max()) if len(sequences_array) > 0 else 0
    
    # Save metadata
    metadata = {
        "num_sequences": len(sequences),
        "seq_len": seq_len,
        "vocab_size": max_token_id + 1,
        "split": split_name,
        "dtype": "int32",
        "shape": list(sequences_array.shape),
        "size_mb": float(size_mb),
        "max_token_id": max_token_id
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"     ✅ {sequences_array.shape}, {size_mb:.1f} MB, max_token: {max_token_id}")


def main():
    """Main function using existing cache"""
    print("🚀 PROCESSING FULL OPENWEBTEXT FROM EXISTING CACHE")
    print("=" * 60)
    
    # Use existing caches
    input_cache = str(Path.home() / ".cache" / "openwebtext")
    output_cache = str(Path.home() / ".cache" / "openwebtext_gpt2_full")
    
    # Check if input cache exists
    if not Path(input_cache).exists():
        print(f"❌ Input cache not found: {input_cache}")
        return False
    
    print(f"✅ Using existing OpenWebText cache: {input_cache}")
    
    # Process the full dataset
    success = process_full_cached_openwebtext(
        input_cache_dir=input_cache,
        output_cache_dir=output_cache,
        seq_len=512,
        batch_size=2000,
        num_workers=8,  # Use 8 workers for optimal performance
        max_documents=None  # Process ALL documents
    )
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Full dataset processed: {output_cache}")
        print(f"🚀 Ready for full SOTA training!")
        
        # Test the streaming loader
        print(f"\n🧪 Testing streaming loader...")
        from streaming_data_loader import create_streaming_loaders
        
        try:
            train_loader, val_loader = create_streaming_loaders(
                output_cache, seq_len=512, batch_size=16
            )
            print(f"✅ Streaming loaders working!")
            print(f"   Train: {train_loader.num_sequences:,} sequences")
            print(f"   Val: {val_loader.num_sequences:,} sequences")
        except Exception as e:
            print(f"❌ Streaming test failed: {e}")
    
    return success


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)