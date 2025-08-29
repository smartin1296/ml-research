#!/usr/bin/env python3
"""
Streaming Data Loader for Full OpenWebText Dataset
Memory-efficient streaming to handle 10M+ sequences without RAM overflow
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import json
import random
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
import logging
import mmap
import struct
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

logger = logging.getLogger(__name__)


class StreamingOpenWebTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for very large OpenWebText datasets.
    
    Key features:
    - Streams data from disk without loading full dataset into RAM
    - Memory-mapped file access for efficiency
    - Shuffling via random file access
    - Buffered loading for smooth training
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train", 
        seq_len: int = 512,
        buffer_size: int = 10000,  # Sequences to keep in memory
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Load metadata
        split_dir = self.data_dir / split
        metadata_path = split_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_sequences = self.metadata['num_sequences']
        self.data_file = split_dir / f"sequences_{seq_len}.npy"
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Open memory-mapped file
        self.data_mmap = np.load(str(self.data_file), mmap_mode='r')
        
        logger.info(f"🌊 Streaming dataset initialized:")
        logger.info(f"   Split: {split}")
        logger.info(f"   Sequences: {self.num_sequences:,}")
        logger.info(f"   Buffer size: {buffer_size:,}")
        logger.info(f"   Data shape: {self.data_mmap.shape}")
        logger.info(f"   File size: {self.data_file.stat().st_size / (1024**3):.2f} GB")
    
    def __len__(self):
        return self.num_sequences
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Stream sequences from the dataset"""
        
        # Create shuffled indices if needed
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed + torch.distributed.get_rank() if torch.distributed.is_initialized() else self.seed)
            indices = list(range(self.num_sequences))
            random.shuffle(indices)
        else:
            indices = range(self.num_sequences)
        
        # Stream sequences with buffering
        buffer = []
        for idx in indices:
            # Load sequence from memory-mapped file
            sequence = self.data_mmap[idx]
            
            input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
            target_ids = torch.tensor(sequence[1:], dtype=torch.long)
            
            buffer.append((input_ids, target_ids))
            
            # Yield from buffer when full
            if len(buffer) >= self.buffer_size:
                if self.shuffle:
                    random.shuffle(buffer)
                
                while buffer:
                    yield buffer.pop()
        
        # Yield remaining buffer
        if self.shuffle and buffer:
            random.shuffle(buffer)
        while buffer:
            yield buffer.pop()


class ChunkedStreamingDataset(IterableDataset):
    """
    Alternative streaming approach: Load data in chunks to balance memory and I/O.
    Better for very large datasets where even memory mapping is slow.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = 512,
        chunk_size: int = 50000,  # Sequences per chunk
        shuffle: bool = True,
        prefetch_chunks: int = 2  # Number of chunks to prefetch
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.prefetch_chunks = prefetch_chunks
        
        # Load metadata
        split_dir = self.data_dir / split
        metadata_path = split_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.num_sequences = self.metadata['num_sequences']
        self.data_file = split_dir / f"sequences_{seq_len}.npy"
        self.num_chunks = (self.num_sequences + chunk_size - 1) // chunk_size
        
        logger.info(f"📦 Chunked streaming dataset:")
        logger.info(f"   Total sequences: {self.num_sequences:,}")
        logger.info(f"   Chunk size: {chunk_size:,}")
        logger.info(f"   Total chunks: {self.num_chunks}")
        logger.info(f"   Prefetch chunks: {prefetch_chunks}")
    
    def _load_chunk(self, chunk_idx: int) -> np.ndarray:
        """Load a specific chunk from disk"""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.num_sequences)
        
        # Load only the required slice
        data_mmap = np.load(str(self.data_file), mmap_mode='r')
        chunk_data = data_mmap[start_idx:end_idx].copy()
        
        return chunk_data
    
    def __len__(self):
        return self.num_sequences
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Stream with chunked loading and prefetching"""
        
        # Prepare chunk order
        chunk_indices = list(range(self.num_chunks))
        if self.shuffle:
            random.shuffle(chunk_indices)
        
        # Prefetch chunks using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            chunk_futures = {}
            
            # Start prefetching first chunks
            for i, chunk_idx in enumerate(chunk_indices[:self.prefetch_chunks]):
                future = executor.submit(self._load_chunk, chunk_idx)
                chunk_futures[chunk_idx] = future
            
            for i, chunk_idx in enumerate(chunk_indices):
                # Get current chunk (blocking if not ready)
                if chunk_idx in chunk_futures:
                    chunk_data = chunk_futures[chunk_idx].result()
                    del chunk_futures[chunk_idx]
                else:
                    chunk_data = self._load_chunk(chunk_idx)
                
                # Start prefetching next chunk
                next_idx = i + self.prefetch_chunks
                if next_idx < len(chunk_indices):
                    next_chunk_idx = chunk_indices[next_idx]
                    future = executor.submit(self._load_chunk, next_chunk_idx)
                    chunk_futures[next_chunk_idx] = future
                
                # Shuffle chunk data if needed
                chunk_sequences = []
                for seq_data in chunk_data:
                    input_ids = torch.tensor(seq_data[:-1], dtype=torch.long)
                    target_ids = torch.tensor(seq_data[1:], dtype=torch.long)
                    chunk_sequences.append((input_ids, target_ids))
                
                if self.shuffle:
                    random.shuffle(chunk_sequences)
                
                # Yield sequences from chunk
                for seq in chunk_sequences:
                    yield seq


class AdaptiveStreamingDataLoader:
    """
    Smart data loader that adapts to memory constraints.
    Automatically chooses between streaming strategies based on dataset size.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = 512,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,  # Streaming datasets work best with 0 workers
        memory_limit_gb: float = 32.0  # Auto-detect strategy based on memory
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.memory_limit_gb = memory_limit_gb
        
        # Load metadata to determine strategy
        split_dir = self.data_dir / split
        metadata_path = split_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        num_sequences = metadata['num_sequences']
        data_size_gb = metadata.get('size_mb', 0) / 1024
        
        logger.info(f"🤖 Adaptive data loader analysis:")
        logger.info(f"   Sequences: {num_sequences:,}")
        logger.info(f"   Data size: {data_size_gb:.2f} GB")
        logger.info(f"   Memory limit: {memory_limit_gb:.1f} GB")
        
        # Choose strategy
        if data_size_gb < memory_limit_gb * 0.3:  # Can fit in 30% of memory
            logger.info("   Strategy: Memory-mapped streaming (fast)")
            dataset = StreamingOpenWebTextDataset(
                data_dir, split, seq_len, 
                buffer_size=min(20000, num_sequences // 100),
                shuffle=shuffle
            )
        elif data_size_gb < memory_limit_gb * 0.8:  # Can fit with careful management
            logger.info("   Strategy: Memory-mapped with small buffer")
            dataset = StreamingOpenWebTextDataset(
                data_dir, split, seq_len,
                buffer_size=5000,  # Smaller buffer
                shuffle=shuffle
            )
        else:  # Too large for memory mapping
            logger.info("   Strategy: Chunked streaming (memory-safe)")
            dataset = ChunkedStreamingDataset(
                data_dir, split, seq_len,
                chunk_size=min(25000, num_sequences // 200),
                shuffle=shuffle
            )
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Handled by dataset
            num_workers=num_workers,
            pin_memory=False,  # Better for streaming
            drop_last=True
        )
        
        self.num_sequences = num_sequences
        self.num_batches = num_sequences // batch_size
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        return iter(self.dataloader)


def create_streaming_loaders(
    data_dir: str,
    seq_len: int = 512,
    batch_size: int = 16,
    memory_limit_gb: float = 32.0
) -> Tuple[AdaptiveStreamingDataLoader, AdaptiveStreamingDataLoader]:
    """
    Create memory-efficient streaming data loaders for train and validation.
    """
    
    logger.info(f"🌊 Creating streaming data loaders...")
    logger.info(f"   Data directory: {data_dir}")
    logger.info(f"   Sequence length: {seq_len}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Memory limit: {memory_limit_gb:.1f} GB")
    
    # Create loaders
    train_loader = AdaptiveStreamingDataLoader(
        data_dir, "train", seq_len, batch_size, 
        shuffle=True, memory_limit_gb=memory_limit_gb
    )
    
    val_loader = AdaptiveStreamingDataLoader(
        data_dir, "validation", seq_len, batch_size,
        shuffle=False, memory_limit_gb=memory_limit_gb
    )
    
    logger.info(f"✅ Streaming loaders created:")
    logger.info(f"   Train: {train_loader.num_sequences:,} sequences, ~{train_loader.num_batches:,} batches")
    logger.info(f"   Val: {val_loader.num_sequences:,} sequences, ~{val_loader.num_batches:,} batches")
    
    return train_loader, val_loader


def test_streaming_performance():
    """Test streaming loader performance and memory usage"""
    import psutil
    import gc
    
    print("🧪 TESTING STREAMING DATA LOADER PERFORMANCE")
    print("=" * 60)
    
    # Test data directory (adjust path as needed)
    cache_dir = str(Path.home() / ".cache" / "openwebtext_gpt2_parallel")
    
    if not Path(cache_dir).exists():
        print(f"❌ Test data not found at {cache_dir}")
        print("   Run parallel_prepare_openwebtext.py first")
        return
    
    # Monitor memory before
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024**3)
    print(f"📊 Memory before: {memory_before:.2f} GB")
    
    # Create streaming loader
    try:
        train_loader, val_loader = create_streaming_loaders(
            cache_dir, seq_len=512, batch_size=8, memory_limit_gb=16.0
        )
        
        memory_after = process.memory_info().rss / (1024**3)
        print(f"📊 Memory after loader creation: {memory_after:.2f} GB")
        print(f"📈 Memory increase: {memory_after - memory_before:.2f} GB")
        
        # Test iteration speed
        print(f"\n⏱️  Testing iteration speed...")
        start_time = time.time()
        batch_count = 0
        
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 100:  # Test first 100 batches
                break
            
            if batch_count % 20 == 0:
                current_memory = process.memory_info().rss / (1024**3)
                elapsed = time.time() - start_time
                batches_per_sec = batch_count / elapsed
                print(f"   Batch {batch_count}: {batches_per_sec:.1f} batches/sec, Memory: {current_memory:.2f} GB")
        
        elapsed = time.time() - start_time
        batches_per_sec = batch_count / elapsed
        final_memory = process.memory_info().rss / (1024**3)
        
        print(f"\n✅ Performance test complete:")
        print(f"   Speed: {batches_per_sec:.1f} batches/sec")
        print(f"   Final memory: {final_memory:.2f} GB")
        print(f"   Memory stable: {abs(final_memory - memory_after) < 1.0}")
        
        # Test batch content
        input_ids, target_ids = batch
        print(f"   Batch shape: input {input_ids.shape}, target {target_ids.shape}")
        print(f"   Token range: {input_ids.min().item()} - {input_ids.max().item()}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Cleanup
    gc.collect()


if __name__ == "__main__":
    test_streaming_performance()