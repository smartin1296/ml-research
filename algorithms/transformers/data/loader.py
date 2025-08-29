#!/usr/bin/env python3
"""
OpenWebText Dataset Loader
Optimized for M1 Max 64GB RAM transformer benchmarking
Downloads and processes OpenWebText for phase evolution testing
"""

import sys
import os
import json
import hashlib
import random
from pathlib import Path
from typing import Optional, List, Dict, Iterator
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.data.tokenizer import StandardTransformerTokenizer


class OpenWebTextDataset(Dataset):
    """OpenWebText dataset with proper train/val splits for transformer benchmarking"""
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        split_type: str = "train",
        subset_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        min_length: int = 50
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_type = split_type
        self.min_length = min_length
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = str(Path.home() / ".cache" / "openwebtext")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🌐 Loading OpenWebText {split_type} dataset...")
        print(f"   Cache: {cache_dir}")
        print(f"   Max length: {max_length}, Min length: {min_length}")
        
        # Load processed data or create it
        self.processed_file = self.cache_dir / f"processed_{split_type}_{max_length}_{subset_size or 'full'}.json"
        
        if self.processed_file.exists():
            print(f"   📦 Loading cached processed data...")
            self.tokenized_texts = self._load_processed()
        else:
            print(f"   🔄 Processing raw dataset...")
            self.tokenized_texts = self._process_dataset(subset_size)
            self._save_processed()
        
        print(f"   ✅ Final {split_type} dataset: {len(self.tokenized_texts):,} examples")
    
    def _process_dataset(self, subset_size: Optional[int]) -> List[List[int]]:
        """Process raw OpenWebText dataset"""
        
        # Load dataset
        print("   📥 Downloading OpenWebText...")
        try:
            dataset = load_dataset("Skylion007/openwebtext", cache_dir=str(self.cache_dir))['train']
        except Exception as e:
            print(f"   ❌ Failed to load OpenWebText: {e}")
            print("   💡 Fallback: Using local text files if available...")
            return self._load_local_fallback()
        
        print(f"   📊 Raw dataset: {len(dataset):,} documents")
        
        # Create content-based train/val split
        train_texts = []
        val_texts = []
        
        total_processed = 0
        target_docs = subset_size or len(dataset)
        
        print(f"   🔄 Processing documents (target: {target_docs:,})...")
        
        for i, example in enumerate(tqdm(dataset, desc="Processing")):
            if total_processed >= target_docs:
                break
            
            text = example.get('text', '').strip()
            
            # Quality filters
            if not text or len(text) < 200:  # Skip very short texts
                continue
            
            # Remove texts that are too repetitive
            if self._is_too_repetitive(text):
                continue
            
            # Content-based splitting using text hash
            text_hash = hashlib.md5(text[:100].encode()).hexdigest()
            hash_int = int(text_hash[:8], 16)
            
            if hash_int % 10 == 0:  # 10% validation
                val_texts.append(text)
            else:  # 90% training
                train_texts.append(text)
            
            total_processed += 1
            
            # Progress update
            if total_processed % 10000 == 0:
                print(f"   📈 Processed: {total_processed:,} docs")
        
        # Select appropriate split
        if self.split_type == "train":
            selected_texts = train_texts
        else:
            selected_texts = val_texts
        
        print(f"   📝 Selected {len(selected_texts):,} {self.split_type} texts")
        
        # Tokenize texts
        tokenized_texts = []
        
        print(f"   🔤 Tokenizing texts...")
        for text in tqdm(selected_texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text, add_special=True)
            
            # Length filtering
            if self.min_length <= len(tokens) <= self.max_length:
                tokenized_texts.append(tokens)
        
        return tokenized_texts
    
    def _is_too_repetitive(self, text: str) -> bool:
        """Check if text is too repetitive (spam, boilerplate, etc.)"""
        
        # Split into words
        words = text.split()
        if len(words) < 10:
            return True
        
        # Check for excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any single word appears more than 30% of the time, it's repetitive
        max_word_pct = max(word_counts.values()) / len(words)
        if max_word_pct > 0.3:
            return True
        
        # Check for repeated phrases (simple heuristic)
        if len(set(words)) / len(words) < 0.3:  # Low diversity
            return True
        
        return False
    
    def _load_local_fallback(self) -> List[List[int]]:
        """Fallback to local text files if OpenWebText download fails"""
        
        print("   🔍 Looking for local text files...")
        
        # Look for any .txt files in data directories
        data_dirs = [
            Path("data/raw/text"),
            Path("data/raw"),
            Path("data")
        ]
        
        texts = []
        for data_dir in data_dirs:
            if data_dir.exists():
                for txt_file in data_dir.rglob("*.txt"):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if len(content) > 500:  # Minimum size
                                # Split into chunks
                                chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
                                texts.extend(chunks[:100])  # Limit per file
                    except Exception as e:
                        continue
        
        if not texts:
            print("   ❌ No suitable local text files found")
            return []
        
        print(f"   📁 Found {len(texts)} local text chunks")
        
        # Process local texts similar to OpenWebText
        tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special=True)
            if self.min_length <= len(tokens) <= self.max_length:
                tokenized_texts.append(tokens)
        
        return tokenized_texts
    
    def _save_processed(self):
        """Save processed tokenized texts to cache"""
        print(f"   💾 Caching processed data to {self.processed_file}")
        with open(self.processed_file, 'w') as f:
            json.dump(self.tokenized_texts, f)
    
    def _load_processed(self) -> List[List[int]]:
        """Load processed tokenized texts from cache"""
        with open(self.processed_file, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx].copy()
        
        # Pad to max length
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input/target pairs for language modeling
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


def download_and_prepare_openwebtext(
    cache_dir: Optional[str] = None,
    subset_size: Optional[int] = 100000,  # 100K documents for testing
    force_redownload: bool = False
):
    """Download and prepare OpenWebText dataset"""
    
    print("🌐 OPENWEBTEXT DATASET PREPARATION")
    print("=" * 50)
    
    if cache_dir is None:
        cache_dir = str(Path.home() / ".cache" / "openwebtext")
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = Path("algorithms/transformers/standard_tokenizer.json")
    if not tokenizer_path.exists():
        print("❌ Standard tokenizer not found. Please create it first:")
        print("   python algorithms/transformers/standard_tokenizer.py --create")
        return None
    
    tokenizer = StandardTransformerTokenizer()
    tokenizer.load(str(tokenizer_path))
    
    print(f"📂 Tokenizer: {len(tokenizer.word_to_idx):,} vocab")
    
    # Test dataset creation
    try:
        print(f"\n🧪 Testing dataset creation (subset: {subset_size:,})...")
        
        train_dataset = OpenWebTextDataset(
            tokenizer=tokenizer,
            max_length=512,
            split_type="train",
            subset_size=subset_size,
            cache_dir=cache_dir
        )
        
        val_dataset = OpenWebTextDataset(
            tokenizer=tokenizer,
            max_length=512,
            split_type="val",
            subset_size=subset_size,
            cache_dir=cache_dir
        )
        
        print(f"\n✅ Dataset ready!")
        print(f"   Train: {len(train_dataset):,} examples")
        print(f"   Val: {len(val_dataset):,} examples")
        
        # Test sample
        if len(train_dataset) > 0:
            input_sample, target_sample = train_dataset[0]
            decoded = tokenizer.decode(input_sample[:50].tolist(), skip_special=True)
            print(f"   Sample text: '{decoded[:100]}...'")
        
        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "cache_dir": cache_dir
        }
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare OpenWebText dataset")
    parser.add_argument("--cache-dir", type=str, help="Cache directory")
    parser.add_argument("--subset-size", type=int, default=100000, help="Number of documents to process")
    parser.add_argument("--force-redownload", action="store_true", help="Force redownload")
    
    args = parser.parse_args()
    
    result = download_and_prepare_openwebtext(
        cache_dir=args.cache_dir,
        subset_size=args.subset_size,
        force_redownload=args.force_redownload
    )
    
    if result:
        print(f"\n🎉 OpenWebText dataset ready for transformer benchmarking!")
    else:
        print(f"\n❌ Failed to prepare dataset")