#!/usr/bin/env python3
"""
Standard Tokenizer for Transformer Phase Comparisons
Ensures consistent tokenization across all phases for fair evaluation
"""

import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

class StandardTransformerTokenizer:
    """
    Standard tokenizer for all transformer phases to ensure fair comparison.
    
    Features:
    - Word-level tokenization optimized for TinyStories
    - Fixed vocabulary size of 2048 (good balance for stories)
    - Consistent special tokens across all phases
    - Reproducible vocabulary building with fixed seed
    - Save/load functionality for exact consistency
    """
    
    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
        # Standard special tokens (same across all phases)
        self.pad_token = 0
        self.unk_token = 1  
        self.bos_token = 2  # Beginning of story
        self.eos_token = 3  # End of story
        
        # Initialize with special tokens
        special_tokens = {
            '<PAD>': self.pad_token,
            '<UNK>': self.unk_token, 
            '<BOS>': self.bos_token,
            '<EOS>': self.eos_token
        }
        
        for token, idx in special_tokens.items():
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
        
        self.built = False
        
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Consistent text tokenization for TinyStories
        - Lowercase normalization
        - Punctuation as separate tokens
        - Word boundaries preserved
        """
        # Normalize text
        text = text.lower().strip()
        
        # Split on whitespace and punctuation, keeping punctuation as separate tokens
        # This regex captures words and common punctuation
        tokens = re.findall(r"\w+|[.!?',;:\-\"()]", text)
        
        return tokens
    
    def build_vocab_from_file(self, data_path: str, max_stories: int = 10000) -> None:
        """
        Build vocabulary from TinyStories file with deterministic results
        
        Args:
            data_path: Path to TinyStories data file
            max_stories: Maximum stories to use for vocab building (for consistency)
        """
        print(f"🔤 Building standard tokenizer vocabulary...")
        print(f"   Data: {data_path}")
        print(f"   Max stories: {max_stories:,}")
        print(f"   Target vocab size: {self.vocab_size:,}")
        
        # Load stories for vocabulary building
        stories = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_stories:
                    break
                if line.strip():
                    try:
                        if line.strip().startswith('{'):
                            data = json.loads(line.strip())
                            story = data.get('story', data.get('text', ''))
                        else:
                            story = line.strip()
                        
                        # Quality filter for consistent vocabulary
                        if story and 50 <= len(story) <= 1000:  # Reasonable story length
                            stories.append(story)
                    except:
                        continue
        
        print(f"   Loaded: {len(stories):,} quality stories")
        
        # Build vocabulary from stories
        self.build_vocab(stories)
        
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts with reproducible results"""
        
        # Collect word frequencies with deterministic ordering
        all_tokens = []
        for text in texts:
            tokens = self._tokenize_text(text)
            all_tokens.extend(tokens)
        
        self.word_freq = Counter(all_tokens)
        
        # Sort by frequency (descending), then alphabetically for deterministic results
        sorted_words = sorted(
            self.word_freq.items(), 
            key=lambda x: (-x[1], x[0])  # Frequency desc, then alphabetical
        )
        
        # Add most frequent words to vocabulary (after special tokens)
        available_slots = self.vocab_size - len(self.word_to_idx)
        
        for word, freq in sorted_words[:available_slots]:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.built = True
        
        print(f"✅ Standard tokenizer built:")
        print(f"   Final vocab size: {len(self.word_to_idx):,}")
        print(f"   Most common words: {list(sorted_words[:10])}")
        
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs with consistent special token handling"""
        if not self.built:
            raise ValueError("Tokenizer not built. Call build_vocab() or load() first.")
        
        tokens = self._tokenize_text(text)
        
        # Convert to IDs
        token_ids = []
        
        if add_special:
            token_ids.append(self.bos_token)
        
        for token in tokens:
            if token in self.word_to_idx:
                token_ids.append(self.word_to_idx[token])
            else:
                token_ids.append(self.unk_token)  # Unknown token
        
        if add_special:
            token_ids.append(self.eos_token)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text with consistent formatting"""
        if not self.built:
            return "<TOKENIZER_NOT_BUILT>"
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.idx_to_word:
                token = self.idx_to_word[token_id]
                
                if skip_special and token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    continue
                    
                tokens.append(token)
            else:
                if not skip_special:
                    tokens.append('<UNK>')
        
        # Rejoin tokens with proper spacing
        text = ' '.join(tokens)
        
        # Fix punctuation spacing (standard formatting)
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'\s+', ' ', text)  # Clean up multiple spaces
        
        return text.strip()
    
    def save(self, path: str) -> None:
        """Save tokenizer for exact reproduction across phases"""
        data = {
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()},
            'word_freq': dict(self.word_freq),
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token, 
                'eos_token': self.eos_token
            },
            'built': self.built
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Standard tokenizer saved to: {path}")
        
    def load(self, path: str) -> None:
        """Load exact tokenizer state for consistency"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        self.word_freq = Counter(data.get('word_freq', {}))
        
        special = data['special_tokens']
        self.pad_token = special['pad_token']
        self.unk_token = special['unk_token']
        self.bos_token = special['bos_token']
        self.eos_token = special['eos_token']
        
        self.built = data.get('built', True)
        
        print(f"📂 Standard tokenizer loaded from: {path}")
        print(f"   Vocab size: {len(self.word_to_idx):,}")
        
    def get_stats(self) -> Dict:
        """Get tokenizer statistics for verification"""
        return {
            'vocab_size': len(self.word_to_idx),
            'total_tokens_seen': sum(self.word_freq.values()),
            'unique_tokens_seen': len(self.word_freq),
            'coverage_ratio': len(self.word_to_idx) / len(self.word_freq) if self.word_freq else 0,
            'special_tokens': {
                'pad': self.pad_token,
                'unk': self.unk_token, 
                'bos': self.bos_token,
                'eos': self.eos_token
            }
        }

def create_standard_tokenizer(data_path: str, save_path: str = None) -> StandardTransformerTokenizer:
    """
    Create and save the standard tokenizer for all phases
    
    Args:
        data_path: Path to TinyStories data
        save_path: Where to save the tokenizer (optional)
        
    Returns:
        Built tokenizer ready for use
    """
    print("🏗️ Creating Standard Transformer Tokenizer")
    print("=" * 50)
    
    # Create tokenizer
    tokenizer = StandardTransformerTokenizer(vocab_size=2048)
    
    # Build from data
    tokenizer.build_vocab_from_file(data_path, max_stories=10000)
    
    # Save if requested
    if save_path:
        tokenizer.save(save_path)
    
    # Print stats
    stats = tokenizer.get_stats()
    print(f"\n📊 Tokenizer Statistics:")
    for key, value in stats.items():
        if key != 'special_tokens':
            print(f"   {key}: {value}")
    
    print("\n✅ Standard tokenizer ready for all phases!")
    return tokenizer

def verify_tokenizer_consistency(tokenizer_path: str) -> bool:
    """
    Verify that a saved tokenizer produces consistent results
    
    Args:
        tokenizer_path: Path to saved tokenizer
        
    Returns:
        True if tokenizer is consistent
    """
    print(f"🔍 Verifying tokenizer consistency: {tokenizer_path}")
    
    # Load tokenizer
    tokenizer = StandardTransformerTokenizer()
    tokenizer.load(tokenizer_path)
    
    # Test with sample stories
    test_stories = [
        "Once upon a time, there was a little girl.",
        "The brave princess went to the magical forest.",
        "A small cat played happily in the garden."
    ]
    
    print("Testing encoding/decoding consistency:")
    all_consistent = True
    
    for i, story in enumerate(test_stories, 1):
        # Encode
        tokens = tokenizer.encode(story, add_special=True)
        
        # Decode
        decoded = tokenizer.decode(tokens, skip_special=True)
        
        # Check consistency (should be very similar after tokenization)
        original_normalized = tokenizer._tokenize_text(story)
        decoded_normalized = tokenizer._tokenize_text(decoded)
        
        consistent = original_normalized == decoded_normalized
        all_consistent = all_consistent and consistent
        
        print(f"  Story {i}: {'✅' if consistent else '❌'}")
        print(f"    Original: {story}")
        print(f"    Tokens: {len(tokens)} tokens")
        print(f"    Decoded: {decoded}")
        if not consistent:
            print(f"    ⚠️  Mismatch in tokenization!")
        print()
    
    if all_consistent:
        print("✅ Tokenizer consistency verified!")
    else:
        print("❌ Tokenizer consistency issues found!")
    
    return all_consistent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Standard Transformer Tokenizer")
    parser.add_argument("--create", action="store_true", help="Create new standard tokenizer")
    parser.add_argument("--verify", type=str, help="Verify existing tokenizer consistency")
    parser.add_argument("--data", type=str, default="data/raw/text/tinystories/TinyStories-small.txt", 
                       help="Path to TinyStories data")
    parser.add_argument("--output", type=str, default="algorithms/transformers/standard_tokenizer.json",
                       help="Output path for tokenizer")
    
    args = parser.parse_args()
    
    if args.create:
        # Create standard tokenizer
        tokenizer = create_standard_tokenizer(args.data, args.output)
        
        # Verify it works
        verify_tokenizer_consistency(args.output)
        
    elif args.verify:
        # Just verify existing tokenizer
        verify_tokenizer_consistency(args.verify)
        
    else:
        print("Use --create to build new tokenizer or --verify <path> to check existing one")
        
        # Quick demo
        print("\n🧪 Quick Demo:")
        data_path = Path(args.data)
        if data_path.exists():
            tokenizer = create_standard_tokenizer(str(data_path))
        else:
            print(f"Data not found at {data_path}, using sample stories")
            sample_stories = [
                "Once upon a time, there was a brave princess who lived in a magical castle.",
                "The little dragon loved to play with the village children every sunny day.",
                "In a beautiful garden, colorful flowers danced in the gentle breeze."
            ]
            tokenizer = StandardTransformerTokenizer()
            tokenizer.build_vocab(sample_stories)
        
        # Test encoding/decoding
        test_text = "Once upon a time, there was a little cat."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"\n📝 Test:")
        print(f"   Original: {test_text}")
        print(f"   Tokens: {tokens}")
        print(f"   Decoded: {decoded}")