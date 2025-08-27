#!/usr/bin/env python3
"""
Proper Word-Level Tokenizer for Scaled Phase 1
Move beyond character-level to word/subword tokenization
"""

import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter


class ScaledWordTokenizer:
    """
    Word-level tokenizer with subword fallback for better story generation
    """
    
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
        # Special tokens
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
        """Convert text to word tokens with punctuation handling"""
        # Lowercase and handle punctuation
        text = text.lower()
        
        # Split on whitespace and punctuation, keeping punctuation as separate tokens
        tokens = re.findall(r"\w+|[.!?',;:-]", text)
        
        return tokens
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from training texts"""
        print(f"Building vocabulary from {len(texts):,} stories...")
        
        # Collect all word frequencies
        for text in texts:
            tokens = self._tokenize_text(text)
            self.word_freq.update(tokens)
        
        # Select most frequent tokens for vocabulary
        # Reserve space for special tokens
        available_slots = self.vocab_size - len(self.word_to_idx)
        most_common = self.word_freq.most_common(available_slots)
        
        # Add to vocabulary
        for word, freq in most_common:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.built = True
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs"""
        if not self.built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
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
        """Decode token IDs to text"""
        if not self.built:
            return "<VOCAB_NOT_BUILT>"
        
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
        
        # Rejoin tokens into text
        text = ' '.join(tokens)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'\s+', ' ', text)  # Clean up multiple spaces
        
        return text.strip()
    
    def batch_encode(self, texts: List[str], max_length: int = 256, add_special: bool = True) -> torch.Tensor:
        """Batch encode with padding"""
        batch_tokens = []
        
        for text in texts:
            tokens = self.encode(text, add_special=add_special)
            
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.eos_token] if add_special else tokens[:max_length]
            
            # Pad if too short
            while len(tokens) < max_length:
                tokens.append(self.pad_token)
            
            batch_tokens.append(tokens)
        
        return torch.tensor(batch_tokens, dtype=torch.long)
    
    def save(self, path: str) -> None:
        """Save tokenizer to file"""
        data = {
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()},  # JSON keys must be strings
            'word_freq': dict(self.word_freq),
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token, 
                'eos_token': self.eos_token
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
    
    def load(self, path: str) -> None:
        """Load tokenizer from file"""
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
        
        self.built = True
        


def create_scaled_tokenizer(stories: List[str], vocab_size: int = 8192) -> ScaledWordTokenizer:
    """Create and train word-level tokenizer"""
    tokenizer = ScaledWordTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(stories)
    return tokenizer


if __name__ == "__main__":
    # Test the tokenizer
    test_stories = [
        "Once upon a time, there was a brave princess who lived in a magical castle.",
        "The little cat played happily in the garden with colorful butterflies.",
        "Every day, the friendly dragon helped the village children learn to read."
    ]
    
    print("🧪 Testing Scaled Word Tokenizer")
    
    tokenizer = create_scaled_tokenizer(test_stories, vocab_size=100)
    
    # Test encoding/decoding
    test_text = "Once upon a time, there was a brave princess."
    print(f"\nOriginal: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)  
    print(f"Decoded: {decoded}")
    
    print("✅ Tokenizer test complete!")