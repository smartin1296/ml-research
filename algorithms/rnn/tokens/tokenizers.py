"""Token-level tokenizers for RNN models."""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class BPETokenizer:
    """Byte-Pair Encoding tokenizer for token-level prediction."""
    
    def __init__(self, vocab_size: int = 1000, min_freq: int = 2):
        """Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            min_freq: Minimum frequency for merging pairs
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # Initialize with special tokens
        self.token_to_id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.id_to_token: Dict[int, str] = {
            v: k for k, v in self.token_to_id.items()
        }
        
        # BPE specific
        self.merges: List[Tuple[str, str]] = []
        self.word_tokenized: Dict[str, List[str]] = {}
        
    def _get_word_frequencies(self, text: str) -> Dict[str, int]:
        """Get frequency of each word in text."""
        words = text.split()
        word_freq = Counter(words)
        return word_freq
    
    def _initialize_word_tokens(self, word_freq: Dict[str, int]) -> Dict[str, List[str]]:
        """Initialize each word as character-level tokens."""
        word_tokens = {}
        for word in word_freq:
            # Add special end-of-word marker
            tokens = list(word) + ["</w>"]
            word_tokens[word] = tokens
        return word_tokens
    
    def _count_pairs(self, word_tokens: Dict[str, List[str]], 
                     word_freq: Dict[str, int]) -> Counter:
        """Count frequency of adjacent token pairs."""
        pair_freq = Counter()
        for word, tokens in word_tokens.items():
            freq = word_freq[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freq[pair] += freq
        return pair_freq
    
    def _merge_pair(self, word_tokens: Dict[str, List[str]], 
                    pair: Tuple[str, str]) -> Dict[str, List[str]]:
        """Merge most frequent pair in word tokens."""
        new_word_tokens = {}
        merged = pair[0] + pair[1]
        
        for word, tokens in word_tokens.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == pair[0] and 
                    tokens[i + 1] == pair[1]):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_word_tokens[word] = new_tokens
            
        return new_word_tokens
    
    def fit(self, text: str):
        """Learn BPE merges from text.
        
        Args:
            text: Training text to learn BPE from
        """
        # Get word frequencies
        word_freq = self._get_word_frequencies(text)
        
        # Initialize word tokens
        word_tokens = self._initialize_word_tokens(word_freq)
        
        # Get initial vocabulary (all characters)
        vocab = set()
        for tokens in word_tokens.values():
            vocab.update(tokens)
        
        # Add initial characters to vocabulary
        for i, char in enumerate(sorted(vocab), start=len(self.token_to_id)):
            if char not in self.token_to_id:
                self.token_to_id[char] = i
                self.id_to_token[i] = char
        
        # Learn merges
        num_merges = self.vocab_size - len(self.token_to_id)
        for _ in range(num_merges):
            pair_freq = self._count_pairs(word_tokens, word_freq)
            
            if not pair_freq:
                break
                
            # Get most frequent pair
            most_frequent = pair_freq.most_common(1)[0]
            pair, freq = most_frequent
            
            if freq < self.min_freq:
                break
            
            # Merge the pair
            self.merges.append(pair)
            word_tokens = self._merge_pair(word_tokens, pair)
            
            # Add merged token to vocabulary
            merged = pair[0] + pair[1]
            if merged not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged] = idx
                self.id_to_token[idx] = merged
        
        # Store final word tokenizations for fast encoding
        self.word_tokenized = word_tokens
        
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned BPE merges."""
        if word in self.word_tokenized:
            return self.word_tokenized[word]
        
        # For unknown words, apply merges
        tokens = list(word) + ["</w>"]
        
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == pair[0] and 
                    tokens[i + 1] == pair[1]):
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        words = text.split()
        token_ids = []
        
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    token_ids.append(self.token_to_id[self.unk_token])
                    
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in [self.pad_token, self.unk_token, 
                                self.bos_token, self.eos_token]:
                    tokens.append(token)
        
        # Join tokens and remove end-of-word markers
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()
    
    def save(self, path: Path):
        """Save tokenizer to disk."""
        path = Path(path)
        data = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'merges': self.merges,
            'word_tokenized': self.word_tokenized,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        """Load tokenizer from disk."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.min_freq = data['min_freq']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.merges = [tuple(m) for m in data['merges']]
        self.word_tokenized = data['word_tokenized']
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.token_to_id)


class WordTokenizer:
    """Simple word-level tokenizer for comparison."""
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        """Initialize word tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum word frequency to include in vocab
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # Initialize with special tokens
        self.token_to_id: Dict[str, int] = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        self.id_to_token: Dict[int, str] = {
            v: k for k, v in self.token_to_id.items()
        }
        
    def fit(self, text: str):
        """Build vocabulary from text.
        
        Args:
            text: Training text to build vocabulary from
        """
        # Count word frequencies
        words = text.lower().split()
        word_freq = Counter(words)
        
        # Filter by minimum frequency and take top vocab_size
        filtered_words = [
            word for word, freq in word_freq.items() 
            if freq >= self.min_freq
        ]
        
        # Sort by frequency and take top vocab_size - len(special_tokens)
        sorted_words = sorted(
            filtered_words, 
            key=lambda w: word_freq[w], 
            reverse=True
        )[:self.vocab_size - len(self.token_to_id)]
        
        # Add to vocabulary
        for word in sorted_words:
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        words = text.lower().split()
        token_ids = []
        
        for word in words:
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                token_ids.append(self.token_to_id[self.unk_token])
                
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                word = self.id_to_token[token_id]
                if word not in [self.pad_token, self.unk_token,
                              self.bos_token, self.eos_token]:
                    words.append(word)
        
        return " ".join(words)
    
    def save(self, path: Path):
        """Save tokenizer to disk."""
        path = Path(path)
        data = {
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        """Load tokenizer from disk."""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.min_freq = data['min_freq']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.token_to_id)