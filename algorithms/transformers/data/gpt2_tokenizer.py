#!/usr/bin/env python3
"""
GPT-2 Compatible Tokenizer for SOTA Transformer Training
Uses the exact same tokenizer as GPT-2 for fair comparison
"""

import torch
from transformers import GPT2Tokenizer
from pathlib import Path
import json
from typing import List, Optional, Union

class GPT2CompatibleTokenizer:
    """
    GPT-2 compatible tokenizer using HuggingFace implementation.
    
    This provides the exact same tokenization as GPT-2 with:
    - 50,257 vocabulary size
    - BPE encoding
    - Proper handling of special tokens
    - Compatible with all existing training code
    """
    
    def __init__(self):
        print("🤖 Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Add padding token (GPT-2 doesn't have one by default)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Store key properties for compatibility
        self.vocab_size = len(self.tokenizer)
        self.pad_token = self.tokenizer.pad_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.bos_token = self.tokenizer.bos_token_id or self.eos_token  # GPT-2 uses EOS as BOS
        self.unk_token = self.tokenizer.unk_token_id
        
        print(f"✅ GPT-2 tokenizer loaded:")
        print(f"   Vocabulary size: {self.vocab_size:,}")
        print(f"   PAD token: {self.pad_token}")
        print(f"   EOS token: {self.eos_token}")
        print(f"   BOS token: {self.bos_token}")
        print(f"   UNK token: {self.unk_token}")
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        if add_special:
            # Add BOS token at start, EOS at end
            tokens = [self.bos_token] + self.tokenizer.encode(text) + [self.eos_token]
        else:
            tokens = self.tokenizer.encode(text)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special: Whether to skip special tokens in output
        
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)
    
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None, 
                    padding: bool = True, truncation: bool = True) -> torch.Tensor:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
        
        Returns:
            Tensor of shape (batch_size, seq_len)
        """
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt'
        )
        
        return encoded['input_ids']
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size
    
    def save(self, path: str):
        """Save tokenizer configuration"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the HuggingFace tokenizer
        self.tokenizer.save_pretrained(str(save_path))
        
        # Save our compatibility info
        config = {
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'eos_token': self.eos_token,
            'bos_token': self.bos_token,
            'unk_token': self.unk_token,
            'tokenizer_type': 'gpt2_compatible'
        }
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from saved path"""
        load_path = Path(path)
        
        # Load the HuggingFace tokenizer
        tokenizer = cls()
        tokenizer.tokenizer = GPT2Tokenizer.from_pretrained(str(load_path))
        
        # Load our compatibility info
        with open(load_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        tokenizer.vocab_size = config['vocab_size']
        tokenizer.pad_token = config['pad_token']
        tokenizer.eos_token = config['eos_token']
        tokenizer.bos_token = config['bos_token']
        tokenizer.unk_token = config['unk_token']
        
        print(f"📂 Tokenizer loaded from {path}")
        return tokenizer


def test_tokenizer():
    """Test the GPT-2 compatible tokenizer"""
    print("\n🧪 TESTING GPT-2 COMPATIBLE TOKENIZER")
    print("=" * 50)
    
    # Create tokenizer
    tokenizer = GPT2CompatibleTokenizer()
    
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog. This is a test of the GPT-2 tokenizer."
    
    # Test encoding
    print(f"\n📝 Original text: '{test_text}'")
    
    tokens = tokenizer.encode(test_text, add_special=True)
    print(f"🔢 Encoded tokens: {tokens[:10]}... (length: {len(tokens)})")
    
    # Test decoding
    decoded = tokenizer.decode(tokens, skip_special=True)
    print(f"📖 Decoded text: '{decoded}'")
    
    # Test batch encoding
    texts = [
        "Hello world!",
        "This is a longer sentence to test tokenization.",
        "Short."
    ]
    
    print(f"\n📦 Batch encoding {len(texts)} texts...")
    batch_encoded = tokenizer.batch_encode(texts, max_length=20, padding=True)
    print(f"   Shape: {batch_encoded.shape}")
    print(f"   Example: {batch_encoded[0].tolist()}")
    
    # Test vocabulary size
    print(f"\n📊 Vocabulary size: {tokenizer.get_vocab_size():,}")
    
    # Compare with original small tokenizer
    print(f"\n📈 Improvement: {tokenizer.get_vocab_size() / 2048:.1f}x larger vocabulary")
    
    return tokenizer


def create_and_save_gpt2_tokenizer():
    """Create and save GPT-2 tokenizer for the project"""
    print("\n🚀 CREATING PROJECT GPT-2 TOKENIZER")
    print("=" * 50)
    
    # Create tokenizer
    tokenizer = GPT2CompatibleTokenizer()
    
    # Save to project location
    save_path = Path("algorithms/transformers/data/gpt2_tokenizer")
    tokenizer.save(str(save_path))
    
    # Test a quick encode/decode cycle
    test_text = "OpenWebText is a dataset for training large language models."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens, skip_special=True)
    
    print(f"\n✅ Tokenizer created and tested:")
    print(f"   📝 Test: '{test_text}'")
    print(f"   🔢 Tokens: {len(tokens)} ({tokens[:5]}...)")
    print(f"   📖 Decoded: '{decoded}'")
    print(f"   💾 Saved to: {save_path}")
    
    return tokenizer, save_path


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = test_tokenizer()
    
    # Create and save for project use
    print("\n" + "="*60)
    project_tokenizer, save_path = create_and_save_gpt2_tokenizer()
    
    print(f"\n🎉 GPT-2 compatible tokenizer ready!")
    print(f"   Import with: from data.gpt2_tokenizer import GPT2CompatibleTokenizer")
    print(f"   Load with: tokenizer = GPT2CompatibleTokenizer.load('{save_path}')")