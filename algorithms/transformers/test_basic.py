#!/usr/bin/env python3
"""
Basic Transformer Test
Test the original "Attention is All You Need" implementation on a simple task
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.core import Transformer, TransformerTrainer
from algorithms.transformers.core.attention import create_padding_mask, create_look_ahead_mask


class SimpleTokenizer:
    """Simple character-level tokenizer for testing"""
    
    def __init__(self, text: str):
        # Create character vocabulary
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}  # +1 to reserve 0 for padding
        self.char_to_idx['<PAD>'] = 0
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        print(f"📚 Vocabulary size: {self.vocab_size}")
        print(f"   Characters: {''.join(chars[:20])}{'...' if len(chars) > 20 else ''}")
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs"""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: list) -> str:
        """Decode token IDs to text"""
        return ''.join([self.idx_to_char.get(t, '<UNK>') for t in tokens if t != 0])


class SequenceDataset(Dataset):
    """Simple sequence-to-sequence dataset for testing"""
    
    def __init__(self, text: str, tokenizer: SimpleTokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Encode entire text
        self.tokens = tokenizer.encode(text)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_len - 1, seq_len // 2):  # 50% overlap
            src = self.tokens[i:i + seq_len]
            tgt = self.tokens[i + 1:i + seq_len + 1]  # Target is shifted by 1
            
            # Pad if necessary
            if len(src) < seq_len:
                src += [0] * (seq_len - len(src))
                tgt += [0] * (seq_len - len(tgt))
            
            self.sequences.append((src, tgt))
        
        print(f"📊 Created {len(self.sequences)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        src, tgt = self.sequences[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def create_masks(src: torch.Tensor, tgt: torch.Tensor) -> tuple:
    """Create masks for Transformer"""
    batch_size, src_len = src.shape
    tgt_len = tgt.shape[1]
    
    # Padding masks
    src_padding_mask = create_padding_mask(src, pad_token=0)
    
    # For decoder: combine look-ahead mask and padding mask
    look_ahead_mask = create_look_ahead_mask(tgt_len - 1, src.device)  # -1 because we use tgt[:-1]
    tgt_padding_mask = create_padding_mask(tgt[:, :-1], pad_token=0)
    
    # Combine masks for decoder self-attention
    tgt_mask = torch.minimum(look_ahead_mask, tgt_padding_mask)
    
    return src_padding_mask, tgt_mask, src_padding_mask  # Use src_padding_mask for encoder-decoder attention


class SimpleTransformer(nn.Module):
    """Simplified Transformer for language modeling"""
    
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 8, 
                 num_layers: int = 4, d_ff: int = 512, max_seq_len: int = 1000):
        super().__init__()
        
        # Use just the encoder for language modeling
        from algorithms.transformers.core.models import TransformerEncoder
        
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            max_seq_length=max_seq_len
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for language modeling"""
        # Encoder forward pass
        encoder_output = self.encoder(x, mask)
        
        # Project to vocabulary
        output = self.output_projection(encoder_output)
        
        return output


def test_transformer_basic():
    """Test basic transformer functionality"""
    print("🔧 Testing Basic Transformer Implementation")
    print("=" * 60)
    
    # Load text data (use Shakespeare as in RNN tests)
    data_path = project_root / "data" / "shakespeare.txt"
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()[:50000]  # Use first 50k characters for testing
    
    print(f"📖 Loaded {len(text):,} characters of text")
    
    # Create tokenizer and dataset
    tokenizer = SimpleTokenizer(text)
    
    # Create datasets
    seq_len = 32  # Shorter sequences for faster testing
    dataset = SequenceDataset(text, tokenizer, seq_len)
    
    # Split data (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 64  # Start with moderate batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Create model (smaller for initial testing)
    model = SimpleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,    # Smaller than original 512
        num_heads=8,    # Original
        num_layers=4,   # Smaller than original 6
        d_ff=512,       # Smaller than original 2048
        max_seq_len=seq_len
    )
    
    print(f"🧠 Model created with {model.encoder.count_parameters():,} parameters")
    
    # Setup trainer
    save_dir = project_root / "algorithms" / "transformers" / "results" / "basic_test"
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        save_dir=str(save_dir)
    )
    
    # Quick training test (just a few epochs)
    print(f"\n🚀 Starting training test...")
    results = trainer.train(
        learning_rate=1e-4,
        max_epochs=10,    # Just a quick test
        warmup_steps=1000,
        d_model=128
    )
    
    print(f"\n✅ Basic transformer test completed!")
    print(f"   Final train loss: {results['train_losses'][-1]:.4f}")
    print(f"   Final val loss: {results['val_losses'][-1]:.4f}")
    print(f"   Final val accuracy: {results['val_accuracies'][-1]:.3f}")
    
    # Test text generation
    print(f"\n🎨 Testing text generation...")
    test_generation(model, tokenizer, trainer.device)


def test_generation(model: nn.Module, tokenizer: SimpleTokenizer, device: torch.device):
    """Test basic text generation"""
    model.eval()
    
    # Start with a seed text
    seed_text = "The "
    seed_tokens = tokenizer.encode(seed_text)
    
    generated_tokens = seed_tokens.copy()
    max_length = 50
    
    with torch.no_grad():
        for _ in range(max_length - len(seed_tokens)):
            # Prepare input
            input_tensor = torch.tensor([generated_tokens], dtype=torch.long).to(device)
            
            # Forward pass
            output = model(input_tensor)
            
            # Get next token (greedy sampling)
            next_token_logits = output[0, -1, :]  # Last position, all vocabulary
            next_token = torch.argmax(next_token_logits).item()
            
            # Stop if we hit padding token
            if next_token == 0:
                break
                
            generated_tokens.append(next_token)
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens)
    print(f"   Seed: '{seed_text}'")
    print(f"   Generated: '{generated_text}'")


if __name__ == "__main__":
    test_transformer_basic()