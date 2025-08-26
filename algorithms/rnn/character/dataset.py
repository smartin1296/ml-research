import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import requests
import string
import re
from pathlib import Path

class CharacterTokenizer:
    """Character-level tokenizer for RNN text modeling"""
    
    def __init__(self, text: Optional[str] = None, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.char_to_idx = vocab
            self.idx_to_char = {idx: char for char, idx in vocab.items()}
        elif text is not None:
            self._build_vocab(text)
        else:
            # Default ASCII vocabulary
            chars = string.printable
            self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.vocab_size = len(self.char_to_idx)
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        
        # Add special tokens if not present
        if self.unk_token not in self.char_to_idx:
            self.char_to_idx[self.unk_token] = len(self.char_to_idx)
            self.idx_to_char[self.char_to_idx[self.unk_token]] = self.unk_token
            self.vocab_size += 1
            
        if self.pad_token not in self.char_to_idx:
            self.char_to_idx[self.pad_token] = len(self.char_to_idx)
            self.idx_to_char[self.char_to_idx[self.pad_token]] = self.pad_token
            self.vocab_size += 1
    
    def _build_vocab(self, text: str):
        """Build character vocabulary from text"""
        unique_chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        return [self.char_to_idx.get(char, self.char_to_idx[self.unk_token]) 
                for char in text]
    
    def decode(self, indices: Union[List[int], torch.Tensor]) -> str:
        """Convert indices back to text"""
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        return ''.join([self.idx_to_char.get(idx, self.unk_token) for idx in indices])
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.char_to_idx, f, indent=2)
    
    @classmethod
    def load_vocab(cls, filepath: str):
        """Load vocabulary from file"""
        import json
        with open(filepath, 'r') as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

class TextSequenceDataset(Dataset):
    """
    Dataset for character-level text sequence modeling
    Creates overlapping sequences for next-token prediction
    """
    
    def __init__(self, 
                 text: str,
                 sequence_length: int = 100,
                 tokenizer: Optional[CharacterTokenizer] = None,
                 stride: int = 1):
        """
        Args:
            text: Input text string
            sequence_length: Length of each sequence
            tokenizer: Character tokenizer (created if None)
            stride: Step size for creating sequences (1 = max overlap)
        """
        self.text = self._preprocess_text(text)
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Create or use tokenizer
        if tokenizer is None:
            self.tokenizer = CharacterTokenizer(self.text)
        else:
            self.tokenizer = tokenizer
        
        # Encode text
        self.encoded_text = self.tokenizer.encode(self.text)
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
        return text.strip()
    
    def _create_sequences(self) -> List[Tuple[List[int], int]]:
        """Create (input_sequence, target) pairs"""
        sequences = []
        
        for i in range(0, len(self.encoded_text) - self.sequence_length, self.stride):
            input_seq = self.encoded_text[i:i + self.sequence_length]
            target = self.encoded_text[i + self.sequence_length]
            sequences.append((input_seq, target))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

class TextDataLoader:
    """Utility class for creating text data loaders with proper collation"""
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for variable length sequences"""
        inputs, targets = zip(*batch)
        
        # Stack inputs and targets
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        return inputs, targets
    
    @staticmethod
    def create_dataloaders(dataset: TextSequenceDataset,
                          train_ratio: float = 0.8,
                          batch_size: int = 32,
                          num_workers: int = 2,
                          shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders"""
        
        # Split dataset
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=TextDataLoader.collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=TextDataLoader.collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader

def download_shakespeare() -> str:
    """Download Shakespeare text for demo purposes"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except (requests.RequestException, Exception) as e:
        print(f"Could not download Shakespeare text: {e}")
        # Return a simple fallback text
        return """
        To be or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause—there's the respect
        That makes calamity of so long life.
        """ * 100  # Repeat to make it longer

def create_sample_dataset(sequence_length: int = 100,
                         batch_size: int = 32,
                         data_source: str = 'shakespeare') -> Tuple[DataLoader, DataLoader, CharacterTokenizer]:
    """
    Create sample dataset for RNN training
    
    Args:
        sequence_length: Length of input sequences
        batch_size: Batch size for data loaders
        data_source: 'shakespeare' or custom text
        
    Returns:
        train_loader, val_loader, tokenizer
    """
    
    if data_source == 'shakespeare':
        text = download_shakespeare()
    else:
        # Use provided text or default
        text = data_source if len(data_source) > 100 else download_shakespeare()
    
    # Create tokenizer and dataset
    tokenizer = CharacterTokenizer(text)
    dataset = TextSequenceDataset(text, sequence_length=sequence_length, 
                                tokenizer=tokenizer)
    
    # Create data loaders
    train_loader, val_loader = TextDataLoader.create_dataloaders(
        dataset, batch_size=batch_size
    )
    
    print(f"Dataset created:")
    print(f"  Text length: {len(text):,} characters")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Training sequences: {len(train_loader.dataset):,}")
    print(f"  Validation sequences: {len(val_loader.dataset):,}")
    print(f"  Sequence length: {sequence_length}")
    
    return train_loader, val_loader, tokenizer

class SequenceGenerator:
    """Generate text using trained RNN models"""
    
    def __init__(self, model: torch.nn.Module, tokenizer: CharacterTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(self, 
                 seed_text: str = "",
                 max_length: int = 200,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> str:
        """
        Generate text using the trained model
        
        Args:
            seed_text: Initial text to start generation
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only consider top k tokens for sampling
            
        Returns:
            Generated text
        """
        
        if not seed_text:
            # Start with a random character from vocab
            seed_text = np.random.choice(list(self.tokenizer.char_to_idx.keys()))
        
        # Encode seed text
        input_seq = self.tokenizer.encode(seed_text)
        generated_text = seed_text
        
        # Initialize hidden state
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                x = torch.tensor([input_seq], dtype=torch.long).to(self.device)
                
                # Forward pass
                output, hidden = self.model(x, hidden)
                
                # Get last output
                logits = output[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Add to sequence
                char = self.tokenizer.decode([next_token])
                generated_text += char
                
                # Update input sequence (sliding window)
                input_seq = input_seq[-99:] + [next_token]  # Keep last 99 + new token
        
        return generated_text