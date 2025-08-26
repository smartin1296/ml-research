"""Token-level dataset for next token prediction."""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from tokens.tokenizers import BPETokenizer, WordTokenizer


class TokenSequenceDataset(Dataset):
    """Dataset for token-level next token prediction."""
    
    def __init__(
        self,
        text: str,
        tokenizer: Union[BPETokenizer, WordTokenizer],
        seq_len: int = 50,
        stride: int = 1,
        train: bool = True,
        train_split: float = 0.8,
    ):
        """Initialize token sequence dataset.
        
        Args:
            text: Input text data
            tokenizer: Tokenizer to use (BPE or Word)
            seq_len: Length of input sequences
            stride: Stride for creating overlapping sequences
            train: Whether this is training or validation dataset
            train_split: Fraction of data to use for training
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.train = train
        
        # Encode entire text
        self.token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        # Split into train/val
        split_idx = int(len(self.token_ids) * train_split)
        if train:
            self.token_ids = self.token_ids[:split_idx]
        else:
            self.token_ids = self.token_ids[split_idx:]
        
        # Calculate number of sequences
        self.num_sequences = max(0, (len(self.token_ids) - seq_len - 1) // stride + 1)
        
    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its target.
        
        Args:
            idx: Index of sequence to retrieve
            
        Returns:
            Tuple of (input_sequence, target_token)
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len
        
        input_seq = self.token_ids[start_idx:end_idx]
        target = self.token_ids[end_idx]
        
        return input_seq, target


class TokenTextGenerator:
    """Generate text using trained token-level models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Union[BPETokenizer, WordTokenizer],
        device: torch.device,
    ):
        """Initialize text generator.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer used for training
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Text prompt to continue from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top k tokens (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt)
        
        # Handle empty prompt
        if not token_ids:
            token_ids = [self.tokenizer.token_to_id[self.tokenizer.bos_token]]
        
        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Generate tokens
        generated_ids = []
        hidden = None
        
        for _ in range(max_tokens):
            # Forward pass
            with torch.cuda.amp.autocast(enabled=False):  # Disabled for stability
                logits, hidden = self.model(input_ids, hidden)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Find cutoff
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                # Set filtered positions to -inf
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Add to generated sequence
            generated_ids.append(next_token_id.item())
            
            # Check for EOS token
            if next_token_id.item() == self.tokenizer.token_to_id.get(self.tokenizer.eos_token, -1):
                break
            
            # Update input for next iteration
            input_ids = next_token_id
        
        # Decode generated tokens
        all_token_ids = token_ids + generated_ids
        generated_text = self.tokenizer.decode(all_token_ids)
        
        return generated_text


def load_shakespeare_data() -> str:
    """Load Shakespeare dataset."""
    import urllib.request
    
    data_path = Path("../../data/shakespeare.txt")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r') as f:
        text = f.read()
    
    return text


def create_token_datasets(
    text: str,
    tokenizer: Union[BPETokenizer, WordTokenizer],
    seq_len: int = 50,
    stride: int = 1,
    train_split: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use
        seq_len: Sequence length
        stride: Stride for overlapping sequences
        train_split: Train/val split ratio
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = TokenSequenceDataset(
        text=text,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        train=True,
        train_split=train_split,
    )
    
    val_dataset = TokenSequenceDataset(
        text=text,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=stride,
        train=False,
        train_split=train_split,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader