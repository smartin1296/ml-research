#!/usr/bin/env python3
"""Training script for token-level RNN models."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .device_utils import get_best_device
from .results_utils import print_results_table
from .token_dataset import (
    TokenSequenceDataset,
    TokenTextGenerator,
    create_token_datasets,
    load_shakespeare_data,
)
from .token_models import TokenRNNModel, OptimizedTokenRNNModel, get_token_model_config
from .tokenizers import BPETokenizer, WordTokenizer
from .trainer import RNNTrainer


def train_token_rnn(
    tokenizer_type: str = "bpe",
    vocab_size: int = 1000,
    model_size: str = "medium",
    seq_len: int = 50,
    batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    use_optimized: bool = True,
    device: Optional[torch.device] = None,
) -> Dict:
    """Train a token-level RNN model.
    
    Args:
        tokenizer_type: Type of tokenizer (bpe or word)
        vocab_size: Vocabulary size
        model_size: Model size (small, medium, large)
        seq_len: Sequence length
        batch_size: Batch size
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        use_optimized: Whether to use optimized model
        device: Device to train on
        
    Returns:
        Dictionary of results
    """
    # Get device
    if device is None:
        device = get_best_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading Shakespeare dataset...")
    text = load_shakespeare_data()
    print(f"Text length: {len(text):,} characters")
    
    # Initialize tokenizer
    print(f"\nInitializing {tokenizer_type.upper()} tokenizer...")
    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size, min_freq=2)
    else:
        tokenizer = WordTokenizer(vocab_size=vocab_size, min_freq=2)
    
    # Fit tokenizer
    print("Fitting tokenizer on text...")
    tokenizer.fit(text)
    actual_vocab_size = len(tokenizer)
    print(f"Actual vocabulary size: {actual_vocab_size:,}")
    
    # Save tokenizer
    tokenizer_path = Path(f"checkpoints/tokenizer_{tokenizer_type}_{vocab_size}.json")
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_loader, val_loader = create_token_datasets(
        text=text,
        tokenizer=tokenizer,
        seq_len=seq_len,
        stride=1,
        train_split=0.8,
        batch_size=batch_size,
        num_workers=0,
    )
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Get model config
    model_config = get_token_model_config(actual_vocab_size, model_size)
    
    # Initialize model
    print(f"\nInitializing {model_size} model...")
    if use_optimized:
        model = OptimizedTokenRNNModel(
            vocab_size=actual_vocab_size,
            **model_config,
        )
    else:
        model = TokenRNNModel(
            vocab_size=actual_vocab_size,
            rnn_type="lstm",
            **model_config,
        )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader),
        eta_min=learning_rate * 0.1,
    )
    
    # Initialize trainer
    criterion = nn.CrossEntropyLoss()
    trainer = RNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        grad_clip_norm=1.0,
        mixed_precision=(device.type == "cuda"),
    )
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        verbose=True,
    )
    
    train_time = time.time() - start_time
    
    # Calculate throughput
    total_samples = len(train_loader.dataset) * num_epochs
    throughput = total_samples / train_time
    
    # Generate sample text
    print("\n" + "=" * 50)
    print("Generating sample text...")
    generator = TokenTextGenerator(model, tokenizer, device)
    
    prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Romeo, Romeo",
    ]
    
    for prompt in prompts:
        generated = generator.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.8,
            top_k=40,
        )
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
    
    # Prepare results
    results = {
        "tokenizer_type": tokenizer_type,
        "vocab_size": actual_vocab_size,
        "model_size": model_size,
        "model_params": num_params,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "train_accuracy": history["train_accuracy"][-1] if history.get("train_accuracy") else None,
        "val_accuracy": history["val_accuracy"][-1] if history.get("val_accuracy") else None,
        "train_time": train_time,
        "throughput_samples_per_sec": throughput,
        "device": str(device),
    }
    
    # Save results
    results_path = Path(f"results/token_{tokenizer_type}_{model_size}_{int(time.time())}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train token-level RNN models")
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bpe",
        choices=["bpe", "word"],
        help="Tokenizer type",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Model size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="Sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-optimized",
        action="store_true",
        help="Use standard model instead of optimized",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None
    
    # Train model
    results = train_token_rnn(
        tokenizer_type=args.tokenizer,
        vocab_size=args.vocab_size,
        model_size=args.model_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_optimized=not args.no_optimized,
        device=device,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print_results_table(results)


if __name__ == "__main__":
    main()