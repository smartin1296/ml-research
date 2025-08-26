#!/usr/bin/env python3
"""Find optimal settings for token-level RNN on specific hardware."""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .device_utils import get_best_device, get_device_info
from .tokenizers import BPETokenizer
from .token_dataset import load_shakespeare_data, TokenSequenceDataset
from .token_models import TokenRNNModel, OptimizedTokenRNNModel


def benchmark_configuration(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 100,
) -> Tuple[float, float]:
    """Benchmark a specific configuration.
    
    Returns:
        Tuple of (throughput_samples_per_sec, avg_batch_time)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= 10:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            logits, _ = model(inputs)
            # For next token prediction, we only need the last position
            # logits shape: [batch_size, seq_len, vocab_size]
            # targets shape: [batch_size] (single next token per sequence)
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            loss = criterion(last_logits, targets)
    
    # Actual benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()
    
    total_samples = 0
    batch_count = 0
    
    for i, (inputs, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            logits, _ = model(inputs)
            # For next token prediction, we only need the last position
            # logits shape: [batch_size, seq_len, vocab_size]
            # targets shape: [batch_size] (single next token per sequence)
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            loss = criterion(last_logits, targets)
        
        total_samples += inputs.size(0)
        batch_count += 1
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = total_samples / total_time
    avg_batch_time = total_time / batch_count
    
    return throughput, avg_batch_time


def find_optimal_settings():
    """Find optimal settings for token-level RNN on current hardware."""
    
    device = get_best_device()
    device_info = get_device_info()
    
    print("=" * 80)
    print("FINDING OPTIMAL SETTINGS FOR TOKEN-LEVEL RNN")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Device Info: {device_info}")
    print()
    
    # Load and prepare data
    print("Loading data...")
    text = load_shakespeare_data()
    text = text[:200000]  # Use subset for faster testing
    
    # Initialize tokenizer with medium vocabulary
    tokenizer = BPETokenizer(vocab_size=1000, min_freq=2)
    tokenizer.fit(text)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Test configurations
    configs = []
    
    # Batch size options
    batch_sizes = [16, 32, 64, 128, 256]
    if device.type == "mps":
        batch_sizes.extend([512, 1024, 2048])
    
    # Sequence length options
    seq_lens = [25, 50, 100, 150]
    
    # Model size options
    model_configs = [
        {"name": "small", "hidden_size": 256, "num_layers": 2, "embed_dim": 128},
        {"name": "medium", "hidden_size": 512, "num_layers": 2, "embed_dim": 256},
        {"name": "large", "hidden_size": 1024, "num_layers": 2, "embed_dim": 512},
    ]
    
    results = []
    
    print("\nTesting configurations...")
    print("-" * 80)
    
    for model_cfg in model_configs:
        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                
                # Skip very large configs that might OOM
                if batch_size * seq_len * model_cfg["hidden_size"] > 50000000:
                    continue
                
                try:
                    # Create dataset
                    dataset = TokenSequenceDataset(
                        text=text,
                        tokenizer=tokenizer,
                        seq_len=seq_len,
                        stride=seq_len,  # Non-overlapping for benchmark
                        train=True,
                    )
                    
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                    )
                    
                    # Skip if not enough batches
                    if len(dataloader) < 50:
                        continue
                    
                    # Create model
                    model = TokenRNNModel(
                        vocab_size=vocab_size,
                        embed_dim=model_cfg["embed_dim"],
                        hidden_size=model_cfg["hidden_size"],
                        num_layers=model_cfg["num_layers"],
                        rnn_type="lstm",
                        dropout=0.0,  # No dropout for benchmarking
                        layer_norm=True,
                    ).to(device)
                    
                    # Benchmark
                    throughput, batch_time = benchmark_configuration(
                        model, dataloader, device, num_batches=50
                    )
                    
                    config = {
                        "model_size": model_cfg["name"],
                        "hidden_size": model_cfg["hidden_size"],
                        "num_layers": model_cfg["num_layers"],
                        "embed_dim": model_cfg["embed_dim"],
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "vocab_size": vocab_size,
                        "throughput_samples_per_sec": throughput,
                        "ms_per_batch": batch_time * 1000,
                        "params": sum(p.numel() for p in model.parameters()),
                    }
                    
                    results.append(config)
                    
                    print(f"{model_cfg['name']:<8} | "
                          f"BS={batch_size:<4} | "
                          f"SL={seq_len:<3} | "
                          f"Throughput={throughput:>8.1f} samples/s | "
                          f"Batch={batch_time*1000:>6.1f}ms")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{model_cfg['name']:<8} | "
                              f"BS={batch_size:<4} | "
                              f"SL={seq_len:<3} | "
                              f"OOM - Skipping")
                    else:
                        raise
                except Exception as e:
                    print(f"Error with config: {e}")
                    continue
    
    # Find optimal configurations
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATIONS")
    print("=" * 80)
    
    if results:
        # Sort by throughput
        results.sort(key=lambda x: x["throughput_samples_per_sec"], reverse=True)
        
        # Best overall
        best_overall = results[0]
        print(f"\n1. BEST OVERALL THROUGHPUT:")
        print(f"   Model: {best_overall['model_size']}")
        print(f"   Batch Size: {best_overall['batch_size']}")
        print(f"   Sequence Length: {best_overall['seq_len']}")
        print(f"   Throughput: {best_overall['throughput_samples_per_sec']:,.1f} samples/sec")
        print(f"   Batch Time: {best_overall['ms_per_batch']:.1f}ms")
        print(f"   Parameters: {best_overall['params']:,}")
        
        # Best for each model size
        for size in ["small", "medium", "large"]:
            size_results = [r for r in results if r["model_size"] == size]
            if size_results:
                best = size_results[0]
                print(f"\n2. BEST {size.upper()} MODEL:")
                print(f"   Batch Size: {best['batch_size']}")
                print(f"   Sequence Length: {best['seq_len']}")
                print(f"   Throughput: {best['throughput_samples_per_sec']:,.1f} samples/sec")
                print(f"   Batch Time: {best['ms_per_batch']:.1f}ms")
                print(f"   Parameters: {best['params']:,}")
        
        # Save results
        output_file = Path("results/token_optimal_settings.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "device": str(device),
            "device_info": device_info,
            "timestamp": time.time(),
            "results": results,
            "best_overall": best_overall,
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n\nResults saved to: {output_file}")
        
        # Create optimized config file
        optimized_config = {
            "tokenizer": {
                "type": "bpe",
                "vocab_size": 1000,
            },
            "small": {
                "batch_size": next((r["batch_size"] for r in results if r["model_size"] == "small"), 64),
                "seq_len": next((r["seq_len"] for r in results if r["model_size"] == "small"), 50),
                "hidden_size": 256,
                "num_layers": 2,
                "embed_dim": 128,
            },
            "medium": {
                "batch_size": next((r["batch_size"] for r in results if r["model_size"] == "medium"), 32),
                "seq_len": next((r["seq_len"] for r in results if r["model_size"] == "medium"), 50),
                "hidden_size": 512,
                "num_layers": 2,
                "embed_dim": 256,
            },
            "large": {
                "batch_size": next((r["batch_size"] for r in results if r["model_size"] == "large"), 16),
                "seq_len": next((r["seq_len"] for r in results if r["model_size"] == "large"), 50),
                "hidden_size": 1024,
                "num_layers": 2,
                "embed_dim": 512,
            },
        }
        
        config_file = Path("algorithms/rnn/token_optimal_config.json")
        with open(config_file, "w") as f:
            json.dump(optimized_config, f, indent=2)
        
        print(f"Optimized config saved to: {config_file}")
        
        return optimized_config
    
    else:
        print("No valid configurations found!")
        return None


if __name__ == "__main__":
    find_optimal_settings()