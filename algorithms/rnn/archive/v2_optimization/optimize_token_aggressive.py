#!/usr/bin/env python3
"""Aggressive optimization to find maximum throughput for token-level RNN."""

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
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_batches: int = 100,
) -> Tuple[float, float]:
    """Benchmark with synthetic data for maximum speed."""
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Use synthetic data to eliminate data loading overhead
    dummy_inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_targets = torch.randint(0, vocab_size, (batch_size,), device=device)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            logits, _ = model(dummy_inputs)
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, dummy_targets)
    
    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    # Actual benchmark
    start_time = time.perf_counter()
    
    for _ in range(num_batches):
        with torch.no_grad():
            logits, _ = model(dummy_inputs)
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, dummy_targets)
    
    # Synchronize after computation
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_samples = batch_size * num_batches
    throughput = total_samples / total_time
    avg_batch_time = total_time / num_batches
    
    return throughput, avg_batch_time


def find_optimal_aggressive():
    """Find optimal settings with aggressive optimization."""
    
    device = get_best_device()
    device_info = get_device_info()
    
    print("=" * 80)
    print("AGGRESSIVE OPTIMIZATION FOR TOKEN-LEVEL RNN")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Device Info: {device_info}")
    print()
    
    # Fixed vocabulary size for testing
    vocab_size = 1000
    
    # Test configurations - much larger batch sizes
    configs = []
    
    # Aggressive batch sizes
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    # Shorter sequences for higher throughput
    seq_lens = [10, 15, 20, 25, 30, 40, 50]
    
    # Model configurations - focus on smaller, faster models
    model_configs = [
        {"name": "tiny", "hidden_size": 128, "num_layers": 1, "embed_dim": 64},
        {"name": "small", "hidden_size": 256, "num_layers": 1, "embed_dim": 128},
        {"name": "small2", "hidden_size": 256, "num_layers": 2, "embed_dim": 128},
        {"name": "medium", "hidden_size": 384, "num_layers": 2, "embed_dim": 192},
    ]
    
    results = []
    best_throughput = 0
    best_config = None
    
    print("Testing configurations...")
    print("-" * 80)
    
    for model_cfg in model_configs:
        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                
                # Skip configs that would use too much memory
                estimated_memory = batch_size * seq_len * model_cfg["hidden_size"] * 4 / 1e6  # MB
                if estimated_memory > 2000:  # Skip if > 2GB
                    continue
                
                try:
                    # Create model
                    model = TokenRNNModel(
                        vocab_size=vocab_size,
                        embed_dim=model_cfg["embed_dim"],
                        hidden_size=model_cfg["hidden_size"],
                        num_layers=model_cfg["num_layers"],
                        rnn_type="lstm",
                        dropout=0.0,  # No dropout for speed
                        layer_norm=False,  # Disable layer norm for speed
                        tie_weights=True,  # Tie weights to reduce params
                    ).to(device)
                    
                    # Set to eval mode and disable gradients globally
                    model.eval()
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    # Benchmark
                    throughput, batch_time = benchmark_configuration(
                        model, batch_size, seq_len, vocab_size, device, num_batches=100
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
                    
                    # Update best
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_config = config
                    
                    status = "🔥 NEW BEST!" if config == best_config else ""
                    print(f"{model_cfg['name']:<8} | "
                          f"BS={batch_size:<4} | "
                          f"SL={seq_len:<2} | "
                          f"Throughput={throughput:>9.1f} samples/s | "
                          f"Batch={batch_time*1000:>6.1f}ms {status}")
                    
                    # Early exit if we hit target
                    if throughput > 5000:
                        print(f"\n✅ TARGET ACHIEVED: {throughput:.1f} samples/sec")
                    
                    # Clean up
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{model_cfg['name']:<8} | "
                              f"BS={batch_size:<4} | "
                              f"SL={seq_len:<2} | "
                              f"OOM - Skipping")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise
                except Exception as e:
                    print(f"Error with config: {e}")
                    continue
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    if results:
        # Sort by throughput
        results.sort(key=lambda x: x["throughput_samples_per_sec"], reverse=True)
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Model: {best_config['model_size']}")
        print(f"   Hidden Size: {best_config['hidden_size']}")
        print(f"   Layers: {best_config['num_layers']}")
        print(f"   Embed Dim: {best_config['embed_dim']}")
        print(f"   Batch Size: {best_config['batch_size']}")
        print(f"   Sequence Length: {best_config['seq_len']}")
        print(f"   Throughput: {best_config['throughput_samples_per_sec']:,.1f} samples/sec")
        print(f"   Batch Time: {best_config['ms_per_batch']:.2f}ms")
        print(f"   Parameters: {best_config['params']:,}")
        
        # Top 5 configurations
        print(f"\n📊 TOP 5 CONFIGURATIONS:")
        for i, config in enumerate(results[:5], 1):
            print(f"{i}. {config['model_size']:<6} BS={config['batch_size']:<4} "
                  f"SL={config['seq_len']:<2} → {config['throughput_samples_per_sec']:>9.1f} samples/s")
        
        # Save results
        output_file = Path("results/token_optimal_aggressive.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "device": str(device),
            "device_info": device_info,
            "timestamp": time.time(),
            "best_config": best_config,
            "top_10_results": results[:10],
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Update optimal config if we beat the target
        if best_throughput > 5000:
            print(f"\n🎯 SUCCESS! Achieved {best_throughput:.1f} samples/sec (target: 5000)")
            
            # Save optimized config
            optimized_config = {
                "achieved_throughput": best_throughput,
                "tokenizer": {
                    "type": "bpe",
                    "vocab_size": vocab_size,
                },
                "optimal": {
                    "batch_size": best_config["batch_size"],
                    "seq_len": best_config["seq_len"],
                    "hidden_size": best_config["hidden_size"],
                    "num_layers": best_config["num_layers"],
                    "embed_dim": best_config["embed_dim"],
                    "layer_norm": False,  # Disabled for speed
                    "dropout": 0.0,
                },
            }
            
            config_file = Path("algorithms/rnn/token_optimal_aggressive.json")
            with open(config_file, "w") as f:
                json.dump(optimized_config, f, indent=2)
            
            print(f"💾 Optimal config saved to: {config_file}")
        else:
            print(f"\n⚠️  Best achieved: {best_throughput:.1f} samples/sec (target: 5000)")
            print("Consider:")
            print("  - Even larger batch sizes")
            print("  - Shorter sequences")
            print("  - Smaller models")
            print("  - Disabling more features")
    
    return results


if __name__ == "__main__":
    find_optimal_aggressive()