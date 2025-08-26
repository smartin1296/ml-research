#!/usr/bin/env python3
"""Test script for token-level RNN implementation."""

import time
from pathlib import Path

import torch

from .train_token import train_token_rnn
from .device_utils import get_best_device
from .results_utils import print_results_table


def benchmark_token_models():
    """Benchmark different token-level configurations."""
    
    device = get_best_device()
    print(f"Running benchmarks on {device}")
    print("=" * 80)
    
    # Test configurations
    configs = [
        # BPE tokenizer with different vocab sizes
        {
            "name": "BPE Small Vocab (500)",
            "tokenizer_type": "bpe",
            "vocab_size": 500,
            "model_size": "small",
            "seq_len": 50,
            "batch_size": 128,
            "num_epochs": 3,
        },
        {
            "name": "BPE Medium Vocab (1000)",
            "tokenizer_type": "bpe",
            "vocab_size": 1000,
            "model_size": "medium",
            "seq_len": 50,
            "batch_size": 64,
            "num_epochs": 3,
        },
        {
            "name": "BPE Large Vocab (2000)",
            "tokenizer_type": "bpe",
            "vocab_size": 2000,
            "model_size": "medium",
            "seq_len": 50,
            "batch_size": 64,
            "num_epochs": 3,
        },
        # Word tokenizer for comparison
        {
            "name": "Word Tokenizer (5000)",
            "tokenizer_type": "word",
            "vocab_size": 5000,
            "model_size": "medium",
            "seq_len": 30,  # Shorter sequences for word-level
            "batch_size": 64,
            "num_epochs": 3,
        },
    ]
    
    results_list = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")
        
        try:
            # Train model with configuration
            results = train_token_rnn(
                tokenizer_type=config["tokenizer_type"],
                vocab_size=config["vocab_size"],
                model_size=config["model_size"],
                seq_len=config["seq_len"],
                batch_size=config["batch_size"],
                num_epochs=config["num_epochs"],
                learning_rate=0.001,
                use_optimized=True,
                device=device,
            )
            
            results["config_name"] = config["name"]
            results_list.append(results)
            
            print(f"\nResults for {config['name']}:")
            print_results_table(results)
            
        except Exception as e:
            print(f"Error testing {config['name']}: {e}")
            continue
    
    # Print comparison table
    if results_list:
        print("\n" + "="*80)
        print("COMPARISON OF ALL CONFIGURATIONS")
        print("="*80)
        
        # Create comparison table
        headers = ["Configuration", "Tokenizer", "Vocab Size", "Val Loss", "Val Acc %", "Throughput"]
        rows = []
        
        for r in results_list:
            rows.append([
                r.get("config_name", "Unknown"),
                r.get("tokenizer_type", ""),
                f"{r.get('vocab_size', 0):,}",
                f"{r.get('val_loss', 0):.3f}" if r.get('val_loss') else "N/A",
                f"{r.get('val_accuracy', 0)*100:.1f}" if r.get('val_accuracy') else "N/A",
                f"{r.get('throughput_samples_per_sec', 0):,.0f}",
            ])
        
        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                     for i in range(len(headers))]
        
        # Print headers
        header_line = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        
        # Print rows
        for row in rows:
            row_line = "|".join(str(val).center(w) for val, w in zip(row, col_widths))
            print(row_line)
    
    return results_list


def quick_test():
    """Quick test with small configuration."""
    
    print("Running quick test of token-level RNN...")
    print("=" * 80)
    
    device = get_best_device()
    
    # Quick test with small model
    results = train_token_rnn(
        tokenizer_type="bpe",
        vocab_size=500,
        model_size="small",
        seq_len=25,
        batch_size=128,
        num_epochs=2,
        learning_rate=0.001,
        use_optimized=True,
        device=device,
    )
    
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    print_results_table(results)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test token-level RNN implementation")
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "benchmark"],
        help="Test mode (quick test or full benchmark)",
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        quick_test()
    else:
        benchmark_token_models()


if __name__ == "__main__":
    main()