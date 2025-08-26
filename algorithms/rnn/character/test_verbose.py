#!/usr/bin/env python3
"""Quick test of verbose logging for token-level RNN."""

import time
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path


def log(msg, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {level}: {msg}")


def create_model(vocab_size=500, hidden_size=256, embed_dim=256, num_layers=2):
    """Create LSTM model."""
    class TokenLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, 
                               batch_first=True, dropout=0.0)
            self.output = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x, hidden=None):
            x = self.embedding(x)
            x, hidden = self.lstm(x, hidden)
            x = self.output(x)
            return x, hidden
    
    return TokenLSTM()


def benchmark_config(model, device, batch_size, seq_len, vocab_size=500, warmup=5, iterations=20):
    """Quick benchmark with verbose logging."""
    
    log(f"Starting benchmark: BS={batch_size}, SL={seq_len}")
    
    model.eval()
    
    # Create synthetic data
    log(f"Creating synthetic data on {device}...")
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    log(f"Data created: shape={inputs.shape}")
    
    # Warmup
    log(f"Running warmup ({warmup} iterations)...")
    for i in range(warmup):
        with torch.no_grad():
            outputs, _ = model(inputs)
        if i % 2 == 0:
            log(f"  Warmup {i+1}/{warmup}")
    
    # Synchronize
    log("Synchronizing...")
    if device.type == 'mps':
        torch.mps.synchronize()
    
    # Benchmark
    log(f"Benchmarking ({iterations} iterations)...")
    start = time.perf_counter()
    
    for i in range(iterations):
        with torch.no_grad():
            outputs, _ = model(inputs)
        
        if (i + 1) % 5 == 0:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed
            log(f"  Progress: {i+1}/{iterations} | {rate:.1f} iter/s")
    
    # Final sync
    log("Final sync...")
    if device.type == 'mps':
        torch.mps.synchronize()
    
    end = time.perf_counter()
    total_time = end - start
    throughput = (batch_size * iterations) / total_time
    
    log(f"Complete! Throughput: {throughput:,.1f} samples/sec")
    return throughput


def main():
    log("=" * 60, "HEADER")
    log("VERBOSE LOGGING TEST", "HEADER")
    log("=" * 60, "HEADER")
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    log(f"Using device: {device}")
    
    # Test 3 configurations
    configs = [
        {'batch_size': 512, 'seq_len': 25},
        {'batch_size': 1024, 'seq_len': 15},
        {'batch_size': 2048, 'seq_len': 10},
    ]
    
    best = 0
    
    for idx, cfg in enumerate(configs, 1):
        log(f"\n{'='*40}")
        log(f"Test {idx}/{len(configs)}")
        log(f"{'='*40}")
        
        log("Creating model...")
        model = create_model().to(device)
        params = sum(p.numel() for p in model.parameters())
        log(f"Model ready: {params:,} params")
        
        log("Starting benchmark...")
        throughput = benchmark_config(
            model, device, 
            cfg['batch_size'], 
            cfg['seq_len']
        )
        
        if throughput > best:
            best = throughput
            log(f"🔥 NEW BEST!", "SUCCESS")
        
        del model
    
    log(f"\nBest: {best:,.1f} samples/sec", "RESULT")


if __name__ == "__main__":
    main()