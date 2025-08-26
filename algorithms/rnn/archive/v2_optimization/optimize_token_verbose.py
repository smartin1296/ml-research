#!/usr/bin/env python3
"""Maximum throughput optimization for token-level RNN with verbose logging."""

import time
import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime


def log(msg, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {level}: {msg}")


def create_fast_lstm(vocab_size=1000, hidden_size=128, embed_dim=64):
    """Create a minimal LSTM for maximum speed."""
    
    class FastLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)
            self.output = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x, hidden=None):
            x = self.embedding(x)
            x, hidden = self.lstm(x, hidden)
            x = self.output(x)
            return x, hidden
    
    return FastLSTM()


def benchmark_config(model, device, batch_size, seq_len, vocab_size, warmup=20, iterations=200):
    """Benchmark a configuration with synthetic data."""
    
    log(f"Starting benchmark: BS={batch_size}, SL={seq_len}, warmup={warmup}, iterations={iterations}")
    
    model.eval()
    
    # Create synthetic data on device
    log(f"Creating synthetic data on {device}...")
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size,), device=device)
    log(f"Data created: inputs shape={inputs.shape}, targets shape={targets.shape}")
    
    # Warmup
    log(f"Running warmup ({warmup} iterations)...")
    warmup_start = time.perf_counter()
    for i in range(warmup):
        with torch.no_grad():
            outputs, _ = model(inputs)
        if i % 5 == 0:
            log(f"  Warmup iteration {i+1}/{warmup}")
    warmup_time = time.perf_counter() - warmup_start
    log(f"Warmup complete in {warmup_time:.3f}s")
    
    # Synchronize
    log("Synchronizing device...")
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    log(f"Starting benchmark ({iterations} iterations)...")
    start = time.perf_counter()
    
    for i in range(iterations):
        with torch.no_grad():
            outputs, _ = model(inputs)
        
        # Log progress every 50 iterations
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed
            log(f"  Progress: {i+1}/{iterations} iterations | {rate:.1f} iter/s | {elapsed:.2f}s elapsed")
    
    # Synchronize
    log("Final synchronization...")
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    total_time = end - start
    samples_processed = batch_size * iterations
    throughput = samples_processed / total_time
    ms_per_batch = (total_time / iterations) * 1000
    
    log(f"Benchmark complete:")
    log(f"  Total time: {total_time:.3f}s")
    log(f"  Samples processed: {samples_processed:,}")
    log(f"  Throughput: {throughput:,.1f} samples/sec")
    log(f"  Time per batch: {ms_per_batch:.2f}ms")
    
    return throughput, ms_per_batch


def main():
    log("=" * 80, "HEADER")
    log("MAXIMUM THROUGHPUT OPTIMIZATION FOR TOKEN-LEVEL RNN", "HEADER")
    log("=" * 80, "HEADER")
    
    # Device setup
    log("Detecting device...")
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        log(f"✅ Using MPS (Apple Silicon)")
        log(f"  MPS built: {torch.backends.mps.is_built()}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        log(f"✅ Using CUDA")
        log(f"  CUDA version: {torch.version.cuda}")
        log(f"  Device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        log(f"⚠️  Using CPU (no GPU available)")
    
    log(f"PyTorch version: {torch.__version__}")
    
    # Test configurations
    configs_to_test = [
        # Start conservative
        {'batch_size': 64, 'seq_len': 10, 'hidden': 128, 'embed': 64},
        {'batch_size': 128, 'seq_len': 10, 'hidden': 128, 'embed': 64},
        {'batch_size': 256, 'seq_len': 10, 'hidden': 128, 'embed': 64},
        {'batch_size': 512, 'seq_len': 10, 'hidden': 128, 'embed': 64},
        {'batch_size': 1024, 'seq_len': 10, 'hidden': 128, 'embed': 64},
        {'batch_size': 2048, 'seq_len': 10, 'hidden': 128, 'embed': 64},
        
        # Try different sequence lengths with best batch size so far
        {'batch_size': 1024, 'seq_len': 15, 'hidden': 128, 'embed': 64},
        {'batch_size': 1024, 'seq_len': 20, 'hidden': 128, 'embed': 64},
        {'batch_size': 1024, 'seq_len': 25, 'hidden': 128, 'embed': 64},
        
        # Try smaller models for higher throughput
        {'batch_size': 2048, 'seq_len': 10, 'hidden': 64, 'embed': 32},
        {'batch_size': 4096, 'seq_len': 10, 'hidden': 64, 'embed': 32},
        
        # Extreme batch sizes with tiny models
        {'batch_size': 8192, 'seq_len': 5, 'hidden': 32, 'embed': 16},
        {'batch_size': 16384, 'seq_len': 5, 'hidden': 32, 'embed': 16},
    ]
    
    log(f"Will test {len(configs_to_test)} configurations")
    
    results = []
    best_throughput = 0
    best_config = None
    
    log("\n" + "=" * 80)
    log("STARTING CONFIGURATION TESTS")
    log("=" * 80)
    
    for idx, config in enumerate(configs_to_test, 1):
        log(f"\n{'='*60}")
        log(f"Configuration {idx}/{len(configs_to_test)}")
        log(f"{'='*60}")
        log(f"Config: BS={config['batch_size']}, SL={config['seq_len']}, "
            f"Hidden={config['hidden']}, Embed={config['embed']}")
        
        try:
            # Create model
            log("Creating model...")
            model = create_fast_lstm(
                vocab_size=1000,
                hidden_size=config['hidden'],
                embed_dim=config['embed']
            )
            
            log(f"Moving model to {device}...")
            model = model.to(device)
            
            num_params = sum(p.numel() for p in model.parameters())
            log(f"Model created with {num_params:,} parameters")
            
            # Benchmark
            log("Starting benchmark...")
            throughput, ms_per_batch = benchmark_config(
                model, device,
                config['batch_size'],
                config['seq_len'],
                1000,
                warmup=20,
                iterations=200
            )
            
            # Record results
            result = {
                **config,
                'throughput': throughput,
                'ms_per_batch': ms_per_batch,
                'params': num_params
            }
            results.append(result)
            
            # Update best
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = result
                log(f"🔥 NEW BEST! {throughput:,.1f} samples/sec", "SUCCESS")
            else:
                log(f"Result: {throughput:,.1f} samples/sec (current best: {best_throughput:,.1f})")
            
            # Check if we hit target
            if throughput > 5000:
                log(f"✅ TARGET ACHIEVED: {throughput:.1f} samples/sec!", "SUCCESS")
            elif throughput > 4000:
                log(f"📈 Getting close! {throughput:.1f} samples/sec", "PROGRESS")
            
            # Clean up
            log("Cleaning up model...")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                log("CUDA cache cleared")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(f"❌ Out of memory with this configuration", "ERROR")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    log("CUDA cache cleared after OOM")
            else:
                log(f"❌ Error: {e}", "ERROR")
        except Exception as e:
            log(f"❌ Unexpected error: {e}", "ERROR")
    
    # Print results
    log("\n" + "=" * 80)
    log("OPTIMIZATION COMPLETE - RESULTS SUMMARY")
    log("=" * 80)
    
    if best_config:
        log("\n🏆 BEST CONFIGURATION:", "RESULT")
        log(f"   Batch Size: {best_config['batch_size']}", "RESULT")
        log(f"   Sequence Length: {best_config['seq_len']}", "RESULT")
        log(f"   Hidden Size: {best_config['hidden']}", "RESULT")
        log(f"   Embedding Dim: {best_config['embed']}", "RESULT")
        log(f"   Throughput: {best_config['throughput']:,.1f} samples/sec", "RESULT")
        log(f"   Batch Time: {best_config['ms_per_batch']:.2f}ms", "RESULT")
        log(f"   Parameters: {best_config['params']:,}", "RESULT")
        
        if best_throughput > 5000:
            log(f"\n🎯 SUCCESS! Achieved target of 5000+ samples/sec", "SUCCESS")
        else:
            log(f"\n⚠️  Current best: {best_throughput:.1f} samples/sec", "WARNING")
            log(f"   Still {5000 - best_throughput:.1f} samples/sec below target", "WARNING")
        
        # Top 5 results
        log("\n📊 TOP 5 CONFIGURATIONS:", "RESULT")
        sorted_results = sorted(results, key=lambda x: x['throughput'], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            log(f"  {i}. BS={r['batch_size']:<5} SL={r['seq_len']:<2} "
                f"H={r['hidden']:<3} E={r['embed']:<2} → "
                f"{r['throughput']:>10.1f} samples/s", "RESULT")
        
        # Save results
        log("\nSaving results...")
        output = {
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'best_config': best_config,
            'all_results': sorted_results
        }
        
        output_file = Path("results/token_max_throughput_verbose.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        log(f"💾 Results saved to: {output_file}", "SUCCESS")
    else:
        log("❌ No successful configurations found", "ERROR")
    
    log("\n" + "=" * 80)
    log("OPTIMIZATION COMPLETE", "HEADER")
    log("=" * 80)


if __name__ == "__main__":
    main()