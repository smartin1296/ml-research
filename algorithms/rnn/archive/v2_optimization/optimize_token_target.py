#!/usr/bin/env python3
"""Optimize token-level RNN for 500 vocab and ~1.4M parameters."""

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


def calculate_params(vocab_size, embed_dim, hidden_size, num_layers):
    """Calculate parameter count for LSTM model."""
    # Embedding layer
    embedding_params = vocab_size * embed_dim
    
    # LSTM parameters per layer
    # input_size -> hidden_size: 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    # First layer
    first_layer = 4 * (embed_dim * hidden_size + hidden_size * hidden_size + hidden_size)
    # Additional layers
    other_layers = (num_layers - 1) * 4 * (hidden_size * hidden_size + hidden_size * hidden_size + hidden_size)
    lstm_params = first_layer + other_layers
    
    # Output layer
    output_params = hidden_size * vocab_size + vocab_size
    
    total = embedding_params + lstm_params + output_params
    return total, {
        'embedding': embedding_params,
        'lstm': lstm_params,
        'output': output_params
    }


def find_model_config(target_params=1_400_000, vocab_size=500):
    """Find model configuration close to target parameter count."""
    
    configs = []
    
    # Test different configurations
    for hidden_size in [256, 384, 512]:
        for num_layers in [1, 2, 3]:
            for embed_dim in [128, 192, 256]:
                total, breakdown = calculate_params(vocab_size, embed_dim, hidden_size, num_layers)
                diff = abs(total - target_params)
                configs.append({
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'total_params': total,
                    'breakdown': breakdown,
                    'diff_from_target': diff
                })
    
    # Sort by closest to target
    configs.sort(key=lambda x: x['diff_from_target'])
    
    return configs[:5]  # Return top 5 closest


def create_model(vocab_size, hidden_size, embed_dim, num_layers):
    """Create LSTM model with specified configuration."""
    
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
            # Take last output for next-token prediction
            last_output = outputs[:, -1, :]
        
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
    log("TOKEN-LEVEL RNN OPTIMIZATION", "HEADER")
    log("Target: 500 vocab size, ~1.4M parameters", "HEADER")
    log("=" * 80, "HEADER")
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        log(f"✅ Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        log(f"✅ Using CUDA")
    else:
        device = torch.device('cpu')
        log(f"⚠️  Using CPU")
    
    # Find configurations close to 1.4M parameters
    log("\nFinding model configurations near 1.4M parameters...")
    configs = find_model_config(target_params=1_400_000, vocab_size=500)
    
    log("\nTop configurations closest to 1.4M parameters:")
    for i, cfg in enumerate(configs, 1):
        log(f"  {i}. Hidden={cfg['hidden_size']}, Layers={cfg['num_layers']}, "
            f"Embed={cfg['embed_dim']} → {cfg['total_params']:,} params "
            f"(diff: {cfg['diff_from_target']:,})")
    
    # Select the closest configuration
    best_model_config = configs[0]
    log(f"\n✅ Selected configuration: {best_model_config['total_params']:,} parameters")
    log(f"   - Embedding: {best_model_config['breakdown']['embedding']:,} params")
    log(f"   - LSTM: {best_model_config['breakdown']['lstm']:,} params")
    log(f"   - Output: {best_model_config['breakdown']['output']:,} params")
    
    # Test different batch sizes and sequence lengths with this model
    log("\n" + "=" * 80)
    log("TESTING THROUGHPUT WITH DIFFERENT BATCH SIZES AND SEQUENCE LENGTHS")
    log("=" * 80)
    
    test_configs = [
        # Conservative
        {'batch_size': 32, 'seq_len': 25},
        {'batch_size': 64, 'seq_len': 25},
        {'batch_size': 128, 'seq_len': 25},
        {'batch_size': 256, 'seq_len': 25},
        {'batch_size': 512, 'seq_len': 25},
        {'batch_size': 1024, 'seq_len': 25},
        {'batch_size': 2048, 'seq_len': 25},
        
        # Different sequence lengths
        {'batch_size': 1024, 'seq_len': 10},
        {'batch_size': 1024, 'seq_len': 15},
        {'batch_size': 1024, 'seq_len': 20},
        {'batch_size': 1024, 'seq_len': 30},
        {'batch_size': 1024, 'seq_len': 50},
        
        # Aggressive batch sizes
        {'batch_size': 4096, 'seq_len': 10},
        {'batch_size': 4096, 'seq_len': 25},
        {'batch_size': 8192, 'seq_len': 10},
        
        # Extreme configurations
        {'batch_size': 2048, 'seq_len': 50},
        {'batch_size': 512, 'seq_len': 100},
    ]
    
    results = []
    best_throughput = 0
    best_config = None
    
    for idx, test_cfg in enumerate(test_configs, 1):
        log(f"\n{'='*60}")
        log(f"Configuration {idx}/{len(test_configs)}")
        log(f"{'='*60}")
        log(f"Config: BS={test_cfg['batch_size']}, SL={test_cfg['seq_len']}")
        
        try:
            # Create model
            log("Creating model...")
            model = create_model(
                vocab_size=500,
                hidden_size=best_model_config['hidden_size'],
                embed_dim=best_model_config['embed_dim'],
                num_layers=best_model_config['num_layers']
            )
            
            log(f"Moving model to {device}...")
            model = model.to(device)
            
            # Verify parameter count
            actual_params = sum(p.numel() for p in model.parameters())
            log(f"Model ready: {actual_params:,} parameters")
            
            # Benchmark
            log("Starting performance benchmark...")
            throughput, ms_per_batch = benchmark_config(
                model, device,
                test_cfg['batch_size'],
                test_cfg['seq_len'],
                500,  # vocab_size
                warmup=20,
                iterations=200
            )
            
            result = {
                **test_cfg,
                **best_model_config,
                'throughput': throughput,
                'ms_per_batch': ms_per_batch,
                'actual_params': actual_params
            }
            results.append(result)
            
            # Update best
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = result
                log(f"🔥 NEW BEST! {throughput:,.1f} samples/sec", "SUCCESS")
            else:
                log(f"Result: {throughput:,.1f} samples/sec (current best: {best_throughput:,.1f})")
            
            # Check against targets
            if throughput > 100000:
                log(f"✅ Exceeded 100K samples/sec!", "SUCCESS")
            elif throughput > 50000:
                log(f"📈 Good performance: {throughput:.1f} samples/sec", "PROGRESS")
            
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
                log(f"❌ Runtime error: {e}", "ERROR")
        except Exception as e:
            log(f"❌ Unexpected error: {e}", "ERROR")
    
    # Print final results
    log("\n" + "=" * 80)
    log("OPTIMIZATION RESULTS", "HEADER")
    log("=" * 80)
    
    if best_config:
        log("\n🏆 BEST CONFIGURATION:")
        log(f"  Model: {best_config['actual_params']:,} parameters")
        log(f"  - Hidden Size: {best_config['hidden_size']}")
        log(f"  - Layers: {best_config['num_layers']}")
        log(f"  - Embedding Dim: {best_config['embed_dim']}")
        log(f"  - Vocab Size: 500")
        log(f"  Optimal Settings:")
        log(f"  - Batch Size: {best_config['batch_size']}")
        log(f"  - Sequence Length: {best_config['seq_len']}")
        log(f"  Performance:")
        log(f"  - Throughput: {best_config['throughput']:,.1f} samples/sec")
        log(f"  - Batch Time: {best_config['ms_per_batch']:.2f}ms")
        
        # Top 5 configurations
        log("\n📊 TOP 5 CONFIGURATIONS:")
        sorted_results = sorted(results, key=lambda x: x['throughput'], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            log(f"  {i}. BS={r['batch_size']:<4} SL={r['seq_len']:<3} → "
                f"{r['throughput']:>10.1f} samples/s ({r['ms_per_batch']:>6.2f}ms)")
        
        # Save results
        output = {
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'target_params': 1_400_000,
            'vocab_size': 500,
            'model_config': {
                'hidden_size': best_config['hidden_size'],
                'num_layers': best_config['num_layers'],
                'embed_dim': best_config['embed_dim'],
                'actual_params': best_config['actual_params']
            },
            'optimal_settings': {
                'batch_size': best_config['batch_size'],
                'seq_len': best_config['seq_len'],
                'throughput': best_config['throughput'],
                'ms_per_batch': best_config['ms_per_batch']
            },
            'all_results': sorted_results
        }
        
        output_file = Path("results/token_optimal_1_4M_params.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        log(f"\n💾 Results saved to: {output_file}")
        
        # Save as the official optimal config
        optimal_config = {
            'vocab_size': 500,
            'model': {
                'hidden_size': best_config['hidden_size'],
                'num_layers': best_config['num_layers'], 
                'embed_dim': best_config['embed_dim'],
                'params': best_config['actual_params']
            },
            'training': {
                'batch_size': best_config['batch_size'],
                'seq_len': best_config['seq_len']
            },
            'performance': {
                'throughput_samples_per_sec': best_config['throughput'],
                'ms_per_batch': best_config['ms_per_batch']
            }
        }
        
        config_file = Path("algorithms/rnn/token_optimal_final.json")
        with open(config_file, 'w') as f:
            json.dump(optimal_config, f, indent=2)
        
        log(f"💾 Optimal config saved to: {config_file}")


if __name__ == "__main__":
    main()