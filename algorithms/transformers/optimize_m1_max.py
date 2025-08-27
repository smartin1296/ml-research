#!/usr/bin/env python3
"""
M1 Max Optimization Testing for Transformers
Find optimal settings like RNN module's successful 5,720 samples/sec achievement
"""

import torch
import torch.nn as nn
import time
import json
import psutil
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.core import TransformerTrainer
from algorithms.transformers.test_basic import SimpleTransformer
from algorithms.transformers.benchmark_dataset import create_benchmark_dataloaders


class M1MaxTransformerOptimizer:
    """
    M1 Max optimization testing following RNN module's successful methodology
    """
    
    def __init__(self):
        self.device = self._get_best_device()
        self.results = []
        
        print(f"🔧 M1 Max Transformer Optimizer")
        print(f"   Device: {self.device}")
        print(f"   Available Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # Test configurations inspired by RNN success (RNN found 2048+ optimal)
        # M1 Max has 64GB unified memory - can go much larger than typical GPU setups
        self.test_configs = [
            # Start moderate, scale up aggressively like RNN testing
            {'batch_size': 128, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 256, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 512, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 1024, 'seq_len': 64, 'd_model': 256},
            
            # RNN found 2048 optimal - test transformer equivalent and beyond
            {'batch_size': 2048, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 4096, 'seq_len': 64, 'd_model': 256},
            {'batch_size': 8192, 'seq_len': 64, 'd_model': 256},
            
            # Test with longer sequences at high batch sizes
            {'batch_size': 2048, 'seq_len': 128, 'd_model': 256},
            {'batch_size': 4096, 'seq_len': 128, 'd_model': 256},
            {'batch_size': 1024, 'seq_len': 256, 'd_model': 256},
            {'batch_size': 2048, 'seq_len': 256, 'd_model': 256},
            
            # Test larger models at high batch sizes (leverage 64GB memory)
            {'batch_size': 2048, 'seq_len': 128, 'd_model': 512},
            {'batch_size': 4096, 'seq_len': 128, 'd_model': 384},
            {'batch_size': 1024, 'seq_len': 128, 'd_model': 768},
            
            # Extreme high batch sizes (unique to 64GB systems)
            {'batch_size': 6144, 'seq_len': 128, 'd_model': 256},
            {'batch_size': 12288, 'seq_len': 64, 'd_model': 256},
        ]
    
    def _get_best_device(self) -> torch.device:
        """Get optimal device following project patterns"""
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def test_configuration(
        self, 
        batch_size: int, 
        seq_len: int, 
        d_model: int,
        test_duration: float = 30.0
    ) -> Dict[str, float]:
        """
        Test a specific configuration for performance
        Similar to RNN module's optimization methodology
        """
        print(f"\n🧪 Testing: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
        
        try:
            # Create simple test data without external downloads
            # For optimization testing, we just need the right shapes
            class SimpleTestTokenizer:
                def __init__(self):
                    self.vocab_size = 1000  # Reasonable vocab size for testing
                    self.pad_token = 0
            
            class SimpleTestDataset:
                def __init__(self, batch_size, seq_len):
                    # Generate random sequences for testing
                    self.data = [(
                        torch.randint(1, 1000, (seq_len,), dtype=torch.long),
                        torch.randint(1, 1000, (seq_len,), dtype=torch.long)
                    ) for _ in range(batch_size * 2)]  # Make enough batches for testing
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            from torch.utils.data import DataLoader
            dataset = SimpleTestDataset(batch_size, seq_len)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            tokenizer = SimpleTestTokenizer()
            
            # Create test model
            model = SimpleTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=d_model,
                num_heads=min(8, d_model // 32),  # Ensure divisibility
                num_layers=2,  # Smaller for testing
                d_ff=d_model * 4,
                max_seq_len=seq_len
            )
            model.to(self.device)
            
            # Test parameters
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Create trainer for consistency
            trainer = TransformerTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=train_loader,  # Use same for testing
                tokenizer=tokenizer
            )
            
            # Memory before training
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            
            memory_before = psutil.virtual_memory().used / (1024**3)
            
            # Training performance test
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            start_time = time.time()
            total_tokens = 0
            total_batches = 0
            
            # Test for specified duration
            for batch_idx, batch in enumerate(train_loader):
                if time.time() - start_time > test_duration:
                    break
                
                input_tokens, target_tokens = batch
                input_tokens = input_tokens.to(self.device)
                target_tokens = target_tokens.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(input_tokens)
                loss = nn.functional.cross_entropy(
                    output.contiguous().view(-1, output.size(-1)),
                    target_tokens.contiguous().view(-1),
                    ignore_index=tokenizer.pad_token
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_tokens += target_tokens.numel()
                total_batches += 1
            
            elapsed_time = time.time() - start_time
            memory_after = psutil.virtual_memory().used / (1024**3)
            
            # Calculate metrics
            tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
            memory_used = memory_after - memory_before
            
            # Success metrics
            result = {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'd_model': d_model,
                'parameters': param_count,
                'tokens_per_sec': tokens_per_sec,
                'memory_used_gb': memory_used,
                'loss': loss.item(),
                'batches_processed': total_batches,
                'status': 'success'
            }
            
            print(f"   ✅ {tokens_per_sec:.0f} tokens/sec, {memory_used:.1f}GB memory")
            
            # Cleanup
            del model, trainer, train_loader, tokenizer, optimizer
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            
            # Cleanup on error
            if 'model' in locals():
                del model
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            
            return {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'd_model': d_model,
                'tokens_per_sec': 0,
                'memory_used_gb': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_optimization_sweep(self) -> Dict[str, any]:
        """
        Run comprehensive optimization sweep
        Following RNN module's successful methodology
        """
        print(f"\n🚀 Starting M1 Max Transformer Optimization Sweep")
        print(f"   Testing {len(self.test_configs)} configurations")
        print("=" * 60)
        
        start_time = time.time()
        
        for i, config in enumerate(self.test_configs, 1):
            print(f"\n📊 Configuration {i}/{len(self.test_configs)}")
            
            result = self.test_configuration(
                config['batch_size'],
                config['seq_len'], 
                config['d_model']
            )
            
            self.results.append(result)
            
            # Show running best
            successful_results = [r for r in self.results if r['status'] == 'success']
            if successful_results:
                best_result = max(successful_results, key=lambda x: x['tokens_per_sec'])
                print(f"   🏆 Current best: {best_result['tokens_per_sec']:.0f} tokens/sec "
                      f"(batch={best_result['batch_size']}, seq={best_result['seq_len']}, "
                      f"d_model={best_result['d_model']})")
        
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Save results
        self._save_results(analysis, total_time)
        
        print(f"\n✅ Optimization sweep complete in {total_time:.1f}s")
        return analysis
    
    def _analyze_results(self) -> Dict[str, any]:
        """Analyze optimization results"""
        successful = [r for r in self.results if r['status'] == 'success']
        
        if not successful:
            return {'error': 'No successful configurations'}
        
        # Find optimal configuration
        best_throughput = max(successful, key=lambda x: x['tokens_per_sec'])
        best_memory = min(successful, key=lambda x: x['memory_used_gb'])
        
        # Find sweet spots by batch size
        batch_analysis = {}
        for result in successful:
            batch_size = result['batch_size']
            if batch_size not in batch_analysis:
                batch_analysis[batch_size] = []
            batch_analysis[batch_size].append(result['tokens_per_sec'])
        
        # Average performance by batch size
        batch_averages = {
            batch: sum(speeds) / len(speeds) 
            for batch, speeds in batch_analysis.items()
        }
        
        analysis = {
            'best_throughput_config': best_throughput,
            'best_memory_config': best_memory,
            'batch_size_analysis': batch_averages,
            'total_successful': len(successful),
            'total_failed': len(self.results) - len(successful),
            'all_results': self.results
        }
        
        return analysis
    
    def _save_results(self, analysis: Dict, total_time: float):
        """Save optimization results"""
        results_dir = Path("algorithms/transformers/results/optimization")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"m1_max_optimization_{timestamp}.json"
        
        save_data = {
            'device': str(self.device),
            'total_time': total_time,
            'timestamp': timestamp,
            'analysis': analysis
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"   💾 Results saved to {results_file}")
    
    def get_recommended_config(self) -> Dict[str, int]:
        """Get recommended configuration for Phase 1"""
        successful = [r for r in self.results if r['status'] == 'success']
        
        if not successful:
            # Fallback conservative config
            return {
                'batch_size': 64,
                'seq_len': 128,
                'd_model': 256,
                'reason': 'fallback_conservative'
            }
        
        # Find best balanced configuration (good throughput, reasonable memory)
        best = max(successful, key=lambda x: x['tokens_per_sec'])
        
        return {
            'batch_size': best['batch_size'],
            'seq_len': best['seq_len'],
            'd_model': best['d_model'],
            'expected_tokens_per_sec': best['tokens_per_sec'],
            'memory_usage_gb': best['memory_used_gb'],
            'reason': 'optimization_tested'
        }


def run_quick_optimization():
    """Run quick optimization test before Phase 1"""
    print("🔍 Quick M1 Max Optimization Test")
    print("=" * 40)
    
    optimizer = M1MaxTransformerOptimizer()
    
    # Quick test of key configurations including high batch sizes
    quick_configs = [
        {'batch_size': 256, 'seq_len': 128, 'd_model': 256},
        {'batch_size': 512, 'seq_len': 128, 'd_model': 256},
        {'batch_size': 1024, 'seq_len': 128, 'd_model': 256},
        {'batch_size': 2048, 'seq_len': 128, 'd_model': 256},
        {'batch_size': 4096, 'seq_len': 128, 'd_model': 256},
        {'batch_size': 8192, 'seq_len': 64, 'd_model': 256},  # Reduce seq_len for extreme batch
    ]
    
    optimizer.test_configs = quick_configs
    results = optimizer.run_optimization_sweep()
    
    recommended = optimizer.get_recommended_config()
    
    print(f"\n🎯 RECOMMENDED CONFIGURATION:")
    print(f"   Batch Size: {recommended['batch_size']}")
    print(f"   Sequence Length: {recommended['seq_len']}")
    print(f"   Model Dimension: {recommended['d_model']}")
    print(f"   Expected Speed: {recommended.get('expected_tokens_per_sec', 'TBD'):.0f} tokens/sec")
    print(f"   Memory Usage: {recommended.get('memory_usage_gb', 'TBD'):.1f} GB")
    
    return recommended


def run_full_optimization():
    """Run comprehensive optimization sweep"""
    optimizer = M1MaxTransformerOptimizer()
    results = optimizer.run_optimization_sweep()
    
    # Display results summary
    if 'best_throughput_config' in results:
        best = results['best_throughput_config']
        print(f"\n🏆 OPTIMAL M1 MAX CONFIGURATION:")
        print(f"   Batch Size: {best['batch_size']}")
        print(f"   Sequence Length: {best['seq_len']}")
        print(f"   Model Dimension: {best['d_model']}")
        print(f"   Peak Throughput: {best['tokens_per_sec']:.0f} tokens/sec")
        print(f"   Memory Usage: {best['memory_used_gb']:.1f} GB")
        print(f"   Parameters: {best['parameters']:,}")
        
        # Compare to RNN achievement
        rnn_best = 5720  # Character-level from RNN module
        improvement = (best['tokens_per_sec'] / rnn_best) * 100
        print(f"   vs RNN Best: {improvement:.1f}% of RNN character speed")
    
    return optimizer.get_recommended_config()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="M1 Max Transformer Optimization")
    parser.add_argument("--quick", action="store_true", help="Run quick optimization test")
    parser.add_argument("--full", action="store_true", help="Run full optimization sweep")
    
    args = parser.parse_args()
    
    if args.quick:
        recommended = run_quick_optimization()
    elif args.full:
        recommended = run_full_optimization()
    else:
        print("Usage: python optimize_m1_max.py --quick or --full")
        print("  --quick: Fast test of key configurations")  
        print("  --full:  Comprehensive optimization sweep")