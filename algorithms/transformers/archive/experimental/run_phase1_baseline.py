#!/usr/bin/env python3
"""
Phase 1 Baseline: "Attention is All You Need" (2017)
Run the original transformer architecture with full benchmarking
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.core import TransformerTrainer
from algorithms.transformers.phase_benchmark import TransformerPhaseBenchmark
from algorithms.transformers.benchmark_dataset import create_benchmark_dataloaders

# Import the simplified model from test_basic
from algorithms.transformers.test_basic import SimpleTransformer


def create_phase1_model(vocab_size: int) -> torch.nn.Module:
    """
    Create Phase 1 model: Optimized for M1 Max performance
    Based on optimization results: d_model=256, seq_len=64 optimal
    """
    # Use M1 Max optimized hyperparameters
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=256,        # Optimal from testing
        num_heads=8,        # Original paper (256/8 = 32 per head)
        num_layers=6,       # Original paper  
        d_ff=1024,          # 4x d_model (standard ratio)
        max_seq_len=128     # Allow some flexibility beyond optimal 64
    )
    
    return model


def run_phase1_baseline():
    """Run Phase 1 baseline benchmark"""
    
    print("🎯 PHASE 1: 'Attention is All You Need' Baseline")
    print("=" * 60)
    print("📋 Original 2017 Transformer Architecture")
    print("   - Multi-head attention (8 heads)")
    print("   - 6-layer encoder architecture")  
    print("   - Sinusoidal positional encoding")
    print("   - ReLU activation in feed-forward")
    print("   - Original learning rate schedule")
    print("   - AdamW optimizer (modern improvement)")
    
    # Initialize benchmark system
    benchmark = TransformerPhaseBenchmark(
        phase_name="1_baseline_2017",
        model_description="Original 'Attention is All You Need' architecture with modern training"
    )
    
    # Create data using M1 Max optimized settings
    print(f"\n📊 Setting up TinyStories benchmark dataset...")
    train_loader, val_loader, tokenizer = create_benchmark_dataloaders(
        batch_size=512,       # Optimal from M1 Max testing (266K tokens/sec)
        max_length=64,        # Optimal sequence length from testing
        train_subset=25000,   # Larger dataset for better training (fast with optimal batch)
        val_subset=3000       # Proportional validation set
    )
    
    # Create Phase 1 model
    print(f"\n🏗️ Creating Phase 1 Transformer...")
    model = create_phase1_model(tokenizer.vocab_size)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: 6-layer encoder, 8-head attention")
    print(f"   Dimensions: d_model=256, d_ff=1024")
    
    # Override benchmark config for Phase 1 with M1 Max optimized settings
    benchmark.config.update({
        'batch_size': 512,          # M1 Max optimal (266K tokens/sec)
        'max_length': 64,           # M1 Max optimal sequence length
        'train_subset': 25000,      # Larger training set (fast with optimal batch)
        'val_subset': 3000,         # Proportional validation set
        'max_epochs': 30,           # Allow good training with fast throughput
        'generation_samples': 10,   # More story samples for evaluation
        'learning_rate': 1e-4       # Standard transformer learning rate
    })
    
    # Run comprehensive benchmark
    print(f"\n🚀 Running Phase 1 comprehensive benchmark...")
    results = benchmark.run_full_benchmark(
        model=model,
        trainer_class=TransformerTrainer,
        learning_rate=benchmark.config['learning_rate']
    )
    
    # Phase 1 specific analysis
    print(f"\n📊 PHASE 1 RESULTS SUMMARY:")
    print(f"   🎓 Training: {results['training_time']:.1f}s ({results['epochs_trained']} epochs)")
    print(f"   📈 Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"   🎯 Best Val Accuracy: {results['best_val_accuracy']:.3f}")
    print(f"   ⚡ Training Speed: {results['avg_tokens_per_sec']:.0f} tokens/sec")
    print(f"   📚 Story Coherence: {results['story_quality_metrics']['avg_coherence_score']:.3f}")
    print(f"   💾 Results: {benchmark.results_dir}")
    
    # Display sample stories
    print(f"\n📖 Sample Generated Stories (Phase 1):")
    print("-" * 40)
    for i, story in enumerate(results['sample_stories'][:3], 1):
        print(f"Story {i}: {story[:150]}{'...' if len(story) > 150 else ''}")
        print()
    
    # Save phase comparison data for future phases
    comparison_data = {
        'phase_1_baseline': {
            'phase_name': '1_baseline_2017',
            'parameters': param_count,
            'train_time': results['training_time'],
            'val_loss': results['best_val_loss'],
            'val_accuracy': results['best_val_accuracy'], 
            'tokens_per_sec': results['avg_tokens_per_sec'],
            'coherence_score': results['story_quality_metrics']['avg_coherence_score'],
            'results_dir': str(benchmark.results_dir)
        }
    }
    
    comparison_file = project_root / "algorithms" / "transformers" / "phase_comparison.json"
    import json
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✅ Phase 1 baseline established!")
    print(f"📊 Comparison data saved for future phase analysis")
    print(f"🎯 Ready for Phase 2 improvements!")
    
    return results, benchmark.results_dir


def quick_generation_test(model_path: str = None):
    """Quick test of story generation quality"""
    print(f"\n🎨 Quick Generation Test")
    
    # Load model if provided
    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path}")
        # Implementation for loading saved model
        pass
    else:
        print("Creating fresh model for quick test...")
        _, _, tokenizer = create_benchmark_dataloaders(batch_size=1, train_subset=100, val_subset=10)
        model = create_phase1_model(tokenizer.vocab_size)
        
        # Quick untrained generation test
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        
        from algorithms.transformers.benchmark_dataset import generate_sample_stories
        stories = generate_sample_stories(model, tokenizer, device, num_stories=3, max_length=50)
        
        print(f"Untrained model stories (for comparison):")
        for i, story in enumerate(stories, 1):
            print(f"{i}. {story}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 Transformer Baseline")
    parser.add_argument("--quick-test", action="store_true", help="Run quick generation test only")
    parser.add_argument("--model-path", type=str, help="Path to saved model for testing")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_generation_test(args.model_path)
    else:
        results, results_dir = run_phase1_baseline()
        print(f"\n🎉 Phase 1 complete! Next: Phase 2 training improvements")
        print(f"📁 Full results: {results_dir}")