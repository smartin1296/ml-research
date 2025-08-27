#!/usr/bin/env python3
"""
Transformer Phase Benchmarking System
Standardized evaluation across all transformer evolution phases
"""

import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from benchmark_dataset import (
    create_benchmark_dataloaders, 
    StoryQualityEvaluator,
    generate_sample_stories
)


class TransformerPhaseBenchmark:
    """
    Comprehensive benchmarking system for tracking transformer improvements
    across evolution phases with consistent metrics and evaluation
    """
    
    def __init__(
        self, 
        phase_name: str,
        model_description: str,
        base_dir: str = "algorithms/transformers/results/phases"
    ):
        self.phase_name = phase_name
        self.model_description = model_description
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results directory
        self.results_dir = Path(base_dir) / f"phase_{phase_name}_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Phase Benchmark: {phase_name}")
        print(f"   Description: {model_description}")
        print(f"   Results dir: {self.results_dir}")
        
        # Benchmark configuration
        self.config = {
            'batch_size': 64,           # Optimized for M1 Max
            'max_length': 128,          # Good balance for stories
            'train_subset': 25000,      # Fast but meaningful training
            'val_subset': 2500,         # Representative validation
            'max_epochs': 20,           # Reasonable training time
            'generation_samples': 10,   # Stories for qualitative evaluation
            'seed': 42                  # Reproducible results
        }
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_perplexity': [],
            'tokens_per_sec': [],
            'learning_rates': [],
            'story_quality': []
        }
    
    def run_full_benchmark(
        self, 
        model: nn.Module, 
        trainer_class: Any,
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Run complete benchmark for a transformer phase
        
        Args:
            model: Transformer model to benchmark
            trainer_class: Trainer class (TransformerTrainer or subclass)
            learning_rate: Learning rate for training
            
        Returns:
            Complete benchmark results
        """
        print(f"\n🚀 Starting Full Benchmark: {self.phase_name}")
        print("=" * 60)
        
        benchmark_start = time.time()
        
        # 1. Setup data and model
        results = self._setup_benchmark(model, trainer_class, learning_rate)
        
        # 2. Training benchmark  
        training_results = self._run_training_benchmark(
            results['trainer'], 
            results['tokenizer'],
            results['device']
        )
        results.update(training_results)
        
        # 3. Performance evaluation
        performance_results = self._evaluate_performance(
            results['best_model'],
            results['val_loader'], 
            results['tokenizer'],
            results['device']
        )
        results.update(performance_results)
        
        # 4. Story generation evaluation
        story_results = self._evaluate_story_generation(
            results['best_model'],
            results['tokenizer'],
            results['device']
        )
        results.update(story_results)
        
        # 5. Save comprehensive results
        results['total_benchmark_time'] = time.time() - benchmark_start
        self._save_results(results)
        
        # 6. Generate report
        self._generate_benchmark_report(results)
        
        print(f"\n✅ Benchmark Complete: {results['total_benchmark_time']:.1f}s")
        print(f"📁 Results saved to: {self.results_dir}")
        
        return results
    
    def _setup_benchmark(self, model: nn.Module, trainer_class: Any, learning_rate: float) -> Dict[str, Any]:
        """Setup data, model, and trainer for benchmarking"""
        print(f"\n🔧 Setting up benchmark environment...")
        
        # Create benchmark dataset
        train_loader, val_loader, tokenizer = create_benchmark_dataloaders(
            batch_size=self.config['batch_size'],
            max_length=self.config['max_length'],
            train_subset=self.config['train_subset'],
            val_subset=self.config['val_subset']
        )
        
        # Initialize trainer
        trainer = trainer_class(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            save_dir=str(self.results_dir / "checkpoints")
        )
        
        # Model info
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        setup_results = {
            'train_loader': train_loader,
            'val_loader': val_loader, 
            'tokenizer': tokenizer,
            'trainer': trainer,
            'device': trainer.device,
            'model_parameters': param_count,
            'vocab_size': tokenizer.vocab_size,
            'learning_rate': learning_rate,
            'config': self.config.copy()
        }
        
        print(f"   ✅ Model parameters: {param_count:,}")
        print(f"   ✅ Vocabulary size: {tokenizer.vocab_size:,}")
        print(f"   ✅ Device: {trainer.device}")
        print(f"   ✅ Mixed precision: {trainer.mixed_precision}")
        
        return setup_results
    
    def _run_training_benchmark(
        self, 
        trainer: Any, 
        tokenizer: Any,
        device: torch.device
    ) -> Dict[str, Any]:
        """Run training and track metrics"""
        print(f"\n🎓 Training Phase Benchmark...")
        
        training_start = time.time()
        
        # Run training
        training_results = trainer.train(
            learning_rate=self.config.get('learning_rate', 1e-4),
            max_epochs=self.config['max_epochs'],
            warmup_steps=2000
        )
        
        training_time = time.time() - training_start
        
        # Load best model for evaluation
        best_checkpoint = trainer.save_dir / "best_checkpoint.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded best model (epoch {checkpoint['epoch']})")
        
        # Calculate training efficiency
        total_tokens = len(trainer.train_loader.dataset) * self.config['max_length'] * checkpoint.get('epoch', 1)
        avg_tokens_per_sec = total_tokens / training_time if training_time > 0 else 0
        
        return {
            'training_results': training_results,
            'training_time': training_time,
            'best_model': trainer.model,
            'final_train_loss': training_results['train_losses'][-1],
            'best_val_loss': trainer.best_val_loss,
            'best_val_accuracy': trainer.best_val_accuracy,
            'epochs_trained': len(training_results['train_losses']),
            'avg_tokens_per_sec': avg_tokens_per_sec,
            'convergence_epoch': self._find_convergence_epoch(training_results['val_losses'])
        }
    
    def _evaluate_performance(
        self, 
        model: nn.Module, 
        val_loader: Any,
        tokenizer: Any,
        device: torch.device
    ) -> Dict[str, Any]:
        """Evaluate model performance metrics"""
        print(f"\n📈 Performance Evaluation...")
        
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                start_time = time.time()
                
                input_tokens, target_tokens = batch
                input_tokens = input_tokens.to(device)
                target_tokens = target_tokens.to(device)
                
                # Forward pass
                if hasattr(model, 'encoder') and not hasattr(model, 'decoder'):
                    # Encoder-only model
                    output = model(input_tokens)
                else:
                    # Handle full transformer
                    output = model(input_tokens, input_tokens)  # Simple case for language modeling
                
                loss = nn.functional.cross_entropy(
                    output.contiguous().view(-1, output.size(-1)),
                    target_tokens.contiguous().view(-1),
                    ignore_index=tokenizer.pad_token
                )
                
                # Calculate accuracy
                predictions = output.argmax(dim=-1)
                mask = (target_tokens != tokenizer.pad_token)
                correct = (predictions == target_tokens) & mask
                
                total_loss += loss.item()
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                inference_times.append(time.time() - start_time)
                
                # Only evaluate subset for speed
                if batch_idx >= 50:  # Evaluate ~50 batches
                    break
        
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        avg_inference_time = np.mean(inference_times)
        
        return {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_perplexity': perplexity,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'tokens_per_sec_inference': (input_tokens.numel() / avg_inference_time) if avg_inference_time > 0 else 0
        }
    
    def _evaluate_story_generation(
        self, 
        model: nn.Module,
        tokenizer: Any,
        device: torch.device
    ) -> Dict[str, Any]:
        """Evaluate story generation quality"""
        print(f"\n📚 Story Generation Evaluation...")
        
        # Generate sample stories
        sample_stories = generate_sample_stories(
            model, tokenizer, device,
            num_stories=self.config['generation_samples'],
            max_length=100,
            temperature=0.8
        )
        
        # Evaluate story quality
        evaluator = StoryQualityEvaluator(tokenizer)
        quality_metrics = evaluator.evaluate_batch(sample_stories)
        
        # Save sample stories
        stories_path = self.results_dir / "sample_stories.json"
        with open(stories_path, 'w') as f:
            json.dump({
                'phase': self.phase_name,
                'timestamp': self.timestamp,
                'stories': sample_stories,
                'quality_metrics': quality_metrics
            }, f, indent=2)
        
        print(f"   ✅ Generated {len(sample_stories)} sample stories")
        print(f"   📊 Avg story length: {quality_metrics['avg_length']:.1f} words")
        print(f"   🎯 Avg coherence: {quality_metrics['avg_coherence_score']:.3f}")
        
        return {
            'sample_stories': sample_stories,
            'story_quality_metrics': quality_metrics
        }
    
    def _find_convergence_epoch(self, val_losses: List[float]) -> int:
        """Find approximate convergence epoch"""
        if len(val_losses) < 5:
            return len(val_losses)
        
        # Simple convergence detection: when loss stops decreasing significantly
        min_loss_epoch = val_losses.index(min(val_losses))
        return min_loss_epoch + 1
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive results"""
        # Prepare serializable results
        save_results = {
            'phase_name': self.phase_name,
            'model_description': self.model_description,
            'timestamp': self.timestamp,
            'config': self.config,
            'model_parameters': results['model_parameters'],
            'vocab_size': results['vocab_size'],
            'device': str(results['device']),
            'training_time': results['training_time'],
            'total_benchmark_time': results['total_benchmark_time'],
            'epochs_trained': results['epochs_trained'],
            'convergence_epoch': results['convergence_epoch'],
            'final_train_loss': results['final_train_loss'],
            'best_val_loss': results['best_val_loss'],
            'best_val_accuracy': results['best_val_accuracy'],
            'eval_accuracy': results['eval_accuracy'],
            'eval_perplexity': results['eval_perplexity'],
            'avg_tokens_per_sec': results['avg_tokens_per_sec'],
            'avg_inference_time_ms': results['avg_inference_time_ms'],
            'story_quality_metrics': results['story_quality_metrics'],
            'training_curves': {
                'train_losses': results['training_results']['train_losses'],
                'val_losses': results['training_results']['val_losses'],
                'val_accuracies': results['training_results']['val_accuracies'],
                'learning_rates': results['training_results']['learning_rates']
            }
        }
        
        # Save main results
        results_path = self.results_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"   💾 Saved results to {results_path}")
    
    def _generate_benchmark_report(self, results: Dict[str, Any]) -> None:
        """Generate human-readable benchmark report"""
        report_path = self.results_dir / "benchmark_report.md"
        
        report = f"""# Transformer Phase Benchmark Report

## Phase: {self.phase_name}
**Model**: {self.model_description}  
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Device**: {results['device']}

## Model Configuration
- **Parameters**: {results['model_parameters']:,}
- **Vocabulary Size**: {results['vocab_size']:,}
- **Max Sequence Length**: {self.config['max_length']}
- **Batch Size**: {self.config['batch_size']}

## Training Results
- **Training Time**: {results['training_time']:.1f}s
- **Epochs Trained**: {results['epochs_trained']}
- **Convergence Epoch**: {results['convergence_epoch']}
- **Final Train Loss**: {results['final_train_loss']:.4f}
- **Best Val Loss**: {results['best_val_loss']:.4f}
- **Best Val Accuracy**: {results['best_val_accuracy']:.3f}

## Performance Metrics
- **Training Speed**: {results['avg_tokens_per_sec']:.0f} tokens/sec
- **Inference Speed**: {results['avg_inference_time_ms']:.1f}ms per batch
- **Eval Accuracy**: {results['eval_accuracy']:.3f}
- **Eval Perplexity**: {results['eval_perplexity']:.2f}

## Story Generation Quality
- **Avg Story Length**: {results['story_quality_metrics']['avg_length']:.1f} words
- **Avg Vocabulary Size**: {results['story_quality_metrics']['avg_vocabulary_size']:.1f} unique words
- **Repetition Ratio**: {results['story_quality_metrics']['avg_repetition_ratio']:.3f} (lower is better)
- **Story Word Coverage**: {results['story_quality_metrics']['avg_story_word_coverage']:.3f}
- **Coherence Score**: {results['story_quality_metrics']['avg_coherence_score']:.3f}

## Sample Generated Stories

"""
        
        # Add sample stories
        for i, story in enumerate(results['sample_stories'][:3], 1):
            report += f"### Story {i}\n```\n{story[:300]}{'...' if len(story) > 300 else ''}\n```\n\n"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"   📋 Generated report: {report_path}")
    
    @staticmethod
    def compare_phases(results_dirs: List[str]) -> Dict[str, Any]:
        """Compare results across multiple phases"""
        print(f"\n🔍 Comparing {len(results_dirs)} transformer phases...")
        
        phase_results = []
        for results_dir in results_dirs:
            results_path = Path(results_dir) / "benchmark_results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    phase_results.append(json.load(f))
        
        if not phase_results:
            print("❌ No valid phase results found")
            return {}
        
        # Create comparison
        comparison = {
            'phases': [r['phase_name'] for r in phase_results],
            'parameters': [r['model_parameters'] for r in phase_results],
            'train_time': [r['training_time'] for r in phase_results],
            'final_val_loss': [r['best_val_loss'] for r in phase_results],
            'final_val_accuracy': [r['best_val_accuracy'] for r in phase_results],
            'story_coherence': [r['story_quality_metrics']['avg_coherence_score'] for r in phase_results],
            'tokens_per_sec': [r['avg_tokens_per_sec'] for r in phase_results]
        }
        
        print(f"✅ Phase comparison ready")
        return comparison


if __name__ == "__main__":
    print("🧪 Testing Phase Benchmark System")
    
    # This would be called from actual phase implementations
    print("✅ Phase benchmark system ready for transformer evolution!")