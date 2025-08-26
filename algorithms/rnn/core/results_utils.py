"""
Standard results formatting utilities for all RNN tests
Provides consistent human-readable output format across all scripts
"""

import json
import torch
from datetime import datetime
from pathlib import Path

def write_standard_results(
    test_name: str,
    model,
    device: torch.device,
    training_time: float,
    final_loss: float,
    samples_per_second: int = None,
    batch_size: int = None,
    generated_samples: list = None,
    additional_metrics: dict = None
):
    """
    Write results in standard human-readable format
    
    Args:
        test_name: Name of the test (e.g., "LSTM Quick Test", "GPU Performance Test")
        model: PyTorch model for parameter counting
        device: Device used for training
        training_time: Total training time in seconds
        final_loss: Final validation or training loss
        samples_per_second: Throughput (optional)
        batch_size: Batch size used (optional) 
        generated_samples: List of generated text samples (optional)
        additional_metrics: Dict of extra metrics to include (optional)
    """
    
    # Create results directory if needed
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name.lower().replace(' ', '_')}_{timestamp_str}.txt"
    filepath = results_dir / filename
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    # Write human-readable summary
    with open(filepath, 'w') as f:
        f.write(f"{test_name} Results\n")
        f.write("=" * (len(test_name) + 8) + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model Parameters: {param_count:,}\n")
        
        if batch_size:
            f.write(f"Batch Size: {batch_size:,}\n")
            
        if samples_per_second:
            f.write(f"Throughput: {samples_per_second:,} samples/second\n")
            
        f.write(f"Final Loss: {final_loss:.6f}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        
        # Additional metrics
        if additional_metrics:
            f.write("\nAdditional Metrics:\n")
            for key, value in additional_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Generated text samples
        if generated_samples:
            f.write("\nText Generations:\n\n")
            for i, (seed, generated) in enumerate(generated_samples):
                f.write(f"Seed: '{seed}'\n")
                f.write(f"Generated: {generated}\n")
                f.write("-" * 50 + "\n\n")
    
    # Also save JSON version for programmatic access
    json_data = {
        'test_name': test_name,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_parameters': param_count,
        'batch_size': batch_size,
        'samples_per_second': samples_per_second,
        'final_loss': final_loss,
        'training_time': training_time,
        'additional_metrics': additional_metrics or {},
        'generated_samples': generated_samples or []
    }
    
    json_filename = f"{test_name.lower().replace(' ', '_')}_{timestamp_str}.json"
    json_filepath = results_dir / json_filename
    
    with open(json_filepath, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"📝 Results saved:")
    print(f"   Human-readable: {filepath}")
    print(f"   Machine-readable: {json_filepath}")
    
    return str(filepath), str(json_filepath)

def print_results_table(results: dict):
    """Print results in a formatted table.
    
    Args:
        results: Dictionary of results to print
    """
    # Print key metrics
    print("\nKey Metrics:")
    print("-" * 40)
    
    metrics = [
        ("Model Size", f"{results.get('model_params', 0):,}" if results.get('model_params') else "N/A"),
        ("Vocab Size", f"{results.get('vocab_size', 0):,}" if results.get('vocab_size') else "N/A"),
        ("Train Loss", f"{results.get('train_loss', 0):.4f}" if results.get('train_loss') else "N/A"),
        ("Val Loss", f"{results.get('val_loss', 0):.4f}" if results.get('val_loss') else "N/A"),
        ("Val Accuracy", f"{results.get('val_accuracy', 0)*100:.2f}%" if results.get('val_accuracy') else "N/A"),
        ("Train Time", f"{results.get('train_time', 0):.2f}s" if results.get('train_time') else "N/A"),
        ("Throughput", f"{results.get('throughput_samples_per_sec', 0):,.0f} samples/sec" if results.get('throughput_samples_per_sec') else "N/A"),
        ("Device", results.get('device', 'N/A')),
    ]
    
    for name, value in metrics:
        print(f"{name:<20} {value}")
    print("-" * 40)


def generate_text_samples(model, tokenizer, device, seeds=None, max_length=100, temperature=0.8):
    """Generate text samples for results"""
    if seeds is None:
        seeds = ['To be', 'The', 'Hello']
    
    from dataset import SequenceGenerator
    generator = SequenceGenerator(model, tokenizer, device)
    
    samples = []
    for seed in seeds:
        try:
            generated = generator.generate(seed, max_length=max_length, temperature=temperature)
            samples.append((seed, generated))
        except Exception as e:
            samples.append((seed, f"Generation failed: {e}"))
    
    return samples