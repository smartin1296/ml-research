#!/usr/bin/env python3
"""
Phase 2: Training & Optimization Improvements
Building on Phase 1 baseline with advanced training techniques

Phase 2 Enhancements:
- Label smoothing: Prevent overconfident predictions
- Advanced LR schedules: Cosine annealing, polynomial decay  
- Gradient accumulation: Larger effective batch sizes
- Enhanced optimizer settings: Better weight decay, beta parameters
"""

import sys
import torch
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.core import TransformerTrainer
from algorithms.transformers.phase_benchmark import TransformerPhaseBenchmark
from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer

# Reuse the dataset from Phase 1
from algorithms.transformers.run_phase1_scaled import ScaledTinyStoriesDataset
from torch.utils.data import Dataset, DataLoader


def create_phase2_model(vocab_size: int, d_model: int = 256, n_layers: int = 4, n_heads: int = 8) -> SimpleTransformer:
    """Create Phase 2 transformer - M1 Max optimized sequence length"""
    return SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_model * 4,  # Standard 4x expansion
        max_seq_len=64  # M1 Max optimized from profiling (+26% performance)
    )


def create_dataloaders(data_path: str, tokenizer: ScaledWordTokenizer, batch_size: int = 512):
    """Create train/val dataloaders - same as Phase 1"""
    
    # Load training data with same subset size as Phase 1 for comparison
    train_dataset = ScaledTinyStoriesDataset(
        data_path, 
        tokenizer, 
        max_length=64,  # M1 Max optimized sequence length
        subset_size=10000  # Same as Phase 1
    )
    
    # Create validation split
    val_size = len(train_dataset) // 10  # 10% validation
    train_size = len(train_dataset) - val_size
    
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Use same batch size as Phase 1 M1 Max optimal
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def run_phase2_training():
    """Run Phase 2 with training improvements"""
    
    print("=== Phase 2: Training & Optimization Improvements ===")
    
    # Setup paths
    data_path = project_root / "data/raw/text/tinystories/TinyStories-small.txt"
    
    # Phase 2 results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/phases/phase_2_training_improvements_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tokenizer - same as Phase 1 for fair comparison
    print("Creating tokenizer...")
    
    # Load stories for tokenizer training
    stories = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 50000:  # Limit for efficiency
                break
            if line.strip():
                try:
                    if line.strip().startswith('{'):
                        data = json.loads(line.strip())
                        story = data.get('story', data.get('text', ''))
                    else:
                        story = line.strip()
                    
                    if story and len(story) > 50:  # Basic quality filter
                        stories.append(story)
                except:
                    continue
    
    tokenizer = ScaledWordTokenizer(vocab_size=8192)
    tokenizer.build_vocab(stories)
    
    # Create model - same architecture as Phase 1
    print("Creating model...")
    model = create_phase2_model(vocab_size=len(tokenizer.word_to_idx))
    
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(str(data_path), tokenizer, batch_size=512)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize trainer with Phase 2 enhancements per roadmap
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        save_dir=str(results_dir / "checkpoints"),
        # Phase 2.1: Advanced Training Techniques
        label_smoothing=0.1,           # Prevent overconfident predictions (roadmap)
        gradient_accumulation_steps=2  # Effective large batch training (roadmap)
    )
    
    # Phase 2 training with advanced techniques
    print("\nStarting Phase 2 training...")
    
    training_start = time.time()
    
    # Train with Phase 2 roadmap improvements
    training_history = trainer.train(
        learning_rate=3e-4,          # Same as Phase 1 for comparison
        max_epochs=50,               # Allow more epochs for convergence
        warmup_steps=500,            # Same as Phase 1
        d_model=256,                 # Same as Phase 1
        # Phase 2.1: Advanced Training Techniques (roadmap)
        lr_schedule='cosine_annealing',  # Cosine decay (roadmap)
        weight_decay=0.01,               # L2 regularization tuning (roadmap 2.2)
        beta1=0.9,                       # Optimizer beta1
        beta2=0.98                       # Optimizer beta2 (Transformer paper standard)
    )
    
    training_time = time.time() - training_start
    
    # Save tokenizer for inference
    tokenizer_path = results_dir / "tokenizer.json" 
    tokenizer.save(str(tokenizer_path))
    
    # Generate test stories for quality assessment
    print("\nGenerating test stories...")
    inference_start = time.time()
    
    model.eval()
    device = next(model.parameters()).device
    
    test_prompts = [
        "Once upon a time",
        "There was a little", 
        "The brave princess",
        "In a magical forest",
        "A small cat"
    ]
    
    generated_stories = []
    
    for prompt in test_prompts:
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt, add_special=True)
        input_ids = torch.tensor([prompt_tokens]).to(device)
        
        # Generate
        with torch.no_grad():
            for _ in range(25):  # Generate 25 more tokens
                if input_ids.size(1) >= 128:  # Max sequence length
                    break
                    
                output = model(input_ids)
                next_token_logits = output[0, -1, :]
                
                # Temperature sampling
                temperature = 0.8
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop at end token
                if next_token.item() == tokenizer.word_to_idx.get('<EOS>', -1):
                    break
        
        # Decode
        story = tokenizer.decode(input_ids[0].cpu().tolist())
        generated_stories.append({
            "prompt": prompt,
            "story": story
        })
        print(f"'{prompt}' -> {story}")
    
    inference_time = time.time() - inference_start
    
    # Collect results
    final_results = {
        "phase": "2_training_improvements",
        "model_description": "Phase 1 baseline + label smoothing + cosine annealing + gradient accumulation",
        "training_results": {
            "epochs_trained": len(training_history['train_losses']),
            "training_time_seconds": training_time,
            "best_val_loss": trainer.best_val_loss,
            "best_val_accuracy": trainer.best_val_accuracy,
            "tokens_per_second": len(train_loader.dataset) * 64 * len(training_history['train_losses']) / training_time  # Approximate
        },
        "model_config": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "d_model": 256,
            "num_layers": 4,
            "num_heads": 8,
            "d_ff": 1024,
            "vocab_size": len(tokenizer.word_to_idx),
            "max_seq_len": 64,  # M1 Max optimized
            "batch_size": 512,
            # Phase 2 specific config
            "effective_batch_size": 1024,  # 512 * 2 gradient accumulation (roadmap)
            "label_smoothing": 0.1,
            "lr_schedule": "cosine_annealing",
            "gradient_accumulation_steps": 2
        },
        "generated_stories": generated_stories,
        "hardware": "M1 Max optimized settings",
        "inference_method": "Actual model inference with temperature sampling",
        "phase_2_improvements": [
            "Label smoothing (0.1) - prevents overconfident predictions (roadmap 2.1)",
            "Gradient accumulation (2x) - effective large batch training (roadmap 2.1)", 
            "Cosine annealing LR schedule - better convergence (roadmap 2.1)",
            "M1 Max sequence optimization - seq_len=64 for +26% performance (roadmap 2.3)",
            "Enhanced weight decay (0.01) - L2 regularization tuning (roadmap 2.2)"
        ]
    }
    
    # Save complete results
    results_file = results_dir / "phase2_complete_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Create markdown report
    report_content = f"""# Phase 2 Transformer Results

## Training Summary
- **Epochs**: {len(training_history['train_losses'])}
- **Training Time**: {training_time:.1f} seconds
- **Best Validation Loss**: {trainer.best_val_loss:.4f}
- **Best Validation Accuracy**: {trainer.best_val_accuracy:.6f}
- **Effective Batch Size**: 1024 (512 × 2 gradient accumulation)
- **Sequence Length**: 64 (M1 Max optimized for +26% performance)

## Phase 2 Roadmap Improvements
- **Label Smoothing**: 0.1 (prevents overconfident predictions - roadmap 2.1)
- **Gradient Accumulation**: 2x (effective large batch training - roadmap 2.1)
- **LR Schedule**: Cosine annealing (better convergence - roadmap 2.1)
- **M1 Max Optimization**: seq_len=64 for performance (roadmap 2.3)
- **Weight Decay**: 0.01 (L2 regularization tuning - roadmap 2.2)

## Model Architecture (Same as Phase 1)
- **d_model**: 256
- **Layers**: 4
- **Heads**: 8
- **Parameters**: {sum(p.numel() for p in model.parameters()):,}
- **Vocabulary**: {len(tokenizer.word_to_idx):,} words

## Generated Stories (Actual Model Inference)

"""
    
    for i, story in enumerate(generated_stories, 1):
        report_content += f"""### Story {i}: "{story['prompt']}"
{story['story']}

"""
    
    report_file = results_dir / "phase2_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\nPhase 2 Results Summary:")
    print(f"Training time: {training_time:.1f}s")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best validation accuracy: {trainer.best_val_accuracy:.6f}")
    print(f"Generated {len(generated_stories)} test stories")
    print(f"Results saved to: {results_dir}")
    
    # Initialize benchmarking for comparison
    benchmark = TransformerPhaseBenchmark()
    
    print(f"\nPhase 2 complete! Ready for Phase 1 vs Phase 2 comparison.")
    print(f"Run phase_benchmark.py to compare results.")
    
    return final_results


if __name__ == "__main__":
    try:
        results = run_phase2_training()
        print("Phase 2 training completed successfully!")
    except Exception as e:
        print(f"Phase 2 training failed: {e}")
        import traceback
        traceback.print_exc()