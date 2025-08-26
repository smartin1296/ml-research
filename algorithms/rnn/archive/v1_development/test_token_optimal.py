#!/usr/bin/env python3
"""
Optimal Token RNN Test Script - M1 Max Optimized
Uses optimal batch_size=4096, with checkpointing and validation like character prediction
"""

import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from trainer import RNNTrainer, create_optimizer, create_scheduler
from token_models import OptimizedTokenRNNModel
from tokenizers import BPETokenizer
from token_dataset import create_token_datasets, load_shakespeare_data, TokenTextGenerator
from device_utils import get_best_device
from results_utils import write_standard_results
import time

def main():
    """Run optimized token RNN training with M1 Max optimal settings"""
    
    # Load optimal configuration from testing
    config_path = Path('token_optimal_final.json')
    if config_path.exists():
        with open(config_path, 'r') as f:
            optimal_config = json.load(f)
        
        # Use tested optimal settings
        BATCH_SIZE = optimal_config['training']['batch_size']  # 4096
        SEQ_LEN = optimal_config['training']['seq_len']        # 10
        VOCAB_SIZE = optimal_config['vocab_size']              # 500
        HIDDEN_SIZE = optimal_config['model']['hidden_size']   # 256
        NUM_LAYERS = optimal_config['model']['num_layers']     # 2
        EMBED_DIM = optimal_config['model']['embed_dim']       # 256
    else:
        # Fallback to similar character model size but with batch_size=4096
        BATCH_SIZE = 4096
        SEQ_LEN = 25
        VOCAB_SIZE = 500
        HIDDEN_SIZE = 256
        NUM_LAYERS = 2
        EMBED_DIM = 256
    
    EPOCHS = 3
    
    device = get_best_device()
    print(f"🚀 Optimal Token RNN Test - M1 Max Optimized")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE:,}")
    print(f"Vocab Size: {VOCAB_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 50)
    
    # Load Shakespeare data
    print("📊 Loading Shakespeare dataset...")
    text = load_shakespeare_data()
    print(f"Text length: {len(text):,} characters")
    
    # Initialize and fit BPE tokenizer
    print(f"🔤 Initializing BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, min_freq=2)
    tokenizer.fit(text)
    actual_vocab_size = len(tokenizer)
    print(f"Actual vocabulary size: {actual_vocab_size:,}")
    
    # Save tokenizer
    os.makedirs('checkpoints', exist_ok=True)
    tokenizer.save(Path('checkpoints/tokenizer_bpe_optimal.json'))
    print(f"💾 Saved tokenizer to checkpoints/")
    
    # Create datasets with optimal settings
    print("📊 Creating datasets...")
    train_loader, val_loader = create_token_datasets(
        text=text,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        stride=1,
        train_split=0.8,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    
    print(f"✅ Dataset ready: {len(train_loader):,} train batches, {len(val_loader):,} val batches")
    
    # Create optimal model (similar to character model architecture)
    print("🏗️  Building optimal token model...")
    model = OptimizedTokenRNNModel(
        vocab_size=actual_vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        dropout=0.1,
        layer_norm=True  # Essential for convergence like character model
    )
    
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready: {param_count:,} parameters")
    
    # Setup training with optimal settings (match character model)
    optimizer = create_optimizer(model, optimizer_type='adamw', lr=0.001)
    scheduler = create_scheduler(optimizer, scheduler_type='cosine', num_epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    trainer = RNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        mixed_precision=False,  # MPS doesn't support mixed precision yet
        grad_clip_norm=1.0,
        patience=EPOCHS + 5,    # Don't stop early for short training
        checkpoint_dir='checkpoints'  # Save model every epoch like character model
    )
    
    # Train
    print(f"🚀 Starting training ({EPOCHS} epochs)...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ Training complete!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
    
    # Save final model manually for test script (like character model)
    print(f"💾 Saving final trained token model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'final_val_loss': history['best_val_loss'],
        'final_val_accuracy': history['best_val_accuracy'],
        'training_time': training_time,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'model_params': param_count,
        'vocab_size': actual_vocab_size,
        'tokenizer_path': 'checkpoints/tokenizer_bpe_optimal.json'
    }, 'checkpoints/final_token_model.pt')
    print(f"✅ Token model saved to checkpoints/final_token_model.pt")
    
    # Calculate throughput
    total_samples = len(train_loader) * BATCH_SIZE * EPOCHS
    samples_per_second = int(total_samples / training_time)
    
    # Generate text samples
    print(f"📝 Generating text samples...")
    generator = TokenTextGenerator(model, tokenizer, device)
    
    prompts = [
        "To be or not to be",
        "All the world's a stage", 
        "Romeo, Romeo",
    ]
    
    text_samples = []
    for prompt in prompts:
        generated = generator.generate(
            prompt=prompt,
            max_tokens=50,
            temperature=0.8,
            top_k=40,
        )
        # Format for results_utils compatibility
        text_samples.append((f"seed_{len(text_samples)}", f"Prompt: {prompt}\nGenerated: {generated}"))
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print()
    
    # Save results using standard format (like character model)
    print(f"💾 Saving results...")
    write_standard_results(
        test_name="Optimal Token RNN Test",
        model=model,
        device=device,
        training_time=training_time,
        final_loss=history['best_val_loss'],
        samples_per_second=samples_per_second,
        batch_size=BATCH_SIZE,
        generated_samples=text_samples,
        additional_metrics={
            'epochs': EPOCHS,
            'sequence_length': SEQ_LEN,
            'vocab_size': actual_vocab_size,
            'final_train_loss': history['train_losses'][-1],
            'best_val_accuracy': history['best_val_accuracy'],
            'final_val_accuracy': history['val_accuracies'][-1],
            'batches_per_epoch': len(train_loader),
            'tokenizer_type': 'bpe'
        }
    )
    
    print(f"\n🎯 SUMMARY")
    print(f"   Device: {device}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Vocabulary: {actual_vocab_size:,}")
    print(f"   Throughput: {samples_per_second:,} samples/second")
    print(f"   Final loss: {history['best_val_loss']:.6f}")
    print(f"   Final accuracy: {history['best_val_accuracy']:.4f}")
    print(f"   Training time: {training_time:.1f}s")

if __name__ == '__main__':
    main()