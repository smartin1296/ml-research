#!/usr/bin/env python3
"""
Primary RNN Test Script - M1 Max Optimized
Uses optimal batch_size=2048, 3 epochs, with standardized results output
"""

import torch
import torch.nn as nn
from core.models import LSTM
from core.trainer import RNNTrainer, create_optimizer, create_scheduler
from character.dataset import create_sample_dataset
from core.device_utils import get_best_device
from core.results_utils import write_standard_results, generate_text_samples
import time

def main():
    """Run optimized RNN training with M1 Max optimal settings"""
    
    # Optimal M1 Max configuration (from benchmarking)
    BATCH_SIZE = 2048
    SEQUENCE_LENGTH = 25
    EPOCHS = 3
    
    device = get_best_device()
    print(f"🚀 Primary RNN Test - M1 Max Optimized")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE:,}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 50)
    
    # Create dataset with optimal settings
    print("📊 Creating dataset...")
    train_loader, val_loader, tokenizer = create_sample_dataset(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        data_source='shakespeare'
    )
    
    vocab_size = tokenizer.vocab_size
    print(f"✅ Dataset ready: {len(train_loader):,} batches, vocab={vocab_size}")
    
    # Create optimal model (from M1 Max benchmarking)
    print("🏗️  Building model...")
    lstm = LSTM(
        input_size=192,      # Large embedding
        hidden_size=384,     # Large hidden size
        num_layers=2,        # Multi-layer
        layer_norm=True,     # Essential for convergence
        dropout=0.1          # Light regularization
    )
    
    class OptimalRNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 192)
            self.lstm = lstm
            self.dropout = nn.Dropout(0.1)
            self.output = nn.Linear(384, vocab_size)
            
        def forward(self, x, hidden=None):
            embedded = self.embedding(x)
            lstm_out, hidden = self.lstm(embedded, hidden)
            lstm_out = self.dropout(lstm_out)
            logits = self.output(lstm_out)
            return logits, hidden
    
    model = OptimalRNNModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready: {param_count:,} parameters")
    
    # Setup training with optimal settings
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
        checkpoint_dir='checkpoints'  # Save model every epoch
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
    
    # Save final model manually for test script
    print(f"💾 Saving final trained model...")
    import os
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'final_val_loss': history['best_val_loss'],
        'training_time': training_time,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'model_params': param_count
    }, 'checkpoints/final_test_model.pt')
    print(f"✅ Model saved to checkpoints/final_test_model.pt")
    
    # Calculate throughput
    total_samples = len(train_loader) * BATCH_SIZE * EPOCHS
    samples_per_second = int(total_samples / training_time)
    
    # Generate text samples
    print(f"📝 Generating text samples...")
    text_samples = generate_text_samples(model, tokenizer, device)
    
    # Save results using standard format
    print(f"💾 Saving results...")
    write_standard_results(
        test_name="Primary RNN Test",
        model=model,
        device=device,
        training_time=training_time,
        final_loss=history['best_val_loss'],
        samples_per_second=samples_per_second,
        batch_size=BATCH_SIZE,
        generated_samples=text_samples,
        additional_metrics={
            'epochs': EPOCHS,
            'sequence_length': SEQUENCE_LENGTH,
            'final_train_loss': history['train_losses'][-1],
            'best_val_accuracy': history['best_val_accuracy'],
            'final_val_accuracy': history['val_accuracies'][-1],
            'vocab_size': vocab_size,
            'batches_per_epoch': len(train_loader)
        }
    )
    
    print(f"\n🎯 SUMMARY")
    print(f"   Device: {device}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Throughput: {samples_per_second:,} samples/second")
    print(f"   Final loss: {history['best_val_loss']:.6f}")
    print(f"   Training time: {training_time:.1f}s")

if __name__ == '__main__':
    main()