#!/usr/bin/env python3
"""
Maximal Accuracy Character RNN Training Script
Optimized for highest possible character-level training accuracy
"""

import torch
import torch.nn as nn
import os
import time
from pathlib import Path
from core.models import LSTM
from core.trainer import RNNTrainer, create_optimizer, create_scheduler
from character.dataset import create_sample_dataset
from core.device_utils import get_best_device
from core.results_utils import write_standard_results, generate_text_samples
import string
import random

class CharAccuracyOptimizedTrainer(RNNTrainer):
    """Extended trainer with character-level accuracy optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_train_accuracy = 0.0
        self.accuracy_patience = 0
        self.accuracy_plateau_threshold = 0.001  # Stop if accuracy improvement < 0.1%
        
    def train(self, train_loader, val_loader, num_epochs, verbose=True, accuracy_target=None):
        """Extended training with character-level accuracy monitoring"""
        print(f"🎯 Training for maximal character accuracy (target: {accuracy_target if accuracy_target else 'maximize'})")
        print(f"Accuracy plateau threshold: {self.accuracy_plateau_threshold:.1%}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self.train_epoch(train_loader, epoch, verbose)
            self.train_losses.append(train_loss)
            
            # Validation phase  
            val_loss, val_metrics = self.validate(val_loader)
            val_accuracy = val_metrics.get('accuracy', 0.0)
            
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                        
            # Track best validation
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_accuracy
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Track best training accuracy
            train_metrics = self.calculate_train_accuracy(train_loader)
            train_accuracy = train_metrics.get('accuracy', 0.0)
            
            accuracy_improved = train_accuracy > self.best_train_accuracy + self.accuracy_plateau_threshold
            if accuracy_improved:
                self.best_train_accuracy = train_accuracy
                self.accuracy_patience = 0
            else:
                self.accuracy_patience += 1
                
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"Epoch [{epoch+1:>3}/{num_epochs}] "
                      f"Train Loss: {train_loss:.6f} | Train Acc: {train_accuracy:.4f} | "
                      f"Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.4f} | "
                      f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")
                      
                if accuracy_improved:
                    print(f"🎯 New best training accuracy: {train_accuracy:.4f}")
                    
            # Save checkpoint (every 5 epochs, best validation, or accuracy improvement)
            if self.checkpoint_dir and (epoch % 5 == 0 or is_best or accuracy_improved):
                self.save_checkpoint(epoch, is_best or accuracy_improved)
                
            # Early stopping based on accuracy plateau or target
            if accuracy_target and train_accuracy >= accuracy_target:
                print(f"🎯 Target accuracy {accuracy_target:.4f} reached!")
                break
                
            if self.accuracy_patience >= 10:  # No accuracy improvement for 10 epochs
                print(f"⏹️  Training stopped due to accuracy plateau (no improvement > {self.accuracy_plateau_threshold:.1%} for 10 epochs)")
                break
                
            if self.patience_counter >= self.patience:
                print(f"⏹️  Early stopping triggered (validation loss plateau)")
                break
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'best_train_accuracy': self.best_train_accuracy,
            'learning_rates': self.learning_rates
        }
        
    def calculate_train_accuracy(self, train_loader):
        """Calculate training accuracy on a subset of training data"""
        self.model.eval()
        correct = 0
        total = 0
        
        # Sample a subset for efficiency (first 10 batches)
        with torch.no_grad():
            for i, (data, target) in enumerate(train_loader):
                if i >= 10:  # Only check first 10 batches for efficiency
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                
                # For character-level, we predict all positions
                if output.dim() == 3:  # (batch, seq, vocab)
                    output = output.reshape(-1, output.size(-1))  # (batch*seq, vocab)
                    target = target.reshape(-1)  # (batch*seq,)
                    
                pred = output.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
        self.model.train()
        return {'accuracy': correct / total if total > 0 else 0.0}

def main():
    """Run maximal accuracy character training"""
    
    # Enhanced configuration for character-level accuracy
    BATCH_SIZE = 2048      # Optimal from previous testing
    SEQUENCE_LENGTH = 50   # Longer sequences for better context
    EPOCHS = 50           # Extended training
    
    # Larger model architecture for character-level
    INPUT_SIZE = 256      # Large character embeddings
    HIDDEN_SIZE = 768     # Very large hidden state
    NUM_LAYERS = 4        # Deep network
    
    device = get_best_device()
    print(f"🎯 Maximal Accuracy Character RNN Training")
    print(f"Device: {device}")
    print(f"Model: {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden, {INPUT_SIZE} embed")
    print(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE:,}, seq_len={SEQUENCE_LENGTH}")
    print("=" * 70)
    
    # Create dataset with longer sequences
    print("📊 Creating enhanced character dataset...")
    train_loader, val_loader, tokenizer = create_sample_dataset(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        data_source='shakespeare'
    )
    
    vocab_size = tokenizer.vocab_size
    print(f"✅ Dataset ready: {len(train_loader):,} batches, vocab={vocab_size}")
    
    # Create enhanced character model
    print("🏗️  Building enhanced character model...")
    lstm = LSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        layer_norm=True,     # Essential for deep networks
        dropout=0.2          # Regularization for large model
    )
    
    class MaximalCharRNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, INPUT_SIZE)
            self.lstm = lstm
            self.dropout = nn.Dropout(0.2)
            self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)  # Additional normalization
            self.output = nn.Linear(HIDDEN_SIZE, vocab_size)
            
            # Initialize embeddings with better distribution
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.output.weight)
            
        def forward(self, x, hidden=None):
            embedded = self.embedding(x)
            lstm_out, hidden = self.lstm(embedded, hidden)
            lstm_out = self.dropout(lstm_out)
            lstm_out = self.layer_norm(lstm_out)  # Normalize before output
            logits = self.output(lstm_out)
            return logits, hidden
    
    model = MaximalCharRNNModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Enhanced character model ready: {param_count:,} parameters")
    
    # Advanced optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.003,              # Higher learning rate for character-level
        weight_decay=0.01,     # L2 regularization
        betas=(0.9, 0.95),     # Better momentum settings
        eps=1e-8
    )
    
    # Advanced learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,         # Warm up for 10% of training
        anneal_strategy='cos',  # Cosine annealing
        div_factor=10,         # Start at max_lr/10
        final_div_factor=200   # End at max_lr/200
    )
    
    # Label smoothing for character-level
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Less smoothing for character-level
    
    # Initialize accuracy-optimized trainer
    trainer = CharAccuracyOptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        mixed_precision=False,  # MPS doesn't support mixed precision
        grad_clip_norm=1.0,
        patience=15,            # Patient for character-level learning
        checkpoint_dir='checkpoints'
    )
    
    # Run extended training
    print(f"🚀 Starting maximal character accuracy training...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        verbose=True,
        accuracy_target=0.7     # Target 70% character accuracy
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Character training complete!")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"Best training accuracy: {history['best_train_accuracy']:.4f}")
    
    # Save final model
    print(f"💾 Saving maximal accuracy character model...")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_val_loss': history['best_val_loss'],
        'final_val_accuracy': history['best_val_accuracy'],
        'best_train_accuracy': history['best_train_accuracy'],
        'training_time': training_time,
        'epochs_trained': len(history['train_losses']),
        'batch_size': BATCH_SIZE,
        'sequence_length': SEQUENCE_LENGTH,
        'model_params': param_count,
        'vocab_size': vocab_size
    }, 'checkpoints/maximal_accuracy_char_model.pt')
    print(f"✅ Character model saved to checkpoints/maximal_accuracy_char_model.pt")
    
    # Calculate final throughput
    total_samples = len(train_loader) * BATCH_SIZE * len(history['train_losses'])
    samples_per_second = int(total_samples / training_time)
    
    # Generate high-quality character samples
    print(f"📝 Generating high-quality character samples...")
    text_samples = []
    
    # Generate diverse samples
    prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Romeo, Romeo, wherefore art thou",
        "Friends, Romans, countrymen",
        "Now is the winter of our discontent",
        "What light through yonder window breaks",
        "The quality of mercy is not strained"
    ]
    
    for i, prompt in enumerate(prompts):
        # Generate with model
        model.eval()
        with torch.no_grad():
            # Convert prompt to tokens
            input_tokens = [tokenizer.char_to_idx.get(c, 0) for c in prompt]
            input_tensor = torch.tensor([input_tokens], device=device)
            
            generated_tokens = input_tokens.copy()
            hidden = None
            
            # Generate 100 characters
            for _ in range(100):
                output, hidden = model(input_tensor, hidden)
                # Use temperature sampling for better quality
                logits = output[0, -1, :] / 0.8  # temperature=0.8
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(next_token)
                input_tensor = torch.tensor([[next_token]], device=device)
                
                # Stop at reasonable length
                if len(generated_tokens) >= len(input_tokens) + 100:
                    break
            
            # Convert back to text
            generated_text = ''.join([tokenizer.idx_to_char.get(t, '?') for t in generated_tokens])
        
        text_samples.append((f"sample_{i}", f"Prompt: {prompt}\nGenerated: {generated_text}"))
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text[:150]}...")  # Show first 150 chars
    
    # Save comprehensive results
    print(f"💾 Saving comprehensive character results...")
    write_standard_results(
        test_name="Maximal Accuracy Character RNN",
        model=model,
        device=device,
        training_time=training_time,
        final_loss=history['best_val_loss'],
        samples_per_second=samples_per_second,
        batch_size=BATCH_SIZE,
        generated_samples=text_samples,
        additional_metrics={
            'epochs_trained': len(history['train_losses']),
            'sequence_length': SEQUENCE_LENGTH,
            'vocab_size': vocab_size,
            'best_train_accuracy': history['best_train_accuracy'],
            'best_val_accuracy': history['best_val_accuracy'],
            'final_val_accuracy': history['val_accuracies'][-1] if history['val_accuracies'] else 0,
            'batches_per_epoch': len(train_loader),
            'tokenizer_type': 'character',
            'model_architecture': f"{NUM_LAYERS}L-{HIDDEN_SIZE}H-{INPUT_SIZE}E",
            'optimization_target': 'maximal_character_accuracy'
        }
    )
    
    print(f"\n🎯 MAXIMAL CHARACTER ACCURACY RESULTS")
    print(f"   Device: {device}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: {NUM_LAYERS} layers × {HIDDEN_SIZE} hidden × {INPUT_SIZE} embed")
    print(f"   Character vocabulary: {vocab_size}")
    print(f"   Epochs trained: {len(history['train_losses'])}")
    print(f"   Best training accuracy: {history['best_train_accuracy']:.4f}")
    print(f"   Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"   Final validation loss: {history['best_val_loss']:.6f}")
    print(f"   Throughput: {samples_per_second:,} samples/second")
    print(f"   Training time: {training_time/60:.1f} minutes")

if __name__ == '__main__':
    main()