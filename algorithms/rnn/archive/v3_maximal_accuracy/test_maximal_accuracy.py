#!/usr/bin/env python3
"""
Maximal Accuracy Token RNN Training Script
Optimized for highest possible training accuracy with advanced techniques
"""

import torch
import torch.nn as nn
import json
import os
import math
from pathlib import Path
from core.trainer import RNNTrainer, create_optimizer, create_scheduler
from tokens.models import OptimizedTokenRNNModel
from tokens.tokenizers import BPETokenizer
from tokens.dataset import create_token_datasets, load_shakespeare_data, TokenTextGenerator
from core.device_utils import get_best_device
from core.results_utils import write_standard_results
import time

class AccuracyOptimizedTrainer(RNNTrainer):
    """Extended trainer with accuracy-focused optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_train_accuracy = 0.0
        self.accuracy_patience = 0
        self.accuracy_plateau_threshold = 0.001  # Stop if accuracy improvement < 0.1%
        
    def train(self, train_loader, val_loader, num_epochs, verbose=True, accuracy_target=None):
        """Extended training with accuracy monitoring"""
        print(f"🎯 Training for maximal accuracy (target: {accuracy_target if accuracy_target else 'maximize'})")
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
                    
            # Save checkpoint (only best validation or accuracy improvement)
            if self.checkpoint_dir and (is_best or accuracy_improved):
                # Always overwrite the same checkpoint to save space
                self.save_latest_checkpoint(epoch, is_best or accuracy_improved)
                
            # Early stopping based on accuracy plateau
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
            'best_val_accuracy': val_accuracy,
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
                
                if output.dim() == 3:
                    output = output[:, -1, :]
                    
                pred = output.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
        self.model.train()
        return {'accuracy': correct / total if total > 0 else 0.0}
        
    def save_latest_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint, always overwriting the same file to save space"""
        if not self.checkpoint_dir:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_train_accuracy': self.best_train_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
        }

        # Always save to the same file - overwrite each time
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        # Also save as best if this is the best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'  
            torch.save(checkpoint, best_path)

def main():
    """Run maximal accuracy training"""
    
    # Enhanced configuration for accuracy
    BATCH_SIZE = 4096  # Keep optimal throughput
    SEQ_LEN = 15       # Slightly longer sequences for better context
    VOCAB_SIZE = 500   # Optimal vocabulary size
    EPOCHS = 50        # Extended training
    
    # Larger model for better capacity
    HIDDEN_SIZE = 512  # Increased from 256
    NUM_LAYERS = 3     # Increased from 2  
    EMBED_DIM = 384    # Increased from 256
    
    device = get_best_device()
    print(f"🎯 Maximal Accuracy Token RNN Training")
    print(f"Device: {device}")
    print(f"Model: {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden, {EMBED_DIM} embed")
    print(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE:,}, seq_len={SEQ_LEN}")
    print("=" * 60)
    
    # Load and prepare data
    print("📊 Loading Shakespeare dataset...")
    text = load_shakespeare_data()
    print(f"Text length: {len(text):,} characters")
    
    # Initialize tokenizer
    print(f"🔤 Initializing BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, min_freq=2)
    tokenizer.fit(text)
    actual_vocab_size = len(tokenizer)
    print(f"Actual vocabulary size: {actual_vocab_size:,}")
    
    # Save tokenizer
    os.makedirs('checkpoints', exist_ok=True)
    tokenizer.save(Path('checkpoints/tokenizer_maximal_accuracy.json'))
    
    # Create datasets
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
    
    # Create enhanced model
    print("🏗️  Building enhanced model for maximal accuracy...")
    model = OptimizedTokenRNNModel(
        vocab_size=actual_vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        dropout=0.2,           # Slightly higher dropout for regularization
        layer_norm=True,       # Essential for convergence
        tie_weights=True       # Tie input/output embeddings for better generalization
    )
    
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Enhanced model ready: {param_count:,} parameters")
    
    # Advanced optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,              # Slightly higher learning rate
        weight_decay=0.01,     # L2 regularization
        betas=(0.9, 0.95),     # Better momentum settings
        eps=1e-8
    )
    
    # Advanced learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,         # Warm up for 10% of training
        anneal_strategy='cos',  # Cosine annealing
        div_factor=10,         # Start at max_lr/10
        final_div_factor=100   # End at max_lr/1000
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    
    # Initialize accuracy-optimized trainer
    trainer = AccuracyOptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        mixed_precision=False,  # MPS doesn't support mixed precision
        grad_clip_norm=1.0,
        patience=15,            # More patient for longer training
        checkpoint_dir='checkpoints'
    )
    
    # Load from best checkpoint if available
    checkpoint_path = Path('checkpoints/best_model.pt')
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"📂 Loading from best checkpoint: {checkpoint_path}")
        try:
            start_epoch = trainer.load_checkpoint(str(checkpoint_path))
            print(f"✅ Resumed from epoch {start_epoch}")
            print(f"   Best val loss: {trainer.best_val_loss:.6f}")
            print(f"   Best train accuracy: {trainer.best_train_accuracy:.4f}")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print(f"   Starting fresh training...")
            start_epoch = 0
    else:
        print(f"📂 No checkpoint found, starting fresh training")
    
    # Run extended training
    print(f"🚀 Starting maximal accuracy training from epoch {start_epoch}...")
    start_time = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        verbose=True,
        accuracy_target=0.8     # Target 80% accuracy
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Training complete!")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"Best training accuracy: {history['best_train_accuracy']:.4f}")
    
    # Save final model
    print(f"💾 Saving maximal accuracy model...")
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
        'model_params': param_count,
        'vocab_size': actual_vocab_size,
        'tokenizer_path': 'checkpoints/tokenizer_maximal_accuracy.json'
    }, 'checkpoints/maximal_accuracy_model.pt')
    print(f"✅ Model saved to checkpoints/maximal_accuracy_model.pt")
    
    # Calculate final throughput
    total_samples = len(train_loader) * BATCH_SIZE * len(history['train_losses'])
    samples_per_second = int(total_samples / training_time)
    
    # Generate high-quality text samples
    print(f"📝 Generating high-quality text samples...")
    generator = TokenTextGenerator(model, tokenizer, device)
    
    prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Romeo, Romeo, wherefore art thou Romeo",
        "Friends, Romans, countrymen",
        "Now is the winter of our discontent"
    ]
    
    text_samples = []
    for i, prompt in enumerate(prompts):
        generated = generator.generate(
            prompt=prompt,
            max_tokens=80,           # Longer generations
            temperature=0.7,         # Lower temperature for higher quality
            top_k=50,                # Larger top_k for diversity
        )
        text_samples.append((f"sample_{i}", f"Prompt: {prompt}\nGenerated: {generated}"))
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
    
    # Save comprehensive results
    print(f"💾 Saving comprehensive results...")
    write_standard_results(
        test_name="Maximal Accuracy Token RNN",
        model=model,
        device=device,
        training_time=training_time,
        final_loss=history['best_val_loss'],
        samples_per_second=samples_per_second,
        batch_size=BATCH_SIZE,
        generated_samples=text_samples,
        additional_metrics={
            'epochs_trained': len(history['train_losses']),
            'sequence_length': SEQ_LEN,
            'vocab_size': actual_vocab_size,
            'best_train_accuracy': history['best_train_accuracy'],
            'best_val_accuracy': history['best_val_accuracy'],
            'final_val_accuracy': history['val_accuracies'][-1] if history['val_accuracies'] else 0,
            'batches_per_epoch': len(train_loader),
            'tokenizer_type': 'bpe',
            'model_architecture': f"{NUM_LAYERS}L-{HIDDEN_SIZE}H-{EMBED_DIM}E",
            'optimization_target': 'maximal_accuracy'
        }
    )
    
    print(f"\n🎯 MAXIMAL ACCURACY RESULTS")
    print(f"   Device: {device}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: {NUM_LAYERS} layers × {HIDDEN_SIZE} hidden × {EMBED_DIM} embed")
    print(f"   Vocabulary: {actual_vocab_size:,}")
    print(f"   Epochs trained: {len(history['train_losses'])}")
    print(f"   Best training accuracy: {history['best_train_accuracy']:.4f}")
    print(f"   Best validation accuracy: {history['best_val_accuracy']:.4f}")
    print(f"   Final validation loss: {history['best_val_loss']:.6f}")
    print(f"   Throughput: {samples_per_second:,} samples/second")
    print(f"   Training time: {training_time/60:.1f} minutes")

if __name__ == '__main__':
    main()