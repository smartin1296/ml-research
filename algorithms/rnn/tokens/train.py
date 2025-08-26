#!/usr/bin/env python3
"""
MPS-Optimized Token RNN Training
Combines simpler compilation modes + manual MPS optimizations + other speed improvements
"""

import torch
import torch.nn as nn
import json
import os
import time
from pathlib import Path
from core.trainer import RNNTrainer, create_optimizer, create_scheduler
from tokens.models import OptimizedTokenRNNModel
from tokens.tokenizers import BPETokenizer
from tokens.dataset import create_token_datasets, load_shakespeare_data, TokenTextGenerator
from core.device_utils import get_best_device
from core.results_utils import write_standard_results

class MPSOptimizedModel(nn.Module):
    """MPS-optimized model with manual optimizations"""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, 
                 layer_norm=True, tie_weights=True, training_mode=True):
        super().__init__()
        self.training_mode = training_mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Use minimal dropout during training, none during inference
        dropout_p = 0.1 if training_mode else 0.0
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers optimized for MPS
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                embed_dim if i == 0 else hidden_size,
                hidden_size,
                batch_first=True,
                dropout=0.0  # Handle dropout manually
            ) for i in range(num_layers)
        ])
        
        # Manual dropout layers (can be disabled easily)
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_p) for _ in range(num_layers)
        ])
        
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None
            
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # Weight tying for better efficiency
        if tie_weights and embed_dim == hidden_size:
            self.output.weight = self.embedding.weight
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better convergence"""
        nn.init.xavier_uniform_(self.embedding.weight)
        if not hasattr(self.output.weight, '_is_shared'):  # Don't reinit if tied
            nn.init.xavier_uniform_(self.output.weight)
            
    def switch_to_inference_mode(self):
        """Remove dropout overhead for inference"""
        self.training_mode = False
        for dropout in self.dropouts:
            dropout.p = 0.0
        print("🚀 Switched to inference mode (dropout disabled)")
        
    def forward(self, x, hidden_states=None):
        # MPS Optimization: Ensure tensors are contiguous
        x = x.contiguous()
        
        x = self.embedding(x)
        
        new_hidden_states = []
        
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropouts)):
            h_0 = hidden_states[i] if hidden_states else None
            x, h_n = lstm(x, h_0)
            
            # MPS Optimization: Explicit memory management
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            
            if self.layer_norms:
                x = self.layer_norms[i](x)
                
            # Only apply dropout if in training mode and p > 0
            if self.training and dropout.p > 0:
                x = dropout(x)
                
            new_hidden_states.append(h_n)
            
        logits = self.output(x)
        return logits, new_hidden_states

def try_mps_compilation(model, device):
    """Try different compilation modes for MPS compatibility"""
    
    if device.type != 'mps':
        print(f"Device is {device.type}, skipping MPS-specific compilation")
        return model
        
    print("⚡ Attempting MPS-compatible compilation...")
    
    # Option 2: Try simpler compilation modes
    compilation_modes = [
        ('reduce-overhead', 'Reduce overhead mode'),
        ('default', 'Default mode'), 
        (None, 'No compilation')
    ]
    
    for mode, description in compilation_modes:
        if mode is None:
            print("❌ All compilation modes failed, using eager execution")
            return model
            
        try:
            print(f"   Trying: {description}")
            
            # Set MPS-friendly compilation settings
            if hasattr(torch._dynamo.config, 'suppress_errors'):
                torch._dynamo.config.suppress_errors = True
                
            compiled_model = torch.compile(model, mode=mode, dynamic=False)
            
            # Test compilation with a dummy forward pass
            dummy_input = torch.randint(0, 100, (2, 8)).to(device)
            with torch.no_grad():
                _ = compiled_model(dummy_input)
                
            print(f"✅ Compilation successful with {description}!")
            return compiled_model
            
        except Exception as e:
            print(f"   ❌ {description} failed: {str(e)[:100]}...")
            continue
    
    print("❌ All compilation modes failed, using eager execution")
    return model

def create_mps_optimized_datasets(text, tokenizer, seq_len, batch_size, train_split=0.8):
    """Create datasets with MPS and DataLoader optimizations"""
    from torch.utils.data import DataLoader
    from tokens.dataset import TokenSequenceDataset
    
    # Split text first, then create datasets
    split_point = int(len(text) * train_split)
    train_text = text[:split_point]
    val_text = text[split_point:]
    
    # Create datasets using existing constructor
    train_dataset = TokenSequenceDataset(train_text, tokenizer, seq_len)
    val_dataset = TokenSequenceDataset(val_text, tokenizer, seq_len)
    
    # MPS + DataLoader optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,           # Optimization: Use multiple workers
        pin_memory=True,         # Optimization: Faster GPU transfer
        persistent_workers=True, # Optimization: Keep workers alive
        drop_last=True,          # Consistent batch sizes for compilation
        prefetch_factor=2        # MPS Optimization: Prefetch batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader

def mps_memory_optimization():
    """Apply MPS-specific memory optimizations"""
    if not torch.backends.mps.is_available():
        return
        
    print("🔧 Applying MPS memory optimizations...")
    
    # Set MPS memory fraction if available
    try:
        torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
        print("   ✅ Set MPS memory fraction to 80%")
    except:
        print("   ⚠️  Could not set MPS memory fraction")
    
    # Enable MPS fallback to CPU for unsupported ops
    try:
        torch.backends.mps.fallback_enabled = True
        print("   ✅ Enabled MPS fallback to CPU")
    except:
        pass

class MPSAccuracyOptimizedTrainer(RNNTrainer):
    """MPS-optimized trainer with accuracy plateau detection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_train_accuracy = 0.0
        self.accuracy_patience = 0
        self.accuracy_plateau_threshold = 0.001  # Stop if accuracy improvement < 0.1%
        
    def train(self, train_loader, val_loader, num_epochs, verbose=True, accuracy_target=None):
        """Extended training with accuracy monitoring until target or plateau"""
        print(f"🎯 Training until {accuracy_target:.1%} accuracy or plateau detected")
        print(f"Accuracy plateau threshold: {self.accuracy_plateau_threshold:.1%}")
        
        epoch = 0
        while True:
            epoch_start = time.time()
            
            # MPS optimization: Sync at epoch start
            if self.device.type == 'mps':
                torch.mps.synchronize()
            
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
                
            # MPS optimization: Cleanup at epoch end
            if self.device.type == 'mps':
                torch.mps.synchronize()
                
            epoch += 1  # Increment for next iteration
                
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
        latest_path = self.checkpoint_dir / 'latest_mps_optimized.pt'
        torch.save(checkpoint, latest_path)
        
        # Also save as best if this is the best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_mps_optimized.pt'  
            torch.save(checkpoint, best_path)

def main():
    """Run MPS-optimized training until 80% accuracy or plateau"""
    
    # Speed Optimization: Use power-of-2 sequence length  
    BATCH_SIZE = 4096        # Keep optimal batch size
    SEQ_LEN = 16             # Power of 2 for better alignment
    MAX_EPOCHS = 1000        # High limit for scheduler - actual training stops on plateau/target
    VOCAB_SIZE = 500
    
    # Model architecture (keep successful config)
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3  
    EMBED_DIM = 384
    
    device = get_best_device()
    print(f"🚀 MPS-Optimized Token RNN Training")
    print(f"Device: {device}")
    print(f"Optimizations: MPS + Compilation + DataLoader + SeqLen + Dropout")
    print(f"Training: unlimited epochs (stops at 80% accuracy or plateau), batch={BATCH_SIZE:,}, seq_len={SEQ_LEN}")
    print("=" * 75)
    
    # Apply MPS-specific optimizations
    mps_memory_optimization()
    
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
    
    # Create MPS-optimized datasets
    print("📊 Creating MPS-optimized datasets...")
    train_loader, val_loader = create_mps_optimized_datasets(
        text=text,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        train_split=0.8
    )
    
    print(f"✅ Dataset ready: {len(train_loader):,} train batches, {len(val_loader):,} val batches")
    print(f"   DataLoader optimizations: workers=2, pin_memory=True, prefetch_factor=2")
    
    # Create MPS-optimized model
    print("🏗️  Building MPS-optimized model...")
    model = MPSOptimizedModel(
        vocab_size=actual_vocab_size,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        layer_norm=True,
        tie_weights=True,
        training_mode=True  # Enable minimal dropout for training
    )
    
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model ready: {param_count:,} parameters")
    
    # MPS Optimization: Explicit memory sync
    if device.type == 'mps':
        torch.mps.synchronize()
        print("   ✅ MPS memory synchronized")
    
    # Try MPS-compatible compilation
    model = try_mps_compilation(model, device)
    
    # Setup optimized training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Scheduler optimized for shorter training
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=len(train_loader),
        epochs=MAX_EPOCHS,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize MPS accuracy-optimized trainer
    trainer = MPSAccuracyOptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        mixed_precision=False,  # MPS doesn't support mixed precision yet
        grad_clip_norm=1.0,
        patience=15,            # Patient for validation loss plateau
        checkpoint_dir='checkpoints'
    )
    
    # Run MPS-optimized training
    print(f"🚀 Starting MPS-optimized training...")
    start_time = time.time()
    
    # MPS Optimization: Warm up
    if device.type == 'mps':
        print("⚡ MPS warmup...")
        torch.mps.synchronize()
        torch.mps.empty_cache()
    
    # Train without epoch limit - will stop on accuracy target or plateau
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=float('inf'),  # Will be ignored due to while loop
        verbose=True,
        accuracy_target=0.8  # Train until 80% accuracy or plateau
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✅ MPS-optimized training complete!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    
    # Calculate optimized throughput
    total_samples = len(train_loader) * BATCH_SIZE * len(history['train_losses'])
    samples_per_second = int(total_samples / training_time)
    
    # Speed Optimization: Switch to inference mode for text generation
    print(f"📝 Generating text with inference-optimized model...")
    model.switch_to_inference_mode()  # Remove dropout overhead
    
    if device.type == 'mps':
        torch.mps.synchronize()
        
    generator = TokenTextGenerator(model, tokenizer, device)
    
    prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Romeo, Romeo",
    ]
    
    text_samples = []
    generation_start = time.time()
    
    for i, prompt in enumerate(prompts):
        generated = generator.generate(
            prompt=prompt,
            max_tokens=80,
            temperature=0.7,
            top_k=50,
        )
        text_samples.append((f"sample_{i}", f"Prompt: {prompt}\nGenerated: {generated}"))
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:100]}...")
    
    generation_time = time.time() - generation_start
    print(f"Text generation time: {generation_time:.2f}s")
    
    # Final MPS cleanup
    if device.type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    
    # Save MPS-optimized results
    print(f"💾 Saving MPS-optimized results...")
    write_standard_results(
        test_name="MPS-Optimized Token RNN",
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
            'final_val_accuracy': history['val_accuracies'][-1] if history['val_accuracies'] else 0,
            'batches_per_epoch': len(train_loader),
            'tokenizer_type': 'bpe',
            'model_architecture': f"{NUM_LAYERS}L-{HIDDEN_SIZE}H-{EMBED_DIM}E",
            'optimizations': 'mps+compilation+dataloader+seqlen+dropout',
            'generation_time': generation_time,
            'device_optimizations': 'mps_sync+memory_fraction+fallback'
        }
    )
    
    print(f"\n⚡ MPS-OPTIMIZED TOKEN RNN RESULTS")
    print(f"   Device: {device}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: {NUM_LAYERS} layers × {HIDDEN_SIZE} hidden × {EMBED_DIM} embed")
    print(f"   Optimizations Applied:")
    print(f"     • MPS-compatible compilation (fallback modes)")
    print(f"     • MPS memory management + synchronization")
    print(f"     • DataLoader (workers=2, pin_memory=True, prefetch)")
    print(f"     • Sequence length: {SEQ_LEN} (power of 2)")
    print(f"     • Dropout removal for inference")
    print(f"   Throughput: {samples_per_second:,} samples/second")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Status: MPS-OPTIMIZED ✅")

if __name__ == '__main__':
    main()