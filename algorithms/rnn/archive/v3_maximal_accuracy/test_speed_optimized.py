#!/usr/bin/env python3
"""
Speed-Optimized Token RNN Training
Implements 4 key optimizations: compilation, dataloader, sequence length, dropout removal
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

class SpeedOptimizedModel(nn.Module):
    """Speed-optimized version with dropout removal capability"""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, 
                 layer_norm=True, tie_weights=True, training_mode=True):
        super().__init__()
        self.training_mode = training_mode
        
        # Use minimal dropout during training, none during inference
        dropout_p = 0.1 if training_mode else 0.0
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                embed_dim if i == 0 else hidden_size,
                hidden_size,
                batch_first=True,
                dropout=0.0  # We'll handle dropout manually for better control
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
        x = self.embedding(x)
        
        new_hidden_states = []
        
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropouts)):
            h_0 = hidden_states[i] if hidden_states else None
            x, h_n = lstm(x, h_0)
            
            if self.layer_norms:
                x = self.layer_norms[i](x)
                
            # Only apply dropout if in training mode and p > 0
            if self.training and dropout.p > 0:
                x = dropout(x)
                
            new_hidden_states.append(h_n)
            
        logits = self.output(x)
        return logits, new_hidden_states

def create_optimized_datasets(text, tokenizer, seq_len, batch_size, train_split=0.8):
    """Create datasets with optimized DataLoader settings"""
    from torch.utils.data import DataLoader
    from tokens.dataset import TokenSequenceDataset
    
    # Split text first, then create datasets
    split_point = int(len(text) * train_split)
    train_text = text[:split_point]
    val_text = text[split_point:]
    
    # Create datasets using existing constructor
    train_dataset = TokenSequenceDataset(train_text, tokenizer, seq_len)
    val_dataset = TokenSequenceDataset(val_text, tokenizer, seq_len)
    
    # Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,           # Optimization 1: Use multiple workers
        pin_memory=True,         # Optimization 2: Faster GPU transfer
        persistent_workers=True, # Optimization 3: Keep workers alive
        drop_last=True          # Consistent batch sizes for compilation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    return train_loader, val_loader

def main():
    """Run speed-optimized training"""
    
    # Speed Optimization 3: Use power-of-2 sequence length
    BATCH_SIZE = 4096        # Keep optimal batch size
    SEQ_LEN = 16             # Changed from 15 to 16 (power of 2)
    EPOCHS = 10              # Shorter test run
    VOCAB_SIZE = 500
    
    # Model architecture (keep successful config)
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3  
    EMBED_DIM = 384
    
    device = get_best_device()
    print(f"🚀 Speed-Optimized Token RNN Training")
    print(f"Device: {device}")
    print(f"Optimizations: Compilation + DataLoader + SeqLen + Dropout")
    print(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE:,}, seq_len={SEQ_LEN}")
    print("=" * 70)
    
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
    
    # Create optimized datasets
    print("📊 Creating optimized datasets...")
    train_loader, val_loader = create_optimized_datasets(
        text=text,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        train_split=0.8
    )
    
    print(f"✅ Dataset ready: {len(train_loader):,} train batches, {len(val_loader):,} val batches")
    print(f"   DataLoader optimizations: num_workers=2, pin_memory=True, persistent_workers=True")
    
    # Create speed-optimized model
    print("🏗️  Building speed-optimized model...")
    model = SpeedOptimizedModel(
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
    
    # Speed Optimization 1: Torch compilation (biggest speedup)
    print("⚡ Compiling model for maximum speed...")
    try:
        compiled_model = torch.compile(model, mode='max-autotune')
        print("✅ Model compiled successfully!")
        model = compiled_model
    except Exception as e:
        print(f"⚠️  Compilation failed: {e}")
        print("   Continuing with uncompiled model...")
    
    # Setup optimized training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Faster scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize trainer with optimized settings
    trainer = RNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        mixed_precision=False,  # MPS doesn't support mixed precision yet
        grad_clip_norm=1.0,
        patience=15,
        checkpoint_dir='checkpoints'
    )
    
    # Run speed-optimized training
    print(f"🚀 Starting speed-optimized training...")
    start_time = time.time()
    
    # Pre-compile first batch (compilation overhead happens here)
    print("⚡ Warming up compiled model...")
    warmup_start = time.time()
    
    try:
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 2:  # Just a few warmup batches
                break
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output[:, -1, :], target)
            loss.backward()
            optimizer.zero_grad()
            
        warmup_time = time.time() - warmup_start
        print(f"✅ Warmup complete ({warmup_time:.2f}s)")
    except Exception as e:
        print(f"⚠️  Warmup failed: {e}")
    
    # Actual training
    print(f"🏁 Starting actual training loop...")
    training_start = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        verbose=True
    )
    
    training_time = time.time() - training_start
    total_time = time.time() - start_time
    
    print(f"\n✅ Speed-optimized training complete!")
    print(f"Total time (including warmup): {total_time:.2f} seconds")
    print(f"Pure training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    
    # Calculate optimized throughput
    total_samples = len(train_loader) * BATCH_SIZE * EPOCHS
    samples_per_second = int(total_samples / training_time)
    
    # Speed Optimization 4: Switch to inference mode for text generation
    print(f"📝 Generating text with inference-optimized model...")
    model.switch_to_inference_mode()  # Remove dropout overhead
    
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
    
    # Save speed-optimized results
    print(f"💾 Saving speed-optimized results...")
    write_standard_results(
        test_name="Speed-Optimized Token RNN",
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
            'optimizations': 'compilation+dataloader+seqlen+dropout',
            'total_time_with_warmup': total_time,
            'warmup_overhead': warmup_time if 'warmup_time' in locals() else 0,
            'generation_time': generation_time
        }
    )
    
    print(f"\n⚡ SPEED-OPTIMIZED TOKEN RNN RESULTS")
    print(f"   Device: {device}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: {NUM_LAYERS} layers × {HIDDEN_SIZE} hidden × {EMBED_DIM} embed")
    print(f"   Optimizations Applied:")
    print(f"     • Torch compilation (mode='max-autotune')")
    print(f"     • DataLoader (workers=2, pin_memory=True)")
    print(f"     • Sequence length: {SEQ_LEN} (power of 2)")
    print(f"     • Dropout removal for inference")
    print(f"   Throughput: {samples_per_second:,} samples/second")
    print(f"   Training time: {training_time:.1f}s")
    print(f"   Speedup vs baseline: TBD (run comparison)")

if __name__ == '__main__':
    main()