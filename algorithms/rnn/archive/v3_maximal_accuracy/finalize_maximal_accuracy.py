#!/usr/bin/env python3
"""
Finalize Maximal Accuracy Training - Run final validation and save results
"""

import torch
import torch.nn as nn
import json
import os
import time
from pathlib import Path
from core.trainer import RNNTrainer
from tokens.models import OptimizedTokenRNNModel
from tokens.tokenizers import BPETokenizer
from tokens.dataset import create_token_datasets, load_shakespeare_data, TokenTextGenerator
from core.device_utils import get_best_device
from core.results_utils import write_standard_results

def main():
    """Finalize the maximal accuracy model with proper validation and saving"""
    
    device = get_best_device()
    print(f"🎯 Finalizing Maximal Accuracy Token RNN")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load the checkpoint to get model configuration
    checkpoint_path = Path('checkpoints/latest_model.pt')
    if not checkpoint_path.exists():
        print(f"❌ No checkpoint found at {checkpoint_path}")
        return
        
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct the model (same config as training)
    VOCAB_SIZE = 500
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3  
    EMBED_DIM = 384
    
    print("🏗️  Reconstructing model...")
    model = OptimizedTokenRNNModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS,
        dropout=0.2,
        layer_norm=True,
        tie_weights=True
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Model loaded: {param_count:,} parameters")
    print(f"   Training accuracy achieved: {checkpoint.get('best_train_accuracy', 0):.4f}")
    print(f"   Epochs trained: {checkpoint.get('epoch', 0)}")
    
    # Load dataset for validation
    print("📊 Loading dataset for final validation...")
    text = load_shakespeare_data()
    
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, min_freq=2)
    tokenizer.fit(text)
    
    train_loader, val_loader = create_token_datasets(
        text=text,
        tokenizer=tokenizer,
        seq_len=15,  # Same as training
        stride=1,
        train_split=0.8,
        batch_size=4096,  # Same as training
        num_workers=0,
    )
    
    # Run final validation
    print("🧪 Running final validation...")
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            
            if output.dim() == 3:
                output = output[:, -1, :]
                
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total_samples += target.size(0)
    
    final_val_loss = total_loss / len(val_loader)
    final_val_accuracy = correct / total_samples
    
    print(f"✅ Final validation complete!")
    print(f"   Validation loss: {final_val_loss:.6f}")
    print(f"   Validation accuracy: {final_val_accuracy:.4f}")
    
    # Generate high-quality text samples
    print(f"📝 Generating final text samples...")
    generator = TokenTextGenerator(model, tokenizer, device)
    
    prompts = [
        "To be or not to be",
        "All the world's a stage",
        "Romeo, Romeo, wherefore art thou Romeo",
        "Friends, Romans, countrymen",
        "Now is the winter of our discontent",
    ]
    
    text_samples = []
    for i, prompt in enumerate(prompts):
        generated = generator.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            top_k=50,
        )
        text_samples.append((f"sample_{i}", f"Prompt: {prompt}\nGenerated: {generated}"))
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:150]}...")  # Show first 150 chars
    
    # Save final model with complete metadata
    print(f"💾 Saving final maximal accuracy model...")
    final_model_path = 'checkpoints/tokens/maximal_accuracy_final.pt'
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': VOCAB_SIZE,
            'embed_dim': EMBED_DIM,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': 0.2,
            'layer_norm': True,
            'tie_weights': True
        },
        'training_results': {
            'epochs_trained': checkpoint.get('epoch', 0),
            'best_train_accuracy': checkpoint.get('best_train_accuracy', 0),
            'final_val_loss': final_val_loss,
            'final_val_accuracy': final_val_accuracy,
            'model_params': param_count
        },
        'tokenizer_config': {
            'type': 'bpe',
            'vocab_size': VOCAB_SIZE,
            'min_freq': 2
        },
        'training_config': {
            'batch_size': 4096,
            'seq_len': 15,
            'architecture': f"{NUM_LAYERS}L-{HIDDEN_SIZE}H-{EMBED_DIM}E",
            'optimization_target': 'maximal_accuracy'
        }
    }, final_model_path)
    
    print(f"✅ Final model saved to {final_model_path}")
    
    # Save tokenizer
    tokenizer_path = 'checkpoints/tokens/tokenizer_maximal_accuracy_final.json'
    tokenizer.save(Path(tokenizer_path))
    print(f"✅ Tokenizer saved to {tokenizer_path}")
    
    # Calculate final throughput estimate
    total_samples_trained = checkpoint.get('epoch', 0) * len(train_loader) * 4096
    
    # Save comprehensive results
    print(f"💾 Saving comprehensive results...")
    write_standard_results(
        test_name="Maximal Accuracy Token RNN - Final",
        model=model,
        device=device,
        training_time=0,  # Not applicable for validation-only run
        final_loss=final_val_loss,
        samples_per_second=0,  # Not applicable
        batch_size=4096,
        generated_samples=text_samples,
        additional_metrics={
            'epochs_trained': checkpoint.get('epoch', 0),
            'sequence_length': 15,
            'vocab_size': VOCAB_SIZE,
            'best_train_accuracy': checkpoint.get('best_train_accuracy', 0),
            'final_val_accuracy': final_val_accuracy,
            'final_val_loss': final_val_loss,
            'batches_per_epoch': len(train_loader),
            'tokenizer_type': 'bpe',
            'model_architecture': f"{NUM_LAYERS}L-{HIDDEN_SIZE}H-{EMBED_DIM}E",
            'optimization_target': 'maximal_accuracy',
            'total_samples_trained': total_samples_trained
        }
    )
    
    print(f"\n🏆 MAXIMAL ACCURACY TOKEN RNN - FINAL RESULTS")
    print(f"   Device: {device}")
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: {NUM_LAYERS} layers × {HIDDEN_SIZE} hidden × {EMBED_DIM} embed")
    print(f"   Vocabulary: {VOCAB_SIZE}")
    print(f"   Epochs trained: {checkpoint.get('epoch', 0)}")
    print(f"   Best training accuracy: {checkpoint.get('best_train_accuracy', 0):.4f}")
    print(f"   Final validation accuracy: {final_val_accuracy:.4f}")
    print(f"   Final validation loss: {final_val_loss:.6f}")
    print(f"   Model saved: {final_model_path}")
    print(f"   Status: COMPLETED ✅")

if __name__ == '__main__':
    main()