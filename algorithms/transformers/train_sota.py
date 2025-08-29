#!/usr/bin/env python3
"""
SOTA Transformer Training Launcher
Ready-to-use script for training a 125M parameter transformer on OpenWebText
"""

import torch
import logging
from pathlib import Path
import json
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sota_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our components
from optimized_config import get_config_125m, get_config_debug
from optimized_transformer import OptimizedSOTATransformer
from data.gpt2_tokenizer import GPT2CompatibleTokenizer
from advanced_train import AdvancedTrainer, create_data_loaders


def main():
    """Main training launcher"""
    logger.info("🚀 SOTA TRANSFORMER TRAINING LAUNCHER")
    logger.info("=" * 80)
    
    # Check system
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"💻 Device: {device}")
    
    if device.type == "cpu":
        logger.warning("⚠️  Running on CPU - training will be slow!")
    
    # Configuration choice
    print("\nChoose your configuration:")
    print("1. Quick Test (16M params, 1K steps) - ~5 minutes")
    print("2. Full Training (125M params, 100K steps) - ~12-24 hours")
    print("3. Custom config")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        logger.info("🧪 Starting quick test training...")
        config = get_config_debug()
        config.seq_len_start = 512
        config.seq_len_end = 512
        config.max_seq_len = 512
        config.max_steps = 1000
        config.eval_interval = 100
        config.log_interval = 20
    elif choice == "2":
        logger.info("🎯 Starting full SOTA training...")
        config = get_config_125m()
    elif choice == "3":
        print("Custom configuration not implemented yet. Using quick test.")
        config = get_config_debug()
        config.seq_len_start = 512
        config.seq_len_end = 512
        config.max_seq_len = 512
    else:
        logger.info("Invalid choice. Using quick test configuration.")
        config = get_config_debug()
        config.seq_len_start = 512
        config.seq_len_end = 512
        config.max_seq_len = 512
    
    # Print configuration
    config.print_config()
    
    # Verify data exists
    cache_dir = Path.home() / ".cache" / "openwebtext_gpt2_parallel"
    if not cache_dir.exists():
        logger.error("❌ Parallel-processed data not found!")
        logger.error("   Please run: python parallel_prepare_openwebtext.py")
        return False
    
    logger.info(f"✅ Data found: {cache_dir}")
    
    # Create model
    logger.info(f"\n🏗️  Creating model...")
    model = OptimizedSOTATransformer(config).to(device)
    
    # Load tokenizer
    logger.info(f"\n🔤 Loading tokenizer...")
    tokenizer = GPT2CompatibleTokenizer.load(str(cache_dir / "tokenizer"))
    
    # Create data loaders
    logger.info(f"\n📁 Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    # Calculate training time estimate
    estimated_hours = (config.max_steps * config.effective_batch_size) / (3600 * 300)  # Rough estimate
    logger.info(f"⏰ Estimated training time: ~{estimated_hours:.1f} hours")
    
    # Confirm training
    if choice == "2":
        confirm = input(f"\nThis will train for ~{estimated_hours:.1f} hours. Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            logger.info("Training cancelled.")
            return False
    
    # Create trainer
    logger.info(f"\n🎯 Creating advanced trainer...")
    trainer = AdvancedTrainer(config, model, tokenizer)
    
    # Save configuration
    with open("training_config.json", 'w') as f:
        config_dict = {
            "model_params": config.total_params,
            "effective_batch_size": config.effective_batch_size,
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "vocab_size": config.vocab_size,
            "device": str(device)
        }
        json.dump(config_dict, f, indent=2)
    logger.info("💾 Configuration saved to training_config.json")
    
    # Start training
    logger.info(f"\n🚀 STARTING TRAINING...")
    logger.info(f"📊 Progress will be logged to sota_training.log")
    logger.info(f"🔥 Press Ctrl+C to stop training and save checkpoint")
    
    start_time = time.time()
    
    try:
        trainer.train(train_loader, val_loader)
        
        # Training completed successfully
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 TRAINING COMPLETED!")
        logger.info(f"   Total time: {elapsed/3600:.2f} hours")
        logger.info(f"   Final step: {trainer.step:,}")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        logger.info(f"\n⏹️  TRAINING INTERRUPTED")
        logger.info(f"   Time elapsed: {elapsed/3600:.2f} hours") 
        logger.info(f"   Step reached: {trainer.step:,}")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        logger.info(f"   Time elapsed: {elapsed/3600:.2f} hours")
        trainer.save_checkpoint("error_checkpoint.pt")
        
    finally:
        # Always save final checkpoint
        trainer.save_checkpoint("final_checkpoint.pt")
        
        # Test generation
        if trainer.step > 100:  # Only if we trained for a bit
            logger.info(f"\n📝 Testing final generation...")
            try:
                sample = trainer.generate_sample("The future of artificial intelligence is")
                logger.info(f"Generated: '{sample[:200]}...'")
            except Exception as e:
                logger.error(f"Generation test failed: {e}")
        
        logger.info(f"\n✅ Training session complete!")
        logger.info(f"📁 Checkpoints saved in current directory")
        logger.info(f"📊 Full log available in sota_training.log")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)