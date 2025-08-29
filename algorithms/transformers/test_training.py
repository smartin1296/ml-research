#!/usr/bin/env python3
"""
Test the complete training pipeline with a small model
"""

import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our components
from optimized_config import get_config_debug
from optimized_transformer import OptimizedSOTATransformer
from data.gpt2_tokenizer import GPT2CompatibleTokenizer
from parallel_prepare_openwebtext import FastGPT2Dataset, create_fast_data_loaders


def test_complete_pipeline():
    """Test the complete training pipeline"""
    logger.info("🧪 TESTING COMPLETE TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # 1. Test configuration
    config = get_config_debug()
    # Fix sequence length to match our processed data
    config.seq_len_start = 512
    config.seq_len_end = 512
    config.max_seq_len = 512
    config.print_config()
    
    # 2. Test model creation
    logger.info(f"\n🏗️  Testing model creation...")
    model = OptimizedSOTATransformer(config)
    
    # 3. Test tokenizer
    logger.info(f"\n🔤 Testing tokenizer...")
    cache_dir = str(Path.home() / ".cache" / "openwebtext_gpt2_parallel")
    tokenizer = GPT2CompatibleTokenizer.load(str(Path(cache_dir) / "tokenizer"))
    
    # 4. Test data loading
    logger.info(f"\n📁 Testing data loading...")
    try:
        train_loader, val_loader = create_fast_data_loaders(
            cache_dir, 
            seq_len=config.seq_len_start, 
            batch_size=config.batch_size
        )
        logger.info(f"✅ Data loaders created successfully")
        
        # Test a batch
        batch = next(iter(train_loader))
        input_ids, target_ids = batch
        logger.info(f"✅ Batch shapes: input {input_ids.shape}, target {target_ids.shape}")
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False
    
    # 5. Test forward pass
    logger.info(f"\n🔬 Testing forward pass...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    try:
        with torch.no_grad():
            logits = model(input_ids)
        logger.info(f"✅ Forward pass successful: {logits.shape}")
        
        # Test loss calculation
        target_ids = target_ids.to(device)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        logger.info(f"✅ Loss calculation: {loss.item():.4f}")
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        return False
    
    # 6. Test generation
    logger.info(f"\n📝 Testing generation...")
    try:
        prompt_text = "The future of artificial intelligence"
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_tensor = torch.tensor([prompt_tokens], device=device)
        
        generated = model.generate_optimized(
            prompt_tensor, 
            max_new_tokens=20, 
            temperature=0.8
        )
        
        generated_text = tokenizer.decode(generated[0].tolist(), skip_special=True)
        logger.info(f"✅ Generated text: '{generated_text[:100]}...'")
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False
    
    logger.info(f"\n🎉 ALL TESTS PASSED!")
    logger.info(f"✅ Training pipeline is ready")
    return True


if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print(f"\n🚀 Ready to start full training!")
    else:
        print(f"\n❌ Pipeline test failed - check logs")