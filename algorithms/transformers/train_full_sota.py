#!/usr/bin/env python3
"""
Full SOTA Transformer Training - 8M Documents
Memory-efficient training with streaming data loader for maximum performance
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import json
import time
import psutil
import gc
import sys

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_sota_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our components
from full_sota_config import FullSOTAConfig, get_full_sota_config, estimate_training_time
from optimized_transformer import OptimizedSOTATransformer
from data.gpt2_tokenizer import GPT2CompatibleTokenizer
from streaming_data_loader import create_streaming_loaders
from advanced_train import AdvancedTrainer, LearningRateScheduler, DropoutScheduler


class FullSOTATrainer(AdvancedTrainer):
    """
    Extended trainer for full 8M document SOTA training with streaming data.
    
    Enhancements:
    - Progressive sequence length training
    - Memory monitoring and management
    - Robust checkpointing for multi-day training
    - Advanced learning rate scheduling
    """
    
    def __init__(self, config: FullSOTAConfig, model, tokenizer):
        # Convert to base config for compatibility
        from optimized_config import OptimizedSOTAConfig
        base_config = OptimizedSOTAConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            grad_clip_norm=config.grad_clip_norm,
            warmup_steps=config.warmup_steps,
            max_steps=config.total_steps,
            batch_size=config.batch_size,
            gradient_accumulation=config.gradient_accumulation,
            label_smoothing=config.label_smoothing,
            tie_embeddings=config.tie_embeddings,
            use_flash_attn=config.use_flash_attn,
            gradient_checkpointing=config.gradient_checkpointing,
            log_interval=config.log_interval,
            eval_interval=config.eval_interval,
            save_interval=config.save_interval
        )
        
        super().__init__(base_config, model, tokenizer)
        
        # Store full config
        self.full_config = config
        self.current_stage = 1
        self.stage_start_step = 0
        
        # Memory monitoring
        self.process = psutil.Process()
        self.max_memory_gb = 0
        self.memory_warnings = 0
        
        # Multi-stage training state
        self.stage_history = []
        
        logger.info(f"🌍 Full SOTA Trainer initialized for {config.total_steps:,} steps")
        logger.info(f"   Progressive training: {config.curriculum_learning}")
        logger.info(f"   Memory monitoring: Enabled")
    
    def get_current_stage_config(self) -> dict:
        """Get configuration for current training stage"""
        return self.full_config.get_stage_config(self.step)
    
    def check_stage_transition(self) -> bool:
        """Check if we should transition to next stage"""
        stage_config = self.get_current_stage_config()
        
        if stage_config['stage'] != self.current_stage:
            # Stage transition
            old_stage = self.current_stage
            self.current_stage = stage_config['stage']
            
            logger.info(f"🎯 STAGE TRANSITION: {old_stage} → {self.current_stage}")
            logger.info(f"   {stage_config['description']}")
            logger.info(f"   Sequence length: {stage_config['seq_len']}")
            logger.info(f"   Batch size: {stage_config['batch_size']}")
            
            # Record stage history
            self.stage_history.append({
                'stage': old_stage,
                'end_step': self.step,
                'duration_steps': self.step - self.stage_start_step
            })
            
            self.stage_start_step = self.step
            return True
        
        return False
    
    def monitor_memory(self):
        """Monitor and log memory usage"""
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024**3)
        
        if memory_gb > self.max_memory_gb:
            self.max_memory_gb = memory_gb
        
        # Memory warning system
        if memory_gb > 50:  # Warning at 50GB
            self.memory_warnings += 1
            if self.memory_warnings % 10 == 1:  # Log every 10th warning
                logger.warning(f"⚠️  High memory usage: {memory_gb:.1f} GB")
                
                if memory_gb > 58:  # Critical at 58GB (leave 6GB buffer)
                    logger.critical(f"🚨 Critical memory usage: {memory_gb:.1f} GB")
                    logger.critical("   Forcing garbage collection...")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return memory_gb
    
    def save_comprehensive_checkpoint(self, filename: str):
        """Save comprehensive checkpoint with full training state"""
        memory_gb = self.monitor_memory()
        
        checkpoint = {
            # Model and optimizer state
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            
            # Configurations
            'full_config': self.full_config,
            'base_config': self.config,
            
            # Training history
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            
            # Stage information
            'current_stage': self.current_stage,
            'stage_start_step': self.stage_start_step,
            'stage_history': self.stage_history,
            
            # System information
            'max_memory_gb': self.max_memory_gb,
            'memory_warnings': self.memory_warnings,
            'current_memory_gb': memory_gb,
            'timestamp': time.time(),
            
            # Training metadata
            'total_steps': self.full_config.total_steps,
            'progress_pct': (self.step / self.full_config.total_steps) * 100
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"💾 Comprehensive checkpoint saved: {filename}")
        logger.info(f"   Progress: {checkpoint['progress_pct']:.1f}%")
        logger.info(f"   Memory: {memory_gb:.1f} GB (max: {self.max_memory_gb:.1f} GB)")
    
    def train_full_sota(self, data_dir: str):
        """
        Main training loop for full SOTA transformer.
        Uses streaming data loaders for memory efficiency.
        """
        logger.info(f"🚀 STARTING FULL SOTA TRAINING")
        logger.info("=" * 80)
        
        # System information
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"💻 System: {memory_gb:.1f}GB RAM, {self.device}")
        
        # Create streaming data loaders
        logger.info(f"📊 Creating streaming data loaders...")
        
        train_loader, val_loader = create_streaming_loaders(
            data_dir=data_dir,
            seq_len=self.full_config.seq_len_stage1,  # Start with stage 1
            batch_size=self.full_config.stage1_batch_size,
            memory_limit_gb=memory_gb * 0.7  # Use 70% of available memory
        )
        
        # Training estimates
        estimates = estimate_training_time(self.full_config, train_loader.num_sequences)
        logger.info(f"⏰ Training estimates:")
        logger.info(f"   Duration: {estimates['training_time_days']:.1f} days")
        logger.info(f"   Training tokens: {estimates['total_training_tokens']:,}")
        logger.info(f"   Memory needed: {estimates['total_memory_gb']:.1f} GB")
        
        # Confirm training
        logger.info(f"\n🎯 Ready to train {self.full_config.param_count_m:.1f}M parameter model")
        logger.info(f"   Total steps: {self.full_config.total_steps:,}")
        logger.info(f"   Training sequences: {train_loader.num_sequences:,}")
        logger.info(f"   Validation sequences: {val_loader.num_sequences:,}")
        
        start_time = time.time()
        accumulated_loss = 0.0
        last_stage_check = 0
        
        try:
            # Main training loop
            for epoch in range(1000):  # Large number - will stop based on steps
                self.epoch = epoch
                
                for batch_idx, batch in enumerate(train_loader):
                    # Stage management
                    if self.step - last_stage_check >= 1000:  # Check every 1000 steps
                        stage_changed = self.check_stage_transition()
                        if stage_changed:
                            # Recreate data loaders with new sequence length
                            stage_config = self.get_current_stage_config()
                            logger.info(f"🔄 Recreating data loaders for stage {stage_config['stage']}")
                            
                            train_loader, val_loader = create_streaming_loaders(
                                data_dir=data_dir,
                                seq_len=stage_config['seq_len'],
                                batch_size=stage_config['batch_size'],
                                memory_limit_gb=memory_gb * 0.7
                            )
                            break  # Restart epoch with new loaders
                        
                        last_stage_check = self.step
                    
                    # Update learning rate
                    current_lr = self.lr_scheduler.get_lr(self.step)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
                    
                    # Update model configuration
                    self.update_model_config()
                    
                    # Training step
                    loss = self.train_step(batch)
                    accumulated_loss += loss
                    
                    # Gradient accumulation
                    if (self.step + 1) % self.config.gradient_accumulation == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    self.step += 1
                    
                    # Logging
                    if self.step % self.config.log_interval == 0:
                        avg_loss = accumulated_loss / self.config.log_interval
                        elapsed = time.time() - start_time
                        
                        # Memory monitoring
                        memory_gb = self.monitor_memory()
                        
                        # Throughput calculation
                        stage_config = self.get_current_stage_config()
                        total_tokens = self.step * self.config.effective_batch_size * stage_config['seq_len']
                        tokens_per_sec = total_tokens / elapsed
                        
                        # Progress calculation
                        progress_pct = (self.step / self.full_config.total_steps) * 100
                        eta_hours = (self.full_config.total_steps - self.step) * elapsed / (self.step * 3600)
                        
                        logger.info(
                            f"Step {self.step:,}/{self.full_config.total_steps:,} ({progress_pct:.1f}%) | "
                            f"Stage {stage_config['stage']} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Tokens/sec: {tokens_per_sec:.0f} | "
                            f"Memory: {memory_gb:.1f}GB | "
                            f"ETA: {eta_hours:.1f}h"
                        )
                        
                        self.train_losses.append(avg_loss)
                        self.learning_rates.append(current_lr)
                        accumulated_loss = 0.0
                    
                    # Validation
                    if self.step % self.config.eval_interval == 0:
                        val_loss = self.validation_step(val_loader.dataloader)
                        self.val_losses.append(val_loss)
                        
                        perplexity = torch.exp(torch.tensor(val_loss)).item()
                        logger.info(f"📊 Validation | Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_comprehensive_checkpoint("best_full_sota.pt")
                            logger.info(f"💾 New best model (perplexity: {perplexity:.2f})")
                    
                    # Generation sample
                    if self.step % self.config.generate_interval == 0:
                        try:
                            sample = self.generate_sample("The future of artificial intelligence")
                            logger.info(f"📝 Generated: '{sample[:150]}...'")
                        except Exception as e:
                            logger.warning(f"Generation failed: {e}")
                    
                    # Checkpointing
                    if self.step % self.config.save_interval == 0:
                        checkpoint_name = f"checkpoint_step_{self.step:06d}.pt"
                        self.save_comprehensive_checkpoint(checkpoint_name)
                    
                    # Training completion check
                    if self.step >= self.full_config.total_steps:
                        logger.info(f"🏁 Training complete after {self.step:,} steps")
                        return
                
                # Check if we changed stages and need to restart epoch
                if self.check_stage_transition():
                    continue
        
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            logger.info(f"⏹️  Training interrupted at step {self.step:,}")
            logger.info(f"   Time elapsed: {elapsed/3600:.1f} hours")
            self.save_comprehensive_checkpoint("interrupted_full_sota.pt")
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Training failed: {e}")
            logger.info(f"   Time elapsed: {elapsed/3600:.1f} hours")
            self.save_comprehensive_checkpoint("error_full_sota.pt")
            raise
        
        finally:
            # Final checkpoint
            self.save_comprehensive_checkpoint("final_full_sota.pt")
            
            # Training summary
            elapsed = time.time() - start_time
            logger.info(f"✅ TRAINING SESSION COMPLETE")
            logger.info(f"   Total time: {elapsed/3600:.1f} hours")
            logger.info(f"   Final step: {self.step:,}")
            logger.info(f"   Best val loss: {self.best_val_loss:.4f}")
            logger.info(f"   Max memory: {self.max_memory_gb:.1f} GB")
            logger.info(f"   Final stage: {self.current_stage}")


def main():
    """Main launcher for full SOTA training"""
    logger.info("🌍 FULL SOTA TRANSFORMER TRAINING - 8M DOCUMENTS")
    logger.info("=" * 80)
    
    # Check system resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    logger.info(f"💻 System resources:")
    logger.info(f"   Memory: {memory_gb:.1f} GB")
    logger.info(f"   CPU cores: {cpu_count}")
    
    # Check data availability
    data_dir = Path.home() / ".cache" / "openwebtext_gpt2_full"
    
    if not data_dir.exists():
        logger.error("❌ Full OpenWebText dataset not found!")
        logger.error(f"   Expected location: {data_dir}")
        logger.error("   Run: python prepare_full_openwebtext.py")
        return False
    
    # Load configuration
    config = get_full_sota_config()
    config.print_full_config()
    
    # Resource validation
    estimates = estimate_training_time(config, 10_000_000)  # Estimate 10M sequences
    
    if estimates['total_memory_gb'] > memory_gb * 0.9:
        logger.warning(f"⚠️  Memory requirements ({estimates['total_memory_gb']:.1f}GB) near system limit ({memory_gb:.1f}GB)")
        logger.warning("   Consider using get_full_sota_conservative() configuration")
    
    # Confirm training
    logger.info(f"\n🎯 FULL TRAINING SUMMARY:")
    logger.info(f"   Model: {config.param_count_m:.1f}M parameters")
    logger.info(f"   Training time: ~{estimates['training_time_days']:.1f} days")
    logger.info(f"   Memory needed: {estimates['total_memory_gb']:.1f} GB")
    logger.info(f"   Training tokens: {estimates['total_training_tokens']:,}")
    
    confirm = input("\nStart full SOTA training on 8M documents? (y/N): ").strip().lower()
    if confirm != 'y':
        logger.info("Training cancelled.")
        return False
    
    # Create model
    logger.info(f"🏗️  Creating {config.param_count_m:.1f}M parameter model...")
    model = OptimizedSOTATransformer(config)
    
    # Model compilation for speed
    if config.compile_model and hasattr(torch, 'compile'):
        logger.info("⚡ Compiling model with PyTorch 2.0...")
        model = torch.compile(model)
    
    # Move to device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load tokenizer
    logger.info(f"🔤 Loading tokenizer...")
    tokenizer = GPT2CompatibleTokenizer.load(str(data_dir / "tokenizer"))
    
    # Create trainer
    logger.info(f"🎯 Creating full SOTA trainer...")
    trainer = FullSOTATrainer(config, model, tokenizer)
    
    # Save initial configuration
    with open("full_sota_config.json", 'w') as f:
        config_dict = {
            "model_params": config.total_params,
            "training_days_estimate": estimates['training_time_days'],
            "total_steps": config.total_steps,
            "memory_estimate_gb": estimates['total_memory_gb'],
            "device": str(device),
            "timestamp": time.time()
        }
        json.dump(config_dict, f, indent=2)
    
    # Start training
    logger.info(f"🚀 LAUNCHING FULL SOTA TRAINING...")
    trainer.train_full_sota(str(data_dir))
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)