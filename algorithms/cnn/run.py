#!/usr/bin/env python3
"""
Unified CNN Entry Point
Single interface for all CNN experiments and training
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="Run CNN experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --train                      # Train CNN with intelligent stopping
  python run.py --resume                     # Resume training from checkpoint
  python run.py --dataset cifar10            # Train on CIFAR-10 (default)
  python run.py --train --model resnet       # Train ResNet variant
        """
    )
    
    # Main actions
    parser.add_argument('--train', action='store_true',
                       help='Train CNN model')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--test', action='store_true',
                       help='Run basic functionality test')
    
    # Configuration arguments
    parser.add_argument('--dataset', choices=['cifar10'], default='cifar10',
                       help='Dataset to use (default: cifar10)')
    parser.add_argument('--model', choices=['simple', 'resnet'], default='simple',
                       help='Model architecture (default: simple)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs (default: 100, uses intelligent stopping)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    
    args = parser.parse_args()
    
    # Validate arguments
    actions = [args.train, args.resume, args.test]
    if sum(actions) != 1:
        parser.error("Must specify exactly one action: --train, --resume, or --test")
    
    try:
        if args.train:
            print(f"🚀 Training CNN on {args.dataset.upper()} with {args.model} architecture")
            from .train_intelligent import main as run_train
            run_train()
            
        elif args.resume:
            print("🔄 Resuming CNN training from checkpoint")
            from .resume_intelligent import main as run_resume
            run_resume()
            
        elif args.test:
            print("🧪 Running basic CNN functionality test")
            from .archive.tests.test_basic import main as run_test
            run_test()
            
    except ImportError as e:
        print(f"❌ Error importing module: {e}")
        print("   Make sure all required files are present")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()