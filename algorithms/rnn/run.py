#!/usr/bin/env python3
"""
Unified RNN Entry Point
Single interface for all RNN experiments (character and token level)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="Run RNN experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode character              # Run character-level RNN
  python run.py --mode character --maximal   # Run maximal accuracy character RNN
  python run.py --mode token                 # Run token-level RNN  
  python run.py --mode token --maximal       # Run maximal accuracy token RNN
        """
    )
    
    # Main arguments
    parser.add_argument('--mode', choices=['character', 'token'], required=True,
                       help='RNN mode: character-level or token-level')
    parser.add_argument('--maximal', action='store_true',
                       help='Use maximal accuracy configuration')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Configuration arguments  
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'character':
            if args.maximal:
                print("🚀 Running Character-Level RNN (Maximal Accuracy)")
                from .character.test_maximal_accuracy import main as run_char_maximal
                run_char_maximal()
            elif args.verbose:
                print("🚀 Running Character-Level RNN (Verbose)")
                from .character.test_verbose import main as run_char_verbose
                run_char_verbose()
            else:
                print("🚀 Running Character-Level RNN (Basic)")
                from .character.test_basic import main as run_char_basic
                run_char_basic()
                
        elif args.mode == 'token':
            if args.maximal:
                print("🚀 Running Token-Level RNN (Maximal Accuracy)")
                # Use the proven maximal accuracy implementation
                from .tokens.train import run_maximal_accuracy_training
                run_maximal_accuracy_training()
            else:
                print("🚀 Running Token-Level RNN (Basic)")
                from .tokens.train_token import main as run_token_basic
                run_token_basic()
                
    except ImportError as e:
        print(f"❌ Error importing module: {e}")
        print("   Make sure all required files are present")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()