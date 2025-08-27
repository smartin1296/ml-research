#!/usr/bin/env python3
"""
ML Environment - Main Entry Point
Unified interface for all machine learning algorithms and experiments
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="ML Environment - Run neural network experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm Examples:
  python run.py rnn --mode character         # Character-level RNN
  python run.py rnn --mode token --maximal  # Maximal accuracy token RNN
  python run.py cnn --train                 # Train CNN on CIFAR-10
  python run.py transformers --phase 1      # Baseline transformer
  python run.py transformers --comparison   # Phase 1 vs 2 comparison
  
Status and Help:
  python run.py --status                    # Show implementation status
  python run.py --help                      # Show this help
        """
    )
    
    # Status and info
    parser.add_argument('--status', action='store_true',
                       help='Show current implementation status')
    
    # Algorithm selection (optional if using --status)
    parser.add_argument('algorithm', nargs='?',
                       choices=['rnn', 'cnn', 'transformers'],
                       help='Choose algorithm to run')
    
    # Parse known args to pass remaining to algorithm
    args, remaining_args = parser.parse_known_args()
    
    if args.status:
        show_status()
        return
    
    if not args.algorithm:
        parser.error("Must specify algorithm or use --status")
    
    # Import and run specific algorithm
    try:
        if args.algorithm == 'rnn':
            from algorithms.rnn.run import main as rnn_main
            sys.argv = ['run.py'] + remaining_args  # Reset argv for subparser
            rnn_main()
            
        elif args.algorithm == 'cnn':
            from algorithms.cnn.run import main as cnn_main
            sys.argv = ['run.py'] + remaining_args
            cnn_main()
            
        elif args.algorithm == 'transformers':
            from algorithms.transformers.run import main as transformer_main
            sys.argv = ['run.py'] + remaining_args
            transformer_main()
            
    except ImportError as e:
        print(f"❌ Error importing {args.algorithm}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running {args.algorithm}: {e}")
        sys.exit(1)

def show_status():
    """Display current implementation status"""
    print("🧠 ML Environment - Implementation Status")
    print("=" * 50)
    
    status = {
        "RNN": {
            "status": "✅ COMPLETED",
            "details": [
                "Character-level: 5,720 samples/sec (M1 Max optimized)",
                "Token-level: 24,121 samples/sec with 39%+ accuracy",
                "Maximal accuracy training with advanced scheduling"
            ],
            "entry": "python run.py rnn --mode character|token"
        },
        "CNN": {
            "status": "✅ COMPLETED", 
            "details": [
                "CIFAR-10: 86.15% validation accuracy",
                "Intelligent stopping criteria",
                "M1 Max optimized training"
            ],
            "entry": "python run.py cnn --train"
        },
        "Transformers": {
            "status": "✅ COMPLETED - DEBUGGED",
            "details": [
                "Phase 1: 99.9% validation accuracy (fixed)",
                "Phase 2: Advanced optimizations working correctly", 
                "Fair comparison framework established"
            ],
            "entry": "python run.py transformers --phase 1|2"
        },
        "Reasoning NNs": {
            "status": "🔄 IN PROGRESS",
            "details": ["Test-time compute models (DeepSeek-R1 style)"],
            "entry": "Not yet available"
        },
        "RCL Algorithm": {
            "status": "📋 PLANNED",
            "details": ["Proprietary algorithm implementation"],
            "entry": "Not yet available"
        }
    }
    
    for name, info in status.items():
        print(f"\n{info['status']} {name}")
        for detail in info['details']:
            print(f"  • {detail}")
        print(f"  🚀 Usage: {info['entry']}")
    
    print(f"\n📁 Results stored in: results/")
    print(f"📚 Documentation: CLAUDE.md")

if __name__ == "__main__":
    main()