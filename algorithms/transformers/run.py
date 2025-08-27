#!/usr/bin/env python3
"""
Unified Transformer Entry Point
Single interface for all transformer experiments and comparisons
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="Run Transformer experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --phase 1                    # Run Phase 1 baseline
  python run.py --phase 2                    # Run Phase 2 optimizations  
  python run.py --comparison                 # Run Phase 1 vs 2 comparison
  python run.py --test                       # Run basic functionality test
  python run.py --story-generation           # Test story generation
        """
    )
    
    # Main action arguments
    parser.add_argument('--phase', type=int, choices=[1, 2],
                       help='Run specific phase (1=baseline, 2=optimizations)')
    parser.add_argument('--comparison', action='store_true',
                       help='Run Phase 1 vs Phase 2 comparison')
    parser.add_argument('--test', action='store_true',
                       help='Run basic functionality test')
    parser.add_argument('--story-generation', action='store_true',
                       help='Test story generation capabilities')
    
    # Configuration arguments
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--subset-size', type=int, default=5000,
                       help='Dataset subset size (default: 5000)')
    
    args = parser.parse_args()
    
    # Validate arguments
    actions = [args.phase is not None, args.comparison, args.test, args.story_generation]
    if sum(actions) != 1:
        parser.error("Must specify exactly one action: --phase, --comparison, --test, or --story-generation")
    
    try:
        if args.phase == 1:
            print("🚀 Running Phase 1 (Baseline Transformer)")
            from .phase1_standard import run_phase1_standard
            run_phase1_standard()
            
        elif args.phase == 2:
            print("🚀 Running Phase 2 (Optimized Transformer)")
            from .phase2_standard import run_phase2_standard
            run_phase2_standard()
            
        elif args.comparison:
            print("📊 Running Phase 1 vs Phase 2 Comparison")
            from .standard_phase_comparison import main as run_comparison
            run_comparison()
            
        elif args.test:
            print("🧪 Running Basic Functionality Test")
            from .test_basic import main as run_test
            run_test()
            
        elif args.story_generation:
            print("📖 Testing Story Generation")
            from .test_story_generation import main as run_story_test
            run_story_test()
            
    except ImportError as e:
        print(f"❌ Error importing module: {e}")
        print("   Make sure all required files are present")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()