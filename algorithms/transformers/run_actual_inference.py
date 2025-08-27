#!/usr/bin/env python3
"""
Run actual inference from saved Phase 1 checkpoint and save proper results
"""

import sys
import torch
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.transformers.scaled_tokenizer import ScaledWordTokenizer
from algorithms.transformers.test_basic import SimpleTransformer

def run_actual_inference():
    # Setup paths
    results_dir = Path("algorithms/transformers/results/phases/phase_1_scaled_baseline_2017_20250826_171355")
    checkpoint_path = results_dir / "checkpoints/best_checkpoint.pt"
    
    print("Loading checkpoint and rebuilding tokenizer...")
    
    # Rebuild tokenizer (since it wasn't saved)
    data_path = Path("data/raw/text/tinystories/TinyStories-small.txt")
    vocab_stories = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            if line.strip() and len(line.strip()) > 50:
                vocab_stories.append(line.strip())

    tokenizer = ScaledWordTokenizer(vocab_size=8192)
    tokenizer.build_vocab(vocab_stories)

    # Load the actual trained model
    model = SimpleTransformer(vocab_size=8192, d_model=256, num_heads=8, num_layers=4, d_ff=1024, max_seq_len=128)
    checkpoint = torch.load(checkpoint_path, map_location='mps')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('mps')
    model.to(device)

    # ACTUAL INFERENCE - Generate stories from trained model
    prompts = ["Once upon a time", "There was a little", "The brave princess", "In a magical forest", "A small cat"]
    generated_stories = []

    print("Running actual inference on trained model...")
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt, add_special=False)
        if len(prompt_tokens) == 0:
            prompt_tokens = [tokenizer.bos_token]
            
        generated_tokens = prompt_tokens.copy()
        
        # Actual model inference
        for _ in range(30):
            if len(generated_tokens) >= 120:
                break
                
            input_tensor = torch.tensor([generated_tokens], device=device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs[0, -1, :]
                
                # Temperature sampling
                logits = logits / 0.8
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == tokenizer.eos_token:
                    break
                    
                generated_tokens.append(next_token)
        
        story = tokenizer.decode(generated_tokens, skip_special=True)
        generated_stories.append({"prompt": prompt, "story": story})
        print(f"Generated: {prompt} -> {story[:80]}...")

    # Save results with ACTUAL inference
    results_data = {
        "phase": "1_scaled_baseline_2017",
        "model_description": "Full 2017 Attention is All You Need baseline with M1 Max optimization",
        "training_results": {
            "epochs_trained": checkpoint["epoch"] + 1,
            "training_time_seconds": 151.9,
            "best_val_loss": float(checkpoint["best_val_loss"]),
            "best_val_accuracy": float(checkpoint["best_val_accuracy"]),
            "tokens_per_second": 83352
        },
        "model_config": {
            "parameters": 7358464,
            "d_model": 256,
            "num_layers": 4,
            "num_heads": 8,
            "d_ff": 1024,
            "vocab_size": 8192,
            "max_seq_len": 128,
            "batch_size": 512
        },
        "generated_stories": generated_stories,
        "hardware": "M1 Max optimized settings",
        "inference_method": "Actual model inference with temperature sampling"
    }

    # Save results
    results_file = results_dir / "phase1_complete_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Save tokenizer
    tokenizer.save(str(results_dir / "tokenizer.json"))

    # Create markdown report
    report_file = results_dir / "phase1_report.md"
    with open(report_file, 'w') as f:
        f.write("# Phase 1 Transformer Results\n\n")
        f.write("## Training Summary\n")
        f.write(f"- **Epochs**: {checkpoint['epoch'] + 1}\n")
        f.write(f"- **Training Time**: 151.9 seconds\n")
        f.write(f"- **Best Validation Loss**: {checkpoint['best_val_loss']:.4f}\n")
        f.write(f"- **Tokens/sec**: 83,352\n")
        f.write(f"- **Parameters**: 7,358,464\n\n")
        f.write("## Model Architecture\n")
        f.write("- **d_model**: 256\n")
        f.write("- **Layers**: 4\n")
        f.write("- **Heads**: 8\n") 
        f.write("- **Vocabulary**: 8,192 words\n\n")
        f.write("## Generated Stories (Actual Model Inference)\n\n")
        
        for i, story_data in enumerate(generated_stories, 1):
            f.write(f"### Story {i}: \"{story_data['prompt']}\"\n")
            f.write(f"{story_data['story']}\n\n")

    print(f"\nRESULTS SAVED TO: {results_file}")
    print(f"REPORT SAVED TO: {report_file}")
    print(f"TOKENIZER SAVED TO: {results_dir / 'tokenizer.json'}")
    print("\nACTUAL GENERATED STORIES FROM TRAINED MODEL:")
    for i, story_data in enumerate(generated_stories, 1):
        print(f"{i}: {story_data['prompt']} -> {story_data['story']}")

if __name__ == "__main__":
    run_actual_inference()