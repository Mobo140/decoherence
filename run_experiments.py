#!/usr/bin/env python3
"""Script to run experiments with different models and compare results."""
import subprocess
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime


def run_experiment(config_path: str, model_type: str, output_dir: Path):
    """Run training and evaluation for a single model.
    
    Args:
        config_path: Path to config file
        model_type: Model type to test
        output_dir: Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_type}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    
    # Create model-specific directories
    model_output_dir = output_dir / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    print(f"Training {model_type}...")
    train_cmd = [
        "python3", "-m", "src.train.train",
        "--config", config_path,
        "--model", model_type
    ]
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for {model_type}:")
        print(e.stdout)
        print(e.stderr)
        return None
    
    # Find the best checkpoint (model-specific directory)
    checkpoint_dir = Path("checkpoints") / model_type
    checkpoints_list = sorted(checkpoint_dir.glob("checkpoint_*"))
    if not checkpoints_list:
        print(f"No checkpoints found for {model_type} in {checkpoint_dir}")
        return None
    
    best_checkpoint = checkpoints_list[-1]  # Latest checkpoint
    print(f"Using checkpoint: {best_checkpoint}")
    
    # Run evaluation
    print(f"\nEvaluating {model_type}...")
    eval_cmd = [
        "python3", "-m", "src.train.eval",
        "--config", config_path,
        "--checkpoint", str(best_checkpoint)
    ]
    
    try:
        result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {model_type}:")
        print(e.stdout)
        print(e.stderr)
        return None
    
    # Results are already saved in model-specific directories (figures/{model_type}/)
    # So we don't need to copy them - they're already organized
    figures_dir = Path("figures") / model_type
    if figures_dir.exists():
        print(f"✓ Figures saved to: {figures_dir}")
    
    runs_dir = Path("runs") / model_type
    if runs_dir.exists():
        print(f"✓ Training history saved to: {runs_dir}")
    
    checkpoints_dir = Path("checkpoints") / model_type
    if checkpoints_dir.exists():
        print(f"✓ Checkpoints saved to: {checkpoints_dir}")
    
    return {
        'model': model_type,
        'checkpoint': str(best_checkpoint),
        'status': 'completed'
    }


def main():
    parser = argparse.ArgumentParser(description="Run experiments with multiple models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment config file")
    parser.add_argument("--models", type=str, nargs="+",
                       default=["mlp", "gru", "lstm", "transformer"],
                       help="List of models to test")
    parser.add_argument("--output-dir", type=str, default="experiments",
                       help="Output directory for results")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training, only evaluate existing checkpoints")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for model_type in args.models:
        if args.skip_training:
            # Just evaluate
            checkpoint_dir = Path("checkpoints")
            checkpoints_list = sorted(checkpoint_dir.glob("checkpoint_*"))
            if checkpoints_list:
                best_checkpoint = checkpoints_list[-1]
                eval_cmd = [
                    "python3", "-m", "src.train.eval",
                    "--config", str(config_path),
                    "--checkpoint", str(best_checkpoint)
                ]
                subprocess.run(eval_cmd)
        else:
            result = run_experiment(str(config_path), model_type, output_dir)
            if result:
                results.append(result)
    
    # Save summary
    summary_path = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'config': str(config_path),
            'models_tested': args.models,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Experiments completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
