"""Unified interface for training and comparing multiple models."""
import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config, parse_args_with_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.splits import load_splits, create_cv_folds, save_cv_folds
from src.train.train import main as train_main
from src.train.cross_validation import run_cross_validation


def load_full_dataset(data_path: Path) -> Dict:
    """Load full dataset (before splitting).
    
    Args:
        data_path: Path to dataset file
        
    Returns:
        Dictionary with full dataset
    """
    data = np.load(data_path, allow_pickle=True)
    
    # Combine all splits into full dataset
    def extract_obj_array(arr):
        if arr.dtype == object:
            if arr.size == 1:
                return arr.item()
            else:
                return [arr[i] for i in range(len(arr))]
        return arr.tolist()
    
    # Combine train, val, test
    all_observables = []
    all_times = []
    all_t_decoh = []
    all_gamma = []
    
    for split in ['train', 'val', 'test']:
        obs_key = f'{split}_observables'
        times_key = f'{split}_times'
        t_decoh_key = f'{split}_t_decoh'
        gamma_key = f'{split}_gamma'
        
        if obs_key in data:
            all_observables.extend(extract_obj_array(data[obs_key]))
            all_times.extend(extract_obj_array(data[times_key]))
            all_t_decoh = np.concatenate([all_t_decoh, data[t_decoh_key]])
            all_gamma.extend(extract_obj_array(data[gamma_key]))
    
    return {
        'observables': all_observables,
        'times': all_times,
        't_decoh': all_t_decoh,
        'gamma': all_gamma
    }


def run_holdout_experiment(
    config: Dict,
    model_type: str,
    seed: int,
    output_dir: Path,
    logger
) -> Dict:
    """Run hold-out validation experiment.
    
    Args:
        config: Configuration dictionary
        model_type: Model type
        seed: Random seed
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(f"Running hold-out experiment for {model_type}")
    
    # Use existing train.py main function
    # This will use train/val/test splits from dataset
    # We'll call it programmatically
    
    # For now, we'll use the existing train.py script
    # In a full implementation, we'd refactor train.py to be callable
    
    # Create experiment directory
    exp_dir = output_dir / model_type / f'seed_{seed}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Note: In a full implementation, we'd call train_main() here
    # For now, we'll return a placeholder structure
    return {
        'model_type': model_type,
        'seed': seed,
        'method': 'holdout',
        'output_dir': str(exp_dir)
    }


def run_cv_experiment(
    config: Dict,
    model_type: str,
    n_folds: int,
    seed: int,
    output_dir: Path,
    logger
) -> Dict:
    """Run cross-validation experiment.
    
    Args:
        config: Configuration dictionary
        model_type: Model type
        n_folds: Number of folds
        seed: Random seed
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dictionary with CV results
    """
    logger.info(f"Running {n_folds}-fold CV experiment for {model_type}")
    
    # Load full dataset
    data_path = config.get('data', {}).get('dataset_path', 'data/dataset.npz')
    all_data = load_full_dataset(Path(data_path))
    
    # Extract features from config or dataset
    data = np.load(data_path, allow_pickle=True)
    features_arr = data['features']
    if features_arr.dtype == object:
        if features_arr.size == 1:
            features = [features_arr.item()]
        else:
            features = features_arr.tolist()
    else:
        features = features_arr.tolist()
    
    if isinstance(features, str):
        features = [features]
    
    config['features'] = features
    
    # Run CV
    cv_output_dir = output_dir / model_type
    cv_results = run_cross_validation(
        all_data=all_data,
        config=config,
        model_type=model_type,
        n_folds=n_folds,
        seed=seed,
        groups=None,  # Can be extended
        output_dir=cv_output_dir,
        logger=logger
    )
    
    return cv_results


def main():
    """Main function for running model comparison suite."""
    parser = argparse.ArgumentParser(description='Run model comparison suite')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['mlp', 'gru', 'lstm', 'transformer'],
                       help='List of models to compare')
    parser.add_argument('--method', type=str, choices=['holdout', 'cv', 'both'],
                       default='holdout', help='Validation method')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory for experiments')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    seed = args.seed if args.seed else config.get('data', {}).get('seed', 42)
    set_seed(seed)
    logger = setup_logging()
    
    logger.info(f"Starting model comparison suite")
    logger.info(f"Config: {args.config}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Seed: {seed}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment name
    exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    exp_config = {
        'config_path': args.config,
        'models': args.models,
        'method': args.method,
        'n_folds': args.n_folds if args.method in ['cv', 'both'] else None,
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(exp_dir / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    all_results = {}
    
    # Run experiments for each model
    for model_type in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model: {model_type}")
        logger.info(f"{'='*60}")
        
        model_results = {}
        
        if args.method in ['holdout', 'both']:
            result = run_holdout_experiment(
                config=config,
                model_type=model_type,
                seed=seed,
                output_dir=exp_dir,
                logger=logger
            )
            model_results['holdout'] = result
        
        if args.method in ['cv', 'both']:
            result = run_cv_experiment(
                config=config,
                model_type=model_type,
                n_folds=args.n_folds,
                seed=seed,
                output_dir=exp_dir,
                logger=logger
            )
            model_results['cv'] = result
        
        all_results[model_type] = model_results
    
    # Save summary
    summary_path = exp_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info("Experiment suite completed!")
    logger.info(f"Results saved to: {exp_dir}")
    
    # Generate report automatically
    try:
        logger.info("\nGenerating comparison report...")
        from src.reports.make_report import generate_metrics_table, generate_latex_table, generate_markdown_report
        
        report_output_dir = exp_dir / 'report'
        report_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results for report generation
        results_for_report = {}
        for model_type in args.models:
            model_results = {}
            
            # Try to load CV results
            cv_results_path = exp_dir / model_type / 'cv_results.json'
            if cv_results_path.exists():
                with open(cv_results_path, 'r') as f:
                    model_results['cv'] = json.load(f)
            
            if model_results:
                results_for_report[model_type] = model_results
        
        if results_for_report:
            # Generate metrics table
            df = generate_metrics_table(
                results_for_report,
                report_output_dir,
                primary_metric='mae'
            )
            
            # Generate LaTeX table
            latex_path = report_output_dir / 'metrics_summary.tex'
            generate_latex_table(df, latex_path)
            
            # Generate markdown report
            markdown_path = report_output_dir / 'report.md'
            generate_markdown_report(
                results_for_report,
                args.config,
                markdown_path,
                primary_metric='mae'
            )
            
            logger.info(f"Report generated successfully in: {report_output_dir}")
        else:
            logger.warning("No results found for report generation")
    except Exception as e:
        logger.warning(f"Could not generate report: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    logger.info(f"{'='*60}\n")
    
    return all_results


if __name__ == '__main__':
    main()
