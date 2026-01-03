"""Evaluation script."""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from flax.training import checkpoints

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config, parse_args_with_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.dataset import QuantumTrajectoryDataset, create_dataloader
from src.data.transforms import Normalizer
from src.models.mlp import MLP
from src.models.rnn import GRU, LSTM
from src.models.transformer import Transformer, TwoStageTransformer
from src.train.train import create_model, create_train_state, evaluate
from src.train.metrics import compute_metrics
from src.utils.plotting import (
    plot_prediction_scatter,
    plot_error_histogram,
    plot_error_vs_gamma,
    plot_observable_trajectories,
    plot_metrics_comparison
)


def main():
    """Main evaluation function."""
    args = parse_args_with_config()
    
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation")
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    seed = args.seed if args.seed else config.get('data', {}).get('seed', 42)
    set_seed(seed)
    logger = setup_logging()
    
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    
    # Load dataset
    data_path = config.get('data', {}).get('dataset_path', 'data/dataset.npz')
    data = np.load(data_path, allow_pickle=True)
    
    # Helper function to extract object arrays
    def extract_obj_array(arr):
        if arr.dtype == object:
            if arr.size == 1:
                return arr.item()
            else:
                result = []
                for i in range(len(arr)):
                    elem = arr[i]
                    if isinstance(elem, np.ndarray) and elem.size == 1:
                        result.append(elem.item())
                    else:
                        result.append(elem)
                return result
        return arr
    
    # Extract features - ensure it's always a list
    features_arr = data['features']
    if features_arr.dtype == object:
        if features_arr.size == 1:
            feat_val = features_arr.item()
            features = [feat_val] if isinstance(feat_val, str) else feat_val
        else:
            features = features_arr.tolist()
    else:
        features = features_arr.tolist()
    
    # Ensure features is a list
    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, list):
        features = list(features)
    
    n_features = len(features)
    seq_length = config.get('training', {}).get('seq_length', 100)
    predict_remaining = config.get('training', {}).get('predict_remaining', True)
    t_max = config.get('physics', {}).get('T_max', 10.0)
    
    # Build normalizer (fit on train data)
    normalizer = Normalizer()
    train_obs = extract_obj_array(data['train_observables'])
    temp_data = {
        'observables': train_obs,
        'times': extract_obj_array(data['train_times']),
        't_decoh': data['train_t_decoh'],
        'gamma': extract_obj_array(data['train_gamma'])
    }
    temp_dataset = QuantumTrajectoryDataset(
        temp_data,
        seq_length=seq_length,
        features=features,
        predict_remaining=predict_remaining,
        normalizer=None,
        t_max=t_max
    )
    normalizer.fit(temp_dataset.sequences)
    
    # Test dataset
    test_dataset = QuantumTrajectoryDataset(
        {
            'observables': extract_obj_array(data['test_observables']),
            'times': extract_obj_array(data['test_times']),
            't_decoh': data['test_t_decoh'],
            'gamma': extract_obj_array(data['test_gamma'])
        },
        seq_length=seq_length,
        features=features,
        predict_remaining=predict_remaining,
        normalizer=normalizer,
        t_max=t_max
    )
    
    # Load checkpoint - ensure absolute path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    
    # If checkpoint_path is a file, use its parent directory
    if checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = checkpoint_path
    
    # Ensure absolute path
    checkpoint_dir = checkpoint_dir.resolve()
    
    # Try to infer model type from checkpoint path (e.g., checkpoints/mlp/checkpoint_246 -> mlp)
    # Check if checkpoint_dir contains a model type subdirectory
    model_type_from_path = None
    checkpoint_parts = checkpoint_dir.parts
    if 'checkpoints' in checkpoint_parts:
        checkpoints_idx = checkpoint_parts.index('checkpoints')
        if checkpoints_idx + 1 < len(checkpoint_parts):
            potential_model_type = checkpoint_parts[checkpoints_idx + 1]
            # Check if it's a known model type
            if potential_model_type in ['mlp', 'gru', 'lstm', 'transformer', 'two_stage']:
                model_type_from_path = potential_model_type
                logger.info(f"Inferred model type from checkpoint path: {model_type_from_path}")
    
    # Create model - use model type from path if available, otherwise from config
    model_type = model_type_from_path or config.get('model', {}).get('type', 'transformer')
    if not model_type:
        model_type = 'transformer'
    
    # Override config model type to match checkpoint
    if model_type_from_path:
        if 'model' not in config:
            config['model'] = {}
        config['model']['type'] = model_type_from_path
        logger.info(f"Using model type from checkpoint path: {model_type_from_path}")
    
    model = create_model(model_type, config, n_features)
    
    logger.info(f"Loading checkpoint from: {checkpoint_dir}")
    logger.info(f"Model type: {model_type}")
    
    rng = jax.random.PRNGKey(seed)
    # For eval, use for_eval=True to skip learning rate schedule
    state, _ = create_train_state(model, config, rng, n_features, for_eval=True)
    
    # Restore checkpoint
    try:
        state = checkpoints.restore_checkpoint(str(checkpoint_dir), state)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to restore checkpoint: {e}")
        logger.error("This usually means:")
        logger.error("  1. The model architecture doesn't match the checkpoint")
        logger.error("  2. The checkpoint path is incorrect")
        logger.error("  3. The checkpoint is corrupted")
        logger.error(f"Model type used: {model_type}")
        logger.error(f"Model config: {config.get('model', {})}")
        raise
    
    logger.info("Loaded checkpoint")
    
    # Evaluate
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
    test_results = evaluate(state, test_loader, t_max=t_max)
    
    # Print metrics
    logger.info("Test Metrics:")
    for key, value in test_results['metrics'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Create plots - use model-specific directory
    base_output_dir = Path(config.get('evaluation', {}).get('output_dir', 'figures'))
    output_dir = base_output_dir / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots and metrics to: {output_dir}")
    
    # Save metrics to file
    import json
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'checkpoint': str(checkpoint_dir),
            'metrics': {k: float(v) for k, v in test_results['metrics'].items()},
            'config': {
                'model': config.get('model', {}),
                'training': {
                    'seq_length': seq_length,
                    'predict_remaining': predict_remaining
                },
                'physics': {
                    'T_max': t_max,
                    'dt': config.get('physics', {}).get('dt', 0.1)
                }
            }
        }, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # If predict_remaining, convert back to absolute times for plotting
    y_pred = test_results['predictions']
    y_true = test_results['targets']
    
    if predict_remaining:
        # Need to compute absolute times from remaining times
        # For simplicity, assume T_obs is seq_length * dt
        dt = config.get('physics', {}).get('dt', 0.1)
        T_obs = seq_length * dt
        y_pred_abs = y_pred + T_obs
        y_true_abs = y_true + T_obs
    else:
        y_pred_abs = y_pred * t_max
        y_true_abs = y_true * t_max
    
    # Plot predictions
    plot_prediction_scatter(
        y_true_abs,
        y_pred_abs,
        save_path=str(output_dir / 'prediction_scatter.png'),
        title="True vs Predicted Decoherence Time"
    )
    
    # Plot error histogram
    errors = y_pred_abs - y_true_abs
    plot_error_histogram(
        errors,
        save_path=str(output_dir / 'error_histogram.png'),
        title="Prediction Error Distribution"
    )
    
    # Plot error vs gamma (if available)
    gamma_values = np.array([
        g if isinstance(g, (int, float)) else np.mean(g) if isinstance(g, np.ndarray) else 0.0
        for g in test_dataset.gamma_values
    ])
    plot_error_vs_gamma(
        errors,
        gamma_values,
        save_path=str(output_dir / 'error_vs_gamma.png'),
        title="Error vs Dissipation Rate"
    )
    
    # Plot detailed trajectories for a few examples
    logger.info("Generating detailed trajectory plots for sample predictions...")
    
    try:
        # Select a few examples with different error magnitudes
        n_examples = min(6, len(test_dataset))
        if n_examples == 0:
            logger.warning("No test examples available for detailed plotting")
        else:
            example_indices = []
            
            # Select examples: best, worst, and random
            sorted_indices = np.argsort(np.abs(errors))
            example_indices.append(sorted_indices[0])  # Best prediction
            if len(sorted_indices) > 1:
                example_indices.append(sorted_indices[-1])  # Worst prediction
                # Add a few random ones
                if len(sorted_indices) > 2:
                    mid_start = len(sorted_indices) // 4
                    mid_end = 3 * len(sorted_indices) // 4
                    if mid_end > mid_start:
                        random_indices = np.random.choice(
                            sorted_indices[mid_start:mid_end],
                            size=min(4, mid_end - mid_start),
                            replace=False
                        )
                        example_indices.extend(random_indices)
            
            example_indices = list(set(example_indices))[:n_examples]  # Remove duplicates
            
            # Load full trajectory data for visualization
            data = np.load(data_path, allow_pickle=True)
            
            def extract_obj_array(arr):
                if arr.dtype == object:
                    if arr.size == 1:
                        return arr.item()
                    else:
                        result = []
                        for i in range(len(arr)):
                            elem = arr[i]
                            if isinstance(elem, np.ndarray) and elem.size == 1:
                                result.append(elem.item())
                            else:
                                result.append(elem)
                        return result
                return arr
            
            test_obs = extract_obj_array(data['test_observables'])
            test_times = extract_obj_array(data['test_times'])
    
            for idx, example_idx in enumerate(example_indices):
                try:
                    # Get full trajectory
                    if example_idx >= len(test_obs) or example_idx >= len(test_times):
                        continue
                    
                    traj_obs = test_obs[example_idx]
                    traj_times = test_times[example_idx]
                    t_decoh_true = y_true_abs[example_idx]
                    t_decoh_pred = y_pred_abs[example_idx]
                    
                    # Extract observables
                    if isinstance(traj_obs, dict):
                        observables_true = {k: v for k, v in traj_obs.items() 
                                          if k in features and isinstance(v, np.ndarray) and len(v) > 0}
                        
                        if observables_true:
                            # Plot observables
                            plot_observable_trajectories(
                                traj_times,
                                observables_true,
                                observables_pred=None,  # We don't predict full trajectories yet
                                t_decoh_true=t_decoh_true,
                                t_decoh_pred=t_decoh_pred,
                                save_path=str(output_dir / f'example_{idx}_observables.png'),
                                title=f"Example {idx+1}: Observables (Error: {errors[example_idx]:.4f})"
                            )
                        
                        # Plot metrics if available
                        metrics_true = {}
                        if 'purity' in traj_obs and isinstance(traj_obs['purity'], np.ndarray) and len(traj_obs['purity']) > 0:
                            metrics_true['purity'] = traj_obs['purity']
                        if 'coherence_l1' in traj_obs and isinstance(traj_obs['coherence_l1'], np.ndarray) and len(traj_obs['coherence_l1']) > 0:
                            metrics_true['coherence_l1'] = traj_obs['coherence_l1']
                        
                        if metrics_true:
                            plot_metrics_comparison(
                                traj_times,
                                metrics_true,
                                metrics_pred=None,
                                t_decoh_true=t_decoh_true,
                                t_decoh_pred=t_decoh_pred,
                                save_path=str(output_dir / f'example_{idx}_metrics.png'),
                                title=f"Example {idx+1}: Metrics (Error: {errors[example_idx]:.4f})"
                            )
                except Exception as e:
                    logger.warning(f"Could not plot example {idx} (index {example_idx}): {e}")
    except Exception as e:
        logger.warning(f"Could not generate detailed trajectory plots: {e}")
    
    logger.info(f"All plots saved to {output_dir}")
    
    # Print detailed statistics
    logger.info("\n" + "="*60)
    logger.info("Detailed Evaluation Statistics:")
    logger.info(f"  Mean Absolute Error: {np.mean(np.abs(errors)):.6f}")
    logger.info(f"  Median Absolute Error: {np.median(np.abs(errors)):.6f}")
    logger.info(f"  Std of Errors: {np.std(errors):.6f}")
    logger.info(f"  Max Error: {np.max(np.abs(errors)):.6f}")
    logger.info(f"  Min Error: {np.min(np.abs(errors)):.6f}")
    logger.info(f"  R² Score: {test_results['metrics']['r2']:.6f}")
    logger.info(f"  Coverage (5% T_max): {test_results['metrics']['coverage']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
