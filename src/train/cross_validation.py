"""Cross-validation utilities for model training and evaluation."""
import os
# Configure JAX before importing to prevent multiprocessing issues
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# Disable Orbax async operations that can cause segmentation faults
os.environ['ORBAX_ASYNC'] = 'false'

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from flax.training import train_state
# Note: checkpoints import removed - we don't save checkpoints during CV to avoid Orbax issues

from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.dataset import QuantumTrajectoryDataset, create_dataloader
from src.data.transforms import Normalizer
from src.data.splits import create_cv_folds, save_cv_folds, load_cv_folds
from src.train.train import (
    create_model,
    create_train_state,
    train_epoch,
    evaluate
)
from src.train.metrics import compute_metrics
from src.utils.plotting import (
    plot_prediction_scatter,
    plot_error_histogram,
    plot_error_vs_gamma,
    plot_training_history,
    save_plot
)
import matplotlib.pyplot as plt


def train_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    all_data: Dict,
    config: Dict,
    model_type: str,
    seed: int,
    output_dir: Path,
    logger
) -> Dict:
    """Train model on a single CV fold.
    
    Args:
        fold_idx: Fold index
        train_idx: Training indices for this fold
        val_idx: Validation indices for this fold
        all_data: Full dataset dictionary
        config: Configuration dictionary
        model_type: Model type
        seed: Random seed
        output_dir: Output directory for this fold (should be model-specific dir)
        logger: Logger instance
        
    Returns:
        Dictionary with fold results (metrics, history, checkpoint path)
    """
    # Ensure output_dir is absolute
    output_dir = Path(output_dir).resolve()
    
    set_seed(seed + fold_idx)
    
    # Extract fold data
    def extract_split(idx):
        return {
            'observables': [all_data['observables'][i] for i in idx],
            'times': [all_data['times'][i] for i in idx],
            't_decoh': all_data['t_decoh'][idx],
            'gamma': [all_data['gamma'][i] for i in idx]
        }
    
    train_data = extract_split(train_idx)
    val_data = extract_split(val_idx)
    
    # Extract features
    features = config.get('features', ['sigma_z'])
    if isinstance(features, str):
        features = [features]
    
    seq_length = config.get('training', {}).get('seq_length', 100)
    predict_remaining = config.get('training', {}).get('predict_remaining', True)
    t_max = config.get('physics', {}).get('T_max', 10.0)
    
    # Build normalizer on train data
    normalizer = Normalizer()
    temp_dataset = QuantumTrajectoryDataset(
        train_data,
        seq_length=seq_length,
        features=features,
        predict_remaining=predict_remaining,
        normalizer=None,
        t_max=t_max
    )
    normalizer.fit(temp_dataset.sequences)
    
    # Create datasets
    train_dataset = QuantumTrajectoryDataset(
        train_data,
        seq_length=seq_length,
        features=features,
        predict_remaining=predict_remaining,
        normalizer=normalizer,
        t_max=t_max
    )
    
    val_dataset = QuantumTrajectoryDataset(
        val_data,
        seq_length=seq_length,
        features=features,
        predict_remaining=predict_remaining,
        normalizer=normalizer,
        t_max=t_max
    )
    
    # Create model
    n_features = len(features)
    model = create_model(model_type, config, n_features)
    
    # Initialize training state
    rng = jax.random.PRNGKey(seed + fold_idx)
    state, train_cfg = create_train_state(
        model, config, rng, n_features, dataset_size=len(train_dataset)
    )
    
    # Training loop
    train_config = config.get('training', {})
    n_epochs = train_config.get('n_epochs', 100)
    batch_size = train_config.get('batch_size', 32)
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    fold_checkpoint_dir = (output_dir / f'fold_{fold_idx}').resolve()
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    train_history = []
    val_history = []
    
    train_rng = jax.random.PRNGKey(seed + fold_idx + 1000)
    
    for epoch in range(n_epochs):
        # Train
        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed + fold_idx + epoch
        )
        train_rng, epoch_rng = jax.random.split(train_rng)
        state, train_metrics = train_epoch(
            state, train_loader, train_cfg, epoch_rng, epoch, n_epochs, verbose=False
        )
        
        # Validate
        val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        val_results = evaluate(state, val_loader, t_max=train_cfg['t_max'])
        val_loss = val_results['metrics']['mae']
        
        # Store history
        train_history.append({
            'loss': float(train_metrics['loss']),
            'base_loss': float(train_metrics['base_loss']),
            'constraint': float(train_metrics['constraint'])
        })
        val_history.append({
            'mae': float(val_results['metrics']['mae']),
            'rmse': float(val_results['metrics']['rmse']),
            'r2': float(val_results['metrics']['r2']),
            'coverage': float(val_results['metrics']['coverage'])
        })
        
        # Track best checkpoint (save only at the end to avoid Orbax issues)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = state  # Keep reference to best state
    
    # Skip checkpoint saving during CV to avoid Orbax/tensorstore segmentation fault
    # Results are saved in JSON format, and we use best_state from memory
    # If checkpoint is really needed, it can be saved after CV completes
    logger.info(f"Fold {fold_idx} completed - best epoch: {best_epoch}, val MAE: {best_val_loss:.6f}")
    logger.info("Skipping checkpoint save during CV (using state from memory)")
    
    # Use best state directly from memory (no checkpoint loading needed)
    if best_state is not None:
        state = best_state
        logger.info(f"Using best state from memory for fold {fold_idx} (epoch {best_epoch})")
    else:
        logger.warning(f"No best state found for fold {fold_idx}, using final state")
    
    # Final evaluation
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    val_results = evaluate(state, val_loader, t_max=train_cfg['t_max'])
    
    # Save fold results
    fold_results = {
        'fold_idx': fold_idx,
        'metrics': val_results['metrics'],
        'best_val_loss': best_val_loss,
        'train_history': train_history,
        'val_history': val_history,
        'checkpoint_path': str(fold_checkpoint_dir),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'predictions': val_results['predictions'],
        'targets': val_results['targets']
    }
    
    # Save fold history
    np.savez(
        fold_checkpoint_dir / 'fold_history.npz',
        train_history=train_history,
        val_history=val_history,
        best_val_loss=best_val_loss
    )
    
    # Generate plots for this fold
    try:
        figures_dir = fold_checkpoint_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert predictions/targets if needed
        y_pred = val_results['predictions']
        y_true = val_results['targets']
        
        # Get config values
        train_cfg = config.get('training', {})
        physics_cfg = config.get('physics', {})
        predict_remaining = train_cfg.get('predict_remaining', True)
        seq_length = train_cfg.get('seq_length', 100)
        t_max = train_cfg.get('t_max', physics_cfg.get('T_max', 10.0))
        
        # Convert to absolute times if predict_remaining
        if predict_remaining:
            dt = physics_cfg.get('dt', 0.1)
            T_obs = seq_length * dt
            y_pred_abs = y_pred + T_obs
            y_true_abs = y_true + T_obs
        else:
            y_pred_abs = y_pred * t_max
            y_true_abs = y_true * t_max
        
        # Scatter plot
        plot_prediction_scatter(
            y_true_abs,
            y_pred_abs,
            save_path=str(figures_dir / 'prediction_scatter.png'),
            title=f"Fold {fold_idx + 1}: True vs Predicted Decoherence Time"
        )
        
        # Error histogram
        errors = y_pred_abs - y_true_abs
        plot_error_histogram(
            errors,
            save_path=str(figures_dir / 'error_histogram.png'),
            title=f"Fold {fold_idx + 1}: Prediction Error Distribution"
        )
        
        # Error vs gamma (if available)
        gamma_values = np.array([
            g if isinstance(g, (int, float)) else np.mean(g) if isinstance(g, np.ndarray) else 0.0
            for g in val_dataset.gamma_values
        ])
        if len(gamma_values) > 0 and np.any(gamma_values > 0):
            plot_error_vs_gamma(
                errors,
                gamma_values,
                save_path=str(figures_dir / 'error_vs_gamma.png'),
                title=f"Fold {fold_idx + 1}: Error vs Dissipation Rate"
            )
        
        # Training history
        plot_training_history(
            train_history,
            val_history,
            save_path=str(figures_dir / 'training_history.png'),
            title=f"Fold {fold_idx + 1}: Training History"
        )
        
        # Plot Pauli operator oscillations for sample predictions
        # Show what model sees (only up to T_obs) vs full trajectory
        try:
            from src.utils.plotting import plot_observable_trajectories
            import matplotlib.pyplot as plt
            from src.utils.plotting import save_plot
            
            # Skip individual sample plots - we'll create aggregated plots after all folds
            # Individual sample plots are too detailed and create many files
            # Uncomment below to enable individual sample plots for debugging
            # n_samples_plot = min(5, len(val_dataset))
            # sample_indices = np.linspace(0, len(val_dataset) - 1, n_samples_plot, dtype=int)
            
            # dt = physics_cfg.get('dt', 0.1)
            # T_obs = seq_length * dt
            
            # for sample_idx in sample_indices:
            #     # Get true trajectory
            #     obs_dict = val_data['observables'][sample_idx]
            #     time_points = val_data['times'][sample_idx] if sample_idx < len(val_data['times']) else None
            #     t_d_true = val_data['t_decoh'][sample_idx]
            #     
            #     # Get predicted remaining time
            #     y_pred_remaining = y_pred[sample_idx]
            #     if predict_remaining:
            #         t_d_pred = y_pred_remaining + T_obs
            #     else:
            #         t_d_pred = y_pred_remaining * t_max
            #     
            #     # Prepare observables for plotting
            #     if isinstance(obs_dict, dict):
            #         # Get all Pauli operators (sigma_x, sigma_y, sigma_z)
            #         pauli_ops = {k: v for k, v in obs_dict.items() if k.startswith('sigma_')}
            #         
            #         # If not all Pauli operators are available, compute them from density matrices
            #         if 'density_matrices' in obs_dict and len(pauli_ops) < 3:
            #             try:
            #                 from src.physics.decoherence_time import compute_observables_single_qubit
            #                 
            #                 density_matrices = obs_dict['density_matrices']
            #                 if density_matrices is not None and len(density_matrices) > 0:
            #                     # Compute all Pauli operators from density matrices
            #                     computed_ops = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
            #                     for rho in density_matrices:
            #                         obs_dict_computed = compute_observables_single_qubit(rho)
            #                         for op_name in ['sigma_x', 'sigma_y', 'sigma_z']:
            #                             if op_name in obs_dict_computed:
            #                                 computed_ops[op_name].append(obs_dict_computed[op_name])
            #                     
            #                     # Convert to numpy arrays
            #                     for op_name, values in computed_ops.items():
            #                         if len(values) > 0:
            #                             pauli_ops[op_name] = np.array(values)
            #             except Exception as e:
            #                 logger.debug(f"Could not compute Pauli operators from density matrices: {e}")
            #         
            #         if not pauli_ops:
            #             continue
            #         
            #         # Get time points
            #         if time_points is None:
            #             # Estimate from first observable
            #             first_obs = list(pauli_ops.values())[0]
            #             time_points = np.arange(len(first_obs)) * dt
            #         
            #         time_points = np.asarray(time_points)
            #         
            #         # Split observables into observed (up to T_obs) and future
            #         obs_mask = time_points <= T_obs
            #         future_mask = time_points > T_obs
            #         
            #         # Create observables that model sees (only up to T_obs)
            #         observables_observed = {}
            #         observables_future = {}
            #         observables_true_full = {}
            #         
            #         for op_name, op_values in pauli_ops.items():
            #             op_values = np.asarray(op_values)
            #             observables_true_full[op_name] = op_values
            #             observables_observed[op_name] = op_values[obs_mask] if len(op_values) == len(time_points) else op_values[:seq_length]
            #             if np.any(future_mask):
            #                 observables_future[op_name] = op_values[future_mask]
            #         
            #         time_observed = time_points[obs_mask] if len(time_points) > 0 else time_points[:seq_length]
            #         time_future = time_points[future_mask] if np.any(future_mask) else np.array([])
            #         
            #         # Plot with observation window highlighted
            #         n_ops = len(pauli_ops)
            #         fig, axes = plt.subplots(n_ops, 1, figsize=(10, 3*n_ops))
            #         if n_ops == 1:
            #             axes = [axes]
            #         
            #         for idx, (op_name, op_values) in enumerate(pauli_ops.items()):
            #             ax = axes[idx]
            #             
            #             # Full true trajectory
            #             ax.plot(time_points, op_values, 'b-', alpha=0.3, linewidth=1, 
            #                    label='Full trajectory (model does not see)', linestyle='--')
            #             
            #             # Observed part (what model sees)
            #             if len(time_observed) > 0:
            #                 obs_values = observables_observed[op_name]
            #                 ax.plot(time_observed, obs_values, 'b-', alpha=0.8, linewidth=2, 
            #                        label='Observed (model input)')
            #             
            #             # Future part (model predicts t_decoh but doesn't see this)
            #             if len(time_future) > 0 and op_name in observables_future:
            #                 future_values = observables_future[op_name]
            #                 ax.plot(time_future, future_values, 'gray', alpha=0.5, linewidth=1, 
            #                        linestyle=':', label='Future (model predicts t_decoh)')
            #             
            #             # Mark observation window
            #             ax.axvline(T_obs, color='g', linestyle=':', linewidth=2, 
            #                       label=f'$T_{{obs}}$ = {T_obs:.3f}', alpha=0.7)
            #             ax.axvspan(0, T_obs, alpha=0.1, color='green')
            #             
            #             # Mark decoherence times
            #             ax.axvline(t_d_true, color='r', linestyle='--', linewidth=2, 
            #                       label=f'True $t_{{decoh}}$ = {t_d_true:.3f}')
            #             ax.axvline(t_d_pred, color='orange', linestyle='--', linewidth=2, 
            #                       label=f'Pred $t_{{decoh}}$ = {t_d_pred:.3f}')
            #             
            #             ax.set_xlabel('Time')
            #             ax.set_ylabel(f'$\\langle {op_name} \\rangle$')
            #             ax.set_title(f'{op_name} - Model sees only up to T_obs, predicts t_decoh')
            #             ax.legend(fontsize=8, loc='best')
            #             ax.grid(True, alpha=0.3)
            #         
            #         plt.suptitle(f"Fold {fold_idx + 1}, Sample {sample_idx}: Model Input vs Full Trajectory\n"
            #                    f"True $t_{{decoh}}$={t_d_true:.3f}, Pred $t_{{decoh}}$={t_d_pred:.3f}",
            #                    fontsize=12, y=1.02)
            #         plt.tight_layout()
            #         save_plot(fig, str(figures_dir / f'pauli_oscillations_sample_{sample_idx}.png'))
                    
        except Exception as e:
            logger.warning(f"Could not generate Pauli oscillation plots for fold {fold_idx}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        logger.info(f"Plots saved for fold {fold_idx} in {figures_dir}")
    except Exception as e:
        logger.warning(f"Could not generate plots for fold {fold_idx}: {e}")
    
    return fold_results


def run_cross_validation(
    all_data: Dict,
    config: Dict,
    model_type: str,
    n_folds: int = 5,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    output_dir: Path = Path('experiments'),
    logger=None
) -> Dict:
    """Run K-fold cross-validation.
    
    Args:
        all_data: Full dataset dictionary
        config: Configuration dictionary
        model_type: Model type
        n_folds: Number of folds
        seed: Random seed
        groups: Optional group labels for GroupKFold
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Dictionary with CV results (aggregated metrics, per-fold results)
    """
    if logger is None:
        logger = setup_logging()
    
    set_seed(seed)
    
    n_samples = len(all_data['observables'])
    
    # Create folds
    folds = create_cv_folds(n_samples, n_folds=n_folds, seed=seed, groups=groups)
    folds_list = list(folds)  # Convert to list to reuse
    
    # Save folds
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    folds_path = output_dir / 'cv_folds.json'
    save_cv_folds(folds_list, folds_path, seed, metadata={'model_type': model_type})
    
    logger.info(f"Starting {n_folds}-fold cross-validation for {model_type}")
    logger.info(f"Total trajectories in dataset: {n_samples}")
    logger.info(f"Each trajectory is a separate sample - model will predict t_decoh for EACH trajectory individually")
    
    # Train on each fold
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(folds_list, desc="CV Folds")):
        logger.info(f"\nFold {fold_idx + 1}/{n_folds}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # output_dir already contains model_type from run_suite.py, so don't add it again
        fold_output_dir = output_dir.resolve()
        result = train_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            all_data=all_data,
            config=config,
            model_type=model_type,
            seed=seed,
            output_dir=fold_output_dir,
            logger=logger
        )
        fold_results.append(result)
        
        logger.info(
            f"Fold {fold_idx + 1} - Val MAE: {result['metrics']['mae']:.6f}, "
            f"RMSE: {result['metrics']['rmse']:.6f}, "
            f"R²: {result['metrics']['r2']:.6f}"
        )
    
    # Aggregate metrics
    metric_names = ['mae', 'rmse', 'r2', 'coverage']
    aggregated = {}
    
    for metric_name in metric_names:
        values = [r['metrics'][metric_name] for r in fold_results]
        aggregated[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': [float(v) for v in values]
        }
    
    cv_results = {
        'n_folds': n_folds,
        'seed': seed,
        'model_type': model_type,
        'fold_results': fold_results,
        'aggregated_metrics': aggregated
    }
    
    # Save CV results
    import json
    # output_dir already contains model_type from run_suite.py
    results_path = output_dir / 'cv_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format (exclude predictions/targets arrays for JSON)
    cv_results_json = {
        'n_folds': n_folds,
        'seed': seed,
        'model_type': model_type,
        'aggregated_metrics': aggregated,
        'fold_results': [
            {
                'fold_idx': r['fold_idx'],
                'metrics': r['metrics'],
                'best_val_loss': r['best_val_loss'],
                'n_train': r['n_train'],
                'n_val': r['n_val']
            }
            for r in fold_results
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(cv_results_json, f, indent=2)
    
    logger.info(f"\nCross-validation results saved to {results_path}")
    logger.info("\nAggregated Metrics:")
    for metric_name in metric_names:
        agg = aggregated[metric_name]
        logger.info(f"  {metric_name.upper()}: {agg['mean']:.6f} ± {agg['std']:.6f}")
    
    # Generate aggregated plots across all folds
    try:
        # output_dir already contains model_type from run_suite.py
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all predictions and targets from all folds
        all_y_pred = []
        all_y_true = []
        all_errors = []
        
        train_cfg = config.get('training', {})
        physics_cfg = config.get('physics', {})
        predict_remaining = train_cfg.get('predict_remaining', True)
        seq_length = train_cfg.get('seq_length', 100)
        t_max = train_cfg.get('t_max', physics_cfg.get('T_max', 10.0))
        dt = physics_cfg.get('dt', 0.1)
        T_obs = seq_length * dt
        
        for r in fold_results:
            y_pred = r['predictions']
            y_true = r['targets']
            
            # Convert to absolute times
            if predict_remaining:
                dt = physics_cfg.get('dt', 0.1)
                T_obs = seq_length * dt
                y_pred_abs = y_pred + T_obs
                y_true_abs = y_true + T_obs
            else:
                y_pred_abs = y_pred * t_max
                y_true_abs = y_true * t_max
            
            all_y_pred.append(y_pred_abs)
            all_y_true.append(y_true_abs)
            all_errors.append(y_pred_abs - y_true_abs)
        
        # Concatenate all folds
        all_y_pred = np.concatenate(all_y_pred)
        all_y_true = np.concatenate(all_y_true)
        all_errors = np.concatenate(all_errors)
        
        # Aggregated scatter plot
        plot_prediction_scatter(
            all_y_true,
            all_y_pred,
            save_path=str(figures_dir / 'cv_aggregated_scatter.png'),
            title=f"CV Aggregated: True vs Predicted Decoherence Time"
        )
        
        # Aggregated error histogram
        plot_error_histogram(
            all_errors,
            save_path=str(figures_dir / 'cv_aggregated_error_histogram.png'),
            title=f"CV Aggregated: Prediction Error Distribution"
        )
        
        # Plot comparison of average Pauli operator values at predicted vs true t_decoh
        try:
            from scipy.interpolate import interp1d
            from src.physics.decoherence_time import compute_observables_single_qubit
            
            logger.info("Generating Pauli operator comparison plots...")
            
            # Collect all observables and compute values at predicted/true t_decoh
            all_pred_ops = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
            all_true_ops = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
            
            # Process each prediction
            for fold_idx, (train_idx, val_idx) in enumerate(folds_list):
                fold_result = fold_results[fold_idx]
                y_pred_fold = fold_result['predictions']
                y_true_fold = fold_result['targets']
                
                # Get fold data
                fold_val_data = {
                    'observables': [all_data['observables'][i] for i in val_idx],
                    'times': [all_data['times'][i] for i in val_idx],
                    't_decoh': all_data['t_decoh'][val_idx]
                }
                
                for local_idx, (y_pred_val, y_true_val) in enumerate(zip(y_pred_fold, y_true_fold)):
                    # Convert to absolute times
                    if predict_remaining:
                        dt = physics_cfg.get('dt', 0.1)
                        T_obs = seq_length * dt
                        t_pred = y_pred_val + T_obs
                        t_true = y_true_val + T_obs
                    else:
                        t_pred = y_pred_val * t_max
                        t_true = y_true_val * t_max
                    
                    # Get observables for this sample
                    obs_dict = fold_val_data['observables'][local_idx]
                    time_points = fold_val_data['times'][local_idx] if local_idx < len(fold_val_data['times']) else None
                    
                    if not isinstance(obs_dict, dict):
                        continue
                    
                    # Get or compute all Pauli operators
                    pauli_ops = {k: v for k, v in obs_dict.items() if k.startswith('sigma_')}
                    
                    # If not all available, compute from density matrices
                    if 'density_matrices' in obs_dict and len(pauli_ops) < 3:
                        density_matrices = obs_dict['density_matrices']
                        if density_matrices is not None and len(density_matrices) > 0:
                            computed_ops = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
                            for rho in density_matrices:
                                obs_dict_computed = compute_observables_single_qubit(rho)
                                for op_name in ['sigma_x', 'sigma_y', 'sigma_z']:
                                    if op_name in obs_dict_computed:
                                        computed_ops[op_name].append(obs_dict_computed[op_name])
                            
                            for op_name, values in computed_ops.items():
                                if len(values) > 0:
                                    pauli_ops[op_name] = np.array(values)
                    
                    if len(pauli_ops) == 0:
                        continue
                    
                    # Get time points
                    if time_points is None:
                        first_obs = list(pauli_ops.values())[0]
                        time_points = np.arange(len(first_obs)) * physics_cfg.get('dt', 0.1)
                    time_points = np.asarray(time_points)
                    
                    # Interpolate to get values at predicted and true t_decoh
                    for op_name in ['sigma_x', 'sigma_y', 'sigma_z']:
                        if op_name not in pauli_ops:
                            continue
                        
                        op_values = np.asarray(pauli_ops[op_name])
                        if len(op_values) != len(time_points):
                            continue
                        
                        # Interpolate
                        interp_func = interp1d(time_points, op_values, kind='linear',
                                              bounds_error=False, fill_value='extrapolate')
                        
                        # Get values at predicted and true t_decoh
                        try:
                            val_at_pred = float(interp_func(t_pred))
                            val_at_true = float(interp_func(t_true))
                            
                            all_pred_ops[op_name].append(val_at_pred)
                            all_true_ops[op_name].append(val_at_true)
                        except:
                            continue
            
            # Compute averages
            avg_pred_ops = {}
            avg_true_ops = {}
            std_pred_ops = {}
            std_true_ops = {}
            
            for op_name in ['sigma_x', 'sigma_y', 'sigma_z']:
                if len(all_pred_ops[op_name]) > 0:
                    avg_pred_ops[op_name] = np.mean(all_pred_ops[op_name])
                    std_pred_ops[op_name] = np.std(all_pred_ops[op_name])
                    avg_true_ops[op_name] = np.mean(all_true_ops[op_name])
                    std_true_ops[op_name] = np.std(all_true_ops[op_name])
            
            # Plot comparison of averages at t_decoh (bar plot)
            if len(avg_pred_ops) > 0:
                fig, axes = plt.subplots(1, len(avg_pred_ops), figsize=(5*len(avg_pred_ops), 5))
                if len(avg_pred_ops) == 1:
                    axes = [axes]
                
                op_names = list(avg_pred_ops.keys())
                
                for idx, op_name in enumerate(op_names):
                    ax = axes[idx] if len(avg_pred_ops) > 1 else axes[0]
                    
                    # Bar plot
                    bars1 = ax.bar(0, avg_pred_ops[op_name], width=0.35, yerr=std_pred_ops[op_name],
                                  label='Predicted (at pred $t_{decoh}$)', alpha=0.8, color='orange')
                    bars2 = ax.bar(1, avg_true_ops[op_name], width=0.35, yerr=std_true_ops[op_name],
                                  label='True (at true $t_{decoh}$)', alpha=0.8, color='blue')
                    
                    ax.set_ylabel(f'$\\langle {op_name} \\rangle$')
                    ax.set_title(f'Average {op_name} at Decoherence Time')
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['Predicted', 'True'])
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels
                    ax.text(0, avg_pred_ops[op_name], f'{avg_pred_ops[op_name]:.3f}',
                           ha='center', va='bottom', fontsize=10)
                    ax.text(1, avg_true_ops[op_name], f'{avg_true_ops[op_name]:.3f}',
                           ha='center', va='bottom', fontsize=10)
                
                plt.suptitle('Comparison: Average Pauli Operators at Predicted vs True Decoherence Time',
                           fontsize=14, y=1.02)
                plt.tight_layout()
                save_plot(fig, str(figures_dir / 'pauli_operators_comparison.png'))
                logger.info(f"Pauli operator comparison bar plot saved")
        except Exception as e:
            logger.warning(f"Could not generate Pauli operator comparison bar plot: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Plot average oscillations of Pauli operators over time
        # This should run independently of the bar plot above
        try:
            logger.info("Starting to collect trajectories for oscillations plot...")
            logger.info(f"Total trajectories in dataset: {len(all_data['observables'])}")
            logger.info(f"Model makes predictions for EACH trajectory separately")
            # Collect all trajectories and compute time-averaged values
            all_trajectories = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
            all_time_points_list = []
            all_pred_t_decoh = []
            all_true_t_decoh = []
            all_true_ops_at_pred_t = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
            all_true_ops_at_true_t = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
            
            samples_processed = 0
            samples_skipped = 0
            total_predictions = 0
            
            for fold_idx, (train_idx, val_idx) in enumerate(folds_list):
                logger.info(f"Processing fold {fold_idx + 1}/{len(folds_list)} for oscillations plot...")
                fold_result = fold_results[fold_idx]
                y_pred_fold = fold_result['predictions']
                y_true_fold = fold_result['targets']
                
                # Get fold data
                fold_val_data = {
                    'observables': [all_data['observables'][i] for i in val_idx],
                    'times': [all_data['times'][i] for i in val_idx],
                    't_decoh': all_data['t_decoh'][val_idx]
                }
                
                for local_idx, (y_pred_val, y_true_val, t_d_true) in enumerate(zip(y_pred_fold, y_true_fold, fold_val_data['t_decoh'])):
                    # Each iteration processes ONE trajectory and ONE prediction
                    total_predictions += 1
                    
                    # Convert to absolute times
                    if predict_remaining:
                        dt = physics_cfg.get('dt', 0.1)
                        T_obs = seq_length * dt
                        t_pred = y_pred_val + T_obs
                        t_true = y_true_val + T_obs
                    else:
                        t_pred = y_pred_val * t_max
                        t_true = y_true_val * t_max
                    
                    # Get observables for this sample
                    obs_dict = fold_val_data['observables'][local_idx]
                    time_points = fold_val_data['times'][local_idx] if local_idx < len(fold_val_data['times']) else None
                    
                    samples_processed += 1
                    
                    if not isinstance(obs_dict, dict):
                        samples_skipped += 1
                        continue
                    
                    # Get or compute all Pauli operators
                    pauli_ops = {k: v for k, v in obs_dict.items() if k.startswith('sigma_')}
                    
                    # If not all available, compute from density matrices
                    if 'density_matrices' in obs_dict and len(pauli_ops) < 3:
                        density_matrices = obs_dict['density_matrices']
                        if density_matrices is not None and len(density_matrices) > 0:
                            computed_ops = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
                            for rho in density_matrices:
                                obs_dict_computed = compute_observables_single_qubit(rho)
                                for op_name in ['sigma_x', 'sigma_y', 'sigma_z']:
                                    if op_name in obs_dict_computed:
                                        computed_ops[op_name].append(obs_dict_computed[op_name])
                            
                            for op_name, values in computed_ops.items():
                                if len(values) > 0:
                                    pauli_ops[op_name] = np.array(values)
                    
                    if len(pauli_ops) == 0:
                        samples_skipped += 1
                        continue
                    
                    # Get time points
                    if time_points is None:
                        first_obs = list(pauli_ops.values())[0]
                        time_points = np.arange(len(first_obs)) * physics_cfg.get('dt', 0.1)
                    time_points = np.asarray(time_points)
                    
                    # Store trajectories and values at t_decoh
                    for op_name in ['sigma_x', 'sigma_y', 'sigma_z']:
                        if op_name in pauli_ops:
                            op_values = np.asarray(pauli_ops[op_name])
                            if len(op_values) == len(time_points):
                                all_trajectories[op_name].append(op_values)
                                
                                # Interpolate to get values at predicted and true t_decoh
                                interp_func = interp1d(time_points, op_values, kind='linear',
                                                      bounds_error=False, fill_value='extrapolate')
                                try:
                                    val_at_pred_t = float(interp_func(t_pred))
                                    val_at_true_t = float(interp_func(t_true))
                                    # True value at predicted t_decoh (what the true operator value is at the time model predicted)
                                    all_true_ops_at_pred_t[op_name].append(val_at_pred_t)
                                    # True value at true t_decoh
                                    all_true_ops_at_true_t[op_name].append(val_at_true_t)
                                except:
                                    pass
                                
                                if op_name == 'sigma_x':  # Store time points only once
                                    all_time_points_list.append(time_points)
                                    all_pred_t_decoh.append(t_pred)
                                    all_true_t_decoh.append(t_true)
            
            # Plot average oscillations
            logger.info(f"Finished collecting trajectories: processed={samples_processed}, skipped={samples_skipped}")
            logger.info(f"Total predictions made by model: {total_predictions} (one per trajectory)")
            logger.info(f"Trajectories collected for plotting: sigma_x={len(all_trajectories['sigma_x'])}, "
                       f"sigma_y={len(all_trajectories['sigma_y'])}, sigma_z={len(all_trajectories['sigma_z'])}, "
                       f"time_points={len(all_time_points_list)}")
            
            if len(all_trajectories['sigma_x']) > 0 or len(all_trajectories['sigma_y']) > 0 or len(all_trajectories['sigma_z']) > 0:
                logger.info(f"Found trajectories, proceeding to create plot...")
                # Find common time grid
                all_times = []
                for tp in all_time_points_list:
                    all_times.extend(tp)
                logger.info(f"Found {len(all_times)} time points from {len(all_time_points_list)} trajectories, "
                           f"range: [{min(all_times) if all_times else 'N/A'}, "
                           f"{max(all_times) if all_times else 'N/A'}]")
                if all_times:
                    logger.info(f"Creating oscillations plot with {len(all_times)} time points...")
                    min_time = min(all_times)
                    max_time = max(all_times)
                    common_times = np.linspace(min_time, max_time, 500)
                    
                    # Interpolate all trajectories to common grid
                    interpolated_trajectories = {'sigma_x': [], 'sigma_y': [], 'sigma_z': []}
                    
                    for idx, time_points in enumerate(all_time_points_list):
                        for op_name in ['sigma_x', 'sigma_y', 'sigma_z']:
                            if idx < len(all_trajectories[op_name]):
                                traj = all_trajectories[op_name][idx]
                                # Ensure traj is a numpy array
                                traj = np.asarray(traj, dtype=np.float64)
                                time_points = np.asarray(time_points, dtype=np.float64)
                                
                                if len(traj) == len(time_points) and len(traj) > 0:
                                    try:
                                        interp_func = interp1d(time_points, traj, kind='linear',
                                                              bounds_error=False, fill_value='extrapolate')
                                        interp_traj = interp_func(common_times)
                                        # Ensure interp_traj is a proper array
                                        interp_traj = np.asarray(interp_traj, dtype=np.float64).flatten()
                                        if len(interp_traj) == len(common_times):
                                            interpolated_trajectories[op_name].append(interp_traj)
                                    except Exception as e:
                                        logger.debug(f"Interpolation failed for {op_name} trajectory {idx}: {e}")
                    
                    # Compute mean and std across all trajectories
                    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
                    
                    # Get observation window parameters
                    dt_local = physics_cfg.get('dt', 0.1)
                    T_obs_local = seq_length * dt_local
                    avg_pred_t = np.mean(all_pred_t_decoh)
                    avg_true_t = np.mean(all_true_t_decoh)
                    
                    for idx, op_name in enumerate(['sigma_x', 'sigma_y', 'sigma_z']):
                        if len(interpolated_trajectories[op_name]) == 0:
                            logger.warning(f"No interpolated trajectories for {op_name}")
                            continue
                        
                        ax = axes[idx]
                        # Convert to numpy array and ensure proper shape
                        traj_list = interpolated_trajectories[op_name]
                        # Filter out any invalid trajectories
                        valid_trajs = []
                        for traj in traj_list:
                            traj_arr = np.asarray(traj, dtype=np.float64)
                            if traj_arr.ndim == 1 and len(traj_arr) == len(common_times):
                                valid_trajs.append(traj_arr)
                        
                        if len(valid_trajs) == 0:
                            logger.warning(f"No valid trajectories for {op_name}")
                            continue
                        
                        trajectories = np.array(valid_trajs)
                        logger.info(f"Computing mean/std for {op_name}: shape={trajectories.shape}")
                        
                        # Compute mean and std
                        mean_traj = np.mean(trajectories, axis=0)
                        std_traj = np.std(trajectories, axis=0, dtype=np.float64)
                        
                        # Split time into observed (model sees) and future (model predicts)
                        obs_mask = common_times <= T_obs_local
                        future_mask = common_times > T_obs_local
                        
                        # Plot FULL theoretical trajectory (true oscillations) - solid line
                        ax.plot(common_times, mean_traj, 'b-', linewidth=2.5, 
                               label=f'Средняя траектория $\\langle {op_name} \\rangle$ (по всем образцам)', 
                               alpha=0.8, zorder=2, linestyle='-')
                        
                        # Plot std band for full trajectory (lighter, behind everything)
                        ax.fill_between(common_times, mean_traj - std_traj, mean_traj + std_traj,
                                       alpha=0.15, color='blue', label='±1 std (разброс траекторий)', zorder=0)
                        
                        # Plot more individual trajectories to show oscillations clearly
                        # This is important because averaging can hide oscillations if phases differ
                        n_sample = min(10, len(trajectories))
                        sample_indices = np.linspace(0, len(trajectories) - 1, n_sample, dtype=int)
                        colors = plt.cm.tab10(np.linspace(0, 1, n_sample))
                        for i, sample_idx in enumerate(sample_indices):
                            if i == 0:  # Label only first one
                                ax.plot(common_times, trajectories[sample_idx], color=colors[i], 
                                       alpha=0.4, linewidth=1.2, linestyle='-', 
                                       label=f'Примеры индивидуальных траекторий (n={n_sample})', zorder=0)
                            else:
                                ax.plot(common_times, trajectories[sample_idx], color=colors[i], 
                                       alpha=0.4, linewidth=1.2, linestyle='-', zorder=0)
                        
                        # Also plot median trajectory (more robust to outliers)
                        median_traj = np.median(trajectories, axis=0)
                        ax.plot(common_times, median_traj, 'purple', linewidth=2, 
                               linestyle=':', alpha=0.7, zorder=2,
                               label=f'Медианная траектория (более устойчива к выбросам)')
                        
                        # Interpolate mean trajectory to get values at avg t_decoh
                        mean_interp = interp1d(common_times, mean_traj, kind='linear',
                                              bounds_error=False, fill_value='extrapolate')
                        val_at_pred_t = mean_interp(avg_pred_t)
                        val_at_true_t = mean_interp(avg_true_t)
                        
                        # Mark observation window boundary (what model sees)
                        ax.axvline(T_obs_local, color='green', linestyle='--', linewidth=2,
                                  label=f'$T_{{obs}}$ = {T_obs_local:.3f} (модель видит только до этой границы)', 
                                  alpha=0.7, zorder=4)
                        ax.axvspan(0, T_obs_local, alpha=0.05, color='green', zorder=0)
                        
                        # Mark predicted t_decoh (model prediction) - thick orange line
                        ax.axvline(avg_pred_t, color='orange', linestyle='-', linewidth=3,
                                  label=f'Предсказание модели: $t_{{decoh}}$ = {avg_pred_t:.3f}', 
                                  alpha=0.9, zorder=5)
                        
                        # Mark true t_decoh - thick red line
                        ax.axvline(avg_true_t, color='red', linestyle='-', linewidth=3,
                                  label=f'Истинное значение: $t_{{decoh}}$ = {avg_true_t:.3f}', 
                                  alpha=0.9, zorder=5)
                        
                        # Values at t_decoh are shown on separate plot (pauli_operators_at_t_decoh_comparison.png)
                        # Removed markers from this plot for clarity
                        
                        ax.set_xlabel('Время', fontsize=12)
                        ax.set_ylabel(f'$\\langle {op_name} \\rangle$', fontsize=12)
                        ax.set_title(f'{op_name}: Осцилляции оператора Паули во времени (n={len(trajectories)} траекторий)', 
                                   fontsize=13, fontweight='bold')
                        # Place legend at bottom, smaller font, multiple columns
                        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                                 ncol=3, framealpha=0.9, fancybox=True, shadow=True)
                        ax.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.suptitle('Осцилляции операторов Паули: что видит модель и что она предсказывает',
                               fontsize=16, y=1.0, fontweight='bold')
                    # Adjust layout to make room for bottom legends
                    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
                    
                    plot_path = figures_dir / 'pauli_operators_average_oscillations.png'
                    save_plot(fig, str(plot_path))
                    
                    logger.info(f"Average Pauli operator oscillations plot saved to {plot_path}")
                    
                    # Create separate plot for operator values at t_decoh (yellow circles and red triangles)
                    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
                    
                    for idx, op_name in enumerate(['sigma_x', 'sigma_y', 'sigma_z']):
                        ax2 = axes2[idx]
                        
                        # Get values at predicted and true t_decoh
                        if len(all_true_ops_at_pred_t[op_name]) > 0 and len(all_true_ops_at_true_t[op_name]) > 0:
                            pred_vals = all_true_ops_at_pred_t[op_name]
                            true_vals = all_true_ops_at_true_t[op_name]
                            
                            # Compute statistics
                            pred_mean = np.mean(pred_vals)
                            pred_std = np.std(pred_vals)
                            true_mean = np.mean(true_vals)
                            true_std = np.std(true_vals)
                            
                            # Bar plot with error bars
                            x_pos = [0, 1]
                            means = [pred_mean, true_mean]
                            stds = [pred_std, true_std]
                            colors = ['orange', 'red']
                            labels = [f'При предсказанном $t_{{decoh}}$\n(среднее по всем)', 
                                     f'При истинном $t_{{decoh}}$\n(среднее по всем)']
                            
                            bars = ax2.bar(x_pos, means, yerr=stds, capsize=10, 
                                          color=colors, alpha=0.7, edgecolor='black', linewidth=2,
                                          width=0.6)
                            
                            # Add value labels on bars
                            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                                        f'{mean:.3f}\n±{std:.3f}',
                                        ha='center', va='bottom', fontsize=11, fontweight='bold')
                            
                            # Scatter plot of individual values
                            n_samples = len(pred_vals)
                            x_scatter_pred = np.random.normal(0, 0.05, n_samples)  # Jitter for visibility
                            x_scatter_true = np.random.normal(1, 0.05, n_samples)
                            
                            ax2.scatter(x_scatter_pred, pred_vals, color='orange', s=30, 
                                       alpha=0.4, edgecolors='black', linewidths=0.5, zorder=3,
                                       label=f'Индивидуальные значения (n={n_samples})')
                            ax2.scatter(x_scatter_true, true_vals, color='red', s=30, 
                                       alpha=0.4, edgecolors='black', linewidths=0.5, zorder=3,
                                       marker='^')
                            
                            ax2.set_xticks(x_pos)
                            ax2.set_xticklabels(['Предсказано\nмоделью', 'Истинное\nзначение'], fontsize=11)
                            ax2.set_ylabel(f'$\\langle {op_name} \\rangle$', fontsize=12, fontweight='bold')
                            ax2.set_title(f'{op_name}: Значения оператора при $t_{{decoh}}$', 
                                         fontsize=13, fontweight='bold')
                            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
                            ax2.legend(fontsize=9, loc='best')
                            
                            # Add difference annotation
                            diff = abs(pred_mean - true_mean)
                            ax2.text(0.5, 0.95, f'Разница: {diff:.4f}',
                                    transform=ax2.transAxes, fontsize=10,
                                    ha='center', va='top',
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                        else:
                            ax2.text(0.5, 0.5, f'Нет данных для {op_name}',
                                    transform=ax2.transAxes, ha='center', va='center',
                                    fontsize=12)
                    
                    plt.suptitle('Сравнение значений операторов Паули при предсказанном и истинном $t_{decoh}$',
                               fontsize=16, y=1.02, fontweight='bold')
                    plt.tight_layout()
                    
                    plot_path2 = figures_dir / 'pauli_operators_at_t_decoh_comparison.png'
                    save_plot(fig2, str(plot_path2))
                    
                    logger.info(f"Pauli operator values at t_decoh comparison plot saved to {plot_path2}")
                else:
                    logger.warning("No time points found for oscillations plot")
            else:
                logger.warning(f"No trajectories found for Pauli operator oscillations plot. "
                             f"sigma_x={len(all_trajectories['sigma_x'])}, "
                             f"sigma_y={len(all_trajectories['sigma_y'])}, "
                             f"sigma_z={len(all_trajectories['sigma_z'])}")
        except Exception as e:
            logger.error(f"Could not generate Pauli operator oscillations plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Also create scatter plot: predicted vs true operator values
        # Note: avg_pred_ops, all_pred_ops, and all_true_ops are defined in the first try block above
        try:
            # Check if variables are available (they should be if first try block succeeded)
            if 'avg_pred_ops' in locals() and 'all_pred_ops' in locals() and 'all_true_ops' in locals() and len(avg_pred_ops) > 0:
                fig, axes = plt.subplots(1, len(avg_pred_ops), figsize=(5*len(avg_pred_ops), 5))
                if len(avg_pred_ops) == 1:
                    axes = [axes]
                
                op_names = list(avg_pred_ops.keys())
                for idx, op_name in enumerate(op_names):
                    ax = axes[idx] if len(avg_pred_ops) > 1 else axes[0]
                    
                    pred_vals = all_pred_ops[op_name]
                    true_vals = all_true_ops[op_name]
                    
                    ax.scatter(true_vals, pred_vals, alpha=0.6, s=50)
                    
                    # Perfect prediction line
                    min_val = min(min(pred_vals), min(true_vals))
                    max_val = max(max(pred_vals), max(true_vals))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                           label='Perfect prediction', linewidth=2)
                    
                    ax.set_xlabel(f'True $\\langle {op_name} \\rangle$ at $t_{{decoh}}$')
                    ax.set_ylabel(f'Predicted $\\langle {op_name} \\rangle$ at pred $t_{{decoh}}$')
                    ax.set_title(f'{op_name}: Predicted vs True Values')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.suptitle('Pauli Operators: Predicted vs True Values at Respective Decoherence Times',
                           fontsize=14, y=1.02)
                plt.tight_layout()
                save_plot(fig, str(figures_dir / 'pauli_operators_scatter.png'))
                
                logger.info(f"Pauli operator comparison plots saved to {figures_dir}")
        except Exception as e:
            logger.warning(f"Could not generate Pauli operator comparison plots: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Training history comparison across folds
        if len(fold_results) > 0 and 'train_history' in fold_results[0]:
            # Plot average training history across folds
            max_epochs = max(len(r['train_history']) for r in fold_results)
            epochs = np.arange(1, max_epochs + 1)
            
            # Average train/val metrics across folds
            avg_train_loss = []
            avg_val_mae = []
            avg_val_r2 = []
            
            for epoch in range(max_epochs):
                train_losses = [r['train_history'][epoch]['loss'] 
                               for r in fold_results 
                               if epoch < len(r['train_history'])]
                val_maes = [r['val_history'][epoch]['mae'] 
                         for r in fold_results 
                         if epoch < len(r['val_history'])]
                val_r2s = [r['val_history'][epoch]['r2'] 
                           for r in fold_results 
                           if epoch < len(r['val_history'])]
                
                if train_losses:
                    avg_train_loss.append(np.mean(train_losses))
                if val_maes:
                    avg_val_mae.append(np.mean(val_maes))
                if val_r2s:
                    avg_val_r2.append(np.mean(val_r2s))
            
            # Create aggregated training history plot
            if avg_train_loss and avg_val_mae:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Loss
                ax = axes[0, 0]
                ax.plot(epochs[:len(avg_train_loss)], avg_train_loss, 'b-', label='Avg Train Loss', linewidth=2)
                ax.plot(epochs[:len(avg_val_mae)], avg_val_mae, 'r-', label='Avg Val MAE', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss/MAE')
                ax.set_title('Average Training History Across Folds')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                
                # R² - показываем каждый фолд отдельно с финальными значениями
                ax = axes[0, 1]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                final_r2_values = []
                
                # Находим разумный диапазон для отображения (обрезаем выбросы)
                all_r2_values = []
                for r in fold_results:
                    if 'val_history' in r and len(r['val_history']) > 0:
                        fold_r2 = [h.get('r2', 0) for h in r['val_history']]
                        all_r2_values.extend(fold_r2)
                
                if all_r2_values:
                    # Используем перцентили для определения разумного диапазона
                    r2_array = np.array(all_r2_values)
                    # Обрезаем значения ниже -2 (выбросы в начале обучения)
                    r2_clipped = np.clip(r2_array, -2, np.max(r2_array))
                    y_min = max(-1.0, np.percentile(r2_clipped, 1))  # Минимум: -1 или 1-й перцентиль
                    y_max = min(1.0, np.percentile(r2_clipped, 99))  # Максимум: 1 или 99-й перцентиль
                    # Убедимся, что диапазон включает финальные значения
                    final_r2s = [r['val_history'][-1]['r2'] for r in fold_results if 'val_history' in r and len(r['val_history']) > 0]
                    if final_r2s:
                        y_min = min(y_min, min(final_r2s) - 0.1)
                        y_max = max(y_max, max(final_r2s) + 0.1)
                else:
                    y_min, y_max = -1.0, 1.0
                
                # Показываем R² для каждого фолда отдельно (обрезаем выбросы при отображении)
                for idx, r in enumerate(fold_results):
                    if 'val_history' in r and len(r['val_history']) > 0:
                        fold_r2 = [h.get('r2', 0) for h in r['val_history']]
                        fold_epochs = np.arange(1, len(fold_r2) + 1)
                        final_r2 = fold_r2[-1]
                        final_r2_values.append(final_r2)
                        
                        # Обрезаем значения для отображения (но сохраняем оригинальные для вычислений)
                        fold_r2_display = np.clip(fold_r2, y_min, y_max)
                        
                        # Рисуем линию для каждого фолда
                        ax.plot(fold_epochs, fold_r2_display, color=colors[idx % len(colors)], 
                               alpha=0.6, linewidth=1.5, label=f'Fold {r["fold_idx"]} (final R²={final_r2:.3f})')
                        
                        # Отмечаем финальное значение
                        ax.scatter(fold_epochs[-1], final_r2, color=colors[idx % len(colors)], 
                                 s=50, zorder=5, edgecolors='black', linewidths=1)
                
                # Показываем медиану R² (более устойчива к выбросам, чем среднее)
                median_r2 = []
                for epoch in range(max_epochs):
                    val_r2s = [r['val_history'][epoch]['r2'] 
                             for r in fold_results 
                             if epoch < len(r['val_history'])]
                    if val_r2s:
                        median_r2.append(np.median(val_r2s))
                
                if median_r2:
                    median_r2_display = np.clip(median_r2, y_min, y_max)
                    ax.plot(epochs[:len(median_r2_display)], median_r2_display, 'k--', 
                           label=f'Median R² (final={median_r2[-1]:.3f})', linewidth=2.5, alpha=0.8)
                
                # Горизонтальная линия на R²=0
                ax.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                
                # Устанавливаем ограничения по Y
                ax.set_ylim([y_min, y_max])
                
                # Аннотация с итоговыми значениями
                if final_r2_values:
                    mean_final_r2 = np.mean(final_r2_values)
                    std_final_r2 = np.std(final_r2_values)
                    ax.text(0.02, 0.98, 
                           f'Final R²: {mean_final_r2:.3f} ± {std_final_r2:.3f}',
                           transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.8))
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('R²')
                ax.set_title('Validation R² by Fold (Individual Traces, Clipped)')
                ax.legend(loc='lower right', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # RMSE
                avg_val_rmse = []
                for epoch in range(max_epochs):
                    val_rmses = [r['val_history'][epoch]['rmse'] 
                               for r in fold_results 
                               if epoch < len(r['val_history'])]
                    if val_rmses:
                        avg_val_rmse.append(np.mean(val_rmses))
                
                ax = axes[1, 0]
                ax.plot(epochs[:len(avg_val_rmse)], avg_val_rmse, 'orange', label='Avg Val RMSE', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('RMSE')
                ax.set_title('Average Validation RMSE Across Folds')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                
                # Coverage
                avg_val_coverage = []
                for epoch in range(max_epochs):
                    val_coverages = [r['val_history'][epoch]['coverage'] 
                                   for r in fold_results 
                                   if epoch < len(r['val_history'])]
                    if val_coverages:
                        avg_val_coverage.append(np.mean(val_coverages))
                
                ax = axes[1, 1]
                ax.plot(epochs[:len(avg_val_coverage)], avg_val_coverage, 'purple', label='Avg Val Coverage', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Coverage')
                ax.set_title('Average Validation Coverage Across Folds')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.1])
                
                plt.tight_layout()
                save_plot(fig, str(figures_dir / 'cv_aggregated_training_history.png'))
        
        logger.info(f"Aggregated plots saved to {figures_dir}")
    except Exception as e:
        logger.warning(f"Could not generate aggregated plots: {e}")
    
    return cv_results
