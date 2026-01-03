"""Evaluation script for multi-task models with visualizations."""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.data.dataset import EarlyWarningDataset, collate_multitask
from src.train.train_multitask import create_multitask_model
from src.train.metrics import (
    compute_risk_metrics,
    compute_lead_time_metrics,
    compute_tpr_at_fpr,
    compute_forecast_metrics,
    compute_calibration_metrics,
    compute_metrics as compute_regression_metrics
)
from src.utils.early_warning_plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_reliability_diagram,
    plot_risk_histogram,
    plot_lead_time_distribution,
    plot_trajectory_with_risk,
    plot_forecast_comparison
)
from flax.training import checkpoints
import torch
from torch.utils.data import DataLoader


def evaluate_and_visualize(
    config_path: str,
    checkpoint_path: str,
    output_dir: Path,
    split: str = 'test'
):
    """Evaluate model and generate visualizations.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint
        output_dir: Output directory for plots and metrics
        split: Which split to evaluate ('val' or 'test')
    """
    # Load config
    config = load_config(config_path)
    
    # Load dataset
    dataset_path = config.get('data', {}).get('dataset_path')
    data = np.load(dataset_path, allow_pickle=True)
    
    # Get config
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    enable_risk = model_config.get('enable_risk', True)
    enable_forecast = model_config.get('enable_forecast', False)
    enable_time = model_config.get('enable_time', True)
    
    # Get split
    if split == 'val':
        indices = data['val_indices']
    elif split == 'test':
        indices = data['test_indices']
    else:
        indices = data['train_indices']
    
    # Create dataset
    sequences = data['sequences']
    t_obs = data['t_obs']
    t_decoh = data['t_decoh']
    future_sequences = data.get('future_sequences', None)
    
    deltas = training_config.get('deltas', [0.5, 1.0, 2.0])
    
    dataset = EarlyWarningDataset(
        sequences=sequences[indices],
        t_obs=t_obs[indices],
        t_decoh=t_decoh[indices],
        future_sequences=future_sequences[indices] if future_sequences is not None else None,
        deltas=deltas,
        enable_risk=enable_risk,
        enable_forecast=enable_forecast,
        enable_time=enable_time
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_multitask
    )
    
    # Load model
    n_features = sequences.shape[-1]
    model = create_multitask_model(config, n_features)
    
    # Initialize and load checkpoint
    seed = config.get('data', {}).get('seed', 42)
    rng = jax.random.PRNGKey(seed)
    
    dummy_input = jnp.ones((1, sequences.shape[1], n_features))
    variables = model.init(rng, dummy_input, training=False)
    
    # Load checkpoint
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=None
    )
    params = restored_state['params']
    
    # Evaluate
    print(f"Evaluating on {split} set...")
    
    all_predictions = {}
    all_targets = {}
    
    for batch in dataloader:
        batch_jax = {
            k: jnp.array(v.numpy()) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Forward pass
        predictions = model.apply(
            {'params': params},
            batch_jax['sequence'],
            training=False
        )
        
        # Accumulate
        for key, value in predictions.items():
            if key not in all_predictions:
                all_predictions[key] = []
            all_predictions[key].append(np.array(value))
        
        for key in ['risk_labels', 'future_sequence', 'remaining_time', 't_obs', 't_decoh']:
            if key in batch_jax:
                if key not in all_targets:
                    all_targets[key] = []
                all_targets[key].append(np.array(batch_jax[key]))
    
    # Concatenate
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key], axis=0)
    
    # Compute metrics
    metrics = {}
    
    # Risk metrics
    if 'risk_logits' in all_predictions:
        logits = all_predictions['risk_logits']
        probs = 1 / (1 + np.exp(-logits))
        labels = all_targets['risk_labels']
        
        print("\n📊 Risk Classification Metrics:")
        risk_metrics = compute_risk_metrics(probs, labels)
        metrics['risk'] = risk_metrics
        
        for key, val in risk_metrics.items():
            print(f"  {key}: {val:.4f}")
        
        # TPR at FPR
        tpr_at_fpr = compute_tpr_at_fpr(
            probs[:, 0] if probs.ndim == 2 else probs,
            labels[:, 0] if labels.ndim == 2 else labels,
            max_fpr=config.get('evaluation', {}).get('max_fpr', 0.05)
        )
        metrics['risk']['tpr_at_fpr'] = tpr_at_fpr
        print(f"  TPR@FPR5%: {tpr_at_fpr:.4f}")
        
        # Lead time metrics
        lead_metrics = compute_lead_time_metrics(
            probs[:, 0] if probs.ndim == 2 else probs,
            all_targets['t_obs'],
            all_targets['t_decoh'],
            threshold=0.5
        )
        metrics['lead_time'] = lead_metrics
        
        print("\n⏱️  Lead Time Metrics:")
        for key, val in lead_metrics.items():
            print(f"  {key}: {val:.4f}")
        
        # Calibration
        calib_metrics = compute_calibration_metrics(
            probs[:, 0] if probs.ndim == 2 else probs,
            labels[:, 0] if labels.ndim == 2 else labels
        )
        metrics['calibration'] = calib_metrics
        print(f"\n📐 Calibration ECE: {calib_metrics['ece']:.4f}")
        
        # Generate plots
        print("\n📈 Generating visualizations...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC curve
        plot_roc_curve(
            labels,
            probs,
            save_path=output_dir / 'roc_curve.png',
            delta_labels=[f"{d}" for d in deltas]
        )
        
        # PR curve
        plot_pr_curve(
            labels,
            probs,
            save_path=output_dir / 'pr_curve.png',
            delta_labels=[f"{d}" for d in deltas]
        )
        
        # Reliability diagram
        plot_reliability_diagram(
            labels[:, 0] if labels.ndim == 2 else labels,
            probs[:, 0] if probs.ndim == 2 else probs,
            save_path=output_dir / 'reliability_diagram.png'
        )
        
        # Risk histogram
        plot_risk_histogram(
            labels[:, 0] if labels.ndim == 2 else labels,
            probs[:, 0] if probs.ndim == 2 else probs,
            save_path=output_dir / 'risk_histogram.png'
        )
        
        # Lead time distribution
        # Extract lead times for positive predictions
        threshold = 0.5
        probs_single = probs[:, 0] if probs.ndim == 2 else probs
        alarms = probs_single >= threshold
        lead_times = []
        for i in range(len(alarms)):
            if alarms[i] and all_targets['t_decoh'][i] >= all_targets['t_obs'][i]:
                lead_times.append(all_targets['t_decoh'][i] - all_targets['t_obs'][i])
        
        if len(lead_times) > 0:
            plot_lead_time_distribution(
                np.array(lead_times),
                save_path=output_dir / 'lead_time_distribution.png'
            )
    
    # Forecast metrics
    if 'forecast' in all_predictions:
        forecast_pred = all_predictions['forecast']
        forecast_true = all_targets['future_sequence']
        
        forecast_metrics = compute_forecast_metrics(forecast_pred, forecast_true)
        metrics['forecast'] = forecast_metrics
        
        print("\n🔮 Forecast Metrics:")
        print(f"  MAE: {forecast_metrics['forecast_mae']:.4f}")
        print(f"  RMSE: {forecast_metrics['forecast_rmse']:.4f}")
        
        # Plot some forecast examples
        n_examples = min(5, len(forecast_pred))
        for i in range(n_examples):
            plot_forecast_comparison(
                forecast_true[i],
                forecast_pred[i],
                times=np.arange(forecast_true.shape[1]),
                t_obs=all_targets['t_obs'][i],
                save_path=output_dir / f'forecast_example_{i}.png'
            )
    
    # Time regression metrics
    if 'time' in all_predictions:
        time_pred = all_predictions['time']
        time_true = all_targets['remaining_time']
        
        time_metrics = compute_regression_metrics(time_pred, time_true)
        metrics['time'] = time_metrics
        
        print("\n⏲️  Time Regression Metrics:")
        print(f"  MAE: {time_metrics['mae']:.4f}")
        print(f"  RMSE: {time_metrics['rmse']:.4f}")
        print(f"  R²: {time_metrics['r2']:.4f}")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"  Metrics saved to: {output_dir / 'metrics.json'}")
    print(f"  Plots saved to: {output_dir}")
    
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate multi-task model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    evaluate_and_visualize(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        split=args.split
    )


if __name__ == '__main__':
    main()
