"""Training script for multi-task early warning models."""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import optax
from flax.training import train_state, checkpoints
from tqdm import tqdm
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.dataset import EarlyWarningDataset, collate_multitask
from src.data.transforms import Normalizer
from src.models.multitask import MultiTaskTransformer, MultiTaskRNN
from src.train.losses import compute_multitask_loss
from src.train.metrics import (
    compute_risk_metrics,
    compute_lead_time_metrics,
    compute_forecast_metrics,
    compute_metrics as compute_regression_metrics
)
import torch
from torch.utils.data import DataLoader


def create_multitask_model(config: dict, n_features: int):
    """Create multi-task model based on config.
    
    Args:
        config: Configuration dictionary
        n_features: Number of input features
        
    Returns:
        Model instance
    """
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'multitask_transformer')
    
    if model_type == 'multitask_transformer':
        return MultiTaskTransformer(
            d_model=model_config.get('d_model', 256),
            n_layers=model_config.get('n_layers', 4),
            n_heads=model_config.get('n_heads', 8),
            d_ff=model_config.get('d_ff', 1024),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            max_seq_len=model_config.get('max_seq_len', 500),
            use_pos_encoding=model_config.get('use_pos_encoding', True),
            pooling=model_config.get('pooling', 'attention'),
            enable_risk=model_config.get('enable_risk', True),
            enable_forecast=model_config.get('enable_forecast', False),
            enable_time=model_config.get('enable_time', True),
            n_deltas=model_config.get('n_deltas', 3),
            forecast_horizon=model_config.get('forecast_horizon', 20),
            forecast_n_features=model_config.get('forecast_n_features', 3),
            head_hidden_dims=tuple(model_config.get('head_hidden_dims', [256, 128, 64]))
        )
    elif model_type == 'multitask_rnn':
        return MultiTaskRNN(
            hidden_dim=model_config.get('hidden_dim', 256),
            n_layers=model_config.get('n_layers', 2),
            dropout_rate=model_config.get('dropout_rate', 0.2),
            cell_type=model_config.get('cell_type', 'lstm'),
            enable_risk=model_config.get('enable_risk', True),
            enable_forecast=model_config.get('enable_forecast', False),
            enable_time=model_config.get('enable_time', True),
            n_deltas=model_config.get('n_deltas', 3),
            forecast_horizon=model_config.get('forecast_horizon', 20),
            forecast_n_features=model_config.get('forecast_n_features', 3),
            head_hidden_dims=tuple(model_config.get('head_hidden_dims', [128, 64]))
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_step(state, batch, weights, loss_config, rng):
    """Single training step for multi-task model.
    
    Args:
        state: Training state
        batch: Batch of data
        weights: Loss weights
        loss_config: Loss configuration
        rng: Random key for dropout
        
    Returns:
        Updated state, loss, and metrics
    """
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
        # Forward pass
        predictions = state.apply_fn(
            {'params': params},
            batch['sequence'],
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        # Prepare targets
        targets = {}
        if 'risk_labels' in batch:
            targets['risk_labels'] = batch['risk_labels']
        if 'future_sequence' in batch:
            targets['future_sequence'] = batch['future_sequence']
        if 'remaining_time' in batch:
            targets['remaining_time'] = batch['remaining_time']
        
        # Compute loss
        total_loss, loss_dict = compute_multitask_loss(
            predictions=predictions,
            targets=targets,
            weights=weights,
            config=loss_config
        )
        
        return total_loss, (loss_dict, predictions)
    
    # Compute gradients
    (loss, (loss_dict, predictions)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss, loss_dict, predictions


def eval_step(state, batch, weights, loss_config):
    """Single evaluation step.
    
    Args:
        state: Training state
        batch: Batch of data
        weights: Loss weights
        loss_config: Loss configuration
        
    Returns:
        Loss, loss dict, and predictions
    """
    # Forward pass (no dropout)
    predictions = state.apply_fn(
        {'params': state.params},
        batch['sequence'],
        training=False
    )
    
    # Prepare targets
    targets = {}
    if 'risk_labels' in batch:
        targets['risk_labels'] = batch['risk_labels']
    if 'future_sequence' in batch:
        targets['future_sequence'] = batch['future_sequence']
    if 'remaining_time' in batch:
        targets['remaining_time'] = batch['remaining_time']
    
    # Compute loss
    total_loss, loss_dict = compute_multitask_loss(
        predictions=predictions,
        targets=targets,
        weights=weights,
        config=loss_config
    )
    
    return total_loss, loss_dict, predictions


@jax.jit
def train_step_jit(state, sequences, targets, weights_dict, rng):
    """JIT-compiled training step (simplified)."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    def loss_fn(params):
        predictions = state.apply_fn(
            {'params': params},
            sequences,
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        # Simplified loss computation
        total_loss = 0.0
        
        # Risk loss
        if 'risk_logits' in predictions and 'risk_labels' in targets:
            # BCE loss
            logits = predictions['risk_logits']
            labels = targets['risk_labels']
            max_val = jnp.maximum(logits, 0)
            risk_loss = jnp.mean(max_val - logits * labels + jnp.log(1 + jnp.exp(-jnp.abs(logits))))
            total_loss = total_loss + weights_dict['lambda_risk'] * risk_loss
        
        # Time loss
        if 'time' in predictions and 'remaining_time' in targets:
            time_loss = jnp.mean(jnp.abs(predictions['time'] - targets['remaining_time']))
            total_loss = total_loss + weights_dict['lambda_time'] * time_loss
        
        return total_loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss


def evaluate_model(
    state,
    dataloader,
    weights,
    loss_config,
    config
):
    """Evaluate model on validation/test set.
    
    Args:
        state: Training state
        dataloader: DataLoader for evaluation
        weights: Loss weights
        loss_config: Loss configuration
        config: Full configuration
        
    Returns:
        Dictionary with evaluation metrics
    """
    all_losses = []
    all_predictions = {}
    all_targets = {}
    
    for batch in dataloader:
        # Convert tensors to JAX arrays
        batch_jax = {
            k: jnp.array(v.numpy()) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        loss, loss_dict, predictions = eval_step(state, batch_jax, weights, loss_config)
        
        all_losses.append(float(loss))
        
        # Accumulate predictions and targets
        for key, value in predictions.items():
            if key not in all_predictions:
                all_predictions[key] = []
            all_predictions[key].append(np.array(value))
        
        for key in ['risk_labels', 'future_sequence', 'remaining_time', 't_obs', 't_decoh']:
            if key in batch_jax:
                if key not in all_targets:
                    all_targets[key] = []
                all_targets[key].append(np.array(batch_jax[key]))
    
    # Concatenate all batches
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key], axis=0)
    
    # Compute metrics
    metrics = {
        'loss': float(np.mean(all_losses))
    }
    
    # Risk metrics
    if 'risk_logits' in all_predictions:
        logits = all_predictions['risk_logits']
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        labels = all_targets['risk_labels']
        
        risk_metrics = compute_risk_metrics(probs, labels)
        metrics.update({f'risk_{k}': v for k, v in risk_metrics.items()})
        
        # Lead time metrics (if we have trajectory info)
        if 't_obs' in all_targets and 't_decoh' in all_targets:
            # Use first delta for lead time
            lead_metrics = compute_lead_time_metrics(
                probs[:, 0] if probs.ndim == 2 else probs,
                all_targets['t_obs'],
                all_targets['t_decoh'],
                threshold=0.5
            )
            metrics.update({f'lead_{k}': v for k, v in lead_metrics.items()})
    
    # Forecast metrics
    if 'forecast' in all_predictions:
        forecast_metrics = compute_forecast_metrics(
            all_predictions['forecast'],
            all_targets['future_sequence']
        )
        metrics.update(forecast_metrics)
    
    # Time regression metrics
    if 'time' in all_predictions:
        time_metrics = compute_regression_metrics(
            all_predictions['time'],
            all_targets['remaining_time']
        )
        metrics.update({f'time_{k}': v for k, v in time_metrics.items()})
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multi-task early warning model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--output-dir', type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    seed = config.get('data', {}).get('seed', 42)
    set_seed(seed)
    
    # Directories
    checkpoint_dir = Path(config.get('training', {}).get('checkpoint_dir', 'checkpoints'))
    history_dir = Path(config.get('training', {}).get('history_dir', 'runs'))
    output_dir = Path(args.output_dir) if args.output_dir else history_dir
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_path = config.get('data', {}).get('dataset_path')
    data = np.load(dataset_path, allow_pickle=True)
    
    # Get training config
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    enable_risk = model_config.get('enable_risk', True)
    enable_forecast = model_config.get('enable_forecast', False)
    enable_time = model_config.get('enable_time', True)
    
    # Create datasets
    sequences = data['sequences']
    t_obs = data['t_obs']
    t_decoh = data['t_decoh']
    future_sequences = data.get('future_sequences', None)
    
    deltas = training_config.get('deltas', [0.5, 1.0, 2.0])
    
    # Split indices
    train_indices = data['train_indices']
    val_indices = data['val_indices']
    test_indices = data['test_indices']
    
    # Create datasets
    train_dataset = EarlyWarningDataset(
        sequences=sequences[train_indices],
        t_obs=t_obs[train_indices],
        t_decoh=t_decoh[train_indices],
        future_sequences=future_sequences[train_indices] if future_sequences is not None else None,
        deltas=deltas,
        enable_risk=enable_risk,
        enable_forecast=enable_forecast,
        enable_time=enable_time
    )
    
    val_dataset = EarlyWarningDataset(
        sequences=sequences[val_indices],
        t_obs=t_obs[val_indices],
        t_decoh=t_decoh[val_indices],
        future_sequences=future_sequences[val_indices] if future_sequences is not None else None,
        deltas=deltas,
        enable_risk=enable_risk,
        enable_forecast=enable_forecast,
        enable_time=enable_time
    )
    
    # DataLoaders
    batch_size = training_config.get('batch_size', 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multitask
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multitask
    )
    
    # Create model
    n_features = sequences.shape[-1]
    model = create_multitask_model(config, n_features)
    
    # Initialize model
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    dummy_input = jnp.ones((1, sequences.shape[1], n_features))
    params = model.init(init_rng, dummy_input, training=False)['params']
    
    # Setup optimizer
    learning_rate = training_config.get('learning_rate', 0.0001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Loss weights
    weights = {
        'lambda_risk': training_config.get('lambda_risk', 1.0),
        'lambda_forecast': training_config.get('lambda_forecast', 1.0),
        'lambda_time': training_config.get('lambda_time', 1.0)
    }
    
    loss_config = {
        'use_focal_loss': training_config.get('use_focal_loss', False),
        'focal_alpha': training_config.get('focal_alpha', 0.25),
        'focal_gamma': training_config.get('focal_gamma', 2.0),
        'forecast_loss_type': training_config.get('forecast_loss_type', 'mae'),
        'forecast_smoothness_weight': training_config.get('forecast_smoothness_weight', 0.0)
    }
    
    # Training loop
    n_epochs = training_config.get('n_epochs', 200)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    print(f"\n🚀 Starting multi-task training for {n_epochs} epochs...")
    print(f"  Tasks enabled: Risk={enable_risk}, Forecast={enable_forecast}, Time={enable_time}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(n_epochs):
        # Training
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            # Convert to JAX
            batch_jax = {
                k: jnp.array(v.numpy()) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            rng, step_rng = jax.random.split(rng)
            state, loss, loss_dict, _ = train_step(state, batch_jax, weights, loss_config, step_rng)
            train_losses.append(float(loss))
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        val_metrics = evaluate_model(state, val_loader, weights, loss_config, config)
        history['val_loss'].append(val_metrics['loss'])
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        if 'risk_auroc' in val_metrics:
            print(f"  Val AUROC: {val_metrics['risk_auroc']:.4f}")
        if 'lead_mean_lead_time' in val_metrics:
            print(f"  Mean Lead Time: {val_metrics['lead_mean_lead_time']:.4f}")
        
        # Save checkpoint if improved
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoints.save_checkpoint(
                ckpt_dir=str(checkpoint_dir),
                target=state,
                step=epoch,
                prefix='best_',
                overwrite=True
            )
            print(f"  ✓ Saved best checkpoint")
    
    # Save final history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print(f"  History saved to: {output_dir / 'history.json'}")


if __name__ == '__main__':
    main()
