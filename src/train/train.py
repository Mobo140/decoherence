"""Training script."""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import optax
from flax.training import train_state, checkpoints
from tqdm import tqdm

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
from src.train.losses import compute_loss
from src.train.metrics import compute_metrics


def create_model(model_type: str, config: dict, n_features: int):
    """Create model based on type.
    
    Args:
        model_type: Model type ('mlp', 'gru', 'lstm', 'transformer', 'two_stage')
        config: Model configuration
        n_features: Number of input features
        
    Returns:
        Model instance
    """
    model_config = config.get('model', {})
    if 'type' in model_config and model_type is None:
        model_type = model_config['type']
    
    if model_type == 'mlp':
        return MLP(
            hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
            dropout_rate=model_config.get('dropout_rate', 0.1)
        )
    elif model_type == 'gru':
        return GRU(
            hidden_dim=model_config.get('hidden_dim', 128),
            n_layers=model_config.get('n_layers', 2),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            head_hidden_dims=model_config.get('head_hidden_dims', [64])
        )
    elif model_type == 'lstm':
        return LSTM(
            hidden_dim=model_config.get('hidden_dim', 128),
            n_layers=model_config.get('n_layers', 2),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            head_hidden_dims=model_config.get('head_hidden_dims', [64])
        )
    elif model_type == 'transformer':
        return Transformer(
            d_model=model_config.get('d_model', 256),
            n_layers=model_config.get('n_layers', 4),
            n_heads=model_config.get('n_heads', 8),
            d_ff=model_config.get('d_ff', 1024),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            max_seq_len=model_config.get('max_seq_len', 500),
            head_hidden_dims=model_config.get('head_hidden_dims', [128, 64])
        )
    elif model_type == 'two_stage':
        return TwoStageTransformer(
            d_model=model_config.get('d_model', 256),
            n_layers=model_config.get('n_layers', 4),
            n_heads=model_config.get('n_heads', 8),
            d_ff=model_config.get('d_ff', 1024),
            dropout_rate=model_config.get('dropout_rate', 0.1),
            n_coeffs=model_config.get('n_coeffs', 3),
            bernstein_degree=model_config.get('bernstein_degree', 2),
            T_max=config.get('physics', {}).get('T_max', 10.0),
            head_hidden_dims=model_config.get('head_hidden_dims', [128, 64])
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Use wrapper to avoid JIT issues with string loss_type
def train_step_wrapper(state, batch, loss_type, t_max, constraint_weight, huber_delta, rng):
    """Wrapper that handles loss_type selection."""
    # For now, just use MAE - can expand later
    if loss_type != 'mae':
        raise NotImplementedError(f"Only 'mae' loss supported in JIT mode, got {loss_type}")
    
    sequences, targets = batch
    # Split RNG for dropout
    dropout_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(params):
        preds = state.apply_fn(
            {'params': params},
            sequences,
            training=True,
            rngs={'dropout': dropout_rng}
        )
        loss, base_loss, constraint = compute_loss(
            preds, targets,
            loss_type=loss_type,
            t_max=t_max,
            constraint_weight=constraint_weight,
            huber_delta=huber_delta
        )
        return loss, (base_loss, constraint)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (base_loss, constraint)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {'loss': loss, 'base_loss': base_loss, 'constraint': constraint}

train_step = jax.jit(train_step_wrapper, static_argnames=['loss_type'])


@jax.jit
def eval_step(state, batch):
    """Single evaluation step.
    
    Args:
        state: Training state
        batch: Batch of (sequences, targets)
        
    Returns:
        Predictions and targets
    """
    sequences, targets = batch
    preds = state.apply_fn({'params': state.params}, sequences, training=False)
    return preds, targets


def train_epoch(state, dataloader, train_cfg, rng, epoch, n_epochs, verbose=True):
    """Train for one epoch.
    
    Args:
        state: Training state
        dataloader: Data loader iterator
        train_cfg: Training configuration dict
        rng: Random key
        epoch: Current epoch number
        n_epochs: Total number of epochs
        verbose: Whether to show progress bar
        
    Returns:
        Updated state and epoch metrics
    """
    metrics_list = []
    n_batches = 0
    
    # Convert dataloader to list to get length (for progress bar)
    if verbose:
        # Estimate batch count
        batch_list = list(dataloader)
        n_batches = len(batch_list)
        pbar = tqdm(batch_list, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
    else:
        pbar = dataloader
    
    for batch in pbar:
        # Create new RNG for this step
        step_rng, rng = jax.random.split(rng)
        state, metrics = train_step(
            state, batch,
            loss_type=train_cfg['loss_type'],
            t_max=train_cfg['t_max'],
            constraint_weight=train_cfg['constraint_weight'],
            huber_delta=train_cfg['huber_delta'],
            rng=step_rng
        )
        metrics_list.append(metrics)
        
        # Update progress bar
        if verbose and hasattr(pbar, 'set_postfix'):
            current_loss = metrics['loss']
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in metrics_list])
        for key in metrics_list[0].keys()
    }
    
    return state, avg_metrics


def evaluate(state, dataloader, t_max):
    """Evaluate model.
    
    Args:
        state: Training state
        dataloader: Data loader iterator
        t_max: Maximum time for metrics
        
    Returns:
        Dictionary with predictions, targets, and metrics
    """
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        preds, targets = eval_step(state, batch)
        all_preds.append(np.array(preds))
        all_targets.append(np.array(targets))
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets, t_max=t_max)
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'metrics': metrics
    }


def create_train_state(model, config, rng, n_features, dataset_size=None, for_eval=False):
    """Create initial training state.
    
    Args:
        model: Model instance
        config: Training configuration
        rng: Random key
        n_features: Number of input features
        dataset_size: Optional dataset size for computing steps_per_epoch
        for_eval: If True, use constant learning rate (for evaluation)
        
    Returns:
        Tuple of (training state, training config dict)
    """
    # Initialize model
    train_config = config.get('training', {})
    dummy_input = jnp.ones((1, train_config.get('seq_length', 100), n_features))
    
    params = model.init(rng, dummy_input, training=False)['params']
    
    # Optimizer with learning rate scheduling
    learning_rate = train_config.get('learning_rate', 1e-4)
    optimizer_name = train_config.get('optimizer', 'adam')
    
    # For eval, use constant learning rate (schedule doesn't matter)
    if for_eval:
        schedule = learning_rate
    else:
        lr_schedule_type = train_config.get('lr_schedule', 'warmup_cosine')  # 'cosine', 'warmup_cosine', 'step', None
        warmup_steps = train_config.get('warmup_steps', 1000)
        n_epochs = train_config.get('n_epochs', 100)
        batch_size = train_config.get('batch_size', 32)
        # Compute steps_per_epoch
        steps_per_epoch = train_config.get('steps_per_epoch', None)
        if steps_per_epoch is None:
            # Estimate from dataset size if provided, otherwise use default
            if dataset_size is not None:
                steps_per_epoch = max(1, dataset_size // batch_size)
            else:
                # Default estimate: assume ~100 samples per epoch
                steps_per_epoch = max(1, 100 // batch_size)
        
        # Ensure decay_steps is positive
        total_steps = n_epochs * steps_per_epoch
        if total_steps <= 0:
            total_steps = max(1000, n_epochs * 10)  # Fallback to reasonable default
        
        # Create learning rate schedule
        if lr_schedule_type == 'cosine':
            # Cosine decay
            schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=total_steps
            )
        elif lr_schedule_type == 'warmup_cosine':
            # Warmup + cosine decay
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=min(warmup_steps, total_steps // 2),  # Ensure warmup < total_steps
                decay_steps=total_steps,
                end_value=learning_rate * 0.01
            )
        elif lr_schedule_type == 'step':
            # Step decay
            schedule = optax.piecewise_constant_schedule(
                init_value=learning_rate,
                boundaries_and_scales={
                    int(n_epochs * 0.5): 0.1,
                    int(n_epochs * 0.75): 0.1
                }
            )
        else:
            # Constant learning rate
            schedule = learning_rate
    
    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adam(schedule)
        )
    elif optimizer_name == 'adamw':
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adamw(schedule, weight_decay=train_config.get('weight_decay', 1e-5))
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Store additional config in a dict
    train_cfg = {
        'loss_type': train_config.get('loss_type', 'mae'),
        't_max': config.get('physics', {}).get('T_max', 10.0),
        'constraint_weight': train_config.get('constraint_weight', 0.0),
        'huber_delta': train_config.get('huber_delta', 1.0)
    }
    
    return state, train_cfg


def main():
    """Main training function."""
    args = parse_args_with_config()
    
    # Load config
    config = load_config(args.config)
    
    # Override model if specified
    if args.model:
        if 'model' not in config:
            config['model'] = {}
        config['model']['type'] = args.model
    elif 'model' not in config or 'type' not in config['model']:
        config.setdefault('model', {})['type'] = 'transformer'
    
    # Setup
    seed = args.seed if args.seed else config.get('data', {}).get('seed', 42)
    set_seed(seed)
    logger = setup_logging()
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Model: {config.get('model', {}).get('type', 'transformer')}")
    
    # Load dataset
    data_path = config.get('data', {}).get('dataset_path', 'data/dataset.npz')
    data = np.load(data_path, allow_pickle=True)
    
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
    
    # Build datasets
    def build_dataset(observables, times, t_decoh, gamma, fit_normalizer=None):
        data_dict = {
            'observables': observables,
            'times': times,
            't_decoh': t_decoh,
            'gamma': gamma
        }
        dataset = QuantumTrajectoryDataset(
            data_dict,
            seq_length=seq_length,
            features=features,
            predict_remaining=predict_remaining,
            normalizer=fit_normalizer,
            t_max=t_max
        )
        return dataset
    
    # Train dataset (fit normalizer)
    normalizer = Normalizer()
    # Fit on train data
    train_obs = extract_obj_array(data['train_observables'])
    # Temporarily create dataset to get shape for fitting
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
    
    train_dataset = build_dataset(
        train_obs,
        extract_obj_array(data['train_times']),
        data['train_t_decoh'],
        extract_obj_array(data['train_gamma']),
        fit_normalizer=normalizer
    )
    
    val_dataset = build_dataset(
        extract_obj_array(data['val_observables']),
        extract_obj_array(data['val_times']),
        data['val_t_decoh'],
        extract_obj_array(data['val_gamma']),
        fit_normalizer=normalizer
    )
    
    # Create model
    model_type = config.get('model', {}).get('type', 'transformer')
    model = create_model(model_type, config, n_features)
    
    # Initialize training state
    rng = jax.random.PRNGKey(seed)
    state, train_cfg = create_train_state(model, config, rng, n_features, dataset_size=len(train_dataset))
    
    # Training loop
    train_config = config.get('training', {})
    n_epochs = train_config.get('n_epochs', 100)
    batch_size = train_config.get('batch_size', 32)
    
    best_val_loss = float('inf')
    base_checkpoint_dir = Path(config.get('training', {}).get('checkpoint_dir', 'checkpoints'))
    # Create model-specific checkpoint directory to avoid conflicts
    checkpoint_dir = (base_checkpoint_dir / model_type).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    logger.info(f"Training for {n_epochs} epochs")
    logger.info(f"Batch size: {batch_size}, Learning rate: {train_config.get('learning_rate', 1e-4)}")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create RNG for training
    train_rng = jax.random.PRNGKey(seed + 1000)
    
    # Training history for logging
    train_history = []
    val_history = []
    
    # Main training loop with progress bar
    epoch_pbar = tqdm(range(n_epochs), desc="Training", position=0)
    
    for epoch in epoch_pbar:
        # Train
        train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed + epoch)
        train_rng, epoch_rng = jax.random.split(train_rng)
        state, train_metrics = train_epoch(state, train_loader, train_cfg, epoch_rng, epoch, n_epochs, verbose=True)
        
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
            'mape': float(val_results['metrics']['mape']),
            'coverage': float(val_results['metrics']['coverage'])
        })
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f"{train_metrics['loss']:.4f}",
            'val_mae': f"{val_loss:.4f}",
            'val_r2': f"{val_results['metrics']['r2']:.4f}",
            'best': f"{best_val_loss:.4f}"
        })
        
        # Detailed logging every N epochs or on improvement
        log_interval = max(1, n_epochs // 20)  # Log ~20 times during training
        if (epoch + 1) % log_interval == 0 or val_loss < best_val_loss:
            logger.info(
                f"Epoch {epoch + 1}/{n_epochs}:\n"
                f"  Train - Loss: {train_metrics['loss']:.6f}, "
                f"Base: {train_metrics['base_loss']:.6f}, "
                f"Constraint: {train_metrics['constraint']:.6f}\n"
                f"  Val   - MAE: {val_results['metrics']['mae']:.6f}, "
                f"RMSE: {val_results['metrics']['rmse']:.6f}, "
                f"R²: {val_results['metrics']['r2']:.6f}, "
                f"MAPE: {val_results['metrics']['mape']:.2f}%, "
                f"Coverage: {val_results['metrics']['coverage']:.4f}"
            )
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoints.save_checkpoint(
                checkpoint_dir,
                state,
                epoch,
                keep=3,
                overwrite=True
            )
            logger.info(f"✓ Saved checkpoint at epoch {epoch + 1} (val MAE: {val_loss:.6f})")
    
    epoch_pbar.close()
    logger.info(f"Training completed. Best val MAE: {best_val_loss:.6f}")
    
    # Save training history - use model-specific directory
    base_history_dir = Path(config.get('training', {}).get('history_dir', 'runs'))
    history_dir = base_history_dir / model_type
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / 'training_history.npz'
    np.savez(
        history_path,
        train_history=train_history,
        val_history=val_history,
        best_val_loss=best_val_loss
    )
    logger.info(f"Training history saved to {history_path}")
    
    # Plot training history
    try:
        from src.utils.plotting import plot_training_history
        plot_path = history_dir / 'training_history.png'
        plot_training_history(train_history, val_history, save_path=str(plot_path))
        logger.info(f"Training history plot saved to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not plot training history: {e}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary:")
    logger.info(f"  Best validation MAE: {best_val_loss:.6f}")
    logger.info(f"  Final validation R²: {val_history[-1]['r2']:.6f}")
    logger.info(f"  Final validation coverage: {val_history[-1]['coverage']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
