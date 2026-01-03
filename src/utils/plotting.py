"""Plotting utilities for visualization."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List

def save_plot(fig, save_path: str) -> None:
    """Save figure to file.
    
    Args:
        fig: Matplotlib figure
        save_path: Path to save figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "True vs Predicted Decoherence Time"
) -> plt.Figure:
    """Plot scatter plot of true vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax.set_xlabel('True $t_{decoh}$')
    ax.set_ylabel('Predicted $t_{decoh}$')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig

def plot_error_histogram(
    errors: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Prediction Error Distribution"
) -> plt.Figure:
    """Plot histogram of prediction errors.
    
    Args:
        errors: Prediction errors
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='Zero error')
    ax.set_xlabel('Error ($t_{pred} - t_{true}$)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig

def plot_error_vs_gamma(
    errors: np.ndarray,
    gamma_values: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Error vs Dissipation Rate"
) -> plt.Figure:
    """Plot prediction error as function of dissipation rate.
    
    Args:
        errors: Prediction errors
        gamma_values: Dissipation rate values
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(gamma_values, np.abs(errors), alpha=0.5, s=20)
    ax.set_xlabel(r'Dissipation rate $\gamma$')
    ax.set_ylabel('Absolute Error')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_observable_trajectories(
    times: np.ndarray,
    observables_true: Dict[str, np.ndarray],
    observables_pred: Optional[Dict[str, np.ndarray]] = None,
    t_decoh_true: Optional[float] = None,
    t_decoh_pred: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "Observable Trajectories"
) -> plt.Figure:
    """Plot trajectories of observables (true and optionally predicted).
    
    Args:
        times: Time points
        observables_true: Dictionary of true observable trajectories
        observables_pred: Optional dictionary of predicted observable trajectories
        t_decoh_true: True decoherence time
        t_decoh_pred: Predicted decoherence time
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_obs = len(observables_true)
    n_cols = min(3, n_obs)
    n_rows = (n_obs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_obs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (obs_name, obs_values) in enumerate(observables_true.items()):
        ax = axes[idx]
        
        # Plot true trajectory
        ax.plot(times, obs_values, 'b-', label='True', linewidth=2, alpha=0.7)
        
        # Plot predicted trajectory if available
        if observables_pred is not None and obs_name in observables_pred:
            pred_values = observables_pred[obs_name]
            # Interpolate to match times if needed
            if len(pred_values) != len(times):
                from scipy.interpolate import interp1d
                pred_times = np.linspace(times[0], times[-1], len(pred_values))
                interp_func = interp1d(pred_times, pred_values, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                pred_values = interp_func(times)
            ax.plot(times, pred_values, 'r--', label='Predicted', linewidth=2, alpha=0.7)
        
        # Mark decoherence times
        if t_decoh_true is not None:
            ax.axvline(t_decoh_true, color='b', linestyle=':', linewidth=2, 
                      label=f'True $t_{{decoh}}$ = {t_decoh_true:.2f}')
        if t_decoh_pred is not None:
            ax.axvline(t_decoh_pred, color='r', linestyle=':', linewidth=2,
                      label=f'Pred $t_{{decoh}}$ = {t_decoh_pred:.2f}')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'$\\langle {obs_name} \\rangle$')
        ax.set_title(obs_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_obs, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_metrics_comparison(
    times: np.ndarray,
    metrics_true: Dict[str, np.ndarray],
    metrics_pred: Optional[Dict[str, np.ndarray]] = None,
    t_decoh_true: Optional[float] = None,
    t_decoh_pred: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "Metrics Comparison"
) -> plt.Figure:
    """Plot comparison of metrics (purity, coherence, etc.).
    
    Args:
        times: Time points
        metrics_true: Dictionary of true metric trajectories
        metrics_pred: Optional dictionary of predicted metric trajectories
        t_decoh_true: True decoherence time
        t_decoh_pred: Predicted decoherence time
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics_true)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (metric_name, metric_values) in enumerate(metrics_true.items()):
        ax = axes[idx]
        
        # Plot true metric
        ax.plot(times, metric_values, 'b-', label='True', linewidth=2, alpha=0.7)
        
        # Plot predicted metric if available
        if metrics_pred is not None and metric_name in metrics_pred:
            pred_values = metrics_pred[metric_name]
            if len(pred_values) != len(times):
                from scipy.interpolate import interp1d
                pred_times = np.linspace(times[0], times[-1], len(pred_values))
                interp_func = interp1d(pred_times, pred_values, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
                pred_values = interp_func(times)
            ax.plot(times, pred_values, 'r--', label='Predicted', linewidth=2, alpha=0.7)
        
        # Mark decoherence times
        if t_decoh_true is not None:
            ax.axvline(t_decoh_true, color='b', linestyle=':', linewidth=2,
                      label=f'True $t_{{decoh}}$ = {t_decoh_true:.2f}')
        if t_decoh_pred is not None:
            ax.axvline(t_decoh_pred, color='r', linestyle=':', linewidth=2,
                      label=f'Pred $t_{{decoh}}$ = {t_decoh_pred:.2f}')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig


def plot_training_history(
    train_history: List[Dict],
    val_history: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Training History"
) -> plt.Figure:
    """Plot training and validation metrics over epochs.
    
    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    epochs = np.arange(1, len(train_history) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [h['loss'] for h in train_history], 'b-', label='Train Loss', alpha=0.7)
    ax.plot(epochs, [h['mae'] for h in val_history], 'r-', label='Val MAE', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # R²
    ax = axes[0, 1]
    ax.plot(epochs, [h['r2'] for h in val_history], 'g-', label='Val R²', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.set_title('Validation R²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1, 0]
    ax.plot(epochs, [h['rmse'] for h in val_history], 'orange', label='Val RMSE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Validation RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Coverage
    ax = axes[1, 1]
    ax.plot(epochs, [h['coverage'] for h in val_history], 'purple', label='Val Coverage', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Coverage')
    ax.set_title('Validation Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    return fig
