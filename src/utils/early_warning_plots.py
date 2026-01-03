"""Visualization functions for early warning system."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
    delta_labels: Optional[List[str]] = None
):
    """Plot ROC curve for risk classification.
    
    Args:
        y_true: True binary labels (n_samples,) or (n_samples, n_deltas)
        y_probs: Predicted probabilities (same shape as y_true)
        save_path: Path to save figure
        title: Plot title
        delta_labels: Labels for each delta (if multi-delta)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle multi-delta case
    if y_probs.ndim == 2:
        n_deltas = y_probs.shape[1]
        for i in range(n_deltas):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            label = f"Δ={delta_labels[i]} (AUC={roc_auc:.3f})" if delta_labels else f"Delta {i} (AUC={roc_auc:.3f})"
            ax.plot(fpr, tpr, label=label, linewidth=2)
    else:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}", linewidth=2)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
    delta_labels: Optional[List[str]] = None
):
    """Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        delta_labels: Labels for each delta
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle multi-delta case
    if y_probs.ndim == 2:
        n_deltas = y_probs.shape[1]
        for i in range(n_deltas):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            pr_auc = auc(recall, precision)
            label = f"Δ={delta_labels[i]} (AUC={pr_auc:.3f})" if delta_labels else f"Delta {i} (AUC={pr_auc:.3f})"
            ax.plot(recall, precision, label=label, linewidth=2)
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"AUC={pr_auc:.3f}", linewidth=2)
    
    # Baseline (random classifier)
    baseline = np.mean(y_true if y_probs.ndim == 1 else y_true[:, 0])
    ax.axhline(baseline, color='k', linestyle='--', label=f'Baseline={baseline:.3f}', alpha=0.5)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Reliability Diagram"
):
    """Plot reliability diagram (calibration curve).
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
        
        ax.plot(prob_pred, prob_true, 's-', label='Model', markersize=8, linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
        
        # Compute ECE
        ece = 0.0
        for i in range(len(prob_true)):
            bin_mask = (y_probs >= prob_pred[i] - 0.5/n_bins) & (y_probs < prob_pred[i] + 0.5/n_bins)
            bin_count = np.sum(bin_mask)
            if bin_count > 0:
                ece += bin_count / len(y_probs) * np.abs(prob_true[i] - prob_pred[i])
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('True Probability', fontsize=12)
        ax.set_title(f"{title} (ECE={ece:.3f})", fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Calibration curve failed:\n{str(e)}", 
                ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_risk_histogram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Risk Score Distribution"
):
    """Plot histogram of risk scores separated by true class.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate by class
    probs_positive = y_probs[y_true == 1]
    probs_negative = y_probs[y_true == 0]
    
    # Plot histograms
    bins = np.linspace(0, 1, 30)
    ax.hist(probs_negative, bins=bins, alpha=0.6, label='Negative class', color='blue', density=True)
    ax.hist(probs_positive, bins=bins, alpha=0.6, label='Positive class', color='red', density=True)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_lead_time_distribution(
    lead_times: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Lead Time Distribution",
    bins: int = 30
):
    """Plot distribution of lead times.
    
    Args:
        lead_times: Array of lead times
        save_path: Path to save figure
        title: Plot title
        bins: Number of bins for histogram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(lead_times, bins=bins, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(np.mean(lead_times), color='red', linestyle='--', 
                label=f'Mean={np.mean(lead_times):.2f}', linewidth=2)
    ax1.axvline(np.median(lead_times), color='orange', linestyle='--', 
                label=f'Median={np.median(lead_times):.2f}', linewidth=2)
    ax1.set_xlabel('Lead Time', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2.boxplot(lead_times, vert=True)
    ax2.set_ylabel('Lead Time', fontsize=12)
    ax2.set_title('Lead Time Statistics', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Min: {np.min(lead_times):.2f}\n"
    stats_text += f"Q1: {np.percentile(lead_times, 25):.2f}\n"
    stats_text += f"Median: {np.median(lead_times):.2f}\n"
    stats_text += f"Q3: {np.percentile(lead_times, 75):.2f}\n"
    stats_text += f"Max: {np.max(lead_times):.2f}"
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_trajectory_with_risk(
    times: np.ndarray,
    observables: np.ndarray,
    risk_probs: np.ndarray,
    t_obs_points: np.ndarray,
    t_decoh: float,
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
    title: str = "Trajectory with Risk Predictions",
    observable_names: Optional[List[str]] = None
):
    """Plot trajectory with risk predictions over time.
    
    Args:
        times: Full time array for trajectory
        observables: Observable values (n_timesteps, n_features)
        risk_probs: Risk probabilities at observation points
        t_obs_points: Observation times where risk was computed
        t_decoh: True decoherence time
        threshold: Risk threshold for alarm
        save_path: Path to save figure
        title: Plot title
        observable_names: Names of observables
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot observables
    n_features = observables.shape[1]
    obs_names = observable_names or [f"Feature {i}" for i in range(n_features)]
    
    for i in range(min(n_features, 3)):  # Plot up to 3 features
        ax1.plot(times, observables[:, i], label=obs_names[i], linewidth=2)
    
    ax1.axvline(t_decoh, color='red', linestyle='--', label='Decoherence time', linewidth=2)
    ax1.set_ylabel('Observable Value', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot risk probabilities
    ax2.plot(t_obs_points, risk_probs, 'o-', label='Risk probability', 
             markersize=6, linewidth=2, color='purple')
    ax2.axhline(threshold, color='orange', linestyle='--', 
                label=f'Threshold={threshold}', linewidth=2)
    ax2.axvline(t_decoh, color='red', linestyle='--', linewidth=2)
    
    # Mark first alarm
    alarms = t_obs_points[risk_probs >= threshold]
    if len(alarms) > 0:
        first_alarm = alarms[0]
        ax2.axvline(first_alarm, color='green', linestyle=':', 
                    label=f'First alarm (lead={t_decoh-first_alarm:.2f})', linewidth=2)
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Risk Probability', fontsize=12)
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_forecast_comparison(
    true_sequence: np.ndarray,
    pred_sequence: np.ndarray,
    times: np.ndarray,
    t_obs: float,
    save_path: Optional[Path] = None,
    title: str = "Forecast Comparison",
    feature_names: Optional[List[str]] = None
):
    """Plot comparison of true vs predicted future sequences.
    
    Args:
        true_sequence: True future sequence (horizon, n_features)
        pred_sequence: Predicted future sequence (horizon, n_features)
        times: Time points for forecast horizon
        t_obs: Observation time (vertical line)
        save_path: Path to save figure
        title: Plot title
        feature_names: Names of features
    """
    n_features = true_sequence.shape[1]
    feat_names = feature_names or [f"Feature {i}" for i in range(n_features)]
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features), sharex=True)
    if n_features == 1:
        axes = [axes]
    
    for i in range(n_features):
        axes[i].plot(times, true_sequence[:, i], 'o-', label='True', 
                     markersize=6, linewidth=2, color='blue')
        axes[i].plot(times, pred_sequence[:, i], 's-', label='Predicted', 
                     markersize=6, linewidth=2, color='red', alpha=0.7)
        axes[i].axvline(t_obs, color='green', linestyle='--', 
                        label='Observation time', linewidth=1.5, alpha=0.7)
        
        axes[i].set_ylabel(feat_names[i], fontsize=12)
        axes[i].legend(loc='best')
        axes[i].grid(True, alpha=0.3)
        
        # Compute MAE for this feature
        mae = np.mean(np.abs(true_sequence[:, i] - pred_sequence[:, i]))
        axes[i].text(0.02, 0.98, f'MAE={mae:.3f}', transform=axes[i].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[0].set_title(title, fontsize=14)
    axes[-1].set_xlabel('Time', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
