"""Evaluation metrics."""
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, t_max: float = 10.0) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        t_max: Maximum time (for coverage metric)
        
    Returns:
        Dictionary of metrics
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # MAE
    mae = np.mean(abs_errors)
    
    # RMSE
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # MAPE (with clipping to avoid division by zero)
    relative_errors = abs_errors / (np.abs(y_true) + 1e-8)
    mape = np.mean(relative_errors) * 100.0
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    
    # Relative error (clipped)
    rel_error = np.mean(np.clip(relative_errors, 0.0, 1.0))
    
    # Coverage: fraction of predictions within delta of true value
    delta = 0.05 * t_max  # 5% of max time
    coverage = np.mean(abs_errors < delta)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        'relative_error': float(rel_error),
        'coverage': float(coverage)
    }


def compute_risk_metrics(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute metrics for risk classification.
    
    Args:
        y_probs: Predicted probabilities (n_samples,) or (n_samples, n_deltas)
        y_true: True binary labels (same shape as y_probs)
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Handle multi-delta case
    if y_probs.ndim == 2:
        # Compute metrics for each delta and average
        metrics = {}
        for i in range(y_probs.shape[1]):
            delta_metrics = compute_risk_metrics(y_probs[:, i], y_true[:, i], threshold)
            for key, val in delta_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(val)
        
        # Average metrics
        return {key: float(np.mean(vals)) for key, vals in metrics.items()}
    
    # Single delta case
    y_pred = (y_probs >= threshold).astype(int)
    
    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_probs)
    except:
        auroc = 0.5
    
    # AUPRC
    try:
        auprc = average_precision_score(y_true, y_probs)
    except:
        auprc = np.mean(y_true)
    
    # Precision, Recall, F1
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Accuracy
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def compute_tpr_at_fpr(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    max_fpr: float = 0.05
) -> float:
    """Compute TPR (True Positive Rate) at a given maximum FPR.
    
    Args:
        y_probs: Predicted probabilities
        y_true: True binary labels
        max_fpr: Maximum false positive rate
        
    Returns:
        TPR at max_fpr
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        # Find TPR at max_fpr
        idx = np.where(fpr <= max_fpr)[0]
        if len(idx) > 0:
            return float(tpr[idx[-1]])
        return 0.0
    except:
        return 0.0


def compute_lead_time_metrics(
    y_probs: np.ndarray,
    t_obs: np.ndarray,
    t_decoh: np.ndarray,
    threshold: float = 0.5,
    trajectory_ids: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute lead time metrics for early warning.
    
    Args:
        y_probs: Predicted probabilities (n_samples,)
        t_obs: Observation times (n_samples,)
        t_decoh: Decoherence times (n_samples,)
        threshold: Alarm threshold
        trajectory_ids: Optional trajectory IDs to group samples
        
    Returns:
        Dictionary with lead time metrics
    """
    # Compute alarms
    alarms = y_probs >= threshold
    
    if trajectory_ids is None:
        # Treat each sample independently
        # Lead time: t_decoh - t_obs when alarm is raised
        lead_times = []
        false_alarms = 0
        
        for i in range(len(y_probs)):
            if alarms[i]:
                if t_decoh[i] >= t_obs[i]:
                    # True alarm
                    lead_times.append(t_decoh[i] - t_obs[i])
                else:
                    # False alarm (event already occurred)
                    false_alarms += 1
        
        if len(lead_times) > 0:
            mean_lead_time = np.mean(lead_times)
            median_lead_time = np.median(lead_times)
            min_lead_time = np.min(lead_times)
            max_lead_time = np.max(lead_times)
        else:
            mean_lead_time = 0.0
            median_lead_time = 0.0
            min_lead_time = 0.0
            max_lead_time = 0.0
        
        false_alarm_rate = false_alarms / len(y_probs) if len(y_probs) > 0 else 0.0
        
    else:
        # Group by trajectory
        unique_trajs = np.unique(trajectory_ids)
        lead_times = []
        false_alarms = 0
        n_detected = 0
        
        for traj_id in unique_trajs:
            mask = trajectory_ids == traj_id
            traj_probs = y_probs[mask]
            traj_t_obs = t_obs[mask]
            traj_t_decoh = t_decoh[mask][0]  # Same for all samples in trajectory
            
            # Find first alarm
            alarm_indices = np.where(traj_probs >= threshold)[0]
            if len(alarm_indices) > 0:
                first_alarm_idx = alarm_indices[0]
                t_alarm = traj_t_obs[first_alarm_idx]
                
                if t_alarm < traj_t_decoh:
                    # Valid early warning
                    lead_times.append(traj_t_decoh - t_alarm)
                    n_detected += 1
                else:
                    # False alarm
                    false_alarms += 1
        
        if len(lead_times) > 0:
            mean_lead_time = np.mean(lead_times)
            median_lead_time = np.median(lead_times)
            min_lead_time = np.min(lead_times)
            max_lead_time = np.max(lead_times)
        else:
            mean_lead_time = 0.0
            median_lead_time = 0.0
            min_lead_time = 0.0
            max_lead_time = 0.0
        
        detection_rate = n_detected / len(unique_trajs) if len(unique_trajs) > 0 else 0.0
        false_alarm_rate = false_alarms / len(unique_trajs) if len(unique_trajs) > 0 else 0.0
    
    return {
        'mean_lead_time': float(mean_lead_time),
        'median_lead_time': float(median_lead_time),
        'min_lead_time': float(min_lead_time),
        'max_lead_time': float(max_lead_time),
        'false_alarm_rate': float(false_alarm_rate),
        'detection_rate': float(detection_rate) if trajectory_ids is not None else 1.0
    }


def compute_calibration_metrics(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Compute calibration metrics (ECE - Expected Calibration Error).
    
    Args:
        y_probs: Predicted probabilities
        y_true: True binary labels
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    try:
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
        
        # ECE: Expected Calibration Error
        ece = 0.0
        for i in range(len(prob_true)):
            # Weight by fraction of samples in bin
            bin_count = np.sum((y_probs >= i / n_bins) & (y_probs < (i + 1) / n_bins))
            ece += bin_count / len(y_probs) * np.abs(prob_true[i] - prob_pred[i])
        
        return {
            'ece': float(ece),
            'n_bins': n_bins
        }
    except:
        return {
            'ece': 0.0,
            'n_bins': n_bins
        }


def compute_forecast_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for forecasting.
    
    Args:
        y_pred: Predicted sequences (n_samples, horizon, n_features)
        y_true: True sequences (same shape)
        
    Returns:
        Dictionary of metrics
    """
    # MAE and RMSE
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # Per-timestep metrics
    mae_per_step = np.mean(np.abs(errors), axis=(0, 2))  # Average over samples and features
    
    # Per-feature metrics
    mae_per_feature = np.mean(np.abs(errors), axis=(0, 1))  # Average over samples and timesteps
    
    return {
        'forecast_mae': float(mae),
        'forecast_rmse': float(rmse),
        'mae_per_step': mae_per_step.tolist(),
        'mae_per_feature': mae_per_feature.tolist()
    }
