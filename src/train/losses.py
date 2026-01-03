"""Loss functions for training."""
import jax.numpy as jnp
from typing import Optional


def mae_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Mean Absolute Error loss.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        MAE loss
    """
    return jnp.mean(jnp.abs(y_pred - y_true))


def mse_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error loss.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        MSE loss
    """
    return jnp.mean((y_pred - y_true) ** 2)


def huber_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss (smooth L1).
    
    Args:
        y_pred: Predicted values
        y_true: True values
        delta: Threshold parameter
        
    Returns:
        Huber loss
    """
    error = y_pred - y_true
    abs_error = jnp.abs(error)
    quadratic = jnp.clip(abs_error, None, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * quadratic ** 2 + delta * linear)


def constraint_loss(y_pred: jnp.ndarray, t_max: float) -> jnp.ndarray:
    """Constraint loss: penalize predictions outside [0, t_max].
    
    Args:
        y_pred: Predicted values
        t_max: Maximum allowed time
        
    Returns:
        Constraint penalty
    """
    penalty = jnp.mean(jnp.maximum(0.0, -y_pred) + jnp.maximum(0.0, y_pred - t_max))
    return penalty


def smoothness_loss(coeffs: jnp.ndarray, order: int = 1) -> jnp.ndarray:
    """Smoothness penalty for Bernstein coefficients (to avoid "sawtooth" patterns).
    
    Args:
        coeffs: Bernstein coefficients (batch_size, n_coeffs)
        order: Order of derivative (1 for first, 2 for second)
        
    Returns:
        Smoothness penalty
    """
    if order == 1:
        # First derivative: difference between adjacent coefficients
        diff = coeffs[:, 1:] - coeffs[:, :-1]
        return jnp.mean(diff ** 2)
    elif order == 2:
        # Second derivative
        diff1 = coeffs[:, 1:] - coeffs[:, :-1]
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        return jnp.mean(diff2 ** 2)
    else:
        raise ValueError(f"Unsupported order: {order}")


def bce_with_logits_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    pos_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Binary cross-entropy loss with logits (numerically stable).
    
    Args:
        logits: Predicted logits (n_samples,) or (n_samples, n_classes)
        targets: True binary labels (same shape as logits)
        pos_weight: Weight for positive class (for class imbalance)
        
    Returns:
        BCE loss
    """
    # Numerically stable BCE: log(sigmoid(x)) = -softplus(-x)
    # BCE = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
    #     = -[y*(-softplus(-x)) + (1-y)*(-x - softplus(-x))]
    #     = softplus(-x) + y*x - x + y*softplus(-x)
    #     = softplus(x) + y*(x - softplus(x) - x)
    #     = softplus(x) - y*softplus(x) + y*x
    # Simplified: max(x, 0) - x*y + log(1 + exp(-|x|))
    
    max_val = jnp.maximum(logits, 0)
    loss = max_val - logits * targets + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
    
    if pos_weight is not None:
        # Apply positive class weight
        loss = loss * (1 + (pos_weight - 1) * targets)
    
    return jnp.mean(loss)


def focal_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> jnp.ndarray:
    """Focal loss for handling class imbalance.
    
    Args:
        logits: Predicted logits
        targets: True binary labels
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        
    Returns:
        Focal loss
    """
    # Compute probabilities
    p = jax.nn.sigmoid(logits)
    
    # Focal loss: -alpha * (1-p)^gamma * log(p) for positive class
    #             -(1-alpha) * p^gamma * log(1-p) for negative class
    ce_loss = bce_with_logits_loss(logits, targets)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    focal_weight = alpha_t * jnp.power(1 - p_t, gamma)
    loss = focal_weight * ce_loss
    
    return jnp.mean(loss)


def forecast_loss(
    pred: jnp.ndarray,
    true: jnp.ndarray,
    loss_type: str = 'mae',
    smoothness_weight: float = 0.0
) -> jnp.ndarray:
    """Loss for forecasting future sequences.
    
    Args:
        pred: Predicted sequences (batch_size, horizon, n_features)
        true: True sequences (batch_size, horizon, n_features)
        loss_type: 'mae' or 'huber'
        smoothness_weight: Weight for smoothness penalty
        
    Returns:
        Forecast loss
    """
    if loss_type == 'mae':
        base_loss = jnp.mean(jnp.abs(pred - true))
    elif loss_type == 'huber':
        base_loss = huber_loss(pred, true, delta=1.0)
    else:
        base_loss = mae_loss(pred, true)
    
    # Smoothness penalty: penalize large changes in predictions
    if smoothness_weight > 0:
        # Second derivative penalty
        if pred.ndim == 3:
            # (batch_size, horizon, n_features)
            diff1 = pred[:, 1:, :] - pred[:, :-1, :]
            diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
            smoothness_penalty = jnp.mean(diff2 ** 2)
        else:
            smoothness_penalty = 0.0
        
        return base_loss + smoothness_weight * smoothness_penalty
    
    return base_loss


def compute_loss(
    y_pred: jnp.ndarray,
    y_true: jnp.ndarray,
    loss_type: str = 'mae',
    t_max: Optional[float] = None,
    constraint_weight: float = 0.0,
    huber_delta: float = 1.0
) -> tuple:
    """Compute total loss.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        loss_type: Type of base loss ('mae', 'mse', 'huber') - must be 'mae' for JIT
        t_max: Maximum time (for constraint loss)
        constraint_weight: Weight for constraint penalty
        huber_delta: Delta parameter for Huber loss
        
    Returns:
        Tuple of (total_loss, base_loss, constraint_penalty)
    """
    # Base loss - only MAE supported in JIT mode
    base_loss = mae_loss(y_pred, y_true)
    
    # Constraint loss - use multiplication instead of conditionals for JIT
    # If t_max is None or constraint_weight is 0, this will be 0
    t_max_val = t_max if t_max is not None else 1e10
    constraint_penalty = constraint_loss(y_pred, t_max_val)
    # Multiply by constraint_weight to enable/disable
    constraint_term = constraint_weight * constraint_penalty
    
    total_loss = base_loss + constraint_term
    
    return total_loss, base_loss, constraint_penalty


def compute_multitask_loss(
    predictions: dict,
    targets: dict,
    weights: dict,
    config: dict
) -> tuple:
    """Compute multi-task loss.
    
    Args:
        predictions: Dictionary with predictions for each task
            - 'risk_logits': (batch_size, n_deltas)
            - 'forecast': (batch_size, horizon, n_features)
            - 'time': (batch_size,)
        targets: Dictionary with ground truth for each task
            - 'risk_labels': (batch_size, n_deltas)
            - 'future_sequence': (batch_size, horizon, n_features)
            - 'remaining_time': (batch_size,)
        weights: Dictionary with loss weights
            - 'lambda_risk': weight for risk loss
            - 'lambda_forecast': weight for forecast loss
            - 'lambda_time': weight for time loss
        config: Additional config (pos_weight, focal params, etc.)
        
    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
    """
    total_loss = 0.0
    loss_dict = {}
    
    # Risk loss (classification)
    if 'risk_logits' in predictions and 'risk_labels' in targets:
        logits = predictions['risk_logits']
        labels = targets['risk_labels']
        
        use_focal = config.get('use_focal_loss', False)
        if use_focal:
            risk_loss = focal_loss(
                logits,
                labels,
                alpha=config.get('focal_alpha', 0.25),
                gamma=config.get('focal_gamma', 2.0)
            )
        else:
            pos_weight = config.get('pos_weight', None)
            if pos_weight is not None:
                pos_weight = jnp.array(pos_weight)
            risk_loss = bce_with_logits_loss(logits, labels, pos_weight)
        
        loss_dict['risk'] = risk_loss
        total_loss = total_loss + weights.get('lambda_risk', 1.0) * risk_loss
    
    # Forecast loss
    if 'forecast' in predictions and 'future_sequence' in targets:
        forecast_pred = predictions['forecast']
        forecast_true = targets['future_sequence']
        
        fc_loss = forecast_loss(
            forecast_pred,
            forecast_true,
            loss_type=config.get('forecast_loss_type', 'mae'),
            smoothness_weight=config.get('forecast_smoothness_weight', 0.0)
        )
        
        loss_dict['forecast'] = fc_loss
        total_loss = total_loss + weights.get('lambda_forecast', 1.0) * fc_loss
    
    # Time regression loss
    if 'time' in predictions and 'remaining_time' in targets:
        time_pred = predictions['time']
        time_true = targets['remaining_time']
        
        time_loss = mae_loss(time_pred, time_true)
        
        loss_dict['time'] = time_loss
        total_loss = total_loss + weights.get('lambda_time', 1.0) * time_loss
    
    return total_loss, loss_dict
