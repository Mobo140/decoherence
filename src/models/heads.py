"""Output heads for models."""
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple


class RegressionHead(nn.Module):
    """Regression head for predicting scalar values (e.g., remaining time)."""
    
    hidden_dims: tuple = (128, 64)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply regression head.
        
        Args:
            x: Input features (batch_size, feature_dim)
            training: Whether in training mode
            
        Returns:
            Scalar predictions (batch_size,)
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output: single scalar
        output = nn.Dense(1)(x)
        return jnp.squeeze(output, axis=-1)


class RiskHead(nn.Module):
    """Risk classification head for early warning.
    
    Predicts probability of event occurring in each time horizon.
    """
    
    n_deltas: int = 3  # Number of time horizons
    hidden_dims: tuple = (128, 64)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply risk head.
        
        Args:
            x: Input features (batch_size, feature_dim)
            training: Whether in training mode
            
        Returns:
            Logits for each delta (batch_size, n_deltas)
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output: logits for each delta (will apply sigmoid in loss)
        logits = nn.Dense(self.n_deltas)(x)
        return logits


class ForecastHead(nn.Module):
    """Forecasting head for predicting future observables.
    
    Predicts future sequence of observables/metrics over horizon H.
    """
    
    horizon: int = 20  # Number of future timesteps
    n_features: int = 3  # Number of features to forecast
    hidden_dims: tuple = (256, 128)
    dropout_rate: float = 0.1
    use_decoder: bool = False  # Whether to use decoder architecture
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply forecast head.
        
        Args:
            x: Input features (batch_size, feature_dim)
            training: Whether in training mode
            
        Returns:
            Forecasted sequence (batch_size, horizon, n_features)
        """
        if self.use_decoder:
            # Decoder-based forecasting (more complex)
            # Project to sequence
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.relu(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
            
            # Generate sequence
            output = nn.Dense(self.horizon * self.n_features)(x)
            output = jnp.reshape(output, (x.shape[0], self.horizon, self.n_features))
        else:
            # Direct prediction (simpler baseline)
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.relu(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
            
            # Direct output
            output = nn.Dense(self.horizon * self.n_features)(x)
            output = jnp.reshape(output, (x.shape[0], self.horizon, self.n_features))
        
        return output


class TwoStageHead(nn.Module):
    """Two-stage head: predict gamma(t) coefficients, then compute t_decoh."""
    
    n_coeffs: int = 3  # Number of Bernstein coefficients
    hidden_dims: list = (128, 64)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply two-stage head.
        
        Args:
            x: Input features (batch_size, feature_dim)
            training: Whether in training mode
            
        Returns:
            Bernstein coefficients (batch_size, n_coeffs)
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output: Bernstein coefficients (non-negative)
        coeffs = nn.Dense(self.n_coeffs)(x)
        # Use softplus to ensure non-negativity
        coeffs = nn.softplus(coeffs)
        
        return coeffs
