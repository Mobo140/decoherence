"""MLP model for regression."""
import jax.numpy as jnp
from flax import linen as nn
from .heads import RegressionHead


class MLP(nn.Module):
    """Multi-layer perceptron for sequence regression."""
    
    hidden_dims: list = (512, 512, 256, 256, 128)  # Deeper and wider network
    dropout_rate: float = 0.3  # More dropout for regularization
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply MLP model.
        
        Args:
            x: Input sequences (batch_size, seq_len, n_features)
            training: Whether in training mode
            
        Returns:
            Scalar predictions (batch_size,)
        """
        # Flatten sequence
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * n_features)
        
        # Apply MLP layers with better activation
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)  # GELU instead of ReLU
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
            x = nn.LayerNorm()(x)  # Layer normalization for stability
        
        # Output head
        output = nn.Dense(1)(x)
        return jnp.squeeze(output, axis=-1)
