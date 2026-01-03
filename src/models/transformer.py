"""Transformer encoder model for sequence regression."""
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from .heads import RegressionHead, TwoStageHead


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    d_model: int
    max_len: int = 5000
    
    def setup(self):
        # Precompute positional encodings
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[None, :, :]  # (1, max_len, d_model)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Add positional encoding to input.
        
        Args:
            x: Input sequences (batch_size, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class TransformerEncoder(nn.Module):
    """Transformer encoder block."""
    
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply transformer encoder block.
        
        Args:
            x: Input sequences (batch_size, seq_len, d_model)
            training: Whether in training mode
            
        Returns:
            Encoded sequences (batch_size, seq_len, d_model)
        """
        # Self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            deterministic=not training
        )(x, x)
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_output)
        x = nn.LayerNorm()(x)
        
        # Feed-forward
        ff_output = nn.Dense(self.d_ff)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)
        ff_output = nn.Dense(self.d_model)(ff_output)
        x = x + nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)
        x = nn.LayerNorm()(x)
        
        return x


class Transformer(nn.Module):
    """Transformer encoder model for regression."""
    
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout_rate: float = 0.1
    max_seq_len: int = 500
    use_pos_encoding: bool = True
    pooling: str = 'attention'  # 'mean' or 'attention' - attention is better
    head_hidden_dims: list = (256, 128, 64)  # Deeper head
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply transformer model.
        
        Args:
            x: Input sequences (batch_size, seq_len, n_features)
            training: Whether in training mode
            
        Returns:
            Scalar predictions (batch_size,)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Project input to d_model
        x = nn.Dense(self.d_model)(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        if self.use_pos_encoding:
            pos_enc = PositionalEncoding(d_model=self.d_model, max_len=self.max_seq_len)
            x = pos_enc(x)
        
        # Apply dropout
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Apply transformer encoder layers
        for _ in range(self.n_layers):
            encoder = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            )
            x = encoder(x, training=training)
        
        # Pooling
        if self.pooling == 'mean':
            x = jnp.mean(x, axis=1)  # (batch_size, d_model)
        elif self.pooling == 'attention':
            # Attention pooling: learn a query vector
            query = self.param('pooling_query', nn.initializers.normal(0.02), (1, 1, self.d_model))
            # Compute attention weights
            attn_weights = jnp.einsum('bld,qld->bq', x, query)
            attn_weights = nn.softmax(attn_weights, axis=1)
            x = jnp.einsum('bld,bq->qd', x, attn_weights)
            x = jnp.squeeze(x, axis=0)  # (batch_size, d_model)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Regression head with residual connections
        for i, dim in enumerate(self.head_hidden_dims):
            residual = x if i == 0 and x.shape[-1] == dim else None
            x_new = nn.Dense(dim)(x)
            x_new = nn.gelu(x_new)  # GELU is better than ReLU
            x_new = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x_new)
            if residual is not None:
                x = x_new + residual
            else:
                x = x_new
        
        # Final output layer
        output = nn.Dense(1)(x)
        return jnp.squeeze(output, axis=-1)


class TwoStageTransformer(nn.Module):
    """Two-stage transformer: predict gamma(t) coefficients, then compute t_decoh."""
    
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout_rate: float = 0.1
    max_seq_len: int = 500
    n_coeffs: int = 3
    bernstein_degree: int = 2
    head_hidden_dims: list = (128, 64)
    T_max: float = 10.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply two-stage transformer.
        
        Args:
            x: Input sequences (batch_size, seq_len, n_features)
            training: Whether in training mode
            
        Returns:
            Bernstein coefficients (batch_size, n_coeffs)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Project input to d_model
        x = nn.Dense(self.d_model)(x)
        
        # Add positional encoding
        pos_enc = PositionalEncoding(d_model=self.d_model, max_len=self.max_seq_len)
        x = pos_enc(x)
        
        # Apply dropout
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Apply transformer encoder layers
        for _ in range(self.n_layers):
            encoder = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            )
            x = encoder(x, training=training)
        
        # Mean pooling
        x = jnp.mean(x, axis=1)  # (batch_size, d_model)
        
        # Two-stage head
        for dim in self.head_hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Output: Bernstein coefficients (non-negative)
        coeffs = nn.Dense(self.n_coeffs)(x)
        coeffs = nn.softplus(coeffs)  # Ensure non-negativity
        
        return coeffs
