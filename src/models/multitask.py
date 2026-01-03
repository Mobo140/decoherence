"""Multi-task models for early warning system."""
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Optional, Tuple
from .transformer import TransformerEncoder, PositionalEncoding
from .rnn import GRUEncoder, LSTMEncoder
from .heads import RegressionHead, RiskHead, ForecastHead


class MultiTaskTransformer(nn.Module):
    """Multi-task Transformer with separate heads for risk, forecast, and time."""
    
    # Encoder config
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    dropout_rate: float = 0.1
    max_seq_len: int = 500
    use_pos_encoding: bool = True
    pooling: str = 'attention'
    
    # Task config
    enable_risk: bool = True
    enable_forecast: bool = False
    enable_time: bool = True
    
    # Head config
    n_deltas: int = 3
    forecast_horizon: int = 20
    forecast_n_features: int = 3
    head_hidden_dims: tuple = (256, 128, 64)
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Apply multi-task transformer.
        
        Args:
            x: Input sequences (batch_size, seq_len, n_features)
            training: Whether in training mode
            
        Returns:
            Dictionary with outputs:
            - 'risk_logits': (batch_size, n_deltas) if enable_risk
            - 'forecast': (batch_size, horizon, n_features) if enable_forecast
            - 'time': (batch_size,) if enable_time
        """
        batch_size, seq_len, n_features = x.shape
        
        # Input projection
        x = nn.Dense(self.d_model)(x)
        
        # Positional encoding
        if self.use_pos_encoding:
            x = PositionalEncoding(
                d_model=self.d_model,
                max_len=self.max_seq_len
            )(x)
        
        # Transformer encoder layers
        for _ in range(self.n_layers):
            x = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            )(x, training=training)
        
        # Pooling
        if self.pooling == 'mean':
            # Mean pooling
            pooled = jnp.mean(x, axis=1)
        elif self.pooling == 'attention':
            # Attention pooling
            attention_weights = nn.Dense(1)(x)  # (batch_size, seq_len, 1)
            attention_weights = nn.softmax(attention_weights, axis=1)
            pooled = jnp.sum(x * attention_weights, axis=1)  # (batch_size, d_model)
        else:
            # Last token
            pooled = x[:, -1, :]
        
        # Apply task-specific heads
        outputs = {}
        
        if self.enable_risk:
            risk_logits = RiskHead(
                n_deltas=self.n_deltas,
                hidden_dims=self.head_hidden_dims,
                dropout_rate=self.dropout_rate
            )(pooled, training=training)
            outputs['risk_logits'] = risk_logits
        
        if self.enable_forecast:
            forecast = ForecastHead(
                horizon=self.forecast_horizon,
                n_features=self.forecast_n_features,
                hidden_dims=self.head_hidden_dims[:2],  # Smaller head for forecast
                dropout_rate=self.dropout_rate,
                use_decoder=False
            )(pooled, training=training)
            outputs['forecast'] = forecast
        
        if self.enable_time:
            time_pred = RegressionHead(
                hidden_dims=self.head_hidden_dims,
                dropout_rate=self.dropout_rate
            )(pooled, training=training)
            outputs['time'] = time_pred
        
        return outputs


class MultiTaskRNN(nn.Module):
    """Multi-task RNN (GRU/LSTM) with separate heads."""
    
    # Encoder config
    hidden_dim: int = 256
    n_layers: int = 2
    dropout_rate: float = 0.1
    cell_type: str = 'lstm'  # 'gru' or 'lstm'
    
    # Task config
    enable_risk: bool = True
    enable_forecast: bool = False
    enable_time: bool = True
    
    # Head config
    n_deltas: int = 3
    forecast_horizon: int = 20
    forecast_n_features: int = 3
    head_hidden_dims: tuple = (128, 64)
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Apply multi-task RNN.
        
        Args:
            x: Input sequences (batch_size, seq_len, n_features)
            training: Whether in training mode
            
        Returns:
            Dictionary with task outputs
        """
        batch_size, seq_len, n_features = x.shape
        
        # RNN encoder
        if self.cell_type == 'gru':
            encoder = GRUEncoder(
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                dropout_rate=self.dropout_rate
            )
        else:  # lstm
            encoder = LSTMEncoder(
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                dropout_rate=self.dropout_rate
            )
        
        # Encode sequence
        encoded = encoder(x, training=training)  # (batch_size, seq_len, hidden_dim)
        
        # Use last hidden state
        pooled = encoded[:, -1, :]  # (batch_size, hidden_dim)
        
        # Apply task-specific heads
        outputs = {}
        
        if self.enable_risk:
            risk_logits = RiskHead(
                n_deltas=self.n_deltas,
                hidden_dims=self.head_hidden_dims,
                dropout_rate=self.dropout_rate
            )(pooled, training=training)
            outputs['risk_logits'] = risk_logits
        
        if self.enable_forecast:
            forecast = ForecastHead(
                horizon=self.forecast_horizon,
                n_features=self.forecast_n_features,
                hidden_dims=self.head_hidden_dims,
                dropout_rate=self.dropout_rate,
                use_decoder=False
            )(pooled, training=training)
            outputs['forecast'] = forecast
        
        if self.enable_time:
            time_pred = RegressionHead(
                hidden_dims=self.head_hidden_dims,
                dropout_rate=self.dropout_rate
            )(pooled, training=training)
            outputs['time'] = time_pred
        
        return outputs
