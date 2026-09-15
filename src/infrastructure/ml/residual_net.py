"""BiLSTM residual corrector network.

Extracted verbatim from lstm_predictor.py (task 19) -- no behaviour change.
Serialisation stores only ``state_dict()`` tensors plus scalar config, never a
reference to this class, so moving it does not affect existing checkpoints.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Neural network (residual corrector on top of physics estimate)
# ---------------------------------------------------------------------------

class _ResidualNet(nn.Module):
    """BiLSTM that predicts residual corrections to the physics estimate."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        n_physics: int = 7,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # physics scalars: t_obs, slope, log_coh, physics_remaining, ols_r2,
        #                  interaction_code, dissipator_code, [optional J]
        self.n_physics = n_physics
        enc_dim = hidden_size * 2 + n_physics

        self.regression_head = nn.Sequential(
            nn.Linear(enc_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_norm: torch.Tensor,
        slope_norm: torch.Tensor,
        logc_norm: torch.Tensor,
        phys_norm: torch.Tensor,
        r2_norm: torch.Tensor,
        int_norm: torch.Tensor,
        dis_norm: torch.Tensor,
        j_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        extras = [
            t_norm.unsqueeze(1),
            slope_norm.unsqueeze(1),
            logc_norm.unsqueeze(1),
            phys_norm.unsqueeze(1),
            r2_norm.unsqueeze(1),
            int_norm.unsqueeze(1),
            dis_norm.unsqueeze(1),
        ]
        if j_norm is not None:
            extras.append(j_norm.unsqueeze(1))
        extras = extras[: self.n_physics]
        pooled = torch.cat([pooled, *extras], dim=1)

        residual  = self.regression_head(pooled).squeeze(-1)
        risk_logit = self.risk_head(pooled).squeeze(-1)
        return residual, risk_logit
