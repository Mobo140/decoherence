"""BiLSTM residual corrector network.

Serialisation stores only ``state_dict()`` tensors plus scalar config, never
a reference to this class, so moving it does not affect checkpoints.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualNet(nn.Module):
    """BiLSTM that predicts a correction to the physics estimate.

    The window is encoded by a bidirectional LSTM and mean-pooled; the
    physics scalars (t_obs, slope, log C, physics remaining time, OLS R^2,
    interaction code, dissipator code and optionally J) are appended before
    the two heads.

    Attributes:
        lstm: The window encoder.
        n_physics: Number of physics scalars the heads expect.
        regression_head: Pooled features -> residual (normalised units).
        risk_head: Pooled features -> risk logit (training auxiliary).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        n_physics: int = 7,
    ) -> None:
        """Builds the network.

        Args:
            n_features: Input channels per time step.
            hidden_size: LSTM hidden size per direction.
            num_layers: Number of LSTM layers.
            dropout: Dropout in the heads and between LSTM layers.
            n_physics: Number of physics scalars appended to the encoding
                (4, 7 or 8; older checkpoints used fewer).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
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
        j_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the network.

        Args:
            x: Standardised windows, shape (N, T, n_features).
            t_norm: Standardised observation time, shape (N,).
            slope_norm: Standardised OLS slope, shape (N,).
            logc_norm: Standardised log coherence at t_obs, shape (N,).
            phys_norm: Standardised physics remaining time, shape (N,).
            r2_norm: Standardised OLS R^2, shape (N,).
            int_norm: Standardised interaction code, shape (N,).
            dis_norm: Standardised dissipator code, shape (N,).
            j_norm: Standardised coupling J, shape (N,); used only by an
                eight-scalar network.

        Returns:
            (residual, risk_logit), each of shape (N,).
        """
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
        extras = extras[:self.n_physics]
        pooled = torch.cat([pooled, *extras], dim=1)

        residual = self.regression_head(pooled).squeeze(-1)
        risk_logit = self.risk_head(pooled).squeeze(-1)
        return residual, risk_logit
