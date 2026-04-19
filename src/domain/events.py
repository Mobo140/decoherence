"""Domain events.

Events are immutable records that something significant happened in the domain.
They can be used for logging, auditing, or triggering side-effects (e.g.
applying a correction pulse when RiskThresholdExceeded fires).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)

@dataclass(frozen=True)
class DecoherenceDetected:
    """Fired when the coherence/purity measure crosses its threshold."""

    t_decoh: float
    criterion: str       # 'coherence' | 'purity'
    threshold: float
    occurred_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class RiskThresholdExceeded:
    """Fired when the predictor's risk score exceeds the alarm threshold.

    Consumers may use this event to schedule a correction pulse or log
    an early-warning alert.
    """

    t_obs: float
    risk_score: float
    alarm_threshold: float
    recommended_action: str = "apply_correction_pulse"
    occurred_at: datetime = field(default_factory=_utcnow)
