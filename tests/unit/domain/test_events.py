"""Domain events are immutable records."""
from __future__ import annotations

import dataclasses
import datetime

import pytest

from src.domain import events


def test_risk_threshold_exceeded_defaults_to_the_correction_pulse():
    event = events.RiskThresholdExceeded(
        t_obs=2.0, risk_score=0.9, alarm_threshold=0.5)
    assert event.recommended_action == "apply_correction_pulse"
    assert (event.t_obs, event.risk_score, event.alarm_threshold) == (
        2.0, 0.9, 0.5)


def test_risk_threshold_exceeded_is_stamped_with_utc_time():
    before = datetime.datetime.now(tz=datetime.timezone.utc)
    event = events.RiskThresholdExceeded(1.0, 0.8, 0.5)
    after = datetime.datetime.now(tz=datetime.timezone.utc)
    assert before <= event.occurred_at <= after
    assert event.occurred_at.utcoffset() == datetime.timedelta(0)


def test_risk_threshold_exceeded_custom_action_is_kept():
    event = events.RiskThresholdExceeded(
        1.0, 0.8, 0.5, recommended_action="log")
    assert event.recommended_action == "log"


# --- Negative cases ---


def test_risk_threshold_exceeded_is_immutable():
    event = events.RiskThresholdExceeded(1.0, 0.8, 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.risk_score = 0.1


def test_risk_threshold_exceeded_missing_fields_raise_type_error():
    with pytest.raises(TypeError, match="alarm_threshold"):
        events.RiskThresholdExceeded(1.0, 0.8)


def test_risk_threshold_exceeded_does_not_validate_the_risk_score():
    # Debatable, kept: the event is a record, not a PredictionResult.
    assert events.RiskThresholdExceeded(1.0, 2.0, 0.5).risk_score == 2.0
