"""Tests for HamiltonianModelRegistry file names, update rule, and fallback."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.value_objects import DissipatorType, InteractionType
from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
from src.infrastructure.persistence.stores import BestModelRegistry, HamiltonianModelRegistry


def _metrics(r2: float, n: int = 10):
    return SimpleNamespace(r2=r2, mae=0.1, rmse=0.2, mape=1.0, risk_auroc=0.9, n_samples=n)


class TestNames:
    def test_1q_uses_dissipator(self):
        assert HamiltonianModelRegistry._name(
            1, InteractionType.XXZ, DissipatorType.SIGMA_MINUS
        ) == "best_1q_sigma_minus"
        assert HamiltonianModelRegistry._name(
            1, InteractionType.TFIM, DissipatorType.SIGMA_Z
        ) == "best_1q_sigma_z"

    def test_2q_uses_interaction_only(self):
        assert HamiltonianModelRegistry._name(
            2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS
        ) == "best_XXZ"
        assert HamiltonianModelRegistry._name(
            2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS
        ) == "best_TFIM"


class TestMaybeUpdate:
    def test_saves_and_rejects_worse(self, tmp_path):
        reg = HamiltonianModelRegistry(root=tmp_path)
        pred = PhysicsOnlyPredictor(t_max=10.0)
        assert reg.maybe_update(
            pred, _metrics(0.50), 2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS
        )
        assert (tmp_path / "best_XXZ.pt").exists()
        assert (tmp_path / "best_XXZ_meta.json").exists()
        meta = reg.get_meta(2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS)
        assert meta["r2"] == pytest.approx(0.50)

        worse = PhysicsOnlyPredictor(t_max=20.0)
        assert not reg.maybe_update(
            worse, _metrics(0.10), 2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS
        )
        assert reg.get_meta(2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS)["r2"] == pytest.approx(0.50)

        better = PhysicsOnlyPredictor(t_max=15.0)
        assert reg.maybe_update(
            better, _metrics(0.80), 2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS
        )
        assert reg.get_meta(2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS)["r2"] == pytest.approx(0.80)

    def test_tfim_is_separate_slot(self, tmp_path):
        reg = HamiltonianModelRegistry(root=tmp_path)
        pred = PhysicsOnlyPredictor()
        reg.maybe_update(pred, _metrics(0.4), 2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS)
        reg.maybe_update(pred, _metrics(0.6), 2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS)
        assert (tmp_path / "best_XXZ.pt").exists()
        assert (tmp_path / "best_TFIM.pt").exists()
        assert reg.get_meta(2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS)["r2"] == pytest.approx(0.6)

    def test_rejects_nan_or_empty(self, tmp_path):
        reg = HamiltonianModelRegistry(root=tmp_path)
        pred = PhysicsOnlyPredictor()
        assert not reg.maybe_update(
            pred, _metrics(float("nan"), n=0), 2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS
        )
        assert not (tmp_path / "best_TFIM.pt").exists()

    def test_missing_predictor_is_none(self, tmp_path):
        reg = HamiltonianModelRegistry(root=tmp_path)
        assert reg.get_predictor(2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS) is None

    def test_summary_lists_empty_slots(self, tmp_path):
        text = HamiltonianModelRegistry(root=tmp_path).summary()
        assert "best_XXZ" not in text or "2q/XXZ" in text
        assert "2q/XXZ: no model saved yet" in text
        assert "2q/TFIM: no model saved yet" in text
        assert "1q/sigma_minus: no model saved yet" in text


class TestBestModelRegistryUnchanged:
    def test_still_keys_by_qubit_count(self, tmp_path):
        reg = BestModelRegistry(root=tmp_path)
        pred = PhysicsOnlyPredictor()
        assert reg.maybe_update(pred, _metrics(0.3), 2)
        assert (tmp_path / "best_2qubit.pt").exists()
        assert reg.get_meta(2)["r2"] == pytest.approx(0.3)
        assert not reg.maybe_update(pred, _metrics(0.1), 2)
