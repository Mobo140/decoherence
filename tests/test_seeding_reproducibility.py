"""Seeding invariants.

Three reproducibility defects were found in this pipeline: training did
not seed the ML framework's RNG, the simulator drew initial states from an
unseeded global stream, and the experiment scripts built their training
commands directly and passed no seed at all. These tests pin the
behaviour so none of the three can come back unnoticed.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.application.train_model import TrainModelCommand
from src.domain.value_objects import (
    DissipatorConfig,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _cfg() -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma=0.4),
        t_max=10.0,
        dt=0.1,
    )


class _RandomSim:
    """Stands in for the real simulator: draws from the global numpy RNG."""

    def simulate(self, config):
        n = int(round(config.t_max / config.dt)) + 1
        times = np.linspace(0.0, config.t_max, n)
        t2 = 2.0 + 4.0 * float(np.random.rand())
        return TrajectoryResult(
            times=times,
            observables={
                "sigma_x": np.random.rand(n).astype(np.float32),
                "coherence_l1": np.exp(-times / t2).astype(np.float32),
                "purity": np.ones(n, dtype=np.float32),
            },
            t_decoh=t2,
            system_config=config,
        )


def _dataset(seed):
    uc = GenerateDatasetUseCase(_RandomSim(), store=None)
    return uc.execute(GenerateDatasetCommand(
        configs=[_cfg() for _ in range(6)],
        window_length=10,
        horizon=1.0,
        samples_per_trajectory=3,
        seed=seed,
        train_ratio=0.5,
        val_ratio=0.25,
    ))


def test_same_seed_gives_an_identical_dataset():
    a, b = _dataset(11), _dataset(11)
    assert np.array_equal(a.sequences, b.sequences)
    assert np.array_equal(a.t_decoh_abs, b.t_decoh_abs)
    assert np.array_equal(a.train_idx, b.train_idx)


def test_the_initial_state_draw_is_seeded_too():
    """The simulator's own randomness must be covered, not just the split.

    This is the defect that made two runs with the same seed differ: the
    split was reproducible while the trajectories were not.
    """
    np.random.seed(999)
    a = _dataset(11)
    np.random.seed(12345)            # different global state before the run
    b = _dataset(11)
    assert np.array_equal(a.t_decoh_abs, b.t_decoh_abs), (
        "trajectory content depended on RNG state outside the command's seed"
    )


def test_different_seeds_give_different_data():
    a, b = _dataset(11), _dataset(12)
    assert not np.array_equal(a.t_decoh_abs, b.t_decoh_abs)


def test_train_model_command_accepts_a_seed():
    assert "seed" in inspect.signature(TrainModelCommand).parameters


def test_training_with_the_same_seed_gives_identical_weights():
    torch = pytest.importorskip("torch")
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    ds = _dataset(5)
    weights = []
    for _ in range(2):
        p = LSTMPredictor(hidden_size=8, num_layers=1, dropout=0.0)
        p.train(ds, n_epochs=1, batch_size=8, verbose=False, seed=123)
        weights.append([w.detach().clone() for w in p._net.parameters()])
    for a, b in zip(*weights):
        assert torch.equal(a, b), "same seed produced different weights"


def test_training_without_a_seed_is_not_pinned():
    """Guards the test above from passing for the wrong reason."""
    torch = pytest.importorskip("torch")
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    ds = _dataset(5)
    weights = []
    for _ in range(2):
        p = LSTMPredictor(hidden_size=32, num_layers=1, dropout=0.0)
        p.train(ds, n_epochs=1, batch_size=8, verbose=False)
        weights.append([w.detach().clone() for w in p._net.parameters()])
    assert any(not torch.equal(a, b) for a, b in zip(*weights))


def _training_command_sites():
    """Every place an experiment script constructs a training command."""
    names = ("TrainModelCommand", "AblationCommand", "CrossSystemCommand",
             "NoiseSweepCommand", "WindowSweepCommand")
    pattern = re.compile(r"\b(" + "|".join(names) + r")\s*\(")
    for path in sorted(EXPERIMENTS.glob("*.py")):
        text = path.read_text()
        for m in pattern.finditer(text):
            i = m.end() - 1
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
            yield path.name, text[:m.start()].count("\n") + 1, m.group(1), text[i:j + 1]


def test_no_experiment_trains_without_a_seed():
    """The scripts build these commands themselves and bypass the use-case
    defaults, so seeding the use case is not enough."""
    unseeded = [
        f"{name}:{line} {cmd}"
        for name, line, cmd, body in _training_command_sites()
        if not re.search(r"\bseed\s*=", body)
    ]
    assert not unseeded, "training command(s) without a seed: " + "; ".join(unseeded)


def test_the_site_scan_actually_finds_sites():
    """Otherwise the test above would hold vacuously."""
    assert len(list(_training_command_sites())) >= 10
