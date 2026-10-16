"""Seeding invariants across data generation, training and the scripts.

Three reproducibility defects were found in this pipeline: training did
not seed the ML framework's RNG, the simulator drew initial states from an
unseeded global stream, and the experiment scripts built their training
commands directly and passed no seed at all. These tests pin the
behaviour so none of the three can come back unnoticed. The dataset-level
checks are in tests/unit/application/test_generate_dataset.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.data.dataset import Dataset
from src.models.lstm import LSTMPredictor
from tests import support

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def _dataset(seed: int) -> Dataset:
    uc = GenerateDatasetUseCase(support.GlobalRngSimulator(), store=None)
    return uc.execute(GenerateDatasetCommand(
        configs=[support.make_system_config() for _ in range(6)],
        window_length=10,
        horizon=1.0,
        samples_per_trajectory=3,
        seed=seed,
        train_ratio=0.5,
        val_ratio=0.25,
    ))


def test_training_with_the_same_seed_gives_identical_weights():
    ds = _dataset(5)
    weights = []
    for _ in range(2):
        p = LSTMPredictor(hidden_size=8, num_layers=1, dropout=0.0)
        p.train(ds, n_epochs=1, batch_size=8, verbose=False, seed=123)
        weights.append([w.detach().clone() for w in p.net.parameters()])
    for a, b in zip(*weights):
        assert torch.equal(a, b), "same seed produced different weights"


def test_training_without_a_seed_is_not_pinned():
    """Guards the test above from passing for the wrong reason."""
    ds = _dataset(5)
    weights = []
    for _ in range(2):
        p = LSTMPredictor(hidden_size=32, num_layers=1, dropout=0.0)
        p.train(ds, n_epochs=1, batch_size=8, verbose=False)
        weights.append([w.detach().clone() for w in p.net.parameters()])
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
            line = text[:m.start()].count("\n") + 1
            yield path.name, line, m.group(1), text[i:j + 1]


def test_experiment_scripts_always_pass_a_seed_to_training_commands():
    """The scripts build these commands themselves and bypass the use-case
    defaults, so seeding the use case is not enough."""
    unseeded = [
        f"{name}:{line} {cmd}"
        for name, line, cmd, body in _training_command_sites()
        if not re.search(r"\bseed\s*=", body)
    ]
    assert not unseeded, (
        "training command(s) without a seed: " + "; ".join(unseeded))


def test_training_command_site_scan_finds_the_sites():
    """Otherwise the test above would hold vacuously."""
    assert len(list(_training_command_sites())) >= 10
