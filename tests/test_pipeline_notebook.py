"""notebooks/pipeline.ipynb runs from top to bottom without Jupyter."""
from __future__ import annotations

import json
import os
import pathlib
import re

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after selecting the backend)

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pipeline.ipynb"
RESULTS = ROOT / "experiments" / "results"
TRAJECTORIES = ROOT / "data" / "trajectories"
STAGES = 10


def _cells(kind: str) -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == kind]


def _snapshot() -> tuple[list[tuple[str, int]], int]:
    results = sorted((p.name, p.stat().st_mtime_ns) for p in RESULTS.iterdir())
    cache = len(os.listdir(TRAJECTORIES)) if TRAJECTORIES.exists() else 0
    return results, cache


def test_notebook_has_one_markdown_cell_per_stage_in_order():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    stages = [int(m.group(1)) for text in _cells("markdown")
              for m in re.finditer(r"^## (\d+)\. ", text, re.MULTILINE)]
    assert stages == list(range(STAGES))


def test_planned_observation_models_are_text_only():
    text = "\n".join(_cells("markdown"))
    for planned in ('"shots"', '"sequential"', '"local"'):
        assert planned in text
    assert not any(
        planned in source for source in _cells("code")
        for planned in ("shots", "sequential", "local"))


def test_code_cells_run_in_order_and_write_nothing(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    threads = torch.get_num_threads()
    before = _snapshot()
    namespace = {"__name__": "pipeline_notebook",
                 "display": lambda *args, **kwargs: None}
    try:
        for number, source in enumerate(_cells("code")):
            lines = [line for line in source.splitlines()
                     if not line.lstrip().startswith(("%", "!"))]
            exec(compile("\n".join(lines), f"<cell {number}>", "exec"),
                 namespace)
    finally:
        plt.close("all")
        torch.set_num_threads(threads)
    assert _snapshot() == before
    # Stage 8 ends with a table of two seeds, two arms and two conditions.
    assert len(namespace["table"]) == 8


@pytest.mark.parametrize("name", [
    "r13_independent_test.csv", "r22_ar1_noise_training.csv",
    "r25_esd_data_scaling.csv"])
def test_published_csvs_read_by_the_notebook_exist(name):
    assert (RESULTS / name).exists()
