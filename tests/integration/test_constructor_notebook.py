"""notebooks/constructor.ipynb runs from top to bottom without Jupyter."""
from __future__ import annotations

import json
import pathlib

import matplotlib
import pytest

from tests import support

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after selecting the backend)

ROOT = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "constructor.ipynb"


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


def test_notebook_is_a_python3_nbformat_4_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"


def test_notebook_uses_blocks_and_no_simulator_classes():
    source = "\n".join(_code_cells())
    assert "src.blocks" in source and "finite_shots" in source
    assert "QuTipSimulator" not in source


@pytest.mark.slow
def test_code_cells_run_in_order_and_write_nothing(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    before = support.snapshot_protected_dirs()
    namespace = {"__name__": "constructor_notebook"}
    try:
        for number, source in enumerate(_code_cells()):
            exec(compile(source, f"<cell {number}>", "exec"), namespace)
    finally:
        plt.close("all")
    assert support.snapshot_protected_dirs() == before
    assert len(namespace["table"]) == 5
    assert set(namespace["results"]) == {"exact", "N=100", "N=1000"}
