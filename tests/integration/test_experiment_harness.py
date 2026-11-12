"""Experiment harness: pipeline configuration, writer and provenance sidecar."""
from __future__ import annotations

import json

from experiments import common
from src import pipeline


def test_write_results_provenance_sidecar_describes_the_run(tmp_path):
    config = pipeline.ExperimentConfig(
        data=pipeline.DataSpec(name="t", scenarios=("B",)), model="lstm_only",
        tags=(("arm", "x"),))
    path = tmp_path / "out.csv"
    common.write_results([{"a": 1}], path, config=[config], seeds=[7, 8])
    sidecar = tmp_path / "out.csv.provenance.json"
    record = json.loads(sidecar.read_text())
    assert record["csv"] == "out.csv"
    assert record["seeds"] == [7, 8]
    assert record["config"][0]["model"] == "lstm_only"
    assert record["config"][0]["data"]["scenarios"] == ["B"]
    assert {"python", "numpy", "git_commit"} <= set(record["environment"])
    assert path.read_bytes() == b"a\r\n1\r\n"
