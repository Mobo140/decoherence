"""Catalog consistency.

The catalog is what the workbench shows, so a label that disagrees with
the papers or a CSV that no longer exists is a bug the user sees.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from src.application.experiment_catalog import (
    CATALOG,
    RESULTS_DIR,
    list_csv_paths,
)

REPO = Path(__file__).resolve().parents[2]
PAPER2 = REPO / "paper" / "paper_2" / "main.tex"
VALID_STATUS = {"cited", "superseded", "negative", "replication"}


def test_ids_are_unique():
    ids = [e.id for e in CATALOG]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"


def test_every_status_is_known():
    bad = {e.id: e.status for e in CATALOG if e.status not in VALID_STATUS}
    assert not bad, f"unknown status values: {bad}"


def test_every_declared_csv_exists():
    missing = {
        e.id: [c for c in e.csv_globs if not (RESULTS_DIR / c).exists()]
        for e in CATALOG
    }
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"catalog points at missing result files: {missing}"


def test_list_csv_paths_agrees_with_the_declaration():
    for entry in CATALOG:
        assert len(list_csv_paths(entry)) == len(entry.csv_globs), entry.id


def test_every_module_is_importable():
    broken = {}
    for entry in CATALOG:
        try:
            importlib.import_module(entry.module)
        except Exception as exc:          # pragma: no cover - failure path
            broken[entry.id] = f"{type(exc).__name__}: {exc}"
    assert not broken, f"catalog modules that do not import: {broken}"


def test_cited_entries_name_a_paper():
    bad = [e.id for e in CATALOG
           if e.status == "cited" and e.paper in ("", "-")]
    assert not bad, f"marked cited but no paper given: {bad}"


def test_uncited_entries_do_not_claim_a_paper():
    bad = [
        e.id for e in CATALOG
        if e.status in {"negative", "replication"} and e.paper not in ("", "-")
    ]
    assert not bad, f"marked uncited but a paper is claimed: {bad}"


def test_replications_live_outside_the_experiment_numbering():
    """R1-R3 check other experiments; numbering them E22-E24 after task ids
    left a phantom gap at E15-E21."""
    for entry in CATALOG:
        if entry.status == "replication":
            assert entry.id.startswith("R"), entry.id
        else:
            assert not entry.id.startswith("R"), entry.id


def test_every_experiment_script_has_a_catalog_entry_in_its_own_folder():
    """A script the catalog does not know is invisible in the workbench, and
    a catalog module outside the folders is a stale path."""
    folders = ("paper1", "paper2", "checks", "new")
    scripts = {
        ".".join(p.relative_to(REPO).with_suffix("").parts)
        for folder in folders
        for p in (REPO / "experiments" / folder).glob("[er][0-9]*.py")
    }
    modules = {e.module for e in CATALOG}
    assert scripts - modules == set(), (
        f"scripts without a catalog entry: {sorted(scripts - modules)}")
    assert modules - scripts == set(), (
        f"catalog modules that are not scripts: {sorted(modules - scripts)}")


@pytest.mark.skipif(not PAPER2.exists(), reason="paper source not present")
def test_cited_ids_appear_in_the_paper_they_claim():
    """A cited entry whose label the paper never mentions is mislabelled."""
    text = PAPER2.read_text()
    missing = []
    for entry in CATALOG:
        if entry.status != "cited" or "2" not in entry.paper:
            continue
        labels = [p for p in entry.id.split("/")]
        if not any(re.search(rf"\b{re.escape(lbl)}\b", text) for lbl in labels):
            missing.append(entry.id)
    assert not missing, (
        f"cited in the catalog but absent from Paper 2: {missing}")
