"""Catalog entries, their CSVs and the flags the workbench relies on."""
import pathlib

from src.application.experiment_catalog import (
    CATALOG,
    by_id,
    compare_summaries,
    list_csv_paths,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_catalog_has_paper_experiments():
    ids = {e.id for e in CATALOG}
    assert {"E1", "E8c", "E14"}.issubset(ids)


def test_e8_csv_exists_and_compare_runs():
    paths = list_csv_paths(by_id("E8c"))
    assert paths
    text = compare_summaries("E8c", "E14")
    assert "E8c" in text and "E14" in text


def test_fast_flag_is_only_offered_to_scripts_that_accept_it():
    # --fast is accepted through experiments.common.parse_args or a script's
    # own parser; R11 is a separate invocation of R7 and opts out.
    for entry in CATALOG:
        if not entry.supports_fast:
            continue
        source = (ROOT / (entry.module.replace(".", "/") + ".py")).read_text()
        assert "common.parse_args(" in source or '"--fast"' in source, (
            f"{entry.id} offers --fast but {entry.module} rejects it")


def test_scripts_on_the_common_parser_offer_fast():
    for entry in CATALOG:
        source = (ROOT / (entry.module.replace(".", "/") + ".py")).read_text()
        if "common.parse_args(" in source and entry.id != "R11":
            assert entry.supports_fast, entry.id
