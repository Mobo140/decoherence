from src.application.experiment_catalog import (
    CATALOG,
    by_id,
    compare_summaries,
    list_csv_paths,
)


def test_catalog_has_paper_experiments():
    ids = {e.id for e in CATALOG}
    assert {"E1", "E8c", "E14"}.issubset(ids)


def test_e8_csv_exists_and_compare_runs():
    paths = list_csv_paths(by_id("E8c"))
    assert paths
    text = compare_summaries("E8c", "E14")
    assert "E8c" in text and "E14" in text
