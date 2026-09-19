from src.application.experiment_catalog import (
    CATALOG,
    by_id,
    catalog_table,
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


def test_catalog_table_rows():
    table = catalog_table()
    assert len(table) == len(CATALOG)
    assert table[0][0] == "E1"


def test_dashboard_html_has_tfim_champion():
    from src.application.workbench_html import dashboard_html, catalog_html
    html = dashboard_html()
    assert "0.575" in html and "TFIM" in html
    assert "E8c" in catalog_html()
