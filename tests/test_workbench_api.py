from fastapi.testclient import TestClient

from src.application.workbench_server import create_app


def _client():
    return TestClient(create_app())


def test_index_has_sidebar_and_screens():
    html = _client().get("/").text
    assert "Результаты" in html
    assert "data-screen=\"runs\"" in html
    assert "Симуляция" in html


def test_meta_has_tfim_champion():
    data = _client().get("/api/meta").json()
    e = next(c for c in data["champions"] if c["scenario"] == "E")
    assert e["r2"] == 0.575
    assert data["cache"]["n_csv"] >= 1


def test_catalog_and_e8_csv():
    cat = _client().get("/api/catalog").json()["experiments"]
    assert any(e["id"] == "E8" for e in cat)
    csv = _client().get("/api/csv/E8").json()
    assert csv["headers"]
    assert csv["rows"]


def test_compare_e8_e14():
    data = _client().post("/api/compare", json={"a": "E8", "b": "E14"}).json()
    assert "E8" in data["text"] and "E14" in data["text"]


def test_models_slots():
    slots = _client().get("/api/models").json()["slots"]
    assert len(slots) == 4


def test_simulate_scenario_a():
    data = _client().post("/api/simulate", json={"scenario": "A", "seed": 1}).json()
    assert data["t_decoh"] > 0
    assert len(data["times"]) == len(data["coherence"])
    assert data["n_qubits"] == 1


def test_simulate_c_sends_l1_and_sigma_x():
    data = _client().post("/api/simulate", json={"scenario": "C", "seed": 42}).json()
    assert data["observable_name"] == "sigma_x"
    assert len(data["observable"]) == len(data["coherence"])
    assert data["coherence"] != data["observable"]


def test_predict_fallback_physics_only():
    data = _client().post("/api/predict", json={"scenario": "A", "seed": 2, "horizon": 1.0, "alarm": 0.7}).json()
    assert "t_pred" in data
    assert data["source"] in ("registry", "physics_only fallback")
    assert 0.0 <= data["risk"] <= 1.0
    assert data["risk_uses_horizon"] is True
