"""The constructor endpoints of the workbench."""
from __future__ import annotations

import copy
import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from src.application import workbench_server

EXAMPLES = pathlib.Path(workbench_server.EXAMPLES)


def _example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text("utf-8"))


@pytest.fixture
def client(tmp_path):
    app = workbench_server.create_app(
        recipes_dir=tmp_path / "recipes", datasets_dir=tmp_path / "datasets")
    return TestClient(app)


class _FakeJobs:
    def __init__(self):
        self.started = None

    def start(self, module, args=()):
        self.started = (module, list(args))
        return f"started {module}"

    def as_dict(self):
        return {"status": "running", "log": "", "module": "src.blocks.cli"}


# --- Negative cases ---


def test_validate_reports_the_message_of_a_bad_recipe(client):
    recipe = _example("kondo_xxz")
    recipe["channels"][0]["params"]["rate"] = 1.0
    data = client.post("/api/recipe/validate", json={"recipe": recipe}).json()
    assert data["ok"] is False
    assert data["error"].startswith("channels[0]:")


def test_validate_rejects_modifier_first(client):
    recipe = _example("kondo_xxz")
    recipe["channels"].reverse()
    data = client.post("/api/recipe/validate", json={"recipe": recipe}).json()
    assert not data["ok"] and "before any channel producer" in data["error"]


def test_validate_without_a_recipe_is_422(client):
    assert client.post("/api/recipe/validate", json={}).status_code == 422


def test_preview_of_an_invalid_recipe_is_422_with_the_message(client):
    recipe = _example("kondo_xxz")
    del recipe["system"]
    response = client.post(
        "/api/recipe/preview", json={"recipe": recipe, "n": 1})
    assert response.status_code == 422
    assert "system: the field is missing" in response.json()["detail"]


def test_preview_with_a_bad_value_is_422(client):
    recipe = _example("scenario_b_shots")
    recipe["channels"][0]["params"]["op_type"] = "BAD"
    response = client.post(
        "/api/recipe/preview", json={"recipe": recipe, "n": 1})
    assert response.status_code == 422
    assert "BAD" in response.json()["detail"]


def test_preview_with_too_many_steps_is_422(client):
    recipe = _example("scenario_b_shots")
    recipe["t_max"], recipe["dt"] = 600.0, 0.1
    response = client.post(
        "/api/recipe/preview", json={"recipe": recipe, "n": 1})
    assert response.status_code == 422
    assert "6000 time steps" in response.json()["detail"]


def test_preview_with_sampled_steps_too_many_is_422(client):
    recipe = _example("kondo_xxz")
    recipe["t_max"] = {"param": "T"}
    recipe["sampler"]["params"]["T"] = [1000.0, 2000.0]
    recipe["dt"] = 0.1
    response = client.post(
        "/api/recipe/preview", json={"recipe": recipe, "n": 2})
    assert response.status_code == 422
    assert "time steps" in response.json()["detail"]


@pytest.mark.parametrize("n", [0, 6, -1])
def test_preview_n_out_of_range_is_422(client, n):
    response = client.post(
        "/api/recipe/preview",
        json={"recipe": _example("kondo_xxz"), "n": n})
    assert response.status_code == 422


def test_preview_negative_seed_is_422(client):
    response = client.post(
        "/api/recipe/preview",
        json={"recipe": _example("kondo_xxz"), "n": 1, "seed": -1})
    assert response.status_code == 422


@pytest.mark.parametrize("name", [
    "", "a b", "../x", "a/b", "a.json", "x" * 65, "ü", "a\n"])
def test_save_with_a_bad_name_is_422(client, tmp_path, name):
    response = client.post(
        "/api/recipes",
        json={"name": name, "recipe": _example("kondo_xxz")})
    assert response.status_code == 422
    assert not (tmp_path / "recipes").exists()


def test_save_of_an_invalid_recipe_is_422(client):
    response = client.post(
        "/api/recipes", json={"name": "bad", "recipe": {"system": 1}})
    assert response.status_code == 422


@pytest.mark.parametrize("name", [
    "..%2f..%2fetc%2fpasswd", "..%5c..", "a%20b", "x" * 65])
def test_load_with_a_bad_name_is_422_or_404(client, name):
    assert client.get(f"/api/recipes/{name}").status_code in (404, 422)


def test_load_of_an_unknown_name_is_404(client):
    assert client.get("/api/recipes/nothing").status_code == 404


def test_dataset_with_a_bad_name_is_422_and_starts_nothing(
        client, monkeypatch):
    fake = _FakeJobs()
    monkeypatch.setattr(workbench_server, "_jobs", fake)
    response = client.post("/api/recipe/dataset", json={
        "name": "../x", "recipe": _example("kondo_xxz"), "n": 2})
    assert response.status_code == 422 and fake.started is None


def test_dataset_without_a_target_is_422(client, monkeypatch):
    fake = _FakeJobs()
    monkeypatch.setattr(workbench_server, "_jobs", fake)
    recipe = _example("scenario_b_shots")
    recipe["target"] = None
    response = client.post(
        "/api/recipe/dataset", json={"name": "x", "recipe": recipe, "n": 2})
    assert response.status_code == 422 and fake.started is None


@pytest.mark.parametrize("field, value", [
    ("n", 0), ("window_length", 0), ("horizon", 0), ("train_ratio", 1.0),
    ("seed", -1)])
def test_dataset_with_a_bad_option_is_422(client, monkeypatch, field, value):
    fake = _FakeJobs()
    monkeypatch.setattr(workbench_server, "_jobs", fake)
    body = {"name": "x", "recipe": _example("kondo_xxz"), "n": 2}
    body[field] = value
    assert client.post(
        "/api/recipe/dataset", json=body).status_code == 422
    assert fake.started is None


# --- Positive cases ---


def test_blocks_lists_every_stage(client):
    data = client.get("/api/blocks").json()
    assert list(data["stages"]) == [
        "system", "channels", "initial", "evolve", "observe", "target",
        "samplers"]
    jump = next(b for b in data["stages"]["channels"]
                if b["fn"] == "channels.jump")
    assert jump["role"] == "producer"
    op = next(p for p in jump["params"] if p["name"] == "op_type")
    assert op["choices"] == ["sigma_minus", "sigma_z"]
    assert data["evolve_output"]["evolve.sequential_record"] == "record"


@pytest.mark.parametrize("name", ["kondo_xxz", "scenario_b_shots"])
def test_validate_accepts_the_examples(client, name):
    data = client.post(
        "/api/recipe/validate", json={"recipe": _example(name)}).json()
    assert data == {"ok": True, "error": None}


def test_preview_of_the_kondo_example(client):
    data = client.post("/api/recipe/preview", json={
        "recipe": _example("kondo_xxz"), "n": 2, "seed": 1}).json()
    assert len(data["trajectories"]) == 2 and data["elapsed_s"] > 0
    assert data["series_names"] == [
        "sigma_x1", "sigma_y1", "sigma_z1", "sigma_x2", "sigma_y2",
        "sigma_z2", "purity", "coherence_l1"]
    for row in data["trajectories"]:
        assert set(row["params"]) == {"J"}
        assert 0.2 <= row["params"]["J"] <= 1.0
        assert set(row["target"]) == {"time", "censored"}
        assert len(row["times"]) == 61
        assert set(row["series"]) == set(data["series_names"])
        assert all(len(v) == 61 for v in row["series"].values())


def test_preview_of_the_shots_example_is_downsampled(client):
    data = client.post("/api/recipe/preview", json={
        "recipe": _example("scenario_b_shots"), "n": 1, "seed": 4}).json()
    row = data["trajectories"][0]
    assert data["series_names"] == [
        "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"]
    assert row["params"] == {}
    assert len(row["times"]) == 201


def test_preview_downsamples_long_runs_to_400_points(client):
    recipe = _example("scenario_b_shots")
    recipe["t_max"], recipe["dt"] = 100.0, 0.1
    row = client.post("/api/recipe/preview", json={
        "recipe": recipe, "n": 1}).json()["trajectories"][0]
    assert len(row["times"]) == 400
    assert row["times"][0] == 0.0 and row["times"][-1] == pytest.approx(100.0)
    assert all(len(v) == 400 for v in row["series"].values())


def test_preview_is_reproducible(client):
    body = {"recipe": _example("kondo_xxz"), "n": 2, "seed": 5}
    a = client.post("/api/recipe/preview", json=body).json()
    b = client.post("/api/recipe/preview", json=body).json()
    assert a["trajectories"] == b["trajectories"]


def test_preview_of_a_record_recipe_has_no_target(client):
    recipe = _example("scenario_b_shots")
    recipe["evolve"] = {
        "fn": "evolve.sequential_record", "params": {"basis": "z"}}
    recipe["observe"] = [{"fn": "observe.record_series"}]
    recipe["target"] = None
    recipe["t_max"], recipe["dt"] = 2.0, 0.1
    row = client.post("/api/recipe/preview", json={
        "recipe": recipe, "n": 1}).json()["trajectories"][0]
    assert row["target"] is None and list(row["series"]) == ["outcome_q1"]


def test_save_list_load_round_trip(client, tmp_path):
    recipe = _example("kondo_xxz")
    saved = client.post(
        "/api/recipes", json={"name": "mine-1_a", "recipe": recipe}).json()
    assert saved == {"name": "mine-1_a", "saved": True}
    assert (tmp_path / "recipes" / "mine-1_a.json").is_file()
    listed = client.get("/api/recipes").json()
    assert listed["saved"] == ["mine-1_a"]
    assert listed["examples"] == ["kondo_xxz", "scenario_b_shots"]
    loaded = client.get("/api/recipes/mine-1_a").json()
    assert loaded["name"] == "mine-1_a"
    assert loaded["recipe"] == recipe


def test_examples_load_without_being_saved(client):
    loaded = client.get("/api/recipes/scenario_b_shots").json()
    assert loaded["recipe"] == _example("scenario_b_shots")


def test_listing_before_any_save_is_empty(client):
    assert client.get("/api/recipes").json()["saved"] == []


def test_dataset_saves_the_recipe_and_starts_the_cli(
        client, tmp_path, monkeypatch):
    fake = _FakeJobs()
    monkeypatch.setattr(workbench_server, "_jobs", fake)
    recipe = _example("kondo_xxz")
    data = client.post("/api/recipe/dataset", json={
        "name": "ds1", "recipe": recipe, "n": 7, "seed": 3,
        "window_length": 12, "include_censored": False}).json()
    module, args = fake.started
    assert module == "src.blocks.cli" and args[0] == "generate"
    saved = tmp_path / "recipes" / "ds1.json"
    assert json.loads(saved.read_text()) == recipe
    assert args[1] == str(saved)
    assert args[args.index("--n") + 1] == "7"
    assert args[args.index("--window-length") + 1] == "12"
    assert args[args.index("--out") + 1] == str(tmp_path / "datasets/ds1.npz")
    assert "--exclude-censored" in args
    assert data["status"] == "running" and data["message"].startswith(
        "started")


def test_the_environment_variable_sets_the_recipes_dir(
        tmp_path, monkeypatch):
    monkeypatch.setenv(workbench_server.RECIPES_ENV, str(tmp_path / "env"))
    client = TestClient(workbench_server.create_app())
    client.post("/api/recipes", json={
        "name": "e", "recipe": copy.deepcopy(_example("kondo_xxz"))})
    assert (tmp_path / "env" / "e.json").is_file()

