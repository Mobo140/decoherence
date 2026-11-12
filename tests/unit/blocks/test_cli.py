"""blocks.cli: generate a dataset file from a recipe file."""
import json

import numpy as np
import pytest

from src.blocks import cli
from src.blocks.recipe import Param, Recipe, block, recipe_to_dict


def _recipe():
    return Recipe(
        system=block("system.single_qubit", omega=Param("w")),
        channels=(block("channels.jump", op_type="sigma_minus", gamma=0.5),),
        initial=block("initial.random_pure"),
        observe=(block("observe.pauli_means"), block("observe.purity"),
                 block("observe.coherence_l1")),
        target=block("target.coherence_crossing", threshold=0.5),
        t_max=4.0, dt=0.25,
        sampler=block("samplers.uniform", w=(0.5, 1.5)))


def _write(tmp_path, data, name="recipe.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _args(tmp_path, recipe_path, *extra):
    return ["generate", str(recipe_path), "--seed", "3",
            "--cache-dir", str(tmp_path / "states"), *extra]


# --- Negative cases ---


def test_missing_recipe_file_is_reported(tmp_path, capsys):
    code = cli.main(_args(
        tmp_path, tmp_path / "none.json", "--n", "2",
        "--out", str(tmp_path / "d.npz")))
    assert code == 2
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "d.npz").exists()


def test_invalid_json_is_reported(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    code = cli.main(_args(
        tmp_path, path, "--n", "2", "--out", str(tmp_path / "d.npz")))
    assert code == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_invalid_recipe_names_the_field(tmp_path, capsys):
    data = recipe_to_dict(_recipe())
    data["channels"][0]["params"]["rate"] = 1.0
    code = cli.main(_args(
        tmp_path, _write(tmp_path, data), "--n", "2",
        "--out", str(tmp_path / "d.npz")))
    assert code == 2
    assert "channels[0]" in capsys.readouterr().err


@pytest.mark.parametrize("n", ["0", "-3", "x", "1.5"])
def test_bad_trajectory_count_exits(tmp_path, capsys, n):
    path = _write(tmp_path, recipe_to_dict(_recipe()))
    with pytest.raises(SystemExit) as info:
        cli.main(_args(tmp_path, path, "--n", n,
                       "--out", str(tmp_path / "d.npz")))
    assert info.value.code == 2
    assert "--n" in capsys.readouterr().err


def test_missing_required_options_exit(tmp_path):
    path = _write(tmp_path, recipe_to_dict(_recipe()))
    with pytest.raises(SystemExit):
        cli.main(["generate", str(path), "--n", "2"])


@pytest.mark.parametrize("option, value", [
    ("--window-length", "0"), ("--horizon", "-1"),
    ("--train-ratio", "1.5"), ("--seed", "-1")])
def test_bad_window_option_exits(tmp_path, option, value):
    path = _write(tmp_path, recipe_to_dict(_recipe()))
    with pytest.raises(SystemExit):
        cli.main(_args(tmp_path, path, "--n", "2", option, value,
                       "--out", str(tmp_path / "d.npz")))


def test_output_must_end_in_npz(tmp_path, capsys):
    path = _write(tmp_path, recipe_to_dict(_recipe()))
    code = cli.main(_args(
        tmp_path, path, "--n", "2", "--out", str(tmp_path / "d.txt")))
    assert code == 2
    assert ".npz" in capsys.readouterr().err


def test_a_recipe_without_target_cannot_make_a_dataset(tmp_path, capsys):
    data = recipe_to_dict(_recipe())
    data["target"] = None
    code = cli.main(_args(
        tmp_path, _write(tmp_path, data), "--n", "2",
        "--out", str(tmp_path / "d.npz")))
    assert code == 2
    assert "no target" in capsys.readouterr().err
    assert not (tmp_path / "d.npz").exists()


# --- Positive run ---


def test_generate_writes_a_dataset_and_prints_progress(tmp_path, capsys):
    path = _write(tmp_path, recipe_to_dict(_recipe()))
    out = tmp_path / "sub" / "d.npz"
    code = cli.main(_args(
        tmp_path, path, "--n", "2", "--out", str(out),
        "--window-length", "4", "--horizon", "1.0",
        "--samples-per-trajectory", "2"))
    printed = capsys.readouterr().out.splitlines()
    assert code == 0
    assert printed[:2] == ["1/2", "2/2"]
    with np.load(out) as data:
        assert data["sequences"].shape[1] == 4
        assert data["sequences"].shape[2] == len(data["feature_names"])
        assert len(data["t_obs"]) == len(data["sequences"])
        assert "context__J" in data.files
        metadata = json.loads(str(data["metadata"]))
    assert metadata["run_seed"] == 3
    assert metadata["recipe"]["system"]["params"]["omega"] == {"param": "w"}
    assert len(list((tmp_path / "states").glob("*.npz"))) == 2


def test_the_same_arguments_give_the_same_dataset(tmp_path):
    path = _write(tmp_path, recipe_to_dict(_recipe()))
    outs = []
    for name in ("a.npz", "b.npz"):
        out = tmp_path / name
        assert cli.main(_args(
            tmp_path, path, "--n", "2", "--out", str(out),
            "--window-length", "4")) == 0
        outs.append(out)
    with np.load(outs[0]) as a, np.load(outs[1]) as b:
        np.testing.assert_array_equal(a["sequences"], b["sequences"])
