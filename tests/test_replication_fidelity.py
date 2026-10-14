"""A replication must replicate the run it claims to check.

R1 was wrong twice for this reason: first it trained unseeded, then it
built predictors from the base use case (hidden 64, 1 layer, tau=0)
instead of E6a's overrides, and its numbers were compared against E6a's
table anyway. These tests pin the couplings that keep R1 and R3 honest.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from experiments import e6_2qubit_improved
from experiments import e7_paper2_extended
from experiments.e11_scaling import run_e11a, run_e11b
from src.application import run_ablation
from src.domain.value_objects import PredictorVariant

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _defaults(func):
    return {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def _calls(path: Path, name: str) -> list:
    tree = ast.parse(path.read_text())
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) == name
             or getattr(n.func, "attr", None) == name)
    ]


def test_run_e11a_defaults_reproduce_the_published_run():
    d = _defaults(run_e11a)
    assert d["seed"] == 42
    assert d["window"] == 20, "published E11a uses window 20"
    assert d["eval_groups"] is None, (
        "by default evaluation must use the training configurations, as "
        "published"
    )


def test_experiment_functions_do_not_choose_a_file():
    """They return rows; only the script's main writes a CSV, so a
    replication cannot overwrite a champion file by calling them."""
    for func in (run_e11a, run_e11b):
        assert "out_path" not in inspect.signature(func).parameters


def test_r4_scores_disjoint_configurations():
    """R4 measures configuration overlap, so its unseen arm must genuinely
    use a different generator seed -- not the training configurations."""
    calls = _calls(EXPERIMENTS / "r4_config_overlap.py", "run_e11a")
    assert calls, "no arm calls found"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert {"seed", "eval_groups"} <= kwargs, (
            f"line {call.lineno}: R4 arm is missing seed/eval_groups"
        )
    from experiments.r4_config_overlap import CONFIG_SEED_OFFSET
    assert CONFIG_SEED_OFFSET != 0, (
        "fresh configs would equal the training ones")


def test_run_e11b_defaults_reproduce_the_published_run():
    assert _defaults(run_e11b)["seed"] == 42


def test_r3_calls_the_experiment_rather_than_copying_it():
    """If R3 rebuilt the configuration itself it could drift from E11."""
    path = EXPERIMENTS / "r3_multiseed_scaling.py"
    assert _calls(path, "run_e11a") and _calls(path, "run_e11b")
    for name in ("run_e11a", "run_e11b"):
        for call in _calls(path, name):
            assert "seed" in {k.arg for k in call.keywords}, (
                f"line {call.lineno}: arm is not seeded")


def test_r1_replicates_e6a_predictors():
    """R1 measures E6a's spread, so it must build the predictors of E6a."""
    r1 = (EXPERIMENTS / "r1_multiseed_ablation.py").read_text()
    assert "e6_2qubit_improved.improved_ablation_predictor" in r1, (
        "R1 would build default predictors instead of E6a's"
    )
    command = run_ablation.AblationCommand(configs=[])
    for variant, threshold in ((PredictorVariant.LSTM_ONLY, 0.0),
                               (PredictorVariant.PHYSICS_LSTM, 0.7)):
        model = e6_2qubit_improved.improved_ablation_predictor(
            variant, command)
        assert (model.hidden_size, model.num_layers, model.dropout) == (
            128, 2, 0.3)
        assert model.adaptive_prior_r2_threshold == threshold


def test_r2_replicates_e7b_predictors():
    r2 = (EXPERIMENTS / "r2_multiseed_tau.py").read_text()
    assert "e7_paper2_extended.tau_predictor" in r2
    model = e7_paper2_extended.tau_predictor(0.3)
    assert (model.hidden_size, model.num_layers, model.dropout) == (
        128, 2, 0.3)
    assert model.adaptive_prior_r2_threshold == 0.3


def _code_strings(path: Path) -> set:
    """String literals a module actually uses, excluding docstrings."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return {
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and n.value not in docstrings
    }


def test_replications_write_only_their_own_results():
    """A probe that overwrote a champion CSV would destroy a published run.

    Mentioning one in a docstring is fine; naming it in code is not.
    """
    champions = {
        "e6_ablation_improved.csv", "e7b_tau_sweep.csv",
        "e7a_transformer_cross.csv", "e8c_cross_system.csv",
        "e11a_cross_system.csv", "e11b_cross_system.csv",
    }
    for probe in ("r1_multiseed_ablation.py", "r2_multiseed_tau.py",
                  "r3_multiseed_scaling.py"):
        used = _code_strings(EXPERIMENTS / probe)
        clash = used & champions
        assert not clash, f"{probe} names champion CSV(s) in code: {clash}"


def test_the_champion_names_are_current():
    """Guards the test above from passing because the names went stale."""
    from src.application.experiment_catalog import RESULTS_DIR
    for name in ("e6_ablation_improved.csv", "e7b_tau_sweep.csv",
                 "e11a_cross_system.csv", "e11b_cross_system.csv"):
        assert (RESULTS_DIR / name).exists(), f"{name} no longer exists"
