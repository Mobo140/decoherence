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

from experiments.e11_scaling import run_e11a, run_e11b

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


def _defaults(func):
    return {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_run_e11a_defaults_reproduce_the_published_run():
    d = _defaults(run_e11a)
    assert d["seed"] == 42
    assert d["window"] == 20, "published E11a uses window 20"
    assert d["out_path"] is None, "by default it must write the champion CSV"
    assert d["eval_groups"] is None, (
        "by default evaluation must use the training configurations, as published"
    )


def test_r4_scores_disjoint_configurations():
    """R4 measures configuration overlap, so its unseen arm must genuinely
    use a different generator seed -- not the training configurations."""
    src = (EXPERIMENTS / "r4_config_overlap.py").read_text()
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "run_e11a"
    ]
    assert calls, "no arm calls found"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert {"out_path", "seed", "eval_groups"} <= kwargs, (
            f"line {call.lineno}: R4 arm is missing out_path/seed/eval_groups"
        )
    assert "CONFIG_SEED_OFFSET" in src
    from experiments.r4_config_overlap import CONFIG_SEED_OFFSET
    assert CONFIG_SEED_OFFSET != 0, "fresh configs would equal the training ones"


def test_run_e11b_defaults_reproduce_the_published_run():
    d = _defaults(run_e11b)
    assert d["seed"] == 42
    assert d["out_path"] is None


def test_r3_calls_the_experiment_rather_than_copying_it():
    """If R3 rebuilt the configuration itself it could drift from E11."""
    src = (EXPERIMENTS / "r3_multiseed_scaling.py").read_text()
    assert "from experiments.e11_scaling import run_e11a, run_e11b" in src
    assert "run_e11a(" in src and "run_e11b(" in src


def test_r3_never_writes_the_champion_csvs():
    """Its arms run the champion functions, so the output path must be
    redirected on every call."""
    src = (EXPERIMENTS / "r3_multiseed_scaling.py").read_text()
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in {"run_e11a", "run_e11b"}
    ]
    assert calls, "no arm calls found"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "out_path" in kwargs, f"line {call.lineno}: champion CSV not redirected"
        assert "seed" in kwargs, f"line {call.lineno}: arm is not seeded"


def test_r1_mirrors_e6a_hyperparameters():
    """R1 measures E6a's spread, so it must build E6a's predictors."""
    r1 = (EXPERIMENTS / "r1_multiseed_ablation.py").read_text()
    e6a = (EXPERIMENTS / "e6_2qubit_improved.py").read_text()
    for marker in ("hidden_size=128", "num_layers=2", "dropout=0.3",
                   "adaptive_prior_r2_threshold=0.7"):
        assert marker in e6a, f"E6a no longer uses {marker}; update R1"
        assert marker in r1, f"R1 does not mirror E6a's {marker}"
    assert "AblationStudyUseCase(simulator=simulator)" not in r1, (
        "R1 would build default predictors instead of E6a's"
    )


def test_r2_mirrors_e7b_hyperparameters():
    r2 = (EXPERIMENTS / "r2_multiseed_tau.py").read_text()
    e7b = (EXPERIMENTS / "e7_paper2_extended.py").read_text()
    for marker in ("hidden_size=128", "num_layers=2", "dropout=0.3"):
        assert marker in e7b, f"E7b no longer uses {marker}; update R2"
        assert marker in r2, f"R2 does not mirror E7b's {marker}"


def _code_strings(path: Path) -> set:
    """String literals a module actually uses, excluding docstrings."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
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
        "e6_ablation_improved.csv", "e7b_tau_sweep.csv", "e7a_transformer_cross.csv",
        "e8c_cross_system.csv", "e11a_cross_system.csv", "e11b_cross_system.csv",
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
