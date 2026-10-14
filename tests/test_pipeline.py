"""ExperimentConfig / run_experiment: declared arms run the same calls."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src import pipeline
from src.application import generate_dataset
from src.data import observation
from src.data import scenarios
from src.evaluation import backtest
from src.models import registry
from src.simulation import simulator
from src.training import trainer

SEED = 3
TINY_TRANSFORMER = (("d_model", 16), ("n_heads", 2), ("n_layers", 1),
                    ("dim_ff", 32))


def _data(**changes) -> pipeline.DataSpec:
    spec = pipeline.DataSpec(
        name="t", scenarios=("B",), n_per_scenario=10, window_length=20,
        samples_per_trajectory=3)
    return dataclasses.replace(spec, **changes)


def _manual_dataset(spec: pipeline.DataSpec, seed: int):
    configs = scenarios.build_configs(spec.n_per_scenario, seed)["B"]
    use_case = generate_dataset.GenerateDatasetUseCase(
        simulator.QuTipSimulator(), store=None)
    return use_case.execute(generate_dataset.GenerateDatasetCommand(
        configs=configs, window_length=spec.window_length, horizon=1.0,
        samples_per_trajectory=spec.samples_per_trajectory, seed=seed,
        dataset_name="t"))


def test_untrained_arm_matches_the_manual_calls():
    spec = _data()
    config = pipeline.ExperimentConfig(
        data=spec, model="physics_only",
        tests=(pipeline.EvalSpec(),
               pipeline.EvalSpec("gaussian", sigma=0.05)))
    rows = pipeline.run_experiment(config, SEED)

    dataset = _manual_dataset(spec, SEED)
    model = registry.build_model("physics_only")
    noisy = observation.OBSERVATIONS["gaussian"](
        dataset, sigma=0.05, rng=np.random.default_rng(SEED))
    expected = [backtest.evaluate(model, dataset),
                backtest.evaluate(model, noisy)]
    assert [r["r2"] for r in rows] == [m.r2 for m in expected]
    assert [r["mae"] for r in rows] == [m.mae for m in expected]
    assert [r["n_samples"] for r in rows] == [m.n_samples for m in expected]
    assert rows[0]["seed"] == SEED
    assert rows[0]["n_train_windows"] == len(dataset.train_idx)


def test_trained_arm_with_augmentation_matches_the_manual_calls():
    spec = _data()
    config = pipeline.ExperimentConfig(
        data=spec, model="transformer", model_overrides=TINY_TRANSFORMER,
        epochs=1, batch_size=8, learning_rate=1e-3,
        augmentation=pipeline.Augmentation(copies=1, phi_max=0.5),
        metrics=("r2", "auroc", "rmse"))
    (row,) = pipeline.run_experiment(config, SEED)

    dataset = _manual_dataset(spec, SEED)
    augmented = observation.noise_augmented(
        dataset, 1, np.random.default_rng(SEED + 7), phi_max=0.5)
    model = registry.build_model("transformer", **dict(TINY_TRANSFORMER))
    trainer.train(model, augmented, epochs=1, batch_size=8, lr=1e-3,
                  seed=SEED)
    scores = backtest.evaluate(model, dataset)
    assert row["r2"] == scores.r2
    assert row["auroc"] == scores.risk_auroc
    assert row["rmse"] == scores.rmse
    assert "mae" not in row
    assert row["n_train_windows"] == len(dataset.train_idx)


def test_independent_test_uses_its_own_configurations_and_all_windows():
    independent = _data(
        n_per_scenario=6, train_ratio=0.0, val_ratio=0.0,
        config_seed_offset=1000, data_seed_offset=1)
    config = pipeline.ExperimentConfig(
        data=_data(), model="physics_only",
        tests=(pipeline.EvalSpec(data=independent),))
    (row,) = pipeline.run_experiment(config, SEED)

    configs = scenarios.build_configs(6, SEED + 1000)["B"]
    use_case = generate_dataset.GenerateDatasetUseCase(
        simulator.QuTipSimulator(), store=None)
    dataset = use_case.execute(generate_dataset.GenerateDatasetCommand(
        configs=configs, window_length=20, horizon=1.0,
        samples_per_trajectory=3, seed=SEED + 1, train_ratio=0.0,
        val_ratio=0.0, dataset_name="t")).as_test_only()
    scores = backtest.evaluate(registry.build_model("physics_only"), dataset)
    assert row["r2"] == scores.r2
    assert row["n_samples"] == scores.n_samples


def test_tags_become_columns_and_arms_share_their_data(monkeypatch):
    calls = []
    original = generate_dataset.GenerateDatasetUseCase.execute

    def counting(self, command):
        calls.append(command.dataset_name)
        return original(self, command)

    monkeypatch.setattr(
        generate_dataset.GenerateDatasetUseCase, "execute", counting)
    spec = _data()
    arms = [
        pipeline.ExperimentConfig(
            data=spec, model="physics_only", tags=(("arm", name),),
            tests=(pipeline.EvalSpec(tags=(("test", "own"),)),))
        for name in ("a", "b")
    ]
    rows = list(pipeline.iter_rows(arms, [SEED, SEED + 1]))
    assert [(r["seed"], r["arm"], r["test"]) for r in rows] == [
        (3, "a", "own"), (3, "b", "own"), (4, "a", "own"), (4, "b", "own")]
    assert calls == ["t_3", "t_4"]


def test_run_seeds_returns_a_table():
    config = pipeline.ExperimentConfig(data=_data(), model="physics_only")
    table = pipeline.run_seeds(config, [1, 2])
    assert list(table["seed"]) == [1, 2]
    assert {"r2", "mae", "auroc", "n_samples"} <= set(table.columns)


def test_coupling_group_replays_the_generator_of_the_sweep():
    sweep = (0.5, 1.0, 2.0)
    rng = np.random.default_rng(SEED)
    drawn = [scenarios.fixed_coupling_configs(4, j, rng) for j in sweep]
    for coupling, expected in zip(sweep, drawn):
        spec = pipeline.DataSpec(
            name="t", coupling=coupling, coupling_sweep=sweep,
            n_per_scenario=4)
        assert pipeline.system_configs(spec, SEED) == expected


def test_nested_sets_are_prefixes_and_the_criterion_is_replaced():
    big = pipeline.DataSpec(
        name="t", scenarios=("D",), n_per_scenario=8,
        criterion="entanglement_death")
    small = dataclasses.replace(big, n_trajectories=3)
    all_configs = pipeline.system_configs(big, SEED)
    assert pipeline.system_configs(small, SEED) == all_configs[:3]
    assert {c.decoherence_criterion.value for c in all_configs} == {
        "entanglement_death"}


def test_scenario_groups_are_concatenated_in_order():
    spec = pipeline.DataSpec(name="t", scenarios=("C", "B"),
                             n_per_scenario=3)
    groups = scenarios.build_configs(3, SEED)
    assert pipeline.system_configs(spec, SEED) == groups["C"] + groups["B"]


@pytest.mark.parametrize("changes", [
    {"scenarios": ()},
    {"scenarios": ("Z",)},
    {"scenarios": ("B",), "cache": "tape"},
    {"scenarios": ("B",), "criterion": "nonsense"},
    {"scenarios": ("B",), "cache": "disk",
     "simulator": pipeline.SimulatorSpec("correlators")},
    {"scenarios": ("B",), "coupling": 1.0, "coupling_sweep": (1.0,)},
    {"scenarios": (), "coupling": 1.0, "coupling_sweep": (0.5,)},
])
def test_invalid_data_specs_are_rejected(changes):
    with pytest.raises(ValueError):
        pipeline.DataSpec(name="t", **changes)


def test_configs_are_hashable_and_comparable():
    a = pipeline.ExperimentConfig(data=_data(), model="lstm_only")
    b = pipeline.ExperimentConfig(data=_data(), model="lstm_only")
    assert a == b and len({a, b}) == 1


def test_stochastic_simulator_gets_the_offset_seed():
    spec = pipeline.SimulatorSpec(
        "stochastic_gamma", (("s", 0.3), ("tau", 2.0)), seed_offset=13)
    sim = pipeline._simulator(spec, 5)
    twin = pipeline._simulator(spec, 5)
    assert (sim.s, sim.tau) == (0.3, 2.0)
    assert sim.rng.standard_normal() == twin.rng.standard_normal()
    other = pipeline._simulator(spec, 6)
    assert sim.rng.standard_normal() != other.rng.standard_normal()


def test_fixed_coupling_group_takes_an_interaction():
    sweep = (0.5, 1.0)
    spec = pipeline.DataSpec(
        name="t", coupling=1.0, coupling_sweep=sweep, interaction="XXZ",
        n_per_scenario=3, criterion="coherence_energy")
    configs = pipeline.system_configs(spec, SEED)
    assert {c.interaction_type.value for c in configs} == {"XXZ"}
    assert {c.decoherence_criterion.value for c in configs} == {
        "coherence_energy"}
    default = pipeline.system_configs(
        dataclasses.replace(spec, interaction=None, criterion=None), SEED)
    assert {c.interaction_type.value for c in default} == {"TFIM"}


@pytest.mark.parametrize("changes", [
    {"scenarios": ("B",), "interaction": "XXZ"},
    {"scenarios": (), "coupling": 1.0, "coupling_sweep": (1.0,),
     "interaction": "ising"},
])
def test_invalid_interactions_are_rejected(changes):
    with pytest.raises(ValueError):
        pipeline.DataSpec(name="t", **changes)


def test_window_gap_reaches_the_dataset_builder():
    base = _data(n_per_scenario=6, samples_per_trajectory=4)
    gapped = dataclasses.replace(base, min_window_gap=15)
    cache = {}
    dataset = pipeline.generate(gapped, SEED, cache)
    by_trajectory = {}
    for t_obs, t_decoh in zip(dataset.t_obs, dataset.t_decoh_abs):
        by_trajectory.setdefault(float(t_decoh), []).append(float(t_obs))
    steps = [np.diff(np.sort(v)) / 0.1 for v in by_trajectory.values()
             if len(v) > 1]
    assert steps and min(s.min() for s in steps) >= 15 - 1e-6


def test_bootstrap_columns_match_the_manual_interval():
    from src.evaluation import metrics, stats
    independent = _data(
        n_per_scenario=6, train_ratio=0.0, val_ratio=0.0,
        config_seed_offset=1000, data_seed_offset=1)
    config = pipeline.ExperimentConfig(
        data=_data(), model="physics_only",
        metrics=("r2_unguarded", "auroc_pairwise"),
        tests=(pipeline.EvalSpec(
            data=independent,
            bootstrap=pipeline.Bootstrap("r2_unguarded", 20)),))
    (row,) = pipeline.run_experiment(config, SEED)

    cache = {}
    dataset = pipeline.generate(independent, SEED, cache).as_test_only()
    model = registry.build_model("physics_only")
    scores = backtest.evaluate(model, dataset)
    resamples = stats.group_resample_indices(
        scores.actuals, 20, np.random.default_rng(SEED))
    values = stats.bootstrap_values(
        metrics.r2_unguarded, resamples, scores.actuals, scores.predictions)
    low, high = stats.percentile_interval(values)
    assert (row["r2_unguarded_lo"], row["r2_unguarded_hi"]) == (low, high)
    assert row["n_groups"] == len(np.unique(dataset.t_decoh_abs))
    assert row["r2_unguarded"] == metrics.r2_unguarded(
        scores.actuals, scores.predictions)
    assert 0.0 <= row["auroc_pairwise"] <= 1.0
