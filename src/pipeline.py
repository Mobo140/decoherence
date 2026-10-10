"""Experiment arms as data, and the function that runs one.

An `ExperimentConfig` names every choice of one experiment arm through the
registries: which systems are simulated and by which simulator, how the
windows are cut, how the training set is augmented, which feature set and
model are trained with which loss, and under which observation models and
on which data the trained model is scored. `run_experiment` executes it
with the same lower-level calls, in the same order, that the experiment
scripts used to make by hand:

    system configurations -> simulation -> windows -> augmentation
    -> feature set -> model -> training -> observation -> metrics

so a script that declares its arms gets the numbers it got before.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from src.application import generate_dataset
from src.data import dataset as dataset_lib
from src.data import features
from src.data import observation
from src.data import scenarios
from src.domain import value_objects
from src.evaluation import backtest
from src.evaluation import metrics
from src.evaluation import stats
from src.models import registry as model_registry
from src.persistence import stores
from src.simulation import registry as simulator_registry
from src.training import trainer

# Constructor arguments or extra columns as (name, value) pairs; a tuple of
# pairs keeps the dataclasses hashable.
Params = tuple[tuple[str, object], ...]

CACHES = ("none", "disk", "memory")
_MEMORY_STORE = "memory trajectory store"


@dataclasses.dataclass(frozen=True)
class SimulatorSpec:
    """Which simulator produces the trajectories.

    Attributes:
        alias: Key of `simulator_registry.SIMULATORS`.
        params: Constructor arguments of the simulator.
        seed_offset: If not None, the simulator is also given
            `seed=run seed + seed_offset` (the stochastic-gamma simulator
            draws its own noise from it).
    """

    alias: str = "plain"
    params: Params = ()
    seed_offset: int | None = None


@dataclasses.dataclass(frozen=True)
class DataSpec:
    """How systems are drawn, simulated and cut into windows.

    The systems are either the scenarios `scenarios` (drawn by
    `scenarios.build_configs`) or, if `coupling` is set, one group of
    two-qubit TFIM systems with that coupling (`fixed_coupling_configs`).

    Attributes:
        name: Prefix of the name under which a trajectory store keeps the
            run's trajectories (followed by the seed).
        scenarios: Labels of `scenarios.SCENARIOS`; their configurations are
            concatenated in this order.
        coupling: Coupling J of a fixed-coupling group, or None.
        coupling_sweep: All J values of a sweep, in drawing order. The
            groups of one sweep are drawn one after another from a single
            generator, so the configurations of a group depend on the
            groups before it; this lists them. Must contain `coupling`.
        interaction: Value of `InteractionType` of a fixed-coupling group,
            or None for the default (TFIM). Not for scenarios, which fix
            their interaction.
        n_per_scenario: Configurations drawn per scenario (or in the
            fixed-coupling group).
        n_trajectories: If set, only the first N configurations are
            simulated; with the same `n_per_scenario` the smaller sets are
            nested in the larger one.
        criterion: Value of `DecoherenceCriterion` that replaces the
            scenario's default, or None to keep it.
        simulator: The simulator.
        cache: Where trajectories are kept between datasets: "none",
            "disk" (the shared content-addressed cache, plain simulator
            only) or "memory" (for the seed's run only).
        window_length: Time steps per window.
        horizon: Risk horizon, also the horizon of the backtest.
        samples_per_trajectory: Windows drawn per trajectory.
        min_window_gap: Minimum distance in time steps between the window
            ends sampled from one trajectory; 0 for none.
        train_ratio: Share of trajectories in the train split.
        val_ratio: Share of trajectories in the validation split.
        config_seed_offset: Added to the run seed to seed the draw of the
            system configurations.
        data_seed_offset: Added to the run seed to seed the simulation
            (initial states) and the windows.
    """

    name: str
    scenarios: tuple[str, ...] = ()
    coupling: float | None = None
    coupling_sweep: tuple[float, ...] = ()
    interaction: str | None = None
    n_per_scenario: int = 100
    n_trajectories: int | None = None
    criterion: str | None = None
    simulator: SimulatorSpec = SimulatorSpec()
    cache: str = "none"
    window_length: int = 50
    horizon: float = 1.0
    samples_per_trajectory: int = 5
    min_window_gap: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    config_seed_offset: int = 0
    data_seed_offset: int = 0

    def __post_init__(self) -> None:
        if self.coupling is None:
            unknown = set(self.scenarios) - set(scenarios.SCENARIOS)
            if not self.scenarios or unknown:
                raise ValueError(
                    f"Scenarios must be labels of {sorted(scenarios.SCENARIOS)}"
                    f", got {self.scenarios!r}")
        elif self.scenarios or self.coupling not in self.coupling_sweep:
            raise ValueError(
                "A coupling group takes no scenarios and must be part of "
                "coupling_sweep")
        if self.interaction is not None:
            if self.coupling is None:
                raise ValueError(
                    "An interaction is chosen for a coupling group only")
            value_objects.InteractionType(self.interaction)
        if self.cache not in CACHES:
            raise ValueError(f"cache must be one of {CACHES}")
        if self.cache == "disk" and self.simulator.alias != "plain":
            raise ValueError(
                "The disk cache key does not name the simulator variant, "
                "so only the plain simulator may use it")
        if self.criterion is not None:
            value_objects.DecoherenceCriterion(self.criterion)


@dataclasses.dataclass(frozen=True)
class Augmentation:
    """Noisy copies added to the train and validation windows.

    Attributes:
        copies: Noisy copies of every window.
        phi_max: 0 for white noise; otherwise the noise is AR(1) with a
            lag-one correlation drawn per window from U[0, phi_max].
        seed_offset: The copies are drawn from a generator seeded with the
            run seed plus this offset.
    """

    copies: int
    phi_max: float = 0.0
    seed_offset: int = 7


@dataclasses.dataclass(frozen=True)
class Bootstrap:
    """A 95% interval of a regression metric from resampled trajectories.

    The windows of one trajectory share its decoherence time, so whole
    groups of windows with equal true T2 are drawn with replacement.

    Attributes:
        metric: Key of the regression metrics in `metrics.METRICS`.
        resamples: Number of bootstrap resamples.
    """

    metric: str = "r2"
    resamples: int = 1000


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    """One condition under which a trained model is scored.

    Attributes:
        observation: Key of `observation.OBSERVATIONS`.
        sigma: Noise level passed to the observation model.
        phi: Noise correlation passed to the observation model.
        data: Independent data to score on (all its windows are test
            windows), or None for the test split of the experiment's own
            dataset.
        bootstrap: If set, the row also holds `<metric>_lo`, `<metric>_hi`
            and `n_groups` (trajectories scored), drawn with a generator
            seeded with the run seed.
        tags: Columns that describe this condition in the result row.
    """

    observation: str = "exact"
    sigma: float = 0.0
    phi: float = 0.0
    data: DataSpec | None = None
    bootstrap: Bootstrap | None = None
    tags: Params = ()


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """One experiment arm: data, training and the conditions it is scored on.

    Attributes:
        data: The experiment's own dataset: the training data (unless
            `train_data` is set) and the base of the "own" test windows.
        model: Key of `model_registry.MODELS`.
        model_overrides: Constructor arguments that replace the alias's
            defaults.
        loss: Key of the regression losses.
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate.
        features: Key of `features.FEATURE_SETS` applied to the training
            and test windows, or None to keep every column.
        train_data: If set, the model is trained on this dataset instead
            of `data` (the noisy-rate simulator of R21); the "own" test
            windows still come from `data`.
        augmentation: Noisy copies of the training windows, or None.
        metrics: Keys of `metrics.METRICS`, one result column each.
        tests: The conditions the trained model is scored on, one result
            row each, in this order.
        tags: Columns that describe the arm in every result row.
    """

    data: DataSpec
    model: str
    model_overrides: Params = ()
    loss: str = "huber"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    features: str | None = None
    train_data: DataSpec | None = None
    augmentation: Augmentation | None = None
    metrics: tuple[str, ...] = ("r2", "mae", "auroc")
    tests: tuple[EvalSpec, ...] = (EvalSpec(),)
    tags: Params = ()


def _memoised(
    cache: dict[Hashable, object],
    key: Hashable,
    build: Callable[[], object],
):
    """Returns cache[key], building and storing it on first use."""
    if key not in cache:
        cache[key] = build()
    return cache[key]


def system_configs(
    spec: DataSpec, seed: int
) -> list[value_objects.SystemConfig]:
    """Draws the system configurations of a data spec.

    Args:
        spec: The data spec.
        seed: Run seed; `spec.config_seed_offset` is added to it.

    Returns:
        The configurations, in simulation order.
    """
    config_seed = seed + spec.config_seed_offset
    if spec.coupling is None:
        groups = scenarios.build_configs(
            n_per_scenario=spec.n_per_scenario, seed=config_seed)
        configs = [c for label in spec.scenarios for c in groups[label]]
    else:
        configs = _coupling_group(spec, config_seed)
    if spec.criterion is not None:
        configs = scenarios.with_criterion(
            configs, value_objects.DecoherenceCriterion(spec.criterion))
    return configs[:spec.n_trajectories]


def _coupling_group(
    spec: DataSpec, config_seed: int
) -> list[value_objects.SystemConfig]:
    """Draws the sweep up to the spec's coupling and returns its last group."""
    interaction = value_objects.InteractionType.TFIM
    if spec.interaction is not None:
        interaction = value_objects.InteractionType(spec.interaction)
    rng = np.random.default_rng(config_seed)
    last = spec.coupling_sweep.index(spec.coupling)
    for coupling in spec.coupling_sweep[:last + 1]:
        configs = scenarios.fixed_coupling_configs(
            spec.n_per_scenario, coupling, rng, interaction=interaction)
    return configs


def _simulator(spec: SimulatorSpec, seed: int):
    params = dict(spec.params)
    if spec.seed_offset is not None:
        params["seed"] = seed + spec.seed_offset
    return simulator_registry.build_simulator(spec.alias, **params)


def _store(spec: DataSpec, cache: dict[Hashable, object]):
    if spec.cache == "disk":
        return scenarios.trajectory_store()
    if spec.cache == "memory":
        return _memoised(
            cache, _MEMORY_STORE, stores.MemoryTrajectoryStore)
    return None


def generate(
    spec: DataSpec, seed: int, cache: dict[Hashable, object]
) -> dataset_lib.Dataset:
    """Simulates and windows the data of a spec.

    The result is kept in `cache`, so arms that share a data spec and seed
    simulate it once.

    Args:
        spec: The data spec.
        seed: Run seed.
        cache: Per-seed memory shared by the arms of one run.

    Returns:
        The labelled windows.
    """
    def build() -> dataset_lib.Dataset:
        use_case = generate_dataset.GenerateDatasetUseCase(
            _simulator(spec.simulator, seed), store=_store(spec, cache))
        return use_case.execute(generate_dataset.GenerateDatasetCommand(
            configs=system_configs(spec, seed),
            window_length=spec.window_length,
            horizon=spec.horizon,
            samples_per_trajectory=spec.samples_per_trajectory,
            min_window_gap=spec.min_window_gap,
            train_ratio=spec.train_ratio,
            val_ratio=spec.val_ratio,
            seed=seed + spec.data_seed_offset,
            dataset_name=f"{spec.name}_{seed}"))

    return _memoised(cache, (spec, seed), build)


def _select(
    config: ExperimentConfig, dataset: dataset_lib.Dataset
) -> dataset_lib.Dataset:
    if config.features is None:
        return dataset
    return features.select(dataset, config.features)


def _train(
    config: ExperimentConfig, training_set: dataset_lib.Dataset, seed: int
):
    model = model_registry.build_model(
        config.model, **dict(config.model_overrides))
    return trainer.train(
        model, training_set, epochs=config.epochs,
        batch_size=config.batch_size, lr=config.learning_rate, seed=seed,
        loss=config.loss)


def _test_set(
    config: ExperimentConfig,
    spec: EvalSpec,
    seed: int,
    cache: dict[Hashable, object],
) -> tuple[dataset_lib.Dataset, float]:
    """Builds the windows of one test condition and their horizon."""
    if spec.data is None:
        data, spec_data = generate(config.data, seed, cache), config.data
    else:
        data = generate(spec.data, seed, cache).as_test_only()
        spec_data = spec.data
    data = _select(config, data)
    data = observation.OBSERVATIONS[spec.observation](
        data, sigma=spec.sigma, phi=spec.phi, rng=np.random.default_rng(seed))
    return data, spec_data.horizon


def _interval(
    scores: value_objects.BacktestMetrics, bootstrap: Bootstrap, seed: int
) -> dict[str, float]:
    """Bootstraps one metric over the scored trajectories."""
    resamples = stats.group_resample_indices(
        scores.actuals, bootstrap.resamples, np.random.default_rng(seed))
    values = stats.bootstrap_values(
        metrics.METRICS[bootstrap.metric], resamples, scores.actuals,
        scores.predictions)
    low, high = stats.percentile_interval(values)
    return {f"{bootstrap.metric}_lo": low, f"{bootstrap.metric}_hi": high,
            "n_groups": int(len(np.unique(scores.actuals)))}


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    cache: dict[Hashable, object] | None = None,
) -> list[dict[str, object]]:
    """Trains the arm's model once and scores it under every test.

    Order of the calls: draw configurations, simulate, window (the
    training data), augment, select features, build the model, train,
    then for each test generate its data, select features, observe with
    `np.random.default_rng(seed)` and backtest.

    Args:
        config: The arm.
        seed: Seed of the data, the split, the noise and the training.
        cache: Memory of datasets shared between arms of the same seed
            (see `iter_rows`); None for a private one.

    Returns:
        One row per test, in order: `seed`, the arm's and the test's tags,
        one column per metric, the bootstrap columns if asked for,
        `n_samples` (scored windows),
        `n_train_windows` and `n_censored` (of the training data before
        augmentation).
    """
    if cache is None:
        cache = {}
    base = generate(config.train_data or config.data, seed, cache)
    training_set = base
    if config.augmentation is not None:
        aug = config.augmentation
        training_set = observation.noise_augmented(
            base, aug.copies, np.random.default_rng(seed + aug.seed_offset),
            phi_max=aug.phi_max)
    model = _train(config, _select(config, training_set), seed)
    rows = []
    for spec in config.tests:
        test, horizon = _test_set(config, spec, seed, cache)
        scores = backtest.evaluate(
            model, test, horizon=horizon, metrics=config.metrics)
        row = {"seed": seed, **dict(config.tags), **dict(spec.tags)}
        row.update(scores.extra)
        if spec.bootstrap is not None:
            row.update(_interval(scores, spec.bootstrap, seed))
        row["n_samples"] = scores.n_samples
        row["n_train_windows"] = len(base.train_idx)
        row["n_censored"] = base.metadata.get("n_censored")
        rows.append(row)
    return rows


def iter_rows(
    configs: ExperimentConfig | Iterable[ExperimentConfig],
    seeds: Iterable[int],
) -> Iterator[dict[str, object]]:
    """Runs every arm for every seed and yields the rows as they appear.

    Seeds are the outer loop. The arms of one seed share their simulated
    data, which is dropped when the next seed starts.

    Args:
        configs: One arm or the arms in the order of the result rows.
        seeds: Seeds to run.

    Yields:
        The rows of `run_experiment`.
    """
    if isinstance(configs, ExperimentConfig):
        configs = (configs,)
    arms = tuple(configs)
    for seed in seeds:
        cache: dict[Hashable, object] = {}
        for config in arms:
            yield from run_experiment(config, seed, cache)


def run_seeds(
    configs: ExperimentConfig | Sequence[ExperimentConfig],
    seeds: Iterable[int],
) -> pd.DataFrame:
    """Runs every arm for every seed.

    Args:
        configs: One arm or several.
        seeds: Seeds to run.

    Returns:
        A table of the rows of `iter_rows`.
    """
    return pd.DataFrame(list(iter_rows(configs, seeds)))
