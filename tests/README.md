# Tests

Run from the repository root with `.venv/bin/python -m pytest`.

## Layout

`tests/unit/` mirrors the packages under `src/` and `experiments/`; one
module (or one class) is under test at a time. Unit tests are fast and use
the deterministic fake simulator instead of QuTiP. A unit test may run at
most one tiny QuTiP simulation or train for a few epochs on a toy dataset.

    tests/unit/
        application/   use cases (generate dataset, train, backtest, ...)
        data/          dataset, windows, scenarios, observation, features
        domain/        value objects
        evaluation/    metrics, statistics, backtest
        experiments/   experiments.common and experiment helpers
        models/        predictors and the model registry
        persistence/   trajectory stores, model stores, champion registries
        physics/       Lindblad systems, measures, observables
        simulation/    QuTiP simulator, variants, registry
        training/      loop, losses, standardiser, trainer
        test_pipeline.py, test_reproducibility.py   (src/pipeline.py, src/reproducibility.py)

`tests/integration/` runs several layers together: simulation, dataset,
training and evaluation, `pipeline.run_experiment`, the experiment harness,
the notebook, the workbench API, replication fidelity, the experiment
catalog against the scripts, and the golden outputs
(`scripts/golden_outputs.json`).

## Shared pieces

- `tests/support.py`: plain builders and test doubles (toy datasets,
  `make_system_config`, `ExponentialDecaySimulator`, `CountingSimulator`,
  `ScriptedSimulator`, `snapshot_protected_dirs`).
- `tests/conftest.py`: the fixtures that hand them out (`toy_dataset`,
  `small_dataset`, `small_config`, `scenario_configs` for scenarios A-E,
  `fake_simulator`, `single_thread`, `trajectory_store`, `results_dir`,
  `isolated_cwd`) and the automatic `integration` marker.
- `tests/integration/conftest.py`: autouse guards. Every integration test
  runs on one torch thread and fails if it changes `experiments/results/`,
  `data/trajectories/` or `checkpoints/`.

Tests never write into `data/`, `checkpoints/` or `experiments/results/`;
use `tmp_path`, `monkeypatch`, or the `results_dir`, `trajectory_store` and
`isolated_cwd` fixtures. `git status` must be clean after a run.

## Markers

`pytest.ini` has `--strict-markers`, so a marker must be declared there.

- `integration`: added automatically to everything under `tests/integration/`.
- `slow`: more than about five seconds; add `@pytest.mark.slow` by hand.

## Selecting tests

    pytest                          # everything
    pytest tests/unit               # unit tests only
    pytest -m "not integration"     # same, by marker
    pytest tests/integration        # integration tests only
    pytest -m "not slow"            # skip the slow ones
    pytest -m "not slow and not integration"   # fastest loop

## Negative tests

The code must fail loudly with a clear error and never return a wrong
result silently. Every test module keeps its positive tests first and ends
with a section headed `# --- Negative cases ---` (or a
`class TestInvalidInputs`). That section holds the error paths: invalid
values, unknown aliases, wrong shapes, degenerate or empty data, corrupted
files, incompatible combinations. Assert the exception type and a
distinctive part of the message (`pytest.raises(..., match=...)`); for
results that are defined as NaN or a cache miss, assert exactly that value.

Name tests `test_<unit>_<condition>_<expected>`, for example
`test_get_loss_unknown_alias_raises_key_error_listing_valid_ones`.
Modules without error-path tests have no such section yet; add it at the
bottom of the file when the first negative test is written.
