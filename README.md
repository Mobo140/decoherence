# Physics-Informed Early Warning of Quantum Decoherence

> *Given only a short partial trajectory of Pauli observables — without knowledge
> of γ(t) — predict the remaining decoherence time T₂ and emit a calibrated risk
> score, recovering the exact analytical formula when γ is constant and
> outperforming pure-physics and pure-ML baselines when γ(t) is unknown.*

**Paper:** *Physics-Informed Early Warning of Quantum Decoherence Under Unknown
Time-Dependent Dissipation* (in preparation, target: npj Quantum Information /
Physical Review Applied)

---

## Contents

- [Physics-Informed Early Warning of Quantum Decoherence](#physics-informed-early-warning-of-quantum-decoherence)
  - [Contents](#contents)
  - [Physics background](#physics-background)
    - [Supported systems](#supported-systems)
    - [Why γ(t) cannot be assumed known](#why-γt-cannot-be-assumed-known)
    - [Analytical baseline (constant γ, σ₋)](#analytical-baseline-constant-γ-σ)
  - [Model architecture](#model-architecture)
  - [Architecture](#architecture)
  - [Installation](#installation)
  - [Quick start](#quick-start)
    - [1. Interactive UI](#1-interactive-ui)
    - [2. Programmatic API](#2-programmatic-api)
    - [3. Single inference](#3-single-inference)
  - [Running experiments](#running-experiments)
    - [Reproducing a paper table](#reproducing-a-paper-table)
    - [Scenario taxonomy](#scenario-taxonomy)
  - [Analysing results](#analysing-results)
  - [Plugging in a custom loss](#plugging-in-a-custom-loss)
  - [Key parameters](#key-parameters)
  - [Tests](#tests)
  - [Scientific motivation (extended)](#scientific-motivation-extended)

---

## Physics background

A quantum system coupled to an environment evolves according to the **Lindblad
master equation** (ℏ = 1):

$$\dot{\rho} = -i[H,\rho] + \gamma(t)\sum_k \left(L_k\rho L_k^\dagger - \tfrac{1}{2}\{L_k^\dagger L_k,\rho\}\right)$$

The **decoherence time T₂** is the time at which the L1 coherence
(sum of |off-diagonal elements|) falls to 1/e of its initial value.

### Supported systems

| System | Hamiltonian | Jump operators |
| --- | --- | --- |
| 1-qubit | H = ω σ_z | σ₋ (amplitude damping), σ_z (pure dephasing) |
| 2-qubit XXZ | H = J(σˣσˣ+σʸσʸ+σᶻσᶻ) | σ₋ per qubit |
| 2-qubit TFIM | H = J σˣσˣ + h(σᶻ+σᶻ) | σ₋ per qubit |

### Why γ(t) cannot be assumed known

In every physical realisation the dissipation rate is **stochastic and non-stationary**:

| Platform | Source of unknown γ(t) |
| --- | --- |
| Superconducting qubits | 1/f flux noise, TLS fluctuators |
| Spin qubits | Hyperfine interaction with nuclear spin bath |
| Trapped ions | Laser power drift, micromotion fluctuations |
| Photonic systems | Cavity leakage rate drifts |

Here γ(t) is parametrised via **degree-4 Bernstein polynomials** (guaranteed non-negative),
enabling controlled generation of arbitrary smooth dissipation profiles.

### Analytical baseline (constant γ, σ₋)

For constant γ the coherence decays as C(t) = C₀ e^{−γt/2}, so:

```
slope = d(log C)/dt = −γ/2   →   T₂ = −1/slope
```

This OLS-slope formula achieves R² ≈ 0.999 and MAE ≈ 0.02 in this regime.
The LSTM residual corrector extends accuracy to all other regimes.

---

## Model architecture

```
Input observables  (batch, W, F)
      │                            F = 5 (1-qubit) or 8 (2-qubit) — Pauli + L1 + purity
      │                                (+ log C, log purity appended inside the net)
      │
Physics prior ──────────────────────────────────────────────────────┐
  OLS (or Hilbert envelope) of log(coherence_l1) → phys_rem         │
  7 scalars: t_obs, slope, log C, phys_rem, OLS R²,                 │
             interaction_code, dissipator_code  [+ J if enabled]    │
                                                                      │
BiLSTM  (hidden=64, bidirectional, 1 layer)                          │
  → mean pool over W → (batch, 128)                                  │
  → concat scalars → (batch, 135)   old Paper-1 ckpt: 4 scalars/132  │
      │                                                               │
      ├── Regression head → residual δT                              │
      │   remaining = phys_rem + δT  ←──────────────────────────────┘
      │
      └── Risk head → logit (train auxiliary at dataset Δt)
              At predict(): risk = σ(−k · (remaining − horizon)), k=5
              Same mapping as physics_only; the UI slider moves risk.
```

When `use_physics_prior=False` (lstm_only ablation) the physics scalars are zeroed
and the network predicts T₂ directly from sequences.

---

## Architecture

The pipeline is the three-layer picture *physics → observation model →
forecast*. Each step is a function or a registry where a variant is chosen
by a string alias; scripts and the notebook assemble the pipeline from
these only. Details: `notes/architecture.md`; style rules:
`notes/code_guidelines.md`.

```
scenario → simulation → observation → features → windows → model → loss → training → metrics
```

```
.
├── app.py                          # FastAPI workbench, 7 screens
├── src/
│   ├── domain/                     # value objects and ports; no torch/qutip
│   ├── physics/                    # Lindblad systems, measures, Pauli operators
│   ├── simulation/                 # QuTipSimulator; registry SIMULATORS
│   ├── data/                       # scenarios A–E, Dataset, windows,
│   │                               #   observation.py (OBSERVATIONS),
│   │                               #   features.py (FEATURE_SETS)
│   ├── models/                     # predictors; registry MODELS, build_model(alias)
│   ├── training/                   # shared loop, standardisation, LOSSES
│   ├── evaluation/                 # METRICS, backtest, paired statistics
│   ├── persistence/                # trajectory cache, checkpoint stores
│   ├── pipeline.py                 # ExperimentConfig / run_experiment
│   ├── reproducibility.py          # seeding, environment_info()
│   └── application/                # use cases, experiment catalog, workbench
├── experiments/
│   ├── common.py                   # harness: CLI, results path, provenance
│   ├── e1_…e14_*.py, r1_…r27_*.py  # one script per experiment
│   ├── run_paper1.py, run_all.py   # chains of experiments
│   └── results/                    # CSV (+ .provenance.json) of finished runs
├── notebooks/                      # pipeline.ipynb (walkthrough), figures.ipynb
├── scripts/golden_outputs.py       # regression check against golden_outputs.json
├── tests/
├── paper/                          # LaTeX sources of the papers
└── checkpoints/
```

**Registries.** `models.registry.build_model("physics_lstm", **overrides)`,
`simulation.registry.build_simulator("correlators")`, and the alias tables
`OBSERVATIONS` (`exact`, `gaussian`, `ar1`), `FEATURE_SETS`, `LOSSES`,
`evaluation.metrics.METRICS`. A new variant is a new entry, not an `if`.

**Pipeline.** `pipeline.ExperimentConfig` describes one arm (data, model,
loss, epochs, feature set, tests); `pipeline.run_experiment(config, seed)`
returns one row per test, `iter_rows` / `run_seeds` run arms over seeds.

**Harness.** Experiment scripts share `experiments.common`: `parse_args`
(`--seeds`, `--fast`, `--tag`, `--out`), `results_path`, and
`ResultsWriter` / `write_results`, which write the CSV and a
`<csv>.provenance.json` with the configuration, seeds and software
versions. `--fast` is a smoke run on reduced data and writes
`*_fast.csv`, which is git-ignored and never touches a published file.

**Notebook.** `notebooks/pipeline.ipynb` walks through the pipeline stage by
stage with the same functions; it writes nothing to disk.

**Golden check.** `python -m scripts.golden_outputs` recomputes a fixed set
of numbers (simulation, windows, training, metrics) and must print
`golden outputs identical`; it guards refactors against changing results.

The dependency rule: `domain` knows no framework, `application` orchestrates
use cases, `models` and `training` hold PyTorch, `physics` and `simulation`
hold QuTiP, `persistence` holds the file stores.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: `qutip`, `torch`, `pandas`, `fastapi`, `numpy`, `scipy`, `scikit-learn`

---

## Quick start

### 1. Interactive UI

```bash
source .venv/bin/activate
python app.py
# http://localhost:7860
```

| Screen | What it does |
| --- | --- |
| **01 Dashboard** | Champion R² A–E (Paper 1 done, TFIM 0.575 / цель 0.70) |
| **02 Runs** | Каталог E1–E14 и R1–R27, CSV, Compare, Replay `--fast` / full |
| **03 Models** | `HamiltonianModelRegistry` (best_XXZ / best_TFIM / 1q) |
| **04 Simulate** | Пресет сценария A–E (тот же `build_configs`) |
| **05 Train** | Очередь eN в фоне, UI не блокируется |
| **06 Predict** | Окно до T₂ + risk alarm (fallback: physics_only) |
| **07 Backtest** | Таблица чемпионов из CSV статей |

### 2. Programmatic API

```python
from src.domain.value_objects import DissipatorConfig, QubitCount, SystemConfig
from src.simulation.simulator import QuTipSimulator
from src.models import registry
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.application.backtest import BacktestCommand, BacktestUseCase

# 1. Define system with unknown time-dependent γ(t)
config = SystemConfig(
    n_qubits=QubitCount.ONE,
    omega=1.0,
    dissipator=DissipatorConfig.time_dependent(
        coeffs=(0.2, 0.8, 1.2, 0.4, 0.1),   # Bernstein coefficients
    ),
)

# 2. Generate dataset
simulator = QuTipSimulator()
dataset = GenerateDatasetUseCase(simulator).execute(
    GenerateDatasetCommand(configs=[config] * 200, window_length=50)
)

# 3. Train physics+LSTM predictor (aliases: physics_only, stretched_exp,
#    lstm_only, physics_lstm, physics_feature, transformer, transformer_large)
predictor = registry.build_model("physics_lstm")
TrainModelUseCase(predictor).execute(
    TrainModelCommand(dataset=dataset, n_epochs=100, seed=0)
)
# or in one call: trainer.train(predictor, dataset, epochs=100,
#                               batch_size=32, lr=1e-3, seed=0)

# 4. Evaluate
metrics = BacktestUseCase(predictor).execute(BacktestCommand(dataset=dataset))
print(metrics.summary())
# → BacktestMetrics(n=... MAE=... RMSE=... R²=... AUROC=...)
```

### 3. Single inference

```python
import numpy as np
from src.application.predict_decoherence import PredictDecoherenceCommand, PredictDecoherenceUseCase

window = ...  # (window_length, n_features) float32 array of recent observables
result, events = PredictDecoherenceUseCase(predictor).execute(
    PredictDecoherenceCommand(window=window, t_obs=3.0, horizon=1.0)
)
print(f"Predicted T₂: {result.t_decoh_predicted:.3f}")
print(f"Risk score:   {result.risk_score:.3f}")
if events:
    print("⚠ Decoherence alarm triggered!")
```

---

## Running experiments

Paper 1 = E1–E5. Paper 2 = E6–E14. Replay from the UI (**02 Runs**) or from
the shell; every script is a module:

```bash
python -m experiments.run_paper1 --fast        # Paper 1 smoke (writes *_fast.csv)
python -m experiments.e8_improved_2qubit --fast
python -m experiments.r22_ar1_noise_training --seeds 42 43 44
make help                                      # the same commands as make targets
```

Common flags: `--seeds`, `--fast`, `--tag` (suffix of the result name),
`--out` (result file name); some scripts add their own (`--scenarios`,
`--n-per`, …; see `--help`). CSV and provenance go to `experiments/results/`.
Catalog and champions: `src/application/experiment_catalog.py`.

- **E1** C1, C2 — physics_only / lstm_only / physics_lstm / transformer × A–E
- **E2** C3 — window fraction sweep
- **E3** C4 — measurement noise σ
- **E4** C5 — train on mixture, eval per scenario
- **E5** — direct T₂ vs inverse γ-reconstruction
- **E6–E8** — 2-qubit scale-up; E8c is the TFIM R² champion (0.575)
- **E9–E11** — context codes, Transformer AUROC, scaling (E11a XXZ R²=0.817)
- **E12–E14** — stretched-exp baseline, survival loss, inject J
- **R1–R27** — replications and follow-up probes that check the experiments
  above across seeds rather than producing results of their own. Every
  published table survived them; several conclusions built on top of those
  tables did not. `paper/description_ru.tex` lists each probe and its
  outcome.

### Reproducing a paper table

1. Run the script that writes the table's CSV (the catalog lists it).
2. Compare with the committed CSV in `experiments/results/`; the provenance
   file next to it records the arguments and the software environment.

| Table | Command | Result file |
| --- | --- | --- |
| Paper 1, ablation (single run) | `python -m experiments.run_paper1` | `e1_ablation_p1.csv` |
| Paper 1, ablation with CIs | `python -m experiments.r13_independent_test` | `r13_independent_test.csv` |
| Paper 2, cross-system champion | `python -m experiments.e11_scaling --e11a` | `e11a_cross_system.csv` |
| Paper 2, champion across seeds | `python -m experiments.r3_multiseed_scaling` | `r3_multiseed_scaling.csv` |

Trajectories are cached in `data/trajectories/`. The cache key does not
contain the simulator variant, and a cache hit does not advance numpy's
global generator, so numbers depend on the cache state; for a before/after
comparison run each side from its own directory with a cold cache.

### Scenario taxonomy

| Label | System | γ(t) | Notes |
| --- | --- | --- | --- |
| A | 1-qubit, σ₋ | constant | Physics formula exact — LSTM residual ≈ 0 |
| B | 1-qubit, σ₋ | time-dependent | Core difficult case |
| C | 1-qubit, σ_z | time-dependent | Pure dephasing, different decay form |
| D | 2-qubit XXZ | time-dependent | Entanglement + decoherence |
| E | 2-qubit TFIM | time-dependent | Different interaction structure |

---

## Analysing results

```bash
jupyter notebook notebooks/figures.ipynb
```

The notebook generates:

| Figure | Content |
| --- | --- |
| Fig 1 | E1 ablation bar charts (R², MAE, AUROC by variant and scenario) |
| Fig 2 | E2 window sweep: R² and AUROC vs. observation fraction |
| Fig 3 | E3 noise sweep: MAE and AUROC vs. noise σ |
| Fig 4 | E4 cross-system bar chart |
| Fig 5 | E5 direct vs. inverse comparison |
| Fig 6 | Live demo: fresh trajectory with online predictions |
| Validation | Physics-only R²≥0.999 check on constant-γ trajectories |

All figures are saved as PDF to `notebooks/figs/`.

---

## Plugging in a custom loss

```python
from src.domain.ports import ILossFunction
from src.training.losses import get_loss

# Built-in losses (aliases in src.training.losses.LOSSES)
loss = get_loss("huber")          # HuberLoss(delta=1.0)
loss = get_loss("mae")            # L1
loss = get_loss("quantile_0.9")   # Pinball loss at q=0.9

# Custom loss — implement ILossFunction
class MyLoss(ILossFunction):
    @property
    def name(self): return "my_loss"
    def __call__(self, preds, targets): ...
    def as_torch(self): return MyTorchModule()

# Use it in training
TrainModelUseCase(predictor).execute(
    TrainModelCommand(dataset=ds, regression_loss="huber")
)
```

---

## Key parameters

| Parameter | Symbol | Default | Meaning |
| --- | --- | --- | --- |
| `omega` | ω | 1.0 | Qubit transition frequency |
| `gamma_const` | γ | — | Constant dissipation rate (mutually exclusive with `gamma_coeffs`) |
| `gamma_coeffs` | — | — | Bernstein coefficients for time-dependent γ(t) ≥ 0 |
| `dissipator` | L | σ₋ | Jump operator: `sigma_minus` or `sigma_z` |
| `t_max` | T | 10.0 | Simulation end time |
| `dt` | dt | 0.1 | Simulation time step |
| `decoherence_threshold` | θ | 1/e | Relative coherence threshold for T₂ definition |
| `window_length` | W | 50 | Timesteps in the observation window |
| `horizon` | Δt | 1.0 | Time window for binary risk label |
| `use_physics_prior` | — | True | Embed OLS-slope estimate as physics prior (False → lstm_only) |
| `noise.sigma` | σ | 0.0 | Gaussian measurement noise standard deviation |

---

## Tests

```bash
python -m pytest tests/ -q        # or: make test
python -m scripts.golden_outputs  # or: make golden
```

246 tests (`pytest.ini` puts the repository root on the path). They cover
physics (T₂, Lindblad, censoring), predictors, the registries, the Hamiltonian
model registry, survival backtest, inject-J, the experiment harness and
catalog, the notebook, and the invariants that make the reported metrics
meaningful:

- **Split integrity** — the train/val/test split is trajectory-level (several
  windows share one trajectory's T₂, so a per-window split would leak), no
  window reaches its own decoherence time, and normalisation statistics are
  fitted on training rows only.
- **Cache robustness** — a run interrupted mid-write must not poison the
  trajectory cache for every later run.
- **Seeding** — same seed gives identical data and identical weights, and no
  experiment script builds a training command without a seed.
- **Catalog integrity** — ids unique and matching the labels the papers cite,
  declared CSVs present, modules importable.
- **Replication fidelity** — R1 and R2 build the predictors of the
  experiments they check, R3 and R4 call those experiments rather than
  copying them, and an experiment function returns rows instead of writing a
  champion CSV.

---

## Scientific motivation (extended)

The closest related work is **arXiv 2505.06928** (*Unraveling Quantum Environments:
Transformer-Assisted Learning in Lindblad Dynamics*, Phys. Rev. A 2025), which solves
the **inverse problem** — reconstructing γ(t) from a full trajectory. Our work is
complementary and distinct:

| Aspect | arXiv 2505.06928 | This work |
| --- | --- | --- |
| Goal | Reconstruct γ(t) | Predict remaining T₂ |
| Task type | Inverse problem (offline) | Early warning (online) |
| Input | Full trajectory required | **Partial early window only** |
| Output | γ(t) Bernstein coefficients | T₂ + risk score |
| Physics prior | None (pure Transformer) | OLS slope = −γ/2 embedded |
| Analytical limit | No — pure ML | Yes — recovers exact formula when γ = const |

The key practical advantage: our approach is **online** (works from f ≈ 0.15 of
the total trajectory) and **directly actionable** (outputs T₂ remaining and a
calibrated alarm probability) without requiring a full simulation forward pass.
