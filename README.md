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

- [Physics background](#physics-background)
- [Model architecture](#model-architecture)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running experiments](#running-experiments)
- [Analysing results](#analysing-results)
- [Interactive UI](#interactive-ui)
- [Key parameters](#key-parameters)
- [Scientific motivation](#scientific-motivation)

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

## Project structure

```
.
├── app.py                              # FastAPI workbench: 7 screens
├── static/workbench.css
├── experiments/
│   ├── _config.py                      # SystemConfig factory, scenarios A–E
│   ├── e1_ablation.py … e14_inject_J.py
│   ├── run_all.py                      # Paper 1 suite (E1–E5)
│   └── results/                        # CSV from completed runs
├── src/
│   ├── domain/                         # value objects, ports — no torch/qutip
│   ├── application/                    # use-cases + experiment_catalog / job_queue
│   ├── infrastructure/                 # QuTiP, LSTM, Transformer, registries
│   └── physics/                        # Lindblad systems, T₂
├── paper/paper_1/  paper/paper_2/
├── docks/                              # plan, architecture, task log
├── tests/
├── checkpoints/
└── requirements.txt
```

Full layer map: `docks/main/ARCHITECTURE.md`. Status: `docks/current/CURRENT_STATUS.md`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: `qutip`, `torch`, `fastapi`, `numpy`, `scipy`, `scikit-learn`

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
| **02 Runs** | Каталог E1–E14 и R1–R5, CSV, Compare, Replay `--fast` / full |
| **03 Models** | `HamiltonianModelRegistry` (best_XXZ / best_TFIM / 1q) |
| **04 Simulate** | Пресет сценария A–E (тот же `build_configs`) |
| **05 Train** | Очередь eN в фоне, UI не блокируется |
| **06 Predict** | Окно до T₂ + risk alarm (fallback: physics_only) |
| **07 Backtest** | Таблица чемпионов из CSV статей |

Интерфейс — макет аудита (сайдбар + blueprint), не Gradio. Перезапусти `python app.py`.

### 2. Programmatic API

```python
from src.domain.value_objects import DissipatorConfig, QubitCount, SystemConfig
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
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

# 3. Train physics+LSTM predictor
predictor = LSTMPredictor(use_physics_prior=True)
TrainModelUseCase(predictor).execute(
    TrainModelCommand(dataset=dataset, n_epochs=100)
)

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

Paper 1 = E1–E5. Paper 2 = E6–E14. Replay from the UI (**02 Runs**) or:

```bash
python experiments/run_all.py --fast          # Paper 1 smoke
python -m experiments.e8_improved_2qubit --fast
python -m experiments.e14_inject_J --fast
```

CSV → `experiments/results/`. Catalog and champions: `src/application/experiment_catalog.py`.

- **E1** C1, C2 — physics_only / lstm_only / physics_lstm / transformer × A–E
- **E2** C3 — window fraction sweep
- **E3** C4 — measurement noise σ
- **E4** C5 — train on mixture, eval per scenario
- **E5** — direct T₂ vs inverse γ-reconstruction
- **E6–E8** — 2-qubit scale-up; E8c is TFIM R² champion (0.575)
- **E9–E11** — context codes, Transformer AUROC, scaling (E11a XXZ R²=0.817)
- **E12–E14** — stretched-exp baseline, survival loss, inject J
- **R1–R5** — replications that check the experiments above across seeds
  rather than producing results of their own. They overturned five of Paper 2's
  contested conclusions (see `docks/current/CURRENT_STATUS.md`); the tables
  themselves were unaffected.

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
from src.infrastructure.ml.loss_functions import get_loss

# Built-in losses
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
python -m pytest tests/
```

138 tests. Covers physics (T₂, Lindblad, censoring), predictors, Hamiltonian
registry, survival backtest, inject-J, and the experiment catalog / dashboard
HTML, plus the invariants that make the reported metrics meaningful:

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
- **Replication fidelity** — R1/R2 mirror the hyperparameters of the
  experiments they check, R3–R5 call those experiments rather than copying
  them, and no probe can write a champion CSV.

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
