"""Behaviour-regression check for refactoring: records or compares golden outputs.

Runs small, deterministic versions of everything the papers depend on --
simulation under every decoherence criterion, the experiment-specific
simulators, dataset construction, noise models and short training runs of
every predictor -- and stores scalar fingerprints of the results. A refactor
must reproduce them exactly.

Usage:
    python -m scripts.golden_outputs --write   # record (before refactoring)
    python -m scripts.golden_outputs           # compare against the record

Only the imports in the "Code under test" block may change when code moves;
the recorded values may not.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Code under test (update these imports if the code moves) -------------
from experiments.r26_esd_occurrence import windows as esd_windows
from src import reproducibility
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.data.features import select as feature_set
from src.data.observation import ar1_filter, noise_augmented
from src.data.scenarios import build_configs
from src.domain.value_objects import DecoherenceCriterion
from src.models.lstm import LSTMPredictor
from src.models.physics_only import PhysicsOnlyPredictor
from src.models.stretched_exp import StretchedExpPredictor
from src.models.transformer import TransformerPredictor
from src.persistence.stores import NumpyTrajectoryStore, PickleModelStore
from src.simulation.simulator import QuTipSimulator
from src.simulation.variants import (
    CORR,
    CorrelatorSimulator,
    EntanglementSimulator,
    StochasticGammaSimulator,
)
# ---------------------------------------------------------------------------

GOLDEN = Path(__file__).with_name("golden_outputs.json")


def _fp(arr) -> list[str]:
    """Fingerprint of an array: size, sum, sum of squares, first and last values."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return ["0"]
    return [str(a.size), repr(float(a.sum())), repr(float((a * a).sum())),
            repr(float(a[0])), repr(float(a[-1]))]


def _traj(traj) -> dict:
    out = {"t_decoh": repr(float(traj.t_decoh)), "censored": bool(traj.censored),
           "features": list(traj.feature_names)}
    for k in traj.feature_names:
        out[k] = _fp(traj.observables[k])
    return out


def _dataset(ds) -> dict:
    return {"sequences": _fp(ds.sequences), "remaining": _fp(ds.remaining_time),
            "t_obs": _fp(ds.t_obs), "risk": _fp(ds.risk_labels),
            "split": [len(ds.train_idx), len(ds.val_idx), len(ds.test_idx)],
            "features": list(ds.feature_names)}


def _metrics(m) -> list[str]:
    return [repr(float(m.r2)), repr(float(m.mae)), repr(float(m.risk_auroc)), str(m.n_samples)]


def compute() -> dict:
    reproducibility.set_num_threads(1)
    out: dict = {}
    g = build_configs(n_per_scenario=6, seed=11)

    # Simulation under every criterion, every scenario.
    sim = QuTipSimulator()
    for sc in "ABCDE":
        crits = [DecoherenceCriterion.COHERENCE, DecoherenceCriterion.PURITY]
        if sc in "DE":
            crits += [DecoherenceCriterion.COHERENCE_ENERGY, DecoherenceCriterion.CONCURRENCE,
                      DecoherenceCriterion.ENTANGLEMENT_DEATH]
        for c in crits:
            reproducibility.seed_numpy_global(5)
            cfg = dataclasses.replace(g[sc][1], decoherence_criterion=c)
            out[f"sim/{sc}/{c.value}"] = _traj(sim.simulate(cfg))

    # Experiment-specific simulators.
    reproducibility.seed_numpy_global(5)
    out["sim/corr"] = _traj(CorrelatorSimulator().simulate(g["E"][2]))
    reproducibility.seed_numpy_global(5)
    out["sim/ent"] = _traj(EntanglementSimulator().simulate(
        dataclasses.replace(g["D"][2], decoherence_criterion=DecoherenceCriterion.ENTANGLEMENT_DEATH)))
    reproducibility.seed_numpy_global(5)
    out["sim/stochastic_gamma"] = _traj(StochasticGammaSimulator(0.3, 0.5, seed=3).simulate(g["B"][2]))

    # Noise model.
    eta = np.random.default_rng(0).standard_normal((50, 20, 3))
    out["noise/ar1_scalar"] = _fp(ar1_filter(eta, 0.7))
    out["noise/ar1_rows"] = _fp(ar1_filter(eta, np.linspace(0.0, 0.9, 50)))

    # Datasets.
    cmd = dict(window_length=20, horizon=1.0, samples_per_trajectory=5, seed=3)
    ds_1q = GenerateDatasetUseCase(QuTipSimulator(), store=None).execute(
        GenerateDatasetCommand(configs=g["B"] + g["C"], dataset_name="golden_1q", **cmd))
    ds_2q = GenerateDatasetUseCase(EntanglementSimulator(), store=None).execute(
        GenerateDatasetCommand(configs=g["D"] + g["E"], dataset_name="golden_2q", **cmd))
    out["ds/1q"] = _dataset(ds_1q)
    out["ds/2q"] = _dataset(ds_2q)
    out["ds/2q_base"] = _dataset(ds_2q.drop_features(CORR))
    out["ds/2q_corr"] = _dataset(feature_set(ds_2q, "corr"))
    out["ds/1q_aug"] = _dataset(noise_augmented(ds_1q, 2, np.random.default_rng(7), phi_max=0.9))

    # Classification windows (R26).
    trajs = []
    for i, c in enumerate(g["D"][:4]):
        reproducibility.seed_numpy_global(100 + i)
        trajs.append(EntanglementSimulator().simulate(
            dataclasses.replace(c, decoherence_criterion=DecoherenceCriterion.ENTANGLEMENT_DEATH)))
    X, owner, tob = esd_windows(trajs, "corr_conc")
    out["r26/windows"] = [_fp(X), owner.tolist(), tob.tolist()]

    # Predictors.
    def run(pred, ds, train=True):
        if train:
            TrainModelUseCase(pred).execute(TrainModelCommand(
                dataset=ds, n_epochs=3, batch_size=16, learning_rate=1e-3,
                regression_loss="huber", verbose=False, seed=1))
        return _metrics(BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0)))

    out["model/physics_only"] = run(PhysicsOnlyPredictor(t_max=20.0), ds_1q, train=False)
    out["model/stretched_exp"] = run(StretchedExpPredictor(t_max=20.0), ds_1q, train=False)
    out["model/physics_lstm"] = run(LSTMPredictor(use_physics_prior=True), ds_1q)
    out["model/lstm_only"] = run(LSTMPredictor(use_physics_prior=False), ds_1q)
    out["model/physics_feature"] = run(LSTMPredictor(use_physics_prior=True, residual_output=False), ds_1q)
    out["model/transformer"] = run(TransformerPredictor(), ds_1q)
    out["model/transformer_2q"] = run(
        TransformerPredictor(d_model=32, n_heads=4, n_layers=1, dim_ff=64), feature_set(ds_2q, "corr"))
    out["model/lstm_2q_gated"] = run(
        LSTMPredictor(use_physics_prior=True, adaptive_prior_r2_threshold=0.7), ds_2q.drop_features(CORR))

    # Stored artefacts must keep loading: every checkpoint predicts on fixed
    # windows, and a few cached trajectories load unchanged.
    store = PickleModelStore("checkpoints")
    for ckpt in sorted(Path("checkpoints").glob("*.pt")):
        pred = store.load(ckpt.stem)
        n_feat = 18 if "2q" in ckpt.stem or "2qubit" in ckpt.stem else 5
        ds = ds_2q if n_feat == 18 else ds_1q
        preds = []
        for i in range(min(5, len(ds.sequences))):
            w = ds.sequences[i]
            try:
                r = pred.predict(w, float(ds.t_obs[i]), 1.0)
                preds.append(r.t_decoh_predicted)
            except Exception as exc:  # record failures too, they must not change
                preds.append(type(exc).__name__)
        out[f"ckpt/{ckpt.stem}"] = [type(pred).__name__] + [
            repr(float(x)) if not isinstance(x, str) else x for x in preds]
    traj_store = NumpyTrajectoryStore()
    for name in sorted(traj_store.list_available())[:5]:
        out[f"cache/{name}"] = [_traj(t) for t in traj_store.load(name)[:2]]
    return out


def _diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
        return out
    return [] if a == b else [f"{path}: expected {a} got {b}"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    got = compute()
    if args.write:
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True))
        print(f"wrote {GOLDEN} ({len(got)} entries)")
        return
    diffs = _diff(json.loads(GOLDEN.read_text()), json.loads(json.dumps(got)))
    if diffs:
        print(f"GOLDEN MISMATCH ({len(diffs)}):")
        print("\n".join(diffs[:50]))
        sys.exit(1)
    print(f"golden outputs identical ({len(got)} entries)")


if __name__ == "__main__":
    main()
