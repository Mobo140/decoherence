"""R20 — the energy-basis target on the Heisenberg model and on the cross-system model.

Why
---
R10 showed that on TFIM the loss of R^2 with J disappears when the target is
the 1/e crossing of the l1-coherence in the eigenbasis of H instead of the
computational basis. The explanation is that the TFIM Hamiltonian rotates
coherence between bases by an angle tan(2 theta) = J / 2h, so a
computational-basis target is modulated by unitary motion.

Two checks of that explanation:

  (a) XXZ J-sweep, both targets. The XXZ Hamiltonian mixes |01> and |10>
      maximally for every J, so the mixing angle does not depend on J.
      Prediction: no J-trend under either target. This is the control.
  (b) Cross-system Transformer (E11a, window 20) trained on D+E with the
      energy-basis target. Compared with R3 arm e11a_w20 (computational
      target, same configurations and seeds). Prediction: E (TFIM) gains,
      D (XXZ) changes little.

Arms
----
  xxz_coh      run_e7c on XXZ, computational-basis target
  xxz_energy   run_e7c on XXZ, energy-basis target
  cross_energy run_e11a on build_configs(500, seed) D+E, energy-basis target

Writes ONLY to experiments/results/r20a_xxz_J_sweep*.csv and
experiments/results/r20b_cross_energy*.csv (suffix set by --tag).

Usage
-----
    python -m experiments.r20_energy_target_xxz --fast
    python -m experiments.r20_energy_target_xxz --arms xxz_coh --seeds 42 --tag s42
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from experiments.e7_paper2_extended import run_e7c
from experiments.e11_scaling import run_e11a
from src.domain.value_objects import DecoherenceCriterion, InteractionType
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
ARMS = ("xxz_coh", "xxz_energy", "cross_energy")
TARGET = {"xxz_coh": DecoherenceCriterion.COHERENCE,
          "xxz_energy": DecoherenceCriterion.COHERENCE_ENERGY}


def _energy_groups(n_per: int, seed: int) -> dict:
    groups = build_configs(n_per_scenario=n_per, seed=seed)
    return {sc: [dataclasses.replace(
                c, decoherence_criterion=DecoherenceCriterion.COHERENCE_ENERGY)
                 for c in groups[sc]]
            for sc in ("D", "E")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()

    n_per_J = 30 if a.fast else 200
    n_per = 50 if a.fast else 500
    epochs = 10 if a.fast else 150
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    out_a = RESULTS_DIR / f"r20a_xxz_J_sweep{sfx}.csv"
    out_b = RESULTS_DIR / f"r20b_cross_energy{sfx}.csv"
    scratch = RESULTS_DIR / f"_r20_scratch{sfx}.csv"
    print(f"R20 | seeds={a.seeds} arms={a.arms} epochs={epochs}", flush=True)

    sim = QuTipSimulator()
    fa = fb = wa = wb = None
    if any(arm in TARGET for arm in a.arms):
        fa = out_a.open("w", newline="")
        wa = csv.DictWriter(fa, fieldnames=["seed", "target", "J", "r2", "mae",
                                            "auroc", "n_samples"])
        wa.writeheader()
    if "cross_energy" in a.arms:
        fb = out_b.open("w", newline="")
        wb = csv.DictWriter(fb, fieldnames=["seed", "scenario", "r2", "mae",
                                            "rmse", "auroc", "n_samples"])
        wb.writeheader()

    for seed in a.seeds:
        for arm in a.arms:
            ts = time.time()
            if arm in TARGET:
                rows = run_e7c(sim, n_per_J=n_per_J, n_epochs=epochs, verbose=False,
                               seed=seed, config_seed=seed, out_path=scratch,
                               criterion=TARGET[arm],
                               interaction=InteractionType.XXZ)
                target = "energy" if arm == "xxz_energy" else "computational"
                for r in rows:
                    wa.writerow({"seed": seed, "target": target, "J": r["J"],
                                 "r2": r["r2"], "mae": r["mae"],
                                 "auroc": r["auroc"], "n_samples": r["n_samples"]})
                    fa.flush()
                    print(f"R20 | {arm} seed={seed} J={r['J']:.1f} "
                          f"R2={r['r2']:.3f}", flush=True)
            else:
                rows = run_e11a(sim, _energy_groups(n_per, seed), n_epochs=epochs,
                                n_per=n_per, verbose=False, seed=seed, window=20,
                                out_path=scratch)
                for r in rows:
                    wb.writerow({"seed": seed, "scenario": r["scenario"],
                                 "r2": r["r2"], "mae": r["mae"], "rmse": r["rmse"],
                                 "auroc": r["auroc"], "n_samples": r["n_samples"]})
                    fb.flush()
                    print(f"R20 | {arm} seed={seed} {r['scenario']} "
                          f"R2={r['r2']:.3f}", flush=True)
            print(f"R20 | {arm} seed={seed} done [{time.time() - ts:.0f}s]", flush=True)

    for f in (fa, fb):
        if f is not None:
            f.close()
    scratch.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
