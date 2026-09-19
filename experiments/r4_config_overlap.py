"""R4 — does E11a's score depend on evaluating the training configurations?

Why
---
E11a trains on ``build_configs(seed=42)`` and then evaluates on the *same*
configurations, regenerating their trajectories under a different seed.
The model therefore sees fresh noise realisations of physical systems it
has already been trained on, not unseen systems.

Whether that inflates the score depends on how much of T2 the
configuration fixes. Measured directly, the spread of T2 within one
configuration is much smaller than the spread across configurations on
XXZ (roughly 0.25--0.53 against 1.90) and comparable on TFIM
(0.73--1.50 against 1.46). So on XXZ a model that recognised a
configuration from its window could recover most of the target without
generalising at all.

Design
------
One trained model per seed, scored twice:

    seen     the training configurations, fresh trajectories (published)
    unseen   disjoint configurations from a different build_configs seed

Everything else -- architecture, window, min-gap, protocol, evaluation
seed -- is identical, so the difference is attributable to configuration
overlap alone. A large gap would mean the published cross-system numbers
are optimistic; a small one closes the question.

Writes ONLY to experiments/results/r4_config_overlap.csv.

Usage
-----
    python -m experiments.r4_config_overlap --fast
    python -m experiments.r4_config_overlap --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from experiments.e11_scaling import run_e11a
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r4_scratch.csv"
CONFIG_SEED_OFFSET = 500      # far from any seed the experiments use


def _append(path: Path, row: dict, first: bool) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default="r4_config_overlap.csv")
    a = p.parse_args()

    n_per = 50 if a.fast else 500
    epochs = 10 if a.fast else 150
    out = RESULTS_DIR / a.out

    print(f"R4 | seeds={a.seeds} n_per={n_per} epochs={epochs}")
    print(f"writing -> {out}  (champion CSVs untouched)\n", flush=True)

    sim, first, t0, collected = QuTipSimulator(), True, time.time(), []

    for seed in a.seeds:
        train_groups = build_configs(n_per_scenario=n_per, seed=seed)
        fresh_groups = build_configs(n_per_scenario=n_per,
                                     seed=seed + CONFIG_SEED_OFFSET)
        for arm, eval_groups in (("seen", None), ("unseen", fresh_groups)):
            ts = time.time()
            print(f"--- seed {seed} | {arm} configs ---", flush=True)
            rows = run_e11a(sim, train_groups, n_epochs=epochs, n_per=n_per,
                            verbose=False, seed=seed, window=20,
                            out_path=SCRATCH, eval_groups=eval_groups)
            for r in rows:
                row = {"seed": seed, "arm": arm, "scenario": r["scenario"],
                       "r2": r["r2"], "mae": r["mae"], "auroc": r["auroc"],
                       "n_samples": r["n_samples"]}
                _append(out, row, first)
                first = False
                collected.append(row)
                print(f"    {arm:<7} {r['scenario']} R2={r['r2']:8.3f} "
                      f"AUROC={r['auroc']:.3f}", flush=True)
            print(f"    [{time.time() - ts:.0f}s]", flush=True)

    SCRATCH.unlink(missing_ok=True)

    g = defaultdict(dict)
    for r in collected:
        g[(r["scenario"], r["arm"])][r["seed"]] = r

    print("\n" + "=" * 58)
    print(f"{'scen':<5}{'arm':<8}{'R2 mean':>9}{'±std':>8}{'AUROC':>8}{'±std':>7}")
    for sc in ("D", "E"):
        for arm in ("seen", "unseen"):
            v = list(g[(sc, arm)].values())
            if not v:
                continue
            r2 = [float(x["r2"]) for x in v]
            au = [float(x["auroc"]) for x in v]
            sd = st.stdev(r2) if len(r2) > 1 else 0.0
            sa = st.stdev(au) if len(au) > 1 else 0.0
            print(f"{sc:<5}{arm:<8}{st.mean(r2):9.3f}{sd:8.3f}{st.mean(au):8.3f}{sa:7.3f}")

    print("\npaired seen - unseen (positive = evaluating on training configs flatters):")
    for sc in ("D", "E"):
        seeds = sorted(set(g[(sc, "seen")]) & set(g[(sc, "unseen")]))
        if not seeds:
            continue
        d = [float(g[(sc, "seen")][s]["r2"]) - float(g[(sc, "unseen")][s]["r2"])
             for s in seeds]
        da = [float(g[(sc, "seen")][s]["auroc"]) - float(g[(sc, "unseen")][s]["auroc"])
              for s in seeds]
        sd = st.stdev(d) if len(d) > 1 else 0.0
        print(f"  {sc}: ΔR²={st.mean(d):+.3f} ± {sd:.3f}  {[f'{x:+.3f}' for x in d]}")
        print(f"     ΔAUROC={st.mean(da):+.3f}  {[f'{x:+.3f}' for x in da]}")

    print(f"\ntotal {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
