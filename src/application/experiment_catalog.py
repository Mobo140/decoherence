"""Catalog of paper experiments and helpers to read their CSV results."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"


@dataclass(frozen=True)
class ExperimentEntry:
    id: str
    title: str
    module: str
    paper: str
    claim: str
    scenarios: str
    csv_globs: tuple
    supports_fast: bool = True
    status: str = "cited"
    """Role of this experiment, so the UI does not present a dead end as a
    result: "cited" (a paper reports it), "superseded" (a later experiment
    replaced it), "negative" (recorded outcome that did not reach its goal,
    kept so it is not retried blindly), or "replication" (checks another
    experiment rather than producing a result of its own)."""


CATALOG: List[ExperimentEntry] = [
    ExperimentEntry("E1", "Ablation", "experiments.e1_ablation", "1", "C1, C2", "A–C",
                    ("e1_ablation_p1.csv",), status="cited"),
    ExperimentEntry("E2", "Window sweep", "experiments.e2_window_sweep", "1", "C3", "B/C",
                    ("e2_window_sweep_p1.csv",), status="cited"),
    ExperimentEntry("E3", "Noise sweep", "experiments.e3_noise_sweep", "1", "C4", "B/C",
                    ("e3_noise_sweep_p1.csv",), status="cited"),
    ExperimentEntry("E4", "Cross-system", "experiments.e4_cross_system", "1", "C5", "A–C",
                    ("e4_cross_system_p1.csv",), status="cited"),
    ExperimentEntry("E5", "Inverse baseline", "experiments.e5_inverse_baseline", "1", "C3", "B",
                    ("e5_inverse_baseline.csv",), status="cited"),
    ExperimentEntry("E6", "2q improved (E6a/b/c)", "experiments.e6_2qubit_improved", "2", "C2", "D, E",
                    ("e6_ablation_improved.csv", "e6_noise_sweep_improved.csv",
                     "e6_cross_system_improved.csv"), status="cited"),
    ExperimentEntry("E7", "Transformer / τ / J (E7a/b/c)", "experiments.e7_paper2_extended", "2", "C2, C5", "D, E",
                    ("e7a_transformer_cross.csv", "e7b_tau_sweep.csv", "e7c_J_sweep.csv"),
                    status="cited"),
    ExperimentEntry("E8c", "Improved 2q training", "experiments.e8_improved_2qubit", "2", "C4, C5", "D, E",
                    ("e8c_cross_system.csv",), status="cited"),
    ExperimentEntry("E9", "Context codes", "experiments.e9_context_injection", "-", "C5", "D, E",
                    ("e9a_cross_system.csv", "e9b_ablation.csv"), status="negative"),
    # The papers' "E10b" is the window-30 cross-system run produced by
    # e10_fixed_context.py -- not e10_tfim_specialized.py, which this catalog
    # used to label E10b. Ids now match what the papers cite.
    ExperimentEntry("E10a/E10b", "Fixed context (E10b cited)", "experiments.e10_fixed_context", "2", "C5", "D, E",
                    ("e10a_cross_system.csv", "e10b_cross_system.csv"), status="cited"),
    ExperimentEntry("E10-TFIM", "TFIM specialized", "experiments.e10_tfim_specialized", "-", "C2", "E",
                    ("e10_tfim_c.csv",), status="negative"),
    ExperimentEntry("E11", "Scaling (E11a/E11b)", "experiments.e11_scaling", "2", "C5", "D, E",
                    ("e11a_cross_system.csv", "e11b_cross_system.csv"), status="cited"),
    ExperimentEntry("E12", "Stretched-exp baseline", "experiments.e12_baseline_comparison", "-", "C2", "B–E",
                    ("e12_baseline_comparison.csv",), status="negative"),
    ExperimentEntry("E13", "Survival TFIM", "experiments.e13_survival_tfim", "-", "C2", "E",
                    ("e13_survival_tfim.csv",), status="negative"),
    ExperimentEntry("E14", "Inject J", "experiments.e14_inject_J", "-", "C2, C5", "D, E",
                    ("e14_inject_J.csv",), status="negative"),

    # Replications: these check other experiments rather than producing
    # results of their own, so they sit outside the E-numbering the papers
    # cite. (They were briefly named E22-E24 after task ids, which left a
    # phantom gap at E15-E21.)
    ExperimentEntry("R1", "Ablation across seeds (checks E6a)", "experiments.r1_multiseed_ablation", "-", "-", "D, E",
                    ("r1_multiseed_ablation_full.csv", "r1_multiseed_ablation_defaultarch.csv"),
                    status="replication"),
    ExperimentEntry("R2", "τ sweep across seeds (checks E7b)", "experiments.r2_multiseed_tau", "-", "-", "D, E",
                    ("r2_multiseed_tau_full_seeded.csv",), status="replication"),
    ExperimentEntry("R3", "Scaling across seeds (checks E11a/E11b)", "experiments.r3_multiseed_scaling", "-", "-", "D, E",
                    ("r3_multiseed_scaling.csv",), status="replication"),
    ExperimentEntry("R4", "Config overlap in E11a evaluation", "experiments.r4_config_overlap", "-", "-", "D, E",
                    ("r4_config_overlap.csv",), status="replication"),
    ExperimentEntry("R5", "J sweep across seeds (checks E7c)", "experiments.r5_multiseed_J_sweep", "-", "-", "E",
                    ("r5_multiseed_J_sweep.csv",), status="replication"),
    ExperimentEntry("R6", "Parity-sector preparation (tests the ceiling mechanism)", "experiments.r6_parity_sector", "-", "-", "E",
                    ("r6_parity_sector.csv",), status="replication"),
    ExperimentEntry("R7", "Paper 1 ablation across seeds (checks E1)", "experiments.r7_multiseed_1qubit", "-", "-", "A-C",
                    ("r7_multiseed_1qubit.csv",), status="replication"),
    ExperimentEntry("R8", "Noise sweep and scaling across seeds (checks E6b/E8c)", "experiments.r8_noise_and_scaling", "-", "-", "D, E",
                    ("r8_noise_and_scaling.csv",), status="replication"),
]


# Paper-cited champions (PROJECT_PLAN / CURRENT_STATUS). Dashboard uses these
# so the UI does not silently pick a weak row from a mixed CSV.
CHAMPIONS = [
    {"scenario": "A", "label": "1q σ₋ const", "r2": 0.9998, "model": "physics_only", "run": "E1", "auroc": 1.000, "goal": 0.999},
    {"scenario": "B", "label": "1q σ₋ γ(t)", "r2": 0.945, "model": "physics_lstm", "run": "E1", "auroc": 0.965, "goal": 0.97},
    {"scenario": "C", "label": "1q σz γ(t)", "r2": 0.981, "model": "physics_lstm", "run": "E1", "auroc": 0.891, "goal": 0.97},
    {"scenario": "D", "label": "2q XXZ", "r2": 0.817, "model": "transformer w20", "run": "E11a", "auroc": 0.937, "goal": 0.80},
    {"scenario": "E", "label": "2q TFIM", "r2": 0.575, "model": "physics_lstm", "run": "E8c", "auroc": 0.759, "goal": 0.70},
]


def by_id(exp_id: str) -> ExperimentEntry:
    for e in CATALOG:
        if e.id == exp_id:
            return e
    raise KeyError(exp_id)


def list_csv_paths(entry: ExperimentEntry) -> List[Path]:
    found = []
    for name in entry.csv_globs:
        p = RESULTS_DIR / name
        if p.exists():
            found.append(p)
    return found


def read_csv_rows(path: Path, limit: int = 40) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def summarize_r2(path: Path) -> str:
    rows = read_csv_rows(path, limit=200)
    if not rows:
        return f"{path.name}: empty"
    r2_key = next((k for k in ("r2", "R²", "R2") if k in rows[0]), None)
    if r2_key is None:
        return f"{path.name}: {len(rows)} rows"
    parts = []
    for row in rows:
        label = row.get("scenario") or row.get("name") or row.get("arm") or row.get("variant") or "?"
        extra = row.get("j_bin") or row.get("variant") or ""
        try:
            r2 = float(row[r2_key])
            r2s = f"{r2:.3f}"
        except (TypeError, ValueError):
            r2s = str(row[r2_key])
        tag = f"{label}" + (f"/{extra}" if extra and extra != label else "")
        parts.append(f"{tag} {r2s}")
    return f"{path.name}: " + "; ".join(parts[:8])


def catalog_table() -> List[List[str]]:
    table = []
    for e in CATALOG:
        csvs = list_csv_paths(e)
        status = f"{len(csvs)} CSV" if csvs else "no CSV"
        table.append([e.id, e.title, e.paper, e.claim, e.scenarios, status, e.module])
    return table


CLAIMS = [
    {"id": "C1", "ok": True, "text": "physics baseline R²≥0.999 ✓"},
    {"id": "C2", "ok": True, "text": "residual > ablations на B, C ✓"},
    {"id": "C3", "ok": True, "text": "AUROC 0.947 @ f=0.15 ✓ (R² плато 0.73)"},
    {"id": "C4", "ok": True, "text": "AUROC 0.942 @ σ=0.05 ✓"},
    {"id": "C5", "ok": False, "text": "mixed слабее per-Hamiltonian"},
]


def cache_stats(root: Path | None = None) -> dict:
    cache_root = root or (Path(__file__).resolve().parents[2] / "data" / "trajectories")
    files = list(cache_root.glob("*.npz")) if cache_root.exists() else []
    bytes_ = sum(p.stat().st_size for p in files)
    return {
        "n_files": len(files),
        "bytes": bytes_,
        "gb": round(bytes_ / (1024 ** 3), 3),
        "n_csv": len(list(RESULTS_DIR.glob("*.csv"))),
    }


def compare_summaries(id_a: str, id_b: str) -> str:
    lines = [f"Compare {id_a} vs {id_b}", ""]
    for exp_id in (id_a, id_b):
        e = by_id(exp_id)
        lines.append(f"### {e.id} — {e.title} (claim {e.claim})")
        paths = list_csv_paths(e)
        if not paths:
            lines.append("нет CSV")
        for p in paths:
            lines.append(summarize_r2(p))
        lines.append("")
    return "\n".join(lines)
