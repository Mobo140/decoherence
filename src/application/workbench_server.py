"""FastAPI workbench matching the agreed audit mockup.

Serves static/index.html and JSON for the 7 screens.
Does not start paper experiments except via the existing JobQueue (--fast).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC = ROOT / "static"

from src.application.experiment_catalog import (
    CATALOG,
    CHAMPIONS,
    CLAIMS,
    by_id,
    cache_stats,
    compare_summaries,
    list_csv_paths,
    read_csv_rows,
)
from src.application.job_queue import JobQueue
from src.application.predict_decoherence import PredictDecoherenceCommand, PredictDecoherenceUseCase
from src.application.simulate_trajectory import SimulateTrajectoryCommand, SimulateTrajectoryUseCase
from src.domain.value_objects import DissipatorType, InteractionType
from src.infrastructure.ml.lstm_predictor import physics_remaining
from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
from src.infrastructure.persistence.stores import BestModelRegistry, HamiltonianModelRegistry
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


_jobs = JobQueue()
_simulator = QuTipSimulator()
_ham = HamiltonianModelRegistry()
_best = BestModelRegistry()
_session: dict[str, Any] = {"trajectory": None, "scenario": None}


class SimulateIn(BaseModel):
    scenario: str = "B"
    seed: int = 42


class PredictIn(BaseModel):
    scenario: str = "B"
    seed: int = 7
    horizon: float = 1.0
    alarm: float = Field(0.7, ge=0.0, le=1.0)
    n_qubits: int = 1
    interaction: str = "XXZ"
    dissipator: str = "sigma_minus"


class JobIn(BaseModel):
    exp_id: str = "E8c"
    mode: str = "fast"


class CompareIn(BaseModel):
    a: str = "E8c"
    b: str = "E14"


def _series(traj) -> dict:
    obs = traj.observables
    names = list(traj.feature_names)
    pauli = next((k for k in ("sigma_x", "sigma_x1") if k in obs), None)
    l1 = obs.get("coherence_l1")
    if l1 is None:
        raise HTTPException(500, "trajectory has no coherence_l1")
    return {
        "times": [float(x) for x in traj.times.tolist()],
        "coherence": [float(x) for x in l1.tolist()],
        "observable": [float(x) for x in obs[pauli].tolist()] if pauli else [],
        "observable_name": pauli or "",
        "t_decoh": float(traj.t_decoh),
        "censored": bool(getattr(traj, "censored", False)),
        "feature_names": names,
    }


def _simulate_preset(scenario: str, seed: int):
    from experiments._config import build_configs

    groups = build_configs(n_per_scenario=1, seed=int(seed))
    if scenario not in groups:
        raise HTTPException(400, f"unknown scenario {scenario}")
    cfg = groups[scenario][0]
    traj = SimulateTrajectoryUseCase(_simulator).execute(
        SimulateTrajectoryCommand(config=cfg, seed=int(seed))
    )
    _session["trajectory"] = traj
    _session["scenario"] = scenario
    _session["config"] = cfg
    return traj, cfg


def _pick_predictor(n_qubits: int, interaction: str, dissipator: str):
    it = InteractionType(interaction)
    dt = DissipatorType(dissipator)
    pred = (
        _ham.get_predictor(n_qubits, it, dt)
        or _best.get_predictor(n_qubits)
    )
    if pred is not None and pred.is_trained:
        source = "registry"
        return pred, source
    return PhysicsOnlyPredictor(), "physics_only fallback"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (value != value or value == float("inf") or value == float("-inf")):
        return None
    return value


def _model_slots() -> list[dict]:
    specs = (
        (1, InteractionType.XXZ, DissipatorType.SIGMA_MINUS, "1q / σ₋ · amplitude damping"),
        (1, InteractionType.XXZ, DissipatorType.SIGMA_Z, "1q / σz · dephasing"),
        (2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS, "2q / XXZ / σ₋"),
        (2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS, "2q / TFIM / σ₋"),
    )
    out = []
    for nq, it, dt, kicker in specs:
        meta = _ham.get_meta(nq, it, dt)
        out.append({
            "kicker": kicker,
            "label": HamiltonianModelRegistry._label(nq, it, dt),
            "n_qubits": nq,
            "interaction": it.value,
            "dissipator": dt.value,
            "meta": _json_safe(meta),
            "warn": nq == 2 and it == InteractionType.TFIM,
            "empty": meta is None,
        })
    return out


def create_app() -> FastAPI:
    app = FastAPI(title="Q-EWS Workbench", docs_url=None, redoc_url=None)

    @app.get("/api/meta")
    def meta():
        return {
            "champions": CHAMPIONS,
            "claims": CLAIMS,
            "cache": cache_stats(),
            "paper": "Paper 1 готов · Paper 2: TFIM R² 0.575 < 0.70 (E8c)",
        }

    @app.get("/api/catalog")
    def catalog():
        rows = []
        for e in CATALOG:
            paths = list_csv_paths(e)
            rows.append({
                "id": e.id,
                "title": e.title,
                "module": e.module,
                "paper": e.paper,
                "claim": e.claim,
                "scenarios": e.scenarios,
                "supports_fast": e.supports_fast,
                "n_csv": len(paths),
                "csv": [p.name for p in paths],
            })
        return {"experiments": rows}

    @app.get("/api/csv/{exp_id}")
    def csv_rows(exp_id: str):
        try:
            entry = by_id(exp_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        paths = list_csv_paths(entry)
        if not paths:
            return {"headers": [], "rows": [], "file": None}
        rows = read_csv_rows(paths[0], limit=40)
        headers = list(rows[0].keys()) if rows else []
        keep = [k for k in headers if k in
                ("scenario", "name", "variant", "arm", "j_bin", "r2", "mae",
                 "auroc", "n_samples", "use_J", "window", "noise_sigma")]
        if not keep:
            keep = headers[:8]
        return {
            "file": paths[0].name,
            "headers": keep,
            "rows": [{k: r.get(k, "") for k in keep} for r in rows],
        }

    @app.post("/api/compare")
    def compare(body: CompareIn):
        try:
            text = compare_summaries(body.a, body.b)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"text": text, "a": body.a, "b": body.b}

    @app.get("/api/models")
    def models():
        return {"slots": _model_slots()}

    @app.get("/api/jobs")
    def jobs():
        return _jobs.as_dict()

    @app.post("/api/jobs")
    def start_job(body: JobIn):
        try:
            entry = by_id(body.exp_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        args = ["--fast"] if body.mode == "fast" and entry.supports_fast else []
        msg = _jobs.start(entry.module, args)
        payload = _jobs.as_dict()
        payload["message"] = msg
        return payload

    @app.post("/api/jobs/cancel")
    def cancel_job():
        msg = _jobs.cancel()
        payload = _jobs.as_dict()
        payload["message"] = msg
        return payload

    @app.post("/api/simulate")
    def simulate(body: SimulateIn):
        traj, cfg = _simulate_preset(body.scenario, body.seed)
        data = _series(traj)
        data.update({
            "scenario": body.scenario,
            "n_qubits": cfg.n_qubits.value,
            "interaction": cfg.interaction_type.value,
            "dissipator": cfg.dissipator.operator_type.value,
            "J": float(cfg.J),
            "omega": float(cfg.omega),
            "t_max": float(cfg.t_max),
            "dt": float(cfg.dt),
        })
        return data

    @app.post("/api/predict")
    def predict(body: PredictIn):
        traj, cfg = _simulate_preset(body.scenario, body.seed)
        nq = cfg.n_qubits.value
        pred, source = _pick_predictor(nq, cfg.interaction_type.value, cfg.dissipator.operator_type.value)
        if hasattr(pred, "inference_interaction_code"):
            if nq == 2:
                pred.inference_interaction_code = 3.0 if cfg.interaction_type.value == "TFIM" else 2.0
            else:
                pred.inference_interaction_code = (
                    1.0 if cfg.dissipator.operator_type.value == "sigma_z" else 0.0
                )
        if hasattr(pred, "inference_dissipator_code"):
            pred.inference_dissipator_code = (
                1.0 if cfg.dissipator.operator_type.value == "sigma_z" else 0.0
            )
        if hasattr(pred, "inference_J"):
            pred.inference_J = float(cfg.J) if nq == 2 else 0.0
        feat = traj.feature_matrix
        window_length = 20 if nq == 2 else 15
        pre = [i for i, t in enumerate(traj.times) if t < traj.t_decoh]
        if len(pre) < window_length:
            raise HTTPException(400, "trajectory shorter than window")
        end_idx = int(pre[-1])
        window = feat[end_idx - window_length : end_idx]
        t_obs = float(traj.times[end_idx])
        result, events = PredictDecoherenceUseCase(pred).execute(
            PredictDecoherenceCommand(
                window=window,
                t_obs=t_obs,
                horizon=float(body.horizon),
                risk_alarm_threshold=float(body.alarm),
            )
        )
        phys_rem, slope, ols_r2 = physics_remaining(window, t_obs, float(cfg.dt), t_max=float(cfg.t_max))
        remaining = float(result.t_decoh_predicted - t_obs)
        risk_uses_horizon = True
        series = _series(traj)
        series.update({
            "scenario": body.scenario,
            "source": source,
            "t_obs": t_obs,
            "window_length": window_length,
            "t_pred": float(result.t_decoh_predicted),
            "remaining": remaining,
            "risk": float(result.risk_score),
            "alarm": float(body.alarm),
            "alarmed": bool(events),
            "phys_rem": float(phys_rem),
            "slope": float(slope),
            "ols_r2": float(ols_r2),
            "horizon": float(body.horizon),
            "risk_uses_horizon": risk_uses_horizon,
        })
        return series

    @app.get("/")
    def index():
        page = STATIC / "index.html"
        if not page.exists():
            raise HTTPException(500, "static/index.html missing")
        return FileResponse(page)

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app
