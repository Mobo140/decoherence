"""FastAPI workbench matching the agreed audit mockup.

Serves static/index.html and JSON for the 7 screens. It starts paper
experiments only through the existing `JobQueue`, with `--fast`.
"""
from __future__ import annotations

import pathlib
from typing import Any

import fastapi
from fastapi import responses
from fastapi import staticfiles
import pydantic

from src.application import experiment_catalog
from src.application import job_queue
from src.application import predict_decoherence
from src.application import simulate_trajectory
from src.data import scenarios
from src.domain import ports
from src.domain import value_objects
from src.models import physics_only
from src.models import physics_prior
from src.persistence import stores
from src.simulation import simulator as simulator_lib

STATIC = pathlib.Path(__file__).resolve().parents[2] / "static"

_jobs = job_queue.JobQueue()
_simulator = simulator_lib.QuTipSimulator()
_ham = stores.HamiltonianModelRegistry()
_best = stores.BestModelRegistry()
_session: dict[str, Any] = {"trajectory": None, "scenario": None}


class SimulateIn(pydantic.BaseModel):
    """Request of `/api/simulate`."""

    scenario: str = "B"
    seed: int = 42


class PredictIn(pydantic.BaseModel):
    """Request of `/api/predict`."""

    scenario: str = "B"
    seed: int = 7
    horizon: float = 1.0
    alarm: float = pydantic.Field(0.7, ge=0.0, le=1.0)
    n_qubits: int = 1
    interaction: str = "XXZ"
    dissipator: str = "sigma_minus"


class JobIn(pydantic.BaseModel):
    """Request of `POST /api/jobs`."""

    exp_id: str = "E8c"
    mode: str = "fast"


class CompareIn(pydantic.BaseModel):
    """Request of `/api/compare`."""

    a: str = "E8c"
    b: str = "E14"


def _series(trajectory: value_objects.TrajectoryResult) -> dict[str, Any]:
    """Returns the plotted series of a trajectory as JSON-ready data."""
    observables = trajectory.observables
    pauli = next(
        (k for k in ("sigma_x", "sigma_x1") if k in observables), None)
    coherence = observables.get("coherence_l1")
    if coherence is None:
        raise fastapi.HTTPException(500, "trajectory has no coherence_l1")
    return {
        "times": [float(x) for x in trajectory.times.tolist()],
        "coherence": [float(x) for x in coherence.tolist()],
        "observable": ([float(x) for x in observables[pauli].tolist()]
                       if pauli else []),
        "observable_name": pauli or "",
        "t_decoh": float(trajectory.t_decoh),
        "censored": bool(getattr(trajectory, "censored", False)),
        "feature_names": list(trajectory.feature_names),
    }


def _simulate_preset(scenario: str, seed: int):
    """Simulates the first configuration of a scenario and keeps it."""
    groups = scenarios.build_configs(n_per_scenario=1, seed=int(seed))
    if scenario not in groups:
        raise fastapi.HTTPException(400, f"unknown scenario {scenario}")
    config = groups[scenario][0]
    trajectory = simulate_trajectory.SimulateTrajectoryUseCase(
        _simulator
    ).execute(simulate_trajectory.SimulateTrajectoryCommand(
        config=config, seed=int(seed)))
    _session["trajectory"] = trajectory
    _session["scenario"] = scenario
    _session["config"] = config
    return trajectory, config


def _pick_predictor(
    n_qubits: int, interaction: str, dissipator: str
) -> tuple[ports.IPredictor, str]:
    """Returns the stored predictor for a system, or the physics fallback."""
    predictor = (
        _ham.get_predictor(
            n_qubits,
            value_objects.InteractionType(interaction),
            value_objects.DissipatorType(dissipator))
        or _best.get_predictor(n_qubits)
    )
    if predictor is not None and predictor.is_trained:
        return predictor, "registry"
    return physics_only.PhysicsOnlyPredictor(), "physics_only fallback"


def _json_safe(value: Any) -> Any:
    """Replaces nan and infinity by None, recursively."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (
            value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _model_slots() -> list[dict[str, Any]]:
    """Describes the four model slots of the registry for the UI."""
    sigma_minus = value_objects.DissipatorType.SIGMA_MINUS
    xxz = value_objects.InteractionType.XXZ
    tfim = value_objects.InteractionType.TFIM
    specs = (
        (1, xxz, sigma_minus, "1q / σ₋ · amplitude damping"),
        (1, xxz, value_objects.DissipatorType.SIGMA_Z,
         "1q / σz · dephasing"),
        (2, xxz, sigma_minus, "2q / XXZ / σ₋"),
        (2, tfim, sigma_minus, "2q / TFIM / σ₋"),
    )
    slots = []
    for n_qubits, interaction, dissipator, kicker in specs:
        meta = _ham.get_meta(n_qubits, interaction, dissipator)
        slots.append({
            "kicker": kicker,
            "label": stores.slot_label(n_qubits, interaction, dissipator),
            "n_qubits": n_qubits,
            "interaction": interaction.value,
            "dissipator": dissipator.value,
            "meta": _json_safe(meta),
            "warn": n_qubits == 2 and interaction == tfim,
            "empty": meta is None,
        })
    return slots


def _set_inference_context(predictor, config) -> None:
    """Gives context-aware predictors the system's codes and coupling."""
    n_qubits = config.n_qubits.value
    is_sigma_z = config.dissipator.operator_type.value == "sigma_z"
    if hasattr(predictor, "inference_interaction_code"):
        if n_qubits == 2:
            predictor.inference_interaction_code = (
                3.0 if config.interaction_type.value == "TFIM" else 2.0)
        else:
            predictor.inference_interaction_code = 1.0 if is_sigma_z else 0.0
    if hasattr(predictor, "inference_dissipator_code"):
        predictor.inference_dissipator_code = 1.0 if is_sigma_z else 0.0
    if hasattr(predictor, "inference_J"):
        predictor.inference_J = float(config.J) if n_qubits == 2 else 0.0


def create_app() -> fastapi.FastAPI:
    """Builds the workbench application.

    Returns:
        The FastAPI app with the JSON endpoints and the static pages.
    """
    app = fastapi.FastAPI(
        title="Q-EWS Workbench", docs_url=None, redoc_url=None)

    @app.get("/api/meta")
    def meta():
        """Champions, claims and cache statistics."""
        return {
            "champions": experiment_catalog.CHAMPIONS,
            "claims": experiment_catalog.CLAIMS,
            "cache": experiment_catalog.cache_stats(),
            "paper": "Paper 1 готов · Paper 2: TFIM R² 0.575 < 0.70 (E8c)",
        }

    @app.get("/api/catalog")
    def catalog():
        """The experiment catalog with the result files that exist."""
        rows = []
        for entry in experiment_catalog.CATALOG:
            paths = experiment_catalog.list_csv_paths(entry)
            rows.append({
                "id": entry.id,
                "title": entry.title,
                "module": entry.module,
                "paper": entry.paper,
                "claim": entry.claim,
                "scenarios": entry.scenarios,
                "supports_fast": entry.supports_fast,
                "note": entry.note,
                "n_csv": len(paths),
                "csv": [p.name for p in paths],
            })
        return {"experiments": rows}

    @app.get("/api/csv/{exp_id}")
    def csv_rows(exp_id: str):
        """The first rows of an experiment's first result file."""
        try:
            entry = experiment_catalog.by_id(exp_id)
        except KeyError as exc:
            raise fastapi.HTTPException(404, str(exc)) from exc
        paths = experiment_catalog.list_csv_paths(entry)
        if not paths:
            return {"headers": [], "rows": [], "file": None}
        rows = experiment_catalog.read_csv_rows(paths[0], limit=40)
        headers = list(rows[0].keys()) if rows else []
        wanted = ("scenario", "name", "variant", "arm", "j_bin", "r2", "mae",
                  "auroc", "n_samples", "use_J", "window", "noise_sigma")
        keep = [k for k in headers if k in wanted] or headers[:8]
        return {
            "file": paths[0].name,
            "headers": keep,
            "rows": [{k: r.get(k, "") for k in keep} for r in rows],
        }

    @app.post("/api/compare")
    def compare(body: CompareIn):
        """A text comparing two experiments' results."""
        try:
            text = experiment_catalog.compare_summaries(body.a, body.b)
        except KeyError as exc:
            raise fastapi.HTTPException(404, str(exc)) from exc
        return {"text": text, "a": body.a, "b": body.b}

    @app.get("/api/models")
    def models():
        """The model slots of the registry."""
        return {"slots": _model_slots()}

    @app.get("/api/jobs")
    def jobs():
        """The state of the background job."""
        return _jobs.as_dict()

    @app.post("/api/jobs")
    def start_job(body: JobIn):
        """Starts an experiment, with --fast where the script has it."""
        try:
            entry = experiment_catalog.by_id(body.exp_id)
        except KeyError as exc:
            raise fastapi.HTTPException(404, str(exc)) from exc
        fast = body.mode == "fast" and entry.supports_fast
        message = _jobs.start(entry.module, ["--fast"] if fast else [])
        payload = _jobs.as_dict()
        payload["message"] = message
        return payload

    @app.post("/api/jobs/cancel")
    def cancel_job():
        """Cancels the background job."""
        message = _jobs.cancel()
        payload = _jobs.as_dict()
        payload["message"] = message
        return payload

    @app.post("/api/simulate")
    def simulate(body: SimulateIn):
        """Simulates a scenario preset and returns its series."""
        trajectory, config = _simulate_preset(body.scenario, body.seed)
        data = _series(trajectory)
        data.update({
            "scenario": body.scenario,
            "n_qubits": config.n_qubits.value,
            "interaction": config.interaction_type.value,
            "dissipator": config.dissipator.operator_type.value,
            "J": float(config.J),
            "omega": float(config.omega),
            "t_max": float(config.t_max),
            "dt": float(config.dt),
        })
        return data

    @app.post("/api/predict")
    def predict(body: PredictIn):
        """Predicts the decoherence time of a simulated preset."""
        trajectory, config = _simulate_preset(body.scenario, body.seed)
        n_qubits = config.n_qubits.value
        predictor, source = _pick_predictor(
            n_qubits, config.interaction_type.value,
            config.dissipator.operator_type.value)
        _set_inference_context(predictor, config)
        window_length = 20 if n_qubits == 2 else 15
        before = [i for i, t in enumerate(trajectory.times)
                  if t < trajectory.t_decoh]
        if len(before) < window_length:
            raise fastapi.HTTPException(400, "trajectory shorter than window")
        end = int(before[-1])
        window = trajectory.feature_matrix[end - window_length:end]
        t_obs = float(trajectory.times[end])
        result, alarms = predict_decoherence.PredictDecoherenceUseCase(
            predictor
        ).execute(predict_decoherence.PredictDecoherenceCommand(
            window=window,
            t_obs=t_obs,
            horizon=float(body.horizon),
            risk_alarm_threshold=float(body.alarm),
        ))
        phys_rem, slope, ols_r2 = physics_prior.physics_remaining(
            window, t_obs, float(config.dt), t_max=float(config.t_max))
        series = _series(trajectory)
        series.update({
            "scenario": body.scenario,
            "source": source,
            "t_obs": t_obs,
            "window_length": window_length,
            "t_pred": float(result.t_decoh_predicted),
            "remaining": float(result.t_decoh_predicted - t_obs),
            "risk": float(result.risk_score),
            "alarm": float(body.alarm),
            "alarmed": bool(alarms),
            "phys_rem": float(phys_rem),
            "slope": float(slope),
            "ols_r2": float(ols_r2),
            "horizon": float(body.horizon),
            "risk_uses_horizon": True,
        })
        return series

    @app.get("/")
    def index():
        """The single-page UI."""
        page = STATIC / "index.html"
        if not page.exists():
            raise fastapi.HTTPException(500, "static/index.html missing")
        return responses.FileResponse(page)

    if STATIC.exists():
        app.mount("/static", staticfiles.StaticFiles(directory=str(STATIC)),
                  name="static")
    return app
