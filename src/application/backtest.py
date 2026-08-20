"""Use-case: evaluate a trained predictor on the held-out test split."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.ports import IPredictor
from ..domain.value_objects import BacktestMetrics
from .generate_dataset import Dataset


@dataclass(frozen=True)
class BacktestCommand:
    """Input for BacktestUseCase."""

    dataset: Dataset
    horizon: float = 1.0


class BacktestUseCase:
    """Run inference on the test split and compute regression + ranking metrics.

    Metrics:
        Regression  – MAE, RMSE, R², MAPE (predicting t_decoh)
        Ranking     – AUROC (predicting binary risk within `horizon`)
    """

    def __init__(self, predictor: IPredictor) -> None:
        self._predictor = predictor

    def execute(self, command: BacktestCommand) -> BacktestMetrics:
        if not self._predictor.is_trained:
            raise RuntimeError(
                "Predictor is not trained. Run TrainModelUseCase first."
            )

        ds = command.dataset
        test_idx = ds.test_idx
        # R² against t_max on censored rows is meaningless — evaluate uncensored only.
        if len(getattr(ds, "censored", [])) == len(ds.sequences) and len(test_idx) > 0:
            uncens = ~ds.censored[test_idx]
            if uncens.any():
                test_idx = test_idx[uncens]
        test_seq = ds.sequences[test_idx]
        test_t_obs = ds.t_obs[test_idx]
        test_t_decoh = ds.t_decoh_abs[test_idx]
        test_risk = ds.risk_labels[test_idx]
        n_all = len(ds.sequences)
        has_J = (
            len(getattr(ds, "J_values", [])) == n_all
            and hasattr(self._predictor, "inference_J")
        )
        has_ic = (
            len(getattr(ds, "interaction_type_codes", [])) == n_all
            and hasattr(self._predictor, "inference_interaction_code")
        )
        has_dc = (
            len(getattr(ds, "dissipator_type_codes", [])) == n_all
            and hasattr(self._predictor, "inference_dissipator_code")
        )

        predictions: list[float] = []
        risk_scores: list[float] = []

        for i, idx in enumerate(test_idx):
            if has_J:
                self._predictor.inference_J = float(ds.J_values[idx])
            if has_ic:
                self._predictor.inference_interaction_code = float(ds.interaction_type_codes[idx])
            if has_dc:
                self._predictor.inference_dissipator_code = float(ds.dissipator_type_codes[idx])
            result = self._predictor.predict(
                test_seq[i], float(test_t_obs[i]), command.horizon
            )
            predictions.append(result.t_decoh_predicted)
            risk_scores.append(result.risk_score)

        preds = np.array(predictions, dtype=np.float64)
        actuals = test_t_decoh.astype(np.float64)

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        mask = actuals > 1e-8
        mape = (
            float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100)
            if mask.any()
            else float("nan")
        )

        risk_scores_arr = np.array(risk_scores)
        try:
            from sklearn.metrics import roc_auc_score
            # roc_auc_score requires both classes present
            if len(np.unique(test_risk)) > 1:
                auroc = float(roc_auc_score(test_risk, risk_scores_arr))
            else:
                auroc = float("nan")
        except Exception:
            auroc = float("nan")

        return BacktestMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            risk_auroc=auroc,
            n_samples=len(test_seq),
            predictions=preds,
            actuals=actuals,
        )
