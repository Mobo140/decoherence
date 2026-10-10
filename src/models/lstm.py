"""Physics-informed decoherence predictor: OLS estimate + BiLSTM residual.

Primary estimate (pure physics, for sigma_minus with constant gamma and a
1/e threshold). The coherence decays as C(t) = C0 * exp(-gamma * t / 2), so
log C is linear in t and gamma = -2 * slope, where slope is the OLS slope of
log(coherence_l1) over the observation window. With the 1/e threshold the
decoherence time is T2 = 2 / gamma, hence::

    remaining = T2 - t_obs = -1 / slope - t_obs

This formula alone gives R^2 ~ 0.999 on held-out single-qubit data.

Residual correction. The estimate deviates for time-dependent gamma, for
sigma_z dephasing and for two qubits. A BiLSTM learns the correction.

Architecture::

    window (N, T, F) -> log C and log purity appended
      -> BiLSTM (hidden_size per direction) -> mean pool (2 * hidden_size)
      -> concatenated with the physics scalars (7, or 8 with use_J_scalar):
         t_obs, slope, log C(t_obs), physics remaining time, OLS R^2,
         interaction code, dissipator code, [J]
      -> regression head -> residual dT
      -> risk head -> logit (training auxiliary)

    remaining = max(0, phys_rem + dT)   (phys_rem is 0 without the prior)
    T2_hat = clip(t_obs + remaining, 0, t_max)

Codes: interaction 0 = 1q sigma_-, 1 = 1q sigma_z, 2 = XXZ, 3 = TFIM;
dissipator 0 = sigma_-, 1 = sigma_z. At training time the risk head learns
P(remaining <= dt) at the dataset horizon; `predict` ignores its logit and
returns `remaining_to_risk(remaining, horizon)`.
"""
from __future__ import annotations

import dataclasses
import io

import numpy as np
import torch
from torch import nn

from src.data import dataset as dataset_lib
from src.domain import ports
from src.domain import value_objects
from src.models import lstm_data
from src.models import physics_prior
from src.models import residual_net
from src.training import loop
from src.training import losses
from src.training import standardize

# R^2_OLS is nearly constant on clean single-qubit windows (s.d. ~1e-3);
# standardising by that s.d. turns any measurement noise into inputs of
# order -100. Differences below 0.05 in a [0, 1] goodness-of-fit carry no
# information, so the scale is floored there (two qubits: s.d. ~0.2,
# unaffected).
R2_SCALE_FLOOR = 0.05

# Scalar inputs, in the order of `ResidualNet.forward`, and the stem of
# their checkpoint keys ("<stem>_mean", "<stem>_std").
SCALAR_KEYS = ("t_obs", "slope", "logc", "phys", "r2", "ic", "dc", "j")
_RESIDUAL_KEY = "residual"
# Scalers stored under their own key pair in a checkpoint ("j" is stored
# last, after the configuration).
_CHECKPOINT_SCALERS = (*SCALAR_KEYS[:-1], _RESIDUAL_KEY)

_MIN_LR = 1e-5


def n_physics_from_state(state_dict: dict, hidden_size: int) -> int | None:
    """Infers how many physics scalars a checkpoint concatenated.

    Args:
        state_dict: The network's state dict.
        hidden_size: LSTM hidden size per direction.

    Returns:
        The number of scalars (4, 7 or 8), or None if the state dict has no
        regression head to read it from.
    """
    weight = state_dict.get("regression_head.0.weight")
    if weight is None:
        return None
    n = int(weight.shape[1] - hidden_size * 2)
    if n < 1:
        return None
    return n


def _identity_scalers() -> dict[str, standardize.Standardizer]:
    keys = (*SCALAR_KEYS, _RESIDUAL_KEY)
    return {key: standardize.Standardizer() for key in keys}


@dataclasses.dataclass
class LSTMPredictor(ports.IPredictor):
    """Physics estimate plus BiLSTM residual correction.

    Attributes:
        hidden_size: LSTM hidden size per direction.
        num_layers: Number of LSTM layers.
        dropout: Dropout in the heads (and between LSTM layers).
        alpha: Weight of the regression loss.
        beta: Weight of the risk (BCE) loss.
        t_max: Hard upper bound on the predicted T2.
        use_physics_prior: If True, the OLS estimate is an input scalar and
            the network predicts a correction to it. If False it is zeroed
            and the network learns the whole prediction from the raw
            sequences (the lstm_only ablation, experiment E1).
        adaptive_prior_r2_threshold: If positive (0.7 for two qubits), the
            OLS prior is replaced by 0 for windows whose own R^2 is below
            it. 0 disables the gate.
        inference_interaction_code: Interaction code at prediction time
            (0 = 1q sigma_-, 1 = 1q sigma_z, 2 = XXZ, 3 = TFIM).
        inference_dissipator_code: Dissipator code at prediction time
            (0 = sigma_-, 1 = sigma_z).
        use_J_scalar: If True, the coupling J is an eighth physics scalar.
            False keeps the 7-scalar network so old checkpoints load.
        inference_J: Coupling at prediction time; `BacktestUseCase` sets it
            per window from `Dataset.J_values`.
        risk_pos_weight: Positive-class weight of the risk BCE. 0 means
            automatic from the label ratio, clamped to [1, 10]. Values > 1
            help AUROC on imbalanced labels.
        augment_sigma: If positive, Gaussian noise of this size (in
            observable units) is added to the inputs during training only,
            a regulariser that helps cross-system generalisation.
        early_stopping_patience: Epochs without improvement before
            stopping; raise it for large datasets (> 2000 samples).
        residual_output: If True, remaining = T_phys + dT. If False, T_phys
            is only an input scalar and the network predicts the whole
            remaining time (separates "useful feature" from "residual
            output").
        net: The trained network, None before training.
        dt: Time step between window rows, synced from the training
            dataset.
        feature_scaler: Per-channel standardiser of the windows.
        scalers: Standardisers of the scalar inputs (`SCALAR_KEYS`) and of
            the regression target (key "residual").
        device: Torch device the network runs on.
    """

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.3
    alpha: float = 1.0
    beta: float = 1.0
    t_max: float = 10.0
    use_physics_prior: bool = True
    adaptive_prior_r2_threshold: float = 0.0
    inference_interaction_code: float = 0.0
    inference_dissipator_code: float = 0.0
    use_J_scalar: bool = False
    inference_J: float = 0.0
    risk_pos_weight: float = 1.0
    augment_sigma: float = 0.0
    early_stopping_patience: int = 25
    residual_output: bool = True

    net: residual_net.ResidualNet | None = dataclasses.field(
        default=None, init=False, repr=False)
    dt: float = dataclasses.field(default=0.1, init=False, repr=False)
    feature_scaler: standardize.Standardizer | None = dataclasses.field(
        default=None, init=False, repr=False)
    scalers: dict[str, standardize.Standardizer] = dataclasses.field(
        default_factory=_identity_scalers, init=False, repr=False)
    device: str = dataclasses.field(default="cpu", init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_trained(self) -> bool:
        return self.net is not None

    def predict(
        self, window: np.ndarray, t_obs: float, horizon: float
    ) -> value_objects.PredictionResult:
        """Predicts the decoherence time of one window.

        Args:
            window: Array of shape (window_length, n_features); the last
                column is coherence_l1.
            t_obs: Absolute time at the end of the window.
            horizon: Time window for the risk score.

        Returns:
            The predicted decoherence time and risk score.

        Raises:
            RuntimeError: If the predictor is not trained.
        """
        if self.net is None:
            raise RuntimeError("LSTMPredictor is not trained.")

        phys_rem, slope, ols_r2 = physics_prior.physics_remaining(
            window, t_obs, self.dt,
            t_max=self.t_max,
            adaptive_r2_threshold=self.adaptive_prior_r2_threshold,
        )
        if not self.use_physics_prior:
            phys_rem = 0.0
            slope = 0.0
        log_coh = float(np.log(np.abs(window[-1, -1]) + 1e-8))
        values = {
            "t_obs": t_obs, "slope": slope, "logc": log_coh,
            "phys": phys_rem, "r2": ols_r2,
            "ic": self.inference_interaction_code,
            "dc": self.inference_dissipator_code,
        }

        self.net.eval()
        with torch.no_grad():
            x = self._standardise_window(window)
            inputs = [
                lstm_data.scalar_tensor(v, self.scalers[k], self.device)
                for k, v in values.items()
            ]
            j_in = None
            if self.use_J_scalar:
                j_in = lstm_data.scalar_tensor(
                    self.inference_J, self.scalers["j"], self.device)

            residual_norm, _risk_logit = self.net(x, *inputs, j_in)

            target = self.scalers[_RESIDUAL_KEY]
            residual = float(residual_norm.item()) * target.std + target.mean
            base = phys_rem if self.residual_output else 0.0
            remaining = max(0.0, base + residual)
            t_decoh_pred = float(np.clip(t_obs + remaining, 0.0, self.t_max))
            risk = physics_prior.remaining_to_risk(remaining, horizon)

        return value_objects.PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=risk,
            horizon=horizon,
        )

    def train(self, dataset: object, **kwargs) -> loop.TrainResult:
        """Fits the network to the training split of a dataset.

        Args:
            dataset: A `Dataset`; its train split is fitted and its val
                split drives early stopping.
            **kwargs: See `loop.FitOptions` (n_epochs, batch_size, lr,
                verbose, regression_loss, seed).

        Returns:
            The predictor and the loss histories.

        Raises:
            TypeError: If `dataset` is not a `Dataset`.
        """
        if not isinstance(dataset, dataset_lib.Dataset):
            raise TypeError("LSTMPredictor.train needs a Dataset.")
        options = loop.FitOptions.from_kwargs(kwargs)
        if options.seed is not None:
            # Weight init, dropout and batch shuffling all draw from
            # torch's global RNG; without this the same command gives
            # different results.
            torch.manual_seed(options.seed)
        train_hist, val_hist = self._fit(dataset, options)
        return loop.TrainResult(
            predictor=self,
            train_loss_history=train_hist,
            val_loss_history=val_hist,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_bytes(self) -> bytes:
        """Serialises the network, scalers and configuration (torch.save).

        Returns:
            The checkpoint bytes read back by `from_bytes`.

        Raises:
            RuntimeError: If the predictor is not trained.
        """
        if self.net is None:
            raise RuntimeError("Cannot serialise an untrained predictor.")
        ckpt = {
            "state_dict": self.net.state_dict(),
            "feature_mean": self.feature_scaler.mean,
            "feature_std": self.feature_scaler.std,
        }
        for key in _CHECKPOINT_SCALERS:
            ckpt[f"{key}_mean"] = self.scalers[key].mean
            ckpt[f"{key}_std"] = self.scalers[key].std
        ckpt.update({
            "dt": self.dt,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "n_features": self.net.lstm.input_size,
            "use_physics_prior": self.use_physics_prior,
            "adaptive_prior_r2_threshold": self.adaptive_prior_r2_threshold,
            "risk_pos_weight": self.risk_pos_weight,
            "augment_sigma": self.augment_sigma,
            "residual_output": self.residual_output,
            "inference_interaction_code": self.inference_interaction_code,
            "inference_dissipator_code": self.inference_dissipator_code,
            "use_J_scalar": self.use_J_scalar,
            "inference_J": self.inference_J,
            "j_mean": self.scalers["j"].mean,
            "j_std": self.scalers["j"].std,
        })
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> LSTMPredictor:
        """Restores a predictor from `state_bytes` (also old checkpoints).

        Args:
            data: Checkpoint bytes.

        Returns:
            The trained predictor, with its network in eval mode.
        """
        ckpt = torch.load(io.BytesIO(data), map_location="cpu",
                          weights_only=False)
        p = cls(
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            use_physics_prior=ckpt.get("use_physics_prior", True),
            adaptive_prior_r2_threshold=ckpt.get(
                "adaptive_prior_r2_threshold", 0.0),
            risk_pos_weight=ckpt.get("risk_pos_weight", 1.0),
            augment_sigma=ckpt.get("augment_sigma", 0.0),
            residual_output=ckpt.get("residual_output", True),
            inference_interaction_code=ckpt.get(
                "inference_interaction_code", 0.0),
            inference_dissipator_code=ckpt.get(
                "inference_dissipator_code", 0.0),
            use_J_scalar=ckpt.get("use_J_scalar", False),
            inference_J=ckpt.get("inference_J", 0.0),
        )
        n_physics = n_physics_from_state(
            ckpt["state_dict"], ckpt["hidden_size"])
        if n_physics is None:
            n_physics = 8 if p.use_J_scalar else 7
        p.use_J_scalar = n_physics >= 8
        net = residual_net.ResidualNet(
            n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"], dropout=ckpt["dropout"],
            n_physics=n_physics)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        p.net = net
        p.feature_scaler = standardize.Standardizer(
            ckpt["feature_mean"], ckpt["feature_std"])
        # Checkpoints from before these inputs existed lack their keys; the
        # identity map is what such a network was trained with.
        for key in ("t_obs", "slope", "logc", "phys", _RESIDUAL_KEY):
            p.scalers[key] = standardize.Standardizer(
                ckpt[f"{key}_mean"], ckpt[f"{key}_std"])
        for key in ("r2", "ic", "dc", "j"):
            p.scalers[key] = standardize.Standardizer(
                ckpt.get(f"{key}_mean", 0.0), ckpt.get(f"{key}_std", 1.0))
        p.dt = ckpt.get("dt", 0.1)
        return p

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _fit(
        self, ds: dataset_lib.Dataset, options: loop.FitOptions
    ) -> tuple[list[float], list[float]]:
        # The physics formulas must use the actual simulation step.
        self.dt = float(ds.metadata.get("dt", self.dt))
        train = self._arrays(ds, ds.train_idx)
        val = self._arrays(ds, ds.val_idx)
        self._fit_scalers(train)
        # predict() standardises every window with these statistics, so the
        # training windows must be standardised the same way.
        train = train._replace(windows=self._standardise_array(train.windows))
        val = val._replace(windows=self._standardise_array(val.windows))

        self.net = residual_net.ResidualNet(
            train.windows.shape[-1], self.hidden_size, self.num_layers,
            self.dropout, n_physics=8 if self.use_J_scalar else 7,
        ).to(self.device)
        objective = _Objective(
            self, train, float(ds.metadata.get("horizon", 1.0)),
            options.regression_loss)
        return loop.fit(
            self.net,
            lstm_data.make_loader(train, options.batch_size, True),
            lstm_data.make_loader(val, options.batch_size, False),
            step=objective,
            n_epochs=options.n_epochs,
            lr=options.lr,
            patience=self.early_stopping_patience,
            min_lr=_MIN_LR,
            verbose=options.verbose,
            label="LSTM",
        )

    def _arrays(
        self, ds: dataset_lib.Dataset, idx: np.ndarray
    ) -> lstm_data.Arrays:
        return lstm_data.prepare_arrays(
            ds, idx,
            dt=self.dt,
            t_max=self.t_max,
            adaptive_r2_threshold=self.adaptive_prior_r2_threshold,
            use_physics_prior=self.use_physics_prior,
            residual_output=self.residual_output,
        )

    def _fit_scalers(self, train: lstm_data.Arrays) -> None:
        fit = standardize.Standardizer.fit
        self.feature_scaler = fit(train.windows, axis=(0, 1))
        self.scalers = {
            "t_obs": fit(train.t_obs),
            "slope": fit(train.slope),
            "logc": fit(train.log_coh),
            "phys": fit(train.phys),
            "r2": fit(train.r2, min_std=R2_SCALE_FLOOR),
            "ic": fit(train.int_code),
            "dc": fit(train.dis_code),
            "j": fit(train.j),
        }
        # Censored rows carry only a lower bound, so they stay out of the
        # target statistics unless every row is censored.
        uncensored = train.censored < 0.5
        target = (train.residual[uncensored] if uncensored.any()
                  else train.residual)
        self.scalers[_RESIDUAL_KEY] = fit(target)

    def _standardise_array(self, windows: np.ndarray) -> np.ndarray:
        return self.feature_scaler.apply(windows).astype(np.float32)

    def _standardise_window(self, window: np.ndarray) -> torch.Tensor:
        # Pad raw features to the dimension seen at training time:
        # coherence_l1 and purity are always the last 2 raw columns, so
        # zeros go before them (same convention as generate_dataset).
        n_raw_expected = self.net.lstm.input_size - 2  # add_log adds 2.
        if window.shape[-1] < n_raw_expected:
            n_pad = n_raw_expected - window.shape[-1]
            window = np.concatenate(
                [window[:, :-2],
                 np.zeros((window.shape[0], n_pad), dtype=window.dtype),
                 window[:, -2:]],
                axis=-1,
            )
        w = lstm_data.add_log(window[np.newaxis], [])[0]
        normed = self.feature_scaler.apply(w)
        return torch.tensor(normed, dtype=torch.float32).unsqueeze(0).to(
            self.device)


class _Objective:
    """The LSTM's loss on one batch; the `step` of `loop.fit`.

    Regression loss on the standardised residual (Huber by default; a
    one-sided Huber on the censored rows when the data has any) plus a
    positive-weighted BCE on the risk logit.
    """

    def __init__(
        self,
        predictor: LSTMPredictor,
        train: lstm_data.Arrays,
        horizon: float,
        regression_loss: str,
    ) -> None:
        self._p = predictor
        self._horizon = horizon
        self._use_surv = bool(train.censored.any())
        # surv_huber on uncensored rows is plain Huber; the censored part
        # is handled by `_surv`.
        name = "huber" if regression_loss == "surv_huber" \
            else regression_loss
        self._reg_loss = losses.get_loss(name).as_torch()
        self._surv = losses.SurvHuberLoss(delta=1.0)
        self._pos_weight = torch.tensor(
            self._positive_weight(train.risk), dtype=torch.float32)
        self._feature_std = torch.tensor(
            predictor.feature_scaler.std, dtype=torch.float32
        ).to(predictor.device)

    def _positive_weight(self, risk: np.ndarray) -> float:
        # Weight 0 asks for the label ratio, clamped to [1, 10].
        weight = self._p.risk_pos_weight
        if weight != 0.0:
            return float(weight)
        n_pos = float(risk.sum()) + 1e-3
        n_neg = float(len(risk) - n_pos) + 1e-3
        return float(np.clip(n_neg / n_pos, 1.0, 10.0))

    def __call__(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        p = self._p
        device = p.device
        (xb, yb, tb, sb, lb, pb, r2b, ib, db, jb, rb, remb, cb) = batch
        xb = xb.to(device)
        rb = rb.to(device)
        remb = remb.to(device)
        cb = cb.to(device)
        pb = pb.to(device)
        if p.net.training and p.augment_sigma > 0.0:
            # augment_sigma is measurement noise in observable units; the
            # batch is standardised, so scale it channel by channel.
            xb = xb + torch.randn_like(xb) * (
                p.augment_sigma / self._feature_std)
        sc = p.scalers
        yn = sc[_RESIDUAL_KEY].apply(yb).to(device)
        tn = sc["t_obs"].apply(tb).to(device)
        sn = sc["slope"].apply(sb).to(device)
        ln = sc["logc"].apply(lb).to(device)
        pn = sc["phys"].apply(pb)
        r2n = sc["r2"].apply(r2b).to(device)
        itn = sc["ic"].apply(ib).to(device)
        dtn = sc["dc"].apply(db).to(device)
        jn = sc["j"].apply(jb).to(device) if p.use_J_scalar else None

        res_pred, risk_pred = p.net(xb, tn, sn, ln, pn, r2n, itn, dtn, jn)
        reg = self._regression(res_pred, yn, pb, remb, cb)
        # A censored window already inside the horizon has no risk label.
        risk_valid = ~((cb > 0.5) & (remb <= self._horizon))
        if risk_valid.any():
            bce = nn.functional.binary_cross_entropy_with_logits(
                risk_pred[risk_valid], rb[risk_valid],
                pos_weight=self._pos_weight.to(risk_pred.device),
            )
        else:
            bce = risk_pred.new_zeros(())
        return p.alpha * reg + p.beta * bce

    def _regression(self, res_pred, yn, pb, remb, cb) -> torch.Tensor:
        if not self._use_surv:
            return self._reg_loss(res_pred, yn)
        target = self._p.scalers[_RESIDUAL_KEY]
        pred_rem = pb + res_pred * target.std + target.mean
        unc = cb < 0.5
        acc = res_pred.new_zeros(())
        if unc.any():
            acc = acc + self._reg_loss(res_pred[unc], yn[unc]) * unc.sum()
        if (~unc).any():
            ones = torch.ones_like(cb[~unc])
            acc = acc + self._surv(
                pred_rem[~unc], remb[~unc], ones) * (~unc).sum()
        return acc / cb.numel()
