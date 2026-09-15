"""Training loop for LSTMPredictor.

Extracted verbatim from LSTMPredictor._fit (task 19) -- no behaviour change.
``self`` became the explicit ``predictor`` parameter; the body is otherwise
untouched, including the nested prep/_reg_loss/_bce closures.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .batching import _add_log, _batch_physics, _loader, _ns
from .residual_net import _ResidualNet


def fit_predictor(predictor, ds, n_epochs, batch_size, lr, verbose, regression_loss: str = "huber"):
    from ...application.generate_dataset import Dataset as DS
    ds: DS = ds

    # Sync dt from dataset so physics formulas use the actual simulation step.
    predictor._dt = float(ds.metadata.get("dt", predictor._dt))

    def prep(idx):
        X   = ds.sequences[idx]
        rem = ds.remaining_time[idx]
        t   = ds.t_obs[idx]
        r   = ds.risk_labels[idx]
        slopes, logcs, phys, r2s = _batch_physics(
            X, ds.feature_names, predictor._dt, t, predictor.t_max,
            adaptive_r2_threshold=predictor.adaptive_prior_r2_threshold,
        )
        if not predictor.use_physics_prior:
            phys    = np.zeros_like(phys)
            slopes  = np.zeros_like(slopes)
        residuals = (rem - phys).astype(np.float32)
        X_aug = _add_log(X, ds.feature_names)
        # Context codes: use dataset arrays if available, else zeros.
        int_codes = (ds.interaction_type_codes[idx]
                     if len(ds.interaction_type_codes) == len(ds.sequences)
                     else np.zeros(len(idx), dtype=np.float32))
        dis_codes = (ds.dissipator_type_codes[idx]
                     if len(ds.dissipator_type_codes) == len(ds.sequences)
                     else np.zeros(len(idx), dtype=np.float32))
        j_vals = (ds.J_values[idx]
                  if len(getattr(ds, "J_values", [])) == len(ds.sequences)
                  else np.zeros(len(idx), dtype=np.float32))
        if len(getattr(ds, "censored", [])) == len(ds.sequences):
            cens = ds.censored[idx].astype(np.float32)
        else:
            cens = np.zeros(len(idx), dtype=np.float32)
        return X_aug, rem, t, slopes, logcs, phys, r2s, int_codes, dis_codes, j_vals, residuals, r, cens

    X_tr, rem_tr, t_tr, sl_tr, lc_tr, ph_tr, r2_tr, ic_tr, dc_tr, j_tr, res_tr, r_tr, c_tr = prep(ds.train_idx)
    X_va, rem_va, t_va, sl_va, lc_va, ph_va, r2_va, ic_va, dc_va, j_va, res_va, r_va, c_va = prep(ds.val_idx)

    # normalisation
    predictor._feature_mean = X_tr.mean(axis=(0,1))
    predictor._feature_std  = X_tr.std(axis=(0,1)) + 1e-8
    predictor._t_obs_mean    = float(t_tr.mean())
    predictor._t_obs_std     = float(t_tr.std()) + 1e-8
    predictor._slope_mean    = float(sl_tr.mean())
    predictor._slope_std     = float(sl_tr.std()) + 1e-8
    predictor._logc_mean     = float(lc_tr.mean())
    predictor._logc_std      = float(lc_tr.std()) + 1e-8
    predictor._phys_mean     = float(ph_tr.mean())
    predictor._phys_std      = float(ph_tr.std()) + 1e-8
    predictor._r2_mean       = float(r2_tr.mean())
    predictor._r2_std        = float(r2_tr.std()) + 1e-8
    predictor._ic_mean       = float(ic_tr.mean())
    predictor._ic_std        = float(ic_tr.std()) + 1e-8
    predictor._dc_mean       = float(dc_tr.mean())
    predictor._dc_std        = float(dc_tr.std()) + 1e-8
    predictor._j_mean        = float(j_tr.mean())
    predictor._j_std         = float(j_tr.std()) + 1e-8
    unc_tr = c_tr < 0.5
    res_for_stats = res_tr[unc_tr] if unc_tr.any() else res_tr
    predictor._residual_mean = float(res_for_stats.mean())
    predictor._residual_std  = float(res_for_stats.std()) + 1e-8

    n_feat = X_tr.shape[-1]
    n_physics = 8 if predictor.use_J_scalar else 7
    predictor._net = _ResidualNet(
        n_feat, predictor.hidden_size, predictor.num_layers, predictor.dropout, n_physics=n_physics,
    ).to(predictor._device)

    opt   = torch.optim.AdamW(predictor._net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
    from .loss_functions import SurvHuberLoss, get_loss
    use_surv = bool(c_tr.any())
    _reg_name = "huber" if regression_loss == "surv_huber" else regression_loss
    reg_loss_fn = get_loss(_reg_name).as_torch()
    surv_fn = SurvHuberLoss(delta=1.0)
    horizon = float(ds.metadata.get("horizon", 1.0))

    def _reg_loss(res_pred, yn, pb, remb, cb):
        if not use_surv:
            return reg_loss_fn(res_pred, yn)
        pred_rem = pb + res_pred * predictor._residual_std + predictor._residual_mean
        unc = cb < 0.5
        acc = res_pred.new_zeros(())
        if unc.any():
            acc = acc + reg_loss_fn(res_pred[unc], yn[unc]) * unc.sum()
        if (~unc).any():
            ones = torch.ones_like(cb[~unc])
            acc = acc + surv_fn(pred_rem[~unc], remb[~unc], ones) * (~unc).sum()
        return acc / cb.numel()

    # Weighted BCE: up-weight the positive (imminent-decoherence) class.
    # risk_pos_weight == 0 → auto-compute from label ratio; clamp to [1, 10].
    if predictor.risk_pos_weight == 0.0:
        n_pos = float(r_tr.sum()) + 1e-3
        n_neg = float(len(r_tr) - n_pos) + 1e-3
        _pw = float(np.clip(n_neg / n_pos, 1.0, 10.0))
    else:
        _pw = float(predictor.risk_pos_weight)
    _pw_tensor = torch.tensor(_pw, dtype=torch.float32)

    def _bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            pred, target, pos_weight=_pw_tensor.to(pred.device),
        )

    tr_ld = _loader(X_tr, res_tr, t_tr, sl_tr, lc_tr, ph_tr, r2_tr, ic_tr, dc_tr, j_tr, r_tr, rem_tr, c_tr, batch_size, True)
    va_ld = _loader(X_va, res_va, t_va, sl_va, lc_va, ph_va, r2_va, ic_va, dc_va, j_va, r_va, rem_va, c_va, batch_size, False)

    train_hist, val_hist = [], []
    best_val, best_st, pat = float("inf"), None, 0
    patience = predictor.early_stopping_patience

    for ep in range(1, n_epochs + 1):
        predictor._net.train()
        tl = 0.0
        for xb, yb, tb, sb, lb, pb, r2b, ib, db, jb, rb, remb, cb in tr_ld:
            xb = xb.to(predictor._device)
            rb = rb.to(predictor._device)
            remb = remb.to(predictor._device)
            cb = cb.to(predictor._device)
            pb = pb.to(predictor._device)
            if predictor.augment_sigma > 0.0:
                xb = xb + torch.randn_like(xb) * predictor.augment_sigma
            yn  = ((yb  - predictor._residual_mean) / predictor._residual_std).to(predictor._device)
            tn  = ((tb  - predictor._t_obs_mean)    / predictor._t_obs_std).to(predictor._device)
            sn  = ((sb  - predictor._slope_mean)    / predictor._slope_std).to(predictor._device)
            ln  = ((lb  - predictor._logc_mean)     / predictor._logc_std).to(predictor._device)
            pn  = ((pb  - predictor._phys_mean)     / predictor._phys_std)
            r2n = ((r2b - predictor._r2_mean)       / predictor._r2_std).to(predictor._device)
            itn = ((ib  - predictor._ic_mean)       / predictor._ic_std).to(predictor._device)
            dtn = ((db  - predictor._dc_mean)       / predictor._dc_std).to(predictor._device)
            jn = (
                ((jb - predictor._j_mean) / predictor._j_std).to(predictor._device)
                if predictor.use_J_scalar else None
            )
            res_pred, risk_pred = predictor._net(xb, tn, sn, ln, pn, r2n, itn, dtn, jn)
            reg = _reg_loss(res_pred, yn, pb, remb, cb)
            risk_valid = ~((cb > 0.5) & (remb <= horizon))
            if risk_valid.any():
                bce = _bce(risk_pred[risk_valid], rb[risk_valid])
            else:
                bce = risk_pred.new_zeros(())
            loss = predictor.alpha * reg + predictor.beta * bce
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(predictor._net.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        avg_tr = tl / max(len(tr_ld), 1)

        predictor._net.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb, tb, sb, lb, pb, r2b, ib, db, jb, rb, remb, cb in va_ld:
                xb = xb.to(predictor._device)
                rb = rb.to(predictor._device)
                remb = remb.to(predictor._device)
                cb = cb.to(predictor._device)
                pb = pb.to(predictor._device)
                yn  = ((yb  - predictor._residual_mean) / predictor._residual_std).to(predictor._device)
                tn  = ((tb  - predictor._t_obs_mean)    / predictor._t_obs_std).to(predictor._device)
                sn  = ((sb  - predictor._slope_mean)    / predictor._slope_std).to(predictor._device)
                ln  = ((lb  - predictor._logc_mean)     / predictor._logc_std).to(predictor._device)
                pn  = ((pb  - predictor._phys_mean)     / predictor._phys_std)
                r2n = ((r2b - predictor._r2_mean)       / predictor._r2_std).to(predictor._device)
                itn = ((ib  - predictor._ic_mean)       / predictor._ic_std).to(predictor._device)
                dtn = ((db  - predictor._dc_mean)       / predictor._dc_std).to(predictor._device)
                jn = (
                    ((jb - predictor._j_mean) / predictor._j_std).to(predictor._device)
                    if predictor.use_J_scalar else None
                )
                res_pred, risk_pred = predictor._net(xb, tn, sn, ln, pn, r2n, itn, dtn, jn)
                reg = _reg_loss(res_pred, yn, pb, remb, cb)
                risk_valid = ~((cb > 0.5) & (remb <= horizon))
                if risk_valid.any():
                    bce = _bce(risk_pred[risk_valid], rb[risk_valid])
                else:
                    bce = risk_pred.new_zeros(())
                vl += (predictor.alpha * reg + predictor.beta * bce).item()
        avg_va = vl / max(len(va_ld), 1)
        sched.step(avg_va)
        train_hist.append(avg_tr)
        val_hist.append(avg_va)

        if avg_va < best_val - 1e-4:
            best_val = avg_va
            best_st  = copy.deepcopy(predictor._net.state_dict())
            pat = 0
        else:
            pat += 1

        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  Epoch {ep:3d}/{n_epochs}  train={avg_tr:.4f}  val={avg_va:.4f}"
                  f"  (best={best_val:.4f}  pat={pat}/{patience})")
        if pat >= patience:
            if verbose:
                print(f"  Early stop at epoch {ep}.")
            break

    if best_st is not None:
        predictor._net.load_state_dict(best_st)
    predictor._net.eval()
    return train_hist, val_hist
