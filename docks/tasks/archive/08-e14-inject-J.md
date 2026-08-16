# E14: впрыснуть J как непрерывный физический скаляр

## Why

E7c показал, что TFIM R² определяется J/h: 0.88 при J=0.2 и 0.32–0.50 у критичности.
Сценарий E сэмплит `J ~ U(0.2, 1.0)` при h=1, модель J не видит.
Context codes (E9) — дискрет XXZ/TFIM, не ось внутри TFIM.
Цель: поднять TFIM cross-system R² с 0.575 к 0.70, не меняя γ(t) и не двигая t_max.

## Scope

**Входит:**
- `Dataset.J_values` (1q → 0.0; 2q → `SystemConfig.J`)
- `LSTMPredictor.use_J_scalar` (дефолт False — E1–E13 без изменений)
- Backtest выставляет `inference_J` на каждое окно; сигнатура `predict()` не меняется
- `experiments/e14_inject_J.py` + CSV
- тесты

**Не входит:**
- менять Paper 2 таблицы до ΔR² > 0.01 на полном прогоне
- UI / Transformer
- survival / другой t_max

## Acceptance Criteria

- [x] `Dataset.J_values` длины N; 1q = 0; 2q ∈ [0.2, 1.0]
- [x] `use_J_scalar=False` — `_ResidualNet` как раньше (7 скаляров), старые `.pt` грузятся
- [x] `use_J_scalar=True` — J нормализуется по train mean/std
- [x] `experiments/results/e14_inject_J.csv`
- [x] E14a: E-only, J vs no-J, те же гиперпараметры что E8 (window=30, hidden=128, τ=0)
- [x] E14b: D+E cross-system, J-aware vs E8c (D=0.740, E=0.575)
- [x] R² по бинам J: `<0.5`, `[0.5, 0.8)`, `≥0.8`

## Invariants

- IV1: `IPredictor.predict(window, t_obs, horizon)` не меняется
- IV2: дефолт `use_J_scalar=False`
- IV3: domain без torch/qutip

## Conclusion

E14a (TFIM-only, n≈600): no-J R²=0.336 → J R²=0.443 (**+0.107**).
Главный выигрыш в бине J<0.5 (−0.09 → 0.22). У критичности (J≥0.8) J не помогает (0.54 → 0.45).

E14b (D+E): D=0.652, E=0.485 — оба хуже E8c (0.740 / 0.575).
Цель 0.70 не взята. `paper/paper_2` не трогал.

J — полезный скаляр внутри фиксированного TFIM, не замена E8c на смеси.

## Status

`done`

## Priority

P0
