# E9: Context Injection — Добавить гамильтонов и диссипаторный коды в модель

## Task

Добавить в `LSTMPredictor` три новых физических скаляра:
`ols_r2`, `interaction_code` (XXZ=0, TFIM=1), `dissipator_code` (σ₋=0, σ_z=1).
Обучить E9 эксперимент и измерить влияние на TFIM R² и cross-system AUROC.

## Why

Модель сейчас слепа к типу гамильтониана (TFIM vs XXZ) и диссипатора.
TFIM осциллирует — физический prior (OLS slope) даёт плохой R²≤0.5 без знания типа системы.
Context codes дают модели явный сигнал о структуре системы → ожидаем TFIM R² +0.05–0.10.

## Scope

**Входит:**
- `src/application/generate_dataset.py` — добавить `interaction_type_codes` и `dissipator_type_codes` в `Dataset`
- `src/infrastructure/ml/lstm_predictor.py` — добавить 3 новых physics scalar (ols_r2, int_code, dis_code)
- `experiments/e9_context_injection.py` — управляемый эксперимент vs E8c
- `docks/current/CURRENT_STATUS.md` — обновить после результатов

**Не входит:**
- Изменение domain layer
- Изменение E1–E8 (только новые дефолты)
- Изменение UI

## Acceptance Criteria

- [ ] `Dataset` содержит `interaction_type_codes` и `dissipator_type_codes` (N,) float32
- [ ] `LSTMPredictor._ResidualNet` принимает 7 physics scalars (было 4)
- [ ] `_batch_physics()` возвращает `ols_r2` как 4-й элемент
- [ ] `state_bytes/from_bytes` сохраняет/загружает новые norm stats
- [ ] E1–E8 эксперименты запускаются без ошибок (новые kwargs=defaults)
- [ ] `experiments/results/e9_context_injection.csv` создан
- [x] TFIM R² в E9 ≥ 0.60 (vs 0.575 в E8c) — **не достигнуто**: валидный E9a E R²=0.452. Коды D/E были 0/1, исправлены на 2/3.

## Invariants

- IV1: `IPredictor.predict()` сигнатура не меняется
- IV2: Старые `.pt` файлы загружаются через `from_bytes()` с fallback defaults
- IV3: 1-qubit модели работают корректно (interaction_code=0.0 по умолчанию)

## Verification

```bash
python experiments/e9_context_injection.py
cat experiments/results/e9_context_injection.csv
# ожидаем TFIM R² ≥ 0.60
```

## Status

`done` — код готов, цель TFIM R²≥0.60 не взята (валидный E9a E=0.452)

## References

- Design: `docs/tasks/fix-2qubit-context.md` (полный план реализации)
- Baseline: E8c cross-system → D R²=0.740, E R²=0.575
