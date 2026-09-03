# Task: Hilbert-огибающая в physics prior

## Why

OLS по log(C) — формула одной экспоненты. На TFIM/XXZ L1 осциллирует, prior врёт.
Огибающая Hilbert → OLS по log|A(t)| даёт тот же −1/slope, но по медленной моде.

## Scope

**Входит:**
- `physics_remaining` / `_batch_physics` — auto: огибающая, если raw R² < 0.7 и у envelope R² выше
- `PhysicsOnlyPredictor` зовёт тот же `physics_remaining`
- тесты на чистую экспоненту и затухающую осцилляцию
- смок physics_only D/E → `loop_measure_envelope.csv`

**Не входит:**
- полный E8, новые архитектуры, paper CSV

## Acceptance Criteria

- [x] 1q-экспонента: remaining близко к −1/slope
- [x] осцилляция |cos|: envelope ближе к T₂, чем raw OLS
- [x] смок: D −10.75 → −9.15; E −1.36 без изменения. Paper CSV не тронуты

## Invariants

- IV1: `IPredictor.predict()` без изменений
- IV3: paper CSV не перезаписываем

## Status

`done`

## Priority

P1
