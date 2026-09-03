# Task: Loop — починить регрессии и замерить метрики

## Why

После кодов гамильтониана E6/бэктест даёт R²≈−7 и AUROC=0.5.
Старые `.pt` не грузятся в UI (132 vs 135). Нужен цикл: замерить → починить → снова замерить.
Paper CSV не перезаписывать.

## Scope

**Входит:**
- `backtest.py` — коды на тесте (уже)
- `lstm_predictor.from_bytes` — n_physics из веса головы
- реестры — не падать на битом `.pt`
- `experiments/loop_measure.py` → `experiments/results/loop_measure.csv`

**Не входит:**
- полный E6/E8 500×150
- перезапись `e6_*` / `e8_*` чемпионов
- погоня за TFIM 0.70

## Acceptance Criteria

- [x] Старый чекпоинт 4 physics-скаляра грузится
- [x] `get_predictor` при size-mismatch возвращает None, не 500
- [x] `loop_measure.csv`: D `lstm_only` R²=0.75 (не −7)
- [x] D `physics_lstm` R² > 0
- [x] точечные тесты чекпоинта/бэктеста зелёные
- [x] `loop_measure_de80.csv` записан (D lstm_only 0.54; E physics_lstm −2.65 при τ=0.7)
- [x] E τ=0: physics_lstm R² −0.19 (было −2.65 при τ=0.7)

## Invariants

- IV1: `IPredictor.predict()` сигнатура не меняется
- IV2: Старые `.pt` загружаются
- IV3: Paper CSV E1–E8 не трогаем

## Status

`done` — tick 1: B physics_lstm R²=0.931, D lstm_only 0.75. Tick 2: E τ=0 physics_lstm −0.19.

## Priority

P0
