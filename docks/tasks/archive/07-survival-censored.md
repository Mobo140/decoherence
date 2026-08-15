# Task: Survival / RUL loss for right-censored T₂

## Why

После задачи 06 censored траектории (порог 1/e не пересечён за t_max) выбрасываются.
На TFIM это съедает длинные T₂, тест пустеет (E10a/d: n=8–28), specialized-модель
неизмерима. Правильная постановка — Remaining Useful Life с right-censoring:
T₂ > t_max это нижняя граница, не мусор.

## Scope

**Входит:**
- `generate_dataset.py` — флаг `include_censored`, поле `Dataset.censored`
- `loss_functions.py` — SurvHuber (штраф censored только если pred < границы)
- `lstm_predictor.py` — лосс в пространстве remaining time + маска risk
- `backtest.py` — R²/MAE только по uncensored
- `experiments/e13_survival_tfim.py` — drop vs keep+surv на TFIM
- тесты

**Не входит:**
- квантильные головы / доверительные интервалы
- ломать E1–E12 (дефолт `include_censored=False`)

## Acceptance Criteria

- [x] `include_censored=False` — drop как в задаче 06
- [x] `include_censored=True` — окна остаются, `Dataset.censored` заполнен
- [x] SurvHuber: uncensored = Huber; pred ≥ bound → 0; pred < bound → > 0
- [x] Backtest R² только по uncensored
- [x] `experiments/results/e13_survival_tfim.csv`
- [x] Тесты зелёные

## Conclusion

E13 (TFIM, n=200, window=30): при текущем t_max=20 почти нет цензуры (5 окон).
drop R²=0.16 (первый прогон 0.58 — шум на n_test=30); surv R²=0.12.
Инфраструктура готова, выигрыша нет: нечего цензурировать. Чтобы survival
проявился, нужны более длинные T₂ (больше t_max / меньше γ).

## Invariants

- IV1: `IPredictor.predict()` не меняется
- IV2: дефолт генерации датасета — drop censored
- IV3: domain без torch/scipy

## Status

`done`

## Priority

P0
