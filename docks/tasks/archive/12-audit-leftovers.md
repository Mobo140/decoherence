# Task: Audit leftovers — T4 kind-load, T5 variant, T2 cache, C3 AUROC

## Why

В плане ещё висели T4/T5/T2/C3. Код почти закрыт, но `_force_all_test`
оказался внутри `_variant_of` после `return` — E4/E7a/E9 падают с AttributeError.
Headline Transformer без `_kind.json` тоже не покрыт тестом.

## Scope

**Входит:**
- `run_cross_system.py` — `_force_all_test` снова метод use-case
- `PickleModelStore` Transformer roundtrip (с sidecar и без)
- docstring E2 / строка C3: success = AUROC, не R²≥0.95

**Не входит:**
- YAML-runner, multi-seed, lineage чемпионов, pydantic CSV schema
- полный E4/E7 rerun

## Acceptance Criteria

- [x] `CrossSystemUseCase._force_all_test` кладёт все окна в `test_idx`
- [x] Transformer `.pt` грузится как `TransformerPredictor` (kind sidecar и infer)
- [x] E7a CSV: `variant=transformer`, не `physics_lstm`
- [x] E2 docstring и PROJECT_PLAN C3 говорят AUROC≥0.92, не R²≥0.95
- [x] `pytest tests/` зелёный (88)

## Invariants

- IV1: `IPredictor.predict()` без изменений
- IV3: paper CSV E1–E8 не перезаписываем (E7a уже поправлен)

## Status

`done` — `_force_all_test` вернули на класс. T4/T2 уже были в коде, добавлены тесты Transformer. C3 переписан в E2.

## Priority

P0
