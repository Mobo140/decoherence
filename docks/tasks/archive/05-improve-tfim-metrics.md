# E10: Улучшить TFIM R² (цель ≥ 0.70 cross-system)

## Task

После E9 (context injection) оценить TFIM R² и если ≤ 0.70 — реализовать E10:
специализированную модель для TFIM с улучшенной архитектурой.

## Why

TFIM физика принципиально отличается от XXZ: когерентность осциллирует из-за квантовых
переходов в поперечном поле. OLS physics prior даёт отрицательный вклад при TFIM.
Текущий лучший результат: R²=0.575 (E8c). Цель Paper 2: ≥ 0.70 cross-system.

## Гипотезы для улучшения (в порядке приоритета)

| # | Идея | Ожидаемый эффект |
|---|------|-----------------|
| H1 | Context codes (E9) | +0.05–0.10 R² | 
| H2 | Longer window (40–50 steps) для TFIM | +0.05 (осцилляции лучше видны) |
| H3 | Отдельная модель только TFIM (не cross-system) | +0.10–0.15 (нет компромисса с XXZ) |
| H4 | Увеличить hidden_size до 256 для 2-qubit | +0.03–0.07 |
| H5 | Убрать adaptive_prior для TFIM (r2_threshold=0.0) | убрать bias от плохого prior |

## Scope

**Входит:**
- `experiments/e10_tfim_specialized.py` — три sub-experiments:
  - E10a: TFIM-only model, window=40, hidden=256
  - E10b: TFIM context codes (из E9) + longer window
  - E10c: cross-system D+E с E10b конфигурацией
- Обновить `paper/paper_2/paper2.tex` если E10 превышает E8/E9

**Не входит:**
- Изменение XXZ результатов
- Изменение Paper 1

## Acceptance Criteria

- [x] `experiments/e10_tfim_specialized.py` создан
- [x] CSV: `e10_tfim_a.csv` / `b` / `c` (d running)
- [ ] TFIM R² cross-system ≥ 0.70 — **не достигнуто** (лучшее E10c E=0.472 < E8c 0.575)
- [x] `paper/paper_2` не трогал: E10 хуже E8c

## Prerequisites

- [ ] E9 завершён и TFIM R² измерен
- [ ] `HamiltonianModelRegistry` реализован (задача 02)

## Status

`done` — цель 0.70 не взята.

- E10a window=40: n=21, R²=−0.97
- E10b window=50: n=8, R²=−1.48
- E10c cross-system: D=0.737, E=0.472
- E10d window=30 TFIM-only: n=28, R²=−0.89

Специализация без survival-пайплайна не работает: тест пустеет.

## Priority

P0 — ключевая метрика для Paper 2
