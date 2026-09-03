# Execution Plan — Quantum Decoherence Predictor

Очередь пуста. Бэклог 01–12 в архиве.

Запуск: `make help` · `make app` · `make test` · `make pdf`.

## Закрытый бэклог

| # | Задача | Итог |
|---|--------|------|
| 01 | E9 context injection | done, TFIM R²=0.452 |
| 02 | Per-Hamiltonian registry | `best_XXZ.pt` / `best_TFIM.pt` |
| 03 | Статьи `paper_1/` `paper_2/` | done |
| 04 | Docstring e1–e14 | done |
| 05 | E10 TFIM specialized | E=0.472, цель 0.70 нет |
| 06 | Audit: коды, censored T₂, E12 | done |
| 07 | Survival/RUL | done |
| 08 | E14 inject J | E-only +0.11; E8c не побит |
| 09 | UI workbench | FastAPI, 7 экранов макета |
| 10 | Loop fix metrics | D lstm_only 0.54; E τ=0 −0.19 |
| 11 | Hilbert envelope | D −10.75 → −9.15; E без сдвига |
| 12 | T4/T5/T2/C3 leftovers | `_force_all_test` вернули; Transformer kind-load; C3 = AUROC |

## Не бэклог — сознательно не делаем

- YAML вместо `experiments/eN_*.py`
- Multi-seed ×5 / eval-v1 (часы CPU)
- Аудит E12–E16 (entanglement, 1/f, J-sweep, ECE)
- Ещё одна архитектура ради TFIM R² 0.70
- Lineage чемпионов / pydantic CSV schema
