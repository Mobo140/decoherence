# Execution Plan — Quantum Decoherence Predictor

Запуск: `make help` · `make app` · `make test` · `make pdf`.

Очередь `docks/tasks/actual/` пуста — бэклог 01–17 в архиве.

**Итог цикла 2026-09-12/13** (задачи 13–17): git-история восстановлена из
9-месячного разрыва; qutip API актуализирован; Paper 1 полностью вылизана
(рисунки, 12 верифицированных ссылок, авторы, факт-чек, явные claims C1–C5);
Paper 2 получила теорию TFIM-потолка (точная диагонализация N=2, Pfeuty 1970)
+ тот же класс правок, что в Paper 1. Реальная аффилиация автора — open item
в обеих статьях (сейчас placeholder «Independent Researcher»).

## Закрытый бэклог

| # | Задача | Итог |
|---|--------|------|
| 17 | Финализация — статус + коммиты | done — `CURRENT_STATUS.md` без over-claim, archive обновлён |
| 16 | Paper 2 — теория потолка TFIM у QPT | done — точная диагонализация N=2 (Pfeuty1970), авторы+библио+overfull fix, commit `b0acc1b` |
| 15 | Paper 1 — библиография, авторы, факт-чек | done — 12 ссылок (веб-верифицированы), авторы, `>15×`→`3–9×` факт-чек, явные C1–C5, commit `b789695` |
| 14 | Paper 1 — publication-рисунки | done — 6 рисунков (E1–E5 + physics_validation), overfull hbox в Table 4 попутно починен, commit `6b46596` |
| 13 | Уборка: `.venv` из git + qutip API | done — репо не коммитился 9 мес (снимок commit `39e4c64`), qutip fix commit `7a88624`, 92 passed 0 warnings |
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
