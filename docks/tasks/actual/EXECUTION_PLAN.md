# Execution Plan — Quantum Decoherence Predictor

Запуск: `make help` · `make app` · `make test` · `make pdf`.

## Активная очередь — 2026-09-15

Разбор итогов цикла 13–17 показал два недотянутых пункта: Paper 2 не дошла до
уровня Paper 1, и «глобально прибраться в коде» свелось к гигиене репозитория,
а не к уборке кода.

| # | Задача | Приоритет | Статус |
|---|--------|-----------|--------|
| 18 | Paper 2 — паритет с Paper 1 (библио 6→14, факт-чек, claims) | P1 | **done** → archive |
| 19 | Рефакторинг `lstm_predictor.py` (780 строк) | P2 | planned |

Порядок: 18 → 19 (Paper 2 ближе к публикации и дешевле, чем рефакторинг).

**Открытые пункты, не оформленные в задачи:**
- Реальная аффилиация автора вместо placeholder «Independent Researcher» —
  в обеих статьях, нужна от пользователя.
- Вычитка обеих статей человеком перед реальной подачей.
- `description_ru.tex` — числа D/E устарели (0.70/0.48 против 0.817/0.575);
  это черновик, не публикация — обновлять по необходимости.
- Пренебрежимо малый overfull hbox (5.4pt, Table 2) в Paper 1.

**Итог цикла 2026-09-12/13** (задачи 13–17): git-история восстановлена из
9-месячного разрыва (33 коммита, реальная хронология по mtime); qutip API
актуализирован; Paper 1 полностью вылизана (рисунки, 12 верифицированных
ссылок, авторы, факт-чек, явные claims C1–C5); Paper 2 получила теорию
TFIM-потолка (точная диагонализация N=2, Pfeuty 1970) + тот же класс правок,
что в Paper 1.

## Закрытый бэклог

| # | Задача | Итог |
|---|--------|------|
| 18 | Paper 2 — паритет с Paper 1 | done — библио 6→14 (веб-верифицированы), факт-чек нашёл 4 ошибки, claims размечены (2 из 4 — не подтверждены), commit `72dfb52` |
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
