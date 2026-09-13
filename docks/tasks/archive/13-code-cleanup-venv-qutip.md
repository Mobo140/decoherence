# Task: Уборка кода — `.venv` из git + актуализация qutip API

## Why

Гигиена репозитория перед финализацией статей. Сейчас `.gitignore` игнорирует
`venv/`, но проект использует `.venv/` → 37 файлов венва трекаются в git (мусор в
`git status`, шум в диффах). Симулятор кидает `FutureWarning` от qutip 5.x
(устаревший `options`-класс, `e_ops` станет keyword-only) — тихий долг, ломается в
будущих версиях.

## Scope

**Входит:**
- `.gitignore` — добавить `.venv/`
- `git rm -r --cached .venv` — убрать венв из индекса (файлы на диске не трогаем)
- `src/infrastructure/quantum/qutip_simulator.py`, `src/physics/simulate_qutip.py` —
  `options` передавать dict'ом, `e_ops` как keyword-аргумент

**Не входит:**
- Рефактор `lstm_predictor.py` (780 LOC) — отдельная будущая задача
- Любые изменения физики/метрик, перезапуск экспериментов
- Domain-слой

## Acceptance Criteria

- [x] `.gitignore` содержит `.venv/`; `git ls-files | grep '^\.venv/'` пусто
- [x] `pytest tests/` зелёный (92) **без** qutip `FutureWarning` в выводе
- [x] Численные результаты симулятора не изменились (тесты `test_lindblad_systems`,
  `test_decoherence_time` проходят как раньше)
- [x] Существующие E1–E8 запускаются без изменений (код не трогали, только API вызова)

## Invariants

- IV1: `ISimulator` сигнатуры не меняются
- IV2: Результаты Lindblad-решателя идентичны (только API-вызов, не физика)
- IV3: Чемпионские CSV в `experiments/results/` не трогаем

## Verification

```bash
git ls-files | grep -c '^\.venv/'          # -> 0
source .venv/bin/activate
python -W error::FutureWarning -c "from src.physics.simulate_qutip import *; print('OK')"
pytest tests/ -q
```

## Status

`done`

Обнаружилось, что репозиторий не коммитился ~9 месяцев (единственный коммит `MVP`
от 2026-01-03 фиксировал ранний прототип, не текущую DDD-структуру) — до правки
`.venv` пришлось сначала landing'ить снимок текущего состояния отдельным коммитом
(с согласия пользователя), убрав по пути чужие staged-файлы из индекса (design-canvas
артефакты не из этого проекта, на диске их не было).

Коммиты:
- `39e4c64` — снимок 9 месяцев работы (DDD, статьи, docks/, experiments) + untrack `.venv`
- `7a88624` — сам фикс: `qt.Options(nsteps=...)` → dict; `e_ops` positional → keyword.
  Численно идентично (max abs diff = 0.0 на тестовом mesolve), 92 passed без warnings.

## Priority

P1
