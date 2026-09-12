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

- [ ] `.gitignore` содержит `.venv/`; `git ls-files | grep '^\.venv/'` пусто
- [ ] `pytest tests/` зелёный (92) **без** qutip `FutureWarning` в выводе
- [ ] Численные результаты симулятора не изменились (тесты `test_lindblad_systems`,
  `test_decoherence_time` проходят как раньше)
- [ ] Существующие E1–E8 запускаются без изменений

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

`planned`

## Priority

P1
