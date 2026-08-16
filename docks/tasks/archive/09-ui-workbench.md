# UI workbench: 7 экранов по макету аудита

## Why

Текущий `app.py` — демо: все слайдеры открыты, только const γ, E1–E14 оттуда не запустить.
Макет (`Анализ и улучшение кода/UI Dashboard.dc.html`) обещал Dashboard / Runs / Models / Simulate / Train / Predict / Backtest.
Научный бэклог 01–08 UI не закрывал.

## Scope

**Входит:**
- 7 экранов в Gradio
- Dashboard чемпионов A–E из известных CSV / plan
- Runs: каталог E1–E14, таблица CSV, replay `--fast`/`full` фоном
- Models: `HamiltonianModelRegistry`
- Simulate: пресеты A–E (как в экспериментах)
- Train: запуск eN или старый custom в accordion
- Predict / Backtest: те же use-case, меньше слайдеров

**Не входит в этот срез:**
- YAML-пайплайн вместо `experiments/eN_*.py`
- content-addressed кэш траекторий
- multi-seed очередь / pause mid-epoch
- MLflow

## Status

`done` — 7 вкладок в `app.py`. Replay = `python -m experiments.eN`, не YAML-пайплайн.

## Priority

P1
