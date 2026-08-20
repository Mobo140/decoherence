# Current Status — Quantum Decoherence Predictor

## Проект готов (2026-08-20)

Очередь `docks/tasks/actual/` пуста — бэклог 01–12 в архиве. Дубликаты 01–08 из actual убраны.
`pytest tests/` зелёный (88).
Paper 1 сдан. Paper 2: лучший валидный TFIM cross-system **E8c R²=0.575** (цель 0.70 не взята — потолок смеси J у QPT).

### Как запустить
```bash
source .venv/bin/activate
python app.py          # http://localhost:7860  — макет аудита, 7 экранов
pytest tests/
```

Replay эксперимента: вкладка **02 Runs** → E8 → fast.

UI = согласованный макет аудита (сайдбар 01–07), не Gradio. Predict работает без чекпоинта (physics_only).

### Чемпионы
- A physics_only E1 ≈ 0.9998
- B physics_lstm E1 0.945
- C physics_lstm E1 0.981
- D transformer E11a 0.817 / E8c 0.740
- E physics_lstm E8c **0.575** (Transformer E7a 0.599)

### Что сознательно не делаем
- YAML-пайплайн вместо `experiments/eN_*.py`
- Multi-seed ×5 / eval-v1 перепрогон (часы CPU)
- Новые E12–E16 из аудита (entanglement-фичи, 1/f, плотный J-sweep, ECE)
- Lineage чемпионов / pydantic-схема CSV

