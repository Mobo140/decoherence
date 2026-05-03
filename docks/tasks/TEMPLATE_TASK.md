# Task: <Название>

## Why

Зачем делаем — какую проблему решаем, какой claim это поддерживает, как улучшает метрики.

## Scope

**Входит:**
- Файлы/модули которые меняем

**Не входит:**
- Что явно не трогаем (особенно domain layer если не нужен)

## Acceptance Criteria

- [ ] Критерий 1 (конкретный, проверяемый)
- [ ] Критерий 2
- [ ] Эксперимент eN запущен, CSV создан в `experiments/results/`
- [ ] Метрика X ≥ Y (конкретное число)
- [ ] Существующие E1–E8 запускаются без ошибок

## Invariants

- IV1: `IPredictor.predict()` сигнатура не меняется
- IV2: Старые `.pt` файлы загружаются через `from_bytes()` с fallback

## Verification

```bash
# Запуск эксперимента
python experiments/eN_name.py

# Проверка результатов
cat experiments/results/eN_*.csv

# Проверка обратной совместимости
python -c "from src.infrastructure.ml.lstm_predictor import LSTMPredictor; print('OK')"
```

## Status

`planned` / `in_progress` / `done`

## Priority

P0 (срочно) / P1 (следующее) / P2 (когда-нибудь)
