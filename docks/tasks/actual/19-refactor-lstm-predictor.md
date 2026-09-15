# Task: Рефакторинг `lstm_predictor.py` (780 строк)

## Why

Исходная просьба была «глобально прибраться в симуляторе и коде», но цикл 13–17 дал
в основном **гигиену репозитория** (`.venv` из git, qutip API, восстановление
истории), а не уборку кода. `lstm_predictor.py` — крупнейший файл проекта (780 строк
против 438 у следующего), в нём смешаны четыре разные ответственности. Он был
сознательно отложен в задаче 13 как рискованный; пора вернуться.

Код в целом здоров (0 TODO/FIXME, чистые DDD-слои, 92 теста) — это рефакторинг
ради читаемости, а не спасение от долга. Поэтому **приоритет P2 и право отменить**,
если риск окажется выше пользы.

## Scope

**Входит:** `src/infrastructure/ml/lstm_predictor.py` разбить по естественным швам:

| Ответственность | Текущие строки | Куда |
|---|---|---|
| Физика/OLS (`_ols_log_slope`, `_coherence_envelope`, `_maybe_envelope_slope`, `physics_remaining`) | 68–186 | `physics_prior.py` |
| Архитектура сети (`_ResidualNet`) | 186–265 | `residual_net.py` |
| Цикл обучения (`_fit`) | 486–692 | `training.py` |
| Батч/данные (`_batch_physics`, `_add_log`, `_ns`, `_loader`) | 714–end | `batching.py` |
| `LSTMPredictor` (порт `IPredictor`) | остаётся | `lstm_predictor.py` |

**Не входит:**
- Любое изменение поведения, гиперпараметров, численных результатов.
- `transformer_predictor.py`, `stores.py` и прочие файлы.
- Переобучение моделей, перезапуск экспериментов.

## Acceptance Criteria

- [ ] `pytest tests/` — 92 passed, 0 warnings (как сейчас)
- [ ] **Существующие чекпоинты грузятся**: `best_1qubit.pt`, `best_1q_sigma_minus.pt`,
      `best_2qubit.pt` (последний — от 18 апреля) через `from_bytes()` без ошибок
- [ ] Публичный импорт-контракт сохранён: `from src.infrastructure.ml.lstm_predictor
      import LSTMPredictor, physics_remaining` продолжает работать
      (либо фасад с ре-экспортом, либо обновлены все импортёры)
- [ ] Численная эквивалентность: `predict()` на одном и том же окне до и после
      рефактора даёт **идентичный** результат (как проверяли qutip в задаче 13)
- [ ] Ни один файл не превышает ~300 строк
- [ ] `python app.py` стартует (workbench импортирует `physics_remaining`)

## Invariants

- IV1: `IPredictor.predict()` сигнатура не меняется
- IV2: `state_bytes`/`from_bytes` обратно совместимы — **главный риск**: если
  сериализация тянет ссылки на модуль/класс (`_ResidualNet`), перенос класса
  сломает загрузку старых `.pt`. Проверить ДО переноса, чем именно сериализуется.
- IV3: Чемпионские CSV и метрики не трогаем
- IV4: `physics_remaining` импортируется в `workbench_server.py` — не сломать

## Verification

```bash
pytest tests/ -q
python -c "
from src.infrastructure.persistence.stores import BestModelRegistry
r = BestModelRegistry(root='checkpoints')
print(r.summary())
print('1q:', r.get_predictor(1) is not None)
print('2q:', r.get_predictor(2) is not None)
"
python -c "from src.infrastructure.ml.lstm_predictor import LSTMPredictor, physics_remaining; print('imports OK')"
```

## Порядок (важен из-за IV2)

1. Сначала выяснить формат сериализации (`state_bytes`) — можно ли двигать классы.
2. Зафиксировать эталон: `predict()` на фиксированном окне → сохранить числа.
3. Резать по одному шву, после каждого — тесты + сверка эталона.
4. Если IV2 нарушается неустранимо — остановиться, оставить `_ResidualNet` на месте,
   вынести только физику/батчинг (частичная победа лучше сломанных чекпоинтов).

## Status

`planned`

## Priority

P2
</content>
