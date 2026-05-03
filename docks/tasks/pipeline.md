# Pipeline — порядок работы агента (Quantum Decoherence Predictor)

## Папки

- `docks/main/` — архитектура, цели, дизайн проекта.
- `docks/current/` — текущий статус выполнения по этапам.
- `docks/tasks/actual/` — активные задачи в работе.
- `docks/tasks/archive/` — завершённые задачи.

## Обязательные документы (читать перед каждой задачей)

| Документ | Когда применяется |
|----------|-------------------|
| `docks/main/ARCHITECTURE.md` | **Всегда** — перед написанием любого кода |
| `docks/main/PROJECT_PLAN.md` | При создании экспериментов, изменении целей, добавлении моделей |
| `docks/current/CURRENT_STATUS.md` | При начале любой задачи — понять, что уже сделано |

## Алгоритм работы

1. **Прочитать `docks/main/ARCHITECTURE.md`** — понять DDD-слои и запрещённые паттерны.
2. Читать `docks/main/PROJECT_PLAN.md` — понять научные цели и текущее состояние метрик.
3. Читать `docks/current/CURRENT_STATUS.md` — что уже сделано и что в работе.
4. Создать/обновить задачу в `docks/tasks/actual/` — один файл = одна задача с acceptance criteria.
5. Во время выполнения фиксировать прогресс в файле задачи.
6. **Перед коммитом — пройти чеклист** из `ARCHITECTURE.md` (раздел "Чеклист").
7. После завершения:
   - перенести задачу в `docks/tasks/archive/`;
   - обновить `docks/current/CURRENT_STATUS.md`;
   - обновить `docks/tasks/actual/EXECUTION_PLAN.md`.

## Когда задача считается выполненной

1. Техническая реализация готова.
2. Эксперимент запущен и CSV-результаты сохранены в `experiments/results/`.
3. Метрики соответствуют acceptance criteria.
4. **Чеклист `ARCHITECTURE.md`** — все пункты зелёные.
5. Файл задачи перемещён в `tasks/archive/`.
6. `docks/current/CURRENT_STATUS.md` обновлён.
7. `docks/tasks/actual/EXECUTION_PLAN.md` обновлён (статус + DONE-метка).

Если хотя бы один пункт не выполнен — задача не завершена.

## Архитектурный чеклист (краткий, полный — в ARCHITECTURE.md)

- [ ] Новый код не смешивает слои (domain не импортирует infrastructure)
- [ ] Новый предиктор реализует `IPredictor` из `src/domain/ports.py`
- [ ] Новый эксперимент сохраняет результаты в `experiments/results/eN_*.csv`
- [ ] Сериализация `state_bytes/from_bytes` обратно совместима (старые .pt грузятся)
- [ ] Существующие эксперименты E1–E8 запускаются без изменений
- [ ] Модели для разных гамильтонианов хранятся отдельно через `HamiltonianModelRegistry`
- [ ] Статьи в `paper/paper_1/` и `paper/paper_2/` обновлены если изменились метрики

## Правило контекста

`docks/tasks/pipeline.md` и `docks/main/ARCHITECTURE.md` всегда должны быть в контексте агента перед началом работы.
