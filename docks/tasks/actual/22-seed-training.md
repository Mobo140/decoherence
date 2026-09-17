# Task: Сидировать обучение (torch не сеется вообще)

## Why

`torch.manual_seed` **не вызывается нигде** в проекте. `AblationCommand.seed=42`
уходит только в генерацию конфигураций, нарезку окон и сплит
(`generate_dataset`, `simulate_trajectory`). Инициализация весов, dropout и
перемешивание батчей неуправляемы.

Доказано напрямую: две подряд созданные `_ResidualNet` различаются в весах до
**0.248**; с `torch.manual_seed(0)` перед каждой — идентичны (0.000).

**Следствия:**

1. «Воспроизводимо с seed=42» — неверно для ML-части. Повтор той же команды
   даёт другие числа. Это утверждение есть в доках и в `PAPERS_EXPLAINED_RU`.
2. E7b генерирует датасет один раз на сценарий и обучает на нём все пять τ без
   сидов → размах 0.46 между соседними τ это лотерея инициализации, а не
   эффект порога. Вывод «оптимальное τ = 0.5» на этом не держится.
3. Чемпионские числа — одна реализация из неуправляемого распределения.

Подтверждено измерением (E22, 3 сида, уменьшенный размер): std по R² у
`physics_lstm` = 0.311 (D) и 0.764 (E), тогда как заявленный в статье разрыв с
`lstm_only` — 0.037.

## Scope

**Входит:**
- `TrainModelCommand` — добавить поле `seed`
- `TrainModelUseCase.execute` — пробрасывать `seed` в `predictor.train(...)`
- `LSTMPredictor.train` / `TransformerPredictor.train` — читать `seed` из
  kwargs и звать `torch.manual_seed(seed)` перед обучением
- Use-cases (`run_ablation`, `run_cross_system`, `run_noise_sweep`,
  `run_window_sweep`) — передавать `seed=command.seed`

**Не входит:**
- Перепрогон чемпионских экспериментов и перезапись их CSV
- Изменение архитектур, гиперпараметров, метрик
- Правки статей (отдельно, после того как станет ясна картина)

## Invariants

- IV1: torch остаётся в infrastructure — application-слой его не импортирует
- IV2: `IPredictor.train(dataset, **kwargs)` сигнатура не меняется
- IV3: предикторы без обучения (`physics_only`, `stretched_exp`) не ломаются
- IV4: чемпионские CSV не трогаем

## Acceptance Criteria

- [ ] Два прогона одной конфигурации с одним сидом дают **идентичные** метрики
- [ ] Разные сиды по-прежнему дают разные метрики (сид реально управляет)
- [ ] `pytest tests/` зелёный (92)
- [ ] Application-слой не импортирует torch

## Verification

```bash
pytest tests/ -q
grep -rn "import torch" src/application/    # -> пусто
```

## Status

`planned`

## Priority

P1
