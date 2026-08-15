# Per-Hamiltonian Model Registry

## Task

Реализовать `HamiltonianModelRegistry` в `src/infrastructure/persistence/stores.py`.
Обновить UI (`app.py`) чтобы при inference автоматически выбиралась лучшая модель
под конкретный гамильтониан (XXZ / TFIM / 1q-sigma_minus / 1q-sigma_z).

## Why

Сейчас `BestModelRegistry` хранит только одну модель на количество кубит.
TFIM и XXZ имеют принципиально разные физики (осцилляции vs монотонное затухание),
поэтому специализированная модель даёт значимо лучшие метрики чем одна общая.
При inference через Gradio UI пользователь задаёт тип взаимодействия →
система должна автоматически выбрать лучшую модель для этого гамильтониана.

## Scope

**Входит:**
- `src/infrastructure/persistence/stores.py`:
  - Новый класс `HamiltonianModelRegistry`
  - Ключ: `(n_qubits, interaction_type, dissipator_type)` → строка-имя файла
  - Методы: `maybe_update()`, `get_predictor()`, `get_meta()`, `summary()`
  - Файлы: `best_XXZ.pt` / `best_TFIM.pt` / `best_1q_sigma_minus.pt` / `best_1q_sigma_z.pt`
- `app.py`:
  - В `run_training()`: после backtest вызывать `HamiltonianModelRegistry.maybe_update()`
  - В `run_prediction()`: выбирать из `HamiltonianModelRegistry` по (n_qubits, interaction_type, dissipator_type)
  - Fallback: `BestModelRegistry` по n_qubits
  - В `run_backtest()`: аналогично
  - Показывать в UI какая модель используется (Hamiltonian + метрики)

**Не входит:**
- Изменение domain layer
- Изменение алгоритмов обучения
- Изменение экспериментов

## Acceptance Criteria

- [x] `HamiltonianModelRegistry` реализован с методами `maybe_update`, `get_predictor`, `get_meta`, `summary`
- [x] Имена: `best_XXZ.pt` / `best_TFIM.pt` / `best_1q_sigma_minus.pt` / `best_1q_sigma_z.pt`
- [x] UI Predict/Backtest/Train выбирает Hamiltonian-specific, fallback на BestModelRegistry
- [x] UI показывает label + R² модели
- [x] `BestModelRegistry` без изменений API
- [x] `maybe_update` не пишет чемпиона при n_samples=0 или NaN R²

## Verification

```bash
# 1. Обучить XXZ модель через UI → убедиться что best_XXZ.pt создан
# 2. Обучить TFIM модель → убедиться что best_TFIM.pt создан
# 3. В Predict выбрать TFIM → убедиться что используется best_TFIM.pt
# 4. В Predict выбрать XXZ → убедиться что используется best_XXZ.pt
ls checkpoints/best_*.pt
cat checkpoints/best_XXZ_meta.json
```

## Registry Key Design

```python
def _name(n_qubits: int, interaction_type: InteractionType,
          dissipator_type: DissipatorType) -> str:
    if n_qubits == 1:
        return f"best_1q_{dissipator_type.value}"
    return f"best_{interaction_type.value}"

# Examples:
# best_1q_sigma_minus.pt
# best_1q_sigma_z.pt
# best_XXZ.pt
# best_TFIM.pt
```

## Status

`done` — UI уже был подключён; имена файлов выровнены (`best_XXZ.pt` / `best_TFIM.pt`), `maybe_update` отвергает NaN/пустой тест, тесты в `tests/test_hamiltonian_registry.py`.

## Priority

P0 — нужно до запуска экспериментов E9/E10 чтобы модели сохранялись правильно
