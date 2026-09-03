# Архитектура проекта: Quantum Decoherence Predictor

Документ задаёт единый контракт для нового кода.
Разработчик/агент **обязан прочитать и пройти чеклист** перед коммитом.

---

## 1. Цели архитектуры

- **Тестируемость**: доменная логика не зависит от ML-фреймворков и квантовых симуляторов.
- **Сменяемость**: QuTip, PyTorch, FastAPI — за границами домена.
- **Воспроизводимость**: каждый эксперимент — отдельный entry-point, результаты — CSV.
- **Расширяемость**: новый гамильтониан/диссипатор = новый `InteractionType`/`DissipatorType` без изменений домена.

---

## 2. Слои и правило зависимостей

```
┌─────────────────────────────────────────────────┐
│  UI / Experiments (app.py, experiments/eN.py)   │  entry-points
└────────────────────┬────────────────────────────┘
                     │ использует только
┌────────────────────▼────────────────────────────┐
│  Application (src/application/)                 │  use-cases, команды
│  GenerateDatasetUseCase, TrainModelUseCase,     │
│  AblationStudyUseCase, WindowSweepUseCase ...   │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Domain (src/domain/)                           │  чистая физическая логика
│  value_objects.py · ports.py · events.py        │  БЕЗ torch/qutip/numpy
└────────────────────▲────────────────────────────┘
                     │ реализуют
┌────────────────────┴────────────────────────────┐
│  Infrastructure (src/infrastructure/)           │  адаптеры
│  ml/lstm_predictor.py · ml/transformer_predictor│
│  quantum/qutip_simulator.py · persistence/      │
└─────────────────────────────────────────────────┘
```

**Строгие запреты:**
- `src/domain/` **не импортирует** ничего из `infrastructure`, `torch`, `qutip`.
- `src/application/` **не содержит** torch-операций — только оркестрация.
- Эксперименты (`experiments/`) **не трогают** domain напрямую — через application use-cases.

---

## 3. Структура каталогов

```
src/
  domain/
    value_objects.py    # SystemConfig, TrajectoryResult, QubitCount, InteractionType...
    ports.py            # ISimulator, IPredictor, IModelStore, ITrajectoryStore
    events.py           # DecoherenceAlarmEvent

  application/
    generate_dataset.py       # GenerateDatasetUseCase + GenerateDatasetCommand
    train_model.py            # TrainModelUseCase + TrainModelCommand
    predict_decoherence.py    # PredictDecoherenceUseCase
    backtest.py               # BacktestUseCase
    run_ablation.py           # AblationStudyUseCase (E1)
    run_window_sweep.py       # WindowSweepUseCase (E2)
    run_noise_sweep.py        # NoiseSweepUseCase (E3)
    run_cross_system.py       # CrossSystemUseCase (E4)

  infrastructure/
    ml/
      lstm_predictor.py       # LSTMPredictor (основная модель)
      transformer_predictor.py
      physics_only_predictor.py
      stretched_exp_predictor.py  # curve_fit baseline (E12)
      loss_functions.py
    quantum/
      qutip_simulator.py
      noise.py
    persistence/
      stores.py               # PickleModelStore, NumpyTrajectoryStore,
                              # BestModelRegistry, HamiltonianModelRegistry

experiments/
  _config.py          # фабрика SystemConfig для сценариев A–E
  e1_ablation.py      # сравнение архитектур (claims C1, C2)
  e2_window_sweep.py  # sweep по длине окна (claim C3)
  e3_noise_sweep.py   # robustness к шуму (claim C4)
  e4_cross_system.py  # cross-system generalisation (claim C5)
  e5_inverse_baseline.py
  e6_2qubit_improved.py
  e7_paper2_extended.py
  e8_improved_2qubit.py
  e9_context_injection.py
  e10_fixed_context.py      # коды 4-class + Transformer (CSV e10a/b_cross_system)
  e10_tfim_specialized.py   # TFIM-only / большое окно (CSV e10_tfim_*)
  e11_scaling.py
  e12_baseline_comparison.py
  e13_survival_tfim.py
  e14_inject_J.py     # J как physics scalar (use_J_scalar)
  results/            # все CSV с результатами

app.py                # FastAPI workbench: 7 экранов макета аудита

paper/
  paper_1/            # статья 1-кубит (main.tex, figures/)
  paper_2/            # статья 2-кубит (paper2.tex, figures/)

checkpoints/          # сохранённые модели (.pt + _meta.json)
data/trajectories/    # кэшированные симуляции (.npz + _meta.json)
```

---

## 4. Ключевые домен-объекты

| Класс | Файл | Назначение |
|-------|------|------------|
| `SystemConfig` | `value_objects.py` | Полное описание квантовой системы |
| `InteractionType` | `value_objects.py` | XXZ / TFIM / NONE |
| `DissipatorType` | `value_objects.py` | sigma_minus / sigma_z |
| `TrajectoryResult` | `value_objects.py` | Результат симуляции (times, observables, t_decoh) |
| `IPredictor` | `ports.py` | Контракт модели: `fit()`, `predict()`, `state_bytes()`, `from_bytes()` |
| `ISimulator` | `ports.py` | Контракт симулятора: `simulate()` |
| `IModelStore` | `ports.py` | `save()` / `load()` |

---

## 5. Реестр моделей

### BestModelRegistry (существующий)
Хранит одну лучшую модель на количество кубит (1-qubit, 2-qubit).
Используется UI для выбора модели при inference.

### HamiltonianModelRegistry
Хранит лучшую модель для каждого гамильтониана:
- `best_XXZ.pt` / `best_XXZ_meta.json`
- `best_TFIM.pt` / `best_TFIM_meta.json`
- `best_1q_sigma_minus.pt` / meta
- `best_1q_sigma_z.pt` / meta

UI при inference выбирает модель по `(n_qubits, interaction_type, dissipator_type)`.
Fallback: `BestModelRegistry` по количеству кубит.

---

## 6. Соглашения по экспериментам

- Каждый эксперимент eN — отдельный файл `experiments/eN_*.py`.
- Результаты сохраняются в `experiments/results/eN_*.csv`.
- Каждый эксперимент содержит docstring: **Цель**, **Claim**, **Метрики**.
- Новые эксперименты не ломают запуск E1–E8.
- Сценарии: A=1q σ₋ const, B=1q σ₋ time-dep, C=1q σ_z time-dep, D=2q XXZ, E=2q TFIM.

---

## 7. Сериализация моделей

Каждый `IPredictor` обязан реализовать:
- `state_bytes() -> bytes` — полный снапшот весов + гиперпараметров + норм.статистик
- `from_bytes(data: bytes) -> Self` — восстановление (classmethod)
- `PickleModelStore` пишет `<name>_kind.json`; без sidecar угадывает класс по байтам
- Обратная совместимость: старые `.pt` файлы должны загружаться через `from_bytes`

---

## 8. Чеклист перед коммитом

- [ ] `domain` не импортирует `torch`, `qutip`, `numpy` (кроме type hints)
- [ ] Новый предиктор реализует `IPredictor` полностью (fit, predict, state_bytes, from_bytes)
- [ ] Новый эксперимент сохраняет результаты в `experiments/results/`
- [ ] Эксперименты E1–E8 не сломаны (новые параметры имеют дефолты)
- [ ] `state_bytes/from_bytes` обратно совместимы
- [ ] Физические скаляры нормализованы (mean/std из training set)
- [ ] Если изменились метрики — обновить таблицу в `paper/paper_1/` или `paper/paper_2/`
- [ ] `docks/current/CURRENT_STATUS.md` обновлён
