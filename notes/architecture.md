# Целевая архитектура

Пайплайн повторяет трёхслойную картину из письма научного руководителя: **физика → модель наблюдения → прогноз**. Каждый шаг — функция или реестр, где вариант выбирается строковым псевдонимом. Скрипты экспериментов и ноутбук `notebooks/pipeline.ipynb` собирают пайплайн только из этих функций.

```
сценарий → симуляция → наблюдение → признаки → окна → модель → потери → обучение → метрики
```

## Пакеты

| Пакет | Содержимое | Ключевые функции и реестры |
|---|---|---|
| `src/domain/` | Типы-значения (`SystemConfig`, `TrajectoryResult`, критерии) и порты | — |
| `src/physics/` | Системы Линдблада, решение уравнения, меры, операторы Паули | `measures.coherence_l1`, `measures.concurrence`, … |
| `src/simulation/` | Адаптер QuTiP и его варианты | `simulator.QuTipSimulator`, `variants.*` |
| `src/data/scenarios.py` | Сценарии A–E и генераторы конфигураций (`experiments/_config.py` оставлен как псевдоним для `paper/`) | `build_configs(n_per_scenario, seed)`, `fixed_coupling_configs`, `SCENARIOS` |
| `src/data/dataset.py` | `Dataset` и операции над ним | `Dataset.with_sequences`, `Dataset.select_features`, `Dataset.drop_features`, `Dataset.as_test_only` |
| `src/data/windows.py` | Нарезка окон и разбиение по траекториям | `build_dataset(trajectories, ...)`, `assign_splits`, `sample_window_ends` |
| `src/data/observation.py` | **Модель наблюдения**: как из точных средних получаются входы | реестр `OBSERVATIONS`: `"exact"`, `"gaussian"`, `"ar1"`; позже `"shots"`, `"sequential"` |
| `src/data/features.py` | Наборы признаков | реестр `FEATURE_SETS`: `"base"`, `"corr"`, `"corr_conc"`; позже `"local"` |
| `src/models/` | Предикторы и сети; реестр по псевдониму | `build_model("transformer", **kw)`; `MODELS`: `"physics_only"`, `"stretched_exp"`, `"lstm_only"`, `"physics_lstm"`, `"physics_feature"`, `"transformer"` |
| `src/training/` | Общий цикл обучения, ранняя остановка, нормировка, функции потерь | `LOSSES`: `"huber"`, `"mse"`, `"mae"`, `"quantile"`, `"surv_huber"`; `train(model, dataset, loss=..., ...)` |
| `src/evaluation/` | Единые метрики и бэктест, парная статистика | `METRICS`: `"r2"`, `"mae"`, `"rmse"`, `"mape"`, `"auroc"`, `"brier"`, `"balanced_accuracy"`; `evaluate(model, dataset, metrics)`; `paired_difference`, `bootstrap_ci` |
| `src/reproducibility.py` | Одна точка фиксации случайности | `set_seed(seed)`, число потоков, сведения о версиях для записи рядом с результатами |
| `src/pipeline.py` | Описание эксперимента и его запуск | `ExperimentConfig` (сценарии, число траекторий, окно, критерий, наблюдение, признаки, модель, потери, эпохи, сиды); `run_experiment(config) -> pandas.DataFrame` |
| `experiments/` | Тонкие скрипты: набор конфигураций → `run_experiment` → CSV | общий CLI (`--seeds`, `--fast`, `--tag`) и запись результатов |
| `notebooks/pipeline.ipynb` | Пошаговое прохождение пайплайна с вызовом этих функций | — |

## Правила

- **Варианты по псевдониму.** Новый вариант — новая запись в реестре, а не ветка `if` в чужом коде.
- **Метрики только из `src/evaluation`.** Никаких локальных реализаций R², AUROC и т. п. в экспериментах.
- **Случайность только через `set_seed` и явные генераторы.** Поведение должно совпадать с прежним, эталон `scripts/golden_outputs.py` — бит в бит.
- **Чистка не меняет числа.** Новая функциональность (`"shots"`, `"sequential"`, `"local"`) добавляется после чистки отдельным изменением.

## Конвенция знака в двухкубитной модели

Для одного кубита оператор затухания — `sigmam()`: релаксация в |1⟩, основное состояние H = ωσz. Для двух кубитов — `destroy(2)`: релаксация в |0⟩, а при поле h·σz это верхнее по энергии состояние.

На результаты это не влияет. Глобальный переворот X⊗X переводит модель в модель с полем −h и затуханием в основное состояние. При этом не меняются:
- ℓ₁-когерентность в вычислительном базисе;
- чистота и конкурентность;
- мера Хаара на начальных состояниях;
- угол смешивания, который зависит только от |h|.

Конвенцию нужно указать в статье 2 и в документе. Код оставлен как есть, потому что от него зависят опубликованные числа.
