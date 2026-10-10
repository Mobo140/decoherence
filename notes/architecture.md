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
| `src/simulation/` | Адаптер QuTiP и его варианты; реестр по псевдониму | `simulator.QuTipSimulator`, `variants.*`; `registry.build_simulator(alias, **params)`; `SIMULATORS`: `"plain"`, `"correlators"`, `"entanglement"`, `"stochastic_gamma"` |
| `src/persistence/` | Файловые хранилища: кэш траекторий и чекпойнты | `NumpyTrajectoryStore`, `PickleModelStore`, `BestModelRegistry`, `HamiltonianModelRegistry` |
| `src/data/scenarios.py` | Сценарии A–E и генераторы конфигураций (`experiments/_config.py` оставлен как псевдоним для `paper/`) | `build_configs(n_per_scenario, seed)`, `fixed_coupling_configs`, `SCENARIOS` |
| `src/data/dataset.py` | `Dataset` и операции над ним | `Dataset.with_sequences`, `Dataset.select_features`, `Dataset.drop_features`, `Dataset.as_test_only` |
| `src/data/windows.py` | Нарезка окон и разбиение по траекториям | `build_dataset(trajectories, ...)`, `assign_splits`, `sample_window_ends` |
| `src/data/observation.py` | **Модель наблюдения**: как из точных средних получаются входы | реестр `OBSERVATIONS`: `"exact"`, `"gaussian"`, `"ar1"`; позже `"shots"`, `"sequential"` |
| `src/data/features.py` | Наборы признаков | реестр `FEATURE_SETS`: `"base"`, `"corr"`, `"corr_conc"`; позже `"local"` |
| `src/models/` | Предикторы и сети; реестр по псевдониму | `build_model(alias, **overrides)`; `MODELS`: `"physics_only"`, `"stretched_exp"`, `"lstm_only"`, `"physics_lstm"`, `"physics_feature"`, `"transformer"`, `"transformer_large"` |
| `src/training/` | Общий цикл обучения (`loop.fit`: AdamW, ранняя остановка, откат к лучшим весам), нормировка (`standardize.Standardizer`), функции потерь | `LOSSES`: `"huber"`, `"mse"`, `"mae"`, `"quantile_median"`, `"surv_huber"` (+ `"quantile_<q>"` в `get_loss`); `trainer.train(model, dataset, epochs=..., batch_size=..., lr=..., seed=..., loss=...)` |
| `src/evaluation/` | Единые метрики, бэктест, парная статистика | `metrics.METRICS`: `"r2"`, `"mae"`, `"rmse"`, `"mape"`, `"auroc"`, `"brier"`, `"balanced_accuracy"` (первые четыре — регрессия, остальные — метки и оценки); `metrics.r2_unguarded`, `metrics.auroc_pairwise` — формы R13/R17; `backtest.evaluate(model, dataset, horizon=..., metrics=...)`, `backtest.predict_windows`; `stats.paired_difference`, `stats.mean_std`, `stats.pearson`, `stats.group_resample_indices`, `stats.bootstrap_values`, `stats.percentile_interval` |
| `src/reproducibility.py` | Тонкие обёртки над вызовами, которые код делает и так; новых обращений к генераторам нет | `seed_numpy_global(seed)`, `seed_torch(seed)`, `set_num_threads(n)`, `environment_info()` (версии, коммит, число потоков — для записи рядом с результатами) |
| `src/pipeline.py` | Описание одного плеча эксперимента и его запуск | `ExperimentConfig` (`data: DataSpec`, модель и переопределения, потери, эпохи, `batch_size`, `learning_rate`, набор признаков, `train_data`, `augmentation`, метрики, `tests: tuple[EvalSpec]`, `tags`); `DataSpec` (сценарии или группа с фиксированной связью, число траекторий, критерий, `SimulatorSpec`, кэш `none`/`disk`/`memory`, окно, горизонт, доли разбиения, смещения сидов); `EvalSpec` (наблюдение, sigma, phi, независимые данные); `run_experiment(config, seed) -> list[dict]` (одна строка на тест); `iter_rows(configs, seeds)`; `run_seeds(configs, seeds) -> pandas.DataFrame` |
| `experiments/common.py` | Общая обвязка скриптов | `parse_args` (`--seeds`, `--fast`, `--tag`, `--out`), `results_path`, `ResultsWriter` и `write_results` (CSV строка за строкой и `<csv>.provenance.json` с конфигурацией, сидами и `environment_info()`), `run_arms` |
| `experiments/` | Тонкие скрипты: набор конфигураций → `pipeline.iter_rows` → CSV | перенесены R15, R16, R21, R22, R23 (a, b), R24 (a), R25; остальные пока вызывают функции напрямую |
| `notebooks/pipeline.ipynb` | Пошаговое прохождение пайплайна с вызовом этих функций | — |

## Правила

- **Варианты по псевдониму.** Новый вариант — новая запись в реестре, а не ветка `if` в чужом коде.
- **Метрики только из `src/evaluation`.** Никаких локальных реализаций R², AUROC и т. п. в экспериментах.
- **Случайность только через `src.reproducibility` и явные генераторы.** Поведение должно совпадать с прежним, эталон `scripts/golden_outputs.py` — бит в бит.
- **Кэш траекторий и побайтовые сравнения.** Ключ кэша не содержит вариант симулятора, а попадание в кэш не расходует случайные числа глобального генератора, поэтому при частичном кэше начальные состояния следующих траекторий сдвигаются: числа зависят от состояния `data/trajectories`. Параллельные запуски в одном каталоге делят кэш. Для сравнения «до/после» каждый запуск ведут из своего каталога с холодным кэшем.
- **Чистка не меняет числа.** Новая функциональность (`"shots"`, `"sequential"`, `"local"`) добавляется после чистки отдельным изменением.

## Конвенция знака в двухкубитной модели

Для одного кубита оператор затухания — `sigmam()`: релаксация в |1⟩, основное состояние H = ωσz. Для двух кубитов — `destroy(2)`: релаксация в |0⟩, а при поле h·σz это верхнее по энергии состояние.

На результаты это не влияет. Глобальный переворот X⊗X переводит модель в модель с полем −h и затуханием в основное состояние. При этом не меняются:
- ℓ₁-когерентность в вычислительном базисе;
- чистота и конкурентность;
- мера Хаара на начальных состояниях;
- угол смешивания, который зависит только от |h|.

Конвенцию нужно указать в статье 2 и в документе. Код оставлен как есть, потому что от него зависят опубликованные числа.
