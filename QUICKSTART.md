# 🚀 Quick Start Guide - Early Warning System

## Быстрый запуск за 3 команды

```bash
# 1. Перейти в директорию проекта
cd /Users/nb/Science/Physics/Quantum_physics_with_DL/Realization

# 2. Сделать скрипт исполняемым (только один раз)
chmod +x run_early_warning_pipeline.sh

# 3. Запустить полный pipeline
./run_early_warning_pipeline.sh E5
```

**Это всё!** Скрипт автоматически:
- ✅ Сгенерирует датасет (500 траекторий)
- ✅ Обучит multi-task модель (200 эпох)
- ✅ Оценит результаты и создаст графики

---

## Пошаговый запуск (если нужен контроль)

### Шаг 1: Генерация датасета (~5 минут)

```bash
python3 -m src.data.generate_dataset \
    --config configs/experiments/E5.yaml \
    --sliding-window
```

**Результат:** Файл `data/E5_dataset.npz` с окнами наблюдения

### Шаг 2: Обучение (~30-60 минут)

```bash
python3 -m src.train.train_multitask \
    --config configs/experiments/E5.yaml \
    --output-dir runs/E5_exp1
```

**Результат:** Чекпоинты в `checkpoints/E5/`

### Шаг 3: Оценка (~2 минуты)

```bash
python3 -m src.train.eval_multitask \
    --config configs/experiments/E5.yaml \
    --checkpoint checkpoints/E5 \
    --output-dir figures/E5_results \
    --split test
```

**Результат:** Метрики и графики в `figures/E5_results/`

---

## Быстрый тест (2 минуты)

Если хотите быстро проверить, что всё работает:

```bash
# Скачать мини-конфиг
cat > configs/experiments/E5_mini.yaml << 'EOF'
data:
  n_samples: 20
  output_path: data/E5_mini_dataset.npz
  dataset_path: data/E5_mini_dataset.npz
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42

physics:
  system_type: single_qubit
  observables: [sigma_x, sigma_y, sigma_z]
  gamma_range: [0.1, 2.0]
  omega_range: [0.5, 2.0]
  time_dependent_gamma: false
  T_max: 10.0
  dt: 0.1
  decoherence_criterion: coherence
  decoherence_threshold: 0.01
  initial_state_type: random

model:
  type: multitask_transformer
  d_model: 64
  n_layers: 2
  n_heads: 2
  d_ff: 256
  dropout_rate: 0.2
  max_seq_len: 500
  enable_risk: true
  enable_forecast: false
  enable_time: true
  n_deltas: 3
  head_hidden_dims: [64, 32]

training:
  window_length: 50
  samples_per_trajectory: 2
  stratified_sampling: true
  deltas: [0.5, 1.0, 2.0]
  batch_size: 16
  n_epochs: 5
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.0001
  lambda_risk: 1.0
  lambda_time: 0.5
  checkpoint_dir: checkpoints/E5_mini
  history_dir: runs/E5_mini

evaluation:
  output_dir: figures/E5_mini
  risk_thresholds: [0.5]
  max_fpr: 0.05
EOF

# Запустить
./run_early_warning_pipeline.sh E5_mini
```

---

## Посмотреть результаты

```bash
# Метрики в JSON
cat figures/E5_results/metrics.json

# Открыть все графики
open figures/E5_results/*.png  # macOS
# или
xdg-open figures/E5_results/*.png  # Linux
```

**Ключевые метрики:**
- `risk_auroc` - должен быть > 0.85
- `lead_mean_lead_time` - должен быть > 0.5
- `risk_precision`, `risk_recall` - баланс точности

---

## Доступные эксперименты

### E5 - Basic Early Warning (Рекомендуется для начала)
```bash
./run_early_warning_pipeline.sh E5
```
- Задачи: Risk + Time
- Система: Single-qubit, constant γ
- Время: ~40 минут

### E6 - With Forecasting
```bash
./run_early_warning_pipeline.sh E6
```
- Задачи: Risk + Forecast + Time
- Система: Single-qubit, time-dependent γ(t)
- Время: ~60 минут (больше модель)

### E7 - Control Baseline
```bash
./run_early_warning_pipeline.sh E7
```
- Задачи: Risk + Time + Control
- Демонстрация улучшения через triggered control
- Время: ~40 минут

---

## Пропустить этапы

Если датасет уже создан:
```bash
./run_early_warning_pipeline.sh E5 true
```

Если и датасет, и модель готовы (только оценка):
```bash
./run_early_warning_pipeline.sh E5 true true
```

---

## Решение проблем

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Permission denied"
```bash
chmod +x run_early_warning_pipeline.sh
```

### "Config not found"
```bash
# Проверить, что вы в правильной директории
pwd
# Должно быть: .../Realization
```

### Проблемы с JAX/GPU
```bash
# Использовать CPU
export JAX_PLATFORM_NAME=cpu
python3 -m src.train.train_multitask ...
```

---

## Что дальше?

После успешного запуска:

1. **Посмотреть графики** в `figures/E5_results/`
2. **Проанализировать метрики** в `metrics.json`
3. **Сравнить эксперименты** E5 vs E6 vs E7
4. **Настроить параметры** в конфигах
5. **Прочитать полную документацию** в `reports/early_warning_report.md`

---

## Полезные команды

```bash
# Посмотреть прогресс обучения
tail -f runs/E5_exp1/history.json

# Проверить размер датасета
python3 -c "import numpy as np; d=np.load('data/E5_dataset.npz'); print(f'Samples: {len(d[\"sequences\"])}')"

# Список всех чекпоинтов
ls -lh checkpoints/E5/

# Очистить старые результаты
rm -rf data/E5_dataset.npz checkpoints/E5/ figures/E5_results/
```

---

**Документация:**
- 📘 Полное руководство: `README_EARLY_WARNING.md`
- 📊 Отчёт: `reports/early_warning_report.md`
- ✅ Статус: `COMPLETION_STATUS.md`

**Вопросы?** См. FAQ в `README_EARLY_WARNING.md`
