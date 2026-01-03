# 🚨 Early Warning System for Quantum Decoherence

Система раннего предупреждения о классикализации (декогеренции) квантовых систем на основе глубокого обучения.

## 🎯 Основные возможности

- **Multi-task Learning**: Одновременное решение трёх задач
  - 🔴 **Risk Classification**: Предсказание вероятности декогеренции в ближайшем окне
  - 🔮 **Forecasting**: Прогнозирование будущих наблюдаемых
  - ⏱️ **Time Regression**: Оценка оставшегося времени

- **Early Warning Metrics**: Специализированные метрики
  - AUROC, AUPRC для классификации риска
  - Lead time (время предупреждения)
  - TPR@FPR для контроля false positives
  - Калибровка вероятностей

- **Control Baseline**: Демонстрация практической пользы
  - Триггерное управление на основе предсказаний риска
  - Pulse control, parameter modification, Hamiltonian control
  - Сравнение траекторий с/без контроля

- **Visualizations**: Comprehensive визуализация результатов
  - ROC/PR curves
  - Reliability diagrams
  - Lead time distributions
  - Trajectory analysis with risk

## 📦 Установка

```bash
# Клонировать репозиторий
cd /path/to/Realization

# Установить зависимости
pip install -r requirements.txt

# Или установить как пакет
pip install -e .
```

### Требования

- Python 3.8+
- JAX / Flax
- PyTorch
- QuTiP
- scikit-learn
- matplotlib
- numpy, scipy

## 🚀 Quick Start

### Шаг 1: Генерация датасета

```bash
# E5: Базовая система раннего предупреждения
python3 -m src.data.generate_dataset \
    --config configs/experiments/E5.yaml \
    --sliding-window

# E6: С прогнозированием
python3 -m src.data.generate_dataset \
    --config configs/experiments/E6.yaml \
    --sliding-window

# E7: Для control baseline
python3 -m src.data.generate_dataset \
    --config configs/experiments/E7.yaml \
    --sliding-window
```

**Параметры датасета** (в конфиге):
- `n_samples`: Число траекторий
- `window_length`: Длина окна наблюдения (L)
- `samples_per_trajectory`: Число окон на траекторию
- `stratified_sampling`: Балансировка классов
- `deltas`: Горизонты для риска [Δ1, Δ2, Δ3]
- `forecast_horizon`: Горизонт прогноза (H)

### Шаг 2: Обучение модели

```bash
# E5: Risk + Time
python3 -m src.train.train_multitask \
    --config configs/experiments/E5.yaml \
    --output-dir runs/E5_exp1

# E6: Risk + Forecast + Time
python3 -m src.train.train_multitask \
    --config configs/experiments/E6.yaml \
    --output-dir runs/E6_exp1

# E7: С фокусом на контроль
python3 -m src.train.train_multitask \
    --config configs/experiments/E7.yaml \
    --output-dir runs/E7_exp1
```

**Параметры обучения**:
- `lambda_risk`, `lambda_forecast`, `lambda_time`: Веса лоссов
- `use_focal_loss`: Использовать Focal Loss для дисбаланса
- `pos_weight`: Вес для положительного класса
- `batch_size`, `n_epochs`, `learning_rate`

### Шаг 3: Оценка и визуализация

```bash
# Оценить на test set
python -m src.train.eval_multitask \
    --config configs/experiments/E5.yaml \
    --checkpoint checkpoints/E5 \
    --output-dir figures/E5_results \
    --split test
```

**Выходные файлы**:
- `metrics.json`: Все метрики
- `roc_curve.png`: ROC кривая
- `pr_curve.png`: Precision-Recall кривая
- `reliability_diagram.png`: Калибровка
- `risk_histogram.png`: Распределение рисков
- `lead_time_distribution.png`: Распределение lead time
- `forecast_example_*.png`: Примеры прогнозов (для E6)

### Шаг 4: Control Baseline (E7)

```bash
# Сравнить траектории с/без контроля
python3 -c "
from src.physics.control import compare_with_without_control, PulseControl
from src.utils.config import load_config
import qutip as qt
import numpy as np

# Загрузить конфиг
config = load_config('configs/experiments/E7.yaml')

# Создать control action
sigma_x = qt.sigmax()
control = PulseControl(pulse_operator=sigma_x, strength=0.2)

# TODO: Загрузить обученную модель как risk_model
# risk_model = lambda window: model.predict(window)

# Сравнить
results = compare_with_without_control(
    system_type='single_qubit',
    system_params={'omega': 1.0, 'gamma_const': 0.8},
    initial_state=qt.basis(2, 0) * qt.basis(2, 0).dag(),
    times=np.arange(0, 10, 0.05),
    observables=['sigma_x', 'sigma_y', 'sigma_z'],
    decoherence_criterion='coherence',
    decoherence_threshold=0.01,
    risk_model=risk_model,  # Ваша обученная модель
    risk_threshold=0.6,
    control_action=control,
    n_trials=20
)

print(results)
"
```

## 📊 Эксперименты

### E5: Basic Early Warning

**Цель**: Базовая система раннего предупреждения

**Конфигурация**:
- Система: Single-qubit, constant γ
- Задачи: Risk + Time
- Модель: Transformer (256d, 4 layers, 8 heads)
- Горизонты: Δ = [0.5, 1.0, 2.0]

**Ожидаемые метрики**:
- AUROC > 0.85
- Mean Lead Time > 0.5
- TPR@FPR5% > 0.70

### E6: Multi-task with Forecasting

**Цель**: Полная система с прогнозированием

**Конфигурация**:
- Система: Single-qubit, time-dependent γ(t)
- Задачи: Risk + Forecast + Time
- Модель: Transformer (512d, 6 layers) - больше для forecast
- Forecast horizon: H = 20

**Ожидаемые метрики**:
- AUROC > 0.80
- Forecast MAE < 0.15
- Улучшение lead time за счёт прогноза

### E7: Control Baseline

**Цель**: Демонстрация практической пользы

**Конфигурация**:
- Система: Single-qubit с triggered control
- Control: Pulse (X-rotation, θ=0.2)
- Trigger threshold: p_risk > 0.6

**Ожидаемые результаты**:
- Относительное улучшение t_decoh > 15%
- Увеличение purity > 10%
- Success rate > 60%

## 📈 Метрики

### Risk Classification

```python
from src.train.metrics import compute_risk_metrics

metrics = compute_risk_metrics(y_probs, y_true)
# {'auroc': 0.87, 'auprc': 0.82, 'precision': 0.78, 'recall': 0.85, 'f1': 0.81}
```

### Lead Time

```python
from src.train.metrics import compute_lead_time_metrics

lead_metrics = compute_lead_time_metrics(y_probs, t_obs, t_decoh, threshold=0.5)
# {'mean_lead_time': 0.65, 'median_lead_time': 0.52, 
#  'detection_rate': 0.88, 'false_alarm_rate': 0.12}
```

### Forecasting

```python
from src.train.metrics import compute_forecast_metrics

forecast_metrics = compute_forecast_metrics(pred_sequence, true_sequence)
# {'forecast_mae': 0.12, 'forecast_rmse': 0.18}
```

## 🎨 Визуализация

### ROC Curve

```python
from src.utils.early_warning_plots import plot_roc_curve

plot_roc_curve(
    y_true=labels,
    y_probs=predictions,
    save_path='figures/roc.png',
    delta_labels=['0.5', '1.0', '2.0']
)
```

### Lead Time Distribution

```python
from src.utils.early_warning_plots import plot_lead_time_distribution

plot_lead_time_distribution(
    lead_times=lead_time_array,
    save_path='figures/lead_time_dist.png'
)
```

### Trajectory with Risk

```python
from src.utils.early_warning_plots import plot_trajectory_with_risk

plot_trajectory_with_risk(
    times=times,
    observables=obs_matrix,
    risk_probs=risk_predictions,
    t_obs_points=observation_times,
    t_decoh=decoherence_time,
    threshold=0.5,
    save_path='figures/trajectory_risk.png'
)
```

## 🔧 Конфигурация

### Пример конфига (E5)

```yaml
model:
  type: multitask_transformer
  d_model: 256
  n_layers: 4
  n_heads: 8
  enable_risk: true
  enable_forecast: false
  enable_time: true
  n_deltas: 3

training:
  window_length: 50
  deltas: [0.5, 1.0, 2.0]
  batch_size: 64
  n_epochs: 200
  lambda_risk: 1.0
  lambda_time: 0.5
  use_focal_loss: false

evaluation:
  risk_thresholds: [0.3, 0.5, 0.7]
  max_fpr: 0.05
```

## 📚 Архитектура кода

```
src/
├── data/                    # Генерация и загрузка данных
│   ├── dataset.py          # Dataset классы
│   ├── generate_dataset.py # Генерация с sliding window
│   └── transforms.py       # Нормализация
├── models/                  # Модели
│   ├── multitask.py        # Multi-task Transformer/RNN
│   ├── heads.py            # Task-specific heads
│   └── transformer.py      # Transformer encoder
├── train/                   # Обучение и оценка
│   ├── train_multitask.py  # Обучение multi-task моделей
│   ├── eval_multitask.py   # Оценка с визуализацией
│   ├── losses.py           # Multi-task losses
│   └── metrics.py          # Early warning метрики
├── physics/                 # Квантовые симуляции
│   ├── control.py          # Control actions и baseline
│   ├── simulate_qutip.py   # QuTiP симуляции
│   └── lindblad_systems.py # Quantum systems
└── utils/                   # Утилиты
    ├── early_warning_plots.py  # Визуализация
    ├── config.py           # Загрузка конфигов
    └── logging.py          # Логирование
```

## 🧪 Тестирование

```bash
# Быстрый тест генерации датасета
python -m src.data.generate_dataset \
    --config configs/experiments/E5.yaml \
    --sliding-window

# Проверить, что датасет создан правильно
python -c "
import numpy as np
data = np.load('data/E5_dataset.npz', allow_pickle=True)
print('Sequences shape:', data['sequences'].shape)
print('T_obs shape:', data['t_obs'].shape)
print('T_decoh shape:', data['t_decoh'].shape)
print('Train samples:', len(data['train_indices']))
print('Val samples:', len(data['val_indices']))
print('Test samples:', len(data['test_indices']))
"

# Тест обучения (1 эпоха)
python3 -m src.train.train_multitask \
    --config configs/experiments/E5.yaml \
    --output-dir test_run
# Затем проверить checkpoints/E5/ и runs/E5/
```

## ❓ FAQ

### Q: Как выбрать горизонты Δ?

**A**: Зависит от вашей системы и требований к lead time:
- Короткие Δ (0.1-0.5): Более точные, но меньше времени на реакцию
- Длинные Δ (1.0-3.0): Больше lead time, но ниже precision
- Рекомендация: Использовать несколько [0.5, 1.0, 2.0]

### Q: Как бороться с дисбалансом классов?

**A**: Несколько способов:
1. `stratified_sampling: true` в конфиге датасета
2. `use_focal_loss: true` в training конфиге
3. Установить `pos_weight` в соответствии с дисбалансом
4. Увеличить `samples_per_trajectory`

### Q: Модель переобучается, что делать?

**A**:
- Увеличить `dropout_rate` (0.2 → 0.3)
- Добавить `weight_decay` (попробовать 0.0001)
- Уменьшить размер модели (`d_model`, `n_layers`)
- Увеличить размер датасета (`n_samples`)
- Early stopping по val loss

### Q: Как интерпретировать lead time?

**A**: Lead time = t_decoh - t_first_alarm
- Положительный: Успешное раннее предупреждение
- Близкий к нулю: Предупреждение "в последний момент"
- Отрицательный: Ложная тревога или слишком поздно

### Q: Какой threshold выбрать для триггера?

**A**: Зависит от приложения:
- Низкий (0.3-0.4): Больше false positives, но выше recall
- Средний (0.5): Баланс precision/recall
- Высокий (0.6-0.7): Меньше false positives, но можно пропустить события
- Рекомендация: Анализировать PR curve и выбирать по требованиям

## 📄 Лицензия

MIT License

## 👥 Авторы

Quantum ML Research Team

## 📞 Контакты

Для вопросов и предложений: создайте issue в репозитории

---

**Полная документация**: См. `reports/early_warning_report.md`

**Дата обновления**: 2026-01-02
