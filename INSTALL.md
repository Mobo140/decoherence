# 📦 Установка зависимостей

## Быстрая установка

```bash
cd <home>/Science/Physics/Quantum_physics_with_DL/Realization

# Вариант 1: Установить все зависимости
pip3 install -r requirements.txt

# Вариант 2: Установить как пакет (рекомендуется)
pip3 install -e .
```

## Пошаговая установка (если возникают проблемы)

### 1. Основные зависимости

```bash
# NumPy, SciPy, matplotlib
pip3 install numpy scipy matplotlib pandas tqdm pyyaml

# Scikit-learn для метрик
pip3 install scikit-learn
```

### 2. PyTorch (для DataLoader)

```bash
# Для CPU (быстрее установка)
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# Для GPU (если есть CUDA)
pip3 install torch
```

### 3. JAX/Flax (для моделей)

```bash
# Для CPU
pip3 install jax jaxlib flax optax

# Для GPU (если есть CUDA)
pip3 install jax[cuda12] flax optax
```

### 4. QuTiP (для квантовых симуляций)

```bash
pip3 install qutip
```

### 5. Опциональные (для логирования)

```bash
# TensorBoard (опционально)
pip3 install tensorboard

# Weights & Biases (опционально)
pip3 install wandb
```

## Проверка установки

После установки проверьте, что всё работает:

```bash
python3 -c "
import numpy as np
import torch
import jax
import qutip
import sklearn
print('✅ Все модули установлены!')
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'JAX: {jax.__version__}')
print(f'QuTiP: {qutip.__version__}')
"
```

## Минимальная установка (только для тестирования)

Если нужна быстрая проверка работоспособности:

```bash
pip3 install numpy scipy matplotlib qutip torch jax flax optax scikit-learn pyyaml tqdm
```

## Решение проблем

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip3 install torch
```

### "ModuleNotFoundError: No module named 'sklearn'"

```bash
pip3 install scikit-learn
```

### "No module named 'jax'"

```bash
pip3 install jax jaxlib flax optax
```

### Проблемы с QuTiP

```bash
# Установить зависимости для QuTiP
pip3 install numpy scipy cython
pip3 install qutip
```

### Конфликты версий

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

## Использование с conda

```bash
# Создать окружение
conda create -n quantum python=3.10
conda activate quantum

# Установить зависимости
conda install numpy scipy matplotlib pandas scikit-learn tqdm pyyaml
conda install pytorch -c pytorch
conda install -c conda-forge qutip
pip install jax flax optax
```

## Версии Python

Рекомендуется Python 3.8 - 3.11

Проверить версию:
```bash
python3 --version
```

## После установки

Запустите быстрый тест:

```bash
# Мини-тест (2 минуты)
./run_early_warning_pipeline.sh E5_mini

# Или полный эксперимент E5 (~40 минут)
./run_early_warning_pipeline.sh E5
```

---

**Важно:** Если у вас есть GPU и CUDA, установите соответствующие версии JAX и PyTorch для ускорения обучения.
