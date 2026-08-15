# Документировать эксперименты (docstring + цель + claim)

## Task

Добавить стандартный docstring в каждый файл `experiments/eN_*.py`:
- Цель эксперимента
- Какой claim доказывает
- Входные параметры / сценарии
- Ожидаемые метрики и где сохраняются результаты

## Why

Сейчас файлы экспериментов E1–E9 не содержат объяснения зачем они нужны.
При рецензировании статьи или воспроизведении результатов неясно что запускать и почему.
Стандартный docstring делает код самодокументированным для co-авторов и рецензентов.

## Scope

**Входит:** docstring в начале каждого файла experiments/eN_*.py
**Не входит:** изменение логики экспериментов, запуск экспериментов

## Стандартный шаблон docstring

```python
"""Experiment EN: <Название>.

Goal
----
<Что измеряем и зачем>

Claim
-----
<CN: Текст утверждения из plan.md>

Scenarios
---------
<A/B/C/D/E — какие системы>

Metrics
-------
<R², MAE, AUROC — что ожидаем>

Output
------
experiments/results/eN_*.csv

Usage
-----
    python experiments/eN_name.py [--scenarios D E] [--n_trajectories 500]
"""
```

## Acceptance Criteria

- [x] `e1_ablation.py` — docstring с Goal, Claim C1+C2, Scenarios A/B/C
- [x] `e2_window_sweep.py` — docstring с Goal, Claim C3
- [x] `e3_noise_sweep.py` — docstring с Goal, Claim C4
- [x] `e4_cross_system.py` — docstring с Goal, Claim C5
- [x] `e5_inverse_baseline.py` — docstring с Goal, Claim C3
- [x] `e6_2qubit_improved.py` — docstring с Goal, почему 500 trajs + adaptive prior
- [x] `e7_paper2_extended.py` — docstring с Goal, τ-sweep + J-sweep
- [x] `e8_improved_2qubit.py` — docstring с Goal, все 4 улучшения
- [x] `e9_context_injection.py` — docstring с Goal, context codes

## Status

`done`

Шаблон Goal/Claim/Scenarios/Metrics/Output/Usage также у e10–e13.
Два E10 не путать: `e10_fixed_context.py` → `e10a/b_cross_system.csv`;
`e10_tfim_specialized.py` → `e10_tfim_*.csv`.

## Priority

P1 — можно делать параллельно с E9
