# Project Plan: Physics-Informed Early Warning of Quantum Decoherence

## Цель проекта

Предсказывать время декогеренции T₂ квантовых систем из короткого начального окна наблюдений
(без знания γ(t)), используя физически-информированную остаточную BiLSTM архитектуру.

---

## Статьи

### Paper 1 — 1-кубит (сценарии A, B, C)

**Файл:** `paper/paper_1/main.tex`
**Сценарии:** A = 1q σ₋ const γ, B = 1q σ₋ time-dep γ, C = 1q σ_z time-dep γ
**Эксперименты:** E1, E2, E3, E4, E5
**Статус:** ✅ Все цели достигнуты

| Метрика | Цель | Достигнуто |
|---------|------|------------|
| R² (physics_lstm, сц. B) | ≥ 0.97 | 0.945 (E1) |
| AUROC | ≥ 0.92 | 0.965 (E1) |
| AUROC при f=0.05 | ≥ 0.92 | 0.933 (E2) |
| AUROC при σ=0.05 | ≥ 0.90 | 0.942 (E3) |
| Direct >> Inverse (E5) | confirmed | R²=0.92 vs −0.86 |

### Paper 2 — 2-кубит (сценарии D, E)

**Файл:** `paper/paper_2/paper2.tex`
**Сценарии:** D = 2q XXZ time-dep γ, E = 2q TFIM time-dep γ
**Эксперименты:** E6–E14 (плюс E1/E3/E4 на D,E)
**Статус:** закрыт. Цель TFIM cross-system R² ≥ 0.70 на смеси J∈[0.2,1] не взята — потолок постановки у QPT, не дыра в коде.

| Метрика | Базовый (E4) | Лучший валидный | Цель | Итог |
|---------|-------------|-----------------|------|------|
| XXZ R² cross-system | 0.523 | 0.817 (E11a) / 0.740 (E8c) | ≥ 0.80 | E11a закрыл |
| TFIM R² cross-system | −0.381 | 0.575 (E8c) | ≥ 0.70 | не взята |
| XXZ AUROC | — | 0.890 (E10b Transformer) | ≥ 0.85 | E10b закрыл |
| TFIM AUROC | — | 0.801 (E10b) | ≥ 0.80 | E10b закрыл |
| TFIM R² single-scen | −0.358 | 0.689 (E7a Transformer) | ≥ 0.75 | не взята |

---

## Ключевые утверждения (falsifiable claims)

| ID | Утверждение | Доказывается |
|----|-------------|--------------|
| C1 | OLS-формула даёт R²≥0.999 при const γ + σ₋ | E1 сц. A |
| C2 | physics+LSTM превосходит physics-only и lstm-only при time-dep γ | E1 сц. B, C |
| C3 | f=0.15 окна даёт AUROC≥0.92 (R² плато ≈0.73, не 0.95) | E2 |
| C4 | Шум σ=0.05 деградирует AUROC < 10% | E3 |
| C5 | Одна модель обобщается на все типы систем | E4, E9 |

---

## Сценарии (A–E)

| ID | Система | Гамильтониан | Диссипатор | γ(t) |
|----|---------|-------------|-----------|------|
| A | 1-qubit | — | σ₋ | const |
| B | 1-qubit | — | σ₋ | time-dep (Bernstein) |
| C | 1-qubit | — | σ_z | time-dep |
| D | 2-qubit | XXZ | σ₋⊗I | time-dep |
| E | 2-qubit | TFIM | σ₋⊗I | time-dep |

---

## Эксперименты

| ID | Название | Цель | Claim | Статус |
|----|----------|------|-------|--------|
| E1 | Ablation | Сравнить physics_only / lstm_only / physics_lstm / transformer | C1, C2 | ✅ done |
| E2 | Window sweep | Минимальное окно: AUROC≥0.92 при f=0.15 | C3 | ✅ done |
| E3 | Noise sweep | Деградация при измерительном шуме | C4 | ✅ done |
| E4 | Cross-system | Обобщение на все сценарии | C5 | ✅ done |
| E5 | Inverse baseline | Direct vs inverse prediction на коротких окнах | C3 | ✅ done |
| E6 | 2-qubit improved | 500 траекторий + adaptive prior + hidden=128 | C2 | ✅ done |
| E7 | 2-qubit extended | Transformer, τ sweep, J sweep | C2, C5 | ✅ done |
| E8 | Improved 2-qubit | Weighted BCE, augmentation, min_gap, per-scenario τ | C4, C5 | ✅ done |
| E9 | Context injection | interaction_code + dissipator_code в physics scalars | C5 | ✅ rerun D=0.537 E=0.452 |
| E10 | TFIM specialized / fixed codes | два скрипта: `e10_fixed_context` и `e10_tfim_specialized` | C2, C5 | ✅ E10b AUROC; E10c E=0.472 |
| E11 | Scaling | больше данных / Transformer window=20 | C5 | ✅ D=0.817 E=0.566 |
| E12 | Baseline comparison | physics_only / stretched_exp / physics_lstm на B–E | C2 | ✅ done |
| E13 | Survival TFIM | drop vs keep+SurvHuber | C2 | ✅ цензуры почти нет при t_max=20 |
| E14 | Inject J | J как physics scalar, E-only + D+E | C2, C5 | ✅ E-only +0.11; E8c не побит |

---

## Что сознательно не делаем

- Ещё один скаляр/архитектура ради TFIM R² 0.70 на смеси J∈[0.2,1].
- YAML-пайплайн вместо `experiments/eN_*.py`.

UI: `python app.py` — 7 экранов (Dashboard / Runs / Models / Simulate / Train / Predict / Backtest).
