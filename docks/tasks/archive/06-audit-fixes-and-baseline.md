# Task: Audit fixes — context-code bug, censored labels, T₂ interpolation, stretched-exp baseline

## Why

Аудит кода нашёл 4 проблемы, которые искажают метрики всех статей, и одну методологическую дыру,
которую первым делом спросит рецензент (отсутствие сильного не-ML baseline):

1. **БАГ: рассинхрон context-кодов train/inference в E9.**
   `generate_dataset.py` (строки 183–192) кодирует 4-классовый код:
   `0 = 1q σ₋, 1 = 1q σ_z, 2 = 2q XXZ, 3 = 2q TFIM`.
   Но `experiments/e9_context_injection.py` (строка 111) на инференсе подаёт
   `("D", 0.0), ("E", 1.0)` — модель обучалась на кодах 2.0/3.0, а на инференсе
   получала 0.0/1.0 (= «1-кубит»). Результаты E9 частично невалидны.
   E10/E11 используют правильные коды 2.0/3.0 — их не трогаем.

2. **Цензурированные метки трактуются как истинные.**
   `QuTipSimulator._compute_t_decoh` (qutip_simulator.py, строка 179) возвращает
   `t_max`, если порог 1/e не пересечён. Это не «T₂ = t_max», это «T₂ > t_max»
   (right-censored). Такие метки занижают ошибку и смещают регрессию.

3. **T₂ квантуется сеткой dt без интерполяции.**
   `_compute_t_decoh` возвращает `times[i]` первого пересечения → метка
   систематически завышена на величину до `dt` (0.1). Дёшево чинится линейной
   интерполяцией между `times[i-1]` и `times[i]`.

4. **BCE по вероятностям вместо logits** в `lstm_predictor.py` (`_bce`, строка 440) —
   численно менее стабильно; `binary_cross_entropy_with_logits` надёжнее.

5. **Нет baseline «нелинейный фит».** physics_only — это OLS по log C (чистая
   экспонента). Для time-dependent γ и 2-кубит честный классический baseline —
   stretched exponential `C(t) = C₀·exp(−(t/τ)^β)`. Без него claim C2
   («ML нужен») недоказуем: вдруг обычный curve_fit даёт те же R².

Поддерживаемые claims: C2, C5. После фикса меток метрики E1–E9 слегка изменятся —
это ожидаемо, обновить таблицы.

## Scope

**Входит:**
- `src/infrastructure/quantum/qutip_simulator.py` — интерполяция T₂ + флаг censored
- `src/domain/value_objects.py` — поле `censored` в `TrajectoryResult`
- `src/application/generate_dataset.py` — исключение censored траекторий, фикс docstring кодов
- `src/infrastructure/ml/lstm_predictor.py` — BCE with logits
- `src/infrastructure/ml/stretched_exp_predictor.py` — НОВЫЙ файл
- `src/application/run_ablation.py` — новый variant + чистка `_apply_noise`
- `experiments/e9_context_injection.py` — фикс кодов
- `experiments/e12_baseline_comparison.py` — НОВЫЙ эксперимент

**Не входит:**
- `src/domain/ports.py` — сигнатуры портов не меняем
- Survival-loss / квантильные головы — отдельная задача 07 (после этой)
- Переобучение E1–E8 целиком (только e12 + повтор e9)

## Шаги (выполнять строго по порядку)

### Шаг 1 — фикс кодов в E9 (5 минут)

В `experiments/e9_context_injection.py` строка 111 заменить:

```python
for scenario_label, int_code in [("D", 0.0), ("E", 1.0)]:
```

на:

```python
for scenario_label, int_code in [("D", 2.0), ("E", 3.0)]:
```

Проверить остальные места в этом файле: везде, где задаётся
`inference_interaction_code` для сценариев D/E, должны быть 2.0/3.0
(сверяться с таблицей кодов в `generate_dataset.py`, строки 183–187).
Также исправить устаревший комментарий в `src/application/generate_dataset.py`
строки 47–49 (`# interaction_type_codes: 0.0 = XXZ ...`) на актуальную
4-классовую схему.

### Шаг 2 — интерполяция T₂ и флаг censored

2a. В `src/domain/value_objects.py` в `TrajectoryResult` добавить поле:

```python
censored: bool = False  # True: порог не пересечён за t_max, t_decoh — нижняя граница
```

(добавлять ПОСЛЕ существующих полей, чтобы не сломать позиционные вызовы).

2b. В `qutip_simulator.py` переписать `_compute_t_decoh` так, чтобы он возвращал
кортеж `(t_decoh: float, censored: bool)`:
- при пересечении порога между i−1 и i вернуть линейную интерполяцию:

```python
frac = (prev_val - abs_threshold) / (prev_val - val + 1e-12)
t_cross = times[i-1] + frac * (times[i] - times[i-1])
return float(t_cross), False
```

- если порог не пересечён — `return float(times[-1]), True`.

2c. В `_simulate_single` и `_simulate_two` прокинуть оба значения в
`TrajectoryResult(..., censored=censored)`.

2d. В `generate_dataset.py` в `_build_dataset` пропускать censored траектории:

```python
if getattr(traj, "censored", False):
    n_censored += 1
    continue
```

и добавить `"n_censored": n_censored` в `metadata`.

### Шаг 3 — BCE with logits

В `lstm_predictor.py`:
- в `_ResidualNet.forward` вернуть сырой `risk_logit` вместо `torch.sigmoid(risk_logit)`;
- в `_bce` использовать `nn.functional.binary_cross_entropy_with_logits(pred, target, pos_weight=_pw_tensor.to(pred.device))` (параметр `weight` убрать, `pos_weight` делает то же для положительного класса);
- в `predict()` применить `torch.sigmoid(...)` к risk перед `PredictionResult`.
Аналогично проверить `transformer_predictor.py` — если там та же схема, применить тот же фикс.

### Шаг 4 — StretchedExpPredictor (новый baseline)

Создать `src/infrastructure/ml/stretched_exp_predictor.py`:

```python
"""Classical baseline: stretched-exponential fit C(t) = C0 * exp(-(t/tau)**beta).

With the relative 1/e threshold, coherence crosses C0/e exactly at t = tau,
so the predicted decoherence time is simply the fitted tau.
No training required (fit happens per-window at predict time).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ...domain.ports import IPredictor
from ...domain.value_objects import PredictionResult


def _stretched_exp(t, log_c0, tau, beta):
    return log_c0 - (t / tau) ** beta


class StretchedExpPredictor(IPredictor):
    def __init__(self, t_max: float = 20.0, dt: float = 0.1) -> None:
        self.t_max = t_max
        self.dt = dt

    @property
    def is_trained(self) -> bool:
        return True

    def train(self, dataset, **kwargs):
        from ...application.train_model import TrainModelResult
        return TrainModelResult(predictor=self, train_loss_history=[], val_loss_history=[])

    def predict(self, window: np.ndarray, t_obs: float, horizon: float) -> PredictionResult:
        coh = np.abs(window[:, -1]) + 1e-8
        T = len(coh)
        # absolute time axis: window ends at t_obs
        t = t_obs - (T - 1 - np.arange(T)) * self.dt
        t = np.clip(t, 1e-6, None)
        log_coh = np.log(coh)
        try:
            popt, _ = curve_fit(
                _stretched_exp, t, log_coh,
                p0=(log_coh[0], max(t_obs, 1.0), 1.0),
                bounds=([-20.0, 1e-3, 0.2], [5.0, 10.0 * self.t_max, 3.0]),
                maxfev=2000,
            )
            tau = float(popt[1])
        except Exception:
            tau = self.t_max  # fit failed: pessimistic fallback
        t_decoh_pred = float(np.clip(tau, t_obs, self.t_max))
        remaining = t_decoh_pred - t_obs
        risk = 1.0 if remaining <= horizon else 0.0
        return PredictionResult(
            t_decoh_predicted=t_decoh_pred,
            risk_score=risk,
            horizon=horizon,
        )

    def state_bytes(self) -> bytes:
        import json
        return json.dumps({"t_max": self.t_max, "dt": self.dt}).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "StretchedExpPredictor":
        import json
        d = json.loads(data.decode())
        return cls(t_max=d["t_max"], dt=d["dt"])
```

Сверить методы с `IPredictor` в `src/domain/ports.py` и с тем, как
`PhysicsOnlyPredictor` реализует no-op train — повторить тот же паттерн,
если он отличается от приведённого выше.

### Шаг 5 — подключить variant

5a. В `src/domain/value_objects.py` в `PredictorVariant` добавить:

```python
STRETCHED_EXP = "stretched_exp"          # curve_fit baseline, no ML
```

5b. В `run_ablation.py` в `_build_predictor` добавить ветку (по аналогии с PHYSICS_ONLY):

```python
if variant == PredictorVariant.STRETCHED_EXP:
    t_max = max((c.t_max for c in command.configs), default=100.0)
    return StretchedExpPredictor(t_max=t_max)
```

и убедиться, что в `execute()` этот variant попадает в ветку «без training»
(условие `variant != PredictorVariant.PHYSICS_ONLY` заменить на
`variant not in (PredictorVariant.PHYSICS_ONLY, PredictorVariant.STRETCHED_EXP)`).

5c. Там же почистить `_apply_noise`: удалить строку с
`object.__setattr__(...) if hasattr(...) else None` (мёртвый код), оставить
`copy.copy(dataset)` + присваивание `noisy_ds.sequences = noisy_seqs`.

### Шаг 6 — эксперимент E12

Создать `experiments/e12_baseline_comparison.py` по образцу
`experiments/e1_ablation.py` (скопировать структуру: импорты, `build_configs`,
`AblationCommand`, сохранение CSV):

- **Цель:** проверить, что physics_lstm обгоняет классический curve_fit baseline.
- **Claim:** C2.
- Сценарии: B, C, D, E (по 100 траекторий, seed=42).
- Variants: `PHYSICS_ONLY`, `STRETCHED_EXP`, `PHYSICS_LSTM`.
- Результат: `experiments/results/e12_baseline_comparison.csv`
  (колонки как в e1: scenario, variant, mae, rmse, r2, mape, auroc, n_samples).

### Шаг 7 — тесты и перезапуск

7a. В `tests/` добавить `test_censoring_and_interpolation.py`:
- тест: константная γ=0.4, σ₋, t_max=30 → T₂ аналитически = 2/γ = 5.0;
  проверить `abs(traj.t_decoh - 5.0) < 0.05` (раньше ошибка была до dt=0.1)
  и `censored is False`;
- тест: γ=0.01, t_max=5 → порог не достигается: `censored is True`;
- тест: датасет из 10 конфигов с одной censored траекторией →
  `metadata["n_censored"] == 1`, censored окон нет.

7b. Запустить: `python -m pytest tests/ -x -q` — все тесты зелёные.

7c. Запустить `python experiments/e12_baseline_comparison.py`.

7d. Перезапустить `python experiments/e9_context_injection.py` (с фиксом кодов)
и сравнить TFIM R² с прежним значением из `experiments/results/e9a_cross_system.csv`
(сохранить старый файл как `e9a_cross_system_buggy_codes.csv` перед перезапуском).

7e. Обновить `docks/current/CURRENT_STATUS.md` и таблицы в
`docks/main/PROJECT_PLAN.md` новыми числами. Если метрики paper-таблиц
изменились > 0.01 — отметить в задаче, какие таблицы в `paper/` требуют обновления
(сами .tex не трогать без отдельного запроса).

## Acceptance Criteria

- [x] В `e9_context_injection.py` коды D/E = 2.0/3.0, совпадают с `generate_dataset.py`
- [x] `_compute_t_decoh` интерполирует T₂ и возвращает флаг censored
- [x] Censored траектории исключены из датасета, `metadata["n_censored"]` заполняется
- [x] BCE считается через `binary_cross_entropy_with_logits`
- [x] `StretchedExpPredictor` реализует `IPredictor` полностью
- [x] `experiments/results/e12_baseline_comparison.csv` создан, содержит 3 варианта × 4 сценария
- [x] На сценарии B: R²(physics_lstm)=0.967 > R²(stretched_exp)=0.820
- [x] Новые тесты зелёные (37 passed). `test_lindblad_systems.py` и `test_decoherence_time.py` падают на `src.utils` — pre-existing, не из этой задачи
- [x] E1 smoke OK (все 5 variant, включая stretched_exp)

## Invariants

- IV1: сигнатура `IPredictor.predict(window, t_obs, horizon)` не меняется
- IV2: старые `.pt` загружаются через `from_bytes()` (поле censored не сериализуется в модель)
- IV3: `TrajectoryResult(times, observables, t_decoh, system_config)` без censored продолжает работать (default=False)
- IV4: domain не импортирует torch/qutip/scipy

## Verification

```bash
python -m pytest tests/ -x -q
python experiments/e12_baseline_comparison.py
python experiments/e9_context_injection.py
python -c "from src.infrastructure.ml.stretched_exp_predictor import StretchedExpPredictor; print('OK')"
```

## Status

`done`

## Conclusion

### E9 rerun vs buggy codes

E9a cross-system:
- D: 0.696 (buggy) → 0.363 (код 0.5) → **0.537** (коды 2/3)
- E: 0.596 (buggy) → 0.469 (код 0.5) → **0.452** (коды 2/3)

E9b single-scenario (коды 2/3):
- D physics_lstm 0.691 (было 0.559), transformer 0.792 (было 0.767)
- E physics_lstm 0.205 (было 0.603), transformer 0.279 (было 0.576)

Context codes не дотягивают TFIM до 0.70. Падение E частично из-за исключения censored траекторий (другой состав датасета).

`.tex` E9 не цитирует — обновлять paper не нужно.

### E12 numbers (n=100, seed=42)

- B: physics_lstm 0.967 > stretched_exp 0.820 > physics_only 0.749
- C: stretched_exp 0.978 ≈ physics_only 0.965 >> physics_lstm −0.182 (default LSTM без E8-гиперпараметров)
- D: physics_lstm 0.210 > stretched_exp −0.155 > physics_only −0.796
- E: physics_lstm −0.149 > stretched_exp −6.41 > physics_only −7.62

### Paper tables

E12 — новый эксперимент, в `.tex` ещё нет. Старые таблицы E1–E8 не пересчитывались (по плану). После рерана E9 обновить числа в `paper/paper_2/` если ΔR² > 0.01.

### Deviations from plan

- `noise.py` прокидывает `censored=` при копировании траектории — иначе флаг терялся бы на шуме.
- `StretchedExpPredictor.train` возвращает `self`, как `PhysicsOnlyPredictor`, а не `TrainModelResult`.
- `_compute_t_decoh` вынесен helper `_interpolate_crossing` (i=0 → `times[0]`).
- E12 получил флаг `--fast` (n=20, 15 эпох) сверх плана; канонический прогон — без флага.
- E9a не выставлял per-scenario код на eval (оставался 0.5 = OOD для 4-class). Добавлен `CrossSystemCommand.eval_interaction_codes`; E9a перезапущен.

## Priority

P0 — шаги 1–3 (баги, влияют на валидность результатов); P1 — шаги 4–7.

## Контекст для следующей задачи (07, не делать сейчас)

После этой задачи логичное «интересное» развитие — переформулировка как
survival/RUL-задача (right-censored time-to-event): censored траектории не
выбрасывать, а учить с асимметричным лоссом (штраф только если предсказание
меньше границы цензурирования) + квантильная голова для доверительных интервалов.
Литература: SurvLoss (PHM Society 2024), «LSTM and Transformers based methods for
RUL Prediction considering Censored Data» (IJPHM), arXiv:2405.01614.
Это даст статье уникальный угол: в quantum-ML такой постановки ещё нет.
