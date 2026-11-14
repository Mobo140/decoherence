# Карта экспериментов

- E — основные эксперименты (один прогон, seed 42), R — повторы и проверки (3–8 seed). Буква говорит, когда эксперимент написан; что он проверяет — в таблицах ниже.
- Папки: `paper1/` и `paper2/` — эксперименты, чьи таблицы и рисунки есть в статье 1 и статье 2; `checks/` — проверки, которые статьи не цитируют; `new/` — новые направления, в статьях их ещё нет. Общее лежит здесь: `common.py`, `j_sweep.py`, `_config.py`, цепочки `run_all.py`, `run_paper1.py`, `run_paper2_improved.py`.
- Запуск одного: `.venv/bin/python -m experiments.paper1.e1_ablation --fast` (флаги `--seeds`, `--fast`, `--tag`, `--out`; остальные — в `--help`).
- Результаты: плоско в `experiments/results/<имя>.csv` рядом с `<имя>.csv.provenance.json`; R28 — в `experiments/results/r28/`; `--fast` пишет `*_fast.csv`, он в `.gitignore`.
- Новый эксперимент собирать из рецептов (`src/blocks/README.md`), а не копировать скрипт.

Числа ниже взяты из CSV; R² — среднее по seed, если не сказано иное. «Документ» — `paper/description_ru.tex`. Сценарии: A — σ₋ с постоянной γ, B — σ₋ с γ(t), C — σ_z с γ(t), D — XXZ, E — TFIM (два кубита).

## Один кубит: какая модель лучше

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| E1 | `paper1/e1_ablation.py` | Абляция: physics_only, lstm_only, physics_lstm, transformer на A–C, 100 траекторий, один прогон | R² physics_lstm на B 0.945, lstm_only 0.009 (`e1_ablation_p1.csv`) | статья 1, документ |
| R7 | `paper1/r7_multiseed_1qubit.py` | Та же абляция, три seed. R11 — тот же скрипт с `--n-per 500` | physics_lstm A/B/C 0.881 / 0.865 / 0.982, lstm_only 0.783 / 0.802 / 0.934 (`r7_multiseed_1qubit.csv`); при 500 траекториях на B 0.976 против 0.959 (`r11_1qubit_500traj.csv`) | статья 1 (таблицы E1 и R11), документ |
| R13 | `paper1/r13_independent_test.py` | Абляция с оценкой на 400 независимых траекториях на сценарий, бутстреп-интервалы; добавлен вариант physics_feature | physics_lstm A/B/C 0.782 / 0.948 / 0.989; lstm_only 0.718 / 0.849 / 0.908 (`r13_independent_test.csv`) | статья 1, документ |

## Один кубит: длина окна

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| E2 | `paper1/e2_window_sweep.py` | Длина окна f = t_obs/T₂ от 0.05 до 0.5 на B+C, один прогон | R² 0.697–0.736, AUROC 0.933 при f=0.05 и 0.990 при f=0.5 (`e2_window_sweep_p1.csv`) | статья 1, документ |
| R12 | `paper1/r12_multiseed_window_sweep.py` | То же, три seed | R² 0.875–0.888, AUROC 0.979–0.991: от длины окна почти не зависит (`r12_multiseed_window_sweep.csv`) | статья 1 (таблица E2), документ |

## Один кубит: шум измерений

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| E3 | `paper1/e3_noise_sweep.py` | physics_lstm на чистых данных, оценка при гауссовом шуме σ, один прогон | R² 0.971 при σ=0, 0.197 при σ=0.05, 0.111 при σ=0.1 (`e3_noise_sweep_p1.csv`) | статья 1 |
| R15 | `paper1/r15_noise_variants.py` | Все модели, обучение на чистых данных, оценка при шуме, три seed | при σ=0.05 R²: physics_only −0.168, physics_lstm 0.312, lstm_only 0.656, transformer 0.715; при σ=0: 0.735 / 0.833 / 0.682 / 0.755 (`r15_noise_variants.csv`) | статья 1 (рисунок), документ |
| R16 | `paper1/r16_noise_aware_training.py` | Обучение на зашумлённых копиях окон, физические скаляры пересчитываются | при σ=0.05 physics_lstm с зашумлёнными копиями 0.701 против 0.312 без них (`r16_noise_aware_training.csv`) | статья 1, документ |
| R17 | `paper1/r17_slope_estimators.py` | Сглаживающие оценки наклона log C без обучения | при σ=0.05 лучшая оценка (exp_fit) R² 0.223, ols_log 0.190: сглаживание не помогает (`r17_slope_estimators.csv`) | статья 1, документ |
| R22 | `paper1/r22_ar1_noise_training.py` | Обучение с коррелированным шумом AR(1), пять seed | при σ=0.05, φ=0.9 physics_lstm на AR(1)-копиях 0.647 против 0.576 на белых; transformer 0.667 (`r22_ar1_noise_training.csv`) | статья 1, документ |

## Один кубит: перенос между системами и обратная задача

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| E4 | `paper1/e4_cross_system.py` | Одна модель, обучение на A+B+C, оценка по сценариям, один прогон | R² A 0.546, B 0.919, C 0.980 (`e4_cross_system_p1.csv`) | статья 1, документ |
| E5 | `paper1/e5_inverse_baseline.py` | Прямой прогноз T₂ против восстановления γ(t) и интегрирования Линдблада | при f=0.1 R² прямого 0.946, обратного −1.150; обратный отрицателен при всех f (`e5_inverse_baseline.csv`) | статья 1, документ |
| R19 | `paper1/r19_replicate_e4_e5.py` | E4 и E5 по трём seed | E4: A 0.452, B 0.835, C 0.978; E5: прямой 0.952–0.970, обратный от −2.184 до −0.444 (`r19_e4.csv`, `r19_e5.csv`) | статья 1 (таблицы E4, E5), документ |

## Один кубит: когда прогноз надёжен; стохастическая γ

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| R18 | `paper1/r18_alarm_timing.py` | Точность и тревога по ходу траектории, p = t_obs/T₂ | physics_lstm на B: тревога вовремя в 95.0% траекторий, на C в 100%; доля оценок T₂ с ошибкой ≤10% растёт с 0.59 (p<0.4) до 0.94 (p 0.6–0.7) на B (`r18_alarm_timing_alarm.csv`, `r18_alarm_timing_bins.csv`) | статья 1 (рисунок, таблица), документ |
| R21 | `paper1/r21_stochastic_gamma_ar1.py` | Шум AR(1) и случайная скорость γ(t) (логнормальный OU) | R² physics_lstm: 0.924 при s=0, 0.661 при s=0.6, τ=0.5 и 0.560 при τ=2 (`r21_stochastic_gamma_ar1.csv`) | статья 1 (рисунок, таблица), документ |

## Два кубита: модели и объём данных

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| E6 | `paper2/e6_2qubit_improved.py` | E6a — абляция на D, E; E6b — шум; E6c — одна модель на D+E; 500 траекторий, адаптивный prior | E6a transformer D/E 0.897 / 0.689, physics_lstm 0.636 / 0.455 (`e6_ablation_improved.csv`); E6c physics_lstm 0.706 / 0.533 (`e6_cross_system_improved.csv`) | статья 2, документ |
| E7 | `paper2/e7_paper2_extended.py` | E7a — transformer на D+E; E7b — порог τ; E7c — развёртка по J на TFIM | E7a D/E 0.758 / 0.599 (`e7a_transformer_cross.csv`); E7c: 0.876 при J=0.2 и 0.319 при J=1.2 (`e7c_J_sweep.csv`) | статья 2, документ |
| E8 | `paper2/e8_improved_2qubit.py` | Улучшения по итогам E7: τ по сценарию, взвешенный BCE, шумовая аугментация, окно 30; E8c — одна модель на D+E | E8c physics_lstm D/E 0.740 / 0.575 (`e8c_cross_system.csv`) | нигде (его импортирует R8; в workbench это E8c) |
| E11 | `paper2/e11_scaling.py` | E11a — transformer, окно 20; E11b — physics_lstm на 1000 траекториях | E11a D/E 0.817 / 0.566 (`e11a_cross_system.csv`); E11b 0.492 / 0.499 (`e11b_cross_system.csv`) | статья 2, документ |
| R1 | `paper2/r1_multiseed_ablation.py` | Абляция E6a по трём seed | D/E: transformer 0.843 / 0.699, physics_lstm 0.816 / 0.644, lstm_only 0.818 / 0.686, physics_only −3.982 / −5.979 (`r1_multiseed_ablation_full.csv`) | статья 2 (рисунок), документ |
| R2 | `checks/r2_multiseed_tau.py` | Развёртка по порогу τ из E7b, три seed | на D среднее R² 0.620–0.688 при всех τ; на E от 0.040 до 0.459 при std до 0.730, эффект τ не отделяется от разброса (`r2_multiseed_tau_full_seeded.csv`) | нигде |
| R3 | `paper2/r3_multiseed_scaling.py` | E11a и E11b по трём seed | E11a D/E 0.814 / 0.703; E11b при 1000 траекториях 0.815 / 0.673, при 500 0.793 / 0.663 (`r3_multiseed_scaling.csv`) | статья 2, документ |
| R4 | `paper2/r4_config_overlap.py` | Оценка E11a на конфигурациях из обучения и на новых | D: 0.814 и 0.814; E: 0.703 и 0.689 (`r4_config_overlap.csv`) | статья 2, документ |
| R8 | `paper2/r8_noise_and_scaling.py` | Шум E6b и масштабирование E8c по seed | E6b: R² 0.493 при σ=0 и 0.420 при σ=0.1; E8c D/E 0.795 / 0.596 (`r8_noise_and_scaling.csv`; seed у E8c не независимы, см. `note` в каталоге) | статья 2, документ |

## Два кубита: почему Изинг хуже, цель T₂ против T₂^E

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| R5 | `paper2/r5_multiseed_J_sweep.py` | Развёртка по J на TFIM, восемь seed | R² 0.793 при J=0.2, минимум 0.461 при J=1.0 (`r5_multiseed_J_sweep.csv`) | статья 2 (рисунок, таблица), документ |
| R6 | `paper2/r6_parity_sector.py` | Начальные состояния в одном секторе чётности: проверка объяснения через смешивание секторов | в секторе R² при J≥0.5 остаётся 0.49–0.66, провал не исчезает (`r6_parity_sector.csv`) | статья 2, документ |
| R9 | `paper2/r9_target_modulation.py` | Только моделирование: как гамильтониан модулирует цель T₂ | на TFIM доля траекторий с немонотонной C(t) растёт с 0.13 (J=0.2) до 0.75 (J=2.0), на XXZ с 0.00 до 0.56 (`r9_target_modulation.csv`) | статья 2 (рисунок, таблица), документ |
| R10 | `paper2/r10_energy_target_J_sweep.py` | Развёртка R5 с целью T₂^E в собственном базисе H, восемь seed | R² 0.787–0.861 при всех J против 0.461–0.793 у R5 (`r10_energy_target_J_sweep.csv`) | статья 2 (рисунок, таблица), документ |
| R20 | `paper2/r20_energy_target_xxz.py` | Цель T₂^E на XXZ (контроль) и на модели D+E | XXZ: среднее по J 0.826 против 0.743 с вычислительной целью; D+E: 0.906 / 0.886 (`r20a_xxz_J_sweep.csv`, `r20b_cross_energy.csv`) | статья 2, документ |

## Два кубита: корреляторы на входе

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| R23 | `paper2/r23_two_body_correlators.py` | Двухчастичные корреляторы Паули как входы; отклик целей на изменение скорости на 2% | R² на независимой выборке: transformer на E 0.687 → 0.718 с корреляторами, lstm_only на D 0.826 → 0.800; T₂ сдвигается на 0.75% (медиана), T₂^E на 1.6% (`r23_two_body_correlators.csv`, `r23c_target_sensitivity.csv`) | статья 2, документ |

## Запутанность

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| R24 | `paper2/r24_entanglement_target.py` | Цель — момент внезапной смерти запутанности (ESD), статистика смерти и возрождения | R² на независимой выборке 0.43–0.66; без смерти до t_max остаётся 46% траекторий на XXZ и 20% на TFIM (`r24_entanglement_target.csv`, `r24b_esd_statistics.csv`) | статья 2, документ |
| R25 | `paper2/r25_esd_data_scaling.py` | Кривая обучения по цели ESD, 250–4000 траекторий | R² с признаками corr_conc: D 0.413 → 0.687, E 0.510 → 0.812 (`r25_esd_data_scaling.csv`) | статья 2, документ |
| R26 | `paper2/r26_esd_occurrence.py` | Классификация: наступит ли смерть запутанности и останется ли окончательной, при фиксированном t_obs | AUROC gbt с corr_conc: наступление D 0.972, E 0.927; окончательность D 0.844 (`r26_esd_occurrence.csv`) | статья 2, документ |
| R27 | `paper2/r27_esd_permanence.py` | То же на XXZ, 8000/4000 траекторий, бутстреп-интервалы | AUROC gbt с corr_conc: наступление 0.983, окончательность 0.924 (`r27_esd_permanence.csv`) | статья 2, документ |

## Kondo-ванна

| ID | Скрипт | Что делает | Главный результат | Где процитирован |
|---|---|---|---|---|
| R28 | `new/r28_kondo_reproduction.py` | Диссипатор Kondo-ванны по Delgado и Fernandez-Rossier: воспроизведение их результатов, затем T₂ и T₂^E двух кубитов от k_F d | замкнутые формы статьи воспроизведены до 2e-11, кроме двух соглашений о нормировке Λ (расхождение 0.5–1.1); формула C6 в статье напечатана с ошибкой; T₂ на XXZ при d=1: 510 при k_F d/π=0 и 1013 при 0.25 (`r28/deviations_*.csv`, `r28/r28_demo_t2.csv`) | нигде |

## Удалено

Коммит `b625a60` (`chore: delete uncited experiments E9, E10, E12-E14 and the old trajectory cache`). Код — `git show b625a60^:experiments/<файл>`, CSV — `git show b625a60^:experiments/results/<файл>`.

- E9 `e9_context_injection.py` — кодовые числа системы на входе (`e9a_cross_system.csv`, `e9b_ablation.csv`).
- E10a/E10b `e10_fixed_context.py` — фиксированный контекст (`e10a_cross_system.csv`, `e10b_cross_system.csv`). В `paper2/e11_scaling.py` остались как константы «Transformer w=30 (earlier run)» значения E10b.
- E10-TFIM `e10_tfim_specialized.py` — модель только на TFIM (`e10_tfim_c.csv`).
- E12 `e12_baseline_comparison.py` — базовая модель stretched-exp (`e12_baseline_comparison.csv`).
- E13 `e13_survival_tfim.py` — функция потерь для цензурированных траекторий (`e13_survival_tfim.csv`).
- E14 `e14_inject_J.py` — J как вход (`e14_inject_J.csv`).
- `r2_multiseed_tau_full.csv` — файл, на который никто не ссылался.
- Кэш `data/trajectories/` (1.6 ГБ): `pipeline.generate` его не читает, скрипты используют кэш в памяти процесса; три траектории, нужные `scripts/golden_outputs.py`, лежат в `scripts/golden_cache/`.
