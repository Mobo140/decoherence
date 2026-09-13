# Task: Paper 1 — publication-рисунки (сейчас 0)

## Why

Paper 1 (`paper/paper_1/main.tex`) содержит только таблицы, ни одного
`\includegraphics`; `paper_1/figures/` пуст. Для сабмита (npj QI / Phys Rev Applied)
нужны рисунки. Все данные уже есть в `experiments/results/*_p1*.csv` — рисуем из них,
**не перезапуская** эксперименты (иначе перезапишем чемпионские CSV).

## Scope

**Входит:**
- `notebooks/figures.ipynb` — добавить/переиспользовать ячейки для Paper 1
- `paper/paper_1/figures/` — сохранить 4–5 PDF-рисунков
- `paper/paper_1/main.tex` — вставить `\includegraphics` + подписи + `\ref` в текст

**Рисунки:**
1. Схема остаточной архитектуры (physics OLS ⊕ LSTM-коррекция) — matplotlib/TikZ
2. E2: AUROC и R² vs окно f (AUROC≥0.93 уже при f=0.05) — из `e2_*.csv`
3. E3: R²/MAE/AUROC vs σ_noise — из `e3_*.csv`
4. E5: direct vs inverse (R²/MAE vs f) — из `e5_*.csv`
5. (опц.) C(t)-распад + OLS-фит — иллюстрация physics-приора

**Не входит:**
- Перезапуск экспериментов, изменение чисел в таблицах
- Рисунки Paper 2 (уже есть)

## Acceptance Criteria

- [x] 4+ PDF в `paper/paper_1/figures/`, сгенерированы из существующих CSV (6 шт.:
  E1/E2/E3/E4/E5 + physics_validation)
- [x] Каждый рисунок встроен в `main.tex` с подписью и упомянут через `\ref`
- [x] Числа на рисунках совпадают с таблицами E1/E2/E3/E4/E5 в тексте
- [x] `make pdf-1` компилируется без ошибок, рисунки видны в `main.pdf`
- [x] Стиль единый — переиспользован stylesheet + палитра из `notebooks/figures.ipynb`,
  1:1 паттерн с Paper 2 (`\includegraphics[width=\columnwidth]` + caption + label
  сразу после соответствующей таблицы)

## Invariants

- IV1: Результаты экспериментов и таблицы не меняются — только визуализация
- IV2: Чемпионские CSV не перезаписываются (только чтение)

## Verification

```bash
ls paper/paper_1/figures/*.pdf
make pdf-1 && echo "PDF OK"
```

## Status

`done`

Рисунки 1:5 для E1–E5 уже существовали как рабочий код в `notebooks/figures.ipynb`
(тот же паттерн, что дал рисунки Paper 2) — но никогда не копировались в
`paper_1/figures/` и не подключались в `main.tex`. Перегенерировал их (без
jupyter/nbconvert в venv — автономный скрипт с тем же кодом) и добавил 6-й:
`fig_physics_validation` (иллюстрация claim C1 в Method) — с явной оговоркой в
caption, что это отдельный demo-прогон (R²≈1.0000), не то же число, что в
Table 4 (R²=0.9995, E1 test set), чтобы не создавать фактического противоречия.

Заодно нашёл и починил настоящий overfull \hbox (71.8pt) в Table 4 — не моя
правка контента, обнаружилось при рендере страниц; сузил tabcolsep/шрифт,
цифры не менял.

Коммит: `6b46596`.

## Priority

P1
