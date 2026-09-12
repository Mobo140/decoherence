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

- [ ] 4+ PDF в `paper/paper_1/figures/`, сгенерированы из существующих CSV
- [ ] Каждый рисунок встроен в `main.tex` с подписью и упомянут через `\ref`
- [ ] Числа на рисунках совпадают с таблицами E2/E3/E5 в тексте
- [ ] `make pdf-1` компилируется без ошибок, рисунки видны в `main.pdf`
- [ ] Стиль единый (шрифты/подписи), пригодно для двухколоночного формата

## Invariants

- IV1: Результаты экспериментов и таблицы не меняются — только визуализация
- IV2: Чемпионские CSV не перезаписываются (только чтение)

## Verification

```bash
ls paper/paper_1/figures/*.pdf
make pdf-1 && echo "PDF OK"
```

## Status

`planned`

## Priority

P1
